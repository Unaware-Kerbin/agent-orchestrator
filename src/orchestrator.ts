import { EventEmitter } from "node:events";
import { statSync } from "node:fs";
import type { WriteAllowlist } from "./allowlist.js";
import { defaultWorkspaceCwd, loadConfig } from "./config.js";
import { LocalModelService } from "./local-models/service.js";
import { specialistPrompt } from "./prompts.js";
import { createProviders } from "./providers/index.js";
import { hasLogo } from "./identity.js";
import { refreshRuntimeEnv } from "./secrets.js";
import { RunStore } from "./store.js";
import { TempAnalyzeAllowlist } from "./temp-allowlist.js";
import type {
  AgentProvider,
  BackendConfig,
  ConversationMode,
  OrchestratedRun,
  OrchestratorConfig,
  ProviderHealth,
  ProviderRunRequest,
  ProviderRunResult,
} from "./types.js";

export interface DispatchInput {
  specialist: string;
  task: string;
  backend?: string;
  cwd?: string;
  model?: string;
  wait?: boolean;
  prUrl?: string;
  repoUrl?: string;
  branch?: string;
  extraContext?: string;
  cloudAutoCreatePr?: boolean;
  /** Override specialist conversation mode (e.g. plan during round-table debate). */
  mode?: ConversationMode;
  onDelta?: (delta: string) => void;
  /** Cap this dispatch. Hung cloud calls should fail fast in a round-table. */
  timeoutMs?: number;
}

export interface FollowUpInput {
  runId: string;
  message: string;
  wait?: boolean;
  onDelta?: (delta: string) => void;
  timeoutMs?: number;
}

export interface WorkflowInput {
  workflow: string;
  task: string;
  cwd?: string;
  prUrl?: string;
  repoUrl?: string;
  branch?: string;
  extraContext?: string;
  stopOnError?: boolean;
}

function formatContext(input: {
  cwd?: string;
  prUrl?: string;
  repoUrl?: string;
  branch?: string;
  extraContext?: string;
}): string {
  const lines: string[] = [];
  if (input.cwd) lines.push(`Workspace: ${input.cwd}`);
  if (input.prUrl) lines.push(`PR: ${input.prUrl}`);
  if (input.repoUrl) lines.push(`Repo: ${input.repoUrl}`);
  if (input.branch) lines.push(`Branch: ${input.branch}`);
  if (input.extraContext) lines.push(input.extraContext);
  return lines.join("\n");
}

export class Orchestrator {
  readonly store = new RunStore();
  readonly events = new EventEmitter();
  readonly localModels: LocalModelService;
  readonly tempAnalyze = new TempAnalyzeAllowlist();
  providers: Map<string, AgentProvider>;
  private configMtime = 0;

  constructor(
    private _config: OrchestratorConfig,
    readonly allowlist: WriteAllowlist,
    readonly configPath: string,
  ) {
    this.providers = createProviders(_config.backends, allowlist);
    this.rememberConfigMtime();
    this.localModels = new LocalModelService(
      allowlist,
      this.events,
      configPath,
      () => this._config,
      () => this.reloadConfig(),
    );
  }

  get config(): OrchestratorConfig {
    return this._config;
  }

  defaultCwd(): string {
    return defaultWorkspaceCwd(this._config);
  }

  syncFromDisk(): void {
    this.allowlist.reloadIfChanged();
    try {
      const mtime = statSync(this.configPath).mtimeMs;
      if (mtime > this.configMtime) this.reloadConfig();
    } catch {
      // keep current config
    }
  }

  reloadConfig(next?: OrchestratorConfig): void {
    this._config = next ?? loadConfig(this.configPath);
    this.providers = createProviders(this._config.backends, this.allowlist);
    this.rememberConfigMtime();
    this.emitCatalog();
  }

  async catalog() {
    refreshRuntimeEnv();
    this.syncFromDisk();
    return this.buildCatalog();
  }

  private emitCatalog(): void {
    void this.buildCatalog().then(
      (catalog) => {
        try {
          this.events.emit("catalog", catalog);
        } catch {
          /* SSE client gone */
        }
      },
      () => undefined,
    );
  }

  private async buildCatalog() {
    const healthById = new Map<string, ProviderHealth>();
    await Promise.all(
      [...this.providers.entries()].map(async ([id, provider]) => {
        healthById.set(id, await providerHealth(provider));
      }),
    );
    const backends = [...this.providers.keys()].map((id) => {
      const health = healthById.get(id)!;
      const cfg = this._config.backends[id];
      return {
        ...health,
        nickname: cfg?.nickname,
        hasLogo: hasLogo(id),
      };
    });
    const specialists = Object.entries(this._config.specialists).map(([id, spec]) => {
      const backendCfg = this._config.backends[spec.backend];
      return {
        id,
        description: spec.description,
        backend: spec.backend,
        fallback: spec.fallback,
        mode: spec.mode,
        backendReady: healthById.get(spec.backend)?.ready ?? false,
        fallbackReady: spec.fallback ? (healthById.get(spec.fallback)?.ready ?? false) : undefined,
        writesLocalFiles: writesLocalFiles(backendCfg),
      };
    });
    const workflows = Object.entries(this._config.workflows).map(([id, workflow]) => ({
      id,
      description: workflow.description,
      mode: workflow.mode ?? "sequence",
      steps: workflow.steps.map((step) =>
        step.backend ? `${step.specialist} (${step.backend})` : step.specialist,
      ),
    }));
    const ollama = backends.find((b) => b.type === "ollama");
    const llamacpp = backends.filter((b) => b.type === "llamacpp");
    return {
      backends,
      specialists,
      workflows,
      writePolicy: {
        allowedDirectories: this.allowlist.list(),
        defaultCwd: this.defaultCwd(),
        fileWrites: "cursor-local-only" as const,
      },
      localRuntime: {
        ...this.localModels.catalogSummary(),
        ollama: ollama
          ? {
              running: ollama.ready,
              model: ollama.model,
              baseUrl: ollama.baseUrl,
              reason: ollama.reason,
              models: ollama.modelChoices,
            }
          : { running: false, reason: "No Ollama backend configured" },
        llamacpp: llamacpp.map((b) => ({
          id: b.id,
          running: b.ready,
          model: b.model,
          baseUrl: b.baseUrl,
          reason: b.reason,
        })),
      },
    };
  }

  async dispatch(input: DispatchInput): Promise<OrchestratedRun> {
    refreshRuntimeEnv();
    this.syncFromDisk();
    const spec = this._config.specialists[input.specialist];
    if (!spec) {
      throw new Error(`Unknown specialist "${input.specialist}". Known: ${Object.keys(this._config.specialists).join(", ")}`);
    }
    const { provider, backendId, backend } = await this.resolveProvider(input.backend ?? spec.backend, spec.fallback);
    const declaredCwd = input.cwd ?? this.defaultCwd();
    const cwd = this.cwdForBackend(backend, declaredCwd);
    const prompt = this.buildPrompt(input.task, { ...input, cwd: declaredCwd });
    const run = this.store.create({
      specialist: input.specialist,
      backend: backendId,
      status: "running",
      prompt,
      cwd,
    });
    this.emitRun(run.id);

    const request: ProviderRunRequest = {
      prompt,
      system: specialistPrompt(input.specialist, spec.system),
      cwd,
      model: input.model ?? (backend?.type === "cursor" ? this._config.defaults?.model : undefined),
      mode: input.mode ?? spec.mode,
      onDelta: input.onDelta,
      timeoutMs: input.timeoutMs,
      cloud: input.repoUrl
        ? {
            repos: [{ url: input.repoUrl, startingRef: input.branch }],
            autoCreatePR: input.cloudAutoCreatePr,
          }
        : undefined,
    };

    const wait = input.wait ?? this._config.defaults?.wait ?? true;
    if (!wait) {
      void this.execute(run.id, provider, request).catch(() => undefined);
      return this.store.get(run.id)!;
    }
    await this.execute(run.id, provider, request);
    return this.store.get(run.id)!;
  }

  async followUp(input: FollowUpInput): Promise<OrchestratedRun> {
    refreshRuntimeEnv();
    this.syncFromDisk();
    const existing = this.store.get(input.runId);
    if (!existing) throw new Error(`Unknown run "${input.runId}"`);
    const spec = this._config.specialists[existing.specialist];
    const { provider, backend } = await this.resolveProvider(existing.backend, spec?.fallback);
    if (writesLocalFiles(backend) && existing.cwd) {
      this.allowlist.assertCwd(existing.cwd);
    }
    const history = [...existing.history];
    const request: ProviderRunRequest = {
      prompt: input.message,
      system: spec ? specialistPrompt(existing.specialist, spec.system) : undefined,
      history,
      resumeAgentId: existing.agentId,
      cwd: existing.cwd,
      onDelta: input.onDelta,
      timeoutMs: input.timeoutMs,
    };
    const child = this.store.create({
      specialist: existing.specialist,
      backend: existing.backend,
      status: "running",
      prompt: input.message,
      agentId: existing.agentId,
      history,
      cwd: existing.cwd,
    });
    this.emitRun(child.id);
    const wait = input.wait ?? true;
    if (!wait) {
      void this.execute(child.id, provider, request).catch(() => undefined);
      return this.store.get(child.id)!;
    }
    await this.execute(child.id, provider, request);
    return this.store.get(child.id)!;
  }

  async runWorkflow(input: WorkflowInput): Promise<{
    workflow: string;
    runs: OrchestratedRun[];
    status: "finished" | "error";
    summary: string;
  }> {
    this.syncFromDisk();
    const workflow = this._config.workflows[input.workflow];
    if (!workflow) {
      throw new Error(`Unknown workflow "${input.workflow}". Known: ${Object.keys(this._config.workflows).join(", ")}`);
    }
    const stopOnError = input.stopOnError !== false;
    const runs: OrchestratedRun[] = [];
    const outputs: string[] = [];

    if (workflow.mode === "parallel") {
      const dispatched = await Promise.all(
        workflow.steps.map((step, index) =>
          this.dispatch({
            specialist: step.specialist,
            backend: step.backend,
            task: input.task,
            cwd: input.cwd,
            prUrl: input.prUrl,
            repoUrl: input.repoUrl,
            branch: input.branch,
            extraContext: input.extraContext,
            wait: true,
          }).then((run) => {
            this.store.update(run.id, { workflowId: input.workflow, stepIndex: index });
            this.emitRun(run.id);
            return this.store.get(run.id)!;
          }),
        ),
      );
      runs.push(...dispatched);
      for (const [index, step] of workflow.steps.entries()) {
        const updated = dispatched[index]!;
        outputs.push(`(${step.specialist})\n${updated.text ?? updated.error ?? ""}`);
      }
      const failed = dispatched.find((run) => run.status === "error");
      if (failed) {
        return {
          workflow: input.workflow,
          runs,
          status: "error",
          summary: `Parallel workflow had errors. ${outputs.join("\n\n-----\n\n")}`,
        };
      }
      return {
        workflow: input.workflow,
        runs,
        status: "finished",
        summary: outputs.join("\n\n-----\n\n"),
      };
    }

    for (const [index, step] of workflow.steps.entries()) {
      const prior = outputs.length
        ? `\n\nPrior specialist output:\n${outputs.map((text, i) => `### Step ${i + 1}\n${text}`).join("\n\n")}`
        : "";
      const run = await this.dispatch({
        specialist: step.specialist,
        backend: step.backend,
        task: input.task,
        cwd: input.cwd,
        prUrl: input.prUrl,
        repoUrl: input.repoUrl,
        branch: input.branch,
        extraContext: `${input.extraContext ?? ""}${prior}`.trim() || undefined,
        wait: true,
      });
      this.store.update(run.id, { workflowId: input.workflow, stepIndex: index });
      this.emitRun(run.id);
      const updated = this.store.get(run.id)!;
      runs.push(updated);
      outputs.push(`(${step.specialist})\n${updated.text ?? updated.error ?? ""}`);
      if (updated.status === "error" && stopOnError) {
        return {
          workflow: input.workflow,
          runs,
          status: "error",
          summary: `Stopped at ${step.specialist}: ${updated.error ?? "unknown error"}`,
        };
      }
    }

    return {
      workflow: input.workflow,
      runs,
      status: "finished",
      summary: outputs.join("\n\n-----\n\n"),
    };
  }

  private cwdForBackend(backend: BackendConfig | undefined, declaredCwd: string): string | undefined {
    if (writesLocalFiles(backend)) {
      const allowed = this.allowlist.tryCwd(declaredCwd);
      if (allowed) return allowed;
      const first = this.allowlist.list()[0];
      if (first) return first;
      return this.allowlist.assertCwd(declaredCwd);
    }
    return undefined;
  }

  private async execute(
    runId: string,
    provider: AgentProvider,
    request: ProviderRunRequest,
  ): Promise<ProviderRunResult> {
    try {
      const result = await provider.run(request);
      const existing = this.store.get(runId);
      this.store.update(runId, {
        status: result.status,
        text: result.text,
        error: result.error,
        agentId: result.agentId ?? existing?.agentId,
        providerRunId: result.providerRunId,
        durationMs: result.durationMs,
        history: [
          ...(existing?.history ?? []),
          { role: "user", content: request.prompt },
          ...(result.text ? [{ role: "assistant" as const, content: result.text }] : []),
        ],
      });
      this.emitRun(runId);
      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const existing = this.store.get(runId);
      this.store.update(runId, {
        status: "error",
        text: existing?.text ?? "",
        error: message,
        agentId: existing?.agentId,
        history: existing?.history ?? [],
      });
      this.emitRun(runId);
      return { status: "error", text: "", error: message, durationMs: 0 };
    }
  }

  private emitRun(runId: string): void {
    const run = this.store.get(runId);
    if (run) this.events.emit("run", run);
  }

  private rememberConfigMtime(): void {
    try {
      this.configMtime = statSync(this.configPath).mtimeMs;
    } catch {
      this.configMtime = Date.now();
    }
  }

  private resolveBackendId(id: string): string {
    if (this.providers.has(id)) return id;
    const specialist = this._config.specialists[id];
    if (specialist) return specialist.backend;
    return id;
  }

  private async resolveProvider(
    preferred: string,
    fallback?: string,
  ): Promise<{ provider: AgentProvider; backendId: string; backend: BackendConfig | undefined }> {
    const preferredId = this.resolveBackendId(preferred);
    const primary = this.providers.get(preferredId);
    const primaryHealth = primary ? await providerHealth(primary) : undefined;
    if (primary && primaryHealth?.ready) {
      return { provider: primary, backendId: preferredId, backend: this._config.backends[preferredId] };
    }
    if (fallback) {
      const fallbackId = this.resolveBackendId(fallback);
      const secondary = this.providers.get(fallbackId);
      const secondaryHealth = secondary ? await providerHealth(secondary) : undefined;
      if (secondary && secondaryHealth?.ready) {
        return { provider: secondary, backendId: fallbackId, backend: this._config.backends[fallbackId] };
      }
    }
    const reason = primaryHealth?.reason ?? `Unknown backend "${preferred}"`;
    throw new Error(
      `Backend "${preferred}" is not ready (${reason})${fallback ? `; fallback "${fallback}" is also unavailable` : ""}`,
    );
  }

  private buildPrompt(task: string, input: DispatchInput): string {
    const context = formatContext({
      cwd: input.cwd ?? this._config.workspace?.cwd,
      prUrl: input.prUrl,
      repoUrl: input.repoUrl,
      branch: input.branch,
      extraContext: input.extraContext,
    });
    return context ? `${task}\n\n${context}` : task;
  }
}

function writesLocalFiles(backend: BackendConfig | undefined): boolean {
  return backend?.type === "cursor" && backend.runtime === "local";
}

async function providerHealth(provider: AgentProvider): Promise<ProviderHealth> {
  return provider.probe ? provider.probe() : provider.health();
}
