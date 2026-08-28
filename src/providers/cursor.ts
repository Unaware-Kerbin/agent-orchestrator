import { Agent, CursorAgentError } from "@cursor/sdk";
import type { WriteAllowlist } from "../allowlist.js";
import { awaitOrTimeout } from "../chat/timeout.js";
import type { AgentProvider, CursorBackendConfig, ProviderHealth, ProviderRunRequest, ProviderRunResult } from "../types.js";
import { missingKeyHealth, readyHealth } from "./util.js";

type LiveAgent = {
  agent: Awaited<ReturnType<typeof Agent.create>>;
  lastUsed: number;
};

const LIVE_TTL_MS = 30 * 60 * 1000;

export class CursorProvider implements AgentProvider {
  readonly type = "cursor" as const;
  readonly capabilities = ["text", "workspace", "follow_up", "cancel"];
  private readonly live = new Map<string, LiveAgent>();

  constructor(
    readonly id: string,
    private readonly config: CursorBackendConfig,
    private readonly allowlist?: WriteAllowlist,
  ) {
    const timer = setInterval(() => {
      try {
        this.reap();
      } catch {
        /* isolate SDK close failures from the HTTP process */
      }
    }, 60_000);
    timer.unref?.();
  }

  health(): ProviderHealth {
    const writesLocalFiles = this.config.runtime === "local";
    const extra = { runtime: this.config.runtime, writesLocalFiles };
    if (!process.env.CURSOR_API_KEY?.trim()) {
      return missingKeyHealth(this.id, "cursor", "CURSOR_API_KEY", this.capabilities, {
        ...extra,
        secretNames: ["CURSOR_API_KEY"],
        needsKey: true,
      });
    }
    return readyHealth(this.id, "cursor", this.capabilities, {
      ...extra,
      secretNames: ["CURSOR_API_KEY"],
      needsKey: true,
      reason: "CURSOR_API_KEY present (masked; never displayed).",
    });
  }

  async run(request: ProviderRunRequest): Promise<ProviderRunResult> {
    const started = Date.now();
    const apiKey = process.env.CURSOR_API_KEY?.trim();
    if (!apiKey) {
      return { status: "error", text: "", error: "CURSOR_API_KEY is not set", durationMs: 0 };
    }
    const timeoutMs = request.timeoutMs ?? 30_000;

    let agent: Awaited<ReturnType<typeof Agent.create>> | undefined;
    try {
      agent = request.resumeAgentId
        ? await this.resume(request.resumeAgentId, apiKey)
        : await this.create(request, apiKey);

      const prompt = request.system
        ? `${request.system}\n\n---\n\n${request.prompt}`
        : request.prompt;

      const run = await agent.send(prompt, request.mode ? { mode: request.mode } : undefined);
      const result = await awaitOrTimeout(
        run.wait(),
        timeoutMs,
        `Cursor timed out after ${Math.round(timeoutMs / 1000)}s`,
      );
      this.live.set(agent.agentId, { agent, lastUsed: Date.now() });

      if (result.status === "error") {
        return {
          status: "error",
          text: result.result ?? "",
          error: result.error?.message ?? "Cursor run failed",
          agentId: agent.agentId,
          providerRunId: result.id,
          durationMs: Date.now() - started,
        };
      }

      return {
        status: result.status,
        text: result.result ?? "",
        agentId: agent.agentId,
        providerRunId: result.id,
        durationMs: Date.now() - started,
      };
    } catch (error) {
      this.safeClose(agent);
      const message =
        error instanceof CursorAgentError
          ? `Cursor startup failed: ${error.message} (retryable=${error.isRetryable})`
          : error instanceof Error
            ? error.message
            : String(error);
      return { status: "error", text: "", error: message, durationMs: Date.now() - started };
    }
  }

  async followUp(agentId: string, message: string): Promise<ProviderRunResult> {
    return this.run({ prompt: message, resumeAgentId: agentId });
  }

  private async create(request: ProviderRunRequest, apiKey: string) {
    const model = { id: request.model ?? this.config.model ?? "composer-2.5" };
    if (this.config.runtime === "cloud") {
      return Agent.create({
        apiKey,
        model,
        mode: request.mode,
        cloud: {
          repos: request.cloud?.repos ?? [],
          autoCreatePR: request.cloud?.autoCreatePR,
        },
      });
    }
    const cwd = request.cwd ?? process.cwd();
    const safeCwd = this.allowlist ? this.allowlist.assertCwd(cwd) : cwd;
    return Agent.create({
      apiKey,
      model,
      mode: request.mode,
      local: { cwd: safeCwd },
    });
  }

  private async resume(agentId: string, apiKey: string) {
    const cached = this.live.get(agentId);
    if (cached) {
      cached.lastUsed = Date.now();
      return cached.agent;
    }
    const agent = await Agent.resume(agentId, { apiKey });
    this.live.set(agentId, { agent, lastUsed: Date.now() });
    return agent;
  }

  private reap(): void {
    const now = Date.now();
    for (const [id, session] of this.live) {
      if (now - session.lastUsed < LIVE_TTL_MS) continue;
      this.live.delete(id);
      this.safeClose(session.agent);
    }
  }

  private safeClose(agent: { close?: () => unknown } | undefined): void {
    if (!agent?.close) return;
    try {
      const closed = agent.close();
      if (closed && typeof (closed as Promise<unknown>).then === "function") {
        void (closed as Promise<unknown>).then(
          () => undefined,
          () => undefined,
        );
      }
    } catch {
      /* isolate SDK close failures */
    }
  }
}
