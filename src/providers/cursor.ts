import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { Agent, AuthenticationError, ConfigurationError, CursorAgentError } from "@cursor/sdk";
import type { WriteAllowlist } from "../allowlist.js";
import { awaitOrTimeout, DEFAULT_CURSOR_TIMEOUT_MS } from "../chat/timeout.js";
import type { AgentProvider, CursorBackendConfig, ProviderHealth, ProviderRunRequest, ProviderRunResult } from "../types.js";
import { missingKeyHealth, readyHealth } from "./util.js";

type LiveAgent = {
  agent: Awaited<ReturnType<typeof Agent.create>>;
  lastUsed: number;
};

const LIVE_TTL_MS = 30 * 60 * 1000;

export const CURSOR_NOT_CONFIGURED = "Cursor not configured";

export function formatCursorError(error: unknown): string {
  if (error instanceof AuthenticationError || error instanceof ConfigurationError) {
    return CURSOR_NOT_CONFIGURED;
  }
  const raw = error instanceof Error ? error.message : String(error ?? "");
  const first = (raw.trim().split(/\n/)[0] ?? "").trim();
  if (/CURSOR_API_KEY|not configured|unauthoriz|\b401\b|api key|authentication/i.test(first)) {
    return CURSOR_NOT_CONFIGURED;
  }
  if (/timed out after \d+s/i.test(first)) return first.slice(0, 160);
  if (/run stream is no longer available/i.test(first)) {
    return "Cursor run stream is no longer available — skipped so other speakers can finish.";
  }
  if (error instanceof CursorAgentError) {
    return `Cursor startup failed: ${first || error.message}`.slice(0, 160);
  }
  if (!first || /^(none|null|undefined)$/i.test(first)) {
    return "Cursor run failed (no error detail from the SDK).";
  }
  return first.slice(0, 160);
}

/** Prefer MCP process cwd; never a hardcoded home path. Fall back to a git work tree if cwd is not one. */
export function resolveCursorLocalCwd(requested?: string): string {
  const candidates = [requested, process.cwd()].filter((p): p is string => Boolean(p?.trim()));
  for (const raw of candidates) {
    const abs = resolve(raw);
    if (existsSync(join(abs, ".git"))) return abs;
  }
  return resolve(requested?.trim() || process.cwd());
}

export function gitOriginHttpsUrl(cwd: string): string | undefined {
  try {
    const out = execFileSync("git", ["-C", cwd, "remote", "get-url", "origin"], {
      timeout: 1500,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
    return normalizeGitHttpsUrl(out);
  } catch {
    return undefined;
  }
}

export function normalizeGitHttpsUrl(url: string): string | undefined {
  const t = url.trim();
  if (!t) return undefined;
  const ssh = t.match(/^git@([^:]+):(.+?)(?:\.git)?$/i);
  if (ssh?.[1] && ssh[2]) {
    return `https://${ssh[1]}/${ssh[2].replace(/\.git$/i, "")}.git`;
  }
  if (/^https?:\/\//i.test(t)) return t;
  return undefined;
}

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
        reason: CURSOR_NOT_CONFIGURED,
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
      return { status: "error", text: "", error: CURSOR_NOT_CONFIGURED, durationMs: 0 };
    }
    const timeoutMs = request.timeoutMs ?? DEFAULT_CURSOR_TIMEOUT_MS;
    const timeoutMessage = `Cursor timed out after ${Math.round(timeoutMs / 1000)}s`;
    const deadline = started + timeoutMs;

    const attempt = async (): Promise<ProviderRunResult> => {
      let agent: Awaited<ReturnType<typeof Agent.create>> | undefined;
      const remaining = Math.max(50, deadline - Date.now());
      const work = (async () => {
        agent = request.resumeAgentId
          ? await this.resume(request.resumeAgentId, apiKey)
          : await this.create(request, apiKey);
        const prompt = request.system ? `${request.system}\n\n---\n\n${request.prompt}` : request.prompt;
        const run = await agent.send(prompt, request.mode ? { mode: request.mode } : undefined);
        return await run.wait();
      })();

      try {
        const result = await awaitOrTimeout(work, remaining, timeoutMessage);
        if (agent) this.live.set(agent.agentId, { agent, lastUsed: Date.now() });

        if (result.status === "error") {
          return {
            status: "error",
            text: result.result ?? "",
            error: formatCursorError(result.error?.message ?? result.error ?? "Cursor run failed"),
            agentId: agent?.agentId,
            providerRunId: result.id,
            durationMs: Date.now() - started,
          };
        }

        return {
          status: result.status,
          text: result.result ?? "",
          agentId: agent?.agentId,
          providerRunId: result.id,
          durationMs: Date.now() - started,
        };
      } catch (error) {
        const message = formatCursorError(error);
        const timedOut = /timed out after \d+s/i.test(message);
        if (timedOut) {
          if (agent) this.live.set(agent.agentId, { agent, lastUsed: Date.now() });
        } else {
          this.safeClose(agent);
        }
        return {
          status: "error",
          text: "",
          error: message,
          agentId: agent?.agentId,
          durationMs: Date.now() - started,
        };
      }
    };

    let result = await attempt();
    if (result.status === "error" && /\b429\b|rate[- ]limit|RESOURCE_EXHAUSTED/i.test(result.error ?? "")) {
      const waitMs = Math.min(800, Math.max(0, deadline - Date.now() - 1));
      if (waitMs > 0) await new Promise((r) => setTimeout(r, waitMs));
      if (Date.now() < deadline) result = await attempt();
    }
    return result;
  }

  async followUp(agentId: string, message: string): Promise<ProviderRunResult> {
    return this.run({ prompt: message, resumeAgentId: agentId });
  }

  private async create(request: ProviderRunRequest, apiKey: string) {
    const model = { id: request.model ?? this.config.model ?? "composer-2.5" };
    if (this.config.runtime === "cloud") {
      const repos = this.cloudRepos(request);
      return Agent.create({
        apiKey,
        model,
        mode: request.mode,
        cloud: {
          ...(repos.length ? { repos } : {}),
          autoCreatePR: request.cloud?.autoCreatePR,
        },
      });
    }
    const cwd = this.localCwd(request);
    return Agent.create({
      apiKey,
      model,
      mode: request.mode,
      local: { cwd },
    });
  }

  private localCwd(request: ProviderRunRequest): string {
    const resolved = resolveCursorLocalCwd(request.cwd);
    if (!this.allowlist) return resolved;
    const allowed = this.allowlist.tryCwd(resolved) ?? this.allowlist.list()[0];
    if (allowed) return allowed;
    return this.allowlist.assertCwd(resolved);
  }

  private cloudRepos(request: ProviderRunRequest): Array<{ url: string; startingRef?: string }> {
    const given = (request.cloud?.repos ?? []).filter((r) => r.url?.trim());
    if (given.length) return given;
    const origin = gitOriginHttpsUrl(resolveCursorLocalCwd(request.cwd));
    if (!origin) return [];
    return [{ url: origin, startingRef: undefined }];
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

  private safeClose(agent: { close?: () => unknown; [Symbol.asyncDispose]?: () => Promise<void> } | undefined): void {
    if (!agent) return;
    try {
      const dispose = agent[Symbol.asyncDispose];
      if (typeof dispose === "function") {
        void dispose.call(agent).then(
          () => undefined,
          () => undefined,
        );
        return;
      }
      if (!agent.close) return;
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
