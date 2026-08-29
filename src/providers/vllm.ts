import type { AgentProvider, ProviderHealth, ProviderRunRequest, ProviderRunResult, VllmBackendConfig } from "../types.js";
import {
  envNamesForBackend,
  isUnreachableError,
  normalizeBaseUrl,
  VLLM_LOCAL_DUMMY_KEY,
  vllmUnreachableReason,
} from "./keys.js";
import { runOpenAiChat } from "./openai.js";
import { readyHealth, secretFrom } from "./util.js";

const DEFAULT_BASE = "http://127.0.0.1:8000/v1";

export class VllmProvider implements AgentProvider {
  readonly type = "vllm" as const;
  readonly capabilities = ["text", "follow_up"];
  private lastProbe: { at: number; health: ProviderHealth } | undefined;

  constructor(
    readonly id: string,
    private readonly config: VllmBackendConfig,
  ) {}

  health(): ProviderHealth {
    if (this.lastProbe && Date.now() - this.lastProbe.at < 15_000) {
      return this.lastProbe.health;
    }
    const snapshot = this.configHealth();
    if (this.config.probe === false) return snapshot;
    return {
      ...snapshot,
      ready: false,
      reason: `Configured at ${normalizeBaseUrl(this.config.baseUrl ?? DEFAULT_BASE)} (will probe /models; API key optional)`,
    };
  }

  async probe(): Promise<ProviderHealth> {
    const snapshot = this.configHealth();
    if (!snapshot.ready || this.config.probe === false) {
      this.lastProbe = { at: Date.now(), health: snapshot };
      return snapshot;
    }
    const baseUrl = normalizeBaseUrl(this.config.baseUrl ?? DEFAULT_BASE);
    const timeoutMs = this.config.probeTimeoutMs ?? 800;
    const key =
      secretFrom(this.config, envNamesForBackend(this.id, this.config)) ??
      this.config.apiKey ??
      VLLM_LOCAL_DUMMY_KEY;
    try {
      const response = await fetch(`${baseUrl}/models`, {
        method: "GET",
        headers: {
          authorization: `Bearer ${key}`,
        },
        signal: AbortSignal.timeout(timeoutMs),
      });
      void response.arrayBuffer().catch(() => undefined);
      if (!response.ok) {
        const health: ProviderHealth = {
          id: this.id,
          type: "vllm",
          ready: false,
          reason: `vLLM not ready at ${baseUrl} (HTTP ${response.status})`,
          capabilities: this.capabilities,
          writesLocalFiles: false,
          ...this.meta(),
        };
        this.lastProbe = { at: Date.now(), health };
        return health;
      }
      const health = readyHealth(this.id, "vllm", this.capabilities, {
        ...this.meta(),
        reason: `vLLM reachable at ${baseUrl} (HTTP ${response.status})`,
      });
      this.lastProbe = { at: Date.now(), health };
      return health;
    } catch (error) {
      const unreachable = vllmUnreachableReason(baseUrl, error);
      const timedOut = error instanceof Error && /timeout|aborted/i.test(error.message);
      const reason =
        unreachable ??
        (timedOut
          ? `vLLM not reachable at ${baseUrl} (timeout)`
          : isUnreachableError(error)
            ? `vLLM not running at ${baseUrl}`
            : `vLLM not reachable at ${baseUrl}: ${error instanceof Error ? error.message : String(error)}`);
      const health: ProviderHealth = {
        id: this.id,
        type: "vllm",
        ready: false,
        reason,
        capabilities: this.capabilities,
        writesLocalFiles: false,
        ...this.meta(),
      };
      this.lastProbe = { at: Date.now(), health };
      return health;
    }
  }

  run(request: ProviderRunRequest): Promise<ProviderRunResult> {
    return runOpenAiChat(this.id, this.config, request, {
      label: "vLLM",
      defaultBaseUrl: DEFAULT_BASE,
      optionalKey: true,
    });
  }

  followUp(agentId: string, message: string): Promise<ProviderRunResult> {
    void agentId;
    return this.run({ prompt: message });
  }

  private configHealth(): ProviderHealth {
    const baseUrl = (this.config.baseUrl ?? DEFAULT_BASE).trim();
    const extra = this.meta();
    if (!baseUrl) {
      return {
        id: this.id,
        type: "vllm",
        ready: false,
        reason: `Set backends.${this.id}.baseUrl (typical ${DEFAULT_BASE})`,
        capabilities: this.capabilities,
        writesLocalFiles: false,
        ...extra,
      };
    }
    const reason =
      this.config.probe === false
        ? `Configured at ${normalizeBaseUrl(baseUrl)} (probe disabled; API key optional)`
        : `Configured at ${normalizeBaseUrl(baseUrl)} (will probe /models; API key optional)`;
    return readyHealth(this.id, "vllm", this.capabilities, { ...extra, reason });
  }

  private meta() {
    return {
      secretNames: envNamesForBackend(this.id, this.config),
      needsKey: false,
      baseUrl: normalizeBaseUrl(this.config.baseUrl ?? DEFAULT_BASE),
      model: this.config.model,
    };
  }
}
