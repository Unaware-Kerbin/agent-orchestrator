import type {
  AgentProvider,
  LlamaCppBackendConfig,
  OllamaBackendConfig,
  ProviderHealth,
  ProviderRunRequest,
  ProviderRunResult,
} from "../types.js";
import { probeLlamaCpp, probeOllama } from "../local-servers/status.js";
import { DEFAULT_LLAMACPP_BASE, DEFAULT_OLLAMA_BASE } from "../local-servers/status.js";
import { normalizeLoopbackOpenAiUrl } from "../local-servers/loopback.js";
import { envNamesForBackend, LOCAL_OPENAI_DUMMY_KEY } from "./keys.js";
import { runOpenAiChat } from "./openai.js";
import { readyHealth } from "./util.js";

export type LocalOpenAiBackendConfig = OllamaBackendConfig | LlamaCppBackendConfig;

function defaultsFor(config: LocalOpenAiBackendConfig): { label: string; defaultBase: string } {
  if (config.type === "ollama") return { label: "Ollama", defaultBase: DEFAULT_OLLAMA_BASE };
  return { label: "llama.cpp", defaultBase: DEFAULT_LLAMACPP_BASE };
}

export class LocalOpenAiCompatProvider implements AgentProvider {
  readonly type: "ollama" | "llamacpp";
  readonly capabilities = ["text", "follow_up"];
  private lastProbe: { at: number; health: ProviderHealth } | undefined;

  constructor(
    readonly id: string,
    private readonly config: LocalOpenAiBackendConfig,
  ) {
    this.type = config.type;
  }

  health(): ProviderHealth {
    if (this.lastProbe && Date.now() - this.lastProbe.at < 15_000) {
      return this.lastProbe.health;
    }
    return this.configHealth();
  }

  async probe(): Promise<ProviderHealth> {
    const snapshot = this.configHealth();
    if (!snapshot.ready || this.config.probe === false) {
      this.lastProbe = { at: Date.now(), health: snapshot };
      return snapshot;
    }
    const { defaultBase } = defaultsFor(this.config);
    const timeoutMs = this.config.probeTimeoutMs ?? 800;
    const apiKey = this.config.apiKey?.trim() || LOCAL_OPENAI_DUMMY_KEY;
    const status =
      this.config.type === "ollama"
        ? await probeOllama({
            baseUrl: this.config.baseUrl ?? defaultBase,
            timeoutMs,
            apiKey,
          })
        : await probeLlamaCpp({
            baseUrl: this.config.baseUrl ?? defaultBase,
            timeoutMs,
            apiKey,
          });
    const health: ProviderHealth = {
      id: this.id,
      type: this.type,
      ready: status.ready,
      reason: status.reason,
      capabilities: this.capabilities,
      writesLocalFiles: false,
      ...this.meta(),
      model: this.config.model,
      modelChoices: status.models.length ? status.models : undefined,
    };
    this.lastProbe = { at: Date.now(), health };
    return health;
  }

  run(request: ProviderRunRequest): Promise<ProviderRunResult> {
    const { label, defaultBase } = defaultsFor(this.config);
    return runOpenAiChat(this.id, this.config, request, {
      label,
      defaultBaseUrl: defaultBase,
      optionalKey: true,
    });
  }

  followUp(agentId: string, message: string): Promise<ProviderRunResult> {
    void agentId;
    return this.run({ prompt: message });
  }

  private configHealth(): ProviderHealth {
    const { label, defaultBase } = defaultsFor(this.config);
    const extra = this.meta();
    const raw = (this.config.baseUrl ?? defaultBase).trim();
    if (!raw) {
      return {
        id: this.id,
        type: this.type,
        ready: false,
        reason: `Set backends.${this.id}.baseUrl (typical ${defaultBase})`,
        capabilities: this.capabilities,
        writesLocalFiles: false,
        ...extra,
      };
    }
    const baseUrl = extra.baseUrl ?? raw;
    const reason =
      this.config.probe === false
        ? `Configured at ${baseUrl} (probe disabled; API key optional)`
        : `Configured at ${baseUrl} (will probe; API key optional)`;
    return readyHealth(this.id, this.type, this.capabilities, { ...extra, reason: `${label}: ${reason}` });
  }

  private meta() {
    const { defaultBase } = defaultsFor(this.config);
    return {
      secretNames: envNamesForBackend(this.id, this.config),
      needsKey: false,
      baseUrl: normalizeLoopbackOpenAiUrl(this.config.baseUrl ?? defaultBase, this.config.type),
      model: this.config.model,
    };
  }
}

export class OllamaProvider extends LocalOpenAiCompatProvider {
  constructor(id: string, config: OllamaBackendConfig) {
    super(id, config);
  }
}

export class LlamaCppProvider extends LocalOpenAiCompatProvider {
  constructor(id: string, config: LlamaCppBackendConfig) {
    super(id, config);
  }
}
