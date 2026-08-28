import type { AgentProvider, AnthropicBackendConfig, ProviderHealth, ProviderRunRequest, ProviderRunResult } from "../types.js";
import { envNamesForBackend } from "./keys.js";
import { extractHttpText, missingKeyHealth, readyHealth, secretFrom } from "./util.js";

export class AnthropicProvider implements AgentProvider {
  readonly type = "anthropic" as const;
  readonly capabilities = ["text", "follow_up"];

  constructor(
    readonly id: string,
    private readonly config: AnthropicBackendConfig,
  ) {}

  health(): ProviderHealth {
    const secretNames = envNamesForBackend(this.id, this.config);
    if (!secretFrom(this.config, secretNames)) {
      return missingKeyHealth(
        this.id,
        "anthropic",
        secretNames.join(" or ") || "ANTHROPIC_API_KEY",
        this.capabilities,
        { secretNames, needsKey: true, model: this.config.model },
      );
    }
    return readyHealth(this.id, "anthropic", this.capabilities, {
      secretNames,
      needsKey: true,
      model: this.config.model,
      reason: "API key present (masked; never displayed).",
    });
  }

  async run(request: ProviderRunRequest): Promise<ProviderRunResult> {
    const started = Date.now();
    const apiKey = secretFrom(this.config, envNamesForBackend(this.id, this.config));
    if (!apiKey) {
      return { status: "error", text: "", error: "ANTHROPIC_API_KEY is not set", durationMs: 0 };
    }
    const baseUrl = (this.config.baseUrl ?? "https://api.anthropic.com").replace(/\/$/, "");
    const messages = [...(request.history ?? []), { role: "user" as const, content: request.prompt }];

    const timeoutMs = request.timeoutMs ?? 30_000;
    try {
      const response = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": apiKey,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: request.model ?? this.config.model,
          max_tokens: this.config.maxTokens ?? 8192,
          system: request.system,
          messages,
        }),
        signal: AbortSignal.timeout(timeoutMs),
      });
      const payload: unknown = await response.json().catch(() => ({ error: response.statusText }));
      if (!response.ok) {
        return {
          status: "error",
          text: "",
          error: `Anthropic error ${response.status}: ${extractHttpText(payload)}`,
          durationMs: Date.now() - started,
        };
      }
      return {
        status: "finished",
        text: extractHttpText(payload),
        durationMs: Date.now() - started,
      };
    } catch (error) {
      return {
        status: "error",
        text: "",
        error: error instanceof Error ? error.message : String(error),
        durationMs: Date.now() - started,
      };
    }
  }

  followUp(agentId: string, message: string): Promise<ProviderRunResult> {
    void agentId;
    return this.run({ prompt: message });
  }
}
