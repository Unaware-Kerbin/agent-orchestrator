import type { AgentProvider, HttpBackendConfig, ProviderHealth, ProviderRunRequest, ProviderRunResult } from "../types.js";
import { extractHttpText, readyHealth } from "./util.js";

export class HttpProvider implements AgentProvider {
  readonly type = "http" as const;
  readonly capabilities = ["text", "follow_up"];

  constructor(
    readonly id: string,
    private readonly config: HttpBackendConfig,
  ) {}

  health(): ProviderHealth {
    if (!this.config.url) {
      return {
        id: this.id,
        type: "http",
        ready: false,
        reason: "url is empty",
        capabilities: this.capabilities,
        writesLocalFiles: false,
      };
    }
    return readyHealth(this.id, "http", this.capabilities);
  }

  async run(request: ProviderRunRequest): Promise<ProviderRunResult> {
    const started = Date.now();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.config.timeoutMs ?? 120_000);
    try {
      const response = await fetch(this.config.url, {
        method: this.config.method ?? "POST",
        headers: {
          "content-type": "application/json",
          ...this.config.headers,
        },
        body: JSON.stringify({
          prompt: request.prompt,
          system: request.system,
          history: request.history ?? [],
          metadata: {
            backend: this.id,
            cwd: request.cwd,
            model: request.model,
          },
        }),
        signal: controller.signal,
      });
      const contentType = response.headers.get("content-type") ?? "";
      const payload: unknown = contentType.includes("json")
        ? await response.json()
        : await response.text();
      if (!response.ok) {
        return {
          status: "error",
          text: "",
          error: `HTTP agent ${response.status}: ${extractHttpText(payload)}`,
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
    } finally {
      clearTimeout(timeout);
    }
  }

  followUp(agentId: string, message: string): Promise<ProviderRunResult> {
    void agentId;
    return this.run({ prompt: message });
  }
}
