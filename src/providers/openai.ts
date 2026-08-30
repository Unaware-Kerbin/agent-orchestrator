import type {
  AgentProvider,
  LlamaCppBackendConfig,
  OpenAIBackendConfig,
  OllamaBackendConfig,
  ProviderHealth,
  ProviderRunRequest,
  ProviderRunResult,
  VllmBackendConfig,
} from "../types.js";
import {
  DEFAULT_GEMINI_MODEL,
  formatGeminiChatError,
  geminiModelChoices,
  isGeminiOpenAiConfig,
  isRetiredGeminiModel,
  listGeminiOpenAiModels,
  resolveGeminiChatModel,
} from "./gemini.js";
import {
  backendNeedsKey,
  envNamesForBackend,
  isLocalOpenAiUrl,
  isUnreachableError,
  localUnreachableReason,
  normalizeBaseUrl,
  VLLM_LOCAL_DUMMY_KEY,
} from "./keys.js";
import { extractHttpText, missingKeyHealth, readyHealth, secretFrom } from "./util.js";

function chatModelFor(id: string, config: OpenAiCompatConfig, override?: string, liveIds?: string[]): string {
  if (config.type === "openai" && isGeminiOpenAiConfig(id, config)) {
    return resolveGeminiChatModel(config.model, override, liveIds);
  }
  return override ?? config.model;
}

export type OpenAiCompatConfig =
  | OpenAIBackendConfig
  | VllmBackendConfig
  | OllamaBackendConfig
  | LlamaCppBackendConfig;

async function readOpenAiStream(response: Response, onDelta: (delta: string) => void): Promise<string> {
  const reader = response.body?.getReader();
  if (!reader) return "";
  const decoder = new TextDecoder();
  let buffer = "";
  let full = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const data = trimmed.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      try {
        const json: unknown = JSON.parse(data);
        const delta = streamDeltaText(json);
        if (delta) {
          full += delta;
          onDelta(delta);
        }
      } catch {
        // ignore incomplete JSON frames
      }
    }
  }
  return full;
}

function streamDeltaText(payload: unknown): string {
  if (!payload || typeof payload !== "object") return "";
  const choices = (payload as { choices?: unknown }).choices;
  if (!Array.isArray(choices) || !choices[0] || typeof choices[0] !== "object") return "";
  const choice = choices[0] as { delta?: { content?: unknown }; message?: { content?: unknown } };
  if (typeof choice.delta?.content === "string") return choice.delta.content;
  if (typeof choice.message?.content === "string") return choice.message.content;
  return "";
}

export async function runOpenAiChat(
  id: string,
  config: OpenAiCompatConfig,
  request: ProviderRunRequest,
  options: { label: string; defaultBaseUrl: string; optionalKey?: boolean; liveGeminiIds?: string[] },
): Promise<ProviderRunResult> {
  const started = Date.now();
  const fallbacks = envNamesForBackend(id, config);
  const apiKey =
    secretFrom(config, fallbacks) ?? config.apiKey ?? (options.optionalKey ? VLLM_LOCAL_DUMMY_KEY : undefined);
  if (!apiKey) {
    return {
      status: "error",
      text: "",
      error: `Set ${fallbacks.join(" or ")} to enable this backend`,
      durationMs: 0,
    };
  }
  const baseUrl = normalizeBaseUrl(config.baseUrl ?? options.defaultBaseUrl);
  const messages: Array<{ role: string; content: string }> = [];
  if (request.system?.trim()) messages.push({ role: "system", content: request.system });
  for (const turn of request.history ?? []) {
    if (typeof turn.content === "string" && turn.content.trim()) messages.push(turn);
  }
  messages.push({ role: "user", content: request.prompt?.trim() ? request.prompt : "(empty operator turn)" });
  const gemini = config.type === "openai" && isGeminiOpenAiConfig(id, config);
  const model = chatModelFor(id, config, request.model, gemini ? options.liveGeminiIds : undefined);
  const stream = typeof request.onDelta === "function";
  const body: Record<string, unknown> = { model, messages };
  if (stream) body.stream = true;
  const timeoutMs = request.timeoutMs ?? 30_000;
  const deadline = started + timeoutMs;
  const payloadJson = JSON.stringify(body);

  const postOnce = (remainingMs: number) =>
    fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${apiKey}`,
      },
      body: payloadJson,
      signal: AbortSignal.timeout(Math.max(1, remainingMs)),
    });

  try {
    let response = await postOnce(timeoutMs);
    if (response.status === 429) {
      const waitMs = Math.min(800, Math.max(0, deadline - Date.now() - 1));
      if (waitMs > 0) await new Promise((r) => setTimeout(r, waitMs));
      const remaining = deadline - Date.now();
      if (remaining > 0) response = await postOnce(remaining);
    }
    const contentType = response.headers.get("content-type") ?? "";
    if (!response.ok) {
      const payload: unknown = await response.json().catch(() => ({ error: response.statusText }));
      const extracted = extractHttpText(payload);
      if (response.status === 429) {
        return {
          status: "error",
          text: "",
          error: gemini
            ? formatGeminiChatError(429, extracted, model)
            : `${options.label} rate-limited (429) — skipped so other speakers can finish.`,
          durationMs: Date.now() - started,
        };
      }
      return {
        status: "error",
        text: "",
        error: gemini
          ? formatGeminiChatError(response.status, extracted, model)
          : `${options.label} error ${response.status}: ${extracted}`,
        durationMs: Date.now() - started,
      };
    }
    if (stream && (contentType.includes("text/event-stream") || contentType.includes("ndjson") || !contentType.includes("json"))) {
      const text = await readOpenAiStream(response, request.onDelta!);
      return {
        status: "finished",
        text,
        durationMs: Date.now() - started,
      };
    }
    const payload: unknown = await response.json().catch(() => ({ error: response.statusText }));
    const text = extractHttpText(payload);
    if (stream && text) request.onDelta?.(text);
    return {
      status: "finished",
      text,
      durationMs: Date.now() - started,
    };
  } catch (error) {
    const unreachable = localUnreachableReason(options.label, baseUrl, error);
    const fallback = error instanceof Error ? error.message : String(error);
    return {
      status: "error",
      text: "",
      error: unreachable ?? (isUnreachableError(error) ? `${options.label} not reachable at ${baseUrl}` : fallback),
      durationMs: Date.now() - started,
    };
  }
}

export class OpenAIProvider implements AgentProvider {
  readonly type = "openai" as const;
  readonly capabilities = ["text", "follow_up"];
  private geminiList?: { at: number; ids: string[] };

  constructor(
    readonly id: string,
    private readonly config: OpenAIBackendConfig,
  ) {}

  health(): ProviderHealth {
    return this.healthFromList(this.geminiList?.ids);
  }

  async probe(): Promise<ProviderHealth> {
    if (!isGeminiOpenAiConfig(this.id, this.config)) return this.health();
    const secretNames = envNamesForBackend(this.id, this.config);
    const key = secretFrom(this.config, secretNames);
    if (!key) return this.health();
    if (!this.geminiList || Date.now() - this.geminiList.at > 60_000) {
      try {
        const ids = await listGeminiOpenAiModels(
          this.config.baseUrl ?? "https://generativelanguage.googleapis.com/v1beta/openai",
          key,
        );
        this.geminiList = { at: Date.now(), ids };
      } catch {
        // Keep the 2026 catalog when ListModels is unavailable.
      }
    }
    return this.healthFromList(this.geminiList?.ids);
  }

  run(request: ProviderRunRequest): Promise<ProviderRunResult> {
    const localish = isLocalOpenAiUrl(this.config.baseUrl) || this.config.apiKey === "ollama";
    return runOpenAiChat(this.id, this.config, request, {
      label: "OpenAI-compatible",
      defaultBaseUrl: "https://api.openai.com/v1",
      optionalKey: localish,
      liveGeminiIds: this.geminiList?.ids,
    });
  }

  followUp(agentId: string, message: string): Promise<ProviderRunResult> {
    void agentId;
    return this.run({ prompt: message });
  }

  private healthFromList(liveIds?: string[]): ProviderHealth {
    const secretNames = envNamesForBackend(this.id, this.config);
    const gemini = isGeminiOpenAiConfig(this.id, this.config);
    const sentModel = gemini ? resolveGeminiChatModel(this.config.model, undefined, liveIds) : this.config.model;
    const extra = {
      secretNames,
      needsKey: backendNeedsKey(this.id, this.config),
      baseUrl: this.config.baseUrl ?? "https://api.openai.com/v1",
      model: sentModel,
      modelChoices: gemini ? geminiModelChoices(liveIds) : undefined,
    };
    const localish = isLocalOpenAiUrl(this.config.baseUrl) || this.config.apiKey === "ollama";
    if (localish) {
      return readyHealth(this.id, "openai", this.capabilities, {
        ...extra,
        needsKey: false,
        reason: `Local OpenAI-compatible API at ${extra.baseUrl} (API key optional)`,
      });
    }
    const key = secretFrom(this.config, secretNames);
    if (!key) {
      return missingKeyHealth(
        this.id,
        "openai",
        secretNames.join(" or ") || "OPENAI_API_KEY",
        this.capabilities,
        extra,
      );
    }
    let reason = `API key present via ${secretNames.join(" / ")} (masked; never displayed).`;
    if (gemini && isRetiredGeminiModel(this.config.model) && sentModel !== this.config.model) {
      reason += ` Configured ${this.config.model} is retired; sending ${sentModel} (e.g. ${DEFAULT_GEMINI_MODEL}).`;
    }
    return readyHealth(this.id, "openai", this.capabilities, { ...extra, reason });
  }
}
