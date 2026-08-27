import type { BackendConfig, OpenAIBackendConfig } from "../types.js";

const ENV_NAME_RE = /^[A-Z][A-Z0-9_]*$/;

export const GEMINI_KEY_ENV_NAMES = [
  "GEMINI_API_KEY",
  "GOOGLE_API_KEY",
  "GOOGLE_GENERATIVE_AI_API_KEY",
] as const;

/** Bearer value for loopback vLLM when the OpenAI client requires a token. Not a real secret. */
export const VLLM_LOCAL_DUMMY_KEY = "sk-local";

export function isEnvVarName(name: string | undefined): name is string {
  return Boolean(name && ENV_NAME_RE.test(name));
}

export function uniqueNames(names: Array<string | undefined>): string[] {
  const out: string[] = [];
  for (const name of names) {
    if (!isEnvVarName(name) || out.includes(name)) continue;
    out.push(name);
  }
  return out;
}

export function isLocalOpenAiUrl(baseUrl?: string): boolean {
  if (!baseUrl) return false;
  try {
    const url = new URL(baseUrl);
    return url.hostname === "127.0.0.1" || url.hostname === "localhost" || url.hostname === "::1";
  } catch {
    return /127\.0\.0\.1|localhost|\[::1\]/.test(baseUrl);
  }
}

export function isGeminiBackend(id: string, config: OpenAIBackendConfig): boolean {
  const base = (config.baseUrl ?? "").toLowerCase();
  const env = (config.apiKeyEnv ?? "").toUpperCase();
  return (
    /^gemini\b/i.test(id) ||
    base.includes("generativelanguage.googleapis.com") ||
    base.includes("aiplatform.googleapis.com") ||
    GEMINI_KEY_ENV_NAMES.includes(env as (typeof GEMINI_KEY_ENV_NAMES)[number])
  );
}

export function isOpenRouterBackend(id: string, config: OpenAIBackendConfig): boolean {
  const base = (config.baseUrl ?? "").toLowerCase();
  return /openrouter/i.test(id) || base.includes("openrouter.ai");
}

export function envNamesForBackend(id: string, config: BackendConfig): string[] {
  switch (config.type) {
    case "cursor":
      return ["CURSOR_API_KEY"];
    case "anthropic":
      return uniqueNames([config.apiKeyEnv, "ANTHROPIC_API_KEY"]);
    case "vllm":
      return uniqueNames([config.apiKeyEnv, "VLLM_API_KEY"]);
    case "openai":
      if (isGeminiBackend(id, config)) {
        return uniqueNames([config.apiKeyEnv, ...GEMINI_KEY_ENV_NAMES]);
      }
      if (isOpenRouterBackend(id, config)) {
        return uniqueNames([config.apiKeyEnv, "OPENROUTER_API_KEY", "OPENAI_API_KEY"]);
      }
      return uniqueNames([config.apiKeyEnv, "OPENAI_API_KEY"]);
    case "http":
      return [];
  }
}

export function backendNeedsKey(id: string, config: BackendConfig): boolean {
  if (config.type === "cursor") return true;
  if (config.type === "anthropic") return true;
  if (config.type === "http") return false;
  if (config.type === "vllm") return false;
  if (config.type === "openai") {
    if (config.apiKey === "ollama") return false;
    if (isLocalOpenAiUrl(config.baseUrl)) return false;
    return true;
  }
  return false;
}

export function defaultBaseUrl(config: BackendConfig): string | undefined {
  if (config.type === "openai") return config.baseUrl ?? "https://api.openai.com/v1";
  if (config.type === "vllm") return config.baseUrl ?? "http://127.0.0.1:8000/v1";
  if (config.type === "anthropic") return config.baseUrl;
  if (config.type === "http") return config.url;
  return undefined;
}

export function backendModel(config: BackendConfig): string | undefined {
  if (config.type === "cursor" || config.type === "openai" || config.type === "anthropic" || config.type === "vllm") {
    return config.model;
  }
  return undefined;
}

export function normalizeBaseUrl(url: string): string {
  return url.replace(/\/$/, "");
}

export function vllmUnreachableReason(baseUrl: string, error: unknown): string | undefined {
  if (isUnreachableError(error)) {
    return `vLLM not running at ${baseUrl}`;
  }
  return undefined;
}

export function isUnreachableError(error: unknown): boolean {
  const codes = new Set(["ECONNREFUSED", "ENOTFOUND", "EHOSTUNREACH", "ECONNRESET", "EAI_AGAIN"]);
  let current: unknown = error;
  for (let i = 0; i < 5 && current; i++) {
    if (typeof current === "string") {
      if (codes.has(current) || /ECONNREFUSED|ENOTFOUND|EHOSTUNREACH|fetch failed/i.test(current)) return true;
      return false;
    }
    if (typeof current !== "object") return false;
    const rec = current as { code?: unknown; cause?: unknown; message?: unknown; name?: unknown };
    if (typeof rec.code === "string" && codes.has(rec.code)) return true;
    if (typeof rec.message === "string" && /ECONNREFUSED|ENOTFOUND|EHOSTUNREACH|ECONNRESET/i.test(rec.message)) {
      return true;
    }
    current = rec.cause;
  }
  return false;
}
