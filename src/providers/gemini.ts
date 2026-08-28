import type { OpenAIBackendConfig } from "../types.js";
import { isGeminiBackend, normalizeBaseUrl } from "./keys.js";

/**
 * Google’s OpenAI-compat endpoint is still `v1beta/openai`.
 * The shim looks the id up as `models/<id>` on generateContent (`v1main` in 404s).
 * Send a **bare** id such as `gemini-3.6-flash` — never `models/models/…`.
 */
export const DEFAULT_GEMINI_MODEL = "gemini-3.6-flash";
export const GEMINI_OPENAI_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai";

/** Documented/live Aug 2026 chat ids (OpenAI-compat). Not 1.5 / 2.0 / 2.5. */
export const KNOWN_GEMINI_MODELS = [
  "gemini-3.6-flash",
  "gemini-3.7-flash",
  "gemini-3.5-flash",
  "gemini-3.5-flash-lite",
  "gemini-3.1-flash-lite",
  "gemini-3.1-pro-preview",
  "gemini-flash-latest",
  "gemini-3-flash-preview",
] as const;

/** Retired or 404-for-new-users on this OpenAI-compat path. Do not offer as defaults. */
export const RETIRED_GEMINI_MODELS = new Set([
  "gemini-1.5-flash",
  "gemini-1.5-pro",
  "gemini-1.5-flash-8b",
  "gemini-1.5-flash-002",
  "gemini-1.5-pro-002",
  "gemini-2.0-flash",
  "gemini-2.0-flash-001",
  "gemini-2.0-flash-lite",
  "gemini-2.0-flash-lite-001",
  "gemini-2.0-pro",
  "gemini-2.5-flash",
  "gemini-2.5-pro",
  "gemini-2.5-flash-lite",
  "gemini-3.6-pro",
]);

export const RETIRED_GEMINI_REPLACEMENTS: Record<string, string> = {
  "gemini-1.5-flash": DEFAULT_GEMINI_MODEL,
  "gemini-1.5-flash-8b": DEFAULT_GEMINI_MODEL,
  "gemini-1.5-flash-002": DEFAULT_GEMINI_MODEL,
  "gemini-1.5-pro": "gemini-3.1-pro-preview",
  "gemini-1.5-pro-002": "gemini-3.1-pro-preview",
  "gemini-2.0-flash": DEFAULT_GEMINI_MODEL,
  "gemini-2.0-flash-001": DEFAULT_GEMINI_MODEL,
  "gemini-2.0-flash-lite": "gemini-3.1-flash-lite",
  "gemini-2.0-flash-lite-001": "gemini-3.1-flash-lite",
  "gemini-2.0-pro": "gemini-3.1-pro-preview",
  "gemini-2.5-flash": DEFAULT_GEMINI_MODEL,
  "gemini-2.5-pro": "gemini-3.1-pro-preview",
  "gemini-2.5-flash-lite": "gemini-3.5-flash-lite",
  "gemini-3.6-pro": "gemini-3.1-pro-preview",
};

export const GEMINI_ONE_ID_ERROR = "use one model id, not a list";

const GEMINI_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const NON_CHAT_GEMINI_RE =
  /image|tts|embed|veo|lyria|live|transcribe|robotics|computer-use|deep-research|aqa|antigravity|native-audio|banana/i;

export function stripGeminiModelPrefix(raw: string): string {
  let text = raw.trim();
  while (/^models\//i.test(text)) {
    text = text.slice("models/".length);
  }
  return text;
}

export function geminiLooksLikeList(raw: string): boolean {
  const text = raw.trim();
  if (!text) return true;
  return /[\s/#,]/.test(stripGeminiModelPrefix(text)) || /\|/.test(text);
}

/**
 * Parse a Gemini OpenAI-compat model id.
 * Strips every leading `models/` (native GenerateContent / ListModels form).
 * Rejects whitespace, leftover `/`, `#`, or multiple names.
 */
export function parseGeminiModelId(raw: string): string {
  if (typeof raw !== "string" || !raw.trim()) {
    throw new Error(`Gemini model is empty; ${GEMINI_ONE_ID_ERROR} (e.g. ${DEFAULT_GEMINI_MODEL})`);
  }
  const stripped = stripGeminiModelPrefix(raw);
  if (geminiLooksLikeList(stripped) || stripped.includes("/") || /\s/.test(stripped) || stripped.includes("#")) {
    throw new Error(`Gemini model: ${GEMINI_ONE_ID_ERROR} (e.g. ${DEFAULT_GEMINI_MODEL})`);
  }
  if (/^composer-/i.test(stripped)) {
    throw new Error(`Gemini model: ${GEMINI_ONE_ID_ERROR} (e.g. ${DEFAULT_GEMINI_MODEL})`);
  }
  if (!GEMINI_ID_RE.test(stripped)) {
    throw new Error(`Gemini model: ${GEMINI_ONE_ID_ERROR} (e.g. ${DEFAULT_GEMINI_MODEL})`);
  }
  return stripped;
}

export function isRetiredGeminiModel(id: string): boolean {
  const bare = stripGeminiModelPrefix(id);
  if (RETIRED_GEMINI_MODELS.has(bare)) return true;
  return /^gemini-1\.5($|-)/.test(bare) || /^gemini-2\.0($|-)/.test(bare);
}

export function replacementForRetiredGemini(id: string): string {
  const bare = stripGeminiModelPrefix(id);
  return RETIRED_GEMINI_REPLACEMENTS[bare] ?? DEFAULT_GEMINI_MODEL;
}

/** Parse + remap retired ids so YAML/GUI never keep 1.5/2.0/2.5 as the saved default. */
export function normalizeGeminiConfigModel(raw: string): string {
  const id = parseGeminiModelId(raw);
  return isRetiredGeminiModel(id) ? replacementForRetiredGemini(id) : id;
}

export function isGeminiChatModelId(id: string): boolean {
  const bare = stripGeminiModelPrefix(id);
  if (!GEMINI_ID_RE.test(bare)) return false;
  if (!/^(gemini|gemma)-/i.test(bare)) return false;
  if (NON_CHAT_GEMINI_RE.test(bare)) return false;
  if (isRetiredGeminiModel(bare)) return false;
  return true;
}

export function parseGeminiIdsFromListPayload(payload: unknown): string[] {
  const ids: string[] = [];
  const push = (value: unknown) => {
    if (typeof value !== "string" || !value.trim()) return;
    try {
      ids.push(parseGeminiModelId(value));
    } catch {
      const stripped = stripGeminiModelPrefix(value);
      if (stripped) ids.push(stripped);
    }
  };
  if (!payload || typeof payload !== "object") return ids;
  const rec = payload as Record<string, unknown>;
  if (Array.isArray(rec.data)) {
    for (const row of rec.data) {
      if (row && typeof row === "object") {
        const item = row as { id?: unknown; name?: unknown };
        push(item.id ?? item.name);
      }
    }
  }
  if (Array.isArray(rec.models)) {
    for (const row of rec.models) {
      if (row && typeof row === "object") {
        const item = row as { id?: unknown; name?: unknown };
        push(item.name ?? item.id);
      }
    }
  }
  return [...new Set(ids)];
}

export function pickDefaultGeminiModel(listed: string[]): string {
  const chat = [...new Set(listed.map(stripGeminiModelPrefix).filter(isGeminiChatModelId))];
  if (chat.includes(DEFAULT_GEMINI_MODEL)) return DEFAULT_GEMINI_MODEL;
  return chat[0] ?? DEFAULT_GEMINI_MODEL;
}

export function geminiModelChoices(liveIds?: string[]): string[] {
  if (liveIds && liveIds.length > 0) {
    const ids = [...new Set(liveIds.map(stripGeminiModelPrefix).filter(isGeminiChatModelId))];
    ids.sort((a, b) => {
      if (a === DEFAULT_GEMINI_MODEL) return -1;
      if (b === DEFAULT_GEMINI_MODEL) return 1;
      return a.localeCompare(b);
    });
    return ids.length ? ids : [...KNOWN_GEMINI_MODELS];
  }
  return [...KNOWN_GEMINI_MODELS];
}

/** Parse Google 404 copy such as “update your code to use models/gemini-3.6-flash”. */
export function suggestGeminiModelFromError(errorText: string, sentModel?: string): string | undefined {
  const text = errorText.trim();
  const update = text.match(/update your code to use\s+(?:models\/)*([A-Za-z0-9][A-Za-z0-9._-]*)/i);
  if (update?.[1]) return stripGeminiModelPrefix(update[1]);
  const useModels = text.match(/\buse models\/([A-Za-z0-9][A-Za-z0-9._-]*)/i);
  if (useModels?.[1]) return stripGeminiModelPrefix(useModels[1]);
  if (sentModel && isRetiredGeminiModel(sentModel)) return replacementForRetiredGemini(sentModel);
  return undefined;
}

export function formatGeminiChatError(status: number, bodyText: string, sentModel: string): string {
  if (status === 429) {
    return "Gemini rate-limited (429) — skipped so other speakers can finish.";
  }
  const suggestion = suggestGeminiModelFromError(bodyText, sentModel);
  const hint = suggestion
    ? ` Google suggests ${suggestion}. Send a bare id (not models/${suggestion}) on v1beta/openai/chat/completions.`
    : "";
  return `OpenAI-compatible error ${status} for model ${sentModel}: ${bodyText}${bodyText.endsWith(".") ? "" : "."}${hint}`;
}

/**
 * Id to send on `v1beta/openai/chat/completions`.
 * Ignores Cursor defaults like composer-2.5. Remaps retired 1.5/2.0/2.5 ids.
 * If ListModels ran, prefer a listed chat id (gemini-3.6-flash when present).
 */
export function resolveGeminiChatModel(configured: string, override?: string, liveIds?: string[]): string {
  const fromConfig = parseGeminiModelId(configured);
  let requested = fromConfig;
  if (override?.trim()) {
    try {
      const parsed = parseGeminiModelId(override);
      if (!/^composer-/i.test(parsed)) requested = parsed;
    } catch {
      requested = fromConfig;
    }
  }
  const live = liveIds?.map(stripGeminiModelPrefix).filter(Boolean) ?? [];
  if (live.length > 0) {
    if (live.includes(requested) && !isRetiredGeminiModel(requested)) return requested;
    return pickDefaultGeminiModel(live);
  }
  if (isRetiredGeminiModel(requested)) return replacementForRetiredGemini(requested);
  return requested;
}

export function isGeminiOpenAiConfig(
  id: string,
  config: { type?: string; baseUrl?: string; apiKeyEnv?: string; model?: string },
): boolean {
  if (config.type && config.type !== "openai") return false;
  return isGeminiBackend(id, {
    type: "openai",
    model: typeof config.model === "string" && config.model ? config.model : DEFAULT_GEMINI_MODEL,
    baseUrl: config.baseUrl,
    apiKeyEnv: config.apiKeyEnv,
  });
}

export function assertGeminiBackendModel(id: string, config: OpenAIBackendConfig): string {
  if (!isGeminiBackend(id, config)) return config.model;
  return parseGeminiModelId(config.model);
}

export async function listGeminiOpenAiModels(baseUrl: string, apiKey: string, timeoutMs = 8000): Promise<string[]> {
  const url = `${normalizeBaseUrl(baseUrl)}/models`;
  const response = await fetch(url, {
    method: "GET",
    headers: { authorization: `Bearer ${apiKey}` },
    signal: AbortSignal.timeout(timeoutMs),
  });
  const payload: unknown = await response.json().catch(() => undefined);
  if (!response.ok) {
    throw new Error(`Gemini ListModels HTTP ${response.status}`);
  }
  return parseGeminiIdsFromListPayload(payload);
}
