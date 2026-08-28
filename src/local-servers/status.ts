import { parseModelId } from "../identity.js";
import { LOCAL_OPENAI_DUMMY_KEY } from "../providers/keys.js";
import { isUnreachableError } from "../providers/keys.js";
import { which } from "../platform.js";
import { DEFAULT_LLAMACPP_BASE, DEFAULT_OLLAMA_BASE, loopbackOrigin, normalizeLoopbackOpenAiUrl } from "./loopback.js";

export { DEFAULT_LLAMACPP_BASE, DEFAULT_OLLAMA_BASE } from "./loopback.js";

export type FetchLike = typeof fetch;

export type LocalServerKind = "ollama" | "llamacpp";

export interface LocalServerStatus {
  kind: LocalServerKind;
  running: boolean;
  ready: boolean;
  baseUrl: string;
  origin: string;
  models: string[];
  reason: string;
}

export interface LlamaServerBinary {
  path?: string;
}

export function llamaServerOnPath(whichFn: (cmd: string) => string | undefined = which): string | undefined {
  return whichFn("llama-server");
}

export function ollamaOnPath(whichFn: (cmd: string) => string | undefined = which): string | undefined {
  return whichFn("ollama");
}

function acceptedModelId(raw: string): string | undefined {
  try {
    return parseModelId(raw);
  } catch {
    return undefined;
  }
}

function openaiModelIds(payload: unknown): string[] {
  if (!payload || typeof payload !== "object") return [];
  const data = (payload as { data?: unknown }).data;
  if (!Array.isArray(data)) return [];
  const ids: string[] = [];
  for (const row of data) {
    if (row && typeof row === "object" && typeof (row as { id?: unknown }).id === "string") {
      const id = acceptedModelId((row as { id: string }).id);
      if (id && !ids.includes(id)) ids.push(id);
    }
  }
  return ids;
}

function ollamaTagNames(payload: unknown): string[] {
  if (!payload || typeof payload !== "object") return [];
  const models = (payload as { models?: unknown }).models;
  if (!Array.isArray(models)) return [];
  const names: string[] = [];
  for (const row of models) {
    if (!row || typeof row !== "object") continue;
    const rec = row as { name?: unknown; model?: unknown };
    const raw = typeof rec.name === "string" ? rec.name : typeof rec.model === "string" ? rec.model : "";
    const name = acceptedModelId(raw);
    if (name && !names.includes(name)) names.push(name);
  }
  return names;
}

async function getJson(
  url: string,
  fetchFn: FetchLike,
  timeoutMs: number,
  apiKey: string,
): Promise<{ ok: boolean; status: number; payload: unknown }> {
  const response = await fetchFn(url, {
    method: "GET",
    headers: {
      authorization: `Bearer ${apiKey}`,
      accept: "application/json",
    },
    signal: AbortSignal.timeout(timeoutMs),
  });
  const payload: unknown = await response.json().catch(() => undefined);
  return { ok: response.ok, status: response.status, payload };
}

function downStatus(
  kind: LocalServerKind,
  baseUrl: string,
  origin: string,
  error: unknown,
  timeoutMs: number,
): LocalServerStatus {
  const timedOut = error instanceof Error && /timeout|aborted/i.test(error.message);
  const label = kind === "ollama" ? "Ollama" : "llama.cpp";
  const reason = timedOut
    ? `${label} not reachable at ${baseUrl} (timeout)`
    : isUnreachableError(error)
      ? `${label} not running at ${baseUrl}`
      : `${label} not reachable at ${baseUrl}: ${error instanceof Error ? error.message : String(error)}`;
  void timeoutMs;
  return { kind, running: false, ready: false, baseUrl, origin, models: [], reason };
}

export async function probeOllama(options: {
  baseUrl?: string;
  fetchFn?: FetchLike;
  timeoutMs?: number;
  apiKey?: string;
} = {}): Promise<LocalServerStatus> {
  const label = "Ollama";
  const baseUrl = normalizeLoopbackOpenAiUrl(options.baseUrl?.trim() || DEFAULT_OLLAMA_BASE, label);
  const origin = loopbackOrigin(baseUrl, label);
  const fetchFn = options.fetchFn ?? fetch;
  const timeoutMs = options.timeoutMs ?? 800;
  const apiKey = options.apiKey?.trim() || "ollama";
  try {
    const tags = await getJson(`${origin}/api/tags`, fetchFn, timeoutMs, apiKey);
    if (tags.ok || tags.status < 500) {
      const models = ollamaTagNames(tags.payload);
      return {
        kind: "ollama",
        running: true,
        ready: true,
        baseUrl,
        origin,
        models,
        reason:
          models.length > 0
            ? `Ollama reachable at ${origin} (${models.length} model${models.length === 1 ? "" : "s"})`
            : `Ollama reachable at ${origin} (no tags yet — run ollama pull)`,
      };
    }
  } catch (error) {
    if (isUnreachableError(error)) {
      return downStatus("ollama", baseUrl, origin, error, timeoutMs);
    }
  }
  try {
    const modelsProbe = await getJson(`${baseUrl}/models`, fetchFn, timeoutMs, apiKey);
    const models = openaiModelIds(modelsProbe.payload);
    return {
      kind: "ollama",
      running: true,
      ready: true,
      baseUrl,
      origin,
      models,
      reason: `Ollama OpenAI API reachable at ${baseUrl} (HTTP ${modelsProbe.status})`,
    };
  } catch (error) {
    return downStatus("ollama", baseUrl, origin, error, timeoutMs);
  }
}

export async function probeLlamaCpp(options: {
  baseUrl?: string;
  fetchFn?: FetchLike;
  timeoutMs?: number;
  apiKey?: string;
} = {}): Promise<LocalServerStatus> {
  const label = "llama.cpp";
  const baseUrl = normalizeLoopbackOpenAiUrl(options.baseUrl?.trim() || DEFAULT_LLAMACPP_BASE, label);
  const origin = loopbackOrigin(baseUrl, label);
  const fetchFn = options.fetchFn ?? fetch;
  const timeoutMs = options.timeoutMs ?? 800;
  const apiKey = options.apiKey?.trim() || LOCAL_OPENAI_DUMMY_KEY;
  try {
    const modelsProbe = await getJson(`${baseUrl}/models`, fetchFn, timeoutMs, apiKey);
    const models = openaiModelIds(modelsProbe.payload);
    return {
      kind: "llamacpp",
      running: true,
      ready: true,
      baseUrl,
      origin,
      models,
      reason: `llama.cpp reachable at ${baseUrl} (HTTP ${modelsProbe.status})`,
    };
  } catch (error) {
    if (isUnreachableError(error)) {
      return downStatus("llamacpp", baseUrl, origin, error, timeoutMs);
    }
    try {
      const health = await fetchFn(`${origin}/health`, {
        method: "GET",
        signal: AbortSignal.timeout(timeoutMs),
      });
      void health.arrayBuffer().catch(() => undefined);
      return {
        kind: "llamacpp",
        running: true,
        ready: health.ok,
        baseUrl,
        origin,
        models: [],
        reason: `llama.cpp health at ${origin}/health (HTTP ${health.status})`,
      };
    } catch {
      return downStatus("llamacpp", baseUrl, origin, error, timeoutMs);
    }
  }
}
