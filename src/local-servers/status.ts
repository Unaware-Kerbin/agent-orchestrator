import { parseModelId } from "../identity.js";
import { LOCAL_OPENAI_DUMMY_KEY } from "../providers/keys.js";
import { isUnreachableError } from "../providers/keys.js";
import { classifyLateInferDevice } from "./gpu-pick.js";
import { findEngineBin, findLateInferBin } from "./bins.js";
import { which } from "../platform.js";
import {
  DEFAULT_LATE_INFER_BASE,
  DEFAULT_LLAMACPP_BASE,
  DEFAULT_OLLAMA_BASE,
  loopbackOrigin,
  normalizeLoopbackOpenAiUrl,
} from "./loopback.js";

export { DEFAULT_LATE_INFER_BASE, DEFAULT_LLAMACPP_BASE, DEFAULT_OLLAMA_BASE } from "./loopback.js";

export type FetchLike = typeof fetch;

export type LocalServerKind = "lateinfer" | "ollama" | "llamacpp";

export interface LocalServerStatus {
  kind: LocalServerKind;
  running: boolean;
  ready: boolean;
  baseUrl: string;
  origin: string;
  models: string[];
  /** Serving model id when known (health or first models[] entry). */
  model?: string;
  reason: string;
  device?: string;
  deviceKind?: string;
  accel?: string;
  weightsInHostRam?: boolean;
  gpuRunning?: boolean;
}

export interface LlamaServerBinary {
  path?: string;
}

export function llamaServerOnPath(whichFn: (cmd: string) => string | undefined = which): string | undefined {
  return findEngineBin("llama-server", whichFn);
}

export function ollamaOnPath(whichFn: (cmd: string) => string | undefined = which): string | undefined {
  return findEngineBin("ollama", whichFn);
}

export function lateInferOnPath(
  whichFn: (cmd: string) => string | undefined = which,
): string | undefined {
  return findLateInferBin(whichFn);
}

function acceptedModelId(raw: string): string | undefined {
  try {
    return parseModelId(raw);
  } catch {
    return undefined;
  }
}

/** OpenAI GET /v1/models JSON (`{ data: [...] }`). HTML or other JSON is not llama.cpp. */
function isOpenAiModelsList(payload: unknown): boolean {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return false;
  return Array.isArray((payload as { data?: unknown }).data);
}

function openaiModelIds(payload: unknown): string[] {
  if (!isOpenAiModelsList(payload)) return [];
  const data = (payload as { data: unknown[] }).data;
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
  const label =
    kind === "lateinfer" ? "Late infer" : kind === "ollama" ? "Ollama" : "llama.cpp";
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
    // HTTP 200 HTML (another process on :8080) is not llama.cpp — require OpenAI models JSON.
    if (!isOpenAiModelsList(modelsProbe.payload)) {
      return {
        kind: "llamacpp",
        running: false,
        ready: false,
        baseUrl,
        origin,
        models: [],
        reason: `llama.cpp not running at ${baseUrl} (/v1/models is not an OpenAI models JSON list)`,
      };
    }
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
        headers: { accept: "application/json" },
        signal: AbortSignal.timeout(timeoutMs),
      });
      const raw = await health.text().catch(() => "");
      let payload: unknown;
      try {
        payload = JSON.parse(raw) as unknown;
      } catch {
        payload = undefined;
      }
      if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
        return downStatus(
          "llamacpp",
          baseUrl,
          origin,
          new Error("/health is not JSON"),
          timeoutMs,
        );
      }
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

function isLateInferHealth(payload: unknown): boolean {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return false;
  const rec = payload as { ok?: unknown; name?: unknown };
  return rec.ok === true && rec.name === "late-infer";
}

export function parseLateInferHealth(payload: unknown): {
  device: string;
  deviceKind: string;
  accel: string;
  weightsInHostRam: boolean;
  gpuRunning: boolean;
} {
  const rec = payload && typeof payload === "object" && !Array.isArray(payload) ? (payload as Record<string, unknown>) : {};
  const device = typeof rec.device === "string" ? rec.device : typeof rec.device_label === "string" ? rec.device_label : "";
  const fromHealthKind =
    typeof rec.device_kind === "string"
      ? rec.device_kind
      : typeof rec.deviceKind === "string"
        ? rec.deviceKind
        : "";
  const classified = classifyLateInferDevice(fromHealthKind || device);
  const flag =
    rec.weights_in_host_ram === true ||
    rec.weightsInHostRam === true ||
    rec.weights_in_host_ram === "true";
  const weightsInHostRam = flag || classified.weightsInHostRam;
  const deviceKind = (fromHealthKind as string) || classified.deviceKind;
  const gpuRunning = !weightsInHostRam && (classified.gpuRunning || deviceKind === "intel-xpu" || deviceKind === "cuda" || deviceKind === "hip");
  return {
    device,
    deviceKind,
    accel: typeof rec.accel === "string" ? rec.accel : classified.accel,
    weightsInHostRam,
    gpuRunning,
  };
}

export async function probeLateInfer(options: {
  baseUrl?: string;
  fetchFn?: FetchLike;
  timeoutMs?: number;
  apiKey?: string;
} = {}): Promise<LocalServerStatus> {
  const label = "Late infer";
  const baseUrl = normalizeLoopbackOpenAiUrl(options.baseUrl?.trim() || DEFAULT_LATE_INFER_BASE, label);
  const origin = loopbackOrigin(baseUrl, label);
  const fetchFn = options.fetchFn ?? fetch;
  const timeoutMs = options.timeoutMs ?? 800;
  const apiKey = options.apiKey?.trim() || LOCAL_OPENAI_DUMMY_KEY;
  try {
    const health = await fetchFn(`${origin}/health`, {
      method: "GET",
      headers: { accept: "application/json" },
      signal: AbortSignal.timeout(timeoutMs),
    });
    const payload: unknown = await health.json().catch(() => undefined);
      if (health.ok && isLateInferHealth(payload)) {
        const modelsProbe = await getJson(`${baseUrl}/models`, fetchFn, timeoutMs, apiKey).catch(() => undefined);
        const models = modelsProbe ? openaiModelIds(modelsProbe.payload) : [];
        const modelFromHealth =
          payload && typeof payload === "object" && typeof (payload as { model?: unknown }).model === "string"
            ? acceptedModelId((payload as { model: string }).model)
            : undefined;
        if (modelFromHealth && !models.includes(modelFromHealth)) models.push(modelFromHealth);
        const meta = parseLateInferHealth(payload);
        const gpuBit = meta.gpuRunning
          ? "GPU running"
          : meta.weightsInHostRam
            ? "weights in host RAM, not idle GPU VRAM"
            : meta.device
              ? `device ${meta.device}`
              : "health ok";
        return {
          kind: "lateinfer",
          running: true,
          ready: true,
          baseUrl,
          origin,
          models,
          model: modelFromHealth ?? models[0],
          device: meta.device,
          deviceKind: meta.deviceKind,
          accel: meta.accel,
          weightsInHostRam: meta.weightsInHostRam,
          gpuRunning: meta.gpuRunning,
          reason:
            models.length > 0
              ? `late-infer running at ${origin} (${models.join(", ")}) · ${gpuBit}`
              : `late-infer running at ${origin} · ${gpuBit}`,
        };
      }
  } catch (error) {
    if (isUnreachableError(error)) {
      return downStatus("lateinfer", baseUrl, origin, error, timeoutMs);
    }
  }
  try {
    const modelsProbe = await getJson(`${baseUrl}/models`, fetchFn, timeoutMs, apiKey);
    if (!isOpenAiModelsList(modelsProbe.payload)) {
      return {
        kind: "lateinfer",
        running: false,
        ready: false,
        baseUrl,
        origin,
        models: [],
        reason: `late-infer not running at ${baseUrl} (/v1/models is not an OpenAI models JSON list)`,
      };
    }
    const models = openaiModelIds(modelsProbe.payload);
    return {
      kind: "lateinfer",
      running: true,
      ready: modelsProbe.ok,
      baseUrl,
      origin,
      models,
      model: models[0],
      reason: `late-infer reachable at ${baseUrl} (HTTP ${modelsProbe.status})`,
    };
  } catch (error) {
    return downStatus("lateinfer", baseUrl, origin, error, timeoutMs);
  }
}
