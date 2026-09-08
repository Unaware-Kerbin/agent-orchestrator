/**
 * Hugging Face Hub catalog for Late infer on this computer.
 * Official Hub REST API only (`/api/models`) — never scrape hf.co HTML.
 * Unknown graphs still fail closed at Download/compile.
 */

import { detectHardware, type AcceleratorVendor, type HardwareSnapshot } from "../hardware.js";
import { resolveHfToken } from "../secrets.js";
import { DEFAULT_LATE_INFER_MODEL } from "./loopback.js";
import {
  hubServeDecision,
  inferHubModelType,
  type HubServeVendor,
} from "./hub-serve.js";
import { attachHubBackends, type HubBackendSupport } from "./hub-backends.js";

/** Official Hub models list. Not the HTML model card. */
export const HF_HUB_MODELS_API = "https://huggingface.co/api/models";

/** Hide or mark models whose estimated weights exceed this fraction of VRAM (or RAM if no GPU). */
export const HUB_FIT_FRACTION = 0.8;

/** Extra VRAM on top of weights for KV cache + activations at a long context (max usage). */
export const VRAM_MAX_HEADROOM = 0.5;

export const HUB_FAMILY_ORDER = ["Qwen", "Mistral", "Gemma", "Llama", "Phi", "other"] as const;
export type HubFamily = (typeof HUB_FAMILY_ORDER)[number];

/** Hub `search=` seeds so empty q is not a single “Instruct” page (Gemma ids are `-it`, not Instruct). */
export const HUB_FAMILY_SEARCHES = ["qwen", "qwen3", "gemma", "mistral", "llama", "phi", "unsloth", "instruct"] as const;

/**
 * Well-known Instruct safetensors always listed on your computer.
 * Hub `/api/models?search=&sort=lastModified` returns community fine-tunes; official ids drop out of the page.
 * Never scrape hf.co HTML. Download still fail-closed for unknown graphs / gated repos.
 */
export const FALLBACK_HUB_SEEDS: readonly {
  id: string;
  gated?: boolean;
  weightsMiB: number;
  params: number;
  modelType: string;
}[] = [
  { id: "Qwen/Qwen2.5-0.5B-Instruct", weightsMiB: 1_000, params: 500_000_000, modelType: "qwen2" },
  { id: "Qwen/Qwen2.5-1.5B-Instruct", weightsMiB: 3_100, params: 1_500_000_000, modelType: "qwen2" },
  { id: "Qwen/Qwen2.5-3B-Instruct", weightsMiB: 6_000, params: 3_000_000_000, modelType: "qwen2" },
  { id: "Qwen/Qwen3-0.6B", weightsMiB: 1_200, params: 600_000_000, modelType: "qwen3" },
  { id: "Qwen/Qwen3-1.7B", weightsMiB: 3_400, params: 1_700_000_000, modelType: "qwen3" },
  { id: "Qwen/Qwen3-1.7B-Instruct", weightsMiB: 3_400, params: 1_700_000_000, modelType: "qwen3" },
  { id: "Qwen/Qwen3-4B-Instruct", weightsMiB: 8_000, params: 4_000_000_000, modelType: "qwen3" },
  { id: "Qwen/Qwen3-8B", weightsMiB: 16_000, params: 8_000_000_000, modelType: "qwen3" },
  { id: "Qwen/Qwen3-8B-Instruct", weightsMiB: 16_000, params: 8_000_000_000, modelType: "qwen3" },
  { id: "Qwen/Qwen2.5-7B-Instruct", weightsMiB: 14_000, params: 7_000_000_000, modelType: "qwen2" },
  { id: "google/gemma-2-2b-it", gated: true, weightsMiB: 5_000, params: 2_600_000_000, modelType: "gemma2" },
  { id: "google/gemma-2-9b-it", gated: true, weightsMiB: 18_000, params: 9_200_000_000, modelType: "gemma2" },
  { id: "google/gemma-3-1b-it", gated: true, weightsMiB: 3_000, params: 1_000_000_000, modelType: "gemma3" },
  { id: "google/gemma-3-4b-it", gated: true, weightsMiB: 8_000, params: 4_300_000_000, modelType: "gemma3" },
  { id: "google/gemma-4-E2B-it", weightsMiB: 10_000, params: 5_100_000_000, modelType: "gemma4" },
  { id: "google/gemma-4-12B-it", weightsMiB: 23_000, params: 12_000_000_000, modelType: "gemma4_unified" },
  { id: "mistralai/Mistral-7B-Instruct-v0.2", weightsMiB: 14_000, params: 7_000_000_000, modelType: "mistral" },
  { id: "mistralai/Mistral-7B-Instruct-v0.3", weightsMiB: 14_000, params: 7_000_000_000, modelType: "mistral" },
  { id: "mistralai/Ministral-8B-Instruct-2410", weightsMiB: 16_000, params: 8_000_000_000, modelType: "mistral" },
  { id: "meta-llama/Llama-3.2-1B-Instruct", gated: true, weightsMiB: 3_000, params: 1_200_000_000, modelType: "llama" },
  { id: "meta-llama/Llama-3.2-3B-Instruct", gated: true, weightsMiB: 7_000, params: 3_200_000_000, modelType: "llama" },
  { id: "microsoft/Phi-3.5-mini-instruct", weightsMiB: 7_600, params: 3_800_000_000, modelType: "phi3" },
  { id: "microsoft/Phi-4-mini-instruct", weightsMiB: 7_500, params: 3_800_000_000, modelType: "phi3" },
];

const CACHE_TTL_MS = 120_000;
const HUB_TIMEOUT_MS = 10_000;
const HUB_LIMIT = 200;
const HUB_PAGES = 2;
const QUERY_MAX = 128;

const DTYPE_BYTES: Record<string, number> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U8: 1,
  BOOL: 1,
  F8_E4M3: 1,
  F8_E5M2: 1,
};

const ALLOWED_PIPELINE = new Set(["text-generation", "text2text-generation", "conversational"]);

const BLOCKED_PIPELINE = new Set([
  "text-to-image",
  "image-to-image",
  "automatic-speech-recognition",
  "text-to-speech",
  "audio-to-audio",
  "text-to-video",
  "video-to-video",
  "feature-extraction",
  "sentence-similarity",
  "fill-mask",
  "token-classification",
  "image-classification",
  "object-detection",
  "zero-shot-image-classification",
  "image-segmentation",
  "depth-estimation",
  "mask-generation",
  "visual-question-answering",
  "document-question-answering",
]);

export interface HubSibling {
  rfilename?: string;
  size?: number;
}

export interface HubSafetensors {
  total?: number;
  parameters?: number | Record<string, number>;
}

/** Subset of Hub `/api/models` JSON. */
export interface HubRawModel {
  id?: string;
  modelId?: string;
  pipeline_tag?: string;
  tags?: string[];
  library_name?: string;
  lastModified?: string;
  downloads?: number;
  gated?: boolean | string;
  siblings?: HubSibling[];
  safetensors?: HubSafetensors;
  config?: { model_type?: string };
}

export interface HubCatalogModel {
  id: string;
  family: HubFamily;
  version?: number;
  lastModified?: string;
  fits: boolean;
  likelyTooBig: boolean;
  sizeUnknown: boolean;
  gated: boolean;
  /** True when Hub list included a `gated` field (expand=gated). False/omit means unknown — keep seed gated on merge. */
  gatedKnown?: boolean;
  /** True when Hub denied config.json (401/403) — accept the license before a GB fetch. */
  gatedNeedsLicense?: boolean;
  /** False when gated without license, or this GPU cannot serve the graph. */
  downloadable?: boolean;
  /** transformers config.model_type (inferred from the Hub id when missing). */
  modelType?: string;
  /** Built-in store row so the GUI is never an empty paste box. */
  fallback?: boolean;
  bytes?: number;
  params?: number;
  sizeMiB?: number;
  /** Peak VRAM estimate: weights + KV/activations at a long context. */
  vramMaxMiB?: number;
  vramMaxGiB?: number;
  vramMaxLabel?: string;
  /** Which local engines can run this id (lateinfer OV vs not, vllm, ollama, llamacpp). */
  backends?: HubBackendSupport;
}

export interface HubCatalogGroup {
  family: HubFamily;
  models: HubCatalogModel[];
}

export interface HubCatalogResponse {
  models: HubCatalogModel[];
  groups: HubCatalogGroup[];
  defaultId: string;
  budgetMiB: number;
  budgetGiB: number;
  hint: string;
  offline: boolean;
  /** True when Hub API failed or returned nothing usable — built-in ids still listed. */
  usedFallback: boolean;
  /** Idle GPU vendor from gpu-pick (`intel` / `nvidia` / `amd`). */
  serveVendor?: HubServeVendor;
  serveLabel?: string;
  runtimeOk?: boolean;
  error?: string;
}

export type HubFetchFn = (input: string | URL, init?: RequestInit) => Promise<Response>;

interface CatalogDeps {
  fetchFn?: HubFetchFn;
  hardware?: HardwareSnapshot;
  now?: () => number;
  /** Test override — gpu-pick vendor, not a SKU. */
  vendor?: HubServeVendor;
  runtimeOk?: boolean;
  serveLabel?: string;
}

let testDeps: CatalogDeps = {};
const hubCache = new Map<string, { at: number; rows: HubRawModel[] }>();

function hubCacheKey(search: string, hasToken: boolean): string {
  return `${hasToken ? "t" : "p"}:${search.toLowerCase()}`;
}

export function resetHubCatalogForTests(deps: CatalogDeps = {}): void {
  hubCache.clear();
  testDeps = deps;
}

export function fallbackHubCatalogModels(): HubCatalogModel[] {
  return FALLBACK_HUB_SEEDS.map((seed) =>
    withVramEstimate({
      id: seed.id,
      family: hubFamily(seed.id),
      version: parseHubVersion(seed.id),
      fits: true,
      likelyTooBig: false,
      sizeUnknown: false,
      gated: seed.gated === true,
      gatedKnown: true,
      gatedNeedsLicense: seed.gated === true,
      downloadable: seed.gated !== true,
      modelType: seed.modelType,
      fallback: true,
      sizeMiB: seed.weightsMiB,
      params: seed.params,
    }),
  );
}

export function fallbackHubCatalogModelsForVendor(
  vendor: HubServeVendor | undefined,
  runtimeOk = true,
): HubCatalogModel[] {
  return fallbackHubCatalogModels().filter((model) => hubServeDecision(vendor, model, runtimeOk).ok);
}

/** Hub safetensors/siblings win on the same id; seed weights fill gaps. Official Instruct seeds always remain and stay listed (not too-big) on Intel Arc. */
export function mergeHubCatalog(fromHub: readonly HubCatalogModel[], fallback = fallbackHubCatalogModels()): HubCatalogModel[] {
  const byId = new Map<string, HubCatalogModel>();
  for (const model of fallback) {
    if (model?.id) byId.set(model.id, model);
  }
  const seedIds = new Set(FALLBACK_HUB_SEEDS.map((s) => s.id));
  for (const model of fromHub) {
    if (!model?.id) continue;
    const prev = byId.get(model.id);
    const isSeed = seedIds.has(model.id) || prev?.fallback === true;
    const next = isSeed ? { ...model, fallback: true } : model;
    const gated =
      next.gatedKnown === false && prev
        ? Boolean(prev.gated) || Boolean(next.gated)
        : Boolean(next.gated);
    byId.set(
      model.id,
      withVramEstimate({
        ...prev,
        ...next,
        gated,
        gatedKnown: next.gatedKnown !== false || prev?.gatedKnown === true,
        gatedNeedsLicense: gated || Boolean(prev?.gatedNeedsLicense),
        downloadable: next.downloadable ?? prev?.downloadable ?? !gated,
        sizeMiB: next.sizeMiB ?? prev?.sizeMiB,
        params: next.params ?? prev?.params,
        bytes: next.bytes ?? prev?.bytes,
        modelType: next.modelType ?? prev?.modelType,
        ...(isSeed ? { fallback: true, likelyTooBig: false, fits: true } : {}),
      }),
    );
  }
  return [...byId.values()].map((model) =>
    withVramEstimate(
      seedIds.has(model.id) || model.fallback === true
        ? { ...model, fallback: true, likelyTooBig: false, fits: true }
        : model,
    ),
  );
}

export function emptyHubCatalog(input: {
  hardware?: HardwareSnapshot;
  error?: string;
  offline?: boolean;
  q?: string;
  vendor?: HubServeVendor;
  runtimeOk?: boolean;
  serveLabel?: string;
}): HubCatalogResponse {
  const hardware = input.hardware ?? testDeps.hardware ?? detectHardware();
  const serve = resolveCatalogServe({ hardware, vendor: input.vendor, runtimeOk: input.runtimeOk, serveLabel: input.serveLabel });
  const budgetMiB = hardwareFitBudgetMiB(hardware);
  const budgetGiB = Math.round((budgetMiB / 1024) * 10) / 10;
  const q = String(input.q ?? "").trim();
  const mapped = fallbackHubCatalogModelsForVendor(serve.vendor, serve.runtimeOk).filter((model) =>
    matchesQuery(model, q),
  );
  const groups = groupHubModels(mapped);
  return {
    models: groups.flatMap((g) => g.models),
    groups,
    defaultId: defaultHubIdForServe(mapped),
    budgetMiB,
    budgetGiB,
    hint: catalogHint(budgetGiB, true, serve),
    offline: input.offline ?? true,
    usedFallback: true,
    serveVendor: serve.vendor,
    serveLabel: serve.label,
    runtimeOk: serve.runtimeOk,
    ...(input.error ? { error: input.error } : {}),
  };
}

function defaultHubIdForServe(models: readonly HubCatalogModel[]): string {
  if (models.some((m) => m.id === DEFAULT_LATE_INFER_MODEL)) return DEFAULT_LATE_INFER_MODEL;
  return models[0]?.id || DEFAULT_LATE_INFER_MODEL;
}

export function catalogHint(
  budgetGiB: number,
  offline = false,
  serve?: { vendor?: HubServeVendor; runtimeOk?: boolean; label?: string },
): string {
  const n = Number.isFinite(budgetGiB) ? budgetGiB : 0;
  const gpu =
    serve?.label?.trim() ||
    (serve?.vendor === "intel"
      ? "the idle Intel GPU on your computer (OpenVINO CausalLM)"
      : serve?.vendor === "nvidia"
        ? "the idle NVIDIA GPU on your computer (Candle / CUDA)"
        : serve?.vendor === "amd"
          ? "the idle AMD GPU on your computer"
          : "the idle GPU on your computer");
  const listed = `Safetensors chat models for OpenVINO / late-infer compile on ${gpu} (~${n} GB) — not the full Hugging Face website. GGUF-only models appear under llama.cpp. vLLM has its own weights pane. Each row can actually compile for this card. VRAM at max usage (weights plus KV cache) is shown. Gated repos need the Hugging Face license accepted first. Search, click a row, or paste any Hub org/model id, then Download.`;
  if (serve?.vendor === "amd" || serve?.runtimeOk === false) {
    return `${listed} This GPU cannot Start Hub Instruct snapshots yet.`;
  }
  if (offline) {
    return `Hugging Face Hub is offline or returned no official Instruct ids. These well-known ids still list on your computer when this GPU can run them. ${listed}`;
  }
  return listed;
}

export function hardwareFitBudgetMiB(hardware: HardwareSnapshot): number {
  const vram = Number(hardware.totalVramMiB) > 0 ? hardware.totalVramMiB : Number(hardware.vramMiB) > 0 ? hardware.vramMiB : 0;
  const ram = Math.max(0, Number(hardware.ramMiB) || 0);
  // Intel Arc / XPU has no nvidia-smi. Unknown GPU VRAM still lists against RAM so the catalog is not empty.
  if (vram > 0) return vram * HUB_FIT_FRACTION;
  return ram * HUB_FIT_FRACTION;
}

export function resolveCatalogServe(input: {
  hardware?: HardwareSnapshot;
  vendor?: HubServeVendor;
  runtimeOk?: boolean;
  serveLabel?: string;
} = {}): { vendor: HubServeVendor; runtimeOk: boolean; label: string } {
  const vendor = (testDeps.vendor ?? input.vendor ?? vendorFromHardware(input.hardware ?? testDeps.hardware) ?? "") as HubServeVendor;
  const runtimeOk =
    testDeps.runtimeOk ??
    input.runtimeOk ??
    (vendor === "nvidia" || vendor === "intel");
  const label =
    testDeps.serveLabel ??
    input.serveLabel ??
    (vendor === "intel"
      ? "the idle Intel GPU on your computer (OpenVINO CausalLM)"
      : vendor === "nvidia"
        ? "the idle NVIDIA GPU on your computer (Candle / CUDA)"
        : vendor === "amd"
          ? "the idle AMD GPU on your computer"
          : "the idle GPU on your computer");
  return { vendor, runtimeOk, label };
}

function vendorFromHardware(hardware?: HardwareSnapshot): AcceleratorVendor | undefined {
  const fromAccel = hardware?.accelerators?.find((row) => row.vendor)?.vendor;
  if (fromAccel) return fromAccel;
  const fromGpu = hardware?.gpus?.find((row) => row.vendor)?.vendor;
  return fromGpu;
}

export function catalogSearchTerms(q: string): string[] {
  const extra = String(q ?? "")
    .trim()
    .slice(0, QUERY_MAX);
  const terms: string[] = [...HUB_FAMILY_SEARCHES];
  if (!extra) return terms;
  const lower = extra.toLowerCase();
  if (!(HUB_FAMILY_SEARCHES as readonly string[]).includes(lower)) terms.push(extra);
  return terms;
}

export function buildHubModelsUrl(search: string, limit = HUB_LIMIT): URL {
  const url = new URL(HF_HUB_MODELS_API);
  url.searchParams.set("search", search);
  // safetensors only — do not AND text-generation (Gemma 3 is image-text-to-text on the Hub).
  url.searchParams.set("filter", "safetensors");
  url.searchParams.set("sort", "lastModified");
  url.searchParams.set("direction", "-1");
  url.searchParams.set("limit", String(limit));
  url.searchParams.append("expand", "safetensors");
  url.searchParams.append("expand", "siblings");
  // List API omits `gated` unless expanded — without this, merge overwrites seed gated:true with false.
  url.searchParams.append("expand", "gated");
  url.searchParams.set("config", "true");
  return url;
}

export function hubFamily(id: string, tags: readonly string[] = [], modelType?: string): HubFamily {
  const blob = `${id} ${tags.join(" ")} ${modelType ?? ""}`.toLowerCase();
  if (blob.includes("qwen")) return "Qwen";
  if (blob.includes("mistral") || blob.includes("mixtral")) return "Mistral";
  if (blob.includes("gemma")) return "Gemma";
  if (blob.includes("llama")) return "Llama";
  if (/(?:^|[^a-z])phi(?:\d|[ab]|\b|-|_)/.test(blob) || /microsoft\/phi/i.test(id)) return "Phi";
  return "other";
}

/** Generation from the Hub id (Qwen2.5 → 2.5, Llama-3.1 → 3.1). Param sizes like 0.5B are stripped. */
/** Billion-param size from a Hub id (`Qwen2.5-7B` → 7, `gemma-4-E2B` → stored ~5.1). */
export function parseHubParamsB(id: string): number | undefined {
  const name = id.includes("/") ? (id.split("/")[1] ?? id) : id;
  const moe = name.match(/[-_]E(\d+(?:\.\d+)?)B/i);
  if (moe?.[1]) {
    const active = Number(moe[1]);
    if (Number.isFinite(active) && active > 0) {
      if (active <= 2) return 5.1;
      if (active <= 4) return 8;
      return active * 2.5;
    }
  }
  const matches = [...name.matchAll(/(\d+(?:\.\d+)?)[Bb](?:[-_]|$)/g)];
  const last = matches[matches.length - 1];
  if (!last?.[1]) return undefined;
  const n = Number(last[1]);
  if (!Number.isFinite(n) || n <= 0 || n > 1000) return undefined;
  return n;
}

export function estimateMaxVramMiB(weightsMiB: number): number {
  if (!Number.isFinite(weightsMiB) || weightsMiB <= 0) return 0;
  return Math.ceil(weightsMiB * (1 + VRAM_MAX_HEADROOM));
}

export function formatVramMaxLabel(vramMaxMiB: number): string {
  if (!Number.isFinite(vramMaxMiB) || vramMaxMiB <= 0) return "VRAM unknown";
  if (vramMaxMiB < 1024) return `~${Math.round(vramMaxMiB)} MiB VRAM max`;
  const gib = vramMaxMiB / 1024;
  const shown =
    gib >= 10 ? String(Math.round(gib)) : (Math.round(gib * 10) / 10).toFixed(1).replace(/\.0$/, "");
  return `~${shown} GB VRAM max`;
}

/** Weights from Hub metadata, then param count, then the `7B` / `E2B` token in the id (BF16 ≈ 2 bytes/param). */
export function inferWeightsMiB(input: { id: string; sizeMiB?: number; params?: number }): number | undefined {
  if (Number.isFinite(input.sizeMiB) && (input.sizeMiB ?? 0) > 0) return input.sizeMiB;
  const params = input.params;
  if (Number.isFinite(params) && (params ?? 0) > 0) {
    const n = params as number;
    const count = n >= 1e6 ? n : n * 1e9;
    return (count * 2) / (1024 * 1024);
  }
  const billions = parseHubParamsB(input.id);
  if (billions != null) return billions * 2_000;
  return undefined;
}

export function withVramEstimate(model: HubCatalogModel): HubCatalogModel {
  const sizeMiB = inferWeightsMiB(model);
  const vramMaxMiB = sizeMiB != null ? estimateMaxVramMiB(sizeMiB) : undefined;
  return attachHubBackends({
    ...model,
    sizeMiB: sizeMiB ?? model.sizeMiB,
    sizeUnknown: sizeMiB == null,
    vramMaxMiB,
    vramMaxGiB: vramMaxMiB != null ? Math.round((vramMaxMiB / 1024) * 10) / 10 : undefined,
    vramMaxLabel: formatVramMaxLabel(vramMaxMiB ?? 0),
  });
}

export function parseHubVersion(id: string): number | undefined {
  const name = id.includes("/") ? (id.split("/")[1] ?? id) : id;
  const stripped = name.replace(/\d+(?:\.\d+)?(?:x\d+(?:\.\d+)?)?[Bb](?:[-_]|$)/gi, "-");
  const familyGen = stripped.match(/(?:qwen|llama|gemma|phi|mistral|mixtral)[-_]?(\d+(?:\.\d+)?)/i);
  if (familyGen?.[1]) {
    const n = Number(familyGen[1]);
    if (Number.isFinite(n)) return n;
  }
  const instructV = stripped.match(/[-_]v(\d+(?:\.\d+)?)/i);
  if (instructV?.[1]) {
    const n = Number(instructV[1]);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

export function estimateHubWeights(row: HubRawModel): { bytes?: number; params?: number; sizeMiB?: number } {
  let params: number | undefined;
  let bytes: number | undefined;
  const st = row.safetensors;
  if (st && typeof st === "object") {
    const p = st.parameters;
    if (typeof p === "number" && Number.isFinite(p) && p > 0) {
      params = p;
    } else if (p && typeof p === "object") {
      let sum = 0;
      let fromDtype = 0;
      for (const [dtype, count] of Object.entries(p)) {
        const n = Number(count);
        if (!Number.isFinite(n) || n <= 0) continue;
        sum += n;
        fromDtype += n * (DTYPE_BYTES[dtype.toUpperCase()] ?? 2);
      }
      if (sum > 0) params = sum;
      if (fromDtype > 0) bytes = fromDtype;
    }
    const total = Number(st.total);
    if (params == null && Number.isFinite(total) && total > 0) params = total;
  }

  let sibBytes = 0;
  for (const sibling of row.siblings ?? []) {
    const name = String(sibling.rfilename ?? "");
    if (!/\.safetensors$/i.test(name)) continue;
    const size = Number(sibling.size);
    if (Number.isFinite(size) && size > 0) sibBytes += size;
  }
  if (sibBytes > 0) bytes = sibBytes;
  else if (bytes == null && params != null) bytes = params * 2;

  const sizeMiB = bytes != null ? bytes / (1024 * 1024) : undefined;
  return { bytes, params, sizeMiB };
}

export function isHubInstructCandidate(row: HubRawModel): boolean {
  const id = String(row.id ?? row.modelId ?? "").trim();
  if (!id.includes("/")) return false;
  const tags = (row.tags ?? []).map((t) => String(t).toLowerCase());
  const pipeline = String(row.pipeline_tag ?? "").toLowerCase();
  const library = String(row.library_name ?? "").toLowerCase();
  const sibNames = (row.siblings ?? []).map((s) => String(s.rfilename ?? "").toLowerCase());

  if (BLOCKED_PIPELINE.has(pipeline)) return false;
  if (pipeline.includes("diffusion") || pipeline.includes("whisper")) return false;
  if (
    tags.some(
      (t) =>
        t === "diffusers" ||
        t.includes("stable-diffusion") ||
        t === "whisper" ||
        t === "text-to-image",
    )
  ) {
    return false;
  }

  const hasSafetensors =
    tags.includes("safetensors") ||
    sibNames.some((n) => n.endsWith(".safetensors")) ||
    Boolean(row.safetensors && typeof row.safetensors === "object");
  const hasGguf = tags.includes("gguf") || library === "gguf" || sibNames.some((n) => n.endsWith(".gguf"));
  if (hasGguf && !hasSafetensors) return false;
  const instructId =
    /\binstruct\b/i.test(id) ||
    /\bchat\b/i.test(id) ||
    /[-_/]it(?:[-_/]|$)/i.test(id);
  // Hub list rows often omit siblings/tags even with filter=safetensors.
  if (!hasSafetensors && !instructId) return false;

  const multimodalLlm =
    pipeline === "image-text-to-text" ||
    pipeline === "image-to-text" ||
    pipeline === "any-to-any" ||
    tags.includes("image-text-to-text");

  if (pipeline) {
    if (!ALLOWED_PIPELINE.has(pipeline) && !multimodalLlm) return false;
  } else if (
    !instructId &&
    !tags.includes("text-generation") &&
    !tags.includes("conversational") &&
    !tags.includes("text2text-generation") &&
    !multimodalLlm
  ) {
    return false;
  }

  const blob = `${id} ${tags.join(" ")}`.toLowerCase();
  return (
    /\binstruct\b/.test(blob) ||
    /\bchat\b/.test(blob) ||
    tags.includes("conversational") ||
    /[-_/]it(?:[-_/]|$)/i.test(id)
  );
}

export function toHubCatalogModel(row: HubRawModel, budgetMiB: number): HubCatalogModel | undefined {
  const id = String(row.id ?? row.modelId ?? "").trim();
  if (!isHubInstructCandidate(row)) return undefined;
  const tags = row.tags ?? [];
  const family = hubFamily(id, tags, row.config?.model_type);
  const weights = estimateHubWeights(row);
  const lastModified = typeof row.lastModified === "string" && row.lastModified.trim() ? row.lastModified : undefined;
  const gatedKnown = row.gated !== undefined && row.gated !== null;
  const gated = hubRowGated(row);
  const modelType = inferHubModelType(id, row.config?.model_type) || undefined;
  const estimated = withVramEstimate({
    id,
    family,
    version: parseHubVersion(id),
    lastModified,
    fits: true,
    likelyTooBig: false,
    sizeUnknown: weights.sizeMiB == null,
    gated,
    gatedKnown,
    gatedNeedsLicense: gated,
    downloadable: !gated,
    modelType,
    bytes: weights.bytes,
    params: weights.params,
    sizeMiB: weights.sizeMiB,
  });
  const sizeUnknown = estimated.sizeUnknown;
  const likelyTooBig = !sizeUnknown && (estimated.sizeMiB ?? 0) > budgetMiB;
  return {
    ...estimated,
    fits: sizeUnknown || !likelyTooBig,
    likelyTooBig,
    sizeUnknown,
  };
}

export function hubRowGated(row: HubRawModel): boolean {
  const gated = row.gated;
  if (gated === true) return true;
  if (typeof gated === "string") {
    const v = gated.trim().toLowerCase();
    return v !== "" && v !== "false" && v !== "0";
  }
  return false;
}

function lastModifiedMs(value: string | undefined): number {
  if (!value) return 0;
  const t = Date.parse(value);
  return Number.isFinite(t) ? t : 0;
}

/** Newest generation first, then Hub lastModified, then fit, then size, then name. Fallback is a seed of ids, not a sort key. */
export function compareHubModels(a: HubCatalogModel, b: HubCatalogModel): number {
  const aSeed = a.fallback === true || FALLBACK_HUB_SEEDS.some((s) => s.id === a.id);
  const bSeed = b.fallback === true || FALLBACK_HUB_SEEDS.some((s) => s.id === b.id);
  if (aSeed !== bSeed) return aSeed ? -1 : 1;
  const av = a.version;
  const bv = b.version;
  if (av != null && bv != null && av !== bv) return bv - av;
  if (av != null && bv == null) return -1;
  if (av == null && bv != null) return 1;
  const modified = lastModifiedMs(b.lastModified) - lastModifiedMs(a.lastModified);
  if (modified !== 0) return modified;
  if (a.fits !== b.fits) return a.fits ? -1 : 1;
  const as = a.sizeMiB;
  const bs = b.sizeMiB;
  if (a.fits && b.fits) {
    if (as != null && bs != null && as !== bs) return bs - as;
    if (as != null && bs == null) return -1;
    if (as == null && bs != null) return 1;
  } else if (!a.fits && !b.fits) {
    if (as != null && bs != null && as !== bs) return as - bs;
  }
  return a.id.localeCompare(b.id);
}

export function groupHubModels(models: HubCatalogModel[]): HubCatalogGroup[] {
  const buckets = new Map<HubFamily, HubCatalogModel[]>();
  for (const family of HUB_FAMILY_ORDER) buckets.set(family, []);
  for (const model of models) {
    const list = buckets.get(model.family) ?? buckets.get("other");
    list?.push(model);
  }
  const groups: HubCatalogGroup[] = [];
  for (const family of HUB_FAMILY_ORDER) {
    const list = buckets.get(family) ?? [];
    if (list.length === 0) continue;
    list.sort(compareHubModels);
    groups.push({ family, models: list });
  }
  return groups;
}

export function matchesQuery(model: HubCatalogModel, q: string): boolean {
  if (!q) return true;
  const needle = q.toLowerCase();
  return model.id.toLowerCase().includes(needle) || model.family.toLowerCase().includes(needle);
}

function parseHubListPayload(body: unknown): HubRawModel[] | undefined {
  if (Array.isArray(body)) return body as HubRawModel[];
  if (body && typeof body === "object" && Array.isArray((body as { models?: unknown }).models)) {
    return (body as { models: HubRawModel[] }).models;
  }
  return undefined;
}

async function fetchHubRows(options: {
  search: string;
  fetchFn: HubFetchFn;
  token?: string;
  offset?: number;
}): Promise<{ rows: HubRawModel[]; error?: string }> {
  const url = buildHubModelsUrl(options.search);
  if (options.offset && options.offset > 0) url.searchParams.set("offset", String(options.offset));
  if (!url.href.startsWith(`${HF_HUB_MODELS_API}?`) && url.origin + url.pathname !== HF_HUB_MODELS_API) {
    return { rows: [], error: "Hub catalog URL is not the official API" };
  }
  const headers: Record<string, string> = {
    accept: "application/json",
    "user-agent": "agent-orchestrator",
  };
  if (options.token) headers.authorization = `Bearer ${options.token}`;
  let res: Response;
  try {
    res = await options.fetchFn(url, {
      method: "GET",
      headers,
      signal: AbortSignal.timeout(HUB_TIMEOUT_MS),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { rows: [], error: "Hugging Face Hub API unavailable" + (message ? ` (${message})` : "") };
  }
  const contentType = res.headers.get("content-type") ?? "";
  if (contentType.includes("text/html")) {
    return { rows: [], error: "Hugging Face Hub API returned HTML; catalog does not scrape model pages" };
  }
  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    return { rows: [], error: "Hugging Face Hub API returned non-JSON" };
  }
  if (!res.ok) {
    return { rows: [], error: `Hugging Face Hub API HTTP ${res.status}` };
  }
  const rows = parseHubListPayload(parsed);
  if (!rows) return { rows: [], error: "Hugging Face Hub API returned an unexpected payload" };
  return { rows };
}

async function fetchMergedHubRows(options: {
  q: string;
  fetchFn: HubFetchFn;
  token?: string;
  now: number;
}): Promise<{ rows: HubRawModel[]; error?: string }> {
  const terms = catalogSearchTerms(options.q);
  const hasToken = Boolean(options.token);
  const jobs = terms.flatMap((search) =>
    Array.from({ length: HUB_PAGES }, (_, page) => {
      const offset = page * HUB_LIMIT;
      return (async () => {
        const key = `${hubCacheKey(search, hasToken)}:${offset}`;
        const cached = hubCache.get(key);
        if (cached && options.now - cached.at < CACHE_TTL_MS) {
          return { rows: cached.rows };
        }
        const fetched = await fetchHubRows({
          search,
          fetchFn: options.fetchFn,
          token: options.token,
          offset,
        });
        if (!fetched.error) hubCache.set(key, { at: options.now, rows: fetched.rows });
        return fetched;
      })();
    }),
  );
  const results = await Promise.all(jobs);
  const byId = new Map<string, HubRawModel>();
  const errors: string[] = [];
  for (const result of results) {
    if (result.error && result.rows.length === 0) errors.push(result.error);
    for (const row of result.rows) {
      const id = String(row.id ?? row.modelId ?? "").trim();
      if (id && !byId.has(id)) byId.set(id, row);
    }
  }
  const rows = [...byId.values()];
  if (rows.length === 0) {
    return { rows: [], error: errors[0] ?? "Hugging Face Hub API unavailable" };
  }
  return { rows };
}

/**
 * List Instruct/chat safetensors from the Hub that are likely to fit this computer.
 * Always includes the built-in fallback catalog (official Qwen/Gemma/Mistral/Llama/Phi Instruct ids).
 * Hub `/api/models` results merge on top when they work; empty/offline Hub still lists fallback.
 * Intel Arc / XPU (no nvidia-smi) still lists: unknown VRAM uses RAM; fallback seeds stay listed (likelyTooBig false) even when size is known.
 * Cloud AI is not required for public Hub listing. Gated Gemma/Llama stay listed; Download still fail-closed.
 */
export async function listHubModels(options: {
  q?: string;
  fetchFn?: HubFetchFn;
  hardware?: HardwareSnapshot;
  token?: string;
} = {}): Promise<HubCatalogResponse> {
  try {
    const hardware = options.hardware ?? testDeps.hardware ?? detectHardware();
    const budgetMiB = hardwareFitBudgetMiB(hardware);
    const budgetGiB = Math.round((budgetMiB / 1024) * 10) / 10;
    const q = String(options.q ?? "")
      .trim()
      .slice(0, QUERY_MAX);
    const now = (testDeps.now ?? Date.now)();
    const fetchFn = options.fetchFn ?? testDeps.fetchFn ?? globalThis.fetch;
    const token = options.token ?? resolveHfToken();
    const fetched = await fetchMergedHubRows({ q, fetchFn, token, now });
    const offline = Boolean(fetched.error && fetched.rows.length === 0);

    const fromHub: HubCatalogModel[] = [];
    for (const row of fetched.rows) {
      try {
        const model = toHubCatalogModel(row, budgetMiB);
        if (!model) continue;
        // Keep likelyTooBig (Intel Arc ~31 GB × 2 still lists Gemma 2B/E2B/4B; Download/compile fail closed).
        fromHub.push(model);
      } catch {
        // Soft-fail untypable Hub rows (bad payload / missing config metadata) — keep the rest of the list.
        continue;
      }
    }
    const mapped = mergeHubCatalog(fromHub).filter((model) => matchesQuery(model, q));
    const groups = groupHubModels(mapped);
    const models = groups.flatMap((g) => g.models);
    return {
      models,
      groups,
      defaultId: DEFAULT_LATE_INFER_MODEL,
      budgetMiB,
      budgetGiB,
      hint: catalogHint(budgetGiB, offline),
      offline,
      usedFallback: offline || fromHub.length === 0,
      ...(offline && fetched.error ? { error: fetched.error } : {}),
    };
  } catch (error) {
    return emptyHubCatalog({
      hardware: options.hardware ?? testDeps.hardware,
      error: error instanceof Error ? error.message : String(error),
      offline: true,
      q: options.q,
    });
  }
}
