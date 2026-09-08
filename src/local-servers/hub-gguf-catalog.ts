/**
 * Hugging Face Hub GGUF catalog for llama.cpp on this computer.
 * Official Hub REST API only (`/api/models?filter=gguf`) — never scrape hf.co HTML.
 * Separate from late-infer safetensors / OpenVINO rules.
 */

import { existsSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { detectHardware, type HardwareSnapshot } from "../hardware.js";
import { resolveHfToken } from "../secrets.js";
import { stateDir } from "../state.js";
import { HF_HUB_MODELS_API, HUB_FIT_FRACTION, hubRowGated, type HubFetchFn, type HubRawModel, type HubSibling } from "./hub-catalog.js";

const CACHE_TTL_MS = 120_000;
const HUB_TIMEOUT_MS = 10_000;
const HUB_LIMIT = 200;
const QUERY_MAX = 128;
const PAGES = 2;

/** Seeds so empty q still finds Unsloth + common chat families (GGUF filter). */
export const GGUF_HUB_SEARCHES = ["unsloth", "qwen", "gemma", "mistral", "llama", "phi", "instruct"] as const;

/** Quant preference for a single downloadable file (Vulkan / llama.cpp on Arc). */
const QUANT_RANK: { re: RegExp; rank: number }[] = [
  { re: /Q4_K_M/i, rank: 100 },
  { re: /Q5_K_M/i, rank: 95 },
  { re: /Q4_K_S/i, rank: 90 },
  { re: /Q5_K_S/i, rank: 85 },
  { re: /Q6_K/i, rank: 80 },
  { re: /Q3_K_M/i, rank: 70 },
  { re: /Q4_0/i, rank: 65 },
  { re: /Q5_0/i, rank: 60 },
  { re: /Q8_0/i, rank: 55 },
  { re: /IQ4_XS/i, rank: 50 },
  { re: /IQ3_M/i, rank: 45 },
  { re: /Q2_K/i, rank: 30 },
];

export interface GgufFileHint {
  filename: string;
  sizeBytes?: number;
  sizeMiB?: number;
  sizeGiB?: number;
  sizeLabel: string;
}

export interface GgufHubModel {
  id: string;
  family: string;
  lastModified?: string;
  gated: boolean;
  downloads?: number;
  fits: boolean;
  likelyTooBig: boolean;
  sizeUnknown: boolean;
  /** Preferred single GGUF file to download/start. */
  preferredFile?: GgufFileHint;
  files: GgufFileHint[];
  /** Local absolute path when already downloaded. */
  localPath?: string;
  downloaded: boolean;
  sizeMiB?: number;
  sizeLabel?: string;
}

export interface GgufHubCatalogResponse {
  models: GgufHubModel[];
  defaultId?: string;
  budgetMiB: number;
  budgetGiB: number;
  hint: string;
  offline: boolean;
  ggufDir: string;
  error?: string;
}

interface CatalogDeps {
  fetchFn?: HubFetchFn;
  hardware?: HardwareSnapshot;
  now?: () => number;
  ggufDir?: string;
}

let testDeps: CatalogDeps = {};
const hubCache = new Map<string, { at: number; rows: HubRawModel[] }>();

export function resetGgufHubCatalogForTests(deps: CatalogDeps = {}): void {
  hubCache.clear();
  testDeps = deps;
}

export function defaultGgufModelsDir(): string {
  return join(stateDir(), "models", "gguf");
}

export function ggufRepoDirName(id: string): string {
  return id.replace(/\//g, "--");
}

export function localGgufDirForRepo(id: string, root = defaultGgufModelsDir()): string {
  return join(root, ggufRepoDirName(id));
}

function formatSizeLabel(bytes: number | undefined): string {
  if (!Number.isFinite(bytes) || (bytes ?? 0) <= 0) return "size unknown";
  const mib = (bytes as number) / (1024 * 1024);
  if (mib < 1024) return `~${Math.round(mib)} MiB`;
  const gib = mib / 1024;
  const shown = gib >= 10 ? String(Math.round(gib)) : (Math.round(gib * 10) / 10).toFixed(1).replace(/\.0$/, "");
  return `~${shown} GB`;
}

export function siblingToGgufFile(sib: HubSibling): GgufFileHint | undefined {
  const filename = String(sib.rfilename ?? "").trim();
  if (!filename || !/\.gguf$/i.test(filename)) return undefined;
  const size = Number(sib.size);
  const sizeBytes = Number.isFinite(size) && size > 0 ? size : undefined;
  const sizeMiB = sizeBytes != null ? sizeBytes / (1024 * 1024) : undefined;
  return {
    filename,
    sizeBytes,
    sizeMiB,
    sizeGiB: sizeMiB != null ? Math.round((sizeMiB / 1024) * 10) / 10 : undefined,
    sizeLabel: formatSizeLabel(sizeBytes),
  };
}

export function quantRank(filename: string): number {
  for (const row of QUANT_RANK) {
    if (row.re.test(filename)) return row.rank;
  }
  if (/[-_][fiq]\d/i.test(filename)) return 20;
  return 10;
}

/** Pick one GGUF: best quant that fits budget, else best quant overall, else first. */
export function pickPreferredGgufFile(files: readonly GgufFileHint[], budgetMiB: number): GgufFileHint | undefined {
  if (!files.length) return undefined;
  const scored = [...files].sort((a, b) => {
    const ar = quantRank(a.filename);
    const br = quantRank(b.filename);
    if (ar !== br) return br - ar;
    const as = a.sizeMiB ?? Number.POSITIVE_INFINITY;
    const bs = b.sizeMiB ?? Number.POSITIVE_INFINITY;
    return as - bs;
  });
  const fitting = scored.filter((f) => f.sizeMiB == null || f.sizeMiB <= budgetMiB);
  return fitting[0] ?? scored[0];
}

export function ggufFamily(id: string): string {
  const blob = id.toLowerCase();
  if (blob.includes("qwen")) return "Qwen";
  if (blob.includes("gemma")) return "Gemma";
  if (blob.includes("mistral") || blob.includes("mixtral") || blob.includes("ministral")) return "Mistral";
  if (blob.includes("llama") || blob.includes("tinyllama")) return "Llama";
  if (/(?:^|[^a-z])phi(?:\d|[ab]|\b|-|_)/.test(blob)) return "Phi";
  if (blob.includes("unsloth")) return "Unsloth";
  return "other";
}

/**
 * Budget for one llama-server load: idle discrete GPU VRAM × fit fraction.
 * Dual Arc B70 (~32 GB each) → use max single-card VRAM (not sum), since default pin is one card.
 */
export function ggufFitBudgetMiB(hardware: HardwareSnapshot): number {
  const cards = [
    ...(hardware.accelerators ?? []).map((a) => Number(a.vramMiB) || 0),
    ...(hardware.gpus ?? []).map((g) => Number(g.vramMiB) || 0),
  ].filter((n) => n > 0);
  const single = cards.length ? Math.max(...cards) : 0;
  if (single > 0) return single * HUB_FIT_FRACTION;
  const total = Number(hardware.totalVramMiB) > 0 ? hardware.totalVramMiB : Number(hardware.vramMiB) || 0;
  if (total > 0) return total * HUB_FIT_FRACTION;
  const ram = Math.max(0, Number(hardware.ramMiB) || 0);
  return ram * HUB_FIT_FRACTION;
}

export function buildGgufHubModelsUrl(search: string, limit = HUB_LIMIT, offset = 0): URL {
  const url = new URL(HF_HUB_MODELS_API);
  url.searchParams.set("search", search);
  url.searchParams.set("filter", "gguf");
  url.searchParams.set("sort", "downloads");
  url.searchParams.set("direction", "-1");
  url.searchParams.set("limit", String(limit));
  if (offset > 0) url.searchParams.set("offset", String(offset));
  url.searchParams.append("expand", "siblings");
  // List API omits `gated` unless expanded.
  url.searchParams.append("expand", "gated");
  return url;
}

export function ggufCatalogSearchTerms(q: string): string[] {
  const extra = String(q ?? "")
    .trim()
    .slice(0, QUERY_MAX);
  const terms: string[] = [...GGUF_HUB_SEARCHES];
  if (!extra) return terms;
  const lower = extra.toLowerCase();
  if (!(GGUF_HUB_SEARCHES as readonly string[]).includes(lower)) terms.push(extra);
  return terms;
}

export function isGgufHubCandidate(row: HubRawModel): boolean {
  const id = String(row.id ?? row.modelId ?? "").trim();
  if (!id.includes("/")) return false;
  const tags = (row.tags ?? []).map((t) => String(t).toLowerCase());
  const pipeline = String(row.pipeline_tag ?? "").toLowerCase();
  const library = String(row.library_name ?? "").toLowerCase();
  const sibNames = (row.siblings ?? []).map((s) => String(s.rfilename ?? "").toLowerCase());

  if (
    pipeline === "text-to-image" ||
    pipeline === "image-to-image" ||
    pipeline.includes("diffusion") ||
    tags.some((t) => t === "diffusers" || t.includes("stable-diffusion") || t === "text-to-image")
  ) {
    return false;
  }

  const hasGguf =
    tags.includes("gguf") ||
    library === "gguf" ||
    sibNames.some((n) => n.endsWith(".gguf")) ||
    /\bgguf\b/i.test(id);
  return hasGguf;
}

export function findLocalGgufPath(id: string, preferredFile?: string, root = defaultGgufModelsDir()): string | undefined {
  const dir = localGgufDirForRepo(id, root);
  if (!existsSync(dir)) return undefined;
  let names: string[] = [];
  try {
    names = readdirSync(dir);
  } catch {
    return undefined;
  }
  const ggufs = names.filter((n) => /\.gguf$/i.test(n) && !n.includes(".."));
  if (!ggufs.length) {
    // Nested snapshot layout
    try {
      for (const sub of names) {
        const subPath = join(dir, sub);
        let st;
        try {
          st = statSync(subPath);
        } catch {
          continue;
        }
        if (!st.isDirectory()) continue;
        const nested = readdirSync(subPath).filter((n) => /\.gguf$/i.test(n) && !n.includes(".."));
        if (preferredFile && nested.includes(preferredFile)) return join(subPath, preferredFile);
        if (nested[0]) return join(subPath, nested[0]!);
      }
    } catch {
      /* ignore */
    }
    return undefined;
  }
  if (preferredFile && ggufs.includes(preferredFile)) return join(dir, preferredFile);
  const ranked = [...ggufs].sort((a, b) => quantRank(b) - quantRank(a));
  return join(dir, ranked[0]!);
}

export function toGgufHubModel(row: HubRawModel, budgetMiB: number, ggufDir: string): GgufHubModel | undefined {
  if (!isGgufHubCandidate(row)) return undefined;
  const id = String(row.id ?? row.modelId ?? "").trim();
  const files = (row.siblings ?? [])
    .map(siblingToGgufFile)
    .filter((f): f is GgufFileHint => Boolean(f));
  // List API often omits siblings even with expand=siblings — still list the repo.
  const preferred = files.length ? pickPreferredGgufFile(files, budgetMiB) : undefined;
  const sizeMiB = preferred?.sizeMiB;
  const sizeUnknown = sizeMiB == null;
  const likelyTooBig = !sizeUnknown && (sizeMiB as number) > budgetMiB;
  const localPath = findLocalGgufPath(id, preferred?.filename, ggufDir);
  const gated = hubRowGated(row);
  return {
    id,
    family: ggufFamily(id),
    lastModified: typeof row.lastModified === "string" ? row.lastModified : undefined,
    gated,
    downloads: typeof row.downloads === "number" ? row.downloads : undefined,
    fits: sizeUnknown || !likelyTooBig,
    likelyTooBig,
    sizeUnknown,
    preferredFile: preferred,
    files: files.slice(0, 24),
    localPath,
    downloaded: Boolean(localPath),
    sizeMiB,
    sizeLabel: preferred?.sizeLabel ?? (files.length ? undefined : "GGUF (file list on download)"),
  };
}

function parseHubListPayload(body: unknown): HubRawModel[] | undefined {
  if (Array.isArray(body)) return body as HubRawModel[];
  if (body && typeof body === "object" && Array.isArray((body as { models?: unknown }).models)) {
    return (body as { models: HubRawModel[] }).models;
  }
  return undefined;
}

async function fetchGgufHubPage(options: {
  search: string;
  offset: number;
  fetchFn: HubFetchFn;
  token?: string;
}): Promise<{ rows: HubRawModel[]; error?: string }> {
  const url = buildGgufHubModelsUrl(options.search, HUB_LIMIT, options.offset);
  if (url.origin + url.pathname !== HF_HUB_MODELS_API) {
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
  if (!res.ok) return { rows: [], error: `Hugging Face Hub API HTTP ${res.status}` };
  const rows = parseHubListPayload(parsed);
  if (!rows) return { rows: [], error: "Hugging Face Hub API returned an unexpected payload" };
  return { rows };
}

async function fetchMergedGgufRows(options: {
  q: string;
  fetchFn: HubFetchFn;
  token?: string;
  now: number;
}): Promise<{ rows: HubRawModel[]; error?: string }> {
  const terms = ggufCatalogSearchTerms(options.q);
  const hasToken = Boolean(options.token);
  const jobs = terms.flatMap((search) =>
    Array.from({ length: PAGES }, (_, page) => {
      const offset = page * HUB_LIMIT;
      const key = `${hasToken ? "t" : "p"}:gguf:${search.toLowerCase()}:${offset}`;
      return (async () => {
        const cached = hubCache.get(key);
        if (cached && options.now - cached.at < CACHE_TTL_MS) return { rows: cached.rows };
        const fetched = await fetchGgufHubPage({
          search,
          offset,
          fetchFn: options.fetchFn,
          token: options.token,
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

function matchesGgufQuery(model: GgufHubModel, q: string): boolean {
  if (!q) return true;
  const needle = q.toLowerCase();
  return (
    model.id.toLowerCase().includes(needle) ||
    model.family.toLowerCase().includes(needle) ||
    (model.preferredFile?.filename.toLowerCase().includes(needle) ?? false)
  );
}

export function ggufCatalogHint(budgetGiB: number, offline = false): string {
  const n = Number.isFinite(budgetGiB) ? budgetGiB : 0;
  const base = `GGUF Hub store for llama.cpp on your computer (~${n} GB per idle GPU). Lists filter=gguf repos (Unsloth welcome). Download picks a preferred quant (Q4_K_M when it fits), then Start llama-server on 127.0.0.1:8080. This is not the late-infer safetensors / OpenVINO store, and not the full Hugging Face website.`;
  if (offline) return `Hugging Face Hub is offline or returned no GGUF rows. ${base}`;
  return base;
}

/**
 * List GGUF Hub repos for llama.cpp download/start on this computer.
 */
export async function listGgufHubModels(options: {
  q?: string;
  fetchFn?: HubFetchFn;
  hardware?: HardwareSnapshot;
  token?: string;
  ggufDir?: string;
} = {}): Promise<GgufHubCatalogResponse> {
  try {
    const hardware = options.hardware ?? testDeps.hardware ?? detectHardware();
    const budgetMiB = ggufFitBudgetMiB(hardware);
    const budgetGiB = Math.round((budgetMiB / 1024) * 10) / 10;
    const q = String(options.q ?? "")
      .trim()
      .slice(0, QUERY_MAX);
    const now = (testDeps.now ?? Date.now)();
    const fetchFn = options.fetchFn ?? testDeps.fetchFn ?? globalThis.fetch;
    const token = options.token ?? resolveHfToken();
    const ggufDir = options.ggufDir ?? testDeps.ggufDir ?? defaultGgufModelsDir();
    const fetched = await fetchMergedGgufRows({ q, fetchFn, token, now });
    const offline = Boolean(fetched.error && fetched.rows.length === 0);
    const models: GgufHubModel[] = [];
    for (const row of fetched.rows) {
      try {
        const model = toGgufHubModel(row, budgetMiB, ggufDir);
        if (!model) continue;
        if (!matchesGgufQuery(model, q)) continue;
        models.push(model);
      } catch {
        // Soft-fail bad Hub rows — GGUF list never requires config.json.
        continue;
      }
    }
    models.sort((a, b) => {
      if (a.downloaded !== b.downloaded) return a.downloaded ? -1 : 1;
      if (a.fits !== b.fits) return a.fits ? -1 : 1;
      const ad = a.downloads ?? 0;
      const bd = b.downloads ?? 0;
      if (ad !== bd) return bd - ad;
      return a.id.localeCompare(b.id);
    });
    return {
      models,
      defaultId: models.find((m) => m.fits)?.id ?? models[0]?.id,
      budgetMiB,
      budgetGiB,
      hint: ggufCatalogHint(budgetGiB, offline),
      offline,
      ggufDir,
      ...(offline && fetched.error ? { error: fetched.error } : {}),
    };
  } catch (error) {
    const hardware = options.hardware ?? testDeps.hardware ?? detectHardware();
    const budgetMiB = ggufFitBudgetMiB(hardware);
    const budgetGiB = Math.round((budgetMiB / 1024) * 10) / 10;
    return {
      models: [],
      budgetMiB,
      budgetGiB,
      hint: ggufCatalogHint(budgetGiB, true),
      offline: true,
      ggufDir: options.ggufDir ?? testDeps.ggufDir ?? defaultGgufModelsDir(),
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
