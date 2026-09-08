import { spawn, type ChildProcess, type SpawnOptions } from "node:child_process";
import { existsSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { isPathInside } from "../allowlist.js";
import { engineLibEnv, findLateInferBin, lateInferMissingMessage } from "./bins.js";
import { DEFAULT_LATE_INFER_MODEL } from "./loopback.js";
import { parseHubModelId, parseModelId } from "../identity.js";
import { loadSecretsIntoEnv, resolveHfToken } from "../secrets.js";
import { gatedRepoHint } from "../local-models/download.js";
import { probeHubGatedAccess } from "../local-models/hf-gated.js";
import {
  convertHostRamBlockedReason,
  lateInferCompileBlockedReason,
  mergeGpuPlanEnv,
  openvinoIrPresent,
  readHostRamProbe,
  resolveLateInferGpuPlan,
  vramMaxMiBForHubId,
  weightsMiBForStart,
  INTEL_IR_MISSING_START,
  type GpuPickPlan,
  type GpuPickVendor,
  type LateInferGpuOptions,
} from "./gpu-pick.js";
import { FALLBACK_HUB_SEEDS } from "./hub-catalog.js";

export type LateInferCompilePhase = "downloading" | "probing" | "compiling" | "ready" | "error";

/** Public compile job. No cwd, allowlist, dest, or filesystem path. */
export interface LateInferCompileJob {
  kind: "lateinfer";
  model: string;
  downloading: boolean;
  phase: LateInferCompilePhase;
  message: string;
  error?: string;
  startedAt: number;
  updatedAt: number;
  /** 0–100 Hub fetch (or compile if known). */
  percent?: number;
  bytes?: number;
  totalBytes?: number;
  etaSec?: number;
  /** Detected compile GPU. Never defaults to nvidia. */
  vendor?: GpuPickVendor;
  busId?: string;
  idle?: boolean;
  display?: boolean;
  gpuLabel?: string;
  /** Intel: OpenVINO IR written under compiled/<slug>/openvino/. */
  ir?: boolean;
  gpuReady?: boolean;
}

export type CompileSpawnFn = (command: string, args: readonly string[], options: SpawnOptions) => ChildProcess;

const FS_JOB_KEYS = ["cwd", "allowlist", "dest", "write_dir", "filesystem", "path", "blobDir", "compiledDir"] as const;

let currentJob: LateInferCompileJob | undefined;
let compileChild: ChildProcess | undefined;
let spawnImpl: CompileSpawnFn = spawn as CompileSpawnFn;
let findBinImpl: typeof findLateInferBin = findLateInferBin;
let gpuOptionsImpl: LateInferGpuOptions = {};
let gatedProbeImpl: ((model: string) => Promise<void>) | undefined;

function gpuOpts(extra: LateInferGpuOptions = {}): LateInferGpuOptions {
  return { ...gpuOptionsImpl, ...extra };
}

export function lateInferCompileSpec(
  model?: string,
  gpu: LateInferGpuOptions = {},
): { args: string[]; env: NodeJS.ProcessEnv; gpu: GpuPickPlan } {
  const id = parseModelId(model?.trim() || DEFAULT_LATE_INFER_MODEL);
  const plan = resolveLateInferGpuPlan({
    ...gpuOpts(gpu),
    model: id,
    vramMaxMiB: gpu.vramMaxMiB ?? gpuOptionsImpl.vramMaxMiB ?? vramMaxMiBForHubId(id),
  });
  return {
    args: ["--compile-only", "--model", id],
    env: lateInferCompileEnv(undefined, plan),
    gpu: plan,
  };
}

export function lateHfHomeDir(env: NodeJS.ProcessEnv = process.env): string {
  const explicit = env.LATE_HF_HOME?.trim();
  if (explicit) return explicit;
  return join(dataLocalDir(env), "late", "hf");
}

/** Compiled blobs on this computer (`late-compile.json`). Hub ids only — never a filesystem path. */
export function lateCompiledDir(env: NodeJS.ProcessEnv = process.env): string {
  const explicit = env.LATE_COMPILED_DIR?.trim();
  if (explicit) return explicit;
  return join(dataLocalDir(env), "late", "compiled");
}

function dataLocalDir(env: NodeJS.ProcessEnv): string {
  if (process.platform === "darwin") return join(homedir(), "Library", "Application Support");
  if (process.platform === "win32") {
    return env.LOCALAPPDATA?.trim() || join(homedir(), "AppData", "Local");
  }
  return env.XDG_DATA_HOME?.trim() || join(homedir(), ".local", "share");
}

function asPublicHubId(raw: string | undefined): string | undefined {
  const id = String(raw ?? "").trim();
  if (!id || id.startsWith("/") || id.includes("\\") || id.includes("..") || id.includes("://")) return undefined;
  try {
    return parseHubModelId(id);
  } catch {
    return undefined;
  }
}

function collectCompiledManifestIds(env: NodeJS.ProcessEnv): string[] {
  const ids: string[] = [];
  let entries: Array<{ isDirectory(): boolean; name: string }>;
  try {
    entries = readdirSync(lateCompiledDir(env), { withFileTypes: true });
  } catch {
    return ids;
  }
  for (const ent of entries) {
    if (!ent.isDirectory()) continue;
    const manifestPath = join(lateCompiledDir(env), ent.name, "late-compile.json");
    if (existsSync(manifestPath)) {
      try {
        const parsed = JSON.parse(readFileSync(manifestPath, "utf8")) as { model_id?: unknown; status?: unknown };
        const id = asPublicHubId(typeof parsed.model_id === "string" ? parsed.model_id : undefined);
        const status = typeof parsed.status === "string" ? parsed.status.trim().toLowerCase() : "ok";
        if (id && (status === "ok" || status === "")) ids.push(id);
        continue;
      } catch {
        /* skip broken manifests — downloaded pane is compile-success only */
      }
    }
  }
  return ids;
}

/**
 * Hub ids compiled successfully on this computer (`late-compile.json` status ok).
 * Loopback GUI only — ids, never cwd / dest / compiledDir. Hub cache leftovers are not listed.
 */
export function listCompiledLateInferIds(env: NodeJS.ProcessEnv = process.env): string[] {
  const ids = new Set<string>();
  for (const id of collectCompiledManifestIds(env)) ids.add(id);
  return [...ids].sort((a, b) => a.localeCompare(b));
}

export interface CompiledLateInferRow {
  id: string;
  irPresent: boolean;
  gpuReady: boolean;
  convertOk: boolean;
  convertReason: string;
  vendor?: GpuPickVendor;
}

function compiledRowVendor(manifest: { vendor?: unknown; accel?: unknown }): GpuPickVendor | undefined {
  const raw = typeof manifest.vendor === "string" ? manifest.vendor : typeof manifest.accel === "string" ? manifest.accel : "";
  const v = raw.trim().toLowerCase();
  if (v === "intel" || v === "nvidia" || v === "amd") return v;
  return undefined;
}

/**
 * Downloaded Hub snapshots plus GPU-Start readiness (Intel needs OpenVINO IR).
 * Ids only — never cwd / dest / compiledDir.
 */
export function listCompiledLateInferRows(
  env: NodeJS.ProcessEnv = process.env,
  liveVendor?: GpuPickVendor,
): CompiledLateInferRow[] {
  const root = lateCompiledDir(env);
  let entries: Array<{ isDirectory(): boolean; name: string }>;
  try {
    entries = readdirSync(root, { withFileTypes: true });
  } catch {
    return [];
  }
  const mem = readHostRamProbe();
  const rows: CompiledLateInferRow[] = [];
  for (const ent of entries) {
    if (!ent.isDirectory()) continue;
    const manifestPath = join(root, ent.name, "late-compile.json");
    if (!existsSync(manifestPath)) continue;
    try {
      const parsed = JSON.parse(readFileSync(manifestPath, "utf8")) as {
        model_id?: unknown;
        status?: unknown;
        vendor?: unknown;
        accel?: unknown;
        ir?: unknown;
        serve?: unknown;
      };
      const id = asPublicHubId(typeof parsed.model_id === "string" ? parsed.model_id : undefined);
      const status = typeof parsed.status === "string" ? parsed.status.trim().toLowerCase() : "ok";
      if (!id || (status !== "ok" && status !== "")) continue;
      const vendor = liveVendor ?? compiledRowVendor(parsed);
      const irPresent =
        parsed.ir === true ||
        String(parsed.serve ?? "").toLowerCase() === "openvino-genai" ||
        openvinoIrPresent(id, root);
      const convertReason =
        vendor === "intel" && !irPresent
          ? convertHostRamBlockedReason({
              vendor,
              weightsMiB: weightsMiBForStart(id),
              ovIrPresent: irPresent,
              mem,
            })
          : undefined;
      const gpuReady = vendor === "nvidia" ? true : vendor === "intel" ? irPresent : vendor === "amd" ? false : irPresent;
      rows.push({
        id,
        irPresent,
        gpuReady,
        convertOk: vendor !== "intel" || irPresent || !convertReason,
        convertReason: convertReason ?? "",
        vendor,
      });
    } catch {
      /* skip broken manifests */
    }
  }
  return rows.sort((a, b) => a.id.localeCompare(b.id));
}

/**
 * GPU overlay model for Start readiness (ovIrPresent / hostRam).
 * Prefer an IR-ready compiled snapshot over a serving/YAML id that has no IR on disk —
 * otherwise the Hub default still serving without a compiled/ tree poisons Start for
 * every IR-ready row (Bug: "OpenVINO IR is missing" after successful compile).
 */
export function pickLateInferGpuPlanModel(input: {
  servingModel?: string;
  yamlModel?: string;
  compiledRows: Array<{ id: string; gpuReady?: boolean; irPresent?: boolean }>;
  compiledIds?: string[];
}): string | undefined {
  const rowFor = (id: string | undefined) =>
    id ? input.compiledRows.find((r) => r.id === id || r.id.trim() === id) : undefined;
  const ready = input.compiledRows.find((r) => r.gpuReady)?.id?.trim();
  const ir = input.compiledRows.find((r) => r.irPresent)?.id?.trim();
  const serving = input.servingModel?.trim();
  if (serving) {
    const row = rowFor(serving);
    if (row?.gpuReady || row?.irPresent) return serving;
    // Serving without IR on disk: use an IR-ready compiled id for Start readiness.
    if (ready) return ready;
    if (ir) return ir;
    return serving;
  }
  if (ready) return ready;
  if (ir) return ir;
  const yaml = input.yamlModel?.trim();
  if (yaml) return yaml;
  return input.compiledIds?.find((id) => id.trim())?.trim();
}

export class LateInferCompiledDeleteError extends Error {
  readonly httpStatus: number;
  constructor(message: string, httpStatus: number) {
    super(message);
    this.name = "LateInferCompiledDeleteError";
    this.httpStatus = httpStatus;
  }
}

export function isLateInferCompiledDeleteError(error: unknown): error is LateInferCompiledDeleteError {
  return (
    error instanceof LateInferCompiledDeleteError ||
    (Boolean(error) &&
      typeof error === "object" &&
      (error as { name?: unknown }).name === "LateInferCompiledDeleteError" &&
      typeof (error as { httpStatus?: unknown }).httpStatus === "number")
  );
}

/** Same slug Late writes: `org/name` → `org--name` under the compiled root. */
export function compiledModelSlug(modelId: string): string {
  return parseCompiledModelRef(modelId).replaceAll("/", "--");
}

/** Hub `org/name`, or the compiled dir slug `org--name`. Never a filesystem path. */
export function parseCompiledModelRef(raw: unknown): string {
  const trimmed = typeof raw === "string" ? raw.trim() : "";
  const hub = asPublicHubId(trimmed);
  if (hub) return hub;
  if (
    trimmed &&
    !trimmed.startsWith("/") &&
    !trimmed.includes("\\") &&
    !trimmed.includes("..") &&
    !trimmed.includes("://") &&
    trimmed.includes("--") &&
    !trimmed.includes("/")
  ) {
    const fromSlug = asPublicHubId(trimmed.replace("--", "/"));
    if (fromSlug) return fromSlug;
  }
  throw new LateInferCompiledDeleteError(
    "model must be a Hugging Face Hub org/model id such as Qwen/Qwen2.5-0.5B-Instruct, not a filesystem path",
    400,
  );
}

export function compiledHubIdsMatch(a: string, b: string): boolean {
  const left = String(a ?? "").trim();
  const right = String(b ?? "").trim();
  if (!left || !right) return false;
  if (left === right) return true;
  try {
    return compiledModelSlug(left) === compiledModelSlug(right);
  } catch {
    return left.replaceAll("/", "--") === right.replaceAll("/", "--");
  }
}

export interface LateInferServingState {
  running: boolean;
  models?: string[];
}

export interface DeleteCompiledLateInferResult {
  ok: true;
  model: string;
  deleted: boolean;
  compiledModels: string[];
  compiledRows: CompiledLateInferRow[];
}

function assertCompiledDest(dest: string, root: string): string {
  if (dest === root || dirname(dest) !== root || !isPathInside(dest, root)) {
    throw new LateInferCompiledDeleteError("Refusing to delete outside the compiled snapshots on your computer.", 400);
  }
  return dest;
}

/**
 * Compiled blob dir for a Hub id. Prefers `compiled/<slug>/`, then any sibling
 * folder whose `late-compile.json` model_id matches (listing uses the manifest).
 */
function compiledBlobDirForModel(modelId: string, env: NodeJS.ProcessEnv): string {
  const slug = compiledModelSlug(modelId);
  const root = resolve(lateCompiledDir(env));
  const slugDest = resolve(root, slug);
  if (existsSync(slugDest)) return assertCompiledDest(slugDest, root);
  let entries: Array<{ isDirectory(): boolean; name: string }>;
  try {
    entries = readdirSync(root, { withFileTypes: true });
  } catch {
    return assertCompiledDest(slugDest, root);
  }
  for (const ent of entries) {
    if (!ent.isDirectory() || ent.name.includes("..") || ent.name.includes("/") || ent.name.includes("\\")) continue;
    const dest = resolve(root, ent.name);
    if (dest === root || dirname(dest) !== root || !isPathInside(dest, root)) continue;
    const manifestPath = join(dest, "late-compile.json");
    if (!existsSync(manifestPath)) continue;
    try {
      const parsed = JSON.parse(readFileSync(manifestPath, "utf8")) as { model_id?: unknown };
      const id = asPublicHubId(typeof parsed.model_id === "string" ? parsed.model_id : undefined);
      if (id && compiledHubIdsMatch(id, modelId)) return dest;
    } catch {
      /* skip broken manifests */
    }
  }
  return assertCompiledDest(slugDest, root);
}

export function servingThisCompiledModel(model: string, serving?: LateInferServingState): boolean {
  if (!serving?.running) return false;
  const models = (serving.models ?? []).map((id) => String(id).trim()).filter(Boolean);
  if (models.length === 0) return true;
  return models.some((id) => compiledHubIdsMatch(id, model));
}

/**
 * Remove one compiled Hub snapshot on this computer (`compiled/<slug>/` only).
 * Does not wipe the Hub cache, `bin/late-infer`, or allowlisted workspace dirs.
 * If that snapshot is serving on 127.0.0.1:8010, refuses until Stop (or the GUI/spawn Stop-then-delete path).
 */
export function deleteCompiledLateInfer(options: {
  model: string;
  confirm?: boolean;
  serving?: LateInferServingState;
  env?: NodeJS.ProcessEnv;
}): DeleteCompiledLateInferResult {
  if (options.confirm !== true) {
    throw new LateInferCompiledDeleteError(
      "delete requires confirm=true (this removes the compiled snapshot from your computer).",
      400,
    );
  }
  const model = parseCompiledModelRef(options.model);
  const env = options.env ?? process.env;
  if (
    currentJob &&
    compiledHubIdsMatch(currentJob.model, model) &&
    (currentJob.phase === "downloading" || currentJob.phase === "probing" || currentJob.phase === "compiling")
  ) {
    throw new LateInferCompiledDeleteError(
      "Download is still running for that snapshot on your computer. Wait until it finishes.",
      409,
    );
  }
  if (servingThisCompiledModel(model, options.serving)) {
    throw new LateInferCompiledDeleteError(
      "Stop late-infer on your computer first. That snapshot is currently serving on 127.0.0.1:8010.",
      409,
    );
  }
  const dest = compiledBlobDirForModel(model, env);
  const hfHome = resolve(lateHfHomeDir(env));
  if (dest === hfHome || isPathInside(dest, hfHome)) {
    throw new LateInferCompiledDeleteError("Refusing to delete the Hub cache on your computer.", 400);
  }
  if (!existsSync(dest)) {
    throw new LateInferCompiledDeleteError(`${model} is not compiled on your computer.`, 404);
  }
  rmSync(dest, { recursive: true, force: true });
  if (currentJob && compiledHubIdsMatch(currentJob.model, model)) currentJob = undefined;
  return {
    ok: true,
    model,
    deleted: true,
    compiledModels: listCompiledLateInferIds(env),
    compiledRows: listCompiledLateInferRows(env),
  };
}

function lateInferCompileEnv(bin?: string, plan?: GpuPickPlan): NodeJS.ProcessEnv {
  loadSecretsIntoEnv(true);
  const base = bin ? engineLibEnv(bin) : { ...process.env };
  const env = plan ? mergeGpuPlanEnv(base, plan) : { ...base };
  env.LATE_HF_HOME = lateHfHomeDir(env);
  const token = resolveHfToken();
  if (token && !env.HF_TOKEN?.trim()) env.HF_TOKEN = token;
  // Intel OpenVINO path: skip mlc-llm PATH preflight (CUDA/MLC is not the idle Arc card).
  const accel = String(env.LATE_INFER_ACCEL ?? plan?.env?.LATE_INFER_ACCEL ?? "").toLowerCase();
  if (accel === "intel") {
    env.LATE_INFER_SKIP_MLC_PREFLIGHT = "1";
    env.LATE_INFER_SKIP_MLC = "1";
  }
  return env;
}

function jobGpuFields(plan: GpuPickPlan): Pick<LateInferCompileJob, "vendor" | "busId" | "idle" | "display" | "gpuLabel"> {
  const card = plan.visible[0] ?? plan.primary;
  if (!card) return { gpuLabel: plan.compileTarget || plan.label };
  return {
    vendor: card.vendor,
    busId: card.busId,
    idle: !card.display,
    display: card.display,
    gpuLabel: plan.compileTarget || plan.label,
  };
}

export function lateInferCompileJob(): LateInferCompileJob | undefined {
  return currentJob ? { ...currentJob } : undefined;
}

export function lateInferCompileView(): {
  downloading: boolean;
  phase?: LateInferCompilePhase;
  message?: string;
  error?: string;
  percent?: number;
  bytes?: number;
  totalBytes?: number;
  etaSec?: number;
} {
  const job = currentJob;
  if (!job) return { downloading: false };
  const failed = job.phase === "error" || Boolean(job.error);
  const busy =
    !failed &&
    (job.phase === "downloading" ||
      job.phase === "probing" ||
      job.phase === "compiling");
  return {
    downloading: busy,
    phase: job.phase,
    message: failed ? compileErrorLine(job.error || job.message) : job.message,
    error: failed ? compileErrorLine(job.error || job.message) || job.error : job.error,
    percent: job.percent,
    bytes: job.bytes,
    totalBytes: job.totalBytes,
    etaSec: failed || job.phase === "ready" ? undefined : job.etaSec,
  };
}

export function compileJobHasFsFields(job: object): boolean {
  const rec = job as Record<string, unknown>;
  for (const key of FS_JOB_KEYS) {
    if (key in rec) return true;
  }
  const dump = JSON.stringify(job);
  return (
    dump.includes('"cwd"') ||
    dump.includes('"allowlist"') ||
    dump.includes('"write_dir"') ||
    dump.includes('"filesystem"')
  );
}

/** One-line Hub/compile error. Strips the “downloading Hub snapshot…” prefix so the card cannot look frozen. */
export function compileErrorLine(text: string): string {
  const cleaned = String(text ?? "")
    .replace(/late-infer:\s*downloading Hub snapshot[^\n]*/gi, "")
    .replace(/late-infer:\s*compiling[^\n]*/gi, "")
    .trim();
  const license = cleaned.match(
    /Token is set but Hugging Face still denied access[\s\S]*?Do not commit the token\.|Repo is gated\.[\s\S]*?Do not commit the token\./i,
  );
  if (license) return license[0].replace(/\s+/g, " ").slice(0, 400);
  // Soft-fail Hub repo issues: surface gated / missing config clearly (not a cryptic config.json dump).
  // late-infer often prints "Error: config.json" on its own line before Caused-by / status codes.
  if (/^\s*Error:\s*config\.json\s*$/i.test(cleaned) || (/Error:\s*config\.json\b/i.test(cleaned) && !/\b(401|403|404|status code|request error|denied|not found|authorization required)\b/i.test(cleaned))) {
    return "Could not read Hub config.json for this model — late-infer needs a safetensors Instruct snapshot with config.json. Check the Hub id (wrong or private repos fail here). GGUF packs belong under llama.cpp.";
  }
  if (/config\.json/i.test(cleaned) && /\b401\b|\b403\b|denied|authorization required/i.test(cleaned)) {
    return "Hub denied config.json (gated or private) — accept the model license on Hugging Face, then set a read token in Settings → Local models.";
  }
  if (/config\.json/i.test(cleaned) && /\b404\b|not found/i.test(cleaned)) {
    return "Hub repo has no config.json (missing on Hugging Face) — late-infer needs a safetensors Instruct snapshot with config.json. GGUF-only packs belong under llama.cpp.";
  }
  if (/config\.json/i.test(cleaned) && /request error|status code/i.test(cleaned)) {
    return "Could not read Hub config.json for this model (Hub/network error). Retry, or pick another Instruct id. GGUF packs do not use config.json — use llama.cpp.";
  }
  const lines = cleaned
    .split(/\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const errLine =
    lines.find((line) =>
      /error|404|401|403|fail closed|not a supported|gated|cannot access|cannot compile|intel blob|not treat that card as NVIDIA|xpu|level zero|IR is missing|system RAM|Start refused/i.test(
        line,
      ),
    ) ??
    lines[0] ??
    "";
  return errLine.replace(/\s+/g, " ").slice(0, 400);
}

export function parseCompilePhase(text: string): LateInferCompilePhase | undefined {
  if (/fail closed|not a supported|cannot compile for the Intel|cannot emit an Intel|not treat that card as NVIDIA/i.test(text)) {
    return "error";
  }
  if (
    /^\s*Error:/im.test(text) ||
    /\bstatus code [45]\d\d\b/i.test(text) ||
    /request error/i.test(text) ||
    /\b404\b/.test(text) ||
    /\b401\b/.test(text) ||
    /\b403\b/.test(text) ||
    /denied access/i.test(text)
  ) {
    return "error";
  }
  if (/\bready\b|compiled on your computer/i.test(text) && !/\bcompiling\b/i.test(text)) return "ready";
  if (/\bcompiling\b|compil(e|ation|er)\b/i.test(text) && !/compiled on your computer/i.test(text)) return "compiling";
  if (/\bprob(e|ing)\b|config\.json|gated access|license/i.test(text) && !/download/i.test(text) && !/compil/i.test(text)) {
    return "probing";
  }
  if (/download/i.test(text)) return "downloading";
  return undefined;
}

export interface LateInferCompileProgress {
  phase?: LateInferCompilePhase;
  percent?: number;
  bytes?: number;
  totalBytes?: number;
  etaSec?: number;
}

/** Hub snapshot size from the built-in store seeds (bytes). Unknown ids stay undefined. */
export function hubSnapshotBytes(model: string): number | undefined {
  const needle = String(model ?? "").trim();
  const seed = FALLBACK_HUB_SEEDS.find((row) => row.id === needle);
  if (!seed) return undefined;
  return Math.round(seed.weightsMiB * 1024 * 1024);
}

export function etaSecFromRate(startedAt: number, bytes: number, totalBytes: number, now: number): number | undefined {
  if (!(totalBytes > 0) || !(bytes > 0)) return undefined;
  if (bytes >= totalBytes) return 0;
  const elapsed = (now - startedAt) / 1000;
  if (elapsed < 0.4) return undefined;
  const rate = bytes / elapsed;
  if (!(rate > 0)) return undefined;
  return Math.max(0, Math.round((totalBytes - bytes) / rate));
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function parseFiniteInt(raw: string | undefined): number | undefined {
  if (raw == null) return undefined;
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 0) return undefined;
  return Math.round(n);
}

/** Parse late-infer `--compile-only` progress lines (bytes / total / percent / ETA). */
export function parseCompileProgress(text: string): LateInferCompileProgress | undefined {
  const phase = parseCompilePhase(text);
  const tagged = /\bprogress\s+phase=(downloading|probing|compiling|ready)\b/i.exec(text);
  const bytes = parseFiniteInt(/\bbytes=(\d+)\b/i.exec(text)?.[1]);
  const totalBytes = parseFiniteInt(/\btotal=(\d+)\b/i.exec(text)?.[1]);
  const percentRaw = parseFiniteInt(/\bpercent=(\d+(?:\.\d+)?)\b/i.exec(text)?.[1]);
  const etaSec = parseFiniteInt(/\beta_sec=(\d+)\b/i.exec(text)?.[1]);
  const slash = tagged || /late-infer:/i.test(text) ? /(\d+)\s*\/\s*(\d+)/.exec(text) : null;
  const slashBytes = slash ? parseFiniteInt(slash[1]) : undefined;
  const slashTotal = slash ? parseFiniteInt(slash[2]) : undefined;
  const out: LateInferCompileProgress = {};
  if (phase) out.phase = phase;
  if (bytes != null) out.bytes = bytes;
  else if (slashBytes != null) out.bytes = slashBytes;
  if (totalBytes != null) out.totalBytes = totalBytes;
  else if (slashTotal != null) out.totalBytes = slashTotal;
  if (percentRaw != null) out.percent = clampPercent(percentRaw);
  else if (out.bytes != null && out.totalBytes && out.totalBytes > 0) {
    out.percent = clampPercent((out.bytes / out.totalBytes) * 100);
  }
  if (etaSec != null) out.etaSec = etaSec;
  if (
    out.phase == null &&
    out.bytes == null &&
    out.totalBytes == null &&
    out.percent == null &&
    out.etaSec == null
  ) {
    return undefined;
  }
  return out;
}

function publicJob(job: LateInferCompileJob): LateInferCompileJob {
  return {
    kind: "lateinfer",
    model: job.model,
    downloading: job.downloading,
    phase: job.phase,
    message: job.message,
    error: job.error,
    startedAt: job.startedAt,
    updatedAt: job.updatedAt,
    percent: job.percent,
    bytes: job.bytes,
    totalBytes: job.totalBytes,
    etaSec: job.etaSec,
    vendor: job.vendor,
    busId: job.busId,
    idle: job.idle,
    display: job.display,
    gpuLabel: job.gpuLabel,
    ir: job.ir,
    gpuReady: job.gpuReady,
  };
}

function killCompileChild(): void {
  if (!compileChild || compileChild.exitCode != null || typeof compileChild.kill !== "function") return;
  try {
    compileChild.kill();
  } catch {
    /* ignore */
  }
}

function markJobError(detail: string, now: number): void {
  if (!currentJob) return;
  currentJob.phase = "error";
  currentJob.downloading = false;
  const line = compileErrorLine(detail) || "This Hub repo is not supported by late-infer on your computer.";
  const gated = /401|403|gated|cannot access|access restricted|authorization required|denied access/i.test(detail);
  const hint = gated ? gatedRepoHint(true) : "";
  const already = /Accept the model license|Repo is gated/i.test(line);
  currentJob.error = already ? line : line + hint;
  currentJob.message = currentJob.error;
  if (gated) {
    currentJob.percent = 0;
    currentJob.bytes = 0;
    currentJob.etaSec = undefined;
  }
  currentJob.updatedAt = now;
}

function applyProgressFields(chunk: string, now: number): void {
  if (!currentJob) return;
  const parsed = parseCompileProgress(chunk);
  if (!parsed) return;
  if (typeof parsed.bytes === "number") currentJob.bytes = parsed.bytes;
  if (typeof parsed.totalBytes === "number" && parsed.totalBytes > 0) currentJob.totalBytes = parsed.totalBytes;
  const total = currentJob.totalBytes;
  const bytes = currentJob.bytes;
  if (typeof parsed.percent === "number") currentJob.percent = parsed.percent;
  else if (typeof bytes === "number" && typeof total === "number" && total > 0) {
    currentJob.percent = clampPercent((bytes / total) * 100);
  }
  if (currentJob.phase === "compiling") {
    if (typeof parsed.etaSec === "number" && parsed.phase === "compiling") currentJob.etaSec = parsed.etaSec;
    else if (parsed.phase === "compiling") currentJob.etaSec = undefined;
  } else if (typeof parsed.etaSec === "number") {
    currentJob.etaSec = parsed.etaSec;
  } else if (typeof bytes === "number" && typeof total === "number") {
    currentJob.etaSec = etaSecFromRate(currentJob.startedAt, bytes, total, now);
  }
  if (currentJob.phase === "downloading") {
    const pct = currentJob.percent;
    currentJob.message =
      typeof pct === "number" ? `downloading Hub snapshot… ${pct}%` : "downloading Hub snapshot…";
  }
  currentJob.updatedAt = now;
}

function applyChunk(chunk: string, now: number): void {
  if (!currentJob) return;
  if (currentJob.phase === "error" || currentJob.phase === "ready") return;
  const phase = parseCompilePhase(chunk);
  if (phase === "error") {
    // Bare first line is often just "Error: config.json"; Caused-by / HTTP status follow.
    // Do not kill yet — finishJob upgrades from full stderr.
    if (/^\s*Error:\s*config\.json\s*$/i.test(chunk.trim())) {
      currentJob.message = chunk.trim();
      currentJob.updatedAt = now;
      return;
    }
    markJobError(chunk, now);
    killCompileChild();
    return;
  }
  applyProgressFields(chunk, now);
  if (phase === "compiling" && currentJob.phase === "downloading") {
    currentJob.phase = "compiling";
    currentJob.downloading = true;
    currentJob.message = "compiling on your computer…";
    currentJob.etaSec = undefined;
    if (typeof currentJob.percent === "number" && currentJob.percent < 100) currentJob.percent = 100;
    currentJob.updatedAt = now;
  }
}

function parseCompileOnlyJson(text: string): { ir?: boolean; serve?: string; vendor?: string } | undefined {
  const lines = String(text ?? "")
    .split(/\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i];
    if (!line || !line.startsWith("{")) continue;
    try {
      const parsed = JSON.parse(line) as { ir?: unknown; serve?: unknown; vendor?: unknown; accel?: unknown };
      return {
        ir: parsed.ir === true,
        serve: typeof parsed.serve === "string" ? parsed.serve : undefined,
        vendor: typeof parsed.vendor === "string" ? parsed.vendor : typeof parsed.accel === "string" ? parsed.accel : undefined,
      };
    } catch {
      /* keep scanning */
    }
  }
  return undefined;
}

function finishJob(code: number | null, stderr: string, stdout: string, now: number): void {
  if (!currentJob) return;
  currentJob.updatedAt = now;
  if (currentJob.phase === "error") {
    currentJob.downloading = false;
    // Prefer full stderr — applyChunk may have seen only the first "Error: config.json" line.
    if (stderr.trim()) markJobError(stderr, now);
    else if (!currentJob.error) markJobError(stderr, now);
    return;
  }
  if (code === 0) {
    const parsed = parseCompileOnlyJson(stdout);
    const vendor = (parsed?.vendor ?? currentJob.vendor ?? "").toLowerCase();
    const ir = parsed?.ir === true || String(parsed?.serve ?? "").toLowerCase() === "openvino-genai";
    // Intel GPU Start needs OpenVINO IR. A green exit without IR is not success.
    if (vendor === "intel" && !ir) {
      markJobError(
        `${INTEL_IR_MISSING_START} Download/compile did not produce IR on your computer.`,
        now,
      );
      return;
    }
    currentJob.phase = "ready";
    currentJob.downloading = false;
    currentJob.percent = 100;
    currentJob.etaSec = undefined;
    currentJob.ir = ir;
    currentJob.gpuReady = vendor === "amd" ? false : true;
    currentJob.message = "ready — compiled on your computer (Start when IR is on the idle GPU)";
    currentJob.error = undefined;
    return;
  }
  markJobError(stderr.trim() || `late-infer compile-only exited ${code ?? "null"}`, now);
}

/**
 * Pull a Hub Instruct id on this computer and compile it (late-infer --compile-only).
 * Does not bind :8010. MCP `pull_late_infer` and GUI POST /api/local-servers/download share this.
 */
export async function pullLateInfer(
  options: {
    model: string;
    spawnFn?: CompileSpawnFn;
    findBin?: typeof findLateInferBin;
  } & LateInferGpuOptions,
): Promise<LateInferCompileJob> {
  const model = parseModelId(options.model);
  if (
    currentJob &&
    (currentJob.phase === "downloading" ||
      currentJob.phase === "probing" ||
      currentJob.phase === "compiling")
  ) {
    return publicJob(currentJob);
  }
  const spec = lateInferCompileSpec(model, options);
  if (spec.args.includes("--bind") || spec.args.some((a) => a.includes("8010"))) {
    throw new Error("compile-only must not bind 127.0.0.1:8010");
  }
  const now = Date.now();
  const gpu = jobGpuFields(spec.gpu);
  const totalBytes = hubSnapshotBytes(model);
  currentJob = {
    kind: "lateinfer",
    model,
    downloading: true,
    phase: "probing",
    message: "probing Hub config / license…",
    startedAt: now,
    updatedAt: now,
    percent: 0,
    bytes: 0,
    totalBytes,
    ...gpu,
  };
  const blocked = lateInferCompileBlockedReason(spec.gpu);
  if (blocked) {
    markJobError(blocked, now);
    return publicJob(currentJob);
  }
  const findBin = options.findBin ?? findBinImpl;
  const bin = findBin();
  if (!bin) {
    throw new Error(lateInferMissingMessage());
  }
  try {
    await (gatedProbeImpl ?? probeHubGatedAccess)(model);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    markJobError(message, Date.now());
    return publicJob(currentJob);
  }
  if (currentJob.phase === "error") return publicJob(currentJob);
  currentJob.phase = "downloading";
  currentJob.message = "downloading Hub snapshot…";
  currentJob.updatedAt = Date.now();
  const env = lateInferCompileEnv(bin, spec.gpu);
  const run = options.spawnFn ?? spawnImpl;
  const child = run(bin, spec.args, {
    env,
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });
  compileChild = child;
  let stderr = "";
  let stdout = "";
  let stdoutCarry = "";
  let stderrCarry = "";
  const feed = (incoming: string, carry: string): string => {
    const buf = carry + incoming;
    const parts = buf.split(/\n/);
    const next = parts.pop() ?? "";
    for (const line of parts) applyChunk(line, Date.now());
    return next;
  };
  child.stdout?.on("data", (buf: Buffer | string) => {
    const text = String(buf);
    stdout += text;
    stdoutCarry = feed(text, stdoutCarry);
  });
  child.stderr?.on("data", (buf: Buffer | string) => {
    const text = String(buf);
    stderr += text;
    stderrCarry = feed(text, stderrCarry);
  });
  child.on("error", (err) => {
    stderr += err instanceof Error ? err.message : String(err);
    if (stderrCarry.trim()) applyChunk(stderrCarry, Date.now());
    finishJob(1, stderr, stdout, Date.now());
    compileChild = undefined;
  });
  child.on("close", (code) => {
    if (stdoutCarry.trim()) applyChunk(stdoutCarry, Date.now());
    if (stderrCarry.trim()) applyChunk(stderrCarry, Date.now());
    finishJob(code, stderr, stdout, Date.now());
    compileChild = undefined;
  });
  return publicJob(currentJob);
}

export function resetLateInferCompileForTests(options?: {
  spawnFn?: CompileSpawnFn;
  findBin?: typeof findLateInferBin;
  gpu?: LateInferGpuOptions;
  gatedProbe?: (model: string) => Promise<void>;
}): void {
  if (compileChild && compileChild.exitCode == null && typeof compileChild.kill === "function") {
    try {
      compileChild.kill();
    } catch {
      /* ignore */
    }
  }
  compileChild = undefined;
  currentJob = undefined;
  spawnImpl = options?.spawnFn ?? (spawn as CompileSpawnFn);
  findBinImpl = options?.findBin ?? findLateInferBin;
  gpuOptionsImpl = options?.gpu ?? {};
  gatedProbeImpl = options ? (options.gatedProbe ?? (async () => {})) : undefined;
}
