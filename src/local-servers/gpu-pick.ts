/**
 * Pick an idle / non-display GPU on this computer for late-infer Start.
 * Vendor-agnostic: NVIDIA, AMD, Intel, mixed. DRM + vendor CLIs. No hardcoded SKU size.
 *
 * Product policy:
 * - Primary = idle discrete card with the most VRAM (100%).
 * - Overflow onto display GPU(s) only when estimated vramMaxMiB > primary VRAM.
 * - Display / desktop cards stay at 70% VRAM. Candle cannot apply that cap — fail closed
 *   rather than OOM the compositor. Operator picker can still pin a card.
 * - Vendor from detect (not a hardcoded intel default): NVIDIA → CUDA, Intel → OpenVINO
 *   GenAI / Level Zero when that stack is on your computer, AMD → HIP when wired.
 *   Do not Start on CPU and call it the idle card, and do not silently use CUDA GPU 0.
 */
import { existsSync, readdirSync, readFileSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import {
  bytesToMib,
  estimateIntelSkuVram,
  isIntelIntegratedGpu,
  parseLspciDisplayDevices,
  type AcceleratorVendor,
} from "../hardware.js";
import { which, runCapture, isWindows } from "../platform.js";
import { FALLBACK_HUB_SEEDS, estimateMaxVramMiB, VRAM_MAX_HEADROOM } from "./hub-catalog.js";

/** Fraction of a display GPU’s VRAM that Start may use (leave ~30% for monitors / desktop). */
export const DISPLAY_GPU_VRAM_CAP = 0.7;

/** Start must not mmap 15–26B Hub weights into DRAM and OOM your computer. */
export const HOST_RAM_VS_VRAM =
  "this convert would use system RAM, not idle GPU VRAM";
export const INTEL_IR_MISSING_START =
  "OpenVINO IR is missing. Start refused to protect your computer. Download/compile must produce IR, or Convert only when MemAvailable is safe.";
export const DESKTOP_HEADROOM_MIB = 8_192;
export const CONVERT_HOST_MULT = 2.2;
export const SWAP_MIN_FREE_MIB = 256;
export const SWAP_FULL_TOTAL_MIB = 1_024;
export const GPU_LOAD_STAGING_MIB = 2_048;

export type GpuPickVendor = AcceleratorVendor;

export interface DrmCardProbe {
  name: string;
  vendorId: string;
  deviceId: string;
  busId?: string;
  driver?: string;
  vramBytes?: number;
  vramUsedBytes?: number;
  renderNode?: string;
  connectors: Record<string, string>;
}

export interface GpuPickProbes {
  nvidiaSmi?: string | null;
  rocmSmi?: string | null;
  lspci?: string | null;
  drmCards?: DrmCardProbe[];
  platform?: NodeJS.Platform;
  /** Injected Intel serve stack. Omitted on GPU-only fixtures → missing (tests stay fail-closed). */
  intelRuntime?: IntelRuntimeProbe | null;
  /** Injected proc drm fdinfo blobs (drm-resident-vram0, not system RAM). */
  drmFdinfo?: string[] | null;
  clinfo?: string | null;
  /** Injected /proc/meminfo. Omitted on GPU-only fixtures → skip (live Start reads MemAvailable). */
  memAvailableMiB?: number;
  swapTotalMiB?: number;
  swapFreeMiB?: number;
  ovIrPresent?: boolean;
  /** Injected Vulkan ICD filenames (e.g. intel_icd.json). Omitted → live scan of /usr/share/vulkan/icd.d. */
  vulkanIcds?: string[] | null;
}

/** OpenVINO GenAI (or a future Intel serve stack) on your computer — not a SKU name. */
export interface IntelRuntimeProbe {
  present: boolean;
  kind?: "openvino-genai";
  python?: string;
  label?: string;
}

export interface GpuPickCard {
  id: string;
  vendor: GpuPickVendor;
  name: string;
  /** Index among this vendor’s discrete cards (CUDA / HIP / XPU-style). */
  index: number;
  /**
   * Runtime affinity index in PCI order among this vendor’s Level Zero / CUDA / HIP GPUs.
   * Intel iGPU is not a Level Zero GPU (i915 vs xe discrete) — it does not consume a slot.
   * Dual B70 + Arrow Lake iGPU: display B70 = 0, idle B70 = 1 (not 2).
   */
  vendorAllIndex: number;
  busId?: string;
  deviceId?: string;
  vramMiB: number;
  freeVramMiB?: number;
  /** Resident VRAM on this PCI card (not host RAM / GTT / drm-resident-system). */
  usedVramMiB?: number;
  usedVramSource?: string;
  display: boolean;
  displayUnknown: boolean;
  connectedDisplays: number;
  igpu: boolean;
  renderNode?: string;
  source: string;
}

export interface GpuPickPlan {
  cards: GpuPickCard[];
  discrete: GpuPickCard[];
  primary?: GpuPickCard;
  overflow?: GpuPickCard[];
  visible: GpuPickCard[];
  /** Env vars for spawn only (not a copy of process.env). */
  env: Record<string, string>;
  needsPicker: boolean;
  overflowWanted: boolean;
  overflowBlocked: boolean;
  capApplied: boolean;
  displayCapPercent: number;
  runtimeOk: boolean;
  runtimeReason: string;
  /** `openvino-genai` / `candle-cuda` when Start can use that vendor’s engine. */
  runtimeKind: string;
  /** Shown when Intel Start serves Hub safetensors instead of a CUDA blob. */
  serveHint: string;
  reason: string;
  label: string;
  useAllGpus: boolean;
  /** False when compile cannot honor this GPU (Intel/AMD, picker, or 70% overflow). */
  compileOk: boolean;
  compileReason: string;
  compileTarget: string;
  /** False when convert/load would mmap weights into DRAM (OOM your computer). */
  hostRamOk: boolean;
  hostRamReason: string;
  /** Intel: OpenVINO IR on disk. NVIDIA/AMD unused (false). */
  ovIrPresent: boolean;
  /** True when a Hub id (or injected probe) was checked for IR. */
  ovIrChecked: boolean;
}

export interface LateInferGpuOptions {
  useAllGpus?: boolean;
  /** Explicit picker id (`intel:0000:08:00.0`). Overrides auto idle-primary. */
  gpuId?: string;
  vramMaxMiB?: number;
  /** Hub id for OpenVINO IR lookup and weight-size refuse. */
  model?: string;
  cards?: GpuPickCard[];
  probes?: GpuPickProbes;
}

export type GpuDeviceKind = "intel-xpu" | "cuda" | "hip" | "cpu" | "";
export type GpuServePhase = "down" | "starting" | "gpu-running" | "host-ram" | "exited" | "ready";

/** Health / GUI overlay from late-infer :8010 plus owned pid. */
export interface GpuServeLiveInput {
  processAlive?: boolean;
  pidFilePresent?: boolean;
  ready?: boolean;
  running?: boolean;
  device?: string;
  deviceKind?: GpuDeviceKind | string;
  weightsInHostRam?: boolean;
  servePhase?: GpuServePhase | string;
}

const NVIDIA_VENDOR = "10de";
const AMD_VENDOR = "1002";
const INTEL_VENDOR = "8086";

let liveCache: { at: number; cards: GpuPickCard[] } | undefined;

export function normalizePciBusId(raw: string | undefined): string {
  if (!raw?.trim()) return "";
  const t = raw.trim().toLowerCase().replace(/^pci@/i, "");
  const match = /(?:([0-9a-f]{2,8}):)?([0-9a-f]{2}):([0-9a-f]{2})\.([0-9a-f])/.exec(t);
  if (!match) return t;
  const domain = (match[1] ?? "0000").slice(-4).padStart(4, "0");
  return `${domain}:${match[2]}:${match[3]}.${match[4]}`;
}

export function gpuCardId(vendor: GpuPickVendor, busId: string | undefined, index: number): string {
  const bus = normalizePciBusId(busId);
  return bus ? `${vendor}:${bus}` : `${vendor}:${index}`;
}

/** Candle CUDA for NVIDIA. Intel Start needs OpenVINO GenAI on your computer. AMD HIP is not wired. */
export function lateInferSupportsVendor(
  vendor: GpuPickVendor | undefined,
  intelRuntime?: IntelRuntimeProbe,
): boolean {
  if (vendor === "nvidia") return true;
  if (vendor === "intel") return intelRuntime?.present === true;
  return false;
}

function lateDataDir(): string {
  if (process.platform === "darwin") return join(homedir(), "Library", "Application Support", "late");
  if (process.platform === "win32") {
    return join(process.env.LOCALAPPDATA?.trim() || join(homedir(), "AppData", "Local"), "late");
  }
  return join(process.env.XDG_DATA_HOME?.trim() || join(homedir(), ".local", "share"), "late");
}

export interface HostRamProbe {
  memAvailableMiB: number;
  swapTotalMiB: number;
  swapFreeMiB: number;
}

export function parseMeminfo(text: string): HostRamProbe | undefined {
  let available: number | undefined;
  let swapTotal = 0;
  let swapFree = 0;
  for (const line of text.split(/\r?\n/)) {
    const [key, raw] = line.split(/\s+/);
    const kib = Number.parseInt(raw ?? "", 10);
    if (!Number.isFinite(kib)) continue;
    const mib = Math.round(kib / 1024);
    if (key === "MemAvailable:") available = mib;
    else if (key === "SwapTotal:") swapTotal = mib;
    else if (key === "SwapFree:") swapFree = mib;
  }
  if (available === undefined) return undefined;
  return { memAvailableMiB: available, swapTotalMiB: swapTotal, swapFreeMiB: swapFree };
}

export function readHostRamProbe(): HostRamProbe | undefined {
  try {
    return parseMeminfo(readFileSync("/proc/meminfo", "utf8"));
  } catch {
    return undefined;
  }
}

export function slugHubModelId(id: string): string {
  return id.trim().replaceAll("/", "--");
}

export function openvinoIrPresent(modelId: string, compiledRoot?: string): boolean {
  const root = compiledRoot ?? (process.env.LATE_COMPILED_DIR?.trim() || join(lateDataDir(), "compiled"));
  return existsSync(join(root, slugHubModelId(modelId), "openvino", "openvino_model.xml"));
}

export function weightsMiBForStart(model?: string, vramMaxMiB?: number): number | undefined {
  const needle = model?.trim();
  if (needle) {
    const seed = FALLBACK_HUB_SEEDS.find((row) => row.id === needle);
    if (seed) return seed.weightsMiB;
  }
  if (typeof vramMaxMiB === "number" && Number.isFinite(vramMaxMiB) && vramMaxMiB > 0) {
    return Math.round(vramMaxMiB / (1 + VRAM_MAX_HEADROOM));
  }
  return undefined;
}

export function convertPeakMiB(weightsMiB: number): number {
  return Math.ceil(weightsMiB * CONVERT_HOST_MULT);
}

/** Convert/Download IR export — RAM/swap only. Start uses {@link hostRamBlockedReason}. */
export function convertHostRamBlockedReason(input: {
  vendor?: GpuPickVendor;
  weightsMiB?: number;
  ovIrPresent: boolean;
  mem?: HostRamProbe;
}): string | undefined {
  const mem = input.mem;
  if (!mem) return undefined;
  if (mem.swapTotalMiB >= SWAP_FULL_TOTAL_MIB && mem.swapFreeMiB < SWAP_MIN_FREE_MIB) {
    return `${HOST_RAM_VS_VRAM} (swap is full on your computer; Start would OOM the desktop)`;
  }
  if (input.vendor !== "intel" || input.ovIrPresent) return undefined;
  const weights = input.weightsMiB;
  if (!weights || weights <= 0) {
    return `${HOST_RAM_VS_VRAM} (OpenVINO IR is missing; convert would mmap Hub safetensors into DRAM)`;
  }
  const peak = convertPeakMiB(weights);
  if (peak + DESKTOP_HEADROOM_MIB > mem.memAvailableMiB) {
    return `${HOST_RAM_VS_VRAM} (convert needs ~${peak} MiB host RAM plus desktop headroom; MemAvailable is ${mem.memAvailableMiB} MiB)`;
  }
  return undefined;
}

/** Fail closed before OpenVINO convert / Candle CPU mmap can OOM your computer. */
export function hostRamBlockedReason(input: {
  vendor?: GpuPickVendor;
  runtimeOk?: boolean;
  weightsMiB?: number;
  ovIrPresent: boolean;
  mem?: HostRamProbe;
}): string | undefined {
  const mem = input.mem;
  if (!mem) return undefined;
  if (mem.swapTotalMiB >= SWAP_FULL_TOTAL_MIB && mem.swapFreeMiB < SWAP_MIN_FREE_MIB) {
    return `${HOST_RAM_VS_VRAM} (swap is full on your computer; Start would OOM the desktop)`;
  }
  const weights = input.weightsMiB;
  const vendor = input.vendor;
  const intelConvert = vendor === "intel" && !input.ovIrPresent;
  if (intelConvert) {
    const convert = convertHostRamBlockedReason(input);
    if (convert) return convert;
    return `${HOST_RAM_VS_VRAM} (OpenVINO IR is missing; Start will not convert in DRAM)`;
  }
  if (vendor === "intel" && input.ovIrPresent) {
    if (GPU_LOAD_STAGING_MIB + DESKTOP_HEADROOM_MIB > mem.memAvailableMiB) {
      return `${HOST_RAM_VS_VRAM} (MemAvailable is too low to stage weights onto idle GPU VRAM)`;
    }
    return undefined;
  }
  if (vendor === "nvidia" && input.runtimeOk) {
    if (GPU_LOAD_STAGING_MIB + DESKTOP_HEADROOM_MIB > mem.memAvailableMiB) {
      return `${HOST_RAM_VS_VRAM} (MemAvailable is too low to stage weights onto idle GPU VRAM)`;
    }
    return undefined;
  }
  if (weights && weights + DESKTOP_HEADROOM_MIB > mem.memAvailableMiB) {
    return HOST_RAM_VS_VRAM;
  }
  return undefined;
}

function resolveHostRam(probes?: GpuPickProbes): HostRamProbe | undefined {
  if (probes && "memAvailableMiB" in probes) {
    return {
      memAvailableMiB: probes.memAvailableMiB ?? 0,
      swapTotalMiB: probes.swapTotalMiB ?? 0,
      swapFreeMiB: probes.swapFreeMiB ?? 0,
    };
  }
  if (probes !== undefined) return undefined;
  return readHostRamProbe();
}

function pythonHasOpenvinoGenai(python: string): boolean {
  const out = runCapture(python, ["-c", "import openvino_genai, openvino; print('openvino-genai')"], 15_000);
  return Boolean(out && out.includes("openvino-genai"));
}

/** Live-detect Intel serve stacks on your computer. OpenVINO GenAI is the wired path. */
export function detectLiveIntelRuntime(): IntelRuntimeProbe {
  const cands = [
    process.env.LATE_INFER_INTEL_PYTHON?.trim(),
    join(lateDataDir(), "intel-ov", "bin", "python"),
    join(lateDataDir(), "intel-ov", "bin", "python3"),
    join(lateDataDir(), "intel-ov", "bin", "python.exe"),
    which("python3.12"),
    which("python3"),
  ].filter((p): p is string => Boolean(p));
  for (const python of cands) {
    if (!existsSync(python)) continue;
    if (pythonHasOpenvinoGenai(python)) {
      return {
        present: true,
        kind: "openvino-genai",
        python,
        label: "OpenVINO GenAI / Level Zero",
      };
    }
  }
  return { present: false };
}

/**
 * Injected GPU fixtures omit intelRuntime → missing (NVIDIA/AMD/Intel fail-closed tests).
 * Live Start (no probes) detects OpenVINO on your computer.
 */
export function resolveIntelRuntime(probes?: GpuPickProbes): IntelRuntimeProbe {
  if (probes && "intelRuntime" in probes) {
    return probes.intelRuntime ?? { present: false };
  }
  if (probes !== undefined) {
    return { present: false };
  }
  return detectLiveIntelRuntime();
}

export function vramMaxMiBForHubId(id: string, override?: number): number | undefined {
  if (typeof override === "number" && Number.isFinite(override) && override > 0) return Math.round(override);
  const needle = id.trim();
  const seed = FALLBACK_HUB_SEEDS.find((row) => row.id === needle);
  if (seed) return estimateMaxVramMiB(seed.weightsMiB);
  return undefined;
}

export function formatGpuVramLabel(vramMiB: number): string {
  if (!Number.isFinite(vramMiB) || vramMiB <= 0) return "VRAM unknown";
  const gib = vramMiB / 1024;
  const shown = gib >= 10 ? String(Math.round(gib)) : (Math.round(gib * 10) / 10).toFixed(1).replace(/\.0$/, "");
  return `${shown} GB`;
}

/** Human vendor for GUI / compile target. Never assume NVIDIA. */
export function formatGpuVendorLabel(vendor: GpuPickVendor | undefined): string {
  if (vendor === "intel") return "Intel (Arc / XPU / Level Zero)";
  if (vendor === "amd") return "AMD";
  if (vendor === "nvidia") return "NVIDIA";
  return "CPU";
}

type ProbeRow = Partial<GpuPickCard> & {
  vendor: GpuPickVendor;
  name: string;
  /** CUDA / HIP index from nvidia-smi or rocm-smi. PCI order is the fallback. */
  vendorRuntimeIndex?: number;
};

function isGenericGpuName(name: string): boolean {
  return /^(nvidia|amd|intel) GPU$/i.test(name) || /^Intel [0-9a-f]{3,4}$/i.test(name);
}

export function parseRocmSmiGpuQuery(text: string): Array<{
  index: number;
  name: string;
  vramMiB: number;
  busId?: string;
}> {
  const byIndex = new Map<number, { index: number; name: string; vramMiB: number; busId?: string }>();
  const row = (index: number) => {
    let current = byIndex.get(index);
    if (!current) {
      current = { index, name: `AMD GPU ${index}`, vramMiB: 0 };
      byIndex.set(index, current);
    }
    return current;
  };
  const vramBytesRe = /GPU\[(\d+)\][^\n]*vram Total Memory \(B\):\s*(\d+)/gi;
  let match: RegExpExecArray | null;
  while ((match = vramBytesRe.exec(text))) {
    const idx = Number(match[1]);
    const vram = bytesToMib(Number(match[2]));
    if (!Number.isInteger(idx) || idx < 0 || vram <= 0) continue;
    row(idx).vramMiB = vram;
  }
  const vramAltRe = /GPU\[(\d+)\][^\n]*VRAM[^\n]*?(\d+)\s*(MiB|MB|GiB|GB)/gi;
  while ((match = vramAltRe.exec(text))) {
    const idx = Number(match[1]);
    let vram = Number(match[2]);
    if ((match[3] ?? "").toLowerCase().startsWith("g")) vram *= 1024;
    if (!Number.isInteger(idx) || idx < 0 || !Number.isFinite(vram) || vram <= 0) continue;
    const current = row(idx);
    if (current.vramMiB <= 0) current.vramMiB = Math.round(vram);
  }
  const busRe = /GPU\[(\d+)\][^\n]*(?:PCI Bus|pci)[:\s]+([0-9a-f:.]+)/gi;
  while ((match = busRe.exec(text))) {
    const idx = Number(match[1]);
    const bus = normalizePciBusId(match[2]);
    if (!Number.isInteger(idx) || idx < 0 || !bus) continue;
    row(idx).busId = bus;
  }
  const nameRe = /GPU\[(\d+)\][^\n]*(?:Card Series|Device Name|Market Name)[:\s]+(.+)$/gim;
  while ((match = nameRe.exec(text))) {
    const idx = Number(match[1]);
    const name = match[2]?.trim();
    if (!Number.isInteger(idx) || idx < 0 || !name) continue;
    row(idx).name = name;
  }
  return [...byIndex.values()].filter((gpu) => gpu.vramMiB > 0 || gpu.busId).sort((a, b) => a.index - b.index);
}

export function parseNvidiaSmiGpuQuery(csv: string): Array<{
  index: number;
  name: string;
  vramMiB: number;
  freeVramMiB?: number;
  busId?: string;
  displayActive?: boolean;
}> {
  const rows: Array<{
    index: number;
    name: string;
    vramMiB: number;
    freeVramMiB?: number;
    busId?: string;
    displayActive?: boolean;
  }> = [];
  for (const raw of csv.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || /^index\s*,/i.test(line)) continue;
    const parts = line.split(",").map((part) => part.trim());
    if (parts.length < 2) continue;
    let idx = Number(parts[0]);
    let shift = 0;
    if (!Number.isInteger(idx)) {
      idx = rows.length;
      shift = -1;
    }
    const name = parts[1 + shift];
    const vram = Number(parts[2 + shift]);
    if (!name || !Number.isFinite(vram) || vram <= 0) continue;
    const freeRaw = parts[3 + shift];
    const free = Number(freeRaw);
    const busPart = parts.find((part) => /[0-9a-f]{2,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]/i.test(part));
    const displayPart = parts.find((part) => /^(enabled|disabled)$/i.test(part));
    rows.push({
      index: idx,
      name,
      vramMiB: Math.round(vram),
      freeVramMiB: Number.isFinite(free) && free >= 0 ? Math.round(free) : undefined,
      busId: busPart ? normalizePciBusId(busPart) : undefined,
      displayActive: displayPart ? /^enabled$/i.test(displayPart) : undefined,
    });
  }
  return rows;
}

/** KiB/MiB/GiB/B token from drm fdinfo or vendor CLI. */
export function parseMemSizeToMib(raw: string | undefined): number {
  if (!raw?.trim()) return 0;
  const match = /(\d+(?:\.\d+)?)\s*(KiB|MiB|GiB|TiB|kB|MB|GB|TB|B)?/i.exec(raw.trim());
  if (!match) return 0;
  let n = Number(match[1]);
  if (!Number.isFinite(n) || n < 0) return 0;
  const unit = (match[2] ?? "B").toLowerCase();
  if (unit === "b") n /= 1024 * 1024;
  else if (unit === "kib" || unit === "kb") n /= 1024;
  else if (unit === "mib" || unit === "mb") n = n;
  else if (unit === "gib" || unit === "gb") n *= 1024;
  else if (unit === "tib" || unit === "tb") n *= 1024 * 1024;
  return Math.round(n);
}

export interface DrmFdinfoClient {
  pci: string;
  clientId: string;
  /** Device local memory (xe vram0 / amdgpu vram). Never GTT/system. */
  vramKiB: number;
  /** Host RAM / GTT mapping — not the idle card’s VRAM. */
  systemKiB: number;
}

/**
 * Parse one `/proc/<pid>/fdinfo/<fd>` DRM client.
 * Intel xe: `drm-resident-vram0`. AMD: `drm-memory-vram`. Ignore `drm-resident-system`.
 */
export function parseDrmFdinfo(text: string): DrmFdinfoClient | undefined {
  const pciRaw = /^drm-pdev:\s*(\S+)/im.exec(text)?.[1];
  const pci = normalizePciBusId(pciRaw);
  if (!pci) return undefined;
  const clientId = /^drm-client-id:\s*(\S+)/im.exec(text)?.[1] ?? pci;
  const vramLine =
    /^drm-resident-vram0:\s*(.+)$/im.exec(text)?.[1] ??
    /^drm-total-vram0:\s*(.+)$/im.exec(text)?.[1] ??
    /^drm-resident-vram:\s*(.+)$/im.exec(text)?.[1] ??
    /^drm-memory-vram:\s*(.+)$/im.exec(text)?.[1];
  const systemLine =
    /^drm-resident-system:\s*(.+)$/im.exec(text)?.[1] ??
    /^drm-total-system:\s*(.+)$/im.exec(text)?.[1] ??
    /^drm-memory-gtt:\s*(.+)$/im.exec(text)?.[1];
  const vramMiB = vramLine ? parseMemSizeToMib(vramLine) : 0;
  const systemMiB = systemLine ? parseMemSizeToMib(systemLine) : 0;
  return {
    pci,
    clientId,
    vramKiB: vramMiB * 1024,
    systemKiB: systemMiB * 1024,
  };
}

export function usedVramMiBFromFdinfo(clients: DrmFdinfoClient[], busId: string | undefined): number {
  const pci = normalizePciBusId(busId);
  if (!pci) return 0;
  const seen = new Set<string>();
  let totalKiB = 0;
  for (const client of clients) {
    if (normalizePciBusId(client.pci) !== pci) continue;
    const key = `${client.pci}:${client.clientId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    totalKiB += client.vramKiB;
  }
  return totalKiB > 0 ? Math.round(totalKiB / 1024) : 0;
}

export function systemResidentMiBFromFdinfo(clients: DrmFdinfoClient[], busId: string | undefined): number {
  const pci = normalizePciBusId(busId);
  if (!pci) return 0;
  const seen = new Set<string>();
  let totalKiB = 0;
  for (const client of clients) {
    if (normalizePciBusId(client.pci) !== pci) continue;
    const key = `${client.pci}:${client.clientId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    totalKiB += client.systemKiB;
  }
  return totalKiB > 0 ? Math.round(totalKiB / 1024) : 0;
}

export function parseRocmSmiUsedQuery(text: string): Array<{ index: number; usedMiB: number; vramMiB: number; busId?: string }> {
  const byIndex = new Map<number, { index: number; usedMiB: number; vramMiB: number; busId?: string }>();
  const row = (index: number) => {
    let current = byIndex.get(index);
    if (!current) {
      current = { index, usedMiB: 0, vramMiB: 0 };
      byIndex.set(index, current);
    }
    return current;
  };
  const usedRe = /GPU\[(\d+)\][^\n]*vram (?:Total )?Used Memory \(B\):\s*(\d+)/gi;
  let match: RegExpExecArray | null;
  while ((match = usedRe.exec(text))) {
    const idx = Number(match[1]);
    const used = bytesToMib(Number(match[2]));
    if (!Number.isInteger(idx) || idx < 0 || used < 0) continue;
    row(idx).usedMiB = used;
  }
  const totalRe = /GPU\[(\d+)\][^\n]*vram Total Memory \(B\):\s*(\d+)/gi;
  while ((match = totalRe.exec(text))) {
    const idx = Number(match[1]);
    const vram = bytesToMib(Number(match[2]));
    if (!Number.isInteger(idx) || idx < 0 || vram <= 0) continue;
    row(idx).vramMiB = vram;
  }
  const busRe = /GPU\[(\d+)\][^\n]*(?:PCI Bus|pci)[:\s]+([0-9a-f:.]+)/gi;
  while ((match = busRe.exec(text))) {
    const idx = Number(match[1]);
    const bus = normalizePciBusId(match[2]);
    if (!Number.isInteger(idx) || idx < 0 || !bus) continue;
    row(idx).busId = bus;
  }
  return [...byIndex.values()].sort((a, b) => a.index - b.index);
}

/** clinfo Global memory for discrete GPUs. iGPU “Global memory” is host RAM — skip it. */
export function parseClinfoDiscreteVram(text: string): Array<{ name: string; busId?: string; vramMiB: number }> {
  const rows: Array<{ name: string; busId?: string; vramMiB: number }> = [];
  const blocks = text.split(/(?=Device Name\s+)/i);
  for (const block of blocks) {
    const name = /^\s*Device Name\s+(.+)$/im.exec(block)?.[1]?.trim();
    if (!name) continue;
    if (isIntelIntegratedGpu(name)) continue;
    const busRaw =
      /PCI bus info[^\n]*?([0-9a-f]{2,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])/i.exec(block)?.[1] ??
      /PCI-E,\s*([0-9a-f]{2,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f])/i.exec(block)?.[1];
    const mem = /Global memory size\s+(\d+)/i.exec(block);
    const vramMiB = mem ? bytesToMib(Number(mem[1])) : 0;
    if (vramMiB <= 0) continue;
    rows.push({ name, busId: busRaw ? normalizePciBusId(busRaw) : undefined, vramMiB });
  }
  return rows;
}

export function classifyLateInferDevice(device: string | undefined): {
  deviceKind: GpuDeviceKind;
  accel: string;
  gpuRunning: boolean;
  weightsInHostRam: boolean;
} {
  const d = String(device ?? "").toLowerCase();
  if (!d.trim()) {
    return { deviceKind: "", accel: "", gpuRunning: false, weightsInHostRam: false };
  }
  if (/\bcpu\b/.test(d) && !/xpu|cuda|hip|openvino|gpu/.test(d)) {
    return { deviceKind: "cpu", accel: "cpu", gpuRunning: false, weightsInHostRam: true };
  }
  if (/intel-xpu|openvino|level.?zero|\bxpu\b/.test(d)) {
    return { deviceKind: "intel-xpu", accel: "intel", gpuRunning: true, weightsInHostRam: false };
  }
  if (/\bcuda\b|nvidia/.test(d)) {
    return { deviceKind: "cuda", accel: "nvidia", gpuRunning: true, weightsInHostRam: false };
  }
  if (/\bhip\b|\brocm\b/.test(d)) {
    return { deviceKind: "hip", accel: "amd", gpuRunning: true, weightsInHostRam: false };
  }
  return { deviceKind: "", accel: "", gpuRunning: false, weightsInHostRam: false };
}

export function resolveGpuServePhase(input: GpuServeLiveInput = {}): GpuServePhase {
  if (
    input.servePhase === "down" ||
    input.servePhase === "starting" ||
    input.servePhase === "gpu-running" ||
    input.servePhase === "host-ram" ||
    input.servePhase === "exited" ||
    input.servePhase === "ready"
  ) {
    return input.servePhase;
  }
  const classified = classifyLateInferDevice(input.device);
  const kind = (input.deviceKind as GpuDeviceKind | undefined) || classified.deviceKind;
  const hostRam = input.weightsInHostRam === true || classified.weightsInHostRam;
  if (input.ready) {
    if (hostRam || kind === "cpu") return "host-ram";
    if (kind === "intel-xpu" || kind === "cuda" || kind === "hip" || classified.gpuRunning) return "gpu-running";
    return "ready";
  }
  if (input.processAlive) return "starting";
  if (input.pidFilePresent && !input.running) return "exited";
  return "down";
}

export function formatGpuVramPair(usedMiB: number | undefined, totalMiB: number | undefined): string {
  const total = formatGpuVramLabel(totalMiB ?? 0);
  if (usedMiB == null || !Number.isFinite(usedMiB) || usedMiB < 0) {
    return total === "VRAM unknown" ? "VRAM unknown" : `? / ${total}`;
  }
  const used = formatGpuVramLabel(usedMiB);
  if (total === "VRAM unknown") return `${used} used`;
  return `${used} / ${total}`;
}

function collectLiveDrmFdinfo(): string[] {
  const proc = "/proc";
  if (!existsSync(proc)) return [];
  const blobs: string[] = [];
  let pids: string[] = [];
  try {
    pids = readdirSync(proc);
  } catch {
    return [];
  }
  for (const pid of pids) {
    if (!/^\d+$/.test(pid)) continue;
    const fdDir = join(proc, pid, "fd");
    const infoDir = join(proc, pid, "fdinfo");
    if (!existsSync(fdDir) || !existsSync(infoDir)) continue;
    let fds: string[] = [];
    try {
      fds = readdirSync(fdDir);
    } catch {
      continue;
    }
    const seenClients = new Set<string>();
    for (const fd of fds) {
      let target = "";
      try {
        target = realpathSync(join(fdDir, fd));
      } catch {
        continue;
      }
      if (!/\/dri\/(card|renderD)\d+$/.test(target) && !/\/dri\//.test(target)) continue;
      try {
        const text = readFileSync(join(infoDir, fd), "utf8");
        if (!/^drm-pdev:/im.test(text) && !/^drm-driver:/im.test(text)) continue;
        const parsed = parseDrmFdinfo(text);
        const key = parsed ? `${parsed.pci}:${parsed.clientId}` : `${pid}:${fd}`;
        if (seenClients.has(key)) continue;
        seenClients.add(key);
        blobs.push(text);
      } catch {
        /* unreadable fdinfo */
      }
    }
  }
  return blobs;
}

function readDrmVramUsedBytes(deviceDir: string): number | undefined {
  const files: string[] = [
    join(deviceDir, "mem_info_vram_used"),
    join(deviceDir, "mem_info_vis_vram_used"),
  ];
  try {
    const drmDir = join(deviceDir, "drm");
    if (existsSync(drmDir)) {
      for (const entry of readdirSync(drmDir)) {
        if (/^card\d+$/.test(entry)) files.push(join(drmDir, entry, "lmem_used_bytes"));
      }
    }
  } catch {
    /* ignore */
  }
  let used = 0;
  for (const file of files) {
    if (!existsSync(file)) continue;
    try {
      const n = Number(readFileSync(file, "utf8").trim());
      if (Number.isFinite(n) && n > 0) used = Math.max(used, n);
    } catch {
      /* ignore */
    }
  }
  return used > 0 ? used : undefined;
}

export function collectLiveDrmCards(): DrmCardProbe[] {
  const drm = "/sys/class/drm";
  if (!existsSync(drm)) return [];
  let names: string[] = [];
  try {
    names = readdirSync(drm);
  } catch {
    return [];
  }
  const cards: DrmCardProbe[] = [];
  for (const name of names) {
    if (!/^card\d+$/.test(name)) continue;
    const deviceDir = join(drm, name, "device");
    try {
      const vendorId = readHexId(join(deviceDir, "vendor"));
      const deviceId = readHexId(join(deviceDir, "device"));
      if (!vendorId || !deviceId) continue;
      let busId: string | undefined;
      let driver: string | undefined;
      try {
        const uevent = readFileSync(join(deviceDir, "uevent"), "utf8");
        driver = /^DRIVER=(.+)$/m.exec(uevent)?.[1]?.trim();
        busId = normalizePciBusId(/^PCI_SLOT_NAME=(.+)$/m.exec(uevent)?.[1]) || undefined;
      } catch {
        /* ignore */
      }
      const connectors: Record<string, string> = {};
      for (const entry of names) {
        if (!entry.startsWith(`${name}-`)) continue;
        try {
          connectors[entry] = readFileSync(join(drm, entry, "status"), "utf8").trim().toLowerCase();
        } catch {
          /* ignore */
        }
      }
      cards.push({
        name,
        vendorId,
        deviceId,
        busId,
        driver,
        vramBytes: readDrmVramBytes(deviceDir),
        vramUsedBytes: readDrmVramUsedBytes(deviceDir),
        connectors,
      });
    } catch {
      /* skip unreadable card */
    }
  }
  const renderByPci = new Map<string, string>();
  for (const name of names) {
    if (!/^renderD\d+$/.test(name)) continue;
    try {
      const uevent = readFileSync(join(drm, name, "device", "uevent"), "utf8");
      const pci = normalizePciBusId(/^PCI_SLOT_NAME=(.+)$/m.exec(uevent)?.[1]);
      if (pci) renderByPci.set(pci, name);
    } catch {
      try {
        const resolved = realpathSync(join(drm, name, "device"));
        const uevent = readFileSync(join(resolved, "uevent"), "utf8");
        const pci = normalizePciBusId(/^PCI_SLOT_NAME=(.+)$/m.exec(uevent)?.[1]);
        if (pci) renderByPci.set(pci, name);
      } catch {
        /* ignore */
      }
    }
  }
  for (const card of cards) {
    const pci = normalizePciBusId(card.busId);
    if (pci && renderByPci.has(pci)) card.renderNode = renderByPci.get(pci);
  }
  return cards;
}

function readHexId(path: string): string {
  const raw = readFileSync(path, "utf8").trim().toLowerCase().replace(/^0x/, "");
  return raw;
}

function readDrmVramBytes(deviceDir: string): number | undefined {
  const files: string[] = [join(deviceDir, "mem_info_vram_total")];
  try {
    for (const entry of readdirSync(deviceDir)) {
      if (/^tile\d+$/.test(entry)) files.push(join(deviceDir, entry, "physical_vram_size_bytes"));
    }
  } catch {
    /* ignore */
  }
  try {
    const drmDir = join(deviceDir, "drm");
    if (existsSync(drmDir)) {
      for (const entry of readdirSync(drmDir)) {
        if (/^card\d+$/.test(entry)) files.push(join(drmDir, entry, "lmem_total_bytes"));
      }
    }
  } catch {
    /* ignore */
  }
  let total = 0;
  for (const file of files) {
    if (!existsSync(file)) continue;
    try {
      const n = Number(readFileSync(file, "utf8").trim());
      if (Number.isFinite(n) && n > 0) total += n;
    } catch {
      /* ignore */
    }
  }
  return total > 0 ? total : undefined;
}

function liveProbe(name: string, args: string[]): string | null {
  const resolved = which(name);
  if (resolved) {
    const fromResolved = runCapture(resolved, args);
    if (fromResolved) return fromResolved;
  }
  return runCapture(name, args);
}

function vendorFromPci(vendorId: string, name: string): GpuPickVendor | undefined {
  const id = vendorId.toLowerCase().replace(/^0x/, "");
  if (id === NVIDIA_VENDOR || /nvidia|geforce|quadro|rtx /i.test(name)) return "nvidia";
  if (id === AMD_VENDOR || /amd|radeon|instinct/i.test(name)) return "amd";
  if (id === INTEL_VENDOR || /intel|arc |battlemage/i.test(name)) return "intel";
  return undefined;
}

function connectedCount(connectors: Record<string, string> | undefined): number {
  if (!connectors) return 0;
  return Object.values(connectors).filter((status) => status.trim().toLowerCase() === "connected").length;
}

/**
 * Detect discrete + integrated GPUs on this computer.
 * Linux: lspci + DRM connectors (vendor-agnostic), plus nvidia-smi / rocm-smi when present.
 */
export function detectGpuCards(probes?: GpuPickProbes): GpuPickCard[] {
  const injected = probes !== undefined;
  if (!injected && liveCache && Date.now() - liveCache.at < 10_000) {
    return overlayLiveVram(
      liveCache.cards.map((card) => ({ ...card })),
      probes,
    );
  }

  const platform = probes?.platform ?? process.platform;
  const lspciText =
    probes && "lspci" in probes
      ? probes.lspci ?? null
      : isWindows(platform)
        ? null
        : liveProbe("lspci", ["-nn"]);
  const drm =
    probes && "drmCards" in probes
      ? probes.drmCards ?? []
      : isWindows(platform)
        ? []
        : collectLiveDrmCards();
  const nvidiaText =
    probes && "nvidiaSmi" in probes
      ? probes.nvidiaSmi ?? null
      : liveProbe("nvidia-smi", [
          "--query-gpu=index,name,memory.total,memory.free,pci.bus_id,display_active,display_mode",
          "--format=csv,noheader,nounits",
        ]);
  const nvidia = nvidiaText ? parseNvidiaSmiGpuQuery(nvidiaText) : [];
  const rocmText =
    probes && "rocmSmi" in probes
      ? probes.rocmSmi ?? null
      : isWindows(platform)
        ? null
        : liveProbe("rocm-smi", ["--showmeminfo", "vram", "--showbus"]);
  const rocm = rocmText ? parseRocmSmiGpuQuery(rocmText) : [];
  const rocmUsed = rocmText ? parseRocmSmiUsedQuery(rocmText) : [];

  const pci = lspciText ? parseLspciDisplayDevices(lspciText) : [];
  const byPci = new Map<string, ProbeRow>();

  const take = (vendor: GpuPickVendor, name: string, extra: Partial<GpuPickCard> & { vendorRuntimeIndex?: number }) => {
    const bus = normalizePciBusId(extra.busId) || undefined;
    const key = bus || `${vendor}:${name}:${byPci.size}`;
    const prev = byPci.get(key);
    const igpu = vendor === "intel" && isIntelIntegratedGpu(name, extra.deviceId ?? prev?.deviceId);
    const sku = vendor === "intel" ? estimateIntelSkuVram(name, extra.deviceId ?? prev?.deviceId) : undefined;
    const vramMiB =
      extra.vramMiB && extra.vramMiB > 0 ? extra.vramMiB : prev?.vramMiB && prev.vramMiB > 0 ? prev.vramMiB : sku?.vramMiB || 0;
    const labeled = sku && vendor === "intel" && !/arc/i.test(name) ? sku.label : name;
    const display = extra.display ?? prev?.display ?? false;
    const displayUnknown =
      extra.displayUnknown ?? (extra.display !== undefined ? false : prev?.displayUnknown) ?? true;
    const prevName = prev?.name ?? "";
    const skuName = /intel graphics/i.test(labeled) && sku ? sku.label : labeled;
    const nextName =
      prevName && !isGenericGpuName(prevName) ? prevName : isGenericGpuName(skuName) ? prevName || skuName : skuName;
    byPci.set(key, {
      ...prev,
      vendor,
      name: nextName,
      busId: bus ?? prev?.busId,
      deviceId: extra.deviceId ?? prev?.deviceId,
      vramMiB,
      freeVramMiB: extra.freeVramMiB ?? prev?.freeVramMiB,
      usedVramMiB: extra.usedVramMiB ?? prev?.usedVramMiB,
      usedVramSource: extra.usedVramSource ?? prev?.usedVramSource,
      display,
      displayUnknown,
      connectedDisplays: extra.connectedDisplays ?? prev?.connectedDisplays ?? 0,
      igpu,
      renderNode: extra.renderNode ?? prev?.renderNode,
      source: extra.source ? (prev?.source ? `${prev.source}+${extra.source}` : extra.source) : (prev?.source ?? "probe"),
      vendorRuntimeIndex:
        typeof extra.vendorRuntimeIndex === "number" ? extra.vendorRuntimeIndex : prev?.vendorRuntimeIndex,
    });
  };

  for (const dev of pci) {
    take(dev.vendor, dev.name, {
      busId: normalizePciBusId(dev.slot),
      deviceId: dev.deviceId,
      vramMiB: dev.vramMiB,
      source: "lspci",
    });
  }

  for (const card of drm) {
    const vendor = vendorFromPci(card.vendorId, card.deviceId);
    if (!vendor) continue;
    const n = connectedCount(card.connectors);
    const known = n > 0 || Object.keys(card.connectors).length > 0;
    take(vendor, vendor === "intel" ? `Intel ${card.deviceId}` : `${vendor} GPU`, {
      busId: card.busId,
      deviceId: card.deviceId,
      vramMiB: card.vramBytes ? bytesToMib(card.vramBytes) : 0,
      usedVramMiB: card.vramUsedBytes ? bytesToMib(card.vramUsedBytes) : undefined,
      usedVramSource: card.vramUsedBytes ? "sysfs" : undefined,
      display: n > 0,
      displayUnknown: !known,
      connectedDisplays: n,
      renderNode: card.renderNode,
      source: "drm",
    });
  }

  for (const gpu of nvidia) {
    take("nvidia", gpu.name, {
      busId: gpu.busId,
      vramMiB: gpu.vramMiB,
      freeVramMiB: gpu.freeVramMiB,
      usedVramMiB:
        gpu.freeVramMiB != null && gpu.vramMiB >= gpu.freeVramMiB ? gpu.vramMiB - gpu.freeVramMiB : undefined,
      usedVramSource: gpu.freeVramMiB != null ? "nvidia-smi" : undefined,
      display: gpu.displayActive === true ? true : gpu.displayActive === false ? false : undefined,
      displayUnknown: gpu.displayActive === undefined ? undefined : false,
      vendorRuntimeIndex: gpu.index,
      source: "nvidia-smi",
    });
  }

  for (const gpu of rocm) {
    const used = rocmUsed.find((row) => row.busId && gpu.busId && normalizePciBusId(row.busId) === normalizePciBusId(gpu.busId))
      ?? rocmUsed.find((row) => row.index === gpu.index);
    take("amd", gpu.name, {
      busId: gpu.busId,
      vramMiB: gpu.vramMiB,
      usedVramMiB: used && used.usedMiB > 0 ? used.usedMiB : undefined,
      usedVramSource: used && used.usedMiB > 0 ? "rocm-smi" : undefined,
      vendorRuntimeIndex: gpu.index,
      source: "rocm-smi",
    });
  }

  const unsorted = [...byPci.values()].filter((row) => row.vendor);
  unsorted.sort((a, b) => (a.busId ?? "").localeCompare(b.busId ?? "") || a.name.localeCompare(b.name));

  const vendorAll = new Map<GpuPickVendor, number>();
  const vendorDiscrete = new Map<GpuPickVendor, number>();
  const cards: GpuPickCard[] = [];
  for (const row of unsorted) {
    const vendor = row.vendor;
    const igpu = row.igpu === true;
    // iGPU is not a Level Zero / CUDA / HIP device. Do not give it an affinity slot —
    // counting it made ZE_AFFINITY_MASK=2 for the idle B70 (display=1, iGPU=0).
    const allIdx = igpu ? -1 : (vendorAll.get(vendor) ?? 0);
    if (!igpu) vendorAll.set(vendor, allIdx + 1);
    const discIdx = igpu ? -1 : (vendorDiscrete.get(vendor) ?? 0);
    if (!igpu) vendorDiscrete.set(vendor, discIdx + 1);
    const displayUnknown = row.displayUnknown !== false && !row.display;
    const runtimeIdx = igpu
      ? -1
      : typeof row.vendorRuntimeIndex === "number"
        ? row.vendorRuntimeIndex
        : Math.max(0, discIdx);
    cards.push({
      id: gpuCardId(vendor, row.busId, igpu ? allIdx : Math.max(0, discIdx)),
      vendor,
      name: row.name,
      index: runtimeIdx,
      vendorAllIndex: igpu ? -1 : allIdx,
      busId: row.busId,
      deviceId: row.deviceId,
      vramMiB: row.vramMiB ?? 0,
      freeVramMiB: row.freeVramMiB,
      usedVramMiB: row.usedVramMiB,
      usedVramSource: row.usedVramSource,
      display: row.display === true,
      displayUnknown,
      connectedDisplays: row.connectedDisplays ?? 0,
      igpu,
      renderNode: row.renderNode,
      source: row.source ?? "probe",
    });
  }

  if (!injected) liveCache = { at: Date.now(), cards: cards.map((card) => ({ ...card })) };
  return overlayLiveVram(cards, probes);
}

function overlayLiveVram(cards: GpuPickCard[], probes?: GpuPickProbes): GpuPickCard[] {
  const injected = probes !== undefined;
  const fdinfoBlobs =
    probes && "drmFdinfo" in probes
      ? probes.drmFdinfo ?? []
      : injected
        ? []
        : collectLiveDrmFdinfo();
  const clients = fdinfoBlobs
    .map((blob) => parseDrmFdinfo(blob))
    .filter((row): row is DrmFdinfoClient => Boolean(row));
  const clinfoText =
    probes && "clinfo" in probes ? probes.clinfo ?? null : null;
  const clinfo = clinfoText ? parseClinfoDiscreteVram(clinfoText) : [];
  for (const card of cards) {
    if (card.igpu) continue;
    const fromFdinfo = usedVramMiBFromFdinfo(clients, card.busId);
    if (fromFdinfo > 0 && card.usedVramSource !== "nvidia-smi" && card.usedVramSource !== "rocm-smi") {
      card.usedVramMiB = fromFdinfo;
      card.usedVramSource = "drm-fdinfo";
    }
    if ((!card.vramMiB || card.vramMiB <= 0) && card.busId) {
      const fromClinfo = clinfo.find((row) => row.busId && normalizePciBusId(row.busId) === normalizePciBusId(card.busId));
      if (fromClinfo && fromClinfo.vramMiB > 0) card.vramMiB = fromClinfo.vramMiB;
    }
  }
  return cards;
}

export function resetGpuPickCacheForTests(): void {
  liveCache = undefined;
}

function pickIdlePrimary(discrete: GpuPickCard[]): GpuPickCard | undefined {
  const idle = discrete.filter((card) => !card.display && !card.displayUnknown);
  if (idle.length === 0) return undefined;
  return [...idle].sort((a, b) => b.vramMiB - a.vramMiB || (a.busId ?? "").localeCompare(b.busId ?? ""))[0];
}

function spawnEnvForVisible(
  visible: GpuPickCard[],
  primary: GpuPickCard | undefined,
  intelRuntime?: IntelRuntimeProbe,
): Record<string, string> {
  const env: Record<string, string> = {};
  const target = visible[0] ?? primary;
  if (!target) return env;
  env.LATE_INFER_ACCEL = target.vendor;
  if (target.busId) env.LATE_INFER_PCI = target.busId;
  env.LATE_INFER_GPU_IDLE = target.display ? "0" : "1";
  env.LATE_INFER_GPU_DISPLAY = target.display ? "1" : "0";
  if (target.vendor === "nvidia") {
    env.CUDA_VISIBLE_DEVICES = visible
      .filter((card) => card.vendor === "nvidia")
      .map((card) => String(card.index))
      .join(",");
  } else if (target.vendor === "amd") {
    const mask = visible
      .filter((card) => card.vendor === "amd")
      .map((card) => String(card.index))
      .join(",");
    env.HIP_VISIBLE_DEVICES = mask;
    env.ROCR_VISIBLE_DEVICES = mask;
  } else if (target.vendor === "intel") {
    env.ZE_FLAT_DEVICE_HIERARCHY = "FLAT";
    env.ONEAPI_DEVICE_SELECTOR = "level_zero:gpu";
    // Discrete XPUs in PCI order (iGPU omitted). Idle card uses that index, not GPU 0.
    env.ZE_AFFINITY_MASK = String(target.vendorAllIndex);
    if (intelRuntime?.present) {
      env.LATE_INFER_INTEL_KIND = intelRuntime.kind ?? "openvino-genai";
      if (intelRuntime.python) env.LATE_INFER_INTEL_PYTHON = intelRuntime.python;
    }
  }
  return env;
}

/** Merge plan env onto a process env. Keep only the matching vendor toolkit. */
export function mergeGpuPlanEnv(base: NodeJS.ProcessEnv, plan: GpuPickPlan): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...base, ...plan.env };
  const accel = (plan.env.LATE_INFER_ACCEL ?? "").toLowerCase();
  if (accel === "intel" || accel === "amd") {
    delete env.CUDA_VISIBLE_DEVICES;
  }
  if (accel === "intel" || accel === "nvidia") {
    delete env.HIP_VISIBLE_DEVICES;
    delete env.ROCR_VISIBLE_DEVICES;
  }
  if (accel === "nvidia" || accel === "amd") {
    delete env.ZE_AFFINITY_MASK;
    delete env.ONEAPI_DEVICE_SELECTOR;
    delete env.ZE_FLAT_DEVICE_HIERARCHY;
    delete env.LATE_INFER_INTEL_PYTHON;
    delete env.LATE_INFER_INTEL_KIND;
  }
  return env;
}

/**
 * Idle-first spawn env + llama.cpp flags for Ollama / llama-server / host vLLM.
 * Same primary as late-infer (`0000:08:00.0` on this dual-B70 box). Fail closed when
 * there is no discrete GPU — do not mmap into DRAM as if that were the card.
 *
 * Ollama on Intel uses the same GGML_VK / ZE idle pin as llama.cpp when a Vulkan ICD
 * is present. Do not refuse Intel solely for lacking CUDA/ROCm.
 */
export function engineGpuSpawnPlan(options: { useAllGpus?: boolean } & LateInferGpuOptions = {}): {
  plan: GpuPickPlan;
  env: Record<string, string>;
  llamaArgs: string[];
  blockedReason?: string;
} {
  const plan = resolveLateInferGpuPlan({
    useAllGpus: options.useAllGpus === true,
    gpuId: options.gpuId,
    vramMaxMiB: options.vramMaxMiB,
    model: options.model,
    cards: options.cards,
    probes: options.probes,
  });
  if (plan.needsPicker) {
    return { plan, env: {}, llamaArgs: [], blockedReason: plan.reason };
  }
  const primary = plan.visible[0] ?? plan.primary;
  if (!primary || plan.discrete.length === 0) {
    return {
      plan,
      env: {},
      llamaArgs: [],
      blockedReason:
        "No discrete GPU on your computer. Refusing to start Ollama/llama.cpp on CPU/RAM as if that were the accelerator.",
    };
  }
  const env: Record<string, string> = { ...plan.env };
  const idx = Math.max(0, primary.vendorAllIndex >= 0 ? primary.vendorAllIndex : primary.index);
  // Vulkan / ggml device index among discrete cards (iGPU omitted) — idle B70 is 1 here.
  env.GGML_VK_VISIBLE_DEVICES = String(idx);
  const llamaArgs = ["-ngl", "99"];
  if (options.useAllGpus === true && plan.discrete.length > 1) {
    const n = plan.discrete.length;
    llamaArgs.push("-sm", "layer", "-ts", Array.from({ length: n }, () => "1").join(","));
  }
  return { plan, env, llamaArgs };
}

const VULKAN_ICD_DIRS = [
  "/usr/share/vulkan/icd.d",
  "/etc/vulkan/icd.d",
  "/usr/local/share/vulkan/icd.d",
];

/**
 * True when a Vulkan ICD JSON for this vendor is installed (Mesa Intel, RADV, NVIDIA, …).
 * Inject `vulkanIcds` on GpuProbes for tests. Empty list → missing.
 */
export function vulkanIcdPresent(
  vendor: GpuPickVendor | undefined,
  probes?: GpuPickProbes,
): boolean {
  if (probes && "vulkanIcds" in probes) {
    const list = probes.vulkanIcds;
    if (!list || list.length === 0) return false;
    if (!vendor) return list.length > 0;
    const re =
      vendor === "intel"
        ? /intel/i
        : vendor === "amd"
          ? /radeon|amd|radv/i
          : vendor === "nvidia"
            ? /nvidia/i
            : /./;
    return list.some((name) => re.test(name));
  }
  const names: string[] = [];
  for (const dir of VULKAN_ICD_DIRS) {
    if (!existsSync(dir)) continue;
    try {
      for (const ent of readdirSync(dir)) {
        if (ent.endsWith(".json")) names.push(ent);
      }
    } catch {
      /* ignore */
    }
  }
  if (!vendor) return names.length > 0;
  const re =
    vendor === "intel"
      ? /intel/i
      : vendor === "amd"
        ? /radeon|amd|radv/i
        : vendor === "nvidia"
          ? /nvidia/i
          : /./;
  return names.some((name) => re.test(name));
}

/**
 * Ollama on Intel Arc uses GGML Vulkan (GGML_VK_VISIBLE_DEVICES) + ZE affinity — same idle
 * pin as llama.cpp. Fail closed only when that GPU path is missing (would be CPU/RAM mmap).
 * NVIDIA CUDA / AMD ROCm boxes do not need a Vulkan ICD check here.
 */
export function ollamaGpuBlockedReason(
  plan: GpuPickPlan,
  probes?: GpuPickProbes,
): string | undefined {
  const vendor = plan.visible[0]?.vendor ?? plan.primary?.vendor;
  if (!vendor || plan.discrete.length === 0) {
    return "No discrete GPU on your computer. Refusing to start Ollama on CPU/RAM as if that were the accelerator.";
  }
  if (vendor === "intel" && !vulkanIcdPresent("intel", probes)) {
    return "Ollama cannot use the idle Intel GPU VRAM on this computer (no Vulkan ICD for GGML_VK). Refusing to load weights into system RAM as that card. Install Mesa Intel Vulkan, or use llama.cpp / late-infer / Intel vLLM Docker.";
  }
  return undefined;
}

/** Host CUDA vLLM on an Intel-only box would mmap into DRAM — refuse before spawn. */
export function vllmCudaOnIntelBlockedReason(backend: string, plan?: GpuPickPlan): string | undefined {
  const live = plan ?? resolveLateInferGpuPlan({});
  const vendor = live.visible[0]?.vendor ?? live.primary?.vendor;
  if (backend === "cuda" && vendor === "intel") {
    return "This vLLM path is CUDA-only and cannot use the Intel Arc GPU on your computer. Use an Intel XPU / Docker image, or Ollama / llama.cpp / late-infer on the idle card. Refusing to load weights into system RAM.";
  }
  if (backend === "cuda" && vendor === "amd") {
    return "This vLLM path is CUDA-only and cannot use the AMD GPU on your computer. Use a ROCm vLLM build or Ollama / llama.cpp. Refusing to load weights into system RAM.";
  }
  return undefined;
}

export function formatCompileTarget(plan: {
  primary?: GpuPickCard;
  visible: GpuPickCard[];
}): string {
  const card = plan.visible[0] ?? plan.primary;
  if (!card) return "CPU on your computer";
  const vendor = formatGpuVendorLabel(card.vendor);
  const role = card.display ? "display · 70% cap" : card.igpu ? "integrated" : "idle · full VRAM";
  const vram = formatGpuVramLabel(card.vramMiB);
  const bus = card.busId ? ` · ${card.busId}` : "";
  return `${vendor} · ${card.name} · ${vram} · ${role}${bus}`;
}

function describeCard(card: GpuPickCard): string {
  const vram = formatGpuVramLabel(card.vramMiB);
  const bus = card.busId ? ` · ${card.busId}` : "";
  if (card.igpu) return `${card.name} (integrated, ignored)${bus}`;
  if (card.display) {
    const n = card.connectedDisplays;
    const monitors = n > 0 ? ` · ${n} display${n === 1 ? "" : "s"}` : " · driving displays";
    return `${card.name} (${vram}${monitors}, 70% cap)${bus}`;
  }
  return `${card.name} (idle · ${vram} · full VRAM)${bus}`;
}

type GpuPlanCore = Omit<
  GpuPickPlan,
  | "compileOk"
  | "compileReason"
  | "compileTarget"
  | "runtimeKind"
  | "serveHint"
  | "hostRamOk"
  | "hostRamReason"
  | "ovIrPresent"
  | "ovIrChecked"
>;

function withCompileMeta(plan: GpuPlanCore, host?: { model?: string; vramMaxMiB?: number; probes?: GpuPickProbes }): GpuPickPlan {
  const compileReason = lateInferCompileBlockedReason(plan);
  const vendor = plan.visible[0]?.vendor ?? plan.primary?.vendor;
  const runtimeKind = !plan.runtimeOk
    ? ""
    : vendor === "intel"
      ? "openvino-genai"
      : vendor === "nvidia"
        ? "candle-cuda"
        : "";
  const ovIrChecked = Boolean(host?.model) || Boolean(host?.probes && "ovIrPresent" in host.probes);
  const ovIr =
    host?.probes && "ovIrPresent" in host.probes
      ? host.probes.ovIrPresent === true
      : host?.model
        ? openvinoIrPresent(host.model)
        : false;
  const serveHint =
    vendor === "intel" && plan.runtimeOk
      ? ovIr
        ? "Start loads OpenVINO IR onto the idle Intel GPU (Level Zero). Hugging Face safetensors were already exported. A CUDA compiled blob is not used as that card."
        : ovIrChecked
          ? "This snapshot is not ready for GPU Start (OpenVINO IR is missing). Start refused to protect your computer. Convert only when MemAvailable is safe."
          : "Start serves Hugging Face safetensors on the idle Intel GPU (OpenVINO GenAI) after Download writes IR. A CUDA compiled blob is not used as that card."
      : "";
  const ramReason = hostRamBlockedReason({
    vendor,
    runtimeOk: plan.runtimeOk,
    weightsMiB: weightsMiBForStart(host?.model, host?.vramMaxMiB),
    ovIrPresent: ovIr,
    mem: resolveHostRam(host?.probes),
  });
  return {
    ...plan,
    runtimeKind,
    serveHint,
    compileOk: !compileReason,
    compileReason: compileReason ?? "",
    compileTarget: formatCompileTarget(plan),
    hostRamOk: !ramReason,
    hostRamReason: ramReason ?? "",
    ovIrPresent: ovIr,
    ovIrChecked,
  };
}

/**
 * Default Start plan: idle discrete card at 100%. Overflow onto display GPUs only when
 * vramMaxMiB > that card. Display cards stay at 70% — Candle cannot apply that, so overflow
 * is blocked unless the operator picks a card. Use-all is an override but still does not
 * dump 100% onto the monitor GPU.
 */
export function resolveLateInferGpuPlan(options: LateInferGpuOptions = {}): GpuPickPlan {
  const cards = options.cards ?? detectGpuCards(options.probes);
  const intelRuntime = resolveIntelRuntime(options.probes);
  const discrete = cards.filter((card) => !card.igpu);
  const displayCapPercent = Math.round(DISPLAY_GPU_VRAM_CAP * 100);
  const useAllGpus = options.useAllGpus === true;
  const host = { model: options.model, vramMaxMiB: options.vramMaxMiB, probes: options.probes };
  const empty = (extra: Partial<GpuPlanCore>): GpuPickPlan =>
    withCompileMeta({
      cards,
      discrete,
      visible: [],
      env: {},
      needsPicker: false,
      overflowWanted: false,
      overflowBlocked: false,
      capApplied: false,
      displayCapPercent,
      runtimeOk: true,
      runtimeReason: "No discrete GPU — late-infer would use CPU.",
      reason: extra.reason ?? "No discrete GPU on your computer.",
      label: extra.label ?? "CPU on your computer",
      useAllGpus,
      ...extra,
    }, host);

  if (discrete.length === 0) {
    return empty({});
  }

  const classified = discrete.filter((card) => !card.displayUnknown);
  const needsPicker = discrete.length > 1 && classified.length === 0;

  const explicit = options.gpuId?.trim()
    ? discrete.find((card) => card.id === options.gpuId?.trim())
    : undefined;

  let primary = explicit ?? pickIdlePrimary(discrete);
  if (!primary && discrete.length === 1) primary = discrete[0];
  if (!primary && !needsPicker) {
    primary = [...discrete].sort((a, b) => b.vramMiB - a.vramMiB)[0];
  }

  if (needsPicker && !explicit) {
    return withCompileMeta({
      cards,
      discrete,
      visible: [],
      env: {},
      needsPicker: true,
      overflowWanted: false,
      overflowBlocked: false,
      capApplied: false,
      displayCapPercent,
      runtimeOk: false,
      runtimeReason: "Pick a GPU on your computer. Auto will not guess GPU 0 (often the display card).",
      reason:
        "Could not tell which GPU drives displays on your computer. Pick a card — Start will not silently use GPU 0.",
      label: "Pick a GPU on your computer",
      useAllGpus,
    }, host);
  }

  if (!primary) {
    return empty({
      needsPicker: true,
      runtimeOk: false,
      runtimeReason: "Pick a GPU on your computer.",
      reason: "Pick a GPU on your computer.",
      label: "Pick a GPU on your computer",
    });
  }

  const vramMax = options.vramMaxMiB;
  const overflowWanted =
    !explicit &&
    typeof vramMax === "number" &&
    Number.isFinite(vramMax) &&
    vramMax > 0 &&
    primary.vramMiB > 0 &&
    vramMax > primary.vramMiB;
  const displayCards = discrete.filter((card) => card.display && card.id !== primary.id);
  const overflow = overflowWanted || useAllGpus ? displayCards : [];
  // Candle / late-infer cannot cap a card at 70% or tensor-parallel. Do not attach
  // the display GPU at 100%. Explicit picker is the only way onto a monitor card.
  const overflowBlocked = (overflowWanted || (useAllGpus && displayCards.length > 0)) && !explicit;
  const visible = explicit ? [explicit] : [primary];
  const runtimeOk = lateInferSupportsVendor(visible[0]?.vendor, intelRuntime);
  const runtimeReason = runtimeOk
    ? visible[0]?.vendor === "intel"
      ? "late-infer can use Intel (OpenVINO GenAI / Level Zero) on your computer."
      : `late-infer can use ${visible[0]!.vendor} on your computer.`
    : visible[0]?.vendor === "intel"
      ? "late-infer cannot use the Intel GPU on your computer (this engine needs OpenVINO GenAI / Level Zero, not CUDA). Start will not run on CPU and call it that card."
      : "late-infer cannot use this GPU on your computer (this engine is CUDA/Metal/CPU). Start will not run on CPU as if it were that card.";

  let reason = `Start uses the idle GPU on your computer: ${describeCard(primary)}.`;
  if (explicit?.display) {
    reason = `You picked the display GPU (${describeCard(explicit)}). This runtime cannot cap it at ${displayCapPercent}% — monitors may stall.`;
  } else if (overflowWanted && overflowBlocked) {
    reason = `This snapshot needs more than the idle ${formatGpuVramLabel(primary.vramMiB)} card. The display GPU would be included at ${displayCapPercent}% only — this runtime cannot apply that cap or split weights, so Start stays on the idle card and will not overflow onto monitors.`;
  } else if (useAllGpus && overflowBlocked) {
    reason = `Use all GPUs still keeps the idle card first. Display GPUs stay at ${displayCapPercent}% — this runtime cannot apply that cap, so they are not attached.`;
  } else if (primary.display && discrete.length === 1) {
    reason = `One GPU on your computer (${describeCard(primary)}). A ${displayCapPercent}% cap for displays cannot be applied by this runtime.`;
  }

  return withCompileMeta({
    cards,
    discrete,
    primary,
    overflow: overflowBlocked ? overflow : [],
    visible,
    env: spawnEnvForVisible(visible, primary, intelRuntime),
    needsPicker: false,
    overflowWanted,
    overflowBlocked,
    capApplied: false,
    displayCapPercent,
    runtimeOk,
    runtimeReason,
    reason,
    label: describeCard(visible[0] ?? primary),
    useAllGpus,
  }, host);
}

export function lateInferStartBlockedReason(plan: GpuPickPlan): string | undefined {
  if (plan.needsPicker) return plan.reason;
  if (!plan.runtimeOk) return plan.runtimeReason;
  const vendor = plan.visible[0]?.vendor ?? plan.primary?.vendor;
  if (vendor === "intel" && plan.ovIrChecked && plan.ovIrPresent === false) {
    const ram = plan.hostRamReason || HOST_RAM_VS_VRAM;
    return ram.includes(INTEL_IR_MISSING_START) ? ram : `${INTEL_IR_MISSING_START} ${ram}`;
  }
  if (plan.hostRamOk === false) return plan.hostRamReason || HOST_RAM_VS_VRAM;
  return undefined;
}

/**
 * Compile/Download fail-closed: Intel without OpenVINO, AMD (HIP not wired),
 * picker unknown, or overflow would need a 70% display cap this compiler cannot apply.
 */
export function lateInferCompileBlockedReason(plan: {
  needsPicker: boolean;
  reason: string;
  runtimeOk: boolean;
  runtimeReason: string;
  overflowBlocked: boolean;
  displayCapPercent: number;
  label: string;
  primary?: GpuPickCard;
  visible: GpuPickCard[];
}): string | undefined {
  if (plan.needsPicker) return plan.reason;
  const vendor = plan.visible[0]?.vendor ?? plan.primary?.vendor;
  if (!plan.runtimeOk) {
    if (vendor === "intel") {
      return "late-infer cannot compile for the Intel GPU on your computer (Arc / XPU / Level Zero). OpenVINO GenAI is not on this computer yet. Download will not treat that card as NVIDIA or start a Hub fetch that CUDA-fails.";
    }
    if (vendor === "amd") {
      return "late-infer cannot compile for the AMD GPU on your computer (this engine is CUDA/Metal/CPU, not ROCm). Download will not treat that card as NVIDIA.";
    }
    return plan.runtimeReason;
  }
  if (plan.overflowBlocked) {
    return `This snapshot needs more than the idle GPU on your computer (${plan.label}). Compile cannot apply a ${plan.displayCapPercent}% cap on the display GPU, so it will not overflow onto monitors or compile as NVIDIA GPU 0.`;
  }
  return undefined;
}

/** GUI / HTTP snapshot: ids and labels, never a home path. */
export function publicGpuPlan(plan: GpuPickPlan, liveInput: GpuServeLiveInput = {}) {
  const cardView = (card: GpuPickCard) => ({
    id: card.id,
    vendor: card.vendor,
    vendorLabel: formatGpuVendorLabel(card.vendor),
    name: card.name,
    index: card.index,
    busId: card.busId,
    vramMiB: card.vramMiB,
    vramLabel: formatGpuVramLabel(card.vramMiB),
    freeVramMiB: card.freeVramMiB,
    usedVramMiB: card.usedVramMiB,
    usedVramSource: card.usedVramSource,
    vramPairLabel: formatGpuVramPair(card.usedVramMiB, card.vramMiB),
    display: card.display,
    connectedDisplays: card.connectedDisplays,
    igpu: card.igpu,
    renderNode: card.renderNode,
    role: card.igpu ? "integrated" : card.display ? "display" : "idle",
    up: true,
  });
  const env = { ...plan.env };
  delete env.LATE_INFER_INTEL_PYTHON;
  const active = plan.visible[0] ?? plan.primary;
  const classified = classifyLateInferDevice(liveInput.device);
  const servePhase = resolveGpuServePhase(liveInput);
  const weightsInHostRam = servePhase === "host-ram" || liveInput.weightsInHostRam === true;
  const gpuRunning = servePhase === "gpu-running";
  const vramPair = active ? formatGpuVramPair(active.usedVramMiB, active.vramMiB) : "VRAM unknown";
  const role = !active ? "none" : active.display ? "display" : active.igpu ? "integrated" : "idle";
  let headline = "GPU down";
  let detail = "late-infer is not serving on 127.0.0.1:8010.";
  if (servePhase === "starting") {
    headline = "starting on idle GPU…";
    detail =
      "Start is loading weights on the idle GPU on your computer. First load can take several minutes. This is not GPU running yet — if the box ran out of memory, Start failed.";
  } else if (servePhase === "gpu-running") {
    headline = "GPU running";
    detail = `late-infer is up on 127.0.0.1:8010 (${classified.deviceKind || liveInput.deviceKind || "GPU"}). VRAM is the idle card, not host RAM.`;
  } else if (servePhase === "host-ram") {
    headline = "Weights in host RAM";
    detail =
      plan.hostRamReason ||
      "Weights landed in host RAM, not VRAM on the idle GPU on your computer. This is not GPU running.";
  } else if (servePhase === "exited") {
    headline = "GPU down";
    detail =
      "Start exited before GPU running. This is not success — if the box ran out of memory, the idle card never came up.";
  } else if (servePhase === "ready") {
    headline = "late-infer answering";
    detail =
      "127.0.0.1:8010 answered /v1/models but the health device is not Intel XPU / CUDA / HIP yet. Not GPU running.";
  }
  return {
    cards: plan.discrete.map(cardView),
    primaryId: plan.primary?.id,
    visibleIds: plan.visible.map((card) => card.id),
    label: plan.label,
    reason: plan.reason,
    runtimeOk: plan.runtimeOk,
    runtimeReason: plan.runtimeReason,
    runtimeKind: plan.runtimeKind,
    serveHint: plan.serveHint,
    needsPicker: plan.needsPicker,
    overflowWanted: plan.overflowWanted,
    overflowBlocked: plan.overflowBlocked,
    capApplied: plan.capApplied,
    displayCapPercent: plan.displayCapPercent,
    env,
    useAllGpus: plan.useAllGpus,
    compileOk: plan.compileOk,
    compileReason: plan.compileReason,
    compileTarget: plan.compileTarget,
    hostRamOk: plan.hostRamOk,
    hostRamReason: plan.hostRamReason,
    ovIrPresent: plan.ovIrPresent === true,
    ovIrChecked: plan.ovIrChecked === true,
    live: {
      servePhase,
      gpuRunning,
      weightsInHostRam,
      headline,
      detail,
      device: liveInput.device ?? "",
      deviceKind: (liveInput.deviceKind as string) || classified.deviceKind,
      processAlive: liveInput.processAlive === true,
      vendor: active?.vendor ?? "",
      vendorLabel: formatGpuVendorLabel(active?.vendor),
      name: active?.name ?? "",
      busId: active?.busId ?? "",
      role,
      up: gpuRunning,
      vramUsedMiB: active?.usedVramMiB,
      vramTotalMiB: active?.vramMiB,
      vramPairLabel: vramPair,
      usedSource: active?.usedVramSource ?? "",
    },
  };
}
