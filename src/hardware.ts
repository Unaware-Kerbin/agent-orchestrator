import { cpus, totalmem } from "node:os";
import { existsSync, readdirSync, readFileSync, realpathSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { join } from "node:path";
import type { IntelDockerCatalog } from "./vllm/intel-docker.js";
import {
  isWindows,
  probeWin32VideoControllers,
  pythonDashArgs,
  pythonInterpreterNames,
  runCapture,
  which,
} from "./platform.js";

export type AcceleratorVendor = "intel" | "nvidia" | "amd";
export type LaunchBackend = "intel-xpu" | "cuda" | "rocm" | "cpu";

export interface Accelerator {
  vendor: AcceleratorVendor;
  name: string;
  vramMiB: number;
  index: number;
  source: string;
  vramEstimated?: boolean;
  deviceId?: string;
}

export interface GpuInfo {
  index: number;
  name: string;
  vramMiB: number;
  driver?: string;
  vendor?: AcceleratorVendor;
  source?: string;
}

export interface HardwareSnapshot {
  accelerators: Accelerator[];
  primaryBackend: LaunchBackend;
  totalVramMiB: number;
  deviceCount: number;
  gpus: GpuInfo[];
  /** Largest single-GPU VRAM (MiB) on the primary backend. 0 if CPU-only. */
  vramMiB: number;
  /** Smallest primary-backend GPU VRAM (MiB). Used for tensor-parallel fitting. */
  minVramMiB: number;
  ramMiB: number;
  cpuCount: number;
  hasNvidiaSmi: boolean;
  /** True when no discrete accelerator was found (CPU last resort). */
  constrained: boolean;
  notes: string[];
  /** Intel vLLM Docker images (llm-scaler-vllm / intel/vllm:*xpu), when probed. */
  intelDocker?: IntelDockerCatalog;
}

export interface SysfsCard {
  name: string;
  vendorId: string;
  deviceId: string;
  driver?: string;
  vramBytes?: number;
}

export interface HardwareProbes {
  nvidiaSmi?: string | null;
  lspci?: string | null;
  lspciVerbose?: string | null;
  xpuSmi?: string | null;
  xpumcli?: string | null;
  syclLs?: string | null;
  syclLsVerbose?: string | null;
  intelGpuTop?: string | null;
  clinfo?: string | null;
  rocmSmi?: string | null;
  amdSmi?: string | null;
  torchXpu?: string | null;
  sysfsCards?: SysfsCard[];
  /** Win32_VideoController dump (`Name|AdapterRAM|PNPDeviceID` or WMI csv). */
  win32Video?: string | null;
}

export interface PciDisplayDevice {
  slot: string;
  vendor: AcceleratorVendor;
  name: string;
  vendorId: string;
  deviceId: string;
  discrete: boolean;
  vramMiB?: number;
}

interface NamedMemory {
  name: string;
  vramMiB: number;
  source: string;
}

/** PCI IDs known to be Intel discrete GPUs (Alchemist / Battlemage / Data Center). */
export const INTEL_DISCRETE_DEVICE_IDS: Readonly<Record<string, { name: string; vramMiB: number }>> = {
  e223: { name: "Intel Arc Pro B70", vramMiB: 32_768 },
  e222: { name: "Intel Arc Pro B70", vramMiB: 32_768 },
  e221: { name: "Intel Arc Pro B65", vramMiB: 32_768 },
  e220: { name: "Intel Arc Pro B70", vramMiB: 32_768 },
  e20b: { name: "Intel Arc B580", vramMiB: 12_288 },
  e20c: { name: "Intel Arc B580", vramMiB: 12_288 },
  e20d: { name: "Intel Arc B570", vramMiB: 10_240 },
  e212: { name: "Intel Arc B580", vramMiB: 12_288 },
  e202: { name: "Intel Arc B580", vramMiB: 12_288 },
  "56a0": { name: "Intel Arc A770", vramMiB: 16_384 },
  "56a1": { name: "Intel Arc A750", vramMiB: 8_192 },
  "56a2": { name: "Intel Arc A580", vramMiB: 8_192 },
  "5690": { name: "Intel Arc A770M", vramMiB: 16_384 },
  "56a5": { name: "Intel Arc A380", vramMiB: 6_144 },
  "56a6": { name: "Intel Arc A310", vramMiB: 4_096 },
};

const INTEL_IGPU_DEVICE_IDS = new Set([
  "7d67",
  "7d45",
  "7d40",
  "7d41",
  "7d55",
  "7d60",
  "a7a0",
  "a7a1",
  "4680",
  "4690",
  "46a6",
  "9a49",
  "9a40",
  "64a0",
  "7d28",
]);

const NVIDIA_VENDOR = "10de";
const AMD_VENDOR = "1002";
const INTEL_VENDOR = "8086";

export function bytesToMib(bytes: number): number {
  if (!Number.isFinite(bytes) || bytes <= 0) return 0;
  return Math.round(bytes / 1024 / 1024);
}

export function backendForVendor(vendor: AcceleratorVendor): LaunchBackend {
  if (vendor === "nvidia") return "cuda";
  if (vendor === "intel") return "intel-xpu";
  return "rocm";
}

export function parseNvidiaSmiCsv(csv: string): GpuInfo[] {
  const gpus: GpuInfo[] = [];
  for (const raw of csv.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const parts = line.split(",").map((part) => part.trim());
    const name = parts[0];
    const vram = Number(parts[1]);
    if (!name || !Number.isFinite(vram) || vram <= 0) continue;
    gpus.push({
      index: gpus.length,
      name,
      vramMiB: Math.round(vram),
      driver: parts[2] && parts[2].length > 0 ? parts[2] : undefined,
      vendor: "nvidia",
      source: "nvidia-smi",
    });
  }
  return gpus;
}

export function isIntelIntegratedGpu(name: string, deviceId?: string): boolean {
  const id = (deviceId ?? "").toLowerCase().replace(/^0x/, "");
  if (id && INTEL_DISCRETE_DEVICE_IDS[id]) return false;
  if (id && INTEL_IGPU_DEVICE_IDS.has(id)) return true;
  const n = name.toLowerCase();
  if (/\barc\s+pro\b|\barc\s*b\d|\barc\s*a\d{3}|battlemage|alchemist|ponte vecchio|data center gpu|flex\s*\d/i.test(n)) {
    return false;
  }
  if (/arrow lake|meteor lake|lunar lake|raptor lake|alder lake|uhd graphics|iris\s*xe|onboard igd/.test(n)) {
    return true;
  }
  if (/^intel(?:\(r\))?\s+graphics$/i.test(name.trim())) return true;
  if (/intel graphics\]/.test(n) && !/battlemage|alchemist|arc/.test(n)) return true;
  return false;
}

export function estimateIntelSkuVram(name: string, deviceId?: string): { vramMiB: number; label: string } | undefined {
  const id = (deviceId ?? "").toLowerCase().replace(/^0x/, "");
  const fromId = id ? INTEL_DISCRETE_DEVICE_IDS[id] : undefined;
  if (fromId) return { vramMiB: fromId.vramMiB, label: fromId.name };
  const n = name.toLowerCase();
  const patterns: Array<[RegExp, number, string]> = [
    [/b70\s*pro|pro\s*b70|arc\s+pro\s*b70/, 32_768, "Intel Arc Pro B70"],
    [/pro\s*b65|arc\s+pro\s*b65/, 32_768, "Intel Arc Pro B65"],
    [/pro\s*b60|arc\s+pro\s*b60/, 24_576, "Intel Arc Pro B60"],
    [/pro\s*b50|arc\s+pro\s*b50/, 16_384, "Intel Arc Pro B50"],
    [/\bb580\b/, 12_288, "Intel Arc B580"],
    [/\bb570\b/, 10_240, "Intel Arc B570"],
    [/\ba770\b/, 16_384, "Intel Arc A770"],
    [/\ba750\b/, 8_192, "Intel Arc A750"],
    [/\ba580\b/, 8_192, "Intel Arc A580"],
    [/\ba380\b/, 6_144, "Intel Arc A380"],
  ];
  for (const [re, vramMiB, label] of patterns) {
    if (re.test(n)) return { vramMiB, label };
  }
  return undefined;
}

export function parseLspciDisplayDevices(text: string): PciDisplayDevice[] {
  const devices: PciDisplayDevice[] = [];
  const lineRe =
    /^([0-9a-f:.]+)\s+(VGA compatible controller|3D controller|Display controller)\s+\[[0-9a-f]{4}\]:\s+(.+?)\s+\[([0-9a-f]{4}):([0-9a-f]{4})\]/i;
  for (const raw of text.split(/\r?\n/)) {
    const match = lineRe.exec(raw.trim());
    if (!match) continue;
    const slot = match[1] ?? "";
    const rest = match[3] ?? "";
    const vendorId = (match[4] ?? "").toLowerCase();
    const deviceId = (match[5] ?? "").toLowerCase();
    const vendor = vendorFromPci(vendorId, rest);
    if (!vendor) continue;
    const discrete = vendor === "intel" ? !isIntelIntegratedGpu(rest, deviceId) : true;
    devices.push({ slot, vendor, name: rest, vendorId, deviceId, discrete });
  }
  return devices;
}

const PNP_PCI_RE = /VEN_([0-9A-Fa-f]{4})&DEV_([0-9A-Fa-f]{4})/;

function adapterRamToMib(raw: string): number {
  const n = Number(raw);
  if (!Number.isFinite(n) || n < 256 * 1024 * 1024) return 0;
  const mib = bytesToMib(n);
  if (mib <= 0 || mib > 256 * 1024) return 0;
  return mib;
}

/**
 * Parse Win32_VideoController dumps from PowerShell (`Name|AdapterRAM|PNPDeviceID`)
 * or `wmic ... /format:csv`. Used on Windows instead of lspci/sysfs.
 */
export function parseWin32VideoControllers(text: string): PciDisplayDevice[] {
  const devices: PciDisplayDevice[] = [];
  const seen = new Set<string>();
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || /^node,/i.test(line) || /^name\s/i.test(line)) continue;
    let name = "";
    let ram = "";
    let pnp = "";
    const pipe = line.split("|");
    if (pipe.length >= 3 && PNP_PCI_RE.test(pipe[2] ?? "")) {
      name = (pipe[0] ?? "").trim();
      ram = (pipe[1] ?? "").trim();
      pnp = (pipe[2] ?? "").trim();
    } else {
      const csv = line.split(",");
      if (csv.length >= 3) {
        pnp = csv.find((part) => PNP_PCI_RE.test(part)) ?? "";
        ram = csv.find((part, i) => i > 0 && /^\d+$/.test(part.trim())) ?? "";
        name =
          csv.find((part, i) => i > 0 && part !== ram && part !== pnp && /nvidia|amd|ati|radeon|intel|arc|geforce|quadro|rtx|radeon/i.test(part)) ??
          csv.find((part, i) => i > 0 && part !== ram && part !== pnp && !/^[\d.]+$/.test(part.trim()) && !/^\\\\/.test(part) && part.trim() !== "") ??
          "";
      }
    }
    const pci = PNP_PCI_RE.exec(pnp);
    if (!name || !pci) continue;
    const vendorId = (pci[1] ?? "").toLowerCase();
    const deviceId = (pci[2] ?? "").toLowerCase();
    const vendor = vendorFromPci(vendorId, name);
    if (!vendor) continue;
    const key = `${vendorId}:${deviceId}:${name}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const discrete = vendor === "intel" ? !isIntelIntegratedGpu(name, deviceId) : true;
    const vramMiB = adapterRamToMib(ram);
    devices.push({
      slot: `win32-${devices.length}`,
      vendor,
      name,
      vendorId,
      deviceId,
      discrete,
      vramMiB: vramMiB > 0 ? vramMiB : undefined,
    });
  }
  return devices;
}

export function parseClinfoGpus(text: string): NamedMemory[] {
  const found: NamedMemory[] = [];
  let currentName: string | undefined;
  for (const raw of text.split(/\r?\n/)) {
    const nameMatch = /^ {2}Device Name\s{2,}(\S.*\S)\s*$/.exec(raw);
    if (nameMatch?.[1]) {
      currentName = nameMatch[1];
      continue;
    }
    const memMatch = /^ {2}Global memory size\s{2,}(\d+)/.exec(raw);
    if (memMatch?.[1] && currentName) {
      const vramMiB = bytesToMib(Number(memMatch[1]));
      if (vramMiB > 0) {
        found.push({ name: currentName, vramMiB, source: "clinfo" });
      }
      currentName = undefined;
    }
  }
  return found;
}

export function parseSyclLsVerbose(text: string): NamedMemory[] {
  const found: NamedMemory[] = [];
  let currentName: string | undefined;
  for (const raw of text.split(/\r?\n/)) {
    const gpuMatch = /\[(?:[^\]]*(?:gpu|level.zero|opencl)[^\]]*)\]\s*(.+)$/i.exec(raw.trim());
    if (gpuMatch?.[1] && /gpu|level.zero|opencl/i.test(raw)) {
      currentName = gpuMatch[1].replace(/,.*/, "").trim();
      continue;
    }
    const nameOnly = /^(?:gpu|device):\s*(.+)$/i.exec(raw.trim());
    if (nameOnly?.[1] && !/cpu/i.test(nameOnly[1])) {
      currentName = nameOnly[1].trim();
      continue;
    }
    const mem = /Global memory size:\s*(\d+)/i.exec(raw);
    if (mem?.[1] && currentName && !/cpu|fpga emulation/i.test(currentName)) {
      const vramMiB = bytesToMib(Number(mem[1]));
      if (vramMiB > 0) found.push({ name: currentName, vramMiB, source: "sycl-ls" });
      currentName = undefined;
    }
  }
  return found;
}

export function parseXpuSmiDiscovery(text: string): NamedMemory[] {
  const blocks = text.split(/(?=Device ID\s*[:=]|^\s*GPU\s+\d+)/im);
  const found: NamedMemory[] = [];
  const chunks = blocks.length > 1 ? blocks : [text];
  for (const block of chunks) {
    const name =
      /Device Name\s*[:=]\s*(.+)$/im.exec(block)?.[1]?.trim() ??
      /GPU Name\s*[:=]\s*(.+)$/im.exec(block)?.[1]?.trim();
    const memLine =
      /Memory Physical Size[^\d]*(\d+(?:\.\d+)?)\s*(MiB|GiB|MB|GB)?/i.exec(block) ??
      /Memory\s*[:=]\s*(\d+(?:\.\d+)?)\s*(MiB|GiB|MB|GB)?/i.exec(block);
    if (!name || /cpu/i.test(name) || !memLine?.[1]) continue;
    let vramMiB = Number(memLine[1]);
    const unit = (memLine[2] ?? "MiB").toLowerCase();
    if (unit.startsWith("g")) vramMiB *= 1024;
    if (Number.isFinite(vramMiB) && vramMiB > 0) {
      found.push({ name, vramMiB: Math.round(vramMiB), source: text.includes("xpumcli") ? "xpumcli" : "xpu-smi" });
    }
  }
  if (found.length === 0) {
    const names = [...text.matchAll(/Device Name\s*[:=]\s*(.+)$/gim)].map((m) => m[1]?.trim() ?? "");
    const mems = [...text.matchAll(/(\d+)\s*MiB/gi)];
    if (names.length > 0 && mems.length >= names.length) {
      names.forEach((name, i) => {
        const vram = Number(mems[i]?.[1]);
        if (name && vram > 0) found.push({ name, vramMiB: vram, source: "xpu-smi" });
      });
    }
  }
  return found;
}

export function parseRocmSmiMeminfo(text: string): NamedMemory[] {
  const found: NamedMemory[] = [];
  const csv = /GPU\[(\d+)].*vram Total Memory \(B\):\s*(\d+)/gi;
  let match: RegExpExecArray | null;
  while ((match = csv.exec(text))) {
    const vramMiB = bytesToMib(Number(match[2]));
    if (vramMiB > 0) {
      found.push({ name: `AMD GPU ${match[1]}`, vramMiB, source: "rocm-smi" });
    }
  }
  if (found.length > 0) return found;
  const alt = /GPU\[(\d+)\][^\n]*VRAM[^\n]*?(\d+)\s*(MiB|MB|GiB|GB)/gi;
  while ((match = alt.exec(text))) {
    let vram = Number(match[2]);
    if ((match[3] ?? "").toLowerCase().startsWith("g")) vram *= 1024;
    if (vram > 0) found.push({ name: `AMD GPU ${match[1]}`, vramMiB: Math.round(vram), source: "rocm-smi" });
  }
  return found;
}

export function parseAmdSmi(text: string): NamedMemory[] {
  const found: NamedMemory[] = [];
  const blocks = text.split(/(?=GPU\s+\d+|^\s*asic)/im);
  for (const block of blocks) {
    const idx = /GPU\s+(\d+)/i.exec(block)?.[1];
    const name = /(?:market_name|card_series|device_name)\s*[:=]\s*(.+)$/im.exec(block)?.[1]?.trim();
    const mem =
      /vram[^\d]*(\d+)\s*(MiB|MB|GiB|GB)/i.exec(block) ??
      /size_mib\s*[:=]\s*(\d+)/i.exec(block);
    if (!mem?.[1]) continue;
    let vramMiB = Number(mem[1]);
    if ((mem[2] ?? "").toLowerCase().startsWith("g")) vramMiB *= 1024;
    if (vramMiB > 0) {
      found.push({
        name: name || `AMD GPU ${idx ?? found.length}`,
        vramMiB: Math.round(vramMiB),
        source: "amd-smi",
      });
    }
  }
  return found;
}

export function parseTorchXpu(text: string): NamedMemory[] {
  const found: NamedMemory[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    const parts = line.split("|");
    if (parts.length >= 3) {
      const name = parts[1]?.trim();
      const bytes = Number(parts[2]);
      if (name && bytes > 0) found.push({ name, vramMiB: bytesToMib(bytes), source: "torch.xpu" });
      continue;
    }
    const jsonish = /name['":\s]+([^"',]+).*?(?:total_memory|memory)['":\s]+(\d+)/i.exec(line);
    if (jsonish?.[1] && jsonish[2]) {
      found.push({ name: jsonish[1].trim(), vramMiB: bytesToMib(Number(jsonish[2])), source: "torch.xpu" });
    }
  }
  return found;
}

export function readRamMiB(readFile: (path: string) => string = readUtf8): number {
  if (!isWindows()) {
    try {
      const text = readFile("/proc/meminfo");
      const match = /^MemTotal:\s+(\d+)\s*kB/m.exec(text);
      if (match?.[1]) {
        const kb = Number(match[1]);
        if (Number.isFinite(kb) && kb > 0) return Math.round(kb / 1024);
      }
    } catch {
      // fall through to os.totalmem
    }
  }
  return Math.max(1, Math.round(totalmem() / 1024 / 1024));
}

function readUtf8(path: string): string {
  return readFileSync(path, "utf8");
}

function vendorFromPci(vendorId: string, name: string): AcceleratorVendor | undefined {
  if (vendorId === NVIDIA_VENDOR || /nvidia/i.test(name)) return "nvidia";
  if (vendorId === AMD_VENDOR || /advanced micro devices|\bamd\b|ati\b|radeon/i.test(name)) return "amd";
  if (vendorId === INTEL_VENDOR || /intel/i.test(name)) return "intel";
  return undefined;
}

function vendorFromSysfs(card: SysfsCard): AcceleratorVendor | undefined {
  const id = card.vendorId.toLowerCase().replace(/^0x/, "");
  if (id === NVIDIA_VENDOR) return "nvidia";
  if (id === AMD_VENDOR) return "amd";
  if (id === INTEL_VENDOR) return "intel";
  return undefined;
}

function normalizeHex(value: string): string {
  return value.toLowerCase().replace(/^0x/, "");
}

function zipNamedMemory(targets: Accelerator[], memories: NamedMemory[], vendor: AcceleratorVendor): void {
  const usable = memories.filter((row) => {
    if (vendor === "intel") return !isIntelIntegratedGpu(row.name);
    return true;
  });
  if (usable.length === 0) return;
  const unmatched = targets.filter((row) => row.vramMiB <= 0);
  const n = Math.min(unmatched.length || targets.length, usable.length);
  const dest = unmatched.length === usable.length || unmatched.length === 0 ? (unmatched.length ? unmatched : targets) : unmatched;
  for (let i = 0; i < n; i++) {
    const gpu = dest[i];
    const mem = usable[i];
    if (!gpu || !mem) continue;
    if (gpu.vramMiB <= 0 && mem.vramMiB > 0) {
      gpu.vramMiB = mem.vramMiB;
      gpu.vramEstimated = false;
      gpu.source = joinSources(gpu.source, mem.source);
    }
    if (mem.name && (gpu.name.length < mem.name.length || /battlemage|device e2|intel graphics/i.test(gpu.name))) {
      gpu.name = mem.name;
    }
  }
}

function joinSources(existing: string, extra: string): string {
  const parts = new Set(
    `${existing}+${extra}`
      .split("+")
      .map((part) => part.trim())
      .filter(Boolean),
  );
  return [...parts].join("+");
}

function applySkuFallback(gpu: Accelerator): void {
  if (gpu.vramMiB > 0) return;
  const sku = gpu.vendor === "intel" ? estimateIntelSkuVram(gpu.name, gpu.deviceId) : undefined;
  if (!sku) return;
  gpu.vramMiB = sku.vramMiB;
  gpu.vramEstimated = true;
  gpu.source = joinSources(gpu.source, "sku-fallback");
  if (!/arc/i.test(gpu.name)) gpu.name = sku.label;
}

function toAccelerator(partial: Omit<Accelerator, "index"> & { index?: number }): Accelerator {
  return {
    vendor: partial.vendor,
    name: partial.name,
    vramMiB: partial.vramMiB,
    index: partial.index ?? 0,
    source: partial.source,
    vramEstimated: partial.vramEstimated,
    deviceId: partial.deviceId,
  };
}

function listSysfsCardsLive(): SysfsCard[] {
  const drm = "/sys/class/drm";
  if (!existsSync(drm)) return [];
  const cards: SysfsCard[] = [];
  let names: string[] = [];
  try {
    names = readdirSync(drm);
  } catch {
    return [];
  }
  for (const name of names) {
    if (!/^card\d+$/.test(name)) continue;
    const deviceDir = join(drm, name, "device");
    try {
      const vendorId = normalizeHex(readFileSync(join(deviceDir, "vendor"), "utf8").trim());
      const deviceId = normalizeHex(readFileSync(join(deviceDir, "device"), "utf8").trim());
      let driver: string | undefined;
      try {
        const uevent = readFileSync(join(deviceDir, "uevent"), "utf8");
        driver = /^DRIVER=(.+)$/m.exec(uevent)?.[1]?.trim();
      } catch {
        // ignore
      }
      const vramBytes = readSysfsVramBytes(deviceDir);
      cards.push({ name, vendorId, deviceId, driver, vramBytes: vramBytes > 0 ? vramBytes : undefined });
    } catch {
      // skip unreadable card
    }
  }
  return cards;
}

function readSysfsVramBytes(deviceDir: string): number {
  const files: string[] = [join(deviceDir, "mem_info_vram_total")];
  try {
    for (const entry of readdirSync(deviceDir)) {
      if (/^tile\d+$/.test(entry)) {
        files.push(join(deviceDir, entry, "physical_vram_size_bytes"));
      }
    }
  } catch {
    // ignore
  }
  try {
    const drmDir = join(deviceDir, "drm");
    if (existsSync(drmDir)) {
      for (const entry of readdirSync(drmDir)) {
        if (/^card\d+$/.test(entry)) files.push(join(drmDir, entry, "lmem_total_bytes"));
      }
    }
  } catch {
    // ignore
  }
  try {
    const resolved = realpathSync(deviceDir);
    files.push(join(resolved, "mem_info_vram_total"));
    try {
      for (const entry of readdirSync(resolved)) {
        if (/^tile\d+$/.test(entry)) files.push(join(resolved, entry, "physical_vram_size_bytes"));
      }
    } catch {
      // ignore
    }
  } catch {
    // ignore
  }
  let total = 0;
  const seen = new Set<string>();
  for (const file of files) {
    if (seen.has(file) || !existsSync(file)) continue;
    seen.add(file);
    try {
      const n = Number(readFileSync(file, "utf8").trim());
      if (Number.isFinite(n) && n > 0) total += n;
    } catch {
      // ignore
    }
  }
  return total;
}

function liveProbe(name: string, args: string[], timeout?: number): string | null {
  const resolved = which(name);
  if (resolved) {
    const fromResolved = runCapture(resolved, args, timeout);
    if (fromResolved) return fromResolved;
  }
  return runCapture(name, args, timeout);
}

function runNvidiaSmi(): string | null {
  return liveProbe("nvidia-smi", [
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
  ]);
}

function collectIntelMemories(probes: HardwareProbes | undefined, live: boolean): NamedMemory[] {
  const memories: NamedMemory[] = [];
  const take = (key: keyof HardwareProbes, runner: () => string | null, parse: (text: string) => NamedMemory[]) => {
    const injected = probes ? probes[key] : undefined;
    const text = probes ? (typeof injected === "string" ? injected : null) : runner();
    if (!text) return;
    memories.push(...parse(text));
  };
  take("xpuSmi", () => liveProbe("xpu-smi", ["discovery"]), (text) =>
    parseXpuSmiDiscovery(text).map((row) => ({ ...row, source: "xpu-smi" })),
  );
  take("xpumcli", () => liveProbe("xpumcli", ["discovery"]), (text) =>
    parseXpuSmiDiscovery(text).map((row) => ({ ...row, source: "xpumcli" })),
  );
  take("syclLsVerbose", () => liveProbe("sycl-ls", ["--verbose"], 6000), parseSyclLsVerbose);
  if (memories.length === 0) {
    take("syclLs", () => liveProbe("sycl-ls", []), (text) => {
      const rows: NamedMemory[] = [];
      for (const line of text.split(/\r?\n/)) {
        const m = /\[(?:[^\]]*gpu[^\]]*)\]\s*(.+)$/i.exec(line.trim());
        if (m?.[1] && !/cpu|fpga emulation/i.test(m[1])) {
          rows.push({ name: m[1].replace(/,.*/, "").trim(), vramMiB: 0, source: "sycl-ls" });
        }
      }
      return rows;
    });
  }
  take("clinfo", () => liveProbe("clinfo", [], 8000), parseClinfoGpus);
  take("torchXpu", () => (live ? probeTorchXpu() : null), parseTorchXpu);
  take("intelGpuTop", () => liveProbe("intel_gpu_top", ["-L"], 2000), () => []);
  return memories;
}

function probeTorchXpu(): string | null {
  const script =
    "import torch\n" +
    "n=torch.xpu.device_count() if hasattr(torch,'xpu') else 0\n" +
    "out=[]\n" +
    "for i in range(n):\n" +
    " p=torch.xpu.get_device_properties(i)\n" +
    " mem=getattr(p,'total_memory',0) or getattr(p,'total_mem',0) or 0\n" +
    " out.append(f'{i}|{getattr(p,\"name\", \"Intel XPU\")}|{int(mem)}')\n" +
    "print('\\n'.join(out))";
  for (const py of pythonInterpreterNames()) {
    const result = spawnSync(py, [...pythonDashArgs(py), "-c", script], { encoding: "utf8", timeout: 8_000, windowsHide: true });
    if (result.status === 0 && result.stdout?.trim()) return result.stdout;
  }
  return null;
}

function mergePciDevices(lspci: PciDisplayDevice[], win32: PciDisplayDevice[]): PciDisplayDevice[] {
  const out = [...lspci];
  for (const row of win32) {
    const dup = out.some(
      (existing) =>
        existing.vendorId === row.vendorId &&
        existing.deviceId === row.deviceId &&
        existing.vendor === row.vendor,
    );
    if (!dup) out.push(row);
  }
  return out;
}

function acceleratorsFromPci(
  pci: PciDisplayDevice[],
  sysfs: SysfsCard[],
): Accelerator[] {
  const usedSysfs = new Set<number>();
  const result: Accelerator[] = [];
  for (const dev of pci) {
    if (!dev.discrete && dev.vendor === "intel") continue;
    let vramMiB = 0;
    let source = dev.slot.startsWith("win32") ? "win32" : "lspci";
    const idx = sysfs.findIndex((card, i) => {
      if (usedSysfs.has(i)) return false;
      return (
        normalizeHex(card.vendorId) === dev.vendorId && normalizeHex(card.deviceId) === dev.deviceId
      );
    });
    if (idx >= 0) {
      usedSysfs.add(idx);
      const card = sysfs[idx];
      if (card?.vramBytes) {
        vramMiB = bytesToMib(card.vramBytes);
        source = "lspci+sysfs";
      }
    }
    const skuName = dev.vendor === "intel" ? INTEL_DISCRETE_DEVICE_IDS[dev.deviceId]?.name : undefined;
    if (!vramMiB && dev.vramMiB) {
      vramMiB = dev.vramMiB;
      source = joinSources(source, "AdapterRAM");
    }
    result.push(
      toAccelerator({
        vendor: dev.vendor,
        name: skuName && /intel graphics/i.test(dev.name) ? skuName : dev.name,
        vramMiB,
        source,
        deviceId: dev.deviceId,
      }),
    );
  }
  for (const [i, card] of sysfs.entries()) {
    if (usedSysfs.has(i)) continue;
    const vendor = vendorFromSysfs(card);
    if (!vendor) continue;
    if (vendor === "intel" && isIntelIntegratedGpu(card.deviceId, card.deviceId)) continue;
    const sku = vendor === "intel" ? INTEL_DISCRETE_DEVICE_IDS[normalizeHex(card.deviceId)] : undefined;
    result.push(
      toAccelerator({
        vendor,
        name: sku?.name ?? `${vendor} GPU ${card.deviceId}`,
        vramMiB: card.vramBytes ? bytesToMib(card.vramBytes) : 0,
        source: card.vramBytes ? "sysfs" : "sysfs",
        deviceId: normalizeHex(card.deviceId),
      }),
    );
  }
  return result;
}

function pickPrimaryBackend(accelerators: Accelerator[]): LaunchBackend {
  if (accelerators.length === 0) return "cpu";
  const vendors: AcceleratorVendor[] = ["nvidia", "intel", "amd"];
  const rank: Record<LaunchBackend, number> = { cuda: 0, "intel-xpu": 1, rocm: 2, cpu: 3 };
  let best: { backend: LaunchBackend; total: number; max: number; count: number } | undefined;
  for (const vendor of vendors) {
    const group = accelerators.filter((row) => row.vendor === vendor);
    if (group.length === 0) continue;
    const total = group.reduce((sum, row) => sum + row.vramMiB, 0);
    const max = group.reduce((m, row) => Math.max(m, row.vramMiB), 0);
    const candidate = { backend: backendForVendor(vendor), total, max, count: group.length };
    if (
      !best ||
      candidate.total > best.total ||
      (candidate.total === best.total && candidate.max > best.max) ||
      (candidate.total === best.total && candidate.max === best.max && candidate.count > best.count) ||
      (candidate.total === best.total &&
        candidate.max === best.max &&
        candidate.count === best.count &&
        rank[candidate.backend] < rank[best.backend])
    ) {
      best = candidate;
    }
  }
  return best?.backend ?? "cpu";
}

function indexByVendor(accelerators: Accelerator[]): Accelerator[] {
  const counts: Partial<Record<AcceleratorVendor, number>> = {};
  return accelerators.map((row) => {
    const index = counts[row.vendor] ?? 0;
    counts[row.vendor] = index + 1;
    return { ...row, index };
  });
}

let liveCache: { at: number; value: HardwareSnapshot } | undefined;

export function detectHardware(options?: {
  nvidiaSmiOutput?: string | null;
  runNvidiaSmi?: () => string | null;
  probes?: HardwareProbes;
  readFile?: (path: string) => string;
}): HardwareSnapshot {
  const injected =
    options?.probes !== undefined ||
    options?.nvidiaSmiOutput !== undefined ||
    options?.runNvidiaSmi !== undefined;
  if (!injected && liveCache && Date.now() - liveCache.at < 10_000) {
    return liveCache.value;
  }
  const snapshot = detectHardwareUncached(options);
  if (!injected) liveCache = { at: Date.now(), value: snapshot };
  return snapshot;
}

function detectHardwareUncached(options?: {
  nvidiaSmiOutput?: string | null;
  runNvidiaSmi?: () => string | null;
  probes?: HardwareProbes;
  readFile?: (path: string) => string;
}): HardwareSnapshot {
  const notes: string[] = [];
  const probes = options?.probes;
  const live = probes === undefined;
  const readFile = options?.readFile ?? readUtf8;

  let smiText: string | null | undefined = options?.nvidiaSmiOutput;
  if (smiText === undefined && probes) smiText = probes.nvidiaSmi ?? null;
  if (smiText === undefined) {
    smiText = options?.runNvidiaSmi ? options.runNvidiaSmi() : runNvidiaSmi();
  }
  const hasNvidiaSmi = smiText !== null && smiText !== undefined;

  const lspciText = probes ? (probes.lspci ?? null) : isWindows() ? null : liveProbe("lspci", ["-nn"]);
  const win32Text = probes ? (probes.win32Video ?? null) : live ? probeWin32VideoControllers() : null;
  const pci = mergePciDevices(
    lspciText ? parseLspciDisplayDevices(lspciText) : [],
    win32Text ? parseWin32VideoControllers(win32Text) : [],
  );
  const sysfs = probes?.sysfsCards ?? (live && !isWindows() ? listSysfsCardsLive() : []);

  const nvidiaFromSmi = smiText ? parseNvidiaSmiCsv(smiText) : [];
  if (smiText !== null && smiText !== undefined && nvidiaFromSmi.length === 0 && hasNvidiaSmi) {
    notes.push("nvidia-smi ran but reported no GPUs.");
  }

  let accelerators = acceleratorsFromPci(pci, sysfs);

  for (const gpu of nvidiaFromSmi) {
    const exists = accelerators.some(
      (row) => row.vendor === "nvidia" && (row.name === gpu.name || accelerators.filter((a) => a.vendor === "nvidia").length === nvidiaFromSmi.length),
    );
    if (!exists || !accelerators.some((row) => row.vendor === "nvidia")) {
      accelerators.push(
        toAccelerator({
          vendor: "nvidia",
          name: gpu.name,
          vramMiB: gpu.vramMiB,
          source: "nvidia-smi",
        }),
      );
    }
  }
  if (nvidiaFromSmi.length > 0) {
    const nvidia = accelerators.filter((row) => row.vendor === "nvidia");
    if (nvidia.length === nvidiaFromSmi.length) {
      nvidia.forEach((row, i) => {
        const smi = nvidiaFromSmi[i];
        if (!smi) return;
        row.vramMiB = smi.vramMiB;
        row.name = smi.name;
        row.source = "nvidia-smi";
        row.vramEstimated = false;
      });
    } else if (nvidia.length === 0) {
      for (const gpu of nvidiaFromSmi) {
        accelerators.push(
          toAccelerator({
            vendor: "nvidia",
            name: gpu.name,
            vramMiB: gpu.vramMiB,
            source: "nvidia-smi",
          }),
        );
      }
    }
  }

  const intelMem = collectIntelMemories(probes, live);
  zipNamedMemory(
    accelerators.filter((row) => row.vendor === "intel"),
    intelMem,
    "intel",
  );

  if (!accelerators.some((row) => row.vendor === "intel")) {
    for (const mem of intelMem.filter((row) => !isIntelIntegratedGpu(row.name))) {
      if (mem.vramMiB <= 0) continue;
      accelerators.push(toAccelerator({ vendor: "intel", name: mem.name, vramMiB: mem.vramMiB, source: mem.source }));
    }
  }

  const rocmText = probes ? (probes.rocmSmi ?? null) : liveProbe("rocm-smi", ["--showmeminfo", "vram"]);
  const amdSmiText = probes ? (probes.amdSmi ?? null) : liveProbe("amd-smi", ["static", "--vram"]);
  const amdMem = [...(rocmText ? parseRocmSmiMeminfo(rocmText) : []), ...(amdSmiText ? parseAmdSmi(amdSmiText) : [])];
  zipNamedMemory(
    accelerators.filter((row) => row.vendor === "amd"),
    amdMem,
    "amd",
  );
  if (!accelerators.some((row) => row.vendor === "amd")) {
    for (const mem of amdMem) {
      accelerators.push(toAccelerator({ vendor: "amd", name: mem.name, vramMiB: mem.vramMiB, source: mem.source }));
    }
  }

  const igpuPci = pci.filter((dev) => dev.vendor === "intel" && !dev.discrete);
  if (igpuPci.length > 0) {
    notes.push(
      `Ignored ${igpuPci.length} Intel integrated GPU(s) for model fitting (shared memory, not a discrete accelerator).`,
    );
  }

  accelerators = accelerators.filter((row) => {
    if (row.vendor !== "intel") return true;
    return !isIntelIntegratedGpu(row.name, row.deviceId);
  });

  for (const gpu of accelerators) applySkuFallback(gpu);
  accelerators = accelerators.filter((row) => row.vramMiB > 0 || row.vendor !== "intel" || Boolean(row.deviceId));
  accelerators = indexByVendor(accelerators.filter((row) => row.vramMiB > 0));

  const primaryBackend = pickPrimaryBackend(accelerators);
  const primary = accelerators.filter((row) => backendForVendor(row.vendor) === primaryBackend);
  const vramMiB = primary.reduce((max, gpu) => Math.max(max, gpu.vramMiB), 0);
  const minVramMiB = primary.length > 0 ? primary.reduce((min, gpu) => Math.min(min, gpu.vramMiB), primary[0]!.vramMiB) : 0;
  const totalVramMiB = primary.reduce((sum, gpu) => sum + gpu.vramMiB, 0);
  const ramMiB = readRamMiB(readFile);
  const cpuCount = Math.max(1, cpus().length);
  const constrained = primaryBackend === "cpu" || primary.length === 0;

  if (constrained) {
    notes.push("No discrete GPU accelerator found. CPU is last resort; tiny CPU-feasible models only.");
    if (isWindows() && live) {
      notes.push(
        "On Windows, GPU detect uses nvidia-smi, AMD/Intel vendor CLIs when present, and Win32 video controllers — not Linux lspci/sysfs. If you have a discrete GPU, put nvidia-smi (or amd-smi / xpu-smi) on PATH.",
      );
    }
  } else {
    const estimated = primary.filter((row) => row.vramEstimated).length;
    if (estimated > 0) {
      notes.push(`${estimated} GPU VRAM value(s) are SKU estimates (live probe did not report memory).`);
    }
    notes.push(
      `Primary launch backend: ${primaryBackend} (${primary.length} device(s), ${vramMiB} MiB per-GPU max, ${totalVramMiB} MiB total).`,
    );
  }

  const gpus: GpuInfo[] = accelerators.map((row) => ({
    index: row.index,
    name: row.name,
    vramMiB: row.vramMiB,
    vendor: row.vendor,
    source: row.source,
  }));

  return {
    accelerators,
    primaryBackend,
    totalVramMiB,
    deviceCount: primary.length,
    gpus,
    vramMiB,
    minVramMiB,
    ramMiB,
    cpuCount,
    hasNvidiaSmi,
    constrained,
    notes,
  };
}
