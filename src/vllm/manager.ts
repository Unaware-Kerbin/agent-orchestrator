import { spawn, spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { accessSync, closeSync, constants, existsSync, openSync, readFileSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { delimiter, dirname, join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import type { EventEmitter } from "node:events";
import { detectHardware, type LaunchBackend } from "../hardware.js";
import { readConfigYaml, writeConfigYaml } from "../config.js";
import { ensureLocalVllmDummyKey } from "../secrets.js";
import { stateDir } from "../state.js";
import {
  patchVllmOrchestratorYaml,
  vllmBackendIdForModel,
  vllmContainerNameForModel,
  vllmSpecialistIdForModel,
} from "./upsert.js";
import type { OrchestratorConfig } from "../types.js";
import {
  buildIntelDockerRunArgs,
  formatDockerCommand,
  dockerContainerPid,
  dockerContainerRunning,
  dockerLogs,
  dockerRmName,
  dockerRunDetached,
  dockerStopContainer,
  intelDockerInstallHint,
  listIntelVllmImages,
  modelsDirFromPath,
  resolveDockerGroups,
  type IntelDockerCatalog,
} from "./intel-docker.js";

export const VLLM_LOOPBACK_HOST = "127.0.0.1" as const;
export const VLLM_PORT_START = 8000;
export const VLLM_PORT_END = 8099;
export const VLLM_INSTALL_HINT =
  "vLLM is not installed. Install with: pip install vllm  (NVIDIA CUDA matching your driver). Then ensure `vllm` or `python3 -m vllm` is on PATH.";
export const VLLM_XPU_INSTALL_HINT = `Intel Arc/XPU: stock \`pip install vllm\` is CUDA-only and will not run on Arc.

Supported path when local images exist: Docker (intel/llm-scaler-vllm:0.21.0-b3 preferred for Arc Pro / Battlemage, then other intel/llm-scaler-vllm tags, then intel/vllm:0.17.0-xpu). Bind 127.0.0.1 only.

Host fallback if Docker is down or those images are missing:
1. Intel GPU driver (xe) working; user in the render group
2. Intel oneAPI 2024.2 or later: source /opt/intel/oneapi/setvars.sh
3. From the vLLM repo: pip install -r requirements/xpu.txt   # PyTorch XPU + IPEX
4. VLLM_TARGET_DEVICE=xpu pip install .     # or: VLLM_TARGET_DEVICE=xpu python setup.py install

Then: vllm serve MODEL --host 127.0.0.1 --device xpu --dtype float16 --enforce-eager --gpu-memory-utilization 0.9
Dual GPU: VLLM_WORKER_MULTIPROC_METHOD=spawn ZE_AFFINITY_MASK=0,1 vllm serve … --tensor-parallel-size 2
FP16 is the default on Arc (BF16 is for Intel Data Center GPU, not Arc).`;
export const VLLM_ROCM_INSTALL_HINT =
  "AMD ROCm: stock pip CUDA wheels will not use the GPU. Install a ROCm vLLM build (vLLM docs: build with HIP / use the ROCm Docker image). Then ensure `vllm` is on PATH. Launch uses HIP_VISIBLE_DEVICES.";
export const VLLM_CPU_INSTALL_HINT =
  "No discrete accelerator detected. CPU vLLM is unsupported here; install a GPU backend or pick a tiny CPU-feasible model knowing it may still fail.";

export type VllmProcessRuntime = "host" | "docker";
export type VllmPhase = "idle" | "starting" | "running" | "error";
export const VLLM_HEALTH_TIMEOUT_MS = 600_000;

export interface VllmStartJob {
  id: string;
  status: "starting" | "running" | "error";
  modelId: string;
  startedAt: number;
  updatedAt: number;
  error?: string;
  lastLog?: string;
  command?: string;
}

export interface VllmStartAccepted {
  status: "starting" | "running";
  jobId: string;
  vllm: VllmRuntimeState;
}

export type VllmStartInput = {
  modelId: string;
  hfRepo: string;
  modelPath: string;
  quantization?: string;
  port?: number;
  host?: string;
  timeoutMs?: number;
  backendId?: string;
  specialistId?: string;
  containerName?: string;
  /** When true, stop the existing instance of this model (same backend id) before starting. Does not stop other models. */
  replace?: boolean;
  backend?: LaunchBackend;
  tensorParallel?: number;
  deviceCount?: number;
  image?: string;
  runtime?: VllmProcessRuntime;
};

export interface VllmInstanceSelector {
  modelId?: string;
  backendId?: string;
  all?: boolean;
}

export interface VllmInstanceState {
  running: boolean;
  pid?: number;
  port?: number;
  host: typeof VLLM_LOOPBACK_HOST;
  modelId?: string;
  hfRepo?: string;
  modelPath?: string;
  servedModelName?: string;
  quantization?: string;
  backendId: string;
  specialistId?: string;
  startedAt?: number;
  lastError?: string;
  lastLog?: string;
  backend?: LaunchBackend;
  runtime?: VllmProcessRuntime;
  image?: string;
  containerId?: string;
  containerName?: string;
  phase: VllmPhase;
  healthy: boolean;
  jobId?: string;
}

export interface VllmRuntimeState extends VllmInstanceState {
  installed: boolean;
  installHint: string;
  intelDocker?: IntelDockerCatalog;
  jobId?: string;
  startJob?: VllmStartJob;
  instances: VllmInstanceState[];
}

export function resolveVllmPhase(input: {
  jobStatus?: VllmStartJob["status"];
  processAlive: boolean;
}): { phase: VllmPhase; healthy: boolean } {
  if (input.jobStatus === "starting") return { phase: "starting", healthy: false };
  if (input.jobStatus === "error") return { phase: "error", healthy: false };
  if (input.jobStatus === "running" || input.processAlive) return { phase: "running", healthy: true };
  return { phase: "idle", healthy: false };
}

function newStartJobId(): string {
  return `vllm-${randomBytes(8).toString("hex")}`;
}

interface VllmStateFile {
  pid: number;
  port: number;
  host: typeof VLLM_LOOPBACK_HOST;
  modelId: string;
  hfRepo: string;
  modelPath: string;
  servedModelName: string;
  quantization?: string;
  backendId: string;
  specialistId?: string;
  startedAt: number;
  runtime?: VllmProcessRuntime;
  image?: string;
  containerId?: string;
  containerName?: string;
}

function statePath(): string {
  return join(stateDir(), "vllm.json");
}

function logPath(backendId?: string): string {
  const slug = backendId?.replace(/[^a-zA-Z0-9_-]/g, "-") || "vllm";
  return join(stateDir(), `${slug}.log`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function assertVllmLoopbackHost(host: string | undefined): typeof VLLM_LOOPBACK_HOST {
  if (!host || host === VLLM_LOOPBACK_HOST || host === "localhost") {
    return VLLM_LOOPBACK_HOST;
  }
  throw new Error(
    `Refusing to bind vLLM to "${host}". Only ${VLLM_LOOPBACK_HOST} is allowed so the server stays on this machine. Do not use 0.0.0.0, public interfaces, or tunnels.`,
  );
}

export function isVllmCmdline(cmdline: string): boolean {
  const text = cmdline.replaceAll("\0", " ").toLowerCase();
  return text.includes("vllm");
}

export function vllmInstallHint(backend: LaunchBackend = "cuda", docker?: IntelDockerCatalog): string {
  if (backend === "intel-xpu") {
    return `${intelDockerInstallHint(docker)}\n\nHost fallback (only if Docker is unavailable):\n${VLLM_XPU_INSTALL_HINT}`;
  }
  if (backend === "rocm") return VLLM_ROCM_INSTALL_HINT;
  if (backend === "cpu") return VLLM_CPU_INSTALL_HINT;
  return VLLM_INSTALL_HINT;
}

export function vllmLaunchEnv(
  backend: LaunchBackend,
  deviceCount: number,
  extra: NodeJS.ProcessEnv = process.env,
): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = { ...extra };
  const mask = Array.from({ length: Math.max(1, deviceCount) }, (_, i) => String(i)).join(",");
  if (backend === "intel-xpu") {
    env.VLLM_TARGET_DEVICE = env.VLLM_TARGET_DEVICE ?? "xpu";
    env.VLLM_WORKER_MULTIPROC_METHOD = env.VLLM_WORKER_MULTIPROC_METHOD ?? "spawn";
    env.ZE_AFFINITY_MASK = env.ZE_AFFINITY_MASK ?? mask;
    env.ONEAPI_DEVICE_SELECTOR = env.ONEAPI_DEVICE_SELECTOR ?? "level_zero:gpu";
  } else if (backend === "cuda") {
    env.CUDA_VISIBLE_DEVICES = env.CUDA_VISIBLE_DEVICES ?? mask;
  } else if (backend === "rocm") {
    env.HIP_VISIBLE_DEVICES = env.HIP_VISIBLE_DEVICES ?? mask;
    env.ROCR_VISIBLE_DEVICES = env.ROCR_VISIBLE_DEVICES ?? mask;
  }
  return env;
}

export function classifyVllmLog(text: string): string | undefined {
  if (/CUDA out of memory|OutOfMemoryError|torch\.OutOfMemoryError/i.test(text)) {
    return "GPU out of memory. Try a smaller catalog model (or a 4-bit AWQ/GPTQ variant on CUDA/ROCm).";
  }
  if (/No CUDA GPUs are available|cuda.*not available|NVIDIA driver/i.test(text)) {
    return "This vLLM build is CUDA-only and did not see an NVIDIA GPU. On Intel Arc install vLLM with VLLM_TARGET_DEVICE=xpu; on AMD use a ROCm build.";
  }
  if (/xpu.*not available|No XPU|Level Zero|zeInit|oneAPI/i.test(text)) {
    return "vLLM XPU did not see an Intel GPU. Source oneAPI (`source /opt/intel/oneapi/setvars.sh`), check `sycl-ls` / `clinfo`, and rebuild with VLLM_TARGET_DEVICE=xpu.";
  }
  if (/HIP.*not available|No HIP GPUs|rocm/i.test(text) && /not available|no gpu/i.test(text)) {
    return "This vLLM build did not see a ROCm GPU. Install a ROCm vLLM wheel/image and check `rocm-smi`.";
  }
  if (/ModuleNotFoundError: No module named 'vllm'|No module named vllm/i.test(text)) {
    return VLLM_INSTALL_HINT;
  }
  return undefined;
}

function which(cmd: string): string | undefined {
  const pathEnv = process.env.PATH ?? "";
  for (const dir of pathEnv.split(delimiter)) {
    const candidate = join(dir, cmd);
    try {
      accessSync(candidate, constants.X_OK);
      return candidate;
    } catch {
      // next
    }
  }
  return undefined;
}

export interface VllmLaunch {
  command: string;
  args: string[];
  backend?: LaunchBackend;
  runtime?: VllmProcessRuntime;
  image?: string;
}

export type VllmDeviceProbe = "cuda" | "xpu" | "rocm" | "unknown" | "missing";

export function detectVllmLaunch(options?: {
  backend?: LaunchBackend;
  whichFn?: (cmd: string) => string | undefined;
  pythonImportOk?: (python: string) => boolean;
  probeDevice?: (python: string) => VllmDeviceProbe;
  docker?: IntelDockerCatalog;
  image?: string;
  runtime?: VllmProcessRuntime;
}): { launch: VllmLaunch } | { missing: true; installHint: string } {
  const backend = options?.backend ?? "cuda";
  const docker = options?.docker;
  const requestedImage = options?.image?.trim();
  const hint = vllmInstallHint(backend, docker);

  const preferDocker =
    options?.runtime === "docker" ||
    Boolean(requestedImage) ||
    (backend === "intel-xpu" && options?.runtime !== "host" && Boolean(docker?.preferred));

  if (preferDocker) {
    const image = requestedImage || docker?.preferred?.ref;
    if (image) {
      return { launch: { command: "docker", args: ["run"], backend, runtime: "docker", image } };
    }
    if (options?.runtime === "docker" || requestedImage) {
      return { missing: true, installHint: hint };
    }
  }

  const whichFn = options?.whichFn ?? which;
  const importOk =
    options?.pythonImportOk ??
    ((python: string) => {
      const result = spawnSync(python, ["-c", "import vllm; print('ok')"], {
        encoding: "utf8",
        timeout: 8_000,
      });
      return result.status === 0 && (result.stdout ?? "").includes("ok");
    });
  const probeDevice = options?.probeDevice ?? defaultProbeVllmDevice;

  const wanted: VllmDeviceProbe =
    backend === "intel-xpu" ? "xpu" : backend === "rocm" ? "rocm" : backend === "cpu" ? "unknown" : "cuda";

  const pythonCandidates = ["python3", "python"].filter((py) => Boolean(whichFn(py)) || py === "python3");
  let probed: VllmDeviceProbe = "missing";
  let pythonWithVllm: string | undefined;
  for (const py of pythonCandidates) {
    try {
      probed = probeDevice(py);
      if (probed !== "missing") {
        pythonWithVllm = py;
        break;
      }
    } catch {
      // next interpreter
    }
  }

  const fromPath = whichFn("vllm");
  if (backend === "cpu") {
    return { missing: true, installHint: hint };
  }

  if (probed !== "missing" && probed !== "unknown" && probed !== wanted) {
    return {
      missing: true,
      installHint:
        `Detected a ${probed.toUpperCase()} vLLM install, which cannot drive ${backend}. ` + hint,
    };
  }

  if (backend === "intel-xpu" && (probed === "cuda" || (fromPath && probed === "unknown"))) {
    return {
      missing: true,
      installHint: `Stock CUDA vLLM cannot run on Intel Arc. ${hint}`,
    };
  }

  if (fromPath && (probed === wanted || probed === "unknown" || probed === "missing")) {
    if (backend === "intel-xpu" && probed !== "xpu") {
      return { missing: true, installHint: hint };
    }
    return { launch: { command: fromPath, args: ["serve"], backend, runtime: "host" } };
  }

  if (pythonWithVllm && importOk(pythonWithVllm) && (probed === wanted || probed === "unknown")) {
    if (backend === "intel-xpu" && probed !== "xpu") {
      return { missing: true, installHint: hint };
    }
    return {
      launch: {
        command: pythonWithVllm,
        args: ["-m", "vllm.entrypoints.openai.api_server"],
        backend,
        runtime: "host",
      },
    };
  }

  if (fromPath && backend === "cuda") {
    return { launch: { command: fromPath, args: ["serve"], backend, runtime: "host" } };
  }
  for (const py of pythonCandidates) {
    try {
      if (importOk(py) && backend === "cuda") {
        return {
          launch: {
            command: py,
            args: ["-m", "vllm.entrypoints.openai.api_server"],
            backend,
            runtime: "host",
          },
        };
      }
    } catch {
      // try next interpreter
    }
  }
  return { missing: true, installHint: hint };
}

function defaultProbeVllmDevice(python: string): VllmDeviceProbe {
  const script = `
import json
out = {"device": "missing"}
try:
    import vllm  # noqa: F401
    out["device"] = "unknown"
except Exception:
    print(json.dumps(out))
    raise SystemExit
try:
    from vllm.platforms import current_platform
    t = (getattr(current_platform, "device_type", None) or str(current_platform) or "").lower()
    if "xpu" in t: out["device"] = "xpu"
    elif "rocm" in t or "hip" in t: out["device"] = "rocm"
    elif "cuda" in t: out["device"] = "cuda"
except Exception:
    pass
if out["device"] == "unknown":
    try:
        import torch
        if hasattr(torch, "xpu"):
            out["device"] = "xpu"
        elif getattr(getattr(torch, "version", None), "hip", None):
            out["device"] = "rocm"
        elif getattr(getattr(torch, "version", None), "cuda", None):
            out["device"] = "cuda"
    except Exception:
        pass
print(json.dumps(out))
`;
  const result = spawnSync(python, ["-c", script], { encoding: "utf8", timeout: 10_000 });
  try {
    const parsed = JSON.parse((result.stdout ?? "").trim().split(/\n/).pop() ?? "{}") as { device?: string };
    if (parsed.device === "xpu" || parsed.device === "cuda" || parsed.device === "rocm" || parsed.device === "unknown") {
      return parsed.device;
    }
  } catch {
    // ignore
  }
  if (result.status !== 0) return "missing";
  return "unknown";
}

export function buildVllmArgv(input: {
  launch: VllmLaunch;
  modelPath: string;
  port: number;
  servedModelName: string;
  quantization?: string;
  backend?: LaunchBackend;
  tensorParallel?: number;
}): { command: string; args: string[] } {
  const host = VLLM_LOOPBACK_HOST;
  const backend = input.backend ?? input.launch.backend ?? "cuda";
  const args: string[] = [];
  if (input.launch.args[0] === "serve") {
    args.push("serve", input.modelPath, "--host", host, "--port", String(input.port), "--served-model-name", input.servedModelName);
  } else {
    args.push(...input.launch.args, "--model", input.modelPath, "--host", host, "--port", String(input.port), "--served-model-name", input.servedModelName);
  }
  if (input.quantization && backend !== "intel-xpu") {
    args.push("--quantization", input.quantization);
  }
  if (backend === "intel-xpu") {
    args.push("--device", "xpu", "--dtype", "float16", "--enforce-eager", "--gpu-memory-utilization", "0.9");
  }
  if (input.tensorParallel && input.tensorParallel > 1) {
    args.push("--tensor-parallel-size", String(input.tensorParallel));
  }
  if (args.includes("0.0.0.0")) {
    throw new Error("Internal error: refused to pass 0.0.0.0 to vLLM");
  }
  return { command: input.launch.command, args };
}

export function findFreePort(
  host: typeof VLLM_LOOPBACK_HOST = VLLM_LOOPBACK_HOST,
  start = VLLM_PORT_START,
  end = VLLM_PORT_END,
  used?: Iterable<number>,
): Promise<number> {
  const skip = new Set(used ?? []);
  const tryPort = (port: number): Promise<number> => {
    if (skip.has(port)) {
      if (port >= end) {
        return Promise.reject(new Error(`No free port in ${start}–${end} on ${host}`));
      }
      return tryPort(port + 1);
    }
    return new Promise((resolve, reject) => {
      const server = createServer();
      server.once("error", (err: NodeJS.ErrnoException) => {
        if (err.code === "EADDRINUSE") {
          if (port >= end) {
            reject(new Error(`No free port in ${start}–${end} on ${host}`));
            return;
          }
          resolve(tryPort(port + 1));
          return;
        }
        reject(err);
      });
      server.listen(port, host, () => {
        server.close((closeErr) => {
          if (closeErr) reject(closeErr);
          else resolve(port);
        });
      });
    });
  };
  return tryPort(start);
}

export function instanceMatches(
  inst: { modelId?: string; hfRepo?: string; backendId: string },
  sel: VllmInstanceSelector,
): boolean {
  if (sel.all) return true;
  if (sel.backendId && inst.backendId === sel.backendId) return true;
  if (sel.modelId && (inst.modelId === sel.modelId || inst.hfRepo === sel.modelId)) return true;
  return false;
}

export function partitionVllmInstances<T extends { modelId?: string; hfRepo?: string; backendId: string }>(
  instances: T[],
  sel: VllmInstanceSelector,
): { matched: T[]; rest: T[] } {
  const matched: T[] = [];
  const rest: T[] = [];
  for (const inst of instances) {
    if (instanceMatches(inst, sel)) matched.push(inst);
    else rest.push(inst);
  }
  return { matched, rest };
}

function readCmdline(pid: number): string {
  try {
    return readFileSync(`/proc/${pid}/cmdline`, "utf8");
  } catch {
    return "";
  }
}

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function parseStateInstance(raw: Record<string, unknown>, legacySingle = false): VllmStateFile | undefined {
  if (typeof raw.port !== "number") return undefined;
  const pid = typeof raw.pid === "number" ? raw.pid : 0;
  const containerId = typeof raw.containerId === "string" && raw.containerId ? raw.containerId : undefined;
  const runtime = raw.runtime === "docker" || raw.runtime === "host" ? raw.runtime : containerId ? "docker" : "host";
  if (!pid && !containerId && runtime !== "docker") return undefined;
  if (raw.host && raw.host !== VLLM_LOOPBACK_HOST) return undefined;
  const containerName =
    typeof raw.containerName === "string" && raw.containerName
      ? raw.containerName
      : legacySingle && runtime === "docker" && !containerId
        ? "orch-vllm"
        : typeof raw.containerName === "string"
          ? raw.containerName
          : undefined;
  if (runtime === "docker" && !pid && !containerId && !containerName) return undefined;
  return {
    pid,
    port: raw.port,
    host: VLLM_LOOPBACK_HOST,
    modelId: typeof raw.modelId === "string" ? raw.modelId : "",
    hfRepo: typeof raw.hfRepo === "string" ? raw.hfRepo : "",
    modelPath: typeof raw.modelPath === "string" ? raw.modelPath : "",
    servedModelName: typeof raw.servedModelName === "string" ? raw.servedModelName : "",
    quantization: typeof raw.quantization === "string" ? raw.quantization : undefined,
    backendId: typeof raw.backendId === "string" ? raw.backendId : vllmBackendIdForModel(String(raw.modelId ?? "local")),
    specialistId: typeof raw.specialistId === "string" ? raw.specialistId : undefined,
    startedAt: typeof raw.startedAt === "number" ? raw.startedAt : Date.now(),
    runtime,
    image: typeof raw.image === "string" ? raw.image : undefined,
    containerId,
    containerName,
  };
}

function readInstances(): VllmStateFile[] {
  const path = statePath();
  if (!existsSync(path)) return [];
  try {
    const raw = JSON.parse(readFileSync(path, "utf8")) as unknown;
    if (!isRecord(raw)) return [];
    if (Array.isArray(raw.instances)) {
      return raw.instances
        .map((row) => (isRecord(row) ? parseStateInstance(row, false) : undefined))
        .filter((row): row is VllmStateFile => Boolean(row));
    }
    const one = parseStateInstance(raw, true);
    return one ? [one] : [];
  } catch {
    return [];
  }
}

function writeInstances(instances: VllmStateFile[]): void {
  writeFileSync(
    statePath(),
    `${JSON.stringify({ version: 2, running: instances.length > 0, instances }, null, 2)}\n`,
    { encoding: "utf8", mode: 0o600 },
  );
}

function upsertInstance(next: VllmStateFile): void {
  const rest = readInstances().filter(
    (row) => row.backendId !== next.backendId && row.modelId !== next.modelId,
  );
  rest.push(next);
  writeInstances(rest);
}

function dropInstances(pred: (row: VllmStateFile) => boolean): VllmStateFile[] {
  const all = readInstances();
  const removed = all.filter(pred);
  writeInstances(all.filter((row) => !pred(row)));
  return removed;
}

function lastLogLines(backendId?: string, maxChars = 4000): string {
  try {
    const text = readFileSync(logPath(backendId), "utf8");
    return text.slice(-maxChars);
  } catch {
    return "";
  }
}

function killProcessGroup(pid: number, signal: NodeJS.Signals): void {
  try {
    process.kill(-pid, signal);
  } catch {
    try {
      process.kill(pid, signal);
    } catch {
      // already gone
    }
  }
}

export async function probeVllmModels(
  baseUrl: string,
  timeoutMs = 800,
): Promise<{ ok: boolean; modelIds: string[]; error?: string }> {
  const url = `${baseUrl.replace(/\/$/, "")}/models`;
  try {
    const response = await fetch(url, { method: "GET", signal: AbortSignal.timeout(timeoutMs) });
    const body = (await response.json().catch(() => ({}))) as { data?: Array<{ id?: string }> };
    const modelIds = Array.isArray(body.data)
      ? body.data.map((row) => row.id).filter((id): id is string => typeof id === "string")
      : [];
    return { ok: response.ok || response.status < 500, modelIds };
  } catch (error) {
    return { ok: false, modelIds: [], error: error instanceof Error ? error.message : String(error) };
  }
}

export class VllmManager {
  private installCache:
    | { at: number; result: ReturnType<typeof detectVllmLaunch>; backend: LaunchBackend; key: string }
    | undefined;
  private dockerCache: { at: number; value: IntelDockerCatalog } | undefined;
  private startJobs = new Map<string, VllmStartJob>();
  private pendingStarts = new Map<string, Promise<VllmRuntimeState>>();
  private abortStarts = new Set<string>();
  private reservedPorts = new Set<number>();

  constructor(
    private readonly events: EventEmitter,
    private readonly configPath: string,
    private readonly getConfig: () => OrchestratorConfig,
    private readonly reloadConfig: () => void,
  ) {}

  dockerCatalog(): IntelDockerCatalog {
    if (this.dockerCache && Date.now() - this.dockerCache.at < 15_000) return this.dockerCache.value;
    const value = listIntelVllmImages();
    this.dockerCache = { at: Date.now(), value };
    return value;
  }

  private detectedLaunch(
    backend?: LaunchBackend,
    extra?: { image?: string; runtime?: VllmProcessRuntime },
  ): ReturnType<typeof detectVllmLaunch> {
    const resolved = backend ?? detectHardware().primaryBackend;
    const docker = this.dockerCatalog();
    const key = `${resolved}:${extra?.runtime ?? ""}:${extra?.image ?? docker.preferred?.ref ?? ""}`;
    if (this.installCache && Date.now() - this.installCache.at < 60_000 && this.installCache.key === key) {
      return this.installCache.result;
    }
    const result = detectVllmLaunch({
      backend: resolved,
      docker,
      image: extra?.image,
      runtime: extra?.runtime,
    });
    this.installCache = { at: Date.now(), result, backend: resolved, key };
    return result;
  }

  private jobFor(modelId: string): VllmStartJob | undefined {
    return this.startJobs.get(modelId);
  }

  private latestJob(): VllmStartJob | undefined {
    let latest: VllmStartJob | undefined;
    for (const job of this.startJobs.values()) {
      if (!latest || job.updatedAt >= latest.updatedAt) latest = job;
    }
    return latest;
  }

  private setJob(modelId: string, job: VllmStartJob): void {
    this.startJobs.set(modelId, job);
  }

  private idsFor(modelId: string, input?: Pick<VllmStartInput, "backendId" | "specialistId" | "containerName">) {
    return {
      backendId: input?.backendId ?? vllmBackendIdForModel(modelId),
      specialistId: input?.specialistId ?? vllmSpecialistIdForModel(modelId),
      containerName: input?.containerName ?? vllmContainerNameForModel(modelId),
    };
  }

  status(): VllmRuntimeState {
    const hardware = detectHardware();
    const backend = hardware.primaryBackend;
    const docker = this.dockerCatalog();
    const detected = this.detectedLaunch(backend);
    const installed = !("missing" in detected);
    const hint = "missing" in detected ? detected.installHint : vllmInstallHint(backend, docker);
    const launchRuntime = !("missing" in detected) ? (detected.launch.runtime ?? "host") : undefined;
    const launchImage = !("missing" in detected) ? detected.launch.image : docker.preferred?.ref;
    const latestJob = this.latestJob();

    const stored = readInstances();
    const live: VllmInstanceState[] = [];
    const dead: VllmStateFile[] = [];
    for (const row of stored) {
      const job = this.jobFor(row.modelId);
      const dockerTarget = row.containerId || row.containerName;
      const dockerAlive = row.runtime === "docker" && dockerTarget ? dockerContainerRunning(dockerTarget) : false;
      const hostAlive = row.pid > 0 && pidAlive(row.pid) && isVllmCmdline(readCmdline(row.pid));
      const alive = row.runtime === "docker" ? dockerAlive : hostAlive;
      const resolved = resolveVllmPhase({
        jobStatus: job?.status,
        processAlive: alive,
      });
      if (!alive && job?.status !== "starting") {
        dead.push(row);
        continue;
      }
      live.push({
        running: alive && resolved.phase !== "starting",
        pid: row.pid || undefined,
        port: row.port,
        host: VLLM_LOOPBACK_HOST,
        modelId: row.modelId,
        hfRepo: row.hfRepo,
        modelPath: row.modelPath,
        servedModelName: row.servedModelName,
        quantization: row.quantization,
        backendId: row.backendId,
        specialistId: row.specialistId ?? vllmSpecialistIdForModel(row.modelId),
        startedAt: row.startedAt,
        lastLog: job?.lastLog,
        lastError: job?.status === "error" ? job.error : undefined,
        backend,
        runtime: row.runtime ?? launchRuntime,
        image: row.image ?? launchImage,
        containerId: row.containerId,
        containerName: row.containerName,
        phase: resolved.phase,
        healthy: resolved.healthy,
        jobId: job?.id,
      });
    }
    if (dead.length) {
      writeInstances(stored.filter((row) => !dead.some((d) => d.backendId === row.backendId && d.port === row.port)));
    }

    for (const [modelId, job] of this.startJobs) {
      if (job.status !== "starting") continue;
      if (live.some((row) => row.modelId === modelId)) continue;
      const ids = this.idsFor(modelId);
      live.push({
        running: false,
        host: VLLM_LOOPBACK_HOST,
        modelId,
        backendId: ids.backendId,
        specialistId: ids.specialistId,
        containerName: ids.containerName,
        phase: "starting",
        healthy: false,
        jobId: job.id,
        lastLog: job.lastLog,
        lastError: job.error,
        backend,
        runtime: launchRuntime,
        image: launchImage,
      });
    }

    const primary =
      live.find((row) => row.healthy) ??
      live.find((row) => row.phase === "starting") ??
      live.find((row) => row.phase === "error") ??
      live[0];
    const anyStarting = live.some((row) => row.phase === "starting") || [...this.startJobs.values()].some((j) => j.status === "starting");
    const anyError = live.some((row) => row.phase === "error") || latestJob?.status === "error";
    const anyRunning = live.some((row) => row.healthy || row.running);
    const phase: VllmPhase = anyStarting ? "starting" : anyRunning ? "running" : anyError ? "error" : "idle";

    return {
      running: anyRunning,
      pid: primary?.pid,
      port: primary?.port,
      host: VLLM_LOOPBACK_HOST,
      modelId: primary?.modelId ?? latestJob?.modelId,
      hfRepo: primary?.hfRepo,
      modelPath: primary?.modelPath,
      servedModelName: primary?.servedModelName,
      quantization: primary?.quantization,
      backendId: primary?.backendId ?? "vllm-local",
      specialistId: primary?.specialistId,
      startedAt: primary?.startedAt,
      lastError: primary?.lastError ?? (latestJob?.status === "error" ? latestJob.error : undefined),
      lastLog: primary?.lastLog ?? latestJob?.lastLog,
      installed,
      installHint: hint,
      backend,
      runtime: primary?.runtime ?? launchRuntime,
      image: primary?.image ?? launchImage,
      containerId: primary?.containerId,
      containerName: primary?.containerName,
      intelDocker: docker,
      phase,
      healthy: live.some((row) => row.healthy),
      jobId: primary?.jobId ?? latestJob?.id,
      startJob: latestJob,
      instances: live,
    };
  }

  beginStart(input: VllmStartInput): VllmStartAccepted {
    const current = this.status();
    const existingJob = this.jobFor(input.modelId);
    const pending = this.pendingStarts.get(input.modelId);
    if (pending && existingJob && existingJob.status === "starting") {
      return {
        status: "starting",
        jobId: existingJob.id,
        vllm: current,
      };
    }
    const already = current.instances.find(
      (row) => row.healthy && (row.modelId === input.modelId || row.hfRepo === input.hfRepo),
    );
    if (already && !input.replace) {
      return {
        status: "running",
        jobId: existingJob?.id ?? "already-running",
        vllm: current,
      };
    }
    this.abortStarts.delete(input.modelId);
    const job: VllmStartJob = {
      id: newStartJobId(),
      status: "starting",
      modelId: input.modelId,
      startedAt: Date.now(),
      updatedAt: Date.now(),
    };
    this.setJob(input.modelId, job);
    this.events.emit("vllm", this.status());
    const pendingStart = this.runStart(input)
      .then((ready) => {
        const currentJob = this.jobFor(input.modelId) ?? job;
        this.setJob(input.modelId, { ...currentJob, status: "running", updatedAt: Date.now() });
        const next = this.status();
        this.events.emit("vllm", next);
        return ready;
      })
      .catch((error) => {
        const message = error instanceof Error ? error.message : String(error);
        const currentJob = this.jobFor(input.modelId) ?? job;
        this.setJob(input.modelId, {
          ...currentJob,
          status: "error",
          error: message,
          lastLog: currentJob.lastLog ?? message,
          updatedAt: Date.now(),
        });
        this.events.emit("vllm", this.status());
        throw error;
      })
      .finally(() => {
        this.pendingStarts.delete(input.modelId);
      });
    this.pendingStarts.set(input.modelId, pendingStart);
    void pendingStart.catch(() => undefined);
    return { status: "starting", jobId: job.id, vllm: this.status() };
  }

  async start(input: VllmStartInput): Promise<VllmRuntimeState> {
    const pending = this.pendingStarts.get(input.modelId);
    if (pending) return pending;
    const accepted = this.beginStart(input);
    return this.pendingStarts.get(input.modelId) ?? Promise.resolve(accepted.vllm);
  }

  private async runStart(input: VllmStartInput): Promise<VllmRuntimeState> {
    assertVllmLoopbackHost(input.host);
    const ids = this.idsFor(input.modelId, input);
    const current = this.status();
    const same = current.instances.find(
      (row) => row.modelId === input.modelId || row.hfRepo === input.hfRepo || row.backendId === ids.backendId,
    );
    if (same?.healthy && !input.replace) {
      return current;
    }
    if (same && input.replace) {
      this.stopStoredAndDrop(same);
    }

    const hardware = detectHardware();
    const backend = input.backend ?? hardware.primaryBackend;
    const deviceCount = input.deviceCount ?? (hardware.deviceCount || 1);
    if (backend === "cpu") {
      throw new Error(VLLM_CPU_INSTALL_HINT);
    }
    if (backend === "intel-xpu" && input.quantization) {
      throw new Error(
        "Intel XPU serving uses FP16 (Arc does not load CUDA AWQ/GPTQ). Start the fp16 catalog snapshot instead.",
      );
    }

    this.installCache = undefined;
    this.dockerCache = undefined;
    const detected = this.detectedLaunch(backend, { image: input.image, runtime: input.runtime });
    if ("missing" in detected) {
      throw new Error(detected.installHint);
    }

    if (!existsSync(input.modelPath)) {
      throw new Error(
        `Model is not downloaded at ${input.modelPath}. Call download_local_model for ${input.modelId} first.`,
      );
    }

    const usedPorts = new Set<number>([
      ...current.instances.map((row) => row.port).filter((port): port is number => typeof port === "number"),
      ...this.reservedPorts,
    ]);
    for (const backend of Object.values(this.getConfig().backends)) {
      if (backend.type !== "vllm" || typeof backend.baseUrl !== "string") continue;
      try {
        const port = Number(new URL(backend.baseUrl).port);
        if (port >= VLLM_PORT_START && port <= VLLM_PORT_END) usedPorts.add(port);
      } catch {
        // ignore malformed baseUrl
      }
    }
    if (same?.port && input.replace) usedPorts.delete(same.port);
    const port = input.port ?? (await findFreePort(VLLM_LOOPBACK_HOST, VLLM_PORT_START, VLLM_PORT_END, usedPorts));
    if (port < VLLM_PORT_START || port > VLLM_PORT_END) {
      throw new Error(`Port ${port} is outside the local range ${VLLM_PORT_START}–${VLLM_PORT_END}`);
    }
    this.reservedPorts.add(port);
    const servedModelName = input.hfRepo;
    const backendId = ids.backendId;
    const specialistId = ids.specialistId;
    const containerName = ids.containerName;

    try {
      if (detected.launch.runtime === "docker" && detected.launch.image) {
        return await this.startDocker({
          input,
          image: detected.launch.image,
          port,
          servedModelName,
          backend,
          backendId,
          specialistId,
          containerName,
          deviceCount,
        });
      }

      const { command, args } = buildVllmArgv({
        launch: detected.launch,
        modelPath: input.modelPath,
        port,
        servedModelName,
        quantization: input.quantization,
        backend,
        tensorParallel: input.tensorParallel,
      });
      if (args.includes("0.0.0.0") || command.includes("0.0.0.0")) {
        throw new Error("Refusing to start vLLM with a public bind address");
      }

      const fd = openSync(logPath(backendId), "a");
      let pid: number;
      try {
        const child = spawn(command, args, {
          cwd: dirname(input.modelPath),
          env: vllmLaunchEnv(backend, deviceCount),
          detached: true,
          stdio: ["ignore", fd, fd],
        });
        if (child.pid == null) {
          throw new Error("Failed to spawn vLLM (no pid)");
        }
        pid = child.pid;
        child.unref();
      } finally {
        closeSync(fd);
      }

      const stored: VllmStateFile = {
        pid,
        port,
        host: VLLM_LOOPBACK_HOST,
        modelId: input.modelId,
        hfRepo: input.hfRepo,
        modelPath: input.modelPath,
        servedModelName,
        quantization: input.quantization,
        backendId,
        specialistId,
        startedAt: Date.now(),
        runtime: "host",
      };
      upsertInstance(stored);
      this.events.emit("vllm", this.status());
      return await this.waitUntilReady(stored, backendId, specialistId, pid, undefined, input.timeoutMs);
    } finally {
      this.reservedPorts.delete(port);
    }
  }

  private async startDocker(params: {
    input: {
      modelId: string;
      hfRepo: string;
      modelPath: string;
      timeoutMs?: number;
      tensorParallel?: number;
    };
    image: string;
    port: number;
    servedModelName: string;
    backend: LaunchBackend;
    backendId: string;
    specialistId: string;
    containerName: string;
    deviceCount: number;
  }): Promise<VllmRuntimeState> {
    const { input, image, port, servedModelName, backendId, specialistId, containerName, deviceCount } = params;
    const groups = resolveDockerGroups();
    const plan = buildIntelDockerRunArgs({
      image,
      hostPort: port,
      modelsDir: modelsDirFromPath(input.modelPath),
      modelPath: input.modelPath,
      servedModelName,
      tensorParallel: input.tensorParallel,
      deviceCount,
      renderGid: groups.renderGid,
      videoGid: groups.videoGid,
      containerName,
    });
    const publishIdx = plan.args.indexOf("--publish");
    const published = publishIdx >= 0 ? plan.args[publishIdx + 1] : "";
    if (!published?.startsWith(`${VLLM_LOOPBACK_HOST}:`) || published.startsWith("0.0.0.0:")) {
      throw new Error("Refusing to start vLLM Docker with a public bind address");
    }

    dockerRmName(plan.containerName);
    const commandLine = formatDockerCommand(plan.args);
    let containerId: string;
    try {
      containerId = dockerRunDetached(plan.args);
    } catch (error) {
      throw new Error(
        `Failed to start Intel vLLM container (${image}): ${error instanceof Error ? error.message : String(error)}\nCommand: ${commandLine}`,
      );
    }
    const pid = dockerContainerPid(containerId) ?? 0;
    const stored: VllmStateFile = {
      pid,
      port,
      host: VLLM_LOOPBACK_HOST,
      modelId: input.modelId,
      hfRepo: input.hfRepo,
      modelPath: input.modelPath,
      servedModelName,
      backendId,
      specialistId,
      startedAt: Date.now(),
      runtime: "docker",
      image,
      containerId,
      containerName: plan.containerName,
    };
    upsertInstance(stored);
    this.events.emit("vllm", this.status());
    return this.waitUntilReady(stored, backendId, specialistId, pid, containerId, input.timeoutMs, commandLine);
  }

  private async waitUntilReady(
    stored: VllmStateFile,
    backendId: string,
    specialistId: string,
    pid: number,
    containerId?: string,
    timeoutMs = VLLM_HEALTH_TIMEOUT_MS,
    commandLine?: string,
  ): Promise<VllmRuntimeState> {
    const baseUrl = `http://${VLLM_LOOPBACK_HOST}:${stored.port}/v1`;
    const deadline = Date.now() + timeoutMs;
    let lastProbe = "";
    let lastEmit = 0;
    const withCommand = (message: string) => (commandLine ? `${message}\nCommand: ${commandLine}` : message);
    const patchJob = (patch: Partial<VllmStartJob>) => {
      const job = this.jobFor(stored.modelId);
      if (job) this.setJob(stored.modelId, { ...job, ...patch, updatedAt: Date.now() });
    };
    if (commandLine) patchJob({ command: commandLine });
    while (Date.now() < deadline) {
      if (this.abortStarts.has(stored.modelId)) {
        this.stopStored(stored);
        dropInstances((row) => row.backendId === stored.backendId && row.port === stored.port);
        throw new Error(withCommand("Stopped while starting."));
      }
      const alive = containerId ? dockerContainerRunning(containerId) : pidAlive(pid);
      if (!alive) {
        const log = containerId ? dockerLogs(containerId) : lastLogLines(backendId);
        const classified = classifyVllmLog(log);
        if (containerId) dockerStopContainer(containerId);
        dropInstances((row) => row.backendId === stored.backendId && row.port === stored.port);
        throw new Error(
          withCommand(
            classified ??
              `vLLM ${containerId ? "container" : "process"} exited before becoming ready. Last log:\n${log.slice(-1500) || lastProbe || "(empty)"}`,
          ),
        );
      }
      const probe = await probeVllmModels(baseUrl, 1000);
      if (probe.ok) {
        const served = probe.modelIds[0] ?? stored.servedModelName;
        stored.servedModelName = served;
        upsertInstance(stored);
        this.upsertConfig(backendId, specialistId, baseUrl, served);
        const ready = this.status();
        this.events.emit("vllm", ready);
        return ready;
      }
      lastProbe = probe.error ?? "";
      if (Date.now() - lastEmit > 4000) {
        const log = containerId ? dockerLogs(containerId) : lastLogLines(backendId);
        patchJob({ lastLog: log.slice(-2000) || lastProbe, command: commandLine });
        this.events.emit("vllm", this.status());
        lastEmit = Date.now();
      }
      await delay(1000);
    }

    const log = containerId ? dockerLogs(containerId) : lastLogLines(backendId);
    const classified = classifyVllmLog(log);
    this.stopStored(stored);
    dropInstances((row) => row.backendId === stored.backendId && row.port === stored.port);
    throw new Error(
      withCommand(
        classified ??
          `Timed out waiting for vLLM at ${baseUrl}/models. Last log:\n${log.slice(-1500) || lastProbe || "(empty)"}`,
      ),
    );
  }

  findInstance(sel: VllmInstanceSelector): VllmInstanceState | undefined {
    return this.status().instances.find((row) => instanceMatches(row, sel));
  }

  stop(selector: VllmInstanceSelector = {}): VllmRuntimeState {
    const current = this.status();
    const instances = readInstances();
    let matched: VllmStateFile[];
    if (selector.all) {
      matched = instances;
      for (const job of this.startJobs.values()) {
        if (job.status === "starting") this.abortStarts.add(job.modelId);
      }
    } else if (selector.modelId || selector.backendId) {
      matched = instances.filter((row) => instanceMatches(row, selector));
      for (const [modelId, job] of this.startJobs) {
        if (job.status !== "starting") continue;
        const ids = this.idsFor(modelId);
        if (instanceMatches({ modelId, backendId: ids.backendId, hfRepo: job.modelId }, selector)) {
          this.abortStarts.add(modelId);
        }
      }
    } else if (instances.length > 1 || [...this.startJobs.values()].filter((j) => j.status === "starting").length > 1) {
      const ids = current.instances.map((row) => row.backendId).join(", ");
      throw new Error(
        `Multiple vLLM instances are running (${ids || "starting"}). Pass model_id or backend_id, or all=true.`,
      );
    } else {
      matched = instances;
      for (const job of this.startJobs.values()) {
        if (job.status === "starting") this.abortStarts.add(job.modelId);
      }
    }

    for (const stored of matched) {
      this.stopStored(stored);
      const job = this.jobFor(stored.modelId);
      if (job?.status === "starting") {
        this.setJob(stored.modelId, {
          ...job,
          status: "error",
          error: "Stopped while starting.",
          updatedAt: Date.now(),
        });
      }
    }
    for (const modelId of this.abortStarts) {
      const job = this.jobFor(modelId);
      if (job?.status === "starting") {
        this.setJob(modelId, {
          ...job,
          status: "error",
          error: "Stopped while starting.",
          updatedAt: Date.now(),
        });
      }
    }
    if (matched.length > 0) {
      writeInstances(instances.filter((row) => !matched.includes(row)));
    }
    const status = this.status();
    this.events.emit("vllm", status);
    return status;
  }

  private stopStoredAndDrop(inst: { backendId: string; modelId?: string; port?: number; containerId?: string; containerName?: string; pid?: number; runtime?: VllmProcessRuntime }): void {
    const stored = readInstances().find(
      (row) =>
        row.backendId === inst.backendId ||
        (inst.modelId && row.modelId === inst.modelId) ||
        (inst.port != null && row.port === inst.port),
    );
    if (stored) this.stopStored(stored);
    dropInstances(
      (row) =>
        row.backendId === inst.backendId ||
        (Boolean(inst.modelId) && row.modelId === inst.modelId),
    );
  }

  private stopStored(stored: VllmStateFile): void {
    if (stored.runtime === "docker") {
      const target = stored.containerId || stored.containerName;
      if (target) dockerStopContainer(target);
    } else if (stored.pid) {
      const cmdline = readCmdline(stored.pid);
      if (pidAlive(stored.pid) && isVllmCmdline(cmdline)) {
        killProcessGroup(stored.pid, "SIGTERM");
        const until = Date.now() + 8000;
        while (Date.now() < until && pidAlive(stored.pid)) {
          Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 200);
        }
        if (pidAlive(stored.pid)) killProcessGroup(stored.pid, "SIGKILL");
      }
    }
  }

  private upsertConfig(backendId: string, specialistId: string, baseUrl: string, model: string): void {
    const yaml = readConfigYaml(this.configPath);
    const next = patchVllmOrchestratorYaml(yaml, { baseUrl, model, backendId, specialistId });
    writeConfigYaml(next, this.configPath);
    ensureLocalVllmDummyKey();
    this.reloadConfig();
  }
}
