import { spawn, type ChildProcess } from "node:child_process";
import { closeSync, existsSync, openSync, readFileSync, unlinkSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { engineLibEnv, findEngineBin, findLateInferBin, lateInferMissingMessage } from "./bins.js";
import { DEFAULT_LATE_INFER_MODEL, LATE_INFER_PORT, LOOPBACK_HOST } from "./loopback.js";
import { probeLateInfer, probeLlamaCpp, probeOllama } from "./status.js";
import {
  isWindows,
  listenerPids,
  normalizeCmdline,
  pidAlive,
  readProcessCmdline,
  stopProcessTree,
  writeSecureFile,
} from "../platform.js";
import { stateDir } from "../state.js";
import {
  pullLateInfer as runLateInferCompile,
  lateInferCompileSpec,
  lateInferCompileJob,
  lateInferCompileView,
  resetLateInferCompileForTests,
  parseCompilePhase,
  compileJobHasFsFields,
  lateHfHomeDir,
  deleteCompiledLateInfer,
  parseCompiledModelRef,
  servingThisCompiledModel,
  type LateInferCompileJob,
  type LateInferServingState,
  type DeleteCompiledLateInferResult,
} from "./compile.js";
import {
  convertHostRamBlockedReason,
  engineGpuSpawnPlan,
  lateInferStartBlockedReason,
  mergeGpuPlanEnv,
  ollamaGpuBlockedReason,
  openvinoIrPresent,
  readHostRamProbe,
  resolveGpuServePhase,
  resolveLateInferGpuPlan,
  vramMaxMiBForHubId,
  weightsMiBForStart,
  INTEL_IR_MISSING_START,
  type GpuServePhase,
  type LateInferGpuOptions,
} from "./gpu-pick.js";

export {
  lateInferCompileSpec,
  lateInferCompileJob,
  lateInferCompileView,
  resetLateInferCompileForTests,
  parseCompilePhase,
  compileJobHasFsFields,
  lateHfHomeDir,
  type LateInferCompileJob,
};

/** MCP `pull_late_infer` and GUI download call this. Compile-only; does not bind :8010. */
export async function pullLateInfer(
  options: { model: string } & LateInferGpuOptions,
): Promise<LateInferCompileJob> {
  return runLateInferCompile(options);
}

/**
 * Optional Convert for a downloaded Intel row missing OpenVINO IR.
 * Same RAM cap as Download. Does not bind :8010. Refuses Gemma/huge convert that would OOM.
 */
export async function convertLateInfer(
  options: { model: string } & LateInferGpuOptions,
): Promise<LateInferCompileJob> {
  const model = (options.model ?? "").trim();
  const plan = resolveLateInferGpuPlan({
    ...options,
    model,
    vramMaxMiB: options.vramMaxMiB ?? vramMaxMiBForHubId(model),
  });
  const vendor = plan.visible[0]?.vendor ?? plan.primary?.vendor;
  if (vendor === "intel") {
    const ir = openvinoIrPresent(model);
    const blocked = convertHostRamBlockedReason({
      vendor,
      weightsMiB: weightsMiBForStart(model, options.vramMaxMiB ?? vramMaxMiBForHubId(model)),
      ovIrPresent: ir,
      mem: options.probes && "memAvailableMiB" in options.probes
        ? {
            memAvailableMiB: options.probes.memAvailableMiB ?? 0,
            swapTotalMiB: options.probes.swapTotalMiB ?? 0,
            swapFreeMiB: options.probes.swapFreeMiB ?? 0,
          }
        : readHostRamProbe(),
    });
    if (blocked) {
      throw new Error(`${INTEL_IR_MISSING_START} ${blocked}`);
    }
  }
  return runLateInferCompile(options);
}

let servingProbeForTests: (() => Promise<LateInferServingState>) | undefined;

export function resetLateInferDeleteForTests(probe?: () => Promise<LateInferServingState>): void {
  servingProbeForTests = probe;
}

/**
 * GUI POST /api/local-servers/delete and MCP `delete_late_infer`.
 * Compiled slug dir only. If that snapshot is serving on 127.0.0.1:8010, Stop then delete
 * (Remove on a running row must work). Injected test probes still refuse so 409 tests stay honest.
 */
export async function deleteLateInferCompiled(options: {
  model: string;
  confirm?: boolean;
}): Promise<DeleteCompiledLateInferResult> {
  const model = parseCompiledModelRef(options.model);
  let serving: LateInferServingState;
  if (servingProbeForTests) {
    serving = await servingProbeForTests();
  } else {
    const probe = await probeLateInfer();
    serving = {
      running: probe.running || probe.ready || ownedLateInferLive(),
      models: probe.models,
    };
    if (servingThisCompiledModel(model, serving)) {
      stopLocalServer("lateinfer");
      serving = { running: false, models: [] };
    }
  }
  return deleteCompiledLateInfer({
    model,
    confirm: options.confirm,
    serving,
  });
}

const OLLAMA_PORT = 11434;
const LLAMA_PORT = 8080;

export type LocalSpawnKind = "lateinfer" | "ollama" | "llamacpp";

export interface LocalSpawnResult {
  kind: LocalSpawnKind;
  pid?: number;
  running: boolean;
  ready: boolean;
  host: string;
  reason: string;
}

const children = new Map<LocalSpawnKind, ChildProcess>();

/** Same record Late writes: pid + /proc comm + starttime so Stop cannot kill a reused PID. */
export interface OwnedPid {
  pid: number;
  starttime: number;
  comm: string;
}

function pidFileName(kind: LocalSpawnKind): string {
  if (kind === "lateinfer") return "late-infer.pid";
  if (kind === "ollama") return "ollama-serve.pid";
  return "llama-server.pid";
}

function pidPath(kind: LocalSpawnKind): string {
  return join(stateDir(), pidFileName(kind));
}

function kindPort(kind: LocalSpawnKind): number {
  if (kind === "lateinfer") return LATE_INFER_PORT;
  if (kind === "ollama") return OLLAMA_PORT;
  return LLAMA_PORT;
}

function fallbackComm(kind: LocalSpawnKind): string {
  if (kind === "lateinfer") return "late-infer";
  if (kind === "ollama") return "ollama";
  return "llama-server";
}

export function commOk(comm: string, kind: LocalSpawnKind): boolean {
  if (kind === "lateinfer") return comm === "late-infer" || comm === "late-infer.exe";
  if (kind === "ollama") return comm === "ollama" || comm === "ollama.exe";
  return comm === "llama-server" || comm === "llama-cpp-server" || comm === "llama-cpp-serve";
}

/** Windows: same idea as vLLM `isVllmCmdline` — only kill if the live command line is still this engine. */
export function isEngineCmdline(cmdline: string, kind: LocalSpawnKind): boolean {
  const text = normalizeCmdline(cmdline).toLowerCase();
  if (kind === "lateinfer") return text.includes("late-infer");
  if (kind === "ollama") return text.includes("ollama");
  return text.includes("llama-server") || text.includes("llama-cpp-server") || text.includes("llama-cpp-serve");
}

export function procComm(pid: number): string | undefined {
  if (!Number.isInteger(pid) || pid <= 0) return undefined;
  try {
    const c = readFileSync(`/proc/${pid}/comm`, "utf8").trim();
    return c || undefined;
  } catch {
    return undefined;
  }
}

/** Field 22 of /proc/pid/stat (clock ticks after boot). Same parse as Late `proc_starttime`. */
export function procStarttime(pid: number): number | undefined {
  if (!Number.isInteger(pid) || pid <= 0) return undefined;
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
    const rest = stat.slice(stat.lastIndexOf(")") + 1).trim();
    const field = rest.split(/\s+/)[19];
    const n = Number.parseInt(field ?? "", 10);
    return Number.isInteger(n) && n > 0 ? n : undefined;
  } catch {
    return undefined;
  }
}

export function formatOwnedPid(rec: OwnedPid): string {
  return `pid=${rec.pid}\nstarttime=${rec.starttime}\ncomm=${rec.comm}\n`;
}

export function parseOwnedPid(raw: string): OwnedPid | undefined {
  let pid: number | undefined;
  let starttime: number | undefined;
  let comm: string | undefined;
  for (const line of raw.split(/\r?\n/)) {
    if (line.startsWith("pid=")) {
      const n = Number.parseInt(line.slice(4).trim(), 10);
      if (Number.isInteger(n) && n > 0) pid = n;
    } else if (line.startsWith("starttime=")) {
      const n = Number.parseInt(line.slice(10).trim(), 10);
      if (Number.isInteger(n) && n >= 0) starttime = n;
    } else if (line.startsWith("comm=")) {
      const c = line.slice(5).trim();
      if (c) comm = c;
    }
  }
  if (pid === undefined || comm === undefined) return undefined;
  if (starttime === undefined) {
    if (isWindows()) starttime = 0;
    else return undefined;
  }
  return { pid, starttime, comm };
}

/**
 * Live process still has the recorded comm + starttime (Linux /proc).
 * Windows: pid still alive (cmdline checked separately, like vLLM).
 */
export function ownedPidIdentityLive(rec: OwnedPid): boolean {
  if (rec.pid === 0) return false;
  if (isWindows()) return pidAlive(rec.pid);
  if (rec.starttime === 0) return false;
  return procStarttime(rec.pid) === rec.starttime && procComm(rec.pid) === rec.comm;
}

/** Kill only when identity still matches this engine. If we cannot match, do not kill. */
export function ownedPidMatchesLive(rec: OwnedPid, kind: LocalSpawnKind): boolean {
  if (!commOk(rec.comm, kind)) return false;
  if (!ownedPidIdentityLive(rec)) return false;
  if (isWindows()) {
    const cmdline = readProcessCmdline(rec.pid);
    if (!cmdline) return false;
    return isEngineCmdline(cmdline, kind);
  }
  return true;
}

function writeOwnedPidFile(kind: LocalSpawnKind, pid: number): void {
  const comm = procComm(pid) ?? fallbackComm(kind);
  if (!commOk(comm, kind)) return;
  if (isWindows()) {
    writeSecureFile(pidPath(kind), formatOwnedPid({ pid, starttime: 0, comm }));
    return;
  }
  const starttime = procStarttime(pid) ?? 0;
  if (starttime === 0) return;
  writeSecureFile(pidPath(kind), formatOwnedPid({ pid, starttime, comm }));
}

function unlinkPidFile(kind: LocalSpawnKind): void {
  try {
    unlinkSync(pidPath(kind));
  } catch {
    /* already gone */
  }
}

function logPath(kind: LocalSpawnKind): string {
  if (kind === "lateinfer") return join(stateDir(), "late-infer.log");
  if (kind === "ollama") return join(stateDir(), "ollama-serve.log");
  return join(stateDir(), "llama-server.log");
}

export function ownedLateInferLive(): boolean {
  try {
    const path = pidPath("lateinfer");
    if (!existsSync(path)) return false;
    const rec = parseOwnedPid(readFileSync(path, "utf8"));
    return Boolean(rec && ownedPidMatchesLive(rec, "lateinfer"));
  } catch {
    return false;
  }
}

export function lateInferPidFilePresent(): boolean {
  try {
    return existsSync(pidPath("lateinfer"));
  } catch {
    return false;
  }
}

export function lateInferServeSnapshot(probe: {
  ready?: boolean;
  running?: boolean;
  device?: string;
  deviceKind?: string;
  weightsInHostRam?: boolean;
  gpuRunning?: boolean;
}): {
  processAlive: boolean;
  pidFilePresent: boolean;
  starting: boolean;
  servePhase: GpuServePhase;
} {
  const processAlive = ownedLateInferLive();
  const pidFilePresent = lateInferPidFilePresent();
  const servePhase = resolveGpuServePhase({
    processAlive,
    pidFilePresent,
    ready: probe.ready === true,
    running: probe.running === true,
    device: probe.device,
    deviceKind: probe.deviceKind,
    weightsInHostRam: probe.weightsInHostRam === true,
  });
  return {
    processAlive,
    pidFilePresent,
    starting: servePhase === "starting",
    servePhase,
  };
}

function libEnv(bin: string): NodeJS.ProcessEnv {
  return engineLibEnv(bin);
}

function spawnLogged(bin: string, args: string[], env: NodeJS.ProcessEnv, kind: LocalSpawnKind): ChildProcess {
  const log = logPath(kind);
  let fd: number | null = null;
  try {
    fd = openSync(log, "a");
  } catch {
    fd = null;
  }
  const child = spawn(bin, args, {
    env,
    stdio: fd == null ? "ignore" : ["ignore", fd, fd],
    windowsHide: true,
  });
  if (fd != null) {
    child.on("exit", () => {
      try {
        closeSync(fd as number);
      } catch {
        /* ignore */
      }
    });
  }
  children.set(kind, child);
  if (typeof child.pid === "number" && child.pid > 0) {
    writeOwnedPidFile(kind, child.pid);
  }
  return child;
}

export function lateInferSpec(
  model?: string,
  gpu: LateInferGpuOptions = {},
): { args: string[]; env: NodeJS.ProcessEnv; host: string; gpu: ReturnType<typeof resolveLateInferGpuPlan> } {
  const id = (model?.trim() || DEFAULT_LATE_INFER_MODEL).trim();
  const plan = resolveLateInferGpuPlan({
    ...gpu,
    model: id,
    vramMaxMiB: gpu.vramMaxMiB ?? vramMaxMiBForHubId(id),
  });
  const env = mergeGpuPlanEnv({ ...process.env }, plan);
  const accel = String(env.LATE_INFER_ACCEL ?? plan.env.LATE_INFER_ACCEL ?? "").toLowerCase();
  if (accel === "intel") {
    env.LATE_INFER_SKIP_MLC_PREFLIGHT = "1";
    env.LATE_INFER_SKIP_MLC = "1";
  }
  return {
    args: ["--bind", `${LOOPBACK_HOST}:${LATE_INFER_PORT}`, "--model", id],
    env,
    host: `${LOOPBACK_HOST}:${LATE_INFER_PORT}`,
    gpu: plan,
  };
}

export async function startLateInfer(
  options: { model?: string } & LateInferGpuOptions = {},
): Promise<LocalSpawnResult> {
  const spec = lateInferSpec(options.model, options);
  const blocked = lateInferStartBlockedReason(spec.gpu);
  if (blocked) {
    throw new Error(blocked);
  }
  const probe = await probeLateInfer();
  if (probe.running) {
    return {
      kind: "lateinfer",
      running: true,
      ready: probe.ready,
      host: spec.host,
      reason: probe.reason,
    };
  }
  if (ownedLateInferLive()) {
    return {
      kind: "lateinfer",
      running: true,
      ready: false,
      host: spec.host,
      reason: `late-infer is up on ${spec.host}; weights still loading`,
    };
  }
  const bin = findLateInferBin();
  if (!bin) {
    throw new Error(lateInferMissingMessage());
  }
  spawnLogged(bin, spec.args, { ...libEnv(bin), ...spec.env }, "lateinfer");
  return {
    kind: "lateinfer",
    pid: children.get("lateinfer")?.pid,
    running: true,
    ready: false,
    host: spec.host,
    reason: `starting late-infer on ${spec.host} — first Hugging Face load can take a few minutes`,
  };
}

export function ollamaServeSpec(
  bin: string,
  options: { useAllGpus?: boolean } & LateInferGpuOptions = {},
): { args: string[]; env: NodeJS.ProcessEnv; host: string; gpuReason: string } {
  const host = `${LOOPBACK_HOST}:${OLLAMA_PORT}`;
  const gpu = engineGpuSpawnPlan({ useAllGpus: options.useAllGpus, ...options });
  if (gpu.blockedReason) {
    throw new Error(gpu.blockedReason);
  }
  // Intel Arc: pin idle card via GGML_VK + ZE (same as llama.cpp). Fail closed only when
  // Vulkan ICD is missing (true CPU/RAM path) — not merely because CUDA/ROCm are absent.
  const ollamaBlocked = ollamaGpuBlockedReason(gpu.plan, options.probes);
  if (ollamaBlocked) {
    throw new Error(ollamaBlocked);
  }
  const plan = gpu.plan;
  const env = mergeGpuPlanEnv({ ...libEnv(bin), OLLAMA_HOST: host }, plan);
  Object.assign(env, gpu.env);
  return {
    args: ["serve"],
    env,
    host,
    gpuReason: plan.reason,
  };
}

export function llamaServerSpec(
  bin: string,
  modelPath: string,
  port = LLAMA_PORT,
  options: { useAllGpus?: boolean } & LateInferGpuOptions = {},
): { args: string[]; env: NodeJS.ProcessEnv; host: string; gpuReason: string } {
  if (!isAbsolute(modelPath) || !modelPath.toLowerCase().endsWith(".gguf")) {
    throw new Error("llama-server needs an absolute path to a .gguf file");
  }
  if (modelPath.includes("..")) {
    throw new Error("llama-server model path must not contain ..");
  }
  const gpu = engineGpuSpawnPlan({ useAllGpus: options.useAllGpus, ...options });
  if (gpu.blockedReason) {
    throw new Error(gpu.blockedReason);
  }
  if (gpu.llamaArgs.length === 0) {
    throw new Error(
      "No discrete GPU on your computer. Refusing llama-server without -ngl (would use system RAM).",
    );
  }
  const plan = gpu.plan;
  const env = mergeGpuPlanEnv(libEnv(bin), plan);
  Object.assign(env, gpu.env);
  return {
    args: [
      "-m",
      modelPath,
      "--host",
      LOOPBACK_HOST,
      "--port",
      String(port),
      ...gpu.llamaArgs,
    ],
    env,
    host: `${LOOPBACK_HOST}:${port}`,
    gpuReason: plan.reason,
  };
}

export async function startOllama(options: { useAllGpus?: boolean } = {}): Promise<LocalSpawnResult> {
  const probe = await probeOllama();
  if (probe.running) {
    return {
      kind: "ollama",
      running: true,
      ready: probe.ready,
      host: `${LOOPBACK_HOST}:${OLLAMA_PORT}`,
      reason: probe.reason,
    };
  }
  const bin = findEngineBin("ollama");
  if (!bin) {
    throw new Error(
      "ollama is missing. Packed installs include it in runtime/bin. Bind stays 127.0.0.1.",
    );
  }
  const spec = ollamaServeSpec(bin, options);
  spawnLogged(bin, spec.args, spec.env, "ollama");
  return {
    kind: "ollama",
    pid: children.get("ollama")?.pid,
    running: true,
    ready: false,
    host: spec.host,
    reason: `starting ollama serve on ${spec.host} — ${spec.gpuReason}`,
  };
}

export async function startLlamaServer(
  modelPath: string,
  port = LLAMA_PORT,
  options: { useAllGpus?: boolean } = {},
): Promise<LocalSpawnResult> {
  const probe = await probeLlamaCpp();
  if (probe.running) {
    return {
      kind: "llamacpp",
      running: true,
      ready: probe.ready,
      host: `${LOOPBACK_HOST}:${port}`,
      reason: probe.reason,
    };
  }
  const bin = findEngineBin("llama-server");
  if (!bin) {
    throw new Error(
      "llama-server is missing. Packed installs include it in runtime/bin (Vulkan/Metal). Bind stays 127.0.0.1.",
    );
  }
  const spec = llamaServerSpec(bin, modelPath, port, options);
  spawnLogged(bin, spec.args, spec.env, "llamacpp");
  return {
    kind: "llamacpp",
    pid: children.get("llamacpp")?.pid,
    running: true,
    ready: false,
    host: spec.host,
    reason: `starting llama-server on ${spec.host} — ${spec.gpuReason}`,
  };
}

/** Brief sync wait (SIGTERM grace) — same pattern as vLLM manager. */
function sleepSyncMs(ms: number): void {
  const n = Math.max(0, Math.floor(ms));
  if (n <= 0) return;
  try {
    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, n);
  } catch {
    /* SharedArrayBuffer may be unavailable; skip grace */
  }
}

/**
 * Kill a PID on our loopback engine port only when the live process still looks like this engine.
 * Orphan late-infer (no pidfile / GUI restarted) must still die on Stop.
 */
export function stopListenerIfEngine(pid: number, kind: LocalSpawnKind): boolean {
  if (!Number.isInteger(pid) || pid <= 0 || pid === process.pid) return false;
  if (!pidAlive(pid)) return false;
  const comm = procComm(pid);
  if (comm && commOk(comm, kind)) {
    stopProcessTree(pid);
    return true;
  }
  const cmdline = readProcessCmdline(pid);
  if (cmdline && isEngineCmdline(cmdline, kind)) {
    stopProcessTree(pid);
    return true;
  }
  return false;
}

export function stopLocalServer(kind: LocalSpawnKind): LocalSpawnResult {
  const port = kindPort(kind);
  const killed = new Set<number>();
  const child = children.get(kind);
  // Only the ChildProcess we still own. After exit, child.pid can be reused by a stranger.
  if (child && typeof child.pid === "number" && child.exitCode == null) {
    stopProcessTree(child.pid);
    killed.add(child.pid);
  }
  children.delete(kind);
  try {
    const path = pidPath(kind);
    if (existsSync(path)) {
      const rec = parseOwnedPid(readFileSync(path, "utf8"));
      if (rec && ownedPidMatchesLive(rec, kind)) {
        stopProcessTree(rec.pid);
        killed.add(rec.pid);
      }
    }
  } catch {
    /* ignore */
  }
  // Orphans: previous GUI exit cleared the ChildProcess map / pidfile but left :port bound.
  for (const pid of listenerPids(port)) {
    if (stopListenerIfEngine(pid, kind)) killed.add(pid);
  }
  unlinkPidFile(kind);
  // SIGTERM grace, then SIGKILL any engine still listening on our port.
  sleepSyncMs(400);
  for (const pid of listenerPids(port)) {
    if (!pidAlive(pid) || pid === process.pid) continue;
    const comm = procComm(pid);
    const cmdline = readProcessCmdline(pid);
    const match =
      (comm && commOk(comm, kind)) || (cmdline !== undefined && cmdline !== "" && isEngineCmdline(cmdline, kind));
    if (!match) continue;
    stopProcessTree(pid, true);
    killed.add(pid);
  }
  unlinkPidFile(kind);
  void killed;
  return {
    kind,
    running: false,
    ready: false,
    host: `${LOOPBACK_HOST}:${port}`,
    reason: `stopped ${kind}`,
  };
}
