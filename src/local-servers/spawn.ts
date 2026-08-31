import { spawn, type ChildProcess } from "node:child_process";
import { closeSync, existsSync, openSync, readFileSync, unlinkSync } from "node:fs";
import { dirname, isAbsolute, join } from "node:path";
import { findEngineBin } from "./bins.js";
import { LOOPBACK_HOST } from "./loopback.js";
import { probeLlamaCpp, probeOllama } from "./status.js";
import {
  isWindows,
  normalizeCmdline,
  pidAlive,
  readProcessCmdline,
  stopProcessTree,
  writeSecureFile,
} from "../platform.js";
import { stateDir } from "../state.js";

const OLLAMA_PORT = 11434;
const LLAMA_PORT = 8080;

export type LocalSpawnKind = "ollama" | "llamacpp";

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

function pidPath(kind: LocalSpawnKind): string {
  return join(stateDir(), kind === "ollama" ? "ollama-serve.pid" : "llama-server.pid");
}

function fallbackComm(kind: LocalSpawnKind): string {
  return kind === "ollama" ? "ollama" : "llama-server";
}

export function commOk(comm: string, kind: LocalSpawnKind): boolean {
  if (kind === "ollama") return comm === "ollama" || comm === "ollama.exe";
  return comm === "llama-server" || comm === "llama-cpp-server" || comm === "llama-cpp-serve";
}

/** Windows: same idea as vLLM `isVllmCmdline` — only kill if the live command line is still this engine. */
export function isEngineCmdline(cmdline: string, kind: LocalSpawnKind): boolean {
  const text = normalizeCmdline(cmdline).toLowerCase();
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
  return join(stateDir(), kind === "ollama" ? "ollama-serve.log" : "llama-server.log");
}

function libEnv(bin: string): NodeJS.ProcessEnv {
  const dir = dirname(bin);
  const lib = join(dirname(dir), "lib");
  const ollamaLib = join(lib, "ollama");
  const extra = [dir, lib, ollamaLib].filter((p) => existsSync(p));
  const env = { ...process.env };
  if (process.platform === "win32") {
    env.PATH = [...extra, process.env.PATH ?? ""].join(";");
  } else if (process.platform === "darwin") {
    env.DYLD_LIBRARY_PATH = [...extra, process.env.DYLD_LIBRARY_PATH ?? ""].join(":");
  } else {
    env.LD_LIBRARY_PATH = [...extra, process.env.LD_LIBRARY_PATH ?? ""].join(":");
  }
  return env;
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

export function ollamaServeSpec(bin: string): { args: string[]; env: NodeJS.ProcessEnv; host: string } {
  const host = `${LOOPBACK_HOST}:${OLLAMA_PORT}`;
  return {
    args: ["serve"],
    env: { ...libEnv(bin), OLLAMA_HOST: host },
    host,
  };
}

export function llamaServerSpec(
  bin: string,
  modelPath: string,
  port = LLAMA_PORT,
): { args: string[]; env: NodeJS.ProcessEnv; host: string } {
  if (!isAbsolute(modelPath) || !modelPath.toLowerCase().endsWith(".gguf")) {
    throw new Error("llama-server needs an absolute path to a .gguf file");
  }
  if (modelPath.includes("..")) {
    throw new Error("llama-server model path must not contain ..");
  }
  return {
    args: ["-m", modelPath, "--host", LOOPBACK_HOST, "--port", String(port)],
    env: libEnv(bin),
    host: `${LOOPBACK_HOST}:${port}`,
  };
}

export async function startOllama(): Promise<LocalSpawnResult> {
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
  const spec = ollamaServeSpec(bin);
  spawnLogged(bin, spec.args, spec.env, "ollama");
  return {
    kind: "ollama",
    pid: children.get("ollama")?.pid,
    running: true,
    ready: false,
    host: spec.host,
    reason: `starting ollama serve on ${spec.host}`,
  };
}

export async function startLlamaServer(modelPath: string, port = LLAMA_PORT): Promise<LocalSpawnResult> {
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
  const spec = llamaServerSpec(bin, modelPath, port);
  spawnLogged(bin, spec.args, spec.env, "llamacpp");
  return {
    kind: "llamacpp",
    pid: children.get("llamacpp")?.pid,
    running: true,
    ready: false,
    host: spec.host,
    reason: `starting llama-server on ${spec.host}`,
  };
}

export function stopLocalServer(kind: LocalSpawnKind): LocalSpawnResult {
  const child = children.get(kind);
  // Only the ChildProcess we still own. After exit, child.pid can be reused by a stranger.
  if (child && typeof child.pid === "number" && child.exitCode == null) {
    stopProcessTree(child.pid);
  }
  children.delete(kind);
  try {
    const path = pidPath(kind);
    if (existsSync(path)) {
      const rec = parseOwnedPid(readFileSync(path, "utf8"));
      if (rec && ownedPidMatchesLive(rec, kind)) {
        stopProcessTree(rec.pid);
      }
    }
  } catch {
    /* ignore */
  }
  unlinkPidFile(kind);
  return {
    kind,
    running: false,
    ready: false,
    host: `${LOOPBACK_HOST}:${kind === "ollama" ? OLLAMA_PORT : LLAMA_PORT}`,
    reason: `stopped ${kind}`,
  };
}
