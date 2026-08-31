import { spawn, spawnSync } from "node:child_process";
import {
  accessSync,
  chmodSync,
  constants,
  existsSync,
  mkdirSync,
  readFileSync,
  readlinkSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

export type PathApi = {
  sep: string;
  join: (...paths: string[]) => string;
  relative: (from: string, to: string) => string;
  isAbsolute: (p: string) => boolean;
};

/** POSIX permission bits. Node ignores most of these on Windows (see README). */
export const SECURE_DIR_MODE = 0o700;
export const SECURE_FILE_MODE = 0o600;

function tryChmod(target: string, mode: number): void {
  try {
    chmodSync(target, mode);
  } catch {
    // Windows and some filesystems ignore or reject POSIX modes.
  }
}

/** mkdir -p, then chmod 0700 when the filesystem honors POSIX bits. */
export function ensureSecureDir(dir: string): void {
  mkdirSync(dir, { recursive: true, mode: SECURE_DIR_MODE });
  tryChmod(dir, SECURE_DIR_MODE);
}

/**
 * Write then chmod 0600. `writeFileSync({ mode })` only applies when creating a
 * new file; chmod closes that hole on POSIX. Windows still uses NTFS ACLs.
 */
export function writeSecureFile(
  filePath: string,
  data: string | NodeJS.ArrayBufferView,
  options?: { encoding?: BufferEncoding },
): void {
  if (typeof data === "string") {
    writeFileSync(filePath, data, {
      encoding: options?.encoding ?? "utf8",
      mode: SECURE_FILE_MODE,
    });
  } else {
    writeFileSync(filePath, data, { mode: SECURE_FILE_MODE });
  }
  tryChmod(filePath, SECURE_FILE_MODE);
}

export function isWindows(platform: NodeJS.Platform = process.platform): boolean {
  return platform === "win32";
}

export function isDarwin(platform: NodeJS.Platform = process.platform): boolean {
  return platform === "darwin";
}

export function pathFor(platform: NodeJS.Platform = process.platform): PathApi {
  if (platform === "win32") return path.win32;
  return path.posix;
}

export function homeDir(): string {
  return homedir();
}

export interface WhichOptions {
  pathEnv?: string;
  extraDirs?: string[];
  platform?: NodeJS.Platform;
  exists?: (candidate: string) => boolean;
  pathImpl?: PathApi;
}

function defaultExists(candidate: string, platform: NodeJS.Platform): boolean {
  try {
    accessSync(candidate, isWindows(platform) ? constants.F_OK : constants.X_OK);
    return true;
  } catch {
    try {
      accessSync(candidate, constants.F_OK);
      return existsSync(candidate);
    } catch {
      return false;
    }
  }
}

export function extraBinaryDirs(platform: NodeJS.Platform = process.platform, pathImpl: PathApi = pathFor(platform)): string[] {
  if (isWindows(platform)) {
    const local = process.env.LOCALAPPDATA?.trim();
    const pf = process.env.ProgramFiles?.trim() || "C:\\Program Files";
    const dirs: string[] = [];
    if (local) {
      dirs.push(pathImpl.join(local, "Programs", "Ollama"));
      dirs.push(pathImpl.join(local, "Programs", "llama.cpp"));
    }
    dirs.push(pathImpl.join(pf, "Ollama"));
    dirs.push(pathImpl.join(pf, "NVIDIA Corporation", "NVSMI"));
    dirs.push(pathImpl.join(pf, "Docker", "Docker", "resources", "bin"));
    dirs.push(pathImpl.join(pf, "llama.cpp"));
    dirs.push(pathImpl.join("C:\\Windows", "System32"));
    return dirs;
  }
  const dirs = ["/usr/local/bin", "/usr/bin"];
  if (isDarwin(platform)) dirs.unshift("/opt/homebrew/bin");
  return dirs;
}

export function candidateFileNames(cmd: string, platform: NodeJS.Platform = process.platform): string[] {
  const trimmed = cmd.trim();
  if (!trimmed) return [];
  if (!isWindows(platform)) return [trimmed];
  if (/\.(exe|cmd|bat|com)$/i.test(trimmed)) return [trimmed];
  const pathext = process.env.PATHEXT ?? ".EXE;.CMD;.BAT;.COM";
  const suffixes = pathext
    .split(";")
    .map((part) => part.trim())
    .filter((part) => part.startsWith("."));
  const names = [trimmed, ...suffixes.map((ext) => `${trimmed}${ext}`)];
  return [...new Set(names)];
}

/** Locate an executable on PATH, PATHEXT (Windows), and typical install dirs. */
export function which(cmd: string, options: WhichOptions = {}): string | undefined {
  const platform = options.platform ?? process.platform;
  const pathImpl = options.pathImpl ?? (options.platform ? pathFor(platform) : path);
  const delim = isWindows(platform) ? ";" : ":";
  const pathEnv = options.pathEnv ?? process.env.PATH ?? "";
  const extra = options.extraDirs ?? extraBinaryDirs(platform, pathImpl);
  const dirs = [...pathEnv.split(delim), ...extra];
  const names = candidateFileNames(cmd, platform);
  const exists = options.exists ?? ((candidate: string) => defaultExists(candidate, platform));
  for (const dir of dirs) {
    if (!dir.trim()) continue;
    for (const name of names) {
      const candidate = pathImpl.join(dir, name);
      if (exists(candidate)) return candidate;
    }
  }
  return undefined;
}

export function pythonInterpreterNames(platform: NodeJS.Platform = process.platform): string[] {
  return isWindows(platform) ? ["py", "python", "python3"] : ["python3", "python"];
}

/** Extra argv before `-c` / `-m` for the Windows `py` launcher. */
export function pythonDashArgs(command: string): string[] {
  const base = command.replace(/\\/g, "/").split("/").pop()?.toLowerCase() ?? "";
  if (base === "py" || base === "py.exe") return ["-3"];
  return [];
}

export function runCapture(command: string, args: string[], timeout = 4000): string | null {
  const result = spawnSync(command, args, { encoding: "utf8", timeout, windowsHide: true });
  if (result.error && (result.error as NodeJS.ErrnoException).code === "ENOENT") return null;
  if (result.stdout?.trim()) return result.stdout;
  return null;
}

export function pidAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  if (pid === process.pid) return true;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export function normalizeCmdline(cmdline: string): string {
  return cmdline.replace(/\0/g, " ").replace(/\\/g, "/");
}

export function readProcessCmdline(pid: number): string {
  if (!Number.isInteger(pid) || pid <= 0) return "";
  if (!isWindows()) {
    try {
      return readFileSync(`/proc/${pid}/cmdline`, "utf8");
    } catch {
      return "";
    }
  }
  const ps = runCapture(
    "powershell.exe",
    [
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-Command",
      `(Get-CimInstance Win32_Process -Filter "ProcessId=${pid}").CommandLine`,
    ],
    5000,
  );
  if (ps?.trim()) return ps.trim();
  const wmic = runCapture("wmic", ["process", "where", `ProcessId=${pid}`, "get", "CommandLine", "/value"], 5000);
  const match = /CommandLine=(.*)/i.exec(wmic ?? "");
  return match?.[1]?.trim() ?? "";
}

export function readProcessCwd(pid: number): string {
  if (!Number.isInteger(pid) || pid <= 0 || isWindows()) return "";
  try {
    return readlinkSync(`/proc/${pid}/cwd`);
  } catch {
    return "";
  }
}

export function stopProcess(pid: number): void {
  if (!Number.isInteger(pid) || pid <= 0) return;
  if (isWindows()) {
    try {
      process.kill(pid);
    } catch {
      spawnSync("taskkill", ["/F", "/PID", String(pid)], { encoding: "utf8", timeout: 8000, windowsHide: true });
    }
    return;
  }
  try {
    process.kill(pid, "SIGTERM");
  } catch {
    // already gone
  }
}

/** SIGTERM process group on POSIX; `taskkill /T` on Windows. */
export function stopProcessTree(pid: number, force = false): void {
  if (!Number.isInteger(pid) || pid <= 0) return;
  if (isWindows()) {
    const args = force ? ["/F", "/T", "/PID", String(pid)] : ["/T", "/PID", String(pid)];
    spawnSync("taskkill", args, { encoding: "utf8", timeout: 8000, windowsHide: true });
    return;
  }
  const signal = force ? "SIGKILL" : "SIGTERM";
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

export function parseSsListenPids(text: string): number[] {
  const found = new Set<number>();
  for (const match of text.matchAll(/pid=(\d+)/g)) found.add(Number(match[1]));
  return [...found].filter((pid) => Number.isInteger(pid) && pid > 0);
}

export function parseLsofListenPids(text: string): number[] {
  const found = new Set<number>();
  for (const line of text.split(/\s+/)) {
    const n = Number(line);
    if (Number.isInteger(n) && n > 0) found.add(n);
  }
  return [...found];
}

export function parseNetstatListenPids(text: string, port: number): number[] {
  const found = new Set<number>();
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!/LISTENING/i.test(line)) continue;
    const match = /(?:^|\s)(?:TCP|UDP)\s+(\S+)\s+\S+\s+LISTENING\s+(\d+)\s*$/i.exec(line);
    if (!match?.[1] || !match[2]) continue;
    const local = match[1];
    const localPort = Number(/:(\d+)$/.exec(local)?.[1] ?? /\[::\]:(\d+)$/.exec(local)?.[1]);
    const pid = Number(match[2]);
    if (localPort === port && Number.isInteger(pid) && pid > 0) found.add(pid);
  }
  return [...found];
}

export function listenerPids(port: number): number[] {
  const found = new Set<number>();
  if (isWindows()) {
    const netstat = runCapture("netstat", ["-ano", "-p", "tcp"], 4000) ?? runCapture("netstat", ["-ano"], 4000);
    for (const pid of parseNetstatListenPids(netstat ?? "", port)) found.add(pid);
    return [...found];
  }
  const ss = runCapture("ss", ["-ltnp", `sport = :${port}`], 4000);
  for (const pid of parseSsListenPids(ss ?? "")) found.add(pid);
  const lsof = runCapture("lsof", ["-t", `-iTCP:${port}`, "-sTCP:LISTEN"], 4000);
  for (const pid of parseLsofListenPids(lsof ?? "")) found.add(pid);
  if (found.size === 0) {
    const fuser = `${runCapture("fuser", ["-n", "tcp", String(port)], 4000) ?? ""}\n`;
    for (const match of fuser.matchAll(/\b(\d+)\b/g)) {
      const n = Number(match[1]);
      if (n > 0 && n !== port) found.add(n);
    }
  }
  return [...found].filter((pid) => Number.isInteger(pid) && pid > 0);
}

export function openUrl(url: string): void {
  if (isDarwin()) {
    spawn("open", [url], { stdio: "ignore", detached: true }).unref();
    return;
  }
  if (isWindows()) {
    spawn("cmd", ["/c", "start", "", url], { stdio: "ignore", detached: true, windowsHide: true }).unref();
    return;
  }
  spawn("xdg-open", [url], { stdio: "ignore", detached: true }).unref();
}

export function spawnHideOptions(): { windowsHide: boolean } {
  return { windowsHide: true };
}

/** Probe Win32_VideoController; returns null when PowerShell/WMI is unavailable. */
export function probeWin32VideoControllers(): string | null {
  if (!isWindows()) return null;
  const ps = runCapture(
    "powershell.exe",
    [
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-Command",
      "Get-CimInstance Win32_VideoController | ForEach-Object { '{0}|{1}|{2}' -f $_.Name, $_.AdapterRAM, $_.PNPDeviceID }",
    ],
    8000,
  );
  if (ps?.trim()) return ps;
  return runCapture("wmic", ["path", "win32_VideoController", "get", "Name,AdapterRAM,PNPDeviceID", "/format:csv"], 8000);
}
