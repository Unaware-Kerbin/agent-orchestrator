import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, readlinkSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { packageRoot } from "./config.js";
import { stateDir } from "./state.js";

const PID_FILE = "gui.pid";
const GUI_SCRIPT = "src/gui.ts";

export function guiPidPath(): string {
  return join(stateDir(), PID_FILE);
}

export function isGuiCmdline(cmdline: string): boolean {
  const text = cmdline.replace(/\0/g, " ");
  return text.includes(GUI_SCRIPT) && !text.includes("gui-process") && !text.includes("--stop");
}

export function guiAddrInUseMessage(port: number): string {
  return [
    `listen EADDRINUSE: address already in use 127.0.0.1:${port}`,
    `That is this project's Node GUI (tsx src/gui.ts), not vLLM (orch-vllm / port 8000).`,
    `Stop it with:  npm run gui:stop`,
    `Then start:    npm run gui`,
    `If you only wanted the existing server, do not start a second copy — open the token URL using .orchestrator/gui.secret.`,
  ].join("\n");
}

export function readGuiPid(): number | undefined {
  const path = guiPidPath();
  if (!existsSync(path)) return undefined;
  try {
    const pid = Number(readFileSync(path, "utf8").trim());
    return Number.isInteger(pid) && pid > 0 ? pid : undefined;
  } catch {
    return undefined;
  }
}

export function writeGuiPid(pid: number): void {
  writeFileSync(guiPidPath(), `${pid}\n`, { encoding: "utf8", mode: 0o600 });
}

export function clearGuiPid(expectedPid?: number): void {
  const path = guiPidPath();
  if (!existsSync(path)) return;
  if (expectedPid != null) {
    const current = readGuiPid();
    if (current !== undefined && current !== expectedPid) return;
  }
  try {
    unlinkSync(path);
  } catch {
    // ignore
  }
}

export function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

export function readCmdline(pid: number): string {
  try {
    return readFileSync(`/proc/${pid}/cmdline`, "utf8");
  } catch {
    return "";
  }
}

function readCwd(pid: number): string {
  try {
    return readlinkSync(`/proc/${pid}/cwd`);
  } catch {
    return "";
  }
}

export function isOurGuiProcess(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0 || !pidAlive(pid)) return false;
  if (!isGuiCmdline(readCmdline(pid))) return false;
  const cwd = readCwd(pid);
  const root = packageRoot();
  if (cwd && cwd !== root) return false;
  return true;
}

function runStdout(command: string, args: string[]): string {
  const result = spawnSync(command, args, { encoding: "utf8" });
  return result.stdout ?? "";
}

function runText(command: string, args: string[]): string {
  const result = spawnSync(command, args, { encoding: "utf8" });
  return `${result.stdout ?? ""}\n${result.stderr ?? ""}`;
}

export function listenerPids(port: number): number[] {
  const found = new Set<number>();
  const ss = runStdout("ss", ["-ltnp", `sport = :${port}`]);
  for (const match of ss.matchAll(/pid=(\d+)/g)) found.add(Number(match[1]));
  const lsof = runStdout("lsof", ["-t", `-iTCP:${port}`, "-sTCP:LISTEN"]);
  for (const line of lsof.split(/\s+/)) {
    const n = Number(line);
    if (Number.isInteger(n) && n > 0) found.add(n);
  }
  if (found.size === 0) {
    const fuser = runText("fuser", ["-n", "tcp", String(port)]);
    for (const match of fuser.matchAll(/\b(\d+)\b/g)) {
      const n = Number(match[1]);
      if (n > 0 && n !== port) found.add(n);
    }
  }
  return [...found].filter((pid) => Number.isInteger(pid) && pid > 0);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function stopOurGui(options: { port: number }): Promise<{
  stopped: number[];
  foreign: number[];
  message: string;
}> {
  const candidates = new Set<number>();
  const filePid = readGuiPid();
  if (filePid) candidates.add(filePid);
  for (const pid of listenerPids(options.port)) candidates.add(pid);

  const ours = [...candidates].filter((pid) => pid !== process.pid && isOurGuiProcess(pid));
  const foreign = [...candidates].filter(
    (pid) => pid !== process.pid && pidAlive(pid) && !isOurGuiProcess(pid) && listenerPids(options.port).includes(pid),
  );

  if (ours.length === 0) {
    clearGuiPid();
    if (foreign.length > 0) {
      return {
        stopped: [],
        foreign,
        message: `Port ${options.port} is in use by pid ${foreign.join(", ")} (not this project's GUI). Not killing it.`,
      };
    }
    return {
      stopped: [],
      foreign: [],
      message: `No orchestrator GUI on 127.0.0.1:${options.port}.`,
    };
  }

  for (const pid of ours) {
    try {
      process.kill(pid, "SIGTERM");
    } catch {
      // already gone
    }
  }

  const until = Date.now() + 5000;
  while (Date.now() < until) {
    const alive = ours.filter((pid) => pidAlive(pid));
    const stillBound = listenerPids(options.port).some((pid) => ours.includes(pid));
    if (alive.length === 0 && !stillBound) break;
    await sleep(100);
  }

  for (const pid of ours) {
    if (!pidAlive(pid)) clearGuiPid(pid);
  }
  if (!ours.some((pid) => pidAlive(pid))) clearGuiPid();

  const still = ours.filter((pid) => pidAlive(pid));
  return {
    stopped: ours,
    foreign: [],
    message:
      still.length === 0
        ? `Stopped GUI pid ${ours.join(", ")} (SIGTERM). 127.0.0.1:${options.port} is free.`
        : `Sent SIGTERM to GUI pid ${ours.join(", ")}; still running: ${still.join(", ")}.`,
  };
}
