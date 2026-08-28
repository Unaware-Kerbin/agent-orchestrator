import { existsSync, readFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { packageRoot } from "./config.js";
import {
  listenerPids,
  normalizeCmdline,
  pidAlive,
  readProcessCmdline,
  readProcessCwd,
  stopProcess,
  writeSecureFile,
} from "./platform.js";
import { stateDir } from "./state.js";

const PID_FILE = "gui.pid";
const GUI_SCRIPT = "src/gui.ts";

export function guiPidPath(): string {
  return join(stateDir(), PID_FILE);
}

export function isGuiCmdline(cmdline: string): boolean {
  const text = normalizeCmdline(cmdline);
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
  writeSecureFile(guiPidPath(), `${pid}\n`);
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

export { listenerPids, pidAlive };

export function readCmdline(pid: number): string {
  return readProcessCmdline(pid);
}

export function isOurGuiProcess(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0 || !pidAlive(pid)) return false;
  if (!isGuiCmdline(readCmdline(pid))) return false;
  const cwd = readProcessCwd(pid);
  const root = packageRoot();
  if (cwd && cwd !== root) return false;
  return true;
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
    stopProcess(pid);
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
        ? `Stopped GUI pid ${ours.join(", ")}. 127.0.0.1:${options.port} is free.`
        : `Sent stop to GUI pid ${ours.join(", ")}; still running: ${still.join(", ")}.`,
  };
}
