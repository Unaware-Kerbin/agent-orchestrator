import { mkdirSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";
import { packageRoot } from "./config.js";

export function stateDir(): string {
  const fromEnv = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  const dir = fromEnv
    ? isAbsolute(fromEnv)
      ? fromEnv
      : resolve(process.cwd(), fromEnv)
    : resolve(packageRoot(), ".orchestrator");
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  return dir;
}
