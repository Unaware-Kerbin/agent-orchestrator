import { isAbsolute, resolve } from "node:path";
import { packageRoot } from "./config.js";
import { ensureSecureDir } from "./platform.js";

export function stateDir(): string {
  const fromEnv = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  const dir = fromEnv
    ? isAbsolute(fromEnv)
      ? fromEnv
      : resolve(process.cwd(), fromEnv)
    : resolve(packageRoot(), ".orchestrator");
  ensureSecureDir(dir);
  return dir;
}
