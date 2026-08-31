import { statSync } from "node:fs";
import { join } from "node:path";
import { packageRoot } from "../config.js";
import { which, type WhichOptions } from "../platform.js";

function isDir(path: string): boolean {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

export function bundledBinDir(): string {
  const fromEnv = process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN?.trim();
  if (fromEnv) return fromEnv;
  return join(packageRoot(), "runtime", "bin");
}

/** Bundled `runtime/bin` first, then PATH. */
export function findEngineBin(
  name: string,
  whichFn: (cmd: string, options?: WhichOptions) => string | undefined = which,
): string | undefined {
  const bundled = bundledBinDir();
  if (isDir(bundled)) {
    const hit = whichFn(name, { extraDirs: [bundled], pathEnv: "" });
    if (hit) return hit;
  }
  return whichFn(name);
}
