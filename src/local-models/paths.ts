import { mkdirSync } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";
import type { WriteAllowlist } from "../allowlist.js";
import { stateDir } from "../state.js";
import { repoDirName } from "./catalog.js";

export const MODELS_DIR_ENV = "AGENT_ORCHESTRATOR_MODELS_DIR";

export function defaultModelsDir(allowlist: WriteAllowlist): string {
  const fromEnv = process.env[MODELS_DIR_ENV]?.trim();
  if (fromEnv) {
    return isAbsolute(fromEnv) ? fromEnv : resolve(process.cwd(), fromEnv);
  }
  const underState = join(stateDir(), "models");
  try {
    allowlist.assertWritable(underState);
    return underState;
  } catch {
    const first = allowlist.list()[0];
    if (first) return join(first, ".orchestrator", "models");
    return underState;
  }
}

/** Resolve dest under the models dir and require it to sit inside the write allowlist. */
export function assertModelDest(
  allowlist: WriteAllowlist,
  modelsDir: string,
  hfRepo: string,
  overrideDest?: string,
): string {
  const dest = overrideDest?.trim()
    ? isAbsolute(overrideDest.trim())
      ? overrideDest.trim()
      : resolve(modelsDir, overrideDest.trim())
    : join(modelsDir, repoDirName(hfRepo));
  return allowlist.assertWritable(dest);
}

export function ensureModelsDir(allowlist: WriteAllowlist, modelsDir: string): string {
  const allowed = allowlist.assertWritable(modelsDir);
  mkdirSync(allowed, { recursive: true, mode: 0o700 });
  return allowed;
}
