import { existsSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { packageRoot } from "../config.js";
import { which, type WhichOptions } from "../platform.js";

function isDir(path: string): boolean {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

function isFile(path: string): boolean {
  try {
    return statSync(path).isFile();
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

const LATE_INFER_NAMES = ["late-infer", "late-infer.exe"] as const;

/**
 * This application's own late-infer locations (packed runtime/bin, `npm run infer:build`
 * output under bin/, dist/bin, cargo target/release). BUNDLE_BIN env wins when set.
 */
export function shippedLateInferDirs(root = packageRoot()): string[] {
  const dirs: string[] = [];
  const fromEnv = process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN?.trim();
  if (fromEnv) dirs.push(fromEnv);
  dirs.push(join(root, "runtime", "bin"));
  dirs.push(join(root, "bin"));
  dirs.push(join(root, "dist", "bin"));
  dirs.push(join(root, "target", "release"));
  return [...new Set(dirs)];
}

/** Dual-repo workspace last resort. Operators are never told to install Late. */
export function siblingLateReleaseDirs(root = packageRoot()): string[] {
  const parent = dirname(root);
  const dirs: string[] = [];
  const checkout = process.env.LATE_CHECKOUT?.trim();
  if (checkout) dirs.push(join(checkout, "target", "release"));
  for (const name of ["Local_AI_Terminal_Emulator", "late"]) {
    dirs.push(join(parent, name, "target", "release"));
  }
  return dirs;
}

function lateInferInDir(dir: string): string | undefined {
  for (const name of LATE_INFER_NAMES) {
    const path = join(dir, name);
    if (isFile(path)) return path;
  }
  return undefined;
}

/**
 * late-infer binary: explicit env, this application's shipped dirs, then (dev
 * last resort) a sibling crate `target/release`, then PATH.
 */
export function findLateInferBin(
  whichFn: (cmd: string, options?: WhichOptions) => string | undefined = which,
  root = packageRoot(),
): string | undefined {
  const explicit = process.env.AGENT_ORCHESTRATOR_LATE_INFER?.trim();
  if (explicit && isFile(explicit)) return explicit;

  for (const dir of shippedLateInferDirs(root)) {
    const hit = lateInferInDir(dir);
    if (hit) return hit;
  }

  for (const dir of siblingLateReleaseDirs(root)) {
    const hit = lateInferInDir(dir);
    if (hit) return hit;
  }

  return whichFn("late-infer") ?? whichFn("late-infer.exe");
}

export function lateInferMissingMessage(): string {
  return (
    "late-infer is missing. On your computer, from this repo run `npm run infer:build`, then Start. " +
    "Packed installs put it in this application's runtime/bin; from source it lands in bin/. " +
    "Bind stays 127.0.0.1:8010. Weights are not in the installer."
  );
}

export function lateInferBinPresent(path: string | undefined): boolean {
  return Boolean(path && existsSync(path) && isFile(path));
}

/** LD_LIBRARY_PATH / PATH extras so a shipped late-infer finds sibling libs. */
export function engineLibEnv(bin: string, base: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const dir = dirname(bin);
  const lib = join(dirname(dir), "lib");
  const ollamaLib = join(lib, "ollama");
  const extra = [dir, lib, ollamaLib].filter((p) => existsSync(p));
  const env = { ...base };
  if (process.platform === "win32") {
    env.PATH = [...extra, base.PATH ?? ""].join(";");
  } else if (process.platform === "darwin") {
    env.DYLD_LIBRARY_PATH = [...extra, base.DYLD_LIBRARY_PATH ?? ""].join(":");
  } else {
    env.LD_LIBRARY_PATH = [...extra, base.LD_LIBRARY_PATH ?? ""].join(":");
  }
  return env;
}
