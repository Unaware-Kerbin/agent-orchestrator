#!/usr/bin/env node
/**
 * Used only by `npm run gui`. Builds late-infer once per install / app-version bump.
 * `npm start` / `mcp:http` / MCP bind must not import or run this file.
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

export const DEFAULT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const BIN_STAMP_NAME = "late-infer.stamp";
export const RUNTIME_STAMP_NAME = ".late-infer-built-for";
export const BINARY_NAMES = ["late-infer", "late-infer.exe"];

function isFile(path) {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
}

export function packageVersion(root = DEFAULT_ROOT) {
  const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
  return String(pkg.version ?? "").trim();
}

export function binStampPath(root) {
  return join(root, "bin", BIN_STAMP_NAME);
}

export function runtimeStampPath(root) {
  return join(root, "runtime", "bin", RUNTIME_STAMP_NAME);
}

export function findShippedLateInfer(root) {
  for (const name of BINARY_NAMES) {
    const binPath = join(root, "bin", name);
    if (isFile(binPath)) {
      return { path: binPath, stamp: binStampPath(root), kind: "bin" };
    }
  }
  for (const name of BINARY_NAMES) {
    const packed = join(root, "runtime", "bin", name);
    if (isFile(packed)) {
      return { path: packed, stamp: runtimeStampPath(root), kind: "runtime" };
    }
  }
  return undefined;
}

export function readStamp(path) {
  try {
    return readFileSync(path, "utf8").trim();
  } catch {
    return undefined;
  }
}

/** Sibling LATE checkout `crates/late-infer` (or LATE_INFER_CRATE / LATE_CHECKOUT). */
export function findLateInferCrate(root = DEFAULT_ROOT) {
  const fromEnv = (process.env.LATE_INFER_CRATE || "").trim();
  if (fromEnv && existsSync(join(fromEnv, "Cargo.toml"))) return fromEnv;
  const checkout = (process.env.LATE_CHECKOUT || "").trim();
  if (checkout) {
    const crate = join(checkout, "crates", "late-infer");
    if (existsSync(join(crate, "Cargo.toml"))) return crate;
  }
  const parent = dirname(root);
  for (const name of ["Local_AI_Terminal_Emulator", "late"]) {
    const crate = join(parent, name, "crates", "late-infer");
    if (existsSync(join(crate, "Cargo.toml"))) return crate;
  }
  return undefined;
}

function mtimeMs(path) {
  try {
    return statSync(path).mtimeMs;
  } catch {
    return 0;
  }
}

/** True when crate sources are newer than the shipped binary (dev dual-repo rebuild). */
export function lateInferSourceNewerThanBinary(root, binPath) {
  const crate = findLateInferCrate(root);
  if (!crate || !binPath) return false;
  const binM = mtimeMs(binPath);
  if (!(binM > 0)) return true;
  const watched = [
    join(crate, "Cargo.toml"),
    join(crate, "src", "main.rs"),
    join(crate, "src", "engine.rs"),
    join(crate, "src", "intel.rs"),
    join(crate, "src", "compiler.rs"),
    join(crate, "src", "device.rs"),
    join(crate, "python", "ov_worker.py"),
  ];
  return watched.some((p) => mtimeMs(p) > binM);
}

/**
 * @returns {{ needed: boolean, reason: "first launch" | "after upgrade" | "source newer" | "ok" }}
 */
export function needsLateInferBuild(root, version) {
  const hit = findShippedLateInfer(root);
  if (!hit) return { needed: true, reason: "first launch" };
  const stamped = readStamp(hit.stamp);
  if (stamped !== version) return { needed: true, reason: "after upgrade" };
  if (lateInferSourceNewerThanBinary(root, hit.path)) {
    return { needed: true, reason: "source newer" };
  }
  return { needed: false, reason: "ok" };
}

export function writeLateInferStamps(root, version) {
  mkdirSync(join(root, "bin"), { recursive: true });
  mkdirSync(join(root, "runtime", "bin"), { recursive: true });
  writeFileSync(binStampPath(root), `${version}\n`);
  writeFileSync(runtimeStampPath(root), `${version}\n`);
}

export function defaultSpawnBuild(root) {
  return spawnSync("npm", ["run", "infer:build"], {
    cwd: root,
    stdio: "inherit",
    env: process.env,
    shell: process.platform === "win32",
  });
}

/**
 * @param {{
 *   root?: string,
 *   version?: string,
 *   spawnBuild?: (root: string) => { status?: number | null },
 *   log?: (line: string) => void,
 * }} [opts]
 */
export function ensureLateInfer(opts = {}) {
  const root = opts.root ?? DEFAULT_ROOT;
  const version = opts.version ?? packageVersion(root);
  const spawnBuild = opts.spawnBuild ?? defaultSpawnBuild;
  const log = opts.log ?? ((line) => console.error(line));

  const decision = needsLateInferBuild(root, version);
  if (!decision.needed) {
    return { built: false, skipped: true, failed: false, reason: decision.reason };
  }

  log(`agent-orchestrator: building late-infer on your computer (${decision.reason})…`);
  const result = spawnBuild(root);
  const status = result.status ?? 1;
  if (status !== 0) {
    return { built: false, skipped: false, failed: true, status, reason: decision.reason };
  }
  writeLateInferStamps(root, version);
  log("agent-orchestrator: late-infer ready");
  return { built: true, skipped: false, failed: false, status: 0, reason: decision.reason };
}

function isDirectRun() {
  const entry = process.argv[1];
  if (!entry) return false;
  try {
    return import.meta.url === pathToFileURL(resolve(entry)).href;
  } catch {
    return false;
  }
}

if (isDirectRun()) {
  if (!existsSync(join(DEFAULT_ROOT, "package.json"))) {
    console.error("agent-orchestrator: package.json missing; cannot ensure late-infer.");
    process.exit(1);
  }
  const result = ensureLateInfer({ root: DEFAULT_ROOT, version: packageVersion(DEFAULT_ROOT) });
  if (result.failed) process.exit(result.status ?? 1);
}
