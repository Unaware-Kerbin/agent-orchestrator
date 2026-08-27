import { existsSync, mkdirSync, readFileSync, realpathSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { stateDir } from "./state.js";

const FILE_NAME = "write-allowlist.json";

export interface AllowlistFile {
  version: 1;
  directories: string[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function allowlistPath(): string {
  const fromEnv = process.env.AGENT_ORCHESTRATOR_ALLOWLIST;
  if (fromEnv) return isAbsolute(fromEnv) ? fromEnv : resolve(process.cwd(), fromEnv);
  return join(stateDir(), FILE_NAME);
}

/** Resolve `..` and follow realpath. Non-existent leafs use the nearest existing ancestor. */
export function resolveContainedPath(input: string): string {
  const absolute = resolve(input);
  if (existsSync(absolute)) {
    return realpathSync(absolute);
  }

  const missing: string[] = [];
  let current = absolute;
  while (!existsSync(current)) {
    missing.unshift(basename(current));
    const parent = dirname(current);
    if (parent === current) {
      throw new Error(`Cannot resolve path "${input}"`);
    }
    current = parent;
  }

  return join(realpathSync(current), ...missing);
}

export function isPathInside(target: string, root: string): boolean {
  const rel = relative(root, target);
  if (rel === "") return true;
  return !rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel);
}

export function canonicalizeDirectory(input: string): string {
  const resolved = resolveContainedPath(input);
  if (!existsSync(resolved)) {
    throw new Error(`Directory does not exist: ${input}`);
  }
  const st = statSync(resolved);
  if (!st.isDirectory()) {
    throw new Error(`Not a directory: ${input}`);
  }
  return resolved;
}

export class WriteAllowlist {
  private directories: string[] = [];
  private mtimeMs = 0;

  constructor(
    readonly filePath: string,
    private readonly defaults: string[],
  ) {
    this.loadFromDisk(true);
  }

  static load(defaults: string[]): WriteAllowlist {
    const normalizedDefaults: string[] = [];
    for (const dir of defaults) {
      try {
        normalizedDefaults.push(canonicalizeDirectory(dir));
        break;
      } catch {
        // try the next candidate (missing WORKSPACE_CWD, etc.)
      }
    }
    if (normalizedDefaults.length === 0) {
      throw new Error("Write allowlist needs at least one existing default directory (the project workspace)");
    }
    return new WriteAllowlist(allowlistPath(), normalizedDefaults);
  }

  list(): string[] {
    this.reloadIfChanged();
    return [...this.directories];
  }

  add(input: string): string[] {
    this.reloadIfChanged();
    const real = canonicalizeDirectory(input);
    if (!this.directories.includes(real)) {
      this.directories.push(real);
      this.directories.sort();
      this.persist();
    }
    return this.list();
  }

  remove(input: string): string[] {
    this.reloadIfChanged();
    let real: string | undefined;
    try {
      real = canonicalizeDirectory(input);
    } catch {
      real = resolve(input);
    }
    this.directories = this.directories.filter((dir) => dir !== real && dir !== resolve(input));
    this.persist();
    return this.list();
  }

  /** cwd for local Cursor agents: must exist and sit inside a granted directory. */
  assertCwd(input: string): string {
    this.reloadIfChanged();
    const real = canonicalizeDirectory(input);
    this.assertInside(real, input);
    return real;
  }

  /** Realpath if `input` is an existing directory inside the allowlist; otherwise undefined. */
  tryCwd(input: string): string | undefined {
    try {
      return this.assertCwd(input);
    } catch {
      return undefined;
    }
  }

  isAllowed(input: string): boolean {
    return this.tryCwd(input) !== undefined;
  }

  /** Target file or directory (may not exist yet). Rejects symlink escapes. */
  assertWritable(input: string): string {
    this.reloadIfChanged();
    const resolved = resolveContainedPath(input);
    this.assertInside(resolved, input);
    return resolved;
  }

  reloadIfChanged(): void {
    if (!existsSync(this.filePath)) return;
    try {
      const mtime = statSync(this.filePath).mtimeMs;
      if (mtime > this.mtimeMs) this.loadFromDisk(false);
    } catch {
      // keep in-memory copy
    }
  }

  private assertInside(resolved: string, original: string): void {
    for (const dir of this.directories) {
      if (isPathInside(resolved, dir)) return;
    }
    throw new Error(
      `Path "${original}" is not inside an allowed directory. Allowed: ${this.directories.join(", ") || "(none)"}`,
    );
  }

  private loadFromDisk(createIfMissing: boolean): void {
    if (!existsSync(this.filePath)) {
      this.directories = [...this.defaults];
      if (createIfMissing) this.persist();
      return;
    }
    const raw = JSON.parse(readFileSync(this.filePath, "utf8")) as unknown;
    const listed = isRecord(raw) && Array.isArray(raw.directories) ? raw.directories : [];
    const resolved: string[] = [];
    for (const item of listed) {
      if (typeof item !== "string" || !item.trim()) continue;
      try {
        resolved.push(canonicalizeDirectory(item));
      } catch {
        // Drop stale paths that no longer exist.
      }
    }
    this.directories = resolved.length > 0 ? [...new Set(resolved)].sort() : [...this.defaults];
    try {
      this.mtimeMs = statSync(this.filePath).mtimeMs;
    } catch {
      this.mtimeMs = Date.now();
    }
  }

  private persist(): void {
    mkdirSync(dirname(this.filePath), { recursive: true, mode: 0o700 });
    const payload: AllowlistFile = { version: 1, directories: [...this.directories] };
    writeFileSync(this.filePath, `${JSON.stringify(payload, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
    try {
      this.mtimeMs = statSync(this.filePath).mtimeMs;
    } catch {
      this.mtimeMs = Date.now();
    }
  }
}
