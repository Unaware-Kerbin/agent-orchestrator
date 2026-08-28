import { existsSync, readFileSync, realpathSync, statSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";
import { resolveContainedPath } from "./allowlist.js";

const DEFAULT_TTL_MS = 10 * 60 * 1000;
const PCAP_EXT = /\.(pcap|pcapng|cap)$/i;

const MAGIC = [
  Buffer.from([0xd4, 0xc3, 0xb2, 0xa1]),
  Buffer.from([0xa1, 0xb2, 0xc3, 0xd4]),
  Buffer.from([0x4d, 0x3c, 0xb2, 0xa1]),
  Buffer.from([0xa1, 0xb2, 0x3c, 0x4d]),
  Buffer.from([0x0a, 0x0d, 0x0d, 0x0a]),
];

export interface TempAnalyzeGrant {
  path: string;
  expiresAt: number;
}

export function looksLikePcapMagic(bytes: Buffer): boolean {
  if (bytes.length < 4) return false;
  return MAGIC.some((m) => bytes.subarray(0, m.length).equals(m));
}

function readHead(path: string, n = 16): Buffer {
  const buf = readFileSync(path);
  return buf.subarray(0, Math.min(n, buf.length));
}

/**
 * Canonicalize one existing file for a temporary read/analyze grant.
 * Rejects directories, `..` segments, NUL, and non-pcap files (unless magic bytes match).
 */
export function assertTempAnalyzeFile(input: string): string {
  if (typeof input !== "string" || !input.trim()) {
    throw new Error("path is required");
  }
  if (input.includes("\0")) {
    throw new Error("path must not contain NUL");
  }
  const trimmed = input.trim();
  if (!isAbsolute(trimmed)) {
    throw new Error("path must be absolute");
  }
  if (trimmed.split(/[\\/]/).includes("..")) {
    throw new Error("path must not contain ..");
  }
  const resolved = resolveContainedPath(trimmed);
  if (resolve(trimmed) !== resolved && trimmed.split(/[\\/]/).includes("..")) {
    throw new Error("path must not contain ..");
  }
  if (!existsSync(resolved)) {
    throw new Error("file does not exist");
  }
  const st = statSync(resolved);
  if (st.isDirectory()) {
    throw new Error("path must be a file, not a directory");
  }
  if (!st.isFile()) {
    throw new Error("path must be a regular file");
  }
  const real = realpathSync(resolved);
  const extOk = PCAP_EXT.test(real);
  let head: Buffer;
  try {
    head = readHead(real, 12);
  } catch {
    throw new Error("unable to read file");
  }
  const magicOk = looksLikePcapMagic(head);
  if (!extOk && !magicOk) {
    throw new Error("not a pcap/pcapng file");
  }
  if (extOk && !magicOk) {
    throw new Error("file extension looks like pcap but magic bytes do not match");
  }
  return real;
}

/** In-memory read-only analyze grants. Never persisted; never grants write. */
export class TempAnalyzeAllowlist {
  private readonly grants = new Map<string, number>();

  constructor(private readonly defaultTtlMs = DEFAULT_TTL_MS) {}

  add(input: string, ttlMs?: number): TempAnalyzeGrant {
    const path = assertTempAnalyzeFile(input);
    const ttl = ttlMs && ttlMs > 0 ? Math.min(ttlMs, 60 * 60 * 1000) : this.defaultTtlMs;
    const expiresAt = Date.now() + ttl;
    this.grants.set(path, expiresAt);
    return { path, expiresAt };
  }

  has(input: string, now = Date.now()): boolean {
    this.purgeExpired(now);
    let real: string;
    try {
      real = assertTempAnalyzeFile(input);
    } catch {
      try {
        real = resolve(input);
      } catch {
        return false;
      }
    }
    const exp = this.grants.get(real);
    return Boolean(exp && exp > now);
  }

  /** Idempotent. */
  remove(input: string): boolean {
    let real: string;
    try {
      real = realpathSync(resolve(input));
    } catch {
      real = resolve(input);
    }
    const had = this.grants.delete(real);
    if (!had) {
      for (const key of this.grants.keys()) {
        if (key === input || key === resolve(input)) {
          this.grants.delete(key);
          return true;
        }
      }
    }
    return had;
  }

  list(now = Date.now()): TempAnalyzeGrant[] {
    this.purgeExpired(now);
    return [...this.grants.entries()].map(([path, expiresAt]) => ({ path, expiresAt }));
  }

  purgeExpired(now = Date.now()): string[] {
    const gone: string[] = [];
    for (const [path, exp] of this.grants) {
      if (exp <= now) {
        this.grants.delete(path);
        gone.push(path);
      }
    }
    return gone;
  }
}
