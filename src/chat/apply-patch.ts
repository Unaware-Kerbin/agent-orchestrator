import { existsSync, mkdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, posix, sep, win32 } from "node:path";
import { isPathInside, resolveContainedPath, type WriteAllowlist } from "../allowlist.js";

export type PatchFile = { path: string; content: string };

const FENCE_RE = /```orchestrator-files\s*\n([\s\S]*?)```/gi;
const BLOCKED = /(^|[\\/])(\.orchestrator|write-allowlist\.json)([\\/]|$)/i;

export function parseOrchestratorFiles(plan: string): PatchFile[] {
  const fromFence = parseFences(plan);
  if (fromFence.length) return fromFence;
  return parseUnifiedDiff(plan);
}

function parseFences(plan: string): PatchFile[] {
  const out: PatchFile[] = [];
  let match: RegExpExecArray | null;
  const re = new RegExp(FENCE_RE.source, "gi");
  while ((match = re.exec(plan))) {
    const raw = (match[1] ?? "").trim();
    if (!raw) continue;
    try {
      const parsed: unknown = JSON.parse(raw);
      out.push(...filesFromUnknown(parsed));
    } catch {
      /* not JSON in this fence */
    }
  }
  return dedupe(out);
}

function filesFromUnknown(value: unknown): PatchFile[] {
  if (!value || typeof value !== "object") return [];
  const rec = value as Record<string, unknown>;
  const rows = Array.isArray(rec.files) ? rec.files : Array.isArray(value) ? value : [value];
  const out: PatchFile[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const item = row as Record<string, unknown>;
    const path = typeof item.path === "string" ? item.path.trim() : "";
    const content = typeof item.content === "string" ? item.content : typeof item.body === "string" ? item.body : "";
    if (!path) continue;
    out.push({ path, content });
  }
  return out;
}

function parseUnifiedDiff(plan: string): PatchFile[] {
  const files: PatchFile[] = [];
  const chunks = plan.split(/^diff --git /m);
  const also = plan.includes("+++ ") ? [plan] : [];
  const bodies = chunks.length > 1 ? chunks.slice(1) : also;
  for (const body of bodies) {
    const plus = body.match(/^\+\+\+\s+(?:b\/)?(.+)$/m);
    const path = (plus?.[1] ?? "").trim();
    if (!path || path === "/dev/null") continue;
    const lines = body.split("\n");
    const contentLines: string[] = [];
    let inHunk = false;
    for (const line of lines) {
      if (line.startsWith("@@")) {
        inHunk = true;
        continue;
      }
      if (!inHunk) continue;
      if (line.startsWith("+") && !line.startsWith("+++")) contentLines.push(line.slice(1));
      else if (line.startsWith(" ")) contentLines.push(line.slice(1));
    }
    if (contentLines.length) files.push({ path, content: contentLines.join("\n") + (contentLines.length ? "\n" : "") });
  }
  return dedupe(files);
}

function dedupe(files: PatchFile[]): PatchFile[] {
  const map = new Map<string, PatchFile>();
  for (const file of files) map.set(file.path, file);
  return [...map.values()];
}

function looksAbsolute(rel: string): boolean {
  return (
    posix.isAbsolute(rel) ||
    win32.isAbsolute(rel) ||
    win32.isAbsolute(rel.replaceAll("/", "\\")) ||
    /^[a-zA-Z]:/.test(rel)
  );
}

export function assertSafeRelPath(rel: string): string {
  const trimmed = rel.trim().replaceAll("\\", "/");
  if (!trimmed) throw new Error("empty file path");
  if (trimmed.includes("\0") || /[\u0001-\u0008\u000b\u000c\u000e-\u001f]/.test(trimmed)) {
    throw new Error("file path contains control characters");
  }
  if (looksAbsolute(trimmed)) throw new Error(`absolute file path refused: ${rel}`);
  if (BLOCKED.test(trimmed)) throw new Error(`refusing to write ${trimmed}`);
  const norm = posix.normalize(trimmed);
  if (!norm || norm === "." || norm === "/") throw new Error("empty file path");
  if (looksAbsolute(norm) || norm.startsWith("..") || norm.split("/").includes("..")) {
    throw new Error(`path escapes workspace: ${rel}`);
  }
  if (BLOCKED.test(norm)) throw new Error(`refusing to write ${rel}`);
  return norm;
}

export function applyParsedFiles(input: {
  cwd: string;
  files: PatchFile[];
  allowlist: WriteAllowlist;
}): { written: string[] } {
  if (!input.files.length) {
    throw new Error(
      'No files to apply. Put a fenced orchestrator-files JSON block in the plan: ```orchestrator-files\\n{"files":[{"path":"README.md","content":"..."}]}\\n```',
    );
  }
  const cwdReal = resolveContainedPath(input.cwd);
  const written: string[] = [];
  for (const file of input.files) {
    if (file.content.includes("\0")) throw new Error(`binary content refused for ${file.path}`);
    if (isAbsolute(file.path) || looksAbsolute(file.path.trim().replaceAll("\\", "/"))) {
      throw new Error(`absolute file path refused: ${file.path}`);
    }
    const rel = assertSafeRelPath(file.path);
    const abs = join(cwdReal, rel.split("/").join(sep));
    if (BLOCKED.test(abs.replaceAll("\\", "/"))) throw new Error(`refusing to write ${file.path}`);
    const target = input.allowlist.assertWritable(abs);
    if (target === input.allowlist.filePath) {
      throw new Error(`refusing to write ${file.path}`);
    }
    if (!isPathInside(target, cwdReal)) {
      throw new Error(`path escapes workspace: ${file.path}`);
    }
    if (existsSync(target) && statSync(target).isDirectory()) {
      throw new Error(`refusing to write directory ${file.path}`);
    }
    try {
      mkdirSync(dirname(target), { recursive: true });
      writeFileSync(target, file.content, { encoding: "utf8", mode: 0o644 });
    } catch (error) {
      const code = (error as NodeJS.ErrnoException).code;
      if (code === "EISDIR") throw new Error(`refusing to write directory ${file.path}`);
      if (code === "EACCES" || code === "EPERM") throw new Error(`permission denied writing ${file.path}`);
      if (code === "ENOTDIR") throw new Error(`parent path is not a directory for ${file.path}`);
      throw error instanceof Error ? error : new Error(String(error));
    }
    written.push(target);
  }
  return { written };
}

export const APPLY_PATCH_INSTRUCTIONS = `When the user Approves, the orchestrator (not you) will write files. End your plan with a fenced block:

\`\`\`orchestrator-files
{"files":[{"path":"relative/path.ext","content":"full file contents"}]}
\`\`\`

Paths are relative to the granted cwd. Do not include binaries, .., .orchestrator, or write-allowlist.json. You cannot run shell commands.`;
