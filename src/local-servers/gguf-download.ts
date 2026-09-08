/**
 * Download a single GGUF (or all *.gguf) from a Hub repo into .orchestrator/models/gguf/.
 * Reuses HF_TOKEN via secrets; never logs the token.
 */

import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, renameSync, statSync } from "node:fs";
import { basename, join } from "node:path";
import { packageRoot } from "../config.js";
import { parseHubModelId } from "../identity.js";
import { ensureSecureDir, pythonDashArgs, pythonInterpreterNames, which } from "../platform.js";
import { redactSecretText } from "../redact.js";
import { loadSecretsIntoEnv, resolveHfToken } from "../secrets.js";
import { stateDir } from "../state.js";
import {
  defaultGgufModelsDir,
  findLocalGgufPath,
  localGgufDirForRepo,
  pickPreferredGgufFile,
  siblingToGgufFile,
  type GgufFileHint,
} from "./hub-gguf-catalog.js";
import { HF_HUB_MODELS_API, type HubFetchFn, type HubRawModel } from "./hub-catalog.js";

export type GgufDownloadStatus = "queued" | "running" | "done" | "error";

export interface GgufDownloadJob {
  id: string;
  repo: string;
  filename?: string;
  destDir: string;
  localPath?: string;
  status: GgufDownloadStatus;
  percent: number;
  message: string;
  error?: string;
  startedAt: number;
  updatedAt: number;
  finishedAt?: number;
}

const jobs = new Map<string, GgufDownloadJob>();
const children = new Map<string, ChildProcess>();

function findPython(): string | undefined {
  for (const name of pythonInterpreterNames()) {
    const path = which(name);
    if (path) return path;
  }
  return undefined;
}

function helperScript(): string {
  return join(packageRoot(), "scripts", "hf_download.py");
}

function hfDownloadChildEnv(): NodeJS.ProcessEnv {
  loadSecretsIntoEnv();
  const env = { ...process.env };
  const token = resolveHfToken();
  if (token) {
    env.HF_TOKEN = token;
    env.HUGGING_FACE_HUB_TOKEN = token;
  }
  env.HF_HUB_DISABLE_PROGRESS_BARS = "1";
  return env;
}

export function listGgufDownloadJobs(): GgufDownloadJob[] {
  return [...jobs.values()].sort((a, b) => b.updatedAt - a.updatedAt);
}

export function getGgufDownloadJob(repo: string): GgufDownloadJob | undefined {
  return jobs.get(repo.trim());
}

function patchJob(id: string, patch: Partial<GgufDownloadJob>): GgufDownloadJob {
  const cur = jobs.get(id);
  if (!cur) throw new Error(`unknown GGUF download job ${id}`);
  const next = { ...cur, ...patch, updatedAt: Date.now() };
  jobs.set(id, next);
  return next;
}

function failJob(job: GgufDownloadJob, message: string): void {
  patchJob(job.id, {
    status: "error",
    error: redactSecretText(message),
    message: redactSecretText(message).slice(0, 400),
    finishedAt: Date.now(),
  });
}

async function resolvePreferredFilename(
  repo: string,
  filename: string | undefined,
  fetchFn: HubFetchFn,
  token?: string,
): Promise<string | undefined> {
  if (filename?.trim()) return filename.trim();
  const url = new URL(`${HF_HUB_MODELS_API}/${repo}`);
  url.searchParams.append("expand", "siblings");
  const headers: Record<string, string> = {
    accept: "application/json",
    "user-agent": "agent-orchestrator",
  };
  if (token) headers.authorization = `Bearer ${token}`;
  let res: Response;
  try {
    res = await fetchFn(url, { method: "GET", headers, signal: AbortSignal.timeout(8_000) });
  } catch {
    return undefined;
  }
  if (!res.ok) return undefined;
  let body: unknown;
  try {
    body = await res.json();
  } catch {
    return undefined;
  }
  const row = body as HubRawModel;
  const files = (row.siblings ?? [])
    .map(siblingToGgufFile)
    .filter((f): f is GgufFileHint => Boolean(f));
  // Prefer a mid-size quant; budget unknown here — pick best rank.
  return pickPreferredGgufFile(files, Number.POSITIVE_INFINITY)?.filename;
}

function flattenDownloadedGguf(destDir: string, preferred?: string): string | undefined {
  if (existsSync(destDir)) {
    try {
      const names = readdirSync(destDir).filter((n) => /\.gguf$/i.test(n) && !n.includes(".."));
      if (preferred && names.includes(preferred)) return join(destDir, preferred);
      if (names[0]) return join(destDir, names[0]!);
      for (const sub of readdirSync(destDir)) {
        const subPath = join(destDir, sub);
        let st;
        try {
          st = statSync(subPath);
        } catch {
          continue;
        }
        if (!st.isDirectory()) continue;
        const nested = readdirSync(subPath).filter((n) => /\.gguf$/i.test(n));
        for (const name of nested) {
          const from = join(subPath, name);
          const to = join(destDir, name);
          if (!existsSync(to)) {
            try {
              renameSync(from, to);
            } catch {
              return from;
            }
          }
          if (preferred && name === preferred) return to;
        }
        const after = readdirSync(destDir).filter((n) => /\.gguf$/i.test(n));
        if (preferred && after.includes(preferred)) return join(destDir, preferred);
        if (after[0]) return join(destDir, after[0]!);
      }
    } catch {
      /* ignore */
    }
  }
  return findLocalGgufPathByScan(destDir, preferred);
}

function findLocalGgufPathByScan(destDir: string, preferred?: string): string | undefined {
  if (!existsSync(destDir)) return undefined;
  try {
    const stack = [destDir];
    const hits: string[] = [];
    while (stack.length) {
      const dir = stack.pop()!;
      for (const name of readdirSync(dir)) {
        if (name === ".." || name === ".") continue;
        const p = join(dir, name);
        let st;
        try {
          st = statSync(p);
        } catch {
          continue;
        }
        if (st.isDirectory()) stack.push(p);
        else if (/\.gguf$/i.test(name)) hits.push(p);
      }
    }
    if (preferred) {
      const match = hits.find((p) => basename(p) === preferred);
      if (match) return match;
    }
    return hits[0];
  } catch {
    return undefined;
  }
}

function spawnHfFileDownload(job: GgufDownloadJob, repo: string, file: string | undefined): void {
  const script = helperScript();
  const python = findPython();
  const env = hfDownloadChildEnv();
  let child: ChildProcess;

  if (python && existsSync(script)) {
    const args = [...pythonDashArgs(python), script, "--repo", repo, "--dest", job.destDir];
    if (file) {
      args.push("--file", file);
    } else {
      args.push("--include", "*.gguf");
    }
    child = spawn(python, args, { env, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  } else {
    const cli = which("hf") ?? which("huggingface-cli");
    if (!cli) {
      failJob(
        job,
        [
          python ? `Download helper missing at ${script}.` : "Python not found on PATH.",
          "Install Hugging Face tools: pip install huggingface_hub",
        ].join(" "),
      );
      return;
    }
    const args = file
      ? ["download", repo, file, "--local-dir", job.destDir]
      : ["download", repo, "--include", "*.gguf", "--local-dir", job.destDir];
    child = spawn(cli, args, { env, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
  }

  children.set(job.id, child);
  let stderr = "";
  child.stdout?.on("data", (chunk: Buffer) => {
    const text = chunk.toString("utf8");
    for (const line of text.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const evt = JSON.parse(trimmed) as { event?: string; percent?: number; message?: string; path?: string };
        if (evt.event === "progress" && typeof evt.percent === "number") {
          patchJob(job.id, { percent: Math.max(0, Math.min(99, evt.percent)), message: `Downloading ${repo}…` });
        } else if (evt.event === "error" && evt.message) {
          failJob(job, evt.message);
        }
      } catch {
        const red = redactSecretText(trimmed).slice(0, 400);
        if (red) patchJob(job.id, { message: red });
      }
    }
  });
  child.stderr?.on("data", (chunk: Buffer) => {
    stderr += chunk.toString("utf8");
    const line = redactSecretText(chunk.toString("utf8")).trim();
    if (line) patchJob(job.id, { message: line.slice(0, 400) });
  });
  child.on("error", (error) => failJob(job, error.message));
  child.on("close", (code) => {
    children.delete(job.id);
    const current = jobs.get(job.id);
    if (!current || current.status === "done" || current.status === "error") return;
    const localPath = flattenDownloadedGguf(job.destDir, file);
    if (code === 0 && localPath) {
      patchJob(job.id, {
        status: "done",
        percent: 100,
        message: "Download complete.",
        localPath,
        filename: basename(localPath),
        finishedAt: Date.now(),
      });
      return;
    }
    failJob(
      current,
      redactSecretText(stderr.trim() || `GGUF download exited with code ${code ?? "unknown"}.`),
    );
  });
}

/**
 * Start (or return in-flight) GGUF download for a Hub org/model id.
 */
export async function startGgufDownload(options: {
  repo: string;
  filename?: string;
  ggufDir?: string;
  fetchFn?: HubFetchFn;
  token?: string;
}): Promise<GgufDownloadJob> {
  const repo = parseHubModelId(options.repo);
  const root = options.ggufDir ?? defaultGgufModelsDir();
  ensureSecureDir(join(stateDir(), "models"));
  ensureSecureDir(root);
  const destDir = localGgufDirForRepo(repo, root);
  mkdirSync(destDir, { recursive: true });

  const existing = jobs.get(repo);
  if (existing?.status === "running" || existing?.status === "queued") return existing;

  const token = options.token ?? resolveHfToken();
  const fetchFn = options.fetchFn ?? globalThis.fetch;
  const filename = await resolvePreferredFilename(repo, options.filename, fetchFn, token);

  const already = findLocalGgufPath(repo, filename, root) ?? findLocalGgufPathByScan(destDir, filename);
  if (already && existing?.status !== "error") {
    const done: GgufDownloadJob = {
      id: repo,
      repo,
      filename: basename(already),
      destDir,
      localPath: already,
      status: "done",
      percent: 100,
      message: "Already downloaded.",
      startedAt: existing?.startedAt ?? Date.now(),
      updatedAt: Date.now(),
      finishedAt: existing?.finishedAt ?? Date.now(),
    };
    jobs.set(repo, done);
    return done;
  }

  const now = Date.now();
  const job: GgufDownloadJob = {
    id: repo,
    repo,
    filename,
    destDir,
    status: "running",
    percent: 0,
    message: filename ? `Downloading ${repo}/${filename}…` : `Downloading GGUF from ${repo}…`,
    startedAt: now,
    updatedAt: now,
  };
  jobs.set(repo, job);
  spawnHfFileDownload(job, repo, filename);
  return job;
}

/** Reset in-memory jobs (tests). */
export function resetGgufDownloadsForTests(): void {
  for (const child of children.values()) {
    try {
      child.kill("SIGTERM");
    } catch {
      /* ignore */
    }
  }
  children.clear();
  jobs.clear();
}
