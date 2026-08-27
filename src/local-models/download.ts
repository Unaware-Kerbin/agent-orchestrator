import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { accessSync, constants, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { delimiter, dirname, join } from "node:path";
import type { EventEmitter } from "node:events";
import type { WriteAllowlist } from "../allowlist.js";
import { packageRoot } from "../config.js";
import { stateDir } from "../state.js";
import type { CatalogModel } from "./catalog.js";
import { findCatalogModel } from "./catalog.js";
import { assertModelDest, defaultModelsDir, ensureModelsDir } from "./paths.js";

export type DownloadStatus = "queued" | "running" | "done" | "error";

export interface DownloadJob {
  id: string;
  modelId: string;
  hfRepo: string;
  dest: string;
  status: DownloadStatus;
  percent: number;
  message: string;
  error?: string;
  startedAt: number;
  updatedAt: number;
  finishedAt?: number;
}

interface JobsFile {
  version: 1;
  jobs: Record<string, DownloadJob>;
}

const JOBS_FILE = "download-jobs.json";

function jobsPath(): string {
  return join(stateDir(), JOBS_FILE);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function loadJobs(): Record<string, DownloadJob> {
  const path = jobsPath();
  if (!existsSync(path)) return {};
  try {
    const raw = JSON.parse(readFileSync(path, "utf8")) as unknown;
    if (!isRecord(raw) || !isRecord(raw.jobs)) return {};
    const jobs: Record<string, DownloadJob> = {};
    for (const [id, value] of Object.entries(raw.jobs)) {
      if (!isRecord(value) || typeof value.modelId !== "string") continue;
      const status = value.status;
      const interrupted = status === "running" || status === "queued";
      jobs[id] = {
        id: typeof value.id === "string" ? value.id : id,
        modelId: value.modelId,
        hfRepo: typeof value.hfRepo === "string" ? value.hfRepo : value.modelId,
        dest: typeof value.dest === "string" ? value.dest : "",
        status: interrupted ? "error" : ((status as DownloadStatus) || "error"),
        percent: typeof value.percent === "number" ? value.percent : 0,
        message: interrupted
          ? "Download interrupted (orchestrator restarted)."
          : typeof value.message === "string"
            ? value.message
            : "",
        error: interrupted
          ? "Download interrupted (orchestrator restarted)."
          : typeof value.error === "string"
            ? value.error
            : undefined,
        startedAt: typeof value.startedAt === "number" ? value.startedAt : Date.now(),
        updatedAt: Date.now(),
        finishedAt: typeof value.finishedAt === "number" ? value.finishedAt : undefined,
      };
    }
    return jobs;
  } catch {
    return {};
  }
}

function persistJobs(jobs: Record<string, DownloadJob>): void {
  mkdirSync(dirname(jobsPath()), { recursive: true, mode: 0o700 });
  const payload: JobsFile = { version: 1, jobs };
  writeFileSync(jobsPath(), `${JSON.stringify(payload, null, 2)}\n`, {
    encoding: "utf8",
    mode: 0o600,
  });
}

export function snapshotLooksDownloaded(dest: string): boolean {
  if (!dest || !existsSync(dest)) return false;
  const markers = ["config.json", "params.json", "tokenizer.json", "tokenizer_config.json"];
  for (const name of markers) {
    if (existsSync(join(dest, name))) return true;
  }
  try {
    return readdirSync(dest).some(
      (name) => name.endsWith(".safetensors") || name.endsWith(".bin") || name.endsWith(".gguf"),
    );
  } catch {
    return false;
  }
}

export function redactSecretText(text: string): string {
  let out = text;
  for (const name of ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]) {
    const value = process.env[name]?.trim();
    if (value && value.length > 3) out = out.split(value).join("***");
  }
  return out;
}

export function hfTokenPresent(): boolean {
  return Boolean(process.env.HF_TOKEN?.trim() || process.env.HUGGING_FACE_HUB_TOKEN?.trim());
}

function helperScript(): string {
  return join(packageRoot(), "scripts", "hf_download.py");
}

function which(cmd: string): string | undefined {
  const pathEnv = process.env.PATH ?? "";
  for (const dir of pathEnv.split(delimiter)) {
    const candidate = join(dir, cmd);
    try {
      accessSync(candidate, constants.X_OK);
      return candidate;
    } catch {
      // try next
    }
  }
  return undefined;
}

function findPython(): string | undefined {
  for (const cmd of ["python3", "python"]) {
    const result = spawnSync(cmd, ["-c", "import sys; print(sys.executable)"], {
      encoding: "utf8",
      timeout: 5000,
    });
    if (result.status === 0 && result.stdout?.trim()) return cmd;
  }
  return which("python3") ?? which("python");
}

export class DownloadManager {
  private jobs: Record<string, DownloadJob>;
  private readonly children = new Map<string, ChildProcess>();

  constructor(
    private readonly allowlist: WriteAllowlist,
    private readonly events: EventEmitter,
  ) {
    this.jobs = loadJobs();
    persistJobs(this.jobs);
  }

  list(): DownloadJob[] {
    return Object.values(this.jobs).sort((a, b) => b.updatedAt - a.updatedAt);
  }

  get(modelId: string): DownloadJob | undefined {
    return this.jobs[modelId];
  }

  emit(): void {
    this.events.emit("local-models", { jobs: this.list() });
  }

  start(input: {
    modelId?: string;
    hfRepo?: string;
    dest?: string;
    dryRun?: boolean;
  }): DownloadJob {
    const model = resolveModel(input.modelId, input.hfRepo);
    const modelsDir = ensureModelsDir(this.allowlist, defaultModelsDir(this.allowlist));
    const dest = assertModelDest(this.allowlist, modelsDir, model.hfRepo, input.dest);

    if (input.dryRun) {
      const now = Date.now();
      return {
        id: model.id,
        modelId: model.id,
        hfRepo: model.hfRepo,
        dest,
        status: "queued",
        percent: 0,
        message: `Dry run: would download ${model.hfRepo} into ${dest}`,
        startedAt: now,
        updatedAt: now,
      };
    }

    const existing = this.jobs[model.id];
    if (existing?.status === "running" || existing?.status === "queued") {
      return existing;
    }
    if (snapshotLooksDownloaded(dest) && existing?.status !== "error") {
      const done: DownloadJob = {
        id: model.id,
        modelId: model.id,
        hfRepo: model.hfRepo,
        dest,
        status: "done",
        percent: 100,
        message: "Already downloaded.",
        startedAt: existing?.startedAt ?? Date.now(),
        updatedAt: Date.now(),
        finishedAt: existing?.finishedAt ?? Date.now(),
      };
      this.jobs[model.id] = done;
      persistJobs(this.jobs);
      this.emit();
      return done;
    }

    const now = Date.now();
    const job: DownloadJob = {
      id: model.id,
      modelId: model.id,
      hfRepo: model.hfRepo,
      dest,
      status: "running",
      percent: 0,
      message: `Downloading ${model.hfRepo}…`,
      startedAt: now,
      updatedAt: now,
    };
    this.jobs[model.id] = job;
    persistJobs(this.jobs);
    this.emit();
    this.spawnDownload(job, model);
    return job;
  }

  private spawnDownload(job: DownloadJob, model: CatalogModel): void {
    const script = helperScript();
    const python = findPython();
    const env = { ...process.env, PYTHONUNBUFFERED: "1" };
    let child: ChildProcess;

    if (python && existsSync(script)) {
      child = spawn(python, [script, "--repo", model.hfRepo, "--dest", job.dest], {
        env,
        stdio: ["ignore", "pipe", "pipe"],
      });
    } else {
      const cli = which("hf") ?? which("huggingface-cli");
      if (!cli) {
        this.fail(
          job,
          [
            python ? `Download helper missing at ${script}.` : "python3 not found on PATH.",
            "Install Hugging Face tools: pip install huggingface_hub",
            "Then retry download_local_model.",
          ].join(" "),
        );
        return;
      }
      const args = ["download", model.hfRepo, "--local-dir", job.dest];
      child = spawn(cli, args, { env, stdio: ["ignore", "pipe", "pipe"] });
    }

    this.children.set(job.id, child);
    let stderr = "";
    child.stdout?.on("data", (chunk: Buffer) => {
      this.onStdout(job, chunk.toString("utf8"));
    });
    child.stderr?.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
      const line = redactSecretText(chunk.toString("utf8")).trim();
      if (line) this.patch(job.id, { message: line.slice(0, 400) });
    });
    child.on("error", (error) => {
      this.fail(job, redactSecretText(error.message));
    });
    child.on("close", (code) => {
      this.children.delete(job.id);
      const current = this.jobs[job.id];
      if (!current || current.status === "done" || current.status === "error") return;
      if (code === 0 && snapshotLooksDownloaded(job.dest)) {
        this.patch(job.id, {
          status: "done",
          percent: 100,
          message: "Download complete.",
          finishedAt: Date.now(),
        });
        return;
      }
      const hint = !hfTokenPresent() && model.gated
        ? " Repo is gated: set HF_TOKEN or HUGGING_FACE_HUB_TOKEN."
        : "";
      this.fail(
        current,
        redactSecretText((stderr.trim() || `Download exited with code ${code ?? "unknown"}.`) + hint),
      );
    });
  }

  private onStdout(job: DownloadJob, chunk: string): void {
    for (const line of chunk.split("\n")) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const payload = JSON.parse(trimmed) as Record<string, unknown>;
        if (payload.event === "progress") {
          const percent =
            typeof payload.percent === "number" ? Math.max(0, Math.min(100, payload.percent)) : job.percent;
          this.patch(job.id, { percent, message: `Downloading… ${percent}%` });
        } else if (payload.event === "done") {
          this.patch(job.id, {
            status: "done",
            percent: 100,
            message: "Download complete.",
            dest: typeof payload.path === "string" ? payload.path : job.dest,
            finishedAt: Date.now(),
          });
        } else if (payload.event === "error") {
          const message = typeof payload.message === "string" ? payload.message : "Download failed";
          this.fail(job, redactSecretText(message));
        }
      } catch {
        this.patch(job.id, { message: redactSecretText(trimmed).slice(0, 400) });
      }
    }
  }

  private fail(job: DownloadJob, error: string): void {
    this.patch(job.id, {
      status: "error",
      error,
      message: error,
      finishedAt: Date.now(),
    });
  }

  private patch(id: string, update: Partial<DownloadJob>): void {
    const current = this.jobs[id];
    if (!current) return;
    this.jobs[id] = { ...current, ...update, id, updatedAt: Date.now() };
    persistJobs(this.jobs);
    this.emit();
  }
}

export function resolveDownloadModel(modelId?: string, hfRepo?: string): CatalogModel {
  const needle = modelId?.trim() || hfRepo?.trim();
  if (!needle) throw new Error("model_id or hf_repo is required");
  const known = findCatalogModel(needle);
  if (known) return known;
  if (!needle.includes("/") && !hfRepo?.includes("/")) {
    throw new Error(
      `Unknown model "${needle}". Use list_local_models for catalog ids, or pass an org/name Hugging Face repo.`,
    );
  }
  const repo = needle.includes("/") ? needle : hfRepo!;
  return {
    id: repo.replaceAll("/", "--").toLowerCase(),
    name: repo,
    hfRepo: repo,
    family: repo,
    paramsB: 0,
    quantization: "fp16",
    weightsMiB: 0,
    sizeClass: "small",
    cpuFeasible: false,
    gated: false,
  };
}

function resolveModel(modelId?: string, hfRepo?: string): CatalogModel {
  return resolveDownloadModel(modelId, hfRepo);
}
