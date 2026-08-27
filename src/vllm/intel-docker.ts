import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { basename, dirname } from "node:path";

/** Host publish address only. Inside the container vLLM still listens on 0.0.0.0:8000. */
const LOOPBACK_HOST = "127.0.0.1";

export const LLM_SCALER_REPO = "intel/llm-scaler-vllm";
export const INTEL_VLLM_REPO = "intel/vllm";
/** Documented Battlemage / Arc Pro path (llm-scaler). */
export const PREFERRED_LLM_SCALER_TAG = "0.21.0-b3";
export const PREFERRED_INTEL_VLLM_TAG = "0.17.0-xpu";
export const PREFERRED_LLM_SCALER_REF = `${LLM_SCALER_REPO}:${PREFERRED_LLM_SCALER_TAG}`;
export const PREFERRED_INTEL_VLLM_REF = `${INTEL_VLLM_REPO}:${PREFERRED_INTEL_VLLM_TAG}`;
export const CONTAINER_MODEL_ROOT = "/llm/models";
export const CONTAINER_API_PORT = 8000;
export const INTEL_VLLM_CONTAINER_NAME = "orch-vllm";
export const INTEL_VLLM_IMAGE_ENV = "AGENT_ORCHESTRATOR_VLLM_IMAGE";

export const EXPECTED_INTEL_IMAGE_REFS = [
  PREFERRED_LLM_SCALER_REF,
  `${LLM_SCALER_REPO}:<any-tag>`,
  PREFERRED_INTEL_VLLM_REF,
] as const;

export type DockerDaemonState = "ok" | "down" | "permission" | "missing";
export type IntelImageKind = "llm-scaler-passthrough" | "llm-scaler-override" | "intel-vllm";

export interface IntelDockerImage {
  repository: string;
  tag: string;
  id: string;
  size: string;
  ref: string;
}

export interface IntelDockerCatalog {
  available: boolean;
  daemon: DockerDaemonState;
  images: IntelDockerImage[];
  preferred?: IntelDockerImage;
  error?: string;
}

export interface DockerExecResult {
  status: number | null;
  stdout: string;
  stderr: string;
  error?: string;
}

export type DockerExec = (command: string, args: string[]) => DockerExecResult;

export interface IntelDockerRunPlan {
  command: "docker";
  args: string[];
  containerName: string;
  containerModelPath: string;
  publish: string;
  image: string;
  kind: IntelImageKind;
  /** Args after `vllm serve` (positional model first). Never includes `--device xpu`. */
  vllmArgs: string[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

export function parseImageRef(ref: string): { repository: string; tag: string } {
  const trimmed = ref.trim();
  const lastSlash = trimmed.lastIndexOf("/");
  const lastColon = trimmed.lastIndexOf(":");
  if (lastColon > lastSlash && lastColon >= 0) {
    return { repository: trimmed.slice(0, lastColon), tag: trimmed.slice(lastColon + 1) };
  }
  return { repository: trimmed, tag: "latest" };
}

export function isIntelVllmImage(repository: string, tag: string): boolean {
  if (tag === "<none>" || !tag) return false;
  if (repository === LLM_SCALER_REPO) return true;
  if (repository === INTEL_VLLM_REPO && /xpu/i.test(tag)) return true;
  return false;
}

/** Compare Intel-style tags like 0.21.0-b3 vs 0.14.0-b8.3.2 (numeric segments, then leftover). */
export function compareIntelTags(a: string, b: string): number {
  const nums = (tag: string) =>
    tag
      .split(/[^0-9]+/)
      .filter(Boolean)
      .map((part) => Number(part));
  const left = nums(a);
  const right = nums(b);
  const n = Math.max(left.length, right.length);
  for (let i = 0; i < n; i++) {
    const lv = left[i] ?? 0;
    const rv = right[i] ?? 0;
    if (lv !== rv) return lv - rv;
  }
  return a.localeCompare(b);
}

export function parseDockerImagesJson(text: string): IntelDockerImage[] {
  const images: IntelDockerImage[] = [];
  const seen = new Set<string>();
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      continue;
    }
    if (!isRecord(parsed)) continue;
    const repository = typeof parsed.Repository === "string" ? parsed.Repository : "";
    const tag = typeof parsed.Tag === "string" ? parsed.Tag : "";
    if (!isIntelVllmImage(repository, tag)) continue;
    const idRaw = typeof parsed.ID === "string" ? parsed.ID : typeof parsed.Id === "string" ? parsed.Id : "";
    const id = idRaw.replace(/^sha256:/, "").slice(0, 12);
    const size = typeof parsed.Size === "string" ? parsed.Size : "";
    const ref = `${repository}:${tag}`;
    if (seen.has(ref)) continue;
    seen.add(ref);
    images.push({ repository, tag, id, size, ref });
  }
  return images;
}

export function selectPreferredIntelImage(
  images: IntelDockerImage[],
  override?: string,
): IntelDockerImage | undefined {
  if (images.length === 0 && !override) return undefined;
  const requested = override?.trim();
  if (requested) {
    const exact = images.find((row) => row.ref === requested || row.id === requested);
    if (exact) return exact;
    const { repository, tag } = parseImageRef(requested);
    return { repository, tag, id: "", size: "", ref: `${repository}:${tag}` };
  }
  const scaler = images.filter((row) => row.repository === LLM_SCALER_REPO);
  const pinned = scaler.find((row) => row.tag === PREFERRED_LLM_SCALER_TAG);
  if (pinned) return pinned;
  if (scaler.length > 0) {
    return [...scaler].sort((a, b) => compareIntelTags(b.tag, a.tag))[0];
  }
  const xpu = images.filter((row) => row.repository === INTEL_VLLM_REPO && /xpu/i.test(row.tag));
  const pinnedXpu = xpu.find((row) => row.tag === PREFERRED_INTEL_VLLM_TAG);
  if (pinnedXpu) return pinnedXpu;
  if (xpu.length > 0) {
    return [...xpu].sort((a, b) => compareIntelTags(b.tag, a.tag))[0];
  }
  return undefined;
}

export function classifyDockerError(stderr: string, status: number | null, error?: string): {
  daemon: DockerDaemonState;
  message: string;
} {
  const text = `${error ?? ""} ${stderr}`.toLowerCase();
  if (/enoent|not found|no such file/.test(text) && /docker/.test(text)) {
    return {
      daemon: "missing",
      message: "docker is not installed or not on PATH. Install Docker Engine and retry.",
    };
  }
  if (/permission denied|dial unix.*permission|got permission denied while trying to connect/.test(text)) {
    return {
      daemon: "permission",
      message:
        "docker images failed: permission denied on the Docker socket. Add this user to the docker group (`sudo usermod -aG docker $USER`), then log out and back in.",
    };
  }
  if (
    /cannot connect|connect: no such file|daemon is not running|is the docker daemon running|docker.sock/.test(text)
  ) {
    return {
      daemon: "down",
      message:
        "Docker daemon is not reachable (unix:///var/run/docker.sock). Start Docker, and confirm this user is in the docker group.",
    };
  }
  if (status !== 0 && status !== null) {
    return {
      daemon: "down",
      message: stderr.trim() || error || `docker images exited with code ${status}`,
    };
  }
  return { daemon: "ok", message: "" };
}

function defaultExec(command: string, args: string[]): DockerExecResult {
  try {
    const result = spawnSync(command, args, { encoding: "utf8", timeout: 15_000 });
    return {
      status: result.status,
      stdout: result.stdout ?? "",
      stderr: result.stderr ?? "",
      error: result.error?.message,
    };
  } catch (error) {
    return {
      status: 1,
      stdout: "",
      stderr: error instanceof Error ? error.message : String(error),
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export function namedMissingImagesHint(found: IntelDockerImage[]): string {
  const have = new Set(found.map((row) => row.ref));
  const missing = [PREFERRED_LLM_SCALER_REF, PREFERRED_INTEL_VLLM_REF].filter((ref) => !have.has(ref));
  const extra =
    found.length === 0
      ? `No local tags matched ${LLM_SCALER_REPO} or ${INTEL_VLLM_REPO}:*-xpu.`
      : `Seen locally: ${found.map((row) => row.ref).join(", ")}.`;
  const pulls = missing.map((ref) => `  docker pull ${ref}`).join("\n");
  return `${extra}${pulls ? `\nPull to add the documented Arc images:\n${pulls}` : ""}`;
}

export function intelDockerInstallHint(catalog?: IntelDockerCatalog): string {
  const expected = `Preferred order: ${PREFERRED_LLM_SCALER_REF} (Arc Pro / Battlemage), then other ${LLM_SCALER_REPO} tags, then ${PREFERRED_INTEL_VLLM_REF}. Override with start_vllm image=… or ${INTEL_VLLM_IMAGE_ENV}.`;
  if (!catalog || catalog.daemon === "missing") {
    return `Intel Arc/XPU: stock \`pip install vllm\` is CUDA-only. The supported path is Intel's Docker images.\n${expected}\n${catalog?.error ?? "docker is not installed."}`;
  }
  if (catalog.daemon === "permission" || catalog.daemon === "down") {
    return `Intel Arc/XPU: stock \`pip install vllm\` is CUDA-only. Docker images are the supported Arc path when present (${PREFERRED_LLM_SCALER_REF}, ${PREFERRED_INTEL_VLLM_REF}).\n${catalog.error ?? "Could not list docker images."}\n${expected}`;
  }
  if (!catalog.preferred) {
    return `Intel Arc/XPU: Docker is running but no Intel vLLM images were found. ${namedMissingImagesHint(catalog.images)}\n${expected}`;
  }
  return `Intel Docker runtime available: ${catalog.preferred.ref}. ${expected}`;
}

export function listIntelVllmImages(options?: {
  exec?: DockerExec;
  preferredImage?: string;
}): IntelDockerCatalog {
  const exec = options?.exec ?? defaultExec;
  const override = options?.preferredImage ?? process.env[INTEL_VLLM_IMAGE_ENV]?.trim();
  const listed = exec("docker", ["images", "--format", "{{json .}}"]);
  if (listed.error && /enoent|not found/i.test(listed.error) && listed.status !== 0) {
    const classified = classifyDockerError(listed.stderr, listed.status, listed.error);
    return {
      available: false,
      daemon: classified.daemon,
      images: [],
      error: classified.message,
    };
  }
  if (listed.status !== 0) {
    const classified = classifyDockerError(listed.stderr, listed.status, listed.error);
    return {
      available: false,
      daemon: classified.daemon,
      images: [],
      error: classified.message,
    };
  }
  const images = parseDockerImagesJson(listed.stdout);
  const preferred = selectPreferredIntelImage(images, override);
  return {
    available: Boolean(preferred),
    daemon: "ok",
    images,
    preferred,
    error: preferred ? undefined : namedMissingImagesHint(images),
  };
}

export function classifyIntelImage(ref: string, entrypoint?: string[] | null): IntelImageKind {
  const ep = (entrypoint ?? []).join(" ");
  if (/exec vllm serve "\$@"|vllm serve "\$@"/.test(ep)) return "llm-scaler-passthrough";
  if (entrypoint?.length === 2 && entrypoint[0] === "vllm" && entrypoint[1] === "serve") {
    return "intel-vllm";
  }
  const { repository, tag } = parseImageRef(ref);
  if (repository === LLM_SCALER_REPO) {
    if (tag === PREFERRED_LLM_SCALER_TAG || /^0\.21\b/.test(tag)) return "llm-scaler-passthrough";
    return "llm-scaler-override";
  }
  return "intel-vllm";
}

export function lookupGroupGid(group: string, exec: DockerExec = defaultExec): string | undefined {
  const result = exec("getent", ["group", group]);
  if (result.status !== 0) return undefined;
  const gid = result.stdout.trim().split(":")[2];
  return gid && /^\d+$/.test(gid) ? gid : undefined;
}

function containerModelPathFor(modelsDir: string, modelPath: string): string {
  const dir = modelsDir.replace(/\/+$/, "");
  const path = modelPath.replace(/\/+$/, "");
  if (path === dir) return CONTAINER_MODEL_ROOT;
  if (path.startsWith(`${dir}/`)) {
    return `${CONTAINER_MODEL_ROOT}/${path.slice(dir.length + 1)}`;
  }
  return `${CONTAINER_MODEL_ROOT}/${basename(path)}`;
}

/**
 * Intel XPU images (llm-scaler and intel/vllm:*-xpu) imply the device.
 * Their CLI rejects `--device xpu` and wants the model as a positional arg
 * (`vllm serve <model> …`), matching Local GPU Terminal Emulator compose.
 */
export function buildIntelVllmArgs(input: {
  containerModelPath: string;
  servedModelName: string;
  tensorParallel?: number;
}): string[] {
  const tp = input.tensorParallel && input.tensorParallel > 1 ? input.tensorParallel : 1;
  return [
    input.containerModelPath,
    "--served-model-name",
    input.servedModelName,
    "--host",
    "0.0.0.0",
    "--port",
    String(CONTAINER_API_PORT),
    "--tensor-parallel-size",
    String(tp),
    "--dtype",
    "float16",
    "--enforce-eager",
    "--trust-remote-code",
    "--gpu-memory-utilization",
    "0.9",
  ];
}

/** True when argv contains the vLLM flag pair `--device xpu` (not Docker `--device /dev/dri`). */
export function hasVllmDeviceXpuFlag(args: string[]): boolean {
  for (let i = 0; i < args.length - 1; i++) {
    if (args[i] === "--device" && args[i + 1] === "xpu") return true;
  }
  return false;
}

export function formatDockerCommand(args: string[]): string {
  const quote = (part: string) => (/[\s"'\\]/.test(part) ? `'${part.replaceAll("'", `'\\''`)}'` : part);
  return ["docker", ...args].map(quote).join(" ");
}

export function buildIntelDockerRunArgs(input: {
  image: string;
  hostPort: number;
  modelsDir: string;
  modelPath: string;
  servedModelName: string;
  tensorParallel?: number;
  deviceCount?: number;
  entrypoint?: string[] | null;
  containerName?: string;
  driDir?: string;
  dxgPath?: string;
  renderGid?: string;
  videoGid?: string;
  shmSize?: string;
}): IntelDockerRunPlan {
  const hostPort = input.hostPort;
  const publish = `${LOOPBACK_HOST}:${hostPort}:${CONTAINER_API_PORT}`;
  if (!publish.startsWith(`${LOOPBACK_HOST}:`)) {
    throw new Error("Refusing to publish vLLM on a non-loopback address");
  }
  const containerName = input.containerName ?? INTEL_VLLM_CONTAINER_NAME;
  const modelsDir = input.modelsDir;
  const containerModelPath = containerModelPathFor(modelsDir, input.modelPath);
  const kind = classifyIntelImage(input.image, input.entrypoint);
  const vllmArgs = buildIntelVllmArgs({
    containerModelPath,
    servedModelName: input.servedModelName,
    tensorParallel: input.tensorParallel,
  });
  const driDir = input.driDir ?? "/dev/dri";
  const dxgPath = input.dxgPath ?? "/dev/dxg";
  const args: string[] = [
    "run",
    "-d",
    "--name",
    containerName,
    "--privileged",
    "--pull",
    "never",
    "--publish",
    publish,
    "--ipc=host",
    "--shm-size",
    input.shmSize ?? "32g",
    // Match Local GPU Terminal Emulator compose: do not inject host ZE_AFFINITY_MASK
    // (the image enumerates discrete XPUs as 0,1). XPU is implied by the image.
    "-e",
    "ZE_FLAT_DEVICE_HIERARCHY=FLAT",
    "-e",
    "VLLM_WORKER_MULTIPROC_METHOD=spawn",
    "-e",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
    "-e",
    "VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1",
    "-e",
    "CCL_ZE_IPC_EXCHANGE=sockets",
    "-e",
    "CCL_ZE_CACHE_OPEN_IPC_HANDLES=0",
    "-e",
    "CCL_ATL_TRANSPORT=ofi",
    "-e",
    "SYCL_CACHE_PERSISTENT=0",
    "-e",
    "SYCL_CACHE_DIR=/tmp/empty-sycl-cache",
    "-e",
    `HF_HOME=${CONTAINER_MODEL_ROOT}`,
    "-v",
    `${modelsDir}:${CONTAINER_MODEL_ROOT}`,
  ];
  if (existsSync(driDir) || input.driDir) {
    args.push("--device", driDir);
    const byPath = `${driDir}/by-path`;
    if (existsSync(byPath) || input.driDir) {
      args.push("-v", `${byPath}:${byPath}`);
    }
  } else {
    args.push("--device", "/dev/dri");
  }
  if (existsSync(dxgPath)) {
    args.push("--device", dxgPath);
  }
  const gids: string[] = [];
  for (const gid of [input.videoGid, input.renderGid]) {
    if (gid && !gids.includes(gid)) gids.push(gid);
  }
  for (const gid of gids) args.push("--group-add", gid);

  if (kind === "llm-scaler-override") {
    args.push("--entrypoint", "bash");
  }

  args.push(input.image);

  if (kind === "llm-scaler-passthrough") {
    // Image ENTRYPOINT is already `bash -c 'exec vllm serve "$@"' --`
    args.push(...vllmArgs);
  } else if (kind === "llm-scaler-override") {
    // Older llm-scaler ENTRYPOINT is `bash -c 'vllm serve'` and drops extra args.
    // Do not source oneAPI setvars.sh (missing / crash-looping on current tags).
    args.push("-lc", 'exec vllm serve "$@"', "--", ...vllmArgs);
  } else {
    args.push("vllm", "serve", ...vllmArgs);
  }

  if (args.some((part) => part === "0.0.0.0" && args[args.indexOf(part) - 1] === "--publish")) {
    throw new Error("Refusing to publish vLLM on 0.0.0.0");
  }
  const publishIdx = args.indexOf("--publish");
  const publishVal = publishIdx >= 0 ? args[publishIdx + 1] : "";
  if (publishVal?.startsWith("0.0.0.0:") || publishVal === "0.0.0.0") {
    throw new Error("Refusing to publish vLLM on 0.0.0.0");
  }

  if (hasVllmDeviceXpuFlag(vllmArgs) || hasVllmDeviceXpuFlag(args.slice(args.lastIndexOf(input.image) + 1))) {
    throw new Error("Refusing to pass --device xpu to Intel vLLM images (CLI rejects it)");
  }

  return {
    command: "docker",
    args,
    containerName,
    containerModelPath,
    publish,
    image: input.image,
    kind,
    vllmArgs,
  };
}

export function dockerContainerRunning(containerId: string, exec: DockerExec = defaultExec): boolean {
  const result = exec("docker", ["inspect", "-f", "{{.State.Running}}", containerId]);
  return result.status === 0 && result.stdout.trim() === "true";
}

export function dockerContainerPid(containerId: string, exec: DockerExec = defaultExec): number | undefined {
  const result = exec("docker", ["inspect", "-f", "{{.State.Pid}}", containerId]);
  if (result.status !== 0) return undefined;
  const pid = Number(result.stdout.trim());
  return Number.isInteger(pid) && pid > 0 ? pid : undefined;
}

export function dockerLogs(containerId: string, exec: DockerExec = defaultExec, tail = 200): string {
  const result = exec("docker", ["logs", "--tail", String(tail), containerId]);
  return `${result.stdout}\n${result.stderr}`.trim();
}

export function dockerStopContainer(containerId: string, exec: DockerExec = defaultExec): void {
  exec("docker", ["stop", "-t", "8", containerId]);
  exec("docker", ["rm", "-f", containerId]);
}

export function dockerRmName(name: string, exec: DockerExec = defaultExec): void {
  exec("docker", ["rm", "-f", name]);
}

export function dockerRunDetached(args: string[], exec?: DockerExec): string {
  const run =
    exec ??
    ((command: string, runArgs: string[]) => {
      try {
        const result = spawnSync(command, runArgs, { encoding: "utf8", timeout: 60_000 });
        return {
          status: result.status,
          stdout: result.stdout ?? "",
          stderr: result.stderr ?? "",
          error: result.error?.message,
        };
      } catch (error) {
        return {
          status: 1,
          stdout: "",
          stderr: error instanceof Error ? error.message : String(error),
          error: error instanceof Error ? error.message : String(error),
        };
      }
    });
  const result = run("docker", args);
  if (result.status !== 0) {
    const detail = (result.stderr || result.stdout || result.error || "docker run failed").trim();
    throw new Error(detail);
  }
  const id = result.stdout.trim().split(/\s+/).pop() ?? "";
  if (!id) throw new Error("docker run did not return a container id");
  return id;
}

export function resolveDockerGroups(exec: DockerExec = defaultExec): { renderGid?: string; videoGid?: string } {
  return {
    renderGid: lookupGroupGid("render", exec),
    videoGid: lookupGroupGid("video", exec),
  };
}

export function modelsDirFromPath(modelPath: string): string {
  return dirname(modelPath);
}
