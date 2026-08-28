import type { EventEmitter } from "node:events";
import { existsSync, rmSync } from "node:fs";
import { join } from "node:path";
import { isPathInside, type WriteAllowlist } from "../allowlist.js";
import { detectHardware, type HardwareSnapshot } from "../hardware.js";
import { readConfigYaml, writeConfigYaml } from "../config.js";
import { refreshRuntimeEnv, hfTokenConfigured } from "../secrets.js";
import type { OrchestratorConfig } from "../types.js";
import { LOCAL_MODEL_CATALOG, findCatalogModel, isNewestHubId, type CatalogModel } from "./catalog.js";
import { DownloadManager, snapshotLooksDownloaded, type DownloadJob } from "./download.js";
import { fitModel, recommendModels, type FitKind } from "./fit.js";
import { assertModelDest, defaultModelsDir, ensureModelsDir } from "./paths.js";
import {
  VllmManager,
  resolveVllmGpuLaunchForFit,
  type VllmInstanceSelector,
  type VllmProcessRuntime,
  type VllmRuntimeState,
  type VllmStartAccepted,
} from "../vllm/manager.js";
import {
  removeVllmOrchestratorYaml,
  vllmBackendIdForModel,
  vllmSpecialistIdForModel,
} from "../vllm/upsert.js";
import type { IntelDockerCatalog } from "../vllm/intel-docker.js";

export interface LocalModelView {
  id: string;
  name: string;
  hfRepo: string;
  family: string;
  paramsB: number;
  quantization: CatalogModel["quantization"];
  sizeClass: CatalogModel["sizeClass"];
  weightsMiB: number;
  vramNeededMiB: number;
  fits: boolean;
  fitKind: FitKind;
  fitReason: string;
  newest: boolean;
  cpuFeasible: boolean;
  gated: boolean;
  downloaded: boolean;
  dest: string;
  running: boolean;
  notes?: string;
  specialist: string;
  backendId: string;
  parallel?: number;
}

export interface LocalModelsSnapshot {
  hardware: HardwareSnapshot;
  modelsDir: string;
  vllm: VllmRuntimeState;
  cloudCursor: { ready: boolean; reason?: string; needsKey: string };
  models: LocalModelView[];
  recommended: LocalModelView[];
  jobs: DownloadJob[];
  hfTokenSet: boolean;
  intelDocker: IntelDockerCatalog;
  preferredRuntime: "docker" | "host";
}

export class LocalModelService {
  readonly downloads: DownloadManager;
  readonly vllm: VllmManager;
  private hardwareCache: { at: number; value: HardwareSnapshot } | undefined;

  constructor(
    private readonly allowlist: WriteAllowlist,
    private readonly events: EventEmitter,
    private readonly configPath: string,
    private readonly getConfig: () => OrchestratorConfig,
    private readonly reloadConfig: () => void,
  ) {
    this.downloads = new DownloadManager(allowlist, events);
    this.vllm = new VllmManager(events, configPath, getConfig, reloadConfig);
  }

  hardware(): HardwareSnapshot {
    if (this.hardwareCache && Date.now() - this.hardwareCache.at < 15_000) {
      return this.hardwareCache.value;
    }
    const value = detectHardware();
    this.hardwareCache = { at: Date.now(), value };
    return value;
  }

  modelsDir(): string {
    return defaultModelsDir(this.allowlist);
  }

  snapshot(hardware = this.hardware()): LocalModelsSnapshot {
    const modelsDir = this.modelsDir();
    try {
      ensureModelsDir(this.allowlist, modelsDir);
    } catch {
      // listing still works; download/start will surface the allowlist error
    }
    const vllm = this.vllm.status();
    const intelDocker = vllm.intelDocker ?? this.vllm.dockerCatalog();
    const hardwareWithDocker: HardwareSnapshot = {
      ...hardware,
      notes: dockerHardwareNotes(hardware, intelDocker),
      intelDocker,
    };
    const cloud = this.getConfig().backends["cursor-cloud"];
    const cloudReady = Boolean(process.env.CURSOR_API_KEY?.trim()) && cloud?.type === "cursor";
    const models = LOCAL_MODEL_CATALOG.map((model) =>
      this.viewFor(model, hardwareWithDocker, modelsDir, vllm),
    );
    const recommended = recommendModels(hardwareWithDocker).map((entry) =>
      this.viewFor(entry.model, hardwareWithDocker, modelsDir, vllm),
    );
    return {
      hardware: hardwareWithDocker,
      modelsDir,
      vllm,
      cloudCursor: {
        ready: cloudReady,
        reason: cloudReady
          ? "CURSOR_API_KEY present (masked; never displayed)."
          : "Set CURSOR_API_KEY to enable Cursor cloud agents. Cloud agents cannot reach local vLLM; the orchestrator passes text between them.",
        needsKey: "CURSOR_API_KEY",
      },
      models,
      recommended,
      jobs: this.downloads.list(),
      hfTokenSet: hfTokenConfigured(),
      intelDocker,
      preferredRuntime: intelDocker.preferred && hardware.primaryBackend === "intel-xpu" ? "docker" : "host",
    };
  }

  listHardware() {
    const hardware = this.hardware();
    const intelDocker = this.vllm.dockerCatalog();
    return {
      ...hardware,
      notes: dockerHardwareNotes(hardware, intelDocker),
      intelDocker,
      preferredRuntime: intelDocker.preferred && hardware.primaryBackend === "intel-xpu" ? "docker" : "host",
    };
  }

  listModels() {
    return this.snapshot();
  }

  recommend() {
    const snap = this.snapshot();
    return {
      hardware: snap.hardware,
      recommendations: snap.recommended,
      constrained: snap.hardware.constrained,
    };
  }

  download(input: { modelId?: string; hfRepo?: string; dest?: string; dryRun?: boolean }): DownloadJob {
    refreshRuntimeEnv();
    return this.downloads.start(input);
  }

  async startVllm(input: {
    modelId?: string;
    hfRepo?: string;
    port?: number;
    quantization?: string;
    host?: string;
    timeoutMs?: number;
    image?: string;
    runtime?: VllmProcessRuntime;
    replace?: boolean;
    useAllGpus?: boolean;
  }): Promise<VllmRuntimeState> {
    const result = await this.vllm.start(this.prepareVllmStart(input));
    this.events.emit("local-models", this.snapshot());
    return result;
  }

  startVllmAsync(input: {
    modelId?: string;
    hfRepo?: string;
    port?: number;
    quantization?: string;
    host?: string;
    timeoutMs?: number;
    image?: string;
    runtime?: VllmProcessRuntime;
    replace?: boolean;
    useAllGpus?: boolean;
  }): VllmStartAccepted {
    const accepted = this.vllm.beginStart(this.prepareVllmStart(input));
    this.events.emit("local-models", this.snapshot());
    return accepted;
  }

  private prepareVllmStart(input: {
    modelId?: string;
    hfRepo?: string;
    port?: number;
    quantization?: string;
    host?: string;
    timeoutMs?: number;
    image?: string;
    runtime?: VllmProcessRuntime;
    replace?: boolean;
    useAllGpus?: boolean;
  }) {
    const model = this.resolve(input.modelId, input.hfRepo);
    const modelsDir = ensureModelsDir(this.allowlist, this.modelsDir());
    const dest = assertModelDest(this.allowlist, modelsDir, model.hfRepo);
    if (!snapshotLooksDownloaded(dest)) {
      throw new Error(
        `Model "${model.id}" is not downloaded at ${dest}. Call download_local_model first (large weights are never fetched implicitly).`,
      );
    }
    const hardware = this.hardware();
    const fit = fitModel(model, hardware);
    const gpu = resolveVllmGpuLaunchForFit({
      useAllGpus: input.useAllGpus,
      deviceCount: hardware.deviceCount || hardware.accelerators.length || 1,
      fitKind: fit.kind,
      fitParallel: fit.parallel,
    });
    const quant =
      hardware.primaryBackend === "intel-xpu"
        ? undefined
        : (input.quantization ??
          (model.vllmQuantization && model.vllmQuantization !== undefined ? model.vllmQuantization : undefined));
    return {
      modelId: model.id,
      hfRepo: model.hfRepo,
      modelPath: dest,
      quantization: quant,
      port: input.port,
      host: input.host,
      timeoutMs: input.timeoutMs,
      backend: hardware.primaryBackend,
      tensorParallel: gpu.tensorParallel,
      deviceCount: gpu.deviceCount,
      image: input.image,
      runtime: input.runtime,
      replace: input.replace,
      backendId: vllmBackendIdForModel(model.id),
      specialistId: vllmSpecialistIdForModel(model.id),
    };
  }

  stopVllm(selector: VllmInstanceSelector = {}): VllmRuntimeState {
    const result = this.vllm.stop(selector);
    this.events.emit("local-models", this.snapshot());
    return result;
  }

  removeVllm(selector: VllmInstanceSelector): VllmRuntimeState {
    const model = selector.modelId ? this.resolve(selector.modelId) : undefined;
    const instance = this.vllm.findInstance(selector);
    const backendIds = new Set<string>();
    if (selector.backendId) backendIds.add(selector.backendId);
    if (instance?.backendId) backendIds.add(instance.backendId);
    if (model) backendIds.add(vllmBackendIdForModel(model.id));
    if (backendIds.size === 0) {
      throw new Error("model_id or backend_id is required to remove a vLLM backend");
    }
    const result = this.vllm.stop({
      modelId: instance?.modelId ?? model?.id ?? selector.modelId,
      backendId: instance?.backendId ?? selector.backendId,
    });
    let yaml = readConfigYaml(this.configPath);
    for (const backendId of backendIds) {
      yaml = removeVllmOrchestratorYaml(yaml, { backendId });
    }
    writeConfigYaml(yaml, this.configPath);
    this.reloadConfig();
    this.events.emit("local-models", this.snapshot());
    return result;
  }

  deleteLocalModel(input: { modelId: string; confirm?: boolean }): { ok: true; dest: string; deleted: boolean } {
    if (input.confirm !== true) {
      throw new Error("delete_local_model requires confirm=true (this permanently deletes weights).");
    }
    const model = this.resolve(input.modelId);
    const modelsDir = this.modelsDir();
    const dest = assertModelDest(this.allowlist, modelsDir, model.hfRepo);
    if (dest === modelsDir || !isPathInside(dest, modelsDir)) {
      throw new Error("Refusing to delete weights outside the allowlisted models directory");
    }
    this.vllm.stop({ modelId: model.id, backendId: vllmBackendIdForModel(model.id) });
    const existed = existsSync(dest);
    if (existed) rmSync(dest, { recursive: true, force: true });
    this.events.emit("local-models", this.snapshot());
    return { ok: true, dest, deleted: existed };
  }

  vllmStatus(): VllmRuntimeState {
    return this.vllm.status();
  }

  catalogSummary() {
    const snap = this.snapshot();
    return {
      hardware: {
        gpuNames: snap.hardware.gpus.map((gpu) => gpu.name),
        vramMiB: snap.hardware.vramMiB,
        ramMiB: snap.hardware.ramMiB,
        cpuCount: snap.hardware.cpuCount,
        constrained: snap.hardware.constrained,
        primaryBackend: snap.hardware.primaryBackend,
        totalVramMiB: snap.hardware.totalVramMiB,
        deviceCount: snap.hardware.deviceCount,
      },
      vllm: {
        running: Boolean(snap.vllm.healthy || snap.vllm.instances.some((row) => row.healthy || row.running)),
        installed: snap.vllm.installed,
        port: snap.vllm.port,
        modelId: snap.vllm.modelId,
        host: snap.vllm.host,
        installHint: snap.vllm.installed ? undefined : snap.vllm.installHint,
        runtime: snap.vllm.runtime,
        image: snap.vllm.image,
        containerId: snap.vllm.containerId,
        containerName: snap.vllm.containerName,
        backendId: snap.vllm.backendId,
        phase: snap.vllm.phase,
        healthy: snap.vllm.healthy,
        jobId: snap.vllm.jobId,
        lastError: snap.vllm.lastError,
        instances: (snap.vllm.instances ?? []).map((row) => ({
          backendId: row.backendId,
          specialistId: row.specialistId,
          modelId: row.modelId,
          hfRepo: row.hfRepo,
          port: row.port,
          host: row.host,
          runtime: row.runtime,
          image: row.image,
          containerId: row.containerId,
          containerName: row.containerName,
          phase: row.phase,
          healthy: row.healthy,
          running: row.running,
        })),
      },
      intelDocker: {
        available: snap.intelDocker.available,
        daemon: snap.intelDocker.daemon,
        preferred: snap.intelDocker.preferred?.ref,
        images: snap.intelDocker.images.map((row) => ({
          ref: row.ref,
          tag: row.tag,
          id: row.id,
          size: row.size,
        })),
      },
      preferredRuntime: snap.preferredRuntime,
      cloudCursor: snap.cloudCursor,
      modelsDir: snap.modelsDir,
      downloadedCount: snap.models.filter((model) => model.downloaded).length,
      recommended: snap.recommended.map((model) => ({
        id: model.id,
        name: model.name,
        quantization: model.quantization,
        fits: model.fits,
        fitKind: model.fitKind,
        newest: model.newest,
        parallel: model.parallel,
      })),
    };
  }

  private resolve(modelId?: string, hfRepo?: string): CatalogModel {
    const needle = modelId?.trim() || hfRepo?.trim();
    if (!needle) throw new Error("model_id or hf_repo is required");
    const known = findCatalogModel(needle);
    if (known) return known;
    if (!needle.includes("/")) {
      throw new Error(`Unknown model "${needle}". See list_local_models for catalog ids.`);
    }
    return {
      id: needle.replaceAll("/", "--").toLowerCase(),
      name: needle,
      hfRepo: needle,
      family: needle,
      lineage: "custom",
      generation: 0,
      paramsB: 0,
      quantization: "fp16",
      weightsMiB: 0,
      sizeClass: "small",
      cpuFeasible: false,
      gated: false,
    };
  }

  private viewFor(
    model: CatalogModel,
    hardware: HardwareSnapshot,
    modelsDir: string,
    vllm: VllmRuntimeState,
  ): LocalModelView {
    const fit = fitModel(model, hardware);
    const dest = join(modelsDir, model.hfRepo.replaceAll("/", "--"));
    const instance = (vllm.instances ?? []).find(
      (row) => row.modelId === model.id || row.hfRepo === model.hfRepo,
    );
    const backendId = instance?.backendId ?? vllmBackendIdForModel(model.id);
    return {
      id: model.id,
      name: model.name,
      hfRepo: model.hfRepo,
      family: model.family,
      paramsB: model.paramsB,
      quantization: model.quantization,
      sizeClass: model.sizeClass,
      weightsMiB: model.weightsMiB,
      vramNeededMiB: fit.vramNeededMiB,
      fits: fit.fits,
      fitKind: fit.kind,
      fitReason: fit.reason,
      newest: isNewestHubId(model),
      cpuFeasible: model.cpuFeasible,
      gated: model.gated,
      downloaded: snapshotLooksDownloaded(dest),
      dest,
      running: Boolean(instance?.healthy || instance?.running),
      notes: model.notes,
      specialist: instance?.specialistId ?? vllmSpecialistIdForModel(model.id),
      backendId,
      parallel: fit.parallel,
    };
  }
}

function dockerHardwareNotes(hardware: HardwareSnapshot, docker: IntelDockerCatalog): string[] {
  const notes = hardware.notes.filter((note) => !note.startsWith("Intel vLLM Docker"));
  if (docker.preferred) {
    notes.push(
      `Intel vLLM Docker: ${docker.images.map((row) => `${row.ref} (${row.size || row.id})`).join(", ")}. Preferred: ${docker.preferred.ref}. Default start path on intel-xpu is Docker.`,
    );
  } else if (hardware.primaryBackend === "intel-xpu") {
    notes.push(`Intel vLLM Docker: ${docker.error ?? "no matching local images."}`);
  }
  return notes;
}
