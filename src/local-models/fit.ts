import type { HardwareSnapshot, LaunchBackend } from "../hardware.js";
import type { CatalogModel, ModelQuantization } from "./catalog.js";
import { LOCAL_MODEL_CATALOG, catalogCompatibleWith, isNewestHubId } from "./catalog.js";

/** Extra VRAM for KV cache and runtime overhead on top of weight estimates. */
export const KV_CACHE_HEADROOM = 0.2;

/**
 * Fraction of each GPU that vLLM is started with (`--gpu-memory-utilization`).
 * Tensor-parallel fit uses the same budget so a model is labeled as fitting only if
 * weight shards fit in the memory vLLM will actually claim per card.
 */
export const GPU_MEMORY_UTILIZATION = 0.9;

export type FitKind = "fits" | "needs_tp" | "too_big" | "incompatible";

export interface FitResult {
  fits: boolean;
  kind: FitKind;
  reason: string;
  weightsMiB: number;
  vramNeededMiB: number;
  vramAvailableMiB: number;
  /** 1 = single GPU; >1 = tensor parallel across primary-backend devices. */
  parallel?: number;
}

export interface RankedModel {
  model: CatalogModel;
  fit: FitResult;
  newest: boolean;
}

export function classifyFit(fit: FitResult): FitKind {
  return fit.kind;
}

export function vramNeededMiB(weightsMiB: number, headroom = KV_CACHE_HEADROOM): number {
  if (!Number.isFinite(weightsMiB) || weightsMiB < 0) {
    throw new Error("weightsMiB must be a non-negative finite number");
  }
  return Math.ceil(weightsMiB * (1 + headroom));
}

export function modelFitsVram(
  weightsMiB: number,
  vramAvailableMiB: number,
  headroom = KV_CACHE_HEADROOM,
): boolean {
  if (!Number.isFinite(vramAvailableMiB) || vramAvailableMiB < 0) return false;
  return vramNeededMiB(weightsMiB, headroom) <= vramAvailableMiB;
}

function backendLabel(backend: LaunchBackend): string {
  if (backend === "intel-xpu") return "Intel XPU";
  if (backend === "cuda") return "CUDA";
  if (backend === "rocm") return "ROCm";
  return "CPU";
}

function primaryDeviceCount(hardware: HardwareSnapshot): number {
  if (hardware.deviceCount > 0) return hardware.deviceCount;
  if (hardware.primaryBackend === "cpu") return 0;
  return hardware.accelerators?.filter((row) => {
    if (hardware.primaryBackend === "cuda") return row.vendor === "nvidia";
    if (hardware.primaryBackend === "intel-xpu") return row.vendor === "intel";
    if (hardware.primaryBackend === "rocm") return row.vendor === "amd";
    return false;
  }).length ?? hardware.gpus.length;
}

function minPrimaryVram(hardware: HardwareSnapshot): number {
  if (hardware.minVramMiB > 0) return hardware.minVramMiB;
  return hardware.vramMiB;
}

/** Combined VRAM for even tensor-parallel shards (limited by the smallest card). */
export function tensorParallelBudgetMiB(minVramMiB: number, deviceCount: number): number {
  if (!Number.isFinite(minVramMiB) || !Number.isFinite(deviceCount)) return 0;
  return Math.max(0, minVramMiB) * Math.max(0, deviceCount);
}

/**
 * True when weight shards fit across `deviceCount` GPUs at the same utilization
 * vLLM will use. KV cache then takes leftover VRAM on each card.
 */
export function tensorParallelShardFits(
  weightsMiB: number,
  minVramMiB: number,
  deviceCount: number,
  utilization = GPU_MEMORY_UTILIZATION,
): boolean {
  if (deviceCount <= 1 || minVramMiB <= 0 || weightsMiB <= 0) return false;
  if (weightsMiB > tensorParallelBudgetMiB(minVramMiB, deviceCount)) return false;
  const shard = Math.ceil(weightsMiB / deviceCount);
  const perGpuBudget = Math.floor(minVramMiB * utilization);
  return shard <= perGpuBudget;
}

export function fitModel(
  model: CatalogModel,
  hardware: HardwareSnapshot,
  headroom = KV_CACHE_HEADROOM,
): FitResult {
  const needed = vramNeededMiB(model.weightsMiB, headroom);
  const backend = hardware.primaryBackend ?? (hardware.gpus.length > 0 ? "cuda" : "cpu");
  const perGpu = hardware.vramMiB;
  const devices = primaryDeviceCount(hardware);
  const minGpu = minPrimaryVram(hardware);
  const headroomPct = Math.round(headroom * 100);

  if (backend === "cpu" || devices <= 0 || perGpu <= 0) {
    if (model.cpuFeasible) {
      return {
        fits: true,
        kind: "fits",
        reason:
          "No discrete accelerator detected; this tiny model is marked CPU-feasible (constrained, local serving may still fail).",
        weightsMiB: model.weightsMiB,
        vramNeededMiB: needed,
        vramAvailableMiB: 0,
        parallel: 1,
      };
    }
    return {
      fits: false,
      kind: "too_big",
      reason: `No discrete accelerator detected; ${model.name} needs ~${needed} MiB VRAM (weights + ${headroomPct}% KV headroom).`,
      weightsMiB: model.weightsMiB,
      vramNeededMiB: needed,
      vramAvailableMiB: 0,
    };
  }

  if (!catalogCompatibleWith(model, backend)) {
    return {
      fits: false,
      kind: "incompatible",
      reason: `${backendLabel(backend)} cannot load this ${model.quantization.toUpperCase()} snapshot (CUDA-oriented kernels). Use the fp16 catalog variant on Intel Arc/XPU.`,
      weightsMiB: model.weightsMiB,
      vramNeededMiB: needed,
      vramAvailableMiB: perGpu,
    };
  }

  if (needed <= perGpu) {
    const extra =
      devices > 1
        ? ` Tensor parallel across ${devices}× ${backendLabel(backend)} is optional (model already fits one card).`
        : "";
    return {
      fits: true,
      kind: "fits",
      reason: `Fits on one ${backendLabel(backend)} GPU: ~${needed} MiB needed (weights ${model.weightsMiB} MiB + ${headroomPct}% KV) ≤ ${perGpu} MiB.${extra}`,
      weightsMiB: model.weightsMiB,
      vramNeededMiB: needed,
      vramAvailableMiB: perGpu,
      parallel: 1,
    };
  }

  const tpBudget = tensorParallelBudgetMiB(minGpu, devices);
  const shard = devices > 1 ? Math.ceil(model.weightsMiB / devices) : model.weightsMiB;
  const utilPct = Math.round(GPU_MEMORY_UTILIZATION * 100);
  if (devices > 1 && needed <= tpBudget) {
    return {
      fits: true,
      kind: "needs_tp",
      reason: `Fits with tensor parallel ${devices}: ~${needed} MiB needed (weights ${model.weightsMiB} MiB + ${headroomPct}% KV) ≤ ${devices}× ${minGpu} MiB ${backendLabel(backend)} (${tpBudget} MiB combined). Does not fit on a single GPU (${perGpu} MiB). Needs all GPUs on this computer.`,
      weightsMiB: model.weightsMiB,
      vramNeededMiB: needed,
      vramAvailableMiB: tpBudget,
      parallel: devices,
    };
  }
  if (devices > 1 && tensorParallelShardFits(model.weightsMiB, minGpu, devices)) {
    return {
      fits: true,
      kind: "needs_tp",
      reason: `Fits with tensor parallel ${devices}: weights ${model.weightsMiB} MiB shard to ~${shard} MiB per GPU ≤ ${utilPct}% of ${minGpu} MiB ${backendLabel(backend)} (combined ${tpBudget} MiB). KV cache uses leftover VRAM. Does not fit on a single GPU (${perGpu} MiB). Needs all GPUs on this computer.`,
      weightsMiB: model.weightsMiB,
      vramNeededMiB: needed,
      vramAvailableMiB: tpBudget,
      parallel: devices,
    };
  }

  const preferQuant =
    backend === "intel-xpu"
      ? " On Intel XPU prefer a smaller fp16 model (AWQ/GPTQ CUDA quants will not load on Arc)."
      : model.quantization === "fp16"
        ? " Prefer a 4-bit (AWQ/GPTQ) variant of this family if one is in the catalog."
        : " Try a smaller model.";
  return {
    fits: false,
    kind: "too_big",
    reason: `Does not fit: ~${needed} MiB needed (weights ${model.weightsMiB} MiB + ${headroomPct}% KV) > ${perGpu} MiB per GPU` +
      (devices > 1 ? ` and weight shards do not fit in ${utilPct}% of ${devices}× ${minGpu} MiB (${tpBudget} MiB combined).` : ".") +
      preferQuant,
    weightsMiB: model.weightsMiB,
    vramNeededMiB: needed,
    vramAvailableMiB: devices > 1 ? tpBudget : perGpu,
  };
}

const QUANT_RANK: Record<ModelQuantization, number> = { fp16: 0, awq: 1, gptq: 2 };

const FIT_KIND_RANK: Record<FitKind, number> = {
  fits: 0,
  needs_tp: 1,
  too_big: 2,
  incompatible: 3,
};

/**
 * Every catalog row for this computer. Fit is labeled (fits / needs TP / too big /
 * incompatible); nothing is hidden. `newest` marks the latest Hub id in each size class
 * (Gemma 4 over Gemma 2/3, Qwen3.8 over Qwen2.5). There is no default top-N cap.
 */
export function recommendModels(
  hardware: HardwareSnapshot,
  catalog: readonly CatalogModel[] = LOCAL_MODEL_CATALOG,
  limit?: number,
): RankedModel[] {
  const backend = hardware.primaryBackend ?? (hardware.gpus.length > 0 ? "cuda" : "cpu");
  const hasGpu = backend !== "cpu" && primaryDeviceCount(hardware) > 0;
  const ranked = catalog
    .map((model) => ({
      model,
      fit: fitModel(model, hardware),
      newest: isNewestHubId(model, catalog),
    }))
    .sort((a, b) => {
      const kind = FIT_KIND_RANK[a.fit.kind] - FIT_KIND_RANK[b.fit.kind];
      if (kind !== 0) return kind;
      if (hasGpu) {
        const cpu = Number(a.model.cpuFeasible) - Number(b.model.cpuFeasible);
        if (cpu !== 0) return cpu;
      }
      if (a.newest !== b.newest) return a.newest ? -1 : 1;
      if (b.model.paramsB !== a.model.paramsB) return b.model.paramsB - a.model.paramsB;
      const aPar = a.fit.parallel ?? 1;
      const bPar = b.fit.parallel ?? 1;
      if (aPar !== bPar) return aPar - bPar;
      return QUANT_RANK[a.model.quantization] - QUANT_RANK[b.model.quantization];
    });
  return limit != null && Number.isFinite(limit) && limit >= 0 ? ranked.slice(0, limit) : ranked;
}
