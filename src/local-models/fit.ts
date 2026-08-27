import type { HardwareSnapshot, LaunchBackend } from "../hardware.js";
import type { CatalogModel, ModelQuantization } from "./catalog.js";
import { LOCAL_MODEL_CATALOG, catalogCompatibleWith } from "./catalog.js";

/** Extra VRAM for KV cache and runtime overhead on top of weight estimates. */
export const KV_CACHE_HEADROOM = 0.2;

export interface FitResult {
  fits: boolean;
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
      reason: `No discrete accelerator detected; ${model.name} needs ~${needed} MiB VRAM (weights + ${headroomPct}% KV headroom).`,
      weightsMiB: model.weightsMiB,
      vramNeededMiB: needed,
      vramAvailableMiB: 0,
    };
  }

  if (!catalogCompatibleWith(model, backend)) {
    return {
      fits: false,
      reason: `${backendLabel(backend)} cannot load this ${model.quantization.toUpperCase()} snapshot (CUDA-oriented kernels). Use the fp16 catalog variant on Intel Arc/XPU.`,
      weightsMiB: model.weightsMiB,
      vramNeededMiB: needed,
      vramAvailableMiB: perGpu,
    };
  }

  if (needed <= perGpu) {
    const extra =
      devices > 1
        ? ` Dual-GPU tensor parallel is optional (${devices}× ${backendLabel(backend)}).`
        : "";
    return {
      fits: true,
      reason: `Fits on one ${backendLabel(backend)} GPU: ~${needed} MiB needed (weights ${model.weightsMiB} MiB + ${headroomPct}% KV) ≤ ${perGpu} MiB.${extra}`,
      weightsMiB: model.weightsMiB,
      vramNeededMiB: needed,
      vramAvailableMiB: perGpu,
      parallel: 1,
    };
  }

  const tpBudget = minGpu * devices;
  if (devices > 1 && needed <= tpBudget) {
    return {
      fits: true,
      reason: `Fits with tensor parallel ${devices}: ~${needed} MiB needed (weights ${model.weightsMiB} MiB + ${headroomPct}% KV) ≤ ${devices}× ${minGpu} MiB ${backendLabel(backend)} (${tpBudget} MiB). Does not fit on a single GPU (${perGpu} MiB).`,
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
    reason: `Does not fit: ~${needed} MiB needed (weights ${model.weightsMiB} MiB + ${headroomPct}% KV) > ${perGpu} MiB per GPU` +
      (devices > 1 ? ` and > ${tpBudget} MiB across ${devices} GPUs.` : ".") +
      preferQuant,
    weightsMiB: model.weightsMiB,
    vramNeededMiB: needed,
    vramAvailableMiB: perGpu,
  };
}

const QUANT_RANK: Record<ModelQuantization, number> = { fp16: 0, awq: 1, gptq: 2 };

export function recommendModels(
  hardware: HardwareSnapshot,
  catalog: readonly CatalogModel[] = LOCAL_MODEL_CATALOG,
  limit = 5,
): RankedModel[] {
  const backend = hardware.primaryBackend ?? (hardware.gpus.length > 0 ? "cuda" : "cpu");
  const hasGpu = backend !== "cpu" && primaryDeviceCount(hardware) > 0;
  const fitting = catalog
    .map((model) => ({ model, fit: fitModel(model, hardware) }))
    .filter((entry) => entry.fit.fits)
    .filter((entry) => catalogCompatibleWith(entry.model, backend))
    .filter((entry) => (hasGpu ? !entry.model.cpuFeasible : true));

  const bestByFamily = new Map<string, RankedModel>();
  for (const entry of fitting) {
    const current = bestByFamily.get(entry.model.family);
    if (!current) {
      bestByFamily.set(entry.model.family, entry);
      continue;
    }
    const betterPrecision =
      QUANT_RANK[entry.model.quantization] < QUANT_RANK[current.model.quantization];
    if (entry.model.paramsB === current.model.paramsB && betterPrecision) {
      bestByFamily.set(entry.model.family, entry);
    }
  }

  return [...bestByFamily.values()]
    .sort((a, b) => {
      if (b.model.paramsB !== a.model.paramsB) return b.model.paramsB - a.model.paramsB;
      const aPar = a.fit.parallel ?? 1;
      const bPar = b.fit.parallel ?? 1;
      if (aPar !== bPar) return aPar - bPar;
      return QUANT_RANK[a.model.quantization] - QUANT_RANK[b.model.quantization];
    })
    .slice(0, limit);
}
