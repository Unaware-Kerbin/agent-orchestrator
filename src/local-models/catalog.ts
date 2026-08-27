import type { LaunchBackend } from "../hardware.js";

export type ModelQuantization = "fp16" | "awq" | "gptq";
export type ModelSizeClass = "small" | "medium" | "large";

export interface CatalogModel {
  id: string;
  name: string;
  hfRepo: string;
  family: string;
  paramsB: number;
  quantization: ModelQuantization;
  /** Approximate on-disk / loaded weights, not including KV cache. */
  weightsMiB: number;
  sizeClass: ModelSizeClass;
  /** Tiny enough to attempt without a GPU (still often slow / unsupported by vLLM). */
  cpuFeasible: boolean;
  gated: boolean;
  /** Passed to `vllm serve --quantization` when set. CUDA/ROCm only. */
  vllmQuantization?: "awq" | "gptq";
  /** Launch backends that can load this snapshot. Default: fp16 → all GPUs; AWQ/GPTQ → CUDA/ROCm. */
  compatibleBackends?: Exclude<LaunchBackend, "cpu">[];
  notes?: string;
}

/** AWQ/GPTQ in this catalog are CUDA/ROCm builds; vLLM XPU on Arc serves FP16 (and IPEX INT4 in custom builds). */
const CUDA_ROCM: Exclude<LaunchBackend, "cpu">[] = ["cuda", "rocm"];

export function catalogCompatibleWith(model: CatalogModel, backend: LaunchBackend): boolean {
  if (backend === "cpu") return model.cpuFeasible;
  if (model.compatibleBackends) return model.compatibleBackends.includes(backend);
  if (model.quantization === "awq" || model.quantization === "gptq") return CUDA_ROCM.includes(backend);
  return true;
}

/**
 * Curated open-weight chat models commonly served by vLLM.
 * VRAM figures are approximate serving weights (fp16 ≈ 2 bytes/param; 4-bit ≈ 0.6–0.7 bytes/param).
 */
export const LOCAL_MODEL_CATALOG: CatalogModel[] = [
  {
    id: "qwen2.5-0.5b-instruct",
    name: "Qwen2.5 0.5B Instruct",
    hfRepo: "Qwen/Qwen2.5-0.5B-Instruct",
    family: "qwen2.5-0.5b",
    paramsB: 0.5,
    quantization: "fp16",
    weightsMiB: 1_000,
    sizeClass: "small",
    cpuFeasible: true,
    gated: false,
    notes: "Tiny; CPU-feasible. Quality is limited; use only when no GPU is present.",
  },
  {
    id: "qwen2.5-1.5b-instruct",
    name: "Qwen2.5 1.5B Instruct",
    hfRepo: "Qwen/Qwen2.5-1.5B-Instruct",
    family: "qwen2.5-1.5b",
    paramsB: 1.5,
    quantization: "fp16",
    weightsMiB: 3_100,
    sizeClass: "small",
    cpuFeasible: true,
    gated: false,
    notes: "Small; marked CPU-feasible. Prefer a 7B GPU model when VRAM allows.",
  },
  {
    id: "qwen2.5-7b-instruct",
    name: "Qwen2.5 7B Instruct",
    hfRepo: "Qwen/Qwen2.5-7B-Instruct",
    family: "qwen2.5-7b",
    paramsB: 7,
    quantization: "fp16",
    weightsMiB: 14_000,
    sizeClass: "small",
    cpuFeasible: false,
    gated: false,
  },
  {
    id: "qwen2.5-7b-instruct-awq",
    name: "Qwen2.5 7B Instruct AWQ",
    hfRepo: "Qwen/Qwen2.5-7B-Instruct-AWQ",
    family: "qwen2.5-7b",
    paramsB: 7,
    quantization: "awq",
    weightsMiB: 5_000,
    sizeClass: "small",
    cpuFeasible: false,
    gated: false,
    vllmQuantization: "awq",
    compatibleBackends: CUDA_ROCM,
    notes: "AWQ: CUDA/ROCm vLLM. Not recommended on Intel XPU (use the fp16 snapshot).",
  },
  {
    id: "llama-3.1-8b-instruct",
    name: "Llama 3.1 8B Instruct",
    hfRepo: "meta-llama/Llama-3.1-8B-Instruct",
    family: "llama-3.1-8b",
    paramsB: 8,
    quantization: "fp16",
    weightsMiB: 16_000,
    sizeClass: "small",
    cpuFeasible: false,
    gated: true,
    notes: "Gated Hugging Face repo; set HF_TOKEN after accepting the license.",
  },
  {
    id: "llama-3.1-8b-instruct-awq",
    name: "Llama 3.1 8B Instruct AWQ",
    hfRepo: "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    family: "llama-3.1-8b",
    paramsB: 8,
    quantization: "awq",
    weightsMiB: 5_500,
    sizeClass: "small",
    cpuFeasible: false,
    gated: false,
    vllmQuantization: "awq",
    compatibleBackends: CUDA_ROCM,
    notes: "AWQ: CUDA/ROCm vLLM. Not recommended on Intel XPU (use the fp16 snapshot).",
  },
  {
    id: "mistral-7b-instruct",
    name: "Mistral 7B Instruct v0.3",
    hfRepo: "mistralai/Mistral-7B-Instruct-v0.3",
    family: "mistral-7b",
    paramsB: 7,
    quantization: "fp16",
    weightsMiB: 14_000,
    sizeClass: "small",
    cpuFeasible: false,
    gated: true,
    notes: "May require Hugging Face access; set HF_TOKEN if gated.",
  },
  {
    id: "mistral-7b-instruct-awq",
    name: "Mistral 7B Instruct AWQ",
    hfRepo: "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    family: "mistral-7b",
    paramsB: 7,
    quantization: "awq",
    weightsMiB: 4_500,
    sizeClass: "small",
    cpuFeasible: false,
    gated: false,
    vllmQuantization: "awq",
  },
  {
    id: "qwen2.5-14b-instruct",
    name: "Qwen2.5 14B Instruct",
    hfRepo: "Qwen/Qwen2.5-14B-Instruct",
    family: "qwen2.5-14b",
    paramsB: 14,
    quantization: "fp16",
    weightsMiB: 28_000,
    sizeClass: "medium",
    cpuFeasible: false,
    gated: false,
  },
  {
    id: "qwen2.5-14b-instruct-awq",
    name: "Qwen2.5 14B Instruct AWQ",
    hfRepo: "Qwen/Qwen2.5-14B-Instruct-AWQ",
    family: "qwen2.5-14b",
    paramsB: 14,
    quantization: "awq",
    weightsMiB: 9_000,
    sizeClass: "medium",
    cpuFeasible: false,
    gated: false,
    vllmQuantization: "awq",
  },
  {
    id: "qwen2.5-32b-instruct",
    name: "Qwen2.5 32B Instruct",
    hfRepo: "Qwen/Qwen2.5-32B-Instruct",
    family: "qwen2.5-32b",
    paramsB: 32,
    quantization: "fp16",
    weightsMiB: 64_000,
    sizeClass: "medium",
    cpuFeasible: false,
    gated: false,
  },
  {
    id: "qwen2.5-32b-instruct-awq",
    name: "Qwen2.5 32B Instruct AWQ",
    hfRepo: "Qwen/Qwen2.5-32B-Instruct-AWQ",
    family: "qwen2.5-32b",
    paramsB: 32,
    quantization: "awq",
    weightsMiB: 20_000,
    sizeClass: "medium",
    cpuFeasible: false,
    gated: false,
    vllmQuantization: "awq",
  },
  {
    id: "llama-3.1-70b-instruct",
    name: "Llama 3.1 70B Instruct",
    hfRepo: "meta-llama/Llama-3.1-70B-Instruct",
    family: "llama-3.1-70b",
    paramsB: 70,
    quantization: "fp16",
    weightsMiB: 140_000,
    sizeClass: "large",
    cpuFeasible: false,
    gated: true,
    notes: "Needs multi-GPU or 80GB+ class cards in fp16. Prefer the 4-bit variant.",
  },
  {
    id: "llama-3.1-70b-instruct-awq",
    name: "Llama 3.1 70B Instruct AWQ",
    hfRepo: "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    family: "llama-3.1-70b",
    paramsB: 70,
    quantization: "awq",
    weightsMiB: 40_000,
    sizeClass: "large",
    cpuFeasible: false,
    gated: false,
    vllmQuantization: "awq",
    notes: "Fits a single 48GB+ GPU with KV-cache headroom.",
  },
  {
    id: "qwen2.5-72b-instruct-awq",
    name: "Qwen2.5 72B Instruct AWQ",
    hfRepo: "Qwen/Qwen2.5-72B-Instruct-AWQ",
    family: "qwen2.5-72b",
    paramsB: 72,
    quantization: "awq",
    weightsMiB: 41_000,
    sizeClass: "large",
    cpuFeasible: false,
    gated: false,
    vllmQuantization: "awq",
    notes: "4-bit 72B-class; typically needs 48GB+ VRAM.",
  },
];

export function findCatalogModel(idOrRepo: string): CatalogModel | undefined {
  const needle = idOrRepo.trim();
  if (!needle) return undefined;
  return LOCAL_MODEL_CATALOG.find(
    (model) => model.id === needle || model.hfRepo === needle || model.hfRepo.toLowerCase() === needle.toLowerCase(),
  );
}

export function repoDirName(hfRepo: string): string {
  return hfRepo.replaceAll("/", "--");
}
