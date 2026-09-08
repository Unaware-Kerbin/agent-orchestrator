/**
 * Per-backend eligibility for Hub catalog rows.
 */
import {
  inferHubModelType,
  openvinoCanExportCausalLm,
  nvidiaCanServeModelType,
  OPENVINO_VISUAL_ONLY_TYPES,
  normalizeHubModelType,
} from "./hub-serve.js";

export type HubBackendId = "lateinfer" | "vllm" | "ollama" | "llamacpp";

export interface HubBackendSupport {
  lateinfer: boolean;
  lateinferOpenVinoCausalLm: boolean;
  lateinferNvidia: boolean;
  vllm: boolean;
  ollama: boolean;
  llamacpp: boolean;
  notes?: string[];
}

function ggufFriendly(modelType: string, id: string): boolean {
  const t = normalizeHubModelType(modelType);
  const blob = id.toLowerCase();
  return (
    t.startsWith("qwen") ||
    t.startsWith("gemma") ||
    t.startsWith("llama") ||
    t.startsWith("mistral") ||
    t.startsWith("mixtral") ||
    t.startsWith("phi") ||
    /qwen|gemma|llama|mistral|phi/.test(blob)
  );
}

/**
 * Declare which local backends can run this Hub id.
 * Gemma3/4 (and other OV visual-only types): lateinfer OpenVINO CausalLM = false;
 * vllm/ollama/llamacpp stay true for text Instruct serving on those stacks.
 * nvidia/amd eligibility is recorded via lateinferNvidia + notes even when this box is Intel-only.
 */
export function hubBackendSupportFor(id: string, modelType?: string): HubBackendSupport {
  const t = inferHubModelType(id, modelType);
  const visualOnly = OPENVINO_VISUAL_ONLY_TYPES.has(t);
  const ovCausal = openvinoCanExportCausalLm(t);
  const nvidia = nvidiaCanServeModelType(t);
  const notes: string[] = [];

  if (visualOnly || (/^(gemma3|gemma3n|gemma4)/.test(t) && !ovCausal)) {
    notes.push("lateinfer OpenVINO: not CausalLM text-generation-with-past (visual/multimodal graph)");
  }
  if (/gemma-4-(12B|31B|26B)/i.test(id) || t === "gemma4_unified" || t.startsWith("gemma4_unified")) {
    notes.push(
      "vLLM Intel XPU: online FP8 + --limit-mm-per-prompt image/video/audio=0 (text chat). Prefer google/gemma-4-E2B-it when possible.",
    );
  }
  if (!nvidia && !ovCausal) {
    notes.push("lateinfer: no Intel OV CausalLM and no NVIDIA Candle/MLC type");
  }
  if (nvidia) {
    notes.push("lateinfer NVIDIA Candle/MLC eligible (not live-tested on this Intel box)");
  }

  return {
    lateinfer: ovCausal, // primary = OpenVINO CausalLM; see lateinferNvidia for NVIDIA Candle/MLC
    lateinferOpenVinoCausalLm: ovCausal,
    lateinferNvidia: nvidia,
    vllm: true,
    ollama: ggufFriendly(t, id),
    llamacpp: ggufFriendly(t, id),
    notes: notes.length ? notes : undefined,
  };
}

export function attachHubBackends<T extends { id: string; modelType?: string }>(
  model: T,
): T & { backends: HubBackendSupport } {
  return { ...model, backends: hubBackendSupportFor(model.id, model.modelType) };
}
