/**
 * Vendor-agnostic Hub serve filter: only list snapshots the idle GPU on this
 * computer can actually compile/serve. Vendor comes from gpu-pick (not a SKU).
 *
 * Intel / OpenVINO CausalLM: Optimum `text-generation-with-past` architectures
 * from the installed stack `~/.local/share/late/intel-ov` (see
 * optimum.exporters.openvino.model_configs). gemma4 is image-text-to-text only —
 * export for text-generation-with-past fails. CUDA blobs and CPU are not that card.
 *
 * NVIDIA: Candle (qwen2 / gemma4) plus the MLC compile path.
 * AMD: HIP serve is not wired — fail closed, do not list as Startable.
 */

import type { AcceleratorVendor } from "../hardware.js";

export type HubServeVendor = AcceleratorVendor | "cpu" | "";

export type HubRepoAccess = "ok" | "denied" | "unknown";

const HF_HUB_ORIGIN = "https://huggingface.co";
const PROBE_TIMEOUT_MS = 5_000;

/**
 * Optimum OpenVINO CausalLM `text-generation-with-past` model_type values.
 * Captured from the installed intel-ov TasksManager (not a B70 special case).
 * Visual-only types (gemma3, gemma4, qwen3_5, …) are intentionally absent.
 */
export const OPENVINO_CAUSAL_MODEL_TYPES: ReadonlySet<string> = new Set([
  "afmoe",
  "aquila",
  "arcee",
  "arctic",
  "baichuan",
  "biogpt",
  "bitnet",
  "bloom",
  "chatglm",
  "codegen",
  "cohere",
  "cohere2",
  "dbrx",
  "deci",
  "deepseek",
  "deepseek_v2",
  "deepseek_v3",
  "exaone",
  "exaone4",
  "falcon",
  "falcon_mamba",
  "gemma",
  "gemma2",
  "gemma3_text",
  "gemma3n_text",
  "gemma4_text",
  "gemma4_unified_text",
  "glm",
  "glm4",
  "gpt2",
  "gpt_bigcode",
  "gpt_neo",
  "gpt_neox",
  "gpt_neox_japanese",
  "gpt_oss",
  "gptj",
  "granite",
  "granitemoe",
  "granitemoehybrid",
  "hunyuan_v1_dense",
  "internlm",
  "internlm2",
  "jais",
  "lfm2",
  "lfm2_moe",
  "llama",
  "llama4",
  "llama4_text",
  "mamba",
  "minicpm",
  "minicpm3",
  "mistral",
  "mixtral",
  "mpt",
  "olmo",
  "olmo2",
  "opt",
  "orion",
  "persimmon",
  "phi",
  "phi3",
  "phimoe",
  "qwen",
  "qwen2",
  "qwen2_moe",
  "qwen3",
  "qwen3_5_moe_text",
  "qwen3_5_text",
  "qwen3_moe",
  "qwen3_next",
  "qwen3_omni_moe_talker_text",
  "qwen3_omni_moe_text",
  "qwen3_vl_text",
  "smollm3",
  "stablelm",
  "starcoder2",
  "xglm",
  "xverse",
  "zamba2",
]);

/** Visual-only in Optimum OV exporter — CausalLM text-generation-with-past fails. */
export const OPENVINO_VISUAL_ONLY_TYPES: ReadonlySet<string> = new Set([
  "gemma3",
  "gemma3n",
  "gemma4",
  "gemma4_unified",
  "qwen2_5_vl",
  "qwen2_vl",
  "qwen3_5",
  "qwen3_5_moe",
  "qwen3_omni_moe",
  "qwen3_vl",
  "phi3_v",
  "phi4_multimodal",
  "phi4mm",
]);

/** Candle in-process serve on NVIDIA (late-infer compiler.rs). */
export const NVIDIA_CANDLE_MODEL_TYPES: ReadonlySet<string> = new Set(["qwen", "qwen2", "gemma4"]);

/** MLC-LLM compile path on NVIDIA. */
export const NVIDIA_MLC_MODEL_TYPES: ReadonlySet<string> = new Set([
  "llama",
  "mistral",
  "mixtral",
  "qwen",
  "qwen2",
  "qwen3",
  "gemma",
  "gemma2",
  "gemma3",
  "phi",
  "phi3",
  "gpt2",
  "gpt_neox",
  "stablelm",
  "internlm",
  "baichuan",
  "chatglm",
  "deepseek",
  "olmo",
  "mpt",
  "falcon",
]);

export interface HubServeProbe {
  id: string;
  modelType?: string;
  tags?: readonly string[];
}

export interface HubServeDecision {
  ok: boolean;
  reason: string;
}

export function normalizeHubModelType(raw: string | undefined): string {
  return String(raw ?? "")
    .trim()
    .toLowerCase()
    .replace(/-/g, "_");
}

/** Infer transformers `model_type` from a Hub id when config is missing. */
export function inferHubModelType(id: string, modelType?: string): string {
  const fromConfig = normalizeHubModelType(modelType);
  if (fromConfig) return fromConfig;
  const name = id.includes("/") ? (id.split("/")[1] ?? id) : id;
  const blob = `${id} ${name}`.toLowerCase();
  if (/gemma-?4|gemma4/.test(blob)) return "gemma4";
  if (/gemma-?3n|gemma3n/.test(blob)) return "gemma3n";
  if (/gemma-?3|gemma3/.test(blob)) return "gemma3";
  if (/gemma-?2|gemma2/.test(blob)) return "gemma2";
  if (/gemma/.test(blob)) return "gemma";
  if (/qwen3\.5|qwen3_5|qwen3-5/.test(blob)) return "qwen3_5";
  if (/qwen3/.test(blob)) return "qwen3";
  if (/qwen2/.test(blob) || /qwen2\.5/.test(blob)) return "qwen2";
  if (/qwen/.test(blob)) return "qwen";
  if (/llama/.test(blob)) return "llama";
  if (/mixtral/.test(blob)) return "mixtral";
  if (/mistral|ministral/.test(blob)) return "mistral";
  if (/phi/.test(blob)) return "phi3";
  return "";
}

/** CUDA-only NVFP4 / NVIDIA FP4 checkpoints cannot load on Intel OpenVINO. */
export function isCudaOnlyHubQuant(id: string, tags: readonly string[] = []): boolean {
  const blob = `${id} ${tags.join(" ")}`.toLowerCase();
  return /(?:^|[-_/.])nvfp4(?:[-_/.]|$)|nv-fp4|nvidia-fp4|\bfp4[-_]?quant/.test(blob);
}

export function nvidiaCanServeModelType(modelType: string): boolean {
  const t = normalizeHubModelType(modelType);
  if (!t) return false;
  return NVIDIA_CANDLE_MODEL_TYPES.has(t) || NVIDIA_MLC_MODEL_TYPES.has(t);
}

export function openvinoCanExportCausalLm(modelType: string): boolean {
  const t = normalizeHubModelType(modelType);
  if (!t) return false;
  if (OPENVINO_VISUAL_ONLY_TYPES.has(t)) return false;
  return OPENVINO_CAUSAL_MODEL_TYPES.has(t);
}

export function hubServeDecision(
  vendor: HubServeVendor | undefined,
  probe: HubServeProbe,
  runtimeOk = true,
): HubServeDecision {
  const id = String(probe.id ?? "").trim();
  const modelType = inferHubModelType(id, probe.modelType);
  const tags = probe.tags ?? [];

  if (!vendor || vendor === "cpu") {
    return {
      ok: false,
      reason: "No discrete GPU on your computer — the Hub store lists snapshots for the idle GPU, not CPU.",
    };
  }

  if (vendor === "amd") {
    return {
      ok: false,
      reason:
        "late-infer cannot serve on the AMD GPU on your computer yet (HIP is not wired). The Hub store will not list Instruct ids as Startable on that card.",
    };
  }

  if (!runtimeOk) {
    if (vendor === "intel") {
      return {
        ok: false,
        reason:
          "OpenVINO GenAI is not on your computer. The Hub store will not list snapshots the idle Intel GPU cannot compile.",
      };
    }
    return {
      ok: false,
      reason: "late-infer cannot use this GPU on your computer. The Hub store will not list snapshots that cannot run on that card.",
    };
  }

  if (isCudaOnlyHubQuant(id, tags)) {
    if (vendor !== "nvidia") {
      return {
        ok: false,
        reason: "This snapshot is CUDA-only NVFP4 — not the idle GPU on your computer.",
      };
    }
  }

  if (vendor === "intel") {
    if (!modelType) {
      return {
        ok: false,
        reason: "Unknown architecture — Optimum OpenVINO CausalLM will not export it on the Intel GPU on your computer.",
      };
    }
    if (!openvinoCanExportCausalLm(modelType)) {
      return {
        ok: false,
        reason: `Optimum OpenVINO cannot export model_type=${modelType} for CausalLM (text-generation-with-past) on the Intel GPU on your computer. A CUDA blob is not that card.`,
      };
    }
    return { ok: true, reason: "" };
  }

  if (vendor === "nvidia") {
    if (!modelType) {
      return {
        ok: false,
        reason: "Unknown architecture — late-infer Candle/MLC will not compile it on the NVIDIA GPU on your computer.",
      };
    }
    if (!nvidiaCanServeModelType(modelType)) {
      return {
        ok: false,
        reason: `late-infer cannot serve model_type=${modelType} on the NVIDIA GPU on your computer.`,
      };
    }
    return { ok: true, reason: "" };
  }

  return { ok: false, reason: "Unknown GPU vendor on your computer." };
}

export function hubServeBlockedReason(
  vendor: HubServeVendor | undefined,
  probe: HubServeProbe,
  runtimeOk = true,
): string | undefined {
  const decision = hubServeDecision(vendor, probe, runtimeOk);
  return decision.ok ? undefined : decision.reason;
}

export function hubConfigResolveUrl(id: string): URL | undefined {
  const trimmed = String(id ?? "").trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/.test(trimmed)) return undefined;
  return new URL(`${HF_HUB_ORIGIN}/${trimmed}/resolve/main/config.json`);
}

/**
 * Cheap license/access probe before a multi-GB Hub fetch.
 * HEAD (then GET) config.json — never the safetensors shards.
 */
export async function probeHubRepoAccess(options: {
  id: string;
  fetchFn: (input: string | URL, init?: RequestInit) => Promise<Response>;
  token?: string;
}): Promise<HubRepoAccess> {
  const url = hubConfigResolveUrl(options.id);
  if (!url) return "unknown";
  const headers: Record<string, string> = {
    accept: "application/json, application/octet-stream, */*",
    "user-agent": "agent-orchestrator",
  };
  if (options.token) headers.authorization = `Bearer ${options.token}`;
  const tryMethod = async (method: "HEAD" | "GET"): Promise<HubRepoAccess> => {
    let res: Response;
    try {
      res = await options.fetchFn(url, {
        method,
        headers,
        redirect: "follow",
        signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
      });
    } catch {
      return "unknown";
    }
    if (res.status === 401 || res.status === 403) return "denied";
    if (res.status === 404) return "unknown";
    if (!res.ok) return "unknown";
    const contentType = res.headers.get("content-type") ?? "";
    if (contentType.includes("text/html")) return "unknown";
    if (contentType.includes("application/json") && method === "GET") {
      try {
        const body: unknown = await res.clone().json();
        if (Array.isArray(body)) return "unknown";
      } catch {
        /* config.json may be empty on HEAD mocks */
      }
    }
    return "ok";
  };
  const head = await tryMethod("HEAD");
  if (head !== "unknown") return head;
  return tryMethod("GET");
}
