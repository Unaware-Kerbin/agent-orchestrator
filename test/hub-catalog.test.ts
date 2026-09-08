import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { createServer } from "node:net";
import { afterEach, test } from "node:test";
import type { HardwareSnapshot } from "../src/hardware.js";
import { startGuiServer } from "../src/gui/http.js";
import {
  HF_HUB_MODELS_API,
  HUB_FAMILY_SEARCHES,
  HUB_FIT_FRACTION,
  FALLBACK_HUB_SEEDS,
  buildHubModelsUrl,
  catalogSearchTerms,
  compareHubModels,
  estimateMaxVramMiB,
  formatVramMaxLabel,
  fallbackHubCatalogModels,
  groupHubModels,
  hardwareFitBudgetMiB,
  hubFamily,
  isHubInstructCandidate,
  listHubModels,
  applyHubLoadability,
  mergeHubCatalog,
  parseHubParamsB,
  parseHubVersion,
  resetHubCatalogForTests,
  stampHubCatalogLoadability,
  toHubCatalogModel,
  withVramEstimate,
  type HubCatalogModel,
  type HubRawModel,
} from "../src/local-servers/hub-catalog.js";
import { openvinoCanExportCausalLm } from "../src/local-servers/hub-serve.js";

afterEach(() => {
  resetHubCatalogForTests();
});

function fakeHardware(opts: { vramMiB?: number; ramMiB?: number }): HardwareSnapshot {
  const vram = opts.vramMiB ?? 0;
  return {
    accelerators: vram
      ? [{ vendor: "nvidia", name: "Fake GPU", vramMiB: vram, index: 0, source: "test" }]
      : [],
    primaryBackend: vram ? "cuda" : "cpu",
    totalVramMiB: vram,
    deviceCount: vram ? 1 : 0,
    gpus: vram ? [{ index: 0, name: "Fake GPU", vramMiB: vram }] : [],
    vramMiB: vram,
    minVramMiB: vram,
    ramMiB: opts.ramMiB ?? 16_000,
    cpuCount: 8,
    hasNvidiaSmi: Boolean(vram),
    constrained: !vram,
    notes: [],
  };
}

function fakeIntelHardware(opts: { vramMiB?: number; ramMiB?: number }): HardwareSnapshot {
  const vram = opts.vramMiB ?? 0;
  return {
    accelerators: [
      { vendor: "intel", name: "Intel Arc Pro B70", vramMiB: vram, index: 0, source: "sysfs" },
      { vendor: "intel", name: "Intel Arc Pro B70", vramMiB: vram, index: 1, source: "sysfs" },
    ],
    primaryBackend: "intel-xpu",
    totalVramMiB: vram > 0 ? vram * 2 : 0,
    deviceCount: 2,
    gpus: [
      { index: 0, name: "Intel Arc Pro B70", vramMiB: vram, vendor: "intel" },
      { index: 1, name: "Intel Arc Pro B70", vramMiB: vram, vendor: "intel" },
    ],
    vramMiB: vram,
    minVramMiB: vram,
    ramMiB: opts.ramMiB ?? 62_276,
    cpuCount: 24,
    hasNvidiaSmi: false,
    constrained: false,
    notes: [],
  };
}

function jsonResponse(body: unknown, status = 200, contentType = "application/json"): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": contentType } });
}

const QWEN25: HubRawModel = {
  id: "Qwen/Qwen2.5-0.5B-Instruct",
  pipeline_tag: "text-generation",
  tags: ["transformers", "safetensors", "qwen2", "text-generation", "conversational"],
  lastModified: "2024-09-01T00:00:00.000Z",
  siblings: [{ rfilename: "model.safetensors", size: 1_000 * 1024 * 1024 }],
  safetensors: { total: 494_032_768, parameters: { BF16: 494_032_768 } },
  config: { model_type: "qwen2" },
};

const QWEN3: HubRawModel = {
  id: "Qwen/Qwen3-8B-Instruct",
  pipeline_tag: "text-generation",
  tags: ["transformers", "safetensors", "qwen3", "text-generation", "instruct"],
  lastModified: "2025-04-01T00:00:00.000Z",
  siblings: [{ rfilename: "model.safetensors", size: 16_000 * 1024 * 1024 }],
  safetensors: { total: 8_000_000_000, parameters: 8_000_000_000 },
  config: { model_type: "qwen3" },
};

const GEMMA_IT: HubRawModel = {
  id: "google/gemma-3-1b-it",
  pipeline_tag: "text-generation",
  tags: ["transformers", "safetensors", "gemma3", "text-generation", "conversational"],
  lastModified: "2025-03-01T00:00:00.000Z",
  siblings: [{ rfilename: "model.safetensors", size: 2_000 * 1024 * 1024 }],
  safetensors: { total: 1_000_000_000, parameters: { BF16: 1_000_000_000 } },
  config: { model_type: "gemma3" },
};

const GEMMA_2B: HubRawModel = {
  id: "google/gemma-2-2b-it",
  gated: true,
  pipeline_tag: "text-generation",
  tags: ["transformers", "safetensors", "gemma2", "text-generation", "conversational"],
  lastModified: "2024-06-24T00:00:00.000Z",
  siblings: [{ rfilename: "model.safetensors", size: 5_000 * 1024 * 1024 }],
  safetensors: { total: 2_000_000_000, parameters: { BF16: 2_000_000_000 } },
  config: { model_type: "gemma2" },
};

const GEMMA_E2B: HubRawModel = {
  id: "google/gemma-4-E2B-it",
  pipeline_tag: "text-generation",
  tags: ["transformers", "safetensors", "gemma4", "text-generation", "conversational"],
  lastModified: "2026-04-01T00:00:00.000Z",
  siblings: [{ rfilename: "model.safetensors", size: 10_000 * 1024 * 1024 }],
  safetensors: { total: 5_100_000_000, parameters: { BF16: 5_100_000_000 } },
  config: { model_type: "gemma4" },
};

const MISTRAL: HubRawModel = {
  id: "mistralai/Mistral-7B-Instruct-v0.3",
  pipeline_tag: "text-generation",
  tags: ["transformers", "safetensors", "mistral", "text-generation", "conversational"],
  lastModified: "2024-05-01T00:00:00.000Z",
  siblings: [{ rfilename: "model.safetensors", size: 14_000 * 1024 * 1024 }],
  safetensors: { total: 7_000_000_000, parameters: 7_000_000_000 },
  config: { model_type: "mistral" },
};

const HUGE: HubRawModel = {
  id: "Qwen/Qwen2.5-72B-Instruct",
  pipeline_tag: "text-generation",
  tags: ["safetensors", "qwen2", "text-generation", "instruct"],
  lastModified: "2024-10-01T00:00:00.000Z",
  siblings: [{ rfilename: "model.safetensors", size: 140_000 * 1024 * 1024 }],
  safetensors: { total: 72_000_000_000, parameters: 72_000_000_000 },
};

const UNKNOWN_SIZE: HubRawModel = {
  id: "example/nightly-instruct-chat",
  pipeline_tag: "text-generation",
  tags: ["safetensors", "qwen2", "text-generation", "instruct"],
  lastModified: "2024-08-01T00:00:00.000Z",
};

const GGUF_ONLY: HubRawModel = {
  id: "Qwen/Qwen2.5-7B-Instruct-GGUF",
  pipeline_tag: "text-generation",
  tags: ["gguf", "qwen2", "text-generation", "instruct"],
  library_name: "gguf",
  siblings: [{ rfilename: "model.gguf", size: 4_000 * 1024 * 1024 }],
};

const DIFFUSION: HubRawModel = {
  id: "stabilityai/stable-diffusion-xl-base-1.0",
  pipeline_tag: "text-to-image",
  tags: ["diffusers", "safetensors", "stable-diffusion"],
};

const WHISPER: HubRawModel = {
  id: "openai/whisper-large-v3",
  pipeline_tag: "automatic-speech-recognition",
  tags: ["safetensors", "whisper"],
};

const VISION: HubRawModel = {
  id: "google/gemma-3-4b-it",
  pipeline_tag: "image-text-to-text",
  tags: ["safetensors", "gemma3", "image-text-to-text", "instruct"],
  lastModified: "2025-03-15T00:00:00.000Z",
  siblings: [{ rfilename: "model.safetensors", size: 8_000 * 1024 * 1024 }],
  safetensors: { total: 4_000_000_000, parameters: { BF16: 4_000_000_000 } },
};

const LLAMA: HubRawModel = {
  id: "meta-llama/Llama-3.1-8B-Instruct",
  gated: true,
  pipeline_tag: "text-generation",
  tags: ["transformers", "safetensors", "llama", "text-generation", "instruct"],
  lastModified: "2024-07-23T00:00:00.000Z",
  siblings: [{ rfilename: "model.safetensors", size: 16_000 * 1024 * 1024 }],
  safetensors: { total: 8_000_000_000, parameters: 8_000_000_000 },
  config: { model_type: "llama" },
};

const PHI: HubRawModel = {
  id: "microsoft/Phi-3.5-mini-instruct",
  pipeline_tag: "text-generation",
  tags: ["transformers", "safetensors", "phi3", "text-generation", "instruct"],
  lastModified: "2024-08-20T00:00:00.000Z",
  siblings: [{ rfilename: "model.safetensors", size: 7_000 * 1024 * 1024 }],
  safetensors: { total: 3_800_000_000, parameters: 3_800_000_000 },
  config: { model_type: "phi3" },
};

const BASE_NO_INSTRUCT: HubRawModel = {
  id: "Qwen/Qwen2.5-7B",
  pipeline_tag: "text-generation",
  tags: ["safetensors", "qwen2", "text-generation"],
};

const CATALOG: HubRawModel[] = [
  QWEN25,
  QWEN3,
  GEMMA_IT,
  GEMMA_2B,
  GEMMA_E2B,
  MISTRAL,
  LLAMA,
  PHI,
  HUGE,
  UNKNOWN_SIZE,
  GGUF_ONLY,
  DIFFUSION,
  WHISPER,
  VISION,
  BASE_NO_INSTRUCT,
];

function mockHubFetch(rows: HubRawModel[] = CATALOG): typeof fetch {
  return (async (input: RequestInfo | URL) => {
    const href = String(input instanceof Request ? input.url : input);
    if (/\/resolve\/main\/config\.json(?:\?|$)/.test(href)) {
      const id = href.replace(/^https:\/\/huggingface\.co\//, "").replace(/\/resolve\/main\/config\.json.*/, "");
      const hit = rows.find((row) => String(row.id ?? row.modelId ?? "") === id);
      if (hit?.config?.model_type) return jsonResponse({ model_type: hit.config.model_type });
      return jsonResponse({ model_type: "qwen2" });
    }
    assert.ok(href.startsWith(HF_HUB_MODELS_API), `must call official Hub API, got ${href}`);
    assert.equal(/huggingface\.co\/models\?/.test(href), false);
    assert.equal(href.includes("scrape"), false);
    return jsonResponse(rows);
  }) as typeof fetch;
}

/** Simulates Hub: a single `Instruct` page is Qwen-only; Gemma only appears on search=gemma. */
function mockHubFetchByFamily(): typeof fetch {
  const seen: string[] = [];
  const fn = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const href = String(input instanceof Request ? input.url : input);
    if (/\/resolve\/main\/config\.json(?:\?|$)/.test(href)) {
      return jsonResponse({ model_type: "qwen2" });
    }
    assert.ok(href.startsWith(HF_HUB_MODELS_API), `must call official Hub API, got ${href}`);
    assert.equal(href.includes("filter=text-generation"), false);
    const url = new URL(href);
    const search = (url.searchParams.get("search") ?? "").toLowerCase();
    seen.push(search);
    void init;
    if (search === "instruct") return jsonResponse([QWEN25, QWEN3, MISTRAL, LLAMA, PHI]);
    if (search === "qwen" || search === "qwen3") return jsonResponse([QWEN25, QWEN3, HUGE, UNKNOWN_SIZE, GGUF_ONLY, BASE_NO_INSTRUCT]);
    if (search === "gemma") return jsonResponse([GEMMA_IT, GEMMA_2B, GEMMA_E2B, VISION]);
    if (search === "mistral") return jsonResponse([MISTRAL]);
    if (search === "llama") return jsonResponse([LLAMA]);
    if (search === "phi") return jsonResponse([PHI]);
    if (search === "unsloth") return jsonResponse([]);
    return jsonResponse([]);
  }) as typeof fetch;
  (fn as typeof fetch & { seen: string[] }).seen = seen;
  return fn;
}

test("buildHubModelsUrl hits the official Hub API, not HTML model pages", () => {
  const url = buildHubModelsUrl("gemma");
  assert.equal(url.origin + url.pathname, HF_HUB_MODELS_API);
  assert.equal(url.searchParams.get("search"), "gemma");
  assert.deepEqual(url.searchParams.getAll("filter"), ["safetensors"]);
  assert.equal(url.searchParams.get("sort"), "lastModified");
  assert.ok(url.searchParams.getAll("expand").includes("safetensors"));
  assert.ok(url.searchParams.getAll("expand").includes("gated"));
  assert.equal(url.pathname.includes("/models/"), false);
});

test("catalogSearchTerms always seeds family searches and adds extra q", () => {
  assert.deepEqual(catalogSearchTerms(""), [...HUB_FAMILY_SEARCHES]);
  assert.deepEqual(catalogSearchTerms("Gemma"), [...HUB_FAMILY_SEARCHES]);
  assert.ok(catalogSearchTerms("olmo").includes("olmo"));
  assert.ok(HUB_FAMILY_SEARCHES.includes("unsloth"));
  assert.ok(HUB_FAMILY_SEARCHES.includes("qwen3"));
  assert.ok(HUB_FAMILY_SEARCHES.includes("instruct"));
  assert.equal(
    catalogSearchTerms("").includes("Instruct"),
    false,
  );
});

test("parseHubVersion reads generation from id, not param size", () => {
  assert.equal(parseHubVersion("Qwen/Qwen2.5-0.5B-Instruct"), 2.5);
  assert.equal(parseHubVersion("Qwen/Qwen3-8B-Instruct"), 3);
  assert.equal(parseHubVersion("meta-llama/Llama-3.1-8B-Instruct"), 3.1);
  assert.equal(parseHubVersion("google/gemma-3-1b-it"), 3);
  assert.equal(parseHubVersion("microsoft/Phi-3.5-mini-instruct"), 3.5);
  assert.equal(parseHubVersion("mistralai/Mistral-7B-Instruct-v0.3"), 0.3);
});

test("hubFamily maps org/tags to Qwen, Mistral, Gemma, Llama, Phi, other", () => {
  assert.equal(hubFamily("Qwen/Qwen2.5-0.5B-Instruct", ["qwen2"]), "Qwen");
  assert.equal(hubFamily("mistralai/Mistral-7B-Instruct-v0.3", ["mistral"]), "Mistral");
  assert.equal(hubFamily("google/gemma-3-1b-it", ["gemma3"]), "Gemma");
  assert.equal(hubFamily("meta-llama/Llama-3.1-8B-Instruct", ["llama"]), "Llama");
  assert.equal(hubFamily("microsoft/Phi-3-mini-4k-instruct", ["phi3"]), "Phi");
  assert.equal(hubFamily("TinyLlama/TinyLlama-1.1B-Chat-v1.0", ["llama"]), "Llama");
  assert.equal(hubFamily("org/some-decoder-instruct", ["text-generation"]), "other");
});

test("isHubInstructCandidate keeps Instruct safetensors and drops GGUF/diffusion", () => {
  assert.equal(isHubInstructCandidate(QWEN25), true);
  assert.equal(isHubInstructCandidate(GEMMA_IT), true);
  assert.equal(isHubInstructCandidate(VISION), true);
  assert.equal(isHubInstructCandidate(GEMMA_2B), true);
  assert.equal(isHubInstructCandidate(GEMMA_E2B), true);
  assert.equal(isHubInstructCandidate(GGUF_ONLY), false);
  assert.equal(isHubInstructCandidate(DIFFUSION), false);
  assert.equal(isHubInstructCandidate(WHISPER), false);
  assert.equal(isHubInstructCandidate(BASE_NO_INSTRUCT), false);
});

test("listHubModels groups at least two families and sorts newest version first", async () => {
  const hw = fakeHardware({ vramMiB: 24_576 });
  const result = await listHubModels({ q: "", fetchFn: mockHubFetch(), hardware: hw });
  assert.equal(result.offline, false);
  const families = result.groups.map((g) => g.family);
  assert.ok(families.includes("Qwen"), JSON.stringify(families));
  assert.ok(families.includes("Gemma"), JSON.stringify(families));
  assert.ok(families.length >= 2);
  const qwen = result.groups.find((g) => g.family === "Qwen");
  assert.ok(qwen);
  assert.equal(qwen.models[0]?.id, "Qwen/Qwen3-8B-Instruct"); // newest Qwen version first
  const qwenIds = qwen.models.map((m) => m.id);
  assert.ok(qwenIds.includes("Qwen/Qwen3-8B-Instruct"));
  assert.ok(qwenIds.includes("Qwen/Qwen2.5-0.5B-Instruct"));
  for (const model of result.models) {
    assert.ok(model.family);
    assert.equal(typeof model.id, "string");
    assert.equal(typeof model.fits, "boolean");
  }
});

test("within a family, lastModified breaks ties when version is missing", () => {
  const older: HubCatalogModel = {
    id: "other/chat-a",
    family: "other",
    lastModified: "2023-01-01T00:00:00.000Z",
    fits: true,
    likelyTooBig: false,
    sizeUnknown: true,
    gated: false,
  };
  const newer: HubCatalogModel = {
    id: "other/chat-b",
    family: "other",
    lastModified: "2026-01-01T00:00:00.000Z",
    fits: true,
    likelyTooBig: false,
    sizeUnknown: true,
    gated: false,
  };
  assert.ok(compareHubModels(newer, older) < 0);
  const grouped = groupHubModels([older, newer]);
  assert.equal(grouped[0]?.models[0]?.id, "other/chat-b");
});

test("8 GB VRAM marks a 72B snapshot likely too big and keeps 0.5B as fitting", async () => {
  const hw = fakeHardware({ vramMiB: 8_192 });
  const result = await listHubModels({ fetchFn: mockHubFetch(), hardware: hw });
  const small = result.models.find((m) => m.id === "Qwen/Qwen2.5-0.5B-Instruct");
  const huge = result.models.find((m) => m.id === "Qwen/Qwen2.5-72B-Instruct");
  assert.equal(small?.fits, true);
  assert.ok((small?.bytes ?? 0) > 0);
  assert.ok((small?.params ?? 0) > 0);
  assert.equal(huge?.fits, false);
  assert.equal(huge?.likelyTooBig, true);
  assert.ok(result.budgetGiB > 0);
  assert.match(result.hint, /OpenVINO|late-infer|idle (?:Intel )?GPU|your computer/i);
  assert.match(result.hint, /VRAM at max usage|Not every Hub repo compiles|Each row can actually compile|GGUF-only/);
});

test("unknown Hub size metadata is included (conservative fit)", async () => {
  const model = toHubCatalogModel(UNKNOWN_SIZE, 4_000);
  assert.ok(model);
  assert.equal(model.sizeUnknown, true);
  assert.equal(model.fits, true);
  assert.equal(model.likelyTooBig, false);
  assert.equal(model.vramMaxLabel, "VRAM unknown");
  const listed = await listHubModels({
    fetchFn: mockHubFetch([UNKNOWN_SIZE]),
    hardware: fakeHardware({ vramMiB: 24_576 }),
  });
  const row = listed.models.find((m) => m.id === UNKNOWN_SIZE.id);
  assert.ok(row, "unknown-size community id still lists");
  assert.equal(row.sizeUnknown, true);
  assert.equal(row.vramMaxLabel, "VRAM unknown");
  assert.equal(row.fits, true);
  assert.ok(listed.models.some((m) => m.id === "Qwen/Qwen2.5-0.5B-Instruct"));
});

test("max VRAM is weights plus 50% KV/activation headroom", () => {
  assert.equal(estimateMaxVramMiB(1_000), 1_500);
  assert.equal(formatVramMaxLabel(1_500), "~1.5 GB VRAM max");
  assert.equal(formatVramMaxLabel(21_000), "~21 GB VRAM max");
  assert.equal(parseHubParamsB("Qwen/Qwen2.5-7B-Instruct"), 7);
  assert.equal(parseHubParamsB("google/gemma-4-E2B-it"), 5.1);
  const qwen = fallbackHubCatalogModels().find((m) => m.id === "Qwen/Qwen2.5-0.5B-Instruct");
  assert.equal(qwen?.sizeMiB, 1_000);
  assert.equal(qwen?.vramMaxMiB, 1_500);
  assert.equal(qwen?.vramMaxLabel, "~1.5 GB VRAM max");
  const mistral = fallbackHubCatalogModels().find((m) => m.id === "mistralai/Mistral-7B-Instruct-v0.2");
  assert.equal(mistral?.sizeMiB, 14_000);
  assert.equal(mistral?.vramMaxMiB, 21_000);
  assert.equal(mistral?.vramMaxLabel, "~21 GB VRAM max");
  const fromId = withVramEstimate({
    id: "Qwen/Qwen2.5-1.5B-Instruct",
    family: "Qwen",
    fits: true,
    likelyTooBig: false,
    sizeUnknown: true,
    gated: false,
  });
  assert.equal(fromId.sizeUnknown, false);
  assert.equal(fromId.sizeMiB, 3_000);
  assert.equal(fromId.vramMaxMiB, 4_500);
});

test("CPU-only machines budget 80% of RAM", async () => {
  const hw = fakeHardware({ ramMiB: 8_192 });
  const result = await listHubModels({ fetchFn: mockHubFetch([QWEN25, HUGE]), hardware: hw });
  assert.equal(Math.round(result.budgetMiB), Math.round(8_192 * 0.8));
  const huge = result.models.find((m) => m.id === HUGE.id);
  assert.equal(huge?.fits, false);
});

test("Intel Arc / XPU without nvidia-smi still lists models using RAM when VRAM is unknown", async () => {
  const hw = fakeIntelHardware({ vramMiB: 0, ramMiB: 62_276 });
  assert.equal(hw.hasNvidiaSmi, false);
  assert.equal(hw.primaryBackend, "intel-xpu");
  assert.equal(Math.round(hardwareFitBudgetMiB(hw)), Math.round(62_276 * HUB_FIT_FRACTION));
  const result = await listHubModels({ fetchFn: mockHubFetch(), hardware: hw });
  assert.equal(result.offline, false);
  assert.ok(result.groups.length >= 2, JSON.stringify(result.groups.map((g) => g.family)));
  assert.ok(result.models.length >= 2);
  assert.ok(result.models.some((m) => m.id === QWEN25.id));
  assert.ok(result.budgetGiB > 40);
});

test("Intel Arc with known VRAM budgets from GPU memory, not nvidia-smi", async () => {
  const hw = fakeIntelHardware({ vramMiB: 31_023, ramMiB: 62_276 });
  assert.equal(hw.hasNvidiaSmi, false);
  assert.equal(Math.round(hardwareFitBudgetMiB(hw)), Math.round(62_046 * HUB_FIT_FRACTION));
  const result = await listHubModels({
    fetchFn: mockHubFetch([QWEN25, GEMMA_IT, GEMMA_2B, GEMMA_E2B, VISION, MISTRAL]),
    hardware: hw,
  });
  assert.equal(result.offline, false);
  assert.ok(result.groups.length >= 2);
  for (const id of ["google/gemma-3-1b-it", "google/gemma-2-2b-it", "google/gemma-4-E2B-it", "google/gemma-3-4b-it"]) {
    const row = result.models.find((m) => m.id === id);
    assert.ok(row, id);
    assert.equal(row.fits, true, id);
  }
});

test("q= filters across families", async () => {
  const hw = fakeHardware({ vramMiB: 24_576 });
  const result = await listHubModels({ q: "gemma", fetchFn: mockHubFetch(), hardware: hw });
  assert.ok(result.models.length >= 1);
  assert.ok(result.models.every((m) => /gemma/i.test(m.id) || m.family === "Gemma"));
  assert.equal(
    result.models.some((m) => m.family === "Qwen"),
    false,
  );
});

test("family Hub searches return Gemma even when Instruct search is Qwen-only", async () => {
  const hw = fakeHardware({ vramMiB: 24_576 });
  const fetchFn = mockHubFetchByFamily();
  const empty = await listHubModels({ q: "", fetchFn, hardware: hw });
  const seen = (fetchFn as typeof fetch & { seen: string[] }).seen;
  for (const family of HUB_FAMILY_SEARCHES) {
    assert.ok(seen.includes(family), `missing family search ${family}: ${JSON.stringify(seen)}`);
  }
  assert.equal(seen.includes("instruct"), true);
  assert.ok(seen.includes("unsloth"));
  assert.ok(seen.includes("qwen3"));
  const families = empty.groups.map((g) => g.family);
  assert.ok(families.includes("Qwen"), JSON.stringify(families));
  assert.ok(families.includes("Gemma"), JSON.stringify(families));
  assert.ok(families.includes("Mistral"), JSON.stringify(families));
  assert.ok(families.includes("Llama"), JSON.stringify(families));
  assert.ok(families.includes("Phi"), JSON.stringify(families));
  assert.ok(empty.models.some((m) => m.id === "google/gemma-3-1b-it"));
  assert.ok(empty.models.some((m) => m.id === "google/gemma-3-4b-it"));
  assert.ok(empty.models.some((m) => m.id === "google/gemma-4-E2B-it"));

  resetHubCatalogForTests();
  const gemma = await listHubModels({ q: "Gemma", fetchFn: mockHubFetchByFamily(), hardware: hw });
  assert.ok(gemma.models.length >= 3, JSON.stringify(gemma.models.map((m) => m.id)));
  assert.ok(gemma.models.every((m) => /gemma/i.test(m.id) || m.family === "Gemma"));
  assert.ok(gemma.models.some((m) => m.id.startsWith("google/gemma-")));
  assert.equal(gemma.models.some((m) => m.family === "Qwen"), false);
});

test("gated Gemma/Llama stay listed", async () => {
  const hw = fakeHardware({ vramMiB: 24_576 });
  const result = await listHubModels({ fetchFn: mockHubFetch(), hardware: hw });
  const gemma2 = result.models.find((m) => m.id === "google/gemma-2-2b-it");
  const llama = result.models.find((m) => m.id === "meta-llama/Llama-3.1-8B-Instruct");
  assert.equal(gemma2?.gated, true);
  assert.equal(llama?.gated, true);
  assert.equal(gemma2?.gatedNeedsLicense, true);
  assert.equal(gemma2?.family, "Gemma");
});

test("Hub gated:manual string marks gated on search rows", async () => {
  const hw = fakeHardware({ vramMiB: 24_576 });
  const row: HubRawModel = {
    id: "meta-llama/Llama-3.2-1B-Instruct",
    gated: "manual",
    pipeline_tag: "text-generation",
    tags: ["transformers", "safetensors", "llama", "text-generation", "instruct"],
    lastModified: "2024-09-01T00:00:00.000Z",
    siblings: [{ rfilename: "model.safetensors", size: 3_000 * 1024 * 1024 }],
    safetensors: { total: 1_200_000_000, parameters: 1_200_000_000 },
    config: { model_type: "llama" },
  };
  const result = await listHubModels({ fetchFn: mockHubFetch([row]), hardware: hw });
  const hit = result.models.find((m) => m.id === row.id);
  assert.equal(hit?.gated, true);
  assert.equal(hit?.gatedKnown, true);
  assert.equal(hit?.gatedNeedsLicense, true);
});

test("merge keeps seed gated when Hub list omits gated field", () => {
  const merged = mergeHubCatalog([
    {
      id: "google/gemma-2-2b-it",
      family: "Gemma",
      fits: true,
      likelyTooBig: false,
      sizeUnknown: false,
      gated: false,
      gatedKnown: false,
      sizeMiB: 5000,
    },
  ]);
  const gemma = merged.find((m) => m.id === "google/gemma-2-2b-it");
  assert.equal(gemma?.gated, true);
  assert.equal(gemma?.fallback, true);
});

test("Hub gated:false wins over stale seed gated when gatedKnown", () => {
  const merged = mergeHubCatalog([
    {
      id: "google/gemma-2-2b-it",
      family: "Gemma",
      fits: true,
      likelyTooBig: false,
      sizeUnknown: false,
      gated: false,
      gatedKnown: true,
      sizeMiB: 5000,
    },
  ]);
  const gemma = merged.find((m) => m.id === "google/gemma-2-2b-it");
  assert.equal(gemma?.gated, false);
});

test("listing sends HF_TOKEN when present and still lists public models without it", async () => {
  const hw = fakeHardware({ vramMiB: 8_192 });
  const headers: string[] = [];
  const fetchFn = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const href = String(input instanceof Request ? input.url : input);
    assert.ok(href.startsWith(HF_HUB_MODELS_API));
    headers.push(String((init?.headers as Record<string, string> | undefined)?.authorization ?? ""));
    return jsonResponse([QWEN25, GEMMA_IT]);
  }) as typeof fetch;
  const withToken = await listHubModels({ fetchFn, hardware: hw, token: "hf_not-a-real-token" });
  assert.equal(withToken.offline, false);
  assert.ok(headers.some((h) => h === "Bearer hf_not-a-real-token"));
  resetHubCatalogForTests();
  const noTokenHeaders: string[] = [];
  const fetchPublic = (async (input: RequestInfo | URL, init?: RequestInit) => {
    noTokenHeaders.push(String((init?.headers as Record<string, string> | undefined)?.authorization ?? ""));
    return jsonResponse([QWEN25, GEMMA_IT]);
  }) as typeof fetch;
  const publicList = await listHubModels({ fetchFn: fetchPublic, hardware: hw, token: "" });
  assert.ok(publicList.models.some((m) => m.id === GEMMA_IT.id));
  assert.ok(noTokenHeaders.every((h) => h === ""));
});

test("Hub API failure still lists fallback families including Gemma", async () => {
  const hw = fakeHardware({ vramMiB: 8_192 });
  const fetchFn = (async () => {
    throw new Error("network down");
  }) as typeof fetch;
  const result = await listHubModels({ fetchFn, hardware: hw });
  assert.equal(result.offline, true);
  assert.ok(result.error);
  const families = result.groups.map((g) => g.family);
  assert.ok(families.includes("Qwen"), JSON.stringify(families));
  assert.ok(families.includes("Gemma"), JSON.stringify(families));
  assert.ok(families.includes("Mistral"), JSON.stringify(families));
  assert.ok(families.includes("Llama"), JSON.stringify(families));
  assert.ok(families.includes("Phi"), JSON.stringify(families));
  assert.ok(result.models.some((m) => m.id === "google/gemma-2-2b-it"));
  assert.ok(result.models.some((m) => m.id === "google/gemma-4-E2B-it"));
  assert.ok(result.models.some((m) => m.id === "Qwen/Qwen2.5-0.5B-Instruct"));
  assert.equal(result.defaultId, "Qwen/Qwen2.5-0.5B-Instruct");
  for (const model of result.models) {
    assert.equal(model.likelyTooBig, false);
    assert.equal(model.sizeUnknown, false);
    assert.match(String(model.vramMaxLabel), /VRAM max/);
  }
});

test("empty Hub page still lists fallback (lastModified search is community noise)", async () => {
  const hw = fakeHardware({ vramMiB: 8_192 });
  const fetchFn = (async () => jsonResponse([])) as typeof fetch;
  const result = await listHubModels({ fetchFn, hardware: hw });
  assert.ok(result.models.length >= FALLBACK_HUB_SEEDS.length);
  assert.ok(result.groups.some((g) => g.family === "Gemma"));
  assert.ok(result.models.some((m) => m.id === "google/gemma-3-1b-it"));
  assert.ok(result.models.some((m) => m.id === "mistralai/Mistral-7B-Instruct-v0.3"));
  assert.ok(result.models.some((m) => m.id === "meta-llama/Llama-3.2-1B-Instruct"));
  assert.ok(result.models.some((m) => m.id === "microsoft/Phi-4-mini-instruct"));
});

test("q=Mistral on fallback-only catalog returns Mistral Instruct ids", async () => {
  const hw = fakeHardware({ vramMiB: 8_192 });
  const result = await listHubModels({
    q: "Mistral",
    fetchFn: (async () => jsonResponse([])) as typeof fetch,
    hardware: hw,
  });
  assert.ok(result.models.length >= 1, JSON.stringify(result.models.map((m) => m.id)));
  assert.ok(result.models.every((m) => /mistral/i.test(m.id) || m.family === "Mistral"));
  assert.ok(result.models.some((m) => m.id === "mistralai/Mistral-7B-Instruct-v0.3"));
  assert.ok(result.models.some((m) => m.id === "mistralai/Ministral-8B-Instruct-2410"));
  assert.equal(result.models.some((m) => m.family === "Qwen"), false);
});

test("q=Gemma on fallback-only catalog returns Gemma rows", async () => {
  const hw = fakeHardware({ vramMiB: 8_192 });
  const result = await listHubModels({
    q: "Gemma",
    fetchFn: (async () => jsonResponse([])) as typeof fetch,
    hardware: hw,
  });
  assert.ok(result.models.length >= 1, JSON.stringify(result.models.map((m) => m.id)));
  assert.ok(result.models.every((m) => /gemma/i.test(m.id) || m.family === "Gemma"));
  assert.equal(result.models.some((m) => m.family === "Qwen"), false);
  assert.ok(result.models.some((m) => m.id === "google/gemma-2-2b-it"));
});

test("Intel Arc with empty Hub still lists fallback including Gemma", async () => {
  const hw = fakeIntelHardware({ vramMiB: 0, ramMiB: 62_276 });
  const result = await listHubModels({
    fetchFn: (async () => {
      throw new Error("ECONNREFUSED");
    }) as typeof fetch,
    hardware: hw,
  });
  assert.equal(hw.hasNvidiaSmi, false);
  assert.ok(result.models.some((m) => m.id === "google/gemma-2-2b-it"));
  assert.ok(result.models.some((m) => m.id === "google/gemma-4-E2B-it"));
  assert.ok(result.groups.some((g) => g.family === "Gemma"));
  for (const seed of FALLBACK_HUB_SEEDS) {
    const row = result.models.find((m) => m.id === seed.id);
    assert.ok(row, seed.id);
    assert.equal(row.likelyTooBig, false, seed.id);
    assert.equal(row.fits, true, seed.id);
    assert.equal(row.sizeUnknown, false, seed.id);
    assert.ok((row.vramMaxMiB ?? 0) > 0, seed.id);
    assert.match(String(row.vramMaxLabel), /VRAM max/);
  }
});

test("community lastModified noise still lists official Instruct families", async () => {
  const community: HubRawModel = {
    id: "someorg/nightly-qwen-instruct-chat",
    pipeline_tag: "text-generation",
    tags: ["safetensors", "qwen2", "text-generation", "instruct"],
    lastModified: "2026-08-01T00:00:00.000Z",
    siblings: [{ rfilename: "model.safetensors", size: 2_000 * 1024 * 1024 }],
  };
  const hw = fakeHardware({ vramMiB: 8_192 });
  const result = await listHubModels({ fetchFn: mockHubFetch([community]), hardware: hw });
  assert.ok(result.models.some((m) => m.id === community.id));
  assert.ok(result.models.some((m) => m.id === "Qwen/Qwen2.5-0.5B-Instruct"));
  assert.ok(result.models.some((m) => m.id === "google/gemma-2-2b-it"));
  assert.ok(result.models.some((m) => m.id === "mistralai/Mistral-7B-Instruct-v0.3"));
  assert.ok(result.models.some((m) => m.id === "meta-llama/Llama-3.2-1B-Instruct"));
  assert.ok(result.models.some((m) => m.id === "microsoft/Phi-4-mini-instruct"));
  const families = result.groups.map((g) => g.family);
  for (const family of ["Qwen", "Gemma", "Mistral", "Llama", "Phi"] as const) {
    assert.ok(families.includes(family), family);
  }
});

test("fallback seeds stay listed and not too-big when Hub size exceeds budget", async () => {
  const hw = fakeHardware({ vramMiB: 8_192 });
  const result = await listHubModels({ fetchFn: mockHubFetch([MISTRAL, HUGE]), hardware: hw });
  const seed = result.models.find((m) => m.id === "mistralai/Mistral-7B-Instruct-v0.3");
  assert.ok(seed);
  assert.equal(seed.fallback, true);
  assert.equal(seed.likelyTooBig, false);
  assert.equal(seed.fits, true);
  assert.equal(seed.vramMaxMiB, 21_000);
  assert.equal(seed.vramMaxLabel, "~21 GB VRAM max");
  const huge = result.models.find((m) => m.id === HUGE.id);
  assert.equal(huge?.likelyTooBig, true);
  assert.ok(result.models.some((m) => m.id === "google/gemma-2-2b-it"));
});

test("Intel Arc still lists fallback seeds when estimated size exceeds VRAM", async () => {
  const hw = fakeIntelHardware({ vramMiB: 2_048, ramMiB: 8_192 });
  const result = await listHubModels({ fetchFn: mockHubFetch([MISTRAL]), hardware: hw });
  assert.equal(hw.hasNvidiaSmi, false);
  for (const seed of FALLBACK_HUB_SEEDS) {
    const row = result.models.find((m) => m.id === seed.id);
    assert.ok(row, seed.id);
    assert.equal(row.likelyTooBig, false, seed.id);
    assert.equal(row.fits, true, seed.id);
    assert.match(String(row.vramMaxLabel), /VRAM max/);
  }
});

test("Hub extras merge on top of fallback without dropping official ids", async () => {
  const hw = fakeHardware({ vramMiB: 24_576 });
  const extra: HubRawModel = {
    id: "Qwen/Qwen3-8B-Instruct",
    pipeline_tag: "text-generation",
    tags: ["safetensors", "qwen3", "text-generation", "instruct"],
    lastModified: "2025-04-01T00:00:00.000Z",
    siblings: [{ rfilename: "model.safetensors", size: 16_000 * 1024 * 1024 }],
  };
  const result = await listHubModels({ fetchFn: mockHubFetch([extra]), hardware: hw });
  assert.ok(result.models.some((m) => m.id === extra.id));
  assert.ok(result.models.some((m) => m.id === "Qwen/Qwen2.5-0.5B-Instruct"));
  assert.ok(result.models.some((m) => m.id === "google/gemma-2-2b-it"));
  const merged = mergeHubCatalog([
    {
      id: "Qwen/Qwen2.5-0.5B-Instruct",
      family: "Qwen",
      fits: true,
      likelyTooBig: false,
      sizeUnknown: false,
      gated: false,
      sizeMiB: 1000,
    },
  ]);
  const overlay = merged.find((m) => m.id === "Qwen/Qwen2.5-0.5B-Instruct");
  assert.equal(overlay?.sizeUnknown, false);
  assert.equal(overlay?.sizeMiB, 1000);
  assert.equal(overlay?.vramMaxMiB, 1500);
  assert.equal(overlay?.vramMaxLabel, "~1.5 GB VRAM max");
});

test("HTML Hub payload is treated as offline, not scraped", async () => {
  const hw = fakeHardware({ vramMiB: 8_192 });
  const fetchFn = (async (input: RequestInfo | URL) => {
    const href = String(input instanceof Request ? input.url : input);
    assert.ok(href.startsWith(HF_HUB_MODELS_API));
    return new Response("<html><body>models</body></html>", {
      status: 200,
      headers: { "content-type": "text/html" },
    });
  }) as typeof fetch;
  const result = await listHubModels({ fetchFn, hardware: hw });
  assert.equal(result.offline, true);
  assert.ok(result.models.some((m) => m.id === "google/gemma-2-2b-it"));
  assert.match(result.error ?? "", /HTML/i);
});

test("listing does not require CURSOR_API_KEY / Cloud AI", async () => {
  const prev = process.env.CURSOR_API_KEY;
  delete process.env.CURSOR_API_KEY;
  try {
    const result = await listHubModels({
      fetchFn: mockHubFetch([QWEN25, GEMMA_IT]),
      hardware: fakeHardware({ vramMiB: 8_192 }),
    });
    assert.equal(result.offline, false);
    assert.ok(result.models.length >= 1);
  } finally {
    if (prev === undefined) delete process.env.CURSOR_API_KEY;
    else process.env.CURSOR_API_KEY = prev;
  }
});

async function freeLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const addr = probe.address();
      const port = typeof addr === "object" && addr ? addr.port : 0;
      probe.close((err) => (err ? reject(err) : resolve(port)));
    });
  });
}

test("GET /api/local-servers/hub-models returns grouped catalog from mocked Hub", async () => {
  resetHubCatalogForTests({
    fetchFn: mockHubFetchByFamily(),
    hardware: fakeHardware({ vramMiB: 24_576 }),
  });
  const events = new EventEmitter();
  const orchestrator = {
    events,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: { snapshot: () => ({ models: [], recommended: [], jobs: [], hardware: {}, intelDocker: {} }) },
    store: { list: () => [], get: () => undefined },
    configPath: "/tmp/agents.config.yaml",
    config: { backends: {}, specialists: {} },
    allowlist: { list: () => [] },
    defaultCwd: () => "/",
    reloadConfig: () => undefined,
  };
  const chat = {
    list: () => [],
    create: () => ({}),
    get: () => ({}),
    delete: () => false,
    setPin: () => ({}),
    send: async () => ({}),
    runAction: async () => ({}),
    resolveApproval: async () => ({}),
  };
  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
  const { server, listen } = startGuiServer({
    orchestrator: orchestrator as never,
    chat: chat as never,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  const base = `http://127.0.0.1:${port}`;
  const auth = { authorization: `Bearer ${token}` };
  try {
    const res = await fetch(`${base}/api/local-servers/hub-models?q=`, { headers: auth });
    assert.equal(res.status, 200);
    const body = (await res.json()) as {
      groups?: { family: string; models: { id: string; family: string; fits: boolean; vramMaxLabel?: string; vramMaxMiB?: number }[] }[];
      models?: { id: string; family: string; vramMaxLabel?: string; vramMaxMiB?: number }[];
      offline?: boolean;
      defaultId?: string;
      hint?: string;
    };
    assert.equal(body.offline, false);
    assert.equal(body.defaultId, "Qwen/Qwen2.5-0.5B-Instruct");
    assert.ok(Array.isArray(body.groups) && body.groups.length >= 2);
    assert.ok(Array.isArray(body.models) && body.models.length >= 1);
    const families = (body.groups ?? []).map((g) => g.family);
    assert.ok(families.includes("Qwen"));
    assert.ok(families.includes("Gemma"), JSON.stringify(families));
    assert.ok(families.includes("Mistral"), JSON.stringify(families));
    assert.match(body.hint ?? "", /your computer/);
    assert.ok((body.models ?? []).some((m) => m.id.startsWith("google/gemma-")));
    const withVram = (body.models ?? []).find((m) => typeof m.vramMaxLabel === "string" && /VRAM max/.test(m.vramMaxLabel));
    assert.ok(withVram, "GET must pass through vramMaxLabel from listHubModels");
    assert.equal(typeof withVram.vramMaxMiB, "number");
    const grouped = (body.groups ?? []).flatMap((g) => g.models);
    assert.ok(grouped.some((m) => typeof m.vramMaxLabel === "string" && m.vramMaxLabel.length > 0));

    const gemma = await fetch(`${base}/api/local-servers/hub-models?q=gemma`, { headers: auth });
    assert.equal(gemma.status, 200);
    const gemmaBody = (await gemma.json()) as {
      groups?: { family: string; models: { id: string }[] }[];
      models: { id: string; family: string; vramMaxLabel?: string; vramMaxMiB?: number }[];
    };
    assert.ok(Array.isArray(gemmaBody.groups) && gemmaBody.groups.length >= 1);
    assert.ok(gemmaBody.models.length >= 1, JSON.stringify(gemmaBody.models));
    assert.ok(gemmaBody.models.some((m) => /gemma/i.test(m.id)));
    assert.ok(gemmaBody.models.every((m) => /gemma/i.test(m.id) || m.family === "Gemma"));
    assert.ok(gemmaBody.models.some((m) => typeof m.vramMaxLabel === "string" && m.vramMaxLabel.length > 0));
  } finally {
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});

test("GET /api/local-servers/hub-models stays 200 when Hub is down", async () => {
  resetHubCatalogForTests({
    fetchFn: (async () => {
      throw new Error("ECONNREFUSED");
    }) as typeof fetch,
    hardware: fakeHardware({ ramMiB: 8_192 }),
  });
  const events = new EventEmitter();
  const orchestrator = {
    events,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: { snapshot: () => ({}) },
    store: { list: () => [], get: () => undefined },
    configPath: "/tmp/agents.config.yaml",
    config: { backends: {}, specialists: {} },
    allowlist: { list: () => [] },
    defaultCwd: () => "/",
    reloadConfig: () => undefined,
  };
  const chat = {
    list: () => [],
    create: () => ({}),
    get: () => ({}),
    delete: () => false,
    setPin: () => ({}),
    send: async () => ({}),
    runAction: async () => ({}),
    resolveApproval: async () => ({}),
  };
  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
  const { server, listen } = startGuiServer({
    orchestrator: orchestrator as never,
    chat: chat as never,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  try {
    const res = await fetch(`http://127.0.0.1:${port}/api/local-servers/hub-models`, {
      headers: { authorization: `Bearer ${token}` },
    });
    assert.equal(res.status, 200);
    const body = (await res.json()) as {
      offline?: boolean;
      models?: { id: string; family: string; vramMaxLabel?: string; vramMaxMiB?: number }[];
      groups?: { family: string; models: { id: string; vramMaxLabel?: string; vramMaxMiB?: number }[] }[];
    };
    assert.equal(res.status, 200);
    assert.equal(body.offline, true);
    const families = (body.groups ?? []).map((g) => g.family);
    assert.ok(families.includes("Gemma"), JSON.stringify(families));
    assert.ok(families.includes("Qwen"), JSON.stringify(families));
    assert.ok((body.models ?? []).some((m) => m.id === "google/gemma-2-2b-it"));
    assert.ok((body.models ?? []).some((m) => m.id === "Qwen/Qwen2.5-0.5B-Instruct"));
    assert.ok((body.models ?? []).some((m) => m.id === "mistralai/Mistral-7B-Instruct-v0.3"));
    const qwen = (body.models ?? []).find((m) => m.id === "Qwen/Qwen2.5-0.5B-Instruct");
    assert.equal(qwen?.vramMaxLabel, "~1.5 GB VRAM max");
    assert.equal(qwen?.vramMaxMiB, 1500);
    const groupedQwen = (body.groups ?? []).flatMap((g) => g.models).find((m) => m.id === "Qwen/Qwen2.5-0.5B-Instruct");
    assert.equal(groupedQwen?.vramMaxLabel, "~1.5 GB VRAM max");
    assert.equal(groupedQwen?.vramMaxMiB, 1500);
  } finally {
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});

test("applyHubLoadability stamps ovExportOk/loadable for Intel OpenVINO CausalLM", () => {
  const qwen = applyHubLoadability(
    {
      id: "Qwen/Qwen3-8B-Instruct",
      family: "Qwen",
      fits: true,
      likelyTooBig: false,
      sizeUnknown: false,
      gated: false,
      modelType: "qwen3",
    },
    "intel",
    true,
  );
  assert.equal(qwen.ovExportOk, true);
  assert.equal(qwen.loadable, true);
  assert.equal(qwen.downloadable, true);
  assert.equal(openvinoCanExportCausalLm("qwen3"), true);

  const gemma3 = applyHubLoadability(
    {
      id: "google/gemma-3-1b-it",
      family: "Gemma",
      fits: true,
      likelyTooBig: false,
      sizeUnknown: false,
      gated: true,
      modelType: "gemma3",
    },
    "intel",
    true,
  );
  assert.equal(gemma3.ovExportOk, false);
  assert.equal(gemma3.loadable, false);
  assert.equal(gemma3.downloadable, false);
  assert.ok(String(gemma3.serveBlockedReason ?? "").length > 0);
});

test("compareHubModels prefers loadable ungated over gated and demotes broken", () => {
  const loadableUngated: HubCatalogModel = {
    id: "Qwen/Qwen3-1.7B-Instruct",
    family: "Qwen",
    version: 3,
    lastModified: "2025-01-01T00:00:00.000Z",
    fits: true,
    likelyTooBig: false,
    sizeUnknown: false,
    gated: false,
    loadable: true,
    ovExportOk: true,
    modelType: "qwen3",
  };
  const loadableGated: HubCatalogModel = {
    id: "meta-llama/Llama-3.2-1B-Instruct",
    family: "Llama",
    version: 3.2,
    lastModified: "2026-01-01T00:00:00.000Z",
    fits: true,
    likelyTooBig: false,
    sizeUnknown: false,
    gated: true,
    loadable: true,
    ovExportOk: true,
    modelType: "llama",
  };
  const broken: HubCatalogModel = {
    id: "org/broken-instruct",
    family: "other",
    lastModified: "2026-06-01T00:00:00.000Z",
    fits: true,
    likelyTooBig: false,
    sizeUnknown: true,
    gated: false,
    loadable: false,
    configStatus: "missing",
    ovExportOk: false,
  };
  assert.ok(compareHubModels(loadableUngated, loadableGated) < 0);
  assert.ok(compareHubModels(loadableGated, broken) < 0);
  assert.ok(compareHubModels(loadableUngated, broken) < 0);
  const gemma2: HubCatalogModel = {
    ...loadableUngated,
    id: "google/gemma-2-2b-it",
    family: "Gemma",
    modelType: "gemma2",
    gated: true,
    version: 2,
  };
  const qwenLoadable = { ...loadableUngated, gated: false };
  assert.ok(compareHubModels(qwenLoadable, gemma2) < 0, "Gemma2 must not sort as primary over loadable Qwen3");
});

test("listHubModels demotes missing config.json via soft-fail probe", async () => {
  const hw = fakeIntelHardware({ vramMiB: 31_000 });
  const broken: HubRawModel = {
    id: "org/missing-config-instruct",
    pipeline_tag: "text-generation",
    tags: ["transformers", "safetensors", "text-generation", "instruct"],
    lastModified: "2026-08-01T00:00:00.000Z",
    siblings: [{ rfilename: "model.safetensors", size: 1_000 * 1024 * 1024 }],
    safetensors: { total: 500_000_000, parameters: { BF16: 500_000_000 } },
    // no config.model_type — forces probe
  };
  const base = mockHubFetch([broken, QWEN3]);
  const fetchFn = (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input instanceof Request ? input.url : input);
    if (url.includes("/resolve/main/config.json") && url.includes("missing-config-instruct")) {
      return new Response("Not Found", { status: 404, headers: { "content-type": "text/plain" } });
    }
    return base(input, init);
  }) as typeof fetch;
  const result = await listHubModels({ fetchFn, hardware: hw });
  const miss = result.models.find((m) => m.id === "org/missing-config-instruct");
  const qwen = result.models.find((m) => m.id === "Qwen/Qwen3-8B-Instruct");
  assert.ok(miss, "broken row still listed (soft-fail)");
  assert.equal(miss?.configStatus, "missing");
  assert.equal(miss?.loadable, false);
  assert.equal(qwen?.loadable, true);
  assert.equal(qwen?.ovExportOk, true);
  const qwenGroup = result.groups.find((g) => g.family === "Qwen");
  assert.ok(qwenGroup);
  assert.equal(qwenGroup?.models[0]?.id, "Qwen/Qwen3-8B-Instruct");
  // default skips non-loadable / gemma2
  assert.notEqual(result.defaultId, "google/gemma-2-2b-it");
});

test("stampHubCatalogLoadability keeps gated downloadable false while architecture loadable", () => {
  const [llama] = stampHubCatalogLoadability(
    [
      {
        id: "meta-llama/Llama-3.2-1B-Instruct",
        family: "Llama",
        fits: true,
        likelyTooBig: false,
        sizeUnknown: false,
        gated: true,
        gatedNeedsLicense: true,
        modelType: "llama",
      },
    ],
    "intel",
    true,
  );
  assert.equal(llama.loadable, true);
  assert.equal(llama.downloadable, false);
  assert.equal(llama.gatedNeedsLicense, true);
  assert.equal(llama.ovExportOk, true);
});
