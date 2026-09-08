import assert from "node:assert/strict";
import { afterEach, test } from "node:test";
import type { HardwareSnapshot } from "../src/hardware.js";
import {
  GGUF_HUB_SEARCHES,
  buildGgufHubModelsUrl,
  ggufCatalogSearchTerms,
  ggufFitBudgetMiB,
  isGgufHubCandidate,
  listGgufHubModels,
  pickPreferredGgufFile,
  quantRank,
  resetGgufHubCatalogForTests,
  toGgufHubModel,
  type GgufFileHint,
} from "../src/local-servers/hub-gguf-catalog.js";
import { HF_HUB_MODELS_API, HUB_FIT_FRACTION, type HubRawModel } from "../src/local-servers/hub-catalog.js";

afterEach(() => {
  resetGgufHubCatalogForTests();
});

function fakeIntelHardware(): HardwareSnapshot {
  return {
    accelerators: [
      { vendor: "intel", name: "Intel Arc Pro B70", vramMiB: 32_000, index: 0, source: "sysfs" },
      { vendor: "intel", name: "Intel Arc Pro B70", vramMiB: 32_000, index: 1, source: "sysfs" },
    ],
    primaryBackend: "intel-xpu",
    totalVramMiB: 64_000,
    deviceCount: 2,
    gpus: [
      { index: 0, name: "Intel Arc Pro B70", vramMiB: 32_000, vendor: "intel" },
      { index: 1, name: "Intel Arc Pro B70", vramMiB: 32_000, vendor: "intel" },
    ],
    vramMiB: 32_000,
    minVramMiB: 32_000,
    ramMiB: 62_000,
    cpuCount: 24,
    hasNvidiaSmi: false,
    constrained: false,
    notes: [],
  };
}

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
}

const UNSLOTH_GGUF: HubRawModel = {
  id: "unsloth/Qwen3-8B-Instruct-GGUF",
  pipeline_tag: "text-generation",
  tags: ["gguf", "qwen3", "text-generation", "unsloth"],
  library_name: "gguf",
  downloads: 50_000,
  siblings: [
    { rfilename: "Qwen3-8B-Instruct-Q2_K.gguf", size: 3_000 * 1024 * 1024 },
    { rfilename: "Qwen3-8B-Instruct-Q4_K_M.gguf", size: 5_000 * 1024 * 1024 },
    { rfilename: "Qwen3-8B-Instruct-Q8_0.gguf", size: 9_000 * 1024 * 1024 },
  ],
};

const HUGE_GGUF: HubRawModel = {
  id: "unsloth/Huge-70B-GGUF",
  tags: ["gguf", "llama"],
  library_name: "gguf",
  siblings: [{ rfilename: "Huge-70B-Q8_0.gguf", size: 80_000 * 1024 * 1024 }],
};

const SAFETENSORS_ONLY: HubRawModel = {
  id: "Qwen/Qwen2.5-0.5B-Instruct",
  tags: ["safetensors", "text-generation", "instruct"],
  siblings: [{ rfilename: "model.safetensors", size: 1_000 * 1024 * 1024 }],
};

const DIFFUSION: HubRawModel = {
  id: "someone/sd-gguf",
  pipeline_tag: "text-to-image",
  tags: ["gguf", "diffusers"],
};

test("buildGgufHubModelsUrl uses official API with filter=gguf", () => {
  const url = buildGgufHubModelsUrl("unsloth");
  assert.equal(url.origin + url.pathname, HF_HUB_MODELS_API);
  assert.equal(url.searchParams.get("filter"), "gguf");
  assert.equal(url.searchParams.get("search"), "unsloth");
  assert.ok(url.searchParams.getAll("expand").includes("siblings"));
  assert.ok(url.searchParams.getAll("expand").includes("gated"));
});

test("ggufCatalogSearchTerms seeds unsloth and families", () => {
  assert.ok(GGUF_HUB_SEARCHES.includes("unsloth"));
  const terms = ggufCatalogSearchTerms("");
  assert.ok(terms.includes("unsloth"));
  assert.ok(terms.includes("qwen"));
  assert.ok(ggufCatalogSearchTerms("deepseek").includes("deepseek"));
});

test("isGgufHubCandidate keeps GGUF and drops safetensors-only / diffusion", () => {
  assert.equal(isGgufHubCandidate(UNSLOTH_GGUF), true);
  assert.equal(isGgufHubCandidate(SAFETENSORS_ONLY), false);
  assert.equal(isGgufHubCandidate(DIFFUSION), false);
});

test("pickPreferredGgufFile prefers Q4_K_M within budget", () => {
  const files: GgufFileHint[] = [
    { filename: "a-Q2_K.gguf", sizeMiB: 3_000, sizeLabel: "~3 GB" },
    { filename: "a-Q4_K_M.gguf", sizeMiB: 5_000, sizeLabel: "~5 GB" },
    { filename: "a-Q8_0.gguf", sizeMiB: 9_000, sizeLabel: "~9 GB" },
  ];
  assert.equal(quantRank("a-Q4_K_M.gguf") > quantRank("a-Q8_0.gguf"), true);
  assert.equal(pickPreferredGgufFile(files, 25_600)?.filename, "a-Q4_K_M.gguf");
});

test("ggufFitBudgetMiB uses single-card VRAM on dual B70, not the sum", () => {
  const hw = fakeIntelHardware();
  const budget = ggufFitBudgetMiB(hw);
  assert.equal(Math.round(budget), Math.round(32_000 * HUB_FIT_FRACTION));
  assert.ok(budget < 40_000);
});

test("listGgufHubModels returns unsloth GGUF and marks huge as too big", async () => {
  const hw = fakeIntelHardware();
  const fetchFn = (async (input: RequestInfo | URL) => {
    const href = String(input instanceof Request ? input.url : input);
    assert.ok(href.startsWith(HF_HUB_MODELS_API));
    assert.ok(href.includes("filter=gguf"));
    return jsonResponse([UNSLOTH_GGUF, HUGE_GGUF, SAFETENSORS_ONLY, DIFFUSION]);
  }) as typeof fetch;
  const result = await listGgufHubModels({ q: "unsloth", fetchFn, hardware: hw, ggufDir: "/tmp/gguf-test-none" });
  assert.equal(result.offline, false);
  assert.ok(result.models.some((m) => m.id === UNSLOTH_GGUF.id));
  assert.equal(
    result.models.some((m) => m.id === SAFETENSORS_ONLY.id),
    false,
  );
  const small = result.models.find((m) => m.id === UNSLOTH_GGUF.id);
  assert.equal(small?.preferredFile?.filename, "Qwen3-8B-Instruct-Q4_K_M.gguf");
  assert.equal(small?.fits, true);
  const huge = result.models.find((m) => m.id === HUGE_GGUF.id);
  assert.equal(huge?.likelyTooBig, true);
  assert.match(result.hint, /llama\.cpp|GGUF/i);
});

test("toGgufHubModel works when siblings are missing (list API gap)", () => {
  const row: HubRawModel = {
    id: "unsloth/Foo-GGUF",
    tags: ["gguf"],
    library_name: "gguf",
  };
  const model = toGgufHubModel(row, 25_000, "/tmp/none");
  assert.ok(model);
  assert.equal(model.sizeUnknown, true);
  assert.equal(model.fits, true);
});

test("GGUF Hub gated:manual propagates to list row", async () => {
  const hw = fakeIntelHardware();
  const gatedRow: HubRawModel = {
    id: "meta-llama/Llama-3.2-1B-Instruct-GGUF",
    gated: "manual",
    tags: ["gguf", "llama"],
    library_name: "gguf",
    siblings: [{ rfilename: "model-Q4_K_M.gguf", size: 800 * 1024 * 1024 }],
  };
  const fetchFn = (async () => jsonResponse([gatedRow, UNSLOTH_GGUF])) as typeof fetch;
  const result = await listGgufHubModels({ q: "", fetchFn, hardware: hw, ggufDir: "/tmp/gguf-gated-test" });
  const hit = result.models.find((m) => m.id === gatedRow.id);
  assert.equal(hit?.gated, true);
  const unsloth = result.models.find((m) => m.id === UNSLOTH_GGUF.id);
  assert.equal(unsloth?.gated, false);
});

test("GGUF list soft-fails bad rows and never needs config.json", async () => {
  const hw = fakeIntelHardware();
  const bad = { id: null, tags: ["gguf"] } as unknown as HubRawModel;
  const fetchFn = (async () => jsonResponse([bad, UNSLOTH_GGUF])) as typeof fetch;
  const result = await listGgufHubModels({ q: "unsloth", fetchFn, hardware: hw, ggufDir: "/tmp/gguf-soft" });
  assert.ok(result.models.some((m) => m.id === UNSLOTH_GGUF.id));
  assert.equal(result.offline, false);
});
