import assert from "node:assert/strict";
import { test } from "node:test";
import { fitModel, recommendModels, vramNeededMiB } from "../src/local-models/fit.js";
import { LOCAL_MODEL_CATALOG, findCatalogModel } from "../src/local-models/catalog.js";
import type { CatalogModel } from "../src/local-models/catalog.js";
import type { Accelerator, HardwareSnapshot, LaunchBackend } from "../src/hardware.js";
import {
  detectHardware,
  parseClinfoGpus,
  parseLspciDisplayDevices,
  parseNvidiaSmiCsv,
  parseWin32VideoControllers,
} from "../src/hardware.js";

function acceleratorsFrom(
  vendor: Accelerator["vendor"],
  vrams: number[],
  name: string,
): Accelerator[] {
  return vrams.map((vramMiB, index) => ({
    vendor,
    name,
    vramMiB,
    index,
    source: "test",
  }));
}

function snapshot(opts: {
  vendor?: Accelerator["vendor"];
  backend?: LaunchBackend;
  vrams: number[];
  name?: string;
}): HardwareSnapshot {
  const vendor = opts.vendor ?? "nvidia";
  const acc = acceleratorsFrom(vendor, opts.vrams, opts.name ?? "Test GPU");
  const backend = opts.backend ?? (vendor === "intel" ? "intel-xpu" : vendor === "amd" ? "rocm" : "cuda");
  const vramMiB = opts.vrams.length ? Math.max(...opts.vrams) : 0;
  const minVramMiB = opts.vrams.length ? Math.min(...opts.vrams) : 0;
  return {
    accelerators: acc,
    primaryBackend: backend,
    totalVramMiB: opts.vrams.reduce((a, b) => a + b, 0),
    deviceCount: acc.length,
    gpus: acc.map((row) => ({
      index: row.index,
      name: row.name,
      vramMiB: row.vramMiB,
      vendor: row.vendor,
      source: row.source,
    })),
    vramMiB,
    minVramMiB,
    ramMiB: 64_000,
    cpuCount: 16,
    hasNvidiaSmi: vendor === "nvidia",
    constrained: backend === "cpu",
    notes: [],
  };
}

function gpu(vramMiB: number): HardwareSnapshot {
  return snapshot({ vendor: "nvidia", backend: "cuda", vrams: [vramMiB], name: "Test GPU" });
}

function cpuOnly(): HardwareSnapshot {
  return snapshot({ vendor: "nvidia", backend: "cpu", vrams: [], name: "none" });
}

function intelDual(vramMiB: number): HardwareSnapshot {
  return snapshot({
    vendor: "intel",
    backend: "intel-xpu",
    vrams: [vramMiB, vramMiB],
    name: "Intel Arc Pro B70",
  });
}

function model(partial: Partial<CatalogModel> & Pick<CatalogModel, "id" | "weightsMiB" | "quantization">): CatalogModel {
  return {
    name: partial.id,
    hfRepo: `org/${partial.id}`,
    family: partial.family ?? partial.id,
    lineage: "test",
    generation: 1,
    paramsB: partial.paramsB ?? 7,
    sizeClass: partial.sizeClass ?? "small",
    cpuFeasible: partial.cpuFeasible ?? false,
    gated: false,
    ...partial,
    lineage: partial.lineage ?? "test",
    generation: partial.generation ?? 1,
  };
}

test("vramNeededMiB applies 20% KV cache headroom", () => {
  assert.equal(vramNeededMiB(10_000), 12_000);
  assert.equal(vramNeededMiB(5_000), 6_000);
});

test("8B fp16 does not fit 8GB; 8B 4-bit does", () => {
  const fp16 = model({ id: "8b-fp16", family: "8b", paramsB: 8, weightsMiB: 16_000, quantization: "fp16" });
  const awq = model({
    id: "8b-awq",
    family: "8b",
    paramsB: 8,
    weightsMiB: 5_500,
    quantization: "awq",
    vllmQuantization: "awq",
  });
  const eightGb = gpu(8192);
  assert.equal(fitModel(fp16, eightGb).fits, false);
  assert.equal(fitModel(awq, eightGb).fits, true);
  assert.match(fitModel(fp16, eightGb).reason, /4-bit/);
});

test("7B fp16 is tight on 16GB; 7B AWQ fits", () => {
  const fp16 = findCatalogModel("qwen2.5-7b-instruct");
  const awq = findCatalogModel("qwen2.5-7b-instruct-awq");
  assert.ok(fp16 && awq);
  const sixteen = gpu(16_384);
  assert.equal(fitModel(fp16, sixteen).fits, false);
  assert.equal(fitModel(awq, sixteen).fits, true);
});

test("70B only fits 4-bit on 48GB+", () => {
  const fp16 = findCatalogModel("llama-3.1-70b-instruct");
  const awq = findCatalogModel("llama-3.1-70b-instruct-awq");
  assert.ok(fp16 && awq);
  const twentyFour = gpu(24_576);
  const fortyEight = gpu(49_152);
  assert.equal(fitModel(fp16, fortyEight).fits, false);
  assert.equal(fitModel(awq, twentyFour).fits, false);
  assert.equal(fitModel(awq, fortyEight).fits, true);
});

test("no GPU only tiny CPU-feasible models fit", () => {
  const tiny = findCatalogModel("qwen2.5-0.5b-instruct");
  const seven = findCatalogModel("qwen2.5-7b-instruct");
  assert.ok(tiny && seven);
  assert.equal(fitModel(tiny, cpuOnly()).fits, true);
  assert.equal(fitModel(seven, cpuOnly()).fits, false);
  assert.match(fitModel(seven, cpuOnly()).reason, /No discrete accelerator/);
  assert.doesNotMatch(fitModel(seven, cpuOnly()).reason, /No NVIDIA GPU/);
  const rec = recommendModels(cpuOnly(), LOCAL_MODEL_CATALOG);
  assert.equal(rec.length, LOCAL_MODEL_CATALOG.length);
  assert.ok(rec.some((entry) => entry.model.id === "qwen2.5-0.5b-instruct" && entry.fit.fits && !entry.newest));
  assert.ok(rec.some((entry) => entry.model.id === "qwen2.5-7b-instruct" && !entry.fit.fits));
  assert.ok(rec.filter((entry) => entry.fit.fits).every((entry) => entry.model.cpuFeasible));
});

test("recommend prefers fp16 when it fits, else 4-bit of the same family", () => {
  const rec24 = recommendModels(gpu(24_576));
  assert.equal(rec24.length, LOCAL_MODEL_CATALOG.length);
  const llama = rec24.find((entry) => entry.model.lineage === "llama" && entry.fit.fits && entry.newest);
  assert.ok(llama);
  assert.equal(llama.model.quantization, "fp16");
  assert.ok(llama.model.generation >= 3.1);

  const rec8 = recommendModels(gpu(8192));
  assert.equal(rec8.length, LOCAL_MODEL_CATALOG.length);
  const llama8 = rec8.find((entry) => entry.model.lineage === "llama" && entry.fit.fits);
  assert.ok(llama8);
  const seventyAwq = rec8.find((entry) => entry.model.id === "llama-3.1-70b-instruct-awq");
  assert.ok(seventyAwq);
  assert.equal(seventyAwq.fit.fits, false);
});

test("recommendations list the full catalog and mark newest Hub ids", () => {
  const catalog: CatalogModel[] = [
    model({ id: "old-s", family: "old-s", lineage: "demo", generation: 1, paramsB: 3, weightsMiB: 4_000, quantization: "fp16" }),
    model({ id: "old-l", family: "old-l", lineage: "demo", generation: 1, paramsB: 22, weightsMiB: 40_000, quantization: "fp16" }),
    model({ id: "new-s", family: "new-s", lineage: "demo", generation: 2, paramsB: 4, weightsMiB: 6_000, quantization: "fp16" }),
    model({ id: "new-m", family: "new-m", lineage: "demo", generation: 2, paramsB: 24, weightsMiB: 16_000, quantization: "fp16" }),
    model({ id: "other", family: "other", lineage: "other", generation: 1, paramsB: 7, weightsMiB: 14_000, quantization: "fp16" }),
  ];
  const rec = recommendModels(gpu(24_576), catalog);
  assert.equal(rec.length, catalog.length);
  assert.deepEqual(
    rec.map((entry) => entry.model.id).sort(),
    ["new-m", "new-s", "old-l", "old-s", "other"].sort(),
  );
  assert.equal(rec.find((entry) => entry.model.id === "new-s")?.newest, true);
  assert.equal(rec.find((entry) => entry.model.id === "old-s")?.newest, false);
  assert.equal(rec.find((entry) => entry.model.id === "old-l")?.fit.fits, false);
});

test("if the newest generation does not fit, it stays listed as too big", () => {
  const catalog: CatalogModel[] = [
    model({ id: "v1", family: "v1", lineage: "demo", generation: 1, paramsB: 8, weightsMiB: 4_000, quantization: "fp16" }),
    model({ id: "v2", family: "v2", lineage: "demo", generation: 2, paramsB: 9, weightsMiB: 80_000, quantization: "fp16" }),
  ];
  const rec = recommendModels(gpu(24_576), catalog);
  assert.equal(rec.length, 2);
  assert.equal(rec.find((entry) => entry.model.id === "v2")?.newest, true);
  assert.equal(rec.find((entry) => entry.model.id === "v2")?.fit.fits, false);
  assert.equal(rec.find((entry) => entry.model.id === "v1")?.fit.fits, true);
  assert.equal(rec.find((entry) => entry.model.id === "v1")?.newest, false);
});

test("Intel-only dual Arc does not recommend tiny CPU-feasible models", () => {
  const tiny = findCatalogModel("qwen2.5-0.5b-instruct");
  const seven = findCatalogModel("qwen2.5-7b-instruct");
  const sevenAwq = findCatalogModel("qwen2.5-7b-instruct-awq");
  assert.ok(tiny && seven && sevenAwq);

  const b70 = intelDual(31_023);
  assert.equal(b70.primaryBackend, "intel-xpu");
  assert.equal(fitModel(seven, b70).fits, true);
  assert.equal(fitModel(seven, b70).parallel, 1);
  assert.equal(fitModel(sevenAwq, b70).fits, false);
  assert.match(fitModel(sevenAwq, b70).reason, /Intel XPU|fp16/);
  const recB70 = recommendModels(b70);
  assert.equal(recB70.length, LOCAL_MODEL_CATALOG.length);
  assert.ok(recB70.some((entry) => entry.model.cpuFeasible));
  assert.ok(recB70.some((entry) => entry.model.paramsB >= 7 && entry.model.quantization === "fp16" && entry.fit.fits));
  assert.ok(recB70.some((entry) => entry.model.quantization === "awq" && !entry.fit.fits));

  const dual12 = intelDual(12_288);
  assert.equal(fitModel(seven, dual12).fits, true);
  assert.equal(fitModel(seven, dual12).parallel, 2);
  const rec12 = recommendModels(dual12);
  assert.equal(rec12.length, LOCAL_MODEL_CATALOG.length);
  assert.ok(rec12.some((entry) => entry.fit.fits && !entry.model.cpuFeasible && entry.model.paramsB >= 4));
  assert.match(fitModel(tiny, dual12).reason, /optional|Fits on one/i);
});

test("14B fp16 on dual B70 uses tensor parallel; 32B fp16 still does not fit; 27B Gemma 2 does", () => {
  const fourteen = findCatalogModel("qwen2.5-14b-instruct");
  const thirtyTwo = findCatalogModel("qwen2.5-32b-instruct");
  const gemma27 = findCatalogModel("gemma-2-27b-it");
  const gemma4_26 = findCatalogModel("gemma-4-26b-a4b-it");
  const qwen38 = findCatalogModel("qwen3.8-27b");
  assert.ok(fourteen && thirtyTwo && gemma27 && gemma4_26 && qwen38);
  const b70 = intelDual(31_023);
  const f14 = fitModel(fourteen, b70);
  assert.equal(f14.fits, true);
  assert.equal(f14.parallel, 2);
  assert.equal(fitModel(thirtyTwo, b70).fits, false);
  const g27 = fitModel(gemma27, b70);
  assert.equal(g27.fits, true);
  assert.equal(g27.parallel, 2);
  assert.equal(fitModel(gemma4_26, b70).fits, true);
  assert.equal(fitModel(qwen38, b70).fits, true);
  const rec = recommendModels(b70);
  assert.equal(rec.length, LOCAL_MODEL_CATALOG.length);
  assert.ok(LOCAL_MODEL_CATALOG.length > 40, "full catalog is not a 2-row slice");
  assert.ok(rec.some((entry) => entry.model.family.startsWith("gemma-2") && !entry.newest));
  assert.ok(rec.some((entry) => entry.model.family.startsWith("gemma-4") && entry.newest));
  assert.ok(rec.some((entry) => entry.model.id === "qwen3.8-27b" && entry.newest && entry.fit.fits));
  assert.equal(fitModel(gemma4_26, b70).kind, "needs_tp");
  assert.ok(rec.some((entry) => entry.fit.fits && entry.model.paramsB >= 24));
});

test("catalog includes Gemma 4, Qwen3.8, Llama 4, and older gens remain downloadable", () => {
  const gemma9 = findCatalogModel("gemma-2-9b-it");
  const gemma27 = findCatalogModel("google/gemma-2-27b-it");
  const gemma3 = findCatalogModel("google/gemma-3-4b-it");
  const gemma4 = findCatalogModel("google/gemma-4-E4B-it");
  const llama32 = findCatalogModel("llama-3.2-3b-instruct");
  const qwen332 = findCatalogModel("qwen3-32b");
  const qwen38 = findCatalogModel("Qwen/Qwen3.8-27B");
  const qwen35 = findCatalogModel("qwen3.5-9b");
  const qwen3 = findCatalogModel("qwen3-8b");
  const phi4 = findCatalogModel("microsoft/phi-4");
  const olmo = findCatalogModel("olmo-2-13b-instruct");
  const olmo3 = findCatalogModel("olmo-3-7b-instruct");
  const granite = findCatalogModel("granite-3.3-8b-instruct");
  const granite42 = findCatalogModel("granite-4.2-8b");
  const r1 = findCatalogModel("deepseek-r1-distill-qwen-14b");
  const r10528 = findCatalogModel("deepseek-r1-0528-qwen3-8b");
  const scout = findCatalogModel("llama-4-scout-instruct");
  const llama33 = findCatalogModel("llama-3.3-70b-instruct");
  const small32 = findCatalogModel("mistral-small-3.2-24b-instruct");
  assert.ok(gemma9 && gemma27 && gemma3 && gemma4 && llama32 && qwen332 && qwen38 && qwen35 && qwen3 && phi4 && olmo && olmo3 && granite && granite42 && r1 && r10528 && scout && llama33 && small32);
  assert.equal(gemma9.gated, true);
  assert.equal(gemma4.gated, false);
  assert.equal(gemma4.generation, 4);
  assert.equal(qwen38.generation, 3.8);
  assert.equal(qwen3.gated, false);
  assert.equal(phi4.gated, false);
  assert.equal(findCatalogModel("qwen3-8b-awq")?.quantization, "awq");
  const llama8 = findCatalogModel("llama-3.1-8b-instruct");
  const llama8awq = findCatalogModel("llama-3.1-8b-instruct-awq");
  const mistral = findCatalogModel("mistral-7b-instruct");
  assert.equal(llama8?.gated, true);
  assert.match(llama8?.notes ?? "", /Llama Community License/);
  assert.equal(llama8awq?.gated, false);
  assert.match(llama8awq?.notes ?? "", /Llama Community License/);
  assert.equal(mistral?.gated, true);
  assert.equal(small32.gated, false);
  assert.equal(scout.gated, true);
  assert.match(gemma9.notes ?? "", /Gemma Terms of Use/);
  assert.match(gemma4.notes ?? "", /Apache-2\.0/);
  assert.equal(LOCAL_MODEL_CATALOG.every((row) => row.lineage.length > 0 && Number.isFinite(row.generation)), true);
});

test("Gemma 2 9B fp16 fits a 24GB NVIDIA card; 27B needs 80GB-class or two ~32GB GPUs", () => {
  const nine = findCatalogModel("gemma-2-9b-it");
  const twentySeven = findCatalogModel("gemma-2-27b-it");
  const qwen3 = findCatalogModel("qwen3-8b");
  const qwen3Awq = findCatalogModel("qwen3-8b-awq");
  const phi4 = findCatalogModel("phi-4");
  assert.ok(nine && twentySeven && qwen3 && qwen3Awq && phi4);
  const rtx4090 = gpu(24_576);
  const h100 = gpu(81_920);
  const eightGb = gpu(8192);
  assert.equal(fitModel(nine, rtx4090).fits, true);
  assert.equal(fitModel(nine, rtx4090).parallel, 1);
  assert.equal(fitModel(qwen3, rtx4090).fits, true);
  assert.equal(fitModel(phi4, rtx4090).fits, false);
  assert.equal(fitModel(twentySeven, rtx4090).fits, false);
  assert.equal(fitModel(twentySeven, h100).fits, true);
  assert.equal(fitModel(qwen3Awq, eightGb).fits, true);
  assert.equal(fitModel(nine, eightGb).fits, false);
  assert.equal(fitModel(twentySeven, intelDual(31_023)).parallel, 2);

  const rec24 = recommendModels(rtx4090);
  assert.equal(rec24.length, LOCAL_MODEL_CATALOG.length);
  assert.ok(rec24.some((entry) => entry.model.family.startsWith("gemma-4") && entry.newest));
  assert.ok(rec24.some((entry) => entry.model.family.startsWith("gemma-2") && !entry.newest));
  assert.ok(rec24.some((entry) => (entry.model.family.startsWith("qwen3.5") || entry.model.family.startsWith("qwen3.8")) && entry.newest));
});

test("dual NVIDIA 24GB: models that miss one card but fit two use tensor parallel", () => {
  const eight = findCatalogModel("llama-3.1-8b-instruct");
  const fourteen = findCatalogModel("qwen2.5-14b-instruct");
  const seventyAwq = findCatalogModel("llama-3.1-70b-instruct-awq");
  const gemma27 = findCatalogModel("gemma-2-27b-it");
  assert.ok(eight && fourteen && seventyAwq && gemma27);
  const dual24 = snapshot({ vendor: "nvidia", backend: "cuda", vrams: [24_576, 24_576], name: "NVIDIA GeForce RTX 4090" });
  assert.equal(fitModel(eight, dual24).fits, true);
  assert.equal(fitModel(eight, dual24).parallel, 1);
  assert.equal(fitModel(fourteen, dual24).fits, true);
  assert.equal(fitModel(fourteen, dual24).parallel, 2);
  assert.equal(fitModel(seventyAwq, dual24).fits, true);
  assert.equal(fitModel(seventyAwq, dual24).parallel, 2);
  assert.equal(fitModel(gemma27, dual24).fits, false);
});

test("dual AMD 24GB: 14B fp16 uses tensor parallel; 8B stays on one GPU", () => {
  const eight = findCatalogModel("qwen2.5-7b-instruct");
  const fourteen = findCatalogModel("qwen2.5-14b-instruct");
  const awq = findCatalogModel("qwen2.5-7b-instruct-awq");
  assert.ok(eight && fourteen && awq);
  const dual = snapshot({ vendor: "amd", backend: "rocm", vrams: [24_576, 24_576], name: "Radeon RX 7900 XTX" });
  assert.equal(fitModel(eight, dual).fits, true);
  assert.equal(fitModel(eight, dual).parallel, 1);
  assert.equal(fitModel(fourteen, dual).fits, true);
  assert.equal(fitModel(fourteen, dual).parallel, 2);
  assert.equal(fitModel(awq, dual).fits, true);
  assert.equal(fitModel(awq, dual).parallel, 1);
});

test("AWQ Gemma/Qwen3 variants are CUDA/ROCm-only; fp16 is used on Intel XPU", () => {
  const gemma9 = findCatalogModel("gemma-2-9b-it");
  const qwen3Awq = findCatalogModel("qwen3-8b-awq");
  assert.ok(gemma9 && qwen3Awq);
  const xpu = intelDual(31_023);
  const rocm = snapshot({ vendor: "amd", backend: "rocm", vrams: [24_576], name: "Radeon RX 7900 XTX" });
  assert.equal(fitModel(gemma9, xpu).fits, true);
  assert.equal(fitModel(qwen3Awq, xpu).fits, false);
  assert.equal(fitModel(qwen3Awq, rocm).fits, true);
});

test("AMD-only uses ROCm VRAM and can take AWQ", () => {
  const seven = findCatalogModel("qwen2.5-7b-instruct");
  const awq = findCatalogModel("qwen2.5-7b-instruct-awq");
  assert.ok(seven && awq);
  const amd = snapshot({ vendor: "amd", backend: "rocm", vrams: [24_576], name: "Radeon RX 7900 XTX" });
  assert.equal(fitModel(seven, amd).fits, true);
  assert.equal(fitModel(awq, amd).fits, true);
  const rec = recommendModels(amd);
  assert.equal(rec.length, LOCAL_MODEL_CATALOG.length);
  assert.ok(rec.some((entry) => entry.model.paramsB >= 7 && entry.fit.fits));
  assert.ok(rec.some((entry) => !entry.model.cpuFeasible));
});

test("NVIDIA-only mock lspci/nvidia-smi is cuda, not CPU", () => {
  const hw = detectHardware({
    probes: {
      nvidiaSmi: "NVIDIA GeForce RTX 4090, 24576, 560.35",
      lspci:
        "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD102 [GeForce RTX 4090] [10de:2684]\n",
    },
  });
  assert.equal(hw.primaryBackend, "cuda");
  assert.equal(hw.constrained, false);
  assert.equal(hw.deviceCount, 1);
  assert.equal(hw.accelerators[0]?.vendor, "nvidia");
  assert.equal(hw.vramMiB, 24_576);
  assert.ok(!hw.notes.join(" ").includes("No NVIDIA GPU"));
});

test("Intel-only mock lspci+clinfo detects two Arc GPUs and VRAM", () => {
  const hw = detectHardware({
    probes: {
      nvidiaSmi: null,
      lspci: [
        "00:02.0 VGA compatible controller [0300]: Intel Corporation Arrow Lake-S [Intel Graphics] [8086:7d67] (rev 06)",
        "04:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]",
        "08:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]",
      ].join("\n"),
      clinfo: [
        "  Device Name                                     Intel(R) Arc(TM) Pro B70 Graphics",
        "  Global memory size                              32530182144 (30.3GiB)",
        "  Device Name                                     Intel(R) Arc(TM) Pro B70 Graphics",
        "  Global memory size                              32530182144 (30.3GiB)",
        "  Device Name                                     Intel(R) Graphics",
        "  Global memory size                              60910755840 (56.73GiB)",
      ].join("\n"),
      sysfsCards: [
        { name: "card0", vendorId: "8086", deviceId: "e223", driver: "xe" },
        { name: "card1", vendorId: "8086", deviceId: "7d67", driver: "i915" },
        { name: "card2", vendorId: "8086", deviceId: "e223", driver: "xe" },
      ],
    },
  });
  assert.equal(hw.primaryBackend, "intel-xpu");
  assert.equal(hw.constrained, false);
  assert.equal(hw.deviceCount, 2);
  assert.equal(hw.accelerators.length, 2);
  assert.ok(hw.accelerators.every((row) => row.vendor === "intel"));
  assert.ok(hw.accelerators.every((row) => /B70|Arc/i.test(row.name)));
  assert.ok(hw.accelerators[0]!.vramMiB > 20_000);
  assert.equal(hw.accelerators[0]!.vramEstimated, false);
  assert.ok(hw.totalVramMiB > 40_000);
  const rec = recommendModels(hw);
  assert.equal(rec.length, LOCAL_MODEL_CATALOG.length);
  assert.ok(rec.some((entry) => entry.fit.fits && !entry.model.cpuFeasible));
  const gemma27 = findCatalogModel("gemma-2-27b-it");
  assert.ok(gemma27);
  assert.equal(fitModel(gemma27, hw).fits, true);
  assert.equal(fitModel(gemma27, hw).parallel, 2);
  assert.doesNotMatch(hw.notes.join(" "), /No NVIDIA GPU/);
});

test("Intel SKU fallback is labeled estimated when VRAM cannot be read", () => {
  const hw = detectHardware({
    probes: {
      nvidiaSmi: null,
      lspci: "04:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]\n",
      sysfsCards: [{ name: "card0", vendorId: "8086", deviceId: "e223", driver: "xe" }],
    },
  });
  assert.equal(hw.primaryBackend, "intel-xpu");
  assert.equal(hw.deviceCount, 1);
  assert.equal(hw.vramMiB, 32_768);
  assert.equal(hw.accelerators[0]?.vramEstimated, true);
  assert.match(hw.accelerators[0]?.source ?? "", /sku-fallback/);
});

test("AMD-only mock sysfs is rocm", () => {
  const hw = detectHardware({
    probes: {
      nvidiaSmi: null,
      lspci:
        "03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] [1002:744c]\n",
      sysfsCards: [
        { name: "card0", vendorId: "1002", deviceId: "744c", driver: "amdgpu", vramBytes: 24_576 * 1024 * 1024 },
      ],
    },
  });
  assert.equal(hw.primaryBackend, "rocm");
  assert.equal(hw.constrained, false);
  assert.equal(hw.deviceCount, 1);
  assert.equal(hw.accelerators[0]?.vendor, "amd");
  assert.equal(hw.vramMiB, 24_576);
});

test("no GPU mock is cpu constrained", () => {
  const hw = detectHardware({
    probes: {
      nvidiaSmi: null,
      lspci: "00:1f.0 ISA bridge [0601]: Intel Corporation Device [8086:ae0d]\n",
      sysfsCards: [],
    },
  });
  assert.equal(hw.primaryBackend, "cpu");
  assert.equal(hw.constrained, true);
  assert.equal(hw.deviceCount, 0);
  assert.equal(hw.accelerators.length, 0);
});

test("parse helpers: lspci Intel pair, clinfo B70 memory, nvidia-smi csv", () => {
  const pci = parseLspciDisplayDevices(
    [
      "0000:00:02.0 VGA compatible controller [0300]: Intel Corporation Arrow Lake-S [Intel Graphics] [8086:7d67]",
      "0000:04:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]",
      "0000:08:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]",
    ].join("\n"),
  );
  assert.equal(pci.filter((row) => row.discrete).length, 2);
  assert.equal(pci.filter((row) => !row.discrete).length, 1);
  const cl = parseClinfoGpus(
    [
      "  Device Name                                     Intel(R) Arc(TM) Pro B70 Graphics",
      "  Global memory size                              32530182144 (30.3GiB)",
      "    Device Name                                   Intel(R) Arc(TM) Pro B70 Graphics",
      "    Global memory size                            32530182144 (30.3GiB)",
    ].join("\n"),
  );
  assert.equal(cl.length, 1);
  assert.ok(cl[0]!.vramMiB > 30_000);
  const nv = parseNvidiaSmiCsv("NVIDIA A100-SXM4-80GB, 81920, 535.104.05\n");
  assert.equal(nv[0]?.vramMiB, 81_920);
});

test("detectHardware reports RAM and CPU; accelerators when present", () => {
  const hw = detectHardware();
  assert.ok(hw.cpuCount >= 1);
  assert.ok(hw.ramMiB > 0);
  assert.ok(["intel-xpu", "cuda", "rocm", "cpu"].includes(hw.primaryBackend));
  if (hw.primaryBackend === "cpu") {
    assert.equal(hw.constrained, true);
    assert.equal(hw.deviceCount, 0);
  } else {
    assert.equal(hw.constrained, false);
    assert.ok(hw.deviceCount >= 1);
    assert.ok((hw.accelerators[0]?.vramMiB ?? 0) > 0);
    assert.ok((hw.accelerators[0]?.name ?? "").length > 0);
    assert.doesNotMatch(hw.notes.join(" "), /No NVIDIA GPU/);
  }
});

test("parseWin32VideoControllers reads Name|AdapterRAM|PNPDeviceID and skips iGPU", () => {
  const devices = parseWin32VideoControllers(
    [
      "NVIDIA GeForce RTX 4090|4293918720|PCI\\VEN_10DE&DEV_2684&SUBSYS_00000000\\0",
      "Intel(R) Arc(TM) Pro B70 Graphics|34359738368|PCI\\VEN_8086&DEV_E223&SUBSYS_00000000\\0",
      "Intel(R) Graphics|1073741824|PCI\\VEN_8086&DEV_7D67&SUBSYS_00000000\\0",
    ].join("\n"),
  );
  assert.equal(devices.filter((row) => row.discrete).length, 2);
  assert.equal(devices.filter((row) => !row.discrete).length, 1);
  assert.equal(devices.find((row) => row.vendor === "nvidia")?.deviceId, "2684");
  assert.equal(devices.find((row) => row.vendor === "intel" && row.discrete)?.deviceId, "e223");
});

test("Windows Win32 + nvidia-smi detect CUDA without lspci/sysfs", () => {
  const hw = detectHardware({
    probes: {
      nvidiaSmi: "NVIDIA GeForce RTX 4090, 24576, 560.35",
      lspci: null,
      sysfsCards: [],
      win32Video:
        "NVIDIA GeForce RTX 4090|4293918720|PCI\\VEN_10DE&DEV_2684&SUBSYS_00000000\\0\nIntel(R) Graphics|1073741824|PCI\\VEN_8086&DEV_7D67&SUBSYS_00000000\\0\n",
    },
  });
  assert.equal(hw.primaryBackend, "cuda");
  assert.equal(hw.constrained, false);
  assert.equal(hw.deviceCount, 1);
  assert.equal(hw.accelerators[0]?.vendor, "nvidia");
  assert.equal(hw.vramMiB, 24_576);
  assert.ok(!hw.notes.join(" ").includes("lspci"));
});

test("Windows Win32 Intel Arc uses SKU VRAM without sysfs", () => {
  const hw = detectHardware({
    probes: {
      nvidiaSmi: null,
      lspci: null,
      sysfsCards: [],
      win32Video: "Intel(R) Arc(TM) Pro B70 Graphics|0|PCI\\VEN_8086&DEV_E223&SUBSYS_00000000\\0\n",
    },
  });
  assert.equal(hw.primaryBackend, "intel-xpu");
  assert.equal(hw.deviceCount, 1);
  assert.equal(hw.vramMiB, 32_768);
  assert.equal(hw.accelerators[0]?.vramEstimated, true);
});
