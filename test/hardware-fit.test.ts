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
    paramsB: partial.paramsB ?? 7,
    sizeClass: partial.sizeClass ?? "small",
    cpuFeasible: partial.cpuFeasible ?? false,
    gated: false,
    ...partial,
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
  assert.ok(rec.length > 0);
  assert.ok(rec.every((entry) => entry.model.cpuFeasible));
});

test("recommend prefers fp16 when it fits, else 4-bit of the same family", () => {
  const rec24 = recommendModels(gpu(24_576));
  const llama8 = rec24.find((entry) => entry.model.family === "llama-3.1-8b");
  assert.ok(llama8);
  assert.equal(llama8.model.quantization, "fp16");

  const rec8 = recommendModels(gpu(8192));
  const llama8q = rec8.find((entry) => entry.model.family === "llama-3.1-8b");
  assert.ok(llama8q);
  assert.equal(llama8q.model.quantization, "awq");
  assert.equal(
    rec8.some((entry) => entry.model.id === "llama-3.1-70b-instruct-awq"),
    false,
  );
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
  assert.ok(recB70.length > 0);
  assert.equal(recB70.every((entry) => entry.model.cpuFeasible), false);
  assert.ok(recB70.some((entry) => entry.model.paramsB >= 7 && entry.model.quantization === "fp16"));
  assert.equal(
    recB70.some((entry) => entry.model.quantization === "awq"),
    false,
  );

  const dual12 = intelDual(12_288);
  assert.equal(fitModel(seven, dual12).fits, true);
  assert.equal(fitModel(seven, dual12).parallel, 2);
  const rec12 = recommendModels(dual12);
  assert.ok(rec12.length > 0);
  assert.ok(rec12.every((entry) => !entry.model.cpuFeasible));
  assert.ok((rec12[0]?.model.paramsB ?? 0) >= 7);
  assert.match(fitModel(tiny, dual12).reason, /optional|Fits on one/i);
});

test("14B fp16 on dual B70 uses tensor parallel; 32B fp16 still does not fit", () => {
  const fourteen = findCatalogModel("qwen2.5-14b-instruct");
  const thirtyTwo = findCatalogModel("qwen2.5-32b-instruct");
  assert.ok(fourteen && thirtyTwo);
  const b70 = intelDual(31_023);
  const f14 = fitModel(fourteen, b70);
  assert.equal(f14.fits, true);
  assert.equal(f14.parallel, 2);
  assert.equal(fitModel(thirtyTwo, b70).fits, false);
});

test("AMD-only uses ROCm VRAM and can take AWQ", () => {
  const seven = findCatalogModel("qwen2.5-7b-instruct");
  const awq = findCatalogModel("qwen2.5-7b-instruct-awq");
  assert.ok(seven && awq);
  const amd = snapshot({ vendor: "amd", backend: "rocm", vrams: [24_576], name: "Radeon RX 7900 XTX" });
  assert.equal(fitModel(seven, amd).fits, true);
  assert.equal(fitModel(awq, amd).fits, true);
  const rec = recommendModels(amd);
  assert.ok(rec.some((entry) => entry.model.paramsB >= 7));
  assert.equal(rec.every((entry) => entry.model.cpuFeasible), false);
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
  assert.ok(rec.length > 0);
  assert.equal(rec.every((entry) => entry.model.cpuFeasible), false);
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
