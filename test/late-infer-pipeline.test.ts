import assert from "node:assert/strict";
import { test } from "node:test";
import {
  lateInferCompileSpec,
  parseCompilePhase,
  resetLateInferCompileForTests,
  pullLateInfer,
} from "../src/local-servers/compile.js";
import { DISPLAY_GPU_VRAM_CAP } from "../src/local-servers/gpu-pick.js";
import { HUB_FIT_FRACTION, hardwareFitBudgetMiB } from "../src/local-servers/hub-catalog.js";
import type { HardwareSnapshot } from "../src/hardware.js";

const INTEL_LSPCI = [
  "00:02.0 VGA compatible controller [0300]: Intel Corporation Arrow Lake-S [Intel Graphics] [8086:7d67] (rev 06)",
  "04:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]",
  "08:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]",
].join("\n");

const INTEL_DRM = [
  {
    name: "card0",
    vendorId: "8086",
    deviceId: "e223",
    busId: "0000:04:00.0",
    driver: "xe",
    connectors: { "card0-DP-4": "connected" },
  },
  {
    name: "card1",
    vendorId: "8086",
    deviceId: "7d67",
    busId: "0000:00:02.0",
    driver: "i915",
    connectors: { "card1-DP-1": "disconnected" },
  },
  {
    name: "card2",
    vendorId: "8086",
    deviceId: "e223",
    busId: "0000:08:00.0",
    driver: "xe",
    connectors: { "card2-DP-8": "disconnected" },
  },
];

test("parseCompilePhase distinguishes probing / downloading / compiling / ready", () => {
  assert.equal(parseCompilePhase("probing Hub config"), "probing");
  assert.equal(parseCompilePhase("downloading Hub snapshot"), "downloading");
  assert.equal(parseCompilePhase("compiling OpenVINO IR"), "compiling");
  assert.equal(parseCompilePhase("ready — compiled on your computer"), "ready");
  assert.equal(parseCompilePhase("Error: config.json"), "error");
});

test("Intel OpenVINO compile env skips MLC preflight and pins idle BDF", () => {
  const probes = {
    lspci: INTEL_LSPCI,
    drmCards: INTEL_DRM,
    nvidiaSmi: null as string | null,
    intelRuntime: { present: true as const, kind: "openvino-genai" as const, python: "/opt/fake-ov/bin/python" },
  };
  const spec = lateInferCompileSpec("Qwen/Qwen2.5-0.5B-Instruct", { probes });
  assert.equal(spec.env.LATE_INFER_ACCEL, "intel");
  assert.equal(spec.env.LATE_INFER_PCI, "0000:08:00.0");
  assert.equal(spec.env.ZE_AFFINITY_MASK, "1");
  assert.equal(spec.env.LATE_INFER_SKIP_MLC_PREFLIGHT, "1");
  assert.equal(spec.env.LATE_INFER_SKIP_MLC, "1");
  assert.equal(spec.env.CUDA_VISIBLE_DEVICES, undefined);
});

test("display GPU budget applies 70% cap before 80% fit fraction", () => {
  const hw = {
    accelerators: [{ vendor: "intel", name: "B70", vramMiB: 32_768, index: 0, source: "t" }],
    primaryBackend: "intel-xpu",
    totalVramMiB: 32_768,
    deviceCount: 1,
    gpus: [{ index: 0, name: "B70", vramMiB: 32_768, vendor: "intel" }],
    vramMiB: 32_768,
    minVramMiB: 32_768,
    ramMiB: 64_000,
    cpuCount: 16,
    hasNvidiaSmi: false,
    constrained: false,
    notes: [],
  } as HardwareSnapshot;
  const idle = hardwareFitBudgetMiB(hw, { idleVramMiB: 32_768 });
  const display = hardwareFitBudgetMiB(hw, { idleVramMiB: 32_768, displayCap: DISPLAY_GPU_VRAM_CAP });
  assert.equal(Math.round(idle), Math.round(32_768 * HUB_FIT_FRACTION));
  assert.equal(Math.round(display), Math.round(32_768 * DISPLAY_GPU_VRAM_CAP * HUB_FIT_FRACTION));
  assert.ok(display < idle);
});

test("pullLateInfer starts in probing then soft-fails closed on Intel without OV runtime", async () => {
  resetLateInferCompileForTests({
    findBin: () => "/usr/bin/false",
    spawnFn: () => {
      throw new Error("must not spawn when blocked");
    },
    gpu: {
      probes: {
        lspci: INTEL_LSPCI,
        drmCards: INTEL_DRM,
        nvidiaSmi: null,
      },
    },
    gatedProbe: async () => {},
  });
  try {
    const job = await pullLateInfer({ model: "Qwen/Qwen2.5-0.5B-Instruct" });
    assert.equal(job.phase, "error");
    assert.match(job.error ?? "", /OpenVINO|Intel|not treat that card as NVIDIA/i);
  } finally {
    resetLateInferCompileForTests();
  }
});
