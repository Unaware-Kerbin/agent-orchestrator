import assert from "node:assert/strict";
import { test } from "node:test";
import {
  DISPLAY_GPU_VRAM_CAP,
  HOST_RAM_VS_VRAM,
  INTEL_IR_MISSING_START,
  classifyLateInferDevice,
  convertHostRamBlockedReason,
  detectGpuCards,
  engineGpuSpawnPlan,
  formatGpuVramPair,
  hostRamBlockedReason,
  lateInferCompileBlockedReason,
  lateInferStartBlockedReason,
  mergeGpuPlanEnv,
  ollamaGpuBlockedReason,
  parseClinfoDiscreteVram,
  parseDrmFdinfo,
  parseMeminfo,
  publicGpuPlan,
  resolveGpuServePhase,
  resolveLateInferGpuPlan,
  usedVramMiBFromFdinfo,
  vllmCudaOnIntelBlockedReason,
  vramMaxMiBForHubId,
} from "../src/local-servers/gpu-pick.js";
import { lateInferCompileSpec, pullLateInfer, resetLateInferCompileForTests } from "../src/local-servers/compile.js";
import { lateInferSpec as spawnLateInferSpec } from "../src/local-servers/spawn.js";

/** Dual Intel Battlemage + Arrow Lake iGPU (this computer’s topology). */
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
    connectors: { "card0-DP-4": "connected", "card0-DP-5": "connected", "card0-HDMI-A-4": "connected" },
    renderNode: "renderD129",
  },
  {
    name: "card1",
    vendorId: "8086",
    deviceId: "7d67",
    busId: "0000:00:02.0",
    driver: "i915",
    connectors: { "card1-DP-1": "disconnected" },
    renderNode: "renderD128",
  },
  {
    name: "card2",
    vendorId: "8086",
    deviceId: "e223",
    busId: "0000:08:00.0",
    driver: "xe",
    connectors: { "card2-DP-8": "disconnected", "card2-HDMI-A-6": "disconnected" },
    renderNode: "renderD130",
  },
];

const AMD_LSPCI = [
  "03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] [1002:744c]",
  "12:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] [1002:744c]",
].join("\n");

const AMD_VRAM_BYTES = 24 * 1024 * 1024 * 1024;

const AMD_DRM = [
  {
    name: "card0",
    vendorId: "1002",
    deviceId: "744c",
    busId: "0000:03:00.0",
    driver: "amdgpu",
    vramBytes: AMD_VRAM_BYTES,
    connectors: { "card0-DP-1": "connected", "card0-DP-2": "connected" },
  },
  {
    name: "card1",
    vendorId: "1002",
    deviceId: "744c",
    busId: "0000:12:00.0",
    driver: "amdgpu",
    vramBytes: AMD_VRAM_BYTES,
    connectors: { "card1-DP-1": "disconnected" },
  },
];

const AMD_ROCM_SMI = [
  "GPU[0]          : PCI Bus: 0000:03:00.0",
  "GPU[0]          : vram Total Memory (B): 25769803776",
  "GPU[1]          : PCI Bus: 0000:12:00.0",
  "GPU[1]          : vram Total Memory (B): 25769803776",
].join("\n");

const NVIDIA_LSPCI = [
  "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD102 [GeForce RTX 4090] [10de:2684]",
  "02:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD102 [GeForce RTX 4090] [10de:2684]",
].join("\n");

const NVIDIA_DRM = [
  {
    name: "card0",
    vendorId: "10de",
    deviceId: "2684",
    busId: "0000:01:00.0",
    driver: "nvidia",
    connectors: { "card0-DP-1": "connected", "card0-HDMI-A-1": "connected" },
  },
  {
    name: "card1",
    vendorId: "10de",
    deviceId: "2684",
    busId: "0000:02:00.0",
    driver: "nvidia",
    connectors: { "card1-DP-1": "disconnected", "card1-HDMI-A-1": "disconnected" },
  },
];

const NVIDIA_SMI_GPU0_DISPLAY = [
  "0, NVIDIA GeForce RTX 4090, 24564, 20000, 00000000:01:00.0, Enabled, Enabled",
  "1, NVIDIA GeForce RTX 4090, 24564, 24000, 00000000:02:00.0, Disabled, Disabled",
].join("\n");

function assertNoForeignToolkit(
  env: Record<string, string | undefined>,
  vendor: "intel" | "nvidia" | "amd",
): void {
  if (vendor === "intel" || vendor === "amd") {
    assert.equal(env.CUDA_VISIBLE_DEVICES, undefined);
  }
  if (vendor === "nvidia" || vendor === "amd") {
    assert.equal(env.ZE_AFFINITY_MASK, undefined);
    assert.equal(env.ONEAPI_DEVICE_SELECTOR, undefined);
    assert.equal(env.ZE_FLAT_DEVICE_HIERARCHY, undefined);
  }
  if (vendor === "intel" || vendor === "nvidia") {
    assert.equal(env.HIP_VISIBLE_DEVICES, undefined);
    assert.equal(env.ROCR_VISIBLE_DEVICES, undefined);
  }
}

const MIXED_LSPCI = [
  "01:00.0 VGA compatible controller [0300]: NVIDIA Corporation AD102 [GeForce RTX 4090] [10de:2684]",
  "03:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]",
].join("\n");

const MIXED_DRM = [
  {
    name: "card0",
    vendorId: "10de",
    deviceId: "2684",
    busId: "0000:01:00.0",
    connectors: { "card0-DP-1": "connected", "card0-HDMI-A-1": "connected" },
  },
  {
    name: "card1",
    vendorId: "8086",
    deviceId: "e223",
    busId: "0000:03:00.0",
    connectors: { "card1-DP-1": "disconnected" },
  },
];

test("Intel dual B70: idle 08:00.0 is primary, display 04:00.0 has 3 connectors, iGPU ignored", () => {
  const cards = detectGpuCards({ lspci: INTEL_LSPCI, drmCards: INTEL_DRM, nvidiaSmi: null, rocmSmi: null });
  const discrete = cards.filter((card) => !card.igpu);
  const igpu = cards.filter((card) => card.igpu);
  assert.equal(igpu.length, 1);
  assert.equal(discrete.length, 2);
  const display = discrete.find((card) => card.busId === "0000:04:00.0");
  const idle = discrete.find((card) => card.busId === "0000:08:00.0");
  assert.equal(display?.display, true);
  assert.equal(display?.connectedDisplays, 3);
  assert.equal(idle?.display, false);
  assert.ok((idle?.vramMiB ?? 0) >= 32_000);
  const plan = resolveLateInferGpuPlan({ probes: { lspci: INTEL_LSPCI, drmCards: INTEL_DRM, nvidiaSmi: null } });
  assert.equal(plan.primary?.busId, "0000:08:00.0");
  assert.equal(plan.env.CUDA_VISIBLE_DEVICES, undefined);
  assert.equal(plan.env.LATE_INFER_ACCEL, "intel");
  assert.equal(plan.env.LATE_INFER_PCI, "0000:08:00.0");
  assert.equal(plan.env.LATE_INFER_GPU_IDLE, "1");
  assert.equal(plan.env.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(idle?.index, 1);
  assert.equal(idle?.vendorAllIndex, 1);
  assert.equal(display?.vendorAllIndex, 0);
  assert.equal(igpu[0]?.vendorAllIndex, -1);
  assert.equal(plan.env.ZE_AFFINITY_MASK, "1");
  assert.notEqual(plan.env.ZE_AFFINITY_MASK, "2");
  assertNoForeignToolkit(plan.env, "intel");
  assert.equal(plan.runtimeOk, false);
  assert.equal(plan.compileOk, false);
  assert.match(plan.compileTarget, /Intel \(Arc \/ XPU \/ Level Zero\)/);
  assert.equal(plan.compileTarget.includes("NVIDIA"), false);
  assert.equal(plan.compileTarget.toLowerCase().includes("cuda"), false);
  assert.match(plan.runtimeReason, /Intel GPU on your computer/);
  assert.equal(lateInferStartBlockedReason(plan)?.includes("Intel"), true);
  assert.match(lateInferCompileBlockedReason(plan) ?? "", /Intel blob|not treat that card as NVIDIA|OpenVINO/);
  assert.equal(plan.runtimeKind, "");
  assert.equal(plan.serveHint, "");
});

test("engineGpuSpawnPlan pins idle Vulkan/ZE index and adds -ngl 99", () => {
  const eng = engineGpuSpawnPlan({
    probes: { lspci: INTEL_LSPCI, drmCards: INTEL_DRM, nvidiaSmi: null },
  });
  assert.equal(eng.blockedReason, undefined);
  assert.equal(eng.plan.primary?.busId, "0000:08:00.0");
  assert.deepEqual(eng.llamaArgs, ["-ngl", "99"]);
  assert.equal(eng.env.GGML_VK_VISIBLE_DEVICES, "1");
  assert.equal(eng.env.ZE_AFFINITY_MASK, "1");
  assert.equal(eng.env.LATE_INFER_PCI, "0000:08:00.0");
  assert.equal(eng.env.CUDA_VISIBLE_DEVICES, undefined);
  assert.match(vllmCudaOnIntelBlockedReason("cuda", eng.plan) ?? "", /CUDA-only/);
  assert.equal(vllmCudaOnIntelBlockedReason("intel-xpu", eng.plan), undefined);
  const all = engineGpuSpawnPlan({
    probes: { lspci: INTEL_LSPCI, drmCards: INTEL_DRM, nvidiaSmi: null },
    useAllGpus: true,
  });
  assert.deepEqual(all.llamaArgs, ["-ngl", "99", "-sm", "layer", "-ts", "1,1"]);
});

test("Ollama on Intel allows idle pin when Vulkan ICD exists; fails closed without ICD", () => {
  const probesVk = {
    lspci: INTEL_LSPCI,
    drmCards: INTEL_DRM,
    nvidiaSmi: null as string | null,
    vulkanIcds: ["intel_icd.json"],
  };
  const eng = engineGpuSpawnPlan({ probes: probesVk });
  assert.equal(ollamaGpuBlockedReason(eng.plan, probesVk), undefined);
  assert.equal(eng.env.GGML_VK_VISIBLE_DEVICES, "1");
  assert.equal(eng.env.ZE_AFFINITY_MASK, "1");
  const noVk = {
    lspci: INTEL_LSPCI,
    drmCards: INTEL_DRM,
    nvidiaSmi: null as string | null,
    vulkanIcds: [] as string[],
  };
  const engNo = engineGpuSpawnPlan({ probes: noVk });
  assert.match(ollamaGpuBlockedReason(engNo.plan, noVk) ?? "", /Vulkan ICD|GGML_VK|system RAM/);
  assert.match(vllmCudaOnIntelBlockedReason("cuda", eng.plan) ?? "", /CUDA-only/);
});

test("Intel dual B70 + iGPU: Level Zero mask is idle B70 (1), iGPU is not a Level Zero GPU", () => {
  const cards = detectGpuCards({ lspci: INTEL_LSPCI, drmCards: INTEL_DRM, nvidiaSmi: null, rocmSmi: null });
  const igpu = cards.find((card) => card.busId === "0000:00:02.0");
  const display = cards.find((card) => card.busId === "0000:04:00.0");
  const idle = cards.find((card) => card.busId === "0000:08:00.0");
  assert.equal(igpu?.igpu, true);
  assert.equal(igpu?.vendorAllIndex, -1);
  assert.equal(igpu?.index, -1);
  assert.equal(display?.igpu, false);
  assert.equal(display?.display, true);
  assert.equal(display?.vendorAllIndex, 0);
  assert.equal(display?.index, 0);
  assert.equal(idle?.display, false);
  assert.equal(idle?.vendorAllIndex, 1);
  assert.equal(idle?.index, 1);
  const plan = resolveLateInferGpuPlan({ probes: { lspci: INTEL_LSPCI, drmCards: INTEL_DRM, nvidiaSmi: null } });
  assert.equal(plan.primary?.busId, "0000:08:00.0");
  assert.equal(plan.env.ZE_AFFINITY_MASK, "1");
  assert.equal(plan.env.ZE_FLAT_DEVICE_HIERARCHY, "FLAT");
  assert.equal(plan.env.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(plan.env.CUDA_VISIBLE_DEVICES, undefined);
  assert.equal(plan.env.LATE_INFER_PCI, "0000:08:00.0");
  const pickedDisplay = resolveLateInferGpuPlan({ cards, gpuId: display?.id });
  assert.equal(pickedDisplay.env.ZE_AFFINITY_MASK, "0");
  assert.equal(pickedDisplay.env.CUDA_VISIBLE_DEVICES, undefined);
});

test("small Hub snapshot stays on idle Intel card; overflow above that VRAM is blocked at 70%", () => {
  const probes = { lspci: INTEL_LSPCI, drmCards: INTEL_DRM, nvidiaSmi: null as string | null };
  const small = vramMaxMiBForHubId("Qwen/Qwen2.5-0.5B-Instruct");
  assert.ok(small && small < 32_768);
  const fit = resolveLateInferGpuPlan({ probes, vramMaxMiB: small });
  assert.equal(fit.overflowWanted, false);
  assert.equal(fit.visible[0]?.busId, "0000:08:00.0");
  const huge = resolveLateInferGpuPlan({ probes, vramMaxMiB: 40_000 });
  assert.equal(huge.overflowWanted, true);
  assert.equal(huge.overflowBlocked, true);
  assert.equal(huge.capApplied, false);
  assert.equal(huge.displayCapPercent, 70);
  assert.equal(DISPLAY_GPU_VRAM_CAP, 0.7);
  assert.equal(huge.visible.length, 1);
  assert.equal(huge.visible[0]?.display, false);
  assert.match(huge.reason, /70%/);
});

test("dual AMD: DRM display vs idle — HIP/ROCR pin idle index, no CUDA", () => {
  const probes = { lspci: AMD_LSPCI, drmCards: AMD_DRM, nvidiaSmi: null as string | null, rocmSmi: AMD_ROCM_SMI };
  const cards = detectGpuCards(probes);
  const display = cards.find((card) => card.busId === "0000:03:00.0");
  const idle = cards.find((card) => card.busId === "0000:12:00.0");
  assert.equal(display?.vendor, "amd");
  assert.equal(display?.display, true);
  assert.equal(display?.connectedDisplays, 2);
  assert.equal(display?.index, 0);
  assert.equal(idle?.display, false);
  assert.equal(idle?.index, 1);
  assert.ok((idle?.vramMiB ?? 0) >= 24_000);
  const plan = resolveLateInferGpuPlan({ probes, vramMaxMiB: 8_000 });
  assert.equal(plan.primary?.vendor, "amd");
  assert.equal(plan.primary?.busId, "0000:12:00.0");
  assert.equal(plan.primary?.display, false);
  assert.equal(plan.env.LATE_INFER_ACCEL, "amd");
  assert.equal(plan.env.LATE_INFER_PCI, "0000:12:00.0");
  assert.equal(plan.env.LATE_INFER_GPU_IDLE, "1");
  assert.equal(plan.env.HIP_VISIBLE_DEVICES, "1");
  assert.equal(plan.env.ROCR_VISIBLE_DEVICES, "1");
  assert.notEqual(plan.env.HIP_VISIBLE_DEVICES, "0");
  assertNoForeignToolkit(plan.env, "amd");
  assert.equal(plan.runtimeOk, false);
  assert.equal(plan.compileOk, false);
  assert.match(plan.compileTarget, /^AMD · /);
  assert.match(plan.runtimeReason, /AMD GPU on your computer|this engine is CUDA/);
  const published = publicGpuPlan(plan);
  assert.equal(published.cards.find((card) => card.busId === "0000:03:00.0")?.vendorLabel, "AMD");
  assert.equal(published.cards.find((card) => card.busId === "0000:03:00.0")?.role, "display");
  assert.equal(published.cards.find((card) => card.busId === "0000:12:00.0")?.role, "idle");
});

test("mixed NVIDIA display + Intel idle prefers the idle Intel card", () => {
  const plan = resolveLateInferGpuPlan({
    probes: { lspci: MIXED_LSPCI, drmCards: MIXED_DRM, nvidiaSmi: null },
  });
  assert.equal(plan.primary?.vendor, "intel");
  assert.equal(plan.primary?.busId, "0000:03:00.0");
  assert.equal(plan.env.CUDA_VISIBLE_DEVICES, undefined);
  assert.equal(plan.env.LATE_INFER_ACCEL, "intel");
});

test("two NVIDIA cards with no display signal fail closed to a picker", () => {
  const smi = [
    "0, NVIDIA GeForce RTX 4090, 24564, 24000, 00000000:01:00.0",
    "1, NVIDIA GeForce RTX 4090, 24564, 24000, 00000000:02:00.0",
  ].join("\n");
  const plan = resolveLateInferGpuPlan({
    probes: { nvidiaSmi: smi, lspci: null, drmCards: [] },
  });
  assert.equal(plan.needsPicker, true);
  assert.equal(plan.env.CUDA_VISIBLE_DEVICES, undefined);
  assert.match(plan.reason, /Pick a (GPU|card) on your computer|will not silently use GPU 0/);
});

test("NVIDIA idle-primary spawn sets CUDA_VISIBLE_DEVICES to that index, never 0.0.0.0", () => {
  const spec = spawnLateInferSpec("Qwen/Qwen2.5-0.5B-Instruct", {
    probes: { nvidiaSmi: NVIDIA_SMI_GPU0_DISPLAY, lspci: NVIDIA_LSPCI, drmCards: NVIDIA_DRM },
  });
  assert.deepEqual(spec.args, ["--bind", "127.0.0.1:8010", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]);
  assert.equal(spec.args.includes("0.0.0.0"), false);
  assert.equal(spec.env.CUDA_VISIBLE_DEVICES, "1");
  assert.equal(spec.env.LATE_INFER_ACCEL, "nvidia");
  assert.equal(spec.env.LATE_INFER_PCI, "0000:02:00.0");
  assert.equal(spec.gpu.runtimeOk, true);
  assert.equal(spec.gpu.primary?.index, 1);
  assert.equal(spec.gpu.primary?.display, false);
  assertNoForeignToolkit(spec.env, "nvidia");
});

test("Use all GPUs still does not attach the display card when 70% cannot be applied", () => {
  const plan = resolveLateInferGpuPlan({
    probes: { lspci: INTEL_LSPCI, drmCards: INTEL_DRM, nvidiaSmi: null },
    useAllGpus: true,
    vramMaxMiB: 8_000,
  });
  assert.equal(plan.useAllGpus, true);
  assert.equal(plan.visible.length, 1);
  assert.equal(plan.visible[0]?.display, false);
  assert.equal(plan.overflowBlocked, true);
});

test("explicit picker can select the display GPU (operator override)", () => {
  const cards = detectGpuCards({ lspci: INTEL_LSPCI, drmCards: INTEL_DRM, nvidiaSmi: null });
  const display = cards.find((card) => card.busId === "0000:04:00.0");
  assert.ok(display);
  const plan = resolveLateInferGpuPlan({ cards, gpuId: display.id });
  assert.equal(plan.visible[0]?.busId, "0000:04:00.0");
  assert.equal(plan.overflowBlocked, false);
  assert.match(plan.reason, /70%/);
});

const INTEL_PROBES = { lspci: INTEL_LSPCI, drmCards: INTEL_DRM, nvidiaSmi: null as string | null, rocmSmi: null as string | null };
const NVIDIA_IDLE_PROBES = {
  nvidiaSmi: NVIDIA_SMI_GPU0_DISPLAY,
  lspci: NVIDIA_LSPCI,
  drmCards: NVIDIA_DRM,
};
const AMD_PROBES = { lspci: AMD_LSPCI, drmCards: AMD_DRM, nvidiaSmi: null as string | null, rocmSmi: AMD_ROCM_SMI };

test("Intel idle-primary with OpenVINO backend: runtimeOk true, no CUDA, warning gone", () => {
  const probes = {
    ...INTEL_PROBES,
    intelRuntime: { present: true, kind: "openvino-genai" as const, python: "/opt/fake-intel-ov/bin/python" },
  };
  const plan = resolveLateInferGpuPlan({ probes });
  assert.equal(plan.primary?.vendor, "intel");
  assert.equal(plan.primary?.display, false);
  assert.equal(plan.env.LATE_INFER_ACCEL, "intel");
  assert.notEqual(plan.env.LATE_INFER_ACCEL, "nvidia");
  assert.equal(plan.env.CUDA_VISIBLE_DEVICES, undefined);
  assert.equal(plan.env.ZE_AFFINITY_MASK, "1");
  assert.equal(plan.env.LATE_INFER_INTEL_KIND, "openvino-genai");
  assert.equal(plan.env.LATE_INFER_INTEL_PYTHON, "/opt/fake-intel-ov/bin/python");
  assert.equal(plan.runtimeOk, true);
  assert.equal(plan.compileOk, true);
  assert.equal(lateInferStartBlockedReason(plan), undefined);
  assert.equal(lateInferCompileBlockedReason(plan), undefined);
  assert.match(plan.runtimeReason, /OpenVINO GenAI/);
  assert.equal(plan.runtimeReason.includes("cannot use the Intel GPU"), false);
  assert.equal(plan.runtimeKind, "openvino-genai");
  assert.match(plan.serveHint, /Hugging Face safetensors/);
  assert.match(plan.serveHint, /OpenVINO/);
  const published = publicGpuPlan(plan);
  assert.equal(published.runtimeOk, true);
  assert.equal(published.env.LATE_INFER_INTEL_PYTHON, undefined);
  assert.equal(published.runtimeKind, "openvino-genai");
  assert.match(published.serveHint, /OpenVINO/);
});

test("NVIDIA idle-primary never shows the Intel XPU warning", () => {
  const plan = resolveLateInferGpuPlan({ probes: NVIDIA_IDLE_PROBES });
  assert.equal(plan.runtimeOk, true);
  assert.equal(plan.env.LATE_INFER_ACCEL, "nvidia");
  assert.match(plan.runtimeReason, /nvidia/i);
  assert.equal(/Intel GPU|not XPU|OpenVINO/i.test(plan.runtimeReason), false);
  assert.equal(plan.serveHint, "");
  assert.equal(plan.runtimeKind, "candle-cuda");
});

test("Intel-only compile spec never sets CUDA_VISIBLE_DEVICES and records intel vendor", () => {
  const spec = lateInferCompileSpec("Qwen/Qwen2.5-0.5B-Instruct", { probes: INTEL_PROBES });
  assert.deepEqual(spec.args, ["--compile-only", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]);
  assert.equal(spec.args.includes("--bind"), false);
  assert.equal(spec.env.CUDA_VISIBLE_DEVICES, undefined);
  assert.equal(spec.env.LATE_INFER_ACCEL, "intel");
  assert.equal(spec.env.LATE_INFER_PCI, "0000:08:00.0");
  assert.equal(spec.env.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(spec.env.ZE_AFFINITY_MASK, "1");
  assert.notEqual(spec.env.ZE_AFFINITY_MASK, "2");
  assert.equal(spec.gpu.primary?.vendor, "intel");
  assert.equal(spec.gpu.compileOk, false);
  assert.match(spec.gpu.compileTarget, /Intel \(Arc \/ XPU \/ Level Zero\)/);
  assert.equal(/nvidia|cuda/i.test(spec.gpu.compileTarget), false);
  const inherited = mergeGpuPlanEnv({ CUDA_VISIBLE_DEVICES: "0", PATH: "/bin" }, spec.gpu);
  assert.equal(inherited.CUDA_VISIBLE_DEVICES, undefined);
  assert.equal(inherited.LATE_INFER_ACCEL, "intel");
});

test("Intel-only pullLateInfer fails closed before spawn (no Hub fetch as NVIDIA)", async () => {
  let spawned = 0;
  resetLateInferCompileForTests({
    findBin: () => {
      throw new Error("must not look up late-infer when Intel compile is blocked");
    },
    spawnFn: () => {
      spawned += 1;
      throw new Error("must not spawn compile-only on Intel-only");
    },
    gpu: { probes: INTEL_PROBES },
  });
  try {
    const job = await pullLateInfer({ model: "Qwen/Qwen2.5-0.5B-Instruct", gpuId: "intel:0000:08:00.0" });
    assert.equal(spawned, 0);
    assert.equal(job.phase, "error");
    assert.equal(job.downloading, false);
    assert.equal(job.vendor, "intel");
    assert.equal(job.busId, "0000:08:00.0");
    assert.equal(job.idle, true);
    assert.equal(job.display, false);
    assert.match(job.error ?? "", /Intel blob|not treat that card as NVIDIA/);
    assert.equal((job.error ?? "").toLowerCase().includes("nvidia") && (job.error ?? "").includes("not treat"), true);
  } finally {
    resetLateInferCompileForTests();
  }
});

test("NVIDIA idle-primary compile sets CUDA_VISIBLE_DEVICES to that index, not GPU 0", () => {
  const spec = lateInferCompileSpec("Qwen/Qwen2.5-0.5B-Instruct", { probes: NVIDIA_IDLE_PROBES });
  assert.equal(spec.env.CUDA_VISIBLE_DEVICES, "1");
  assert.equal(spec.env.LATE_INFER_ACCEL, "nvidia");
  assert.equal(spec.env.LATE_INFER_PCI, "0000:02:00.0");
  assert.equal(spec.gpu.compileOk, true);
  assert.equal(spec.gpu.primary?.display, false);
  assert.equal(lateInferCompileBlockedReason(spec.gpu), undefined);
  assertNoForeignToolkit(spec.env, "nvidia");
});

test("compile fails closed when overflow would need a 70% display cap", () => {
  const spec = lateInferCompileSpec("Qwen/Qwen2.5-0.5B-Instruct", {
    probes: NVIDIA_IDLE_PROBES,
    vramMaxMiB: 40_000,
  });
  assert.equal(spec.gpu.overflowWanted, true);
  assert.equal(spec.gpu.overflowBlocked, true);
  assert.equal(spec.gpu.compileOk, false);
  assert.match(lateInferCompileBlockedReason(spec.gpu) ?? "", /70%|NVIDIA GPU 0/);
});

test("AMD-only compile spec never sets CUDA_VISIBLE_DEVICES", () => {
  const spec = lateInferCompileSpec("Qwen/Qwen2.5-0.5B-Instruct", { probes: AMD_PROBES });
  assert.equal(spec.env.CUDA_VISIBLE_DEVICES, undefined);
  assert.equal(spec.env.LATE_INFER_ACCEL, "amd");
  assert.equal(spec.env.HIP_VISIBLE_DEVICES, "1");
  assert.equal(spec.env.ROCR_VISIBLE_DEVICES, "1");
  assert.notEqual(spec.env.HIP_VISIBLE_DEVICES, "0");
  assert.equal(spec.gpu.primary?.busId, "0000:12:00.0");
  assert.equal(spec.gpu.primary?.display, false);
  assert.equal(spec.gpu.compileOk, false);
  assertNoForeignToolkit(spec.env, "amd");
  assert.match(lateInferCompileBlockedReason(spec.gpu) ?? "", /AMD|not treat that card as NVIDIA/);
});

test("mixed NVIDIA display + Intel idle compile targets Intel, never CUDA_VISIBLE_DEVICES", () => {
  const spec = lateInferCompileSpec("Qwen/Qwen2.5-0.5B-Instruct", {
    probes: { lspci: MIXED_LSPCI, drmCards: MIXED_DRM, nvidiaSmi: null },
  });
  assert.equal(spec.gpu.primary?.vendor, "intel");
  assert.equal(spec.gpu.primary?.busId, "0000:03:00.0");
  assert.equal(spec.env.LATE_INFER_ACCEL, "intel");
  assert.equal(spec.env.CUDA_VISIBLE_DEVICES, undefined);
  assert.equal(spec.env.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(/nvidia|cuda/i.test(spec.gpu.compileTarget), false);
  assert.equal(spec.gpu.compileOk, false);
});

test("dual NVIDIA: GPU0 display_active, GPU1 idle — CUDA_VISIBLE_DEVICES is idle index, no ZE/ONEAPI", () => {
  const probes = NVIDIA_IDLE_PROBES;
  const cards = detectGpuCards(probes);
  const display = cards.find((card) => card.busId === "0000:01:00.0");
  const idle = cards.find((card) => card.busId === "0000:02:00.0");
  assert.equal(display?.vendor, "nvidia");
  assert.equal(display?.display, true);
  assert.equal(display?.index, 0);
  assert.equal(idle?.display, false);
  assert.equal(idle?.index, 1);
  assert.ok((idle?.vramMiB ?? 0) >= 24_000);
  const plan = resolveLateInferGpuPlan({ probes, vramMaxMiB: 8_000 });
  assert.equal(plan.primary?.vendor, "nvidia");
  assert.equal(plan.primary?.busId, "0000:02:00.0");
  assert.equal(plan.primary?.index, 1);
  assert.equal(plan.env.LATE_INFER_ACCEL, "nvidia");
  assert.equal(plan.env.CUDA_VISIBLE_DEVICES, "1");
  assert.notEqual(plan.env.CUDA_VISIBLE_DEVICES, "0");
  assert.equal(plan.env.LATE_INFER_GPU_IDLE, "1");
  assert.equal(plan.runtimeOk, true);
  assert.equal(plan.compileOk, true);
  assert.match(plan.compileTarget, /^NVIDIA · /);
  assertNoForeignToolkit(plan.env, "nvidia");
  const published = publicGpuPlan(plan);
  assert.equal(published.cards.find((card) => card.busId === "0000:01:00.0")?.vendorLabel, "NVIDIA");
  assert.equal(published.cards.find((card) => card.busId === "0000:01:00.0")?.role, "display");
  assert.equal(published.cards.find((card) => card.busId === "0000:02:00.0")?.role, "idle");
});

test("dual NVIDIA: CUDA_VISIBLE_DEVICES uses nvidia-smi idle index, not PCI order", () => {
  const smi = [
    "0, NVIDIA GeForce RTX 4090, 24564, 24000, 00000000:02:00.0, Disabled, Disabled",
    "1, NVIDIA GeForce RTX 4090, 24564, 20000, 00000000:01:00.0, Enabled, Enabled",
  ].join("\n");
  const plan = resolveLateInferGpuPlan({
    probes: { nvidiaSmi: smi, lspci: NVIDIA_LSPCI, drmCards: NVIDIA_DRM },
  });
  assert.equal(plan.primary?.busId, "0000:02:00.0");
  assert.equal(plan.primary?.display, false);
  assert.equal(plan.primary?.index, 0);
  assert.equal(plan.env.CUDA_VISIBLE_DEVICES, "0");
  assert.equal(plan.env.LATE_INFER_ACCEL, "nvidia");
  assertNoForeignToolkit(plan.env, "nvidia");
});

test("dual AMD overflow above idle VRAM is blocked at 70%", () => {
  const small = resolveLateInferGpuPlan({ probes: AMD_PROBES, vramMaxMiB: 8_000 });
  assert.equal(small.overflowWanted, false);
  assert.equal(small.visible[0]?.busId, "0000:12:00.0");
  assert.equal(small.env.HIP_VISIBLE_DEVICES, "1");
  const huge = resolveLateInferGpuPlan({ probes: AMD_PROBES, vramMaxMiB: 40_000 });
  assert.equal(huge.overflowWanted, true);
  assert.equal(huge.overflowBlocked, true);
  assert.equal(huge.capApplied, false);
  assert.equal(huge.displayCapPercent, 70);
  assert.equal(huge.visible.length, 1);
  assert.equal(huge.visible[0]?.display, false);
  assert.equal(huge.env.HIP_VISIBLE_DEVICES, "1");
  assert.match(huge.reason, /70%/);
  assert.equal(huge.compileOk, false);
  assert.match(lateInferCompileBlockedReason(huge) ?? "", /AMD|not treat that card as NVIDIA/);
});

test("AMD idle-primary spawn pins HIP/ROCR to idle index, never 0.0.0.0", () => {
  const spec = spawnLateInferSpec("Qwen/Qwen2.5-0.5B-Instruct", { probes: AMD_PROBES });
  assert.deepEqual(spec.args, ["--bind", "127.0.0.1:8010", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]);
  assert.equal(spec.args.includes("0.0.0.0"), false);
  assert.equal(spec.env.HIP_VISIBLE_DEVICES, "1");
  assert.equal(spec.env.ROCR_VISIBLE_DEVICES, "1");
  assert.equal(spec.env.LATE_INFER_ACCEL, "amd");
  assert.equal(spec.env.LATE_INFER_PCI, "0000:12:00.0");
  assert.equal(spec.gpu.primary?.index, 1);
  assert.equal(spec.gpu.runtimeOk, false);
  assertNoForeignToolkit(spec.env, "amd");
});

test("publicGpuPlan lists vendor + idle vs display for Intel, NVIDIA, and AMD", () => {
  const intel = publicGpuPlan(resolveLateInferGpuPlan({ probes: INTEL_PROBES }));
  assert.equal(intel.cards.find((card) => card.busId === "0000:04:00.0")?.vendorLabel, "Intel (Arc / XPU / Level Zero)");
  assert.equal(intel.cards.find((card) => card.busId === "0000:04:00.0")?.role, "display");
  assert.equal(intel.cards.find((card) => card.busId === "0000:08:00.0")?.role, "idle");
  assert.equal(intel.primaryId, "intel:0000:08:00.0");

  const nvidia = publicGpuPlan(resolveLateInferGpuPlan({ probes: NVIDIA_IDLE_PROBES }));
  assert.equal(nvidia.cards.find((card) => card.busId === "0000:01:00.0")?.vendorLabel, "NVIDIA");
  assert.equal(nvidia.cards.find((card) => card.busId === "0000:01:00.0")?.role, "display");
  assert.equal(nvidia.cards.find((card) => card.busId === "0000:02:00.0")?.role, "idle");
  assert.equal(nvidia.primaryId, "nvidia:0000:02:00.0");

  const amd = publicGpuPlan(resolveLateInferGpuPlan({ probes: AMD_PROBES }));
  assert.equal(amd.cards.find((card) => card.busId === "0000:03:00.0")?.vendorLabel, "AMD");
  assert.equal(amd.cards.find((card) => card.busId === "0000:03:00.0")?.role, "display");
  assert.equal(amd.cards.find((card) => card.busId === "0000:12:00.0")?.role, "idle");
  assert.equal(amd.primaryId, "amd:0000:12:00.0");
});

test("mergeGpuPlanEnv keeps only the matching vendor toolkit", () => {
  const dirty = {
    CUDA_VISIBLE_DEVICES: "0",
    HIP_VISIBLE_DEVICES: "0",
    ROCR_VISIBLE_DEVICES: "0",
    ZE_AFFINITY_MASK: "0",
    ONEAPI_DEVICE_SELECTOR: "level_zero:gpu",
    ZE_FLAT_DEVICE_HIERARCHY: "FLAT",
    PATH: "/bin",
  };
  const intel = mergeGpuPlanEnv(dirty, resolveLateInferGpuPlan({ probes: INTEL_PROBES }));
  assert.equal(intel.CUDA_VISIBLE_DEVICES, undefined);
  assert.equal(intel.HIP_VISIBLE_DEVICES, undefined);
  assert.equal(intel.ZE_AFFINITY_MASK, "1");
  assert.equal(intel.LATE_INFER_ACCEL, "intel");

  const nvidia = mergeGpuPlanEnv(dirty, resolveLateInferGpuPlan({ probes: NVIDIA_IDLE_PROBES }));
  assert.equal(nvidia.CUDA_VISIBLE_DEVICES, "1");
  assert.equal(nvidia.HIP_VISIBLE_DEVICES, undefined);
  assert.equal(nvidia.ZE_AFFINITY_MASK, undefined);
  assert.equal(nvidia.ONEAPI_DEVICE_SELECTOR, undefined);
  assert.equal(nvidia.LATE_INFER_ACCEL, "nvidia");

  const amd = mergeGpuPlanEnv(dirty, resolveLateInferGpuPlan({ probes: AMD_PROBES }));
  assert.equal(amd.CUDA_VISIBLE_DEVICES, undefined);
  assert.equal(amd.HIP_VISIBLE_DEVICES, "1");
  assert.equal(amd.ROCR_VISIBLE_DEVICES, "1");
  assert.equal(amd.ZE_AFFINITY_MASK, undefined);
  assert.equal(amd.LATE_INFER_ACCEL, "amd");
});

test("parseMeminfo reads MemAvailable and swap", () => {
  const mem = parseMeminfo(
    "MemTotal: 65000000 kB\nMemAvailable: 25165824 kB\nSwapTotal: 8388608 kB\nSwapFree: 102400 kB\n",
  );
  assert.ok(mem);
  assert.equal(mem.memAvailableMiB, Math.round(25165824 / 1024));
  assert.equal(mem.swapTotalMiB, Math.round(8388608 / 1024));
  assert.equal(mem.swapFreeMiB, Math.round(102400 / 1024));
});

test("swap full refuses Start (Gemma convert would OOM your computer)", () => {
  const reason = hostRamBlockedReason({
    vendor: "intel",
    runtimeOk: true,
    weightsMiB: 10_000,
    ovIrPresent: false,
    mem: { memAvailableMiB: 24_000, swapTotalMiB: 8_192, swapFreeMiB: 100 },
  });
  assert.match(reason ?? "", /system RAM, not idle GPU VRAM/);
  assert.match(reason ?? "", /swap is full/);
});

test("Intel convert of 10 GB E2B without IR refuses on 24 GB MemAvailable", () => {
  const reason = hostRamBlockedReason({
    vendor: "intel",
    runtimeOk: true,
    weightsMiB: 10_000,
    ovIrPresent: false,
    mem: { memAvailableMiB: 24_000, swapTotalMiB: 8_192, swapFreeMiB: 4_096 },
  });
  assert.match(reason ?? "", /system RAM, not idle GPU VRAM/);
  assert.match(reason ?? "", /convert/);
});

test("Intel missing IR and unknown size refuses convert", () => {
  const reason = hostRamBlockedReason({
    vendor: "intel",
    runtimeOk: true,
    ovIrPresent: false,
    mem: { memAvailableMiB: 40_000, swapTotalMiB: 8_192, swapFreeMiB: 4_096 },
  });
  assert.match(reason ?? "", /system RAM, not idle GPU VRAM/);
  assert.match(reason ?? "", /IR is missing/);
});

test("Intel missing IR still blocks Start when MemAvailable could convert", () => {
  const reason = hostRamBlockedReason({
    vendor: "intel",
    runtimeOk: true,
    weightsMiB: 1_000,
    ovIrPresent: false,
    mem: { memAvailableMiB: 40_000, swapTotalMiB: 8_192, swapFreeMiB: 4_096 },
  });
  assert.match(reason ?? "", /IR is missing/);
  assert.match(reason ?? "", /Start will not convert in DRAM/);
  const convert = convertHostRamBlockedReason({
    vendor: "intel",
    weightsMiB: 1_000,
    ovIrPresent: false,
    mem: { memAvailableMiB: 40_000, swapTotalMiB: 8_192, swapFreeMiB: 4_096 },
  });
  assert.equal(convert, undefined);
});

test("tiny Qwen IR on Intel is ok when RAM and swap are free", () => {
  const reason = hostRamBlockedReason({
    vendor: "intel",
    runtimeOk: true,
    weightsMiB: 1_000,
    ovIrPresent: true,
    mem: { memAvailableMiB: 40_000, swapTotalMiB: 8_192, swapFreeMiB: 4_096 },
  });
  assert.equal(reason, undefined);
});

test("NVIDIA CUDA Start refuses if MemAvailable cannot stage off host RAM", () => {
  const reason = hostRamBlockedReason({
    vendor: "nvidia",
    runtimeOk: true,
    weightsMiB: 14_000,
    ovIrPresent: true,
    mem: { memAvailableMiB: 6_000, swapTotalMiB: 8_192, swapFreeMiB: 4_096 },
  });
  assert.match(reason ?? "", /system RAM, not idle GPU VRAM/);
});

test("Intel Start plan with swap-full probe blocks Start, not compile", () => {
  const probes = {
    ...INTEL_PROBES,
    intelRuntime: { present: true, kind: "openvino-genai" as const, python: "/opt/fake-intel-ov/bin/python" },
    memAvailableMiB: 24_000,
    swapTotalMiB: 8_192,
    swapFreeMiB: 80,
    ovIrPresent: false,
  };
  const plan = resolveLateInferGpuPlan({
    probes,
    model: "google/gemma-4-E2B-it",
    vramMaxMiB: 15_000,
  });
  assert.equal(plan.runtimeOk, true);
  assert.equal(plan.hostRamOk, false);
  assert.match(plan.hostRamReason, /system RAM, not idle GPU VRAM/);
  assert.equal(lateInferStartBlockedReason(plan)?.includes("system RAM"), true);
  assert.equal(lateInferStartBlockedReason(plan)?.includes(INTEL_IR_MISSING_START.split(".")[0]), true);
  assert.equal(lateInferCompileBlockedReason(plan), undefined);
  const published = publicGpuPlan(plan);
  assert.equal(published.hostRamOk, false);
  assert.match(published.hostRamReason ?? "", /system RAM/);
  assert.equal(HOST_RAM_VS_VRAM.includes("system RAM"), true);
});

const INTEL_FDINFO_DISPLAY = [
  "drm-driver:\txe",
  "drm-client-id:\t115",
  "drm-pdev:\t0000:04:00.0",
  "drm-resident-system:\t0",
  "drm-resident-vram0:\t671840 KiB",
].join("\n");

const INTEL_FDINFO_IDLE = [
  "drm-driver:\txe",
  "drm-client-id:\t9",
  "drm-pdev:\t0000:08:00.0",
  "drm-resident-system:\t12000 KiB",
  "drm-resident-vram0:\t15360000 KiB",
].join("\n");

const CLINFO_INTEL = [
  "Device Name                                     Intel(R) Arc(TM) Pro B70 Graphics",
  "  Device PCI bus info (KHR)                       PCI-E, 0000:08:00.0",
  "  Global memory size                              32530182144 (30.3GiB)",
  "Device Name                                     Intel(R) Graphics",
  "  Device PCI bus info (KHR)                       PCI-E, 0000:00:02.0",
  "  Global memory size                              60910755840 (56.73GiB)",
].join("\n");

test("Intel xe drm-fdinfo VRAM is resident-vram0 on the idle PCI, not drm-resident-system", () => {
  const display = parseDrmFdinfo(INTEL_FDINFO_DISPLAY);
  const idle = parseDrmFdinfo(INTEL_FDINFO_IDLE);
  assert.equal(display?.pci, "0000:04:00.0");
  assert.equal(idle?.pci, "0000:08:00.0");
  const clients = [display, idle].filter(Boolean);
  const idleUsed = usedVramMiBFromFdinfo(clients, "0000:08:00.0");
  const displayUsed = usedVramMiBFromFdinfo(clients, "0000:04:00.0");
  assert.ok(idleUsed > 10_000, `idle VRAM used ${idleUsed}`);
  assert.ok(displayUsed > 500 && displayUsed < 2_000, `display VRAM used ${displayUsed} (not host RAM)`);
  assert.ok(idleUsed !== 12, "must not treat 12000 KiB system as idle VRAM");
  const cards = detectGpuCards({
    lspci: INTEL_LSPCI,
    drmCards: INTEL_DRM,
    nvidiaSmi: null,
    rocmSmi: null,
    drmFdinfo: [INTEL_FDINFO_DISPLAY, INTEL_FDINFO_IDLE],
  });
  const idleCard = cards.find((card) => card.busId === "0000:08:00.0");
  assert.equal(idleCard?.usedVramSource, "drm-fdinfo");
  assert.ok((idleCard?.usedVramMiB ?? 0) > 10_000);
  assert.equal(idleCard?.usedVramMiB, idleUsed);
});

test("clinfo iGPU Global memory is host RAM and is not used as discrete VRAM", () => {
  const rows = parseClinfoDiscreteVram(CLINFO_INTEL);
  assert.equal(rows.some((row) => row.busId === "0000:00:02.0"), false);
  const idle = rows.find((row) => row.busId === "0000:08:00.0");
  assert.ok(idle);
  assert.ok((idle?.vramMiB ?? 0) > 20_000);
  assert.ok((idle?.vramMiB ?? 0) < 40_000);
});

test("NVIDIA used VRAM is total-minus-free from nvidia-smi, not system RAM", () => {
  const cards = detectGpuCards({
    nvidiaSmi: NVIDIA_SMI_GPU0_DISPLAY,
    lspci: NVIDIA_LSPCI,
    drmCards: NVIDIA_DRM,
  });
  const idle = cards.find((card) => card.busId === "0000:02:00.0");
  const display = cards.find((card) => card.busId === "0000:01:00.0");
  assert.equal(idle?.usedVramSource, "nvidia-smi");
  assert.equal(idle?.usedVramMiB, 24564 - 24000);
  assert.equal(display?.usedVramMiB, 24564 - 20000);
});

test("AMD rocm-smi used VRAM is vram Used Memory, not host RAM", () => {
  const smi = [
    AMD_ROCM_SMI,
    "GPU[0]          : vram Total Used Memory (B): 1073741824",
    "GPU[1]          : vram Total Used Memory (B): 8589934592",
  ].join("\n");
  const cards = detectGpuCards({
    lspci: AMD_LSPCI,
    drmCards: AMD_DRM,
    nvidiaSmi: null,
    rocmSmi: smi,
  });
  const idle = cards.find((card) => card.busId === "0000:12:00.0");
  const display = cards.find((card) => card.busId === "0000:03:00.0");
  assert.equal(idle?.usedVramSource, "rocm-smi");
  assert.equal(idle?.usedVramMiB, 8192);
  assert.equal(display?.usedVramMiB, 1024);
});

test("classifyLateInferDevice: Intel XPU / CUDA / HIP are GPU running; cpu is host RAM", () => {
  assert.equal(classifyLateInferDevice("intel-xpu:openvino-genai · 0000:08:00.0").gpuRunning, true);
  assert.equal(classifyLateInferDevice("intel-xpu:openvino-genai").weightsInHostRam, false);
  assert.equal(classifyLateInferDevice("intel-xpu:openvino-genai").deviceKind, "intel-xpu");
  assert.equal(classifyLateInferDevice("cuda:0").deviceKind, "cuda");
  assert.equal(classifyLateInferDevice("cuda:0").gpuRunning, true);
  assert.equal(classifyLateInferDevice("hip:0").deviceKind, "hip");
  assert.equal(classifyLateInferDevice("cpu").weightsInHostRam, true);
  assert.equal(classifyLateInferDevice("cpu").gpuRunning, false);
  assert.equal(formatGpuVramPair(2048, 32768), "2 GB / 32 GB");
});

test("resolveGpuServePhase: starting vs GPU running vs host RAM vs exited (not success on OOM)", () => {
  assert.equal(resolveGpuServePhase({ processAlive: true, ready: false }), "starting");
  assert.equal(
    resolveGpuServePhase({ ready: true, device: "intel-xpu:openvino-genai · 0000:08:00.0" }),
    "gpu-running",
  );
  assert.equal(resolveGpuServePhase({ ready: true, device: "cuda:0" }), "gpu-running");
  assert.equal(resolveGpuServePhase({ ready: true, weightsInHostRam: true, device: "cpu" }), "host-ram");
  assert.equal(resolveGpuServePhase({ pidFilePresent: true, processAlive: false, running: false }), "exited");
  assert.equal(resolveGpuServePhase({}), "down");
  const live = publicGpuPlan(resolveLateInferGpuPlan({ probes: INTEL_PROBES }), {
    processAlive: true,
    ready: false,
  }).live;
  assert.equal(live.servePhase, "starting");
  assert.match(live.headline, /starting on idle GPU/);
  assert.equal(live.gpuRunning, false);
  assert.match(live.detail, /not GPU running/);
  const up = publicGpuPlan(resolveLateInferGpuPlan({ probes: INTEL_PROBES }), {
    ready: true,
    device: "intel-xpu:openvino-genai · 0000:08:00.0",
  }).live;
  assert.equal(up.servePhase, "gpu-running");
  assert.equal(up.headline, "GPU running");
  assert.equal(up.busId, "0000:08:00.0");
  assert.equal(up.role, "idle");
});
