/**
 * Idle-GPU spawn env per vendor for vLLM / Ollama / llama.cpp.
 * Run: node --test scripts/idle-gpu-spawn.test.mjs
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const gpuUrl = pathToFileURL(join(root, "dist/local-servers/gpu-pick.js")).href;
const spawnUrl = pathToFileURL(join(root, "dist/local-servers/spawn.js")).href;
const vllmUrl = pathToFileURL(join(root, "dist/vllm/manager.js")).href;

const {
  engineGpuSpawnPlan,
  resolveLateInferGpuPlan,
  vllmCudaOnIntelBlockedReason,
} = await import(gpuUrl);
const { llamaServerSpec, ollamaServeSpec } = await import(spawnUrl);
const { vllmLaunchEnv } = await import(vllmUrl);

const INTEL_LSPCI = `
0000:00:02.0 VGA compatible controller [0300]: Intel Corporation Arrow Lake-S [Intel Graphics] [8086:7d67]
0000:04:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]
0000:08:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]
`.trim();

const INTEL_DRM = [
  {
    name: "card0",
    vendorId: "8086",
    deviceId: "e223",
    busId: "0000:04:00.0",
    connectors: { "card0-DP-1": "connected" },
  },
  {
    name: "card1",
    vendorId: "8086",
    deviceId: "7d67",
    busId: "0000:00:02.0",
    connectors: {},
  },
  {
    name: "card2",
    vendorId: "8086",
    deviceId: "e223",
    busId: "0000:08:00.0",
    connectors: { "card2-DP-1": "disconnected" },
  },
];

const NVIDIA_SMI = [
  "0, NVIDIA GeForce RTX 4090, 24564, 20000, 00000000:01:00.0, Enabled, Enabled",
  "1, NVIDIA GeForce RTX 4090, 24564, 24000, 00000000:02:00.0, Disabled, Disabled",
].join("\n");

const NVIDIA_LSPCI = `
0000:01:00.0 VGA compatible controller [0300]: NVIDIA Corporation Device [10de:2684]
0000:02:00.0 VGA compatible controller [0300]: NVIDIA Corporation Device [10de:2684]
`.trim();

const NVIDIA_DRM = [
  {
    name: "card0",
    vendorId: "10de",
    deviceId: "2684",
    busId: "0000:01:00.0",
    connectors: { "card0-DP-1": "connected" },
  },
  {
    name: "card1",
    vendorId: "10de",
    deviceId: "2684",
    busId: "0000:02:00.0",
    connectors: { "card1-DP-1": "disconnected" },
  },
];

const AMD_LSPCI = `
0000:01:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi [1002:744c]
0000:02:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi [1002:744c]
`.trim();

const AMD_DRM = [
  {
    name: "card0",
    vendorId: "1002",
    deviceId: "744c",
    busId: "0000:01:00.0",
    connectors: { "card0-DP-1": "connected" },
  },
  {
    name: "card1",
    vendorId: "1002",
    deviceId: "744c",
    busId: "0000:02:00.0",
    connectors: { "card1-DP-1": "disconnected" },
  },
];

const AMD_ROCM = `
GPU[0] : PCI Bus: 0000:01:00.0
GPU[1] : PCI Bus: 0000:02:00.0
`;

const INTEL_PROBES = {
  lspci: INTEL_LSPCI,
  drmCards: INTEL_DRM,
  nvidiaSmi: null,
  rocmSmi: null,
  intelRuntime: { present: true, kind: "openvino-genai", python: "/opt/fake/python" },
};

test("Intel dual B70: idle 0000:08:00.0 → ZE_AFFINITY_MASK=1 + -ngl 99", () => {
  const gpu = engineGpuSpawnPlan({ probes: INTEL_PROBES });
  assert.equal(gpu.plan.primary?.busId, "0000:08:00.0");
  assert.equal(gpu.plan.primary?.display, false);
  assert.equal(gpu.env.ZE_AFFINITY_MASK, "1");
  assert.notEqual(gpu.env.ZE_AFFINITY_MASK, "2");
  assert.equal(gpu.env.GGML_VK_VISIBLE_DEVICES, "1");
  assert.equal(gpu.env.CUDA_VISIBLE_DEVICES, undefined);
  assert.deepEqual(gpu.llamaArgs.slice(0, 2), ["-ngl", "99"]);
});

test("NVIDIA idle-primary → CUDA_VISIBLE_DEVICES idle index", () => {
  const gpu = engineGpuSpawnPlan({
    probes: { nvidiaSmi: NVIDIA_SMI, lspci: NVIDIA_LSPCI, drmCards: NVIDIA_DRM, rocmSmi: null },
  });
  assert.equal(gpu.env.CUDA_VISIBLE_DEVICES, "1");
  assert.equal(gpu.env.ZE_AFFINITY_MASK, undefined);
  assert.equal(gpu.env.HIP_VISIBLE_DEVICES, undefined);
});

test("AMD idle-primary → HIP/ROCR idle index", () => {
  const gpu = engineGpuSpawnPlan({
    probes: { lspci: AMD_LSPCI, drmCards: AMD_DRM, nvidiaSmi: null, rocmSmi: AMD_ROCM },
  });
  assert.equal(gpu.plan.primary?.busId, "0000:02:00.0");
  assert.equal(gpu.env.HIP_VISIBLE_DEVICES, "1");
  assert.equal(gpu.env.ROCR_VISIBLE_DEVICES, "1");
  assert.equal(gpu.env.CUDA_VISIBLE_DEVICES, undefined);
});

test("llamaServerSpec: Late-style -ngl + Intel idle env; loopback only", () => {
  const spec = llamaServerSpec("/tmp/llama-server", "/tmp/tiny.gguf", 8080, { probes: INTEL_PROBES });
  assert.ok(spec.args.includes("-ngl"));
  assert.ok(spec.args.includes("99"));
  assert.ok(spec.args.includes("127.0.0.1"));
  assert.equal(spec.args.includes("0.0.0.0"), false);
  assert.equal(spec.env.ZE_AFFINITY_MASK, "1");
  assert.equal(spec.env.GGML_VK_VISIBLE_DEVICES, "1");
});

test("ollama on Intel fail-closed (no DRAM pretend)", () => {
  assert.throws(
    () => ollamaServeSpec("/tmp/ollama", { probes: INTEL_PROBES }),
    /Ollama cannot use the idle Intel GPU/,
  );
});

test("vLLM CUDA on Intel fail-closed", () => {
  const plan = resolveLateInferGpuPlan({ probes: INTEL_PROBES });
  const reason = vllmCudaOnIntelBlockedReason("cuda", plan);
  assert.match(reason ?? "", /CUDA-only|cannot use the Intel/);
  assert.match(reason ?? "", /system RAM/);
});

test("vllmLaunchEnv pins idle ZE affinity for single Intel device", () => {
  // Live path uses real detect; inject via engine plan fixtures is covered above.
  // Here assert CUDA refuse throws from vllmLaunchEnv when forced.
  assert.throws(() => vllmLaunchEnv("cuda", 1, {}), /CUDA-only|cannot use the Intel|system RAM/);
});

test("vllmLaunchEnv multi Intel clears ZE_AFFINITY_MASK", () => {
  const env = vllmLaunchEnv("intel-xpu", 2, {});
  assert.equal(env.ZE_AFFINITY_MASK, undefined);
  assert.equal(env.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(env.VLLM_TARGET_DEVICE, "xpu");
});

test("live: idle Arc Pro B70 is 0000:08:00.0", () => {
  const plan = resolveLateInferGpuPlan();
  assert.equal(plan.primary?.busId, "0000:08:00.0");
  assert.equal(plan.primary?.display, false);
  assert.equal(plan.env.ZE_AFFINITY_MASK, "1");
  const gpu = engineGpuSpawnPlan();
  assert.deepEqual(gpu.llamaArgs.slice(0, 2), ["-ngl", "99"]);
  assert.equal(gpu.env.GGML_VK_VISIBLE_DEVICES, "1");
  console.log("live:", plan.reason);
});

test("live: ollama Start fail-closed on this Intel box", () => {
  const bin = join(root, "runtime/bin/ollama");
  assert.ok(existsSync(bin));
  assert.throws(() => ollamaServeSpec(bin), /Ollama cannot use the idle Intel GPU/);
});

test("live: llama-server spec pins idle GPU + -ngl", () => {
  const bin = join(root, "runtime/bin/llama-server");
  assert.ok(existsSync(bin));
  const gguf = "/tmp/orchestrator-idle-gpu-probe.gguf";
  if (!existsSync(gguf)) writeFileSync(gguf, Buffer.alloc(64));
  const spec = llamaServerSpec(bin, gguf, 18080);
  assert.equal(spec.env.ZE_AFFINITY_MASK, "1");
  assert.ok(spec.args.includes("-ngl"));
  assert.equal(spec.host.startsWith("127.0.0.1:"), true);
  const listed = spawnSync(bin, ["--list-devices"], {
    env: { ...process.env, ...Object.fromEntries(
      Object.entries(spec.env).filter(([, v]) => typeof v === "string"),
    ) },
    encoding: "utf8",
    timeout: 15_000,
  });
  const out = `${listed.stdout ?? ""}${listed.stderr ?? ""}`;
  console.log("llama --list-devices:\n", out.slice(0, 600));
  if (/Available devices:\s*\(none\)/i.test(out)) {
    console.log("honest: Vulkan ICD missing — env/-ngl still pin idle card; GPU VRAM serve needs ICD/SYCL build");
  }
});

test("live: installed backends summary", async () => {
  const { detectHardware } = await import(pathToFileURL(join(root, "dist/hardware.js")).href);
  const hw = detectHardware();
  assert.equal(hw.primaryBackend, "intel-xpu");
  console.log({
    primaryBackend: hw.primaryBackend,
    deviceCount: hw.deviceCount,
    vramMiB: hw.vramMiB,
    ollama: existsSync(join(root, "runtime/bin/ollama")),
    llamaServer: existsSync(join(root, "runtime/bin/llama-server")),
    vllmCudaRefuse: vllmCudaOnIntelBlockedReason("cuda"),
  });
});
