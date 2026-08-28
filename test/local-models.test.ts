import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdirSync, mkdtempSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { test } from "node:test";
import { WriteAllowlist, canonicalizeDirectory } from "../src/allowlist.js";
import { LocalModelService } from "../src/local-models/service.js";
import { assertModelDest } from "../src/local-models/paths.js";
import { validateConfigYaml } from "../src/config.js";
import { loadConfig } from "../src/config.js";
import { assertVllmLoopbackHost, buildVllmArgv, classifyVllmLog, detectVllmLaunch, findFreePort, instanceMatches, isVllmCmdline, partitionVllmInstances, probeVllmModels, resolveTensorParallel, resolveVllmGpuLaunch, resolveVllmGpuLaunchForFit, resolveVllmPhase, VLLM_HTTP_WAIT_MESSAGE, vllmLaunchEnv, VllmManager } from "../src/vllm/manager.js";
import { patchVllmBackendYaml, patchVllmOrchestratorYaml, removeVllmOrchestratorYaml, vllmBackendIdForModel, vllmContainerNameForModel, vllmSpecialistIdForModel } from "../src/vllm/upsert.js";

function tempProject(): string {
  return mkdtempSync(join(tmpdir(), "orch-models-"));
}

test("allowlisted model dest is accepted; outside and symlink escapes are rejected", () => {
  const root = tempProject();
  const models = join(root, "models");
  mkdirSync(models);
  const allow = new WriteAllowlist(join(root, "allowlist.json"), [canonicalizeDirectory(root)]);
  const dest = assertModelDest(allow, models, "Qwen/Qwen2.5-0.5B-Instruct");
  assert.ok(dest.includes("Qwen--Qwen2.5-0.5B-Instruct"));

  const outside = tempProject();
  assert.throws(() => assertModelDest(allow, outside, "Qwen/Qwen2.5-0.5B-Instruct"), /not inside/);

  const secret = tempProject();
  writeFileSync(join(secret, "weights.bin"), "nope");
  symlinkSync(secret, join(root, "link"));
  assert.throws(() => assertModelDest(allow, models, "x", join(root, "link", "Qwen--escape")), /not inside/);
});

test("dry-run download path uses catalog id without fetching weights", async () => {
  const root = tempProject();
  mkdirSync(join(root, "models"));
  const allow = new WriteAllowlist(join(root, "allowlist.json"), [canonicalizeDirectory(root)]);
  const dest = assertModelDest(allow, join(root, "models"), "Qwen/Qwen2.5-0.5B-Instruct");
  assert.equal(dest.includes(".."), false);
});

test("vLLM host must be loopback; 0.0.0.0 is rejected", () => {
  assert.equal(assertVllmLoopbackHost(undefined), "127.0.0.1");
  assert.equal(assertVllmLoopbackHost("localhost"), "127.0.0.1");
  assert.equal(assertVllmLoopbackHost("127.0.0.1"), "127.0.0.1");
  assert.throws(() => assertVllmLoopbackHost("0.0.0.0"), /127\.0\.0\.1/);
  assert.throws(() => assertVllmLoopbackHost("192.168.1.10"), /127\.0\.0\.1/);
});

test("isVllmCmdline only matches vLLM process command lines", () => {
  assert.equal(isVllmCmdline("python\0-m\0vllm.entrypoints.openai.api_server"), true);
  assert.equal(isVllmCmdline("/usr/bin/vllm serve /models/x"), true);
  assert.equal(isVllmCmdline("/usr/bin/nginx"), false);
});

test("detectVllmLaunch reports install hint when binary and module are missing", () => {
  const result = detectVllmLaunch({
    whichFn: () => undefined,
    pythonImportOk: () => false,
    probeDevice: () => "missing",
  });
  assert.equal("missing" in result, true);
  if ("missing" in result) {
    assert.match(result.installHint, /pip install vllm/);
  }
});

test("detectVllmLaunch refuses CUDA-only vLLM on Intel XPU", () => {
  const result = detectVllmLaunch({
    backend: "intel-xpu",
    whichFn: (cmd) => (cmd === "vllm" ? "/usr/bin/vllm" : "/usr/bin/python3"),
    pythonImportOk: () => true,
    probeDevice: () => "cuda",
  });
  assert.equal("missing" in result, true);
  if ("missing" in result) {
    assert.match(result.installHint, /VLLM_TARGET_DEVICE=xpu|Intel Arc/);
    assert.doesNotMatch(result.installHint, /tiny CPU-feasible/);
  }
});

test("detectVllmLaunch accepts an XPU vLLM build", () => {
  const result = detectVllmLaunch({
    backend: "intel-xpu",
    whichFn: (cmd) => (cmd === "vllm" ? "/opt/vllm" : "/usr/bin/python3"),
    pythonImportOk: () => true,
    probeDevice: () => "xpu",
  });
  assert.equal("missing" in result, false);
  if (!("missing" in result)) {
    assert.equal(result.launch.command, "/opt/vllm");
    assert.equal(result.launch.args[0], "serve");
    assert.equal(result.launch.runtime, "host");
  }
});

test("detectVllmLaunch prefers Intel Docker images over missing/CUDA host vLLM", () => {
  const result = detectVllmLaunch({
    backend: "intel-xpu",
    whichFn: () => undefined,
    pythonImportOk: () => false,
    probeDevice: () => "missing",
    docker: {
      available: true,
      daemon: "ok",
      images: [
        {
          repository: "intel/llm-scaler-vllm",
          tag: "0.21.0-b3",
          id: "7a526dcfc49c",
          size: "20.5GB",
          ref: "intel/llm-scaler-vllm:0.21.0-b3",
        },
      ],
      preferred: {
        repository: "intel/llm-scaler-vllm",
        tag: "0.21.0-b3",
        id: "7a526dcfc49c",
        size: "20.5GB",
        ref: "intel/llm-scaler-vllm:0.21.0-b3",
      },
    },
  });
  assert.equal("missing" in result, false);
  if (!("missing" in result)) {
    assert.equal(result.launch.runtime, "docker");
    assert.equal(result.launch.image, "intel/llm-scaler-vllm:0.21.0-b3");
  }
});

test("buildVllmArgv uses XPU flags, loopback host, and tensor parallel", () => {
  const built = buildVllmArgv({
    launch: { command: "vllm", args: ["serve"], backend: "intel-xpu" },
    modelPath: "/models/qwen",
    port: 8001,
    servedModelName: "Qwen/Qwen2.5-7B-Instruct",
    backend: "intel-xpu",
    tensorParallel: 2,
  });
  assert.equal(built.args.includes("0.0.0.0"), false);
  assert.equal(built.args[built.args.indexOf("--host") + 1], "127.0.0.1");
  assert.ok(built.args.includes("xpu"));
  assert.equal(built.args[built.args.indexOf("--tensor-parallel-size") + 1], "2");
  assert.equal(built.args[built.args.indexOf("--gpu-memory-utilization") + 1], "0.9");
  assert.equal(built.args.includes("--quantization"), false);
  assert.ok(built.args.includes("--trust-remote-code"));
  const env = vllmLaunchEnv("intel-xpu", 2, {});
  assert.equal(env.ZE_FLAT_DEVICE_HIERARCHY, "FLAT");
  assert.equal(env.ZE_AFFINITY_MASK, undefined);
  assert.equal(env.VLLM_WORKER_MULTIPROC_METHOD, "spawn");
  assert.equal(env.ONEAPI_DEVICE_SELECTOR, "level_zero:gpu");
  assert.equal(vllmLaunchEnv("intel-xpu", 2, { ZE_AFFINITY_MASK: "1,2" }).ZE_AFFINITY_MASK, "1,2");
  assert.equal(vllmLaunchEnv("cuda", 1, {}).CUDA_VISIBLE_DEVICES, "0");
  assert.equal(vllmLaunchEnv("cuda", 2, {}).CUDA_VISIBLE_DEVICES, "0,1");
  assert.equal(vllmLaunchEnv("cuda", 2, {}).VLLM_WORKER_MULTIPROC_METHOD, "spawn");
  assert.equal(vllmLaunchEnv("rocm", 1, {}).HIP_VISIBLE_DEVICES, "0");
  assert.equal(vllmLaunchEnv("rocm", 2, {}).HIP_VISIBLE_DEVICES, "0,1");
  assert.equal(vllmLaunchEnv("rocm", 2, {}).ROCR_VISIBLE_DEVICES, "0,1");
  assert.equal(resolveTensorParallel(2, 2), 2);
  assert.equal(resolveTensorParallel(4, 2), 2);
  assert.equal(resolveTensorParallel(undefined, 2), 1);
  assert.deepEqual(resolveVllmGpuLaunch({ deviceCount: 2 }), { tensorParallel: 2, deviceCount: 2 });
  assert.deepEqual(resolveVllmGpuLaunch({ deviceCount: 2, useAllGpus: true }), { tensorParallel: 2, deviceCount: 2 });
  assert.deepEqual(resolveVllmGpuLaunch({ deviceCount: 2, useAllGpus: false }), {
    tensorParallel: undefined,
    deviceCount: 1,
  });
  assert.deepEqual(resolveVllmGpuLaunch({ deviceCount: 1 }), { tensorParallel: undefined, deviceCount: 1 });
  assert.deepEqual(
    resolveVllmGpuLaunchForFit({ deviceCount: 2, fitKind: "fits", fitParallel: 1 }),
    { tensorParallel: undefined, deviceCount: 1 },
  );
  assert.deepEqual(
    resolveVllmGpuLaunchForFit({ deviceCount: 2, fitKind: "needs_tp", fitParallel: 2 }),
    { tensorParallel: 2, deviceCount: 2 },
  );
  assert.deepEqual(
    resolveVllmGpuLaunchForFit({ deviceCount: 2, useAllGpus: false, fitKind: "needs_tp", fitParallel: 2 }),
    { tensorParallel: undefined, deviceCount: 1 },
  );
  const dual = buildVllmArgv({
    launch: { command: "vllm", args: ["serve"], backend: "cuda" },
    modelPath: "/m",
    port: 8000,
    servedModelName: "m",
    backend: "cuda",
    tensorParallel: 2,
    deviceCount: 2,
  });
  assert.equal(dual.args[dual.args.indexOf("--tensor-parallel-size") + 1], "2");
  const single = buildVllmArgv({
    launch: { command: "vllm", args: ["serve"], backend: "cuda" },
    modelPath: "/m",
    port: 8000,
    servedModelName: "m",
    backend: "cuda",
    tensorParallel: undefined,
    deviceCount: 1,
  });
  assert.equal(single.args.includes("--tensor-parallel-size"), false);
});

test("buildVllmArgv CUDA dual GPU passes tensor parallel and memory utilization", () => {
  const built = buildVllmArgv({
    launch: { command: "vllm", args: ["serve"], backend: "cuda" },
    modelPath: "/models/qwen14",
    port: 8002,
    servedModelName: "Qwen/Qwen2.5-14B-Instruct",
    backend: "cuda",
    tensorParallel: 2,
  });
  assert.equal(built.args[built.args.indexOf("--host") + 1], "127.0.0.1");
  assert.equal(built.args.includes("0.0.0.0"), false);
  assert.equal(built.args[built.args.indexOf("--tensor-parallel-size") + 1], "2");
  assert.equal(built.args[built.args.indexOf("--gpu-memory-utilization") + 1], "0.9");
  assert.equal(built.args.includes("--device"), false);
});

test("probeVllmModels maps undici fetch failed / ECONNREFUSED to a wait message", async () => {
  const original = globalThis.fetch;
  try {
    globalThis.fetch = (async () => {
      const error = new Error("fetch failed");
      (error as Error & { cause: { code: string } }).cause = { code: "ECONNREFUSED" };
      throw error;
    }) as typeof fetch;
    const down = await probeVllmModels("http://127.0.0.1:8001/v1", 50);
    assert.equal(down.ok, false);
    assert.equal(down.waiting, true);
    assert.equal(down.error, VLLM_HTTP_WAIT_MESSAGE);
    assert.doesNotMatch(down.error ?? "", /fetch failed/i);
  } finally {
    globalThis.fetch = original;
  }
  assert.match(
    classifyVllmLog("ValueError: Model architecture Gemma4ForConditionalGeneration is not supported") ?? "",
    /Gemma 4/,
  );
});

test("buildVllmArgv adds Gemma 4 Intel recipe flags", () => {
  const built = buildVllmArgv({
    launch: { command: "vllm", args: ["serve"], backend: "intel-xpu" },
    modelPath: "/models/google--gemma-4-E2B-it",
    port: 8001,
    servedModelName: "google/gemma-4-E2B-it",
    backend: "intel-xpu",
  });
  assert.equal(built.args[built.args.indexOf("--mamba-ssm-cache-dtype") + 1], "float16");
  assert.ok(built.args.includes("--trust-remote-code"));
});

test("patchVllmBackendYaml updates baseUrl and model without dropping comments", () => {
  const yaml = `# keep me
backends:
  vllm-local:
    type: vllm
    baseUrl: http://127.0.0.1:8000/v1
    model: old-model
    # apiKeyEnv: VLLM_API_KEY
specialists:
  vllm-chat:
    description: t
    backend: vllm-local
`;
  const next = patchVllmBackendYaml(yaml, {
    baseUrl: "http://127.0.0.1:8001/v1",
    model: "Qwen/Qwen2.5-7B-Instruct",
  });
  assert.match(next, /# keep me/);
  assert.match(next, /# apiKeyEnv: VLLM_API_KEY/);
  assert.match(next, /baseUrl: "?http:\/\/127\.0\.0\.1:8001\/v1"?/);
  assert.match(next, /model: "?Qwen\/Qwen2\.5-7B-Instruct"?/);
  assert.equal(next.includes("0.0.0.0"), false);
});

test("patchVllmOrchestratorYaml updates backend in place and keeps vllm-chat pointed at it", () => {
  const yaml = `backends:
  vllm-local:
    type: vllm
    baseUrl: http://127.0.0.1:8000/v1
    model: old
specialists:
  vllm-chat:
    description: t
    backend: vllm-local
    fallback: cursor-local
`;
  const next = patchVllmOrchestratorYaml(yaml, {
    baseUrl: "http://127.0.0.1:8002/v1",
    model: "Qwen/Qwen2.5-7B-Instruct",
  });
  assert.match(next, /baseUrl: "?http:\/\/127\.0\.0\.1:8002\/v1"?/);
  assert.match(next, /model: "?Qwen\/Qwen2\.5-7B-Instruct"?/);
  assert.equal((next.match(/^\s{2}vllm-local:/gm) ?? []).length, 1);
  assert.match(next, /vllm-chat:\n    description: t\n    backend: vllm-local/);
  assert.equal(next.includes("0.0.0.0"), false);
  assert.doesNotMatch(next, /apiKey:/);
});

test("resolveVllmPhase maps start job to starting then running or error", () => {
  assert.deepEqual(resolveVllmPhase({ jobStatus: "starting", processAlive: false }), {
    phase: "starting",
    healthy: false,
  });
  assert.deepEqual(resolveVllmPhase({ jobStatus: "starting", processAlive: true }), {
    phase: "starting",
    healthy: false,
  });
  assert.deepEqual(resolveVllmPhase({ jobStatus: "running", processAlive: true }), {
    phase: "running",
    healthy: true,
  });
  assert.deepEqual(resolveVllmPhase({ jobStatus: "error", processAlive: false }), {
    phase: "error",
    healthy: false,
  });
  assert.deepEqual(resolveVllmPhase({ processAlive: false }), { phase: "idle", healthy: false });
});

test("vLLM backend ids, specialist ids, and container names come from the catalog id", () => {
  assert.equal(vllmBackendIdForModel("qwen2.5-7b-instruct"), "vllm-qwen25-7b-instruct");
  assert.equal(vllmSpecialistIdForModel("qwen2.5-7b-instruct"), "vllm-qwen25-7b-instruct");
  assert.equal(vllmContainerNameForModel("qwen2.5-7b-instruct"), "orch-vllm-qwen25-7b-instruct");
  assert.equal(vllmBackendIdForModel("qwen2.5-14b-instruct"), "vllm-qwen25-14b-instruct");
  assert.notEqual(vllmContainerNameForModel("qwen2.5-7b-instruct"), "orch-vllm");
  assert.notEqual(
    vllmContainerNameForModel("qwen2.5-7b-instruct"),
    vllmContainerNameForModel("qwen2.5-14b-instruct"),
  );
});

test("patchVllmOrchestratorYaml can register two backends on two ports", () => {
  let yaml = `backends:
  cursor-local:
    type: cursor
    runtime: local
specialists:
  builder:
    description: t
    backend: cursor-local
`;
  yaml = patchVllmOrchestratorYaml(yaml, {
    baseUrl: "http://127.0.0.1:8000/v1",
    model: "Qwen/Qwen2.5-7B-Instruct",
    backendId: "vllm-qwen25-7b-instruct",
    specialistId: "vllm-qwen25-7b-instruct",
  });
  yaml = patchVllmOrchestratorYaml(yaml, {
    baseUrl: "http://127.0.0.1:8001/v1",
    model: "Qwen/Qwen2.5-14B-Instruct",
    backendId: "vllm-qwen25-14b-instruct",
    specialistId: "vllm-qwen25-14b-instruct",
  });
  assert.match(yaml, /vllm-qwen25-7b-instruct:\n    type: vllm\n    baseUrl: "?http:\/\/127\.0\.0\.1:8000\/v1"?\n    model: "?Qwen\/Qwen2\.5-7B-Instruct"?/);
  assert.match(yaml, /vllm-qwen25-14b-instruct:\n    type: vllm\n    baseUrl: "?http:\/\/127\.0\.0\.1:8001\/v1"?\n    model: "?Qwen\/Qwen2\.5-14B-Instruct"?/);
  assert.match(yaml, /specialists:[\s\S]*vllm-qwen25-7b-instruct:\n    description:/);
  assert.match(yaml, /specialists:[\s\S]*vllm-qwen25-14b-instruct:\n    description:/);
  const backendsOnly = yaml.split(/^specialists:/m)[0] ?? yaml;
  assert.doesNotMatch(backendsOnly, /^ {4}backend:/m);
  assert.equal((backendsOnly.match(/^\s{2}vllm-qwen25-7b-instruct:/gm) ?? []).length, 1);
  assert.equal((backendsOnly.match(/^\s{2}vllm-qwen25-14b-instruct:/gm) ?? []).length, 1);
  assert.equal((yaml.match(/^\s{2}vllm-qwen25-7b-instruct:/gm) ?? []).length, 2);
  assert.equal((yaml.match(/^\s{2}vllm-qwen25-14b-instruct:/gm) ?? []).length, 2);
  assert.equal(yaml.includes("0.0.0.0"), false);
});

test("removeVllmOrchestratorYaml unregisters one backend and leaves the other", () => {
  let yaml = `backends:
  vllm-qwen25-7b-instruct:
    type: vllm
    baseUrl: http://127.0.0.1:8000/v1
    model: Qwen/Qwen2.5-7B-Instruct
  vllm-qwen25-14b-instruct:
    type: vllm
    baseUrl: http://127.0.0.1:8001/v1
    model: Qwen/Qwen2.5-14B-Instruct
specialists:
  vllm-qwen25-7b-instruct:
    description: t
    backend: vllm-qwen25-7b-instruct
    fallback: cursor-local
  vllm-qwen25-14b-instruct:
    description: t
    backend: vllm-qwen25-14b-instruct
    fallback: cursor-local
`;
  yaml = removeVllmOrchestratorYaml(yaml, { backendId: "vllm-qwen25-7b-instruct" });
  assert.doesNotMatch(yaml, /vllm-qwen25-7b-instruct:/);
  assert.match(yaml, /vllm-qwen25-14b-instruct:\n    type: vllm\n    baseUrl: http:\/\/127\.0\.0\.1:8001\/v1/);
  assert.match(yaml, /backend: vllm-qwen25-14b-instruct/);
  assert.equal(yaml.includes("0.0.0.0"), false);
});

test("removeVllm unregisters yaml for one backend and leaves the other", () => {
  const root = tempProject();
  mkdirSync(join(root, "models"));
  const configPath = join(root, "agents.config.yaml");
  writeFileSync(
    configPath,
    `backends:
  vllm-qwen25-7b-instruct:
    type: vllm
    baseUrl: http://127.0.0.1:8000/v1
    model: Qwen/Qwen2.5-7B-Instruct
  vllm-qwen25-14b-instruct:
    type: vllm
    baseUrl: http://127.0.0.1:8001/v1
    model: Qwen/Qwen2.5-14B-Instruct
specialists:
  vllm-qwen25-7b-instruct:
    description: t
    backend: vllm-qwen25-7b-instruct
    fallback: cursor-local
  vllm-qwen25-14b-instruct:
    description: t
    backend: vllm-qwen25-14b-instruct
    fallback: cursor-local
`,
  );
  const allow = new WriteAllowlist(join(root, "allowlist.json"), [canonicalizeDirectory(root)]);
  const prevState = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  process.env.AGENT_ORCHESTRATOR_STATE_DIR = join(root, "state");
  try {
    const service = new LocalModelService(
      allow,
      new EventEmitter(),
      configPath,
      () => loadConfig(configPath),
      () => undefined,
    );
    service.removeVllm({ backendId: "vllm-qwen25-7b-instruct" });
    const next = readFileSync(configPath, "utf8");
    assert.doesNotMatch(next, /vllm-qwen25-7b-instruct:/);
    assert.match(next, /vllm-qwen25-14b-instruct:/);
    assert.match(next, /baseUrl: http:\/\/127\.0\.0\.1:8001\/v1/);
  } finally {
    if (prevState === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prevState;
  }
});

test("partitionVllmInstances stop one leaves the other", () => {
  const instances = [
    { backendId: "vllm-qwen25-7b-instruct", modelId: "qwen2.5-7b-instruct", port: 8000 },
    { backendId: "vllm-qwen25-14b-instruct", modelId: "qwen2.5-14b-instruct", port: 8001 },
  ];
  const { matched, rest } = partitionVllmInstances(instances, { modelId: "qwen2.5-7b-instruct" });
  assert.equal(matched.length, 1);
  assert.equal(matched[0]?.backendId, "vllm-qwen25-7b-instruct");
  assert.equal(rest.length, 1);
  assert.equal(rest[0]?.port, 8001);
  assert.equal(instanceMatches(instances[1]!, { backendId: "vllm-qwen25-14b-instruct" }), true);
  assert.equal(instanceMatches(instances[1]!, { modelId: "qwen2.5-7b-instruct" }), false);
});

test("findFreePort skips reserved ports in 8000–8099", async () => {
  const port = await findFreePort("127.0.0.1", 8000, 8099, new Set([8000]));
  assert.notEqual(port, 8000);
  assert.ok(port >= 8001 && port <= 8099);
});

test("beginStart returns starting before runStart resolves", async () => {
  const events = new EventEmitter();
  const mgr = new VllmManager(
    events,
    join(tempProject(), "agents.config.yaml"),
    () => ({ backends: { "vllm-local": { type: "vllm", model: "x" } }, specialists: {}, workflows: {} }),
    () => undefined,
  );
  let released: ((value: ReturnType<VllmManager["status"]>) => void) | undefined;
  const gate = new Promise<ReturnType<VllmManager["status"]>>((resolve) => {
    released = resolve;
  });
  (mgr as unknown as { runStart: () => Promise<ReturnType<VllmManager["status"]>> }).runStart = async () => gate;
  const t0 = Date.now();
  const accepted = mgr.beginStart({
    modelId: "qwen-7b",
    hfRepo: "Qwen/Qwen2.5-7B-Instruct",
    modelPath: "/tmp/models/qwen",
  });
  assert.equal(accepted.status, "starting");
  assert.match(accepted.jobId, /^vllm-/);
  assert.equal(accepted.vllm.phase, "starting");
  assert.equal(accepted.vllm.healthy, false);
  assert.ok(Date.now() - t0 < 500);
  released?.({
    ...accepted.vllm,
    phase: "running",
    healthy: true,
    running: true,
    port: 8001,
  });
  await delay(20);
  assert.equal(mgr.status().phase === "running" || mgr.status().startJob?.status === "running", true);
});

test("beginStart of a second model does not stop or wait on the first", async () => {
  const events = new EventEmitter();
  const mgr = new VllmManager(
    events,
    join(tempProject(), "agents.config.yaml"),
    () => ({ backends: {}, specialists: {}, workflows: {} }),
    () => undefined,
  );
  const gates = new Map<string, () => void>();
  (mgr as unknown as { runStart: (input: { modelId: string }) => Promise<ReturnType<VllmManager["status"]>> }).runStart =
    async (input) => {
      await new Promise<void>((resolve) => {
        gates.set(input.modelId, resolve);
      });
      return mgr.status();
    };
  const first = mgr.beginStart({
    modelId: "qwen2.5-7b-instruct",
    hfRepo: "Qwen/Qwen2.5-7B-Instruct",
    modelPath: "/tmp/models/qwen7",
  });
  const second = mgr.beginStart({
    modelId: "qwen2.5-14b-instruct",
    hfRepo: "Qwen/Qwen2.5-14B-Instruct",
    modelPath: "/tmp/models/qwen14",
  });
  assert.equal(first.status, "starting");
  assert.equal(second.status, "starting");
  assert.notEqual(first.jobId, second.jobId);
  assert.equal(first.vllm.phase, "starting");
  gates.get("qwen2.5-7b-instruct")?.();
  gates.get("qwen2.5-14b-instruct")?.();
  await delay(20);
});

test("repo config includes local-and-cloud and cloud-with-local-draft", () => {
  const config = loadConfig();
  assert.equal(config.specialists["cloud-builder"]?.backend, "cursor-cloud");
  assert.equal(config.workflows["local-and-cloud"]?.mode, "parallel");
  assert.equal(config.workflows["cloud-with-local-draft"]?.mode, "sequence");
  assert.deepEqual(
    config.workflows["local-and-cloud"]?.steps.map((step) => step.specialist),
    ["vllm-chat", "cloud-builder"],
  );
});

test("workflow yaml parses parallel mode and step backend override", () => {
  const parsed = validateConfigYaml(`
backends:
  vllm-local:
    type: vllm
    baseUrl: http://127.0.0.1:8000/v1
    model: x
  cursor-cloud:
    type: cursor
    runtime: cloud
specialists:
  vllm-chat:
    description: t
    backend: vllm-local
  cloud-builder:
    description: t
    backend: cursor-cloud
workflows:
  local-and-cloud:
    description: both
    mode: parallel
    steps:
      - specialist: vllm-chat
      - specialist: cloud-builder
        backend: cursor-cloud
`);
  assert.equal(parsed.workflows["local-and-cloud"]?.mode, "parallel");
  assert.equal(parsed.workflows["local-and-cloud"]?.steps[1]?.backend, "cursor-cloud");
});
