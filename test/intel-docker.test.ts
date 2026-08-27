import assert from "node:assert/strict";
import { test } from "node:test";
import {
  PREFERRED_LLM_SCALER_REF,
  PREFERRED_INTEL_VLLM_REF,
  buildIntelDockerRunArgs,
  classifyDockerError,
  classifyIntelImage,
  compareIntelTags,
  hasVllmDeviceXpuFlag,
  listIntelVllmImages,
  parseDockerImagesJson,
  selectPreferredIntelImage,
  type IntelDockerImage,
  type IntelDockerRunPlan,
} from "../src/vllm/intel-docker.js";
import { detectVllmLaunch } from "../src/vllm/manager.js";

const MOCK_IMAGES_JSON = [
  '{"Containers":"1","CreatedAt":"2026-08-10 10:14:46 -0400 EDT","ID":"7a526dcfc49c","Repository":"intel/llm-scaler-vllm","Size":"20.5GB","Tag":"0.21.0-b3"}',
  '{"Containers":"1","CreatedAt":"2026-06-26 01:10:31 -0400 EDT","ID":"4454b316ba83","Repository":"intel/llm-scaler-vllm","Size":"47.9GB","Tag":"0.14.0-b8.3.2"}',
  '{"Containers":"0","CreatedAt":"2026-03-20 05:30:46 -0400 EDT","ID":"e961d08135a6","Repository":"intel/vllm","Size":"30.1GB","Tag":"0.17.0-xpu"}',
  '{"Containers":"0","ID":"deadbeef","Repository":"ubuntu","Size":"80MB","Tag":"24.04"}',
].join("\n");

function mockImages(text = MOCK_IMAGES_JSON): IntelDockerImage[] {
  return parseDockerImagesJson(text);
}

test("parseDockerImagesJson keeps Intel vLLM tags and drops unrelated images", () => {
  const images = mockImages();
  assert.deepEqual(
    images.map((row) => row.ref),
    [
      "intel/llm-scaler-vllm:0.21.0-b3",
      "intel/llm-scaler-vllm:0.14.0-b8.3.2",
      "intel/vllm:0.17.0-xpu",
    ],
  );
  assert.equal(images[0]?.id, "7a526dcfc49c");
  assert.equal(images[0]?.size, "20.5GB");
});

test("selectPreferredIntelImage prefers llm-scaler 0.21.0-b3 over older scaler and intel/vllm xpu", () => {
  const preferred = selectPreferredIntelImage(mockImages());
  assert.equal(preferred?.ref, PREFERRED_LLM_SCALER_REF);
  assert.equal(compareIntelTags("0.21.0-b3", "0.14.0-b8.3.2") > 0, true);

  const withoutPinned = mockImages().filter((row) => row.tag !== "0.21.0-b3");
  assert.equal(selectPreferredIntelImage(withoutPinned)?.ref, "intel/llm-scaler-vllm:0.14.0-b8.3.2");

  const onlyXpu = mockImages().filter((row) => row.repository === "intel/vllm");
  assert.equal(selectPreferredIntelImage(onlyXpu)?.ref, PREFERRED_INTEL_VLLM_REF);

  const override = selectPreferredIntelImage(mockImages(), "intel/vllm:0.17.0-xpu");
  assert.equal(override?.ref, "intel/vllm:0.17.0-xpu");
});

test("listIntelVllmImages uses mocked docker images JSON as the chosen runtime catalog", () => {
  const catalog = listIntelVllmImages({
    exec: (command, args) => {
      assert.equal(command, "docker");
      assert.equal(args[0], "images");
      return { status: 0, stdout: MOCK_IMAGES_JSON, stderr: "" };
    },
  });
  assert.equal(catalog.daemon, "ok");
  assert.equal(catalog.available, true);
  assert.equal(catalog.images.length, 3);
  assert.equal(catalog.preferred?.ref, PREFERRED_LLM_SCALER_REF);
});

test("listIntelVllmImages reports docker group / daemon failures clearly", () => {
  const permission = listIntelVllmImages({
    exec: () => ({
      status: 1,
      stdout: "",
      stderr: "permission denied while trying to connect to the docker API at unix:///var/run/docker.sock",
    }),
  });
  assert.equal(permission.available, false);
  assert.equal(permission.daemon, "permission");
  assert.match(permission.error ?? "", /docker group/);

  const down = classifyDockerError(
    "failed to connect to the docker API at unix:///var/run/docker.sock; check if the path is correct and if the daemon is running",
    1,
  );
  assert.equal(down.daemon, "down");
  assert.match(down.message, /daemon|docker group/i);
});

test("detectVllmLaunch on intel-xpu chooses Docker even when host vLLM is CUDA-only", () => {
  const docker = listIntelVllmImages({
    exec: () => ({ status: 0, stdout: MOCK_IMAGES_JSON, stderr: "" }),
  });
  const result = detectVllmLaunch({
    backend: "intel-xpu",
    whichFn: (cmd) => (cmd === "vllm" ? "/usr/bin/vllm" : "/usr/bin/python3"),
    pythonImportOk: () => true,
    probeDevice: () => "cuda",
    docker,
  });
  assert.equal("missing" in result, false);
  if (!("missing" in result)) {
    assert.equal(result.launch.runtime, "docker");
    assert.equal(result.launch.image, PREFERRED_LLM_SCALER_REF);
    assert.equal(result.launch.command, "docker");
  }
});

test("detectVllmLaunch image override selects intel/vllm:0.17.0-xpu", () => {
  const docker = listIntelVllmImages({
    exec: () => ({ status: 0, stdout: MOCK_IMAGES_JSON, stderr: "" }),
  });
  const result = detectVllmLaunch({
    backend: "intel-xpu",
    docker,
    image: "intel/vllm:0.17.0-xpu",
    runtime: "docker",
  });
  assert.equal("missing" in result, false);
  if (!("missing" in result)) {
    assert.equal(result.launch.image, "intel/vllm:0.17.0-xpu");
  }
});

function assertIntelServeCli(plan: IntelDockerRunPlan) {
  assert.equal(hasVllmDeviceXpuFlag(plan.args), false);
  assert.equal(hasVllmDeviceXpuFlag(plan.vllmArgs), false);
  assert.equal(plan.vllmArgs.includes("--device"), false);
  assert.equal(plan.vllmArgs.includes("xpu"), false);
  assert.equal(plan.vllmArgs.includes("--model"), false);
  assert.equal(plan.vllmArgs[0], plan.containerModelPath);
  const afterImage = plan.args.slice(plan.args.lastIndexOf(plan.image) + 1);
  assert.equal(afterImage.includes("--model"), false);
  assert.equal(hasVllmDeviceXpuFlag(afterImage), false);
}

test("buildIntelDockerRunArgs publishes 127.0.0.1 only and mounts models + DRI", () => {
  const plan = buildIntelDockerRunArgs({
    image: PREFERRED_LLM_SCALER_REF,
    hostPort: 8001,
    modelsDir: "/home/user/.orchestrator/models",
    modelPath: "/home/user/.orchestrator/models/Qwen--Qwen2.5-7B-Instruct",
    servedModelName: "Qwen/Qwen2.5-7B-Instruct",
    tensorParallel: 2,
    deviceCount: 2,
    renderGid: "990",
    videoGid: "44",
    driDir: "/dev/dri",
  });
  assert.equal(plan.command, "docker");
  assert.equal(plan.publish, "127.0.0.1:8001:8000");
  assert.equal(plan.containerModelPath, "/llm/models/Qwen--Qwen2.5-7B-Instruct");
  assert.equal(plan.kind, "llm-scaler-passthrough");
  const publishIdx = plan.args.indexOf("--publish");
  assert.equal(plan.args[publishIdx + 1], "127.0.0.1:8001:8000");
  assert.equal(plan.args.includes("0.0.0.0:8001:8000"), false);
  assert.ok(plan.args.includes("--device"));
  assert.ok(plan.args.includes("/dev/dri"));
  assert.ok(plan.args.some((arg) => arg.includes("/home/user/.orchestrator/models:/llm/models")));
  assert.ok(plan.args.includes("--group-add"));
  assert.ok(plan.args.includes("44"));
  assert.ok(plan.args.includes("990"));
  assert.equal(plan.args.includes("ZE_AFFINITY_MASK=0,1"), false);
  assert.equal(plan.args.includes("ONEAPI_DEVICE_SELECTOR=level_zero:gpu"), false);
  assert.ok(plan.args.includes("ZE_FLAT_DEVICE_HIERARCHY=FLAT"));
  assert.ok(plan.args.includes("SYCL_CACHE_PERSISTENT=0"));
  assert.ok(plan.args.includes("--ipc=host"));
  assert.ok(plan.args.includes("--privileged"));
  assert.equal(plan.args.includes("--net=host"), false);
  assert.equal(plan.args[plan.args.indexOf("--tensor-parallel-size") + 1], "2");
  assert.equal(plan.args[0], "run");
  const imageIdx = plan.args.lastIndexOf(PREFERRED_LLM_SCALER_REF);
  assert.equal(plan.args[imageIdx + 1], plan.containerModelPath);
  assertIntelServeCli(plan);
});

test("Intel docker serve args are positional and omit --device xpu", () => {
  for (const image of [PREFERRED_LLM_SCALER_REF, "intel/llm-scaler-vllm:0.14.0-b8.3.2", PREFERRED_INTEL_VLLM_REF]) {
    const plan = buildIntelDockerRunArgs({
      image,
      hostPort: 8000,
      modelsDir: "/models",
      modelPath: "/models/foo",
      servedModelName: "foo",
      driDir: "/dev/dri",
    });
    assertIntelServeCli(plan);
  }
});

test("older llm-scaler tags override entrypoint because Cmd does not forward args", () => {
  assert.equal(classifyIntelImage("intel/llm-scaler-vllm:0.14.0-b8.3.2"), "llm-scaler-override");
  const plan = buildIntelDockerRunArgs({
    image: "intel/llm-scaler-vllm:0.14.0-b8.3.2",
    hostPort: 8002,
    modelsDir: "/models",
    modelPath: "/models/foo",
    servedModelName: "foo",
    renderGid: "990",
  });
  assert.equal(plan.kind, "llm-scaler-override");
  assert.ok(plan.args.includes("--entrypoint"));
  assert.ok(plan.args.includes("bash"));
  assert.ok(plan.args.some((arg) => arg.includes("vllm serve")));
  assert.equal(
    plan.args.some((arg) => arg.includes("setvars.sh")),
    false,
  );
  assertIntelServeCli(plan);
});

test("intel/vllm xpu uses vllm serve as the container command", () => {
  const plan = buildIntelDockerRunArgs({
    image: PREFERRED_INTEL_VLLM_REF,
    hostPort: 8003,
    modelsDir: "/models",
    modelPath: "/models/foo",
    servedModelName: "foo",
  });
  assert.equal(plan.kind, "intel-vllm");
  const imageIdx = plan.args.lastIndexOf(PREFERRED_INTEL_VLLM_REF);
  assert.equal(plan.args[imageIdx + 1], "vllm");
  assert.equal(plan.args[imageIdx + 2], "serve");
  assert.equal(plan.args[imageIdx + 3], plan.containerModelPath);
  assertIntelServeCli(plan);
});

test("per-model container names do not reuse orch-vllm", () => {
  const seven = buildIntelDockerRunArgs({
    image: PREFERRED_LLM_SCALER_REF,
    hostPort: 8000,
    modelsDir: "/models",
    modelPath: "/models/qwen7",
    servedModelName: "Qwen/Qwen2.5-7B-Instruct",
    containerName: "orch-vllm-qwen25-7b-instruct",
    driDir: "/dev/dri",
  });
  const fourteen = buildIntelDockerRunArgs({
    image: PREFERRED_LLM_SCALER_REF,
    hostPort: 8001,
    modelsDir: "/models",
    modelPath: "/models/qwen14",
    servedModelName: "Qwen/Qwen2.5-14B-Instruct",
    containerName: "orch-vllm-qwen25-14b-instruct",
    driDir: "/dev/dri",
  });
  assert.equal(seven.containerName, "orch-vllm-qwen25-7b-instruct");
  assert.equal(fourteen.containerName, "orch-vllm-qwen25-14b-instruct");
  assert.notEqual(seven.containerName, fourteen.containerName);
  assert.equal(seven.args[seven.args.indexOf("--name") + 1], "orch-vllm-qwen25-7b-instruct");
  assert.equal(fourteen.publish, "127.0.0.1:8001:8000");
  assertIntelServeCli(seven);
  assertIntelServeCli(fourteen);
});

test("live docker lists the three Intel images when the daemon is up", () => {
  const catalog = listIntelVllmImages();
  if (catalog.daemon !== "ok") {
    return;
  }
  const refs = catalog.images.map((row) => row.ref);
  if (!refs.includes(PREFERRED_LLM_SCALER_REF)) {
    return;
  }
  assert.ok(refs.includes("intel/llm-scaler-vllm:0.21.0-b3"));
  assert.ok(refs.includes("intel/llm-scaler-vllm:0.14.0-b8.3.2"));
  assert.ok(refs.includes("intel/vllm:0.17.0-xpu"));
  assert.equal(catalog.preferred?.ref, PREFERRED_LLM_SCALER_REF);
});

