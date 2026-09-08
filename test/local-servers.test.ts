import assert from "node:assert/strict";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, test } from "node:test";
import { parseOrchestratorConfig, validateConfigYaml } from "../src/config.js";
import { pidAlive } from "../src/platform.js";
import {
  isLoopbackHttpUrl,
  isLoopbackHostname,
  normalizeLoopbackOpenAiUrl,
} from "../src/local-servers/loopback.js";
import { llamaServerOnPath, ollamaOnPath, probeLlamaCpp, probeOllama } from "../src/local-servers/status.js";
import { findEngineBin, findLateInferBin, lateInferMissingMessage } from "../src/local-servers/bins.js";
import {
  compileJobHasFsFields,
  compileErrorLine,
  lateCompiledDir,
  lateHfHomeDir,
  lateInferCompileJob,
  lateInferCompileSpec,
  lateInferCompileView,
  listCompiledLateInferIds,
  listCompiledLateInferRows,
  pickLateInferGpuPlanModel,
  deleteCompiledLateInfer,
  LateInferCompiledDeleteError,
  parseCompilePhase,
  parseCompileProgress,
  hubSnapshotBytes,
  etaSecFromRate,
  pullLateInfer,
  resetLateInferCompileForTests,
} from "../src/local-servers/compile.js";
import {
  formatOwnedPid,
  lateInferSpec,
  llamaServerSpec,
  ollamaServeSpec,
  ownedPidIdentityLive,
  ownedPidMatchesLive,
  parseOwnedPid,
  procComm,
  procStarttime,
  stopLocalServer,
  stopListenerIfEngine,
} from "../src/local-servers/spawn.js";
import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { setTimeout as delay } from "node:timers/promises";
import type { ChildProcess } from "node:child_process";
import {
  DEFAULT_OLLAMA_BACKEND_ID,
  DEFAULT_OLLAMA_SPECIALIST_ID,
  ollamaSpecialistDescription,
  patchLocalOrchestratorYaml,
} from "../src/local-servers/upsert.js";
import { LlamaCppProvider, OllamaProvider } from "../src/providers/local-openai.js";

test("loopback host validation accepts 127.0.0.1 and localhost, refuses 0.0.0.0", () => {
  assert.equal(isLoopbackHostname("127.0.0.1"), true);
  assert.equal(isLoopbackHostname("localhost"), true);
  assert.equal(isLoopbackHostname("::1"), true);
  assert.equal(isLoopbackHostname("0.0.0.0"), false);
  assert.equal(isLoopbackHostname("192.168.1.10"), false);
  assert.equal(isLoopbackHttpUrl("http://127.0.0.1:11434/v1"), true);
  assert.equal(isLoopbackHttpUrl("http://0.0.0.0:11434/v1"), false);
  assert.equal(normalizeLoopbackOpenAiUrl("http://localhost:11434/v1", "Ollama"), "http://127.0.0.1:11434/v1");
  assert.throws(() => normalizeLoopbackOpenAiUrl("http://0.0.0.0:11434/v1", "Ollama"), /127\.0\.0\.1/);
  assert.throws(() => normalizeLoopbackOpenAiUrl("http://example.com:11434/v1", "Ollama"), /example.com/);
});

test("parseOrchestratorConfig reads ollama and llamacpp types", () => {
  const parsed = validateConfigYaml(`
backends:
  ollama:
    type: ollama
    baseUrl: http://127.0.0.1:11434/v1
    model: llama3.1
    apiKey: ollama
  llamacpp:
    type: llamacpp
    baseUrl: http://127.0.0.1:8080/v1
    model: local
    probe: false
specialists:
  chat:
    description: t
    backend: ollama
`);
  assert.equal(parsed.backends.ollama?.type, "ollama");
  assert.equal(parsed.backends.llamacpp?.type, "llamacpp");
  if (parsed.backends.ollama?.type === "ollama") {
    assert.equal(parsed.backends.ollama.baseUrl, "http://127.0.0.1:11434/v1");
    assert.equal(parsed.backends.ollama.model, "llama3.1");
  }
  if (parsed.backends.llamacpp?.type === "llamacpp") {
    assert.equal(parsed.backends.llamacpp.probe, false);
    assert.equal(parsed.backends.llamacpp.model, "local");
  }
});

test("ollama and llamacpp config reject non-loopback baseUrl without echoing a secret", () => {
  assert.throws(
    () =>
      parseOrchestratorConfig({
        backends: {
          ollama: {
            type: "ollama",
            baseUrl: "http://8.8.8.8:11434/v1",
            model: "llama3.1",
          },
        },
        specialists: { chat: { description: "x", backend: "ollama" } },
      }),
    /127\.0\.0\.1/,
  );
  assert.throws(
    () =>
      parseOrchestratorConfig({
        backends: {
          llamacpp: {
            type: "llamacpp",
            baseUrl: "http://0.0.0.0:8080/v1",
            model: "local",
          },
        },
        specialists: { chat: { description: "x", backend: "llamacpp" } },
      }),
    /0\.0\.0\.0/,
  );
});

test("probeOllama uses mocked /api/tags and does not need a live daemon", async () => {
  const fetchFn = (async (url: string | URL | Request) => {
    const href = String(url);
    assert.match(href, /127\.0\.0\.1:11434\/api\/tags/);
    return new Response(JSON.stringify({ models: [{ name: "llama3.1:latest" }, { name: "qwen2.5:7b" }] }), {
      status: 200,
    });
  }) as typeof fetch;
  const status = await probeOllama({ fetchFn, timeoutMs: 50 });
  assert.equal(status.running, true);
  assert.equal(status.ready, true);
  assert.deepEqual(status.models, ["llama3.1:latest", "qwen2.5:7b"]);
  assert.match(status.reason, /2 models/);
});

test("probeOllama marks not running on mocked ECONNREFUSED", async () => {
  const fetchFn = (async () => {
    const error = new Error("fetch failed");
    (error as Error & { cause: { code: string } }).cause = { code: "ECONNREFUSED" };
    throw error;
  }) as typeof fetch;
  const status = await probeOllama({ fetchFn, timeoutMs: 50 });
  assert.equal(status.running, false);
  assert.equal(status.ready, false);
  assert.match(status.reason, /Ollama not running at http:\/\/127\.0\.0\.1:11434\/v1/);
});

test("probeLlamaCpp uses mocked /v1/models", async () => {
  const fetchFn = (async (url: string | URL | Request) => {
    const href = String(url);
    assert.match(href, /127\.0\.0\.1:8080\/v1\/models/);
    return new Response(JSON.stringify({ data: [{ id: "qwen2.5-7b" }] }), { status: 200 });
  }) as typeof fetch;
  const status = await probeLlamaCpp({ fetchFn, timeoutMs: 50 });
  assert.equal(status.running, true);
  assert.deepEqual(status.models, ["qwen2.5-7b"]);
});

test("probeLlamaCpp does not treat HTML 200 on /v1/models as llama.cpp running", async () => {
  const html = "<!DOCTYPE html><html><head><title>App</title></head><body>not llama.cpp</body></html>";
  const fetchFn = (async () =>
    new Response(html, {
      status: 200,
      headers: { "content-type": "text/html; charset=utf-8" },
    })) as typeof fetch;
  const status = await probeLlamaCpp({ fetchFn, timeoutMs: 50 });
  assert.equal(status.running, false);
  assert.equal(status.ready, false);
  assert.deepEqual(status.models, []);
  assert.match(status.reason, /not an OpenAI models JSON list/);
});

test("OllamaProvider probe ready/not-ready with mocked fetch", async () => {
  const original = globalThis.fetch;
  const provider = new OllamaProvider("ollama", {
    type: "ollama",
    baseUrl: "http://127.0.0.1:11434/v1",
    model: "llama3.1",
  });
  try {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ models: [{ name: "llama3.1" }] }), { status: 200 })) as typeof fetch;
    const ok = await provider.probe();
    assert.equal(ok.ready, true);
    assert.equal(ok.type, "ollama");
    assert.ok(ok.modelChoices?.includes("llama3.1"));

    globalThis.fetch = (async () => {
      const error = new Error("fetch failed");
      (error as Error & { cause: { code: string } }).cause = { code: "ECONNREFUSED" };
      throw error;
    }) as typeof fetch;
    const down = await provider.probe();
    assert.equal(down.ready, false);
    assert.match(down.reason ?? "", /Ollama not running/);
  } finally {
    globalThis.fetch = original;
  }
});

test("LlamaCppProvider is ready without a key when probe is disabled", () => {
  const provider = new LlamaCppProvider("llamacpp", {
    type: "llamacpp",
    baseUrl: "http://127.0.0.1:8080/v1",
    model: "local",
    probe: false,
  });
  const health = provider.health();
  assert.equal(health.ready, true);
  assert.equal(health.needsKey, false);
  assert.equal(health.type, "llamacpp");
});

test("patchLocalOrchestratorYaml inserts ollama backend and specialist", () => {
  const yaml = `backends:
  gemini:
    type: openai
    model: gemini-3.6-flash
specialists:
  planner:
    description: t
    backend: gemini
`;
  const next = patchLocalOrchestratorYaml(yaml, {
    backendId: DEFAULT_OLLAMA_BACKEND_ID,
    type: "ollama",
    baseUrl: "http://127.0.0.1:11434/v1",
    model: "llama3.1",
    apiKey: "ollama",
    specialistId: DEFAULT_OLLAMA_SPECIALIST_ID,
    description: ollamaSpecialistDescription(),
  });
  assert.match(next, /type: ollama/);
  assert.match(next, /ollama-chat:/);
  assert.match(next, /backend: ollama/);
  assert.match(next, /gemini-3\.6-flash/);
  assert.match(next, /model: "llama3\.1"/);
});

test("patchLocalOrchestratorYaml quotes colon tags and rejects YAML injection", () => {
  const yaml = `backends:
  gemini:
    type: openai
    model: gemini-3.6-flash
specialists:
  planner:
    description: t
    backend: gemini
`;
  const poison =
    "llama3.1\n    type: openai\n    baseUrl: https://api.openai.com/v1\n    apiKeyEnv: OPENAI_API_KEY";
  assert.throws(
    () =>
      patchLocalOrchestratorYaml(yaml, {
        backendId: DEFAULT_OLLAMA_BACKEND_ID,
        type: "ollama",
        baseUrl: "http://127.0.0.1:11434/v1",
        model: poison,
        apiKey: "ollama",
        specialistId: DEFAULT_OLLAMA_SPECIALIST_ID,
        description: ollamaSpecialistDescription(),
      }),
    /line breaks/,
  );

  const next = patchLocalOrchestratorYaml(yaml, {
    backendId: DEFAULT_OLLAMA_BACKEND_ID,
    type: "ollama",
    baseUrl: "http://127.0.0.1:11434/v1",
    model: "llama3.2:latest",
    apiKey: "ollama",
    specialistId: DEFAULT_OLLAMA_SPECIALIST_ID,
    description: ollamaSpecialistDescription(),
  });
  assert.match(next, /model: "llama3\.2:latest"/);
  const ollamaBlock = /^  ollama:\n(?: {4}.*\n)*/m.exec(next)?.[0] ?? "";
  assert.match(ollamaBlock, /type: ollama/);
  assert.doesNotMatch(ollamaBlock, /type: openai/);
  assert.doesNotMatch(ollamaBlock, /api\.openai\.com/);
  const parsed = validateConfigYaml(next);
  assert.equal(parsed.backends.ollama?.type, "ollama");
  if (parsed.backends.ollama?.type === "ollama") {
    assert.equal(parsed.backends.ollama.model, "llama3.2:latest");
    assert.equal(parsed.backends.ollama.baseUrl, "http://127.0.0.1:11434/v1");
  }
  assert.equal(parsed.backends.gemini?.type, "openai");
});

test("probeOllama drops tags with newlines so they cannot be registered", async () => {
  const fetchFn = (async () =>
    new Response(
      JSON.stringify({
        models: [
          {
            name: "evil\n    type: openai\n    baseUrl: https://api.openai.com/v1\n    apiKeyEnv: OPENAI_API_KEY",
          },
          { name: "llama3.2:latest" },
        ],
      }),
      { status: 200 },
    )) as typeof fetch;
  const status = await probeOllama({ fetchFn, timeoutMs: 50 });
  assert.deepEqual(status.models, ["llama3.2:latest"]);
});

test("llamaServerOnPath and ollamaOnPath use injected which including .exe names", () => {
  assert.equal(llamaServerOnPath(() => undefined), undefined);
  assert.equal(llamaServerOnPath((cmd) => (cmd === "llama-server" ? "C:\\\\tools\\\\llama-server.exe" : undefined)), "C:\\\\tools\\\\llama-server.exe");
  assert.equal(ollamaOnPath((cmd) => (cmd === "ollama" ? "C:\\\\Users\\\\me\\\\AppData\\\\Local\\\\Programs\\\\Ollama\\\\ollama.exe" : undefined)), "C:\\\\Users\\\\me\\\\AppData\\\\Local\\\\Programs\\\\Ollama\\\\ollama.exe");
});

test("bundled runtime/bin wins over PATH for findEngineBin", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-bin-"));
  const bundled = join(dir, "ollama");
  writeFileSync(bundled, "#!/bin/sh\n");
  chmodSync(bundled, 0o755);
  const prev = process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN;
  process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN = dir;
  try {
    const found = findEngineBin("ollama");
    assert.equal(found, bundled);
  } finally {
    if (prev === undefined) delete process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN;
    else process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN = prev;
    rmSync(dir, { recursive: true, force: true });
  }
});

test("late-infer spawn spec binds 127.0.0.1:8010", () => {
  const spec = lateInferSpec("Qwen/Qwen2.5-0.5B-Instruct");
  assert.deepEqual(spec.args, ["--bind", "127.0.0.1:8010", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]);
  assert.equal(spec.host, "127.0.0.1:8010");
  assert.equal(spec.args.includes("0.0.0.0"), false);
});

describe("late-infer compile-only pull", { concurrency: false }, () => {
const NVIDIA_COMPILE_GPU = {
  probes: {
    nvidiaSmi: [
      "0, NVIDIA GeForce RTX 4090, 24564, 20000, 00000000:01:00.0, Enabled, Enabled",
      "1, NVIDIA GeForce RTX 4090, 24564, 24000, 00000000:02:00.0, Disabled, Disabled",
    ].join("\n"),
    lspci: null as string | null,
    drmCards: [] as [],
  },
};

test("late-infer compile-only spec has no bind and stays off :8010", () => {
  const spec = lateInferCompileSpec("Qwen/Qwen2.5-0.5B-Instruct", NVIDIA_COMPILE_GPU);
  assert.deepEqual(spec.args, ["--compile-only", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]);
  assert.equal(spec.args.includes("--bind"), false);
  assert.equal(spec.args.some((a) => a.includes("8010")), false);
  assert.equal(spec.args.some((a) => a.includes("0.0.0.0")), false);
  assert.equal(spec.env.LATE_INFER_ACCEL, "nvidia");
  assert.equal(spec.env.CUDA_VISIBLE_DEVICES, "1");
  assert.match(lateHfHomeDir(), /late[/\\]hf/);
});

function mockCompileChild(behavior: "ok" | "fail-closed" | "hang" | "gated-401" | "hub-404"): ChildProcess {
  const child = new EventEmitter() as ChildProcess;
  const stdout = new PassThrough();
  const stderr = new PassThrough();
  child.stdout = stdout;
  child.stderr = stderr;
  child.pid = 4242;
  child.kill = (() => {
    child.emit("close", 1);
    return true;
  }) as ChildProcess["kill"];
  setImmediate(() => {
    if (behavior === "hang") {
      stderr.write("late-infer: downloading Hub snapshot…\n");
      return;
    }
    if (behavior === "hub-404") {
      stderr.write("late-infer: downloading Hub snapshot…\n");
      stderr.write(
        "Error: config.json Caused by: 0: request error: https://huggingface.co/Ali-Mhrez/Qwen3-4B-Instruct-2507-SD-FNC-512-2/resolve/main/config.json: status code 404\n",
      );
      child.emit("close", 1);
      return;
    }
    if (behavior === "gated-401") {
      stderr.write("401 Client Error. Cannot access gated repo google/gemma-2-2b-it\n");
      child.emit("close", 1);
      return;
    }
    if (behavior === "fail-closed") {
      stderr.write(
        'this Hub snapshot is not a supported graph (model_type="my_custom_moe"). unknown types fail closed instead of a new hand-written decoder.\n',
      );
      child.emit("close", 1);
      return;
    }
    stderr.write("late-infer: downloading Hub snapshot…\n");
    stderr.write("late-infer: compiling on your computer…\n");
    stdout.write(
      JSON.stringify({
        ok: true,
        compiler: "mlc-llm",
        serve: "candle-fallback",
        model_type: "qwen2",
        model: "Qwen/Qwen2.5-0.5B-Instruct",
        status: "ok",
      }) + "\n",
    );
    child.emit("close", 0);
  });
  return child;
}

test("pullLateInfer job has no FS fields and reports download then compile", async () => {
  resetLateInferCompileForTests({
    findBin: () => "/tmp/fake-late-infer",
    spawnFn: (_cmd, args, options) => {
      assert.deepEqual(args, ["--compile-only", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]);
      assert.equal(args.includes("--bind"), false);
      const env = (options?.env ?? {}) as NodeJS.ProcessEnv;
      assert.equal(env.CUDA_VISIBLE_DEVICES, "1");
      assert.equal(env.LATE_INFER_ACCEL, "nvidia");
      return mockCompileChild("ok");
    },
    gpu: NVIDIA_COMPILE_GPU,
  });
  try {
    const job = await pullLateInfer({ model: "Qwen/Qwen2.5-0.5B-Instruct" });
    assert.equal(job.kind, "lateinfer");
    assert.equal(job.model, "Qwen/Qwen2.5-0.5B-Instruct");
    assert.equal(compileJobHasFsFields(job), false);
    assert.equal("cwd" in job, false);
    assert.equal("allowlist" in job, false);
    assert.equal("dest" in job, false);
    assert.equal(job.phase === "downloading" || job.phase === "compiling" || job.phase === "done", true);
    const start = Date.now();
    let latest = job;
    while (Date.now() - start < 1000) {
      latest = lateInferCompileJob() ?? latest;
      if (latest.phase === "done") break;
      await delay(15);
    }
    assert.equal(latest.phase, "done");
    assert.equal(latest.downloading, false);
    assert.match(latest.message, /ready for Start/i);
    assert.equal(compileJobHasFsFields(latest), false);
  } finally {
    resetLateInferCompileForTests();
  }
});

test("pullLateInfer surfaces fail-closed unsupported Hub graphs without downloading GB", async () => {
  resetLateInferCompileForTests({
    findBin: () => "/tmp/fake-late-infer",
    spawnFn: () => mockCompileChild("fail-closed"),
    gpu: NVIDIA_COMPILE_GPU,
  });
  try {
    await pullLateInfer({ model: "org/fake-moe" });
    const start = Date.now();
    let latest = lateInferCompileJob();
    while (Date.now() - start < 1000) {
      latest = lateInferCompileJob();
      if (latest?.phase === "error") break;
      await delay(15);
    }
    assert.equal(latest?.phase, "error");
    assert.equal(latest?.downloading, false);
    assert.match(latest?.error ?? "", /fail closed|not a supported/);
    assert.equal(compileJobHasFsFields(latest ?? {}), false);
  } finally {
    resetLateInferCompileForTests();
  }
});

test("pullLateInfer gated 401 explains HF_TOKEN without downloading GB", async () => {
  resetLateInferCompileForTests({
    findBin: () => "/tmp/fake-late-infer",
    spawnFn: () => mockCompileChild("gated-401"),
    gpu: NVIDIA_COMPILE_GPU,
  });
  try {
    await pullLateInfer({ model: "google/gemma-2-2b-it" });
    const start = Date.now();
    let latest = lateInferCompileJob();
    while (Date.now() - start < 1000) {
      latest = lateInferCompileJob();
      if (latest?.phase === "error") break;
      await delay(15);
    }
    assert.equal(latest?.phase, "error");
    assert.match(latest?.error ?? "", /401|gated/i);
    assert.match(latest?.error ?? "", /HF_TOKEN|license|Settings|token/i);
    assert.equal(compileJobHasFsFields(latest ?? {}), false);
  } finally {
    resetLateInferCompileForTests();
  }
});

test("parseCompilePhase maps Hub download and compile lines", () => {
  assert.equal(parseCompilePhase("late-infer: downloading Hub snapshot…"), "downloading");
  assert.equal(parseCompilePhase("late-infer: compiling on your computer…"), "compiling");
  assert.equal(parseCompilePhase('model_type="x" fail closed'), "error");
  assert.equal(
    parseCompilePhase(
      "Error: config.json Caused by: 0: request error: https://huggingface.co/Ali-Mhrez/Qwen3-4B-Instruct-2507-SD-FNC-512-2/resolve/main/config.json: status code 404",
    ),
    "error",
  );
  assert.equal(
    compileErrorLine(
      "late-infer: downloading Hub snapshot…\nError: config.json Caused by: 0: request error: https://huggingface.co/Ali-Mhrez/Qwen3-4B-Instruct-2507-SD-FNC-512-2/resolve/main/config.json: status code 404",
    ).includes("downloading Hub snapshot"),
    false,
  );
  assert.match(
    compileErrorLine(
      "late-infer: downloading Hub snapshot…\nError: config.json Caused by: 0: request error: https://huggingface.co/Ali-Mhrez/Qwen3-4B-Instruct-2507-SD-FNC-512-2/resolve/main/config.json: status code 404",
    ),
    /no config\.json|missing on Hugging Face|llama\.cpp/i,
  );
  assert.match(
    compileErrorLine(
      "Error: config.json Caused by: 0: request error: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/config.json: status code 401",
    ),
    /gated|license|token/i,
  );
  assert.match(
    compileErrorLine("Error: config.json"),
    /Could not read Hub config\.json|safetensors Instruct|llama\.cpp/i,
  );
  assert.equal(compileErrorLine("Error: config.json").includes("Error: config.json"), false);
});

test("parseCompileProgress reads percent bytes totalBytes etaSec from compile-only lines", () => {
  const line = parseCompileProgress(
    "late-infer: progress phase=downloading bytes=2500 total=10000 percent=25 eta_sec=12",
  );
  assert.equal(line?.phase, "downloading");
  assert.equal(line?.bytes, 2500);
  assert.equal(line?.totalBytes, 10000);
  assert.equal(line?.percent, 25);
  assert.equal(line?.etaSec, 12);
  const compile = parseCompileProgress("late-infer: progress phase=compiling");
  assert.equal(compile?.phase, "compiling");
  assert.equal(compile?.percent, undefined);
  const slash = parseCompileProgress("late-infer: downloading 7500/10000");
  assert.equal(slash?.bytes, 7500);
  assert.equal(slash?.totalBytes, 10000);
  assert.equal(slash?.percent, 75);
  assert.equal(hubSnapshotBytes("Qwen/Qwen2.5-0.5B-Instruct"), 1000 * 1024 * 1024);
  assert.equal(etaSecFromRate(0, 2500, 10000, 1000), 3);
  assert.equal(parseCompileProgress("hello")?.phase, undefined);
});

test("pullLateInfer records Hub fetch percent and ETA from mocked progress bytes", async () => {
  const child = new EventEmitter() as ChildProcess;
  const stdout = new PassThrough();
  const stderr = new PassThrough();
  child.stdout = stdout;
  child.stderr = stderr;
  child.pid = 4243;
  child.kill = (() => true) as ChildProcess["kill"];
  resetLateInferCompileForTests({
    findBin: () => "/tmp/fake-late-infer",
    spawnFn: () => {
      setImmediate(() => {
        stderr.write("late-infer: downloading Hub snapshot…\n");
        stderr.write("late-infer: progress phase=downloading bytes=2500 total=10000 percent=25 eta_sec=12\n");
      });
      return child;
    },
    gpu: NVIDIA_COMPILE_GPU,
  });
  try {
    const job = await pullLateInfer({ model: "Qwen/Qwen2.5-0.5B-Instruct" });
    assert.equal(job.phase, "downloading");
    assert.equal(job.percent, 0);
    assert.equal(job.bytes, 0);
    assert.equal(job.totalBytes, 1000 * 1024 * 1024);
    const start = Date.now();
    let latest = job;
    while (Date.now() - start < 1000) {
      latest = lateInferCompileJob() ?? latest;
      if (latest.percent === 25 && latest.bytes === 2500) break;
      await delay(15);
    }
    assert.equal(latest.phase, "downloading");
    assert.equal(latest.percent, 25);
    assert.equal(latest.bytes, 2500);
    assert.equal(latest.totalBytes, 10000);
    assert.equal(latest.etaSec, 12);
    assert.equal(latest.downloading, true);
    const view = lateInferCompileView();
    assert.equal(view.percent, 25);
    assert.equal(view.bytes, 2500);
    assert.equal(view.totalBytes, 10000);
    assert.equal(view.etaSec, 12);
    assert.equal(compileJobHasFsFields(latest), false);
    stderr.write("late-infer: compiling on your computer…\n");
    stderr.write("late-infer: progress phase=compiling\n");
    await delay(40);
    const compiling = lateInferCompileJob();
    assert.equal(compiling?.phase, "compiling");
    assert.equal(compiling?.downloading, true);
    assert.equal(compiling?.etaSec, undefined);
    stdout.write(
      JSON.stringify({
        ok: true,
        compiler: "mlc-llm",
        serve: "candle-fallback",
        model_type: "qwen2",
        model: "Qwen/Qwen2.5-0.5B-Instruct",
        status: "ok",
      }) + "\n",
    );
    child.emit("close", 0);
    const doneStart = Date.now();
    let done = compiling;
    while (Date.now() - doneStart < 1000) {
      done = lateInferCompileJob();
      if (done?.phase === "done") break;
      await delay(15);
    }
    assert.equal(done?.phase, "done");
    assert.equal(done?.percent, 100);
    assert.equal(done?.downloading, false);
  } finally {
    resetLateInferCompileForTests();
  }
});

test("pullLateInfer Hub 404 clears downloading so the card is not frozen", async () => {
  resetLateInferCompileForTests({
    findBin: () => "/tmp/fake-late-infer",
    spawnFn: () => mockCompileChild("hub-404"),
    gpu: NVIDIA_COMPILE_GPU,
  });
  try {
    await pullLateInfer({ model: "google/gemma-4-E2B-it" });
    const start = Date.now();
    let latest = lateInferCompileJob();
    while (Date.now() - start < 1000) {
      latest = lateInferCompileJob();
      if (latest?.phase === "error" && latest.downloading === false) break;
      await delay(15);
    }
    assert.equal(latest?.phase, "error");
    assert.equal(latest?.downloading, false);
    assert.match(latest?.error ?? "", /no config\.json|missing on Hugging Face|404|llama\.cpp/i);
    assert.equal((latest?.error ?? "").includes("downloading Hub snapshot"), false);
    const view = lateInferCompileView();
    assert.equal(view.downloading, false);
    assert.equal(view.phase, "error");
    assert.equal(compileJobHasFsFields(latest ?? {}), false);
  } finally {
    resetLateInferCompileForTests();
  }
});

test("listCompiledLateInferRows marks Intel candle-fallback as not ready for GPU Start", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-late-rows-"));
  const blob = join(dir, "Qwen--Qwen2.5-0.5B-Instruct");
  mkdirSync(blob, { recursive: true });
  writeFileSync(
    join(blob, "late-compile.json"),
    JSON.stringify({
      model_id: "Qwen/Qwen2.5-0.5B-Instruct",
      status: "ok",
      serve: "candle-fallback",
      vendor: "intel",
    }),
  );
  const rows = listCompiledLateInferRows({ ...process.env, LATE_COMPILED_DIR: dir }, "intel");
  assert.equal(rows.length, 1);
  assert.equal(rows[0].id, "Qwen/Qwen2.5-0.5B-Instruct");
  assert.equal(rows[0].irPresent, false);
  assert.equal(rows[0].gpuReady, false);
  mkdirSync(join(blob, "openvino"), { recursive: true });
  writeFileSync(join(blob, "openvino", "openvino_model.xml"), "<net/>");
  const ready = listCompiledLateInferRows({ ...process.env, LATE_COMPILED_DIR: dir }, "intel");
  assert.equal(ready[0].irPresent, true);
  assert.equal(ready[0].gpuReady, true);
  rmSync(dir, { recursive: true, force: true });
});

test("pickLateInferGpuPlanModel prefers IR-ready compiled row over YAML without IR", () => {
  const rows = [
    {
      id: "Qwen/Qwen2.5-0.5B-Instruct",
      gpuReady: true,
      irPresent: true,
    },
  ];
  assert.equal(
    pickLateInferGpuPlanModel({
      yamlModel: "google/gemma-4-E2B-it",
      compiledRows: rows,
      compiledIds: ["Qwen/Qwen2.5-0.5B-Instruct"],
    }),
    "Qwen/Qwen2.5-0.5B-Instruct",
  );
  assert.equal(
    pickLateInferGpuPlanModel({
      servingModel: "Qwen/Qwen2.5-0.5B-Instruct",
      yamlModel: "google/gemma-4-E2B-it",
      compiledRows: [{ id: "Qwen/Qwen2.5-0.5B-Instruct", gpuReady: false, irPresent: false }],
    }),
    "Qwen/Qwen2.5-0.5B-Instruct",
  );
  // Serving Hub default without IR must not hide an IR-ready compiled snapshot.
  assert.equal(
    pickLateInferGpuPlanModel({
      servingModel: "Qwen/Qwen2.5-0.5B-Instruct",
      yamlModel: "Qwen/Qwen2.5-0.5B-Instruct",
      compiledRows: [
        { id: "Qwen/Qwen2.5-0.5B-Instruct", gpuReady: false, irPresent: false },
        { id: "Qwen/Qwen3-0.6B", gpuReady: true, irPresent: true },
      ],
    }),
    "Qwen/Qwen3-0.6B",
  );
});

test("listCompiledLateInferIds returns Hub ids only from compiled dir", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-late-compiled-"));
  const blob = join(dir, "google--gemma-4-E2B-it");
  mkdirSync(blob, { recursive: true });
  writeFileSync(
    join(blob, "late-compile.json"),
    JSON.stringify({
      model_id: "google/gemma-4-E2B-it",
      status: "ok",
      snapshot: { config: "/home/someone/.local/share/late/hf/config.json" },
    }),
  );
  const ids = listCompiledLateInferIds({ ...process.env, LATE_COMPILED_DIR: dir, LATE_HF_HOME: join(dir, "missing-hf") });
  assert.deepEqual(ids, ["google/gemma-4-E2B-it"]);
  assert.equal(ids.some((id) => id.includes("/home/") || id.includes("share/late")), false);
  assert.match(lateCompiledDir({ LATE_COMPILED_DIR: dir }), new RegExp(dir.replace(/[\\^$*+?.()|[\]{}]/g, "\\$&")));
  rmSync(dir, { recursive: true, force: true });
});

test("deleteCompiledLateInfer removes the slug dir and updates compiledModels", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-late-del-"));
  const blob = join(dir, "google--gemma-4-E2B-it");
  const hf = join(dir, "hf");
  mkdirSync(blob, { recursive: true });
  mkdirSync(hf, { recursive: true });
  writeFileSync(
    join(blob, "late-compile.json"),
    JSON.stringify({ model_id: "google/gemma-4-E2B-it", status: "ok" }),
  );
  writeFileSync(join(hf, "keep.txt"), "hub-cache");
  const env = { ...process.env, LATE_COMPILED_DIR: dir, LATE_HF_HOME: hf };
  const result = deleteCompiledLateInfer({
    model: "google/gemma-4-E2B-it",
    confirm: true,
    env,
  });
  assert.equal(result.ok, true);
  assert.equal(result.model, "google/gemma-4-E2B-it");
  assert.equal(result.deleted, true);
  assert.deepEqual(result.compiledModels, []);
  assert.deepEqual(result.compiledRows, []);
  assert.equal(existsSync(blob), false);
  assert.equal(existsSync(join(hf, "keep.txt")), true);
  assert.equal("dest" in result, false);
  assert.equal("cwd" in result, false);
  assert.equal("compiledDir" in result, false);
  assert.equal(compileJobHasFsFields(result), false);
  assert.deepEqual(listCompiledLateInferIds(env), []);
  rmSync(dir, { recursive: true, force: true });
});

test("deleteCompiledLateInfer missing id fails closed", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-late-del-miss-"));
  const env = { ...process.env, LATE_COMPILED_DIR: dir, LATE_HF_HOME: join(dir, "hf") };
  assert.throws(
    () => deleteCompiledLateInfer({ model: "Qwen/Qwen2.5-0.5B-Instruct", confirm: true, env }),
    (err: unknown) => {
      assert.ok(err instanceof LateInferCompiledDeleteError);
      assert.equal(err.httpStatus, 404);
      assert.match(err.message, /not compiled on your computer/);
      return true;
    },
  );
  rmSync(dir, { recursive: true, force: true });
});

test("deleteCompiledLateInfer finds a Hub id when the folder name is not the compile slug", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-late-del-slug-"));
  const blob = join(dir, "google-gemma-4-E2B-it");
  mkdirSync(blob, { recursive: true });
  writeFileSync(
    join(blob, "late-compile.json"),
    JSON.stringify({ model_id: "google/gemma-4-E2B-it", status: "ok" }),
  );
  const env = { ...process.env, LATE_COMPILED_DIR: dir, LATE_HF_HOME: join(dir, "hf") };
  const viaSlug = deleteCompiledLateInfer({
    model: "google--gemma-4-E2B-it",
    confirm: true,
    env,
  });
  assert.equal(viaSlug.ok, true);
  assert.equal(viaSlug.model, "google/gemma-4-E2B-it");
  assert.equal(existsSync(blob), false);
  assert.deepEqual(viaSlug.compiledModels, []);
  assert.deepEqual(viaSlug.compiledRows, []);
  rmSync(dir, { recursive: true, force: true });
});

test("deleteCompiledLateInfer refuses while that snapshot is serving, including a slug probe id", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-late-del-run-"));
  const blob = join(dir, "Qwen--Qwen2.5-0.5B-Instruct");
  mkdirSync(blob, { recursive: true });
  writeFileSync(
    join(blob, "late-compile.json"),
    JSON.stringify({ model_id: "Qwen/Qwen2.5-0.5B-Instruct", status: "ok" }),
  );
  const env = { ...process.env, LATE_COMPILED_DIR: dir, LATE_HF_HOME: join(dir, "hf") };
  assert.throws(
    () =>
      deleteCompiledLateInfer({
        model: "Qwen/Qwen2.5-0.5B-Instruct",
        confirm: true,
        serving: { running: true, models: ["Qwen--Qwen2.5-0.5B-Instruct"] },
        env,
      }),
    (err: unknown) => {
      assert.ok(err instanceof LateInferCompiledDeleteError);
      assert.equal(err.httpStatus, 409);
      assert.match(err.message, /Stop late-infer on your computer first/);
      assert.match(err.message, /127\.0\.0\.1:8010/);
      return true;
    },
  );
  assert.equal(existsSync(blob), true);
  rmSync(dir, { recursive: true, force: true });
});

test("deleteCompiledLateInfer requires confirm and rejects a path", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-late-del-conf-"));
  const env = { ...process.env, LATE_COMPILED_DIR: dir, LATE_HF_HOME: join(dir, "hf") };
  assert.throws(
    () => deleteCompiledLateInfer({ model: "Qwen/Qwen2.5-0.5B-Instruct", confirm: false, env }),
    /confirm=true/,
  );
  assert.throws(
    () => deleteCompiledLateInfer({ model: "/tmp/weights", confirm: true, env }),
    /Hub org\/model id|filesystem path/,
  );
  assert.equal(existsSync(dir), true);
  rmSync(dir, { recursive: true, force: true });
});
});

test("findLateInferBin prefers this application's bin over a sibling crate", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-late-infer-"));
  const orch = join(dir, "MCP_Server_AI_Agent_Project");
  const shipped = join(orch, "bin");
  const sibling = join(dir, "Local_AI_Terminal_Emulator", "target", "release");
  mkdirSync(shipped, { recursive: true });
  mkdirSync(sibling, { recursive: true });
  const ours = join(shipped, "late-infer");
  writeFileSync(ours, "orch");
  writeFileSync(join(sibling, "late-infer"), "sibling");
  const prevBin = process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN;
  const prevExplicit = process.env.AGENT_ORCHESTRATOR_LATE_INFER;
  delete process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN;
  delete process.env.AGENT_ORCHESTRATOR_LATE_INFER;
  try {
    assert.equal(findLateInferBin(() => undefined, orch), ours);
  } finally {
    if (prevBin === undefined) delete process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN;
    else process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN = prevBin;
    if (prevExplicit === undefined) delete process.env.AGENT_ORCHESTRATOR_LATE_INFER;
    else process.env.AGENT_ORCHESTRATOR_LATE_INFER = prevExplicit;
    rmSync(dir, { recursive: true, force: true });
  }
});

test("findLateInferBin uses sibling crate target/release only as last resort", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-late-infer-sib-"));
  const orch = join(dir, "MCP_Server_AI_Agent_Project");
  const sibling = join(dir, "Local_AI_Terminal_Emulator", "target", "release");
  mkdirSync(join(orch, "bin"), { recursive: true });
  mkdirSync(sibling, { recursive: true });
  const fallback = join(sibling, "late-infer");
  writeFileSync(fallback, "sibling");
  const prevBin = process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN;
  const prevExplicit = process.env.AGENT_ORCHESTRATOR_LATE_INFER;
  delete process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN;
  delete process.env.AGENT_ORCHESTRATOR_LATE_INFER;
  try {
    assert.equal(findLateInferBin(() => undefined, orch), fallback);
  } finally {
    if (prevBin === undefined) delete process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN;
    else process.env.AGENT_ORCHESTRATOR_BUNDLE_BIN = prevBin;
    if (prevExplicit === undefined) delete process.env.AGENT_ORCHESTRATOR_LATE_INFER;
    else process.env.AGENT_ORCHESTRATOR_LATE_INFER = prevExplicit;
    rmSync(dir, { recursive: true, force: true });
  }
});

test("lateInferMissingMessage tells operators to build from this repo, never install Late", () => {
  const message = lateInferMissingMessage();
  assert.match(message, /npm run infer:build/);
  assert.match(message, /your computer/);
  assert.match(message, /127\.0\.0\.1:8010/);
  assert.doesNotMatch(message, /Late repo/);
  assert.doesNotMatch(message, /open Late/i);
  assert.doesNotMatch(message, /install Late/i);
  assert.doesNotMatch(message, /cargo run -p late-infer/);
});

test("GUI missing-binary copy points at npm run infer:build and bin/late-infer", () => {
  const js = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", "gui/public/app.js"), "utf8");
  assert.match(js, /npm run infer:build/);
  assert.match(js, /bin\/late-infer/);
  assert.doesNotMatch(js, /cargo run -p late-infer --release/);
});

test("build-late-infer.sh copies into this repo bin/ and runtime/bin", () => {
  const src = readFileSync(
    join(dirname(fileURLToPath(import.meta.url)), "..", "scripts", "build-late-infer.sh"),
    "utf8",
  );
  assert.match(src, /ROOT\/bin/);
  assert.match(src, /runtime\/bin/);
  assert.match(src, /cargo build -p late-infer --release/);
  assert.match(src, /CARGO_TARGET_DIR/);
  assert.match(src, /late-infer\.stamp/);
  assert.match(src, /\.late-infer-built-for/);
  assert.doesNotMatch(src, /open Late/);
  assert.doesNotMatch(src, /Late \.deb/);
});

test("ollama serve and llama-server spawn specs bind 127.0.0.1", () => {
  const probes = {
    lspci: [
      "00:02.0 VGA compatible controller [0300]: Intel Corporation Arrow Lake-S [Intel Graphics] [8086:7d67] (rev 06)",
      "04:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]",
      "08:00.0 VGA compatible controller [0300]: Intel Corporation Battlemage G31 [Intel Graphics] [8086:e223]",
    ].join("\n"),
    drmCards: [
      {
        name: "card0",
        vendorId: "8086",
        deviceId: "e223",
        busId: "0000:04:00.0",
        driver: "xe",
        connectors: {
          "card0-DP-4": "connected",
          "card0-DP-5": "connected",
          "card0-HDMI-A-4": "connected",
        },
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
    ],
    nvidiaSmi: null as string | null,
    vulkanIcds: ["intel_icd.json"],
  };
  const ollama = ollamaServeSpec("/tmp/ollama", { probes });
  assert.equal(ollama.args[0], "serve");
  assert.equal(ollama.env.OLLAMA_HOST, "127.0.0.1:11434");
  assert.equal(ollama.host, "127.0.0.1:11434");
  assert.equal(ollama.env.GGML_VK_VISIBLE_DEVICES, "1");
  assert.equal(ollama.env.ZE_AFFINITY_MASK, "1");
  assert.equal(ollama.env.LATE_INFER_PCI, "0000:08:00.0");
  assert.equal(ollama.env.CUDA_VISIBLE_DEVICES, undefined);
  const llama = llamaServerSpec("/tmp/llama-server", "/tmp/model.gguf", 8080, { probes });
  assert.deepEqual(llama.args.slice(0, 6), [
    "-m",
    "/tmp/model.gguf",
    "--host",
    "127.0.0.1",
    "--port",
    "8080",
  ]);
  assert.ok(llama.args.includes("-ngl"));
  assert.throws(() => llamaServerSpec("/tmp/llama-server", "relative.gguf"), /absolute/);
  assert.throws(() => llamaServerSpec("/tmp/llama-server", "/tmp/../etc/passwd.gguf"), /\.\./);
  assert.throws(
    () =>
      ollamaServeSpec("/tmp/ollama", {
        probes: { ...probes, vulkanIcds: [] },
      }),
    /Vulkan ICD|GGML_VK|system RAM/,
  );
});

test("owned pid identity matches live comm+starttime and rejects a stranger", () => {
  const pid = process.pid;
  const comm = procComm(pid);
  const starttime = procStarttime(pid);
  assert.ok(comm);
  assert.ok(starttime && starttime > 0);
  const rec = { pid, starttime, comm };
  assert.equal(ownedPidIdentityLive(rec), true);
  assert.equal(ownedPidIdentityLive({ ...rec, starttime: 1 }), false);
  assert.equal(ownedPidIdentityLive({ ...rec, comm: "ollama" }), false);
  assert.equal(ownedPidMatchesLive({ pid, starttime, comm: "ollama" }, "ollama"), false);
  assert.equal(parseOwnedPid(String(pid)), undefined);
  assert.deepEqual(parseOwnedPid(formatOwnedPid(rec)), rec);
});

test("stopLocalServer does not kill a reused PID from a stale pidfile and unlinks it", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-pid-"));
  const prev = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  process.env.AGENT_ORCHESTRATOR_STATE_DIR = dir;
  try {
    const victim = process.pid;
    assert.ok(pidAlive(victim));
    const pidFile = join(dir, "ollama-serve.pid");
    writeFileSync(pidFile, `pid=${victim}\nstarttime=1\ncomm=ollama\n`);
    const result = stopLocalServer("ollama");
    assert.equal(result.running, false);
    assert.equal(pidAlive(victim), true);
    assert.equal(existsSync(pidFile), false);
  } finally {
    if (prev === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prev;
    rmSync(dir, { recursive: true, force: true });
  }
});

test("stopLocalServer unlinks a pidfile whose process is already gone", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-pid-gone-"));
  const prev = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  process.env.AGENT_ORCHESTRATOR_STATE_DIR = dir;
  try {
    const pidFile = join(dir, "llama-server.pid");
    writeFileSync(pidFile, "pid=999999\nstarttime=1\ncomm=llama-server\n");
    stopLocalServer("llamacpp");
    assert.equal(existsSync(pidFile), false);
  } finally {
    if (prev === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prev;
    rmSync(dir, { recursive: true, force: true });
  }
});

test("stopLocalServer does not kill from a legacy numeric pidfile", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-pid-legacy-"));
  const prev = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  process.env.AGENT_ORCHESTRATOR_STATE_DIR = dir;
  try {
    const victim = process.pid;
    assert.ok(pidAlive(victim));
    const pidFile = join(dir, "ollama-serve.pid");
    writeFileSync(pidFile, `${victim}\n`);
    stopLocalServer("ollama");
    assert.equal(pidAlive(victim), true);
    assert.equal(existsSync(pidFile), false);
  } finally {
    if (prev === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prev;
    rmSync(dir, { recursive: true, force: true });
  }
});


test("stopListenerIfEngine refuses non-engine PIDs", () => {
  assert.equal(stopListenerIfEngine(process.pid, "lateinfer"), false);
  assert.equal(pidAlive(process.pid), true);
});

test("fetch-inference-bins.sh pins find/curl/sha256sum off PATH", () => {
  const src = readFileSync(
    join(dirname(fileURLToPath(import.meta.url)), "..", "scripts", "fetch-inference-bins.sh"),
    "utf8",
  );
  assert.match(src, /FIND="\$\(secure_bin find\)"/);
  assert.match(src, /CURL="\$\(secure_bin curl\)"/);
  assert.match(src, /SHA256SUM="\$\(secure_bin sha256sum\)"/);
  assert.match(src, /\$FIND/);
  assert.match(src, /\$CURL/);
  assert.match(src, /\$SHA256SUM/);
  assert.match(src, /\$\{expect\}-\$\("\$BASENAME" "\$file"\)/);
  assert.match(src, /SHA-256 mismatch/);
  const code = src
    .split("\n")
    .filter((line) => !line.trim().startsWith("#"))
    .join("\n");
  assert.doesNotMatch(code, /(?:^|[\n;|&])\s*find\s+"/m);
  assert.doesNotMatch(code, /(?:^|[\n;|&])\s*curl\s+-/m);
  assert.doesNotMatch(code, /(?:^|[\n;|&])\s*sha256sum\s+"/m);
  assert.match(src, /usage: fetch-inference-bins.sh DEST \[linux-x64\|linux-arm64\|mac-arm64\|mac-x64\|win-x64\|darwin-arm64\|darwin-x64\]/);
  assert.match(src, /mac-arm64/);
  assert.match(src, /win-x64/);
  assert.match(src, /darwin-arm64/);
  assert.match(src, /INFERENCE_TARGET/);
  assert.match(src, /INFERENCE_BINS_KEY/);
  assert.match(src, /resources-win/);
  assert.match(src, /mlx_metal\*/);
  assert.match(src, /copy_engine_dir "\$ollama_bin" "\$DEST\/bin" "\$DEST\/lib\/ollama"/);
  assert.match(src, /copy_engine_dir "\$llama_bin" "\$DEST\/bin"/);
  const strip = src.slice(src.indexOf("strip_ollama_gpu_libs"), src.indexOf("\nwork="));
  assert.match(strip, /cuda\*/);
  assert.match(strip, /rocm\*/);
  assert.doesNotMatch(strip, /mlx\*/);
});

test("pack.sh stages win and mac targets without host uname", () => {
  const src = readFileSync(
    join(dirname(fileURLToPath(import.meta.url)), "..", "scripts", "pack.sh"),
    "utf8",
  );
  assert.match(src, /win-x64/);
  assert.match(src, /mac-arm64/);
  assert.match(src, /mac-x64/);
  assert.match(src, /PACK_OVERWRITE/);
  assert.match(src, /--win/);
  assert.match(src, /--mac/);
  assert.match(src, /label=darwin/);
  assert.match(src, /Distro-agnostic tarball/);
  assert.match(src, /no fpm\/electron-builder stack/);
  assert.match(src, /install_late_infer/);
  assert.match(src, /find_late_infer_bin/);
  assert.match(src, /late-infer/);
  assert.match(src, /npm run infer:build/);
  assert.match(src, /127\.0\.0\.1:8010/);
  assert.doesNotMatch(src, /fetch-inference-bins\.sh/);
  assert.doesNotMatch(src, /install_engines/);
  assert.doesNotMatch(src, /0\.0\.0\.0/);
});

test("gitignore ignores infer:build late-infer binary and target/, not scripts", () => {
  const gi = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "..", ".gitignore"), "utf8");
  assert.match(gi, /\/bin\/late-infer/);
  assert.match(gi, /\/bin\/late-infer\.stamp/);
  assert.match(gi, /\/runtime\/bin\/late-infer/);
  assert.match(gi, /\/runtime\/bin\/\.late-infer-built-for/);
  assert.match(gi, /^\/target\/$/m);
  assert.doesNotMatch(gi, /^scripts\//m);
  assert.doesNotMatch(gi, /^test\//m);
});
