import assert from "node:assert/strict";
import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { parseOrchestratorConfig, validateConfigYaml } from "../src/config.js";
import { pidAlive } from "../src/platform.js";
import {
  isLoopbackHttpUrl,
  isLoopbackHostname,
  normalizeLoopbackOpenAiUrl,
} from "../src/local-servers/loopback.js";
import { llamaServerOnPath, ollamaOnPath, probeLlamaCpp, probeOllama } from "../src/local-servers/status.js";
import { findEngineBin } from "../src/local-servers/bins.js";
import {
  formatOwnedPid,
  llamaServerSpec,
  ollamaServeSpec,
  ownedPidIdentityLive,
  ownedPidMatchesLive,
  parseOwnedPid,
  procComm,
  procStarttime,
  stopLocalServer,
} from "../src/local-servers/spawn.js";
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

test("ollama serve and llama-server spawn specs bind 127.0.0.1", () => {
  const ollama = ollamaServeSpec("/tmp/ollama");
  assert.equal(ollama.args[0], "serve");
  assert.equal(ollama.env.OLLAMA_HOST, "127.0.0.1:11434");
  assert.equal(ollama.host, "127.0.0.1:11434");
  const llama = llamaServerSpec("/tmp/llama-server", "/tmp/model.gguf");
  assert.deepEqual(llama.args, ["-m", "/tmp/model.gguf", "--host", "127.0.0.1", "--port", "8080"]);
  assert.throws(() => llamaServerSpec("/tmp/llama-server", "relative.gguf"), /absolute/);
  assert.throws(() => llamaServerSpec("/tmp/llama-server", "/tmp/../etc/passwd.gguf"), /\.\./);
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
  assert.match(src, /fetch-inference-bins\.sh/);
  assert.match(src, /PACK_OVERWRITE/);
  assert.match(src, /--win/);
  assert.match(src, /--mac/);
  assert.match(src, /runtime-win/);
  assert.match(src, /label=darwin/);
  assert.match(src, /Distro-agnostic tarball/);
  assert.match(src, /no fpm\/electron-builder stack/);
});
