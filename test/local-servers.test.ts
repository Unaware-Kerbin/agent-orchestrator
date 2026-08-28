import assert from "node:assert/strict";
import { test } from "node:test";
import { parseOrchestratorConfig, validateConfigYaml } from "../src/config.js";
import {
  isLoopbackHttpUrl,
  isLoopbackHostname,
  normalizeLoopbackOpenAiUrl,
} from "../src/local-servers/loopback.js";
import { llamaServerOnPath, ollamaOnPath, probeLlamaCpp, probeOllama } from "../src/local-servers/status.js";
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
