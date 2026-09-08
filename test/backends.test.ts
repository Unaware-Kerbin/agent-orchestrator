import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { loadConfig, parseOrchestratorConfig, patchBackendModelYaml, patchBackendNicknameYaml, patchMcpListenHostYaml, validateConfigYaml } from "../src/config.js";
import { loadEnvFile } from "../src/env.js";
import { GEMINI_ONE_ID_ERROR, parseGeminiModelId } from "../src/providers/gemini.js";
import { OpenAIProvider } from "../src/providers/openai.js";
import { LateInferProvider } from "../src/providers/local-openai.js";
import { VllmProvider } from "../src/providers/vllm.js";
import { createProvider } from "../src/providers/index.js";
import { defaultBaseUrl, envNamesForBackend, isEnvVarName, isGrokBackend } from "../src/providers/keys.js";

const GEMINI_CFG = {
  type: "openai" as const,
  baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai",
  model: "gemini-3.6-flash",
  apiKeyEnv: "GEMINI_API_KEY",
};

function withEnv<T>(vars: Record<string, string | undefined>, fn: () => T): T {
  const prev: Record<string, string | undefined> = {};
  for (const key of Object.keys(vars)) {
    prev[key] = process.env[key];
    const value = vars[key];
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  try {
    return fn();
  } finally {
    for (const key of Object.keys(vars)) {
      if (prev[key] === undefined) delete process.env[key];
      else process.env[key] = prev[key];
    }
  }
}

const GEMINI_ENV_CLEAR = {
  GEMINI_API_KEY: undefined,
  GOOGLE_API_KEY: undefined,
  GOOGLE_GENERATIVE_AI_API_KEY: undefined,
  OPENAI_API_KEY: undefined,
};

test("isEnvVarName rejects secret-looking values", () => {
  assert.equal(isEnvVarName("GEMINI_API_KEY"), true);
  assert.equal(isEnvVarName("GOOGLE_API_KEY"), true);
  assert.equal(isEnvVarName("AQ.not-a-real-key"), false);
  assert.equal(isEnvVarName("not.an.ENV"), false);
});

test("grok / xAI key aliases and default base URL", () => {
  const cfg = {
    type: "openai" as const,
    baseUrl: "https://api.x.ai/v1",
    model: "grok-4.6",
    apiKeyEnv: "XAI_API_KEY",
  };
  assert.equal(isGrokBackend("grok", cfg), true);
  assert.equal(isGrokBackend("openai", { ...cfg, baseUrl: "https://api.openai.com/v1", apiKeyEnv: "OPENAI_API_KEY" }), false);
  const names = envNamesForBackend("grok", cfg);
  assert.ok(names.includes("XAI_API_KEY"));
  assert.ok(names.includes("GROK_API_KEY"));

  withEnv({ ...GEMINI_ENV_CLEAR, XAI_API_KEY: undefined, GROK_API_KEY: undefined, OPENAI_API_KEY: undefined }, () => {
    const health = new OpenAIProvider("grok", cfg).health();
    assert.equal(health.ready, false);
    assert.equal(health.needsKey, true);
    assert.ok(health.secretNames?.includes("XAI_API_KEY"));
    assert.ok(health.secretNames?.includes("GROK_API_KEY"));
    assert.equal(health.baseUrl, "https://api.x.ai/v1");
  });

  withEnv({ XAI_API_KEY: undefined, GROK_API_KEY: "xai-test-key", OPENAI_API_KEY: undefined }, () => {
    const health = new OpenAIProvider("grok", cfg).health();
    assert.equal(health.ready, true);
  });
});


test("gemini model must be a single id, not a list", () => {
  assert.equal(parseGeminiModelId("gemini-3.6-flash"), "gemini-3.6-flash");
  assert.equal(parseGeminiModelId("models/gemini-3.6-flash"), "gemini-3.6-flash");
  for (const bad of ["gemini-1.5-pro / gemini-1.5-flash", "gemini-2.0-flash # fastest", "gemini-1.5-pro, gemini-1.5-flash", "composer-2.5"]) {
    assert.throws(() => parseGeminiModelId(bad), (error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      assert.match(message, new RegExp(GEMINI_ONE_ID_ERROR));
      return true;
    });
  }
  assert.throws(
    () =>
      parseOrchestratorConfig({
        backends: {
          gemini: {
            type: "openai",
            baseUrl: "https://generativelanguage.googleapis.com/v1beta/openai",
            model: "gemini-1.5-pro / gemini-1.5-flash",
            apiKeyEnv: "GEMINI_API_KEY",
          },
        },
        specialists: { planner: { description: "x", backend: "gemini" } },
      }),
    (error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      assert.match(message, /use one model id, not a list/);
      return true;
    },
  );
});

test("gemini OpenAI-compat sends the configured id, not composer-2.5 or models/", async () => {
  const original = globalThis.fetch;
  const prev = process.env.GEMINI_API_KEY;
  let sentModel = "";
  let sentBody: Record<string, unknown> = {};
  try {
    process.env.GEMINI_API_KEY = "test-key-not-real";
    globalThis.fetch = (async (_url, init) => {
      sentBody = JSON.parse(String(init?.body ?? "{}")) as Record<string, unknown>;
      sentModel = typeof sentBody.model === "string" ? sentBody.model : "";
      return new Response(JSON.stringify({ choices: [{ message: { content: "ok" } }] }), { status: 200 });
    }) as typeof fetch;
    const result = await new OpenAIProvider("gemini", GEMINI_CFG).run({ prompt: "hi", model: "composer-2.5" });
    assert.equal(result.status, "finished");
    assert.equal(sentModel, "gemini-3.6-flash");
    assert.equal(sentModel.includes("composer"), false);
    assert.equal(sentModel.startsWith("models/"), false);
    assert.deepEqual(Object.keys(sentBody).sort(), ["messages", "model"]);
  } finally {
    globalThis.fetch = original;
    if (prev === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = prev;
  }
});

test("gemini OpenAI-compat remaps retired 2.0-flash and formats 404 with Google’s suggestion", async () => {
  const original = globalThis.fetch;
  const prev = process.env.GEMINI_API_KEY;
  let sentModel = "";
  try {
    process.env.GEMINI_API_KEY = "test-key-not-real";
    globalThis.fetch = (async (_url, init) => {
      const body = JSON.parse(String(init?.body ?? "{}")) as { model?: string };
      sentModel = body.model ?? "";
      return new Response(
        JSON.stringify([
          {
            error: {
              code: 404,
              message:
                "This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.6-flash for the latest features and improvements.",
              status: "NOT_FOUND",
            },
          },
        ]),
        { status: 404, headers: { "content-type": "application/json" } },
      );
    }) as typeof fetch;
    const retired = { ...GEMINI_CFG, model: "gemini-2.0-flash" };
    const result = await new OpenAIProvider("gemini", retired).run({ prompt: "hi" });
    assert.equal(sentModel, "gemini-3.6-flash");
    assert.equal(sentModel.startsWith("models/"), false);
    assert.equal(result.status, "error");
    assert.match(result.error ?? "", /for model gemini-3\.6-flash/);
    assert.match(result.error ?? "", /Google suggests gemini-3\.6-flash/);
  } finally {
    globalThis.fetch = original;
    if (prev === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = prev;
  }
});

test("patchBackendModelYaml updates gemini model without rewriting other backends", () => {
  const yaml = `backends:
  gemini:
    type: openai
    model: gemini-1.5-flash
    apiKeyEnv: GEMINI_API_KEY
  vllm-local:
    type: vllm
    model: Qwen/Qwen2.5-0.5B-Instruct
specialists:
  chat:
    description: t
    backend: gemini
`;
  const next = patchBackendModelYaml(yaml, "gemini", "gemini-3.6-flash");
  assert.match(next, /model: "gemini-3\.6-flash"/);
  assert.match(next, /vllm-local:/);
  assert.match(next, /Qwen\/Qwen2\.5-0\.5B-Instruct/);
  const parsed = validateConfigYaml(next);
  assert.equal(parsed.backends.gemini?.type, "openai");
  if (parsed.backends.gemini?.type === "openai") {
    assert.equal(parsed.backends.gemini.model, "gemini-3.6-flash");
  }
});

test("patchBackendModelYaml quotes colon tags and rejects newline injection", () => {
  const yaml = `backends:
  ollama:
    type: ollama
    baseUrl: http://127.0.0.1:11434/v1
    model: llama3.1
specialists:
  chat:
    description: t
    backend: ollama
`;
  const poison =
    "llama3.1\n    type: openai\n    baseUrl: https://api.openai.com/v1\n    apiKeyEnv: OPENAI_API_KEY";
  assert.throws(() => patchBackendModelYaml(yaml, "ollama", poison), /line breaks/);
  const next = patchBackendModelYaml(yaml, "ollama", "llama3.2:latest");
  assert.match(next, /model: "llama3\.2:latest"/);
  assert.doesNotMatch(next, /type: openai/);
  assert.doesNotMatch(next, /api\.openai\.com/);
  const parsed = validateConfigYaml(next);
  assert.equal(parsed.backends.ollama?.type, "ollama");
  if (parsed.backends.ollama?.type === "ollama") {
    assert.equal(parsed.backends.ollama.model, "llama3.2:latest");
    assert.equal(parsed.backends.ollama.baseUrl, "http://127.0.0.1:11434/v1");
  }
});

test("parseOrchestratorConfig reads optional nickname", () => {
  const config = parseOrchestratorConfig({
    backends: {
      gemini: {
        type: "openai",
        model: "gemini-3.6-flash",
        apiKeyEnv: "GEMINI_API_KEY",
        nickname: "  Flash  ",
      },
    },
    specialists: { planner: { description: "x", backend: "gemini" } },
  });
  assert.equal(config.backends.gemini?.nickname, "Flash");
});

test("parseOrchestratorConfig rejects overlong nicknames", () => {
  assert.throws(
    () =>
      parseOrchestratorConfig({
        backends: {
          gemini: {
            type: "openai",
            model: "gemini-3.6-flash",
            apiKeyEnv: "GEMINI_API_KEY",
            nickname: "x".repeat(49),
          },
        },
        specialists: { planner: { description: "x", backend: "gemini" } },
      }),
    /48 characters/,
  );
});

test("patchBackendNicknameYaml sets and clears nickname without rewriting other backends", () => {
  const yaml = `backends:
  gemini:
    type: openai
    model: gemini-3.6-flash
    apiKeyEnv: GEMINI_API_KEY
  vllm-local:
    type: vllm
    model: Qwen/Qwen2.5-0.5B-Instruct
specialists:
  chat:
    description: t
    backend: gemini
`;
  const named = patchBackendNicknameYaml(yaml, "vllm-local", "Arc Qwen");
  assert.match(named, /nickname: "Arc Qwen"/);
  assert.match(named, /model: gemini-3\.6-flash/);
  const cleared = patchBackendNicknameYaml(named, "vllm-local", undefined);
  assert.equal(/nickname:/.test(cleared.split("vllm-local:")[1]?.split("specialists:")[0] ?? "nickname:"), false);
  assert.match(cleared, /Qwen\/Qwen2\.5-0\.5B-Instruct/);
});

test("mcp.listen_host accepts one RFC1918 IP and refuses 0.0.0.0", () => {
  const base = {
    backends: { gemini: { type: "openai", model: "gemini-3.6-flash", apiKeyEnv: "GEMINI_API_KEY" } },
    specialists: { planner: { description: "x", backend: "gemini" } },
  };
  const ok = parseOrchestratorConfig({ ...base, mcp: { listen_host: "192.168.2.139" } });
  assert.equal(ok.mcp?.listenHost, "192.168.2.139");
  assert.equal(parseOrchestratorConfig(base).mcp, undefined);
  assert.throws(() => parseOrchestratorConfig({ ...base, mcp: { listen_host: "0.0.0.0" } }), /0\.0\.0\.0|private/);
  assert.throws(() => parseOrchestratorConfig({ ...base, mcp: { listen_host: "8.8.8.8" } }), /private/);
  const yaml = `backends:
  gemini:
    type: openai
    model: gemini-3.6-flash
    apiKeyEnv: GEMINI_API_KEY
specialists:
  planner:
    description: x
    backend: gemini
`;
  const patched = patchMcpListenHostYaml(yaml, "192.168.2.139");
  assert.match(patched, /listen_host: "192\.168\.2\.139"/);
  assert.match(patched, /type: openai/);
  const parsed = validateConfigYaml(patched);
  assert.equal(parsed.mcp?.listenHost, "192.168.2.139");
  const autoYaml = patchMcpListenHostYaml(yaml, "auto");
  assert.match(autoYaml, /listen_host: "auto"/);
  assert.equal(validateConfigYaml(autoYaml).mcp?.listenHost, "auto");
  const emptyYaml = patchMcpListenHostYaml(yaml, "");
  assert.match(emptyYaml, /listen_host: "auto"/);
  const autoCfg = parseOrchestratorConfig({ ...base, mcp: { listen_host: "auto" } });
  assert.equal(autoCfg.mcp?.listenHost, "auto");
});

test("gemini is ready when GEMINI_API_KEY is set", () => {
  withEnv({ ...GEMINI_ENV_CLEAR, GEMINI_API_KEY: "test-key-not-real" }, () => {
    const health = new OpenAIProvider("gemini", GEMINI_CFG).health();
    assert.equal(health.ready, true);
    assert.ok(health.secretNames?.includes("GEMINI_API_KEY"));
    assert.ok(health.secretNames?.includes("GOOGLE_API_KEY"));
    assert.equal(health.model, "gemini-3.6-flash");
    assert.equal(health.modelChoices?.includes("gemini-1.5-flash"), false);
    assert.equal(health.modelChoices?.includes("gemini-2.0-flash"), false);
    assert.equal(health.modelChoices?.[0], "gemini-3.6-flash");
  });
});

test("gemini is ready when GOOGLE_API_KEY is set even if config names GEMINI_API_KEY", () => {
  withEnv({ ...GEMINI_ENV_CLEAR, GOOGLE_API_KEY: "test-key-not-real" }, () => {
    const health = new OpenAIProvider("gemini", GEMINI_CFG).health();
    assert.equal(health.ready, true);
  });
});

test("gemini is not ready when no Google/Gemini key is set", () => {
  withEnv(GEMINI_ENV_CLEAR, () => {
    const health = new OpenAIProvider("gemini", GEMINI_CFG).health();
    assert.equal(health.ready, false);
    assert.match(health.reason ?? "", /GEMINI_API_KEY/);
    assert.equal((health.reason ?? "").includes("test-key"), false);
  });
});

test("invalid apiKeyEnv is ignored; GEMINI_API_KEY still enables gemini", () => {
  const cfg = { ...GEMINI_CFG, apiKeyEnv: "not.an.ENV" };
  withEnv(GEMINI_ENV_CLEAR, () => {
    assert.equal(new OpenAIProvider("gemini", cfg).health().ready, false);
  });
  withEnv({ ...GEMINI_ENV_CLEAR, GEMINI_API_KEY: "test-key-not-real" }, () => {
    assert.equal(new OpenAIProvider("gemini", cfg).health().ready, true);
  });
});

test("config parse rejects secret stuffed into apiKeyEnv without echoing it", () => {
  const stuffed = "AQ.not-a-real-key-value";
  assert.throws(
    () =>
      parseOrchestratorConfig({
        backends: {
          gemini: {
            type: "openai",
            model: "gemini-3.6-flash",
            apiKeyEnv: stuffed,
          },
        },
        specialists: { planner: { description: "x", backend: "gemini" } },
      }),
    (error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      assert.match(message, /environment variable name/);
      assert.equal(message.includes(stuffed), false);
      return true;
    },
  );
});

test("repo agents.config.yaml names GEMINI_API_KEY and defaults local to late-infer :8010", () => {
  const config = loadConfig();
  const gemini = config.backends.gemini;
  assert.equal(gemini?.type, "openai");
  if (gemini?.type === "openai") {
    assert.equal(gemini.apiKeyEnv, "GEMINI_API_KEY");
    assert.equal(gemini.model, "gemini-3.6-flash");
    assert.match(gemini.baseUrl ?? "", /generativelanguage\.googleapis\.com/);
  }
  const late = config.backends["late-infer"];
  assert.equal(late?.type, "lateinfer");
  if (late?.type === "lateinfer") {
    assert.equal(late.baseUrl, "http://127.0.0.1:8010/v1");
    assert.ok(late.model.length > 0);
  }
  assert.equal(config.specialists["late-infer-chat"]?.backend, "late-infer");
  const vllm = config.backends["vllm-local"];
  assert.equal(vllm?.type, "vllm");
  if (vllm?.type === "vllm") {
    assert.equal(vllm.baseUrl, "http://127.0.0.1:8000/v1");
    assert.ok(vllm.model.length > 0);
  }
  assert.equal(config.specialists["vllm-chat"]?.backend, "vllm-local");
  const ollama = config.backends.ollama;
  assert.equal(ollama?.type, "ollama");
  if (ollama?.type === "ollama") {
    assert.equal(ollama.baseUrl, "http://127.0.0.1:11434/v1");
    assert.ok(ollama.model.length > 0);
  }
  assert.equal(config.specialists["ollama-chat"]?.backend, "ollama");
  const grok = config.backends.grok;
  assert.equal(grok?.type, "openai");
  if (grok?.type === "openai") {
    assert.equal(grok.apiKeyEnv, "XAI_API_KEY");
    assert.equal(grok.baseUrl, "https://api.x.ai/v1");
    assert.match(grok.model ?? "", /^grok-/);
  }
  assert.equal(config.specialists["grok-chat"]?.backend, "grok");
});

test("lateinfer type defaults to 127.0.0.1:8010 and does not need Cloud AI", () => {
  const parsed = validateConfigYaml(`
backends:
  late-infer:
    type: lateinfer
    model: Qwen/Qwen2.5-0.5B-Instruct
specialists:
  chat:
    description: t
    backend: late-infer
`);
  const late = parsed.backends["late-infer"];
  assert.equal(late?.type, "lateinfer");
  if (late?.type !== "lateinfer") throw new Error("expected lateinfer");
  assert.equal(late.baseUrl, "http://127.0.0.1:8010/v1");
  assert.equal(defaultBaseUrl(late), "http://127.0.0.1:8010/v1");
  assert.notEqual(late.baseUrl, "http://127.0.0.1:8000/v1");

  withEnv(
    {
      CURSOR_API_KEY: undefined,
      GEMINI_API_KEY: undefined,
      GOOGLE_API_KEY: undefined,
      OPENAI_API_KEY: undefined,
    },
    () => {
      const provider = new LateInferProvider("late-infer", { ...late, probe: false });
      const health = provider.health();
      assert.equal(health.ready, true);
      assert.equal(health.needsKey, false);
      assert.equal(health.writesLocalFiles, false);
      assert.equal(health.type, "lateinfer");
      assert.match(health.baseUrl ?? "", /127\.0\.0\.1:8010\/v1/);
      assert.doesNotMatch(health.baseUrl ?? "", /:8000\//);
      assert.equal(health.reason?.includes("Cloud AI"), false);
    },
  );
});

test("lateinfer health is not ready until /models probe succeeds", () => {
  const provider = new LateInferProvider("late-infer", {
    type: "lateinfer",
    baseUrl: "http://127.0.0.1:8010/v1",
    model: "Qwen/Qwen2.5-0.5B-Instruct",
  });
  const health = provider.health();
  assert.equal(health.ready, false);
  assert.equal(health.needsKey, false);
  assert.match(health.reason ?? "", /will probe/);
  assert.match(health.baseUrl ?? "", /8010\/v1/);
});

test("lateinfer probe marks ready on HTTP success and not-ready on ECONNREFUSED", async () => {
  const original = globalThis.fetch;
  const provider = new LateInferProvider("late-infer", {
    type: "lateinfer",
    baseUrl: "http://127.0.0.1:8010/v1",
    model: "x",
  });
  try {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ data: [{ id: "Qwen/Qwen2.5-0.5B-Instruct" }] }), { status: 200 })) as typeof fetch;
    const ok = await provider.probe();
    assert.equal(ok.ready, true);
    assert.equal(ok.writesLocalFiles, false);
    assert.equal(ok.needsKey, false);
    assert.match(ok.reason ?? "", /late-infer running|8010/i);

    globalThis.fetch = (async () => {
      const error = new Error("fetch failed");
      (error as Error & { cause: { code: string } }).cause = { code: "ECONNREFUSED" };
      throw error;
    }) as typeof fetch;
    const down = await provider.probe();
    assert.equal(down.ready, false);
    assert.match(down.reason ?? "", /Late infer not running at http:\/\/127\.0\.0\.1:8010\/v1/i);
    assert.doesNotMatch(down.reason ?? "", /Cloud AI|CURSOR_API_KEY|8000\/v1/);
  } finally {
    globalThis.fetch = original;
  }
});

test("createProvider lateinfer talks OpenAI /v1 on :8010, not vLLM :8000", async () => {
  const original = globalThis.fetch;
  let posted = "";
  try {
    globalThis.fetch = (async (url) => {
      posted = String(url);
      return new Response(JSON.stringify({ choices: [{ message: { content: "ok" } }] }), { status: 200 });
    }) as typeof fetch;
    const provider = createProvider("late-infer", {
      type: "lateinfer",
      baseUrl: "http://127.0.0.1:8010/v1",
      model: "Qwen/Qwen2.5-0.5B-Instruct",
      probe: false,
    });
    assert.equal(provider.type, "lateinfer");
    const result = await provider.run({ prompt: "hi" });
    assert.equal(result.status, "finished");
    assert.match(posted, /127\.0\.0\.1:8010\/v1\/chat\/completions/);
    assert.doesNotMatch(posted, /:8000\//);
  } finally {
    globalThis.fetch = original;
  }
});

test("vllm type parses and allows multiple endpoints", () => {
  const parsed = validateConfigYaml(`
backends:
  vllm-a:
    type: vllm
    baseUrl: http://127.0.0.1:8000/v1
    model: model-a
  vllm-b:
    type: vllm
    baseUrl: http://127.0.0.1:8001/v1
    model: model-b
    probe: false
specialists:
  chat:
    description: t
    backend: vllm-a
`);
  assert.equal(parsed.backends["vllm-a"]?.type, "vllm");
  assert.equal(parsed.backends["vllm-b"]?.type, "vllm");
  if (parsed.backends["vllm-b"]?.type === "vllm") {
    assert.equal(parsed.backends["vllm-b"].probe, false);
    assert.equal(parsed.backends["vllm-b"].model, "model-b");
  }
});

test("vllm is ready without a cloud key when probe is disabled", () => {
  withEnv({ VLLM_API_KEY: undefined }, () => {
    const provider = new VllmProvider("vllm-local", {
      type: "vllm",
      baseUrl: "http://127.0.0.1:8000/v1",
      model: "meta-llama/Llama-3.1-8B-Instruct",
      probe: false,
    });
    const health = provider.health();
    assert.equal(health.ready, true);
    assert.equal(health.needsKey, false);
    assert.equal(health.writesLocalFiles, false);
    assert.match(health.reason ?? "", /8000\/v1/);
    assert.equal((health.reason ?? "").includes("missing API key"), false);
  });
});

test("vllm health is not ready until /models probe succeeds", () => {
  const provider = new VllmProvider("vllm-gemma-4-e2b-it", {
    type: "vllm",
    baseUrl: "http://127.0.0.1:8002/v1",
    model: "google/gemma-4-E2B-it",
  });
  const health = provider.health();
  assert.equal(health.ready, false);
  assert.equal(health.writesLocalFiles, false);
  assert.match(health.reason ?? "", /will probe/);
});

test("vllm probe marks ready on HTTP success and not-ready on ECONNREFUSED", async () => {
  const original = globalThis.fetch;
  const provider = new VllmProvider("vllm-local", {
    type: "vllm",
    baseUrl: "http://127.0.0.1:8000/v1",
    model: "x",
  });
  try {
    globalThis.fetch = (async () => new Response("{}", { status: 200 })) as typeof fetch;
    const ok = await provider.probe();
    assert.equal(ok.ready, true);
    assert.equal(ok.writesLocalFiles, false);
    assert.match(ok.reason ?? "", /reachable/);

    globalThis.fetch = (async () => {
      const error = new Error("fetch failed");
      (error as Error & { cause: { code: string } }).cause = { code: "ECONNREFUSED" };
      throw error;
    }) as typeof fetch;
    const down = await provider.probe();
    assert.equal(down.ready, false);
    assert.match(down.reason ?? "", /vLLM not running at http:\/\/127\.0\.0\.1:8000\/v1/);
    assert.doesNotMatch(down.reason ?? "", /API key|VLLM_API_KEY/);
  } finally {
    globalThis.fetch = original;
  }
});

test("vllm run maps connection refused to a clear error", async () => {
  const original = globalThis.fetch;
  const provider = new VllmProvider("vllm-local", {
    type: "vllm",
    baseUrl: "http://127.0.0.1:9",
    model: "x",
    probe: false,
  });
  try {
    globalThis.fetch = (async () => {
      const error = new Error("fetch failed");
      (error as Error & { cause: { code: string } }).cause = { code: "ECONNREFUSED" };
      throw error;
    }) as typeof fetch;
    const result = await provider.run({ prompt: "hi" });
    assert.equal(result.status, "error");
    assert.match(result.error ?? "", /vLLM not running at http:\/\/127\.0\.0\.1:9/);
  } finally {
    globalThis.fetch = original;
  }
});

test("loadEnvFile fills empty process env from file without overwrite of non-empty", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-env-"));
  const file = join(dir, ".env");
  writeFileSync(file, "GEMINI_API_KEY=from-file\nOTHER_KEY=keep-me\n");
  withEnv({ GEMINI_API_KEY: "", OTHER_KEY: "already" }, () => {
    loadEnvFile(file, false);
    assert.equal(process.env.GEMINI_API_KEY, "from-file");
    assert.equal(process.env.OTHER_KEY, "already");
    loadEnvFile(file, true);
    assert.equal(process.env.GEMINI_API_KEY, "from-file");
    assert.equal(process.env.OTHER_KEY, "keep-me");
  });
});
