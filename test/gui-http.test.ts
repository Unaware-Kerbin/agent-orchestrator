import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { originAllowed, mcpOriginAllowed, hostAllowed, startGuiServer } from "../src/gui/http.js";

async function freeLoopbackPort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const probe = createServer();
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const addr = probe.address();
      const port = typeof addr === "object" && addr ? addr.port : 0;
      probe.close((err) => (err ? reject(err) : resolve(port)));
    });
  });
}

test("originAllowed accepts 127.0.0.1 and localhost on the GUI port", () => {
  assert.equal(originAllowed(undefined, 8787), true);
  assert.equal(originAllowed("http://127.0.0.1:8787", 8787), true);
  assert.equal(originAllowed("http://localhost:8787", 8787), true);
  assert.equal(originAllowed("http://127.0.0.1:9999", 8787), false);
  assert.equal(originAllowed("http://example.com:8787", 8787), false);
  assert.equal(originAllowed("https://127.0.0.1:8787", 8787), false);
  assert.equal(hostAllowed("127.0.0.1:8787", 8787), true);
  assert.equal(hostAllowed("localhost:8787", 8787), true);
  assert.equal(hostAllowed("0.0.0.0:8787", 8787), false);
});

test("mcpOriginAllowed accepts any loopback Origin; GUI /api stays same-port", () => {
  assert.equal(mcpOriginAllowed(undefined), true);
  assert.equal(mcpOriginAllowed("http://127.0.0.1:5173"), true);
  assert.equal(mcpOriginAllowed("http://localhost:7430"), true);
  assert.equal(mcpOriginAllowed("http://10.0.0.12:5173"), false);
  assert.equal(mcpOriginAllowed("https://127.0.0.1:5173"), false);
  assert.equal(originAllowed("http://127.0.0.1:5173", 8787), false);
});

test("mcpOriginAllowed accepts the bound private host; evil.com is rejected", () => {
  assert.equal(mcpOriginAllowed("http://192.168.2.139:8790", "192.168.2.139"), true);
  assert.equal(mcpOriginAllowed("http://192.168.2.139:5173", "192.168.2.139"), true);
  assert.equal(mcpOriginAllowed(undefined, "192.168.2.139"), true);
  assert.equal(mcpOriginAllowed("http://evil.com", "192.168.2.139"), false);
  assert.equal(mcpOriginAllowed("https://evil.com", "192.168.2.139"), false);
  assert.equal(originAllowed("http://192.168.2.139:8787", 8787, "192.168.2.139"), true);
  assert.equal(originAllowed("http://evil.com:8787", 8787, "192.168.2.139"), false);
  assert.equal(hostAllowed("192.168.2.139:8787", 8787, "192.168.2.139"), true);
  assert.equal(hostAllowed("192.168.3.116:8787", 8787, "192.168.2.139"), false);
  assert.equal(hostAllowed("0.0.0.0:8787", 8787, "192.168.2.139"), false);
});

test("startGuiServer Copy MCP URL uses the entered private host", () => {
  const events = new EventEmitter();
  const orchestrator = {
    events,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: { snapshot: () => ({}) },
    store: { list: () => [], get: () => undefined },
    configPath: "/tmp/agents.config.yaml",
    config: { backends: {}, specialists: {}, mcp: { listenHost: "192.168.2.139" } },
    allowlist: { list: () => [] },
    defaultCwd: () => "/",
    reloadConfig: () => undefined,
  };
  const chat = {
    list: () => [],
    create: () => ({}),
    get: () => ({}),
    delete: () => false,
    setPin: () => ({}),
    send: async () => ({}),
    runAction: async () => ({}),
    resolveApproval: async () => ({}),
  };
  const { server, listen } = startGuiServer({
    orchestrator: orchestrator as never,
    chat: chat as never,
    token: "test-token",
    port: 8787,
    host: "192.168.2.139",
  });
  try {
    assert.equal(listen.host, "192.168.2.139");
    assert.equal(listen.mcpUrl, "http://192.168.2.139:8787/mcp");
    assert.equal(listen.url, "http://192.168.2.139:8787");
  } finally {
    server.close();
  }
});

test("POST /api/vllm/start returns 202 with jobId without waiting for health", async () => {
  const events = new EventEmitter();
  let phase: "idle" | "starting" | "running" = "idle";
  const vllm = () => ({
    running: phase === "running",
    healthy: phase === "running",
    phase,
    host: "127.0.0.1",
    backendId: "vllm-local",
    installed: true,
    installHint: "",
    jobId: "job-test",
  });
  const orchestrator = {
    events,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: {
      snapshot: () => ({ vllm: vllm(), models: [], recommended: [], jobs: [], hardware: {}, intelDocker: {} }),
      vllmStatus: () => vllm(),
      startVllmAsync: (input: { modelId: string }) => {
        phase = "starting";
        events.emit("vllm", vllm());
        void delay(250).then(() => {
          phase = "running";
          events.emit("vllm", { ...vllm(), modelId: input.modelId, port: 8001 });
        });
        return { status: "starting" as const, jobId: "job-test", vllm: vllm() };
      },
      stopVllm: () => {
        phase = "idle";
        return vllm();
      },
      listHardware: () => ({}),
      listModels: () => ({ vllm: vllm(), models: [] }),
      recommend: () => ({ recommendations: [] }),
      download: () => ({}),
    },
    store: { list: () => [], get: () => undefined },
    configPath: "/tmp/agents.config.yaml",
    config: { backends: {}, specialists: {} },
    allowlist: { list: () => [] },
    defaultCwd: () => "/",
    reloadConfig: () => undefined,
    dispatch: async () => ({}),
    followUp: async () => ({}),
    runWorkflow: async () => ({ workflow: "", status: "ok", summary: "", runs: [] }),
  };
  const chat = {
    list: () => [],
    create: () => ({}),
    get: () => ({}),
    delete: () => false,
    setPin: () => ({}),
    send: async () => ({}),
    runAction: async () => ({}),
    resolveApproval: async () => ({}),
  };
  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
  const { server, listen } = startGuiServer({
    orchestrator: orchestrator as never,
    chat: chat as never,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  const base = `http://127.0.0.1:${port}`;
  try {
    const t0 = Date.now();
    const startRes = await fetch(`${base}/api/vllm/start`, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ modelId: "qwen-7b", runtime: "docker" }),
    });
    const elapsed = Date.now() - t0;
    assert.equal(startRes.status, 202);
    assert.ok(elapsed < 400, `start POST took ${elapsed}ms`);
    const body = (await startRes.json()) as { status: string; jobId: string };
    assert.equal(body.status, "starting");
    assert.equal(body.jobId, "job-test");

    const statusRes = await fetch(`${base}/api/vllm`, {
      headers: { authorization: `Bearer ${token}` },
    });
    assert.equal(statusRes.status, 200);
    const statusBody = (await statusRes.json()) as { phase: string };
    assert.equal(statusBody.phase, "starting");

    const unauthorized = await fetch(`${base}/api/vllm/start`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ modelId: "qwen-7b" }),
    });
    assert.equal(unauthorized.status, 401);
  } finally {
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});

test("GET /api/secrets never echoes HF_TOKEN; PUT stores; DELETE clears", async () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-gui-secrets-"));
  const prevState = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  const prevHf = process.env.HF_TOKEN;
  const prevHub = process.env.HUGGING_FACE_HUB_TOKEN;
  process.env.AGENT_ORCHESTRATOR_STATE_DIR = dir;
  delete process.env.HF_TOKEN;
  delete process.env.HUGGING_FACE_HUB_TOKEN;

  const events = new EventEmitter();
  const orchestrator = {
    events,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: {
      snapshot: () => ({ models: [], recommended: [], jobs: [], hardware: {}, intelDocker: {}, hfTokenSet: false }),
    },
    store: { list: () => [], get: () => undefined },
    configPath: "/tmp/agents.config.yaml",
    config: { backends: {}, specialists: {} },
    allowlist: { list: () => [] },
    defaultCwd: () => "/",
    reloadConfig: () => undefined,
    dispatch: async () => ({}),
    followUp: async () => ({}),
    runWorkflow: async () => ({ workflow: "", status: "ok", summary: "", runs: [] }),
  };
  const chat = {
    list: () => [],
    create: () => ({}),
    get: () => ({}),
    delete: () => false,
    setPin: () => ({}),
    send: async () => ({}),
    runAction: async () => ({}),
    resolveApproval: async () => ({}),
  };
  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
  const { server, listen } = startGuiServer({
    orchestrator: orchestrator as never,
    chat: chat as never,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  const base = `http://127.0.0.1:${port}`;
  const hfToken = "hf_guiHttpTestTokenNotReal88";
  const auth = { authorization: `Bearer ${token}`, "content-type": "application/json" };
  try {
    const put = await fetch(`${base}/api/secrets`, {
      method: "PUT",
      headers: auth,
      body: JSON.stringify({ name: "HF_TOKEN", value: hfToken }),
    });
    assert.equal(put.status, 200);
    const putBody = await put.text();
    assert.equal(putBody.includes(hfToken), false);
    const putJson = JSON.parse(putBody) as { secrets: Array<{ name: string; set: boolean }> };
    assert.equal(putJson.secrets.find((s) => s.name === "HF_TOKEN")?.set, true);
    assert.equal("value" in (putJson.secrets.find((s) => s.name === "HF_TOKEN") ?? {}), false);

    const get = await fetch(`${base}/api/secrets`, { headers: { authorization: `Bearer ${token}` } });
    assert.equal(get.status, 200);
    const getText = await get.text();
    assert.equal(getText.includes(hfToken), false);
    const getJson = JSON.parse(getText) as { secrets: Array<{ name: string; set: boolean }> };
    assert.equal(getJson.secrets.find((s) => s.name === "HF_TOKEN")?.set, true);

    const cleared = await fetch(`${base}/api/secrets`, {
      method: "DELETE",
      headers: auth,
      body: JSON.stringify({ name: "HF_TOKEN" }),
    });
    assert.equal(cleared.status, 200);
    const clearedText = await cleared.text();
    assert.equal(clearedText.includes(hfToken), false);
    const clearedJson = JSON.parse(clearedText) as { secrets: Array<{ name: string; set: boolean }>; cleared: string[] };
    assert.equal(clearedJson.secrets.find((s) => s.name === "HF_TOKEN")?.set, false);
    assert.ok(clearedJson.cleared.includes("HF_TOKEN"));
  } finally {
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
    if (prevState === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prevState;
    if (prevHf === undefined) delete process.env.HF_TOKEN;
    else process.env.HF_TOKEN = prevHf;
    if (prevHub === undefined) delete process.env.HUGGING_FACE_HUB_TOKEN;
    else process.env.HUGGING_FACE_HUB_TOKEN = prevHub;
    rmSync(dir, { recursive: true, force: true });
  }
});

test("POST /api/allowlist and workspace reject a missing drop path with 400", async () => {
  const events = new EventEmitter();
  const missing = new Error("Directory does not exist: /no/such/orchestrator-drop");
  const orchestrator = {
    events,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: {
      snapshot: () => ({ models: [], recommended: [], jobs: [], hardware: {}, intelDocker: {} }),
    },
    store: { list: () => [], get: () => undefined },
    configPath: "/tmp/agents.config.yaml",
    config: { backends: {}, specialists: {} },
    allowlist: {
      list: () => ["/tmp"],
      add: (path: string) => {
        throw new Error(`Directory does not exist: ${path}`);
      },
      assertCwd: (path: string) => {
        throw new Error(`Directory does not exist: ${path}`);
      },
    },
    defaultCwd: () => "/tmp",
    reloadConfig: () => undefined,
    dispatch: async () => ({}),
    followUp: async () => ({}),
    runWorkflow: async () => ({ workflow: "", status: "ok", summary: "", runs: [] }),
  };
  const chat = {
    list: () => [],
    create: () => ({}),
    get: () => ({}),
    delete: () => false,
    setPin: () => ({}),
    setWorkspaceDir: () => {
      throw missing;
    },
    send: async () => ({}),
    runAction: async () => ({}),
    resolveApproval: async () => ({}),
  };
  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
  const { server, listen } = startGuiServer({
    orchestrator: orchestrator as never,
    chat: chat as never,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  const base = `http://127.0.0.1:${port}`;
  const auth = { authorization: `Bearer ${token}`, "content-type": "application/json" };
  try {
    const grant = await fetch(`${base}/api/allowlist`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ path: "/no/such/orchestrator-drop" }),
    });
    assert.equal(grant.status, 400);
    const grantBody = (await grant.json()) as { error?: string };
    assert.match(grantBody.error ?? "", /does not exist/i);

    const workspace = await fetch(`${base}/api/chats/thread-1/workspace`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ path: "/no/such/orchestrator-drop" }),
    });
    assert.equal(workspace.status, 400);
  } finally {
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});

test("POST /api/updates/apply without confirm does not download", async () => {
  const events = new EventEmitter();
  const orchestrator = {
    events,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: {
      snapshot: () => ({ models: [], recommended: [], jobs: [], hardware: {}, intelDocker: {} }),
    },
    store: { list: () => [], get: () => undefined },
    configPath: "/tmp/agents.config.yaml",
    config: { backends: {}, specialists: {} },
    allowlist: { list: () => [] },
    defaultCwd: () => "/",
    reloadConfig: () => undefined,
    dispatch: async () => ({}),
    followUp: async () => ({}),
    runWorkflow: async () => ({ workflow: "", status: "ok", summary: "", runs: [] }),
  };
  const chat = {
    list: () => [],
    create: () => ({}),
    get: () => ({}),
    delete: () => false,
    setPin: () => ({}),
    send: async () => ({}),
    runAction: async () => ({}),
    resolveApproval: async () => ({}),
  };
  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
  const { server, listen } = startGuiServer({
    orchestrator: orchestrator as never,
    chat: chat as never,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  const base = `http://127.0.0.1:${port}`;
  const auth = { authorization: `Bearer ${token}`, "content-type": "application/json" };
  try {
    const denied = await fetch(`${base}/api/updates/apply`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ choice: "both", confirm: false }),
    });
    assert.equal(denied.status, 400);
    const deniedBody = (await denied.json()) as { error?: string };
    assert.match(deniedBody.error ?? "", /Say yes first/i);

    const missing = await fetch(`${base}/api/updates/apply`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ choice: "both" }),
    });
    assert.equal(missing.status, 400);
  } finally {
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});

test("GET /api/local-servers includes late-infer snapshot fields", async () => {
  const events = new EventEmitter();
  const orchestrator = {
    events,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: { snapshot: () => ({ models: [], recommended: [], jobs: [], hardware: {}, intelDocker: {} }) },
    store: { list: () => [], get: () => undefined },
    configPath: "/tmp/agents.config.yaml",
    config: {
      backends: {
        "late-infer": { type: "lateinfer", baseUrl: "http://127.0.0.1:8010/v1", model: "google/gemma-3-4b-it" },
      },
      specialists: {},
    },
    allowlist: { list: () => [] },
    defaultCwd: () => "/",
    reloadConfig: () => undefined,
  };
  const chat = {
    list: () => [],
    create: () => ({}),
    get: () => ({}),
    delete: () => false,
    setPin: () => ({}),
    send: async () => ({}),
    runAction: async () => ({}),
    resolveApproval: async () => ({}),
  };
  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
  const { server, listen } = startGuiServer({
    orchestrator: orchestrator as never,
    chat: chat as never,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  const base = `http://127.0.0.1:${port}`;
  const auth = { authorization: `Bearer ${token}`, "content-type": "application/json" };
  try {
    const res = await fetch(`${base}/api/local-servers`, { headers: auth });
    assert.equal(res.status, 200);
    const body = (await res.json()) as {
      lateinfer?: {
        running?: boolean;
        reason?: string;
        downloading?: boolean;
        phase?: string;
      };
      lateInferBinary?: string | null;
      ollama?: unknown;
      llamacpp?: unknown[];
      gpu?: {
        reason?: string;
        cards?: unknown;
        displayCapPercent?: number;
        runtimeOk?: boolean;
        compileOk?: boolean;
        compileTarget?: string;
        live?: {
          headline?: string;
          servePhase?: string;
          gpuRunning?: boolean;
          weightsInHostRam?: boolean;
        };
      };
    };
    assert.ok(body.lateinfer);
    assert.equal(typeof body.lateinfer?.running, "boolean");
    assert.ok("downloading" in (body.lateinfer ?? {}));
    assert.ok(Array.isArray((body.lateinfer as { compiledModels?: unknown })?.compiledModels));
    for (const id of (body.lateinfer as { compiledModels?: string[] }).compiledModels ?? []) {
      assert.equal(typeof id, "string");
      assert.match(id, /^[^/]+\/[^/]+$/);
      assert.equal(id.startsWith("/"), false);
      assert.equal(id.includes("\\"), false);
    }
    assert.ok("lateInferBinary" in body);
    assert.ok(body.gpu);
    assert.equal(typeof body.gpu.reason, "string");
    assert.ok(Array.isArray(body.gpu.cards));
    assert.equal(body.gpu.displayCapPercent, 70);
    assert.match(body.gpu.reason ?? "", /your computer/);
    assert.equal(typeof body.gpu.compileOk, "boolean");
    assert.equal(typeof body.gpu.compileTarget, "string");
    assert.ok(body.gpu.live);
    assert.equal(typeof body.gpu.live.headline, "string");
    assert.equal(typeof body.gpu.live.servePhase, "string");
    assert.ok("gpuRunning" in body.gpu.live);
    assert.ok("weightsInHostRam" in body.gpu.live);
    assert.ok("starting" in (body.lateinfer ?? {}));
    assert.ok("servePhase" in (body.lateinfer ?? {}));

    const badKind = await fetch(`${base}/api/local-servers/start`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ kind: "vllm" }),
    });
    assert.equal(badKind.status, 400);
    const badBody = (await badKind.json()) as { error?: string };
    assert.match(badBody.error ?? "", /kind must be lateinfer/i);
    const badEngine = await fetch(`${base}/api/local-servers/start`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ kind: "not-an-engine" }),
    });
    assert.equal(badEngine.status, 400);
    const badEngineBody = (await badEngine.json()) as { error?: string };
    assert.match(badEngineBody.error ?? "", /kind must be lateinfer, ollama, or llamacpp/i);
    // ollama kind is intentional (GUI Start Ollama) — may 200 or 400 depending on binary/probe, never "lateinfer-only"
    const ollamaKind = await fetch(`${base}/api/local-servers/start`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ kind: "ollama" }),
    });
    assert.notEqual(ollamaKind.status, 404);
    const ollamaBody = (await ollamaKind.json()) as { error?: string };
    assert.equal(/kind must be lateinfer$/i.test(ollamaBody.error ?? ""), false);
  } finally {
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});

test("POST /api/local-servers/download kind lateinfer starts compile-only and GET exposes phase", async () => {
  const { resetLateInferCompileForTests } = await import("../src/local-servers/compile.js");
  const { EventEmitter } = await import("node:events");
  const { PassThrough } = await import("node:stream");
  const child = new EventEmitter() as import("node:child_process").ChildProcess;
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.pid = 77;
  child.kill = (() => true) as import("node:child_process").ChildProcess["kill"];
  resetLateInferCompileForTests({
    findBin: () => "/tmp/fake-late-infer",
    spawnFn: (_cmd, args, options) => {
      assert.deepEqual(args, ["--compile-only", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]);
      assert.equal(args.includes("--bind"), false);
      const env = (options?.env ?? {}) as NodeJS.ProcessEnv;
      assert.notEqual(env.LATE_INFER_ACCEL, "intel");
      assert.equal(env.CUDA_VISIBLE_DEVICES, "1");
      setImmediate(() => {
        child.stderr?.write("late-infer: downloading Hub snapshot…\n");
        child.stderr?.write("late-infer: progress phase=downloading bytes=2500 total=10000 percent=25 eta_sec=12\n");
      });
      return child;
    },
    gpu: {
      probes: {
        nvidiaSmi:
          "0, NVIDIA GeForce RTX 4090, 24564, 20000, 00000000:01:00.0, Enabled, Enabled\n1, NVIDIA GeForce RTX 4090, 24564, 24000, 00000000:02:00.0, Disabled, Disabled",
        lspci: null,
        drmCards: [],
      },
    },
  });
  const events = new EventEmitter();
  const orchestrator = {
    events,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: { snapshot: () => ({ models: [], recommended: [], jobs: [], hardware: {}, intelDocker: {} }) },
    store: { list: () => [], get: () => undefined },
    configPath: "/tmp/agents.config.yaml",
    config: {
      backends: {
        "late-infer": { type: "lateinfer", baseUrl: "http://127.0.0.1:8010/v1", model: "Qwen/Qwen2.5-0.5B-Instruct" },
      },
      specialists: {},
    },
    allowlist: { list: () => [] },
    defaultCwd: () => "/",
    reloadConfig: () => undefined,
  };
  const chat = {
    list: () => [],
    create: () => ({}),
    get: () => ({}),
    delete: () => false,
    setPin: () => ({}),
    send: async () => ({}),
    runAction: async () => ({}),
    resolveApproval: async () => ({}),
  };
  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
  const { server, listen } = startGuiServer({
    orchestrator: orchestrator as never,
    chat: chat as never,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  const base = `http://127.0.0.1:${port}`;
  const auth = { authorization: `Bearer ${token}`, "content-type": "application/json" };
  try {
    const posted = await fetch(`${base}/api/local-servers/download`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ kind: "lateinfer", model: "Qwen/Qwen2.5-0.5B-Instruct" }),
    });
    assert.equal(posted.status, 200);
    const job = (await posted.json()) as {
      kind?: string;
      model?: string;
      downloading?: boolean;
      phase?: string;
      percent?: number;
      bytes?: number;
      totalBytes?: number;
      etaSec?: number;
      cwd?: unknown;
      allowlist?: unknown;
      dest?: unknown;
    };
    assert.equal(job.kind, "lateinfer");
    assert.equal(job.model, "Qwen/Qwen2.5-0.5B-Instruct");
    assert.equal(job.downloading, true);
    assert.equal(job.phase, "downloading");
    assert.equal(job.percent, 0);
    assert.equal(job.bytes, 0);
    assert.equal("cwd" in job, false);
    assert.equal("allowlist" in job, false);
    assert.equal("dest" in job, false);

    const badKind = await fetch(`${base}/api/local-servers/download`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ kind: "ollama", model: "llama3.1" }),
    });
    assert.equal(badKind.status, 400);

    await delay(40);
    const snapStart = Date.now();
    let body: {
      lateinfer?: {
        downloading?: boolean;
        phase?: string;
        message?: string;
        percent?: number;
        bytes?: number;
        totalBytes?: number;
        etaSec?: number;
      };
    } = {};
    while (Date.now() - snapStart < 1000) {
      const snap = await fetch(`${base}/api/local-servers`, { headers: auth });
      assert.equal(snap.status, 200);
      body = (await snap.json()) as typeof body;
      if (body.lateinfer?.percent === 25 && body.lateinfer?.bytes === 2500) break;
      await delay(15);
    }
    assert.equal(body.lateinfer?.downloading, true);
    assert.equal(body.lateinfer?.phase === "downloading" || body.lateinfer?.phase === "compiling", true);
    assert.equal(body.lateinfer?.percent, 25);
    assert.equal(body.lateinfer?.bytes, 2500);
    assert.equal(body.lateinfer?.totalBytes, 10000);
    assert.equal(body.lateinfer?.etaSec, 12);
  } finally {
    child.emit("close", 1);
    resetLateInferCompileForTests();
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});

test("POST /api/local-servers/download passes gpuId through to compile env", async () => {
  const { resetLateInferCompileForTests } = await import("../src/local-servers/compile.js");
  const { EventEmitter } = await import("node:events");
  const { PassThrough } = await import("node:stream");
  const child = new EventEmitter() as import("node:child_process").ChildProcess;
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.pid = 78;
  child.kill = (() => true) as import("node:child_process").ChildProcess["kill"];
  let seenCuda: string | undefined;
  resetLateInferCompileForTests({
    findBin: () => "/tmp/fake-late-infer",
    spawnFn: (_cmd, args, options) => {
      assert.deepEqual(args, ["--compile-only", "--model", "Qwen/Qwen2.5-0.5B-Instruct"]);
      const env = (options?.env ?? {}) as NodeJS.ProcessEnv;
      seenCuda = env.CUDA_VISIBLE_DEVICES;
      assert.equal(env.CUDA_VISIBLE_DEVICES, "0");
      assert.equal(env.LATE_INFER_PCI, "0000:01:00.0");
      setImmediate(() => {
        child.stderr?.write("late-infer: downloading Hub snapshot…\n");
      });
      return child;
    },
    gpu: {
      probes: {
        nvidiaSmi:
          "0, NVIDIA GeForce RTX 4090, 24564, 20000, 00000000:01:00.0, Enabled, Enabled\n1, NVIDIA GeForce RTX 4090, 24564, 24000, 00000000:02:00.0, Disabled, Disabled",
        lspci: null,
        drmCards: [],
      },
    },
  });
  const events = new EventEmitter();
  const orchestrator = {
    events,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: { snapshot: () => ({ models: [], recommended: [], jobs: [], hardware: {}, intelDocker: {} }) },
    store: { list: () => [], get: () => undefined },
    configPath: "/tmp/agents.config.yaml",
    config: {
      backends: {
        "late-infer": { type: "lateinfer", baseUrl: "http://127.0.0.1:8010/v1", model: "Qwen/Qwen2.5-0.5B-Instruct" },
      },
      specialists: {},
    },
    allowlist: { list: () => [] },
    defaultCwd: () => "/",
    reloadConfig: () => undefined,
  };
  const chat = {
    list: () => [],
    create: () => ({}),
    get: () => ({}),
    delete: () => false,
    setPin: () => ({}),
    send: async () => ({}),
    runAction: async () => ({}),
    resolveApproval: async () => ({}),
  };
  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
  const { server, listen } = startGuiServer({
    orchestrator: orchestrator as never,
    chat: chat as never,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  const base = `http://127.0.0.1:${port}`;
  const auth = { authorization: `Bearer ${token}`, "content-type": "application/json" };
  try {
    const posted = await fetch(`${base}/api/local-servers/download`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({
        kind: "lateinfer",
        model: "Qwen/Qwen2.5-0.5B-Instruct",
        gpuId: "nvidia:0000:01:00.0",
      }),
    });
    assert.equal(posted.status, 200);
    const job = (await posted.json()) as { busId?: string; phase?: string };
    assert.equal(job.busId, "0000:01:00.0");
    assert.equal(seenCuda, "0");
  } finally {
    child.emit("close", 1);
    resetLateInferCompileForTests();
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});

test("POST /api/local-servers/delete removes compiled snapshot; missing id and serving refuse", async () => {
  const { resetLateInferDeleteForTests } = await import("../src/local-servers/spawn.js");
  const dir = mkdtempSync(join(tmpdir(), "orch-gui-del-"));
  const blob = join(dir, "google--gemma-4-E2B-it");
  mkdirSync(blob, { recursive: true });
  writeFileSync(
    join(blob, "late-compile.json"),
    JSON.stringify({ model_id: "google/gemma-4-E2B-it", status: "ok" }),
  );
  const prevCompiled = process.env.LATE_COMPILED_DIR;
  const prevHf = process.env.LATE_HF_HOME;
  process.env.LATE_COMPILED_DIR = dir;
  process.env.LATE_HF_HOME = join(dir, "hf");
  resetLateInferDeleteForTests(async () => ({ running: false, models: [] }));
  const events = new EventEmitter();
  const orchestrator = {
    events,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: { snapshot: () => ({ models: [], recommended: [], jobs: [], hardware: {}, intelDocker: {} }) },
    store: { list: () => [], get: () => undefined },
    configPath: "/tmp/agents.config.yaml",
    config: {
      backends: {
        "late-infer": { type: "lateinfer", baseUrl: "http://127.0.0.1:8010/v1", model: "Qwen/Qwen2.5-0.5B-Instruct" },
      },
      specialists: {},
    },
    allowlist: { list: () => [] },
    defaultCwd: () => "/",
    reloadConfig: () => undefined,
  };
  const chat = {
    list: () => [],
    create: () => ({}),
    get: () => ({}),
    delete: () => false,
    setPin: () => ({}),
    send: async () => ({}),
    runAction: async () => ({}),
    resolveApproval: async () => ({}),
  };
  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
  const { server, listen } = startGuiServer({
    orchestrator: orchestrator as never,
    chat: chat as never,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  const base = `http://127.0.0.1:${port}`;
  const auth = { authorization: `Bearer ${token}`, "content-type": "application/json" };
  try {
    const listed = await fetch(`${base}/api/local-servers`, { headers: auth });
    assert.equal(listed.status, 200);
    const before = (await listed.json()) as { lateinfer?: { compiledModels?: string[] } };
    assert.ok(before.lateinfer?.compiledModels?.includes("google/gemma-4-E2B-it"));

    const unauth = await fetch(`${base}/api/local-servers/delete`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ kind: "lateinfer", model: "google/gemma-4-E2B-it", confirm: true }),
    });
    assert.equal(unauth.status, 401);

    resetLateInferDeleteForTests(async () => ({
      running: true,
      models: ["google/gemma-4-E2B-it"],
    }));
    const busy = await fetch(`${base}/api/local-servers/delete`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ kind: "lateinfer", model: "google/gemma-4-E2B-it", confirm: true }),
    });
    assert.equal(busy.status, 409);
    const busyBody = (await busy.json()) as { error?: string };
    assert.match(busyBody.error ?? "", /Stop late-infer on your computer first/);
    assert.equal(existsSync(blob), true);

    resetLateInferDeleteForTests(async () => ({ running: false, models: [] }));
    const missing = await fetch(`${base}/api/local-servers/delete`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ kind: "lateinfer", model: "Qwen/Qwen2.5-0.5B-Instruct", confirm: true }),
    });
    assert.equal(missing.status, 404);
    const missingBody = (await missing.json()) as { error?: string };
    assert.match(missingBody.error ?? "", /not compiled on your computer/);

    const posted = await fetch(`${base}/api/local-servers/delete`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ kind: "lateinfer", model: "google--gemma-4-E2B-it", confirm: true }),
    });
    assert.equal(posted.status, 200);
    const result = (await posted.json()) as {
      ok?: boolean;
      model?: string;
      compiledModels?: string[];
      compiledRows?: unknown[];
      dest?: unknown;
      cwd?: unknown;
    };
    assert.equal(result.ok, true);
    assert.equal(result.model, "google/gemma-4-E2B-it");
    assert.deepEqual(result.compiledModels, []);
    assert.deepEqual(result.compiledRows, []);
    assert.equal("dest" in result, false);
    assert.equal("cwd" in result, false);
    assert.equal(existsSync(blob), false);

    const after = await fetch(`${base}/api/local-servers`, { headers: auth });
    assert.equal(after.status, 200);
    const snap = (await after.json()) as { lateinfer?: { compiledModels?: string[] } };
    assert.equal(snap.lateinfer?.compiledModels?.includes("google/gemma-4-E2B-it"), false);
  } finally {
    resetLateInferDeleteForTests();
    if (prevCompiled === undefined) delete process.env.LATE_COMPILED_DIR;
    else process.env.LATE_COMPILED_DIR = prevCompiled;
    if (prevHf === undefined) delete process.env.LATE_HF_HOME;
    else process.env.LATE_HF_HOME = prevHf;
    rmSync(dir, { recursive: true, force: true });
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});
