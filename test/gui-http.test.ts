import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtempSync, rmSync } from "node:fs";
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
