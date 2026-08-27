import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { createServer } from "node:net";
import { test } from "node:test";
import { setTimeout as delay } from "node:timers/promises";
import { originAllowed, hostAllowed, startGuiServer } from "../src/gui/http.js";

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
