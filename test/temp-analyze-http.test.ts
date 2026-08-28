import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtempSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { startGuiServer } from "../src/gui/http.js";
import { TEMP_ANALYZE_PATH } from "../src/temp-analyze-http.js";
import { TempAnalyzeAllowlist } from "../src/temp-allowlist.js";

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

function tinyPcap(dir: string): string {
  const hdr = Buffer.alloc(24);
  hdr.writeUInt32LE(0xa1b2c3d4, 0);
  hdr.writeUInt16LE(2, 4);
  hdr.writeUInt16LE(4, 6);
  hdr.writeUInt32LE(65535, 16);
  hdr.writeUInt32LE(1, 20);
  const pktHdr = Buffer.alloc(16);
  pktHdr.writeUInt32LE(14, 8);
  pktHdr.writeUInt32LE(14, 12);
  const file = join(dir, "sample.pcap");
  writeFileSync(file, Buffer.concat([hdr, pktHdr, Buffer.alloc(14)]));
  return file;
}

function mockOrch(allow: TempAnalyzeAllowlist) {
  const events = new EventEmitter();
  return {
    events,
    tempAnalyze: allow,
    catalog: async () => ({ backends: [], specialists: [], workflows: [] }),
    localModels: {
      snapshot: () => ({ vllm: {}, models: [], recommended: [], jobs: [], hardware: {}, intelDocker: {} }),
      vllmStatus: () => ({}),
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
}

test("temp-analyze webhook requires Bearer and grants then removes one pcap", async () => {
  const allow = new TempAnalyzeAllowlist();
  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
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
    orchestrator: mockOrch(allow) as never,
    chat: chat as never,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  const base = `http://127.0.0.1:${port}${TEMP_ANALYZE_PATH}`;
  const dir = mkdtempSync(join(tmpdir(), "orch-pcap-http-"));
  const file = tinyPcap(dir);
  try {
    const noAuth = await fetch(base, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ path: file }),
    });
    assert.equal(noAuth.status, 401);

    const wrong = await fetch(base, {
      method: "POST",
      headers: { authorization: "Bearer nope-nope-nope-no", "content-type": "application/json" },
      body: JSON.stringify({ path: file }),
    });
    assert.equal(wrong.status, 401);

    const granted = await fetch(base, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ path: file }),
    });
    const grantedText = await granted.text();
    assert.equal(granted.status, 200, grantedText);
    const body = JSON.parse(grantedText) as { ok: boolean; path: string; write?: boolean };
    assert.equal(body.ok, true);
    assert.equal(body.write, false);
    assert.equal(allow.has(file), true);

    const escaped = await fetch(base, {
      method: "POST",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ path: `${dir}/../secret.pcap` }),
    });
    assert.equal(escaped.status, 400);

    const del = await fetch(base, {
      method: "DELETE",
      headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
      body: JSON.stringify({ path: file }),
    });
    assert.equal(del.status, 200);
    assert.equal(allow.has(file), false);
  } finally {
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});
