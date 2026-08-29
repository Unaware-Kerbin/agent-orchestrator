import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import http from "node:http";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { WriteAllowlist, canonicalizeDirectory } from "../src/allowlist.js";
import { applyParsedFiles } from "../src/chat/apply-patch.js";
import { ChatService } from "../src/chat/service.js";
import { startGuiServer } from "../src/gui/http.js";
import type { DispatchInput } from "../src/orchestrator.js";
import type { Orchestrator } from "../src/orchestrator.js";
import type { OrchestratedRun } from "../src/types.js";

const BUILD_MSG = "implement a commands.txt file in this workspace with show ip route";

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

function finished(input: DispatchInput, text: string): OrchestratedRun {
  return {
    id: `run-${input.backend ?? "x"}`,
    status: "finished",
    text,
    specialist: input.specialist,
    backend: input.backend ?? "vllm-local",
    prompt: input.task,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    history: [],
  };
}

function crashHarness(cwd: string, allow: WriteAllowlist) {
  const events = new EventEmitter();
  const dispatches: DispatchInput[] = [];
  const orchestrator = {
    events,
    defaultCwd: () => cwd,
    allowlist: allow,
    catalog: async () => ({
      backends: [
        {
          id: "vllm-local",
          type: "vllm",
          ready: true,
          writesLocalFiles: false,
          runtime: "local",
          model: "gemma",
        },
      ],
      specialists: [{ id: "vllm-chat", backend: "vllm-local" }],
      localRuntime: {
        vllm: {
          running: true,
          healthy: true,
          backendId: "vllm-local",
          modelId: "gemma",
          instances: [{ backendId: "vllm-local", healthy: true, running: true }],
        },
      },
    }),
    localModels: {
      snapshot: () => ({
        vllm: { running: true, modelId: "gemma" },
        models: [],
        recommended: [],
        jobs: [],
        hardware: {},
        intelDocker: {},
      }),
    },
    store: { list: () => [], get: () => undefined },
    configPath: join(cwd, "agents.config.yaml"),
    config: { backends: {}, specialists: {} },
    reloadConfig: () => undefined,
    dispatch: async (input: DispatchInput) => {
      dispatches.push(input);
      return finished(
        input,
        `Plan.\n\`\`\`orchestrator-files\n{"files":[{"path":"commands.txt","content":"show ip route\\n"}]}\n\`\`\``,
      );
    },
    followUp: async () => {
      throw new Error("followUp should not run");
    },
    runWorkflow: async () => ({ workflow: "", status: "ok", summary: "", runs: [] }),
  } as unknown as Orchestrator;
  const chat = new ChatService(orchestrator);
  return { orchestrator, chat, events, dispatches };
}

test("grant missing path, dead SSE, delete-during-send, and apply-patch into granted dir do not kill the GUI", async () => {
  const root = mkdtempSync(join(tmpdir(), "orch-crash-"));
  const granted = join(root, "Aruba_Test_Configs");
  mkdirSync(granted);
  const prevState = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  process.env.AGENT_ORCHESTRATOR_STATE_DIR = mkdtempSync(join(tmpdir(), "orch-crash-state-"));
  const allow = new WriteAllowlist(join(root, "allowlist.json"), [canonicalizeDirectory(root)]);
  const { orchestrator, chat } = crashHarness(root, allow);
  const crashes: unknown[] = [];
  const onRej = (err: unknown) => {
    crashes.push(err);
  };
  const onEx = (err: unknown) => {
    crashes.push(err);
  };
  process.on("unhandledRejection", onRej);
  process.on("uncaughtException", onEx);

  const token = "test-token-not-secret-16";
  const port = await freeLoopbackPort();
  const { server, listen } = startGuiServer({
    orchestrator,
    chat,
    token,
    port,
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(listen.port, listen.host, () => resolve());
  });
  const base = `http://127.0.0.1:${port}`;
  const auth = { authorization: `Bearer ${token}`, "content-type": "application/json" };
  const pidBefore = process.pid;

  try {
    const health0 = await fetch(`${base}/health`);
    assert.equal(health0.status, 200);
    const health0Body = (await health0.json()) as { ok: boolean; pid: number };
    assert.equal(health0Body.pid, pidBefore);

    const missing = await fetch(`${base}/api/allowlist`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ path: join(root, "does-not-exist-grant") }),
    });
    const missingText = await missing.text();
    assert.equal(missing.status, 400, missingText);
    const missingBody = JSON.parse(missingText) as { error: string };
    assert.match(missingBody.error, /exist|directory|Invalid/i);

    const sseReq = http.get(
      `${base}/api/events?token=${encodeURIComponent(token)}`,
      { headers: { host: `127.0.0.1:${port}` } },
      (res) => {
        res.resume();
      },
    );
    await new Promise((r) => setTimeout(r, 40));
    sseReq.destroy();

    orchestrator.events.emit("chat-heartbeat", { threadId: "gone", now: Date.now(), thinking: [] });
    orchestrator.events.emit("chats", chat.list());

    const grant = await fetch(`${base}/api/allowlist`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ path: granted }),
    });
    const grantText = await grant.text();
    assert.equal(grant.status, 200, grantText);
    const grantBody = JSON.parse(grantText) as { granted: string; allowedDirectories: string[] };
    assert.equal(grantBody.granted, canonicalizeDirectory(granted));
    assert.ok(grantBody.allowedDirectories.includes(grantBody.granted));

    const created = await fetch(`${base}/api/chats`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ pin: "single" }),
    });
    assert.equal(created.status, 200);
    const thread = (await created.json()) as { id: string };
    const workspace = await fetch(`${base}/api/chats/${encodeURIComponent(thread.id)}/workspace`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ path: granted }),
    });
    assert.equal(workspace.status, 200, await workspace.text());

    const sent = await fetch(`${base}/api/chats/${encodeURIComponent(thread.id)}/messages`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ message: BUILD_MSG, pin: "single", cwd: granted, wait: false }),
    });
    assert.equal(sent.status, 200, await sent.text());

    const del = await fetch(`${base}/api/chats/${encodeURIComponent(thread.id)}`, {
      method: "DELETE",
      headers: auth,
    });
    assert.equal(del.status, 200);
    await new Promise((r) => setTimeout(r, 120));

    const afterDeleteSse = await fetch(`${base}/api/chats`, { headers: { authorization: `Bearer ${token}` } });
    assert.equal(afterDeleteSse.status, 200);

    const created2 = await fetch(`${base}/api/chats`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ pin: "single" }),
    });
    const thread2 = (await created2.json()) as { id: string };
    await fetch(`${base}/api/chats/${encodeURIComponent(thread2.id)}/workspace`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ path: granted }),
    });
    const sent2 = await fetch(`${base}/api/chats/${encodeURIComponent(thread2.id)}/messages`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ message: BUILD_MSG, pin: "single", cwd: granted, wait: true }),
    });
    const sent2Text = await sent2.text();
    assert.equal(sent2.status, 200, sent2Text);
    const pendingThread = JSON.parse(sent2Text) as {
      id: string;
      pendingApproval?: { status?: string; applyPatch?: boolean };
    };
    assert.equal(pendingThread.pendingApproval?.status, "pending");
    assert.equal(pendingThread.pendingApproval?.applyPatch, true);

    const approved = await fetch(`${base}/api/chats/${encodeURIComponent(thread2.id)}/approval`, {
      method: "POST",
      headers: auth,
      body: JSON.stringify({ decision: "approve" }),
    });
    assert.equal(approved.status, 200, await approved.text());
    const written = join(granted, "commands.txt");
    assert.equal(existsSync(written), true);
    assert.equal(readFileSync(written, "utf8"), "show ip route\n");

    const health1 = await fetch(`${base}/health`);
    assert.equal(health1.status, 200);
    const health1Body = (await health1.json()) as { ok: boolean; pid: number };
    assert.equal(health1Body.pid, pidBefore);
    assert.equal(crashes.length, 0, String(crashes[0]));
  } finally {
    process.off("unhandledRejection", onRej);
    process.off("uncaughtException", onEx);
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
    if (prevState === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prevState;
    rmSync(root, { recursive: true, force: true });
  }
});

test("applyParsedFiles writes a text file into a newly granted folder", () => {
  const parent = mkdtempSync(join(tmpdir(), "orch-grant-write-"));
  const folder = join(parent, "Aruba_Test_Configs");
  mkdirSync(folder);
  const allow = new WriteAllowlist(join(parent, "allowlist.json"), [canonicalizeDirectory(parent)]);
  allow.add(folder);
  const { written } = applyParsedFiles({
    cwd: folder,
    files: [{ path: "show-ip-route.txt", content: "show ip route\n" }],
    allowlist: allow,
  });
  assert.equal(written.length, 1);
  assert.equal(readFileSync(join(folder, "show-ip-route.txt"), "utf8"), "show ip route\n");
});

test("apply-patch Approve does not throw if the chat is deleted mid-write", async () => {
  const root = mkdtempSync(join(tmpdir(), "orch-del-patch-"));
  const prevState = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  process.env.AGENT_ORCHESTRATOR_STATE_DIR = mkdtempSync(join(tmpdir(), "orch-del-patch-state-"));
  const realAllow = new WriteAllowlist(join(root, "allowlist.json"), [canonicalizeDirectory(root)]);
  let chat: ChatService | undefined;
  let doomed = "";
  const wrapped = Object.create(realAllow) as WriteAllowlist;
  wrapped.assertCwd = (path: string) => {
    const real = realAllow.assertCwd(path);
    if (doomed && chat) chat.delete(doomed);
    return real;
  };
  const { orchestrator } = crashHarness(root, wrapped);
  orchestrator.allowlist = wrapped;
  chat = new ChatService(orchestrator);
  const crashes: unknown[] = [];
  const onRej = (err: unknown) => {
    crashes.push(err);
  };
  process.on("unhandledRejection", onRej);
  try {
    const thread = await chat.send({ message: BUILD_MSG, pin: "single", cwd: root, wait: true });
    assert.equal(thread.pendingApproval?.applyPatch, true);
    doomed = thread.id;
    await assert.doesNotReject(() => chat!.resolveApproval({ threadId: thread.id, decision: "approve" }));
    assert.equal(chat.store.get(thread.id), undefined);
    assert.equal(crashes.length, 0, String(crashes[0]));
  } finally {
    process.off("unhandledRejection", onRej);
    if (prevState === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prevState;
  }
});
