import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { createServer as createHttpServer, request as httpRequest } from "node:http";
import { createServer } from "node:net";
import { test } from "node:test";
import {
  canonicalMcpUrl,
  createOrchestratorMcpHandler,
  isMcpHttpPath,
  isStandaloneMcpPath,
  pipeMcpHttpRequest,
  writeMcpHealth,
} from "../src/mcp-http-handler.js";
import { isMcpLivenessGet } from "../src/mcp/paths.js";
import { loopbackHostOk, loopbackOriginOk } from "../src/temp-analyze-http.js";

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

function mockOrch() {
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
  return { orchestrator, chat };
}

const PROTOCOL = "2025-03-26";

function parseTools(ctype: string, text: string): { name?: string }[] {
  if (ctype.includes("text/event-stream")) {
    const data = text
      .split("\n")
      .filter((l) => l.startsWith("data:"))
      .map((l) => l.slice(5).trim())
      .filter(Boolean);
    for (const row of data) {
      try {
        const msg = JSON.parse(row) as { result?: { tools?: { name?: string }[] } };
        if (msg.result?.tools) return msg.result.tools;
      } catch {
        /* skip */
      }
    }
    return [];
  }
  const body = JSON.parse(text) as { result?: { tools?: { name?: string }[] } };
  return body.result?.tools ?? [];
}

function getLoopbackWithHost(
  port: number,
  path: string,
  hostHeader: string,
): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const req = httpRequest(
      { host: "127.0.0.1", port, path, headers: { host: hostHeader } },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c) => chunks.push(c as Buffer));
        res.on("end", () =>
          resolve({ status: res.statusCode ?? 0, body: Buffer.concat(chunks).toString("utf8") }),
        );
      },
    );
    req.on("error", reject);
    req.end();
  });
}

async function mcpRpc(url: string, id: number, method: string, params: unknown = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      accept: "application/json, text/event-stream",
      "mcp-protocol-version": PROTOCOL,
    },
    body: JSON.stringify({ jsonrpc: "2.0", id, method, params }),
  });
  const text = await res.text();
  return { res, text, ctype: res.headers.get("content-type") ?? "" };
}

test("MCP paths: /mcp and /MCP; GUI root is not MCP", () => {
  assert.equal(isMcpHttpPath("/mcp"), true);
  assert.equal(isMcpHttpPath("/MCP"), true);
  assert.equal(isMcpHttpPath("/"), false);
  assert.equal(isStandaloneMcpPath("/"), true);
  assert.equal(canonicalMcpUrl(new URL("http://127.0.0.1:8790/MCP")).pathname, "/mcp");
});

test("standalone Streamable HTTP: initialize + tools/list on /mcp and /MCP without a token", async () => {
  const { orchestrator, chat } = mockOrch();
  const handler = createOrchestratorMcpHandler(orchestrator as never, chat as never);
  const port = await freeLoopbackPort();
  const server = createHttpServer(async (req, res) => {
    const url = new URL(req.url ?? "/", `http://127.0.0.1:${port}`);
    if (!isStandaloneMcpPath(url.pathname)) {
      res.writeHead(404);
      res.end();
      return;
    }
    await pipeMcpHttpRequest(handler, req, res, url);
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => resolve());
  });
  try {
    for (const path of ["/mcp", "/MCP"]) {
      const mcp = `http://127.0.0.1:${port}${path}`;
      const init = await mcpRpc(mcp, 1, "initialize", {
        protocolVersion: PROTOCOL,
        capabilities: { tools: {} },
        clientInfo: { name: "test", version: "0.0.0" },
      });
      assert.equal(init.res.ok, true, `${path} initialize: ${init.text}`);
      await fetch(mcp, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          accept: "application/json, text/event-stream",
          "mcp-protocol-version": PROTOCOL,
        },
        body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
      });
      const listed = await mcpRpc(mcp, 2, "tools/list");
      assert.equal(listed.res.status, 200, `${path} tools/list: ${listed.text}`);
      const tools = parseTools(listed.ctype, listed.text);
      const names = tools.map((t) => t.name);
      assert.ok(tools.some((t) => t.name === "list_agents"), JSON.stringify(names));
      assert.ok(names.includes("chat_send"), JSON.stringify(names));
      assert.ok(names.includes("list_allowed_dirs"), JSON.stringify(names));
      const called = await mcpRpc(mcp, 3, "tools/call", { name: "list_agents", arguments: {} });
      assert.equal(called.res.status, 200, `${path} tools/call list_agents: ${called.text}`);
      assert.match(called.text, /backends|specialists/);
    }
  } finally {
    await handler.close();
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});

test("loopback Host and Origin accept localhost and 127.0.0.1", () => {
  assert.equal(loopbackHostOk("127.0.0.1:8790", 8790), true);
  assert.equal(loopbackHostOk("localhost:8790", 8790), true);
  assert.equal(loopbackHostOk("[::1]:8790", 8790), true);
  assert.equal(loopbackHostOk("0.0.0.0:8790", 8790), false);
  assert.equal(loopbackHostOk("example.com:8790", 8790), false);
  assert.equal(loopbackOriginOk(undefined), true);
  assert.equal(loopbackOriginOk("http://127.0.0.1:8790"), true);
  assert.equal(loopbackOriginOk("http://localhost:5173"), true);
  assert.equal(loopbackOriginOk("http://evil.example:8790"), false);
  assert.equal(loopbackOriginOk("null"), false);
});

test("GET /mcp/health is {ok:true} without Bearer; GET /mcp is not a 404; tools/list still works", async () => {
  const { orchestrator, chat } = mockOrch();
  const handler = createOrchestratorMcpHandler(orchestrator as never, chat as never);
  const port = await freeLoopbackPort();
  const server = createHttpServer(async (req, res) => {
    if (!loopbackHostOk(req.headers.host, port)) {
      res.writeHead(403, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "Invalid Host header" }));
      return;
    }
    const origin = typeof req.headers.origin === "string" ? req.headers.origin : undefined;
    if (!loopbackOriginOk(origin)) {
      res.writeHead(403, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: "Origin not allowed" }));
      return;
    }
    const url = new URL(req.url ?? "/", `http://127.0.0.1:${port}`);
    const method = req.method ?? "GET";
    const accept = typeof req.headers.accept === "string" ? req.headers.accept : undefined;
    if (isMcpLivenessGet(method, url.pathname, accept, { standalone: true })) {
      writeMcpHealth(res);
      return;
    }
    if (!isStandaloneMcpPath(url.pathname)) {
      res.writeHead(404, { "content-type": "text/plain" });
      res.end("Not found. MCP Streamable HTTP is /mcp");
      return;
    }
    await pipeMcpHttpRequest(handler, req, res, url);
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => resolve());
  });
  try {
    const health = await fetch(`http://127.0.0.1:${port}/mcp/health`);
    assert.equal(health.status, 200);
    const healthBody = (await health.json()) as { ok?: boolean };
    assert.deepEqual(healthBody, { ok: true });
    assert.doesNotMatch(JSON.stringify(healthBody), /list_agents|chat_send|"tools"/);

    const getMcp = await fetch(`http://127.0.0.1:${port}/mcp`);
    assert.equal(getMcp.status, 200, "GET /mcp without SSE must not 404/405 as down");
    assert.deepEqual(await getMcp.json(), { ok: true });

    const withBearer = await fetch(`http://127.0.0.1:${port}/mcp/health`, {
      headers: { authorization: "Bearer unused-token" },
    });
    assert.equal(withBearer.status, 200);
    assert.deepEqual(await withBearer.json(), { ok: true });

    const localHost = await getLoopbackWithHost(port, "/mcp/health", `localhost:${port}`);
    assert.equal(localHost.status, 200);
    assert.deepEqual(JSON.parse(localHost.body), { ok: true });

    const originOk = await fetch(`http://127.0.0.1:${port}/mcp/health`, {
      headers: { origin: `http://localhost:${port}` },
    });
    assert.equal(originOk.status, 200);

    const originBad = await fetch(`http://127.0.0.1:${port}/mcp/health`, {
      headers: { origin: "http://evil.example" },
    });
    assert.equal(originBad.status, 403);

    const mcp = `http://127.0.0.1:${port}/mcp`;
    const init = await mcpRpc(mcp, 1, "initialize", {
      protocolVersion: PROTOCOL,
      capabilities: { tools: {} },
      clientInfo: { name: "test", version: "0.0.0" },
    });
    assert.equal(init.res.ok, true, init.text);
    await fetch(mcp, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
        "mcp-protocol-version": PROTOCOL,
      },
      body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
    });
    const listed = await mcpRpc(mcp, 2, "tools/list");
    assert.equal(listed.res.status, 200, listed.text);
    const tools = parseTools(listed.ctype, listed.text);
    assert.ok(tools.some((t) => t.name === "list_agents"), JSON.stringify(tools));

    const listedAuth = await fetch(mcp, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
        "mcp-protocol-version": PROTOCOL,
        authorization: "Bearer unused-token",
      },
      body: JSON.stringify({ jsonrpc: "2.0", id: 3, method: "tools/list", params: {} }),
    });
    assert.equal(listedAuth.status, 200);
  } finally {
    await handler.close();
    await new Promise<void>((resolve, reject) => server.close((err) => (err ? reject(err) : resolve())));
  }
});
