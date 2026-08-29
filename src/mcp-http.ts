#!/usr/bin/env node
import { createServer as createHttpServer, type IncomingMessage, type ServerResponse } from "node:http";
import { createOrchestrator } from "./bootstrap.js";
import { loadOrCreateGuiToken } from "./gui-auth.js";
import {
  createOrchestratorMcpHandler,
  isStandaloneMcpPath,
  pipeMcpHttpRequest,
  writeMcpHealth,
} from "./mcp-http-handler.js";
import { writeAdvertisedMcpUrl } from "./mcp/advertise.js";
import { bindMcpLoopbackOnly, lateMcpCopyLines, loopbackMcpUrl } from "./mcp/bind.js";
import { isMcpLivenessGet } from "./mcp/paths.js";
import { installMcpProcessGuards } from "./mcp/process-guard.js";
import {
  handleBackendLogoGet,
  handleTempAnalyzeApi,
  loopbackHostOk,
  loopbackOriginOk,
  readJsonBody,
  restAuthorized,
  TEMP_ANALYZE_PATH,
} from "./temp-analyze-http.js";

installMcpProcessGuards();

let HOST: "127.0.0.1";
try {
  HOST = bindMcpLoopbackOnly(process.env.AGENT_ORCHESTRATOR_MCP_HOST);
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}
const port = Number(process.env.AGENT_ORCHESTRATOR_MCP_PORT ?? "8790");
if (!Number.isInteger(port) || port < 1 || port > 65535) {
  console.error("AGENT_ORCHESTRATOR_MCP_PORT must be an integer 1–65535");
  process.exit(1);
}

const { orchestrator, chat } = createOrchestrator();
const handler = createOrchestratorMcpHandler(orchestrator, chat);
const secret = loadOrCreateGuiToken();

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  res.end(JSON.stringify(body));
}

async function tryHandleRest(req: IncomingMessage, res: ServerResponse, url: URL): Promise<boolean> {
  const path = url.pathname;
  if (path !== TEMP_ANALYZE_PATH && !/^\/api\/backends\/[^/]+\/logo$/.test(path)) {
    return false;
  }
  if (!restAuthorized(req, secret.token)) {
    sendJson(res, 401, { error: "unauthorized" });
    return true;
  }
  const method = req.method ?? "GET";
  if (handleBackendLogoGet({ method, pathname: path, res })) return true;
  let body: unknown = {};
  if (method !== "GET" && method !== "HEAD") {
    try {
      body = await readJsonBody(req);
    } catch {
      sendJson(res, 400, { error: "Invalid JSON" });
      return true;
    }
  }
  return handleTempAnalyzeApi({
    method,
    pathname: path,
    body,
    res,
    allowlist: orchestrator.tempAnalyze,
  });
}

async function handleMcpHttpRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const host = req.headers.host ?? `${HOST}:${port}`;
  if (!loopbackHostOk(req.headers.host, port)) {
    sendJson(res, 403, { error: "Invalid Host header" });
    return;
  }
  const origin = typeof req.headers.origin === "string" ? req.headers.origin : undefined;
  if (!loopbackOriginOk(origin)) {
    sendJson(res, 403, { error: "Origin not allowed" });
    return;
  }
  const url = new URL(req.url ?? "/", `http://${host}`);
  const method = req.method ?? "GET";
  const accept = typeof req.headers.accept === "string" ? req.headers.accept : undefined;
  if (isMcpLivenessGet(method, url.pathname, accept, { standalone: true })) {
    writeMcpHealth(res);
    return;
  }
  if (await tryHandleRest(req, res, url)) return;
  if (!isStandaloneMcpPath(url.pathname)) {
    res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    res.end("Not found. MCP Streamable HTTP is /mcp");
    return;
  }
  await pipeMcpHttpRequest(handler, req, res, url);
}

const server = createHttpServer((req, res) => {
  void handleMcpHttpRequest(req, res).catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    if (!res.headersSent) {
      sendJson(res, 500, { error: "internal error" });
      return;
    }
    res.end();
  });
});

server.on("error", (error) => {
  const err = error as NodeJS.ErrnoException;
  if (err.code === "EADDRINUSE") {
    console.error(`MCP HTTP already listening on ${HOST}:${port}`);
    process.exit(1);
  }
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});

const shutdown = (): void => {
  void handler.close();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1500).unref();
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

server.listen(port, HOST, () => {
  const mcpUrl = loopbackMcpUrl(port);
  writeAdvertisedMcpUrl(mcpUrl, { kind: "http" });
  console.error(`agent-orchestrator MCP Streamable HTTP`);
  console.error(`  bind:  ${HOST}:${port}`);
  console.error(`  url:   ${mcpUrl}`);
  console.error(`  health: GET ${mcpUrl}/health  → {"ok":true} (loopback, no tools)`);
  for (const line of lateMcpCopyLines(mcpUrl)) console.error(`  ${line}`);
  console.error(`  notes: The GUI also serves /mcp on AGENT_ORCHESTRATOR_GUI_PORT (same-process as the web UI).`);
  console.error(`         Temp pcap allowlist: POST ${TEMP_ANALYZE_PATH} with Bearer (GUI token).`);
  console.error(`         Bound to 127.0.0.1 only. You start this process. Late will not start it.`);
});
