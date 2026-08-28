import { createMcpHandler, type McpHttpHandler } from "@modelcontextprotocol/server";
import type { IncomingMessage, ServerResponse } from "node:http";
import { Readable } from "node:stream";
import type { ChatService } from "./chat/service.js";
import type { Orchestrator } from "./orchestrator.js";
import { createServer } from "./server.js";

/** Loopback liveness. Never includes tools — those stay on POST tools/list. */
export function writeMcpHealth(res: ServerResponse): void {
  res.writeHead(200, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
  });
  res.end(JSON.stringify({ ok: true }));
}

/** Streamable HTTP MCP on `/mcp` (any case). Not the GUI page at `/`. */
export function isMcpHttpPath(pathname: string): boolean {
  let path = pathname;
  try {
    path = decodeURIComponent(pathname);
  } catch {
    path = pathname;
  }
  const lower = path.toLowerCase();
  return lower === "/mcp" || lower.startsWith("/mcp/");
}

/** Standalone `mcp:http` process: `/` is MCP too because that process has no GUI. */
export function isStandaloneMcpPath(pathname: string): boolean {
  let path = pathname;
  try {
    path = decodeURIComponent(pathname);
  } catch {
    path = pathname;
  }
  const stripped = path.replace(/\/+$/, "") || "/";
  return stripped === "/" || isMcpHttpPath(path);
}

export function canonicalMcpUrl(url: URL): URL {
  const out = new URL(url.href);
  const lower = out.pathname.toLowerCase();
  if (lower === "/mcp" || lower.startsWith("/mcp/")) {
    out.pathname = `/mcp${out.pathname.slice(4)}`;
  } else if ((out.pathname.replace(/\/+$/, "") || "/") === "/") {
    out.pathname = "/mcp";
  }
  return out;
}

export function createOrchestratorMcpHandler(orchestrator: Orchestrator, chat: ChatService): McpHttpHandler {
  return createMcpHandler(() => createServer(orchestrator, chat));
}

export async function pipeMcpHttpRequest(
  handler: McpHttpHandler,
  req: IncomingMessage,
  res: ServerResponse,
  url: URL,
): Promise<void> {
  const headers = new Headers();
  for (const [k, v] of Object.entries(req.headers)) {
    if (v === undefined) continue;
    headers.set(k, Array.isArray(v) ? v.join(", ") : v);
  }
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(chunk as Buffer);
  }
  const body = Buffer.concat(chunks);
  const method = req.method ?? "GET";
  const init: RequestInit = { method, headers };
  if (method !== "GET" && method !== "HEAD" && body.length) {
    init.body = new Uint8Array(body);
  }
  const mcpUrl = canonicalMcpUrl(url);
  try {
    const response = await handler.fetch(new Request(mcpUrl, init));
    const out: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      out[key] = value;
    });
    res.writeHead(response.status, out);
    if (!response.body) {
      res.end();
      return;
    }
    Readable.fromWeb(response.body as import("node:stream/web").ReadableStream).pipe(res);
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    if (!res.headersSent) {
      res.writeHead(500, { "content-type": "text/plain; charset=utf-8" });
    }
    res.end("MCP HTTP error");
  }
}
