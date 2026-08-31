import type { IncomingMessage, ServerResponse } from "node:http";
import type { McpHttpHandler } from "@modelcontextprotocol/server";
import { pipeMcpHttpRequest, writeMcpHealth } from "../mcp-http-handler.js";
import type { McpAuth } from "./auth/index.js";
import { isMcpLivenessGet, isMcpLoginPath, isMcpPath } from "./paths.js";

function sendJson(
  res: ServerResponse,
  status: number,
  body: unknown,
  extra: Record<string, string> = {},
): void {
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    ...extra,
  });
  res.end(JSON.stringify(body));
}

async function readRawBody(req: IncomingMessage, max = 1_000_000): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buf = chunk as Buffer;
    size += buf.length;
    if (size > max) {
      req.destroy();
      throw new Error("Request body too large");
    }
    chunks.push(buf);
  }
  return Buffer.concat(chunks);
}

/**
 * Handle /mcp and /MCP (and /mcp/login). Returns true when the request was consumed.
 * Host/Origin must already have been checked by the GUI server.
 * Streamable HTTP does not require the GUI token — Late never sends it.
 * On a private-IP bind, /mcp is a trusted-LAN model (firewall to the laptop); still not 0.0.0.0.
 * Optional POST /mcp/login stays available for LDAP/RADIUS session tokens.
 */
export async function tryHandleMcpRequest(opts: {
  req: IncomingMessage;
  res: ServerResponse;
  url: URL;
  auth: McpAuth;
  handler: McpHttpHandler;
}): Promise<boolean> {
  const { req, res, url, auth, handler } = opts;
  if (!isMcpPath(url.pathname)) return false;
  const method = req.method ?? "GET";
  const accept = typeof req.headers.accept === "string" ? req.headers.accept : undefined;
  if (isMcpLivenessGet(method, url.pathname, accept)) {
    writeMcpHealth(res);
    return true;
  }

  if (isMcpLoginPath(url.pathname) && method === "POST") {
    let body: unknown;
    try {
      const raw = await readRawBody(req);
      body = raw.length ? JSON.parse(raw.toString("utf8")) : {};
    } catch {
      sendJson(res, 400, { error: "Invalid JSON" });
      return true;
    }
    const rec = body && typeof body === "object" ? (body as Record<string, unknown>) : {};
    const username = typeof rec.username === "string" ? rec.username : "";
    const password = typeof rec.password === "string" ? rec.password : "";
    const result = await auth.login({ username, password });
    if (!result.ok) {
      sendJson(res, result.status, { error: result.error });
      return true;
    }
    sendJson(res, 200, {
      token: result.token,
      expiresAt: new Date(result.expiresAt).toISOString(),
      tokenType: "Bearer",
      subject: result.principal.subject,
    });
    return true;
  }

  if (isMcpLoginPath(url.pathname)) {
    sendJson(res, 405, { error: "POST only" });
    return true;
  }

  await pipeMcpHttpRequest(handler, req, res, url);
  return true;
}
