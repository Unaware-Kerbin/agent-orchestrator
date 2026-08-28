import type { IncomingMessage, ServerResponse } from "node:http";
import { extractBearerToken, tokensEqual } from "./gui-auth.js";
import { readLogo } from "./identity.js";
import { redactSecretText } from "./redact.js";
import type { TempAnalyzeAllowlist } from "./temp-allowlist.js";

export const TEMP_ANALYZE_PATH = "/api/temp-analyze";

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function sendJson(res: ServerResponse, status: number, body: unknown, extra: Record<string, string> = {}): void {
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    ...extra,
  });
  res.end(JSON.stringify(body));
}

export function loopbackHostOk(hostHeader: string | undefined, port: number): boolean {
  if (!hostHeader) return false;
  const host = hostHeader.trim().toLowerCase();
  return (
    host === `127.0.0.1:${port}` ||
    host === `localhost:${port}` ||
    host === "127.0.0.1" ||
    host === "localhost" ||
    host === `[::1]:${port}` ||
    host === "::1"
  );
}

/** Missing Origin is OK (non-browser MCP clients). A present Origin must be loopback. */
export function loopbackOriginOk(origin: string | undefined): boolean {
  if (!origin) return true;
  if (origin.trim().toLowerCase() === "null") return false;
  try {
    const url = new URL(origin);
    if (url.protocol !== "http:" && url.protocol !== "https:") return false;
    const host = url.hostname.replace(/^\[|\]$/g, "").toLowerCase();
    return host === "127.0.0.1" || host === "localhost" || host === "::1" || host === "localhost.localdomain";
  } catch {
    return false;
  }
}

export function restAuthorized(req: IncomingMessage, token: string): boolean {
  const provided =
    extractBearerToken(
      typeof req.headers.authorization === "string" ? req.headers.authorization : undefined,
      null,
    ) ??
    (typeof req.headers["x-orchestrator-token"] === "string" ? req.headers["x-orchestrator-token"].trim() : "");
  return Boolean(token) && tokensEqual(provided, token);
}

export async function readJsonBody(req: IncomingMessage, max = 64_000): Promise<unknown> {
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
  const raw = Buffer.concat(chunks).toString("utf8");
  if (!raw.trim()) return {};
  return JSON.parse(raw);
}

/**
 * POST/DELETE/GET `/api/temp-analyze` and GET `/api/backends/:id/logo`.
 * Caller must already have checked loopback Host and Bearer.
 */
export function handleTempAnalyzeApi(opts: {
  method: string;
  pathname: string;
  body: unknown;
  res: ServerResponse;
  allowlist: TempAnalyzeAllowlist;
}): boolean {
  const { method, pathname, body, res, allowlist } = opts;
  if (pathname !== TEMP_ANALYZE_PATH) return false;

  try {
    if (method === "GET") {
      sendJson(res, 200, { grants: allowlist.list() });
      return true;
    }
    if (method === "POST") {
      const path = isRecord(body) && typeof body.path === "string" ? body.path : "";
      const ttlMs = isRecord(body) && typeof body.ttlMs === "number" ? body.ttlMs : undefined;
      const grant = allowlist.add(path, ttlMs);
      sendJson(res, 200, { ok: true, path: grant.path, expiresAt: grant.expiresAt, write: false });
      return true;
    }
    if (method === "DELETE") {
      const path = isRecord(body) && typeof body.path === "string" ? body.path : "";
      if (!path) {
        sendJson(res, 400, { error: "path required" });
        return true;
      }
      const removed = allowlist.remove(path);
      sendJson(res, 200, { ok: true, removed, path });
      return true;
    }
    sendJson(res, 405, { error: "GET, POST, or DELETE" });
    return true;
  } catch (error) {
    sendJson(res, 400, { error: redactSecretText(error instanceof Error ? error.message : String(error)) });
    return true;
  }
}

export function handleBackendLogoGet(opts: { method: string; pathname: string; res: ServerResponse }): boolean {
  const match = /^\/api\/backends\/([^/]+)\/logo$/.exec(opts.pathname);
  if (!match || opts.method !== "GET") return false;
  const id = decodeURIComponent(match[1] ?? "").trim();
  const logo = readLogo(id);
  if (!logo) {
    sendJson(opts.res, 404, { error: "No logo" });
    return true;
  }
  opts.res.writeHead(200, {
    "content-type": logo.mime,
    "cache-control": "no-store",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "content-length": String(logo.bytes.length),
  });
  opts.res.end(logo.bytes);
  return true;
}
