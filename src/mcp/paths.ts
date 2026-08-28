/** Streamable HTTP MCP lives at /mcp. /MCP is accepted so clients that capitalize still hit MCP, not the GUI. */
export function isMcpPath(pathname: string): boolean {
  const path = normalizeMcpPath(pathname);
  return path === "/mcp" || path.startsWith("/mcp/");
}

export function isMcpLoginPath(pathname: string): boolean {
  return normalizeMcpPath(pathname) === "/mcp/login";
}

/** Loopback liveness only. Does not list tools. */
export function isMcpHealthPath(pathname: string): boolean {
  return normalizeMcpPath(pathname) === "/mcp/health";
}

/**
 * Cheap GET liveness so clients do not treat Streamable HTTP as down.
 * `/mcp/health` always. GET `/mcp` (and standalone GET `/`) without SSE Accept
 * is `{ok:true}` — POST initialize / tools/list still go to MCP.
 */
export function isMcpLivenessGet(
  method: string,
  pathname: string,
  acceptHeader: string | undefined,
  opts: { standalone?: boolean } = {},
): boolean {
  const m = (method || "GET").toUpperCase();
  if (m !== "GET" && m !== "HEAD") return false;
  if (isMcpHealthPath(pathname)) return true;
  const accept = (acceptHeader ?? "").toLowerCase();
  if (accept.includes("text/event-stream")) return false;
  const path = normalizeMcpPath(pathname);
  if (path === "/mcp") return true;
  return Boolean(opts.standalone) && path === "/";
}

export function normalizeMcpPath(pathname: string): string {
  const raw = pathname.split("?")[0] ?? pathname;
  let path = raw.replace(/\/+$/, "") || "/";
  if (path.length > 1) path = path.toLowerCase();
  else path = path.toLowerCase();
  return path;
}
