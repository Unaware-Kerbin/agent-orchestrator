export const MCP_LOOPBACK_HOST = "127.0.0.1" as const;

const LOOPBACK = new Set([MCP_LOOPBACK_HOST, "localhost", "::1"]);

/** Exact Streamable HTTP URL for Late Settings. Always 127.0.0.1 and the port this process bound. */
export function loopbackMcpUrl(port: number): string {
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("MCP port must be an integer 1–65535");
  }
  return `http://${MCP_LOOPBACK_HOST}:${port}/mcp`;
}

/** Stderr lines: the URL to paste, and that Late does not need this process. */
export function lateMcpCopyLines(mcpUrl: string): string[] {
  return [
    `mcp:    ${mcpUrl}  (Streamable HTTP, no GUI token — copy for Late Settings)`,
    `        Late works with MCP off. This URL is the port this process bound, not a hardcoded default.`,
  ];
}

/** HTTP MCP binds loopback only. Remote users reach this machine over VPN/NAC, then 127.0.0.1. */
export function bindMcpLoopbackOnly(requestedHost: string | undefined): typeof MCP_LOOPBACK_HOST {
  const host = (requestedHost ?? MCP_LOOPBACK_HOST).trim();
  if (LOOPBACK.has(host)) return MCP_LOOPBACK_HOST;
  throw new Error(
    `Refusing to bind MCP HTTP to "${host}". Only ${MCP_LOOPBACK_HOST} is allowed. Reach this host over VPN or NAC, then connect to http://${MCP_LOOPBACK_HOST}:<port>/mcp. Do not use 0.0.0.0, tunnels, or a public bind.`,
  );
}
