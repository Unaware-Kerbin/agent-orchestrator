const LOOPBACK = new Set(["127.0.0.1", "localhost", "::1"]);

/** HTTP MCP binds loopback only. Remote users reach this machine over VPN/NAC, then 127.0.0.1. */
export function bindMcpLoopbackOnly(requestedHost: string | undefined): "127.0.0.1" {
  const host = (requestedHost ?? "127.0.0.1").trim();
  if (LOOPBACK.has(host)) return "127.0.0.1";
  throw new Error(
    `Refusing to bind MCP HTTP to "${host}". Only 127.0.0.1 is allowed. Reach this host over VPN or NAC, then connect to http://127.0.0.1:<port>/mcp. Do not use 0.0.0.0, tunnels, or a public bind.`,
  );
}
