import { readFileSync } from "node:fs";
import { isIP } from "node:net";
import { networkInterfaces, type NetworkInterfaceInfo } from "node:os";

export const MCP_LOOPBACK_HOST = "127.0.0.1" as const;
export const MCP_AUTO_LISTEN_HOST = "auto" as const;

const LOOPBACK_NAMES = new Set([MCP_LOOPBACK_HOST, "localhost", "::1", "localhost.localdomain"]);

function stripHost(raw: string): string {
  return raw.trim().toLowerCase().replace(/^\[|\]$/g, "").split("%")[0] ?? "";
}

function isLoopbackName(host: string): boolean {
  const h = stripHost(host);
  if (LOOPBACK_NAMES.has(h)) return true;
  if (isIP(h) === 4) {
    const oct = h.split(".").map(Number);
    return oct[0] === 127;
  }
  if (isIP(h) === 6) {
    const parts = expandIpv6(h);
    return Boolean(parts && parts.every((p, i) => (i === 7 ? p === 1 : p === 0)));
  }
  return false;
}

function expandIpv6(addr: string): number[] | undefined {
  const stripped = stripHost(addr);
  if (isIP(stripped) !== 6) return undefined;
  if (stripped.includes(".")) return undefined;
  const [head, tail] = stripped.split("::");
  const headParts = head ? head.split(":") : [];
  const tailParts = tail !== undefined && tail !== "" ? tail.split(":") : [];
  if (stripped.includes("::")) {
    const missing = 8 - headParts.length - tailParts.length;
    if (missing < 0) return undefined;
    const parts = [...headParts, ...Array(missing).fill("0"), ...tailParts];
    if (parts.length !== 8) return undefined;
    return parts.map((p) => parseInt(p, 16));
  }
  const parts = stripped.split(":");
  if (parts.length !== 8) return undefined;
  return parts.map((p) => parseInt(p, 16));
}

function isUnspecified(host: string): boolean {
  const h = stripHost(host);
  return h === "0.0.0.0" || h === "::" || h === "*" || h === "255.255.255.255";
}

function isRfc1918(host: string): boolean {
  if (isIP(host) !== 4) return false;
  const oct = host.split(".").map(Number);
  const a = oct[0] ?? 0;
  const b = oct[1] ?? 0;
  if (a === 10) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;
  return false;
}

function isLinkLocalV4(host: string): boolean {
  if (isIP(host) !== 4) return false;
  const oct = host.split(".").map(Number);
  return (oct[0] ?? 0) === 169 && (oct[1] ?? 0) === 254;
}

function isUlaV6(host: string): boolean {
  const parts = expandIpv6(host);
  if (!parts) return false;
  const first = parts[0] ?? 0;
  return (first & 0xff00) === 0xfd00;
}

function isLinkLocalV6(host: string): boolean {
  const parts = expandIpv6(host);
  if (!parts) return false;
  const first = parts[0] ?? 0;
  return (first & 0xffc0) === 0xfe80;
}

function refuseMessage(requested: string): string {
  return (
    `Refusing to bind to "${requested}". Use loopback (${MCP_LOOPBACK_HOST}), ${MCP_AUTO_LISTEN_HOST} (this computer's primary RFC1918 IPv4), ` +
    `or one private address (RFC1918 10/8, 172.16/12, 192.168/16, or IPv6 ULA fd00::/8). Do not use 0.0.0.0, ::, or a public address.`
  );
}

const SKIP_IFACE = /^(lo|docker|br-|virbr|veth|cni|flannel|kube|tailscale|zt|wg)/i;

function isIpv4Family(family: string | number): boolean {
  return family === "IPv4" || family === 4;
}

function defaultRouteInterface(): string | undefined {
  try {
    const text = readFileSync("/proc/net/route", "utf8");
    for (const line of text.split("\n").slice(1)) {
      const cols = line.trim().split(/\s+/);
      if (cols[1] === "00000000" && cols[0]) return cols[0];
    }
  } catch {
    /* Windows / macOS have no /proc/net/route */
  }
  return undefined;
}

export type PrivateIfaceMap = NodeJS.Dict<NetworkInterfaceInfo[]>;

/**
 * This computer's primary RFC1918 IPv4 (default-route iface first; skip docker/virbr).
 * Never 0.0.0.0 or a public address. Undefined if none.
 */
export function primaryPrivateIpv4(
  ifaces: PrivateIfaceMap = networkInterfaces(),
  defaultIface: string | undefined = defaultRouteInterface(),
): string | undefined {
  const candidates: { address: string; score: number }[] = [];
  for (const [name, addrs] of Object.entries(ifaces)) {
    if (!addrs) continue;
    for (const a of addrs) {
      if (a.internal || !isIpv4Family(a.family)) continue;
      const address = stripHost(a.address);
      if (!isRfc1918(address)) continue;
      let score = 10;
      if (defaultIface && name === defaultIface) score += 100;
      if (SKIP_IFACE.test(name)) score -= 50;
      const oct = address.split(".").map(Number);
      if ((oct[0] ?? 0) === 172 && (oct[1] ?? 0) >= 17 && (oct[1] ?? 0) <= 19) score -= 20;
      if (address.startsWith("192.168.122.")) score -= 20;
      candidates.push({ address, score });
    }
  }
  candidates.sort((a, b) => b.score - a.score);
  return candidates[0]?.address;
}

export function isAutoListenHost(raw: string | undefined): boolean {
  return stripHost(raw ?? "") === MCP_AUTO_LISTEN_HOST;
}

function resolveAutoListenHost(): string {
  const lan = primaryPrivateIpv4();
  if (!lan) {
    throw new Error(
      `listen_host=${MCP_AUTO_LISTEN_HOST} needs one RFC1918 IPv4 on this computer. Set an explicit private address or ${MCP_LOOPBACK_HOST}.`,
    );
  }
  return lan;
}

/**
 * One listen address: loopback, `auto` (primary RFC1918 IPv4), or a single private IP.
 * Never 0.0.0.0 / :: / public. Empty stays loopback so `npm run mcp:http` is local unless auto is set.
 * Loopback aliases bind as 127.0.0.1 so the printed Late URL stays stable.
 */
export function bindPrivateListenHost(requestedHost: string | undefined): string {
  const raw = (requestedHost ?? MCP_LOOPBACK_HOST).trim();
  if (!raw) return MCP_LOOPBACK_HOST;
  const host = stripHost(raw);
  if (!host) return MCP_LOOPBACK_HOST;
  if (isAutoListenHost(host)) return resolveAutoListenHost();
  if (isUnspecified(host) || host === "0:0:0:0:0:0:0:0") {
    throw new Error(refuseMessage(raw));
  }
  if (isLoopbackName(host)) return MCP_LOOPBACK_HOST;
  if (isIP(host) === 4) {
    if (isLinkLocalV4(host) || isRfc1918(host) === false) {
      throw new Error(refuseMessage(raw));
    }
    return host;
  }
  if (isIP(host) === 6) {
    if (isLinkLocalV6(host) || !isUlaV6(host)) {
      throw new Error(refuseMessage(raw));
    }
    return host;
  }
  throw new Error(refuseMessage(raw));
}

export function bindMcpListenHost(requestedHost: string | undefined): string {
  return bindPrivateListenHost(requestedHost);
}

export function bindGuiListenHost(requestedHost: string | undefined): string {
  return bindPrivateListenHost(requestedHost);
}

/** @deprecated Use bindMcpListenHost — still loopback by default; one private IP is allowed. */
export function bindMcpLoopbackOnly(requestedHost: string | undefined): string {
  return bindMcpListenHost(requestedHost);
}

export function resolveMcpListenHost(configHost?: string): string {
  const env = process.env.AGENT_ORCHESTRATOR_MCP_HOST?.trim();
  return bindMcpListenHost(env || configHost);
}

export function resolveGuiListenHost(configHost?: string): string {
  const env = process.env.AGENT_ORCHESTRATOR_GUI_HOST?.trim();
  return bindGuiListenHost(env || configHost);
}

export function formatListenHostForUrl(host: string): string {
  const h = host.replace(/^\[|\]$/g, "");
  return h.includes(":") ? `[${h}]` : h;
}

export function httpListenUrl(host: string, port: number): string {
  return `http://${formatListenHostForUrl(host)}:${port}`;
}

/** Exact Streamable HTTP URL for Late Settings: the host this process bound, plus /mcp. */
export function listenMcpUrl(port: number, host: string = MCP_LOOPBACK_HOST): string {
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error("MCP port must be an integer 1–65535");
  }
  const bound = bindPrivateListenHost(host);
  return `${httpListenUrl(bound, port)}/mcp`;
}

/** Exact Streamable HTTP URL for Late Settings. Always 127.0.0.1 and the port this process bound. */
export function loopbackMcpUrl(port: number): string {
  return listenMcpUrl(port, MCP_LOOPBACK_HOST);
}

export function isLoopbackListenHost(host: string): boolean {
  return isLoopbackName(host);
}

/** Hostname is loopback or the bound private listen host. */
export function listenHostnameOk(hostname: string, listenHost: string): boolean {
  const host = stripHost(hostname);
  if (!host) return false;
  if (isLoopbackName(host)) return true;
  const listen = stripHost(listenHost);
  if (!listen || isLoopbackName(listen)) return false;
  return host === listen;
}

export function listenHostHeaderOk(
  hostHeader: string | undefined,
  port: number,
  listenHost: string = MCP_LOOPBACK_HOST,
): boolean {
  if (!hostHeader) return false;
  const raw = hostHeader.trim().toLowerCase();
  const loopbackHeaders = new Set([
    `127.0.0.1:${port}`,
    `localhost:${port}`,
    "127.0.0.1",
    "localhost",
    `[::1]:${port}`,
    "::1",
    `[::1]`,
  ]);
  if (loopbackHeaders.has(raw)) return true;
  const listen = stripHost(listenHost);
  if (!listen || isLoopbackName(listen)) return false;
  if (listen.includes(":")) {
    return raw === `[${listen}]:${port}` || raw === `[${listen}]` || raw === listen;
  }
  return raw === `${listen}:${port}` || raw === listen;
}

/**
 * Missing Origin is OK (Late sidecar / non-browser). A present Origin must be loopback
 * or the bound private host — not a random website. `null` Origin is rejected.
 */
export function listenOriginOk(
  origin: string | undefined,
  listenHost: string = MCP_LOOPBACK_HOST,
  opts?: { httpOnly?: boolean },
): boolean {
  if (!origin) return true;
  if (origin.trim().toLowerCase() === "null") return false;
  try {
    const url = new URL(origin);
    if (opts?.httpOnly) {
      if (url.protocol !== "http:") return false;
    } else if (url.protocol !== "http:" && url.protocol !== "https:") {
      return false;
    }
    return listenHostnameOk(url.hostname, listenHost);
  } catch {
    return false;
  }
}

/** Stderr lines: the URL to paste, and that Late does not need this process. */
export function lateMcpCopyLines(mcpUrl: string): string[] {
  const hostPart = mcpUrl.replace(/^https?:\/\//i, "").split("/")[0] ?? "";
  const hostname = hostPart.replace(/:\d+$/, "").replace(/^\[|\]$/g, "");
  const lan = hostname !== MCP_LOOPBACK_HOST && hostname !== "localhost";
  const lines = [
    `mcp:    ${mcpUrl}  (Streamable HTTP, no GUI token — copy for Late Settings)`,
    `        Late works with MCP off. This URL is the port this process bound, not a hardcoded default.`,
  ];
  if (lan) {
    lines.push(
      `        One private IP on your computer. Trusted LAN only — firewall to the laptop. Late will not start this process.`,
    );
  }
  return lines;
}
