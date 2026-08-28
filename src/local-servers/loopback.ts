/** Loopback bind/connect host for local model HTTP. Never 0.0.0.0. */
export const LOOPBACK_HOST = "127.0.0.1" as const;

export const DEFAULT_OLLAMA_BASE = `http://${LOOPBACK_HOST}:11434/v1`;
export const DEFAULT_LLAMACPP_BASE = `http://${LOOPBACK_HOST}:8080/v1`;

const LOOPBACK_NAMES = new Set(["127.0.0.1", "localhost", "::1"]);

export function isLoopbackHostname(hostname: string): boolean {
  const host = hostname.trim().toLowerCase().replace(/^\[|\]$/g, "");
  return LOOPBACK_NAMES.has(host);
}

/**
 * True when the URL is HTTP(S) on loopback. Invalid URLs are not loopback.
 */
export function isLoopbackHttpUrl(raw: string | undefined): boolean {
  if (!raw?.trim()) return false;
  try {
    parseLoopbackHttpUrl(raw, "local server");
    return true;
  } catch {
    return false;
  }
}

export function parseLoopbackHttpUrl(raw: string, label: string): URL {
  const trimmed = raw.trim();
  let url: URL;
  try {
    url = new URL(trimmed);
  } catch {
    throw new Error(
      `${label} baseUrl must be a valid http URL on ${LOOPBACK_HOST} (typical http://${LOOPBACK_HOST}:11434/v1)`,
    );
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`${label} baseUrl must use http or https`);
  }
  if (!isLoopbackHostname(url.hostname)) {
    throw new Error(
      `Refusing ${label} host "${url.hostname}". Only ${LOOPBACK_HOST} is allowed so the server stays on this machine. Do not use 0.0.0.0, public interfaces, or tunnels.`,
    );
  }
  return url;
}

/** Canonical OpenAI-compat base such as `http://127.0.0.1:11434/v1` (no trailing slash). */
export function normalizeLoopbackOpenAiUrl(raw: string, label: string): string {
  const url = parseLoopbackHttpUrl(raw, label);
  let path = url.pathname.replace(/\/+$/, "");
  if (!path) path = "/v1";
  const port = url.port ? `:${url.port}` : "";
  return `http://${LOOPBACK_HOST}${port}${path}`;
}

/** Origin only, e.g. `http://127.0.0.1:11434`. */
export function loopbackOrigin(raw: string, label: string): string {
  const url = parseLoopbackHttpUrl(raw, label);
  const port = url.port ? `:${url.port}` : "";
  return `http://${LOOPBACK_HOST}${port}`;
}
