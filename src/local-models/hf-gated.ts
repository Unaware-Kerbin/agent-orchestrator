import { parseHubModelId } from "../identity.js";
import { redactSecretText } from "../redact.js";
import { resolveHfToken } from "../secrets.js";

const HF_ORIGIN = "https://huggingface.co";
const PROBE_TIMEOUT_MS = 8_000;

export const GATED_LICENSE_TOKEN_SET =
  "Token is set but Hugging Face still denied access. Accept the model license on the model card while logged into the same Hugging Face account, then retry. Do not commit the token.";

export const GATED_LICENSE_TOKEN_MISSING =
  "Repo is gated. Accept the license on the Hugging Face model card while logged in, then paste a read token in Settings → Local models (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN). Create a token at https://huggingface.co/settings/tokens. Do not commit the token.";

export type HubProbeFetch = (url: string | URL, init?: RequestInit) => Promise<Response>;

export function hubGatedDeniedMessage(tokenPresent: boolean): string {
  return tokenPresent ? GATED_LICENSE_TOKEN_SET : GATED_LICENSE_TOKEN_MISSING;
}

export function isDeniedHubStatus(status: number | undefined): boolean {
  return status === 401 || status === 403;
}

/** Live Hub URLs for the early gated probe. Never includes a token. */
export function hubGatedProbeUrls(repo: string, revision = "main"): { card: string; config: string } {
  const id = parseHubModelId(repo);
  const rev = revision.trim() || "main";
  if (rev.includes("..") || rev.includes("/") || rev.includes("\\") || rev.includes(":")) {
    throw new Error("invalid Hugging Face revision");
  }
  return {
    card: `${HF_ORIGIN}/api/models/${id}`,
    config: `${HF_ORIGIN}/${id}/resolve/${rev}/config.json`,
  };
}

/**
 * HEAD-equivalent GET of the model card and config.json with the saved HF token.
 * 401/403 aborts before a Hub snapshot download. Does not log the token.
 */
export async function probeHubGatedAccess(
  repo: string,
  options?: {
    revision?: string;
    fetchFn?: HubProbeFetch;
    token?: string | null;
    timeoutMs?: number;
  },
): Promise<void> {
  const token = options?.token === undefined ? resolveHfToken() : options.token;
  const present = Boolean(token && String(token).trim());
  const urls = hubGatedProbeUrls(repo, options?.revision ?? "main");
  const fetchFn = options?.fetchFn ?? fetch;
  const timeoutMs = options?.timeoutMs ?? PROBE_TIMEOUT_MS;
  const headers: Record<string, string> = {
    accept: "application/json",
    "user-agent": "agent-orchestrator",
  };
  if (present) headers.authorization = `Bearer ${String(token).trim()}`;

  const cardStatus = await hubProbeStatus(fetchFn, urls.card, headers, timeoutMs);
  if (isDeniedHubStatus(cardStatus)) {
    throw new Error(redactSecretText(`401 ${hubGatedDeniedMessage(present)}`));
  }
  const configHeaders = { ...headers, Range: "bytes=0-0" };
  const configStatus = await hubProbeStatus(fetchFn, urls.config, configHeaders, timeoutMs);
  if (isDeniedHubStatus(configStatus)) {
    throw new Error(redactSecretText(`401 ${hubGatedDeniedMessage(present)}`));
  }
}

async function hubProbeStatus(
  fetchFn: HubProbeFetch,
  url: string,
  headers: Record<string, string>,
  timeoutMs: number,
): Promise<number | undefined> {
  try {
    const res = await fetchFn(url, {
      method: "GET",
      headers,
      signal: AbortSignal.timeout(timeoutMs),
      redirect: "follow",
    });
    try {
      await res.arrayBuffer();
    } catch {
      /* drain only */
    }
    return res.status;
  } catch {
    return undefined;
  }
}
