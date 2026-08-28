import { readdirSync, readFileSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { ensureSecureDir, writeSecureFile } from "./platform.js";
import { stateDir } from "./state.js";

export const LOGO_MAX_BYTES = 512 * 1024;

const LOGO_TYPES: Record<string, string> = {
  "image/png": ".png",
  "image/jpeg": ".jpg",
  "image/webp": ".webp",
};

const BACKEND_ID = /^[a-zA-Z][a-zA-Z0-9_-]*$/;

/** Magic-byte sniff so SVG/HTML/script cannot be stored as a "PNG". */
export function sniffLogoMime(buffer: Buffer): "image/png" | "image/jpeg" | "image/webp" | undefined {
  if (buffer.length < 12) return undefined;
  if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4e && buffer[3] === 0x47) {
    return "image/png";
  }
  if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) return "image/jpeg";
  if (
    buffer[0] === 0x52 &&
    buffer[1] === 0x49 &&
    buffer[2] === 0x46 &&
    buffer[3] === 0x46 &&
    buffer[8] === 0x57 &&
    buffer[9] === 0x45 &&
    buffer[10] === 0x42 &&
    buffer[11] === 0x50
  ) {
    return "image/webp";
  }
  return undefined;
}

export function parseNickname(raw: unknown): string | undefined {
  if (raw === undefined || raw === null) return undefined;
  if (typeof raw !== "string") throw new Error("nickname must be a string");
  if (/[\r\n\u0000]/.test(raw)) throw new Error("nickname cannot contain line breaks");
  const trimmed = raw.trim().replace(/\s+/g, " ");
  if (!trimmed) return undefined;
  if (trimmed.length > 48) throw new Error("nickname must be 48 characters or fewer");
  if (/[\u0001-\u0008\u000b\u000c\u000e-\u001f\u007f<>]/.test(trimmed)) {
    throw new Error("nickname cannot contain control characters or markup");
  }
  return trimmed;
}

const MODEL_ID_MAX = 256;
/** HF `org/name`, Ollama `llama3.2:latest`, and similar catalog ids. */
const MODEL_ID_RE = /^[A-Za-z0-9][A-Za-z0-9._/:@+-]*$/;

export function parseModelId(raw: unknown): string {
  if (typeof raw !== "string") throw new Error("model must be a string");
  if (/[\r\n\u0000]/.test(raw)) throw new Error("model cannot contain line breaks");
  const trimmed = raw.trim();
  if (!trimmed) throw new Error("model is required");
  if (trimmed.length > MODEL_ID_MAX) throw new Error("model must be 256 characters or fewer");
  if (/[\u0001-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(trimmed)) {
    throw new Error("model cannot contain control characters");
  }
  if (!MODEL_ID_RE.test(trimmed)) throw new Error("model contains unsupported characters");
  return trimmed;
}

export function logosDir(): string {
  const dir = join(stateDir(), "logos");
  ensureSecureDir(dir);
  return dir;
}

export function hasLogo(backendId: string): boolean {
  return Boolean(findLogo(backendId));
}

/** Loopback GUI URL — Late fetches with Bearer. Never put the token in this string. */
export function backendLogoUrl(backendId: string): string {
  const port = Number(process.env.AGENT_ORCHESTRATOR_GUI_PORT ?? "8787");
  const safe = Number.isInteger(port) && port > 0 && port < 65536 ? port : 8787;
  return `http://127.0.0.1:${safe}/api/backends/${encodeURIComponent(backendId)}/logo`;
}

export function findLogo(backendId: string): { path: string; mime: string } | undefined {
  if (!BACKEND_ID.test(backendId)) return undefined;
  const dir = logosDir();
  let names: string[] = [];
  try {
    names = readdirSync(dir);
  } catch {
    return undefined;
  }
  const prefix = `${backendId}.`;
  const match = names.find((name) => name.startsWith(prefix) && !name.includes("..") && !name.includes("/") && !name.includes("\\"));
  if (!match) return undefined;
  const ext = match.slice(backendId.length);
  const mime = Object.entries(LOGO_TYPES).find(([, suffix]) => suffix === ext)?.[0];
  if (!mime) return undefined;
  return { path: join(dir, match), mime };
}

export function saveLogo(backendId: string, buffer: Buffer, mime: string): { mime: string } {
  if (!BACKEND_ID.test(backendId)) throw new Error("Invalid backend id");
  const ext = LOGO_TYPES[mime];
  if (!ext) throw new Error("Logo must be PNG, JPEG, or WebP");
  const sniffed = sniffLogoMime(buffer);
  if (!sniffed) throw new Error("Logo must be PNG, JPEG, or WebP (not SVG or script)");
  if (sniffed !== mime) throw new Error("Logo type does not match file contents");
  if (buffer.length === 0) throw new Error("Logo file is empty");
  if (buffer.length > LOGO_MAX_BYTES) throw new Error("Logo must be 512 KiB or smaller");
  removeLogo(backendId);
  const dest = join(logosDir(), `${backendId}${ext}`);
  writeSecureFile(dest, buffer);
  return { mime };
}

export function decodeLogoDataUrl(data: string): { buffer: Buffer; mime: string } {
  const match = /^data:(image\/(?:png|jpeg|webp));base64,([A-Za-z0-9+/=\s]+)$/i.exec(data.trim());
  if (!match) throw new Error("Logo must be a PNG, JPEG, or WebP data URL");
  const mime = match[1]!.toLowerCase();
  const buffer = Buffer.from(match[2]!.replace(/\s/g, ""), "base64");
  if (!buffer.length) throw new Error("Logo file is empty");
  const sniffed = sniffLogoMime(buffer);
  if (!sniffed || sniffed !== mime) throw new Error("Logo must be a PNG, JPEG, or WebP data URL");
  return { buffer, mime };
}

export function removeLogo(backendId: string): boolean {
  const found = findLogo(backendId);
  if (!found) {
    if (!BACKEND_ID.test(backendId)) return false;
    try {
      for (const name of readdirSync(logosDir())) {
        if (name.startsWith(`${backendId}.`)) unlinkSync(join(logosDir(), name));
      }
    } catch {
      return false;
    }
    return false;
  }
  unlinkSync(found.path);
  return true;
}

export function readLogo(backendId: string): { bytes: Buffer; mime: string } | undefined {
  const found = findLogo(backendId);
  if (!found) return undefined;
  return { bytes: readFileSync(found.path), mime: found.mime };
}
