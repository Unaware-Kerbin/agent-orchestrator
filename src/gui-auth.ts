import { existsSync, readFileSync } from "node:fs";
import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import { join } from "node:path";
import { writeSecureFile } from "./platform.js";
import { stateDir } from "./state.js";

const FILE_NAME = "gui.secret";

export function guiSecretPath(): string {
  return join(stateDir(), FILE_NAME);
}

export function loadOrCreateGuiToken(): { token: string; path: string; created: boolean } {
  const path = guiSecretPath();
  if (existsSync(path)) {
    const token = readFileSync(path, "utf8").trim();
    if (token.length >= 16) return { token, path, created: false };
  }
  const token = randomBytes(32).toString("base64url");
  writeSecureFile(path, `${token}\n`);
  return { token, path, created: true };
}

export function tokensEqual(provided: string, expected: string): boolean {
  const left = createHash("sha256").update(provided).digest();
  const right = createHash("sha256").update(expected).digest();
  return timingSafeEqual(left, right);
}

export function extractBearerToken(header: string | undefined, queryToken: string | null): string | undefined {
  if (header) {
    const match = /^Bearer\s+(.+)$/i.exec(header.trim());
    if (match?.[1]) return match[1].trim();
  }
  if (queryToken && queryToken.trim()) return queryToken.trim();
  return undefined;
}
