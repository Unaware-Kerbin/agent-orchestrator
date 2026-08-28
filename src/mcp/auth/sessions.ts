import { randomBytes } from "node:crypto";
import { tokensEqual } from "../../gui-auth.js";
import type { McpPrincipal } from "./types.js";

type SessionRow = McpPrincipal & { token: string; expiresAt: number };

const DEFAULT_TTL_MS = 60 * 60 * 1000;

export class McpSessionStore {
  private readonly sessions = new Map<string, SessionRow>();

  constructor(private readonly ttlMs = DEFAULT_TTL_MS) {}

  issue(principal: Omit<McpPrincipal, "method"> & { method?: McpPrincipal["method"] }): { token: string; expiresAt: number } {
    this.gc();
    const token = randomBytes(32).toString("base64url");
    const expiresAt = Date.now() + this.ttlMs;
    this.sessions.set(token, {
      token,
      expiresAt,
      method: "session",
      subject: principal.subject,
      groups: [...principal.groups],
    });
    return { token, expiresAt };
  }

  lookup(token: string): McpPrincipal | undefined {
    const row = this.sessions.get(token);
    if (!row) return undefined;
    if (row.expiresAt <= Date.now()) {
      this.sessions.delete(token);
      return undefined;
    }
    return { method: "session", subject: row.subject, groups: row.groups };
  }

  findEqual(token: string): McpPrincipal | undefined {
    for (const row of this.sessions.values()) {
      if (row.expiresAt <= Date.now()) continue;
      if (tokensEqual(token, row.token)) {
        return { method: "session", subject: row.subject, groups: row.groups };
      }
    }
    return undefined;
  }

  private gc(): void {
    const now = Date.now();
    for (const [key, row] of this.sessions) {
      if (row.expiresAt <= now) this.sessions.delete(key);
    }
  }
}
