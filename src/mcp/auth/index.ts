import { extractBearerToken, tokensEqual } from "../../gui-auth.js";
import { loadSecretsFile } from "../../secrets.js";
import { createLdapAuthenticator } from "./ldap.js";
import { createRadiusAuthenticator } from "./radius.js";
import { McpSessionStore } from "./sessions.js";
import type {
  McpAuthConfig,
  McpAuthMethod,
  McpCredentials,
  McpPrincipal,
  PasswordLogin,
} from "./types.js";

const DEFAULT_TTL_MS = 60 * 60 * 1000;

function splitList(raw: string | undefined): string[] {
  if (!raw?.trim()) return [];
  return raw
    .split(/[,;]/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function secretValue(...names: string[]): string {
  const stored = loadSecretsFile();
  for (const name of names) {
    const env = process.env[name]?.trim();
    if (env) return env;
    const file = stored[name]?.trim();
    if (file) return file;
  }
  return "";
}

function parseMethods(raw: string | undefined): McpAuthMethod[] {
  const parts = (raw ?? "local-token")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
  const out: McpAuthMethod[] = [];
  for (const part of parts) {
    if (part === "local-token" || part === "ldap" || part === "ldaps" || part === "radius") {
      const method: McpAuthMethod = part === "ldaps" ? "ldap" : part;
      if (!out.includes(method)) out.push(method);
    }
  }
  return out.length ? out : ["local-token"];
}

export function loadMcpAuthConfig(localToken: string): McpAuthConfig {
  const methods = parseMethods(process.env.AGENT_ORCHESTRATOR_MCP_AUTH);
  const ldapUrl = process.env.AGENT_ORCHESTRATOR_LDAP_URL?.trim() ?? "";
  const radiusHost = process.env.AGENT_ORCHESTRATOR_RADIUS_HOST?.trim() ?? "";
  const ldapAllowedGroups = splitList(process.env.AGENT_ORCHESTRATOR_LDAP_ALLOWED_GROUPS);
  const radiusAllowedFilterIds = splitList(process.env.AGENT_ORCHESTRATOR_RADIUS_ALLOWED_FILTER_IDS);
  const config: McpAuthConfig = {
    methods,
    localToken,
    ldapAllowedGroups,
    radiusAllowedFilterIds,
    sessionTtlMs: DEFAULT_TTL_MS,
  };
  if (methods.includes("ldap") && ldapUrl) {
    config.ldap = createLdapAuthenticator({
      url: ldapUrl,
      bindDnTemplate:
        process.env.AGENT_ORCHESTRATOR_LDAP_BIND_DN?.trim() || "CN={username},CN=Users,DC=example,DC=com",
      serviceDn: process.env.AGENT_ORCHESTRATOR_LDAP_SERVICE_DN?.trim() || undefined,
      servicePassword: secretValue("LDAP_BIND_PASSWORD", "AGENT_ORCHESTRATOR_LDAP_PASSWORD") || undefined,
      baseDn: process.env.AGENT_ORCHESTRATOR_LDAP_BASE_DN?.trim() || "",
      filter: process.env.AGENT_ORCHESTRATOR_LDAP_FILTER?.trim() || "(sAMAccountName={username})",
      groupAttr: process.env.AGENT_ORCHESTRATOR_LDAP_GROUP_ATTR?.trim() || "memberOf",
      allowedGroups: ldapAllowedGroups,
      tlsRejectUnauthorized: process.env.AGENT_ORCHESTRATOR_LDAP_TLS_REJECT_UNAUTHORIZED !== "0",
    });
  }
  if (methods.includes("radius") && radiusHost) {
    const port = Number(process.env.AGENT_ORCHESTRATOR_RADIUS_PORT ?? "1812");
    config.radius = createRadiusAuthenticator({
      host: radiusHost,
      port: Number.isInteger(port) ? port : 1812,
      secret: secretValue("RADIUS_SECRET", "AGENT_ORCHESTRATOR_RADIUS_SECRET"),
      nasIdentifier: process.env.AGENT_ORCHESTRATOR_RADIUS_NAS_IDENTIFIER?.trim() || "agent-orchestrator",
      allowedFilterIds: radiusAllowedFilterIds,
    });
  }
  return config;
}

export function extractMcpCredentials(
  authorization: string | undefined,
  headerToken: string | undefined,
): McpCredentials {
  if (authorization) {
    const basic = /^Basic\s+(.+)$/i.exec(authorization.trim());
    if (basic?.[1]) {
      try {
        const decoded = Buffer.from(basic[1], "base64").toString("utf8");
        const idx = decoded.indexOf(":");
        if (idx >= 0) {
          return {
            kind: "basic",
            username: decoded.slice(0, idx),
            password: decoded.slice(idx + 1),
          };
        }
      } catch {
        return { kind: "none" };
      }
    }
  }
  const bearer = extractBearerToken(authorization, null) ?? headerToken?.trim();
  if (bearer) return { kind: "bearer", token: bearer };
  return { kind: "none" };
}

export class McpAuth {
  readonly sessions: McpSessionStore;

  constructor(readonly config: McpAuthConfig) {
    this.sessions = new McpSessionStore(config.sessionTtlMs);
  }

  usesPasswordAuth(): boolean {
    return this.config.methods.includes("ldap") || this.config.methods.includes("radius");
  }

  async authenticate(creds: McpCredentials): Promise<McpPrincipal | undefined> {
    if (creds.kind === "none") return undefined;
    if (creds.kind === "bearer") {
      if (this.config.methods.includes("local-token") && this.config.localToken) {
        if (tokensEqual(creds.token, this.config.localToken)) {
          return { method: "local-token", subject: "local-token", groups: [] };
        }
      }
      const session = this.sessions.findEqual(creds.token);
      if (session) return session;
      return undefined;
    }
    const login = await this.login(creds);
    return login.ok ? login.principal : undefined;
  }

  async login(
    creds: PasswordLogin,
  ): Promise<{ ok: true; principal: McpPrincipal; token: string; expiresAt: number } | { ok: false; status: number; error: string }> {
    const username = creds.username.trim();
    if (!username || !creds.password) {
      return { ok: false, status: 400, error: "username and password required" };
    }
    const errors: string[] = [];
    for (const method of this.config.methods) {
      const plugin = method === "ldap" ? this.config.ldap : method === "radius" ? this.config.radius : undefined;
      if (!plugin) continue;
      const result = await plugin.login({ username, password: creds.password });
      if (!result.ok) {
        errors.push(result.reason);
        continue;
      }
      const issued = this.sessions.issue({ subject: result.subject, groups: result.groups });
      return {
        ok: true,
        principal: { method: "session", subject: result.subject, groups: result.groups },
        token: issued.token,
        expiresAt: issued.expiresAt,
      };
    }
    if (!this.usesPasswordAuth()) {
      return { ok: false, status: 400, error: "Password login is off. Use Authorization: Bearer with the loopback GUI token." };
    }
    return { ok: false, status: 401, error: errors[0] || "unauthorized" };
  }
}

export function mcpUnauthorizedHeaders(): Record<string, string> {
  return {
    "www-authenticate": 'Bearer realm="agent-orchestrator"',
    "cache-control": "no-store",
  };
}
