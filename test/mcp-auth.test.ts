import assert from "node:assert/strict";
import { test } from "node:test";
import { McpAuth, extractMcpCredentials } from "../src/mcp/auth/index.js";
import type { McpAuthConfig, PasswordAuthenticator } from "../src/mcp/auth/types.js";

function config(over: Partial<McpAuthConfig> = {}): McpAuthConfig {
  return {
    methods: ["local-token"],
    localToken: "local-token-value-16",
    ldapAllowedGroups: [],
    radiusAllowedFilterIds: [],
    sessionTtlMs: 60_000,
    ...over,
  };
}

test("extractMcpCredentials reads Bearer and Basic, ignores empty", () => {
  assert.equal(extractMcpCredentials(undefined, undefined).kind, "none");
  const bearer = extractMcpCredentials("Bearer abcdefghijklmnop", undefined);
  assert.equal(bearer.kind, "bearer");
  if (bearer.kind === "bearer") assert.equal(bearer.token, "abcdefghijklmnop");
  const header = extractMcpCredentials(undefined, "header-token-value");
  assert.equal(header.kind, "bearer");
  const basic = extractMcpCredentials(`Basic ${Buffer.from("alice:secret").toString("base64")}`, undefined);
  assert.equal(basic.kind, "basic");
  if (basic.kind === "basic") {
    assert.equal(basic.username, "alice");
    assert.equal(basic.password, "secret");
  }
});

test("local-token Bearer is accepted; missing and wrong tokens are not", async () => {
  const auth = new McpAuth(config());
  const ok = await auth.authenticate({ kind: "bearer", token: "local-token-value-16" });
  assert.equal(ok?.method, "local-token");
  assert.equal(await auth.authenticate({ kind: "none" }), undefined);
  assert.equal(await auth.authenticate({ kind: "bearer", token: "nope-nope-nope-no" }), undefined);
});

test("password login issues a session Bearer; wrong password is 401", async () => {
  const ldap: PasswordAuthenticator = {
    method: "ldap",
    async login(creds) {
      if (creds.username === "alice" && creds.password === "correct-horse") {
        return { ok: true, subject: "alice", groups: ["CN=MCP Users,DC=example,DC=com"] };
      }
      return { ok: false, reason: "LDAP rejected the user" };
    },
  };
  const auth = new McpAuth(
    config({ methods: ["local-token", "ldap"], ldap }),
  );
  const denied = await auth.login({ username: "alice", password: "wrong" });
  assert.equal(denied.ok, false);
  if (!denied.ok) assert.equal(denied.status, 401);
  const issued = await auth.login({ username: "alice", password: "correct-horse" });
  assert.equal(issued.ok, true);
  if (!issued.ok) return;
  const principal = await auth.authenticate({ kind: "bearer", token: issued.token });
  assert.equal(principal?.subject, "alice");
  assert.equal(principal?.method, "session");
});
