import assert from "node:assert/strict";
import { createServer, type Socket } from "node:net";
import { test } from "node:test";
import {
  createLdapAuthenticator,
  encodeLdapBindResponse,
  encodeLdapSearchDone,
  encodeLdapSearchEntry,
  parseLdapMessages,
} from "../src/mcp/auth/ldap.js";

function mockLdapServer(users: Record<string, { password: string; groups: string[] }>): Promise<{
  port: number;
  close: () => Promise<void>;
}> {
  return new Promise((resolve, reject) => {
    const server = createServer((sock: Socket) => {
      sock.on("data", (buf) => {
        const msgs = parseLdapMessages(buf);
        for (const msg of msgs) {
          if (msg.tag === 0x60) {
            const dn = msg.children[1]?.value.toString("utf8") ?? "";
            const pass = msg.children[2]?.value.toString("utf8") ?? "";
            const user = Object.entries(users).find(([name]) => dn.toLowerCase().includes(name.toLowerCase()));
            const ok = user && user[1].password === pass;
            sock.write(encodeLdapBindResponse(msg.id, ok ? 0 : 49));
          } else if (msg.tag === 0x63) {
            const filterVal = msg.children[6]?.children[1]?.value.toString("utf8") ?? "";
            const user = users[filterVal];
            if (user) {
              sock.write(
                Buffer.concat([
                  encodeLdapSearchEntry(msg.id, `CN=${filterVal},CN=Users,DC=example,DC=com`, user.groups),
                  encodeLdapSearchDone(msg.id, 0),
                ]),
              );
            } else {
              sock.write(encodeLdapSearchDone(msg.id, 0));
            }
          }
        }
      });
    });
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      const port = typeof addr === "object" && addr ? addr.port : 0;
      resolve({
        port,
        close: () => new Promise((done, fail) => server.close((err) => (err ? fail(err) : done()))),
      });
    });
  });
}

test("LDAPS plugin binds on a mock directory and enforces group allowlist", async () => {
  const dir = await mockLdapServer({
    alice: { password: "alice-pass", groups: ["CN=MCP Users,OU=Groups,DC=example,DC=com"] },
    bob: { password: "bob-pass", groups: ["CN=Guests,OU=Groups,DC=example,DC=com"] },
  });
  try {
    const connect = async () => {
      const { connect: netConnect } = await import("node:net");
      return new Promise<import("node:net").Socket>((resolve, reject) => {
        const sock = netConnect({ host: "127.0.0.1", port: dir.port });
        sock.once("connect", () => resolve(sock));
        sock.once("error", reject);
      });
    };
    const auth = createLdapAuthenticator({
      url: "ldaps://127.0.0.1:636",
      bindDnTemplate: "CN={username},CN=Users,DC=example,DC=com",
      baseDn: "DC=example,DC=com",
      filter: "(sAMAccountName={username})",
      groupAttr: "memberOf",
      allowedGroups: ["CN=MCP Users,OU=Groups,DC=example,DC=com"],
      tlsRejectUnauthorized: true,
      connect,
    });
    const ok = await auth.login({ username: "alice", password: "alice-pass" });
    assert.equal(ok.ok, true);
    const wrongPass = await auth.login({ username: "alice", password: "nope" });
    assert.equal(wrongPass.ok, false);
    const groupDeny = await auth.login({ username: "bob", password: "bob-pass" });
    assert.equal(groupDeny.ok, false);
    if (!groupDeny.ok) assert.match(groupDeny.reason, /group/i);
  } finally {
    await dir.close();
  }
});

test("plain ldap:// is refused", async () => {
  const auth = createLdapAuthenticator({
    url: "ldap://dc.example.com:389",
    bindDnTemplate: "CN={username},CN=Users,DC=example,DC=com",
    baseDn: "",
    filter: "",
    groupAttr: "memberOf",
    allowedGroups: [],
    tlsRejectUnauthorized: true,
  });
  const result = await auth.login({ username: "alice", password: "x" });
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.reason, /LDAPS/i);
});
