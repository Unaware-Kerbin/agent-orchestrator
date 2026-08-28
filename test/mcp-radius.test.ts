import assert from "node:assert/strict";
import { test } from "node:test";
import {
  RADIUS_ACCESS_ACCEPT,
  RADIUS_ACCESS_REJECT,
  createRadiusAuthenticator,
  encodeAccessResponse,
  parseRadiusAttributes,
} from "../src/mcp/auth/radius.js";

test("RADIUS Access-Accept with matching Filter-Id succeeds", async () => {
  const auth = createRadiusAuthenticator({
    host: "127.0.0.1",
    port: 1812,
    secret: "radius-shared-secret",
    nasIdentifier: "agent-orchestrator",
    allowedFilterIds: ["mcp-users"],
    send: async (packet) => {
      const parsed = parseRadiusAttributes(packet);
      assert.equal(parsed.code, 1);
      return encodeAccessResponse({
        code: RADIUS_ACCESS_ACCEPT,
        identifier: packet[1] ?? 1,
        filterIds: ["mcp-users"],
      });
    },
  });
  const ok = await auth.login({ username: "alice", password: "alice-pass" });
  assert.equal(ok.ok, true);
  if (ok.ok) {
    assert.equal(ok.subject, "alice");
    assert.deepEqual(ok.groups, ["mcp-users"]);
  }
});

test("RADIUS reject and Filter-Id miss are denied", async () => {
  const rejectAuth = createRadiusAuthenticator({
    host: "127.0.0.1",
    port: 1812,
    secret: "radius-shared-secret",
    nasIdentifier: "agent-orchestrator",
    allowedFilterIds: [],
    send: async (packet) =>
      encodeAccessResponse({ code: RADIUS_ACCESS_REJECT, identifier: packet[1] ?? 1 }),
  });
  const denied = await rejectAuth.login({ username: "bob", password: "nope" });
  assert.equal(denied.ok, false);

  const filterAuth = createRadiusAuthenticator({
    host: "127.0.0.1",
    port: 1812,
    secret: "radius-shared-secret",
    nasIdentifier: "agent-orchestrator",
    allowedFilterIds: ["mcp-users"],
    send: async (packet) =>
      encodeAccessResponse({
        code: RADIUS_ACCESS_ACCEPT,
        identifier: packet[1] ?? 1,
        filterIds: ["guest"],
      }),
  });
  const filterDenied = await filterAuth.login({ username: "alice", password: "alice-pass" });
  assert.equal(filterDenied.ok, false);
  if (!filterDenied.ok) assert.match(filterDenied.reason, /Filter-Id/);
});
