import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { defaultWorkspaceCwd } from "../src/config.js";
import {
  CURSOR_NOT_CONFIGURED,
  CursorProvider,
  formatCursorError,
  normalizeGitHttpsUrl,
  resolveCursorLocalCwd,
} from "../src/providers/cursor.js";

test("missing CURSOR_API_KEY is a one-line Cursor not configured", async () => {
  const prev = process.env.CURSOR_API_KEY;
  delete process.env.CURSOR_API_KEY;
  try {
    const provider = new CursorProvider("cursor-local", { type: "cursor", runtime: "local" });
    const health = provider.health();
    assert.equal(health.ready, false);
    assert.equal(health.reason, CURSOR_NOT_CONFIGURED);
    const result = await provider.run({ prompt: "pong" });
    assert.equal(result.status, "error");
    assert.equal(result.error, CURSOR_NOT_CONFIGURED);
    assert.doesNotMatch(result.error ?? "", /at |Error:|stack/i);
  } finally {
    if (prev === undefined) delete process.env.CURSOR_API_KEY;
    else process.env.CURSOR_API_KEY = prev;
  }
});

test("formatCursorError never dumps a stack; auth becomes Cursor not configured", () => {
  assert.equal(formatCursorError(new Error("CURSOR_API_KEY is not set")), CURSOR_NOT_CONFIGURED);
  assert.equal(formatCursorError(new Error("Unauthorized 401 invalid API key")), CURSOR_NOT_CONFIGURED);
  assert.match(
    formatCursorError(new Error("Run stream is no longer available\n    at wait (sdk.js:1:1)")),
    /skipped so other speakers/,
  );
  assert.equal(formatCursorError(new Error("Cursor timed out after 90s")), "Cursor timed out after 90s");
  const stacked = formatCursorError(new Error("boom\n    at Object.run (cursor.ts:12:3)\n    at async"));
  assert.equal(stacked.includes("cursor.ts"), false);
  assert.equal(stacked, "boom");
  assert.equal(formatCursorError("none"), "Cursor run failed (no error detail from the SDK).");
  assert.equal(formatCursorError(""), "Cursor run failed (no error detail from the SDK).");
  assert.equal(formatCursorError(undefined), "Cursor run failed (no error detail from the SDK).");
});

test("local cwd is process.cwd / git work tree, never a hardcoded home path", () => {
  const cwd = resolveCursorLocalCwd();
  assert.ok(cwd === process.cwd() || cwd.startsWith(process.cwd()));
  const tmp = mkdtempSync(join(tmpdir(), "orch-cwd-"));
  assert.equal(resolveCursorLocalCwd(tmp), process.cwd());
  const cfg = defaultWorkspaceCwd({ backends: {}, specialists: {}, workflows: {} });
  assert.equal(cfg, process.cwd());
});

test("ssh git remotes normalize to https for Cursor cloud", () => {
  assert.equal(
    normalizeGitHttpsUrl("git@github.com:Unaware-Kerbin/agent-orchestrator.git"),
    "https://github.com/Unaware-Kerbin/agent-orchestrator.git",
  );
  assert.equal(
    normalizeGitHttpsUrl("https://github.com/Unaware-Kerbin/agent-orchestrator.git"),
    "https://github.com/Unaware-Kerbin/agent-orchestrator.git",
  );
  assert.equal(normalizeGitHttpsUrl(""), undefined);
});
