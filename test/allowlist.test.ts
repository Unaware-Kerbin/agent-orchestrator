import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { WriteAllowlist, canonicalizeDirectory, isPathInside, resolveContainedPath } from "../src/allowlist.js";

function tempProject(): string {
  return mkdtempSync(join(tmpdir(), "orch-allow-"));
}

test("cwd inside allowed directory is accepted", () => {
  const root = tempProject();
  const child = join(root, "src");
  mkdirSync(child);
  const allow = new WriteAllowlist(join(root, "allowlist.json"), [canonicalizeDirectory(root)]);
  assert.equal(allow.assertCwd(child), resolveContainedPath(child));
});

test(".. escape is rejected", () => {
  const root = tempProject();
  const outside = tempProject();
  const allow = new WriteAllowlist(join(root, "allowlist.json"), [canonicalizeDirectory(root)]);
  assert.throws(() => allow.assertCwd(outside), /not inside/);
  assert.throws(() => allow.assertWritable(join(outside, "secret.txt")), /not inside/);
});

test("symlink that escapes the allowlist is rejected", () => {
  const root = tempProject();
  const secret = tempProject();
  writeFileSync(join(secret, "key.txt"), "nope");
  symlinkSync(secret, join(root, "link"));
  const allow = new WriteAllowlist(join(root, "allowlist.json"), [canonicalizeDirectory(root)]);
  assert.throws(() => allow.assertWritable(join(root, "link", "key.txt")), /not inside/);
  assert.throws(() => allow.assertCwd(join(root, "link")), /not inside/);
});

test("add/remove persist and isPathInside handles exact root", () => {
  const root = tempProject();
  const extra = tempProject();
  const allow = new WriteAllowlist(join(root, "allowlist.json"), [canonicalizeDirectory(root)]);
  assert.ok(isPathInside(root, root));
  allow.add(extra);
  assert.ok(allow.list().includes(resolveContainedPath(extra)));
  assert.equal(allow.tryCwd(extra), resolveContainedPath(extra));
  allow.remove(extra);
  assert.equal(allow.list().includes(resolveContainedPath(extra)), false);
  assert.equal(allow.tryCwd(extra), undefined);
});
