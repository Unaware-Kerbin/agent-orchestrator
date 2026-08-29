import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, symlinkSync, writeFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { WriteAllowlist, canonicalizeDirectory } from "../src/allowlist.js";
import { applyParsedFiles, assertSafeRelPath, parseOrchestratorFiles } from "../src/chat/apply-patch.js";

test("parseOrchestratorFiles reads fenced JSON and unified diff", () => {
  const fenced = parseOrchestratorFiles(`plan
\`\`\`orchestrator-files
{"files":[{"path":"src/a.ts","content":"export const n = 1;\\n"}]}
\`\`\`
`);
  assert.equal(fenced.length, 1);
  assert.equal(fenced[0]?.path, "src/a.ts");
  assert.match(fenced[0]?.content ?? "", /export const n/);

  const diff = parseOrchestratorFiles(`diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -0,0 +1,2 @@
+hello
+world
`);
  assert.equal(diff[0]?.path, "README.md");
  assert.match(diff[0]?.content ?? "", /hello/);
});

test("generic json fences are not treated as apply-patch files", () => {
  const fence = ["example", "```json", JSON.stringify({ files: [{ path: "pwn.txt", content: "hi" }] }), "```"].join("\n");
  assert.deepEqual(parseOrchestratorFiles(fence), []);
});

test("assertSafeRelPath rejects escapes and orchestrator internals", () => {
  assert.equal(assertSafeRelPath("src/ok.ts"), "src/ok.ts");
  assert.throws(() => assertSafeRelPath("../secret"), /escapes/);
  assert.throws(() => assertSafeRelPath(".orchestrator/x"), /refusing/);
  assert.throws(() => assertSafeRelPath("write-allowlist.json"), /refusing/);
  assert.throws(() => assertSafeRelPath("./write-allowlist.json"), /refusing/);
  assert.throws(() => assertSafeRelPath("src/../.orchestrator/x"), /refusing/);
  assert.throws(() => assertSafeRelPath("/etc/passwd"), /absolute/);
  assert.throws(() => assertSafeRelPath("C:/Windows/notepad.exe"), /absolute/);
  assert.throws(() => assertSafeRelPath("."), /empty/);
  assert.throws(() => assertSafeRelPath("src/ok.txt\0../x"), /control/);
});

test("applyParsedFiles writes utf8 inside the allowlist and rejects outside", () => {
  const root = mkdtempSync(join(tmpdir(), "orch-patch-"));
  mkdirSync(join(root, "src"));
  const allow = new WriteAllowlist(join(root, "allowlist.json"), [canonicalizeDirectory(root)]);
  const { written } = applyParsedFiles({
    cwd: root,
    files: [{ path: "src/hello.txt", content: "hi\n" }],
    allowlist: allow,
  });
  assert.equal(written.length, 1);
  assert.equal(readFileSync(join(root, "src/hello.txt"), "utf8"), "hi\n");
  assert.throws(
    () =>
      applyParsedFiles({
        cwd: root,
        files: [{ path: "../outside.txt", content: "nope" }],
        allowlist: allow,
      }),
    /escapes|not inside|absolute/,
  );
});

test("applyParsedFiles refuses absolute paths even inside a broader allowlist", () => {
  const parent = mkdtempSync(join(tmpdir(), "orch-patch-parent-"));
  const cwd = join(parent, "proj");
  mkdirSync(cwd);
  const sibling = join(parent, "SECRET.txt");
  writeFileSync(sibling, "KEEPME");
  const allow = new WriteAllowlist(join(parent, "allowlist.json"), [canonicalizeDirectory(parent)]);
  assert.throws(
    () =>
      applyParsedFiles({
        cwd,
        files: [{ path: sibling, content: "PWNED" }],
        allowlist: allow,
      }),
    /absolute/,
  );
  assert.equal(readFileSync(sibling, "utf8"), "KEEPME");
});

test("applyParsedFiles refuses binaries, allowlist file, symlink escapes, and cwd overwrite", () => {
  const root = mkdtempSync(join(tmpdir(), "orch-patch-gate-"));
  mkdirSync(join(root, "src"));
  const allowFile = join(root, "allowlist.json");
  const allow = new WriteAllowlist(allowFile, [canonicalizeDirectory(root)]);
  const outside = mkdtempSync(join(tmpdir(), "orch-patch-out-"));

  assert.throws(
    () => applyParsedFiles({ cwd: root, files: [{ path: "src/bin.txt", content: "a\0b" }], allowlist: allow }),
    /binary/,
  );
  assert.throws(
    () => applyParsedFiles({ cwd: root, files: [{ path: "write-allowlist.json", content: "{}" }], allowlist: allow }),
    /refusing/,
  );
  assert.throws(
    () =>
      applyParsedFiles({
        cwd: root,
        files: [{ path: join(root, "write-allowlist.json"), content: "{}" }],
        allowlist: allow,
      }),
    /absolute|refusing/,
  );
  symlinkSync(outside, join(root, "link"));
  assert.throws(
    () => applyParsedFiles({ cwd: root, files: [{ path: "link/pwn.txt", content: "PWN" }], allowlist: allow }),
    /not inside/,
  );
  assert.equal(existsSync(join(outside, "pwn.txt")), false);
  assert.throws(
    () => applyParsedFiles({ cwd: root, files: [{ path: ".", content: "nope" }], allowlist: allow }),
    /empty|directory/,
  );
});
