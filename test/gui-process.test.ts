import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  clearGuiPid,
  guiAddrInUseMessage,
  guiPidPath,
  isGuiCmdline,
  readGuiPid,
  writeGuiPid,
} from "../src/gui-process.js";

function withStateDir<T>(fn: () => T): T {
  const dir = mkdtempSync(join(tmpdir(), "gui-pid-"));
  const prev = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  process.env.AGENT_ORCHESTRATOR_STATE_DIR = dir;
  try {
    return fn();
  } finally {
    if (prev === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prev;
  }
}

test("isGuiCmdline matches this project's GUI entrypoint only", () => {
  assert.equal(isGuiCmdline("tsx src/gui.ts"), true);
  assert.equal(isGuiCmdline("/home/x/.local/bin/node --import loader.mjs src/gui.ts"), true);
  assert.equal(isGuiCmdline("tsx src/gui.ts --open"), true);
  assert.equal(isGuiCmdline("C:\\Users\\me\\AppData\\tsx src\\gui.ts"), true);
  assert.equal(isGuiCmdline("tsx src/gui.ts --stop"), false);
  assert.equal(isGuiCmdline("vllm serve /models/foo"), false);
  assert.equal(isGuiCmdline("node dist/index.js"), false);
});

test("guiAddrInUseMessage tells GUI apart from vLLM and how to stop", () => {
  const text = guiAddrInUseMessage(8787);
  assert.match(text, /EADDRINUSE/);
  assert.match(text, /8787/);
  assert.match(text, /not vLLM/);
  assert.match(text, /npm run gui:stop/);
  assert.match(text, /gui\.secret/);
});

test("writeGuiPid stores a gitignored pid file under the state dir", () => {
  withStateDir(() => {
    writeGuiPid(4242);
    assert.equal(readGuiPid(), 4242);
    assert.equal(readFileSync(guiPidPath(), "utf8").trim(), "4242");
    assert.ok(guiPidPath().endsWith("gui.pid"));
    clearGuiPid(4242);
    assert.equal(readGuiPid(), undefined);
  });
});
