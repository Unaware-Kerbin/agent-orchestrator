import assert from "node:assert/strict";
import { test } from "node:test";
import {
  installMcpProcessGuards,
  installProcessGuards,
  MCP_PROCESS_GUARD_EXCEPTION,
  MCP_PROCESS_GUARD_REJECTION,
  processGuardException,
  processGuardRejection,
} from "../src/mcp/process-guard.js";

test("mcp:http process guards register and uninstall listeners", () => {
  const beforeRejection = process.listenerCount("unhandledRejection");
  const beforeException = process.listenerCount("uncaughtException");
  const stop = installMcpProcessGuards();
  try {
    assert.equal(process.listenerCount("unhandledRejection"), beforeRejection + 1);
    assert.equal(process.listenerCount("uncaughtException"), beforeException + 1);
  } finally {
    stop();
  }
  assert.equal(process.listenerCount("unhandledRejection"), beforeRejection);
  assert.equal(process.listenerCount("uncaughtException"), beforeException);
});

test("guard handler logs a Cursor-like failure without throwing", () => {
  const lines: string[] = [];
  const stop = installMcpProcessGuards((msg) => lines.push(String(msg)));
  try {
    const listeners = process.listeners("unhandledRejection");
    const last = listeners[listeners.length - 1];
    assert.equal(typeof last, "function");
    (last as (reason: unknown) => void)(new Error("cursor sdk late fail"));
    assert.ok(lines.some((l) => l.includes(MCP_PROCESS_GUARD_REJECTION)));
    assert.equal(MCP_PROCESS_GUARD_EXCEPTION.length > 0, true);
  } finally {
    stop();
  }
});

test("gui process guards use a distinct label and do not throw", () => {
  const lines: string[] = [];
  const stop = installProcessGuards("gui", (msg) => lines.push(String(msg)));
  try {
    const listeners = process.listeners("unhandledRejection");
    const last = listeners[listeners.length - 1];
    (last as (reason: unknown) => void)(new Error("sse write after destroy"));
    assert.ok(lines.some((l) => l.includes(processGuardRejection("gui"))));
    assert.equal(processGuardException("gui").includes("gui"), true);
  } finally {
    stop();
  }
});
