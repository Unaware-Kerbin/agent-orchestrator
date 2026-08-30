import assert from "node:assert/strict";
import { test } from "node:test";
import { resolveWriterCwd } from "../src/chat/service.js";

const allow = new Set(["/home/allowed", "/home/other"]);
const tryCwd = (path: string) => (allow.has(path) ? path : undefined);

test("writer cwd returns the allowlisted candidate", () => {
  assert.equal(
    resolveWriterCwd({
      writesLocalFiles: true,
      granted: "/home/allowed",
      defaultCwd: "/home/other",
      tryCwd,
    }),
    "/home/allowed",
  );
});

test("writer cwd does not fall back to another allowlisted folder", () => {
  assert.equal(
    resolveWriterCwd({
      writesLocalFiles: true,
      granted: "/tmp/outside",
      fallback: "/tmp/also-outside",
      defaultCwd: "/home/allowed",
      tryCwd,
    }),
    undefined,
  );
});

test("late wrap uses the granted allowlisted path and does not need defaultCwd", () => {
  assert.equal(
    resolveWriterCwd({
      writesLocalFiles: true,
      lateWrap: true,
      granted: "/home/allowed",
      defaultCwd: "/home/other",
      tryCwd,
    }),
    "/home/allowed",
  );
  assert.equal(
    resolveWriterCwd({
      writesLocalFiles: true,
      lateWrap: true,
      fallback: "/home/allowed",
      defaultCwd: "/home/other",
      tryCwd,
    }),
    "/home/allowed",
  );
});

test("late wrap refuses when no granted path is allowlisted", () => {
  assert.equal(
    resolveWriterCwd({
      writesLocalFiles: true,
      lateWrap: true,
      granted: "/tmp/outside",
      defaultCwd: "/home/allowed",
      tryCwd,
    }),
    undefined,
  );
  assert.equal(
    resolveWriterCwd({
      writesLocalFiles: true,
      lateWrap: true,
      defaultCwd: "/home/allowed",
      tryCwd,
    }),
    undefined,
  );
});

test("non-writers never get a cwd", () => {
  assert.equal(
    resolveWriterCwd({
      writesLocalFiles: false,
      granted: "/home/allowed",
      tryCwd,
    }),
    undefined,
  );
});
