import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import {
  decodeLogoDataUrl,
  findLogo,
  hasLogo,
  parseModelId,
  parseNickname,
  readLogo,
  removeLogo,
  saveLogo,
} from "../src/identity.js";

const PNG_1X1 = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==",
  "base64",
);

test("parseNickname trims, rejects line breaks, markup, and overlong values", () => {
  assert.equal(parseNickname(undefined), undefined);
  assert.equal(parseNickname(""), undefined);
  assert.equal(parseNickname("  Arc   Qwen  "), "Arc Qwen");
  assert.throws(() => parseNickname("no\npe"), /line breaks/);
  assert.throws(() => parseNickname("<script>"), /markup/);
  assert.throws(() => parseNickname("x".repeat(49)), /48 characters/);
});

test("parseModelId allows HF and Ollama tags and rejects line breaks", () => {
  assert.equal(parseModelId("llama3.2:latest"), "llama3.2:latest");
  assert.equal(parseModelId("Qwen/Qwen2.5-0.5B-Instruct"), "Qwen/Qwen2.5-0.5B-Instruct");
  assert.equal(parseModelId("  gemini-3.6-flash  "), "gemini-3.6-flash");
  assert.throws(() => parseModelId("no\npe"), /line breaks/);
  assert.throws(() => parseModelId("bad\rtype"), /line breaks/);
  assert.throws(() => parseModelId("x\u0000y"), /line breaks/);
  assert.throws(() => parseModelId(""), /required/);
  assert.throws(() => parseModelId("x".repeat(257)), /256 characters/);
  assert.throws(() => parseModelId("has space"), /unsupported/);
});

test("logo helpers reject path traversal and persist under the state dir", () => {
  const dir = mkdtempSync(join(tmpdir(), "orch-logo-"));
  const prev = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  process.env.AGENT_ORCHESTRATOR_STATE_DIR = dir;
  try {
    assert.equal(findLogo("../etc/passwd"), undefined);
    assert.equal(hasLogo("not a id"), false);
    assert.throws(() => saveLogo("../etc/passwd", PNG_1X1, "image/png"), /Invalid backend id/);
    saveLogo("vllm-local", PNG_1X1, "image/png");
    assert.equal(hasLogo("vllm-local"), true);
    const found = findLogo("vllm-local");
    assert.ok(found?.path.startsWith(dir));
    assert.equal(found?.path.includes(".."), false);
    const read = readLogo("vllm-local");
    assert.equal(read?.mime, "image/png");
    assert.ok(read && read.bytes.equals(PNG_1X1));
    const decoded = decodeLogoDataUrl(`data:image/png;base64,${PNG_1X1.toString("base64")}`);
    assert.equal(decoded.mime, "image/png");
    assert.throws(() => decodeLogoDataUrl("data:image/svg+xml;base64,PHN2Zz4="), /PNG, JPEG, or WebP/);
    const fakePng = `data:image/png;base64,${Buffer.from("<svg></svg>").toString("base64")}`;
    assert.throws(() => decodeLogoDataUrl(fakePng), /PNG, JPEG, or WebP/);
    assert.throws(() => saveLogo("vllm-local", Buffer.from("<svg></svg>"), "image/png"), /not SVG or script/);
    assert.equal(removeLogo("vllm-local"), true);
    assert.equal(hasLogo("vllm-local"), false);
  } finally {
    if (prev === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prev;
    rmSync(dir, { recursive: true, force: true });
  }
});
