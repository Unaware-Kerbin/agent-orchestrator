import assert from "node:assert/strict";
import test from "node:test";
import { estimateCompletionTokens, estimateTokensPerSec } from "../src/chat/service.js";

test("estimateCompletionTokens uses chars/4 research heuristic", () => {
  assert.equal(estimateCompletionTokens(""), 0);
  assert.equal(estimateCompletionTokens("abcd"), 1);
  assert.equal(estimateCompletionTokens("a".repeat(40)), 10);
});

test("estimateTokensPerSec divides by elapsed seconds", () => {
  const started = 1_000;
  const finished = 2_000; // 1s
  const tps = estimateTokensPerSec("a".repeat(40), started, finished);
  assert.equal(tps, 10);
  assert.equal(estimateTokensPerSec("hi", undefined, finished), undefined);
});
