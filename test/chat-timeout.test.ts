import assert from "node:assert/strict";
import { test } from "node:test";
import {
  DEFAULT_CURSOR_TIMEOUT_MS,
  DEFAULT_SPEAKER_TIMEOUT_MS,
  awaitOrTimeout,
  isCursorSpeaker,
  isSpeakerSkipError,
  LEFTOVER_SPEAKER_SKIP,
  looksLikeLateToolJson,
  raceTimeout,
  speakerSkipLine,
  speakerErrorText,
  speakerTimeoutMs,
  timeoutErrorMessage,
} from "../src/chat/timeout.js";

test("raceTimeout returns the real value when it finishes first", async () => {
  const value = await raceTimeout(Promise.resolve("ok"), 200, () => "skip");
  assert.equal(value, "ok");
});

test("raceTimeout skips a hanging speaker", async () => {
  const hanging = new Promise<string>((resolve) => {
    setTimeout(() => resolve("too late"), 80);
  });
  const value = await raceTimeout(hanging, 15, () => "skip");
  assert.equal(value, "skip");
  await new Promise((resolve) => setTimeout(resolve, 90));
});

test("awaitOrTimeout times out a hanging speaker", async () => {
  const hanging = new Promise<string>((resolve) => {
    setTimeout(() => resolve("too late"), 80);
  });
  await assert.rejects(() => awaitOrTimeout(hanging, 15, "Cursor timed out after 0s"), /timed out/);
  await new Promise((resolve) => setTimeout(resolve, 90));
});

test("awaitOrTimeout swallows a late rejection after skip", async () => {
  const hanging = new Promise<string>((_, reject) => {
    setTimeout(() => reject(new Error("cursor sdk late fail")), 25);
  });
  await assert.rejects(() => awaitOrTimeout(hanging, 10, "Cursor timed out after 0s"), /timed out/);
  await new Promise((resolve) => setTimeout(resolve, 50));
});

test("timeout copy names the skipped speaker", () => {
  assert.equal(DEFAULT_SPEAKER_TIMEOUT_MS, 25_000);
  assert.equal(DEFAULT_CURSOR_TIMEOUT_MS, 90_000);
  assert.match(timeoutErrorMessage("Gemma", 25_000), /Gemma timed out after 25s/);
  assert.equal(isSpeakerSkipError(timeoutErrorMessage("Cursor cloud", 25_000)), true);
  assert.equal(isSpeakerSkipError("Gemini rate-limited (429) — skipped so other speakers can finish."), true);
  assert.equal(isSpeakerSkipError("show version on aos-cx"), false);
  assert.equal(isSpeakerSkipError("Cursor run stream is no longer available — skipped so other speakers can finish."), true);
  assert.equal(isSpeakerSkipError(LEFTOVER_SPEAKER_SKIP), true);
  assert.equal(speakerSkipLine("Cursor cloud", timeoutErrorMessage("Cursor cloud", 25_000)), "Cursor cloud: timed out after 25s — skipped");
  assert.equal(speakerSkipLine("Flash", "OpenAI-compatible error 429: quota"), "Flash: rate-limited (429) — skipped");
  assert.equal(speakerSkipLine("Cursor cloud", "Run stream is no longer available"), "Cursor cloud: skipped (stream ended)");
  assert.equal(speakerSkipLine("Cursor cloud", "none"), "Cursor cloud: failed (no error detail)");
  assert.equal(speakerErrorText("Cursor cloud", "none"), "Cursor cloud failed (no error detail).");
  assert.equal(speakerErrorText("Cursor cloud", ""), "Cursor cloud failed (no error detail).");
  assert.equal(speakerErrorText("Cursor cloud", undefined), "Cursor cloud failed (no error detail).");
});

test("Cursor speakers get a longer default timeout than Gemini/vLLM", () => {
  const prev = process.env.AGENT_ORCHESTRATOR_SPEAKER_TIMEOUT_MS;
  const prevCursor = process.env.AGENT_ORCHESTRATOR_CURSOR_TIMEOUT_MS;
  delete process.env.AGENT_ORCHESTRATOR_SPEAKER_TIMEOUT_MS;
  delete process.env.AGENT_ORCHESTRATOR_CURSOR_TIMEOUT_MS;
  try {
    assert.equal(isCursorSpeaker("cursor-local"), true);
    assert.equal(isCursorSpeaker("cursor-cloud"), true);
    assert.equal(isCursorSpeaker("gemini"), false);
    assert.equal(speakerTimeoutMs("gemini"), DEFAULT_SPEAKER_TIMEOUT_MS);
    assert.equal(speakerTimeoutMs("vllm-local"), DEFAULT_SPEAKER_TIMEOUT_MS);
    assert.equal(speakerTimeoutMs("cursor-local"), DEFAULT_CURSOR_TIMEOUT_MS);
    assert.equal(speakerTimeoutMs("cursor-cloud"), DEFAULT_CURSOR_TIMEOUT_MS);
  } finally {
    if (prev === undefined) delete process.env.AGENT_ORCHESTRATOR_SPEAKER_TIMEOUT_MS;
    else process.env.AGENT_ORCHESTRATOR_SPEAKER_TIMEOUT_MS = prev;
    if (prevCursor === undefined) delete process.env.AGENT_ORCHESTRATOR_CURSOR_TIMEOUT_MS;
    else process.env.AGENT_ORCHESTRATOR_CURSOR_TIMEOUT_MS = prevCursor;
  }
});

test("Late propose_command JSON is recognized for early flush", () => {
  assert.equal(looksLikeLateToolJson('{"tool":"propose_command","command":"show version"}'), true);
  assert.equal(looksLikeLateToolJson("interface descriptions look empty"), false);
});
