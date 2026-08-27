import assert from "node:assert/strict";
import { test } from "node:test";
import { extractHttpText } from "../src/providers/util.js";
import {
  DEFAULT_GEMINI_MODEL,
  formatGeminiChatError,
  geminiModelChoices,
  isRetiredGeminiModel,
  KNOWN_GEMINI_MODELS,
  normalizeGeminiConfigModel,
  parseGeminiIdsFromListPayload,
  parseGeminiModelId,
  pickDefaultGeminiModel,
  resolveGeminiChatModel,
  stripGeminiModelPrefix,
  suggestGeminiModelFromError,
} from "../src/providers/gemini.js";

test("parseGeminiModelId strips repeated models/ prefixes", () => {
  assert.equal(parseGeminiModelId("gemini-3.6-flash"), "gemini-3.6-flash");
  assert.equal(parseGeminiModelId("models/gemini-3.6-flash"), "gemini-3.6-flash");
  assert.equal(parseGeminiModelId("models/models/gemini-3.6-flash"), "gemini-3.6-flash");
  assert.equal(stripGeminiModelPrefix("models/models/foo"), "foo");
  assert.equal(parseGeminiModelId("  models/gemini-3.5-flash-lite  "), "gemini-3.5-flash-lite");
});

test("normalizeGeminiConfigModel remaps retired 1.5/2.0/2.5 ids", () => {
  assert.equal(normalizeGeminiConfigModel("models/gemini-2.0-flash"), DEFAULT_GEMINI_MODEL);
  assert.equal(normalizeGeminiConfigModel("gemini-1.5-flash"), DEFAULT_GEMINI_MODEL);
  assert.equal(normalizeGeminiConfigModel("gemini-2.5-flash"), DEFAULT_GEMINI_MODEL);
  assert.equal(normalizeGeminiConfigModel("gemini-3.6-flash"), "gemini-3.6-flash");
  assert.equal(isRetiredGeminiModel("gemini-2.0-flash"), true);
  assert.equal(isRetiredGeminiModel("gemini-3.6-flash"), false);
});

test("suggestGeminiModelFromError parses Google 404 replacement text", () => {
  const flash404 =
    "This model models/gemini-2.0-flash is no longer available. Please update your code to use models/gemini-3.6-flash for the latest features and improvements. We recommend you to use the Interactions API.";
  assert.equal(suggestGeminiModelFromError(flash404, "gemini-2.0-flash"), "gemini-3.6-flash");
  const missing =
    "models/gemini-1.5-flash is not found for API version v1main, or is not supported for generateContent. Call ModelService.ListModels to see the list of available models and their supported methods.";
  assert.equal(suggestGeminiModelFromError(missing, "gemini-1.5-flash"), DEFAULT_GEMINI_MODEL);
  const formatted = formatGeminiChatError(404, flash404, "gemini-2.0-flash");
  assert.match(formatted, /for model gemini-2\.0-flash/);
  assert.match(formatted, /Google suggests gemini-3\.6-flash/);
  assert.equal(formatted.includes("models/models/"), false);
});

test("extractHttpText reads Google OpenAI-compat error arrays", () => {
  const payload = [
    {
      error: {
        code: 404,
        message:
          "This model models/gemini-2.0-flash is no longer available. Please update your code to use models/gemini-3.6-flash for the latest features and improvements.",
        status: "NOT_FOUND",
      },
    },
  ];
  assert.match(extractHttpText(payload), /update your code to use models\/gemini-3\.6-flash/);
});

test("2026 catalog omits retired 1.5/2.0 defaults and prefers ListModels 3.6-flash", () => {
  const catalog = geminiModelChoices();
  assert.equal(catalog[0], DEFAULT_GEMINI_MODEL);
  assert.equal(catalog.includes("gemini-1.5-flash"), false);
  assert.equal(catalog.includes("gemini-2.0-flash"), false);
  assert.ok(KNOWN_GEMINI_MODELS.includes("gemini-3.6-flash"));
  const listed = parseGeminiIdsFromListPayload({
    data: [
      { id: "models/gemini-2.5-flash" },
      { id: "models/gemini-3.7-flash" },
      { id: "models/gemini-3.6-flash" },
      { id: "models/gemini-2.0-flash" },
      { id: "models/gemini-3.5-flash-image" },
    ],
  });
  assert.equal(pickDefaultGeminiModel(listed), "gemini-3.6-flash");
  const live = geminiModelChoices(listed);
  assert.equal(live.includes("gemini-3.6-flash"), true);
  assert.equal(live.includes("gemini-3.7-flash"), true);
  assert.equal(live.includes("gemini-2.5-flash"), false);
  assert.equal(live.includes("gemini-2.0-flash"), false);
  assert.equal(live.includes("gemini-3.5-flash-image"), false);
});

test("resolveGeminiChatModel ignores composer override and remaps retired ids", () => {
  assert.equal(resolveGeminiChatModel("gemini-2.0-flash", "composer-2.5"), DEFAULT_GEMINI_MODEL);
  assert.equal(resolveGeminiChatModel("models/gemini-3.6-flash"), "gemini-3.6-flash");
  assert.equal(
    resolveGeminiChatModel("gemini-1.5-flash", undefined, ["gemini-3.7-flash", "gemini-3.6-flash"]),
    "gemini-3.6-flash",
  );
  assert.equal(resolveGeminiChatModel("gemini-3.5-flash", undefined, ["gemini-3.5-flash", "gemini-3.6-flash"]), "gemini-3.5-flash");
});
