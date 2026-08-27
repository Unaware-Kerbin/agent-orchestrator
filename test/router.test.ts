import assert from "node:assert/strict";
import { test } from "node:test";
import {
  DEFAULT_ROUNDS,
  detectControl,
  detectIntent,
  detectNamedBackend,
  detectVisual3dIntent,
  extractFilesystemPaths,
  routeChat,
  wantsHostInstall,
} from "../src/chat/router.js";
import type { RouterBackend, RouterContext, RouterSpecialist } from "../src/chat/types.js";

const SPECIALISTS: RouterSpecialist[] = [
  { id: "planner", backend: "anthropic" },
  { id: "builder", backend: "cursor-local" },
  { id: "reviewer", backend: "openai" },
  { id: "pr-triage", backend: "cursor-local" },
  { id: "gemini-planner", backend: "gemini" },
  { id: "procedural-3d-artist", backend: "gemini" },
  { id: "procedural-3d-local", backend: "vllm-mistral-7b-instruct" },
  { id: "vllm-mistral-7b-instruct", backend: "vllm-mistral-7b-instruct" },
  { id: "vllm-chat", backend: "vllm-local" },
  { id: "cloud-builder", backend: "cursor-cloud" },
];

function backend(id: string, extra: Partial<RouterBackend> = {}): RouterBackend {
  const type =
    extra.type ??
    (id.startsWith("vllm") ? "vllm" : id.startsWith("cursor") ? "cursor" : id === "anthropic" ? "anthropic" : "openai");
  return {
    id,
    type,
    ready: extra.ready ?? true,
    writesLocalFiles: extra.writesLocalFiles ?? id === "cursor-local",
    runtime: extra.runtime ?? (id === "cursor-cloud" ? "cloud" : id === "cursor-local" ? "local" : undefined),
    model: extra.model,
    reason: extra.reason,
  };
}

function ctx(partial: Partial<RouterContext> & Pick<RouterContext, "message">): RouterContext {
  return {
    specialists: SPECIALISTS,
    backends: partial.backends ?? [],
    pin: partial.pin ?? "auto",
    vllmRunning: partial.vllmRunning,
    vllmModelId: partial.vllmModelId,
    prior: partial.prior,
    followUp: partial.followUp,
    message: partial.message,
    allowedDirectories: partial.allowedDirectories,
    workspace: partial.workspace,
  };
}

test("hardware / Arc GPU questions are control, not a chat model", () => {
  assert.equal(detectIntent("what models fit my Arc GPUs?"), "control");
  assert.equal(detectControl("what models fit my Arc GPUs?"), "hardware");
  const decision = routeChat(
    ctx({
      message: "what models fit my Arc GPUs?",
      backends: [backend("vllm-local"), backend("gemini"), backend("cursor-local")],
      vllmRunning: true,
    }),
  );
  assert.equal(decision.kind, "control");
  assert.equal(decision.control, "hardware");
  assert.equal(decision.chip, "orchestrator");
});

test("start vLLM is a control action", () => {
  const decision = routeChat(ctx({ message: "start the recommended local model", backends: [backend("vllm-local", { ready: false })] }));
  assert.equal(decision.kind, "control");
  assert.equal(decision.control, "start_vllm");
});

test("allowlist questions stay on control tools", () => {
  assert.equal(detectIntent("show the write allowlist"), "control");
  const decision = routeChat(ctx({ message: "show the write allowlist", backends: [backend("gemini")] }));
  assert.equal(decision.kind, "control");
  assert.equal(decision.control, "allowlist");
});

test("code/fix/PR with two or more ready backends is a round-table debate", () => {
  const decision = routeChat(
    ctx({
      message: "troubleshoot this PR and propose a fix",
      backends: [
        backend("vllm-local", { model: "Qwen/Qwen2.5-14B-Instruct" }),
        backend("gemini"),
        backend("cursor-local"),
      ],
      vllmRunning: true,
      vllmModelId: "Qwen/Qwen2.5-14B-Instruct",
    }),
  );
  assert.equal(decision.kind, "debate");
  assert.equal(decision.rounds, DEFAULT_ROUNDS);
  assert.ok((decision.speakers?.length ?? 0) >= 2);
  assert.ok(decision.speakers?.some((s) => s.backendId === "vllm-local"));
  assert.ok(decision.speakers?.some((s) => s.backendId === "gemini"));
  assert.equal(decision.closer?.backendId, "cursor-local");
  assert.equal(decision.chip, "Debate");
  for (const speaker of decision.speakers ?? []) {
    assert.equal(speaker.label.includes(" · "), false, "speaker labels must not merge model names");
  }
});

test("cloud-with-local-draft shape becomes bounce debate, not a one-way workflow", () => {
  const decision = routeChat(
    ctx({
      message: "implement the login rate limiter",
      backends: [backend("vllm-local"), backend("cursor-cloud", { writesLocalFiles: false })],
      vllmRunning: true,
    }),
  );
  assert.equal(decision.kind, "debate");
  assert.ok(decision.speakers?.some((s) => s.backendId === "vllm-local"));
  assert.ok(decision.speakers?.some((s) => s.backendId === "cursor-cloud"));
  assert.equal(decision.closer?.backendId, "cursor-cloud");
});

test("pin skips debate even when several backends are ready", () => {
  const decision = routeChat(
    ctx({
      message: "draft a plan for the cache layer",
      pin: "gemini",
      backends: [backend("vllm-local"), backend("gemini"), backend("cursor-local")],
      vllmRunning: true,
    }),
  );
  assert.equal(decision.kind, "single");
  assert.equal(decision.speakers?.[0]?.backendId, "gemini");
  assert.equal(decision.rounds, undefined);
});

test("explicitly naming a backend honors it", () => {
  assert.equal(detectNamedBackend("use gemini to draft a plan"), "gemini");
  const decision = routeChat(
    ctx({
      message: "use gemini to draft a plan",
      backends: [backend("vllm-local"), backend("gemini"), backend("cursor-local")],
      vllmRunning: true,
    }),
  );
  assert.equal(decision.kind, "single");
  assert.equal(decision.speakers?.[0]?.backendId, "gemini");
});

test("only one ready backend is single-agent chat", () => {
  const decision = routeChat(
    ctx({
      message: "review this diff for merge readiness",
      backends: [backend("gemini"), backend("vllm-local", { ready: false }), backend("cursor-local", { ready: false })],
      vllmRunning: false,
    }),
  );
  assert.equal(decision.kind, "single");
  assert.equal(decision.speakers?.[0]?.backendId, "gemini");
});

test("general Q&A prefers ready local vLLM, then Gemini, then Cursor", () => {
  const withVllm = routeChat(
    ctx({
      message: "what is MCP?",
      backends: [backend("vllm-local"), backend("gemini"), backend("cursor-local")],
      vllmRunning: true,
    }),
  );
  assert.equal(withVllm.kind, "single");
  assert.equal(withVllm.speakers?.[0]?.backendId, "vllm-local");

  const withGemini = routeChat(
    ctx({
      message: "what is MCP?",
      backends: [backend("vllm-local", { ready: false }), backend("gemini"), backend("cursor-local")],
      vllmRunning: false,
    }),
  );
  assert.equal(withGemini.speakers?.[0]?.backendId, "gemini");
});

test("plan/review with two text backends debates; closer is last speaker when Cursor cannot write", () => {
  const decision = routeChat(
    ctx({
      message: "draft a plan for migrating off Redux",
      backends: [backend("vllm-local"), backend("gemini")],
      vllmRunning: true,
    }),
  );
  assert.equal(decision.kind, "debate");
  assert.equal(decision.intent, "reason");
  assert.ok(decision.closer);
  assert.notEqual(decision.closer?.backendId, "cursor-local");
});

test("pin local when vLLM is down returns a start action", () => {
  const decision = routeChat(
    ctx({
      message: "summarize src/server.ts",
      pin: "local",
      backends: [backend("vllm-local", { ready: false, reason: "vLLM not running at http://127.0.0.1:8000/v1" })],
      vllmRunning: false,
    }),
  );
  assert.equal(decision.kind, "error");
  assert.equal(decision.suggestedAction?.action, "start_vllm");
});

test("follow-up debate uses a single extra round", () => {
  const decision = routeChat(
    ctx({
      message: "implement the safer variant we discussed",
      followUp: true,
      backends: [backend("vllm-local"), backend("gemini"), backend("cursor-local")],
      vllmRunning: true,
    }),
  );
  assert.equal(decision.kind, "debate");
  assert.equal(decision.rounds, 1);
});

test("code with only Cursor ready is a single builder, not a debate", () => {
  const decision = routeChat(
    ctx({
      message: "implement rate limiting on the login route",
      backends: [backend("cursor-local"), backend("gemini", { ready: false })],
    }),
  );
  assert.equal(decision.kind, "single");
  assert.equal(decision.speakers?.[0]?.specialist, "builder");
});

test("build + allowlisted path uses Cursor write closer, not vLLM-only", () => {
  const path = "/tmp/example-app";
  const decision = routeChat(
    ctx({
      message: `I want you to build a stock trading bot here ${path}`,
      backends: [
        backend("vllm-local", { model: "Qwen/Qwen2.5-0.5B-Instruct" }),
        backend("vllm-0.25b", { type: "vllm", model: "Qwen/Qwen2.5-0.25B-Instruct" }),
        backend("cursor-local"),
      ],
      vllmRunning: true,
      vllmModelId: "Qwen/Qwen2.5-0.5B-Instruct",
      allowedDirectories: [path],
    }),
  );
  assert.equal(detectIntent(`I want you to build a stock trading bot here ${path}`), "code");
  assert.deepEqual(extractFilesystemPaths(`here ${path}`), [path]);
  assert.equal(decision.kind, "debate");
  assert.equal(decision.needsWrites, true);
  assert.equal(decision.cwd, path);
  assert.equal(decision.closer?.backendId, "cursor-local");
  assert.equal(decision.closer?.writesLocalFiles, true);
  assert.equal(decision.chip, "Debate");
  const labels = (decision.speakers ?? []).map((s) => s.label);
  assert.equal(new Set(labels).size, labels.length);
  assert.ok(!labels.some((l) => l.includes(" · ")));
  assert.ok(decision.speakers?.some((s) => s.backendId === "cursor-local") || decision.closer?.backendId === "cursor-local");
});

test("build + path not on allowlist prompts add_allowed_dir", () => {
  const path = "/tmp/example-app";
  const decision = routeChat(
    ctx({
      message: `build a stock trading bot here ${path}`,
      backends: [backend("vllm-local"), backend("cursor-local")],
      vllmRunning: true,
      allowedDirectories: ["/tmp/orchestrator-workspace"],
    }),
  );
  assert.equal(decision.kind, "error");
  assert.equal(decision.suggestedAction?.action, "add_allowed_dir");
  assert.equal(decision.suggestedAction?.payload?.path, path);
  assert.match(decision.error ?? "", /allowlist/i);
});

test("install Unity for me needs approval and is not Q&A", () => {
  const message = "Are you guys able to install unity for me and set this all up?";
  assert.equal(detectIntent(message), "code");
  assert.equal(wantsHostInstall(message), true);
  const decision = routeChat(
    ctx({
      message,
      pin: "single",
      backends: [backend("cursor-local"), backend("gemini")],
    }),
  );
  assert.equal(decision.kind, "single");
  assert.equal(decision.needsApproval, true);
  assert.equal(decision.needsHostInstall, true);
  assert.equal(decision.speakers?.[0]?.backendId, "cursor-local");
});

test("how do I install Unity stays Q&A without approval", () => {
  const message = "how do I install unity on Linux?";
  assert.equal(detectIntent(message), "general");
  assert.equal(wantsHostInstall(message), false);
  const decision = routeChat(
    ctx({
      message,
      backends: [backend("vllm-local"), backend("cursor-local")],
      vllmRunning: true,
    }),
  );
  assert.equal(decision.needsApproval, false);
  assert.equal(decision.needsWrites, false);
});

test("Q&A prefers local vLLM and does not require Cursor writes", () => {
  const decision = routeChat(
    ctx({
      message: "what is a stock trading bot?",
      backends: [backend("vllm-local"), backend("cursor-local")],
      vllmRunning: true,
    }),
  );
  assert.equal(decision.kind, "single");
  assert.equal(decision.intent, "general");
  assert.equal(decision.speakers?.[0]?.backendId, "vllm-local");
  assert.equal(decision.needsWrites, false);
});

test("Debate pin forces round-table even for general Q&A", () => {
  const decision = routeChat(
    ctx({
      message: "what is MCP?",
      pin: "debate",
      backends: [backend("vllm-local"), backend("gemini"), backend("cursor-local")],
      vllmRunning: true,
    }),
  );
  assert.equal(decision.kind, "debate");
  assert.ok((decision.speakers?.length ?? 0) >= 2);
  assert.equal(decision.chip, "Debate");
  assert.notEqual(decision.closer?.backendId, undefined);
});

test("Single pin skips debate on a plan", () => {
  const decision = routeChat(
    ctx({
      message: "draft a plan for the cache layer",
      pin: "single",
      backends: [backend("vllm-local"), backend("gemini"), backend("cursor-local")],
      vllmRunning: true,
    }),
  );
  assert.equal(decision.kind, "single");
  assert.equal(decision.speakers?.length, 1);
});

test("build with two tiny vLLMs and no Cursor asks for CURSOR_API_KEY", () => {
  const decision = routeChat(
    ctx({
      message: "build a stock trading bot here /tmp/demo-app",
      backends: [
        backend("vllm-local", { model: "Qwen/Qwen2.5-0.5B-Instruct" }),
        backend("vllm-tiny", { type: "vllm", model: "Qwen/Qwen2.5-0.25B-Instruct" }),
      ],
      vllmRunning: true,
      allowedDirectories: ["/tmp/demo-app"],
    }),
  );
  assert.equal(decision.kind, "error");
  assert.match(decision.error ?? "", /CURSOR_API_KEY/);
  assert.equal(decision.suggestedAction?.action, "open_settings");
});

test("debate speaker labels stay unmerged", () => {
  const decision = routeChat(
    ctx({
      message: "review this diff for merge readiness",
      backends: [
        backend("vllm-local", { model: "Qwen/Qwen2.5-0.5B-Instruct" }),
        backend("gemini"),
      ],
      vllmRunning: true,
      vllmModelId: "Qwen/Qwen2.5-0.5B-Instruct",
    }),
  );
  assert.equal(decision.kind, "debate");
  for (const speaker of [...(decision.speakers ?? []), decision.closer].filter(Boolean)) {
    assert.equal(speaker!.label.includes(" · "), false);
  }
  assert.equal(decision.chip, "Debate");
});

test("3D art questions route the procedural-3d-artist specialist in debate", () => {
  assert.equal(detectVisual3dIntent("create all the 3D renders for ships and planets"), true);
  assert.equal(detectIntent("create procedural mesh factories for corvettes"), "code");

  const decision = routeChat(
    ctx({
      message: "create procedural 3D ship meshes and export OBJ files",
      backends: [
        backend("vllm-mistral-7b-instruct", { model: "mistralai/Mistral-7B-Instruct-v0.3" }),
        backend("gemini"),
        backend("cursor-local"),
      ],
      vllmRunning: true,
      allowedDirectories: ["/tmp/example-app"],
    }),
  );
  assert.equal(decision.kind, "debate");
  assert.ok(decision.speakers?.some((s) => s.specialist === "procedural-3d-artist"));
  assert.ok(decision.speakers?.some((s) => s.specialist === "procedural-3d-local"));
  assert.equal(decision.closer?.backendId, "cursor-local");
  for (const speaker of decision.speakers ?? []) {
    assert.equal(speaker.label.includes(" · "), false);
  }
});

test("3D Q&A without writes prefers procedural-3d-artist on Gemini", () => {
  const decision = routeChat(
    ctx({
      message: "Are there procedural shaders and textures in the game?",
      backends: [backend("gemini"), backend("vllm-mistral-7b-instruct"), backend("cursor-local")],
      vllmRunning: true,
    }),
  );
  assert.equal(decision.intent, "reason");
  assert.equal(decision.kind, "debate");
  assert.ok(decision.speakers?.some((s) => s.specialist === "procedural-3d-artist"));
});
