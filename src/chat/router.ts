import { homedir } from "node:os";
import { isAbsolute, join, resolve } from "node:path";
import { isPathInside } from "../allowlist.js";
import { backendLogoUrl } from "../identity.js";
import { wantsHostInstall } from "./approval.js";
import type {
  ChatIntent,
  ChatSuggestedAction,
  ControlKind,
  RouteDecision,
  RouteSpeaker,
  RouterBackend,
  RouterContext,
  RouterSpecialist,
  WorkspaceHint,
} from "./types.js";

export { wantsHostInstall } from "./approval.js";

export const DEFAULT_ROUNDS = 2;
export const MAX_ROUNDS = 3;

const DEBATE_INTENTS = new Set<ChatIntent>(["code", "review", "reason"]);

/**
 * Late (and similar MCP clients) wrap the operator turn after SYSTEM + untrusted
 * device output. Intent/control must use that turn — the wrapper mentions
 * "allowlist" / vLLM tools and must not become the chat reply.
 */
/** Preamble only — not enough to route. Missing END must fail closed. */
export function lateWrapHasPreamble(message: string): boolean {
  return /^SYSTEM:/m.test(message) && /UNTRUSTED DEVICE OUTPUT follows/i.test(message);
}

export function lateWrapMissingEnd(message: string): boolean {
  return lateWrapHasPreamble(message) && !/END UNTRUSTED DEVICE OUTPUT/i.test(message);
}

/** Complete Late wrap: SYSTEM + untrusted header + END fence. */
export function isLateDeviceWrap(message: string): boolean {
  return lateWrapHasPreamble(message) && /END UNTRUSTED DEVICE OUTPUT/i.test(message);
}

export function extractRoutableMessage(message: string): string {
  const text = message.trim();
  if (!text) return text;
  if (lateWrapMissingEnd(text)) return "";
  if (!isLateDeviceWrap(text)) return text;

  const endRe = /END UNTRUSTED DEVICE OUTPUT[^\n]*/gi;
  let last = -1;
  let match: RegExpExecArray | null;
  while ((match = endRe.exec(text))) {
    last = match.index + match[0].length;
  }
  if (last < 0) return "";
  return text.slice(last).trim();
}

/** Device scrollback between BEGIN/END. Empty when the wrap is incomplete. */
export function extractUntrustedDeviceOutput(message: string): string {
  if (!isLateDeviceWrap(message)) return "";
  const begin = message.match(/BEGIN UNTRUSTED DEVICE OUTPUT[^\n]*/i);
  const end = message.match(/END UNTRUSTED DEVICE OUTPUT[^\n]*/i);
  if (!begin || !end || begin.index === undefined || end.index === undefined) return "";
  const start = begin.index + begin[0].length;
  if (end.index <= start) return "";
  return message.slice(start, end.index).trim();
}

export function detectIntent(message: string): ChatIntent {
  const text = message.trim();
  if (!text) return "general";

  if (
    /\b(allowlist|allowed director(?:y|ies)|grant (?:a )?director)/i.test(text) ||
    /\b(stop vllm|vllm status|is vllm (?:running|up))\b/i.test(text) ||
    /\b(start (?:the )?(?:recommended )?(?:local )?(?:model|vllm|server)|start vllm)\b/i.test(text) ||
    /\bdownload (?:the )?(?:local |recommended )?model\b/i.test(text) ||
    /\b(what models? fit|recommend(?:ed)? (?:local )?models?|list (?:local )?models)\b/i.test(text) ||
    /\b(hardware|vram|\bgpus?\b|accelerators?|intel (?:arc|xpu)|nvidia|rocm|what (?:gpu|models) (?:do i|have))\b/i.test(
      text,
    )
  ) {
    return "control";
  }

  if (wantsHostInstall(text)) {
    return "code";
  }

  if (
    /\b(implement|patch|refactor|write (?:the )?code|edit (?:the )?(?:file|repo)|apply (?:the )?fix|ship (?:this|it|a feature)|open a pr|pull request|\bprs?\b|fix this (?:bug|pr|failing)|add tests?|commit )\b/i.test(
      text,
    ) ||
    /\b(?:write|create|generate|draft) (?:a |the )?(?:cli |ansible )?playbook\b/i.test(text) ||
    /\b(build|scaffold)\b/i.test(text) ||
    /\bcreate (?:a |the )?(?:file|readme|project|app|bot|dir(?:ectory)?|folder)\b/i.test(text) ||
    /\bcreate \S+\.\w{1,8}\b/i.test(text) ||
    (/\b(here|in|into|at)\s+(?:~|\/|[A-Za-z]:[\\/])/i.test(text) &&
      /\b(build|create|write|implement|put|scaffold|generate)\b/i.test(text)) ||
    (detectVisual3dIntent(text) &&
      /\b(create|generate|export|implement|build|add|make|render)\b/i.test(text))
  ) {
    return "code";
  }

  if (/\b(review|critique|merge[- ]ready|look at (?:this )?diff|code review)\b/i.test(text)) {
    return "review";
  }

  if (
    /\b(plan|draft|design|architect|outline|troubleshoot|how should we|what(?:'s| is) the (?:best|right) (?:approach|fix))\b/i.test(
      text,
    ) ||
    detectVisual3dIntent(text)
  ) {
    return "reason";
  }

  return "general";
}

/** Procedural 3D art, meshes, shaders, textures-at-scale — routes the 3D specialist into round-table. */
export function detectVisual3dIntent(message: string): boolean {
  return /\b(3d|three[- ]d|mesh(?:es| factory)?|shader|shaders|texture|textures|render(?:s|ing)?|\.obj\b|gltf|glb|fbx|ur[ph]|procedural (?:3d|mesh|art|graphics|render)|visual metadata|icosphere|hardpoint|wavefront|starmap overlay|planet procedural|energy shield|atmosphere scatter|graphics pipeline|3d art|3d asset)\b/i.test(
    message,
  );
}

const LOCAL_FAMILY_NEEDLES: Array<{ re: RegExp; needle: string }> = [
  { re: /\bgemma\b/, needle: "gemma" },
  { re: /\bqwen\b/, needle: "qwen" },
  { re: /\bmistral\b/, needle: "mistral" },
  { re: /\bphi-?4\b/, needle: "phi" },
  { re: /\bolmo\b/, needle: "olmo" },
  { re: /\bgranite\b/, needle: "granite" },
  { re: /\bdeepseek\b/, needle: "deepseek" },
  { re: /\bllama\b/, needle: "llama" },
];

function backendsMatchingFamily(backends: RouterBackend[], needle: string): RouterBackend[] {
  return backends.filter((b) => `${b.id} ${b.model ?? ""}`.toLowerCase().includes(needle));
}

export function detectNamedBackend(message: string, backends: RouterBackend[] = []): string | undefined {
  const text = message.toLowerCase();
  if (/\b(cursor cloud|cloud cursor|cloud[- ]builder)\b/.test(text)) return "cursor-cloud";
  if (/\b(cursor local|local cursor)\b/.test(text)) return "cursor-local";
  if (/\b(llama\.cpp|llamacpp|llama-server|llama cpp)\b/.test(text) && !/\bcursor\b/.test(text)) return "llamacpp";
  if (/\bollama\b/.test(text)) return "ollama";
  const families = LOCAL_FAMILY_NEEDLES.filter((row) => row.re.test(text));
  if (families.length === 1 && backends.length) {
    const matches = backendsMatchingFamily(backends, families[0]!.needle);
    const pick = matches.find((b) => b.ready) ?? matches[0];
    if (pick) return pick.id;
  }
  if (families.length > 1 && backends.length) {
    // Two named locals (e.g. Qwen + Gemma): do not pin the first YAML row.
    return undefined;
  }
  if (
    (/\b(vllm|local vllm|local model)\b/.test(text) || families.length > 0) &&
    !/\bcursor\b/.test(text)
  ) {
    return "local";
  }
  if (/\b(gemini|google gemini)\b/.test(text)) return "gemini";
  if (/\b(anthropic|claude)\b/.test(text)) return "anthropic";
  if (/\b(openai|gpt-4|gpt4)\b/.test(text)) return "openai";
  return undefined;
}

export function detectControl(message: string): ControlKind | undefined {
  const text = message.trim();
  if (/\b(allowlist|allowed director)/i.test(text)) return "allowlist";
  if (/\bstop vllm\b/i.test(text)) return "stop_vllm";
  if (/\b(start (?:the )?(?:recommended )?(?:local )?(?:model|vllm|server)|start vllm)\b/i.test(text)) {
    return "start_vllm";
  }
  if (/\b(vllm status|is vllm (?:running|up))\b/i.test(text)) return "vllm_status";
  if (/\bdownload (?:the )?(?:local |recommended )?model\b/i.test(text) || /\blist (?:local )?models\b/i.test(text)) {
    return "models";
  }
  if (
    /\b(hardware|vram|\bgpus?\b|accelerators?|intel (?:arc|xpu)|nvidia|rocm|what models? fit|recommend(?:ed)? (?:local )?models?)\b/i.test(
      text,
    )
  ) {
    return "hardware";
  }
  return undefined;
}

const MODE_PINS = new Set(["auto", "debate", "single"]);

/** Absolute or home-relative filesystem paths mentioned in a chat message. */
export function extractFilesystemPaths(message: string): string[] {
  const found: string[] = [];
  const patterns = [
    /(?:^|[\s"'`=(])((?:~|\/)[^\s"'`<>|]+)/g,
    /(?:^|[\s"'`=(])([A-Za-z]:[\\/][^\s"'`<>|]+)/g,
    /(?:^|[\s"'`=(])(\\\\[^\s"'`<>|]+)/g,
  ];
  for (const re of patterns) {
    let match: RegExpExecArray | null;
    while ((match = re.exec(message))) {
      let raw = match[1] ?? "";
      raw = raw.replace(/[.,;:!?)]+$/g, "");
      if (raw.length < 4) continue;
      if (raw.startsWith("//") && !raw.startsWith("\\\\")) continue;
      found.push(raw);
    }
  }
  return [...new Set(found)];
}

export function expandUserPath(input: string): string {
  if (input === "~") return homedir();
  if (input.startsWith("~/")) return join(homedir(), input.slice(2));
  return input;
}

function directoryAllowed(target: string, allowed: string[]): boolean {
  const resolved = isAbsolute(target) ? resolve(target) : resolve(expandUserPath(target));
  return allowed.some((dir) => {
    const root = resolve(dir);
    return resolved === root || isPathInside(resolved, root);
  });
}

export function resolveWorkspaceHint(ctx: RouterContext): WorkspaceHint | undefined {
  if (ctx.workspace) return ctx.workspace;
  const raw = extractFilesystemPaths(ctx.message)[0];
  if (!raw) return undefined;
  const expanded = expandUserPath(raw);
  const allowed = directoryAllowed(expanded, ctx.allowedDirectories ?? []);
  return {
    path: raw,
    allowed,
    cwd: allowed ? resolve(expanded) : undefined,
  };
}

export function wantsFileWrites(intent: ChatIntent, message: string): boolean {
  if (wantsHostInstall(message)) return true;
  if (intent !== "code") return false;
  const planOnly =
    /\b(just |only )?(?:a )?(?:plan|outline|draft a plan)\b/i.test(message) &&
    !/\b(implement|build|scaffold|apply|write files?|create \S+\.\w)/i.test(message);
  return !planOnly;
}

export function shortModelName(modelId?: string): string {
  if (!modelId) return "vLLM";
  const tail = modelId.split("/").pop() ?? modelId;
  return tail.replace(/-Instruct$/i, "").replace(/-/g, " ");
}

export function speakerLabel(backend: RouterBackend, vllmModelId?: string, specialistId?: string): string {
  if (backend.nickname?.trim()) return backend.nickname.trim();
  if (specialistId === "procedural-3d-artist") return "Procedural 3D (Gemini)";
  if (specialistId === "procedural-3d-local") {
    const model = backend.model || vllmModelId;
    return `Procedural 3D (${shortModelName(model)} local)`;
  }
  if (backend.type === "vllm" || backend.id.startsWith("vllm")) {
    const model = backend.model || vllmModelId;
    return `${shortModelName(model)} local`;
  }
  if (isOllamaBackend(backend)) {
    return backend.model ? `${shortModelName(backend.model)} (Ollama)` : "Ollama";
  }
  if (isLlamaCppBackend(backend)) {
    return backend.model ? `${shortModelName(backend.model)} (llama.cpp)` : "llama.cpp";
  }
  if (backend.id === "gemini" || /gemini/i.test(backend.id)) return "Gemini";
  if (backend.id === "cursor-cloud" || backend.runtime === "cloud") return "Cursor cloud";
  if (backend.id === "cursor-local" || (backend.type === "cursor" && backend.runtime === "local")) {
    return "Cursor local";
  }
  if (backend.id === "anthropic") return "Anthropic";
  if (backend.id === "openai") return "OpenAI";
  return backend.id;
}

function specialistFor(
  backendId: string,
  intent: ChatIntent,
  specialists: RouterSpecialist[],
  visual3d: boolean,
): string {
  const matching = specialists.filter((s) => s.backend === backendId);
  const pick = (ids: string[]) => {
    for (const id of ids) {
      if (matching.some((s) => s.id === id)) return id;
    }
    return undefined;
  };

  if (visual3d) {
    if (backendId === "gemini") {
      return pick(["procedural-3d-artist", "gemini-planner"]) ?? "procedural-3d-artist";
    }
    if (backendId.startsWith("vllm") || backendId === "local") {
      return (
        pick(["procedural-3d-local", "vllm-mistral-7b-instruct", "vllm-chat"]) ??
        matching[0]?.id ??
        "procedural-3d-local"
      );
    }
  }

  if (backendId === "cursor-cloud") return pick(["cloud-builder"]) ?? matching[0]?.id ?? "cloud-builder";
  if (backendId === "cursor-local" || backendId === "cursor") {
    if (intent === "code") return pick(["builder", "pr-triage"]) ?? "builder";
    if (intent === "review") return pick(["reviewer", "builder"]) ?? "builder";
    return pick(["planner", "builder"]) ?? "builder";
  }
  if (backendId.startsWith("vllm") || backendId === "local") {
    return matching[0]?.id ?? pick(["vllm-chat"]) ?? matching[0]?.id ?? "vllm-chat";
  }
  if (isOllamaId(backendId)) {
    return matching[0]?.id ?? pick(["ollama-chat"]) ?? matching[0]?.id ?? "ollama-chat";
  }
  if (isLlamaCppId(backendId)) {
    return matching[0]?.id ?? pick(["llamacpp-chat"]) ?? matching[0]?.id ?? "llamacpp-chat";
  }
  if (backendId === "gemini") return pick(["gemini-planner"]) ?? matching[0]?.id ?? "gemini-planner";
  if (intent === "review") return pick(["reviewer"]) ?? matching[0]?.id ?? backendId;
  if (intent === "reason" || intent === "general") {
    return pick(["planner", "gemini-planner", "vllm-chat"]) ?? matching[0]?.id ?? backendId;
  }
  return matching[0]?.id ?? backendId;
}

function findBackend(backends: RouterBackend[], id: string): RouterBackend | undefined {
  return backends.find((b) => b.id === id);
}

function vllmBackend(backends: RouterBackend[]): RouterBackend | undefined {
  return backends.find((b) => isVllmBackend(b) && b.ready) ?? backends.find((b) => isVllmBackend(b));
}

function isOllamaId(id: string): boolean {
  return id === "ollama" || id.startsWith("ollama");
}

function isLlamaCppId(id: string): boolean {
  return (
    id === "llamacpp" ||
    id === "llama-server" ||
    id.startsWith("llamacpp") ||
    /^llama[-.]?cpp/i.test(id)
  );
}

function isOllamaBackend(backend: RouterBackend): boolean {
  return backend.type === "ollama" || isOllamaId(backend.id);
}

function isLlamaCppBackend(backend: RouterBackend): boolean {
  return backend.type === "llamacpp" || isLlamaCppId(backend.id);
}

function ollamaBackend(backends: RouterBackend[]): RouterBackend | undefined {
  return backends.find(isOllamaBackend);
}

function llamaCppBackend(backends: RouterBackend[]): RouterBackend | undefined {
  return backends.find(isLlamaCppBackend);
}

function resolvePinTarget(
  pin: string,
  backends: RouterBackend[],
  _vllmRunning: boolean,
): { backend?: RouterBackend; error?: string; suggestedAction?: ChatSuggestedAction } {
  const normalized = pin.trim().toLowerCase();
  if (normalized === "local") {
    const vllm = vllmBackend(backends);
    if (vllm?.ready) return { backend: vllm };
    return {
      error: "Local vLLM is not running.",
      suggestedAction: {
        label: "Start recommended local model",
        action: "start_vllm",
      },
      backend: undefined,
    };
  }
  if (normalized === "cloud") {
    const cloud = findBackend(backends, "cursor-cloud");
    if (cloud?.ready) return { backend: cloud };
    return {
      error: "Cursor cloud is not ready. Set CURSOR_API_KEY in Settings → Backends.",
      suggestedAction: { label: "Open backends", action: "open_settings", payload: { page: "backends" } },
    };
  }
  if (normalized === "ollama") {
    const found = ollamaBackend(backends) ?? findBackend(backends, "ollama");
    if (found?.ready) return { backend: found };
    return {
      error: found?.reason ?? "Ollama is not running on 127.0.0.1:11434.",
      suggestedAction: { label: "Open backends", action: "open_settings", payload: { page: "backends" } },
      backend: undefined,
    };
  }
  if (normalized === "llamacpp" || normalized === "llama.cpp" || normalized === "llama-server") {
    const found = llamaCppBackend(backends) ?? findBackend(backends, "llamacpp");
    if (found?.ready) return { backend: found };
    return {
      error: found?.reason ?? "llama.cpp is not running on 127.0.0.1. Start llama-server bound to 127.0.0.1, then add it in Settings → Backends.",
      suggestedAction: { label: "Open backends", action: "open_settings", payload: { page: "backends" } },
      backend: undefined,
    };
  }
  const direct = findBackend(backends, pin) ?? findBackend(backends, normalized);
  if (direct) {
    if (direct.ready) return { backend: direct };
    const suggested =
      direct.type === "vllm"
        ? { label: "Start recommended local model", action: "start_vllm" as const }
        : { label: "Open backends", action: "open_settings" as const, payload: { page: "backends" } };
    return { error: direct.reason ?? `Backend "${direct.id}" is not ready.`, suggestedAction: suggested };
  }
  return { error: `Unknown backend "${pin}".` };
}

function runningVllmIds(ctx: RouterContext): Set<string> {
  const ids = new Set((ctx.vllmBackendIds ?? []).filter(Boolean));
  if (ids.size > 0) return ids;
  if (ctx.vllmRunning && ctx.vllmModelId) {
    for (const backend of ctx.backends) {
      if (!isVllmBackend(backend)) continue;
      const model = (backend.model ?? "").toLowerCase();
      const needle = ctx.vllmModelId.toLowerCase();
      if (
        backend.id === ctx.vllmModelId ||
        model === needle ||
        model.endsWith(`/${needle}`) ||
        backend.id.endsWith(needle.replaceAll("/", "-"))
      ) {
        ids.add(backend.id);
      }
    }
  }
  return ids;
}

function readyPool(ctx: RouterContext): RouterBackend[] {
  const skip = new Set(ctx.skipBackendIds ?? []);
  const runningIds = runningVllmIds(ctx);
  const ready = ctx.backends.filter((b) => {
    if (skip.has(b.id)) return false;
    // Probe-ready vLLM stays eligible even when the docker manager is stopped —
    // the operator may have started that /v1 themselves (Late does not).
    if (isVllmBackend(b) && !b.ready && ctx.vllmRunning === false && !runningIds.has(b.id)) {
      return false;
    }
    return b.ready;
  });
  // Promote orchestrator-started instances that have not probed yet — never the first YAML row.
  if (ctx.vllmRunning) {
    for (const id of runningIds) {
      if (skip.has(id) || ready.some((b) => b.id === id)) continue;
      const backend = findBackend(ctx.backends, id);
      if (backend && isVllmBackend(backend)) ready.push({ ...backend, ready: true });
    }
  }
  return ready;
}

function asSpeaker(backend: RouterBackend, intent: ChatIntent, ctx: RouterContext, visual3d: boolean): RouteSpeaker {
  const specialist = specialistFor(backend.id, intent, ctx.specialists ?? [], visual3d);
  const nickname = backend.nickname?.trim() || undefined;
  const hasLogo = Boolean(backend.hasLogo);
  return {
    backendId: backend.id,
    specialist,
    label: speakerLabel(backend, ctx.vllmModelId, specialist),
    writesLocalFiles: backend.writesLocalFiles,
    nickname,
    hasLogo,
    logoUrl: hasLogo ? backendLogoUrl(backend.id) : undefined,
  };
}

const DEBATE_PREFERENCE = ["vllm", "ollama", "llamacpp", "gemini", "anthropic", "openai", "cursor-local", "cursor-cloud"];

function debateRank(backend: RouterBackend): number {
  if (isLocalServerBackend(backend)) return 0;
  const idx = DEBATE_PREFERENCE.indexOf(backend.id);
  return idx === -1 ? 50 : idx;
}

function isVllmBackend(backend: RouterBackend): boolean {
  return backend.type === "vllm" || backend.id.startsWith("vllm");
}

function isLocalServerBackend(backend: RouterBackend): boolean {
  return isVllmBackend(backend) || isOllamaBackend(backend) || isLlamaCppBackend(backend);
}

/** Cloud/API specialists that join a round-table when ready. Not SKIP_UNLESS_NAMED. */
function isRoundtableCloud(backend: RouterBackend): boolean {
  const id = backend.id;
  if (id === "gemini" || id === "anthropic" || id === "openai") return true;
  if (id === "cursor-local" || id === "cursor-cloud" || backend.type === "cursor") return true;
  if (/gemini/i.test(id)) return true;
  return false;
}

function writerBackend(ready: RouterBackend[], preferLocal: boolean): RouterBackend | undefined {
  const local = ready.find((b) => b.id === "cursor-local");
  if (local) return local;
  if (preferLocal) return undefined;
  return ready.find((b) => b.id === "cursor-cloud");
}

function pickDebateSpeakers(
  intent: ChatIntent,
  ready: RouterBackend[],
  ctx: RouterContext,
  needsWrites: boolean,
  preferLocal: boolean,
  visual3d: boolean,
): RouteSpeaker[] {
  const sorted = [...ready].sort((a, b) => {
    if (visual3d && isVllmBackend(a) && isVllmBackend(b)) {
      if (a.id === "vllm-mistral-7b-instruct") return -1;
      if (b.id === "vllm-mistral-7b-instruct") return 1;
    }
    return debateRank(a) - debateRank(b);
  });
  const locals = sorted.filter(isLocalServerBackend);
  const clouds = sorted.filter((b) => !isLocalServerBackend(b) && isRoundtableCloud(b));
  const others = sorted.filter((b) => !isLocalServerBackend(b) && !isRoundtableCloud(b));
  const maxSpeakers = Math.max(8, locals.length + clouds.length);
  const chosen: RouterBackend[] = [];
  const take = (backend: RouterBackend) => {
    if (chosen.length >= maxSpeakers) return;
    if (chosen.some((c) => c.id === backend.id)) return;
    chosen.push(backend);
  };
  for (const backend of locals) take(backend);
  // Cursor and Gemini join by default when keys/ready — not SKIP_UNLESS_NAMED.
  for (const backend of clouds) take(backend);
  for (const backend of others) take(backend);
  if (needsWrites || intent === "code") {
    const cursor = writerBackend(ready, preferLocal);
    if (cursor) take(cursor);
  }
  return chosen.map((b) => asSpeaker(b, intent, ctx, visual3d));
}

function pickCloser(
  intent: ChatIntent,
  speakers: RouteSpeaker[],
  ready: RouterBackend[],
  ctx: RouterContext,
  needsWrites: boolean,
  preferLocal: boolean,
  visual3d: boolean,
): RouteSpeaker {
  if (needsWrites || intent === "code") {
    const cursor = writerBackend(ready, preferLocal);
    if (cursor) return asSpeaker(cursor, intent, ctx, visual3d);
  }
  const last = speakers[speakers.length - 1];
  if (last) return last;
  return asSpeaker(ready[0]!, intent, ctx, visual3d);
}

function pickSingle(
  intent: ChatIntent,
  ready: RouterBackend[],
  ctx: RouterContext,
  needsWrites: boolean,
  preferLocal: boolean,
  visual3d: boolean,
): RouteSpeaker | undefined {
  if (needsWrites || intent === "code") {
    const cursor = writerBackend(ready, preferLocal);
    if (cursor) return asSpeaker(cursor, intent, ctx, visual3d);
  }
  if (visual3d) {
    const gemini = ready.find((b) => b.id === "gemini");
    if (gemini) return asSpeaker(gemini, intent, ctx, visual3d);
    const mistral = ready.find((b) => b.id === "vllm-mistral-7b-instruct");
    if (mistral && ctx.vllmRunning !== false) return asSpeaker(mistral, intent, ctx, visual3d);
  }
  const vllm = ready.find((b) => b.type === "vllm" || b.id.startsWith("vllm"));
  if (vllm && ctx.vllmRunning !== false) return asSpeaker(vllm, intent, ctx, visual3d);
  const ollama = ready.find(isOllamaBackend);
  if (ollama) return asSpeaker(ollama, intent, ctx, visual3d);
  const llamaCpp = ready.find(isLlamaCppBackend);
  if (llamaCpp) return asSpeaker(llamaCpp, intent, ctx, visual3d);
  const gemini = ready.find((b) => b.id === "gemini");
  if (gemini) return asSpeaker(gemini, intent, ctx, visual3d);
  const cursor = ready.find((b) => b.id === "cursor-local") ?? ready.find((b) => b.id === "cursor-cloud");
  if (cursor) return asSpeaker(cursor, intent, ctx, visual3d);
  const first = ready[0];
  return first ? asSpeaker(first, intent, ctx, visual3d) : undefined;
}

function formatChip(kind: RouteDecision["kind"], speakers: RouteSpeaker[], closer?: RouteSpeaker): string {
  if (kind === "debate") return "Debate";
  const labels: string[] = [];
  for (const s of speakers) {
    if (!labels.includes(s.label)) labels.push(s.label);
  }
  if (closer && !labels.includes(closer.label)) labels.push(closer.label);
  return labels[0] ?? "Auto";
}

function startVllmAction(): ChatSuggestedAction {
  return { label: "Start recommended local model", action: "start_vllm" };
}

function cursorKeyError(preferLocal: boolean): { error: string; suggestedAction: ChatSuggestedAction } {
  return {
    error: preferLocal
      ? "File-changing work needs Cursor local with a write-allowlisted cwd. Tiny local vLLM cannot edit the repo. Set CURSOR_API_KEY in Settings → Backends (or .env), then Reload env. Get a key from Cursor Dashboard → Integrations."
      : "File-changing work needs Cursor (local or cloud). Tiny local vLLM cannot edit the repo. Set CURSOR_API_KEY in Settings → Backends (or .env), then Reload env. Get a key from Cursor Dashboard → Integrations.",
    suggestedAction: { label: "Open backends", action: "open_settings", payload: { page: "backends" } },
  };
}

function allowlistAction(path: string): ChatSuggestedAction {
  return {
    label: `Add ${path} to allowlist`,
    action: "add_allowed_dir",
    payload: { path },
  };
}

export function routeChat(ctx: RouterContext): RouteDecision {
  const rawPin = (ctx.pin?.trim() || "auto").trim();
  if (lateWrapMissingEnd(ctx.message)) {
    return {
      kind: "error",
      pin: rawPin.toLowerCase() || "auto",
      intent: "general",
      chip: "late-wrap",
      error: "Late wrap is missing END UNTRUSTED DEVICE OUTPUT. Refusing to route.",
    };
  }
  const message = extractRoutableMessage(ctx.message);
  const routed = { ...ctx, message };
  const pinLower = rawPin.toLowerCase();
  const mode: "auto" | "debate" | "single" | "backend" = MODE_PINS.has(pinLower)
    ? (pinLower as "auto" | "debate" | "single")
    : "backend";
  const named = mode === "auto" ? detectNamedBackend(message, ctx.backends) : undefined;
  const effectivePin = mode === "backend" ? rawPin : (named ?? pinLower);
  const intent = detectIntent(message);
  const visual3d = detectVisual3dIntent(message);
  const workspace = resolveWorkspaceHint(routed);
  const needsHostInstall = wantsHostInstall(message);
  const lateWrap = isLateDeviceWrap(ctx.message);
  const preferLocal = Boolean(workspace?.path);
  const writeCwd = workspace?.allowed ? workspace.cwd : undefined;
  const wantsWrites = wantsFileWrites(intent, message);
  const needsWrites = lateWrap ? Boolean(wantsWrites && writeCwd) : wantsWrites;
  const needsApproval = needsWrites || needsHostInstall;
  const approval = { needsWrites, needsHostInstall, needsApproval };

  if (intent === "control") {
    const control = detectControl(message) ?? "hardware";
    return {
      kind: "control",
      pin: effectivePin,
      intent,
      control,
      chip: "orchestrator",
    };
  }

  if (needsWrites && workspace && workspace.missing) {
    return {
      kind: "error",
      pin: pinLower,
      intent,
      chip: "allowlist",
      error: `Directory "${workspace.path}" does not exist, so Cursor cannot use it as cwd.`,
      ...approval,
    };
  }

  if (needsWrites && workspace && !workspace.allowed) {
    return {
      kind: "error",
      pin: pinLower,
      intent,
      chip: "allowlist",
      error:
        `"${workspace.path}" is not on the write allowlist. Local Cursor can only edit granted directories. Add it, then the team will implement there.`,
      suggestedAction: allowlistAction(workspace.path),
      ...approval,
    };
  }

  const ready = readyPool(ctx);

  if (mode === "backend" || (mode === "auto" && named)) {
    const resolved = resolvePinTarget(effectivePin, ctx.backends, Boolean(ctx.vllmRunning));
    if (!resolved.backend) {
      return {
        kind: "error",
        pin: effectivePin,
        intent,
        chip: effectivePin,
        error: resolved.error ?? `Backend "${effectivePin}" is not ready.`,
        suggestedAction: resolved.suggestedAction,
      };
    }
    const speaker = asSpeaker(resolved.backend, intent, ctx, visual3d);
    const sameBackend = ctx.prior?.backend === speaker.backendId && ctx.prior.runId;
    const readyNow = readyPool(ctx);
    const pinCursor = writerBackend(readyNow, preferLocal);
    const pinApplyPatch = Boolean(
      needsWrites && !speaker.writesLocalFiles && !pinCursor && readyNow.some(isLocalServerBackend),
    );
    if (needsWrites && !speaker.writesLocalFiles && !pinCursor && !pinApplyPatch) {
      const { error, suggestedAction } = cursorKeyError(preferLocal);
      return {
        kind: "error",
        pin: effectivePin,
        intent,
        chip: "cursor",
        error,
        suggestedAction,
        cwd: writeCwd,
        ...approval,
      };
    }
    const writer = pinCursor && needsWrites ? asSpeaker(pinCursor, intent, ctx, visual3d) : undefined;
    return {
      kind: "single",
      pin: effectivePin,
      intent,
      speakers: [speaker],
      closer: writer,
      chip: speaker.label,
      followUpRunId: sameBackend ? ctx.prior?.runId : undefined,
      cwd: speaker.writesLocalFiles || pinApplyPatch || Boolean(writer) ? writeCwd : undefined,
      ...approval,
      needsWrites: speaker.writesLocalFiles ? needsWrites : needsApproval,
      ...(pinApplyPatch ? { applyPatch: true } : {}),
    };
  }

  if (ready.length === 0) {
    const vllm = vllmBackend(ctx.backends);
    const gemini = findBackend(ctx.backends, "gemini");
    const suggested = !vllm?.ready || ctx.vllmRunning === false ? startVllmAction() : {
      label: "Open backends",
      action: "open_settings" as const,
      payload: { page: "backends" },
    };
    return {
      kind: "error",
      pin: pinLower,
      intent,
      chip: "none ready",
      error: gemini && !gemini.ready
        ? `No backends are ready. ${gemini.reason ?? "Gemini is not ready."}`
        : "No backends are ready. Start local vLLM, connect Ollama or llama.cpp, or add an API key in Settings.",
      suggestedAction: suggested,
    };
  }

  const cursorWriter = writerBackend(ready, preferLocal);
  const applyPatch = Boolean(
    needsWrites &&
      ready.some(isLocalServerBackend) &&
      (!cursorWriter || lateWrap),
  );
  if (needsWrites && !cursorWriter && !applyPatch) {
    const { error, suggestedAction } = cursorKeyError(preferLocal);
    return {
      kind: "error",
      pin: pinLower,
      intent,
      chip: "cursor",
      error,
      suggestedAction,
      cwd: writeCwd,
      ...approval,
    };
  }

  const forceDebate = mode === "debate";
  const debateReady =
    ready.length >= 2 &&
    (!needsWrites || Boolean(cursorWriter) || applyPatch);
  const multipleVllm = ready.filter(isVllmBackend).length >= 2;
  const multipleOtherLocal =
    ready.filter(isVllmBackend).length === 0 &&
    ready.filter((b) => isOllamaBackend(b) || isLlamaCppBackend(b)).length >= 2;
  const multipleLocalModels = multipleVllm || multipleOtherLocal;
  const autoDebate =
    mode === "auto" && debateReady && (DEBATE_INTENTS.has(intent) || multipleLocalModels);

  if (forceDebate || autoDebate) {
    const speakers = pickDebateSpeakers(intent, ready, ctx, needsWrites, preferLocal, visual3d);
    if (speakers.length >= 2) {
      const closer = pickCloser(intent, speakers, ready, ctx, needsWrites, preferLocal, visual3d);
      const rounds = ctx.followUp ? 1 : DEFAULT_ROUNDS;
      return {
        kind: "debate",
        pin: pinLower,
        intent,
        speakers,
        closer,
        rounds: Math.min(Math.max(rounds, 1), MAX_ROUNDS),
        chip: formatChip("debate", speakers, closer),
        note: applyPatch
          ? "round-table · after Approve, the orchestrator writes files inside the granted folder"
          : "round-table",
        cwd: closer.writesLocalFiles || applyPatch ? writeCwd : undefined,
        ...approval,
        ...(applyPatch ? { applyPatch: true } : {}),
      };
    }
  }

  const speaker = pickSingle(intent, ready, ctx, needsWrites, preferLocal, visual3d);
  if (!speaker) {
    return {
      kind: "error",
      pin: pinLower,
      intent,
      chip: "none ready",
      error: "No backends are ready.",
      suggestedAction: startVllmAction(),
    };
  }

  const sameBackend = ctx.prior?.backend === speaker.backendId && ctx.prior.runId && ctx.followUp;
  let note: string | undefined;
  if (applyPatch) {
    note = "After you Approve, the orchestrator writes files inside the granted folder. Cursor is not required.";
  } else if (needsWrites && !speaker.writesLocalFiles) {
    note = "Text only — this backend cannot edit the repo. Set CURSOR_API_KEY so Cursor local can apply the change.";
  }
  if (forceDebate && ready.length < 2) {
    note = [note, "Debate needs two ready backends; using a single speaker."].filter(Boolean).join(" ");
  }

  return {
    kind: "single",
    pin: pinLower,
    intent,
    speakers: [speaker],
    chip: speaker.label,
    followUpRunId: sameBackend ? ctx.prior?.runId : undefined,
    note,
    cwd: speaker.writesLocalFiles || applyPatch ? writeCwd : undefined,
    ...approval,
    ...(applyPatch ? { applyPatch: true } : {}),
    suggestedAction:
      speaker.backendId !== "vllm-local" && ctx.vllmRunning === false && !needsWrites
        ? startVllmAction()
        : undefined,
  };
}
