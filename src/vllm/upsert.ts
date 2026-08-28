import { parseModelId } from "../identity.js";

export const DEFAULT_VLLM_BACKEND_ID = "vllm-local";
export const DEFAULT_VLLM_SPECIALIST_ID = "vllm-chat";
export const VLLM_CONTAINER_PREFIX = "orch-vllm";

/** Catalog id `qwen2.5-7b-instruct` → `qwen25-7b-instruct`. */
export function vllmModelSlug(modelId: string): string {
  const trimmed = modelId.trim().toLowerCase();
  const withoutPrefix = trimmed.startsWith("vllm-") ? trimmed.slice("vllm-".length) : trimmed;
  const slug = withoutPrefix
    .replaceAll(".", "")
    .replaceAll("/", "-")
    .replace(/[^a-z0-9-]+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
  return (slug || "local").slice(0, 40);
}

/** Backend id from catalog id, e.g. `qwen2.5-7b-instruct` → `vllm-qwen25-7b-instruct`. */
export function vllmBackendIdForModel(modelId: string): string {
  return `vllm-${vllmModelSlug(modelId)}`;
}

export function vllmSpecialistIdForModel(modelId: string): string {
  return vllmBackendIdForModel(modelId);
}

/** Docker name from catalog id, e.g. `orch-vllm-qwen25-7b-instruct`. Never the bare `orch-vllm`. */
export function vllmContainerNameForModel(modelId: string): string {
  return `${VLLM_CONTAINER_PREFIX}-${vllmModelSlug(modelId)}`;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function yamlQuote(value: string): string {
  return JSON.stringify(value);
}

function setBlockField(block: string, key: string, value: string): string {
  const re = new RegExp(`^( {4}${escapeRegExp(key)}:)\\s*.*$`, "m");
  if (re.test(block)) return block.replace(re, `$1 ${value}`);
  if (/^ {4}type:\s*vllm\s*$/m.test(block)) {
    return block.replace(/^( {4}type:\s*vllm\s*)$/m, `$1\n    ${key}: ${value}`);
  }
  return block.replace(/^( {2}\S+:\n)/, `$1    ${key}: ${value}\n`);
}

/**
 * Update `backends.<id>.baseUrl` and `model` in agents.config.yaml text.
 * Preserves surrounding comments when the backend already exists.
 */
export function patchVllmBackendYaml(
  yamlText: string,
  fields: { baseUrl: string; model: string; backendId?: string },
): string {
  const id = fields.backendId ?? DEFAULT_VLLM_BACKEND_ID;
  const model = yamlQuote(parseModelId(fields.model));
  const baseUrl = yamlQuote(fields.baseUrl);
  const found = mappingBlocksInSection(yamlText, "backends").find((block) => block.id === id);
  if (found) {
    let block = found.block;
    block = setBlockField(block, "baseUrl", baseUrl);
    block = setBlockField(block, "model", model);
    if (!/^ {4}type:/m.test(block)) {
      block = block.replace(`  ${id}:\n`, `  ${id}:\n    type: vllm\n`);
    }
    return yamlText.slice(0, found.index) + block + yamlText.slice(found.index + found.block.length);
  }

  const insertion = `  ${id}:\n    type: vllm\n    baseUrl: ${baseUrl}\n    model: ${model}\n`;
  if (/^backends:\s*\n/m.test(yamlText)) {
    return yamlText.replace(/^backends:\s*\n/m, `backends:\n${insertion}`);
  }
  return `${yamlText.trimEnd()}\n\nbackends:\n${insertion}`;
}

export function vllmBackendIdFromConfig(backends: Record<string, { type?: string }>): string {
  if (backends[DEFAULT_VLLM_BACKEND_ID]) return DEFAULT_VLLM_BACKEND_ID;
  const found = Object.entries(backends).find(([, backend]) => backend.type === "vllm");
  return found?.[0] ?? DEFAULT_VLLM_BACKEND_ID;
}

/**
 * Ensure `specialists.vllm-chat` exists and points at the vLLM backend (no duplicate ids).
 */
export function patchVllmSpecialistYaml(
  yamlText: string,
  fields: { backendId?: string; specialistId?: string } = {},
): string {
  const id = fields.specialistId ?? DEFAULT_VLLM_SPECIALIST_ID;
  const backendId = fields.backendId ?? DEFAULT_VLLM_BACKEND_ID;
  const found = mappingBlocksInSection(yamlText, "specialists").find((block) => block.id === id);
  if (found) {
    const block = setBlockField(found.block, "backend", backendId);
    return yamlText.slice(0, found.index) + block + yamlText.slice(found.index + found.block.length);
  }

  const insertion =
    `  ${id}:\n` +
    `    description: Local vLLM OpenAI-compatible model (text only; file writes stay on Cursor)\n` +
    `    backend: ${backendId}\n` +
    `    fallback: cursor-local\n`;
  if (/^specialists:\s*\n/m.test(yamlText)) {
    return yamlText.replace(/^specialists:\s*\n/m, `specialists:\n${insertion}`);
  }
  return `${yamlText.trimEnd()}\n\nspecialists:\n${insertion}`;
}

export function patchVllmOrchestratorYaml(
  yamlText: string,
  fields: { baseUrl: string; model: string; backendId?: string; specialistId?: string },
): string {
  const withBackend = patchVllmBackendYaml(yamlText, fields);
  return patchVllmSpecialistYaml(withBackend, {
    backendId: fields.backendId ?? DEFAULT_VLLM_BACKEND_ID,
    specialistId: fields.specialistId,
  });
}

function mappingBlocksInSection(
  yamlText: string,
  section: "backends" | "specialists",
): Array<{ id: string; block: string; index: number }> {
  const header = new RegExp(`^${section}:\\s*\\n`, "m");
  const headerMatch = header.exec(yamlText);
  if (!headerMatch || headerMatch.index === undefined) return [];
  const start = headerMatch.index + headerMatch[0].length;
  const rest = yamlText.slice(start);
  const nextSection = rest.search(/^[A-Za-z][A-Za-z0-9_-]*:\s*$/m);
  const sectionText = nextSection >= 0 ? rest.slice(0, nextSection) : rest;
  const blockRe = /^( {2}([A-Za-z][A-Za-z0-9_-]*):\n(?: {4}.*\n)*)/gm;
  const blocks: Array<{ id: string; block: string; index: number }> = [];
  let match: RegExpExecArray | null;
  while ((match = blockRe.exec(sectionText))) {
    if (match.index === undefined || match[1] === undefined || match[2] === undefined) continue;
    blocks.push({ id: match[2], block: match[1], index: start + match.index });
  }
  return blocks;
}

export function removeYamlMappingBlock(
  yamlText: string,
  section: "backends" | "specialists",
  id: string,
): string {
  const blockRe = new RegExp(`^( {2}${escapeRegExp(id)}:\\n(?: {4}.*\\n)*)`, "m");
  const header = new RegExp(`^${section}:\\s*\\n`, "m");
  const headerMatch = header.exec(yamlText);
  if (!headerMatch || headerMatch.index === undefined) return yamlText;
  const start = headerMatch.index + headerMatch[0].length;
  const rest = yamlText.slice(start);
  const nextSection = rest.search(/^[A-Za-z][A-Za-z0-9_-]*:\s*$/m);
  const sectionText = nextSection >= 0 ? rest.slice(0, nextSection) : rest;
  const match = blockRe.exec(sectionText);
  if (!match?.[1] || match.index === undefined) return yamlText;
  const abs = start + match.index;
  return yamlText.slice(0, abs) + yamlText.slice(abs + match[1].length);
}

export function specialistIdsPointingAt(yamlText: string, backendId: string): string[] {
  const ids: string[] = [];
  const backendLine = new RegExp(`^ {4}backend:\\s*${escapeRegExp(backendId)}\\s*$`, "m");
  for (const block of mappingBlocksInSection(yamlText, "specialists")) {
    if (backendLine.test(block.block)) ids.push(block.id);
  }
  return ids;
}

/**
 * Delete `backends.<id>` and every specialist whose `backend` is that id.
 * Leaves other vLLM backends/specialists (and comments) in place.
 */
export function removeVllmOrchestratorYaml(
  yamlText: string,
  fields: { backendId: string },
): string {
  const id = fields.backendId.trim();
  if (!id) return yamlText;
  let next = yamlText;
  const specIds = specialistIdsPointingAt(next, id);
  next = removeYamlMappingBlock(next, "backends", id);
  for (const specId of specIds) {
    next = removeYamlMappingBlock(next, "specialists", specId);
  }
  return next;
}
