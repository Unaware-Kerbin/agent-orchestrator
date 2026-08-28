import { parse as parseYaml } from "yaml";
import { parseModelId } from "../identity.js";
import { normalizeLoopbackOpenAiUrl } from "./loopback.js";

export const DEFAULT_OLLAMA_BACKEND_ID = "ollama";
export const DEFAULT_OLLAMA_SPECIALIST_ID = "ollama-chat";
export const DEFAULT_LLAMACPP_BACKEND_ID = "llamacpp";
export const DEFAULT_LLAMACPP_SPECIALIST_ID = "llamacpp-chat";

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function yamlQuote(value: string): string {
  return JSON.stringify(value);
}

function setBlockField(block: string, key: string, value: string): string {
  const re = new RegExp(`^( {4}${escapeRegExp(key)}:)\\s*.*$`, "m");
  if (re.test(block)) return block.replace(re, `$1 ${value}`);
  if (/^ {4}type:\s*\S+\s*$/m.test(block)) {
    return block.replace(/^( {4}type:\s*\S+\s*)$/m, `$1\n    ${key}: ${value}`);
  }
  return block.replace(/^( {2}\S+:\n)/, `$1    ${key}: ${value}\n`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

/** Re-parse after a text patch and refuse type/baseUrl drift off loopback. */
export function assertLocalBackendPatch(
  yamlText: string,
  fields: { backendId: string; type: "ollama" | "llamacpp"; baseUrl: string; model: string },
): void {
  const parsed = parseYaml(yamlText);
  if (!isRecord(parsed) || !isRecord(parsed.backends)) {
    throw new Error("Config root must be a mapping after local backend patch");
  }
  const backend = parsed.backends[fields.backendId];
  if (!isRecord(backend)) {
    throw new Error(`Backend "${fields.backendId}" missing after patch`);
  }
  if (backend.type !== fields.type) {
    throw new Error(`Refusing config write: backend "${fields.backendId}" type must remain ${fields.type}`);
  }
  const label = fields.type === "ollama" ? "Ollama" : "llama.cpp";
  if (typeof backend.baseUrl !== "string") {
    throw new Error(`Refusing config write: ${label} baseUrl missing`);
  }
  const got = normalizeLoopbackOpenAiUrl(backend.baseUrl, label);
  const want = normalizeLoopbackOpenAiUrl(fields.baseUrl, label);
  if (got !== want) {
    throw new Error(`Refusing config write: ${label} baseUrl must stay on loopback`);
  }
  parseModelId(backend.model);
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

export function patchLocalBackendYaml(
  yamlText: string,
  fields: {
    backendId: string;
    type: "ollama" | "llamacpp";
    baseUrl: string;
    model: string;
    apiKey?: string;
  },
): string {
  const id = fields.backendId.trim();
  if (!id) throw new Error("backend id required");
  const label = fields.type === "ollama" ? "Ollama" : "llama.cpp";
  const model = parseModelId(fields.model);
  const baseUrl = normalizeLoopbackOpenAiUrl(fields.baseUrl, label);
  const found = mappingBlocksInSection(yamlText, "backends").find((block) => block.id === id);
  let next: string;
  if (found) {
    let block = found.block;
    if (!/^ {4}type:/m.test(block)) {
      block = block.replace(`  ${id}:\n`, `  ${id}:\n    type: ${fields.type}\n`);
    } else {
      block = setBlockField(block, "type", fields.type);
    }
    block = setBlockField(block, "baseUrl", yamlQuote(baseUrl));
    block = setBlockField(block, "model", yamlQuote(model));
    if (fields.apiKey) block = setBlockField(block, "apiKey", yamlQuote(fields.apiKey));
    next = yamlText.slice(0, found.index) + block + yamlText.slice(found.index + found.block.length);
  } else {
    const apiKeyLine = fields.apiKey ? `    apiKey: ${yamlQuote(fields.apiKey)}\n` : "";
    const insertion =
      `  ${id}:\n` +
      `    type: ${fields.type}\n` +
      `    baseUrl: ${yamlQuote(baseUrl)}\n` +
      `    model: ${yamlQuote(model)}\n` +
      apiKeyLine;
    if (/^backends:\s*\n/m.test(yamlText)) {
      next = yamlText.replace(/^backends:\s*\n/m, `backends:\n${insertion}`);
    } else {
      next = `${yamlText.trimEnd()}\n\nbackends:\n${insertion}`;
    }
  }
  assertLocalBackendPatch(next, { backendId: id, type: fields.type, baseUrl, model });
  return next;
}

export function patchLocalSpecialistYaml(
  yamlText: string,
  fields: { specialistId: string; backendId: string; description: string },
): string {
  const id = fields.specialistId.trim();
  if (!id) throw new Error("specialist id required");
  const found = mappingBlocksInSection(yamlText, "specialists").find((block) => block.id === id);
  if (found) {
    const block = setBlockField(found.block, "backend", fields.backendId);
    return yamlText.slice(0, found.index) + block + yamlText.slice(found.index + found.block.length);
  }

  const insertion =
    `  ${id}:\n` +
    `    description: ${fields.description}\n` +
    `    backend: ${fields.backendId}\n` +
    `    fallback: cursor-local\n`;
  if (/^specialists:\s*\n/m.test(yamlText)) {
    return yamlText.replace(/^specialists:\s*\n/m, `specialists:\n${insertion}`);
  }
  return `${yamlText.trimEnd()}\n\nspecialists:\n${insertion}`;
}

export function patchLocalOrchestratorYaml(
  yamlText: string,
  fields: {
    backendId: string;
    type: "ollama" | "llamacpp";
    baseUrl: string;
    model: string;
    apiKey?: string;
    specialistId: string;
    description: string;
  },
): string {
  const withBackend = patchLocalBackendYaml(yamlText, fields);
  return patchLocalSpecialistYaml(withBackend, {
    specialistId: fields.specialistId,
    backendId: fields.backendId,
    description: fields.description,
  });
}

export function ollamaSpecialistDescription(): string {
  return "Local Ollama OpenAI-compatible model (text only; file writes stay on Cursor)";
}

export function llamaCppSpecialistDescription(): string {
  return "Local llama.cpp llama-server (text only; file writes stay on Cursor)";
}
