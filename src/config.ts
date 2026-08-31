import { readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse as parseYaml } from "yaml";
import type {
  BackendConfig,
  OrchestratorConfig,
  SpecialistConfig,
  WorkflowConfig,
} from "./types.js";
import { isEnvVarName } from "./providers/keys.js";
import { isGeminiOpenAiConfig, parseGeminiModelId } from "./providers/gemini.js";
import { parseModelId, parseNickname } from "./identity.js";
import { normalizeLoopbackOpenAiUrl } from "./local-servers/loopback.js";
import { DEFAULT_LLAMACPP_BASE, DEFAULT_OLLAMA_BASE } from "./local-servers/loopback.js";
import { bindMcpListenHost, MCP_LOOPBACK_HOST } from "./mcp/bind.js";

const PACKAGE_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function interpolate(value: string): string {
  return value.replace(/\$\{([A-Z0-9_]+)\}/g, (_match, name: string) => {
    return process.env[name] ?? "";
  });
}

function walkStrings<T>(value: T): T {
  if (typeof value === "string") return interpolate(value) as T;
  if (Array.isArray(value)) return value.map((item) => walkStrings(item)) as T;
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, nested] of Object.entries(value as Record<string, unknown>)) {
      out[key] = walkStrings(nested);
    }
    return out as T;
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function asString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`Config field "${field}" must be a non-empty string`);
  }
  return value;
}

function optionalApiKeyEnv(id: string, raw: unknown): string | undefined {
  if (typeof raw !== "string" || !raw.trim()) return undefined;
  if (!isEnvVarName(raw.trim())) {
    throw new Error(
      `Config field "backends.${id}.apiKeyEnv" must be an environment variable name (e.g. GEMINI_API_KEY), not a secret value`,
    );
  }
  return raw.trim();
}

function withNickname<T extends BackendConfig>(id: string, raw: Record<string, unknown>, backend: T): T {
  if (!("nickname" in raw) || raw.nickname === undefined || raw.nickname === null || raw.nickname === "") {
    return backend;
  }
  const nickname = parseNickname(raw.nickname);
  return nickname ? { ...backend, nickname } : backend;
}

function parseBackend(id: string, raw: unknown): BackendConfig {
  if (!isRecord(raw) || typeof raw.type !== "string") {
    throw new Error(`Backend "${id}" is missing a type`);
  }
  const type = raw.type;
  if (type === "cursor") {
    const runtime = raw.runtime === "cloud" ? "cloud" : "local";
    return withNickname(id, raw, {
      type: "cursor",
      runtime,
      model: typeof raw.model === "string" ? raw.model : undefined,
    });
  }
  if (type === "openai") {
    const model = asString(raw.model, `backends.${id}.model`);
    const baseUrl = typeof raw.baseUrl === "string" ? raw.baseUrl : undefined;
    const apiKeyEnv = optionalApiKeyEnv(id, raw.apiKeyEnv);
    const resolvedModel = isGeminiOpenAiConfig(id, { type: "openai", baseUrl, apiKeyEnv, model })
      ? parseGeminiModelId(model)
      : model;
    return withNickname(id, raw, {
      type: "openai",
      baseUrl,
      model: resolvedModel,
      apiKeyEnv,
      apiKey: typeof raw.apiKey === "string" ? raw.apiKey : undefined,
    });
  }
  if (type === "anthropic") {
    return withNickname(id, raw, {
      type: "anthropic",
      baseUrl: typeof raw.baseUrl === "string" ? raw.baseUrl : undefined,
      model: asString(raw.model, `backends.${id}.model`),
      apiKeyEnv: optionalApiKeyEnv(id, raw.apiKeyEnv),
      apiKey: typeof raw.apiKey === "string" ? raw.apiKey : undefined,
      maxTokens: typeof raw.maxTokens === "number" ? raw.maxTokens : undefined,
    });
  }
  if (type === "vllm") {
    const baseUrl =
      typeof raw.baseUrl === "string" && raw.baseUrl.trim()
        ? raw.baseUrl.trim()
        : "http://127.0.0.1:8000/v1";
    return withNickname(id, raw, {
      type: "vllm",
      baseUrl,
      model: asString(raw.model, `backends.${id}.model`),
      apiKeyEnv: optionalApiKeyEnv(id, raw.apiKeyEnv),
      apiKey: typeof raw.apiKey === "string" ? raw.apiKey : undefined,
      probe: raw.probe === false ? false : true,
      probeTimeoutMs: typeof raw.probeTimeoutMs === "number" ? raw.probeTimeoutMs : undefined,
    });
  }
  if (type === "ollama" || type === "llamacpp") {
    const label = type === "ollama" ? "Ollama" : "llama.cpp";
    const fallback = type === "ollama" ? DEFAULT_OLLAMA_BASE : DEFAULT_LLAMACPP_BASE;
    const rawUrl = typeof raw.baseUrl === "string" && raw.baseUrl.trim() ? raw.baseUrl.trim() : fallback;
    return withNickname(id, raw, {
      type,
      baseUrl: normalizeLoopbackOpenAiUrl(rawUrl, label),
      model: asString(raw.model, `backends.${id}.model`),
      apiKeyEnv: optionalApiKeyEnv(id, raw.apiKeyEnv),
      apiKey: typeof raw.apiKey === "string" ? raw.apiKey : undefined,
      probe: raw.probe === false ? false : true,
      probeTimeoutMs: typeof raw.probeTimeoutMs === "number" ? raw.probeTimeoutMs : undefined,
    });
  }
  if (type === "http") {
    return withNickname(id, raw, {
      type: "http",
      url: asString(raw.url, `backends.${id}.url`),
      method: raw.method === "PUT" ? "PUT" : "POST",
      headers: isRecord(raw.headers)
        ? Object.fromEntries(
            Object.entries(raw.headers).filter(
              (entry): entry is [string, string] => typeof entry[1] === "string",
            ),
          )
        : undefined,
      timeoutMs: typeof raw.timeoutMs === "number" ? raw.timeoutMs : undefined,
    });
  }
  throw new Error(`Backend "${id}" has unsupported type "${type}"`);
}

function parseSpecialist(id: string, raw: unknown): SpecialistConfig {
  if (!isRecord(raw)) throw new Error(`Specialist "${id}" must be an object`);
  return {
    description: asString(raw.description, `specialists.${id}.description`),
    backend: asString(raw.backend, `specialists.${id}.backend`),
    fallback: typeof raw.fallback === "string" ? raw.fallback : undefined,
    system: typeof raw.system === "string" ? raw.system : undefined,
    mode: raw.mode === "plan" || raw.mode === "agent" ? raw.mode : undefined,
  };
}

function parseWorkflow(id: string, raw: unknown): WorkflowConfig {
  if (!isRecord(raw)) throw new Error(`Workflow "${id}" must be an object`);
  if (!Array.isArray(raw.steps) || raw.steps.length === 0) {
    throw new Error(`Workflow "${id}" needs at least one step`);
  }
  return {
    description: asString(raw.description, `workflows.${id}.description`),
    mode: raw.mode === "parallel" ? "parallel" : "sequence",
    steps: raw.steps.map((step, index) => {
      if (!isRecord(step)) {
        throw new Error(`Workflow "${id}" step ${index} must be an object`);
      }
      return {
        specialist: asString(step.specialist, `workflows.${id}.steps.${index}.specialist`),
        backend: typeof step.backend === "string" && step.backend.trim() ? step.backend.trim() : undefined,
      };
    }),
  };
}

export function resolveConfigPath(): string {
  const fromEnv = process.env.AGENT_ORCHESTRATOR_CONFIG;
  if (fromEnv) return isAbsolute(fromEnv) ? fromEnv : resolve(process.cwd(), fromEnv);
  const inCwd = resolve(process.cwd(), "agents.config.yaml");
  try {
    readFileSync(inCwd);
    return inCwd;
  } catch {
    return resolve(PACKAGE_ROOT, "agents.config.yaml");
  }
}

function parseMcpListenHost(raw: unknown): string | undefined {
  if (!isRecord(raw)) return undefined;
  const value = raw.listen_host ?? raw.listenHost;
  if (typeof value !== "string" || !value.trim()) return undefined;
  return bindMcpListenHost(value);
}

export function parseOrchestratorConfig(parsed: unknown): OrchestratorConfig {
  if (!isRecord(parsed)) throw new Error("Config root must be a mapping");

  const backendsRaw = isRecord(parsed.backends) ? parsed.backends : {};
  const specialistsRaw = isRecord(parsed.specialists) ? parsed.specialists : {};
  const workflowsRaw = isRecord(parsed.workflows) ? parsed.workflows : {};
  const workspace = isRecord(parsed.workspace) ? parsed.workspace : {};
  const defaults = isRecord(parsed.defaults) ? parsed.defaults : {};
  const listenHost = parseMcpListenHost(parsed.mcp);

  const backends: Record<string, BackendConfig> = {};
  for (const [id, value] of Object.entries(backendsRaw)) {
    backends[id] = parseBackend(id, value);
  }
  const specialists: Record<string, SpecialistConfig> = {};
  for (const [id, value] of Object.entries(specialistsRaw)) {
    specialists[id] = parseSpecialist(id, value);
  }
  const workflows: Record<string, WorkflowConfig> = {};
  for (const [id, value] of Object.entries(workflowsRaw)) {
    workflows[id] = parseWorkflow(id, value);
  }

  if (Object.keys(backends).length === 0) {
    throw new Error("Config must define at least one backend");
  }
  if (Object.keys(specialists).length === 0) {
    throw new Error("Config must define at least one specialist");
  }

  return {
    workspace: {
      cwd: typeof workspace.cwd === "string" && workspace.cwd ? workspace.cwd : undefined,
    },
    defaults: {
      wait: defaults.wait !== false,
      model: typeof defaults.model === "string" ? defaults.model : undefined,
    },
    mcp: listenHost ? { listenHost } : undefined,
    backends,
    specialists,
    workflows,
  };
}

export function parseConfigYaml(text: string, interpolateEnv = true): unknown {
  const parsed = parseYaml(text);
  return interpolateEnv ? walkStrings(parsed) : parsed;
}

export function validateConfigYaml(text: string): OrchestratorConfig {
  return parseOrchestratorConfig(parseConfigYaml(text, true));
}

export function loadConfig(path = resolveConfigPath()): OrchestratorConfig {
  return validateConfigYaml(readFileSync(path, "utf8"));
}

export function readConfigYaml(path = resolveConfigPath()): string {
  return readFileSync(path, "utf8");
}

export function writeConfigYaml(text: string, path = resolveConfigPath()): OrchestratorConfig {
  const parsed = validateConfigYaml(text);
  const body = text.endsWith("\n") ? text : `${text}\n`;
  writeFileSync(path, body, "utf8");
  return parsed;
}

export function defaultWorkspaceCwd(config: OrchestratorConfig): string {
  const fromConfig = config.workspace?.cwd?.trim();
  if (fromConfig) return resolve(fromConfig);
  return process.cwd();
}

export function packageRoot(): string {
  return PACKAGE_ROOT;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Replace `backends.<id>.model` in YAML text, preserving surrounding comments. */
export function patchBackendModelYaml(yamlText: string, backendId: string, model: string): string {
  const id = backendId.trim();
  if (!id) throw new Error("backend id required");
  const quoted = JSON.stringify(parseModelId(model));
  const header = /^backends:\s*\n/m.exec(yamlText);
  if (!header || header.index === undefined) {
    throw new Error(`Backend "${id}" not found in config`);
  }
  const start = header.index + header[0].length;
  const rest = yamlText.slice(start);
  const nextSection = rest.search(/^[A-Za-z][A-Za-z0-9_-]*:\s*$/m);
  const sectionText = nextSection >= 0 ? rest.slice(0, nextSection) : rest;
  const blockRe = new RegExp(`^( {2}${escapeRegExp(id)}:\\n(?: {4}.*\\n)*)`, "m");
  const match = blockRe.exec(sectionText);
  if (!match?.[1] || match.index === undefined) {
    throw new Error(`Backend "${id}" not found in config`);
  }
  const block = match[1];
  const modelRe = /^( {4}model:)\s*.*$/m;
  const nextBlock = modelRe.test(block)
    ? block.replace(modelRe, `$1 ${quoted}`)
    : block.replace(/^( {2}\S+:\n)/, `$1    model: ${quoted}\n`);
  const abs = start + match.index;
  return yamlText.slice(0, abs) + nextBlock + yamlText.slice(abs + block.length);
}

/** Set `mcp.listen_host` without rewriting backends. Loopback writes 127.0.0.1. */
export function patchMcpListenHostYaml(yamlText: string, listenHost: string): string {
  const host = bindMcpListenHost(listenHost || MCP_LOOPBACK_HOST);
  const value = JSON.stringify(host);
  const mcpHeader = /^mcp:\s*\n/m.exec(yamlText);
  if (mcpHeader && mcpHeader.index !== undefined) {
    const start = mcpHeader.index + mcpHeader[0].length;
    const rest = yamlText.slice(start);
    const nextSection = rest.search(/^[A-Za-z][A-Za-z0-9_-]*:\s*$/m);
    const sectionText = nextSection >= 0 ? rest.slice(0, nextSection) : rest;
    const listenRe = /^( {2}listen_host:)\s*.*$/m;
    const camelRe = /^( {2}listenHost:)\s*.*$/m;
    let nextSectionText: string;
    if (listenRe.test(sectionText)) {
      nextSectionText = sectionText.replace(listenRe, `$1 ${value}`);
    } else if (camelRe.test(sectionText)) {
      nextSectionText = sectionText.replace(camelRe, `  listen_host: ${value}`);
    } else {
      nextSectionText = `  listen_host: ${value}\n${sectionText}`;
    }
    return yamlText.slice(0, start) + nextSectionText + yamlText.slice(start + sectionText.length);
  }
  const block = `mcp:\n  listen_host: ${value}\n\n`;
  const backends = /^backends:\s*$/m.exec(yamlText);
  if (backends && backends.index !== undefined) {
    return yamlText.slice(0, backends.index) + block + yamlText.slice(backends.index);
  }
  const body = yamlText.endsWith("\n") ? yamlText : `${yamlText}\n`;
  return `${body}${block}`;
}

/** Set or clear `backends.<id>.nickname`. Empty nickname removes the field. */
export function patchBackendNicknameYaml(yamlText: string, backendId: string, nickname?: string): string {
  const id = backendId.trim();
  if (!id) throw new Error("backend id required");
  const parsed = nickname === undefined || nickname === "" ? undefined : parseNickname(nickname);
  const header = /^backends:\s*\n/m.exec(yamlText);
  if (!header || header.index === undefined) {
    throw new Error(`Backend "${id}" not found in config`);
  }
  const start = header.index + header[0].length;
  const rest = yamlText.slice(start);
  const nextSection = rest.search(/^[A-Za-z][A-Za-z0-9_-]*:\s*$/m);
  const sectionText = nextSection >= 0 ? rest.slice(0, nextSection) : rest;
  const blockRe = new RegExp(`^( {2}${escapeRegExp(id)}:\\n(?: {4}.*\\n)*)`, "m");
  const match = blockRe.exec(sectionText);
  if (!match?.[1] || match.index === undefined) {
    throw new Error(`Backend "${id}" not found in config`);
  }
  let block = match[1];
  const nickRe = /^( {4}nickname:)\s*.*$/m;
  if (!parsed) {
    block = block.replace(/^( {4}nickname:)\s*.*\n/m, "");
  } else if (nickRe.test(block)) {
    block = block.replace(nickRe, `$1 ${JSON.stringify(parsed)}`);
  } else {
    block = block.replace(/^( {2}\S+:\n)/, `$1    nickname: ${JSON.stringify(parsed)}\n`);
  }
  const abs = start + match.index;
  return yamlText.slice(0, abs) + block + yamlText.slice(abs + match[1].length);
}
