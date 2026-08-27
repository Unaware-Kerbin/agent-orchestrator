import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { loadEnvFiles, parseEnvText } from "./env.js";
import { isEnvVarName, VLLM_LOCAL_DUMMY_KEY } from "./providers/keys.js";
import { stateDir } from "./state.js";

const FILE_NAME = "secrets.env";

export const KNOWN_SECRET_NAMES = [
  "CURSOR_API_KEY",
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "OPENROUTER_API_KEY",
  "GEMINI_API_KEY",
  "GOOGLE_API_KEY",
  "GOOGLE_GENERATIVE_AI_API_KEY",
  "VLLM_API_KEY",
  "HF_TOKEN",
  "HUGGING_FACE_HUB_TOKEN",
] as const;

export function secretsPath(): string {
  return join(stateDir(), FILE_NAME);
}

export function loadSecretsFile(): Record<string, string> {
  const path = secretsPath();
  if (!existsSync(path)) return {};
  return parseEnvText(readFileSync(path, "utf8"));
}

export function loadSecretsIntoEnv(overwrite = true): void {
  const vars = loadSecretsFile();
  for (const [key, value] of Object.entries(vars)) {
    if (!isEnvVarName(key) || !value) continue;
    if (overwrite || !process.env[key]?.trim()) process.env[key] = value;
  }
}

export function refreshRuntimeEnv(): void {
  loadEnvFiles(true);
  loadSecretsIntoEnv(true);
}

export interface SecretStatus {
  name: string;
  set: boolean;
}

export function secretStatus(names: string[]): SecretStatus[] {
  const stored = loadSecretsFile();
  const unique = [...new Set(names.filter(isEnvVarName))];
  return unique.map((name) => ({
    name,
    set: Boolean(process.env[name]?.trim() || stored[name]?.trim()),
  }));
}

/** Store a dummy loopback Bearer if none is set. Never overwrite a real key. */
export function ensureLocalVllmDummyKey(): string {
  const existing = process.env.VLLM_API_KEY?.trim() || loadSecretsFile().VLLM_API_KEY?.trim();
  if (existing) return existing;
  upsertSecrets({ VLLM_API_KEY: VLLM_LOCAL_DUMMY_KEY });
  return VLLM_LOCAL_DUMMY_KEY;
}

export function upsertSecrets(updates: Record<string, string>): string[] {
  const stored = loadSecretsFile();
  const changed: string[] = [];
  for (const [name, value] of Object.entries(updates)) {
    if (!isEnvVarName(name)) {
      throw new Error(`Invalid secret name "${name}". Use an env var like GEMINI_API_KEY.`);
    }
    const trimmed = value.trim();
    if (!trimmed) continue;
    stored[name] = trimmed;
    process.env[name] = trimmed;
    changed.push(name);
  }
  persistSecrets(stored);
  return changed;
}

function persistSecrets(vars: Record<string, string>): void {
  const dir = stateDir();
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  const lines = Object.entries(vars)
    .filter(([key, value]) => isEnvVarName(key) && value)
    .map(([key, value]) => `${key}=${escapeEnvValue(value)}`);
  writeFileSync(secretsPath(), `${lines.join("\n")}${lines.length ? "\n" : ""}`, {
    encoding: "utf8",
    mode: 0o600,
  });
}

function escapeEnvValue(value: string): string {
  if (/[\s#"']/.test(value)) {
    return `"${value.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
  }
  return value;
}
