import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { loadEnvFiles, parseEnvText } from "./env.js";
import { ensureSecureDir, writeSecureFile } from "./platform.js";
import { isEnvVarName, VLLM_LOCAL_DUMMY_KEY } from "./providers/keys.js";
import { stateDir } from "./state.js";

const FILE_NAME = "secrets.env";

export const HF_HUB_TOKEN_NAMES = ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] as const;

export const KNOWN_SECRET_NAMES = [
  "CURSOR_API_KEY",
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "OPENROUTER_API_KEY",
  "GEMINI_API_KEY",
  "GOOGLE_API_KEY",
  "GOOGLE_GENERATIVE_AI_API_KEY",
  "VLLM_API_KEY",
  ...HF_HUB_TOKEN_NAMES,
  "RADIUS_SECRET",
  "LDAP_BIND_PASSWORD",
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

/** Hugging Face Hub token from env or the gitignored secrets file. Never log the return value. */
export function resolveHfToken(): string | undefined {
  for (const name of HF_HUB_TOKEN_NAMES) {
    const fromEnv = process.env[name]?.trim();
    if (fromEnv) return fromEnv;
  }
  const stored = loadSecretsFile();
  for (const name of HF_HUB_TOKEN_NAMES) {
    const fromFile = stored[name]?.trim();
    if (fromFile) return fromFile;
  }
  return undefined;
}

export function hfTokenConfigured(): boolean {
  return Boolean(resolveHfToken());
}

/** Clear removes HF_TOKEN and HUGGING_FACE_HUB_TOKEN together (same Hub credential). */
export function expandSecretClearNames(name: string): string[] {
  if ((HF_HUB_TOKEN_NAMES as readonly string[]).includes(name)) {
    return [...HF_HUB_TOKEN_NAMES];
  }
  return [name];
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

/** Remove secrets from the store and process.env. Does not log values. */
export function deleteSecrets(names: string[]): string[] {
  const stored = loadSecretsFile();
  const changed: string[] = [];
  const unique = [...new Set(names.flatMap(expandSecretClearNames))];
  for (const name of unique) {
    if (!isEnvVarName(name)) {
      throw new Error(`Invalid secret name "${name}". Use an env var like GEMINI_API_KEY.`);
    }
    const hadStored = Boolean(stored[name]);
    const hadEnv = Boolean(process.env[name]);
    if (hadStored) delete stored[name];
    if (hadEnv) delete process.env[name];
    if (hadStored || hadEnv) changed.push(name);
  }
  persistSecrets(stored);
  return changed;
}

function persistSecrets(vars: Record<string, string>): void {
  ensureSecureDir(stateDir());
  const lines = Object.entries(vars)
    .filter(([key, value]) => isEnvVarName(key) && value)
    .map(([key, value]) => `${key}=${escapeEnvValue(value)}`);
  writeSecureFile(secretsPath(), `${lines.join("\n")}${lines.length ? "\n" : ""}`);
}

function escapeEnvValue(value: string): string {
  if (/[\s#"']/.test(value)) {
    return `"${value.replaceAll("\\", "\\\\").replaceAll('"', '\\"')}"`;
  }
  return value;
}
