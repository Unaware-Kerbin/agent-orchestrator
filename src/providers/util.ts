import type {
  AgentProvider,
  AnthropicBackendConfig,
  BackendConfig,
  CursorBackendConfig,
  HttpBackendConfig,
  LlamaCppBackendConfig,
  OllamaBackendConfig,
  OpenAIBackendConfig,
  ProviderHealth,
  ProviderRunRequest,
  ProviderRunResult,
  VllmBackendConfig,
} from "../types.js";
import { isEnvVarName } from "./keys.js";

export function secretFrom(
  config: { apiKey?: string; apiKeyEnv?: string },
  fallbackEnv: string | string[],
): string | undefined {
  if (config.apiKey && config.apiKey.trim()) return config.apiKey.trim();
  const names = [
    isEnvVarName(config.apiKeyEnv) ? config.apiKeyEnv : undefined,
    ...(Array.isArray(fallbackEnv) ? fallbackEnv : [fallbackEnv]),
  ].filter(isEnvVarName);
  const seen = new Set<string>();
  for (const name of names) {
    if (seen.has(name)) continue;
    seen.add(name);
    const value = process.env[name]?.trim();
    if (value) return value;
  }
  return undefined;
}

type HealthExtra = Partial<
  Pick<
    ProviderHealth,
    "runtime" | "writesLocalFiles" | "reason" | "secretNames" | "needsKey" | "baseUrl" | "model" | "modelChoices" | "nickname" | "hasLogo"
  >
>;

export function missingKeyHealth(
  id: string,
  type: ProviderHealth["type"],
  envName: string,
  capabilities: string[],
  extra?: HealthExtra,
): ProviderHealth {
  return {
    id,
    type,
    ready: false,
    reason: extra?.reason ?? `Set ${envName} to enable this backend`,
    capabilities,
    writesLocalFiles: extra?.writesLocalFiles ?? false,
    runtime: extra?.runtime,
    secretNames: extra?.secretNames,
    needsKey: extra?.needsKey ?? true,
    baseUrl: extra?.baseUrl,
    model: extra?.model,
    modelChoices: extra?.modelChoices,
  };
}

export function readyHealth(
  id: string,
  type: ProviderHealth["type"],
  capabilities: string[],
  extra?: HealthExtra,
): ProviderHealth {
  return {
    id,
    type,
    ready: true,
    reason: extra?.reason,
    capabilities,
    writesLocalFiles: extra?.writesLocalFiles ?? false,
    runtime: extra?.runtime,
    secretNames: extra?.secretNames,
    needsKey: extra?.needsKey ?? false,
    baseUrl: extra?.baseUrl,
    model: extra?.model,
    modelChoices: extra?.modelChoices,
  };
}

function googleErrorMessage(value: unknown): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  const rec = value as Record<string, unknown>;
  const err = rec.error;
  if (typeof err === "string" && err.trim()) return err;
  if (err && typeof err === "object") {
    const message = (err as { message?: unknown }).message;
    if (typeof message === "string" && message.trim()) return message;
  }
  return undefined;
}

export function extractHttpText(payload: unknown): string {
  if (typeof payload === "string") return payload;
  if (Array.isArray(payload) && payload[0]) {
    const fromItem = googleErrorMessage(payload[0]);
    if (fromItem) return fromItem;
  }
  if (!payload || typeof payload !== "object") return JSON.stringify(payload);
  const fromGoogle = googleErrorMessage(payload);
  if (fromGoogle) return fromGoogle;
  const record = payload as Record<string, unknown>;
  if (typeof record.text === "string") return record.text;
  if (typeof record.result === "string") return record.result;
  if (typeof record.output === "string") return record.output;
  const choices = record.choices;
  if (Array.isArray(choices) && choices[0] && typeof choices[0] === "object") {
    const message = (choices[0] as Record<string, unknown>).message;
    if (message && typeof message === "object") {
      const content = (message as Record<string, unknown>).content;
      if (typeof content === "string") return content;
    }
  }
  const content = record.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (typeof block === "string") return block;
        if (block && typeof block === "object" && typeof (block as { text?: unknown }).text === "string") {
          return (block as { text: string }).text;
        }
        return "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return JSON.stringify(payload, null, 2);
}

export type {
  AnthropicBackendConfig,
  BackendConfig,
  CursorBackendConfig,
  HttpBackendConfig,
  LlamaCppBackendConfig,
  OllamaBackendConfig,
  OpenAIBackendConfig,
  ProviderHealth,
  ProviderRunRequest,
  ProviderRunResult,
  VllmBackendConfig,
  AgentProvider,
};
