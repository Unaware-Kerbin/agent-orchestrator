import type { WriteAllowlist } from "../allowlist.js";
import type { AgentProvider, BackendConfig } from "../types.js";
import { AnthropicProvider } from "./anthropic.js";
import { CursorProvider } from "./cursor.js";
import { HttpProvider } from "./http.js";
import { OpenAIProvider } from "./openai.js";
import { VllmProvider } from "./vllm.js";

export function createProvider(id: string, config: BackendConfig, allowlist?: WriteAllowlist): AgentProvider {
  switch (config.type) {
    case "cursor":
      return new CursorProvider(id, config, allowlist);
    case "openai":
      return new OpenAIProvider(id, config);
    case "anthropic":
      return new AnthropicProvider(id, config);
    case "http":
      return new HttpProvider(id, config);
    case "vllm":
      return new VllmProvider(id, config);
  }
}

export function createProviders(
  backends: Record<string, BackendConfig>,
  allowlist?: WriteAllowlist,
): Map<string, AgentProvider> {
  const map = new Map<string, AgentProvider>();
  for (const [id, config] of Object.entries(backends)) {
    map.set(id, createProvider(id, config, allowlist));
  }
  return map;
}
