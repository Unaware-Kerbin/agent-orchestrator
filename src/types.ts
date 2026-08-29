export type RuntimeKind = "cursor" | "openai" | "anthropic" | "http" | "vllm" | "ollama" | "llamacpp";
export type CursorRuntime = "local" | "cloud";
export type RunStatus = "queued" | "running" | "finished" | "error" | "cancelled";
export type ConversationMode = "plan" | "agent";

export interface CursorBackendConfig {
  type: "cursor";
  runtime: CursorRuntime;
  model?: string;
}

export interface OpenAIBackendConfig {
  type: "openai";
  baseUrl?: string;
  model: string;
  apiKeyEnv?: string;
  apiKey?: string;
}

export interface AnthropicBackendConfig {
  type: "anthropic";
  baseUrl?: string;
  model: string;
  apiKeyEnv?: string;
  apiKey?: string;
  maxTokens?: number;
}

export interface HttpBackendConfig {
  type: "http";
  url: string;
  method?: "POST" | "PUT";
  headers?: Record<string, string>;
  timeoutMs?: number;
}

export interface VllmBackendConfig {
  type: "vllm";
  baseUrl?: string;
  model: string;
  apiKeyEnv?: string;
  apiKey?: string;
  /** When true (default), catalog/dispatch probe the OpenAI-compatible /models endpoint. */
  probe?: boolean;
  probeTimeoutMs?: number;
}

export interface OllamaBackendConfig {
  type: "ollama";
  baseUrl?: string;
  model: string;
  apiKeyEnv?: string;
  apiKey?: string;
  probe?: boolean;
  probeTimeoutMs?: number;
}

export interface LlamaCppBackendConfig {
  type: "llamacpp";
  baseUrl?: string;
  model: string;
  apiKeyEnv?: string;
  apiKey?: string;
  probe?: boolean;
  probeTimeoutMs?: number;
}

export type BackendConfig = (
  | CursorBackendConfig
  | OpenAIBackendConfig
  | AnthropicBackendConfig
  | HttpBackendConfig
  | VllmBackendConfig
  | OllamaBackendConfig
  | LlamaCppBackendConfig
) & { nickname?: string };

export interface SpecialistConfig {
  description: string;
  backend: string;
  fallback?: string;
  system?: string;
  mode?: ConversationMode;
}

export interface WorkflowStepConfig {
  specialist: string;
  /** Optional backend id override for this step (e.g. cursor-cloud). */
  backend?: string;
}

export type WorkflowMode = "sequence" | "parallel";

export interface WorkflowConfig {
  description: string;
  /** parallel: all steps run at once with the same prompt; sequence (default) passes prior output forward. */
  mode?: WorkflowMode;
  steps: WorkflowStepConfig[];
}

export interface OrchestratorConfig {
  workspace?: { cwd?: string };
  defaults?: { wait?: boolean; model?: string };
  backends: Record<string, BackendConfig>;
  specialists: Record<string, SpecialistConfig>;
  workflows: Record<string, WorkflowConfig>;
}

export interface ProviderRunRequest {
  prompt: string;
  system?: string;
  cwd?: string;
  model?: string;
  mode?: ConversationMode;
  resumeAgentId?: string;
  history?: Array<{ role: "user" | "assistant"; content: string }>;
  /** When set, OpenAI-compat backends request SSE and forward token deltas. */
  onDelta?: (delta: string) => void;
  /** Per-run HTTP/SDK cap. Cursor speakers use a longer default than Gemini/vLLM. */
  timeoutMs?: number;
  cloud?: {
    repos?: Array<{ url: string; startingRef?: string }>;
    autoCreatePR?: boolean;
  };
}

export interface ProviderRunResult {
  status: Exclude<RunStatus, "queued" | "running">;
  text: string;
  agentId?: string;
  providerRunId?: string;
  error?: string;
  durationMs?: number;
}

export interface ProviderHealth {
  id: string;
  type: RuntimeKind;
  ready: boolean;
  reason?: string;
  capabilities: string[];
  runtime?: CursorRuntime;
  writesLocalFiles: boolean;
  secretNames?: string[];
  needsKey?: boolean;
  baseUrl?: string;
  model?: string;
  /** Known ids for GUI dropdowns (Gemini OpenAI-compat). */
  modelChoices?: string[];
  nickname?: string;
  hasLogo?: boolean;
}

export interface AgentProvider {
  id: string;
  type: RuntimeKind;
  capabilities: string[];
  health(): ProviderHealth;
  probe?(): Promise<ProviderHealth>;
  run(request: ProviderRunRequest): Promise<ProviderRunResult>;
  followUp?(agentId: string, message: string): Promise<ProviderRunResult>;
  cancel?(handle: unknown): Promise<void>;
}

export interface OrchestratedRun {
  id: string;
  specialist: string;
  backend: string;
  status: RunStatus;
  prompt: string;
  text?: string;
  error?: string;
  agentId?: string;
  providerRunId?: string;
  workflowId?: string;
  stepIndex?: number;
  cwd?: string;
  createdAt: number;
  updatedAt: number;
  durationMs?: number;
  history: Array<{ role: "user" | "assistant"; content: string }>;
}

export interface WritePolicy {
  allowedDirectories: string[];
  defaultCwd: string;
  fileWrites: "cursor-local-only";
}
