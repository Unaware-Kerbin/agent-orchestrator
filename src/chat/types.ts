export type ChatPin = "auto" | "debate" | "single" | "local" | "cloud" | "gemini" | string;

export type ChatIntent = "control" | "code" | "review" | "reason" | "general";

export type ChatPhase = "single" | "debate" | "synthesis" | "control" | "approval";

export type ThinkingPhase = "waiting" | "streaming" | "debating";

export interface ChatSuggestedAction {
  label: string;
  action: "start_vllm" | "download_model" | "open_settings" | "reload_env" | "add_allowed_dir";
  payload?: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  speaker: string;
  label: string;
  content: string;
  createdAt: number;
  status?: "thinking" | "streaming" | "finished" | "error";
  runId?: string;
  agentId?: string;
  round?: number;
  phase?: ChatPhase;
  thinkingPhase?: ThinkingPhase;
  thinkingStartedAt?: number;
  chip?: string;
  error?: string;
  suggestedAction?: ChatSuggestedAction;
  /** Backend nickname when set — prefer this over the raw model id. */
  nickname?: string;
  hasLogo?: boolean;
  /** Loopback URL (no token). Late/GUI fetch with Bearer. */
  logoUrl?: string;
}

export interface ChatHeartbeatPayload {
  threadId: string;
  now: number;
  thinking: Array<{
    id: string;
    label: string;
    speaker: string;
    status: NonNullable<ChatMessage["status"]>;
    thinkingPhase: ThinkingPhase;
    thinkingStartedAt: number;
  }>;
}

export interface PendingApproval {
  id: string;
  status: "pending" | "approved" | "rejected";
  cwd?: string;
  specialist: string;
  backendId: string;
  label: string;
  userMessage: string;
  summary: string;
  commands: string[];
  systemWideInstall: boolean;
  systemWideNote?: string;
  comment?: string;
  pin?: string;
  extraContext?: string;
  prUrl?: string;
  repoUrl?: string;
  branch?: string;
  createdAt: number;
}

export interface ChatThread {
  id: string;
  title: string;
  messages: ChatMessage[];
  agents: string[];
  pin: string;
  lastRunId?: string;
  lastAgentId?: string;
  lastBackend?: string;
  pendingApproval?: PendingApproval;
  createdAt: number;
  updatedAt: number;
  /** True while respond() is in flight. Not persisted; MCP/Late poll this. */
  busy?: boolean;
}

export interface ChatThreadSummary {
  id: string;
  title: string;
  updatedAt: number;
  pin: string;
  agents: string[];
}

export interface RouterBackend {
  id: string;
  type: string;
  ready: boolean;
  writesLocalFiles: boolean;
  runtime?: "local" | "cloud";
  model?: string;
  reason?: string;
  nickname?: string;
  hasLogo?: boolean;
}

export interface RouterSpecialist {
  id: string;
  backend: string;
}

export interface WorkspaceHint {
  /** Path as it appeared in the message (or explicit cwd). */
  path: string;
  allowed: boolean;
  /** Canonical cwd when allowlisted. */
  cwd?: string;
  /** True when the path does not exist as a directory. */
  missing?: boolean;
}

export interface RouterContext {
  message: string;
  pin?: string;
  backends: RouterBackend[];
  specialists?: RouterSpecialist[];
  vllmRunning?: boolean;
  vllmModelId?: string;
  prior?: { runId: string; agentId?: string; backend: string };
  /** True when this thread already has assistant turns (follow-up). */
  followUp?: boolean;
  /** Canonical write-allowlist directories (realpath). */
  allowedDirectories?: string[];
  /** Pre-resolved workspace from ChatService (realpath + allowlist). */
  workspace?: WorkspaceHint;
  /** Backends that already timed out or 429'd on this thread — skip on follow-up. */
  skipBackendIds?: string[];
}

export interface RouteSpeaker {
  backendId: string;
  specialist: string;
  label: string;
  writesLocalFiles: boolean;
  nickname?: string;
  hasLogo?: boolean;
  logoUrl?: string;
}

export type ControlKind = "hardware" | "models" | "start_vllm" | "stop_vllm" | "vllm_status" | "allowlist";

export interface RouteDecision {
  kind: "control" | "single" | "debate" | "error";
  pin: string;
  intent: ChatIntent;
  chip: string;
  control?: ControlKind;
  speakers?: RouteSpeaker[];
  closer?: RouteSpeaker;
  rounds?: number;
  followUpRunId?: string;
  error?: string;
  suggestedAction?: ChatSuggestedAction;
  note?: string;
  /** Allowlisted cwd for Cursor local writes. */
  cwd?: string;
  needsWrites?: boolean;
  needsHostInstall?: boolean;
  needsApproval?: boolean;
}
