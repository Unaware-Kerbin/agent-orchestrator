import { existsSync, mkdirSync, readdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { stateDir } from "../state.js";
import type { ChatMessage, ChatThread, ChatThreadSummary } from "./types.js";

function chatsDir(): string {
  const dir = join(stateDir(), "chats");
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  return dir;
}

function threadPath(id: string): string {
  if (!/^[a-zA-Z0-9_-]+$/.test(id)) {
    throw new Error("Invalid chat id");
  }
  return join(chatsDir(), `${id}.json`);
}

function isThread(value: unknown): value is ChatThread {
  if (!value || typeof value !== "object") return false;
  const rec = value as Record<string, unknown>;
  return typeof rec.id === "string" && typeof rec.title === "string" && Array.isArray(rec.messages);
}

export class ChatStore {
  private readonly memory = new Map<string, ChatThread>();

  create(pin = "auto"): ChatThread {
    const now = Date.now();
    const thread: ChatThread = {
      id: crypto.randomUUID(),
      title: "New chat",
      messages: [],
      agents: [],
      pin,
      createdAt: now,
      updatedAt: now,
    };
    this.save(thread);
    return thread;
  }

  get(id: string): ChatThread | undefined {
    const cached = this.memory.get(id);
    if (cached) return cached;
    const path = threadPath(id);
    if (!existsSync(path)) return undefined;
    try {
      const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
      if (!isThread(parsed)) return undefined;
      this.memory.set(id, parsed);
      return parsed;
    } catch {
      return undefined;
    }
  }

  save(thread: ChatThread, persist = true): ChatThread {
    thread.updatedAt = Date.now();
    this.memory.set(thread.id, thread);
    if (persist) {
      const path = threadPath(thread.id);
      writeFileSync(path, `${JSON.stringify(thread, null, 2)}\n`, { encoding: "utf8", mode: 0o600 });
    }
    return thread;
  }

  delete(id: string): boolean {
    this.memory.delete(id);
    const path = threadPath(id);
    if (!existsSync(path)) return false;
    unlinkSync(path);
    return true;
  }

  list(): ChatThreadSummary[] {
    const dir = chatsDir();
    const summaries: ChatThreadSummary[] = [];
    for (const file of readdirSync(dir)) {
      if (!file.endsWith(".json")) continue;
      const parsed = this.get(file.slice(0, -5));
      if (!parsed) continue;
      summaries.push({
        id: parsed.id,
        title: parsed.title,
        updatedAt: parsed.updatedAt,
        pin: parsed.pin,
        agents: parsed.agents,
      });
    }
    return summaries.sort((a, b) => b.updatedAt - a.updatedAt);
  }

  append(id: string, message: Omit<ChatMessage, "id" | "createdAt"> & { id?: string; createdAt?: number }): ChatMessage {
    const thread = this.require(id);
    const full: ChatMessage = {
      id: message.id ?? crypto.randomUUID(),
      createdAt: message.createdAt ?? Date.now(),
      ...message,
    };
    thread.messages.push(full);
    if (full.speaker && full.role === "assistant" && !thread.agents.includes(full.speaker)) {
      thread.agents.push(full.speaker);
    }
    if (thread.title === "New chat" && full.role === "user" && full.content.trim()) {
      thread.title = titleFrom(full.content);
    }
    this.save(thread);
    return full;
  }

  patchMessage(threadId: string, messageId: string, patch: Partial<ChatMessage>, persist = true): ChatMessage {
    const thread = this.require(threadId);
    const idx = thread.messages.findIndex((m) => m.id === messageId);
    if (idx < 0) throw new Error(`Unknown message ${messageId}`);
    const current = thread.messages[idx]!;
    const next = { ...current, ...patch, id: messageId };
    thread.messages[idx] = next;
    if (next.runId) thread.lastRunId = next.runId;
    if (next.agentId) thread.lastAgentId = next.agentId;
    if (next.speaker && next.role === "assistant") thread.lastBackend = next.speaker;
    this.save(thread, persist);
    return next;
  }

  setPin(id: string, pin: string): ChatThread {
    const thread = this.require(id);
    thread.pin = pin.trim() || "auto";
    return this.save(thread);
  }

  setPendingApproval(id: string, pending: ChatThread["pendingApproval"] | undefined): ChatThread {
    const thread = this.require(id);
    if (pending) thread.pendingApproval = pending;
    else delete thread.pendingApproval;
    return this.save(thread);
  }

  require(id: string): ChatThread {
    const thread = this.get(id);
    if (!thread) throw new Error(`Unknown chat "${id}"`);
    return thread;
  }
}

function titleFrom(content: string): string {
  const line = content.trim().split(/\n/)[0] ?? "New chat";
  return line.length > 72 ? `${line.slice(0, 69)}…` : line;
}
