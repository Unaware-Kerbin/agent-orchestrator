/** True when a string is an orchestrator thread dump / Late SYSTEM wrap — never a tool error. */
export function looksLikeChatDump(raw: string): boolean {
  const t = raw.trim();
  if (!t) return false;
  if (/^SYSTEM:/m.test(t) && /UNTRUSTED DEVICE OUTPUT/i.test(t)) return true;
  if ((t.startsWith("{") || t.startsWith("[")) && /"messages"\s*:/.test(t)) return true;
  return false;
}

/**
 * MCP chat_send/chat_get catch payload. Never the thread JSON or SYSTEM wrap.
 * Speaker timeouts/429s stay on the thread as rows; they are not this string.
 */
export function compactChatToolError(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error);
  if (looksLikeChatDump(raw)) return "Chat failed.";
  const first = raw.trim().split(/\n/)[0] ?? "Chat failed.";
  return first.length > 240 ? `${first.slice(0, 220)}…` : first || "Chat failed.";
}
