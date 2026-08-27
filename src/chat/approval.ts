import type { PendingApproval, RouteDecision } from "./types.js";

const QA_ONLY =
  /\b(how (?:do|can|should) (?:i|we)|what(?:'s| is) the (?:best )?(?:way|command)|explain(?: how)?|tell me)\b/i;
const PLEASE_DO = /\b(for me|please (?:install|do|set)|go ahead|set this all up|and set this(?: all)? up)\b/i;
const HOST_PKG =
  /\b(?:sudo\s+)?(?:apt(?:-get)?|dnf|yum|pacman|zypper|brew|snap|winget|choco)\s+install\b/i;
const SUDO_INSTALL = /\bsudo\b/i;
const UNITY = /\b(unity(?:\s*hub)?|unityhub)\b/i;
const INSTALL_VERB = /\b(install|set(?: this)? (?:all )?up|setup)\b/i;
const INSTALL_TARGET = /\binstall\s+(?:unity|docker|node|nodejs|python|packages?)\b/i;

/** Direct request to install host software (Unity Hub, apt, sudo). Q&A stays unblocked. */
export function wantsHostInstall(message: string): boolean {
  const text = message.trim();
  if (!text) return false;
  if (QA_ONLY.test(text) && !PLEASE_DO.test(text)) return false;
  if (HOST_PKG.test(text)) return true;
  if (SUDO_INSTALL.test(text) && /\b(install|apt|dnf|yum|pacman)\b/i.test(text)) return true;
  if (UNITY.test(text) && INSTALL_VERB.test(text)) return true;
  if (INSTALL_TARGET.test(text)) return true;
  return false;
}

export function extractProposedCommands(plan: string): string[] {
  const out: string[] = [];
  for (const raw of plan.split("\n")) {
    const line = raw.replace(/^[\s>*`-]+/, "").trim();
    if (!line || line.length > 240) continue;
    if (
      /^(sudo\s+|apt(?:-get)?\s+|dnf\s+|yum\s+|brew\s+|snap\s+|winget\s+|choco\s+|unity)/i.test(line) ||
      /\b(apt(?:-get)?\s+install|unity\s*hub|installer)\b/i.test(line)
    ) {
      if (!out.includes(line)) out.push(line);
    }
    if (out.length >= 12) break;
  }
  return out;
}

export function systemWideNote(message: string): string | undefined {
  if (!wantsHostInstall(message)) return undefined;
  if (UNITY.test(message)) {
    return "This plan needs a system-wide Unity / Unity Hub install (host packages, not only repo files). Nothing runs until you click Approve. After Approve, writes stay inside the allowlisted cwd; host installs still require this confirmation.";
  }
  return "This plan wants host package installs (apt, sudo, or similar). Nothing runs until you click Approve. After Approve, file writes stay inside the allowlisted cwd.";
}

export function buildPendingApproval(input: {
  decision: RouteDecision;
  userMessage: string;
  planText: string;
  cwd?: string;
  extraContext?: string;
  prUrl?: string;
  repoUrl?: string;
  branch?: string;
  pin?: string;
}): PendingApproval {
  const closer = input.decision.closer ?? input.decision.speakers?.[0];
  const systemWide = Boolean(input.decision.needsHostInstall) || wantsHostInstall(input.userMessage);
  const commands = extractProposedCommands(input.planText);
  if (systemWide && UNITY.test(input.userMessage) && !commands.some((c) => /unity/i.test(c))) {
    commands.unshift("Install Unity Hub / Unity (system-wide; not executed until Approve)");
  }
  return {
    id: crypto.randomUUID(),
    status: "pending",
    cwd: input.cwd ?? input.decision.cwd,
    specialist: closer?.specialist ?? "builder",
    backendId: closer?.backendId ?? "cursor-local",
    label: closer?.label ?? "Cursor local",
    userMessage: input.userMessage,
    summary: input.planText.trim() || input.userMessage,
    commands,
    systemWideInstall: systemWide,
    systemWideNote: systemWideNote(input.userMessage),
    pin: input.pin,
    extraContext: input.extraContext,
    prUrl: input.prUrl,
    repoUrl: input.repoUrl,
    branch: input.branch,
    createdAt: Date.now(),
  };
}

export function pendingCardText(pending: PendingApproval): string {
  const lines = [
    "Pending actions — waiting for Approve.",
    `Proposed cwd: ${pending.cwd ?? "(default workspace)"}`,
    `Specialist: ${pending.specialist} (${pending.label} / ${pending.backendId})`,
    "",
    pending.summary,
  ];
  if (pending.commands.length) {
    lines.push("", "Proposed commands:", ...pending.commands.map((c) => `• ${c}`));
  }
  if (pending.systemWideNote) {
    lines.push("", pending.systemWideNote);
  } else {
    lines.push("", "Q&A and debate text already ran plan-only. File writes and host installs stay blocked until Approve.");
  }
  return lines.join("\n");
}
