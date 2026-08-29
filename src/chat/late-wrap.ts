import type { PatchFile } from "./apply-patch.js";

/** Late Agent=MCP wrap: speakers must emit one Late JSON tool, not a repo lecture. */
export const LATE_JSON_SYSTEM = `You are answering Late on the operator's computer (Agent=MCP).
Reply with a single JSON object and no other prose. Do not wrap it in a lecture. Markdown fences are ok only if the object is the entire fenced body.
Do not cite orchestrator source files (src/chat/service.ts, timeout.ts, test/chat-busy.test.ts). Do not explain debate, flush, wrap, or routing internals.
session_id must be the live UUID after id= in UNTRUSTED DEVICE OUTPUT. Never invent aos-cx, a hostname, or a nickname as session_id.
If the operator asked for Ansible, a playbook, Netmiko, Salt, or Chef, call propose_staged_artifact with that format (never format=cli). Omit body so Late fills the vendor template. format=cli is one-line CLI only.
AOS-CX (Aruba 6200) is not Cisco IOS.
Example playbook: {"tool":"propose_staged_artifact","format":"ansible","intent":"configure VLAN 2000","session_id":"<uuid from id=>"}
Example show: {"tool":"propose_command","session_id":"<uuid from id=>","command":"show vlan","reason":"need vlan table","intent":"investigate"}
Allowed tools: propose_command, propose_api_get, propose_staged_artifact, list_open_sessions, read_scrollback, query_pcap, ask_user.`;

const VLAN_RE = /\bvlan\s*(?:id\s*)?(?:of\s*)?(\d{1,4})\b/i;

export function operatorWantsPlaybook(turn: string): boolean {
  return /\bansible\b|\bplaybook\b/i.test(turn);
}

export function vlanIdFromText(text: string): string | undefined {
  const m = text.match(VLAN_RE);
  if (!m?.[1]) return undefined;
  const n = Number(m[1]);
  if (!Number.isInteger(n) || n < 1 || n > 4094) return undefined;
  return String(n);
}

export function aosCxVlanCli(vlanId: string): string {
  return `vlan ${vlanId}\nname VLAN${vlanId}`;
}

export function aosCxVlanPlaybookYaml(intent: string, vlanId?: string): string {
  const id = vlanId ?? "2000";
  const cli = aosCxVlanCli(id)
    .split("\n")
    .map((line) => `          ${line}`)
    .join("\n");
  return `---
- name: ${intent.replace(/[\n\r:]/g, " ").trim() || `configure VLAN ${id}`}
  hosts: late_targets
  gather_facts: false
  vars:
    ansible_network_os: arubanetworks.aoscx
    ansible_connection: network_cli
  tasks:
    - name: apply VLAN ${id}
      ansible.netcommon.cli_config:
        config: |
${cli}
`;
}

function extractStagedBody(text: string): string | undefined {
  const objects: string[] = [];
  let depth = 0;
  let start = -1;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === "{") {
      if (depth === 0) start = i;
      depth++;
    } else if (c === "}" && depth > 0) {
      depth--;
      if (depth === 0 && start >= 0) {
        objects.push(text.slice(start, i + 1));
        start = -1;
      }
    }
  }
  for (const raw of objects) {
    try {
      const obj = JSON.parse(raw) as Record<string, unknown>;
      if (obj.tool !== "propose_staged_artifact" && obj.name !== "propose_staged_artifact") continue;
      const body = typeof obj.body === "string" ? obj.body : "";
      if (/\bhosts\s*:/.test(body) && /vlan\s+\d+/i.test(body) && !/#\s*replace with vendor syntax/i.test(body)) {
        return body.endsWith("\n") ? body : `${body}\n`;
      }
    } catch {
      /* skip */
    }
  }
  return undefined;
}

/** Playbook files for apply-patch after Approve when a folder was granted. */
export function latePlaybookPatchFiles(operatorTurn: string, transcriptTexts: string[]): PatchFile[] {
  if (!operatorWantsPlaybook(operatorTurn)) return [];
  const vlanId = vlanIdFromText(operatorTurn);
  const fromModel = transcriptTexts.map(extractStagedBody).find(Boolean);
  const content = fromModel ?? aosCxVlanPlaybookYaml(operatorTurn.trim() || "configure VLAN", vlanId);
  if (/#\s*replace with vendor syntax/i.test(content) || /\bPLACEHOLDER\b/.test(content)) {
    const filled = aosCxVlanPlaybookYaml(operatorTurn.trim() || "configure VLAN", vlanId);
    const path = vlanId ? `playbooks/vlan-${vlanId}.yml` : "playbooks/vlan.yml";
    return [{ path, content: filled.endsWith("\n") ? filled : `${filled}\n` }];
  }
  const path = vlanId ? `playbooks/vlan-${vlanId}.yml` : "playbooks/vlan.yml";
  return [{ path, content: content.endsWith("\n") ? content : `${content}\n` }];
}

export function withOrchestratorFilesFence(plan: string, files: PatchFile[]): string {
  if (!files.length) return plan;
  return `${plan.trim()}\n\n\`\`\`orchestrator-files\n${JSON.stringify({ files })}\n\`\`\`\n`;
}
