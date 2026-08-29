import { homedir } from "node:os";
import { join } from "node:path";
import { ensureSecureDir, writeSecureFile } from "../platform.js";
import { stateDir } from "../state.js";

export const MCP_URL_BASENAME = "mcp.url";
export const MCP_GUI_URL_BASENAME = "mcp.gui.url";
export const MCP_HTTP_URL_BASENAME = "mcp.http.url";

/** XDG (or ~/.config) so Late can find the bound /mcp URL without a hardcoded repo path. */
export function xdgOrchestratorDir(env: NodeJS.Dict<string> = process.env, home = homedir()): string {
  const xdg = env.XDG_CONFIG_HOME?.trim();
  if (xdg) return join(xdg, "agent-orchestrator");
  return join(home, ".config", "agent-orchestrator");
}

function kindBasename(kind: "gui" | "http" | undefined): string | undefined {
  if (kind === "gui") return MCP_GUI_URL_BASENAME;
  if (kind === "http") return MCP_HTTP_URL_BASENAME;
  return undefined;
}

function tryWrite(dir: string, name: string, line: string): void {
  ensureSecureDir(dir);
  writeSecureFile(join(dir, name), line);
}

/**
 * Write the bound Streamable HTTP URL. Not a default port.
 * `mcp.url` is last-writer. GUI and dedicated mcp:http also write their own files so both survive.
 * GUI still starts if a write fails.
 */
export function writeAdvertisedMcpUrl(
  mcpUrl: string,
  opts?: { stateDir?: string; env?: NodeJS.Dict<string>; home?: string; kind?: "gui" | "http" },
): void {
  const line = `${mcpUrl.trim()}\n`;
  const names = [MCP_URL_BASENAME, kindBasename(opts?.kind)].filter((n): n is string => Boolean(n));
  try {
    const state = opts?.stateDir ?? stateDir();
    for (const name of names) tryWrite(state, name, line);
  } catch (err) {
    console.error(
      `Could not write ${MCP_URL_BASENAME} in state dir: ${err instanceof Error ? err.message : err}`,
    );
  }
  try {
    const xdgDir = xdgOrchestratorDir(opts?.env ?? process.env, opts?.home ?? homedir());
    for (const name of names) tryWrite(xdgDir, name, line);
  } catch (err) {
    console.error(
      `Could not advertise MCP URL for Late: ${err instanceof Error ? err.message : err}`,
    );
  }
}
