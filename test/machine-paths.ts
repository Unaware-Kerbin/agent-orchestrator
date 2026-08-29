import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, relative, sep } from "node:path";
import { fileURLToPath } from "node:url";

export const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

const SKIP_DIR = new Set(["node_modules", ".git", ".orchestrator", "dist", "coverage"]);

/** This clone's home directory. Never paste it into shipped files or tests as a literal. */
export function machineHome(): string {
  return homedir();
}

export function assertNoMachineHome(text: string, label: string): void {
  const home = machineHome();
  if (!home) return;
  assert.equal(
    text.includes(home),
    false,
    `${label} must not embed this computer's home path (use cwd, env, or \${WORKSPACE_CWD})`,
  );
  const posix = home.split(sep).join("/");
  if (posix !== home) {
    assert.equal(text.includes(posix), false, `${label} must not embed this computer's home path`);
  }
}

function walkFiles(dir: string, acc: string[]): void {
  for (const name of readdirSync(dir)) {
    if (SKIP_DIR.has(name)) continue;
    const full = join(dir, name);
    const st = statSync(full);
    if (st.isDirectory()) {
      walkFiles(full, acc);
      continue;
    }
    if (/\.(ts|js|mjs|cjs|json|ya?ml|md|mdc|example|html|css)$/i.test(name) || name === ".env.example") {
      acc.push(full);
    }
  }
}

/** Shipped + test sources in this repo — fail if this machine's home leaked in. */
export function repoTextFiles(root = REPO_ROOT): string[] {
  const out: string[] = [];
  walkFiles(root, out);
  return out;
}

export function assertRepoHasNoMachineHome(root = REPO_ROOT): void {
  for (const file of repoTextFiles(root)) {
    const text = readFileSync(file, "utf8");
    assertNoMachineHome(text, relative(root, file));
  }
}
