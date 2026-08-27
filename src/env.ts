import { existsSync, readFileSync } from "node:fs";
import { packageRoot } from "./config.js";

export function parseEnvText(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const body = line.startsWith("export ") ? line.slice(7).trim() : line;
    const eq = body.indexOf("=");
    if (eq <= 0) continue;
    const key = body.slice(0, eq).trim();
    let value = body.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (key) out[key] = value;
  }
  return out;
}

function applyEnvMap(vars: Record<string, string>, overwrite: boolean): void {
  for (const [key, value] of Object.entries(vars)) {
    if (overwrite) {
      if (value) process.env[key] = value;
      continue;
    }
    if (!process.env[key]?.trim()) process.env[key] = value;
  }
}

export function loadEnvFile(path: string, overwrite = false): void {
  if (!existsSync(path)) return;
  applyEnvMap(parseEnvText(readFileSync(path, "utf8")), overwrite);
}

export function envFilePaths(): string[] {
  const root = `${packageRoot()}/.env`;
  const cwd = `${process.cwd()}/.env`;
  return root === cwd ? [root] : [root, cwd];
}

export function loadEnvFiles(overwrite = false): void {
  for (const path of envFilePaths()) loadEnvFile(path, overwrite);
}
