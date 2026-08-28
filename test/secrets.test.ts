import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { gatedRepoHint, hfDownloadChildEnv, hfTokenPresent } from "../src/local-models/download.js";
import { SECURE_FILE_MODE } from "../src/platform.js";
import { redactSecretText } from "../src/redact.js";
import {
  deleteSecrets,
  hfTokenConfigured,
  loadSecretsFile,
  resolveHfToken,
  secretStatus,
  secretsPath,
  upsertSecrets,
} from "../src/secrets.js";

const TOKEN = "hf_testAuditTokenNotReal9x7k2m";

function withIsolatedSecrets(fn: () => void): void {
  const dir = mkdtempSync(join(tmpdir(), "orch-secrets-"));
  const prevState = process.env.AGENT_ORCHESTRATOR_STATE_DIR;
  const prevHf = process.env.HF_TOKEN;
  const prevHub = process.env.HUGGING_FACE_HUB_TOKEN;
  process.env.AGENT_ORCHESTRATOR_STATE_DIR = dir;
  delete process.env.HF_TOKEN;
  delete process.env.HUGGING_FACE_HUB_TOKEN;
  try {
    fn();
  } finally {
    if (prevState === undefined) delete process.env.AGENT_ORCHESTRATOR_STATE_DIR;
    else process.env.AGENT_ORCHESTRATOR_STATE_DIR = prevState;
    if (prevHf === undefined) delete process.env.HF_TOKEN;
    else process.env.HF_TOKEN = prevHf;
    if (prevHub === undefined) delete process.env.HUGGING_FACE_HUB_TOKEN;
    else process.env.HUGGING_FACE_HUB_TOKEN = prevHub;
    rmSync(dir, { recursive: true, force: true });
  }
}

test("HF_TOKEN is stored write-only: status has set, never the raw value", () => {
  withIsolatedSecrets(() => {
    assert.equal(hfTokenConfigured(), false);
    upsertSecrets({ HF_TOKEN: TOKEN });
    const status = secretStatus(["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY"]);
    const hf = status.find((row) => row.name === "HF_TOKEN");
    assert.equal(hf?.set, true);
    const serialized = JSON.stringify(status);
    assert.equal(serialized.includes(TOKEN), false);
    assert.equal(JSON.stringify(status).includes("hf_test"), false);
    const stored = loadSecretsFile();
    assert.equal(stored.HF_TOKEN, TOKEN);
    if (process.platform !== "win32") {
      assert.equal(statSync(secretsPath()).mode & 0o777, SECURE_FILE_MODE);
    }
    assert.equal(readFileSync(secretsPath(), "utf8").includes(TOKEN), true);
  });
});

test("HUGGING_FACE_HUB_TOKEN in the secrets file counts as configured", () => {
  withIsolatedSecrets(() => {
    upsertSecrets({ HUGGING_FACE_HUB_TOKEN: TOKEN });
    assert.equal(hfTokenConfigured(), true);
    assert.equal(hfTokenPresent(), true);
    assert.equal(resolveHfToken(), TOKEN);
  });
});

test("clear HF_TOKEN removes both Hub names without logging the value", () => {
  withIsolatedSecrets(() => {
    upsertSecrets({ HF_TOKEN: TOKEN, HUGGING_FACE_HUB_TOKEN: TOKEN });
    const cleared = deleteSecrets(["HF_TOKEN"]);
    assert.ok(cleared.includes("HF_TOKEN"));
    assert.ok(cleared.includes("HUGGING_FACE_HUB_TOKEN"));
    assert.equal(hfTokenConfigured(), false);
    assert.equal(loadSecretsFile().HF_TOKEN, undefined);
    assert.equal(process.env.HF_TOKEN, undefined);
    assert.equal(process.env.HUGGING_FACE_HUB_TOKEN, undefined);
  });
});

test("download child env loads HF_TOKEN from the secrets file when process.env was cleared", () => {
  withIsolatedSecrets(() => {
    upsertSecrets({ HF_TOKEN: TOKEN });
    delete process.env.HF_TOKEN;
    assert.equal(process.env.HF_TOKEN, undefined);
    const env = hfDownloadChildEnv({ PYTHONPATH: "" });
    assert.equal(env.HF_TOKEN, TOKEN);
    assert.equal(env.PYTHONUNBUFFERED, "1");
    const leak = JSON.stringify({ env: { HF_TOKEN: redactSecretText(env.HF_TOKEN ?? "") } });
    assert.equal(leak.includes(TOKEN), false);
  });
});

test("gated download hint is helpful when unset and still redacts tokens", () => {
  withIsolatedSecrets(() => {
    const unset = gatedRepoHint(true);
    assert.match(unset, /Settings → Local models/);
    assert.match(unset, /HF_TOKEN/);
    assert.match(unset, /huggingface\.co\/settings\/tokens/);
    assert.doesNotMatch(unset, /0\.0\.0\.0/);
    assert.equal(gatedRepoHint(false), "");

    upsertSecrets({ HF_TOKEN: TOKEN });
    const denied = gatedRepoHint(true);
    assert.match(denied, /Accept the model license/);
    assert.equal(denied.includes(TOKEN), false);
    const leaked = redactSecretText(`401 Cannot access gated repo token=${TOKEN}`);
    assert.equal(leaked.includes(TOKEN), false);
    assert.match(leaked, /\*\*\*/);
  });
});
