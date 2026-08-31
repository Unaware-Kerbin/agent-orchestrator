import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { assertNoMachineHome, assertRepoHasNoMachineHome, REPO_ROOT } from "./machine-paths.js";

const THEME_IDS = ["grove", "noir", "linen", "harbor", "ember", "paper"] as const;
const REQUIRED_VARS = [
  "--bg",
  "--bg-2",
  "--panel",
  "--line",
  "--text",
  "--muted",
  "--accent",
  "--accent-2",
  "--ok",
  "--warn",
  "--bad",
  "--accent-ink",
  "--accent-soft",
  "--user-bubble",
];

test("GUI CSS defines data-theme palettes with required variables", () => {
  const css = readFileSync(join(REPO_ROOT, "gui/public/styles.css"), "utf8");
  for (const id of THEME_IDS) {
    const marker = `[data-theme="${id}"]`;
    const start = css.indexOf(marker);
    assert.ok(start >= 0, `missing ${marker}`);
    const open = css.indexOf("{", start);
    const close = css.indexOf("}", open);
    assert.ok(open > start && close > open, `${id} theme block is incomplete`);
    const block = css.slice(open, close + 1);
    for (const name of REQUIRED_VARS) {
      assert.ok(block.includes(name), `${id} missing ${name}`);
    }
  }
});

test("GUI JS lists theme ids, storage key, and Grove fallback", () => {
  const js = readFileSync(join(REPO_ROOT, "gui/public/app.js"), "utf8");
  assert.ok(js.includes('THEME_KEY = "orchestrator.gui.theme"'));
  assert.ok(js.includes('DEFAULT_THEME = "grove"'));
  assert.ok(js.includes("isThemeId(id) ? id : DEFAULT_THEME"));
  for (const id of THEME_IDS) {
    assert.ok(js.includes(`id: "${id}"`), `app.js missing theme id ${id}`);
  }
});

test("index.html applies stored theme before stylesheet paint", () => {
  const html = readFileSync(join(REPO_ROOT, "gui/public/index.html"), "utf8");
  const headEnd = html.indexOf("</head>");
  const head = html.slice(0, headEnd);
  const scriptAt = head.indexOf("<script>");
  const cssAt = head.indexOf('href="/styles.css"');
  assert.ok(scriptAt >= 0 && cssAt > scriptAt, "theme boot script must precede stylesheet");
  assert.ok(head.includes("orchestrator.gui.theme"));
  assert.ok(head.includes("data-theme"));
  assert.ok(html.includes('id="theme-select-rail"'));
  for (const id of THEME_IDS) {
    assert.ok(head.includes(`"${id}"`) || html.includes(`value="${id}"`), `index.html missing ${id}`);
  }
});

test("Local models Settings UI pastes HF_TOKEN and links to the Hub token page", () => {
  const js = readFileSync(join(REPO_ROOT, "gui/public/app.js"), "utf8");
  assert.ok(js.includes("https://huggingface.co/settings/tokens"));
  assert.ok(js.includes('data-name="HF_TOKEN"'));
  assert.ok(js.includes('data-clear-secret="HF_TOKEN"'));
  assert.ok(js.includes("On this computer"));
  assert.ok(js.includes("Not downloaded"));
  assert.ok(js.includes("leftover YAML rows that are not serving do not join"));
  assert.ok(js.includes('id="model-search"'));
  assert.ok(js.includes("matchesQuery"));
  assert.ok(js.includes("hfRepo"));
  assert.ok(js.includes("data-delete-thread"));
  assert.ok(js.includes('id="grant-card"'));
  assert.ok(js.includes("data-grant-folder"));
  assert.ok(js.includes('addEventListener("paste"'));
  assert.ok(js.includes("looksLikeAbsPath"));
  assert.ok(js.includes("/api/chats/"));
  assert.ok(js.includes("workspaceDir"));
  assert.ok(js.includes('method: "DELETE"'));
  const checked = spawnSync(process.execPath, ["--check", join(REPO_ROOT, "gui/public/app.js")], { encoding: "utf8" });
  assert.equal(checked.status, 0, checked.stderr || checked.stdout);
  assert.equal(js.includes("oauth/callback"), false);
  assert.equal(js.includes("0.0.0.0"), false);
  const readme = readFileSync(join(REPO_ROOT, "README.md"), "utf8");
  assert.ok(readme.includes("Gemma Terms of Use"));
  assert.ok(readme.includes("Settings → Local models"));
  assert.ok(readme.includes("huggingface.co/settings/tokens"));
  assert.equal(/\bhf_[A-Za-z0-9]{16,}\b/.test(readme), false);
});

test("GUI Settings copies session mcpUrl for Late; bound port, not a machine home path", () => {
  const js = readFileSync(join(REPO_ROOT, "gui/public/app.js"), "utf8");
  assert.ok(js.includes("data-copy-mcp"));
  assert.ok(js.includes("mcpUrlForLate"));
  assert.ok(js.includes("sessionInfo.mcpUrl"));
  assert.ok(js.includes("mcp-listen-host"));
  assert.ok(js.includes("Copy MCP URL"));
  assert.ok(js.includes("Late works") || js.includes("does not need this server"));
  assert.equal(js.includes("http://${escapeHtml(location.host)}/mcp"), false);
  assertNoMachineHome(js, "gui/public/app.js");
  const readme = readFileSync(join(REPO_ROOT, "README.md"), "utf8");
  assert.match(readme, /printed/i);
  assertNoMachineHome(readme, "README.md");
  const envEx = readFileSync(join(REPO_ROOT, ".env.example"), "utf8");
  assertNoMachineHome(envEx, ".env.example");
});

test("repo text files do not embed this computer's home path", () => {
  assertRepoHasNoMachineHome();
});
