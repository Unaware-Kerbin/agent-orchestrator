import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
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
  const css = readFileSync(join(root, "gui/public/styles.css"), "utf8");
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
  const js = readFileSync(join(root, "gui/public/app.js"), "utf8");
  assert.ok(js.includes('THEME_KEY = "orchestrator.gui.theme"'));
  assert.ok(js.includes('DEFAULT_THEME = "grove"'));
  assert.ok(js.includes("isThemeId(id) ? id : DEFAULT_THEME"));
  for (const id of THEME_IDS) {
    assert.ok(js.includes(`id: "${id}"`), `app.js missing theme id ${id}`);
  }
});

test("index.html applies stored theme before stylesheet paint", () => {
  const html = readFileSync(join(root, "gui/public/index.html"), "utf8");
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
  const js = readFileSync(join(root, "gui/public/app.js"), "utf8");
  assert.ok(js.includes("https://huggingface.co/settings/tokens"));
  assert.ok(js.includes('data-name="HF_TOKEN"'));
  assert.ok(js.includes('data-clear-secret="HF_TOKEN"'));
  assert.ok(js.includes("Gemma Terms of Use"));
  assert.ok(js.includes('method: "DELETE"'));
  assert.equal(js.includes("oauth/callback"), false);
  assert.equal(js.includes("0.0.0.0"), false);
  const readme = readFileSync(join(root, "README.md"), "utf8");
  assert.ok(readme.includes("Gemma Terms of Use"));
  assert.ok(readme.includes("Settings → Local models"));
  assert.ok(readme.includes("huggingface.co/settings/tokens"));
  assert.equal(/\bhf_[A-Za-z0-9]{16,}\b/.test(readme), false);
});
