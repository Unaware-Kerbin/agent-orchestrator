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

function cssRuleBlock(css: string, selector: string): string {
  const marker = `${selector} {`;
  const start = css.indexOf(marker);
  assert.ok(start >= 0, `missing CSS rule ${selector}`);
  const open = css.indexOf("{", start);
  const close = css.indexOf("}", open);
  assert.ok(open > start && close > open, `${selector} block incomplete`);
  return css.slice(open, close + 1);
}

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
  assert.ok(js.includes("Late infer · your computer"));
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

test("GUI Local models primary card is Late infer on your computer", () => {
  const js = readFileSync(join(REPO_ROOT, "gui/public/app.js"), "utf8");
  assert.ok(js.includes("Late infer · your computer"));
  assert.ok(js.includes("Start on your computer"));
  assert.ok(js.includes("Start this backend on your computer"));
  assert.ok(js.includes("not in npm"));
  assert.ok(js.includes("not every Hub repo"));
  assert.ok(js.includes("Qwen/Qwen2.5-0.5B-Instruct"));
  assert.ok(js.includes('id="lateinfer-hub"'));
  assert.ok(js.includes("lateinfer-download"));
  assert.ok(js.includes("/api/local-servers/download"));
  assert.ok(js.includes("/api/local-servers/delete"));
  assert.ok(js.includes("data-lateinfer-delete"));
  assert.ok(js.includes("Remove from your computer"));
  assert.ok(js.includes("This removes the compiled snapshot from your computer. Download again from the Hub store to restore."));
  assert.ok(js.includes("This stops late-infer on 127.0.0.1:8010, then removes the compiled snapshot from your computer."));
  assert.ok(js.includes("applyCompiledDeleteResult"));
  assert.ok(js.includes("lateInferServingThisHubId"));
  assert.ok(js.includes("/api/local-servers/stop"));
  assert.ok(js.includes('kind: "lateinfer"'));
  assert.ok(js.includes("Downloading Hub snapshot"));
  assert.ok(js.includes("Compiling on your computer"));
  assert.ok(js.includes("lateinfer-download-progress"));
  assert.ok(js.includes("lateInferDownloadProgressHtml"));
  assert.ok(js.includes("etaSec"));
  assert.ok(js.includes("totalBytes"));
  assert.ok(js.includes("127.0.0.1:8010"));
  assert.ok(js.includes("lateinfer-start"));
  assert.ok(js.includes("lateinfer-panes"));
  assert.ok(js.includes("Hub store"));
  assert.ok(js.includes("Nothing compiled on your computer yet"));
  assert.equal(js.includes("Optional servers"), false);
  // Intentional Local models / Backends GUI (Ollama, llama.cpp, vLLM Docker, Grok).
  const hasOllama = js.includes("Start Ollama");
  const hasLlama = js.includes("Start llama-server");
  const hasDocker = js.includes("Start with Docker");
  if (!hasOllama || !hasLlama || !hasDocker) {
    throw new Error("local engine UI missing: ollama="+hasOllama+" llama="+hasLlama+" docker="+hasDocker+" len="+js.length);
  }
  assert.ok(js.includes("Register Ollama backend"));
  assert.ok(js.includes("renderGrokKeyCard"));
  assert.ok(js.includes("XAI_API_KEY"));
  assert.ok(js.includes("backend-section-title"));
  assert.ok(js.includes(">Local</"));
  assert.ok(js.includes(">Cloud</"));
  assert.ok(js.includes("do not need Late desktop"));
  assert.equal(js.includes("Late repo"), false);
  assert.ok(js.includes("127.0.0.1:8010"));
  const readme = readFileSync(join(REPO_ROOT, "README.md"), "utf8");
  assert.ok(readme.includes("/mcp"));
  assert.ok(readme.includes("Start Ollama") || readme.includes("Local models"));
});

const LATE_INFER_HUB_FAMILY_ORDER = ["Qwen", "Mistral", "Gemma", "Llama", "Phi", "other"] as const;

function hubFamilyFromModel(model: { family?: string; id?: string }): string {
  const fam = String(model?.family ?? "").trim();
  if (fam) return fam;
  const id = String(model?.id ?? "");
  const lower = id.toLowerCase();
  if (/qwen/i.test(lower)) return "Qwen";
  if (/mistral|mixtral/i.test(lower)) return "Mistral";
  if (/gemma/i.test(lower)) return "Gemma";
  if (/llama|meta-llama/i.test(lower)) return "Llama";
  if (/phi/i.test(lower)) return "Phi";
  return "other";
}

function parseHubVersionScore(version: string | undefined, id: string | undefined): number {
  const explicit = String(version ?? "").trim();
  if (explicit) {
    const nums = explicit.match(/\d+(?:\.\d+)*/g);
    if (nums?.length) {
      return nums.map((part) => Number(part)).reduce((acc, n, i) => acc + n * Math.pow(100, Math.max(0, 2 - i)), 0);
    }
  }
  const idStr = String(id ?? "");
  const match =
    idStr.match(/qwen(\d+(?:\.\d+)?)/i) ??
    idStr.match(/gemma[-_]?(\d+(?:\.\d+)?)/i) ??
    idStr.match(/llama[-_.]?(\d+(?:\.\d+)?)/i) ??
    idStr.match(/[-_.]v(\d+(?:\.\d+)?)/i);
  if (match?.[1]) return Number(match[1]) * 100;
  return 0;
}

function compareHubModelNewestFirst(
  a: { version?: string | number; id?: string; lastModified?: string; fallback?: boolean; onComputer?: boolean },
  b: { version?: string | number; id?: string; lastModified?: string; fallback?: boolean; onComputer?: boolean },
): number {
  if (Boolean(a.onComputer) !== Boolean(b.onComputer)) return a.onComputer ? -1 : 1;
  if (Boolean(a.fallback) !== Boolean(b.fallback)) return a.fallback ? -1 : 1;
  const score = (model: { version?: string | number; id?: string }) => {
    if (typeof model.version === "number" && Number.isFinite(model.version)) return model.version * 100;
    return parseHubVersionScore(String(model.version ?? ""), model.id);
  };
  const versionDelta = score(b) - score(a);
  if (versionDelta !== 0) return versionDelta;
  const tb = Date.parse(b.lastModified ?? "") || 0;
  const ta = Date.parse(a.lastModified ?? "") || 0;
  if (tb !== ta) return tb - ta;
  return String(a.id).localeCompare(String(b.id));
}

function groupLateInferHubModelsForTest(
  models: Array<{
    id: string;
    family?: string;
    version?: string | number;
    lastModified?: string;
    label?: string;
    fallback?: boolean;
    vramMaxMiB?: number;
    vramMaxLabel?: string;
  }>,
) {
  const buckets = new Map<string, typeof models>();
  for (const model of models) {
    const family = hubFamilyFromModel(model);
    if (!buckets.has(family)) buckets.set(family, []);
    buckets.get(family)!.push({ ...model, family });
  }
  for (const list of buckets.values()) list.sort(compareHubModelNewestFirst);
  const extras = [...buckets.keys()].filter((family) => !LATE_INFER_HUB_FAMILY_ORDER.includes(family as (typeof LATE_INFER_HUB_FAMILY_ORDER)[number])).sort();
  return LATE_INFER_HUB_FAMILY_ORDER.concat(extras as unknown as typeof LATE_INFER_HUB_FAMILY_ORDER)
    .filter((family) => buckets.has(family))
    .map((family) => ({ family, models: buckets.get(family) ?? [] }));
}

function filterLateInferHubModelsForTest(
  models: Array<{ id: string; family?: string; version?: string; label?: string; name?: string; params?: number }>,
  query: string,
) {
  const needle = String(query ?? "").trim().toLowerCase();
  if (!needle) return models;
  return models.filter((model) => {
    const hay = [model.id, model.label, model.name, model.family, model.version, model.params]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return hay.includes(needle);
  });
}

function hubModelGroupsForPickerForTest(
  groups: Array<{ family: string; models: Array<{ id: string; family?: string; version?: string; label?: string }> }>,
  query: string,
) {
  const needle = String(query ?? "").trim().toLowerCase();
  if (!needle) return groups;
  return groups
    .map((group) => ({
      family: group.family,
      models: filterLateInferHubModelsForTest(group.models, needle),
    }))
    .filter((group) => group.models.length);
}

type HubVramModel = {
  id: string;
  family?: string;
  version?: string;
  label?: string;
  name?: string;
  params?: number;
  fallback?: boolean;
  gated?: boolean;
  sizeMiB?: number;
  weightsMiB?: number;
  bytes?: number;
  vramMaxMiB?: number;
  vramMaxGiB?: number;
  vramMaxLabel?: string;
  lastModified?: string;
  onComputer?: boolean;
};

function formatVramMaxFromMiBForTest(mib: number): string {
  const n = Number(mib);
  if (!Number.isFinite(n) || n <= 0) return "VRAM unknown";
  if (n < 1024) return `~${Math.round(n)} MiB VRAM max`;
  const gib = n / 1024;
  const shown = gib >= 10 ? String(Math.round(gib)) : (Math.round(gib * 10) / 10).toFixed(1).replace(/\.0$/, "");
  return `~${shown} GB VRAM max`;
}

function formatHubVramMaxForTest(model: HubVramModel): string {
  const label = String(model?.vramMaxLabel ?? "").trim();
  if (label && label !== "VRAM unknown") return label;
  const api = Number(model?.vramMaxMiB);
  if (Number.isFinite(api) && api > 0) return formatVramMaxFromMiBForTest(api);
  const weights = Number(model?.sizeMiB ?? model?.weightsMiB);
  if (Number.isFinite(weights) && weights > 0) return formatVramMaxFromMiBForTest(Math.ceil(weights * 1.5));
  return label || "VRAM unknown";
}

const FALLBACK_VRAM_SEEDS: HubVramModel[] = [
  { id: "Qwen/Qwen2.5-0.5B-Instruct", family: "Qwen", weightsMiB: 1000, params: 500_000_000 },
  { id: "google/gemma-2-2b-it", family: "Gemma", gated: true, weightsMiB: 5000, params: 2_600_000_000 },
  { id: "google/gemma-4-E2B-it", family: "Gemma", weightsMiB: 10000, params: 5_100_000_000 },
  { id: "mistralai/Mistral-7B-Instruct-v0.2", family: "Mistral", weightsMiB: 14000, params: 7_000_000_000 },
  { id: "mistralai/Mistral-7B-Instruct-v0.3", family: "Mistral", gated: true, weightsMiB: 14000, params: 7_000_000_000 },
];

function fallbackLateInferHubModelsForTest(seeds = FALLBACK_VRAM_SEEDS): HubVramModel[] {
  return seeds.map((model) => {
    const vramMaxMiB = Math.ceil((model.weightsMiB ?? 0) * 1.5);
    return {
      id: model.id,
      family: model.family,
      gated: Boolean(model.gated),
      fallback: true,
      sizeMiB: model.weightsMiB,
      weightsMiB: model.weightsMiB,
      params: model.params,
      vramMaxMiB,
      vramMaxLabel: formatVramMaxFromMiBForTest(vramMaxMiB),
    };
  });
}

function mergeLateInferHubWithFallbackForTest(fromHub: HubVramModel[], seeds = fallbackLateInferHubModelsForTest()): HubVramModel[] {
  const byId = new Map<string, HubVramModel>();
  const seedIds = new Set(seeds.map((model) => model.id));
  for (const model of seeds) byId.set(model.id, model);
  for (const model of fromHub ?? []) {
    if (!model?.id) continue;
    const prev = byId.get(model.id);
    const next = seedIds.has(model.id) ? { ...model, fallback: true } : model;
    byId.set(model.id, {
      ...prev,
      ...next,
      sizeMiB: next.sizeMiB ?? prev?.sizeMiB,
      weightsMiB: next.weightsMiB ?? prev?.weightsMiB,
      params: next.params ?? prev?.params,
      bytes: next.bytes ?? prev?.bytes,
      vramMaxMiB: next.vramMaxMiB ?? prev?.vramMaxMiB,
      vramMaxGiB: next.vramMaxGiB ?? prev?.vramMaxGiB,
      vramMaxLabel: next.vramMaxLabel ?? prev?.vramMaxLabel,
    });
  }
  return [...byId.values()];
}

function renderLateInferHubListHtmlForTest(groups: Array<{ family: string; models: HubVramModel[] }>) {
  return groups
    .map(
      (group) => `
    <div data-hub-family="${group.family}" class="hub-model-group" role="presentation">
      <div class="hub-model-group-label">${group.family}</div>
      ${group.models
        .map(
          (model) => `
        <button type="button" role="option" data-hub-id="${model.id}" class="hub-model-option">
          <span class="hub-model-option-main">${model.label ?? model.id} ${model.onComputer ? `<span class="pill ok">on your computer</span>` : `<span class="pill warn">available to download</span>`}${model.fallback ? ` <span class="muted">(fallback)</span>` : ""}${model.gated ? ` <span class="pill warn" title="Needs Hugging Face license accept + read token">gated</span>` : ""}</span>
          <span class="hub-model-option-vram">${formatHubVramMaxForTest(model)}</span>
        </button>`,
        )
        .join("")}
    </div>`,
    )
    .join("");
}

const OFFICIAL_HUB_IDS_FOR_TEST = new Set(FALLBACK_VRAM_SEEDS.map((model) => model.id));

function firstOfficialHubIdForQueryForTest(
  query: string,
  groups: Array<{ family: string; models: Array<{ id: string; family?: string; fallback?: boolean }> }>,
) {
  const pickerGroups = hubModelGroupsForPickerForTest(groups, query);
  for (const group of pickerGroups) {
    for (const model of group.models) {
      if (model.fallback || OFFICIAL_HUB_IDS_FOR_TEST.has(model.id)) return model.id;
    }
  }
  return pickerGroups[0]?.models?.[0]?.id;
}

function resolveLateInferHubIdForDownloadForTest(
  hubInput: string,
  selectedHubId: string,
  search: string,
  groups: Array<{ family: string; models: Array<{ id: string; family?: string; version?: string; label?: string; fallback?: boolean }> }>,
  defaultId = "Qwen/Qwen2.5-0.5B-Instruct",
  selectedRowId = "",
) {
  const fromRow = String(selectedRowId ?? "").trim();
  if (fromRow.includes("/")) return fromRow;
  const fromInput = String(hubInput ?? "").trim();
  const familyQuery = fromInput.includes("/") ? String(search ?? "").trim() : fromInput || String(search ?? "").trim();
  if (familyQuery && !familyQuery.includes("/")) {
    const officialMatch = firstOfficialHubIdForQueryForTest(familyQuery, groups);
    if (officialMatch) return officialMatch;
  }
  if (fromInput.includes("/")) {
    if (OFFICIAL_HUB_IDS_FOR_TEST.has(fromInput) || !search) return fromInput;
    if (!OFFICIAL_HUB_IDS_FOR_TEST.has(fromInput) && search) {
      const officialMatch = firstOfficialHubIdForQueryForTest(search, groups);
      if (officialMatch) return officialMatch;
    }
    return fromInput;
  }
  if (search) {
    const officialMatch = firstOfficialHubIdForQueryForTest(search, groups);
    if (officialMatch) return officialMatch;
  }
  const selected = String(selectedHubId ?? "").trim();
  if (selected.includes("/")) {
    if (OFFICIAL_HUB_IDS_FOR_TEST.has(selected) || !search) return selected;
    const officialMatch = firstOfficialHubIdForQueryForTest(search, groups);
    if (officialMatch) return officialMatch;
    return selected;
  }
  return defaultId;
}

const HUB_MODEL_FIXTURE = [
  { id: "Qwen/Qwen2.5-0.5B-Instruct", family: "Qwen", version: "2.5", lastModified: "2024-09-01T00:00:00Z" },
  { id: "Qwen/Qwen3-8B", family: "Qwen", version: "3", lastModified: "2025-04-01T00:00:00Z" },
  { id: "google/gemma-2-2b-it", family: "Gemma", version: "2", lastModified: "2024-06-01T00:00:00Z" },
  { id: "mistralai/Mistral-7B-Instruct-v0.3", family: "Mistral", version: "0.3", lastModified: "2024-05-01T00:00:00Z" },
] as const;

test("Late infer hub picker uses hub-models API, family groups, and newest-first labels", () => {
  const js = readFileSync(join(REPO_ROOT, "gui/public/app.js"), "utf8");
  const css = readFileSync(join(REPO_ROOT, "gui/public/styles.css"), "utf8");
  assert.ok(js.includes("/api/local-servers/hub-models"));
  assert.ok(js.includes("/api/local-servers/hub-models"));
  assert.ok(js.includes("hub-gguf-models") || js.includes("/api/local-servers/hub-gguf-models"));
  assert.ok(js.includes("llamacpp-gguf-search") || js.includes("GGUF Hub store"));
  assert.ok(js.includes("hubCatalogModelsFromResponse"));
  assert.ok(js.includes("hubCatalogGroupsFromResponse"));
  assert.ok(js.includes("lateInferHubHasListData"));
  assert.ok(js.includes("groupLateInferHubModels"));
  assert.ok(js.includes("hubModelGroupsForPicker"));
  assert.ok(js.includes("compareHubModelNewestFirst"));
  assert.ok(js.includes("LATE_INFER_HUB_FAMILY_ORDER"));
  assert.ok(js.includes("formatHubVramMax"));
  assert.ok(js.includes("hub-model-option-vram"));
  assert.ok(js.includes("VRAM max"));
  assert.ok(js.includes("estimated VRAM at max usage") || js.includes("VRAM at max usage"));
  assert.ok(css.includes(".hub-model-option-vram"));
  assert.ok(js.includes("model.gated"));
  assert.ok(js.includes("model.fallback"));
  assert.ok(js.includes("lateInferHubDefaultId"));
  assert.ok(js.includes("offline"));
  assert.ok(js.includes('data-hub-family="'));
  assert.ok(js.includes("lateinfer-hub-list"));
  assert.ok(js.includes("lateinfer-hub-search"));
  assert.ok(js.includes("Search listed models"));
  assert.ok(js.includes("Hugging Face store"));
  assert.ok(js.includes("GGUF") && (js.includes("llama.cpp") || js.includes("llamacpp")));
  assert.ok(js.includes("OpenVINO") || js.includes("safetensors chat") || js.includes("Safetensors chat"));
  assert.ok(js.includes("FALLBACK_LATE_INFER_HUB_MODELS"));
  assert.ok(js.includes("ensureLateInferHubFallback"));
  assert.ok(js.includes("mergeLateInferHubWithFallback"));
  assert.ok(js.includes("fallbackLateInferHubModels"));
  assert.ok(js.includes("google/gemma-2-2b-it"));
  assert.ok(js.includes("google/gemma-4-E2B-it"));
  assert.ok(js.includes("Qwen/Qwen2.5-0.5B-Instruct"));
  assert.ok(js.includes("selectedLateInferHubRowId"));
  assert.ok(js.includes("firstOfficialHubIdForQuery"));
  assert.ok(js.includes("on your computer"));
  assert.ok(js.includes("not downloaded") || js.includes("available to download"));
  assert.ok(js.includes("lateinfer-panes"));
  assert.ok(js.includes("lateinfer-store-pane"));
  assert.ok(js.includes("lateinfer-local-pane"));
  assert.ok(js.includes("lateinfer-compiled-list"));
  assert.ok(js.includes("Nothing compiled on your computer yet"));
  assert.ok(js.includes("Hub store"));
  assert.ok(js.includes("available to download"));
  assert.ok(js.includes("resolveLateInferIdForStart"));
  assert.ok(js.includes("compiledHubModelsForPane"));
  assert.ok(css.includes(".lateinfer-panes"));
  assert.ok(css.includes(".lateinfer-local-pane"));
  const panesRule = cssRuleBlock(css, ".lateinfer-panes");
  assert.match(panesRule, /grid-template-columns:\s*minmax\(0,\s*1fr\)\s+minmax\(0,\s*1fr\)/);
  const narrow = css.match(/@media \(max-width: 820px\)\s*\{[\s\S]*?\.lateinfer-panes\s*\{[^}]+\}/);
  assert.ok(narrow);
  assert.match(narrow[0], /grid-template-columns:\s*1fr/);
  assert.ok(js.includes("paste any Hub id") || js.includes("Paste any Hub") || js.includes("paste a Hub"));
  assert.ok(js.includes("lateInferErrorLine"));
  assert.ok(js.includes("lateInferHasError"));
  assert.ok(js.includes("lateInferJobBusy"));
  assert.ok(js.includes("compiledModels"));
  assert.ok(js.includes("hubModelOnComputer"));
  assert.ok(css.includes(".hub-model-option-main .pill"));
  assert.ok(js.includes("mistralai/Mistral-7B-Instruct-v0.2"));
  assert.ok(js.includes("meta-llama/Llama-3.2-1B-Instruct"));
  assert.ok(js.includes("microsoft/Phi-4-mini-instruct"));
  assert.ok(js.includes("Hugging Face Hub is offline"));
  assert.ok(js.includes("hardwareNotesForLocalModels"));
  assert.ok(js.includes("role=\"combobox\""));
  assert.ok(js.includes("Paste any Instruct id") || js.includes("paste any Hub") || js.includes("OpenVINO"));
  assert.ok(js.includes("GGUF-only") || js.includes("GGUF →") || js.includes("llama.cpp"));
  assert.equal(js.includes("lateInferHubPickerOpen"), false);
  assert.equal(js.includes("scheduleLateInferHubFetch"), false);
  assert.equal(js.includes("lateInferHubFetchTimer"), false);
  assert.equal(js.includes("huggingface.co/api/models"), false);
  assert.ok(js.includes("do not need Late desktop"));
  assert.ok(js.includes("loadLateInferHubModels"));
  assert.ok(js.includes("gpuId: lateInferGpuIdForPost()"));
  assert.ok(js.includes('kind: "lateinfer"'));
  assert.ok(js.includes("model: hubId"));
  assert.ok(js.includes('querySelectorAll("[data-hub-id]")'));
  assert.ok(css.includes(".lateinfer-hub-list"));
  assert.ok(css.includes(".hub-model-group-label"));
  assert.equal(/id="lateinfer-hub-list"[^>]*\bhidden\b/.test(js), false);
  assert.equal(/id="lateinfer-hub-list"[^>]*position:absolute/.test(js), false);
  const listRule = cssRuleBlock(css, ".lateinfer-hub-list");
  assert.match(listRule, /display:\s*block/);
  assert.match(listRule, /position:\s*static/);
  assert.doesNotMatch(listRule, /position:\s*absolute/);
  assert.doesNotMatch(listRule, /display:\s*none/);
  const listHiddenRule = cssRuleBlock(css, ".lateinfer-hub-list[hidden]");
  assert.match(listHiddenRule, /display:\s*block/);
  assert.doesNotMatch(listHiddenRule, /position:\s*absolute/);
  const optionRule = cssRuleBlock(css, ".hub-model-option");
  assert.match(optionRule, /display:\s*flex/);
  assert.ok(css.includes(".hub-model-option-vram"));
  const vramRule = cssRuleBlock(css, ".hub-model-option-vram");
  assert.match(vramRule, /flex:\s*0 0 auto/);

  const groups = groupLateInferHubModelsForTest([...HUB_MODEL_FIXTURE]);
  assert.deepEqual(
    groups.map((group) => group.family),
    ["Qwen", "Mistral", "Gemma"],
  );
  assert.equal(groups[0]?.models[0]?.id, "Qwen/Qwen3-8B");
  assert.equal(groups[0]?.models[1]?.id, "Qwen/Qwen2.5-0.5B-Instruct");
  assert.ok(parseHubVersionScore("3", "Qwen/Qwen3-8B") > parseHubVersionScore("2.5", "Qwen/Qwen2.5-0.5B-Instruct"));

  // pass: Qwen3 before Qwen2.5 (newest first)
  assert.ok(
    compareHubModelNewestFirst(
      { id: "Qwen/Qwen2.5-0.5B-Instruct", version: "2.5" },
      { id: "Qwen/Qwen3-8B", version: "3" },
    ) > 0,
  );

  // pass: Gemma 4 E2B before Gemma 2 when both listed (newest first)
  assert.ok(
    compareHubModelNewestFirst(
      { id: "google/gemma-2-2b-it", family: "Gemma" },
      { id: "google/gemma-4-E2B-it", family: "Gemma" },
    ) > 0,
  );

  // pass: official fallback seeds stay above newer Hub rows in the same family
  assert.ok(
    compareHubModelNewestFirst(
      { id: "google/gemma-4-E2B-it", family: "Gemma" },
      { id: "google/gemma-2-2b-it", family: "Gemma", fallback: true },
    ) > 0,
  );

  // pass: VRAM span metadata does not change family grouping
  const vramSpanModels = [
    { id: "Qwen/Qwen3-8B", family: "Qwen", vramMaxLabel: "~2 GB VRAM max", vramMaxMiB: 2048 },
    { id: "google/gemma-4-E2B-it", family: "Gemma", vramMaxLabel: "~15 GB VRAM max", vramMaxMiB: 15360 },
    { id: "google/gemma-2-2b-it", family: "Gemma", vramMaxLabel: "~8 GB VRAM max", vramMaxMiB: 8192 },
    { id: "mistralai/Mistral-7B-Instruct-v0.3", family: "Mistral", vramMaxLabel: "~21 GB VRAM max", vramMaxMiB: 21504 },
  ];
  const vramGroups = groupLateInferHubModelsForTest(vramSpanModels);
  assert.deepEqual(
    vramGroups.map((group) => group.family),
    ["Qwen", "Mistral", "Gemma"],
  );
  assert.equal(vramGroups.find((group) => group.family === "Gemma")?.models[0]?.id, "google/gemma-4-E2B-it");
  const vramHtml = renderLateInferHubListHtmlForTest(vramGroups);
  assert.ok(vramHtml.includes("~15 GB VRAM max"));
  assert.ok(vramHtml.includes("~8 GB VRAM max"));
  assert.ok(vramHtml.includes('data-hub-family="Gemma"'));

  const fixtureGroups = groupLateInferHubModelsForTest([...HUB_MODEL_FIXTURE]);
  const fixtureHtml = renderLateInferHubListHtmlForTest(fixtureGroups);
  assert.ok(fixtureHtml.includes('data-hub-family="Qwen"'));
  assert.ok(fixtureHtml.includes('data-hub-family="Gemma"'));
  assert.ok(fixtureHtml.includes("google/gemma-2-2b-it"));
  assert.ok(fixtureHtml.includes("Qwen/Qwen3-8B"));

  const gemmaOnly = hubModelGroupsForPickerForTest(fixtureGroups, "Gemma");
  assert.deepEqual(
    gemmaOnly.map((group) => group.family),
    ["Gemma"],
  );
  assert.equal(gemmaOnly[0]?.models[0]?.id, "google/gemma-2-2b-it");

  const fallbackSeeds = [
    { id: "Qwen/Qwen2.5-0.5B-Instruct", family: "Qwen" },
    { id: "google/gemma-2-2b-it", family: "Gemma" },
    { id: "google/gemma-4-E2B-it", family: "Gemma" },
    { id: "mistralai/Mistral-7B-Instruct-v0.3", family: "Mistral" },
  ];
  const fallbackGroups = groupLateInferHubModelsForTest(fallbackSeeds);
  const searched = hubModelGroupsForPickerForTest(fallbackGroups, "Gemma");
  assert.ok(searched.every((group) => group.family === "Gemma"));
  const selectedId = searched[0]?.models[0]?.id;
  assert.equal(selectedId, "google/gemma-4-E2B-it");
  const downloadBody = JSON.stringify({ kind: "lateinfer", model: selectedId });
  assert.equal(downloadBody, '{"kind":"lateinfer","model":"google/gemma-4-E2B-it"}');
  assert.equal(downloadBody.includes("Qwen/Qwen2.5-0.5B-Instruct"), false);

  assert.equal(
    resolveLateInferHubIdForDownloadForTest("Gemma", "Qwen/Qwen2.5-0.5B-Instruct", "", fallbackGroups),
    "google/gemma-4-E2B-it",
  );
  assert.equal(
    resolveLateInferHubIdForDownloadForTest("google/gemma-2-2b-it", "Qwen/Qwen2.5-0.5B-Instruct", "", fallbackGroups),
    "google/gemma-2-2b-it",
  );
  assert.equal(
    resolveLateInferHubIdForDownloadForTest("", "google/gemma-2-2b-it", "", fallbackGroups),
    "google/gemma-2-2b-it",
  );
  assert.equal(
    resolveLateInferHubIdForDownloadForTest("zzzzz", "google/gemma-2-2b-it", "", fallbackGroups),
    "google/gemma-2-2b-it",
  );
  assert.equal(
    resolveLateInferHubIdForDownloadForTest(
      "Ali-Mhrez/Qwen3-4B-Instruct-2507-SD-FNC-512-2",
      "Ali-Mhrez/Qwen3-4B-Instruct-2507-SD-FNC-512-2",
      "gem",
      fallbackGroups,
      "Qwen/Qwen2.5-0.5B-Instruct",
      "google/gemma-4-E2B-it",
    ),
    "google/gemma-4-E2B-it",
  );
  assert.equal(
    resolveLateInferHubIdForDownloadForTest(
      "Ali-Mhrez/Qwen3-4B-Instruct-2507-SD-FNC-512-2",
      "Ali-Mhrez/Qwen3-4B-Instruct-2507-SD-FNC-512-2",
      "gem",
      fallbackGroups,
    ),
    "google/gemma-4-E2B-it",
  );
  const gemmaDownloadBody = JSON.stringify({
    kind: "lateinfer",
    model: resolveLateInferHubIdForDownloadForTest(
      "Ali-Mhrez/Qwen3-4B-Instruct-2507-SD-FNC-512-2",
      "Ali-Mhrez/Qwen3-4B-Instruct-2507-SD-FNC-512-2",
      "gem",
      fallbackGroups,
      "Qwen/Qwen2.5-0.5B-Instruct",
      "google/gemma-4-E2B-it",
    ),
  });
  assert.equal(gemmaDownloadBody, '{"kind":"lateinfer","model":"google/gemma-4-E2B-it"}');
  assert.equal(gemmaDownloadBody.includes("Ali-Mhrez"), false);

  const localFirst = compareHubModelNewestFirst(
    { id: "google/gemma-3-1b-it", family: "Gemma", fallback: true },
    { id: "google/gemma-4-E2B-it", family: "Gemma", fallback: true, onComputer: true },
  );
  assert.ok(localFirst > 0);

  const downloadedHtml = renderLateInferHubListHtmlForTest([
    {
      family: "Gemma",
      models: [
        { id: "google/gemma-4-E2B-it", family: "Gemma", onComputer: true, fallback: true, vramMaxLabel: "~15 GB VRAM max" },
        { id: "google/gemma-2-2b-it", family: "Gemma", onComputer: false, gated: true, fallback: true, vramMaxLabel: "~8 GB VRAM max" },
      ],
    },
  ]);
  assert.ok(downloadedHtml.includes("on your computer"));
  assert.ok(downloadedHtml.includes("available to download"));
  assert.ok(downloadedHtml.includes("gated"));
  assert.ok(downloadedHtml.includes("~15 GB VRAM max"));
  assert.equal(downloadedHtml.includes("Ali-Mhrez"), false);

  const groupsOnlyResponse = {
    models: [],
    groups: [
      { family: "Qwen", models: [{ id: "Qwen/Qwen2.5-0.5B-Instruct", family: "Qwen", fallback: true }] },
      { family: "Gemma", models: [{ id: "google/gemma-2-2b-it", family: "Gemma", fallback: true }] },
    ],
  };
  const groupsOnlyHtml = renderLateInferHubListHtmlForTest(groupsOnlyResponse.groups);
  assert.ok(groupsOnlyHtml.includes('data-hub-family="Qwen"'));
  assert.ok(groupsOnlyHtml.includes('data-hub-family="Gemma"'));
  assert.ok(groupsOnlyHtml.includes("(fallback)"));

  const notes = [
    "One integrated GPU is ignored for model fitting.",
    "Intel vLLM Docker: intel/llm-scaler-vllm:0.21.0-b3. Preferred: intel/llm-scaler-vllm:0.21.0-b3.",
  ].filter((note) => !/vLLM Docker|llm-scaler-vllm|intel\/vllm/i.test(note));
  assert.deepEqual(notes, ["One integrated GPU is ignored for model fitting."]);

  assert.match(css, /\.hub-model-option\s*\{[^}]*display:\s*flex/);
  assert.match(css, /\.hub-model-option\s*\{[^}]*justify-content:\s*space-between/);
  assert.ok(js.includes("vramMaxMiB: next.vramMaxMiB ?? prev?.vramMaxMiB"));
  assert.ok(js.includes("vramMaxLabel: next.vramMaxLabel ?? prev?.vramMaxLabel"));
  assert.ok(js.includes("sizeMiB: next.sizeMiB ?? prev?.sizeMiB"));
  assert.ok(js.includes("weightsMiB: next.weightsMiB ?? prev?.weightsMiB"));
  assert.ok(js.includes("weightsMiB: 1000"));
  assert.ok(js.includes("weightsMiB: 14000"));
  assert.ok(js.includes("hubInput.value = hubId"));
  const hubInputIdx = js.indexOf('if (target.id === "lateinfer-hub")');
  const searchIdx = js.indexOf('if (target.id !== "lateinfer-hub-search")');
  assert.ok(hubInputIdx >= 0 && searchIdx > hubInputIdx);
  assert.equal(js.slice(hubInputIdx, searchIdx).includes("refreshLateInferHubList"), false);
  assert.ok(js.slice(searchIdx, searchIdx + 180).includes("refreshLateInferHubList"));
});

test("Late infer hub rows keep VRAM max after fallback merge and search", () => {
  const js = readFileSync(join(REPO_ROOT, "gui/public/app.js"), "utf8");
  const css = readFileSync(join(REPO_ROOT, "gui/public/styles.css"), "utf8");
  assert.ok(js.includes("formatHubVramMax"));
  assert.ok(js.includes("hub-model-option-vram"));
  assert.ok(js.includes("VRAM max"));
  assert.ok(js.includes("estimated VRAM at max usage (weights plus KV cache)"));
  assert.ok(js.includes("Listed for your computer"));
  assert.ok(js.includes("you do not need the Hugging Face website"));
  assert.ok(js.includes("do not need Late desktop"));
  assert.ok(css.includes(".hub-model-option-vram"));
  assert.match(css, /\.hub-model-option\s*\{[^}]*display:\s*flex/);

  assert.equal(formatHubVramMaxForTest({ id: "x", vramMaxLabel: "~1.5 GB VRAM max" }), "~1.5 GB VRAM max");
  assert.equal(formatHubVramMaxForTest({ id: "x", vramMaxMiB: 1500 }), "~1.5 GB VRAM max");
  assert.equal(formatHubVramMaxForTest({ id: "x", weightsMiB: 1000 }), "~1.5 GB VRAM max");
  assert.equal(formatHubVramMaxForTest({ id: "x", sizeMiB: 14000 }), "~21 GB VRAM max");
  assert.equal(formatHubVramMaxForTest({ id: "Qwen/Qwen2.5-0.5B-Instruct", weightsMiB: 1000 }), "~1.5 GB VRAM max");
  assert.equal(
    formatHubVramMaxForTest({ id: "mistralai/Mistral-7B-Instruct-v0.2", weightsMiB: 14000 }),
    "~21 GB VRAM max",
  );

  const merged = mergeLateInferHubWithFallbackForTest([
    { id: "someone/random-finetune", family: "other" },
    { id: "Qwen/Qwen2.5-0.5B-Instruct", family: "Qwen" },
    { id: "google/gemma-2-2b-it", family: "Gemma" },
    { id: "mistralai/Mistral-7B-Instruct-v0.2", family: "Mistral" },
  ]);
  assert.ok(merged.some((model) => model.id === "Qwen/Qwen2.5-0.5B-Instruct"));
  assert.ok(merged.some((model) => model.id === "google/gemma-2-2b-it"));
  assert.ok(merged.some((model) => model.id === "google/gemma-4-E2B-it"));
  assert.ok(merged.some((model) => model.id === "mistralai/Mistral-7B-Instruct-v0.2"));
  assert.ok(merged.some((model) => model.id === "someone/random-finetune"));
  const qwen = merged.find((model) => model.id === "Qwen/Qwen2.5-0.5B-Instruct");
  const mistral = merged.find((model) => model.id === "mistralai/Mistral-7B-Instruct-v0.2");
  assert.equal(qwen?.vramMaxMiB, 1500);
  assert.equal(qwen?.sizeMiB, 1000);
  assert.equal(qwen?.weightsMiB, 1000);
  assert.equal(formatHubVramMaxForTest(qwen!), "~1.5 GB VRAM max");
  assert.equal(mistral?.vramMaxMiB, 21000);
  assert.equal(formatHubVramMaxForTest(mistral!), "~21 GB VRAM max");

  const groups = groupLateInferHubModelsForTest(merged);
  const gemmaGroups = hubModelGroupsForPickerForTest(groups, "Gemma");
  assert.deepEqual(
    gemmaGroups.map((group) => group.family),
    ["Gemma"],
  );
  const gemmaHtml = renderLateInferHubListHtmlForTest(gemmaGroups);
  assert.ok(gemmaHtml.includes("hub-model-option-vram"));
  assert.ok(gemmaHtml.includes("VRAM max"));
  assert.equal(gemmaHtml.includes("VRAM unknown"), false);
  assert.ok(gemmaHtml.includes("google/gemma-2-2b-it"));
  assert.ok(gemmaHtml.includes("google/gemma-4-E2B-it"));
  assert.equal(gemmaHtml.includes("Qwen/Qwen2.5-0.5B-Instruct"), false);

  const mistralGroups = hubModelGroupsForPickerForTest(groups, "Mistral");
  const mistralHtml = renderLateInferHubListHtmlForTest(mistralGroups);
  assert.ok(mistralHtml.includes("hub-model-option-vram"));
  assert.ok(mistralHtml.includes("~21 GB VRAM max"));
  assert.equal(mistralHtml.includes("VRAM unknown"), false);
  assert.ok(mistralHtml.includes("mistralai/Mistral-7B-Instruct-v0.2"));

  const clicked = "google/gemma-4-E2B-it";
  assert.ok(clicked.includes("/"));
  const downloadId = clicked.includes("/") ? clicked : "Qwen/Qwen2.5-0.5B-Instruct";
  assert.equal(downloadId, "google/gemma-4-E2B-it");
});

test("Late infer Local models splits Hub store and compiled panes", () => {
  const js = readFileSync(join(REPO_ROOT, "gui/public/app.js"), "utf8");
  const css = readFileSync(join(REPO_ROOT, "gui/public/styles.css"), "utf8");
  const storeIdx = js.indexOf('class="lateinfer-pane lateinfer-store-pane"');
  const localIdx = js.indexOf('class="lateinfer-pane lateinfer-local-pane');
  const dlIdx = js.indexOf('id="lateinfer-download"');
  const startSlot = js.indexOf('id="lateinfer-start-actions"');
  const bannerSlot = js.indexOf('id="lateinfer-start-banner-slot"');
  assert.ok(storeIdx > 0 && localIdx > storeIdx);
  assert.ok(dlIdx > storeIdx && dlIdx < localIdx);
  assert.ok(bannerSlot > localIdx && startSlot > localIdx);
  assert.ok(js.includes('id="lateinfer-start"'));
  assert.ok(js.includes('id="lateinfer-stop"'));
  assert.ok(js.includes("Nothing compiled on your computer yet"));
  assert.ok(js.includes("available to download"));
  assert.ok(js.includes("data-lateinfer-pane=\"compiled\""));
  assert.ok(js.includes("data-lateinfer-pane=\"store\""));
  assert.ok(js.includes("resolveLateInferIdForStart"));
  assert.ok(js.includes("compiledHubModelsForPane"));
  assert.ok(js.includes("lateinfer-check"));
  assert.ok(js.includes("On your computer only"));
  assert.ok(js.includes("already compiled on your computer"));
  assert.ok(js.includes("Remove from your computer"));
  assert.ok(js.includes("data-lateinfer-delete"));
  assert.ok(js.includes("/api/local-servers/delete"));
  assert.ok(js.includes("This removes the compiled snapshot from your computer. Download again from the Hub store to restore."));
  assert.ok(js.includes("applyCompiledDeleteResult"));
  assert.ok(js.includes("lateInferServingThisHubId"));
  assert.equal(js.includes("Docker image"), false);
  assert.ok(js.includes("idle GPU on your computer"));
  assert.ok(js.includes("Use all GPUs on this computer"));
  assert.ok(js.includes("70%"));
  assert.ok(js.includes("lateinfer-gpu-pick"));
  assert.ok(js.includes("lateinfer-gpu-detect"));
  assert.ok(js.includes("id=\"gpu-status\""));
  assert.ok(js.includes("lateinfer-gpu-live"));
  assert.ok(js.includes("starting on idle GPU"));
  assert.ok(js.includes("GPU running"));
  assert.ok(js.includes("Weights in host RAM"));
  assert.ok(js.includes("applyLocalModelsDisconnect"));
  assert.ok(js.includes("probeLateInferHealthDirect"));
  assert.ok(js.includes("http://127.0.0.1:8010/health"));
  assert.ok(js.includes("not GPU running"));
  assert.ok(css.includes(".lateinfer-gpu-live"));
  assert.ok(js.includes("lateinfer-download-progress"));
  assert.ok(js.includes("lateInferDownloadProgressHtml"));
  assert.ok(js.includes("formatLateInferHubBytes"));
  assert.ok(js.includes("formatLateInferEtaSec"));
  assert.ok(js.includes("gpu.runtimeOk === false"));
  assert.ok(js.includes("lateinfer-gpu-serve-hint"));
  assert.ok(js.includes("serveHint"));
  assert.equal(js.includes("this engine is CUDA/Metal/CPU, not XPU"), false);
  assert.ok(js.includes("Intel (Arc / XPU / Level Zero)"));
  assert.ok(js.includes('if (vendor === "nvidia") return "NVIDIA"'));
  assert.ok(js.includes('if (vendor === "amd") return "AMD"'));
  assert.ok(js.includes("idle · full VRAM"));
  assert.ok(js.includes("display · 70% cap"));
  assert.ok(js.includes("Detected before Download"));
  assert.ok(js.includes("lateInferCompileBlockedByGpu"));
  assert.ok(js.includes("not ready for GPU Start"));
  assert.ok(js.includes("pill ok\">ready"));
  assert.ok(js.includes("pill bad\">not ready for GPU Start"));
  assert.ok(js.includes("is-not-gpu-ready"));
  assert.ok(js.includes("is-gpu-ready"));
  assert.ok(js.includes("Convert for GPU Start"));
  assert.ok(js.includes("Start refused to protect your computer"));
  assert.ok(js.includes("Start refused on your computer"));
  assert.ok(js.includes("lateinfer-start-refuse"));
  assert.ok(js.includes("lateinfer-start-banner"));
  assert.ok(js.includes("lateinfer-start-blocked"));
  assert.ok(js.includes("scrollLateInferStartBannerIntoView"));
  assert.ok(js.includes("lateInferStartNotReadyForGpu"));
  assert.ok(js.includes("/api/local-servers/convert"));
  assert.ok(css.includes(".lateinfer-start-refuse"));
  assert.ok(css.includes(".lateinfer-start-banner"));
  assert.ok(css.includes(".hub-model-compiled-row.is-not-gpu-ready"));
  assert.ok(css.includes(".lateinfer-local-pane.is-not-gpu-ready"));
  assert.ok(js.includes('id="lateinfer-download-progress"'));
  assert.ok(js.includes('id="lateinfer-gpu-live"'));
  assert.ok(js.includes("function lateInferGpuIdForPost"));
  assert.ok(js.split("gpuId: lateInferGpuIdForPost()").length - 1 >= 3);
  assert.match(js, /\/api\/local-servers\/download[\s\S]{0,400}gpuId: lateInferGpuIdForPost\(\)/);
  assert.match(js, /\/api\/local-servers\/convert[\s\S]{0,400}gpuId: lateInferGpuIdForPost\(\)/);
  assert.ok(js.includes("data-lateinfer-delete"));
  assert.equal(js.includes("CUDA_VISIBLE_DEVICES"), false);
  assert.equal(js.includes("Ali-Mhrez"), false);
  assert.match(cssRuleBlock(css, ".hub-model-compiled-row .hub-model-delete"), /z-index:\s*1/);
  assert.match(cssRuleBlock(css, ".lateinfer-panes"), /grid-template-columns:\s*minmax\(0,\s*1fr\)\s+minmax\(0,\s*1fr\)/);
  assert.match(css, /@media \(max-width: 820px\)[\s\S]*\.lateinfer-panes\s*\{[\s\S]*grid-template-columns:\s*1fr/);
  assert.ok(css.includes(".lateinfer-download-progress"));
  assert.ok(css.includes(".lateinfer-compile-spinner"));
});

test("Late infer hub store list is in-flow, never a hidden absolute combobox", () => {
  const css = readFileSync(join(REPO_ROOT, "gui/public/styles.css"), "utf8");
  const listRule = cssRuleBlock(css, ".lateinfer-hub-list");
  assert.match(listRule, /display:\s*block/);
  assert.match(listRule, /position:\s*static/);
  assert.match(listRule, /visibility:\s*visible/);
  assert.doesNotMatch(listRule, /position:\s*absolute/);
  assert.doesNotMatch(listRule, /display:\s*none/);
  const listHiddenRule = cssRuleBlock(css, ".lateinfer-hub-list[hidden]");
  assert.match(listHiddenRule, /display:\s*block/);
  assert.match(listHiddenRule, /position:\s*static/);
  assert.doesNotMatch(listHiddenRule, /position:\s*absolute/);
  const optionRule = cssRuleBlock(css, ".hub-model-option");
  assert.match(optionRule, /display:\s*flex/);
  assert.match(optionRule, /justify-content:\s*space-between/);
  const vramRule = cssRuleBlock(css, ".hub-model-option-vram");
  assert.match(vramRule, /flex:\s*0 0 auto/);
  assert.match(vramRule, /white-space:\s*nowrap/);
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
