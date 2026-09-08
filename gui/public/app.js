const TOKEN_KEY = "orchestrator.gui.token";
const THEME_KEY = "orchestrator.gui.theme";
const DEFAULT_THEME = "grove";
const THEMES = [
  { id: "grove", label: "Grove" },
  { id: "noir", label: "Noir" },
  { id: "linen", label: "Linen" },
  { id: "harbor", label: "Harbor" },
  { id: "ember", label: "Ember" },
  { id: "paper", label: "Paper" },
];

function isThemeId(value) {
  return THEMES.some((theme) => theme.id === value);
}

function readStoredTheme() {
  try {
    const stored = localStorage.getItem(THEME_KEY);
    return isThemeId(stored) ? stored : DEFAULT_THEME;
  } catch {
    return DEFAULT_THEME;
  }
}

function themeOptionsHtml(selected) {
  const current = isThemeId(selected) ? selected : DEFAULT_THEME;
  return THEMES.map(
    (theme) =>
      `<option value="${escapeHtml(theme.id)}"${theme.id === current ? " selected" : ""}>${escapeHtml(theme.label)}</option>`,
  ).join("");
}

function themePickerMarkup(selectId) {
  const current = readStoredTheme();
  return `<label class="theme-field" for="${escapeHtml(selectId)}">
      Theme
      <select id="${escapeHtml(selectId)}" class="theme-select" aria-label="Theme">${themeOptionsHtml(current)}</select>
    </label>`;
}

function applyTheme(id) {
  const theme = isThemeId(id) ? id : DEFAULT_THEME;
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* private mode / blocked storage */
  }
  for (const select of document.querySelectorAll(".theme-select")) {
    if (![...select.options].some((option) => option.value === theme) || select.options.length !== THEMES.length) {
      select.innerHTML = themeOptionsHtml(theme);
    } else {
      select.value = theme;
    }
  }
}

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function loadSession() {
  const data = await api("/api/session");
  sessionInfo = {
    mcpUrl: typeof data.mcpUrl === "string" ? data.mcpUrl : "",
    bind: typeof data.bind === "string" ? data.bind : "",
    listenHost: typeof data.listenHost === "string" ? data.listenHost : "",
    configListenHost: typeof data.configListenHost === "string" ? data.configListenHost : "",
    suggestedLanHost: typeof data.suggestedLanHost === "string" ? data.suggestedLanHost : "",
    envListenHostSet: data.envListenHostSet === true,
  };
  return data;
}

/** Exact Late Settings URL from this GUI process (bound host + port). Never a hardcoded GUI default. */
function mcpUrlForLate() {
  if (sessionInfo.mcpUrl) return sessionInfo.mcpUrl;
  const host = location.hostname === "localhost" || location.hostname === "[::1]" ? "127.0.0.1" : location.hostname;
  const port = location.port;
  return `http://${host}${port ? `:${port}` : ""}/mcp`;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const el = document.createElement("textarea");
    el.value = text;
    el.setAttribute("readonly", "");
    el.style.position = "fixed";
    el.style.left = "-9999px";
    document.body.appendChild(el);
    el.select();
    try {
      return document.execCommand("copy");
    } finally {
      el.remove();
    }
  }
}

function backendEntry(id) {
  return (catalog.backends ?? []).find((b) => b.id === id);
}

function backendDisplayName(id) {
  const known = backendEntry(id);
  return (known?.nickname || id || "").trim() || id;
}

function logoSrc(id) {
  return `/api/backends/${encodeURIComponent(id)}/logo?token=${encodeURIComponent(token)}`;
}

function avatarMarkup(id, hasLogo) {
  if (!id || id === "user" || id === "orchestrator") {
    const letter = id === "user" ? "Y" : id === "orchestrator" ? "O" : "?";
    return `<span class="avatar avatar-letter" aria-hidden="true">${letter}</span>`;
  }
  const known = backendEntry(id);
  const showLogo = hasLogo ?? known?.hasLogo;
  const letter = String((known?.nickname || id).trim().charAt(0) || "?").toUpperCase();
  if (showLogo) {
    return `<img class="avatar" alt="" src="${escapeHtml(logoSrc(id))}" />`;
  }
  return `<span class="avatar avatar-letter" aria-hidden="true">${escapeHtml(letter)}</span>`;
}

function tokenFromLocation() {
  const params = new URLSearchParams(location.search);
  const fromQuery = params.get("token");
  if (fromQuery) {
    sessionStorage.setItem(TOKEN_KEY, fromQuery);
    params.delete("token");
    const next = `${location.pathname}${params.toString() ? `?${params}` : ""}${location.hash}`;
    history.replaceState({}, "", next);
    return fromQuery;
  }
  return sessionStorage.getItem(TOKEN_KEY) ?? "";
}

let token = tokenFromLocation();
let catalog = { backends: [], specialists: [], workflows: [], writePolicy: { allowedDirectories: [], defaultCwd: "" }, localRuntime: {} };
let sessionInfo = { mcpUrl: "", bind: "", listenHost: "", configListenHost: "", suggestedLanHost: "", envListenHostSet: false };
let runs = [];
let localModels = null;
let localServers = { lateinfer: null, ollama: null, llamacpp: [], lateInferBinary: null, llamaServerBinary: null, ollamaBinary: null, gpu: null };
let selectedRunId = null;
let events = null;
let threads = [];
let currentThread = null;
let sending = false;
let vllmPollTimer = null;
let lateInferPollTimer = null;
const DEFAULT_LATE_INFER_HUB = "Qwen/Qwen2.5-0.5B-Instruct";
/** Official Instruct ids listed immediately on Local models — Hub lastModified search is community noise. */
const FALLBACK_LATE_INFER_HUB_MODELS = [
  { id: "Qwen/Qwen2.5-0.5B-Instruct", family: "Qwen", weightsMiB: 1000, params: 500000000 },
  { id: "Qwen/Qwen2.5-1.5B-Instruct", family: "Qwen", weightsMiB: 3100, params: 1500000000 },
  { id: "Qwen/Qwen2.5-3B-Instruct", family: "Qwen", weightsMiB: 6000, params: 3000000000 },
  { id: "Qwen/Qwen3-0.6B", family: "Qwen", weightsMiB: 1200, params: 600000000 },
  { id: "Qwen/Qwen3-1.7B", family: "Qwen", weightsMiB: 3400, params: 1700000000 },
  { id: "Qwen/Qwen3-4B-Instruct-2507", family: "Qwen", weightsMiB: 8000, params: 4000000000 },
  { id: "Qwen/Qwen3-8B", family: "Qwen", weightsMiB: 16000, params: 8000000000 },
  { id: "Qwen/Qwen2.5-7B-Instruct", family: "Qwen", weightsMiB: 14000, params: 7000000000 },
  { id: "google/gemma-2-2b-it", family: "Gemma", gated: true, weightsMiB: 5000, params: 2600000000 },
  { id: "google/gemma-2-9b-it", family: "Gemma", gated: true, weightsMiB: 18000, params: 9200000000 },
  { id: "google/gemma-3-1b-it", family: "Gemma", gated: true, weightsMiB: 3000, params: 1000000000 },
  { id: "google/gemma-3-4b-it", family: "Gemma", gated: true, weightsMiB: 8000, params: 4300000000 },
  { id: "google/gemma-4-E2B-it", family: "Gemma", weightsMiB: 10000, params: 5100000000 },
  { id: "mistralai/Mistral-7B-Instruct-v0.2", family: "Mistral", weightsMiB: 14000, params: 7000000000 },
  { id: "mistralai/Mistral-7B-Instruct-v0.3", family: "Mistral", weightsMiB: 14000, params: 7000000000 },
  { id: "mistralai/Ministral-8B-Instruct-2410", family: "Mistral", weightsMiB: 16000, params: 8000000000 },
  { id: "meta-llama/Llama-3.2-1B-Instruct", family: "Llama", gated: true, weightsMiB: 3000, params: 1200000000 },
  { id: "meta-llama/Llama-3.2-3B-Instruct", family: "Llama", gated: true, weightsMiB: 7000, params: 3200000000 },
  { id: "microsoft/Phi-3.5-mini-instruct", family: "Phi", weightsMiB: 7600, params: 3800000000 },
  { id: "microsoft/Phi-4-mini-instruct", family: "Phi", weightsMiB: 7500, params: 3800000000 },
];
function fallbackLateInferHubModels() {
  return FALLBACK_LATE_INFER_HUB_MODELS.map((model) => {
    const vramMaxMiB = Math.ceil(model.weightsMiB * 1.5);
    return {
      id: model.id,
      family: model.family,
      gated: Boolean(model.gated),
      fallback: true,
      fits: true,
      likelyTooBig: false,
      sizeUnknown: false,
      sizeMiB: model.weightsMiB,
      weightsMiB: model.weightsMiB,
      params: model.params,
      vramMaxMiB,
      vramMaxLabel: formatVramMaxFromMiB(vramMaxMiB),
    };
  });
}
let lateInferHubId = (() => {
  try {
    return sessionStorage.getItem("orchestrator.lateinfer.hub") || DEFAULT_LATE_INFER_HUB;
  } catch {
    return DEFAULT_LATE_INFER_HUB;
  }
})();
let lateInferCompiledId = "";
let lateInferDownloadBusy = false;
/** Research opt-in: show estimated tokens/sec on chat replies (default off). */
let chatShowTokensPerSec = (() => {
  try {
    return localStorage.getItem("orchestrator.chat.showTps") === "1";
  } catch {
    return false;
  }
})();

const LATE_INFER_HUB_FAMILY_ORDER = ["Qwen", "Mistral", "Gemma", "Llama", "Phi", "other"];
let lateInferHubCatalog = {
  models: fallbackLateInferHubModels(),
  groups: [],
  hint: "",
  defaultId: DEFAULT_LATE_INFER_HUB,
  budgetGiB: null,
  query: "",
  offline: false,
  loaded: false,
  loading: false,
  error: "",
};
let lateInferHubSearch = "";
let lateInferHubSearchTimer = null;
let lateInferOnComputerOnly = false;
/** When false, hide gated Hub rows. When true, show gated but locked until license + HF token. */
let lateInferShowGated = (() => {
  try {
    const v = sessionStorage.getItem("orchestrator.lateinfer.showGated");
    if (v === "0") return false;
    if (v === "1") return true;
  } catch {
    /* ignore */
  }
  return true;
})();
let lateInferHfTokenSet = false;
let llamaGgufCatalog = {
  models: [],
  hint: "",
  budgetGiB: null,
  offline: false,
  loaded: false,
  loading: false,
  error: "",
  ggufDir: "",
  defaultId: "",
};
let llamaGgufSearch = "unsloth";
let llamaGgufSearchTimer = null;
let llamaGgufSelectedId = "";
let llamaGgufPath = "";
let llamaGgufJobs = [];
let lateInferUseAllGpus = false;
let lateInferGpuId = "auto";
let lateInferStartError = "";
const LATE_INFER_STARTING_KEY = "orchestrator.lateinfer.starting";
let updateCheck = null;
let updatePick = null;
let updateBusy = false;
let updateFlash = "";
let updateFlashKind = "ok";

function anyVllmStarting(status) {
  if (!status) return false;
  if (status.phase === "starting") return true;
  return (status.instances ?? []).some((row) => row.phase === "starting");
}

function syncVllmPoll(phaseOrStatus) {
  const starting = typeof phaseOrStatus === "string" ? phaseOrStatus === "starting" : anyVllmStarting(phaseOrStatus);
  if (starting) {
    if (vllmPollTimer) return;
    vllmPollTimer = setInterval(async () => {
      try {
        const status = await api("/api/vllm");
        if (localModels) localModels = { ...localModels, vllm: status };
        if (pageId() === "local-models") renderLocalModels();
        if (!anyVllmStarting(status)) {
          clearInterval(vllmPollTimer);
          vllmPollTimer = null;
          await loadLocalModels();
          if (pageId() === "local-models") renderLocalModels();
          if (pageId() === "backends") await renderBackends();
        }
      } catch {
        /* keep polling while starting */
      }
    }, 2500);
    return;
  }
  if (vllmPollTimer) {
    clearInterval(vllmPollTimer);
    vllmPollTimer = null;
  }
}

function lateInferErrorLine(text) {
  const cleaned = String(text ?? "")
    .replace(/late-infer:\s*downloading Hub snapshot[^\n]*/gi, "")
    .replace(/late-infer:\s*compiling[^\n]*/gi, "")
    .trim();
  const license = cleaned.match(
    /Token is set but Hugging Face still denied access[\s\S]*?Do not commit the token\.|Repo is gated\.[\s\S]*?Do not commit the token\./i,
  );
  if (license) return license[0].replace(/\s+/g, " ").slice(0, 400);
  if (/^\s*Error:\s*config\.json\s*$/i.test(cleaned) || (/Error:\s*config\.json\b/i.test(cleaned) && !/\b(401|403|404|status code|request error|denied|not found|authorization required)\b/i.test(cleaned))) {
    return "Could not read Hub config.json for this model — late-infer needs a safetensors Instruct snapshot with config.json. Check the Hub id (wrong or private repos fail here). GGUF packs belong under llama.cpp.";
  }
  if (/config\.json/i.test(cleaned) && /\b401\b|\b403\b|denied|authorization required/i.test(cleaned)) {
    return "Hub denied config.json (gated or private) — accept the model license on Hugging Face, then set a read token in Settings → Local models.";
  }
  if (/config\.json/i.test(cleaned) && /\b404\b|not found/i.test(cleaned)) {
    return "Hub repo has no config.json (missing on Hugging Face) — late-infer needs a safetensors Instruct snapshot with config.json. GGUF-only packs belong under llama.cpp.";
  }
  if (/config\.json/i.test(cleaned) && /request error|status code/i.test(cleaned)) {
    return "Could not read Hub config.json for this model (Hub/network error). Retry, or pick another Instruct id. GGUF packs do not use config.json — use llama.cpp.";
  }
  const lines = cleaned
    .split(/\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const errLine =
    lines.find((line) =>
      /error|404|401|403|fail closed|not a supported|gated|cannot access|cannot compile|intel blob|not treat that card as NVIDIA|xpu|level zero|IR is missing|system RAM|Start refused/i.test(
        line,
      ),
    ) ??
    lines[0] ??
    "";
  return errLine.replace(/\s+/g, " ").slice(0, 400);
}

function lateInferHasError(infer) {
  if (!infer) return false;
  const phase = String(infer.phase ?? "").toLowerCase();
  if (phase === "error") return true;
  const err = String(infer.error ?? "").trim();
  const msg = String(infer.message ?? infer.reason ?? "").trim();
  if (err) return true;
  return /status code 404|\bError:|\bfail closed|request error/i.test(msg);
}

function lateInferJobBusy(infer) {
  if (!infer || lateInferHasError(infer)) return false;
  if (infer.downloading) return true;
  const phase = String(infer.phase ?? "").toLowerCase();
  return phase === "downloading" || phase === "compiling";
}

function lateInferBusy(infer) {
  return lateInferJobBusy(infer) || lateInferDownloadBusy;
}

function lateInferDownloadStatusHtml(infer) {
  const msg = String(infer?.message ?? infer?.reason ?? infer?.progress ?? "").trim();
  const err = String(infer?.error ?? "").trim();
  const phase = String(infer?.phase ?? "").toLowerCase();
  if (lateInferHasError(infer) || err || phase === "error") {
    const text =
      lateInferErrorLine(err || msg) || "This Hub repo is not supported by late-infer on your computer.";
    return `<p class="error">${escapeHtml(text)}</p>`;
  }
  if (/unsupported|not every hub/i.test(msg)) {
    return `<p class="error">${escapeHtml(lateInferErrorLine(msg) || msg)}</p>`;
  }
  return "";
}

function formatLateInferHubBytes(n) {
  const bytes = Number(n);
  if (!Number.isFinite(bytes) || bytes < 0) return "";
  if (bytes < 1024) return `${Math.round(bytes)} B`;
  const mb = bytes / (1024 * 1024);
  if (mb < 1024) {
    const shown = mb >= 10 ? String(Math.round(mb)) : (Math.round(mb * 10) / 10).toFixed(1).replace(/\.0$/, "");
    return `${shown} MB`;
  }
  const gb = mb / 1024;
  const shown = gb >= 10 ? String(Math.round(gb)) : (Math.round(gb * 10) / 10).toFixed(1).replace(/\.0$/, "");
  return `${shown} GB`;
}

function formatLateInferEtaSec(sec) {
  const n = Number(sec);
  if (!Number.isFinite(n) || n < 0) return "";
  if (n < 90) return `about ${Math.round(n)}s left`;
  const min = Math.round(n / 60);
  if (min < 90) return `about ${min} min left`;
  return `about ${Math.round(min / 60)} h left`;
}

/** Hub Download progress (percent / ETA / compiling). GPU-status sibling: leave #lateinfer-download-progress alone. */
function lateInferDownloadProgressHtml(infer) {
  if (lateInferHasError(infer)) return "";
  if (!lateInferJobBusy(infer) && !lateInferDownloadBusy) return "";
  const msg = String(infer?.message ?? "").trim();
  const phase = String(infer?.phase ?? "").toLowerCase();
  const compiling = phase === "compiling" || (infer?.downloading && /compil/i.test(msg));
  const percent = Number(infer?.percent);
  const hasPercent = Number.isFinite(percent) && percent >= 0;
  const width = hasPercent ? Math.max(0, Math.min(100, percent)) : 0;
  const bytes = Number(infer?.bytes);
  const total = Number(infer?.totalBytes);
  const eta = Number(infer?.etaSec);
  const bar = compiling && !hasPercent
    ? ""
    : `<div class="progress" id="lateinfer-download-progress-bar" title="${hasPercent ? `${Math.round(percent)}%` : ""}"><span style="width:${width}%"></span></div>`;
  if (compiling) {
    const pct = hasPercent ? ` ${escapeHtml(String(Math.round(percent)))}%` : "";
    return `<div class="lateinfer-download-progress-inner" aria-live="polite"><p class="muted"><span class="lateinfer-compile-spinner" aria-hidden="true"></span> <span class="pill warn">compiling</span> Compiling on your computer…${pct}${msg && !/compil/i.test(msg) ? ` ${escapeHtml(msg)}` : ""}</p>${bar}</div>`;
  }
  const parts = ["Downloading Hub snapshot…"];
  if (hasPercent) parts.push(`${Math.round(percent)}%`);
  if (Number.isFinite(bytes) && Number.isFinite(total) && total > 0) {
    const from = formatLateInferHubBytes(bytes);
    const to = formatLateInferHubBytes(total);
    if (from && to) parts.push(`${from} / ${to}`);
  }
  if (Number.isFinite(eta) && eta >= 0 && !(hasPercent && percent >= 100)) {
    const etaText = formatLateInferEtaSec(eta);
    if (etaText) parts.push(etaText);
  }
  return `<div class="lateinfer-download-progress-inner" aria-live="polite"><p class="muted"><span class="pill warn">downloading</span> ${escapeHtml(parts.join(" · "))}</p>${bar}</div>`;
}

function lateInferCompiledIdSet() {
  const raw = localServers.lateinfer?.compiledModels ?? localServers.compiledModels ?? [];
  return new Set((Array.isArray(raw) ? raw : []).map((id) => String(id)));
}

function lateInferCompiledRows() {
  const raw = localServers.lateinfer?.compiledRows;
  if (Array.isArray(raw) && raw.length) return raw;
  return [...lateInferCompiledIdSet()].map((id) => ({
    id,
    irPresent: false,
    gpuReady: lateInferGpuState().vendor === "nvidia" || lateInferGpuState().cards?.[0]?.vendor === "nvidia",
    convertOk: false,
    convertReason: "",
  }));
}

function compiledHubIdsMatch(a, b) {
  const left = String(a ?? "").trim();
  const right = String(b ?? "").trim();
  if (!left || !right) return false;
  if (left === right) return true;
  return left.replaceAll("/", "--") === right.replaceAll("/", "--");
}

function applyCompiledDeleteResult(result, hubId) {
  if (lateInferCompiledId && compiledHubIdsMatch(lateInferCompiledId, hubId)) lateInferCompiledId = "";
  if (!localServers.lateinfer) localServers.lateinfer = {};
  if (Array.isArray(result?.compiledModels)) localServers.lateinfer.compiledModels = result.compiledModels;
  if (Array.isArray(result?.compiledRows)) {
    localServers.lateinfer.compiledRows = result.compiledRows;
  } else if (Array.isArray(result?.compiledModels)) {
    const keep = new Set(result.compiledModels);
    const rows = Array.isArray(localServers.lateinfer.compiledRows) ? localServers.lateinfer.compiledRows : [];
    localServers.lateinfer.compiledRows = rows.filter((row) => keep.has(String(row?.id ?? "")));
  }
}

function lateInferServingThisHubId(hubId) {
  const late = localServers.lateinfer ?? {};
  const up = Boolean(late.ready || late.running || late.processAlive);
  if (!up) return false;
  const needle = String(hubId ?? "").trim();
  if (!needle) return false;
  const listed = Array.isArray(late.models) ? late.models.map((id) => String(id).trim()).filter(Boolean) : [];
  if (listed.some((id) => compiledHubIdsMatch(id, needle))) return true;
  const healthModel = String(late.model ?? "").trim();
  return Boolean(healthModel) && compiledHubIdsMatch(healthModel, needle);
}

function compiledRowForId(id) {
  const needle = String(id ?? "").trim();
  return lateInferCompiledRows().find((row) => compiledHubIdsMatch(String(row.id), needle));
}

function intelIrMissingStartMessage() {
  const gpu = lateInferGpuState();
  const ram = gpu.hostRamReason || "this convert would use system RAM, not idle GPU VRAM";
  const loud =
    "OpenVINO IR is missing. Start refused to protect your computer. Download/compile must produce IR, or Convert only when MemAvailable is safe.";
  if (String(ram).includes("OpenVINO IR is missing") && String(ram).includes("Start refused")) return ram;
  return `${loud} ${ram}`;
}

function hubModelOnComputer(model) {
  return lateInferCompiledIdSet().has(String(model?.id ?? ""));
}

function hubFamilyFromModel(model) {
  const fam = String(model?.family ?? "").trim();
  if (fam) return fam.toLowerCase() === "other" ? "other" : fam;
  const id = String(model?.id ?? "");
  const lower = id.toLowerCase();
  if (/qwen/i.test(lower)) return "Qwen";
  if (/mistral|mixtral/i.test(lower)) return "Mistral";
  if (/gemma/i.test(lower)) return "Gemma";
  if (/llama|meta-llama/i.test(lower)) return "Llama";
  if (/phi/i.test(lower)) return "Phi";
  return "other";
}

function parseHubVersionScore(version, id) {
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

function hubModelIsLoadable(model) {
  if (model?.loadable === false) return false;
  if (model?.configStatus === "missing") return false;
  if (model?.ovExportOk === false && model?.backends?.lateinfer === false) return false;
  return model?.loadable !== false;
}

function hubModelIsGemma2(model) {
  const t = String(model?.modelType ?? "").toLowerCase();
  if (t === "gemma2") return true;
  return /gemma-?2/i.test(String(model?.id ?? ""));
}

function compareHubModelNewestFirst(a, b) {
  if (Boolean(a.onComputer) !== Boolean(b.onComputer)) return a.onComputer ? -1 : 1;
  const aLoad = hubModelIsLoadable(a);
  const bLoad = hubModelIsLoadable(b);
  if (aLoad !== bLoad) return aLoad ? -1 : 1;
  const aGate = Boolean(a.gated || a.gatedNeedsLicense);
  const bGate = Boolean(b.gated || b.gatedNeedsLicense);
  if (aLoad && bLoad && aGate !== bGate) return aGate ? 1 : -1;
  const aGemma2 = hubModelIsGemma2(a);
  const bGemma2 = hubModelIsGemma2(b);
  if (aLoad && bLoad && aGemma2 !== bGemma2) return aGemma2 ? 1 : -1;
  if (Boolean(a.fallback) !== Boolean(b.fallback)) return a.fallback ? -1 : 1;
  const score = (model) => {
    if (typeof model.version === "number" && Number.isFinite(model.version)) return model.version * 100;
    return parseHubVersionScore(model.version, model.id);
  };
  const versionDelta = score(b) - score(a);
  if (versionDelta !== 0) return versionDelta;
  const tb = Date.parse(b.lastModified ?? "") || 0;
  const ta = Date.parse(a.lastModified ?? "") || 0;
  if (tb !== ta) return tb - ta;
  return String(a.id).localeCompare(String(b.id));
}

function groupLateInferHubModels(models) {
  const buckets = new Map();
  for (const model of models ?? []) {
    if (!model?.id) continue;
    const family = hubFamilyFromModel(model);
    if (!buckets.has(family)) buckets.set(family, []);
    buckets.get(family).push({ ...model, family, onComputer: hubModelOnComputer(model) });
  }
  for (const list of buckets.values()) list.sort(compareHubModelNewestFirst);
  const extras = [...buckets.keys()].filter((family) => !LATE_INFER_HUB_FAMILY_ORDER.includes(family)).sort();
  return LATE_INFER_HUB_FAMILY_ORDER.concat(extras)
    .filter((family) => buckets.has(family))
    .map((family) => ({ family, models: buckets.get(family) ?? [] }));
}

function filterLateInferHubModels(models, query) {
  const needle = String(query ?? "").trim().toLowerCase();
  if (!needle) return models ?? [];
  return (models ?? []).filter((model) => {
    const hay = [model.id, model.label, model.name, model.family, model.version, model.params]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return hay.includes(needle);
  });
}

function normalizeHubModelsResponse(data) {
  if (Array.isArray(data)) return data;
  if (Array.isArray(data?.groups)) {
    return data.groups.flatMap((group) =>
      (group.models ?? []).map((model) => ({ ...model, family: model.family ?? group.family })),
    );
  }
  if (Array.isArray(data?.models)) return data.models;
  return [];
}

function hubCatalogModelsFromResponse(data) {
  const direct = Array.isArray(data?.models) ? data.models.filter((model) => model?.id) : [];
  if (direct.length) return direct;
  return normalizeHubModelsResponse(data).filter((model) => model?.id);
}

function hubCatalogGroupsFromResponse(data, models) {
  if (Array.isArray(data?.groups) && data.groups.length) {
    return data.groups
      .map((group) => ({
        family: group.family,
        models: (group.models ?? []).filter((model) => model?.id),
      }))
      .filter((group) => group.models.length);
  }
  return groupLateInferHubModels(models);
}

function lateInferHubHasListData() {
  if ((lateInferHubCatalog.models?.length ?? 0) > 0) return true;
  return (lateInferHubCatalog.groups ?? []).some((group) => (group.models?.length ?? 0) > 0);
}

function mergeLateInferHubWithFallback(fromHub) {
  const byId = new Map();
  const seeds = fallbackLateInferHubModels();
  const seedIds = new Set(seeds.map((model) => model.id));
  for (const model of seeds) byId.set(model.id, model);
  for (const model of fromHub ?? []) {
    if (!model?.id) continue;
    const prev = byId.get(model.id);
    const next = seedIds.has(model.id) ? { ...model, fallback: true } : model;
    const gated =
      next.gatedKnown === false && prev
        ? Boolean(prev.gated) || Boolean(next.gated)
        : next.gated != null
          ? Boolean(next.gated)
          : Boolean(prev?.gated);
    byId.set(model.id, {
      ...prev,
      ...next,
      gated,
      gatedNeedsLicense: gated || Boolean(prev?.gatedNeedsLicense || next.gatedNeedsLicense),
      sizeMiB: next.sizeMiB ?? prev?.sizeMiB,
      weightsMiB: next.weightsMiB ?? prev?.weightsMiB,
      params: next.params ?? prev?.params,
      bytes: next.bytes ?? prev?.bytes,
      vramMaxMiB: next.vramMaxMiB ?? prev?.vramMaxMiB,
      vramMaxGiB: next.vramMaxGiB ?? prev?.vramMaxGiB,
      vramMaxLabel: next.vramMaxLabel ?? prev?.vramMaxLabel,
      modelType: next.modelType ?? prev?.modelType,
      ovExportOk: next.ovExportOk ?? prev?.ovExportOk,
      loadable: next.loadable ?? prev?.loadable,
      configStatus: next.configStatus ?? prev?.configStatus,
      serveBlockedReason: next.serveBlockedReason ?? prev?.serveBlockedReason,
      downloadable: next.downloadable ?? prev?.downloadable,
      backends: next.backends ?? prev?.backends,
    });
  }
  return [...byId.values()];
}

function ensureLateInferHubFallback() {
  if (lateInferHubHasListData()) return;
  const models = fallbackLateInferHubModels();
  lateInferHubCatalog.models = models;
  lateInferHubCatalog.groups = groupLateInferHubModels(models);
}

function hubFamilyDisplayName(family) {
  const name = String(family ?? "").trim();
  if (!name || name.toLowerCase() === "other") return "Other";
  return name;
}

function formatHubParams(params) {
  const n = Number(params);
  if (!Number.isFinite(n) || n <= 0) return "";
  if (n >= 1e9) {
    const b = n / 1e9;
    return `${Number.isInteger(b) ? b : b.toFixed(1).replace(/\.0$/, "")}B params`;
  }
  if (n >= 1e6) return `${Math.round(n / 1e6)}M params`;
  return `${Math.round(n).toLocaleString()} params`;
}

function formatVramMaxFromMiB(mib) {
  const n = Number(mib);
  if (!Number.isFinite(n) || n <= 0) return "VRAM unknown";
  if (n < 1024) return `~${Math.round(n)} MiB VRAM max`;
  const gib = n / 1024;
  const shown = gib >= 10 ? String(Math.round(gib)) : (Math.round(gib * 10) / 10).toFixed(1).replace(/\.0$/, "");
  return `~${shown} GB VRAM max`;
}

function formatHubVramMax(model) {
  const label = String(model?.vramMaxLabel ?? "").trim();
  if (label && label !== "VRAM unknown") return label;
  const api = Number(model?.vramMaxMiB);
  if (Number.isFinite(api) && api > 0) return formatVramMaxFromMiB(api);
  const weights = Number(model?.sizeMiB ?? model?.weightsMiB);
  if (Number.isFinite(weights) && weights > 0) return formatVramMaxFromMiB(Math.ceil(weights * 1.5));
  return label || "VRAM unknown";
}

function hubModelPickerLabel(model) {
  if (model.label) return model.label;
  if (model.name) return model.name;
  const family = hubFamilyDisplayName(hubFamilyFromModel(model));
  const parts = [];
  const version = model.version != null && model.version !== "" ? String(model.version) : "";
  if (version) parts.push(`${family} ${version}`);
  else if (family !== "Other") parts.push(family);
  const params = formatHubParams(model.params);
  if (params) parts.push(params);
  parts.push(model.id);
  return parts.join(" · ");
}

function hubModelGroupsForPicker(query) {
  const needle = String(query ?? lateInferHubSearch ?? "").trim().toLowerCase();
  const merged = mergeLateInferHubWithFallback([
    ...(lateInferHubCatalog.models ?? []),
    ...(lateInferHubCatalog.groups ?? []).flatMap((group) => group.models ?? []),
  ]);
  const sourceGroups = groupLateInferHubModels(merged);
  const grouped = sourceGroups.map((group) => ({
    family: hubFamilyDisplayName(group.family),
    models: (group.models ?? []).filter((model) => {
      if (lateInferOnComputerOnly && !(model.onComputer === true || hubModelOnComputer(model))) return false;
      if (!lateInferShowGated && (model.gated || model.gatedNeedsLicense)) return false;
      return true;
    }),
  })).filter((group) => group.models.length);
  if (!needle) return grouped;
  return grouped
    .map((group) => ({
      family: group.family,
      models: filterLateInferHubModels(group.models ?? [], needle),
    }))
    .filter((group) => group.models.length);
}

function lateInferHubDefaultId() {
  return lateInferHubCatalog.defaultId?.trim() || DEFAULT_LATE_INFER_HUB;
}

function officialLateInferHubIds() {
  return new Set(FALLBACK_LATE_INFER_HUB_MODELS.map((model) => model.id));
}

function firstOfficialHubIdForQuery(query) {
  const groups = hubModelGroupsForPicker(query);
  const official = officialLateInferHubIds();
  for (const group of groups) {
    for (const model of group.models ?? []) {
      if (model.fallback || official.has(model.id)) return model.id;
    }
  }
  return groups[0]?.models?.[0]?.id;
}

function selectedLateInferHubRowId() {
  const list = $("lateinfer-hub-list");
  const selected = list?.querySelector?.(".hub-model-option.is-selected, .hub-model-option[aria-selected='true']");
  const fromDom = selected?.getAttribute?.("data-hub-id")?.trim() ?? "";
  if (fromDom.includes("/")) return fromDom;
  return "";
}

function resolveLateInferHubIdForDownload() {
  // Typed/pasted Hub id always wins — even when not in the listed rows.
  const fromInput = String($("lateinfer-hub")?.value ?? lateInferHubId ?? "").trim();
  if (fromInput.includes("/")) return fromInput;

  const fromRow = selectedLateInferHubRowId();
  if (fromRow) return fromRow;

  const search = String($("lateinfer-hub-search")?.value ?? lateInferHubSearch ?? "").trim();
  const familyQuery = fromInput || search;
  if (familyQuery && !familyQuery.includes("/")) {
    const officialMatch = firstOfficialHubIdForQuery(familyQuery);
    if (officialMatch) return officialMatch;
  }

  if (search) {
    const officialMatch = firstOfficialHubIdForQuery(search);
    if (officialMatch) return officialMatch;
  }

  const selected = String(lateInferHubId ?? "").trim();
  if (selected.includes("/")) return selected;
  return lateInferHubDefaultId();
}

function lateInferHubBudgetGiB() {
  if (lateInferHubCatalog.budgetGiB != null && Number.isFinite(lateInferHubCatalog.budgetGiB)) {
    return Math.round(lateInferHubCatalog.budgetGiB);
  }
  const vramMiB = localModels?.hardware?.totalVramMiB ?? localModels?.hardware?.vramMiB;
  if (vramMiB != null && Number.isFinite(vramMiB) && vramMiB > 0) return Math.round(vramMiB / 1024);
  const ramMiB = localModels?.hardware?.ramMiB;
  if (ramMiB != null && Number.isFinite(ramMiB) && ramMiB > 0) return Math.round(ramMiB / 1024);
  return null;
}

function lateInferHubHintText() {
  if (lateInferHubCatalog.offline && (lateInferHubCatalog.models ?? []).length) {
    return "Hugging Face Hub is offline. Search listed models, click a row, then Download on your computer — you do not need the Hugging Face website. Paste any other Instruct id if yours is missing.";
  }
  if (lateInferHubCatalog.offline) {
    return "Hugging Face Hub is offline. Search listed models, click a row, then Download. Paste any Instruct id if yours is missing.";
  }
  if (lateInferHubCatalog.hint) {
    return lateInferHubCatalog.hint;
  }
  const gb = lateInferHubBudgetGiB();
  const budget = gb != null ? ` (~${gb} GB)` : "";
  return `Listed for your computer${budget}, grouped by family (newest first). Each row shows estimated VRAM at max usage (weights plus KV cache). Search listed models, click a row, then Download — you do not need the Hugging Face website. Not every Hub repo compiles. Paste any Instruct id if yours is missing.`;
}

function hardwareNotesForLocalModels(notes) {
  return (notes ?? []).filter((note) => !/vLLM Docker|llm-scaler-vllm|intel\/vllm/i.test(String(note)));
}

function renderLateInferHubListHtml(query) {
  ensureLateInferHubFallback();
  const groups = hubModelGroupsForPicker(query);
  if (!groups.length) {
    const fallback = lateInferHubDefaultId();
    return `<p class="muted hub-model-empty">No safetensors chat matches here. Paste a Hub org/model id to Download anyway. GGUF-only models appear under llama.cpp.</p>`;
  }
  return groups
    .map(
      (group) => `
    <div data-hub-family="${escapeHtml(group.family)}" class="hub-model-group" role="presentation">
      <div class="hub-model-group-label">${escapeHtml(group.family)}</div>
      ${group.models
        .map((model) => {
          const selected = model.id === lateInferHubId;
          const onComputer = model.onComputer === true || hubModelOnComputer(model);
          const loadable = hubModelIsLoadable(model);
          const gated = Boolean(model.gated || model.gatedNeedsLicense);
          const demoted = !loadable;
          let availability;
          if (onComputer) {
            availability = `<span class="pill ok">on your computer</span>`;
          } else if (!loadable) {
            const why =
              model.configStatus === "missing"
                ? "missing config.json"
                : model.serveBlockedReason
                  ? "not for this GPU"
                  : "not loadable";
            availability = `<span class="pill bad" title="${escapeHtml(model.serveBlockedReason || why)}">${escapeHtml(why)}</span>`;
          } else if (gated) {
            availability = `<span class="pill warn" title="Accept the Hugging Face license, then set a read token in Settings → Local models">gated · locked</span>`;
          } else {
            availability = `<span class="pill ok">ready to download</span>`;
          }
          const ovPill =
            model.ovExportOk === true
              ? ` <span class="pill ok" title="OpenVINO CausalLM export ok">OV ok</span>`
              : model.ovExportOk === false
                ? ` <span class="pill bad" title="${escapeHtml(model.serveBlockedReason || "Not OpenVINO CausalLM on this GPU")}">OV no</span>`
                : "";
          return `
        <button type="button" role="option" data-hub-id="${escapeHtml(model.id)}" data-lateinfer-pane="store" data-on-computer="${onComputer ? "true" : "false"}" data-loadable="${loadable ? "true" : "false"}" data-gated="${gated ? "true" : "false"}" class="hub-model-option${selected ? " is-selected" : ""}${onComputer ? " is-on-computer" : ""}${demoted ? " is-demoted" : ""}${gated && loadable ? " is-gated-locked" : ""}" aria-selected="${selected ? "true" : "false"}">
          <span class="hub-model-option-main">${escapeHtml(hubModelPickerLabel(model))} ${availability}${ovPill}${model.fallback ? ` <span class="muted">(fallback)</span>` : ""}${model.likelyTooBig || model.fits === false ? ` <span class="muted">(likely too big)</span>` : model.sizeUnknown ? ` <span class="muted">(size unknown)</span>` : ""}</span>
          <span class="hub-model-option-vram">${escapeHtml(formatHubVramMax(model))}</span>
        </button>`;
        })
        .join("")}
    </div>`,
    )
    .join("");
}

function compiledHubModelsForPane() {
  const ids = [...lateInferCompiledIdSet()].filter((id) => id.includes("/") && !id.startsWith("/"));
  const catalog = mergeLateInferHubWithFallback([
    ...(lateInferHubCatalog.models ?? []),
    ...(lateInferHubCatalog.groups ?? []).flatMap((group) => group.models ?? []),
  ]);
  const byId = new Map(catalog.map((model) => [model.id, model]));
  return ids.map((id) => {
    const known = byId.get(id);
    return known ? { ...known, onComputer: true } : { id, family: hubFamilyFromModel({ id }), onComputer: true };
  });
}

function lateInferServingModelId() {
  const li = localServers.lateinfer ?? {};
  const fromModel = String(li.model ?? "").trim();
  if (fromModel) return fromModel;
  const listed = Array.isArray(li.models) ? li.models : [];
  for (const id of listed) {
    const s = String(id ?? "").trim();
    if (s) return s;
  }
  return "";
}

function renderLateInferCompiledListHtml() {
  const models = compiledHubModelsForPane();
  if (!models.length) {
    return `<p class="muted hub-model-empty" id="lateinfer-compiled-empty">Nothing compiled on your computer yet.</p>`;
  }
  const running = lateInferServingModelId();
  const li = localServers.lateinfer ?? {};
  const live = Boolean(li.ready || li.running || li.processAlive) && Boolean(running);
  const selectedId =
    (lateInferCompiledId && models.some((model) => model.id === lateInferCompiledId) && lateInferCompiledId) ||
    (running && models.some((model) => model.id === running) && running) ||
    models[0].id;
  lateInferCompiledId = selectedId;
  return `<div class="hub-model-group" role="presentation">
      <div class="hub-model-group-label">Compiled</div>
      ${models
        .map((model) => {
          const selected = model.id === selectedId;
          const isRunning = live && model.id === running;
          const row = compiledRowForId(model.id);
          const gpuReady = row?.gpuReady === true;
          const statePill = isRunning
            ? `<span class="pill ok">running</span>`
            : gpuReady
              ? `<span class="pill ok">ready</span>`
              : `<span class="pill bad">not ready for GPU Start</span>`;
          const readyClass = gpuReady ? " is-gpu-ready" : " is-not-gpu-ready";
          const cross = lateInferHubId && model.id === lateInferHubId;
          return `
        <div class="hub-model-compiled-row${selected ? " is-selected" : ""}${cross ? " is-cross-selected" : ""}${readyClass}${isRunning ? " is-running" : ""}" role="option" data-hub-id="${escapeHtml(model.id)}" data-lateinfer-pane="compiled" aria-selected="${selected ? "true" : "false"}">
          <button type="button" data-hub-id="${escapeHtml(model.id)}" data-lateinfer-pane="compiled" class="hub-model-option${selected ? " is-selected" : ""}${cross ? " is-cross-selected" : ""}${readyClass}" aria-selected="${selected ? "true" : "false"}">
            <span class="hub-model-option-main">${escapeHtml(hubModelPickerLabel(model))} ${statePill}</span>
            <span class="hub-model-option-vram">${escapeHtml(formatHubVramMax(model))}</span>
          </button>
          <button type="button" class="btn secondary hub-model-delete" data-lateinfer-delete="${escapeHtml(model.id)}">Remove from your computer</button>
        </div>`;
        })
        .join("")}
    </div>`;
}

function refreshLateInferLocalPaneChrome() {
  const slot = $("lateinfer-start-banner-slot");
  if (slot) slot.innerHTML = renderLateInferStartBannerHtml();
  const actions = $("lateinfer-start-actions");
  if (actions) actions.innerHTML = renderLateInferLocalStartInnerHtml();
  const pane = $("lateinfer-local-pane");
  if (pane) pane.classList.toggle("is-not-gpu-ready", lateInferStartNotReadyForGpu());
}

function refreshLateInferCompiledList() {
  const list = $("lateinfer-compiled-list");
  if (list) list.innerHTML = renderLateInferCompiledListHtml();
  refreshLateInferLocalPaneChrome();
}

function refreshLateInferHubList() {
  const list = $("lateinfer-hub-list");
  if (!list) return;
  list.hidden = false;
  list.innerHTML = renderLateInferHubListHtml(lateInferHubSearch);
  const search = $("lateinfer-hub-search");
  if (search) search.setAttribute("aria-expanded", "true");
  refreshLateInferCompiledList();
  refreshLateInferCheckStatus();
}

function resolveLateInferIdForStart() {
  const compiled = compiledHubModelsForPane().map((model) => model.id);
  const fromRow = $("lateinfer-compiled-list")
    ?.querySelector?.(".hub-model-option.is-selected, .hub-model-option[aria-selected='true']")
    ?.getAttribute?.("data-hub-id")
    ?.trim();
  if (fromRow && compiled.includes(fromRow)) return fromRow;
  if (lateInferCompiledId && compiled.includes(lateInferCompiledId)) return lateInferCompiledId;
  return compiled[0] || "";
}

function selectedCompiledVramMaxMiB() {
  const id = resolveLateInferIdForStart();
  const model = compiledHubModelsForPane().find((row) => row.id === id);
  const n = Number(model?.vramMaxMiB);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

function findLateInferHubCatalogModel(hubId) {
  const id = String(hubId ?? "").trim();
  if (!id) return undefined;
  const merged = mergeLateInferHubWithFallback([
    ...(lateInferHubCatalog.models ?? []),
    ...(lateInferHubCatalog.groups ?? []).flatMap((group) => group.models ?? []),
  ]);
  return merged.find((model) => model.id === id);
}

function lateInferCheckStatusText(hubId) {
  const id = String(hubId ?? "").trim();
  if (!id) return "Select a Hub snapshot to see if it is already compiled on your computer.";
  if (lateInferCompiledIdSet().has(id)) {
    return `${id} is already compiled on your computer — it is in the pane on the right.`;
  }
  const model = findLateInferHubCatalogModel(id);
  if (model?.configStatus === "missing") {
    return `${id} has no config.json on Hugging Face — late-infer cannot compile it. Use a safetensors Instruct snapshot (GGUF → llama.cpp).`;
  }
  if (model && hubModelIsLoadable(model) === false) {
    return `${id} is not loadable on this GPU${model.serveBlockedReason ? ` — ${model.serveBlockedReason}` : "."}`;
  }
  if (model?.gated || model?.gatedNeedsLicense) {
    if (!lateInferHfTokenSet) {
      return `${id} is gated. Accept the license on the Hugging Face model card, then save a read token in Settings → Local models before Download.`;
    }
    return `${id} is gated. Accept the Hugging Face license (if you have not), then Download with your saved read token.`;
  }
  return `${id} is not compiled on your computer yet. Download first.`;
}

function refreshLateInferCheckStatus() {
  const el = $("lateinfer-check-status");
  if (!el) return;
  const hubId = resolveLateInferHubIdForDownload();
  const on = Boolean(hubId) && lateInferCompiledIdSet().has(hubId);
  el.classList.toggle("ok", on);
  el.textContent = lateInferCheckStatusText(hubId);
}

function runLateInferHubCheck() {
  const hubId = resolveLateInferHubIdForDownload();
  const on = Boolean(hubId) && lateInferCompiledIdSet().has(hubId);
  if (hubId) {
    lateInferHubId = hubId;
    const hubInput = $("lateinfer-hub");
    if (hubInput) hubInput.value = hubId;
  }
  if (on && hubId) {
    lateInferCompiledId = hubId;
    refreshLateInferCompiledList();
  }
  refreshLateInferHubList();
  refreshLateInferCheckStatus();
  const status = $("local-models-status");
  if (status && hubId) {
    status.innerHTML = flash(
      on ? `${hubId} is already compiled on your computer.` : `${hubId} is not compiled on your computer yet.`,
      on ? "ok" : "bad",
    );
  }
}

function lateInferGpuState() {
  return localServers.gpu ?? {};
}

/** Picker card, else idle-primary on your computer. Download/Start both send this so compile pins that PCI. */
function lateInferGpuIdForPost() {
  if (lateInferGpuId && lateInferGpuId !== "auto") return lateInferGpuId;
  const primary = lateInferGpuState().primaryId;
  return typeof primary === "string" && primary.trim() ? primary : undefined;
}

function gpuVendorLabel(vendor) {
  if (vendor === "intel") return "Intel (Arc / XPU / Level Zero)";
  if (vendor === "amd") return "AMD";
  if (vendor === "nvidia") return "NVIDIA";
  return "";
}

function lateInferStartHardBlocked() {
  const gpu = lateInferGpuState();
  if (gpu.runtimeOk === false && gpu.cards?.[0]?.vendor !== "intel") return true;
  if (gpu.needsPicker && (!lateInferGpuId || lateInferGpuId === "auto")) return true;
  return false;
}

function lateInferSelectedCompiledGpuReady() {
  const selected = compiledRowForId(resolveLateInferIdForStart());
  return Boolean(selected && selected.gpuReady === true);
}

function lateInferStartBlockedByGpu() {
  const gpu = lateInferGpuState();
  if (gpu.runtimeOk === false) return true;
  // Selected IR-ready snapshot wins over serving/YAML overlay without IR.
  if (lateInferSelectedCompiledGpuReady()) {
    if (gpu.needsPicker && (!lateInferGpuId || lateInferGpuId === "auto")) return true;
    return false;
  }
  if (gpu.hostRamOk === false) return true;
  if (gpu.ovIrPresent === false && (gpu.cards?.[0]?.vendor === "intel" || gpu.live?.vendor === "intel" || gpu.visibleIds?.[0]?.startsWith?.("intel:"))) {
    return true;
  }
  if (gpu.needsPicker && (!lateInferGpuId || lateInferGpuId === "auto")) return true;
  return false;
}

function lateInferStartRefuseText() {
  const gpu = lateInferGpuState();
  const selected = compiledRowForId(resolveLateInferIdForStart());
  if (selected && selected.gpuReady === false) return intelIrMissingStartMessage();
  if (selected && selected.gpuReady === true) {
    if (gpu.runtimeOk === false) return gpu.runtimeReason || gpu.reason || gpu.compileReason || "";
    if (gpu.needsPicker) return gpu.reason || "Pick a GPU on your computer.";
    return "";
  }
  if (gpu.ovIrPresent === false && (gpu.live?.vendor === "intel" || gpu.cards?.[0]?.vendor === "intel")) {
    return intelIrMissingStartMessage();
  }
  if (gpu.hostRamOk === false) return gpu.hostRamReason || intelIrMissingStartMessage();
  if (gpu.runtimeOk === false) return gpu.runtimeReason || gpu.reason || gpu.compileReason || "";
  if (gpu.needsPicker) return gpu.reason || "Pick a GPU on your computer.";
  return "";
}

function lateInferSelectedRowNotGpuReady() {
  const selected = compiledRowForId(resolveLateInferIdForStart());
  if (selected) return selected.gpuReady === false;
  const gpu = lateInferGpuState();
  return gpu.ovIrPresent === false && (gpu.live?.vendor === "intel" || gpu.cards?.[0]?.vendor === "intel");
}

function lateInferStartNotReadyForGpu() {
  if (!compiledHubModelsForPane().length) return false;
  if (lateInferSelectedRowNotGpuReady()) return true;
  if (lateInferSelectedCompiledGpuReady()) return false;
  return lateInferGpuState().hostRamOk === false;
}

function renderLateInferStartBannerHtml() {
  const notReady = lateInferStartNotReadyForGpu();
  const text = lateInferStartError || lateInferStartRefuseText();
  if (!text && !notReady) {
    return `<div id="lateinfer-start-banner" class="lateinfer-start-banner" hidden></div>`;
  }
  const body = text || intelIrMissingStartMessage();
  const title = notReady
    ? `<span class="pill bad">not ready for GPU Start</span> <strong>Start refused on your computer</strong>`
    : `<span class="pill bad">Start refused</span> <strong>Start failed on your computer</strong>`;
  return `<div class="lateinfer-start-banner lateinfer-start-refuse" id="lateinfer-start-banner" role="alert" tabindex="-1">
    <p class="lateinfer-start-banner-title">${title}</p>
    <p class="lateinfer-start-banner-body">${escapeHtml(body)}</p>
  </div>`;
}

function renderLateInferLocalStartInnerHtml() {
  const row = compiledRowForId(resolveLateInferIdForStart());
  const lateInferDlBusy = lateInferBusy(localServers.lateinfer);
  const convertBlocked = row && row.gpuReady !== true && row.convertOk === false;
  const convertBtn =
    row && row.gpuReady !== true
      ? `<button type="button" class="btn secondary" id="lateinfer-convert" ${lateInferDlBusy || lateInferIsStarting() ? "disabled" : ""}>Convert for GPU Start</button>`
      : "";
  const startHard = lateInferStartHardBlocked();
  const notReady = lateInferStartNotReadyForGpu();
  const startDisabled =
    localServers.lateinfer?.running ||
    lateInferIsStarting() ||
    lateInferDlBusy ||
    !compiledHubModelsForPane().length ||
    startHard;
  const startErr = lateInferStartError || lateInferStartRefuseText();
  const startErrHtml = startErr
    ? `<p class="error lateinfer-start-refuse" id="lateinfer-start-btn-error" role="alert">${escapeHtml(startErr)}</p>`
    : "";
  const convertNote = convertBlocked
    ? `<p class="error" id="lateinfer-convert-reason">${escapeHtml(row.convertReason || intelIrMissingStartMessage())}</p>`
    : "";
  const blockedClass = notReady && !startDisabled ? " lateinfer-start-blocked" : "";
  const blockedAttr =
    notReady && !startDisabled
      ? ` aria-disabled="true" title="${escapeHtml(startErr || intelIrMissingStartMessage())}"`
      : "";
  return `${convertNote}${startErrHtml}<div class="actions">
              ${convertBtn}
              <button type="button" class="btn${blockedClass}" id="lateinfer-start" ${startDisabled ? "disabled" : ""}${blockedAttr}>Start on your computer</button>
              <button type="button" class="btn secondary" id="lateinfer-stop" ${localServers.lateinfer?.running ? "" : "disabled"}>Stop</button>
            </div>`;
}

function scrollLateInferStartBannerIntoView() {
  const banner = $("lateinfer-start-banner");
  if (!banner || banner.hidden) return;
  try {
    banner.focus({ preventScroll: true });
  } catch {
    /* ignore */
  }
  banner.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function lateInferCompileBlockedByGpu() {
  const gpu = lateInferGpuState();
  return gpu.compileOk === false;
}

function renderLateInferCompileDetectHtml() {
  const gpu = lateInferGpuState();
  const cards = gpu.cards ?? [];
  const items = cards
    .map((card) => {
      const vendor = card.vendorLabel || gpuVendorLabel(card.vendor) || card.vendor || "";
      const role = card.igpu ? "integrated" : card.display ? "display · 70% cap" : "idle · full VRAM";
      const bus = card.busId ? ` · ${card.busId}` : "";
      return `<li>${escapeHtml(vendor)} · ${escapeHtml(card.name)} · ${escapeHtml(card.vramLabel || "")} · ${escapeHtml(role)}${escapeHtml(bus)}</li>`;
    })
    .join("");
  const list = items
    ? `<ul class="lateinfer-gpu-cards" id="lateinfer-gpu-detect-list">${items}</ul>`
    : `<p class="muted" id="lateinfer-gpu-detect-list">No discrete GPU on your computer — compile would use CPU.</p>`;
  const blocked = gpu.compileOk === false;
  const warn = blocked
    ? `<p class="error" id="lateinfer-compile-gpu-reason">${escapeHtml(gpu.compileReason || gpu.runtimeReason || "")}</p>`
    : `<p class="muted" id="lateinfer-compile-gpu-reason">Compile target on your computer: ${escapeHtml(gpu.compileTarget || gpu.label || "idle GPU")}. Detected before Download — not assumed NVIDIA.</p>`;
  return `
    <div class="lateinfer-gpu lateinfer-gpu-detect" id="lateinfer-gpu-detect">
      <p><strong>GPUs on your computer</strong></p>
      ${list}
      ${warn}
    </div>`;
}

function renderLateInferGpuControlsHtml() {
  const gpu = lateInferGpuState();
  const cards = gpu.cards ?? [];
  const selected = lateInferGpuId || "auto";
  const options = [`<option value="auto"${selected === "auto" ? " selected" : ""}>Auto (idle GPU on your computer)</option>`]
    .concat(
      cards.map((card) => {
        const vendor = card.vendorLabel || gpuVendorLabel(card.vendor);
        const where = card.display
          ? `display${card.connectedDisplays ? ` · ${card.connectedDisplays} displays` : ""} · 70% cap`
          : "idle · full VRAM";
        const bus = card.busId ? ` · ${card.busId}` : "";
        const labeled = vendor ? `${vendor} · ${card.name}` : card.name;
        return `<option value="${escapeHtml(card.id)}"${selected === card.id ? " selected" : ""}>${escapeHtml(labeled)} · ${escapeHtml(card.vramLabel || "")} · ${where}${escapeHtml(bus)}</option>`;
      }),
    )
    .join("");
  const runtime =
    gpu.runtimeOk === false
      ? `<p class="error" id="lateinfer-gpu-runtime">${escapeHtml(gpu.runtimeReason || gpu.reason || gpu.compileReason || "")}</p>`
      : gpu.needsPicker
        ? `<p class="error" id="lateinfer-gpu-runtime">${escapeHtml(gpu.reason || "Pick a GPU on your computer.")}</p>`
        : "";
  const serveHint =
    gpu.runtimeOk !== false && gpu.serveHint
      ? `<p class="muted" id="lateinfer-gpu-serve-hint">${escapeHtml(gpu.serveHint)}</p>`
      : "";
  return `
    <div class="lateinfer-gpu" id="lateinfer-gpu">
      <p id="lateinfer-gpu-label"><strong>${escapeHtml(gpu.label || "GPUs on your computer")}</strong></p>
      <p class="muted" id="lateinfer-gpu-reason">${escapeHtml(gpu.reason || "Start prefers the idle GPU on your computer (the card not driving displays).")}</p>
      ${runtime}
      ${serveHint}
      <label class="field" for="lateinfer-gpu-pick">GPU on your computer
        <select id="lateinfer-gpu-pick" aria-label="GPU on your computer">${options}</select>
      </label>
      <label class="check-row"><input type="checkbox" id="lateinfer-use-all-gpus"${lateInferUseAllGpus ? " checked" : ""} /> Use all GPUs on this computer</label>
      <p class="muted">Idle card: 100% VRAM. Display GPU only if the snapshot needs more than that card, and then at 70% so the desktop keeps headroom. This runtime cannot apply a 70% cap, so Start will not overflow onto the display GPU.</p>
    </div>`;
}

function lateInferStartingFlag() {
  try {
    return sessionStorage.getItem(LATE_INFER_STARTING_KEY) === "1";
  } catch {
    return false;
  }
}

function setLateInferStartingFlag(on) {
  try {
    if (on) sessionStorage.setItem(LATE_INFER_STARTING_KEY, "1");
    else sessionStorage.removeItem(LATE_INFER_STARTING_KEY);
  } catch {
    /* private mode */
  }
}

function lateInferIsStarting() {
  const li = localServers.lateinfer;
  if (li?.starting === true || li?.servePhase === "starting") return true;
  const live = lateInferGpuState().live;
  if (live?.servePhase === "starting") return true;
  return lateInferStartingFlag();
}

function applyDirectLateInferHealth(health) {
  if (!health || health.ok !== true || health.name !== "late-infer") return;
  const device = typeof health.device === "string" ? health.device : "";
  const deviceKind = typeof health.device_kind === "string" ? health.device_kind : "";
  const weightsInHostRam = health.weights_in_host_ram === true;
  const gpuRunning =
    !weightsInHostRam &&
    (/intel-xpu|openvino|\bxpu\b/i.test(deviceKind || device) ||
      /\bcuda\b/i.test(deviceKind || device) ||
      /\bhip\b|\brocm\b/i.test(deviceKind || device));
  if (localServers.lateinfer) {
    localServers.lateinfer = {
      ...localServers.lateinfer,
      running: true,
      ready: true,
      device,
      deviceKind,
      weightsInHostRam,
      gpuRunning,
      servePhase: gpuRunning ? "gpu-running" : weightsInHostRam ? "host-ram" : "ready",
    };
  }
  if (!localServers.gpu) localServers.gpu = {};
  localServers.gpu.live = {
    ...(localServers.gpu.live ?? {}),
    servePhase: gpuRunning ? "gpu-running" : weightsInHostRam ? "host-ram" : "ready",
    gpuRunning,
    weightsInHostRam,
    device,
    deviceKind,
    headline: gpuRunning ? "GPU running" : weightsInHostRam ? "Weights in host RAM" : "late-infer answering",
    detail: gpuRunning
      ? `late-infer is up on 127.0.0.1:8010 (${deviceKind || device || "GPU"}). VRAM is the idle card, not host RAM.`
      : weightsInHostRam
        ? "Weights landed in host RAM, not VRAM on the idle GPU on your computer. This is not GPU running."
        : "127.0.0.1:8010 answered. Not GPU running until health reports Intel XPU / CUDA / HIP.",
    up: gpuRunning,
  };
  if (gpuRunning || weightsInHostRam) setLateInferStartingFlag(false);
}

async function probeLateInferHealthDirect() {
  try {
    const response = await fetch("http://127.0.0.1:8010/health", {
      method: "GET",
      headers: { accept: "application/json" },
      signal: AbortSignal.timeout(800),
    });
    if (!response.ok) return null;
    const payload = await response.json();
    applyDirectLateInferHealth(payload);
    return payload;
  } catch {
    return null;
  }
}

function renderLateInferGpuLiveHtml() {
  const gpu = lateInferGpuState();
  const live = gpu.live ?? {};
  const li = localServers.lateinfer ?? {};
  const starting = lateInferIsStarting();
  const disconnected = li.guiDisconnected === true;
  const servePhase = live.servePhase || li.servePhase || (starting ? "starting" : li.ready ? "ready" : "down");
  const gpuRunning = live.gpuRunning === true || servePhase === "gpu-running";
  const hostRam = live.weightsInHostRam === true || servePhase === "host-ram";
  const vendor = live.vendorLabel || gpuVendorLabel(live.vendor) || "";
  const bus = live.busId || "";
  const role = live.role === "display" ? "display" : live.role === "integrated" ? "integrated" : "idle";
  const vram = live.vramPairLabel || "";
  let pill = `<span class="pill warn">GPU down</span>`;
  let headline = live.headline || "GPU down";
  let detail = live.detail || "late-infer is not serving on 127.0.0.1:8010.";
  if (gpuRunning) {
    pill = `<span class="pill ok">GPU running</span>`;
    headline = "GPU running";
    detail = live.detail || "late-infer answered /v1/models on the idle GPU on your computer (Intel XPU / CUDA / HIP).";
  } else if (hostRam) {
    pill = `<span class="pill warn">host RAM</span>`;
    headline = "Weights in host RAM";
    detail =
      live.detail ||
      gpu.hostRamReason ||
      "Weights landed in host RAM, not VRAM on the idle GPU on your computer. This is not GPU running.";
  } else if (servePhase === "exited") {
    pill = `<span class="pill warn">GPU down</span>`;
    headline = "GPU down";
    detail =
      "Start exited before GPU running. This is not success — if the box ran out of memory, the idle card never came up.";
  } else if (starting || servePhase === "starting") {
    pill = `<span class="pill warn">starting</span>`;
    headline = "starting on idle GPU…";
    detail = disconnected
      ? "Lost connection to the orchestrator GUI (127.0.0.1). Start may still be loading on the idle GPU on your computer. This is not GPU running — if the box ran out of memory, Start failed. If you opened this page without ?token=, paste the token from the terminal."
      : live.detail ||
        "Start is loading weights on the idle GPU on your computer. First load can take several minutes. This is not GPU running yet.";
  }
  const meta = [
    vendor,
    live.name || gpu.label || "",
    bus,
    role,
    gpuRunning || live.up ? "up" : "down",
    vram ? `VRAM ${vram}` : "",
    live.usedSource ? `(${live.usedSource})` : "",
  ]
    .filter(Boolean)
    .join(" · ");
  return `
    <div class="lateinfer-gpu lateinfer-gpu-live" id="gpu-status">
      <div id="lateinfer-gpu-live">
        <p id="lateinfer-gpu-live-headline">${pill} <strong>${escapeHtml(headline)}</strong></p>
        <p class="muted" id="lateinfer-gpu-live-meta">${escapeHtml(meta || "GPUs on your computer")}</p>
        <p class="muted" id="lateinfer-gpu-live-state">${escapeHtml(detail)}</p>
      </div>
    </div>`;
}

function applyLocalModelsDisconnect(error) {
  const raw = error instanceof Error ? error.message : String(error);
  const starting = lateInferIsStarting();
  if (localServers.lateinfer) localServers.lateinfer.guiDisconnected = true;
  if (!$("hf-token-form") || !$("gpu-status")) renderLocalModels();
  const status = $("local-models-status");
  if (status) {
    const extra = starting
      ? " Start may still be loading on the idle GPU on your computer. This is not GPU running — if the box ran out of memory, Start failed."
      : " Stay on Local models. If you opened this page without ?token=, paste the token from the terminal.";
    status.innerHTML = flash(`${raw}${extra}`, "bad");
  }
}

async function loadLateInferHubModels(query) {
  lateInferHubCatalog.loading = true;
  const q = String(query ?? lateInferHubSearch ?? "").trim();
  try {
    const data = await api(`/api/local-servers/hub-models?q=${encodeURIComponent(q)}`);
    const models = mergeLateInferHubWithFallback(hubCatalogModelsFromResponse(data));
    lateInferHubCatalog = {
      models,
      groups: groupLateInferHubModels(models),
      hint: typeof data.hint === "string" ? data.hint : "",
      defaultId: typeof data.defaultId === "string" && data.defaultId.trim() ? data.defaultId.trim() : DEFAULT_LATE_INFER_HUB,
      budgetGiB: data.budgetGiB ?? data.budgetGb ?? null,
      query: q,
      offline: data.offline === true,
      loaded: true,
      loading: false,
      error: typeof data.error === "string" ? data.error : "",
    };
  } catch {
    const models = fallbackLateInferHubModels();
    lateInferHubCatalog = {
      models,
      groups: groupLateInferHubModels(models),
      hint: "",
      defaultId: DEFAULT_LATE_INFER_HUB,
      budgetGiB: null,
      query: q,
      offline: true,
      loaded: true,
      loading: false,
      error: "offline",
    };
  }
  if (pageId() === "local-models") {
    refreshLateInferHubList();
    const hint = $("lateinfer-hub-hint");
    if (hint) hint.textContent = lateInferHubHintText();
    const status = $("lateinfer-hub-status");
    if (status) {
      status.hidden = !lateInferHubCatalog.offline;
      status.textContent = lateInferHubCatalog.offline
        ? "Hugging Face Hub is offline. Listed ids below still Download on your computer."
        : "";
    }
  }
}

function syncLateInferPoll(infer) {
  const busy = lateInferJobBusy(infer) || lateInferIsStarting();
  if (busy) {
    lateInferDownloadBusy = lateInferJobBusy(infer);
    if (lateInferPollTimer) return;
    lateInferPollTimer = setInterval(async () => {
      try {
        await loadLocalServers();
        await probeLateInferHealthDirect();
        const li = localServers.lateinfer;
        if (pageId() === "local-models") {
          renderLocalModels();
          if (lateInferStartError && $("local-models-status")) {
            $("local-models-status").innerHTML = flash(lateInferStartError, "bad");
          }
        }
        if (!lateInferJobBusy(li) && !lateInferIsStarting()) {
          clearInterval(lateInferPollTimer);
          lateInferPollTimer = null;
          lateInferDownloadBusy = false;
          await loadLocalModels();
          if (pageId() === "local-models") {
            renderLocalModels();
            if (lateInferStartError && $("local-models-status")) {
              $("local-models-status").innerHTML = flash(lateInferStartError, "bad");
            }
          }
        }
      } catch (error) {
        await probeLateInferHealthDirect();
        if (pageId() === "local-models") applyLocalModelsDisconnect(error);
      }
    }, 800);
    return;
  }
  lateInferDownloadBusy = false;
  if (lateInferPollTimer) {
    clearInterval(lateInferPollTimer);
    lateInferPollTimer = null;
  }
}

function threadIdFromHash() {
  const raw = location.hash.replace("#", "");
  const parts = raw.split("/");
  if (parts[0] === "chat" && parts[1]) return parts[1];
  return null;
}

async function api(path, options = {}) {
  if (!path.startsWith("/")) {
    throw new Error("API calls must stay on this GUI origin (127.0.0.1).");
  }
  if (!token) {
    const err = new Error(
      "Missing session token. Open the GUI URL printed by npm run gui (it includes ?token=), or paste the token from the terminal.",
    );
    err.status = 401;
    throw err;
  }
  const headers = { ...(options.headers ?? {}) };
  headers.Authorization = `Bearer ${token}`;
  if (options.body && !headers["content-type"]) headers["content-type"] = "application/json";
  let response;
  try {
    response = await fetch(path, { ...options, headers, credentials: "same-origin" });
  } catch (error) {
    const raw = error instanceof Error ? error.message : String(error);
    throw new Error(
      /NetworkError|Failed to fetch|Load failed|fetch failed|network/i.test(raw)
        ? "Lost connection to the orchestrator GUI (127.0.0.1). Stay on Local models while late-infer starts (first load can take several minutes). If you opened this page without ?token=, paste the token from the terminal."
        : raw,
    );
  }
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { error: text || response.statusText };
  }
  if (response.status === 401) {
    const err = new Error(
      "Session token missing or rejected. Open the URL printed by npm run gui (it includes ?token=).",
    );
    err.status = 401;
    throw err;
  }
  if (!response.ok && response.status !== 202) {
    const err = new Error(data.error || `HTTP ${response.status}`);
    err.status = response.status;
    throw err;
  }
  return data;
}

function pageId() {
  return (location.hash.replace("#", "") || "chat").split("/")[0];
}

function settingsPages() {
  return ["overview", "specialists", "backends", "runs", "dispatch", "local-models", "allowlist", "updates", "config"];
}

function setActiveNav() {
  const current = pageId();
  for (const link of document.querySelectorAll("#nav a")) {
    link.classList.toggle("active", link.dataset.page === current);
  }
  const details = $("settings-nav");
  if (details) details.open = settingsPages().includes(current);
}

function flash(message, kind = "ok") {
  return `<div class="flash ${kind}">${escapeHtml(message)}</div>`;
}

function isTransientVllmWaitLog(text) {
  if (!text) return true;
  const t = String(text).trim();
  if (/^(fetch failed|Failed to fetch|Load failed|NetworkError|ECONNREFUSED|UND_ERR_CONNECT)/i.test(t)) return true;
  if (/fetch failed/i.test(t) && t.length < 120) return true;
  return /Waiting for GET \/v1\/models/i.test(t) && /connection refused/i.test(t);
}

function vllmStartingDetail(vllm) {
  const raw = vllm.startJob?.lastLog || vllm.lastLog || "";
  if (isTransientVllmWaitLog(raw)) {
    return "Waiting for GET /v1/models… Intel Docker is loading weights (often several minutes). Connection refused is normal until the API binds.";
  }
  return raw;
}

function vllmErrorDetail(vllm) {
  const raw = vllm.startJob?.error || vllm.lastError || "vLLM failed to start";
  const cleaned = String(raw).replace(/(Last log:\s*)fetch failed/gi, "$1(no HTTP yet while loading weights)");
  if (isTransientVllmWaitLog(cleaned)) {
    return "vLLM exited before GET /v1/models was ready. That is not a GUI network error — inspect the container log. Connection refused during weight load is expected.";
  }
  return cleaned;
}

function pill(ready, reason) {
  if (ready) return `<span class="pill ok">ready</span>`;
  const label = /vLLM not running|Ollama not running|llama\.cpp not running/i.test(reason ?? "") ? "not ready" : "missing";
  return `<span class="pill warn" title="${escapeHtml(reason ?? "")}">${label}</span>`;
}

function writePill(writes) {
  return writes
    ? `<span class="pill">local writes</span>`
    : `<span class="pill">text only</span>`;
}

function formatTime(ms) {
  if (!ms) return "";
  try {
    return new Date(ms).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  } catch {
    return "";
  }
}

function upsertThread(summary) {
  threads = [summary, ...threads.filter((t) => t.id !== summary.id)];
}

function renderThreadList() {
  const el = $("thread-list");
  if (!el) return;
  const active = threadIdFromHash() ?? currentThread?.id;
  if (!threads.length) {
    el.innerHTML = `<p class="muted" style="padding:0.4rem 0.55rem;margin:0">No chats yet</p>`;
    return;
  }
  el.innerHTML = threads
    .map((t) => {
      const agents = (t.agents ?? [])
        .filter((a) => a && a !== "user" && a !== "orchestrator")
        .slice(0, 3)
        .map((a) => backendDisplayName(a))
        .join(" · ");
      return `<div class="thread-row ${t.id === active ? "active" : ""}" role="listitem">
        <button type="button" class="thread-item ${t.id === active ? "active" : ""}" data-thread="${escapeHtml(t.id)}">
        <span>${escapeHtml(t.title || "New chat")}</span>
        ${agents ? `<small>${escapeHtml(agents)}</small>` : ""}
      </button>
        <button type="button" class="thread-delete" data-delete-thread="${escapeHtml(t.id)}" aria-label="Delete chat">Delete</button>
      </div>`;
    })
    .join("");
}

function isLegacyLocalEngineBackend(b) {
  const t = b?.type ?? "";
  const id = String(b?.id ?? "");
  return t === "vllm" || t === "ollama" || t === "llamacpp" || /^(vllm|ollama|llamacpp)/.test(id);
}

function pinOptions(selected) {
  const extras = (catalog.backends ?? [])
    .filter((b) => !["cursor-local", "cursor-cloud", "gemini"].includes(b.id))
    .filter((b) => !isLegacyLocalEngineBackend(b))
    .map((b) => `<option value="${escapeHtml(b.id)}" ${selected === b.id ? "selected" : ""}>${escapeHtml(b.nickname ? `${b.nickname} (${b.id})` : b.id)}</option>`)
    .join("");
  const backendPin = selected && !["auto", "debate", "single"].includes(selected) ? selected : "";
  return `
    <option value="" ${!backendPin ? "selected" : ""}>Pin backend</option>
    <option value="local" ${backendPin === "local" ? "selected" : ""}>Local</option>
    <option value="cloud" ${backendPin === "cloud" ? "selected" : ""}>Cloud</option>
    <option value="gemini" ${backendPin === "gemini" ? "selected" : ""}>Gemini</option>
    ${extras}
  `;
}

function modeFromPin(pin) {
  if (pin === "debate" || pin === "single" || pin === "auto" || !pin) return pin || "auto";
  return "single";
}

function composerPin() {
  const modeBtn = document.querySelector(".mode-btn[aria-pressed='true']");
  const mode = modeBtn?.getAttribute("data-mode") || "auto";
  const backend = $("route-pin")?.value;
  if (backend) return backend;
  return mode;
}

function latestChip(thread) {
  const msgs = [...(thread?.messages ?? [])].reverse();
  return msgs.find((m) => m.chip)?.chip || "Auto";
}

function suggestedButton(action) {
  if (!action) return "";
  return `<div class="suggested"><button type="button" class="btn" data-chat-action="${escapeHtml(action.action)}" data-payload="${escapeHtml(JSON.stringify(action.payload ?? {}))}">${escapeHtml(action.label)}</button></div>`;
}

function estimateCompletionTokens(text) {
  const s = String(text ?? "");
  if (!s) return 0;
  // Rough research estimate (not tokenizer-accurate): ~4 chars per token for English/code mix.
  return Math.max(1, Math.round(s.length / 4));
}

function formatTokensPerSec(tps) {
  const n = Number(tps);
  if (!Number.isFinite(n) || n <= 0) return "";
  if (n >= 100) return `${Math.round(n)} tok/s`;
  if (n >= 10) return `${n.toFixed(1)} tok/s`;
  return `${n.toFixed(2)} tok/s`;
}

function messageTokensPerSec(m, now = Date.now()) {
  if (!chatShowTokensPerSec || !m || m.role === "user") return "";
  if (typeof m.tokensPerSec === "number" && Number.isFinite(m.tokensPerSec) && m.tokensPerSec > 0) {
    return formatTokensPerSec(m.tokensPerSec);
  }
  const body = String(m.content || "");
  if (!body.trim()) return "";
  const started = m.thinkingStartedAt || m.createdAt;
  if (!started) return "";
  const elapsedSec = Math.max(0.05, (now - started) / 1000);
  const tokens = typeof m.completionTokensEst === "number" && m.completionTokensEst > 0
    ? m.completionTokensEst
    : estimateCompletionTokens(body);
  return formatTokensPerSec(tokens / elapsedSec);
}

function thinkingChipLabel(m, now = Date.now()) {

  const phase = m.thinkingPhase || (m.status === "streaming" ? "streaming" : "waiting");
  const started = m.thinkingStartedAt || m.createdAt || now;
  const elapsed = Math.max(0, Math.floor((now - started) / 1000));
  const clock = elapsed < 60 ? `${elapsed}s` : `${Math.floor(elapsed / 60)}m ${elapsed % 60}s`;
  return `${m.label || m.speaker || "Model"} · ${clock} · ${phase}`;
}

function renderMessages() {
  const list = $("thread-messages");
  if (!list) return;
  const messages = (currentThread?.messages ?? []).filter((m) => m.phase !== "approval");
  if (!messages.length) {
    list.innerHTML = `
      <div class="empty-chat">
        <h2>Ask the team</h2>
        <p>Type naturally. Auto debates build/fix/review when Cursor or two backends are ready. Speakers show a thinking chip (name · elapsed · waiting/streaming/debating) until the reply lands. Use Debate for a round-table (one bubble per speaker). Implement/install shows a pending card — Approve before writes or host installs (Unity, apt). Q&A stays unblocked.</p>
      </div>`;
    return;
  }
  const nearBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 80;
  const now = Date.now();
  list.innerHTML = messages
    .map((m) => {
      const role = m.role === "user" ? "user" : "assistant";
      const thinking = m.status === "thinking" || m.status === "streaming";
      const round = m.round ? ` · round ${m.round}` : m.phase === "synthesis" ? " · synthesis" : "";
      const chip = thinking
        ? `<div class="thinking-chip" data-thinking-id="${escapeHtml(m.id)}"><span class="thinking-dot" aria-hidden="true"></span><span class="thinking-label">${escapeHtml(thinkingChipLabel(m, now))}</span></div>`
        : "";
      const body = m.content || m.error || "";
      const bubble =
        !thinking || body
          ? `<div class="bubble ${m.status === "error" ? "error" : ""} ${m.status === "streaming" ? "streaming" : ""}">${escapeHtml(body || (thinking ? "" : "…"))}</div>`
          : "";
      return `<article class="msg ${role}" data-id="${escapeHtml(m.id)}">
        <div class="msg-meta">
          ${avatarMarkup(m.speaker)}
          <span class="speaker">${escapeHtml(m.label || m.speaker || role)}</span>
          <span>${escapeHtml(formatTime(m.createdAt))}${escapeHtml(round)}</span>
          ${(() => {
            const tps = messageTokensPerSec(m, now);
            return tps
              ? `<span class="tok-per-sec" title="Research estimate (chars÷4 / elapsed) — not a billing meter">${escapeHtml(tps)}</span>`
              : "";
          })()}
        </div>
        ${chip}
        ${bubble}
        ${m.suggestedAction ? suggestedButton(m.suggestedAction) : ""}
      </article>`;
    })
    .join("");
  const pendingHtml = pendingCard(currentThread?.pendingApproval);
  if (pendingHtml) list.insertAdjacentHTML("beforeend", pendingHtml);
  if (nearBottom || sending) list.scrollTop = list.scrollHeight;
}

function pendingCard(pending) {
  if (!pending || pending.status !== "pending") return "";
  const cmds = (pending.commands ?? []).map((c) => `<li class="mono">${escapeHtml(c)}</li>`).join("");
  const warn = pending.systemWideNote
    ? `<p class="pending-warn">${escapeHtml(pending.systemWideNote)}</p>`
    : `<p class="muted">Implement/install needs Approve. Q&A and debate text already ran plan-only.</p>`;
  return `<aside class="pending-card" data-pending="${escapeHtml(pending.id)}">
    <h3>Pending actions</h3>
    <p><strong>Proposed cwd</strong> <span class="mono">${escapeHtml(pending.cwd ?? "")}</span></p>
    <p><strong>Specialist</strong> ${escapeHtml(pending.specialist)} · ${escapeHtml(pending.label)}</p>
    <pre class="pending-plan">${escapeHtml(pending.summary ?? "")}</pre>
    ${cmds ? `<p><strong>Commands</strong></p><ul>${cmds}</ul>` : ""}
    ${warn}
    <label class="field">Optional comment
      <input id="approval-comment" type="text" placeholder="Optional note" autocomplete="off" />
    </label>
    <div class="actions">
      <button type="button" class="btn" data-approval="approve">Approve</button>
      <button type="button" class="btn danger" data-approval="reject">Reject</button>
    </div>
  </aside>`;
}

function threadPickOptions() {
  const active = threadIdFromHash() ?? currentThread?.id;
  if (!threads.length) {
    return `<option value="">New chat</option>`;
  }
  return threads
    .map((t) => {
      const selected = t.id === active ? "selected" : "";
      return `<option value="${escapeHtml(t.id)}" ${selected}>${escapeHtml(t.title || "New chat")}</option>`;
    })
    .join("");
}

function renderChatHeader() {
  const chip = $("route-chip");
  const title = $("chat-title");
  if (title) title.textContent = currentThread?.title || "New chat";
  if (chip) chip.textContent = latestChip(currentThread);
  const pick = $("chat-thread-pick");
  if (pick && document.activeElement !== pick) {
    pick.innerHTML = threadPickOptions();
  }
  const pin = $("route-pin");
  if (pin && document.activeElement !== pin) {
    pin.innerHTML = pinOptions(currentThread?.pin || "auto");
  }
  const mode = modeFromPin(currentThread?.pin || "auto");
  for (const btn of document.querySelectorAll(".mode-btn")) {
    btn.setAttribute("aria-pressed", btn.getAttribute("data-mode") === mode ? "true" : "false");
  }
}

function ensureChatLayout() {
  const main = $("main");
  main.classList.add("chat-main");
  if ($("chat-root") && $("chat-thread-pick")) {
    renderChatHeader();
    renderMessages();
    renderThreadList();
    return;
  }
  main.innerHTML = `
    <div id="chat-root" class="chat-root">
      <header class="chat-header">
        <div>
          <h1 id="chat-title">New chat</h1>
          <p class="muted" style="margin:0">Enter to send · Shift+Enter for a newline</p>
        </div>
        <div class="chip-row">
          <span class="route-chip" id="route-chip" title="Informational — Auto chooses speakers">Auto</span>
          <label class="sr-only" for="chat-thread-pick">Switch chat</label>
          <select id="chat-thread-pick" class="chat-thread-pick" aria-label="Switch chat">${threadPickOptions()}</select>
          <button type="button" class="btn secondary" data-new-chat>New chat</button>
          <details class="chat-settings">
            <summary>Settings</summary>
            <nav class="chat-settings-menu" aria-label="Settings">
              <a href="#chat" data-page="chat">Chat</a>
              <a href="#overview" data-page="overview">Overview</a>
              <a href="#backends" data-page="backends">Backends</a>
              <a href="#local-models" data-page="local-models">Local models</a>
              <a href="#allowlist" data-page="allowlist">Allowlist</a>
              <a href="#updates" data-page="updates">Updates</a>
              <a href="#specialists" data-page="specialists">Specialists</a>
              <a href="#config" data-page="config">Config</a>
              <a href="#dispatch" data-page="dispatch">Run workflow</a>
              <a href="#runs" data-page="runs">Runs</a>
              ${themePickerMarkup("theme-select-chat")}
            </nav>
          </details>
        </div>
      </header>
      <div id="thread-messages" class="messages" aria-live="polite"></div>
      <form id="composer-form" class="composer">
        <div class="composer-row">
          <div class="mode-toggle" role="group" aria-label="Chat mode">
            <button type="button" class="mode-btn" data-mode="auto" aria-pressed="true">Auto</button>
            <button type="button" class="mode-btn" data-mode="debate" aria-pressed="false">Debate</button>
            <button type="button" class="mode-btn" data-mode="single" aria-pressed="false">Single</button>
          </div>
          <label class="sr-only" for="route-pin">Pin backend</label>
          <select id="route-pin" name="pin" class="pin" aria-label="Pin backend">${pinOptions("auto")}</select>
          <label class="sr-only" for="composer-input">Message</label>
          <textarea id="composer-input" name="message" required placeholder="Troubleshoot this PR, draft a plan, ask what fits your GPUs…"></textarea>
          <button type="submit" id="composer-send">Send</button>
        </div>
        <div class="composer-extras">
          <label class="research-toggle" title="Research: estimate tokens per second for the active reply (chars÷4 / elapsed). Default off.">
            <input type="checkbox" id="chat-show-tps" ${chatShowTokensPerSec ? "checked" : ""} />
            Show tokens/sec <span class="muted">(research)</span>
          </label>
        </div>
        <p class="hint">Auto | Debate | Single chooses speakers. Implement/install requires Approve before writes or host installs; Q&A and debate text stay unblocked. Drag a folder here to grant it (path must exist on this computer). Repo writes go to Cursor local or Approve apply-patch inside the allowlist.</p>
        <div id="grant-card" class="grant-card hidden">
          <p class="muted">Grant this folder for writes. It must already exist on this computer (the one running this GUI).</p>
          <label class="field">Folder path
            <input id="grant-path" type="text" autocomplete="off" spellcheck="false" placeholder="/home/you/project" />
          </label>
          <div class="actions">
            <button type="button" class="btn" data-grant-folder>Add to allowlist</button>
            <button type="button" class="btn secondary" data-grant-cancel>Cancel</button>
          </div>
          <div id="grant-status"></div>
        </div>
      </form>
    </div>
  `;
  const input = $("composer-input");
  input?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("composer-form")?.requestSubmit();
    }
  });
  $("route-pin")?.addEventListener("change", () => {
    if ($("route-pin")?.value) {
      for (const btn of document.querySelectorAll(".mode-btn")) {
        btn.setAttribute("aria-pressed", btn.getAttribute("data-mode") === "single" ? "true" : "false");
      }
    }
  });
  renderChatHeader();
  renderMessages();
  renderThreadList();
  bindChatDrop();
  const tpsToggle = $("chat-show-tps");
  if (tpsToggle) {
    tpsToggle.checked = chatShowTokensPerSec;
    tpsToggle.addEventListener("change", () => {
      chatShowTokensPerSec = Boolean(tpsToggle.checked);
      try {
        localStorage.setItem("orchestrator.chat.showTps", chatShowTokensPerSec ? "1" : "0");
      } catch {
        /* ignore */
      }
      renderMessages();
    });
  }
}

function droppedFilePath(file) {
  if (file && typeof file.path === "string" && file.path.trim()) return file.path.trim();
  return "";
}

function looksLikeAbsPath(text) {
  const t = String(text ?? "").trim();
  if (!t || t.includes("\n") || t.length > 1024) return false;
  if (t.startsWith("/") && !t.startsWith("//")) return true;
  return /^[A-Za-z]:[\\/]/.test(t);
}

function bindChatDrop() {
  const root = $("chat-root");
  const card = $("grant-card");
  const pathInput = $("grant-path");
  if (!root || root.dataset.dropBound === "1") return;
  root.dataset.dropBound = "1";
  const showGrant = (path) => {
    if (!card) return;
    card.classList.remove("hidden");
    if (pathInput) pathInput.value = path || pathInput.value || "";
    pathInput?.focus();
  };
  root.addEventListener("dragover", (event) => {
    if (!event.dataTransfer?.types?.includes("Files")) return;
    event.preventDefault();
    root.classList.add("drop-over");
  });
  root.addEventListener("dragleave", (event) => {
    if (event.target === root) root.classList.remove("drop-over");
  });
  root.addEventListener("drop", (event) => {
    const types = event.dataTransfer?.types;
    if (!types?.includes("Files") && !event.dataTransfer?.files?.length) return;
    event.preventDefault();
    root.classList.remove("drop-over");
    const file = event.dataTransfer.files?.[0];
    showGrant(droppedFilePath(file));
  });
  root.addEventListener("paste", (event) => {
    if (event.target === pathInput) return;
    const files = event.clipboardData?.files;
    if (files?.length) {
      event.preventDefault();
      showGrant(droppedFilePath(files[0]));
      return;
    }
    const text = event.clipboardData?.getData("text/plain") ?? "";
    if (!looksLikeAbsPath(text)) return;
    event.preventDefault();
    showGrant(text.trim());
  });
}

function renderOverview() {
  $("main").classList.remove("chat-main");
  const policy = catalog.writePolicy ?? {};
  const runtime = catalog.localRuntime ?? {};
  const ready = (catalog.backends ?? []).filter((b) => b.ready).length;
  const total = (catalog.backends ?? []).length;
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Overview</h1>
        <p>Backend health and the local write sandbox. Chat on the home screen routes this automatically.</p>
      </div>
    </div>
    <div class="cards">
      <article class="card">
        <h2>Backends</h2>
        <p>${ready} / ${total} ready</p>
        <p class="muted">Keys stay in <span class="mono">.env</span> or the GUI secrets store (gitignored). This UI never shows full secrets. Use Reload env after editing <span class="mono">.env</span>.</p>
      </article>
      <article class="card">
        <h2>Your computer</h2>
        <p>${
          runtime.lateinfer?.running
            ? `<span class="pill ok">late-infer running</span> · ${escapeHtml(runtime.lateinfer.model ?? "")} · 127.0.0.1:8010`
            : "late-infer idle"
        }</p>
        <p class="muted">Late infer on this computer (loopback 127.0.0.1:8010). Cloud agents cannot reach it. Auto prefers it for drafts when it is running. Download a Hub Instruct snapshot, compile, then Start on your computer.</p>
      </article>
      <article class="card">
        <h2>Default cwd</h2>
        <p class="mono">${escapeHtml(policy.defaultCwd ?? "")}</p>
        <p class="muted">Used when chat omits cwd. Must sit inside the allowlist.</p>
      </article>
      <article class="card">
        <h2>Allowed directories</h2>
        <p>${(policy.allowedDirectories ?? []).length} granted</p>
        <p class="muted">Local Cursor agents cannot write elsewhere.</p>
      </article>
      <article class="card">
        <h2>Theme</h2>
        <p class="muted">Look for this browser. Stored locally, not in git.</p>
        ${themePickerMarkup("theme-select-overview")}
      </article>
      <article class="card">
        <h2>Updates</h2>
        <p class="muted">Ask GitHub if this app or Late has a newer file. Cloud AI is not required. <a href="#updates">Open Updates</a></p>
      </article>
    </div>
    <div class="card" style="margin-top:0.85rem">
      <h2>Write allowlist</h2>
      <ul>${(policy.allowedDirectories ?? []).map((d) => `<li class="mono">${escapeHtml(d)}</li>`).join("") || "<li class='muted'>None</li>"}</ul>
    </div>
  `;
}

function renderSpecialists() {
  $("main").classList.remove("chat-main");
  const rows = (catalog.specialists ?? [])
    .map(
      (s) => `
      <tr>
        <td class="mono">${escapeHtml(s.id)}</td>
        <td>${escapeHtml(s.description)}</td>
        <td class="mono">${escapeHtml(s.backend)} ${pill(s.backendReady)}</td>
        <td class="mono">${s.fallback ? `${escapeHtml(s.fallback)} ${pill(s.fallbackReady)}` : "—"}</td>
        <td>${writePill(s.writesLocalFiles)}</td>
      </tr>`,
    )
    .join("");
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Specialists</h1>
        <p>Named agents from <span class="mono">agents.config.yaml</span>. Auto chat picks these; you do not need this page for the default path.</p>
      </div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Id</th><th>Role</th><th>Backend</th><th>Fallback</th><th>Writes</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}


function isLocalBackendType(type) {
  return type === "lateinfer" || type === "vllm" || type === "ollama" || type === "llamacpp";
}

function backendGroup(b) {
  if (isLocalBackendType(b?.type)) return "local";
  return "cloud";
}

function sortBackendsForGroup(list, group) {
  const localOrder = { lateinfer: 0, vllm: 1, ollama: 2, llamacpp: 3 };
  const cloudOrder = (b) => {
    const id = String(b.id ?? "").toLowerCase();
    if (b.type === "cursor" && /local/i.test(id)) return 0;
    if (b.type === "cursor") return 1;
    if (id === "openai" || (b.type === "openai" && /api\.openai\.com/i.test(b.baseUrl ?? ""))) return 2;
    if (b.type === "anthropic") return 3;
    if (/gemini/i.test(id) || /generativelanguage\.googleapis/i.test(b.baseUrl ?? "")) return 4;
    if (/openrouter/i.test(id) || /openrouter\.ai/i.test(b.baseUrl ?? "")) return 5;
    if (/grok|xai/i.test(id) || /api\.x\.ai/i.test(b.baseUrl ?? "")) return 6;
    return 50;
  };
  return [...list].sort((a, b) => {
    if (group === "local") {
      const da = localOrder[a.type] ?? 9;
      const db = localOrder[b.type] ?? 9;
      if (da !== db) return da - db;
    } else {
      const da = cloudOrder(a);
      const db = cloudOrder(b);
      if (da !== db) return da - db;
    }
    return String(a.id).localeCompare(String(b.id));
  });
}

function renderBackendCard(b, secretSet) {
  const names = b.secretNames ?? [];
  const primary = names[0];
  const keySet = names.some((n) => secretSet.get(n));
  const keyForm = primary && b.needsKey
    ? `<form class="secret-form" data-name="${escapeHtml(primary)}">
        <label class="field">Set ${escapeHtml(primary)} (stored locally, never shown in full)
          <input name="value" type="password" autocomplete="off" placeholder="${keySet ? "set — paste to replace" : "paste key"}" />
        </label>
        <div class="actions"><button type="submit">Save key</button>${keySet ? ` <button type="button" class="btn secondary" data-clear-secret="${escapeHtml(primary)}">Clear</button>` : ""}</div>
        <div class="secret-status"></div>
      </form>`
    : "";
  const geminiModels = b.modelChoices ?? [];
  const isGemini =
    b.id === "gemini" ||
    /generativelanguage\.googleapis\.com/i.test(b.baseUrl ?? "") ||
    geminiModels.length > 0;
  const localCompat = b.type === "lateinfer";
  const localModelsList = b.modelChoices ?? [];
  const modelForm = isGemini
    ? `<form class="backend-model-form" data-backend="${escapeHtml(b.id)}">
        <label class="field">Model (one id, not a list)
          <input name="model" list="gemini-models-${escapeHtml(b.id)}" value="${escapeHtml(b.model ?? "gemini-3.6-flash")}" required placeholder="gemini-3.6-flash" autocomplete="off" />
        </label>
        <datalist id="gemini-models-${escapeHtml(b.id)}">${geminiModels.map((m) => `<option value="${escapeHtml(m)}"></option>`).join("")}</datalist>
        <p class="muted">Google OpenAI-compat expects a bare id such as gemini-3.6-flash (not models/…, not 1.5/2.0). Datalist is live ListModels when the key works, else the 2026 catalog.</p>
        <div class="actions"><button type="submit">Save model</button></div>
      </form>`
    : localCompat
      ? `<form class="backend-model-form" data-backend="${escapeHtml(b.id)}">
        <label class="field">Model
          <input name="model" list="local-models-${escapeHtml(b.id)}" value="${escapeHtml(b.model ?? "")}" required placeholder="Qwen/Qwen2.5-0.5B-Instruct" autocomplete="off" />
        </label>
        <datalist id="local-models-${escapeHtml(b.id)}">${localModelsList.map((m) => `<option value="${escapeHtml(m)}"></option>`).join("")}</datalist>
        <p class="muted">Hugging Face Instruct id served by late-infer on 127.0.0.1:8010.</p>
        <div class="actions"><button type="submit">Save model</button></div>
      </form>`
      : "";
  const identityForm = `<form class="backend-nick-form" data-backend="${escapeHtml(b.id)}">
        <label class="field">Nickname
          <input name="nickname" type="text" maxlength="48" value="${escapeHtml(b.nickname ?? "")}" placeholder="Display name in chat" autocomplete="off" />
        </label>
        <div class="actions"><button type="submit">Save nickname</button></div>
      </form>
      <div class="backend-logo">
        <label class="field">Logo (PNG, JPEG, or WebP · 512 KiB max)
          <input class="backend-logo-input" type="file" accept="image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp" data-backend="${escapeHtml(b.id)}" />
        </label>
        ${b.hasLogo ? `<button type="button" class="btn secondary" data-logo-remove="${escapeHtml(b.id)}">Remove logo</button>` : ""}
      </div>`;
  return `
  <article class="card">
    <div class="row backend-head">
      ${avatarMarkup(b.id, b.hasLogo)}
      <div class="backend-title">
        <h3 class="${b.nickname ? "" : "mono"}">${escapeHtml(b.nickname || b.id)}</h3>
        ${b.nickname ? `<p class="muted mono" style="margin:0">${escapeHtml(b.id)}</p>` : ""}
      </div>
      ${pill(b.ready, b.reason)}
      ${writePill(b.writesLocalFiles)}
    </div>
    <p class="muted">${escapeHtml(b.type)}${b.runtime ? ` · ${escapeHtml(b.runtime)}` : ""}${b.model ? ` · ${escapeHtml(b.model)}` : ""}</p>
    ${b.baseUrl ? `<p class="mono muted">${escapeHtml(b.baseUrl)}</p>` : ""}
    <p>${escapeHtml(b.reason ?? (b.ready ? "API key present (masked; never displayed)." : ""))}</p>
    <p class="muted">capabilities: ${escapeHtml((b.capabilities ?? []).join(", "))}</p>
    ${modelForm}
    ${identityForm}
    ${keyForm}
  </article>`;
}

function renderGrokKeyCard(secretSet) {
  const xaiSet = Boolean(secretSet.get("XAI_API_KEY") || secretSet.get("GROK_API_KEY"));
  return `
  <article class="card">
    <div class="row backend-head">
      <div class="backend-title">
        <h3>Grok · xAI</h3>
        <p class="muted mono" style="margin:0">XAI_API_KEY</p>
      </div>
      ${xaiSet ? `<span class="pill ok">key set</span>` : `<span class="pill warn">missing</span>`}
      ${writePill(false)}
    </div>
    <p class="muted">Cloud · paste a Grok / xAI key for OpenAI-compat calls to api.x.ai. Stored in <span class="mono">.orchestrator/secrets.env</span> like other keys. Add a backend in Config if you want it in Auto (yaml is separate).</p>
    <form class="secret-form" data-name="XAI_API_KEY">
      <label class="field">XAI_API_KEY (stored locally, never shown in full)
        <input name="value" type="password" autocomplete="off" placeholder="${xaiSet ? "set — paste to replace" : "xai-…"}" />
      </label>
      <div class="actions">
        <button type="submit">Save Grok key</button>
        ${xaiSet ? `<button type="button" class="btn secondary" data-clear-secret="XAI_API_KEY">Clear</button>` : ""}
      </div>
      <div class="secret-status"></div>
    </form>
    <p class="muted"><span class="mono">GROK_API_KEY</span> in <span class="mono">.env</span> is also accepted after Reload env.</p>
  </article>`;
}

function dockerAvailableForVllm(data) {
  const intel = data?.intelDocker ?? data?.vllm?.intelDocker ?? data?.hardware?.intelDocker;
  if (!intel) return false;
  if (intel.available || intel.preferred) return true;
  // daemon ok means Docker can run containers (CUDA/ROCm host images too)
  return intel.daemon === "ok";
}

function vllmPhasePill(phase, healthy) {
  if (healthy || phase === "running") return `<span class="pill ok">running</span>`;
  if (phase === "starting") return `<span class="pill warn">starting</span>`;
  if (phase === "error") return `<span class="pill bad">error</span>`;
  return `<span class="pill warn">idle</span>`;
}

function renderVllmSectionHtml(data) {
  const vllm = data.vllm ?? {};
  const models = data.models ?? [];
  const recommended = data.recommended ?? [];
  const jobs = data.jobs ?? [];
  const showDocker = dockerAvailableForVllm(data);
  const instances = vllm.instances ?? [];
  const preferDocker = data.preferredRuntime === "docker";
  const byId = new Map(models.map((m) => [m.id, m]));
  for (const m of recommended) if (!byId.has(m.id)) byId.set(m.id, m);
  const list = [...byId.values()].sort((a, b) => {
    if (a.running !== b.running) return a.running ? -1 : 1;
    if (a.downloaded !== b.downloaded) return a.downloaded ? -1 : 1;
    if (a.fits !== b.fits) return a.fits ? -1 : 1;
    return String(a.name || a.id).localeCompare(String(b.name || b.id));
  });
  const instanceRows = instances.length
    ? instances
        .map((row) => {
          const mid = row.modelId || row.servedModelName || row.backendId || "?";
          return `<div class="local-engine-row">
            <div>
              <strong class="mono">${escapeHtml(mid)}</strong>
              ${vllmPhasePill(row.phase, row.healthy)}
              <p class="muted mono" style="margin:0.15rem 0 0">${escapeHtml(row.host || "127.0.0.1")}:${escapeHtml(String(row.port ?? "?"))} · ${escapeHtml(row.runtime || "host")} · ${escapeHtml(row.backendId || "")}</p>
              ${row.lastError || row.phase === "error" ? `<p class="error">${escapeHtml(vllmErrorDetail(row))}</p>` : ""}
              ${row.phase === "starting" ? `<p class="muted">${escapeHtml(vllmStartingDetail({ ...vllm, ...row }))}</p>` : ""}
            </div>
            <div class="actions">
              <button type="button" class="btn secondary" data-vllm-stop="${escapeHtml(mid)}" data-vllm-backend="${escapeHtml(row.backendId || "")}">Stop</button>
              <button type="button" class="btn secondary" data-vllm-remove="${escapeHtml(mid)}" data-vllm-backend="${escapeHtml(row.backendId || "")}">Remove from mix</button>
            </div>
          </div>`;
        })
        .join("")
    : `<p class="muted">No vLLM instance running. Download weights, then Start (host) or Start with Docker when Docker is on this computer.</p>`;
  const catalogRows = list
    .slice(0, 24)
    .map((m) => {
      const job = jobs.find((j) => j.modelId === m.id || j.hfRepo === m.hfRepo);
      const busy = job && (job.status === "running" || job.status === "starting");
      const actions = [];
      if (!m.downloaded) {
        actions.push(`<button type="button" class="btn secondary" data-local-download="${escapeHtml(m.id)}"${busy ? " disabled" : ""}>${busy ? "Downloading…" : "Download"}</button>`);
      } else if (m.running) {
        actions.push(`<button type="button" class="btn secondary" data-vllm-stop="${escapeHtml(m.id)}" data-vllm-backend="${escapeHtml(m.backendId || "")}">Stop</button>`);
        actions.push(`<button type="button" class="btn secondary" data-vllm-remove="${escapeHtml(m.id)}" data-vllm-backend="${escapeHtml(m.backendId || "")}">Remove from mix</button>`);
      } else {
        actions.push(`<button type="button" class="btn" data-vllm-start="${escapeHtml(m.id)}" data-vllm-runtime="host">Start</button>`);
        if (showDocker) {
          actions.push(`<button type="button" class="btn secondary" data-vllm-start="${escapeHtml(m.id)}" data-vllm-runtime="docker">Start with Docker</button>`);
        }
        actions.push(`<button type="button" class="btn danger" data-vllm-delete-weights="${escapeHtml(m.id)}">Delete weights</button>`);
      }
      return `<div class="local-engine-row">
        <div>
          <strong>${escapeHtml(m.name || m.id)}</strong>
          ${m.running ? `<span class="pill ok">serving</span>` : m.downloaded ? `<span class="pill ok">on disk</span>` : `<span class="pill warn">not downloaded</span>`}
          ${m.fits ? "" : `<span class="pill warn">tight fit</span>`}
          <p class="muted mono" style="margin:0.15rem 0 0">${escapeHtml(m.hfRepo || m.id)} · ~${escapeHtml(String(m.weightsMiB ?? "?"))} MiB${m.gated ? " · gated" : ""}</p>
          ${m.fitReason ? `<p class="muted">${escapeHtml(m.fitReason)}</p>` : ""}
        </div>
        <div class="actions">${actions.join(" ")}</div>
      </div>`;
    })
    .join("");
  const phase = vllm.phase || (anyVllmStarting(vllm) ? "starting" : vllm.healthy ? "running" : "idle");
  return `
    <article class="card local-engine-card">
      <h2>vLLM</h2>
      <p>${vllmPhasePill(phase, vllm.healthy)} <span class="muted">${escapeHtml(vllm.installHint || (vllm.installed ? "vLLM tooling detected" : "Install host vLLM or use Docker when available."))}</span></p>
      <p class="muted">OpenAI-compat on 127.0.0.1 (ports 8000–8099). Start returns immediately; wait until healthy. ${showDocker ? (preferDocker ? "Docker preferred on this computer." : "Docker is available — Start with Docker shows per model.") : "Docker not detected — host Start only (Start with Docker hidden)."} Weights are not in the archive.</p>
      <h3>Running instances</h3>
      <div class="local-engine-list">${instanceRows}</div>
      <h3>Catalog</h3>
      <div class="local-engine-list">${catalogRows || `<p class="muted">No catalog models.</p>`}</div>
    </article>`;
}

function renderOllamaSectionHtml() {
  const ollama = localServers.ollama ?? {};
  const running = Boolean(ollama.ready || ollama.running);
  const models = ollama.models ?? [];
  const bin = localServers.ollamaBinary;
  return `
    <article class="card local-engine-card">
      <h2>Ollama</h2>
      <p>${running ? `<span class="pill ok">running</span>` : `<span class="pill warn">idle</span>`} <span class="mono muted">${escapeHtml(ollama.origin || ollama.baseUrl || "http://127.0.0.1:11434")}</span></p>
      <p class="muted">${escapeHtml(ollama.reason || (running ? "Ollama reachable" : "Ollama not running on loopback."))}${
        bin ? ` Found <span class="mono">${escapeHtml(bin)}</span>.` : " Binary not found on PATH / runtime/bin."
      }</p>
      <p class="muted">Start serve on 127.0.0.1:11434 only. Pull weights yourself (<span class="mono">ollama pull llama3.1</span>) — this GUI does not pull. Then Register Ollama backend.</p>
      <div class="actions">
        <button type="button" class="btn" id="ollama-start"${running ? " disabled" : ""}>Start Ollama</button>
        <button type="button" class="btn secondary" id="ollama-stop"${running ? "" : " disabled"}>Stop</button>
        <button type="button" class="btn secondary" id="ollama-register">Register Ollama backend</button>
      </div>
      <p class="muted" style="margin-top:0.55rem">Tags: ${models.length ? models.map((m) => `<span class="mono">${escapeHtml(m)}</span>`).join(", ") : "(none yet)"}</p>
    </article>`;
}

function llamacppPrimaryStatus() {
  const rows = Array.isArray(localServers.llamacpp) ? localServers.llamacpp : [];
  if (rows.length) return rows[0];
  return { running: false, ready: false, baseUrl: "http://127.0.0.1:8080/v1", origin: "http://127.0.0.1:8080", models: [], reason: "llama.cpp not configured — Start with a GGUF path." };
}

function llamacppPhasePill(running) {
  return running ? `<span class="pill ok">running</span>` : `<span class="pill warn">idle</span>`;
}

function llamaGgufJobFor(repo) {
  return (llamaGgufJobs ?? []).find((j) => j.repo === repo || j.id === repo);
}

function renderLlamaGgufCatalogRowsHtml() {
  const list = llamaGgufCatalog.models ?? [];
  if (llamaGgufCatalog.loading && !list.length) {
    return `<p class="muted">Loading GGUF Hub store…</p>`;
  }
  if (!list.length) {
    return `<p class="muted">No GGUF Hub matches. Try search <span class="mono">unsloth</span> or paste a Hub org/model id. Budget ~${escapeHtml(String(llamaGgufCatalog.budgetGiB ?? "?"))} GB per idle GPU.</p>`;
  }
  return list
    .slice(0, 40)
    .map((m) => {
      const job = llamaGgufJobFor(m.id);
      const busy = job && (job.status === "running" || job.status === "queued");
      const path = m.localPath || (job?.status === "done" ? job.localPath : "") || "";
      const onDisk = Boolean(m.downloaded || path);
      const selected = m.id === llamaGgufSelectedId;
      const file = m.preferredFile?.filename || "";
      const size = m.preferredFile?.sizeLabel || m.sizeLabel || "size unknown";
      const actions = [];
      if (!onDisk) {
        actions.push(
          `<button type="button" class="btn secondary" data-llamacpp-gguf-download="${escapeHtml(m.id)}"${
            file ? ` data-llamacpp-gguf-file="${escapeHtml(file)}"` : ""
          }${busy ? " disabled" : ""}>${busy ? "Downloading…" : "Download"}</button>`,
        );
      } else {
        actions.push(
          `<button type="button" class="btn" data-llamacpp-gguf-start="${escapeHtml(m.id)}" data-llamacpp-gguf-path="${escapeHtml(path)}">Start</button>`,
        );
        actions.push(
          `<button type="button" class="btn secondary" data-llamacpp-gguf-use="${escapeHtml(path)}" data-llamacpp-gguf-id="${escapeHtml(m.id)}">Use path</button>`,
        );
      }
      return `<div class="local-engine-row${selected ? " is-selected" : ""}" data-llamacpp-gguf-row="${escapeHtml(m.id)}">
        <div>
          <strong>${escapeHtml(m.id)}</strong>
          ${onDisk ? `<span class="pill ok">on disk</span>` : `<span class="pill warn">not downloaded</span>`}
          ${m.fits ? "" : `<span class="pill warn">likely too big</span>`}
          ${m.gated ? `<span class="pill warn">gated</span>` : ""}
          <p class="muted mono" style="margin:0.15rem 0 0">${escapeHtml(m.family || "")}${
            file ? ` · ${escapeHtml(file)}` : " · GGUF"
          } · ${escapeHtml(size)}${path ? ` · ${escapeHtml(path)}` : ""}</p>
          ${m.likelyTooBig ? `<p class="muted">Preferred quant may exceed ~${escapeHtml(String(llamaGgufCatalog.budgetGiB ?? "?"))} GB idle GPU VRAM.</p>` : ""}
        </div>
        <div class="actions">${actions.join(" ")}</div>
      </div>`;
    })
    .join("");
}

async function loadLlamaGgufCatalog(query) {
  llamaGgufCatalog.loading = true;
  const q = String(query ?? llamaGgufSearch ?? "").trim();
  try {
    const data = await api(`/api/local-servers/hub-gguf-models?q=${encodeURIComponent(q)}`);
    const jobsRes = await api("/api/local-servers/gguf-download").catch(() => ({ jobs: [] }));
    llamaGgufJobs = Array.isArray(jobsRes?.jobs) ? jobsRes.jobs : [];
    llamaGgufCatalog = {
      models: Array.isArray(data.models) ? data.models : [],
      hint: typeof data.hint === "string" ? data.hint : "",
      budgetGiB: data.budgetGiB ?? null,
      offline: data.offline === true,
      loaded: true,
      loading: false,
      error: typeof data.error === "string" ? data.error : "",
      ggufDir: typeof data.ggufDir === "string" ? data.ggufDir : "",
      defaultId: typeof data.defaultId === "string" ? data.defaultId : "",
    };
    if (!llamaGgufSelectedId && llamaGgufCatalog.defaultId) llamaGgufSelectedId = llamaGgufCatalog.defaultId;
  } catch (error) {
    llamaGgufCatalog = {
      ...llamaGgufCatalog,
      models: [],
      loaded: true,
      loading: false,
      offline: true,
      error: error instanceof Error ? error.message : String(error),
    };
  }
  if (pageId() === "local-models") {
    const slot = $("llamacpp-gguf-catalog");
    if (slot) slot.innerHTML = renderLlamaGgufCatalogRowsHtml();
    const hint = $("llamacpp-gguf-hint");
    if (hint) hint.textContent = llamaGgufCatalog.hint || "";
  }
}

function renderLlamaCppSectionHtml() {
  const status = llamacppPrimaryStatus();
  const running = Boolean(status.ready || status.running);
  const bin = localServers.llamaServerBinary;
  const models = status.models ?? [];
  const origin = status.origin || status.baseUrl || "http://127.0.0.1:8080";
  const registered = (catalog.backends ?? []).filter((b) => b.type === "llamacpp");
  const pathValue = llamaGgufPath || "";
  const instanceRows = running
    ? `<div class="local-engine-row">
        <div>
          <strong class="mono">llama-server</strong>
          ${llamacppPhasePill(true)}
          <p class="muted mono" style="margin:0.15rem 0 0">${escapeHtml(origin)}${
            models.length ? ` · ${models.map((m) => escapeHtml(m)).join(", ")}` : ""
          }</p>
          ${status.reason ? `<p class="muted">${escapeHtml(status.reason)}</p>` : ""}
        </div>
        <div class="actions">
          <button type="button" class="btn secondary" id="llamacpp-stop">Stop</button>
          <button type="button" class="btn secondary" id="llamacpp-register">Register backend</button>
        </div>
      </div>`
    : `<p class="muted">No llama-server instance running. Download a GGUF below or set a path, then Start. Register adds the OpenAI-compat backend to the mix when healthy.</p>`;
  const registeredRows = registered.length
    ? registered
        .map((b) => {
          const mid = b.model || b.id || "llamacpp";
          return `<div class="local-engine-row">
            <div>
              <strong class="mono">${escapeHtml(b.id || "llamacpp")}</strong>
              <span class="pill ok">in mix</span>
              <p class="muted mono" style="margin:0.15rem 0 0">${escapeHtml(b.baseUrl || origin)} · ${escapeHtml(mid)}</p>
            </div>
          </div>`;
        })
        .join("")
    : `<p class="muted">No llama.cpp backend in agents.config.yaml yet. Start llama-server, then Register backend.</p>`;
  return `
    <article class="card local-engine-card">
      <h2>llama.cpp</h2>
      <p>${llamacppPhasePill(running)} <span class="muted">${
        bin
          ? `llama-server detected · <span class="mono">${escapeHtml(bin)}</span>`
          : "Install llama-server on PATH / runtime/bin, or use the binary shipped under runtime/bin."
      }</span></p>
      <p class="muted">OpenAI-compat on 127.0.0.1:8080. Hub store lists <strong>GGUF</strong> repos (Unsloth welcome) sized for the idle GPU on this computer (~32 GB Arc). Download into <span class="mono">.orchestrator/models/gguf</span>, then Start / Register — same pattern as vLLM. Not the late-infer safetensors store.</p>
      <h3>Running instance</h3>
      <div class="local-engine-list">${instanceRows}</div>
      <h3>GGUF Hub store</h3>
      <label class="field" for="llamacpp-gguf-search">Search GGUF models
        <input id="llamacpp-gguf-search" type="search" value="${escapeHtml(llamaGgufSearch)}" placeholder="unsloth, qwen, gemma…" autocomplete="off" />
      </label>
      <p class="muted" id="llamacpp-gguf-hint">${escapeHtml(llamaGgufCatalog.hint || "Lists filter=gguf from Hugging Face Hub for llama.cpp on your GPUs.")}</p>
      <div id="llamacpp-gguf-catalog" class="local-engine-list">${renderLlamaGgufCatalogRowsHtml()}</div>
      <h3>Start</h3>
      <label class="field" for="llamacpp-gguf">GGUF path
        <input id="llamacpp-gguf" type="text" class="mono" value="${escapeHtml(pathValue)}" placeholder="/home/…/model.gguf" autocomplete="off" spellcheck="false" />
      </label>
      <label class="field" for="llamacpp-hub-id">Hub id (optional paste)
        <input id="llamacpp-hub-id" type="text" class="mono" value="${escapeHtml(llamaGgufSelectedId)}" placeholder="unsloth/Qwen3-8B-Instruct-GGUF" autocomplete="off" spellcheck="false" />
      </label>
      <label class="field" for="llamacpp-model">Model id (optional, for Register)
        <input id="llamacpp-model" type="text" class="mono" placeholder="local or tag from /v1/models" autocomplete="off" spellcheck="false" />
      </label>
      <div class="actions">
        <button type="button" class="btn secondary" id="llamacpp-hub-download">Download Hub id</button>
        <button type="button" class="btn" id="llamacpp-start"${running ? " disabled" : ""}>Start</button>
        <button type="button" class="btn secondary" id="llamacpp-stop"${running ? "" : " disabled"}>Stop</button>
        <button type="button" class="btn secondary" id="llamacpp-register"${running ? "" : " disabled"}>Register backend</button>
      </div>
      <h3>Registered backends</h3>
      <div class="local-engine-list">${registeredRows}</div>
    </article>`;
}

async function renderBackends() {
  $("main").classList.remove("chat-main");
  let secrets = [];
  try {
    const data = await api("/api/secrets");
    secrets = data.secrets ?? [];
  } catch {
    secrets = [];
  }
  const secretSet = new Map(secrets.map((s) => [s.name, s.set]));
  const all = catalog.backends ?? [];
  const localList = sortBackendsForGroup(all.filter((b) => backendGroup(b) === "local"), "local");
  const cloudList = sortBackendsForGroup(all.filter((b) => backendGroup(b) === "cloud"), "cloud");
  const localCards = localList.map((b) => renderBackendCard(b, secretSet)).join("");
  const cloudCards = cloudList.map((b) => renderBackendCard(b, secretSet)).join("") + renderGrokKeyCard(secretSet);
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Backends</h1>
        <p>Ready vs missing keys. Local engines stay on this computer; cloud keys stay in <span class="mono">.orchestrator/secrets.env</span>. Nicknames and logos appear in chat. Logos stay in <span class="mono">.orchestrator/logos</span> (not git). Secrets are never shown in full. Adding a key here or in <span class="mono">.env</span> takes effect after Reload env.</p>
      </div>
      <button type="button" id="reload-env" class="btn secondary">Reload env</button>
    </div>
    <div id="backends-status"></div>
    <section class="backend-section">
      <h2 class="backend-section-title">Local</h2>
      <p class="muted backend-section-lede">late-infer · vLLM · Ollama · llama.cpp — loopback only. Start / stop weights on Local models.</p>
      <div class="cards">${localCards || `<p class="muted">No local backends in config yet.</p>`}</div>
    </section>
    <section class="backend-section">
      <h2 class="backend-section-title">Cloud</h2>
      <p class="muted backend-section-lede">Cursor · OpenAI · Anthropic · Gemini · OpenRouter · Grok</p>
      <div class="cards">${cloudCards}</div>
    </section>
    <div class="card" style="margin-top:0.85rem">
      <h2>MCP HTTP (any client)</h2>
      <p class="muted">Streamable HTTP on this GUI process. Path is <span class="mono">/mcp</span> (also <span class="mono">/MCP</span>). Late does not send the GUI token and does not need this server. Optional dedicated process: <span class="mono">npm run mcp:http</span> (<span class="mono">AGENT_ORCHESTRATOR_MCP_PORT</span>).</p>
      <label class="field">Late Settings URL (this process)
        <input id="mcp-url" class="mono" type="text" readonly value="${escapeHtml(mcpUrlForLate())}" />
      </label>
      <div class="actions">
        <button type="button" class="btn" data-copy-mcp>Copy MCP URL</button>
        <span id="mcp-copy-status" class="muted"></span>
      </div>
      <form id="mcp-listen-host-form">
        <label class="field">Listen host (this computer)
          <input id="mcp-listen-host" name="listenHost" class="mono" type="text" autocomplete="off" ${sessionInfo.envListenHostSet ? "disabled" : ""} value="${escapeHtml(sessionInfo.configListenHost)}" placeholder="${escapeHtml(sessionInfo.suggestedLanHost || "192.168.2.139")}" />
        </label>
        <p class="muted">Empty or <span class="mono">auto</span> binds this computer's LAN address (example <span class="mono">${escapeHtml(sessionInfo.suggestedLanHost || "192.168.2.139")}</span>). Or type that IP, or <span class="mono">127.0.0.1</span>. Env <span class="mono">AGENT_ORCHESTRATOR_MCP_HOST</span> / <span class="mono">AGENT_ORCHESTRATOR_GUI_HOST</span> override. Restart after save. One private IP only — not every interface. Trusted LAN only — firewall to the laptop. Late still Approve.</p>
        <div class="actions"><button type="submit"${sessionInfo.envListenHostSet ? " disabled" : ""}>Save listen host</button></div>
        <div id="mcp-listen-status"></div>
      </form>
      <p class="muted">Optional ClearPass / ISE (RADIUS) and Active Directory (LDAPS) are off until you set <span class="mono">AGENT_ORCHESTRATOR_MCP_AUTH</span> in <span class="mono">.env</span>. Secrets below never go in YAML.</p>
      <form class="secret-form" data-name="RADIUS_SECRET">
        <label class="field">RADIUS_SECRET (ClearPass / ISE shared secret)
          <input name="value" type="password" autocomplete="off" placeholder="${secretSet.get("RADIUS_SECRET") ? "set — paste to replace" : "optional"}" />
        </label>
        <div class="actions"><button type="submit">Save</button>${secretSet.get("RADIUS_SECRET") ? ` <button type="button" class="btn secondary" data-clear-secret="RADIUS_SECRET">Clear</button>` : ""}</div>
      </form>
      <form class="secret-form" data-name="LDAP_BIND_PASSWORD">
        <label class="field">LDAP_BIND_PASSWORD (optional AD service bind)
          <input name="value" type="password" autocomplete="off" placeholder="${secretSet.get("LDAP_BIND_PASSWORD") ? "set — paste to replace" : "optional"}" />
        </label>
        <div class="actions"><button type="submit">Save</button>${secretSet.get("LDAP_BIND_PASSWORD") ? ` <button type="button" class="btn secondary" data-clear-secret="LDAP_BIND_PASSWORD">Clear</button>` : ""}</div>
      </form>
    </div>
  `;
}

function runStatus(status) {
  const kind = status === "finished" ? "ok" : status === "error" ? "bad" : "warn";
  return `<span class="pill ${kind}">${escapeHtml(status)}</span>`;
}

function renderRuns() {
  $("main").classList.remove("chat-main");
  const rows = runs
    .map(
      (r) => `
      <tr>
        <td class="mono"><a href="#runs/${escapeHtml(r.id)}">${escapeHtml(r.id.slice(0, 8))}</a></td>
        <td>${escapeHtml(r.specialist)}</td>
        <td>${escapeHtml(r.backend)}</td>
        <td>${runStatus(r.status)}</td>
        <td class="muted">${escapeHtml(r.cwd ?? "—")}</td>
        <td class="muted">${escapeHtml(r.createdAt ?? "")}</td>
      </tr>`,
    )
    .join("");
  const selected = runs.find((r) => r.id === selectedRunId);
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Runs</h1>
        <p>Dispatch history for this GUI process. Chat threads live under Settings-adjacent chat history in the sidebar.</p>
      </div>
    </div>
    <div class="card">
      <table>
        <thead><tr><th>Id</th><th>Specialist</th><th>Backend</th><th>Status</th><th>cwd</th><th>Created</th></tr></thead>
        <tbody>${rows || `<tr><td colspan="6" class="muted">No runs yet.</td></tr>`}</tbody>
      </table>
    </div>
    ${selected ? renderRunDetail(selected) : ""}
  `;
}

function renderRunDetail(run) {
  return `
    <div class="card" style="margin-top:0.85rem">
      <div class="row">
        <h2 class="mono">${escapeHtml(run.id)}</h2>
        ${runStatus(run.status)}
      </div>
      <p class="muted">${escapeHtml(run.specialist)} · ${escapeHtml(run.backend)} · cwd ${escapeHtml(run.cwd ?? "n/a")}</p>
      ${run.error ? `<p class="error">${escapeHtml(run.error)}</p>` : ""}
      <pre class="out">${escapeHtml(run.text || run.prompt || "")}</pre>
      <form id="follow-form">
        <label class="field">Follow up
          <textarea name="message" required placeholder="Continue this run…"></textarea>
        </label>
        <div class="actions"><button type="submit">Send follow-up</button></div>
      </form>
      <div id="follow-status"></div>
    </div>
  `;
}

function renderLocalModels() {
  $("main").classList.remove("chat-main");
  const data = localModels ?? {};
  const hw = data.hardware ?? {};
  const cloud = data.cloudCursor ?? {};
  const accelerators = hw.accelerators ?? hw.gpus ?? [];
  const backend = hw.primaryBackend ?? (accelerators.length ? "gpu" : "cpu");
  const backendLabel =
    backend === "intel-xpu" ? "Intel XPU" : backend === "cuda" ? "CUDA" : backend === "rocm" ? "ROCm" : "CPU";
  const gpuLine = accelerators.length
    ? accelerators
        .map((g) => {
          const vendor = g.vendor ? `${g.vendor}` : "";
          const src = g.source ? ` · ${g.source}` : "";
          const est = g.vramEstimated ? " estimated" : "";
          return `${g.name} · ${g.vramMiB} MiB${est}${vendor ? ` · ${vendor}` : ""}${src}${g.driver ? ` · driver ${g.driver}` : ""}`;
        })
        .join("; ")
    : "No discrete accelerator (CPU only)";
  syncLateInferPoll(localServers.lateinfer);
  syncVllmPoll(data.vllm);
  const lateInferDlBusy = lateInferBusy(localServers.lateinfer);
  const lateInferCompiledListHtml = renderLateInferCompiledListHtml();
  const hfTokenSet = Boolean(data.hfTokenSet);
  lateInferHfTokenSet = hfTokenSet;
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Local models</h1>
        <p>Late infer, vLLM, Ollama, and llama.cpp on this computer. Download / compile / Start stay on loopback.</p>
      </div>
    </div>
    <div id="local-models-status"></div>
    <div class="card" style="margin-bottom:0.85rem">
      <h2>Hugging Face token</h2>
      <p class="muted">Gated models (Gemma 2, Llama, Mistral 7B, and others) need two steps: accept the license on the model card while logged into your Hugging Face account, then paste a <strong>read</strong> access token here. Stored in gitignored <span class="mono">.orchestrator/secrets.env</span> (POSIX mode 0600). This UI never shows the value again. Gemma 2 weights are under Google’s Gemma Terms of Use. Gemma 4 is Apache 2.0 and ungated on Hugging Face.</p>
      <p>${hfTokenSet ? `<span class="pill ok">configured</span> <span class="muted">Token is set (value never displayed). Paste a new one to rotate.</span>` : `<span class="pill warn">not configured</span> <span class="muted">Downloads of gated repos will fail with 401 until a token is saved.</span>`}</p>
      <p class="muted"><a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener noreferrer">Open Hugging Face token page</a> — create a token, copy it, paste below. This app does not use OAuth or a public callback.</p>
      <form id="hf-token-form" class="secret-form" data-name="HF_TOKEN">
        <label class="field">HF_TOKEN (stored locally, never shown in full)
          <input name="value" type="password" autocomplete="off" placeholder="${hfTokenSet ? "set — paste to replace" : "hf_…"}" />
        </label>
        <div class="actions">
          <button type="submit">Save HF token</button>
          ${hfTokenSet ? `<button type="button" class="btn secondary" data-clear-secret="HF_TOKEN">Clear token</button>` : ""}
        </div>
        <div class="secret-status"></div>
      </form>
    </div>
    <article class="card lateinfer-card">
        <h2>Late infer · your computer</h2>
        <p>${
          localServers.lateinfer?.ready || localServers.lateinfer?.running
            ? `<span class="pill ok">late-infer running</span> ${escapeHtml(localServers.lateinfer?.origin ?? "http://127.0.0.1:8010")}`
            : `<span class="pill warn">late-infer idle</span> http://127.0.0.1:8010`
        }</p>
        <p class="muted">Start this backend on your computer. It listens on <span class="mono">127.0.0.1:8010</span> only — nothing on your LAN can reach it. <strong>Download</strong> fetches a Hugging Face Instruct snapshot and compiles it here. Weights are not in npm or this zip. You do not need Late desktop. Many Instruct LLMs work; <strong>not every Hub repo</strong>.</p>
        <p class="muted">${
          localServers.lateinfer?.ready || localServers.lateinfer?.running
            ? escapeHtml(localServers.lateinfer?.reason ?? "late-infer running")
            : "late-infer idle. Download from the Hub store, then Start a compiled model on your computer."
        }${
          localServers.lateInferBinary
            ? ` Found <span class="mono">${escapeHtml(localServers.lateInferBinary)}</span>.`
            : " late-infer binary not found. On your computer, from this repo run <span class=\"mono\">npm run infer:build</span> so Start can find <span class=\"mono\">bin/late-infer</span>."
        }</p>
        ${renderLateInferGpuLiveHtml()}
        <div class="lateinfer-panes">
          <section class="lateinfer-pane lateinfer-store-pane" data-lateinfer-pane="store">
            <h3>Hub store</h3>
            <p class="muted">Safetensors CausalLM chat models for OpenVINO / late-infer — not the full Hugging Face website. Compatible / loadable rows sort first (newest among those). Broken config.json and OV-incompatible graphs are demoted. GGUF → llama.cpp. vLLM has its own weights pane. Use the gated checkbox to hide or show locked gated repos.</p>
            <div class="lateinfer-hub-picker">
              <label class="field lateinfer-hub-field" for="lateinfer-hub">Hub id (Hugging Face Instruct)
                <input id="lateinfer-hub" type="text" value="${escapeHtml(lateInferHubId)}" placeholder="${escapeHtml(lateInferHubDefaultId())}" autocomplete="off" spellcheck="false" />
              </label>
              <label class="field" for="lateinfer-hub-search">Search listed models
                <input id="lateinfer-hub-search" type="search" value="${escapeHtml(lateInferHubSearch)}" placeholder="Qwen, Mistral, Gemma…" autocomplete="off" role="combobox" aria-expanded="true" aria-controls="lateinfer-hub-list" aria-autocomplete="list" />
              </label>
              <div class="lateinfer-hub-filters" role="group" aria-label="Hub store filters">
                <label class="check-row"><input type="checkbox" id="lateinfer-on-computer-only"${lateInferOnComputerOnly ? " checked" : ""} /> On your computer only</label>
                <label class="check-row" title="When off, gated repos are hidden. When on, they stay listed as locked until you accept the license and set a Hugging Face read token."><input type="checkbox" id="lateinfer-show-gated"${lateInferShowGated ? " checked" : ""} /> Show gated (locked until license + HF token)</label>
              </div>
              <p class="muted" id="lateinfer-gated-help">Gated rows need two steps before Download: accept the model license on Hugging Face, then save a read token under Settings → Local models. Unsloth GGUF stays under llama.cpp — this list is safetensors CausalLM only.</p>
              <p class="error" id="lateinfer-hub-status"${lateInferHubCatalog.offline ? "" : " hidden"}>${lateInferHubCatalog.offline ? "Hugging Face Hub is offline. Listed ids below still Download on your computer." : ""}</p>
              <p class="muted" id="lateinfer-hub-list-label">Hugging Face store (loadable + ungated first)</p>
              <div id="lateinfer-hub-list" class="lateinfer-hub-list" role="listbox" aria-labelledby="lateinfer-hub-list-label">${renderLateInferHubListHtml(lateInferHubSearch)}</div>
              <p class="muted" id="lateinfer-hub-hint">${escapeHtml(lateInferHubHintText())}</p>
              <p class="muted" id="lateinfer-check-status">${escapeHtml(lateInferCheckStatusText(lateInferHubId))}</p>
            </div>
            ${renderLateInferCompileDetectHtml()}
            <div id="lateinfer-download-progress" class="lateinfer-download-progress">${lateInferDownloadProgressHtml(localServers.lateinfer)}</div>
            <div id="lateinfer-download-status">${lateInferDownloadStatusHtml(localServers.lateinfer)}</div>
            <div class="actions">
              <button type="button" class="btn secondary" id="lateinfer-check">Check</button>
              <button type="button" class="btn secondary" id="lateinfer-download" ${lateInferDlBusy || lateInferCompileBlockedByGpu() ? "disabled" : ""}>Download</button>
            </div>
          </section>
          <section class="lateinfer-pane lateinfer-local-pane${lateInferStartNotReadyForGpu() ? " is-not-gpu-ready" : ""}" id="lateinfer-local-pane" data-lateinfer-pane="compiled">
            <h3>On your computer</h3>
            <div id="lateinfer-start-banner-slot">${renderLateInferStartBannerHtml()}</div>
            <p class="muted">Downloaded Hub snapshots on your computer. A row marked ready can GPU Start on 127.0.0.1:8010. A row marked not ready for GPU Start is on your computer but OpenVINO IR is missing — Start will not use system RAM. Remove from your computer deletes that compiled snapshot (Stop first if it is serving on 127.0.0.1:8010). Download again from the Hub store to restore.</p>
            <div id="lateinfer-compiled-list" class="lateinfer-hub-list lateinfer-compiled-list" role="listbox" aria-label="Compiled models on your computer">${lateInferCompiledListHtml}</div>
            ${renderLateInferGpuControlsHtml()}
            <div id="lateinfer-start-actions">${renderLateInferLocalStartInnerHtml()}</div>
          </section>
        </div>
    </article>

    ${renderVllmSectionHtml(data)}
    ${renderOllamaSectionHtml()}
    ${renderLlamaCppSectionHtml()}
    <div class="cards">
      <article class="card">
        <h2>Hardware</h2>
        <p>${escapeHtml(gpuLine)}</p>
        <p class="muted">Backend ${escapeHtml(backendLabel)} · ${escapeHtml(String(hw.deviceCount ?? accelerators.length ?? 0))} device(s) · ${escapeHtml(String(hw.totalVramMiB ?? hw.vramMiB ?? "?"))} MiB total · RAM ${escapeHtml(String(hw.ramMiB ?? "?"))} MiB · ${escapeHtml(String(hw.cpuCount ?? "?"))} CPUs${hw.constrained ? " · CPU fallback (no accelerator)" : ""}</p>
        ${hardwareNotesForLocalModels(hw.notes).map((n) => `<p class="muted">${escapeHtml(n)}</p>`).join("")}
      </article>
      <article class="card">
        <h2>Cursor cloud</h2>
        <p>${cloud.ready ? "CURSOR_API_KEY present" : "CURSOR_API_KEY missing"}</p>
        <p class="muted">${escapeHtml(cloud.reason ?? "")}</p>
        <p class="muted">Auto debate can include cloud as a speaker; it never calls localhost late-infer.</p>
      </article>
    </div>
  `;
}

function renderDispatch() {
  $("main").classList.remove("chat-main");
  const specs = (catalog.specialists ?? [])
    .map((s) => `<option value="${escapeHtml(s.id)}">${escapeHtml(s.id)}</option>`)
    .join("");
  const backends = [`<option value="">(specialist default)</option>`]
    .concat((catalog.backends ?? []).map((b) => `<option value="${escapeHtml(b.id)}">${escapeHtml(b.id)}</option>`))
    .join("");
  const workflows = (catalog.workflows ?? [])
    .map((w) => `<option value="${escapeHtml(w.id)}">${escapeHtml(w.id)}</option>`)
    .join("");
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Run workflow</h1>
        <p>Advanced: pin a specialist or a named pipeline. The default path is Chat with Auto (round-table). <code>cloud-with-local-draft</code> in chat is a bounce debate, not this one-way handoff.</p>
      </div>
    </div>
    <div class="cards">
      <form id="dispatch-form" class="card">
        <h2>Specialist</h2>
        <label class="field">Specialist <select name="specialist">${specs}</select></label>
        <label class="field">Task <textarea name="task" required></textarea></label>
        <label class="field">Backend override <select name="backend">${backends}</select></label>
        <label class="field">cwd (optional) <input name="cwd" type="text" placeholder="${escapeHtml(catalog.writePolicy?.defaultCwd ?? "")}" /></label>
        <label class="field">Model (optional) <input name="model" type="text" /></label>
        <label class="field row"><input type="checkbox" name="wait" checked /> Wait for completion</label>
        <div class="actions"><button type="submit">Dispatch</button></div>
        <div id="dispatch-status"></div>
      </form>
      <form id="workflow-form" class="card">
        <h2>Workflow</h2>
        <label class="field">Workflow <select name="workflow">${workflows}</select></label>
        <label class="field">Task <textarea name="task" required></textarea></label>
        <label class="field">cwd (optional) <input name="cwd" type="text" /></label>
        <div class="actions"><button type="submit">Run workflow</button></div>
        <div id="workflow-status"></div>
      </form>
    </div>
  `;
}

function renderAllowlist() {
  $("main").classList.remove("chat-main");
  const dirs = catalog.writePolicy?.allowedDirectories ?? [];
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Write allowlist</h1>
        <p>Local Cursor agents may only use a cwd inside these directories. External backends stay text-only.</p>
      </div>
    </div>
    <div class="card">
      <p class="muted">Default cwd: <span class="mono">${escapeHtml(catalog.writePolicy?.defaultCwd ?? "")}</span></p>
      <table>
        <thead><tr><th>Directory</th><th></th></tr></thead>
        <tbody>
          ${dirs
            .map(
              (d) => `<tr>
                <td class="mono">${escapeHtml(d)}</td>
                <td><button class="btn danger" data-remove="${escapeHtml(d)}">Remove</button></td>
              </tr>`,
            )
            .join("")}
        </tbody>
      </table>
      <form id="allow-form">
        <label class="field">Add directory
          <input name="path" type="text" required placeholder="/absolute/path/to/repo" />
        </label>
        <div class="actions"><button type="submit">Grant directory</button></div>
      </form>
      <div id="allow-status"></div>
    </div>
  `;
}

function updateProductLine(client) {
  const local = client.localVersion || "unknown";
  const remote = client.remoteVersion ? `GitHub ${client.remoteVersion}` : "no GitHub release";
  if (client.newer) return `${local} on your computer → ${remote}`;
  return `${local} on your computer · ${remote}`;
}

function updateClientCard(id) {
  const client = id === "late" ? updateCheck?.late : updateCheck?.orchestrator;
  const title = id === "late" ? "Late" : "Orchestrator";
  if (!client) {
    return `<article class="card">
      <h2>${title}</h2>
      <p class="muted">Press Check for updates. I will ask GitHub for Late and Agent Orchestrator.</p>
    </article>`;
  }
  const pillHtml = client.newer
    ? `<span class="pill warn">newer on GitHub</span>`
    : client.error
      ? `<span class="pill bad">could not check</span>`
      : `<span class="pill ok">current</span>`;
  const asset = client.asset?.name
    ? `<p class="mono muted">${escapeHtml(client.asset.name)}</p>`
    : client.newer
      ? `<p class="muted">GitHub tag has no matching file for this computer. I will not build from the tag.</p>`
      : "";
  const err = client.error ? `<p class="muted">${escapeHtml(client.error)}</p>` : "";
  const link = client.releaseUrl
    ? `<p class="muted"><a href="${escapeHtml(client.releaseUrl)}" target="_blank" rel="noopener noreferrer">Open the GitHub page</a></p>`
    : "";
  return `<article class="card">
    <h2>${title} ${pillHtml}</h2>
    <p>${escapeHtml(updateProductLine(client))}</p>
    ${asset}
    ${err}
    ${link}
    <p class="muted">${client.unsigned ? "These files are not signed on Mac or Windows." : "Same files as Releases. I will not run them as root."}</p>
  </article>`;
}

function updateConfirmCopy(choice) {
  const names = choice === "both" ? "Late and Agent Orchestrator" : choice === "late" ? "Late" : "Agent Orchestrator";
  const note = updateCheck?.unsignedNote ? ` ${updateCheck.unsignedNote} This is on your computer.` : "";
  return `Download ${names} from GitHub onto your computer? Late uses the AppImage / .deb / .dmg / .exe already on the release. Orchestrator uses the portable archive.${note}`;
}

function renderUpdates() {
  $("main").classList.remove("chat-main");
  const checked = Boolean(updateCheck);
  const confirmOpen = Boolean(updatePick);
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>${confirmOpen ? "Download on your computer?" : "Updates"}</h1>
        <p>GitHub releases for Late and Agent Orchestrator. Nothing downloads until you confirm. This check does not need Cloud AI.</p>
      </div>
    </div>
    <div class="card" style="margin-bottom:0.85rem">
      <p class="muted">This talks to github.com only. Cloud AI stays off. Bind stays 127.0.0.1. Keys never leave this computer.</p>
      <div class="actions">
        <button type="button" class="btn" data-update-check${updateBusy ? " disabled" : ""}>${updateBusy ? "Checking…" : "Check for updates"}</button>
      </div>
      <div id="update-status">${updateFlash ? flash(updateFlash, updateFlashKind) : updateCheck ? flash(updateCheck.message, "ok") : ""}</div>
    </div>
    <div class="cards">
      ${updateClientCard("late")}
      ${updateClientCard("orchestrator")}
    </div>
    <div class="card" style="margin-top:0.85rem">
      <h2>What should I update?</h2>
      <p class="muted">${checked ? "Late and Orchestrator on your computer match GitHub unless a card says newer. You can still pick Late, Orchestrator, or both." : "Check first. Then you can pick Update Late, Update Orchestrator, or Update both."}</p>
      <div class="actions">
        <button type="button" class="btn" data-update-pick="late"${checked && !updateBusy ? "" : " disabled"}>Update Late</button>
        <button type="button" class="btn" data-update-pick="orchestrator"${checked && !updateBusy ? "" : " disabled"}>Update Orchestrator</button>
        <button type="button" class="btn" data-update-pick="both"${checked && !updateBusy ? "" : " disabled"}>Update both</button>
      </div>
    </div>
    <div id="update-confirm" class="grant-card${confirmOpen ? "" : " hidden"}">
      <h3>Download on your computer?</h3>
      <p class="muted">${confirmOpen ? escapeHtml(updateConfirmCopy(updatePick)) : ""}</p>
      <div class="actions">
        <button type="button" class="btn" data-update-confirm>Download</button>
        <button type="button" class="btn secondary" data-update-cancel>Back</button>
      </div>
    </div>
  `;
}

function renderConfig(state = {}) {
  $("main").classList.remove("chat-main");
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Config</h1>
        <p>Edit <span class="mono">agents.config.yaml</span>. Validated before save. Do not paste live API keys here; use the Backends page or env vars (<span class="mono">GEMINI_API_KEY</span> / <span class="mono">GOOGLE_API_KEY</span>).</p>
      </div>
    </div>
    ${state.message ? flash(state.message, state.kind ?? "ok") : ""}
    <form id="config-form" class="card">
      <label class="field">YAML
        <textarea class="yaml" name="yaml">${escapeHtml(state.yaml ?? "")}</textarea>
      </label>
      <div class="actions">
        <button type="button" id="validate-config" class="btn secondary">Validate</button>
        <button type="submit">Save</button>
      </div>
    </form>
  `;
}

async function loadCatalog() {
  catalog = await api("/api/catalog");
}

async function loadLocalModels() {
  localModels = await api("/api/local-models");
}

async function loadLocalServers() {
  try {
    const prevGpu = localServers.gpu;
    const data = await api("/api/local-servers");
    localServers = data;
    if (!localServers.gpu && prevGpu) localServers.gpu = prevGpu;
    if (!Array.isArray(localServers.llamacpp) || localServers.llamacpp.length === 0) {
      try {
        const probe = await api("/api/llamacpp?baseUrl=" + encodeURIComponent("http://127.0.0.1:8080/v1"));
        if (probe && !Array.isArray(probe.endpoints)) {
          localServers.llamacpp = [{ id: "llamacpp", ...probe }];
        }
      } catch {
        /* optional probe */
      }
    }
    const phase = data.lateinfer?.servePhase || data.gpu?.live?.servePhase;
    if (phase === "gpu-running" || phase === "host-ram" || phase === "exited" || (phase === "down" && !data.lateinfer?.processAlive)) {
      setLateInferStartingFlag(false);
    }
    if (data.lateinfer?.guiDisconnected) delete data.lateinfer.guiDisconnected;
  } catch (error) {
    if (localServers.gpu || localServers.lateinfer) {
      if (localServers.lateinfer) localServers.lateinfer = { ...localServers.lateinfer, guiDisconnected: true };
      throw error;
    }
    localServers = { lateinfer: { running: false, models: [], guiDisconnected: true }, ollama: { running: false, models: [] }, llamacpp: [], lateInferBinary: null, llamaServerBinary: null, ollamaBinary: null, gpu: null };
    throw error;
  }
}

async function loadRuns() {
  runs = await api("/api/runs?limit=100");
  if (selectedRunId) {
    try {
      const detail = await api(`/api/runs/${encodeURIComponent(selectedRunId)}`);
      runs = runs.map((r) => (r.id === detail.id ? detail : r));
      if (!runs.some((r) => r.id === detail.id)) runs.unshift(detail);
    } catch {
      selectedRunId = null;
    }
  }
}

async function loadThreads() {
  threads = await api("/api/chats");
}

async function loadThread(id) {
  currentThread = await api(`/api/chats/${encodeURIComponent(id)}`);
  upsertThread({
    id: currentThread.id,
    title: currentThread.title,
    updatedAt: currentThread.updatedAt,
    pin: currentThread.pin,
    agents: currentThread.agents,
  });
}

async function openNewChat() {
  const created = await api("/api/chats", { method: "POST", body: JSON.stringify({ pin: "auto" }) });
  currentThread = created;
  upsertThread(created);
  location.hash = `#chat/${created.id}`;
}

async function deleteChat(id) {
  if (!window.confirm("Delete this chat? This cannot be undone.")) return;
  await api(`/api/chats/${encodeURIComponent(id)}`, { method: "DELETE" });
  threads = threads.filter((t) => t.id !== id);
  if (currentThread?.id === id || threadIdFromHash() === id) {
    currentThread = null;
    const next = threads[0];
    if (next) location.hash = `#chat/${next.id}`;
    else {
      history.replaceState({}, "", "#chat");
      await openNewChat();
    }
  } else {
    renderThreadList();
  }
}

async function renderChat() {
  const id = threadIdFromHash();
  if (id && currentThread?.id !== id) {
    try {
      await loadThread(id);
    } catch {
      currentThread = { id, title: "New chat", messages: [], pin: "auto", agents: [] };
    }
  }
  if (!id) currentThread = currentThread?.messages?.length ? currentThread : null;
  ensureChatLayout();
}

async function render() {
  setActiveNav();
  const page = pageId();
  const hashParts = location.hash.replace("#", "").split("/");
  selectedRunId = page === "runs" && hashParts[1] ? hashParts[1] : selectedRunId;

  if (page === "chat" || page === "") await renderChat();
  else if (page === "overview") renderOverview();
  else if (page === "specialists") renderSpecialists();
  else if (page === "backends") await renderBackends();
  else if (page === "runs") {
    await loadRuns();
    renderRuns();
  } else if (page === "dispatch") renderDispatch();
  else if (page === "local-models") {
    try {
      await loadLocalModels();
      await loadLocalServers();
      renderLocalModels();
    } catch (error) {
      applyLocalModelsDisconnect(error);
      try {
        await probeLateInferHealthDirect();
        if ($("gpu-status") && pageId() === "local-models") renderLocalModels();
      } catch {
        /* keep Local models + token banner */
      }
    }
    if (!lateInferHubCatalog.loaded) void loadLateInferHubModels();
    if (!llamaGgufCatalog.loaded) void loadLlamaGgufCatalog(llamaGgufSearch);
  } else if (page === "allowlist") renderAllowlist();
  else if (page === "updates") renderUpdates();
  else if (page === "config") {
    const cfg = await api("/api/config");
    renderConfig({ yaml: cfg.yaml });
  } else await renderChat();
}

function showGate(error) {
  $("shell").classList.add("hidden");
  $("gate").classList.remove("hidden");
  if (error) {
    $("gate-error").hidden = false;
    $("gate-error").textContent = error;
  }
}

function showShell() {
  $("gate").classList.add("hidden");
  $("shell").classList.remove("hidden");
}

function connectEvents() {
  if (events) events.close();
  events = new EventSource(`/api/events?token=${encodeURIComponent(token)}`);
  events.addEventListener("run", (ev) => {
    const run = JSON.parse(ev.data);
    runs = [run, ...runs.filter((r) => r.id !== run.id)].slice(0, 100);
    if (pageId() === "runs") renderRuns();
  });
  events.addEventListener("catalog", (ev) => {
    catalog = JSON.parse(ev.data);
    if (["overview", "specialists", "backends", "allowlist", "updates", "dispatch", "local-models"].includes(pageId())) {
      render();
    } else if (pageId() === "chat") {
      renderChatHeader();
      renderMessages();
      renderThreadList();
    }
  });
  events.addEventListener("local-models", (ev) => {
    localModels = JSON.parse(ev.data);
    if (pageId() === "local-models") renderLocalModels();
    if (pageId() === "overview") renderOverview();
  });
  events.addEventListener("vllm", (ev) => {
    const status = JSON.parse(ev.data);
    if (localModels) localModels = { ...localModels, vllm: status };
    syncVllmPoll(status);
    if (pageId() === "local-models") renderLocalModels();
    else if (pageId() === "overview" || pageId() === "backends") render();
  });
  events.addEventListener("chats", (ev) => {
    threads = JSON.parse(ev.data);
    renderThreadList();
    if (pageId() === "chat") renderChatHeader();
  });
  events.addEventListener("chat", (ev) => {
    const thread = JSON.parse(ev.data);
    upsertThread(thread);
    const active = threadIdFromHash();
    if (thread.id === active || thread.id === currentThread?.id) {
      currentThread = thread;
      if (pageId() === "chat") {
        ensureChatLayout();
      }
    } else {
      renderThreadList();
    }
  });
  events.addEventListener("chat-heartbeat", (ev) => {
    const data = JSON.parse(ev.data);
    if (pageId() !== "chat" || !currentThread || data.threadId !== currentThread.id) return;
    for (const row of data.thinking ?? []) {
      const msg = (currentThread.messages ?? []).find((m) => m.id === row.id);
      if (msg) {
        msg.status = row.status;
        msg.thinkingPhase = row.thinkingPhase;
        msg.thinkingStartedAt = row.thinkingStartedAt;
      }
      const label = document.querySelector(`[data-thinking-id="${CSS.escape(row.id)}"] .thinking-label`);
      if (label) label.textContent = thinkingChipLabel(row, data.now);
    }
  });
}

$("gate-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  token = $("token-input").value.trim();
  sessionStorage.setItem(TOKEN_KEY, token);
  try {
    await loadSession();
    showShell();
    await loadCatalog();
    await loadThreads();
    connectEvents();
    await render();
    void api("/api/updates")
      .then((data) => {
        updateCheck = data;
        if (data?.updateLate || data?.updateOrchestrator || data?.bothNewer) {
          updateFlash = data.message || "GitHub has a newer file. Open Settings → Updates.";
          updateFlashKind = "ok";
        }
      })
      .catch(() => {
        /* start check must not block chat or MCP */
      });
  } catch (error) {
    showGate(error.message);
  }
});

function startNewChat() {
  openNewChat().catch((error) => {
    $("main").innerHTML = flash(error.message, "bad");
  });
}

$("shell").addEventListener("click", (event) => {
  if (event.target.closest?.("[data-new-chat]")) {
    event.preventDefault();
    startNewChat();
  }
});

document.addEventListener("change", (event) => {
  const select = event.target?.closest?.(".theme-select");
  if (!select) return;
  applyTheme(select.value);
});

$("main").addEventListener("change", async (event) => {
  if (event.target?.id === "lateinfer-on-computer-only") {
    lateInferOnComputerOnly = event.target.checked === true;
    refreshLateInferHubList();
    return;
  }
  if (event.target?.id === "lateinfer-show-gated") {
    lateInferShowGated = event.target.checked === true;
    try {
      sessionStorage.setItem("orchestrator.lateinfer.showGated", lateInferShowGated ? "1" : "0");
    } catch {
      /* ignore */
    }
    refreshLateInferHubList();
    return;
  }
  if (event.target?.id === "lateinfer-use-all-gpus") {
    lateInferUseAllGpus = event.target.checked === true;
    return;
  }
  if (event.target?.id === "lateinfer-gpu-pick") {
    lateInferGpuId = event.target.value || "auto";
    return;
  }
  if (event.target?.id === "chat-thread-pick") {
    const id = event.target.value;
    if (id) location.hash = `#chat/${id}`;
    return;
  }
  const logoInput = event.target?.closest?.(".backend-logo-input");
  if (!(logoInput instanceof HTMLInputElement) || !logoInput.files?.[0]) return;
  const backendId = logoInput.getAttribute("data-backend");
  const file = logoInput.files[0];
  if (!backendId) return;
  if (file.size > 512 * 1024) {
    $("backends-status").innerHTML = flash("Logo must be 512 KiB or smaller", "bad");
    logoInput.value = "";
    return;
  }
  try {
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result ?? ""));
      reader.onerror = () => reject(new Error("Could not read logo file"));
      reader.readAsDataURL(file);
    });
    const result = await api(`/api/backends/${encodeURIComponent(backendId)}/logo`, {
      method: "POST",
      body: JSON.stringify({ data: dataUrl }),
    });
    catalog = result.catalog ?? catalog;
    await renderBackends();
    $("backends-status").innerHTML = flash("Logo saved", "ok");
  } catch (error) {
    $("backends-status").innerHTML = flash(error.message, "bad");
    logoInput.value = "";
  }
});

$("thread-list").addEventListener("click", (event) => {
  const del = event.target.closest?.("[data-delete-thread]");
  if (del) {
    event.preventDefault();
    event.stopPropagation();
    const id = del.getAttribute("data-delete-thread");
    if (id) deleteChat(id).catch((error) => {
      $("main").insertAdjacentHTML("afterbegin", flash(error.message, "bad"));
    });
    return;
  }
  const btn = event.target.closest?.("[data-thread]");
  if (!btn) return;
  location.hash = `#chat/${btn.getAttribute("data-thread")}`;
});

$("main").addEventListener("submit", async (event) => {
  const form = event.target;
  if (!(form instanceof HTMLFormElement)) return;
  event.preventDefault();
  const data = new FormData(form);
  try {
    if (form.id === "composer-form") {
      const message = String(data.get("message") ?? "").trim();
      if (!message || sending) return;
      sending = true;
      const pin = composerPin();
      let id = threadIdFromHash() ?? currentThread?.id;
      if (!id) {
        const created = await api("/api/chats", { method: "POST", body: JSON.stringify({ pin }) });
        id = created.id;
        currentThread = created;
        history.replaceState({}, "", `#chat/${id}`);
      }
      $("composer-input").value = "";
      const thread = await api(`/api/chats/${encodeURIComponent(id)}/messages`, {
        method: "POST",
        body: JSON.stringify({ message, pin, wait: false, cwd: currentThread?.workspaceDir || undefined }),
      });
      currentThread = thread;
      ensureChatLayout();
      sending = false;
      return;
    }
    if (form.id === "dispatch-form") {
      $("dispatch-status").innerHTML = flash("Running…", "ok");
      const run = await api("/api/dispatch", {
        method: "POST",
        body: JSON.stringify({
          specialist: data.get("specialist"),
          task: data.get("task"),
          backend: data.get("backend") || undefined,
          cwd: data.get("cwd") || undefined,
          model: data.get("model") || undefined,
          wait: data.get("wait") === "on",
        }),
      });
      $("dispatch-status").innerHTML = flash(`Run ${run.id} · ${run.status}`, run.status === "error" ? "bad" : "ok");
      location.hash = `#runs/${run.id}`;
    } else if (form.id === "workflow-form") {
      $("workflow-status").innerHTML = flash("Running workflow…", "ok");
      const result = await api("/api/workflows", {
        method: "POST",
        body: JSON.stringify({
          workflow: data.get("workflow"),
          task: data.get("task"),
          cwd: data.get("cwd") || undefined,
        }),
      });
      $("workflow-status").innerHTML = flash(`${result.workflow} · ${result.status}`, result.status === "error" ? "bad" : "ok");
    } else if (form.id === "follow-form") {
      $("follow-status").innerHTML = flash("Sending…", "ok");
      const run = await api("/api/follow-up", {
        method: "POST",
        body: JSON.stringify({ runId: selectedRunId, message: data.get("message") }),
      });
      location.hash = `#runs/${run.id}`;
      await loadRuns();
      renderRuns();
    } else if (form.id === "allow-form") {
      const result = await api("/api/allowlist", {
        method: "POST",
        body: JSON.stringify({ path: data.get("path") }),
      });
      catalog.writePolicy = { ...catalog.writePolicy, allowedDirectories: result.allowedDirectories };
      $("allow-status").innerHTML = flash("Directory granted", "ok");
      renderAllowlist();
    } else if (form.id === "mcp-listen-host-form") {
      const result = await api("/api/mcp/listen-host", {
        method: "POST",
        body: JSON.stringify({ listenHost: String(data.get("listenHost") ?? "").trim() }),
      });
      if (typeof result.mcpUrl === "string") sessionInfo.mcpUrl = result.mcpUrl;
      if (typeof result.savedListenHost === "string") {
        sessionInfo.configListenHost = result.savedListenHost;
      }
      const urlInput = $("mcp-url");
      if (urlInput) urlInput.value = mcpUrlForLate();
      const saved = result.savedListenHost || "";
      const msg = result.restartRequired
        ? `Saved ${saved}. Restart the GUI on your computer so Late can use ${result.mcpUrl}.`
        : `Listen host ${saved}.`;
      $("mcp-listen-status").innerHTML = flash(msg, "ok");
    } else if (form.classList.contains("backend-model-form")) {
      const backendId = form.getAttribute("data-backend");
      const model = String(data.get("model") ?? "").trim();
      if (!backendId || !model) return;
      const result = await api(`/api/backends/${encodeURIComponent(backendId)}`, {
        method: "PATCH",
        body: JSON.stringify({ model }),
      });
      catalog = result.catalog ?? catalog;
      await renderBackends();
      $("backends-status").innerHTML = flash(`Saved model ${result.model}`, "ok");
    } else if (form.classList.contains("backend-nick-form")) {
      const backendId = form.getAttribute("data-backend");
      if (!backendId) return;
      const nickname = String(data.get("nickname") ?? "").trim();
      const result = await api(`/api/backends/${encodeURIComponent(backendId)}`, {
        method: "PATCH",
        body: JSON.stringify({ nickname }),
      });
      catalog = result.catalog ?? catalog;
      await renderBackends();
      $("backends-status").innerHTML = flash(nickname ? `Nickname saved as ${nickname}` : "Nickname cleared", "ok");
    } else if (form.classList.contains("secret-form")) {
      const name = form.getAttribute("data-name");
      const value = data.get("value");
      if (!name || !value) return;
      const result = await api("/api/secrets", {
        method: "PUT",
        body: JSON.stringify({ name, value }),
      });
      catalog = result.catalog ?? catalog;
      if (pageId() === "local-models") {
        await loadLocalModels();
        renderLocalModels();
        $("local-models-status").innerHTML = flash(`${name} saved locally`, "ok");
      } else {
        await renderBackends();
        $("backends-status").innerHTML = flash(`${name} saved locally`, "ok");
      }
    } else if (form.id === "config-form") {
      await api("/api/config", { method: "PUT", body: JSON.stringify({ yaml: data.get("yaml") }) });
      await loadCatalog();
      renderConfig({ yaml: data.get("yaml"), message: "Saved.", kind: "ok" });
    }
  } catch (error) {
    sending = false;
    const statusId =
      form.id === "dispatch-form"
        ? "dispatch-status"
        : form.id === "workflow-form"
          ? "workflow-status"
          : form.id === "follow-form"
            ? "follow-status"
            : form.id === "allow-form"
              ? "allow-status"
              : form.id === "mcp-listen-host-form"
                ? "mcp-listen-status"
                : pageId() === "local-models"
                  ? "local-models-status"
                  : "backends-status";
    if (form.id === "composer-form") {
      const list = $("thread-messages");
      if (list) list.insertAdjacentHTML("beforeend", flash(error.message, "bad"));
      return;
    }
    if (form.id === "config-form") {
      renderConfig({ yaml: data.get("yaml"), message: error.message, kind: "bad" });
      return;
    }
    if (statusId && $(statusId)) $(statusId).innerHTML = flash(error.message, "bad");
  }
});

$("main").addEventListener("click", async (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const copyMcp = target.closest?.("[data-copy-mcp]");
  if (copyMcp) {
    const url = mcpUrlForLate();
    const status = $("mcp-copy-status");
    const ok = await copyText(url);
    if (status) status.textContent = ok ? "Copied." : url;
    return;
  }
  const clearSecret = target.closest?.("[data-clear-secret]");
  if (clearSecret) {
    const name = clearSecret.getAttribute("data-clear-secret");
    if (!name) return;
    try {
      const payload =
        name === "XAI_API_KEY" || name === "GROK_API_KEY"
          ? { names: ["XAI_API_KEY", "GROK_API_KEY"] }
          : { name };
      const result = await api("/api/secrets", {
        method: "DELETE",
        body: JSON.stringify(payload),
      });
      catalog = result.catalog ?? catalog;
      if (pageId() === "backends") {
        await renderBackends();
        if ($("backends-status")) $("backends-status").innerHTML = flash(`${name} cleared`, "ok");
      } else {
        await loadLocalModels();
        renderLocalModels();
        if ($("local-models-status")) $("local-models-status").innerHTML = flash(`${name} cleared`, "ok");
      }
    } catch (error) {
      const statusId = pageId() === "backends" ? "backends-status" : "local-models-status";
      if ($(statusId)) $(statusId).innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const delCompiled = target.closest?.("[data-lateinfer-delete]");
  if (delCompiled instanceof HTMLElement) {
    event.preventDefault();
    event.stopPropagation();
    const hubId = delCompiled.getAttribute("data-lateinfer-delete")?.trim();
    if (!hubId) return;
    const serving = lateInferServingThisHubId(hubId);
    const ok = window.confirm(
      serving
        ? "This stops late-infer on 127.0.0.1:8010, then removes the compiled snapshot from your computer. Download again from the Hub store to restore."
        : "This removes the compiled snapshot from your computer. Download again from the Hub store to restore.",
    );
    if (!ok) return;
    try {
      if (serving) {
        await api("/api/local-servers/stop", {
          method: "POST",
          body: JSON.stringify({ kind: "lateinfer" }),
        });
      }
      const result = await api("/api/local-servers/delete", {
        method: "POST",
        body: JSON.stringify({ kind: "lateinfer", model: hubId, confirm: true }),
      });
      applyCompiledDeleteResult(result, hubId);
      await loadLocalServers();
      syncLateInferPoll(localServers.lateinfer);
      renderLocalModels();
      if ($("local-models-status")) {
        $("local-models-status").innerHTML = flash(
          `${hubId} removed from your computer. Download again from the Hub store to restore.`,
          "ok",
        );
      }
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const hubPick = target.closest?.("[data-hub-id]");
  if (hubPick instanceof HTMLElement) {
    const hubId = hubPick.getAttribute("data-hub-id")?.trim();
    if (!hubId) return;
    const pane = hubPick.getAttribute("data-lateinfer-pane") || hubPick.closest("[data-lateinfer-pane]")?.getAttribute("data-lateinfer-pane");
    if (pane === "compiled") {
      lateInferCompiledId = hubId;
      refreshLateInferCompiledList();
      return;
    }
    lateInferHubId = hubId;
    const hubInput = $("lateinfer-hub");
    if (hubInput) hubInput.value = hubId;
    try {
      sessionStorage.setItem("orchestrator.lateinfer.hub", hubId);
    } catch {
      /* ignore */
    }
    refreshLateInferHubList();
    refreshLateInferCheckStatus();
    return;
  }
  if (target.id === "lateinfer-check") {
    runLateInferHubCheck();
    return;
  }
  if (target.id === "lateinfer-download") {
    const hubInput = $("lateinfer-hub");
    const hubId = resolveLateInferHubIdForDownload();
    if (hubInput) hubInput.value = hubId;
    if (!hubId) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Enter a Hugging Face Hub id.", "bad");
      return;
    }
    if (lateInferCompileBlockedByGpu()) {
      const gpu = lateInferGpuState();
      const reason = gpu.compileReason || gpu.runtimeReason || "This compiler cannot compile for the GPU on your computer.";
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(reason, "bad");
      return;
    }
    const catalogModel = findLateInferHubCatalogModel(hubId);
    if (catalogModel && hubModelIsLoadable(catalogModel) === false) {
      const reason =
        catalogModel.serveBlockedReason ||
        (catalogModel.configStatus === "missing"
          ? "Hub config.json is missing — pick a safetensors Instruct snapshot. GGUF belongs under llama.cpp."
          : "This snapshot is not loadable for late-infer on this GPU.");
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(reason, "bad");
      refreshLateInferCheckStatus();
      return;
    }
    if (catalogModel?.gated || catalogModel?.gatedNeedsLicense) {
      if (!lateInferHfTokenSet) {
        if ($("local-models-status")) {
          $("local-models-status").innerHTML = flash(
            `${hubId} is gated. Accept the license on the Hugging Face model card, then save a read token in Settings → Local models before Download.`,
            "bad",
          );
        }
        refreshLateInferCheckStatus();
        const tokenForm = $("hf-token-form");
        if (tokenForm?.scrollIntoView) tokenForm.scrollIntoView({ behavior: "smooth", block: "center" });
        return;
      }
      const proceed = window.confirm(
        `${hubId} is gated on Hugging Face.\n\nBefore Download:\n1) Accept the model license on the Hub model card (while logged in)\n2) Ensure a read HF token is saved in Settings → Local models\n\nContinue Download?`,
      );
      if (!proceed) {
        refreshLateInferCheckStatus();
        return;
      }
    }
    try {
      lateInferHubId = hubId;
      try {
        sessionStorage.setItem("orchestrator.lateinfer.hub", hubId);
      } catch {
        /* ignore */
      }
      lateInferDownloadBusy = true;
      if ($("local-models-status")) {
        $("local-models-status").innerHTML = flash(`Downloading ${hubId} from Hugging Face…`, "ok");
      }
      await api("/api/local-servers/download", {
        method: "POST",
        body: JSON.stringify({
          kind: "lateinfer",
          model: hubId,
          gpuId: lateInferGpuIdForPost(),
        }),
      });
      await loadLocalServers();
      syncLateInferPoll(localServers.lateinfer);
      renderLocalModels();
    } catch (error) {
      lateInferDownloadBusy = false;
      if (localServers.lateinfer) {
        localServers.lateinfer = {
          ...localServers.lateinfer,
          downloading: false,
          phase: "error",
          error: lateInferErrorLine(error.message) || error.message,
        };
      }
      syncLateInferPoll(localServers.lateinfer);
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(lateInferErrorLine(error.message) || error.message, "bad");
    }
    return;
  }
  if (target.id === "lateinfer-convert") {
    const model = resolveLateInferIdForStart();
    if (!model) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Nothing compiled on your computer yet.", "bad");
      return;
    }
    const row = compiledRowForId(model);
    if (row && row.convertOk === false) {
      const reason = row.convertReason || intelIrMissingStartMessage();
      lateInferStartError = reason;
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(reason, "bad");
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(reason, "bad");
      return;
    }
    try {
      lateInferDownloadBusy = true;
      if ($("local-models-status")) {
        $("local-models-status").innerHTML = flash(`Converting ${model} to OpenVINO IR on your computer…`, "ok");
      }
      renderLocalModels();
      await api("/api/local-servers/convert", {
        method: "POST",
        body: JSON.stringify({
          kind: "lateinfer",
          model,
          gpuId: lateInferGpuIdForPost(),
        }),
      });
      await loadLocalServers();
      syncLateInferPoll(localServers.lateinfer);
      renderLocalModels();
    } catch (error) {
      lateInferDownloadBusy = false;
      lateInferStartError = lateInferErrorLine(error.message) || error.message;
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(lateInferStartError, "bad");
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(lateInferStartError, "bad");
    }
    return;
  }
  if (target.id === "lateinfer-start") {
    try {
      const model = resolveLateInferIdForStart();
      if (!model) {
        lateInferStartError = "Nothing compiled on your computer yet.";
        if ($("local-models-status")) $("local-models-status").innerHTML = flash(lateInferStartError, "bad");
        renderLocalModels();
        if ($("local-models-status")) $("local-models-status").innerHTML = flash(lateInferStartError, "bad");
        scrollLateInferStartBannerIntoView();
        return;
      }
      const refuse = lateInferStartRefuseText();
      if (refuse) {
        lateInferStartError = refuse;
        if ($("local-models-status")) $("local-models-status").innerHTML = flash(refuse, "bad");
        renderLocalModels();
        if ($("local-models-status")) $("local-models-status").innerHTML = flash(refuse, "bad");
        scrollLateInferStartBannerIntoView();
        try {
          await api("/api/local-servers/start", {
            method: "POST",
            body: JSON.stringify({
              kind: "lateinfer",
              model,
              useAllGpus: lateInferUseAllGpus,
              gpuId: lateInferGpuIdForPost(),
              vramMaxMiB: selectedCompiledVramMaxMiB(),
            }),
          });
        } catch (startErr) {
          lateInferStartError = lateInferErrorLine(startErr.message) || startErr.message || refuse;
          if ($("local-models-status")) $("local-models-status").innerHTML = flash(lateInferStartError, "bad");
        }
        renderLocalModels();
        if ($("local-models-status")) $("local-models-status").innerHTML = flash(lateInferStartError, "bad");
        scrollLateInferStartBannerIntoView();
        return;
      }
      lateInferCompiledId = model;
      lateInferStartError = "";
      setLateInferStartingFlag(true);
      if (localServers.lateinfer) {
        localServers.lateinfer = { ...localServers.lateinfer, starting: true, servePhase: "starting" };
      }
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("starting on idle GPU…", "ok");
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("starting on idle GPU…", "ok");
      await api("/api/local-servers/start", {
        method: "POST",
        body: JSON.stringify({
          kind: "lateinfer",
          model,
          useAllGpus: lateInferUseAllGpus,
          gpuId: lateInferGpuIdForPost(),
          vramMaxMiB: selectedCompiledVramMaxMiB(),
        }),
      });
      await loadLocalServers();
      syncLateInferPoll(localServers.lateinfer);
      await loadCatalog();
      renderLocalModels();
    } catch (error) {
      if (/Lost connection to the orchestrator GUI/i.test(error.message)) {
        applyLocalModelsDisconnect(error);
        syncLateInferPoll(localServers.lateinfer);
        return;
      }
      setLateInferStartingFlag(false);
      lateInferStartError = lateInferErrorLine(error.message) || error.message;
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(lateInferStartError, "bad");
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(lateInferStartError, "bad");
      scrollLateInferStartBannerIntoView();
    }
    return;
  }
  if (target.id === "lateinfer-stop") {
    try {
      await api("/api/local-servers/stop", { method: "POST", body: JSON.stringify({ kind: "lateinfer" }) });
      setLateInferStartingFlag(false);
      await loadLocalServers();
      renderLocalModels();
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }

  const localDownload = target.closest?.("[data-local-download]");
  if (localDownload instanceof HTMLElement) {
    const modelId = localDownload.getAttribute("data-local-download");
    if (!modelId) return;
    try {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(`Downloading ${modelId}…`, "ok");
      await api("/api/local-models/download", {
        method: "POST",
        body: JSON.stringify({ modelId }),
      });
      await loadLocalModels();
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(`Download started for ${modelId}`, "ok");
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const vllmStart = target.closest?.("[data-vllm-start]");
  if (vllmStart instanceof HTMLElement) {
    const modelId = vllmStart.getAttribute("data-vllm-start");
    const runtime = vllmStart.getAttribute("data-vllm-runtime") || "host";
    if (!modelId) return;
    try {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(`Starting vLLM ${modelId} (${runtime})…`, "ok");
      const body = { modelId, useAllGpus: true };
      if (runtime === "docker") body.runtime = "docker";
      else body.runtime = "host";
      const result = await api("/api/vllm/start", { method: "POST", body: JSON.stringify(body) });
      if (localModels) localModels = { ...localModels, vllm: result.vllm ?? localModels.vllm };
      syncVllmPoll(result.vllm ?? localModels?.vllm);
      await loadLocalModels();
      await loadCatalog();
      renderLocalModels();
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const vllmStop = target.closest?.("[data-vllm-stop]");
  if (vllmStop instanceof HTMLElement) {
    const modelId = vllmStop.getAttribute("data-vllm-stop") || undefined;
    const backendId = vllmStop.getAttribute("data-vllm-backend") || undefined;
    try {
      const result = await api("/api/vllm/stop", {
        method: "POST",
        body: JSON.stringify({ modelId, backendId: backendId || undefined }),
      });
      catalog = result.catalog ?? catalog;
      if (localModels) localModels = { ...localModels, vllm: result.vllm ?? localModels.vllm };
      await loadLocalModels();
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("vLLM stopped", "ok");
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const vllmRemove = target.closest?.("[data-vllm-remove]");
  if (vllmRemove instanceof HTMLElement) {
    const modelId = vllmRemove.getAttribute("data-vllm-remove") || undefined;
    const backendId = vllmRemove.getAttribute("data-vllm-backend") || undefined;
    const ok = window.confirm(
      "Remove this vLLM backend from the mix? This stops the instance and drops it from agents.config.yaml. Weights on disk stay until you Delete weights.",
    );
    if (!ok) return;
    try {
      const result = await api("/api/vllm/remove", {
        method: "POST",
        body: JSON.stringify({ modelId, backendId: backendId || undefined }),
      });
      catalog = result.catalog ?? catalog;
      if (localModels) localModels = { ...localModels, vllm: result.vllm ?? localModels.vllm };
      await loadLocalModels();
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Removed from mix", "ok");
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const vllmDelete = target.closest?.("[data-vllm-delete-weights]");
  if (vllmDelete instanceof HTMLElement) {
    const modelId = vllmDelete.getAttribute("data-vllm-delete-weights");
    if (!modelId) return;
    const ok = window.confirm(
      `Permanently delete downloaded weights for ${modelId} from this computer? This cannot be undone.`,
    );
    if (!ok) return;
    try {
      const result = await api("/api/local-models/delete", {
        method: "POST",
        body: JSON.stringify({ modelId, confirm: true }),
      });
      catalog = result.catalog ?? catalog;
      await loadLocalModels();
      renderLocalModels();
      if ($("local-models-status")) {
        $("local-models-status").innerHTML = flash(
          result.deleted ? `${modelId} weights deleted` : `${modelId} had no weights on disk`,
          "ok",
        );
      }
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  if (target.id === "ollama-start") {
    try {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Starting Ollama…", "ok");
      await api("/api/local-servers/start", { method: "POST", body: JSON.stringify({ kind: "ollama" }) });
      await loadLocalServers();
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Ollama start requested", "ok");
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  if (target.id === "ollama-stop") {
    try {
      await api("/api/local-servers/stop", { method: "POST", body: JSON.stringify({ kind: "ollama" }) });
      await loadLocalServers();
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Ollama stopped", "ok");
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  if (target.id === "ollama-register") {
    try {
      const result = await api("/api/ollama/connect", { method: "POST", body: JSON.stringify({}) });
      catalog = result.catalog ?? catalog;
      await loadLocalServers();
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Ollama backend registered", "ok");
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const ggufDl = target.closest?.("[data-llamacpp-gguf-download]");
  if (ggufDl) {
    const repo = ggufDl.getAttribute("data-llamacpp-gguf-download")?.trim();
    const filename = ggufDl.getAttribute("data-llamacpp-gguf-file")?.trim() || undefined;
    if (!repo) return;
    try {
      llamaGgufSelectedId = repo;
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(`Downloading GGUF ${repo}…`, "ok");
      const job = await api("/api/local-servers/gguf-download", {
        method: "POST",
        body: JSON.stringify({ repo, filename }),
      });
      llamaGgufJobs = [job, ...(llamaGgufJobs || []).filter((j) => j.repo !== repo)];
      const poll = setInterval(async () => {
        try {
          const res = await api("/api/local-servers/gguf-download");
          llamaGgufJobs = Array.isArray(res?.jobs) ? res.jobs : llamaGgufJobs;
          const cur = llamaGgufJobFor(repo);
          if (cur?.status === "done" && cur.localPath) {
            clearInterval(poll);
            llamaGgufPath = cur.localPath;
            const gguf = $("llamacpp-gguf");
            if (gguf) gguf.value = cur.localPath;
            await loadLlamaGgufCatalog(llamaGgufSearch);
            renderLocalModels();
            if ($("local-models-status")) $("local-models-status").innerHTML = flash(`Downloaded ${repo} → ${cur.localPath}`, "ok");
          } else if (cur?.status === "error") {
            clearInterval(poll);
            if ($("local-models-status")) $("local-models-status").innerHTML = flash(cur.error || cur.message || "GGUF download failed", "bad");
            await loadLlamaGgufCatalog(llamaGgufSearch);
            renderLocalModels();
          } else {
            const slot = $("llamacpp-gguf-catalog");
            if (slot) slot.innerHTML = renderLlamaGgufCatalogRowsHtml();
          }
        } catch {
          clearInterval(poll);
        }
      }, 1500);
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const ggufUse = target.closest?.("[data-llamacpp-gguf-use]");
  if (ggufUse) {
    const p = ggufUse.getAttribute("data-llamacpp-gguf-use")?.trim();
    const id = ggufUse.getAttribute("data-llamacpp-gguf-id")?.trim();
    if (p) {
      llamaGgufPath = p;
      const gguf = $("llamacpp-gguf");
      if (gguf) gguf.value = p;
    }
    if (id) llamaGgufSelectedId = id;
    if ($("local-models-status")) $("local-models-status").innerHTML = flash(`Using ${p}`, "ok");
    return;
  }
  const ggufStartRow = target.closest?.("[data-llamacpp-gguf-start]");
  if (ggufStartRow) {
    const p = ggufStartRow.getAttribute("data-llamacpp-gguf-path")?.trim();
    const id = ggufStartRow.getAttribute("data-llamacpp-gguf-start")?.trim();
    if (p) {
      llamaGgufPath = p;
      const gguf = $("llamacpp-gguf");
      if (gguf) gguf.value = p;
    }
    if (id) llamaGgufSelectedId = id;
    target.id = "llamacpp-start";
  }
  if (target.id === "llamacpp-hub-download") {
    const repo = String($("llamacpp-hub-id")?.value ?? llamaGgufSelectedId ?? "").trim();
    if (!repo.includes("/")) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Paste a Hub org/model id first.", "bad");
      return;
    }
    try {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(`Downloading GGUF ${repo}…`, "ok");
      const job = await api("/api/local-servers/gguf-download", {
        method: "POST",
        body: JSON.stringify({ repo }),
      });
      llamaGgufJobs = [job, ...(llamaGgufJobs || []).filter((j) => j.repo !== repo)];
      const poll = setInterval(async () => {
        try {
          const res = await api("/api/local-servers/gguf-download");
          llamaGgufJobs = Array.isArray(res?.jobs) ? res.jobs : llamaGgufJobs;
          const cur = llamaGgufJobFor(repo);
          if (cur?.status === "done" && cur.localPath) {
            clearInterval(poll);
            llamaGgufPath = cur.localPath;
            const gguf = $("llamacpp-gguf");
            if (gguf) gguf.value = cur.localPath;
            await loadLlamaGgufCatalog(llamaGgufSearch);
            renderLocalModels();
            if ($("local-models-status")) $("local-models-status").innerHTML = flash(`Downloaded ${repo}`, "ok");
          } else if (cur?.status === "error") {
            clearInterval(poll);
            if ($("local-models-status")) $("local-models-status").innerHTML = flash(cur.error || cur.message || "GGUF download failed", "bad");
          }
        } catch {
          clearInterval(poll);
        }
      }, 1500);
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  if (target.id === "llamacpp-start") {
    const modelPath = String($("llamacpp-gguf")?.value ?? llamaGgufPath ?? "").trim();
    if (!modelPath) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Enter an absolute .gguf path or Download from the GGUF Hub store.", "bad");
      return;
    }
    try {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Starting llama-server…", "ok");
      await api("/api/local-servers/start", {
        method: "POST",
        body: JSON.stringify({ kind: "llamacpp", modelPath }),
      });
      await loadLocalServers();
      renderLocalModels();
      const gguf = $("llamacpp-gguf");
      if (gguf) gguf.value = modelPath;
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("llama-server start requested", "ok");
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  if (target.id === "llamacpp-stop") {
    try {
      await api("/api/local-servers/stop", { method: "POST", body: JSON.stringify({ kind: "llamacpp" }) });
      await loadLocalServers();
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("llama-server stopped", "ok");
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  if (target.id === "llamacpp-register") {
    try {
      const model = String($("llamacpp-model")?.value ?? "").trim();
      const result = await api("/api/llamacpp/connect", {
        method: "POST",
        body: JSON.stringify(model ? { model } : {}),
      });
      catalog = result.catalog ?? catalog;
      await loadLocalServers();
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("llama.cpp backend registered", "ok");
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }

  const logoRemove = target.closest?.("[data-logo-remove]");
  if (logoRemove) {
    const backendId = logoRemove.getAttribute("data-logo-remove");
    if (!backendId) return;
    try {
      const result = await api(`/api/backends/${encodeURIComponent(backendId)}/logo`, { method: "DELETE" });
      catalog = result.catalog ?? catalog;
      await renderBackends();
      $("backends-status").innerHTML = flash("Logo removed", "ok");
    } catch (error) {
      $("backends-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const modeBtn = target.closest?.(".mode-btn");
  if (modeBtn) {
    const mode = modeBtn.getAttribute("data-mode") || "auto";
    for (const btn of document.querySelectorAll(".mode-btn")) {
      btn.setAttribute("aria-pressed", btn === modeBtn ? "true" : "false");
    }
    const pinSelect = $("route-pin");
    if (pinSelect && mode !== "single") pinSelect.value = "";
    const id = threadIdFromHash() ?? currentThread?.id;
    if (id && currentThread) {
      currentThread.pin = mode;
      try {
        await api(`/api/chats/${encodeURIComponent(id)}/pin`, {
          method: "POST",
          body: JSON.stringify({ pin: mode }),
        });
      } catch {
        /* keep local selection */
      }
    }
    return;
  }
  if (target.closest?.("[data-grant-cancel]")) {
    $("grant-card")?.classList.add("hidden");
    return;
  }
  if (target.closest?.("[data-grant-folder]")) {
    const path = $("grant-path")?.value?.trim();
    const status = $("grant-status");
    if (!path) {
      if (status) status.innerHTML = flash("Paste the folder path on this computer.", "bad");
      return;
    }
    try {
      if (status) status.innerHTML = flash("Granting…", "ok");
      const result = await api("/api/allowlist", { method: "POST", body: JSON.stringify({ path }) });
      catalog.writePolicy = { ...catalog.writePolicy, allowedDirectories: result.allowedDirectories };
      const id = threadIdFromHash() ?? currentThread?.id;
      if (id) {
        currentThread = await api(`/api/chats/${encodeURIComponent(id)}/workspace`, {
          method: "POST",
          body: JSON.stringify({ path: result.granted ?? path }),
        });
        upsertThread(currentThread);
      }
      if (status) status.innerHTML = flash(`Granted ${result.granted ?? path}. Chat will use it as cwd.`, "ok");
      $("grant-card")?.classList.add("hidden");
    } catch (error) {
      if (status) status.innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const approval = target.getAttribute("data-approval");
  if (approval === "approve" || approval === "reject") {
    const id = threadIdFromHash() ?? currentThread?.id;
    if (!id) return;
    const comment = $("approval-comment")?.value?.trim() || undefined;
    try {
      const thread = await api(`/api/chats/${encodeURIComponent(id)}/approval`, {
        method: "POST",
        body: JSON.stringify({ decision: approval, comment }),
      });
      currentThread = thread;
      ensureChatLayout();
    } catch (error) {
      const list = $("thread-messages");
      if (list) list.insertAdjacentHTML("beforeend", flash(error.message, "bad"));
    }
    return;
  }
  const chatAction = target.getAttribute("data-chat-action");
  if (chatAction) {
    if (chatAction === "open_settings") {
      const payload = JSON.parse(target.getAttribute("data-payload") || "{}");
      location.hash = `#${payload.page || "backends"}`;
      return;
    }
    try {
      const payload = JSON.parse(target.getAttribute("data-payload") || "{}");
      const result = await api("/api/chat/actions", {
        method: "POST",
        body: JSON.stringify({
          threadId: threadIdFromHash() ?? currentThread?.id,
          action: chatAction,
          payload,
        }),
      });
      if (result && result.id && result.messages) {
        currentThread = result;
        ensureChatLayout();
      }
      if (chatAction === "add_allowed_dir") {
        await loadCatalog();
      }
    } catch (error) {
      const list = $("thread-messages");
      if (list) list.insertAdjacentHTML("beforeend", flash(error.message, "bad"));
    }
    return;
  }
  if (target.closest?.("[data-update-check]")) {
    updateBusy = true;
    updatePick = null;
    updateFlash = "Asking GitHub…";
    updateFlashKind = "ok";
    renderUpdates();
    try {
      updateCheck = await api("/api/updates");
      updateFlash = updateCheck.message;
      updateFlashKind = "ok";
    } catch (error) {
      updateCheck = null;
      updateFlash = error.message;
      updateFlashKind = "bad";
    } finally {
      updateBusy = false;
      if (pageId() === "updates") renderUpdates();
    }
    return;
  }
  if (target.getAttribute?.("data-update-pick")) {
    if (!updateCheck) return;
    updatePick = target.getAttribute("data-update-pick");
    renderUpdates();
    return;
  }
  if (target.closest?.("[data-update-cancel]")) {
    updatePick = null;
    renderUpdates();
    return;
  }
  if (target.closest?.("[data-update-confirm]")) {
    const choice = updatePick;
    if (!choice) return;
    updateBusy = true;
    updateFlash = "Saving the GitHub file on your computer…";
    updateFlashKind = "ok";
    renderUpdates();
    try {
      const result = await api("/api/updates/apply", {
        method: "POST",
        body: JSON.stringify({ choice, which: choice, confirm: true, confirmed: true }),
      });
      updatePick = null;
      updateFlash = result.message;
      updateFlashKind = "ok";
    } catch (error) {
      updateFlash = error.message;
      updateFlashKind = "bad";
    } finally {
      updateBusy = false;
      if (pageId() === "updates") renderUpdates();
    }
    return;
  }
  if (target.id === "reload-env") {
    try {
      const result = await api("/api/env/reload", { method: "POST", body: "{}" });
      catalog = result.catalog ?? catalog;
      await renderBackends();
      $("backends-status").innerHTML = flash("Reloaded .env and local secrets.", "ok");
    } catch (error) {
      $("backends-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  if (target.id === "validate-config") {
    const yaml = document.querySelector("textarea[name=yaml]")?.value ?? "";
    try {
      await api("/api/config/validate", { method: "POST", body: JSON.stringify({ yaml }) });
      renderConfig({ yaml, message: "Valid.", kind: "ok" });
    } catch (error) {
      renderConfig({ yaml, message: error.message, kind: "bad" });
    }
  }
  const remove = target.getAttribute("data-remove");
  if (remove) {
    try {
      const result = await api(`/api/allowlist?path=${encodeURIComponent(remove)}`, { method: "DELETE" });
      catalog.writePolicy = { ...catalog.writePolicy, allowedDirectories: result.allowedDirectories };
      renderAllowlist();
    } catch (error) {
      $("allow-status").innerHTML = flash(error.message, "bad");
    }
  }
});

$("main").addEventListener("input", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLInputElement)) return;
  if (target.id === "lateinfer-hub") {
    lateInferHubId = target.value;
    const list = $("lateinfer-hub-list");
    if (list) {
      for (const option of list.querySelectorAll("[data-hub-id]")) {
        const match = option.getAttribute("data-hub-id") === target.value.trim();
        option.classList.toggle("is-selected", match);
        option.setAttribute("aria-selected", match ? "true" : "false");
      }
    }
    return;
  }
  if (target.id === "llamacpp-gguf-search") {
    llamaGgufSearch = target.value;
    if (llamaGgufSearchTimer) clearTimeout(llamaGgufSearchTimer);
    llamaGgufSearchTimer = setTimeout(() => {
      void loadLlamaGgufCatalog(llamaGgufSearch);
    }, 350);
    return;
  }
  if (target.id === "llamacpp-gguf") {
    llamaGgufPath = target.value;
    return;
  }
  if (target.id === "llamacpp-hub-id") {
    llamaGgufSelectedId = target.value.trim();
    return;
  }
  if (target.id !== "lateinfer-hub-search") return;
  lateInferHubSearch = target.value;
  refreshLateInferHubList();
  if (lateInferHubSearchTimer) clearTimeout(lateInferHubSearchTimer);
  lateInferHubSearchTimer = setTimeout(() => {
    void loadLateInferHubModels(lateInferHubSearch);
  }, 400);
});

window.addEventListener("hashchange", () => {
  render().catch((error) => {
    if (pageId() === "local-models") {
      applyLocalModelsDisconnect(error);
      return;
    }
    $("main").innerHTML = flash(error.message, "bad");
  });
});

async function boot() {
  if (!token) {
    showGate("Open this GUI from the URL printed by npm run gui (it includes ?token=). Without a token, API calls cannot start.");
    return;
  }
  try {
    await loadSession();
    showShell();
    await loadCatalog();
    await loadThreads();
    connectEvents();
    await render();
    void api("/api/updates")
      .then((data) => {
        updateCheck = data;
        if (data?.updateLate || data?.updateOrchestrator || data?.bothNewer) {
          updateFlash = data.message || "GitHub has a newer file. Open Settings → Updates.";
          updateFlashKind = "ok";
        }
      })
      .catch(() => {
        /* start check must not block chat or MCP */
      });
  } catch {
    sessionStorage.removeItem(TOKEN_KEY);
    token = "";
    showGate("Token rejected. Copy it from the terminal that started the GUI.");
  }
}

applyTheme(readStoredTheme());
boot();
