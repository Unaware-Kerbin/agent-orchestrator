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
  };
  return data;
}

/** Exact Late Settings URL from this GUI process (127.0.0.1 + bound port). Never a hardcoded GUI default. */
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
let sessionInfo = { mcpUrl: "", bind: "" };
let runs = [];
let localModelsQuery = "";
let localModels = null;
let localServers = { ollama: null, llamacpp: [], llamaServerBinary: null, ollamaBinary: null };
let selectedRunId = null;
let events = null;
let threads = [];
let currentThread = null;
let sending = false;
let vllmPollTimer = null;

function dockerLaunchFromSnapshot(data) {
  const snap = data ?? localModels ?? {};
  const hw = snap.hardware ?? {};
  const docker = snap.intelDocker ?? hw.intelDocker ?? {};
  const preferred = typeof docker.preferred === "string" ? docker.preferred : docker.preferred?.ref;
  const backend = hw.primaryBackend;
  return {
    preferDocker: snap.preferredRuntime === "docker" || Boolean(preferred && backend === "intel-xpu"),
    image: preferred,
  };
}

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
        ? "Lost connection to the orchestrator GUI (127.0.0.1). Stay on Local models while vLLM starts (often several minutes). If you opened this page without ?token=, paste the token from the terminal."
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
  return ["overview", "specialists", "backends", "runs", "dispatch", "local-models", "allowlist", "config"];
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

function pinOptions(selected) {
  const extras = (catalog.backends ?? [])
    .filter((b) => !["cursor-local", "cursor-cloud", "gemini"].includes(b.id))
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
        <h2>Local vLLM</h2>
        <p>${runtime.vllm?.running ? `running · ${escapeHtml(runtime.vllm.modelId ?? "")} · 127.0.0.1:${escapeHtml(String(runtime.vllm.port ?? ""))}` : "stopped"}</p>
        <p class="muted">Bound to localhost only. Cloud agents cannot reach it. Auto prefers it for drafts when it is running.</p>
      </article>
      <article class="card">
        <h2>Ollama</h2>
        <p>${runtime.ollama?.running ? `running · ${escapeHtml(runtime.ollama.model ?? "")}` : "not running"}</p>
        <p class="mono muted">${escapeHtml(runtime.ollama?.baseUrl ?? "http://127.0.0.1:11434/v1")}</p>
        <p class="muted">${escapeHtml(runtime.ollama?.reason ?? "Detect a daemon on 127.0.0.1:11434, then Register on Local models.")}</p>
      </article>
      <article class="card">
        <h2>llama.cpp</h2>
        <p>${
          (runtime.llamacpp ?? []).some((row) => row.running)
            ? (runtime.llamacpp ?? [])
                .filter((row) => row.running)
                .map((row) => `${row.id ?? "llamacpp"} · ${row.model ?? ""}`)
                .join("; ")
            : "no loopback llama-server registered"
        }</p>
        <p class="muted">Connect to a user-started <span class="mono">llama-server</span> on 127.0.0.1. GGUF weights are not the vLLM catalog.</p>
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
  const cards = (catalog.backends ?? [])
    .map((b) => {
      const names = b.secretNames ?? [];
      const primary = names[0];
      const keySet = names.some((n) => secretSet.get(n));
      const keyForm = primary && b.needsKey
        ? `<form class="secret-form" data-name="${escapeHtml(primary)}">
            <label class="field">Set ${escapeHtml(primary)} (stored locally, never shown in full)
              <input name="value" type="password" autocomplete="off" placeholder="${keySet ? "set — paste to replace" : "paste key"}" />
            </label>
            <div class="actions"><button type="submit">Save key</button></div>
            <div class="secret-status"></div>
          </form>`
        : "";
      const geminiModels = b.modelChoices ?? [];
      const isGemini =
        b.id === "gemini" ||
        /generativelanguage\.googleapis\.com/i.test(b.baseUrl ?? "") ||
        geminiModels.length > 0;
      const localCompat = b.type === "ollama" || b.type === "llamacpp";
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
              <input name="model" list="local-models-${escapeHtml(b.id)}" value="${escapeHtml(b.model ?? "")}" required placeholder="${b.type === "ollama" ? "llama3.1" : "local"}" autocomplete="off" />
            </label>
            <datalist id="local-models-${escapeHtml(b.id)}">${localModelsList.map((m) => `<option value="${escapeHtml(m)}"></option>`).join("")}</datalist>
            <p class="muted">${b.type === "ollama" ? "Ollama tag from /api/tags when the daemon is up." : "Model id llama-server reports on /v1/models (often the GGUF filename or --alias)."}</p>
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
    })
    .join("");
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Backends</h1>
        <p>Ready vs missing keys. Nicknames and logos appear in chat. Logos stay in <span class="mono">.orchestrator/logos</span> (not git). Secrets are never shown in full. Adding a key here or in <span class="mono">.env</span> takes effect after Reload env.</p>
      </div>
      <button type="button" id="reload-env" class="btn secondary">Reload env</button>
    </div>
    <div id="backends-status"></div>
    <div class="cards">${cards}</div>
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
    <form id="vllm-form" class="card" style="margin-top:0.85rem">
      <h2>Add vLLM backend</h2>
      <p class="muted">OpenAI-compatible HTTP API. Typical URL <span class="mono">http://127.0.0.1:8000/v1</span>. API key optional. Multiple endpoints = multiple backend ids / ports.</p>
      <label class="field">Backend id <input name="id" type="text" required placeholder="vllm-local" /></label>
      <label class="field">Base URL <input name="baseUrl" type="text" placeholder="http://127.0.0.1:8000/v1" /></label>
      <label class="field">Model <input name="model" type="text" required placeholder="meta-llama/Llama-3.1-8B-Instruct" /></label>
      <label class="field">Optional nickname <input name="nickname" type="text" maxlength="48" placeholder="Arc Qwen" autocomplete="off" /></label>
      <label class="field">Optional API key env name <input name="apiKeyEnv" type="text" placeholder="VLLM_API_KEY" /></label>
      <label class="field">Optional API key <input name="apiKey" type="password" autocomplete="off" /></label>
      <label class="field">Optional specialist id <input name="specialistId" type="text" placeholder="vllm-chat" /></label>
      <div class="actions"><button type="submit">Add vLLM backend</button></div>
      <div id="vllm-status"></div>
    </form>
    <form id="ollama-form" class="card" style="margin-top:0.85rem">
      <h2>Add Ollama backend</h2>
      <p class="muted">Loopback OpenAI-compat API. Typical URL <span class="mono">http://127.0.0.1:11434/v1</span>. Install and run Ollama yourself; this GUI only detects and registers it. Non-loopback hosts are refused.</p>
      <label class="field">Backend id <input name="id" type="text" required placeholder="ollama" /></label>
      <label class="field">Base URL <input name="baseUrl" type="text" placeholder="http://127.0.0.1:11434/v1" /></label>
      <label class="field">Model <input name="model" type="text" required placeholder="llama3.1" /></label>
      <label class="field">Optional nickname <input name="nickname" type="text" maxlength="48" placeholder="Ollama" autocomplete="off" /></label>
      <label class="field">Optional specialist id <input name="specialistId" type="text" placeholder="ollama-chat" /></label>
      <div class="actions"><button type="submit">Add Ollama backend</button></div>
      <div id="ollama-status"></div>
    </form>
    <form id="llamacpp-form" class="card" style="margin-top:0.85rem">
      <h2>Add llama.cpp backend</h2>
      <p class="muted">Connect to a user-started <span class="mono">llama-server</span> OpenAI API. Example: <span class="mono">llama-server -m model.gguf --host 127.0.0.1 --port 8080</span>. GGUF files are not vLLM Hugging Face snapshots. Bind 127.0.0.1 only.</p>
      <label class="field">Backend id <input name="id" type="text" required placeholder="llamacpp" /></label>
      <label class="field">Base URL <input name="baseUrl" type="text" placeholder="http://127.0.0.1:8080/v1" /></label>
      <label class="field">Model <input name="model" type="text" required placeholder="local" /></label>
      <label class="field">Optional nickname <input name="nickname" type="text" maxlength="48" placeholder="llama.cpp" autocomplete="off" /></label>
      <label class="field">Optional specialist id <input name="specialistId" type="text" placeholder="llamacpp-chat" /></label>
      <div class="actions"><button type="submit">Add llama.cpp backend</button></div>
      <div id="llamacpp-status"></div>
    </form>
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
  const vllm = data.vllm ?? {};
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
  const docker = data.intelDocker ?? hw.intelDocker ?? {};
  const dockerImages = docker.images ?? [];
  const preferredDocker = docker.preferred?.ref ?? docker.preferred;
  const preferDocker = data.preferredRuntime === "docker" || Boolean(preferredDocker && backend === "intel-xpu");
  const recIds = new Set(
    (data.recommended ?? []).filter((m) => m.fits && m.newest).map((m) => m.id),
  );
  const jobs = new Map((data.jobs ?? []).map((j) => [j.modelId, j]));
  const phase = vllm.phase ?? (vllm.healthy ? "running" : vllm.running ? "starting" : "idle");
  const instances = Array.isArray(vllm.instances) && vllm.instances.length
    ? vllm.instances
    : vllm.running || vllm.healthy || phase === "starting"
      ? [vllm]
      : [];
  const instanceFor = (m) =>
    instances.find((row) => row.modelId === m.id || row.hfRepo === m.hfRepo);
  const starting = instances.some((row) => row.phase === "starting") || phase === "starting";
  const healthy = instances.some((row) => row.healthy || (row.phase === "running" && row.running));
  syncVllmPoll(vllm);
  const catalogRow = (m) => {
      const job = jobs.get(m.id);
      const rec = recIds.has(m.id) ? `<span class="pill ok">recommended</span>` : "";
      const newest = m.newest ? `<span class="pill">newest</span>` : `<span class="pill">previous</span>`;
      const gated = m.gated ? `<span class="pill warn">gated</span>` : "";
      const fit =
        m.fitKind === "needs_tp" || m.parallel > 1
          ? `<span class="pill warn">needs TP</span>`
          : m.fits
            ? `<span class="pill ok">fits</span>`
            : m.fitKind === "incompatible"
              ? `<span class="pill warn">incompatible</span>`
              : `<span class="pill warn">too big</span>`;
      const dl = m.downloaded ? `<span class="pill ok">downloaded</span>` : `<span class="pill">not downloaded</span>`;
      const inst = instanceFor(m);
      const thisStarting = Boolean(inst && inst.phase === "starting");
      const thisRunning = Boolean(m.running || inst?.healthy || inst?.running);
      const run = thisRunning
        ? `<span class="pill ok">vLLM running</span>`
        : thisStarting
          ? `<span class="pill warn">starting</span>`
          : "";
      const progress =
        job && (job.status === "running" || job.status === "queued")
          ? `<div class="progress" title="${escapeHtml(job.message ?? "")}"><span style="width:${Number(job.percent) || 0}%"></span></div>
             <p class="muted">${escapeHtml(job.message ?? job.status)}</p>`
          : job?.status === "error"
            ? `<p class="error">${escapeHtml(job.error || job.message || "error")}</p>`
            : "";
      const startLabel = preferDocker ? "Start with Docker" : "Start";
      let actions = `<button type="button" class="btn secondary" data-download="${escapeHtml(m.id)}">Download</button>`;
      if (thisRunning || thisStarting) {
        actions += `<button type="button" class="btn danger" data-vllm-stop="${escapeHtml(m.id)}">Stop</button>`;
        actions += `<button type="button" class="btn danger" data-vllm-remove="${escapeHtml(m.id)}">Remove from mix</button>`;
      } else {
        actions += `<button type="button" class="btn" data-vllm-start="${escapeHtml(m.id)}" ${m.downloaded && !thisStarting ? "" : "disabled"}>${thisStarting ? "Starting…" : startLabel}</button>`;
      }
      if (m.downloaded) {
        actions += `<button type="button" class="btn danger" data-vllm-delete="${escapeHtml(m.id)}">Delete weights</button>`;
      }
      return `<tr>
        <td>
          <div class="row backend-head">
            ${avatarMarkup(m.backendId)}
            <div>
              <strong>${escapeHtml(m.name)}</strong>
              ${backendEntry(m.backendId)?.nickname ? `<div class="muted">${escapeHtml(backendDisplayName(m.backendId))}</div>` : ""}
              <div class="muted mono">${escapeHtml(m.id)}</div>
            </div>
          </div>
          ${rec} ${newest} ${gated} ${fit} ${dl} ${run}
          ${m.gated && !data.hfTokenSet ? `<p class="muted">Gated: accept the Hugging Face license, then save a token above.</p>` : ""}
          ${progress}
        </td>
        <td class="mono">${escapeHtml(m.quantization)} · ${escapeHtml(String(m.paramsB))}B<br/>~${escapeHtml(String(m.vramNeededMiB))} MiB w/ KV${m.parallel > 1 ? ` · TP ${escapeHtml(String(m.parallel))}` : ""}</td>
        <td>${escapeHtml(m.fitReason)}</td>
        <td>
          <div class="actions">
            ${actions}
          </div>
          <p class="muted">specialist <a href="#specialists">${escapeHtml(m.specialist)}</a> · backend <a href="#backends">${escapeHtml(m.backendId)}</a>${inst?.port ? ` · 127.0.0.1:${escapeHtml(String(inst.port))}` : ""}</p>
        </td>
      </tr>`;
  };
  const matchesQuery = (m) => {
    const q = localModelsQuery.trim().toLowerCase();
    if (!q) return true;
    const hay = `${m.id ?? ""} ${m.name ?? ""} ${m.hfRepo ?? ""} ${m.quantization ?? ""}`.toLowerCase();
    return hay.includes(q);
  };
  const downloadedModels = (data.models ?? [])
    .filter((m) => m.downloaded)
    .filter(matchesQuery)
    .slice()
    .sort((a, b) => {
      const aRun = a.running ? 0 : 1;
      const bRun = b.running ? 0 : 1;
      if (aRun !== bRun) return aRun - bRun;
      return String(a.name).localeCompare(String(b.name));
    });
  const notDownloadedModels = (data.models ?? []).filter((m) => !m.downloaded).filter(matchesQuery);
  const recommendedModels = (data.recommended ?? []).filter(matchesQuery);
  const catalogTable = (models, empty) =>
    `<table>
        <thead><tr><th>Model</th><th>Size / VRAM</th><th>Fit</th><th></th></tr></thead>
        <tbody>${models.map(catalogRow).join("") || `<tr><td colspan="4" class="muted">${empty}</td></tr>`}</tbody>
      </table>`;
  const instanceLines = instances
    .map((row) => {
      const label = row.phase === "starting" ? "starting" : row.healthy || row.running ? "running" : row.phase || "idle";
      const rt = row.runtime === "docker" ? "docker" : row.runtime === "host" ? "host" : "";
      const id = row.containerId ? String(row.containerId).slice(0, 12) : row.containerName ?? (row.pid ? `pid ${row.pid}` : "");
      return `<li><span class="mono">${escapeHtml(row.backendId ?? "")}</span> · ${escapeHtml(label)} ${escapeHtml(rt)} · 127.0.0.1:${escapeHtml(String(row.port ?? "?"))} · ${escapeHtml(row.modelId ?? row.hfRepo ?? "")}${row.image ? ` · ${escapeHtml(String(row.image))}` : ""}${id ? ` · ${escapeHtml(id)}` : ""}</li>`;
    })
    .join("");
  const statusFlash = starting
    ? flash("Starting… Intel containers often take several minutes. Stay on this page. Other models can still be started.", "ok")
    : phase === "error" && !healthy
      ? flash(vllmErrorDetail(vllm), "bad")
      : healthy
        ? flash(
            `Ready backends: ${instances
              .filter((row) => row.healthy || row.running)
              .map((row) => `${row.backendId} @ 127.0.0.1:${row.port ?? "?"}`)
              .join(", ")}. Chat Auto can use each ready vLLM as a debate speaker.`,
            "ok",
          )
        : "";
  const hfTokenSet = Boolean(data.hfTokenSet);
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Local models</h1>
        <p>Hardware-aware catalog for vLLM (Hugging Face snapshots). Downloaded snapshots are listed first so you can Start / Stop them without scrolling the full catalog. Ollama and llama.cpp are separate loopback OpenAI APIs — register them below or on Backends. Start several vLLM models at once (each gets its own 127.0.0.1 port, container, and backend id). Chat Auto uses each <strong>running</strong> vLLM as a speaker — leftover YAML rows that are not serving do not join.</p>
      </div>
    </div>
    <div id="local-models-status">${statusFlash}</div>
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
    <div class="cards">
      <article class="card">
        <h2>Hardware</h2>
        <p>${escapeHtml(gpuLine)}</p>
        <p class="muted">Backend ${escapeHtml(backendLabel)} · ${escapeHtml(String(hw.deviceCount ?? accelerators.length ?? 0))} device(s) · ${escapeHtml(String(hw.totalVramMiB ?? hw.vramMiB ?? "?"))} MiB total · RAM ${escapeHtml(String(hw.ramMiB ?? "?"))} MiB · ${escapeHtml(String(hw.cpuCount ?? "?"))} CPUs${hw.constrained ? " · CPU fallback (no accelerator)" : ""}${(hw.deviceCount ?? accelerators.length ?? 0) > 1 ? " · tensor parallel available for models that need more than one card" : ""}</p>
        ${(hw.notes ?? []).map((n) => `<p class="muted">${escapeHtml(n)}</p>`).join("")}
      </article>
      <article class="card">
        <h2>Local servers</h2>
        ${
          instanceLines
            ? `<ul>${instanceLines}</ul>`
            : `<p>stopped</p>`
        }
        <p class="muted">${
          starting
            ? escapeHtml(vllmStartingDetail(vllm) || vllm.modelId || "Waiting for GET /v1/models…")
            : healthy
              ? "Each row is a separate container/process. Stop removes that instance only; Remove from mix also deletes its backend from agents.config.yaml."
              : vllm.installed
                ? (preferDocker ? `GPU Docker runtime ready (${escapeHtml(String(preferredDocker ?? vllm.image ?? ""))}). Download a fitting model, then Start with Docker.` : "Installed. Download a fitting model, then Start.")
                : escapeHtml(vllm.installHint ?? "Install the serving stack that matches this GPU.")
        }</p>
        ${phase === "error" && (vllm.startJob?.lastLog || vllm.lastLog) ? `<pre class="muted" style="white-space:pre-wrap">${escapeHtml(String(vllm.startJob?.lastLog || vllm.lastLog).slice(-1500))}</pre>` : ""}
      </article>
      <article class="card">
        <h2>Ollama</h2>
        <p>${localServers.ollama?.running ? `<span class="pill ok">running</span>` : `<span class="pill warn">not running</span>`} ${escapeHtml(localServers.ollama?.origin ?? "http://127.0.0.1:11434")}</p>
        <p class="muted">${escapeHtml(localServers.ollama?.reason ?? "Probe loopback 11434. Install Ollama yourself — this app does not download it.")}${
          localServers.ollamaBinary ? ` Found <span class="mono">${escapeHtml(localServers.ollamaBinary)}</span> on PATH.` : ""
        }</p>
        ${
          (localServers.ollama?.models ?? []).length
            ? `<ul>${localServers.ollama.models.map((m) => `<li class="mono">${escapeHtml(m)}</li>`).join("")}</ul>`
            : `<p class="muted">No tags yet. Run <span class="mono">ollama pull llama3.1</span> then Register.</p>`
        }
        <div class="actions">
          <button type="button" class="btn" id="ollama-connect" ${localServers.ollama?.running ? "" : "disabled"}>Register Ollama backend</button>
        </div>
      </article>
      <article class="card">
        <h2>llama.cpp</h2>
        ${
          (localServers.llamacpp ?? []).length
            ? `<ul>${localServers.llamacpp
                .map(
                  (row) =>
                    `<li><span class="mono">${escapeHtml(row.id ?? "llamacpp")}</span> · ${row.running ? "ready" : "not running"} · ${escapeHtml(row.baseUrl ?? "")} · ${escapeHtml(row.model || (row.models ?? [])[0] || "")}</li>`,
                )
                .join("")}</ul>`
            : `<p>No llama.cpp backend configured.</p>`
        }
        <p class="muted">${
          localServers.llamaServerBinary
            ? `Found <span class="mono">llama-server</span> at ${escapeHtml(localServers.llamaServerBinary)}. Start it yourself with <span class="mono">--host 127.0.0.1</span>, then add the backend on Settings → Backends. This app does not download GGUF files.`
            : `Start <span class="mono">llama-server -m model.gguf --host 127.0.0.1 --port 8080</span> then add a backend. GGUF is not the vLLM Hugging Face catalog.`
        }</p>
      </article>
      <article class="card">
        <h2>Cursor cloud</h2>
        <p>${cloud.ready ? "CURSOR_API_KEY present" : "CURSOR_API_KEY missing"}</p>
        <p class="muted">${escapeHtml(cloud.reason ?? "")}</p>
        <p class="muted">Auto debate can include cloud as a speaker; it never calls localhost vLLM.</p>
      </article>
      <article class="card">
        <h2>GPU serving images</h2>
        ${
          dockerImages.length
            ? `<ul>${dockerImages
                .map((img) => {
                  const ref = img.ref ?? `${img.repository ?? ""}:${img.tag ?? ""}`;
                  const preferred = ref === preferredDocker ? ` <span class="pill ok">default</span>` : "";
                  return `<li><span class="mono">${escapeHtml(ref)}</span>${preferred}<div class="muted">${escapeHtml(img.tag ?? "")}${img.size ? ` · ${escapeHtml(img.size)}` : ""}${img.id ? ` · ${escapeHtml(img.id)}` : ""}</div></li>`;
                })
                .join("")}</ul>
               <p class="muted">${preferDocker ? "Start with Docker is the default when a vendor image matches this GPU." : "Available as a launch backend."}</p>`
            : `<p>${escapeHtml(docker.error ?? "No local GPU serving images found.")}</p>
               <p class="muted">Optional: pull a vLLM image that matches your GPU vendor, stay in the docker group, and keep publish on 127.0.0.1 only.</p>`
        }
      </article>
    </div>
    <div class="card" style="margin-top:0.85rem">
      <h2>Find a model</h2>
      <label class="field" for="model-search">Search catalog (name, id, Hugging Face repo, quantization)
        <input id="model-search" type="search" value="${escapeHtml(localModelsQuery)}" placeholder="qwen, gemma, 7b…" autocomplete="off" />
      </label>
      <p class="muted">${localModelsQuery.trim() ? `Showing matches in recommended, downloaded, and not downloaded.` : "Empty search shows every catalog row."}</p>
    </div>
    <div class="card" style="margin-top:0.85rem">
      <h2>Recommended for this machine</h2>
      <ul>${
        recommendedModels
          .map((m) => {
            const fit =
              m.fitKind === "needs_tp" || m.parallel > 1
                ? "needs TP"
                : m.fits
                  ? "fits"
                  : m.fitKind === "incompatible"
                    ? "incompatible"
                    : "too big";
            const gen = m.newest ? "newest" : "previous";
            return `<li><span class="mono">${escapeHtml(m.id)}</span> · ${escapeHtml(m.name)} · ${escapeHtml(m.quantization)} · ${escapeHtml(fit)} · ${escapeHtml(gen)} · ~${escapeHtml(String(m.vramNeededMiB))} MiB${m.parallel > 1 ? ` · TP ${escapeHtml(String(m.parallel))}` : ""}</li>`;
          })
          .join("") || `<li class='muted'>${localModelsQuery.trim() ? "No recommended models match that search." : "Nothing in the catalog fits. Use a smaller model, or add GPU VRAM."}</li>`
      }</ul>
      <p class="muted">Every catalog snapshot that fits this hardware (no top-N cap), newest generation per family (Gemma 4 over Gemma 2/3, Qwen3.8 over 2.5). Older gens stay in the table below for download. Models dir: <span class="mono">${escapeHtml(data.modelsDir ?? "")}</span>${hfTokenSet ? " · HF token configured" : " · save an HF token above for gated repos"}</p>
    </div>
    <div class="card" style="margin-top:0.85rem">
      <h2>On this computer</h2>
      <p class="muted">Downloaded weights. Running models are first. Use Stop / Remove from mix here — Chat Auto only talks to a model after Start succeeds.</p>
      ${catalogTable(downloadedModels, localModelsQuery.trim() ? "No downloaded models match that search." : "Nothing downloaded yet. Download a snapshot below.")}
    </div>
    <div class="card" style="margin-top:0.85rem">
      <h2>Not downloaded</h2>
      <p class="muted">Full catalog, including older generations. Gated repos show a gated pill; Gemma 4 is ungated Apache 2.0. Download, then it moves to On this computer.</p>
      ${catalogTable(notDownloadedModels, localModelsQuery.trim() ? "No catalog rows match that search." : "Every catalog snapshot is already on this computer.")}
    </div>
  `;
  const search = $("model-search");
  if (search instanceof HTMLInputElement) {
    search.value = localModelsQuery;
  }
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
    localServers = await api("/api/local-servers");
  } catch {
    localServers = { ollama: { running: false, models: [] }, llamacpp: [], llamaServerBinary: null, ollamaBinary: null };
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
    await loadLocalModels();
    await loadLocalServers();
    renderLocalModels();
  } else if (page === "allowlist") renderAllowlist();
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
    if (["overview", "specialists", "backends", "allowlist", "dispatch", "local-models"].includes(pageId())) {
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

document.addEventListener("input", (event) => {
  const el = event.target;
  if (!(el instanceof HTMLInputElement) || el.id !== "model-search") return;
  localModelsQuery = el.value;
  const pos = el.selectionStart;
  renderLocalModels();
  const next = $("model-search");
  if (next instanceof HTMLInputElement) {
    next.focus();
    if (typeof pos === "number") next.setSelectionRange(pos, pos);
  }
});

document.addEventListener("change", (event) => {
  const select = event.target?.closest?.(".theme-select");
  if (!select) return;
  applyTheme(select.value);
});

$("main").addEventListener("change", async (event) => {
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
    } else if (form.id === "vllm-form") {
      const specialistId = data.get("specialistId");
      const result = await api("/api/backends", {
        method: "POST",
        body: JSON.stringify({
          id: data.get("id"),
          type: "vllm",
          baseUrl: data.get("baseUrl") || undefined,
          model: data.get("model"),
          nickname: String(data.get("nickname") ?? "").trim() || undefined,
          apiKeyEnv: data.get("apiKeyEnv") || undefined,
          apiKey: data.get("apiKey") || undefined,
          specialist: specialistId
            ? { id: specialistId, description: `Local vLLM specialist (${data.get("id")})` }
            : undefined,
        }),
      });
      catalog = result.catalog ?? catalog;
      await renderBackends();
      $("vllm-status").innerHTML = flash(`Saved backend ${result.id}`, "ok");
    } else if (form.id === "ollama-form") {
      const specialistId = data.get("specialistId") || "ollama-chat";
      const result = await api("/api/backends", {
        method: "POST",
        body: JSON.stringify({
          id: data.get("id"),
          type: "ollama",
          baseUrl: data.get("baseUrl") || undefined,
          model: data.get("model"),
          nickname: String(data.get("nickname") ?? "").trim() || undefined,
          specialist: { id: specialistId, description: `Local Ollama specialist (${data.get("id")})` },
        }),
      });
      catalog = result.catalog ?? catalog;
      await renderBackends();
      $("ollama-status").innerHTML = flash(`Saved backend ${result.id}`, "ok");
    } else if (form.id === "llamacpp-form") {
      const specialistId = data.get("specialistId") || "llamacpp-chat";
      const result = await api("/api/backends", {
        method: "POST",
        body: JSON.stringify({
          id: data.get("id"),
          type: "llamacpp",
          baseUrl: data.get("baseUrl") || undefined,
          model: data.get("model"),
          nickname: String(data.get("nickname") ?? "").trim() || undefined,
          specialist: { id: specialistId, description: `Local llama.cpp specialist (${data.get("id")})` },
        }),
      });
      catalog = result.catalog ?? catalog;
      await renderBackends();
      $("llamacpp-status").innerHTML = flash(`Saved backend ${result.id}`, "ok");
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
              : form.id === "vllm-form"
                ? "vllm-status"
                : form.id === "ollama-form"
                  ? "ollama-status"
                  : form.id === "llamacpp-form"
                    ? "llamacpp-status"
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
      const result = await api("/api/secrets", {
        method: "DELETE",
        body: JSON.stringify({ name }),
      });
      catalog = result.catalog ?? catalog;
      await loadLocalModels();
      renderLocalModels();
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(`${name} cleared`, "ok");
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  if (target.id === "ollama-connect") {
    try {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Registering Ollama…", "ok");
      const result = await api("/api/ollama/connect", { method: "POST", body: JSON.stringify({}) });
      catalog = result.catalog ?? catalog;
      await loadLocalServers();
      await loadCatalog();
      renderLocalModels();
      if ($("local-models-status")) {
        $("local-models-status").innerHTML = flash(`Registered Ollama backend ${result.id} · ${result.model}`, "ok");
      }
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
  const downloadId = target.getAttribute("data-download");
  if (downloadId) {
    try {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Starting download…", "ok");
      await api("/api/local-models/download", {
        method: "POST",
        body: JSON.stringify({ modelId: downloadId }),
      });
      await loadLocalModels();
      renderLocalModels();
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const stopId = target.getAttribute("data-vllm-stop");
  if (stopId) {
    try {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Stopping vLLM…", "ok");
      const result = await api("/api/vllm/stop", {
        method: "POST",
        body: JSON.stringify(stopId === "1" ? { all: true } : { modelId: stopId }),
      });
      catalog = result.catalog ?? catalog;
      await loadLocalModels();
      renderLocalModels();
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const removeMixId = target.getAttribute("data-vllm-remove");
  if (removeMixId) {
    if (!window.confirm(`Stop this instance and remove backend ${removeMixId} from agents.config.yaml? Other running models stay up.`)) {
      return;
    }
    try {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Removing from mix…", "ok");
      const result = await api("/api/vllm/remove", {
        method: "POST",
        body: JSON.stringify({ modelId: removeMixId }),
      });
      catalog = result.catalog ?? catalog;
      await loadLocalModels();
      renderLocalModels();
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const deleteId = target.getAttribute("data-vllm-delete");
  if (deleteId) {
    if (!window.confirm(`Permanently delete downloaded weights for ${deleteId} from the models directory? This cannot be undone.`)) {
      return;
    }
    try {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash("Deleting weights…", "ok");
      const result = await api("/api/local-models/delete", {
        method: "POST",
        body: JSON.stringify({ modelId: deleteId, confirm: true }),
      });
      catalog = result.catalog ?? catalog;
      await loadLocalModels();
      renderLocalModels();
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
  }
  const startId = target.getAttribute("data-vllm-start");
  if (startId) {
    try {
      if ($("local-models-status")) {
        $("local-models-status").innerHTML = flash("Starting… Intel Docker can take several minutes. Stay on this page.", "ok");
      }
      const result = await api("/api/vllm/start", {
        method: "POST",
        body: JSON.stringify({
          modelId: startId,
          runtime: dockerLaunchFromSnapshot().preferDocker ? "docker" : undefined,
          image: dockerLaunchFromSnapshot().image,
        }),
        signal: AbortSignal.timeout(30_000),
      });
      if (localModels && result.vllm) localModels = { ...localModels, vllm: result.vllm };
      syncVllmPoll(result.vllm ?? result.status);
      await loadLocalModels();
      renderLocalModels();
    } catch (error) {
      if ($("local-models-status")) $("local-models-status").innerHTML = flash(error.message, "bad");
    }
    return;
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

window.addEventListener("hashchange", () => {
  render().catch((error) => {
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
  } catch {
    sessionStorage.removeItem(TOKEN_KEY);
    token = "";
    showGate("Token rejected. Copy it from the terminal that started the GUI.");
  }
}

applyTheme(readStoredTheme());
boot();
