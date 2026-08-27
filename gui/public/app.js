const TOKEN_KEY = "orchestrator.gui.token";

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
let runs = [];
let localModels = null;
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
      /NetworkError|Failed to fetch|Load failed|network/i.test(raw)
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

function pill(ready, reason) {
  if (ready) return `<span class="pill ok">ready</span>`;
  const label = /vLLM not running/i.test(reason ?? "") ? "not ready" : "missing";
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
      const agents = (t.agents ?? []).filter((a) => a && a !== "user" && a !== "orchestrator").slice(0, 3).join(" · ");
      return `<button type="button" class="thread-item ${t.id === active ? "active" : ""}" data-thread="${escapeHtml(t.id)}" role="listitem">
        <span>${escapeHtml(t.title || "New chat")}</span>
        ${agents ? `<small>${escapeHtml(agents)}</small>` : ""}
      </button>`;
    })
    .join("");
}

function pinOptions(selected) {
  const extras = (catalog.backends ?? [])
    .filter((b) => !["cursor-local", "cursor-cloud", "gemini"].includes(b.id))
    .map((b) => `<option value="${escapeHtml(b.id)}" ${selected === b.id ? "selected" : ""}>${escapeHtml(b.id)}</option>`)
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
        <p class="hint">Auto | Debate | Single chooses speakers. Implement/install requires Approve before writes or host installs; Q&A and debate text stay unblocked. Repo writes go to Cursor local inside the allowlist.</p>
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
        <h2>Default cwd</h2>
        <p class="mono">${escapeHtml(policy.defaultCwd ?? "")}</p>
        <p class="muted">Used when chat omits cwd. Must sit inside the allowlist.</p>
      </article>
      <article class="card">
        <h2>Allowed directories</h2>
        <p>${(policy.allowedDirectories ?? []).length} granted</p>
        <p class="muted">Local Cursor agents cannot write elsewhere.</p>
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
      const modelForm = isGemini
        ? `<form class="backend-model-form" data-backend="${escapeHtml(b.id)}">
            <label class="field">Model (one id, not a list)
              <input name="model" list="gemini-models-${escapeHtml(b.id)}" value="${escapeHtml(b.model ?? "gemini-3.6-flash")}" required placeholder="gemini-3.6-flash" autocomplete="off" />
            </label>
            <datalist id="gemini-models-${escapeHtml(b.id)}">${geminiModels.map((m) => `<option value="${escapeHtml(m)}"></option>`).join("")}</datalist>
            <p class="muted">Google OpenAI-compat expects a bare id such as gemini-3.6-flash (not models/…, not 1.5/2.0). Datalist is live ListModels when the key works, else the 2026 catalog.</p>
            <div class="actions"><button type="submit">Save model</button></div>
          </form>`
        : "";
      return `
      <article class="card">
        <div class="row">
          <h3 class="mono">${escapeHtml(b.id)}</h3>
          ${pill(b.ready, b.reason)}
          ${writePill(b.writesLocalFiles)}
        </div>
        <p class="muted">${escapeHtml(b.type)}${b.runtime ? ` · ${escapeHtml(b.runtime)}` : ""}${b.model ? ` · ${escapeHtml(b.model)}` : ""}</p>
        ${b.baseUrl ? `<p class="mono muted">${escapeHtml(b.baseUrl)}</p>` : ""}
        <p>${escapeHtml(b.reason ?? (b.ready ? "API key present (masked; never displayed)." : ""))}</p>
        <p class="muted">capabilities: ${escapeHtml((b.capabilities ?? []).join(", "))}</p>
        ${modelForm}
        ${keyForm}
      </article>`;
    })
    .join("");
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Backends</h1>
        <p>Ready vs missing keys. Secrets are never shown in full. Adding a key here or in <span class="mono">.env</span> takes effect after Reload env.</p>
      </div>
      <button type="button" id="reload-env" class="btn secondary">Reload env</button>
    </div>
    <div id="backends-status"></div>
    <div class="cards">${cards}</div>
    <form id="vllm-form" class="card" style="margin-top:0.85rem">
      <h2>Add vLLM backend</h2>
      <p class="muted">OpenAI-compatible HTTP API. Typical URL <span class="mono">http://127.0.0.1:8000/v1</span>. API key optional. Multiple endpoints = multiple backend ids / ports.</p>
      <label class="field">Backend id <input name="id" type="text" required placeholder="vllm-local" /></label>
      <label class="field">Base URL <input name="baseUrl" type="text" placeholder="http://127.0.0.1:8000/v1" /></label>
      <label class="field">Model <input name="model" type="text" required placeholder="meta-llama/Llama-3.1-8B-Instruct" /></label>
      <label class="field">Optional API key env name <input name="apiKeyEnv" type="text" placeholder="VLLM_API_KEY" /></label>
      <label class="field">Optional API key <input name="apiKey" type="password" autocomplete="off" /></label>
      <label class="field">Optional specialist id <input name="specialistId" type="text" placeholder="vllm-chat" /></label>
      <div class="actions"><button type="submit">Add vLLM backend</button></div>
      <div id="vllm-status"></div>
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
  const recIds = new Set((data.recommended ?? []).map((m) => m.id));
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
  const rows = (data.models ?? [])
    .map((m) => {
      const job = jobs.get(m.id);
      const rec = recIds.has(m.id) ? `<span class="pill ok">recommended</span>` : "";
      const fit = m.fits ? `<span class="pill ok">fits</span>` : `<span class="pill warn">no fit</span>`;
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
          <strong>${escapeHtml(m.name)}</strong>
          <div class="muted mono">${escapeHtml(m.id)}</div>
          ${rec} ${fit} ${dl} ${run}
          ${progress}
        </td>
        <td class="mono">${escapeHtml(m.quantization)} · ${escapeHtml(String(m.paramsB))}B<br/>~${escapeHtml(String(m.vramNeededMiB))} MiB w/ KV</td>
        <td>${escapeHtml(m.fitReason)}</td>
        <td>
          <div class="actions">
            ${actions}
          </div>
          <p class="muted">specialist <a href="#specialists">${escapeHtml(m.specialist)}</a> · backend <a href="#backends">${escapeHtml(m.backendId)}</a>${inst?.port ? ` · 127.0.0.1:${escapeHtml(String(inst.port))}` : ""}</p>
        </td>
      </tr>`;
    })
    .join("");
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
      ? flash(vllm.startJob?.error || vllm.lastError || "vLLM failed to start", "bad")
      : healthy
        ? flash(
            `Ready backends: ${instances
              .filter((row) => row.healthy || row.running)
              .map((row) => `${row.backendId} @ 127.0.0.1:${row.port ?? "?"}`)
              .join(", ")}. Chat Auto can use each ready vLLM as a debate speaker.`,
            "ok",
          )
        : "";
  $("main").innerHTML = `
    <div class="page-title">
      <div>
        <h1>Local models</h1>
        <p>Hardware-aware catalog. Start several models at once (each gets its own 127.0.0.1 port, container, and backend id). Stop or Remove from mix one without tearing the others down. Chat Auto uses every ready vLLM as a round-table speaker.</p>
      </div>
    </div>
    <div id="local-models-status">${statusFlash}</div>
    <div class="cards">
      <article class="card">
        <h2>Hardware</h2>
        <p>${escapeHtml(gpuLine)}</p>
        <p class="muted">Backend ${escapeHtml(backendLabel)} · ${escapeHtml(String(hw.deviceCount ?? accelerators.length ?? 0))} device(s) · ${escapeHtml(String(hw.totalVramMiB ?? hw.vramMiB ?? "?"))} MiB total · RAM ${escapeHtml(String(hw.ramMiB ?? "?"))} MiB · ${escapeHtml(String(hw.cpuCount ?? "?"))} CPUs${hw.constrained ? " · CPU fallback (no accelerator)" : ""}</p>
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
            ? escapeHtml(vllm.startJob?.lastLog || vllm.lastLog || vllm.modelId || "Waiting for GET /v1/models…")
            : healthy
              ? "Each row is a separate container/process. Stop removes that instance only; Remove from mix also deletes its backend from agents.config.yaml."
              : vllm.installed
                ? (preferDocker ? `GPU Docker runtime ready (${escapeHtml(String(preferredDocker ?? vllm.image ?? ""))}). Download a fitting model, then Start with Docker.` : "Installed. Download a fitting model, then Start.")
                : escapeHtml(vllm.installHint ?? "Install the serving stack that matches this GPU.")
        }</p>
        ${phase === "error" && (vllm.startJob?.lastLog || vllm.lastLog) ? `<pre class="muted" style="white-space:pre-wrap">${escapeHtml(String(vllm.startJob?.lastLog || vllm.lastLog).slice(-1500))}</pre>` : ""}
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
      <h2>Recommended for this machine</h2>
      <ul>${
        (data.recommended ?? [])
          .map(
            (m) =>
              `<li><span class="mono">${escapeHtml(m.id)}</span> · ${escapeHtml(m.name)} · ${escapeHtml(m.quantization)} · ~${escapeHtml(String(m.vramNeededMiB))} MiB${m.parallel > 1 ? ` · TP ${escapeHtml(String(m.parallel))}` : ""}</li>`,
          )
          .join("") || "<li class='muted'>Nothing in the catalog fits. Use a smaller model, or add GPU VRAM.</li>"
      }</ul>
      <p class="muted">Models dir: <span class="mono">${escapeHtml(data.modelsDir ?? "")}</span>${data.hfTokenSet ? " · HF token set" : " · optional HF_TOKEN for gated repos"}</p>
      <form id="hf-token-form" class="secret-form" data-name="HF_TOKEN">
        <label class="field">HF_TOKEN (gated repos; stored locally, never shown in full)
          <input name="value" type="password" autocomplete="off" placeholder="${data.hfTokenSet ? "set — paste to replace" : "hf_…"}" />
        </label>
        <div class="actions"><button type="submit">Save HF token</button></div>
      </form>
    </div>
    <div class="card" style="margin-top:0.85rem">
      <h2>Catalog</h2>
      <table>
        <thead><tr><th>Model</th><th>Size / VRAM</th><th>Fit</th><th></th></tr></thead>
        <tbody>${rows || `<tr><td colspan="4" class="muted">Catalog unavailable.</td></tr>`}</tbody>
      </table>
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
    await api("/api/session");
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

$("main").addEventListener("change", (event) => {
  if (event.target?.id === "chat-thread-pick") {
    const id = event.target.value;
    if (id) location.hash = `#chat/${id}`;
  }
});

$("thread-list").addEventListener("click", (event) => {
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
        body: JSON.stringify({ message, pin, wait: false }),
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
    await api("/api/session");
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

boot();
