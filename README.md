# Agent Orchestrator

MCP server and localhost GUI that routes work across **Cursor agents** and **external LLM backends** (OpenAI-compatible APIs, Anthropic, local vLLM, Ollama, or a custom HTTP agent).

```
Chat (GUI or MCP)
  → Auto router (control tools | single agent | multi-agent debate)
      → Cursor (local or cloud) — can edit allowlisted directories
      → External models — text only
      → Local model server on 127.0.0.1 — never public
```

## Quick start

Requires Node.js 22.13+.

```bash
npm install
cp .env.example .env
```

Put API keys in `.env` or in the GUI **Backends** page — never in `agents.config.yaml`. `apiKeyEnv` is the **variable name** (`GEMINI_API_KEY`), not the secret.

```bash
# GUI (loopback only)
npm run gui
```

Open the URL printed on stderr (`http://127.0.0.1:8787?token=…`). The session token is stored at `.orchestrator/gui.secret` (gitignored).

| Command | Purpose |
| --- | --- |
| `npm run gui` | Start the control plane on `127.0.0.1:8787` |
| `npm run gui:stop` | Stop that process |
| `npm run gui:restart` | Stop then start |
| `npm start` | Stdio MCP server |

If port 8787 is already in use, the GUI is already running — use `gui:stop` or open the existing token URL. Stopping a local model container does **not** stop the GUI.

Cursor: this repo includes [`.cursor/mcp.json`](.cursor/mcp.json). Reload MCP once after clone. `list_agents` re-reads env and GUI secrets without a full IDE restart.

## Chat

Home is a chat thread. The header (new chat, thread switcher, settings) and composer stay on screen; only messages scroll.

- **Auto** (default) — control tools for hardware/download/start; debate for plan/fix/review when two or more backends are ready; otherwise a single agent.
- **Debate** — round-table: each ready model speaks in turn (one bubble per speaker), then a closer synthesizes.
- **Single** / pin a backend — that backend only.

While a speaker runs, a **thinking** chip shows name, elapsed time, and phase so the UI does not look hung.

**Writes and installs wait for Approve.** Implement/install stays plan-only until you click **Approve** on the pending-actions card. After that, Cursor may write **only inside the write allowlist**. Host-wide installs (package managers, game engines, `sudo`) are called out and still wait. External models never edit files.

## Settings

| Page | What it does |
| --- | --- |
| Backends | Ready/not-ready, paste keys (masked), Gemini model id |
| Local models | Detect GPU VRAM, recommend weights that fit, download, start/stop/remove local servers |
| Allowlist | Directories Cursor may write to |
| Config | Edit `agents.config.yaml` (validated; no live keys) |
| Run workflow | Optional named pipelines |

Keys live in `.env` and `.orchestrator/secrets.env` (gitignored, mode `0600`). **Reload env** picks up a key added after start.

## Security

| Property | Behavior |
| --- | --- |
| Bind | GUI and local model HTTP bind **`127.0.0.1` only**. Non-loopback host exits. |
| Auth | GUI requires a Bearer token. Unauthenticated `/api/*` is 401. |
| Origin | Non-loopback `Host` / `Origin` is rejected. |
| Secrets | Never logged or shown in full. Not committed. |
| Writes | Realpath + allowlist; `..` and symlink escapes fail. |

Do not tunnel the GUI or vLLM. Cloud Cursor agents cannot reach localhost; the orchestrator passes **text** between local and cloud.

## Write allowlist

Default: this workspace (`WORKSPACE_CWD` / `workspace.cwd`). Add more via Settings → Allowlist or `add_allowed_dir`. Chat offers one-click add when you name an absolute path that is not listed.

## Local models (vendor-agnostic)

`list_hardware` probes **whatever accelerators are present** (NVIDIA CUDA, AMD ROCm, Intel XPU, or CPU if none). Recommendations use **measured VRAM**, not a single vendor. Missing NVIDIA is not treated as “CPU only” when another GPU exists.

A catalog model **fits** when estimated weights plus ~20% KV-cache headroom are ≤ per-GPU VRAM. Multi-GPU can use tensor parallel when a model misses one card but fits two.

Download snapshots into `.orchestrator/models` (gitignored, must stay on the allowlist). Gated Hugging Face repos need `HF_TOKEN` in env or the GUI — never in git.

`start_vllm` picks a serving stack from the detected backend:

- **CUDA** — host `vllm serve` when the CUDA wheel is installed
- **ROCm** — ROCm vLLM when present
- **XPU** — vendor Docker images if they are already local; otherwise a host XPU build
- **CPU** — not used as a serve path

The API is published on `127.0.0.1` only (ports 8000–8099). Start returns immediately (`202`); wait on the Local models page until `/v1/models` is healthy. The running server is registered as a backend automatically (dummy loopback token if the client requires Bearer — you do not copy a key from the container).

You can run **several models at once**. Each catalog id gets its own container, port, and backend (`vllm-<catalog-slug>`). **Stop** one instance; **Remove from mix** also drops that backend from YAML; **Delete weights** is a separate confirm.

```bash
pip install -r scripts/requirements-hf.txt   # downloads
# Then install the vLLM build that matches your GPU (CUDA, ROCm, or vendor XPU/Docker).
```

## MCP tools

| Tool | Purpose |
| --- | --- |
| `list_agents` | Specialists, backends, allowlist, local runtime |
| `chat_send` / `chat_approve` / `chat_list` | Same router as the GUI |
| `dispatch` / `follow_up` / `run_workflow` | Named specialist or pipeline |
| `get_run` / `list_runs` | Async run status |
| `list_allowed_dirs` / `add_allowed_dir` / `remove_allowed_dir` | Write sandbox |
| `list_hardware` / `list_local_models` / `recommend_local_models` | Fit and catalog |
| `download_local_model` | Hugging Face snapshot |
| `start_vllm` / `stop_vllm` / `remove_vllm` / `vllm_status` / `delete_local_model` | Local servers |

## Default specialists

| Id | Typical backend | Role |
| --- | --- | --- |
| `planner` | Anthropic | Implementation plan |
| `builder` | Cursor local | Writes code |
| `reviewer` | OpenAI | Review |
| `pr-triage` | Cursor local | Failing checks |
| `gemini-planner` | Gemini | Extra external planner |
| `vllm-chat` | Local vLLM | Text-only local model |
| `cloud-builder` | Cursor cloud | Isolated cloud agent |

Only **Cursor** backends edit files. Point `backend` at any id in `agents.config.yaml`.

### Add a backend

```yaml
backends:
  groq:
    type: openai
    baseUrl: https://api.groq.com/openai/v1
    model: llama-3.3-70b-versatile
    apiKeyEnv: GROQ_API_KEY

specialists:
  groq-reviewer:
    description: Fast external review
    backend: groq
    fallback: reviewer
```

Gemini uses Google’s OpenAI-compatible endpoint. Set **one** current model id (the GUI lists ids from Google when the key works). Do not put comments or `pro / flash` lists in `model`.

`${ENV_NAME}` in YAML expands from the process environment.

### Use from another repo

```json
{
  "mcpServers": {
    "agent-orchestrator": {
      "type": "stdio",
      "command": "/absolute/path/to/this-repo/node_modules/.bin/tsx",
      "args": ["/absolute/path/to/this-repo/src/index.ts"],
      "env": {
        "AGENT_ORCHESTRATOR_CONFIG": "/absolute/path/to/this-repo/agents.config.yaml",
        "WORKSPACE_CWD": "${workspaceFolder}",
        "CURSOR_API_KEY": "${env:CURSOR_API_KEY}"
      }
    }
  }
}
```

## What is not in git

`.env`, `.orchestrator/` (GUI token, secrets, chats, allowlist, model weights, vLLM state), `node_modules/`, and logs. See `.gitignore`.
