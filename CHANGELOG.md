# Changelog

What shipped in each tag, in plain language. Newest first.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dates are America/New_York.

This is a window on **your computer**. Bind defaults to loopback (`127.0.0.1`); you may set one private IP. Packed archives include Node 22 and `late-infer`. Ollama / llama-server / vLLM remain optional local engines; vLLM Start still needs Docker.

## [0.1.6] - 2026-09-08

New GitHub tag [v0.1.6](https://github.com/Unaware-Kerbin/agent-orchestrator/releases/tag/v0.1.6) so this build is trackable. Tag [v0.1.5](https://github.com/Unaware-Kerbin/agent-orchestrator/releases/tag/v0.1.5) stays frozen — future installers are new tags; do not rewrite old tags.

### Added

- **Fits my GPU (idle compile target).** Hub store budgets ~80% of the **idle** card VRAM (not dual-GPU total). Display compile target keeps the 70% cap before that fraction. **Fits my GPU** filter (default on) auto-hides over-budget rows; peak `vramMaxMiB` drives the mark (`13697fb`).
- **Download vs compile pipeline states.** Distinct phases: `probing` → `downloading` → `compiling` → `ready` (+ soft-fail error messages) in compile jobs and Local models progress UI (`13697fb`).
- **Research metrics panel (opt-in).** Composer **Show research metrics** expands tokens/sec with latency, VRAM label, and model id on replies (still default off) (`13697fb`).
- **Intel Arc OpenVINO defaults.** Compile/Start set `LATE_INFER_SKIP_MLC_PREFLIGHT` when `LATE_INFER_ACCEL=intel`; idle BDF + `ZE_AFFINITY_MASK` stay pinned to the non-display B70 (`13697fb`).

### Also since v0.1.5

- Soft-fail human messages for missing/denied Hub `config.json` on Download, and opt-in chat tokens/sec (`7d04d10`); Windows late-infer stamp path fix for CI (`9b97e1d`).

## [0.1.5] - 2026-09-08

New GitHub tag [v0.1.5](https://github.com/Unaware-Kerbin/agent-orchestrator/releases/tag/v0.1.5) so this build is trackable. Tag [v0.1.4](https://github.com/Unaware-Kerbin/agent-orchestrator/releases/tag/v0.1.4) stays frozen — future installers are new tags; do not rewrite old tags.

### Documented

- **Inference engines on your computer.** README local-inference section: late-infer (OpenVINO / Candle), Ollama, llama.cpp, vLLM Docker; loopback-only binds; Start/Stop and demo clips (`6f39611`).

### Added

- **late-infer Hub pre-screen.** Local models HF store stamps `ovExportOk` / `loadable` from known `model_type` plus a soft-fail cached `config.json` probe; loadable ungated rows sort first; broken config / OV-incompatible graphs are demoted (`a776eee`).
- **Chat tokens/sec (research, opt-in).** Composer checkbox **Show tokens/sec (research)** — default off. Estimates tok/s for the active reply (chars÷4 / elapsed); finished replies store `tokensPerSec` / `completionTokensEst` on the message. Not a billing meter (`7d04d10`).

### Fixed

- **Build installers CI.** Packaging after v0.1.4 requires a host-built `late-infer` binary; CI had no sibling Late checkout, so every `main` pack failed. Workflow now checks out `Unaware-Kerbin/late`, builds `late-infer` on each runner, and packs one native target per job (linux / darwin-arm64 / darwin-x64 / win-x64). Windows stamp uses a relative `package.json` require so Git Bash paths work.
- **Instant `Error: config.json` on Download.** Fallback seed `Qwen/Qwen3-4B-Instruct` had no public Hub `config.json` (real Instruct id is `Qwen/Qwen3-4B-Instruct-2507`); screening skipped probes when `model_type` was inferred, so the row showed ready/OV ok while Download failed. Also: late-infer often prints bare `Error: config.json` on the first stderr line before Caused-by / HTTP status — the job froze that cryptic line. Fix: use real Hub seed ids, always probe fallback seeds, soft-fail bare/missing/denied config with a human message, upgrade the job error from full stderr on exit, and fail closed earlier on Hub 404 in the gated probe (`7d04d10`).

## [0.1.4] - 2026-08-31

New GitHub tag [v0.1.4](https://github.com/Unaware-Kerbin/agent-orchestrator/releases/tag/v0.1.4) so this build is trackable. Tag [v0.1.3](https://github.com/Unaware-Kerbin/agent-orchestrator/releases/tag/v0.1.3) stays frozen — future installers are new tags; do not rewrite old tags.

**listen_host auto (primary RFC1918 IPv4).**

- `listen_host: auto` or `AGENT_ORCHESTRATOR_MCP_HOST=auto` binds this computer's primary RFC1918 IPv4 (here `192.168.2.139`) and prints `http://192.168.2.139:8790/mcp`. Settings empty save is `auto`. Bare `npm run mcp:http` stays loopback unless auto or an IP is set.
- GUI Settings → Listen host: empty or `auto` writes `listen_host: "auto"`. Placeholder shows this computer's LAN address.

## [0.1.3] - 2026-08-31

New GitHub tag [v0.1.3](https://github.com/Unaware-Kerbin/agent-orchestrator/releases/tag/v0.1.3) so this build is trackable. Tag [v0.1.2](https://github.com/Unaware-Kerbin/agent-orchestrator/releases/tag/v0.1.2) stays frozen — future installers are new tags; do not rewrite old tags.

**One private IP for Late on another computer.**

- GUI and `npm run mcp:http` can bind **one** RFC1918 / ULA address you type (`mcp.listen_host` in YAML, Settings → Listen host, or `AGENT_ORCHESTRATOR_MCP_HOST` / `AGENT_ORCHESTRATOR_GUI_HOST`). Default stays `127.0.0.1`. `0.0.0.0`, `::`, and public IPs are refused.
- Printed / Copy MCP URL is `http://<that-host>:<port>/mcp` so Late on another computer can paste it. Host/Origin allow that bind host plus loopback; random websites are rejected. Late still does not send a GUI token — trusted LAN, firewall to the laptop.
- Example: `AGENT_ORCHESTRATOR_MCP_HOST=192.168.2.139 AGENT_ORCHESTRATOR_MCP_PORT=8790 npm run mcp:http` then paste `http://192.168.2.139:8790/mcp`. You start Orchestrator; Late will not start it. SSH is not required.

**Packed Ollama + llama-server; GitHub packs linux/win/darwin.**

- Packed installs include Ollama and `llama-server` (Vulkan on Linux/Windows, Metal on Apple silicon; Intel Mac is CPU/BLAS) under `runtime/bin`. Apple silicon packs Ollama `mlx_metal_v3`/`v4` under `lib/ollama`. `scripts/pack.sh` can stage `win-x64` / `mac-arm64` / `mac-x64` from Linux without replacing the Linux tarball. The Linux archive is distro-agnostic (Debian, Fedora, Arch, and others use the same `.tar.gz`; there is no `.deb`/`.rpm`). Start them from Local models (loopback only). Model weights stay out of the archive. vLLM Start still needs Docker; if Docker is missing the Start-with-Docker control stays hidden. README lists `darwin-*.zip` (not `mac-….tar.gz`), Local models Start/Stop, token-gated `/api`, and that there is no `start_ollama` MCP tool.
- **Build installers** now runs on `main` (workflow artifacts) as well as `v*` tags (GitHub Release). Ubuntu packs `linux-x64`, macOS packs both Darwin zips, Windows packs `win-x64`. See [release/README.md](release/README.md) for the filenames. Binaries stay gitignored.

## [0.1.2] - 2026-08-30

Portable GUI + Streamable HTTP `/mcp`. Each archive includes Node 22. Copy the printed `/mcp` URL for Late. Stop with `--stop`.

### Security

- **Wrap isolation.** Cloud speakers never see `UNTRUSTED DEVICE OUTPUT`. Only local inference on your computer gets that block.
- **Fail-closed missing END.** If Late’s wrap has BEGIN but no END, the orchestrator refuses to route.
- **Digest pin.** Check for updates downloads the official Unaware-Kerbin GitHub asset and verifies the SHA-256 digest so a redirect or swapped file cannot land (`e709911`).
- **IDE `mcp.json` is not the Late path.** [`.cursor/mcp.json`](.cursor/mcp.json) is for this repo’s Cursor IDE. It does not carry API keys. Late pastes the printed `/mcp` URL and keeps Approve.

### Added

- README clips recaptured on **your computer**: Debate (local Gemma + Cursor; Gemini 429 skipped), Approve, Updates confirm, and the other GUI walks.

## [0.1.1] - 2026-08-29

Portable GUI + `/mcp` archives with apply-patch and a Debate README clip.

### Added

- Apply approved file patches (Node apply-patch) so chat can write in the granted folder without Cursor.
- Late Agent=MCP replies wrap as JSON tools so playbooks stage instead of lecturing.
- README Debate clip: several models on your computer actually reply.

## [0.1.0] - 2026-08-29

First portable GUI + Streamable HTTP `/mcp` archives (Linux, macOS, Windows). Loopback only. Extract, then `./bin/agent-orchestrator-gui` (Windows: `bin\agent-orchestrator-gui.cmd`). Copy the printed `/mcp` URL for Late.

[0.1.6]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Unaware-Kerbin/agent-orchestrator/releases/tag/v0.1.0
