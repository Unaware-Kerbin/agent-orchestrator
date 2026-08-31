# Changelog

What shipped in each tag, in plain language. Newest first.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dates are America/New_York.

This is a window on **your computer**. Bind defaults to loopback (`127.0.0.1`); you may set one private IP. Packed archives include Ollama and llama-server. vLLM Start still needs Docker.

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

[0.1.4]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Unaware-Kerbin/agent-orchestrator/releases/tag/v0.1.0
