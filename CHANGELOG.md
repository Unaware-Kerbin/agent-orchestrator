# Changelog

What shipped in each tag, in plain language. Newest first.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Dates are America/New_York.

This is a window on **your computer**. Bind stays loopback (`127.0.0.1`). This does not start vLLM.

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

[0.1.2]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Unaware-Kerbin/agent-orchestrator/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Unaware-Kerbin/agent-orchestrator/releases/tag/v0.1.0
