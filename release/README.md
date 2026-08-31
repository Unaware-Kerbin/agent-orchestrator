# Agent Orchestrator pack artifacts

Archives are **not** stored in git (`release/` is gitignored except this README). GitHub Actions **Build installers** runs `scripts/pack.sh` on `main` and on `v*` tags, then uploads workflow artifacts. A `v*` tag also attaches the files to the GitHub Release.

Do not commit the 100MB+ tarballs/zips.

## Names (`scripts/pack.sh`)

| File | What it is |
|---|---|
| `agent-orchestrator-<ver>-linux-x64.tar.gz` | Distro-agnostic Linux (no `.deb` / `.rpm`). Node 22 + Ollama + llama-server (Vulkan). |
| `agent-orchestrator-<ver>-darwin-arm64.zip` | macOS Apple silicon. Unsigned zip — **not** `mac-….tar.gz`. Metal / mlx. |
| `agent-orchestrator-<ver>-darwin-x64.zip` | macOS Intel. llama.cpp is CPU/BLAS. |
| `agent-orchestrator-<ver>-win-x64.zip` | Windows. `bin\agent-orchestrator-gui.cmd`. |

Each archive includes Node 22 under `runtime/`. Model weights are not included. vLLM Start still needs Docker on your computer. Bind stays `127.0.0.1`. After extract: `./bin/agent-orchestrator-gui` (Windows: `bin\agent-orchestrator-gui.cmd`). Copy the printed `/mcp` URL for Late.

## How CI publishes

1. Ubuntu packs `--target linux-x64`, macOS packs both Darwin zips, Windows packs `win-x64`.
2. `actions/upload-artifact` keeps `release/*.tar.gz` and `release/*.zip` on the workflow run.
3. On a `v*` tag, the `publish` job attaches `dist/*` to the GitHub Release.
