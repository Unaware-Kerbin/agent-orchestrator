# Agent Orchestrator pack artifacts

Archives are **not** stored in git (`release/` is gitignored except this README). GitHub Actions **Build installers** runs `scripts/pack.sh` on `main` and on `v*` tags, then uploads workflow artifacts. A `v*` tag also attaches the files to the GitHub Release.

Future installers are **new tags** (`v0.1.6`, then `v0.1.7`, …). Do not rewrite old tags.

Do not commit the 100MB+ tarballs/zips.

## Names (`scripts/pack.sh`)

| File | What it is |
|---|---|
| `agent-orchestrator-<ver>-linux-x64.tar.gz` | Distro-agnostic Linux (no `.deb` / `.rpm`). Node 22 + packed `late-infer`. |
| `agent-orchestrator-<ver>-darwin-arm64.zip` | macOS Apple silicon. Unsigned zip — **not** `mac-….tar.gz`. Packed `late-infer`. |
| `agent-orchestrator-<ver>-darwin-x64.zip` | macOS Intel. Packed `late-infer`. |
| `agent-orchestrator-<ver>-win-x64.zip` | Windows. `bin\agent-orchestrator-gui.cmd`. Packed `late-infer`. |

Each archive includes Node 22 under `runtime/` and `runtime/bin/late-infer` (or `.exe`). Model weights are not included. Ollama / llama-server / vLLM are optional engines you may already run; vLLM Start still needs Docker on your computer. Bind stays `127.0.0.1` by default. After extract: `./bin/agent-orchestrator-gui` (Windows: `bin\agent-orchestrator-gui.cmd`). Copy the printed `/mcp` URL for Late.

## How CI publishes

1. Ubuntu packs `--target linux-x64`, macOS-14 packs `mac-arm64`, macOS-14 also cross-packs `mac-x64` (`x86_64-apple-darwin`), Windows packs `win-x64`.
2. Each job checks out [Unaware-Kerbin/late](https://github.com/Unaware-Kerbin/late), builds `late-infer` with Rust (native or cross), then runs `scripts/pack.sh` for that target.
3. `actions/upload-artifact` keeps `release/*.tar.gz` and `release/*.zip` on the workflow run.
4. On a `v*` tag, the `publish` job attaches `dist/*` to the GitHub Release.
