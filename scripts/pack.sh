#!/usr/bin/env bash
# Portable Agent Orchestrator archive (GUI + /mcp) with a Node 22 runtime.
# Distro-agnostic tarball on Linux — not a .deb/.rpm (no fpm/electron-builder stack).
# Default: current OS/arch. Optional targets: linux-x64 linux-arm64 mac-x64 mac-arm64 win-x64
# (darwin-* aliases map to mac-*). GitHub Release workflow runs this with no args on each runner.
# Cross-pack: bash scripts/pack.sh --win --mac
#            bash scripts/pack.sh --target win-x64 --target mac-arm64 --target mac-x64
# Win zip: agent-orchestrator-*-win-x64.zip  Mac zips: *-darwin-arm64.zip *-darwin-x64.zip
# Existing linux-x64 tarball is never replaced unless PACK_OVERWRITE=1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NODE_VER="${NODE_VER:-22.13.0}"
VERSION="$(node -p "require('./package.json').version")"

host_uname_s="$(uname -s)"
host_uname_m="$(uname -m)"
case "$host_uname_s" in
  Linux*) host_os=linux ;;
  Darwin*) host_os=mac ;;
  MINGW*|MSYS*|CYGWIN*) host_os=win ;;
  *) echo "agent-orchestrator: unsupported uname $host_uname_s" >&2; exit 1 ;;
esac
case "$host_uname_m" in
  x86_64|amd64) host_arch=x64 ;;
  aarch64|arm64) host_arch=arm64 ;;
  *) echo "agent-orchestrator: unsupported arch $host_uname_m" >&2; exit 1 ;;
esac

normalize_target() {
  local raw="$1"
  case "$raw" in
    linux-x64|linux-arm64|mac-x64|mac-arm64|win-x64)
      printf '%s' "$raw"
      ;;
    darwin-arm64)
      printf '%s' "mac-arm64"
      ;;
    darwin-x64)
      printf '%s' "mac-x64"
      ;;
    *)
      echo "agent-orchestrator: unsupported target $raw (want linux-x64|linux-arm64|mac-x64|mac-arm64|win-x64)" >&2
      exit 1
      ;;
  esac
}

TARGETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --win)
      TARGETS+=("win-x64")
      shift
      ;;
    --mac)
      TARGETS+=("mac-arm64" "mac-x64")
      shift
      ;;
    --target)
      if [[ -z "${2:-}" ]]; then
        echo "agent-orchestrator: --target needs linux-x64|mac-arm64|mac-x64|win-x64" >&2
        exit 1
      fi
      TARGETS+=("$(normalize_target "$2")")
      shift 2
      ;;
    --target=*)
      TARGETS+=("$(normalize_target "${1#--target=}")")
      shift
      ;;
    -*)
      echo "agent-orchestrator: unknown flag $1 (want --win --mac --target KEY)" >&2
      exit 1
      ;;
    *)
      TARGETS+=("$(normalize_target "$1")")
      shift
      ;;
  esac
done
if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS+=("${host_os}-${host_arch}")
fi

mkdir -p "$ROOT/release"

fetch_node() {
  local dest="$1"
  local os="$2"
  local arch="$3"
  mkdir -p "$dest"
  local tmp url file
  tmp="$(mktemp -d)"
  case "$os-$arch" in
    linux-x64)
      file="node-v${NODE_VER}-linux-x64.tar.xz"
      url="https://nodejs.org/dist/v${NODE_VER}/${file}"
      ;;
    linux-arm64)
      file="node-v${NODE_VER}-linux-arm64.tar.xz"
      url="https://nodejs.org/dist/v${NODE_VER}/${file}"
      ;;
    mac-x64)
      file="node-v${NODE_VER}-darwin-x64.tar.gz"
      url="https://nodejs.org/dist/v${NODE_VER}/${file}"
      ;;
    mac-arm64)
      file="node-v${NODE_VER}-darwin-arm64.tar.gz"
      url="https://nodejs.org/dist/v${NODE_VER}/${file}"
      ;;
    win-x64)
      file="node-v${NODE_VER}-win-x64.zip"
      url="https://nodejs.org/dist/v${NODE_VER}/${file}"
      ;;
    *)
      echo "agent-orchestrator: no Node build for $os-$arch" >&2
      rm -rf "$tmp"
      return 1
      ;;
  esac
  if ! curl -fsSL "$url" -o "$tmp/$file"; then
    echo "agent-orchestrator: Node download failed for $os-$arch" >&2
    rm -rf "$tmp"
    return 1
  fi
  case "$file" in
    *.tar.xz) tar -xJf "$tmp/$file" -C "$tmp" || { rm -rf "$tmp"; return 1; } ;;
    *.tar.gz) tar -xzf "$tmp/$file" -C "$tmp" || { rm -rf "$tmp"; return 1; } ;;
    *.zip)
      if command -v unzip >/dev/null; then
        unzip -q "$tmp/$file" -d "$tmp" || { rm -rf "$tmp"; return 1; }
      else
        powershell -NoProfile -Command "Expand-Archive -LiteralPath '$tmp/$file' -DestinationPath '$tmp'" || { rm -rf "$tmp"; return 1; }
      fi
      ;;
  esac
  local extracted
  extracted="$(find "$tmp" -maxdepth 1 -type d -name "node-v${NODE_VER}-*" | head -n 1)"
  if [[ -z "$extracted" ]]; then
    echo "agent-orchestrator: Node extract failed" >&2
    rm -rf "$tmp"
    return 1
  fi
  cp -a "$extracted/." "$dest/"
  rm -rf "$tmp"
  return 0
}

write_launchers() {
  local stage="$1"
  local os="$2"
  mkdir -p "$stage/bin"
  cat > "$stage/bin/agent-orchestrator-gui" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODE="$ROOT/runtime/bin/node"
if [[ ! -x "$NODE" ]]; then
  NODE="$ROOT/runtime/node.exe"
fi
exec "$NODE" "$ROOT/dist/gui.js" "$@"
EOF
  cat > "$stage/bin/agent-orchestrator" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODE="$ROOT/runtime/bin/node"
if [[ ! -x "$NODE" ]]; then
  NODE="$ROOT/runtime/node.exe"
fi
exec "$NODE" "$ROOT/dist/index.js" "$@"
EOF
  cat > "$stage/bin/agent-orchestrator-mcp" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODE="$ROOT/runtime/bin/node"
if [[ ! -x "$NODE" ]]; then
  NODE="$ROOT/runtime/node.exe"
fi
exec "$NODE" "$ROOT/dist/mcp-http.js" "$@"
EOF
  chmod +x "$stage/bin/agent-orchestrator-gui" "$stage/bin/agent-orchestrator" "$stage/bin/agent-orchestrator-mcp"

  if [[ "$os" == win ]]; then
    cat > "$stage/bin/agent-orchestrator-gui.cmd" <<'EOF'
@echo off
setlocal
set ROOT=%~dp0..
if exist "%ROOT%\runtime\node.exe" (
  "%ROOT%\runtime\node.exe" "%ROOT%\dist\gui.js" %*
) else (
  "%ROOT%\runtime\bin\node.exe" "%ROOT%\dist\gui.js" %*
)
EOF
    cat > "$stage/bin/agent-orchestrator.cmd" <<'EOF'
@echo off
setlocal
set ROOT=%~dp0..
if exist "%ROOT%\runtime\node.exe" (
  "%ROOT%\runtime\node.exe" "%ROOT%\dist\index.js" %*
) else (
  "%ROOT%\runtime\bin\node.exe" "%ROOT%\dist\index.js" %*
)
EOF
    cat > "$stage/bin/agent-orchestrator-mcp.cmd" <<'EOF'
@echo off
setlocal
set ROOT=%~dp0..
if exist "%ROOT%\runtime\node.exe" (
  "%ROOT%\runtime\node.exe" "%ROOT%\dist\mcp-http.js" %*
) else (
  "%ROOT%\runtime\bin\node.exe" "%ROOT%\dist\mcp-http.js" %*
)
EOF
  fi
}

copy_committed_agents_config() {
  local stage="$1"
  if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    && git -C "$ROOT" cat-file -e HEAD:agents.config.yaml 2>/dev/null; then
    git -C "$ROOT" show HEAD:agents.config.yaml > "$stage/agents.config.yaml"
  else
    cp -a "$ROOT/agents.config.yaml" "$stage/"
  fi
}

archive_one() {
  local name="$1"
  local os="$2"
  (
    cd "$ROOT/release"
    if [[ "$os" == linux ]]; then
      tar -czf "${name}.tar.gz" "$name"
      echo "agent-orchestrator: ${ROOT}/release/${name}.tar.gz"
    elif command -v zip >/dev/null; then
      rm -f "${name}.zip"
      zip -rq "${name}.zip" "$name"
      echo "agent-orchestrator: ${ROOT}/release/${name}.zip"
    else
      powershell -NoProfile -Command "Compress-Archive -Path '$name' -DestinationPath '${name}.zip'"
      echo "agent-orchestrator: ${ROOT}/release/${name}.zip"
    fi
  )
}

engine_stage_dir() {
  local os="$1"
  local arch="$2"
  case "$os-$arch" in
    win-x64) printf '%s' "$ROOT/release/runtime-win" ;;
    mac-arm64) printf '%s' "$ROOT/release/runtime-mac-arm64" ;;
    mac-x64) printf '%s' "$ROOT/release/runtime-mac-x64" ;;
    *) printf '%s' "$ROOT/release/runtime-${os}-${arch}" ;;
  esac
}

install_engines() {
  local dest="$1"
  local os="$2"
  local arch="$3"
  local engine_stage
  engine_stage="$(engine_stage_dir "$os" "$arch")"
  mkdir -p "$engine_stage/bin" "$engine_stage/lib"
  local need_fetch=1
  if [[ "$os" == win && -f /tmp/jian-yang-win-inference/bin/ollama.exe && -f /tmp/jian-yang-win-inference/bin/llama-server.exe ]]; then
    echo "agent-orchestrator: copy /tmp/jian-yang-win-inference -> $engine_stage"
    cp -a /tmp/jian-yang-win-inference/. "$engine_stage/"
    need_fetch=0
  elif [[ "$os" == win && -f "$engine_stage/bin/ollama.exe" && -f "$engine_stage/bin/llama-server.exe" ]]; then
    need_fetch=0
  elif [[ "$os" != win && -f "$engine_stage/bin/ollama" && -f "$engine_stage/bin/llama-server" ]]; then
    need_fetch=0
  fi
  if [[ "$need_fetch" == 1 ]]; then
    echo "agent-orchestrator: Ollama + llama-server ($os-$arch) -> $engine_stage"
    INFERENCE_TARGET="$os-$arch" bash "$ROOT/scripts/fetch-inference-bins.sh" "$engine_stage" "$os-$arch"
  fi
  mkdir -p "$dest/bin" "$dest/lib"
  cp -a "$engine_stage/bin/." "$dest/bin/"
  if [[ -d "$engine_stage/lib" ]]; then
    cp -a "$engine_stage/lib/." "$dest/lib/"
  fi
}

pack_one() {
  local os="$1"
  local arch="$2"
  local label="$os"
  if [[ "$os" == mac ]]; then
    label=darwin
  fi
  local name="agent-orchestrator-${VERSION}-${label}-${arch}"
  local stage="$ROOT/release/$name"
  local archive
  if [[ "$os" == linux ]]; then
    archive="$ROOT/release/${name}.tar.gz"
  else
    archive="$ROOT/release/${name}.zip"
  fi

  if [[ "$os-$arch" == "linux-x64" && -f "$archive" && "${PACK_OVERWRITE:-}" != 1 ]]; then
    echo "agent-orchestrator: keep existing $archive (set PACK_OVERWRITE=1 to replace)"
    return 0
  fi

  echo "agent-orchestrator: stage $name"
  rm -rf "$stage"
  mkdir -p "$stage"

  local linux_stage="$ROOT/release/agent-orchestrator-${VERSION}-linux-x64"
  if [[ "$os-$arch" != "linux-x64" && -d "$linux_stage/dist" && -d "$linux_stage/node_modules" ]]; then
    tar -C "$linux_stage" --exclude=runtime -cf - . | tar -C "$stage" -xf -
  else
    cp -a dist gui package.json package-lock.json .env.example README.md LICENSE "$stage/"
    mkdir -p "$stage/scripts"
    cp -a scripts/hf_download.py scripts/requirements-hf.txt "$stage/scripts/"
    local npm_os npm_cpu
    case "$os" in
      win) npm_os=win32 ;;
      mac) npm_os=darwin ;;
      *) npm_os=linux ;;
    esac
    case "$arch" in
      arm64) npm_cpu=arm64 ;;
      *) npm_cpu=x64 ;;
    esac
    (cd "$stage" && npm ci --omit=dev --os "$npm_os" --cpu "$npm_cpu")
  fi
  copy_committed_agents_config "$stage"

  echo "agent-orchestrator: Node ${NODE_VER} runtime ($os-$arch)"
  if ! fetch_node "$stage/runtime" "$os" "$arch"; then
    echo "agent-orchestrator: Node ${NODE_VER} for $os-$arch missing; shipping ollama + llama-server only" >&2
    mkdir -p "$stage/runtime"
    printf '%s\n' "Node ${NODE_VER} for $os-$arch was not bundled. Install Node 22 and run dist/gui.js, or re-pack on that OS." > "$stage/runtime/NODE-GAP.txt"
  fi

  install_engines "$stage/runtime" "$os" "$arch"

  write_launchers "$stage" "$os"
  archive_one "$name" "$os"
}

if [[ "${PACK_SKIP_BUILD:-}" == 1 ]]; then
  test -f "$ROOT/dist/gui.js"
  test -f "$ROOT/dist/mcp-http.js"
else
  echo "agent-orchestrator: npm ci + build"
  npm ci
  npm run build
  test -f "$ROOT/dist/gui.js"
  test -f "$ROOT/dist/mcp-http.js"
fi

for key in "${TARGETS[@]}"; do
  pack_one "${key%-*}" "${key#*-}"
done
