#!/usr/bin/env bash
# Portable Agent Orchestrator archive (GUI + /mcp) with a Node 22 runtime.
# Packs the current OS/arch. GitHub Release workflow runs this on Linux, macOS, and Windows.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NODE_VER="${NODE_VER:-22.13.0}"
VERSION="$(node -p "require('./package.json').version")"

uname_s="$(uname -s)"
uname_m="$(uname -m)"
case "$uname_s" in
  Linux*) os=linux ;;
  Darwin*) os=mac ;;
  MINGW*|MSYS*|CYGWIN*) os=win ;;
  *) echo "agent-orchestrator: unsupported uname $uname_s" >&2; exit 1 ;;
esac
case "$uname_m" in
  x86_64|amd64) arch=x64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "agent-orchestrator: unsupported arch $uname_m" >&2; exit 1 ;;
esac
NAME="agent-orchestrator-${VERSION}-${os}-${arch}"
STAGE="$ROOT/release/$NAME"
rm -rf "$STAGE"
mkdir -p "$STAGE" "$ROOT/release"

echo "agent-orchestrator: npm ci + build"
npm ci
npm run build
test -f "$ROOT/dist/gui.js"
test -f "$ROOT/dist/mcp-http.js"

echo "agent-orchestrator: stage $NAME"
cp -a dist gui package.json package-lock.json agents.config.yaml .env.example README.md LICENSE "$STAGE/"
mkdir -p "$STAGE/scripts"
cp -a scripts/hf_download.py scripts/requirements-hf.txt "$STAGE/scripts/"
# Fresh production deps in the archive (no tsx / typescript).
(cd "$STAGE" && npm ci --omit=dev)

fetch_node() {
  local dest="$1"
  mkdir -p "$dest"
  local tmp url file
  tmp="$(mktemp -d)"
  case "$os-$arch" in
    linux-x64)
      file="node-v${NODE_VER}-linux-x64.tar.xz"
      url="https://nodejs.org/dist/v${NODE_VER}/${file}"
      curl -fsSL "$url" -o "$tmp/$file"
      tar -xJf "$tmp/$file" -C "$tmp"
      ;;
    linux-arm64)
      file="node-v${NODE_VER}-linux-arm64.tar.xz"
      url="https://nodejs.org/dist/v${NODE_VER}/${file}"
      curl -fsSL "$url" -o "$tmp/$file"
      tar -xJf "$tmp/$file" -C "$tmp"
      ;;
    mac-x64)
      file="node-v${NODE_VER}-darwin-x64.tar.gz"
      url="https://nodejs.org/dist/v${NODE_VER}/${file}"
      curl -fsSL "$url" -o "$tmp/$file"
      tar -xzf "$tmp/$file" -C "$tmp"
      ;;
    mac-arm64)
      file="node-v${NODE_VER}-darwin-arm64.tar.gz"
      url="https://nodejs.org/dist/v${NODE_VER}/${file}"
      curl -fsSL "$url" -o "$tmp/$file"
      tar -xzf "$tmp/$file" -C "$tmp"
      ;;
    win-x64)
      file="node-v${NODE_VER}-win-x64.zip"
      url="https://nodejs.org/dist/v${NODE_VER}/${file}"
      curl -fsSL "$url" -o "$tmp/$file"
      if command -v unzip >/dev/null; then
        unzip -q "$tmp/$file" -d "$tmp"
      else
        powershell -NoProfile -Command "Expand-Archive -LiteralPath '$tmp/$file' -DestinationPath '$tmp'"
      fi
      ;;
    *)
      echo "agent-orchestrator: no Node build for $os-$arch" >&2
      exit 1
      ;;
  esac
  local extracted
  extracted="$(find "$tmp" -maxdepth 1 -type d -name "node-v${NODE_VER}-*" | head -n 1)"
  if [[ -z "$extracted" ]]; then
    echo "agent-orchestrator: Node extract failed" >&2
    exit 1
  fi
  cp -a "$extracted/." "$dest/"
  rm -rf "$tmp"
}

echo "agent-orchestrator: Node ${NODE_VER} runtime"
fetch_node "$STAGE/runtime"

mkdir -p "$STAGE/bin"
cat > "$STAGE/bin/agent-orchestrator-gui" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODE="$ROOT/runtime/bin/node"
if [[ ! -x "$NODE" ]]; then
  NODE="$ROOT/runtime/node.exe"
fi
exec "$NODE" "$ROOT/dist/gui.js" "$@"
EOF
cat > "$STAGE/bin/agent-orchestrator" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODE="$ROOT/runtime/bin/node"
if [[ ! -x "$NODE" ]]; then
  NODE="$ROOT/runtime/node.exe"
fi
exec "$NODE" "$ROOT/dist/index.js" "$@"
EOF
cat > "$STAGE/bin/agent-orchestrator-mcp" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NODE="$ROOT/runtime/bin/node"
if [[ ! -x "$NODE" ]]; then
  NODE="$ROOT/runtime/node.exe"
fi
exec "$NODE" "$ROOT/dist/mcp-http.js" "$@"
EOF
chmod +x "$STAGE/bin/agent-orchestrator-gui" "$STAGE/bin/agent-orchestrator" "$STAGE/bin/agent-orchestrator-mcp"

if [[ "$os" == win ]]; then
  cat > "$STAGE/bin/agent-orchestrator-gui.cmd" <<'EOF'
@echo off
setlocal
set ROOT=%~dp0..
if exist "%ROOT%\runtime\node.exe" (
  "%ROOT%\runtime\node.exe" "%ROOT%\dist\gui.js" %*
) else (
  "%ROOT%\runtime\bin\node.exe" "%ROOT%\dist\gui.js" %*
)
EOF
  cat > "$STAGE/bin/agent-orchestrator.cmd" <<'EOF'
@echo off
setlocal
set ROOT=%~dp0..
if exist "%ROOT%\runtime\node.exe" (
  "%ROOT%\runtime\node.exe" "%ROOT%\dist\index.js" %*
) else (
  "%ROOT%\runtime\bin\node.exe" "%ROOT%\dist\index.js" %*
)
EOF
  cat > "$STAGE/bin/agent-orchestrator-mcp.cmd" <<'EOF'
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

echo "agent-orchestrator: archive"
cd "$ROOT/release"
if [[ "$os" == win ]]; then
  if command -v zip >/dev/null; then
    zip -r "${NAME}.zip" "$NAME"
  else
    powershell -NoProfile -Command "Compress-Archive -Path '$NAME' -DestinationPath '${NAME}.zip'"
  fi
  echo "agent-orchestrator: ${ROOT}/release/${NAME}.zip"
else
  tar -czf "${NAME}.tar.gz" "$NAME"
  echo "agent-orchestrator: ${ROOT}/release/${NAME}.tar.gz"
fi
