#!/usr/bin/env bash
# Build this application's late-infer into bin/ and runtime/bin.
# Source crate: LATE_INFER_CRATE, LATE_CHECKOUT, or a sibling Local_AI_Terminal_Emulator checkout.
# Not part of npm test (Candle is heavy). Weights are not copied. Loopback bind is the binary's job.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST_DIR="$ROOT/bin"
RUNTIME_DIR="$ROOT/runtime/bin"
mkdir -p "$DEST_DIR" "$RUNTIME_DIR"

crate_dir=""
if [[ -n "${LATE_INFER_CRATE:-}" ]]; then
  crate_dir="${LATE_INFER_CRATE}"
elif [[ -n "${LATE_CHECKOUT:-}" ]]; then
  crate_dir="${LATE_CHECKOUT}/crates/late-infer"
else
  for sibling in Local_AI_Terminal_Emulator late; do
    candidate="$(cd "$ROOT/.." && pwd)/${sibling}/crates/late-infer"
    if [[ -f "$candidate/Cargo.toml" ]]; then
      crate_dir="$candidate"
      break
    fi
  done
fi

if [[ -z "$crate_dir" || ! -f "$crate_dir/Cargo.toml" ]]; then
  echo "agent-orchestrator: late-infer crate not found." >&2
  echo "agent-orchestrator: set LATE_INFER_CRATE or LATE_CHECKOUT, or keep the crate checkout next to this repo, then run npm run infer:build." >&2
  echo "agent-orchestrator: do not install a desktop helper — this binary ships with Orchestrator." >&2
  exit 1
fi

workspace="$(cd "$crate_dir/../.." && pwd)"
if [[ ! -f "$workspace/Cargo.toml" ]]; then
  echo "agent-orchestrator: expected a Cargo workspace at $workspace (parent of crates/late-infer)" >&2
  exit 1
fi

if ! command -v cargo >/dev/null 2>&1; then
  echo "agent-orchestrator: cargo is missing. Install Rust, then run npm run infer:build on your computer." >&2
  exit 1
fi

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target}"
echo "agent-orchestrator: cargo build -p late-infer --release (crate $crate_dir)"
cargo build -p late-infer --release --manifest-path "$workspace/Cargo.toml"

src=""
dest_name="late-infer"
if [[ -f "$CARGO_TARGET_DIR/release/late-infer.exe" ]]; then
  src="$CARGO_TARGET_DIR/release/late-infer.exe"
  dest_name="late-infer.exe"
elif [[ -f "$CARGO_TARGET_DIR/release/late-infer" ]]; then
  src="$CARGO_TARGET_DIR/release/late-infer"
else
  echo "agent-orchestrator: cargo finished but late-infer was not under $CARGO_TARGET_DIR/release" >&2
  exit 1
fi

cp -a "$src" "$DEST_DIR/$dest_name"
cp -a "$src" "$RUNTIME_DIR/$dest_name"
if [[ "$dest_name" == "late-infer" ]]; then
  chmod +x "$DEST_DIR/$dest_name" "$RUNTIME_DIR/$dest_name"
fi

pkg_ver="$(cd "$ROOT" && node -p "require('./package.json').version")"
printf '%s\n' "$pkg_ver" > "$DEST_DIR/late-infer.stamp"
printf '%s\n' "$pkg_ver" > "$RUNTIME_DIR/.late-infer-built-for"

echo "agent-orchestrator: $DEST_DIR/$dest_name"
echo "agent-orchestrator: $RUNTIME_DIR/$dest_name"
