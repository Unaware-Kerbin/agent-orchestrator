#!/usr/bin/env bash
# Download pinned Ollama + llama-server into DEST/{bin,lib}. Verify SHA-256.
# Usage: fetch-inference-bins.sh DEST [linux-x64|linux-arm64|mac-arm64|mac-x64|win-x64|darwin-arm64|darwin-x64]
# DEST is typically apps/desktop/resources (Linux Late), resources-win, resources-mac, or orchestrator runtime.
# Second arg, INFERENCE_TARGET, TARGET, INFERENCE_BINS_KEY, or INFERENCE_OS+INFERENCE_ARCH selects a manifest row.
# darwin-* aliases map to mac-*. Default is uname (Linux pack unchanged). Do not spoof uname.
# Do not pass a non-host key with DEST=apps/desktop/resources — that folder is Linux ELFs.
# Layout: DEST/bin holds ollama + llama-server (llama.cpp shared libs beside llama-server).
# DEST/lib/ollama holds Ollama GPU libs (Linux Vulkan, Darwin mlx_metal*). CUDA/ROCm are stripped; mlx is not.
set -euo pipefail

# Resolve tools from the POSIX default PATH so a poisoned PATH cannot swap binaries.
# Fail closed if a required tool is missing or not an absolute executable.
secure_bin() {
  local name="$1"
  local resolved
  resolved="$(command -p -v "$name" 2>/dev/null || true)"
  if [[ -z "$resolved" || "$resolved" != /* || ! -x "$resolved" ]]; then
    echo "late: missing $name (need it on the default PATH, e.g. /usr/bin/$name)" >&2
    exit 1
  fi
  printf '%s' "$resolved"
}

optional_bin() {
  local name="$1"
  local resolved
  resolved="$(command -p -v "$name" 2>/dev/null || true)"
  if [[ -n "$resolved" && "$resolved" == /* && -x "$resolved" ]]; then
    printf '%s' "$resolved"
  fi
  return 0
}

MKDIR="$(secure_bin mkdir)"
UNAME="$(secure_bin uname)"
AWK="$(secure_bin awk)"
CURL="$(secure_bin curl)"
CP="$(secure_bin cp)"
RM="$(secure_bin rm)"
TAR="$(secure_bin tar)"
FIND="$(secure_bin find)"
HEAD="$(secure_bin head)"
BASENAME="$(secure_bin basename)"
DIRNAME="$(secure_bin dirname)"
CHMOD="$(secure_bin chmod)"
MKTEMP="$(secure_bin mktemp)"
GREP="$(secure_bin grep)"
ZSTD="$(optional_bin zstd)"
UNZIP="$(optional_bin unzip)"
PYTHON3="$(optional_bin python3)"

if command -p -v sha256sum >/dev/null 2>&1; then
  SHA256SUM="$(secure_bin sha256sum)"
  sha256_file() {
    local f="$1"
    "$SHA256SUM" "$f" | "$AWK" '{print $1}'
  }
else
  SHA256SUM="$(secure_bin shasum)"
  sha256_file() {
    local f="$1"
    "$SHA256SUM" -a 256 "$f" | "$AWK" '{print $1}'
  }
fi

DEST="${1:-}"
if [[ -z "$DEST" ]]; then
  echo "usage: fetch-inference-bins.sh DEST [linux-x64|linux-arm64|mac-arm64|mac-x64|win-x64|darwin-arm64|darwin-x64]" >&2
  exit 2
fi
"$MKDIR" -p "$DEST/bin" "$DEST/lib"

SCRIPT_DIR="$(cd "$("$DIRNAME" "$0")" && pwd)"
MANIFEST="${INFERENCE_BINS_MANIFEST:-$SCRIPT_DIR/inference-bins.manifest}"
if [[ ! -f "$MANIFEST" ]]; then
  echo "late: missing $MANIFEST" >&2
  exit 1
fi

uname_s="$("$UNAME" -s)"
uname_m="$("$UNAME" -m)"
case "$uname_s" in
  Linux*) os=linux ;;
  Darwin*) os=mac ;;
  MINGW*|MSYS*|CYGWIN*) os=win ;;
  *) echo "late: unsupported uname $uname_s" >&2; exit 1 ;;
esac
case "$uname_m" in
  x86_64|amd64) arch=x64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "late: unsupported arch $uname_m" >&2; exit 1 ;;
esac
KEY="$os-$arch"
host_key="$KEY"
OVERRIDE="${2:-}"
if [[ -z "$OVERRIDE" && -n "${INFERENCE_TARGET:-}" ]]; then
  OVERRIDE="$INFERENCE_TARGET"
elif [[ -z "$OVERRIDE" && -n "${TARGET:-}" ]]; then
  OVERRIDE="$TARGET"
elif [[ -z "$OVERRIDE" && -n "${INFERENCE_BINS_KEY:-}" ]]; then
  OVERRIDE="$INFERENCE_BINS_KEY"
elif [[ -z "$OVERRIDE" && -n "${INFERENCE_OS:-}" ]]; then
  OVERRIDE="${INFERENCE_OS}-${INFERENCE_ARCH:-}"
fi
if [[ -n "$OVERRIDE" ]]; then
  case "$OVERRIDE" in
    linux-x64|linux-arm64|mac-arm64|mac-x64|win-x64)
      KEY="$OVERRIDE"
      ;;
    darwin-arm64)
      KEY="mac-arm64"
      ;;
    darwin-x64)
      KEY="mac-x64"
      ;;
    *)
      echo "late: unknown inference key $OVERRIDE (want linux-x64, linux-arm64, mac-arm64, mac-x64, win-x64, darwin-arm64, darwin-x64)" >&2
      exit 1
      ;;
  esac
fi
os="${KEY%%-*}"
arch="${KEY##*-}"

# Never write another OS into apps/desktop/resources (Linux ollama / llama-server).
dest_norm="${DEST%/}"
case "$dest_norm" in
  */apps/desktop/resources|apps/desktop/resources)
    if [[ "$KEY" != "$host_key" ]]; then
      echo "late: refusing $KEY into $DEST (that folder holds this computer's Linux bins; use resources-win or resources-mac)" >&2
      exit 1
    fi
    ;;
esac

OLLAMA_VERSION="$("$AWK" -F= '/^OLLAMA_VERSION=/{print $2}' "$MANIFEST")"
LLAMA_CPP_TAG="$("$AWK" -F= '/^LLAMA_CPP_TAG=/{print $2}' "$MANIFEST")"
OLLAMA_REPO="$("$AWK" -F= '/^OLLAMA_REPO=/{print $2}' "$MANIFEST")"
LLAMA_REPO="$("$AWK" -F= '/^LLAMA_REPO=/{print $2}' "$MANIFEST")"

lookup() {
  local kind="$1"
  "$AWK" -v key="$KEY" -v kind="$kind" '
    $1 == key && $2 == kind { print $3, $4; found=1 }
    END { if (!found) exit 1 }
  ' "$MANIFEST"
}

CACHE="${LATE_INFERENCE_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/late/inference-bins}"
"$MKDIR" -p "$CACHE"

download_verify() {
  local url="$1" file="$2" expect="$3"
  local cache="$CACHE/${expect}-$("$BASENAME" "$file")"
  if [[ -f "$cache" ]]; then
    local got
    got="$(sha256_file "$cache")"
    if [[ "$got" == "$expect" ]]; then
      "$CP" "$cache" "$file"
      return 0
    fi
    "$RM" -f "$cache"
  fi
  echo "late: download $url"
  "$CURL" -fsSL "$url" -o "$file"
  local got
  got="$(sha256_file "$file")"
  if [[ "$got" != "$expect" ]]; then
    echo "late: SHA-256 mismatch for $file" >&2
    echo "late: expected $expect" >&2
    echo "late: got      $got" >&2
    "$RM" -f "$file"
    exit 1
  fi
  "$CP" "$file" "$cache"
}

tar_xf() {
  # Darwin archives record uid 501; as root on Linux GNU tar fails closed without this.
  if "$TAR" --help 2>&1 | "$GREP" -q no-same-owner; then
    "$TAR" --no-same-owner "$@"
  else
    "$TAR" "$@"
  fi
}

extract_archive() {
  local archive="$1" out="$2"
  "$MKDIR" -p "$out"
  case "$archive" in
    *.tar.zst)
      if "$TAR" --help 2>&1 | "$GREP" -q zstd; then
        tar_xf --zstd -xf "$archive" -C "$out"
      else
        if [[ -z "$ZSTD" ]]; then
          echo "late: need zstd to unpack $archive (apt install zstd)" >&2
          exit 1
        fi
        "$ZSTD" -d -c "$archive" | tar_xf -xf - -C "$out"
      fi
      ;;
    *.tgz|*.tar.gz)
      tar_xf -xzf "$archive" -C "$out"
      ;;
    *.zip)
      if [[ -n "$UNZIP" ]]; then
        "$UNZIP" -q "$archive" -d "$out"
      elif [[ -n "$PYTHON3" ]]; then
        "$PYTHON3" - "$archive" "$out" <<'PY'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
PY
      else
        echo "late: need unzip or python3 to unpack $archive" >&2
        exit 1
      fi
      ;;
    *)
      echo "late: unknown archive $archive" >&2
      exit 1
      ;;
  esac
}

find_one() {
  local root="$1"
  shift
  local name
  for name in "$@"; do
    local hit
    hit="$("$FIND" "$root" -type f -name "$name" 2>/dev/null | "$HEAD" -n 1 || true)"
    if [[ -n "$hit" ]]; then
      printf '%s' "$hit"
      return 0
    fi
  done
  return 1
}

# Copy the engine binary into dest_bin. Sibling shared libs go to dest_lib
# (default: dest_bin). mlx_metal* dirs (Darwin Ollama GPU) are copied too.
# Pass dest_lib=DEST/lib/ollama for Ollama so its libggml does not clobber llama.cpp.
copy_engine_dir() {
  local src_bin="$1"
  local dest_bin="$2"
  local dest_lib="${3:-$2}"
  local dir
  dir="$("$DIRNAME" "$src_bin")"
  "$MKDIR" -p "$dest_bin" "$dest_lib"
  "$CP" -a "$src_bin" "$dest_bin/"
  local f
  for f in "$dir"/*; do
    [[ -e "$f" ]] || continue
    local base
    base="$("$BASENAME" "$f")"
    case "$base" in
      llama-server|llama-server.exe|ollama|ollama.exe) continue ;;
    esac
    if [[ -f "$f" ]]; then
      case "$f" in
        *.so|*.so.*|*.dylib|*.dll|*.metal|*.metallib) "$CP" -a "$f" "$dest_lib/" ;;
      esac
    elif [[ -d "$f" ]]; then
      case "$base" in
        mlx_metal*) "$CP" -a "$f" "$dest_lib/" ;;
      esac
    fi
  done
}

# Drop CUDA/ROCm (and CUDA cousins) from packed Ollama. Keep Darwin mlx_metal*.
strip_ollama_gpu_libs() {
  local lib="$1"
  [[ -d "$lib" ]] || return 0
  local d
  "$FIND" "$lib" -mindepth 1 -maxdepth 2 -type d 2>/dev/null | while read -r d; do
    local base
    base="$("$BASENAME" "$d")"
    case "$base" in
      cuda*|rocm*|hip*|cuda_v*|cublas*) "$RM" -rf "$d" ;;
    esac
  done
}

work="$("$MKTEMP" -d)"
trap '"$RM" -rf "$work"' EXIT

read -r ollama_file ollama_sha <<<"$(lookup ollama)"
read -r llama_file llama_sha <<<"$(lookup llama)"

echo "late: Ollama ${OLLAMA_VERSION} + llama.cpp ${LLAMA_CPP_TAG} for $KEY"
download_verify "${OLLAMA_REPO}/${OLLAMA_VERSION}/${ollama_file}" "$work/$ollama_file" "$ollama_sha"
download_verify "${LLAMA_REPO}/${LLAMA_CPP_TAG}/${llama_file}" "$work/$llama_file" "$llama_sha"

extract_archive "$work/$ollama_file" "$work/ollama"
extract_archive "$work/$llama_file" "$work/llama"

ollama_bin="$(find_one "$work/ollama" ollama ollama.exe)" || {
  echo "late: ollama binary missing in $ollama_file" >&2
  exit 1
}
llama_bin="$(find_one "$work/llama" llama-server llama-server.exe)" || {
  echo "late: llama-server missing in $llama_file" >&2
  exit 1
}

copy_engine_dir "$ollama_bin" "$DEST/bin" "$DEST/lib/ollama"
copy_engine_dir "$llama_bin" "$DEST/bin"

ollama_lib="$("$FIND" "$work/ollama" -type d -name ollama 2>/dev/null | while read -r d; do
  if [[ "$("$BASENAME" "$("$DIRNAME" "$d")")" == lib ]]; then echo "$d"; fi
done | "$HEAD" -n 1 || true)"
if [[ -n "${ollama_lib:-}" && -d "$ollama_lib" ]]; then
  "$MKDIR" -p "$DEST/lib/ollama"
  "$CP" -a "$ollama_lib/." "$DEST/lib/ollama/"
fi
strip_ollama_gpu_libs "$DEST/lib/ollama"

if [[ "$os" != win ]]; then
  "$CHMOD" +x "$DEST/bin/ollama" "$DEST/bin/llama-server" 2>/dev/null || true
fi

if [[ "$os" == win ]]; then
  test -f "$DEST/bin/ollama.exe"
  test -f "$DEST/bin/llama-server.exe"
else
  test -x "$DEST/bin/ollama"
  test -x "$DEST/bin/llama-server"
fi
echo "late: inference bins in $DEST/bin"
