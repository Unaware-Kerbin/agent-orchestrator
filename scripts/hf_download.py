#!/usr/bin/env python3
"""Download a Hugging Face snapshot. Progress is JSON lines on stdout. Never prints tokens.

Do not pass a custom tqdm class into huggingface_hub.snapshot_download.
huggingface_hub calls tqdm_class.get_lock() (tqdm.std.tqdm API); a duck-typed
JsonTqdm wrapper raises: type object 'JsonTqdm' has no attribute 'get_lock'.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from typing import Any, Callable


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def dir_nbytes(path: str) -> int:
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def snapshot_kwargs(repo: str, dest: str, revision: str | None = None) -> dict[str, Any]:
    """Arguments for huggingface_hub.snapshot_download. Never includes tqdm_class."""
    kwargs: dict[str, Any] = {"repo_id": repo, "local_dir": dest}
    if revision:
        kwargs["revision"] = revision
    return kwargs


def _disable_hf_progress_bars() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass


def estimate_repo_bytes(repo: str, revision: str | None = None) -> int:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, revision=revision, files_metadata=True)
        siblings = getattr(info, "siblings", None) or []
        return sum(int(getattr(item, "size", 0) or 0) for item in siblings)
    except Exception:
        return 0


def _watch_progress(dest: str, total: int, stop: threading.Event) -> None:
    last_pct = -1
    while not stop.wait(0.75):
        downloaded = dir_nbytes(dest)
        if total > 0:
            pct = int(downloaded * 100 / total)
            pct = max(0, min(pct, 99))
        else:
            pct = 0
        if pct != last_pct:
            last_pct = pct
            emit(
                {
                    "event": "progress",
                    "downloaded": downloaded,
                    "total": total,
                    "percent": pct,
                }
            )


def download_snapshot(
    repo: str,
    dest: str,
    revision: str | None = None,
    *,
    snapshot_download: Callable[..., str] | None = None,
    estimate_bytes: Callable[[str, str | None], int] | None = None,
    watch: bool = True,
) -> str:
    dest = os.path.abspath(dest)
    os.makedirs(dest, exist_ok=True)
    kwargs = snapshot_kwargs(repo, dest, revision)
    if "tqdm_class" in kwargs:
        raise RuntimeError("tqdm_class must not be passed to snapshot_download")

    fn = snapshot_download
    if fn is None:
        _disable_hf_progress_bars()
        from huggingface_hub import snapshot_download as hf_snapshot_download

        fn = hf_snapshot_download

    total = 0
    if estimate_bytes is not None:
        total = estimate_bytes(repo, revision)
    elif snapshot_download is None:
        total = estimate_repo_bytes(repo, revision)
        if total:
            emit({"event": "progress", "downloaded": dir_nbytes(dest), "total": total, "percent": 0})

    stop = threading.Event()
    watcher: threading.Thread | None = None
    if watch:
        watcher = threading.Thread(target=_watch_progress, args=(dest, total, stop), daemon=True)
        watcher.start()
    try:
        path = fn(**kwargs)
    finally:
        stop.set()
        if watcher is not None:
            watcher.join(timeout=1.5)
    return path


def hf_token_present() -> bool:
    token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return bool(token)


def redact_text(text: str) -> str:
    out = text
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = (os.environ.get(name) or "").strip()
        if len(value) > 3:
            out = out.replace(value, "***")
    return re.sub(r"\bhf_[A-Za-z0-9]{8,}\b", "hf_***", out)


def gated_auth_message(message: str, token_present: bool) -> str:
    lowered = message.lower()
    if not any(token in lowered for token in ("401", "gated", "restricted", "authentication", "403")):
        return message
    if token_present:
        return (
            f"{message}. Token is set but Hugging Face still denied access. "
            "Accept the model license on the model card while logged into the same Hugging Face account, then retry. "
            "Do not commit the token."
        )
    return (
        f"{message}. If this repo is gated: accept the license on the model card while logged into Hugging Face, "
        "then paste a read token in Settings → Local models (or set HF_TOKEN / HUGGING_FACE_HUB_TOKEN). "
        "Create a token at https://huggingface.co/settings/tokens. Do not commit the token."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download as _hf  # noqa: F401
    except ImportError:
        emit(
            {
                "event": "error",
                "message": "huggingface_hub is not installed. Install with: pip install huggingface_hub",
            }
        )
        return 2

    try:
        path = download_snapshot(args.repo, args.dest, args.revision)
        emit({"event": "done", "path": path, "percent": 100})
        return 0
    except Exception as exc:  # noqa: BLE001 — surface HF/auth errors to the orchestrator
        msg = gated_auth_message(str(exc), hf_token_present())
        emit({"event": "error", "message": redact_text(msg)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
