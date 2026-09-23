"""Run manifest: enough info to reconstruct every stage."""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path


def git_commit(repo: Path | None = None) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, cwd=str(repo or Path.cwd()), timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def package_lock_hash() -> str:
    # Best-effort: hash installed tasksource version marker.
    try:
        import tasksource  # noqa: F401
        return hashlib.sha256(str(getattr(__import__("tasksource"), "__file__", "")).encode()).hexdigest()[:16]
    except Exception:
        return "unknown"


def build_manifest(cfg, preflight=None, counts: dict | None = None,
                   prompt_hashes: dict | None = None) -> dict:
    return {
        "run_name": cfg.run_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": cfg.to_dict(),
        "provider": {"name": cfg.provider.name, "model": cfg.provider.model,
                     "base_url": cfg.provider.base_url,
                     "returned_model": getattr(preflight, "returned_model", None)},
        "prompt_hashes": prompt_hashes or {},
        "git_commit": git_commit(),
        "package_lock_hash": package_lock_hash(),
        "counts": counts or {},
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
