"""Thin git helpers: revisions, ``git show``, and temporary worktrees."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from cyberwheel.tests.framework.core import REPO_ROOT


def _git(*args: str, check: bool = True) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        if check:
            raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return None
    return proc.stdout


def current_commit() -> Optional[str]:
    out = _git("rev-parse", "HEAD", check=False)
    return out.strip() if out else None


def is_dirty() -> bool:
    out = _git("status", "--porcelain", check=False)
    return bool(out and out.strip())


def resolve_rev(rev: str) -> Optional[str]:
    out = _git("rev-parse", "--verify", f"{rev}^{{commit}}", check=False)
    return out.strip() if out else None


def show_file(rev: str, relpath: str) -> Optional[str]:
    """Return the contents of ``relpath`` at ``rev``, or None if absent."""
    return _git("show", f"{rev}:{relpath}", check=False)


def deps_changed_since(rev: str) -> list[str]:
    """Dependency manifests that differ between ``rev`` and the working tree."""
    out = _git(
        "diff", "--name-only", rev, "--", "pyproject.toml", "uv.lock", check=False
    )
    return out.strip().splitlines() if out and out.strip() else []


@contextmanager
def worktree(rev: str):
    """Check out ``rev`` into a temporary git worktree; always cleans up."""
    sha = resolve_rev(rev)
    if sha is None:
        raise RuntimeError(f"cannot resolve revision {rev!r}")
    parent = Path(tempfile.mkdtemp(prefix=f"cyberwheel-perf-{sha[:8]}-"))
    path = parent / "wt"
    _git("worktree", "add", "--detach", str(path), sha)
    try:
        yield path
    finally:
        _git("worktree", "remove", "--force", str(path), check=False)
        _git("worktree", "prune", check=False)
        shutil.rmtree(parent, ignore_errors=True)
