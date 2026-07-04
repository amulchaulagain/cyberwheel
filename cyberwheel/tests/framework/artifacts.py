"""Naming and cleanup of test artifacts.

Everything the framework (or a subprocess it launches) writes under
``cyberwheel/data/`` uses a ``TEST_`` name prefix, so cleanup can remove
exactly those entries and nothing else.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from cyberwheel.tests.framework.core import REPO_ROOT

_DATA_SUBDIRS = ("models", "runs", "action_logs", "graphs")


def test_run_id() -> str:
    return f"TEST_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}"


def cleanup(root: Path = REPO_ROOT) -> list[str]:
    """Remove ``TEST_*`` artifacts under ``<root>/cyberwheel/data``.

    ``root`` is parameterized so perf comparisons can clean a parent-commit
    worktree as well as this repo. Returns the removed paths.
    """
    removed: list[str] = []
    for sub in _DATA_SUBDIRS:
        base = root / "cyberwheel" / "data" / sub
        if not base.is_dir():
            continue
        for entry in base.glob("TEST_*"):
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            removed.append(str(entry))
    return removed
