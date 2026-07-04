"""Function-level hotspot capture via cProfile.

Produces structured rows (JSON-friendly) rather than raw pstats text, so
reports can rank by internal time and cumulative time and shorten paths.
"""

from __future__ import annotations

import cProfile
import pstats
from pathlib import Path
from typing import Callable


def _shorten(path: str) -> str:
    index = path.find("/cyberwheel/")
    if index != -1:
        return path[index + 1 :]
    if "/site-packages/" in path:
        return path.split("/site-packages/", 1)[1]
    if "/lib/python" in path:
        return path.split("/lib/python", 1)[1].split("/", 1)[-1]
    return path


def _rows_from_stats(stats: pstats.Stats, top: int) -> dict:
    entries = []
    for (filename, lineno, func), (
        cc,
        ncalls,
        tottime,
        cumtime,
        _,
    ) in stats.stats.items():
        entries.append(
            {
                "function": f"{_shorten(filename)}:{lineno}({func})",
                "calls": ncalls,
                "internal_s": round(tottime, 6),
                "cumulative_s": round(cumtime, 6),
            }
        )
    by_internal = sorted(entries, key=lambda e: e["internal_s"], reverse=True)[:top]
    by_cumulative = sorted(entries, key=lambda e: e["cumulative_s"], reverse=True)[:top]
    return {"by_internal": by_internal, "by_cumulative": by_cumulative}


def profile_callable(fn: Callable[[], None], top: int) -> dict:
    """Run ``fn`` under cProfile and return top rows."""
    profile = cProfile.Profile()
    profile.enable()
    try:
        fn()
    finally:
        profile.disable()
    return _rows_from_stats(pstats.Stats(profile), top)


def load_stats_file(path: Path, top: int) -> dict:
    """Extract top rows from a cProfile stats dump (e.g. from a subprocess)."""
    return _rows_from_stats(pstats.Stats(str(path)), top)
