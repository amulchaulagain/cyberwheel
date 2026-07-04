"""Perf baseline storage and comparison.

Parent-commit convention
------------------------
The committed baseline (``cyberwheel/tests/baselines/baseline.json``) is
re-recorded with ``--record-baseline`` in the SAME commit as any change that
intentionally shifts performance. Therefore, while developing, the
working-tree baseline always holds the parent commit's recorded results, and
for any commit C, ``git show C~1:<baseline>`` is exactly what C was gated
against. ``--baseline-rev HEAD~1`` makes that explicit; CI avoids
cross-machine noise entirely by re-measuring the parent on the same runner
via ``--compare-rev``.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cyberwheel.tests.framework import gitio
from cyberwheel.tests.framework.core import BASELINE_RELPATH, REPO_ROOT

SCHEMA_VERSION = 1

DEFAULT_BASELINE_PATH = REPO_ROOT / BASELINE_RELPATH


def fingerprint() -> dict:
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


@dataclass
class MetricComparison:
    name: str
    baseline_value: Optional[float]
    current_value: Optional[float]
    unit: str
    delta_pct: Optional[float]  # positive = current is "more" than baseline
    verdict: str  # OK | REGRESSION | IMPROVED | NEW | MISSING

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "unit": self.unit,
            "delta_pct": self.delta_pct,
            "verdict": self.verdict,
        }


def load_baseline(
    explicit_path: Optional[Path] = None,
    rev: Optional[str] = None,
) -> tuple[Optional[dict], str]:
    """Resolve the baseline document. Returns (doc-or-None, source-label)."""
    if rev:
        text = gitio.show_file(rev, BASELINE_RELPATH)
        if text is None:
            return None, f"git:{rev} (no baseline at that revision)"
        return _parse(text), f"git:{rev}:{BASELINE_RELPATH}"
    path = explicit_path or DEFAULT_BASELINE_PATH
    if not path.is_file():
        return None, f"{path} (missing)"
    return _parse(path.read_text()), str(path)


def _parse(text: str) -> dict:
    doc = json.loads(text)
    version = doc.get("schema_version")
    if version != SCHEMA_VERSION:
        raise RuntimeError(
            f"baseline schema_version {version!r} unsupported (expected {SCHEMA_VERSION})"
        )
    return doc


def fingerprint_mismatches(doc: dict) -> list[str]:
    recorded = doc.get("environment", {})
    current = fingerprint()
    return [
        f"{key}: baseline={recorded.get(key)!r} current={current[key]!r}"
        for key in current
        if recorded.get(key) != current[key]
    ]


def compare(
    baseline_metrics: dict,
    current_metrics: dict,
    tolerance: float,
) -> list[MetricComparison]:
    """Compare metric dicts of ``{name: {value, unit, higher_is_better, ...}}``."""
    comparisons: list[MetricComparison] = []
    for name in sorted(set(baseline_metrics) | set(current_metrics)):
        base = baseline_metrics.get(name)
        cur = current_metrics.get(name)
        if base is None:
            comparisons.append(
                MetricComparison(
                    name,
                    None,
                    cur["value"],
                    cur.get("unit", ""),
                    None,
                    "NEW",
                )
            )
            continue
        if cur is None:
            comparisons.append(
                MetricComparison(
                    name,
                    base["value"],
                    None,
                    base.get("unit", ""),
                    None,
                    "MISSING",
                )
            )
            continue
        base_value = float(base["value"])
        cur_value = float(cur["value"])
        higher_is_better = bool(base.get("higher_is_better", True))
        delta_pct = (
            ((cur_value - base_value) / base_value * 100.0) if base_value else None
        )
        if base_value <= 0:
            verdict = "OK"  # degenerate baseline; nothing sane to gate on
        else:
            ratio = cur_value / base_value
            if higher_is_better:
                regressed = ratio < 1.0 - tolerance
                improved = ratio > 1.0 + tolerance
            else:
                regressed = ratio > 1.0 + tolerance
                improved = ratio < 1.0 - tolerance
            verdict = "REGRESSION" if regressed else "IMPROVED" if improved else "OK"
        comparisons.append(
            MetricComparison(
                name,
                base_value,
                cur_value,
                cur.get("unit", ""),
                delta_pct,
                verdict,
            )
        )
    return comparisons


def has_regression(comparisons: list[MetricComparison]) -> bool:
    return any(c.verdict in ("REGRESSION", "MISSING") for c in comparisons)


def record(metrics: dict, path: Optional[Path] = None) -> Path:
    path = path or DEFAULT_BASELINE_PATH
    commit = gitio.current_commit() or "unknown"
    if gitio.is_dirty():
        commit = f"dirty+{commit}"
    doc = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recorded_at_commit": commit,
        "environment": fingerprint(),
        "metrics": {name: metrics[name] for name in sorted(metrics)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path
