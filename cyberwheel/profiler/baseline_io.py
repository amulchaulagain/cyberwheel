"""Committed profiler baseline: load, record, and floor-aware comparison.

Reuses the test framework's baseline machinery (same schema, same
parent-commit convention: re-record in the SAME commit as any intentional
perf change). Unlike the perf suite gate, sub-floor metrics are reported but
never fail the check, because at that scale a relative tolerance only trips
on timer noise:

- phase metrics below ``PHASE_GATE_FLOOR_MS`` (instrumented, per-step);
- one-shot ``ms`` metrics below ``ONE_SHOT_GATE_FLOOR_MS`` — these time a
  single invocation, so below ~1 ms allocator/cache-state noise dominates
  (observed 10x sample spread on identical code). Per-step ``ms/step``
  metrics stay gated at any magnitude: they are averaged over thousands of
  iterations and stable even at microsecond scale.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from cyberwheel.tests.framework import baseline as framework_baseline
from cyberwheel.profiler.scenarios import PHASE_GATE_FLOOR_MS

# One-shot ("ms" unit) measurements below this are reported but not gated.
ONE_SHOT_GATE_FLOOR_MS = 1.0

PROFILE_BASELINE_PATH = (
    Path(__file__).resolve().parent / "baselines" / "profile_baseline.json"
)


def load(path: Optional[Path] = None) -> tuple[Optional[dict], str]:
    path = path or PROFILE_BASELINE_PATH
    if not path.is_file():
        return None, f"{path} (missing)"
    doc = json.loads(path.read_text())
    version = doc.get("schema_version")
    if version != framework_baseline.SCHEMA_VERSION:
        raise RuntimeError(
            f"profile baseline schema_version {version!r} unsupported "
            f"(expected {framework_baseline.SCHEMA_VERSION})"
        )
    return doc, str(path)


def record(metrics: dict, path: Optional[Path] = None) -> Path:
    return framework_baseline.record(metrics, path=path or PROFILE_BASELINE_PATH)


def fingerprint_mismatches(doc: dict) -> list[str]:
    return framework_baseline.fingerprint_mismatches(doc)


def compare(baseline_metrics: dict, current_metrics: dict, tolerance: float):
    """Framework comparison plus the sub-floor policy for phase metrics."""
    comparisons = framework_baseline.compare(
        baseline_metrics, current_metrics, tolerance
    )
    for comparison in comparisons:
        if "/phase/" in comparison.name:
            floor = PHASE_GATE_FLOOR_MS
        elif comparison.unit == "ms":
            floor = ONE_SHOT_GATE_FLOOR_MS
        else:
            floor = None
        below_floor = (
            floor is not None
            and (comparison.baseline_value or 0) < floor
            and (comparison.current_value or 0) < floor
        )
        if below_floor and comparison.verdict in ("REGRESSION", "IMPROVED"):
            comparison.verdict = "OK"
        # New/vanished phases come from instrumentation coverage, not perf.
        if "/phase/" in comparison.name and comparison.verdict in ("NEW", "MISSING"):
            comparison.verdict = "OK"
    return comparisons


def has_regression(comparisons) -> bool:
    return any(c.verdict in ("REGRESSION", "MISSING") for c in comparisons)
