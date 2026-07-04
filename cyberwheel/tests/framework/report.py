"""Console and JSON reporting for the test framework."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from cyberwheel.tests.framework.core import GATING_STATUSES, Status, TestResult

_STATUS_DECOR = {
    Status.PASS: "ok  ",
    Status.FAIL: "FAIL",
    Status.ERROR: "ERR ",
    Status.SKIP: "skip",
    Status.XFAIL: "xfl ",
    Status.XPASS_WARN: "XPS!",
    Status.INFO: "info",
}


def print_result_line(result: TestResult, verbose: bool = False) -> None:
    decor = _STATUS_DECOR[result.status]
    line = f"[{decor}] {result.name:<58} {result.duration_s:7.2f}s"
    if result.message:
        line += f"  {result.message}"
    print(line, flush=True)
    if verbose and result.details.get("traceback_tail"):
        print(result.details["traceback_tail"], flush=True)


def print_summary(results: list[TestResult]) -> None:
    counts = Counter(r.status for r in results)
    total = len(results)
    parts = [f"{counts[s]} {s.value.lower()}" for s in Status if counts[s]]
    gating = [r for r in results if r.status in GATING_STATUSES]
    print("\n" + "-" * 78)
    print(f"{total} cases: " + ", ".join(parts))
    if gating:
        print("\nGating failures:")
        for r in gating:
            print(f"  [{r.status.value}] {r.name}: {r.message}")
    print("-" * 78, flush=True)


def print_perf_comparison(
    comparisons: list, baseline_source: str, tolerance: float
) -> None:
    print(
        f"\nPerf comparison (baseline: {baseline_source}, tolerance: {tolerance:.0%})"
    )
    header = f"{'metric':<28} {'baseline':>12} {'current':>12} {'delta':>9}  verdict"
    print(header)
    print("-" * len(header))
    for c in comparisons:
        base = f"{c.baseline_value:.2f}" if c.baseline_value is not None else "-"
        cur = f"{c.current_value:.2f}" if c.current_value is not None else "-"
        delta = f"{c.delta_pct:+.1f}%" if c.delta_pct is not None else "-"
        print(f"{c.name:<28} {base:>12} {cur:>12} {delta:>9}  {c.verdict}")
    print(flush=True)


def write_json_report(
    path: Path,
    results: list[TestResult],
    header: dict,
    perf: dict | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **header,
        "results": [r.to_dict() for r in results],
    }
    if perf is not None:
        payload["perf"] = perf
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"JSON report written to {path}", flush=True)
