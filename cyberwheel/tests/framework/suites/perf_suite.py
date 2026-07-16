"""Perf suite: run the standalone benchmarks and collect their metrics.

Each benchmark is executed in its own subprocess with ``PYTHONPATH`` pointed
at a code root, so the same benchmark scripts (always taken from this
checkout) can measure either this repo or a parent-commit worktree — that is
how ``--compare-rev`` produces two same-machine measurements to compare.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from cyberwheel.tests.framework.core import (
    REPO_ROOT,
    Context,
    Outcome,
    Registry,
    Status,
    TestCase,
    TestFailure,
    run_cli,
)

SUITE = "perf"

_BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
_BENCHMARKS = (
    ("bench_network_build.py", 600.0),
    ("bench_sim_step.py", 900.0),
    ("bench_rl_step.py", 900.0),
    ("bench_rl_step_green.py", 900.0),
    ("bench_train_sps.py", 1200.0),
)


def run_benchmark(
    script: str,
    timeout: float,
    quick: bool,
    code_root: Path = REPO_ROOT,
) -> dict:
    """Run one benchmark against ``code_root``; return its metric payload."""
    argv = [str(_BENCH_DIR / script)]
    if quick:
        argv.append("--quick")
    proc = run_cli(
        argv,
        timeout=timeout,
        env_extra={
            "PYTHONPATH": str(code_root),
            "CYBERWHEEL_DETERMINISTIC": "true",
        },
        cwd=code_root,
    )
    if proc.returncode != 0:
        raise TestFailure(
            f"{script} exited {proc.returncode}; stderr tail: {proc.stderr[-600:]}"
        )
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError:
        raise TestFailure(
            f"{script} did not print JSON on its last line: {last_line[:200]!r}"
        )
    for key in ("metric", "value", "unit", "higher_is_better", "samples"):
        if key not in payload:
            raise TestFailure(f"{script} payload missing key {key!r}")
    return payload


def measure_metrics(
    quick: bool, code_root: Path = REPO_ROOT, skip_failing: bool = False
) -> dict:
    """Run all benchmarks against ``code_root`` (used for --compare-rev).

    With ``skip_failing``, a benchmark that cannot run at ``code_root`` is
    skipped instead of raising — e.g. when the compared (parent) revision
    still has a bug that this checkout's benchmark would trip on. Its metric
    is then absent and shows up as NEW in the comparison rather than gating.
    """
    metrics = {}
    for script, timeout in _BENCHMARKS:
        try:
            payload = run_benchmark(script, timeout, quick, code_root)
        except TestFailure as failure:
            if not skip_failing:
                raise
            print(
                f"NOTE: {script} could not run against {code_root} "
                f"({failure}); reporting its metric as NEW.",
                flush=True,
            )
            continue
        metrics[payload["metric"]] = {
            "value": payload["value"],
            "samples": payload["samples"],
            "unit": payload["unit"],
            "higher_is_better": payload["higher_is_better"],
        }
    return metrics


def _bench_case(ctx: Context, script: str, timeout: float) -> Outcome:
    payload = run_benchmark(script, timeout, ctx.quick)
    ctx.perf_metrics[payload["metric"]] = {
        "value": payload["value"],
        "samples": payload["samples"],
        "unit": payload["unit"],
        "higher_is_better": payload["higher_is_better"],
    }
    return Outcome(
        Status.PASS,
        f"{payload['metric']} = {payload['value']:.2f} {payload['unit']}",
        {"samples": payload["samples"]},
    )


def register(registry: Registry, ctx: Context) -> None:
    for script, timeout in _BENCHMARKS:
        registry.add(
            TestCase(
                name=f"perf:{script.removesuffix('.py')}",
                suite=SUITE,
                fn=(lambda s=script, t=timeout: _bench_case(ctx, s, t)),
                timeout_s=timeout,
            )
        )
