"""CLI for the Cyberwheel test framework: ``python -m cyberwheel.tests``.

Exit codes:
    0  everything passed (within perf tolerance)
    1  config/smoke failure or benchmark error (hard, no tolerance)
    2  perf regression beyond tolerance, or missing baseline with
       --require-baseline
    3  usage or framework error
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import traceback
from pathlib import Path

from cyberwheel.tests.framework import artifacts, baseline as baseline_mod, gitio
from cyberwheel.tests.framework.core import (
    GATING_STATUSES,
    REPO_ROOT,
    SAFE_ENV,
    Context,
    Registry,
    run_cases,
)
from cyberwheel.tests.framework.report import (
    print_perf_comparison,
    print_result_line,
    print_summary,
    write_json_report,
)
from cyberwheel.tests.framework.suites import (
    config_suite,
    frontend_suite,
    perf_suite,
    smoke_suite,
)

_SUITE_MODULES = {
    "config": config_suite,
    "smoke": smoke_suite,
    "frontend": frontend_suite,
    "perf": perf_suite,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cyberwheel.tests",
        description="Cyberwheel test framework: config validation, e2e smoke tests, perf gate.",
    )
    parser.add_argument(
        "--suite",
        choices=[*_SUITE_MODULES, "all"],
        default="all",
        help="which suite to run (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="skip large networks; fewer benchmark samples",
    )
    parser.add_argument(
        "--filter", metavar="SUBSTR", help="only run cases whose name contains SUBSTR"
    )
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument(
        "--json", metavar="PATH", type=Path, help="write a machine-readable JSON report"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.20,
        help="perf regression tolerance as a fraction (default: 0.20)",
    )
    parser.add_argument(
        "--baseline",
        metavar="PATH",
        type=Path,
        help="explicit baseline file (default: committed baseline)",
    )
    parser.add_argument(
        "--baseline-rev",
        metavar="REV",
        help="read the baseline as committed at REV (git show)",
    )
    parser.add_argument(
        "--record-baseline",
        action="store_true",
        help="overwrite the baseline file with this run's metrics",
    )
    parser.add_argument(
        "--compare-rev",
        metavar="REV",
        help="also measure REV in a temporary worktree and compare "
        "run-vs-run on this machine (ignores the baseline file)",
    )
    parser.add_argument(
        "--require-baseline",
        action="store_true",
        help="fail (exit 2) if no baseline can be resolved",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="do not delete TEST_* artifacts afterwards",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 3
    try:
        return _run(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 3
    except Exception:
        traceback.print_exc()
        return 3


def _run(args: argparse.Namespace) -> int:
    os.environ.update(SAFE_ENV)

    ctx = Context(
        run_id=artifacts.test_run_id(),
        quick=args.quick,
        filter=args.filter,
        verbose=args.verbose,
        keep_artifacts=args.keep_artifacts,
    )
    registry = Registry()
    suites = list(_SUITE_MODULES) if args.suite == "all" else [args.suite]
    for name in suites:
        _SUITE_MODULES[name].register(registry, ctx)

    if args.list:
        for case in registry.cases:
            marks = []
            if case.quick_skip:
                marks.append("quick-skip")
            if case.default_skip_reason:
                marks.append("default-skip")
            if case.known_issue:
                marks.append("known-issue")
            if case.depends_on:
                marks.append(f"after {case.depends_on}")
            suffix = f"  [{', '.join(marks)}]" if marks else ""
            print(f"{case.name}{suffix}")
        print(f"\n{len(registry.cases)} cases")
        return 0

    commit = gitio.current_commit() or "unknown"
    dirty = gitio.is_dirty()
    print(
        f"cyberwheel test framework | commit {commit[:12]}{'+dirty' if dirty else ''} "
        f"| python {platform.python_version()} | {platform.system()}/{platform.machine()} "
        f"| run id {ctx.run_id}"
    )
    print(
        f"suites: {', '.join(suites)}"
        + (f" | filter: {args.filter}" if args.filter else "")
        + (" | quick" if args.quick else "")
    )
    print("-" * 78, flush=True)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    perf_report: dict | None = None
    exit_code = 0

    try:
        results = run_cases(
            registry.cases,
            quick=args.quick,
            fltr=args.filter,
            on_result=lambda r: print_result_line(r, verbose=args.verbose),
        )

        if any(r.status in GATING_STATUSES for r in results):
            exit_code = 1

        if "perf" in suites and ctx.perf_metrics:
            perf_report = {"metrics": ctx.perf_metrics}
            perf_exit = _gate_perf(args, ctx, perf_report)
            if exit_code == 0 and perf_exit:
                exit_code = perf_exit

            if args.record_baseline:
                path = baseline_mod.record(ctx.perf_metrics, args.baseline)
                print(f"Baseline recorded to {path}", flush=True)
                perf_report["recorded_to"] = str(path)
        elif (
            "perf" in suites
            and not ctx.perf_metrics
            and exit_code == 0
            and not args.filter
        ):
            # perf suite selected but produced nothing (all benches failed).
            exit_code = 1

        print_summary(results)
    finally:
        if not args.keep_artifacts:
            removed = artifacts.cleanup(REPO_ROOT)
            if removed and args.verbose:
                print(f"cleaned {len(removed)} TEST_* artifacts")

    if args.json:
        write_json_report(
            args.json,
            results,
            header={
                "started_at": started_at,
                "commit": commit,
                "dirty": dirty,
                "suites": suites,
                "args": {
                    "quick": args.quick,
                    "filter": args.filter,
                    "tolerance": args.tolerance,
                    "baseline_rev": args.baseline_rev,
                    "compare_rev": args.compare_rev,
                },
            },
            perf=perf_report,
        )

    print(f"exit code: {exit_code}")
    return exit_code


def _gate_perf(args: argparse.Namespace, ctx: Context, perf_report: dict) -> int:
    """Compare current metrics against the parent's; return 0 or 2."""
    if args.compare_rev:
        changed = gitio.deps_changed_since(args.compare_rev)
        if changed:
            print(
                f"WARNING: {changed} differ from {args.compare_rev}; the parent "
                "is measured against this checkout's installed dependencies.",
                flush=True,
            )
        print(f"Measuring {args.compare_rev} in a temporary worktree ...", flush=True)
        with gitio.worktree(args.compare_rev) as parent_root:
            try:
                parent_metrics = perf_suite.measure_metrics(
                    ctx.quick, parent_root, skip_failing=True
                )
            finally:
                artifacts.cleanup(parent_root)
        source = f"same-machine re-measurement of {args.compare_rev}"
        baseline_metrics = parent_metrics
        fingerprint_warnings: list[str] = []
    else:
        doc, source = baseline_mod.load_baseline(args.baseline, args.baseline_rev)
        if doc is None:
            message = f"No baseline available ({source})."
            if args.require_baseline:
                print(
                    f"{message} Failing because --require-baseline is set.", flush=True
                )
                perf_report["comparison"] = {
                    "source": source,
                    "error": "missing baseline",
                }
                return 2
            print(
                f"{message} Perf run is informational; record one with "
                "--record-baseline.",
                flush=True,
            )
            perf_report["comparison"] = {"source": source, "informational": True}
            return 0
        baseline_metrics = doc.get("metrics", {})
        fingerprint_warnings = baseline_mod.fingerprint_mismatches(doc)

    if fingerprint_warnings:
        print("WARNING: baseline was recorded on a different environment:", flush=True)
        for warning in fingerprint_warnings:
            print(f"  {warning}", flush=True)

    if args.filter:
        # A filtered run measures a subset; don't flag unmeasured metrics.
        baseline_metrics = {
            k: v for k, v in baseline_metrics.items() if k in ctx.perf_metrics
        }

    comparisons = baseline_mod.compare(
        baseline_metrics, ctx.perf_metrics, args.tolerance
    )
    print_perf_comparison(comparisons, source, args.tolerance)
    perf_report["comparison"] = {
        "source": source,
        "tolerance": args.tolerance,
        "fingerprint_warnings": fingerprint_warnings,
        "results": [c.to_dict() for c in comparisons],
    }
    if baseline_mod.has_regression(comparisons):
        print(
            "PERF GATE FAILED: regression beyond tolerance "
            f"({args.tolerance:.0%}). If intentional, re-record with "
            "--record-baseline and commit the baseline with this change.",
            flush=True,
        )
        return 2
    return 0
