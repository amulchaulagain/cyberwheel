"""CLI for the environment profiler: ``python3 -m cyberwheel.profiler``.

Default run executes the deterministic scenarios (network-build, sim-step,
rl-step) and prints per-phase tables plus cProfile hotspots. ``train`` is
opt-in (slower; function-level only).

Baseline workflow (same-machine, parent-commit convention -- mirrors
``cyberwheel/tests/baselines/baseline.json``):

    python3 -m cyberwheel.profiler --record-baseline   # commit alongside perf changes
    python3 -m cyberwheel.profiler --check             # exit 2 on phase regression

Exit codes: 0 ok / 2 perf regression / 3 profiler error.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from cyberwheel.profiler import baseline_io, report
from cyberwheel.profiler.scenarios import (
    DEFAULT_SCENARIOS,
    SCENARIOS,
    ScenarioOptions,
)
from cyberwheel.tests.framework import gitio
from cyberwheel.tests.framework.baseline import fingerprint

EXIT_OK = 0
EXIT_REGRESSION = 2
EXIT_ERROR = 3

# Metric-name prefix per scenario, used to scope --check to what actually ran.
_METRIC_PREFIXES = {
    "network-build": "network_build/",
    "sim-step": "sim_step/",
    "rl-step": "rl_step/",
    "train": "train/",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m cyberwheel.profiler",
        description="Profile the Cyberwheel environment (phases, hotspots, baselines).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[*SCENARIOS, "all"],
        help="scenario to run (repeatable); default: network-build, sim-step, rl-step",
    )
    parser.add_argument(
        "--network", default="15-host-network.yaml", help="network config filename"
    )
    parser.add_argument(
        "--env-config",
        default="train_rl_red_agent_vs_rl_blue.yaml",
        help="environment config filename (for rl-step and train)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--network-size",
        choices=["small", "medium", "large"],
        help="override network_size_compatibility (needed to profile networks "
        "larger than the env config's default compatibility size)",
    )
    parser.add_argument("--steps", type=int, help="override rl-step steps per sample")
    parser.add_argument("--quick", action="store_true", help="smaller sample sizes")
    parser.add_argument("--top", type=int, default=12, help="hotspot rows per table")
    parser.add_argument(
        "--no-hotspots", action="store_true", help="skip cProfile passes"
    )
    parser.add_argument("--json", type=Path, help="write the full JSON report here")
    parser.add_argument(
        "--record-baseline",
        action="store_true",
        help="write measured metrics to the committed profile baseline",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed profile baseline; exit 2 on regression",
    )
    parser.add_argument("--baseline", type=Path, help="override the baseline path")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.25,
        help="relative regression tolerance for --check (default 0.25)",
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.record_baseline and args.check:
        print("--record-baseline and --check are mutually exclusive", file=sys.stderr)
        return EXIT_ERROR

    scenario_names = list(dict.fromkeys(args.scenario or DEFAULT_SCENARIOS))
    if "all" in scenario_names:
        scenario_names = list(SCENARIOS)

    options = ScenarioOptions(
        env_config=args.env_config,
        network_config=args.network,
        seed=args.seed,
        quick=args.quick,
        top=args.top,
        with_hotspots=not args.no_hotspots,
        rl_steps=args.steps,
        network_size=args.network_size,
    )

    print(report.header(scenario_names, options))
    results = []
    metrics: dict = {}
    try:
        for name in scenario_names:
            start = time.perf_counter()
            result = SCENARIOS[name](options)
            result.params["scenario_wall_s"] = round(time.perf_counter() - start, 2)
            results.append(result)
            metrics.update(result.all_metrics())
            print(report.render_scenario(result))
    except Exception as error:  # surface, then fail with the framework-error code
        print(f"profiler error in scenario run: {error!r}", file=sys.stderr)
        raise

    exit_code = EXIT_OK
    comparison_rows = None
    baseline_source = None
    if args.check:
        doc, baseline_source = baseline_io.load(args.baseline)
        if doc is None:
            print(
                f"no profile baseline to check against ({baseline_source})",
                file=sys.stderr,
            )
            return EXIT_ERROR
        prefixes = tuple(_METRIC_PREFIXES[name] for name in scenario_names)
        baseline_metrics = {
            name: payload
            for name, payload in doc.get("metrics", {}).items()
            if name.startswith(prefixes)
        }
        comparisons = baseline_io.compare(baseline_metrics, metrics, args.tolerance)
        mismatches = baseline_io.fingerprint_mismatches(doc)
        print(
            report.render_comparison(
                comparisons, baseline_source, args.tolerance, mismatches
            )
        )
        comparison_rows = [c.to_dict() for c in comparisons]
        if baseline_io.has_regression(comparisons):
            exit_code = EXIT_REGRESSION

    if args.record_baseline:
        path = baseline_io.record(metrics, args.baseline)
        print(f"profile baseline recorded to {path}")

    if args.json:
        document = {
            "schema_version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "commit": gitio.current_commit(),
            "dirty": gitio.is_dirty(),
            "python": platform.python_version(),
            "environment": fingerprint(),
            "options": vars(args)
            | {
                "json": str(args.json),
                "baseline": str(args.baseline) if args.baseline else None,
            },
            "scenarios": {
                result.name: {
                    "params": result.params,
                    "metrics": result.metrics,
                    "phase_metrics": result.phase_metrics,
                    "phases": result.phases,
                    "hotspots": result.hotspots,
                    "notes": result.notes,
                }
                for result in results
            },
            "metrics": metrics,
            "comparison": comparison_rows,
            "exit_code": exit_code,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(document, indent=2) + "\n")
        print(f"JSON report written to {args.json}")

    return exit_code
