"""Plain-text rendering for profiler results (tables aligned for terminals)."""

from __future__ import annotations

import platform
import time

from cyberwheel.profiler.scenarios import ScenarioResult
from cyberwheel.tests.framework import gitio

RULE = "-" * 78


def header(scenario_names, options) -> str:
    commit = gitio.current_commit() or "unknown"
    commit = f"{commit[:12]}+dirty" if gitio.is_dirty() else commit[:12]
    lines = [
        f"cyberwheel profiler | commit {commit} | python {platform.python_version()} "
        f"| {platform.system()}/{platform.machine()}",
        f"scenarios: {', '.join(scenario_names)} | network {options.network_config} "
        f"| env {options.env_config} | seed {options.seed}"
        + (" | quick" if options.quick else ""),
        RULE,
    ]
    return "\n".join(lines)


def _fmt_ms(value: float) -> str:
    if value >= 100:
        return f"{value:10.1f}"
    if value >= 1:
        return f"{value:10.3f}"
    return f"{value:10.4f}"


def render_scenario(result: ScenarioResult) -> str:
    lines = [f"== {result.name}  {result.params}"]
    for note in result.notes:
        lines.append(f"   note: {note}")

    if result.metrics:
        lines.append("   metrics:")
        for name, payload in sorted(result.metrics.items()):
            samples = payload.get("samples")
            sample_str = (
                "  samples=[" + ", ".join(f"{s:.4f}" for s in samples) + "]"
                if samples
                else ""
            )
            lines.append(
                f"     {name:<44} {_fmt_ms(payload['value'])} {payload['unit']}{sample_str}"
            )

    if result.phases:
        lines.append("")
        lines.append(
            f"   {'phase':<24} {'excl ms/step':>13} {'incl ms/step':>13} "
            f"{'% of step':>10} {'calls/step':>11}"
        )
        for row in sorted(
            result.phases, key=lambda r: r["exclusive_ms_per_unit"], reverse=True
        ):
            pct = (
                f"{row['pct_of_anchor']:9.1f}%"
                if row["pct_of_anchor"] is not None
                else "         -"
            )
            lines.append(
                f"   {row['phase']:<24} {_fmt_ms(row['exclusive_ms_per_unit'])}    "
                f"{_fmt_ms(row['inclusive_ms_per_unit'])}    {pct} {row['calls_per_unit']:11.2f}"
            )

    if result.hotspots:
        for key, title in (
            ("by_internal", "top functions by internal time"),
            ("by_cumulative", "top functions by cumulative time"),
        ):
            lines.append("")
            lines.append(f"   {title}:")
            lines.append(f"   {'internal s':>10} {'cumul s':>9} {'calls':>9}  function")
            for row in result.hotspots[key]:
                lines.append(
                    f"   {row['internal_s']:10.4f} {row['cumulative_s']:9.4f} "
                    f"{row['calls']:9d}  {row['function']}"
                )
    lines.append(RULE)
    return "\n".join(lines)


def render_comparison(
    comparisons, baseline_source: str, tolerance: float, mismatches
) -> str:
    lines = [
        f"Baseline check (baseline: {baseline_source}, tolerance: {tolerance:.0%})",
    ]
    for mismatch in mismatches:
        lines.append(f"  WARNING fingerprint mismatch -- {mismatch}")
    lines.append(
        f"{'metric':<44} {'baseline':>10} {'current':>10} {'delta':>8}  verdict"
    )
    lines.append(RULE)
    for c in comparisons:
        base = (
            _fmt_ms(c.baseline_value) if c.baseline_value is not None else "         -"
        )
        cur = _fmt_ms(c.current_value) if c.current_value is not None else "         -"
        delta = f"{c.delta_pct:+7.1f}%" if c.delta_pct is not None else "       -"
        lines.append(f"{c.name:<44} {base} {cur} {delta}  {c.verdict}")
    lines.append(RULE)
    return "\n".join(lines)


def run_id() -> str:
    return time.strftime("PROFILE_%Y%m%d%H%M%S")
