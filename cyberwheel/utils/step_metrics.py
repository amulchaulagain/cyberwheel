"""Evaluation summary statistics (``data/action_logs/<graph_name>.summary.json``).

Shared by the evaluator, which writes the summary at the end of a run, and by
tests. Deliberately stdlib-only (json/math/os) so importing it never drags in
torch/pandas — the server reads the JSON file directly and must not import
this module (``cyberwheel.utils.__init__`` pulls in torch).
"""

import json
import math
import os

SUMMARY_FORMAT_VERSION = 1

# Two-sided 95% Student-t critical values for df 1..30; larger df ~ z = 1.96.
_T_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t_critical_95(df: int) -> float:
    """Two-sided 95% Student-t critical value for the given degrees of freedom."""
    if df < 1:
        return 1.96
    return _T_95.get(df, 1.96)


def mean_std_ci95(values: list) -> dict:
    """Stat block {mean, std, min, max, ci95_lo, ci95_hi, n} for a sample.

    Sample std (ddof=1); n==1 collapses the CI to the mean; n==0 yields nulls.
    """
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "min": None, "max": None,
                "ci95_lo": None, "ci95_hi": None, "n": 0}
    mean = sum(values) / n
    if n == 1:
        std = 0.0
        half = 0.0
    else:
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        half = t_critical_95(n - 1) * std / math.sqrt(n)
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "ci95_lo": round(mean - half, 4),
        "ci95_hi": round(mean + half, 4),
        "n": n,
    }


def build_evaluation_summary(
    *,
    seeds: list,
    explicit_seeds: bool,
    deterministic: bool,
    num_episodes: int,
    num_steps: int,
    per_episode: list,
    metric_names: list,
    graph_name: str,
    experiment_name: str,
) -> dict:
    """Aggregate per-episode reward totals into per-seed and overall stat blocks.

    ``per_episode`` holds one dict per episode in seed-block order; seeds are
    grouped positionally (num_episodes-sized slices), which stays correct even
    if the same seed value appears twice.
    """
    per_seed = []
    for i, seed in enumerate(seeds):
        block = per_episode[i * num_episodes:(i + 1) * num_episodes]
        per_seed.append({
            "seed": seed,
            "episodes": len(block),
            "metrics": {
                m: mean_std_ci95([ep[m] for ep in block]) for m in metric_names
            },
        })
    return {
        "format_version": SUMMARY_FORMAT_VERSION,
        "graph_name": graph_name,
        "experiment_name": experiment_name,
        "seeds": list(seeds),
        "explicit_seeds": explicit_seeds,
        "deterministic": deterministic,
        "num_episodes": num_episodes,
        "num_steps": num_steps,
        "total_episodes": len(per_episode),
        "metrics": list(metric_names),
        "per_episode": per_episode,
        "per_seed": per_seed,
        "overall": {
            m: mean_std_ci95([ep[m] for ep in per_episode]) for m in metric_names
        },
    }


def write_summary(path, summary: dict) -> None:
    """Atomically write the summary JSON (tmp + rename) so readers never see a partial file."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(summary, f, indent=2)
    os.replace(tmp, str(path))
