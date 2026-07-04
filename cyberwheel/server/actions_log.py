"""Read evaluation action logs (``data/action_logs/<graph_name>.csv``).

The evaluator appends one row per step while running, so row counts double
as live evaluation progress.
"""

from __future__ import annotations

import csv

from cyberwheel.server.paths import ACTION_LOGS_DIR
from cyberwheel.server.validation import not_found


def _path(graph_name: str):
    return ACTION_LOGS_DIR / f"{graph_name}.csv"


def row_count(graph_name: str) -> int | None:
    path = _path(graph_name)
    if not path.is_file():
        return None
    with open(path, newline="") as f:
        rows = sum(1 for _ in f)
    return max(0, rows - 1)  # header


def actions(graph_name: str, episode: int | None = None) -> dict:
    path = _path(graph_name)
    if not path.is_file():
        raise not_found(f"no action log for {graph_name!r}")
    episodes: list[int] = []
    totals: dict[int, dict] = {}
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        # Only RL agents get per-agent CSV columns (a heuristic red has no
        # policy stepped by the evaluator) — report what actually exists.
        reward_columns = [
            c for c in (reader.fieldnames or []) if c.endswith("_reward")
        ]
        for row in reader:
            try:
                row_episode = int(float(row.get("episode", -1)))
            except (TypeError, ValueError):
                continue
            if row_episode not in totals:
                episodes.append(row_episode)
                totals[row_episode] = {"total": 0.0, "steps": 0}
                totals[row_episode].update({c: 0.0 for c in reward_columns})
            bucket = totals[row_episode]
            bucket["steps"] += 1
            bucket["total"] += _num(row.get("reward"))
            for column in reward_columns:
                bucket[column] += _num(row.get(column))
            if episode is None or row_episode == episode:
                rows.append(row)
    for bucket in totals.values():
        for key in ("total", *reward_columns):
            bucket[key] = round(bucket[key], 4)
    return {
        "episodes": episodes,
        "reward_totals": {str(k): v for k, v in totals.items()},
        "rows": rows,
    }


def _num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
