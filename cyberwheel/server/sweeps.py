"""Experiment sweeps: a group of training runs over a parameter grid.

One ``sweep.json`` per sweep under ``data/frontend/sweeps/<sweep_id>/`` holds
the base config, the grid, and the child run ids. The child runs are ordinary
entries in the run registry (tagged with ``sweep_id``); the sweep record is
just the grouping + provenance. Atomic writes (tmp + rename), mirroring the
run registry.
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

from cyberwheel.server.paths import SWEEPS_DIR
from cyberwheel.server.validation import require

MAX_CELLS = 16


def sweep_dir(sweep_id: str) -> Path:
    return SWEEPS_DIR / sweep_id


def save_sweep(record: dict) -> None:
    directory = sweep_dir(record["id"])
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / "sweep.json.tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, directory / "sweep.json")


def load_sweep(sweep_id: str) -> dict | None:
    path = sweep_dir(sweep_id) / "sweep.json"
    if not path.is_file():
        return None
    with open(path) as f:
        return json.load(f)


def list_sweeps() -> list[dict]:
    records = []
    if SWEEPS_DIR.is_dir():
        for entry in SWEEPS_DIR.iterdir():
            path = entry / "sweep.json"
            if not path.is_file():
                continue
            try:
                with open(path) as f:
                    records.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return records


def expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of a {param: [values]} grid → list of param dicts.

    Sorted keys keep cell ordering deterministic. Rejects empty value lists and
    grids larger than MAX_CELLS.
    """
    require(isinstance(grid, dict) and grid, "grid must be a non-empty object")
    keys = sorted(grid)
    value_lists = []
    for key in keys:
        values = grid[key]
        require(
            isinstance(values, list) and values,
            f"grid['{key}'] must be a non-empty list of values",
        )
        value_lists.append(values)
    total = 1
    for values in value_lists:
        total *= len(values)
    require(total <= MAX_CELLS, f"grid expands to {total} cells (max {MAX_CELLS})")
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def rollup_status(child_statuses: list[str]) -> str:
    if any(s in ("queued", "running") for s in child_statuses):
        return "running"
    if child_statuses and all(s == "succeeded" for s in child_statuses):
        return "succeeded"
    return "failed"
