"""Training metrics straight from the TensorBoard event files the trainer
already writes live to ``data/runs/<experiment_name>/``.

EventAccumulator reloads are incremental but not free, so each run gets a
cached accumulator with a minimum interval between reloads; responses are
downsampled server-side (the UI never needs more points than pixels) and
support ``after_step`` slicing so a 1-2 s poll costs almost nothing.
"""

from __future__ import annotations

import threading
import time

from cyberwheel.server.paths import RUNS_DIR
from cyberwheel.server.validation import not_found

RELOAD_INTERVAL_S = 1.0


class _RunMetrics:
    def __init__(self, experiment_name: str):
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )

        self.accumulator = EventAccumulator(
            str(RUNS_DIR / experiment_name), size_guidance={"scalars": 0}
        )
        self.last_reload = 0.0
        self.lock = threading.Lock()

    def reload(self) -> None:
        with self.lock:
            now = time.time()
            if now - self.last_reload >= RELOAD_INTERVAL_S:
                self.accumulator.Reload()
                self.last_reload = now


_cache: dict[str, _RunMetrics] = {}
_cache_lock = threading.Lock()


def _metrics(experiment_name: str) -> _RunMetrics:
    if not (RUNS_DIR / experiment_name).is_dir():
        raise not_found(f"no training metrics for {experiment_name!r}")
    with _cache_lock:
        entry = _cache.get(experiment_name)
        if entry is None:
            entry = _RunMetrics(experiment_name)
            _cache[experiment_name] = entry
    entry.reload()
    return entry


def _downsample(points: list, max_points: int) -> list:
    if len(points) <= max_points:
        return points
    stride = (len(points) + max_points - 1) // max_points
    sampled = points[::stride]
    if sampled[-1] is not points[-1]:
        sampled.append(points[-1])  # always keep the freshest point
    return sampled


def summary(experiment_name: str) -> dict:
    entry = _metrics(experiment_name)
    tags = sorted(entry.accumulator.Tags().get("scalars", []))
    grouped: dict[str, list[str]] = {}
    for tag in tags:
        grouped.setdefault(tag.split("/", 1)[0], []).append(tag)
    last_step = 0
    for tag in tags:
        events = entry.accumulator.Scalars(tag)
        if events:
            last_step = max(last_step, events[-1].step)
    return {"tags": grouped, "last_step": last_step}


def scalars(
    experiment_name: str,
    tags: list[str],
    after_step: int = -1,
    max_points: int = 1000,
) -> dict:
    entry = _metrics(experiment_name)
    available = set(entry.accumulator.Tags().get("scalars", []))
    series = {}
    for tag in tags:
        if tag not in available:
            series[tag] = []
            continue
        points = [
            [event.step, round(event.wall_time, 3), float(event.value)]
            for event in entry.accumulator.Scalars(tag)
            if event.step > after_step
        ]
        series[tag] = _downsample(points, max_points)
    return {"series": series}


def last_value(experiment_name: str, tag: str):
    """Freshest value of one tag, or None — cheap helper for run listings."""
    try:
        entry = _metrics(experiment_name)
    except Exception:
        return None
    try:
        events = entry.accumulator.Scalars(tag)
    except KeyError:
        return None
    return float(events[-1].value) if events else None


def last_step(experiment_name: str) -> int | None:
    try:
        entry = _metrics(experiment_name)
    except Exception:
        return None
    steps = []
    for tag in entry.accumulator.Tags().get("scalars", []):
        events = entry.accumulator.Scalars(tag)
        if events:
            steps.append(events[-1].step)
    return max(steps) if steps else None
