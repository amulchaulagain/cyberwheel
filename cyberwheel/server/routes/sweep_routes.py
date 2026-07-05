from __future__ import annotations

import shutil

from fastapi import APIRouter, Body

from cyberwheel.server import jobs, metrics, registry, sweeps
from cyberwheel.server.options import env_config_params, validate_config_refs
from cyberwheel.server.paths import (
    GENERATED_CONFIG_DIR,
    MODELS_DIR,
    RUNS_DIR,
)
from cyberwheel.server.validation import ApiError, not_found, require

router = APIRouter(prefix="/api/sweeps", tags=["sweeps"])


def _child_status(run_id: str) -> dict:
    record = registry.load_run(run_id)
    if record is None:
        return {"status": "missing", "progress": None}
    status = record.get("status", "unknown")
    total = record.get("params", {}).get("total_timesteps") or 0
    progress = None
    if status == "succeeded":
        progress = 1.0
    elif status in ("running", "queued"):
        step = metrics.last_step(record["experiment_name"]) or 0
        progress = min(1.0, step / total) if total else None
    return {"status": status, "progress": progress, "experiment_name": record.get("experiment_name")}


def _cell_metrics(experiment_name: str) -> dict:
    """Final episodic returns + SPS for a train cell; empty until metrics exist."""
    try:
        summary = metrics.summary(experiment_name)
    except ApiError:
        return {}  # no tfevents yet (queued / just started)
    tags = [t for group in summary["tags"].values() for t in group]
    wanted = [t for t in tags if t.endswith("_episodic_return")]
    if "charts/SPS" in tags:
        wanted.append("charts/SPS")
    return {t: metrics.last_value(experiment_name, t) for t in wanted}


def _decorate(sweep: dict) -> dict:
    out = dict(sweep)
    statuses = []
    for cell in sweep["cells"]:
        info = _child_status(cell["run_id"])
        cell["status"] = info["status"]
        cell["progress"] = info["progress"]
        statuses.append(info["status"])
    out["status"] = sweeps.rollup_status(statuses)
    return out


@router.post("", status_code=201)
def create_sweep(body: dict = Body(...)) -> dict:
    display_name = str(body.get("display_name") or "").strip()
    base_config = body.get("base_config") or ""
    base_params = dict(body.get("params") or {})
    grid = body.get("grid") or {}
    require(display_name, "display_name is required")
    require(base_config, "base_config is required")
    base_merged = {**env_config_params(base_config), **base_params}
    require("total_timesteps" in base_merged, f"{base_config} is not a training config")

    cells = sweeps.expand_grid(grid)
    sweep_id = jobs.make_run_id(display_name)
    # make_run_id only checks the run registry; a sweep id itself never gets a
    # run.json, so guard against a same-second double submit separately.
    require(sweeps.load_sweep(sweep_id) is None, f"sweep {sweep_id} already exists", 409)

    # Pass 1: validate every cell before anything is written or launched, so
    # a bad cell can't leave earlier cells running under a sweep that was
    # never saved (invisible to and undeletable through the sweep APIs), nor
    # leave stray generated configs behind.
    staged = []
    cell_records = []
    for index, cell in enumerate(cells):
        run_id = f"{sweep_id}-c{index}"
        require(registry.load_run(run_id) is None, f"run {run_id} already exists", 409)
        params = {**base_params, **cell, "experiment_name": run_id}
        params.setdefault("track", False)
        merged = {**env_config_params(base_config), **params}
        require("total_timesteps" in merged, f"{base_config} is not a training config")
        validate_config_refs(merged)
        record = {
            "id": run_id,
            "kind": "train",
            "display_name": f"{display_name} · " + ", ".join(f"{k}={v}" for k, v in cell.items()),
            "status": "queued",
            "pid": None,
            "exit_code": None,
            "created_at": registry.now_iso(),
            "started_at": None,
            "ended_at": None,
            "base_config": base_config,
            "generated_config": None,
            "params": merged,
            "experiment_name": run_id,
            "sweep_id": sweep_id,
            "sweep_cell": cell,
        }
        staged.append((record, params))
        cell_records.append({"run_id": run_id, "params": cell})

    # Persist the sweep before rendering/launching so every child is
    # reachable from the sweep APIs from the moment it exists.
    sweep = {
        "id": sweep_id,
        "display_name": display_name,
        "base_config": base_config,
        "base_params": base_params,
        "grid": grid,
        "varied_keys": sorted(grid),
        "cells": cell_records,
        "created_at": registry.now_iso(),
    }
    sweeps.save_sweep(sweep)

    # Pass 2: render each validated cell's config and launch (or queue) it.
    for record, params in staged:
        config_ref = jobs.generate_config(record["id"], base_config, params)
        record["generated_config"] = config_ref
        jobs.launch_or_queue(record, "train", config_ref)
    return _decorate(sweep)


@router.get("")
def list_sweeps() -> dict:
    return {"sweeps": [_decorate(s) for s in sweeps.list_sweeps()]}


def _get(sweep_id: str) -> dict:
    sweep = sweeps.load_sweep(sweep_id)
    if sweep is None:
        raise not_found(f"sweep {sweep_id!r} not found")
    return sweep


@router.get("/{sweep_id}")
def get_sweep(sweep_id: str) -> dict:
    sweep = _decorate(_get(sweep_id))
    for cell in sweep["cells"]:
        cell["metrics"] = _cell_metrics(cell["run_id"])
    return sweep


@router.delete("/{sweep_id}")
def delete_sweep(sweep_id: str, artifacts: bool = False) -> dict:
    sweep = _get(sweep_id)
    for cell in sweep["cells"]:
        run_id = cell["run_id"]
        record = registry.load_run(run_id)
        require(
            record is None or record.get("status") != "running",
            f"stop child run {run_id} before deleting the sweep",
            409,
        )
    # Drop queued cells from the launch queue first, so the reaper can't
    # spawn (and thereby resurrect) a cell while its files are being removed.
    for cell in sweep["cells"]:
        jobs.cancel_queued(cell["run_id"])
    for cell in sweep["cells"]:
        run_id = cell["run_id"]
        if artifacts:
            for target in (
                GENERATED_CONFIG_DIR / f"{run_id}.yaml",
                MODELS_DIR / run_id,
                RUNS_DIR / run_id,
            ):
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.is_file():
                    target.unlink(missing_ok=True)
        shutil.rmtree(registry.run_dir(run_id), ignore_errors=True)
    shutil.rmtree(sweeps.sweep_dir(sweep_id), ignore_errors=True)
    return {"deleted": sweep_id, "artifacts_removed": artifacts}
