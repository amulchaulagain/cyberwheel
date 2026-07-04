from __future__ import annotations

import shutil

from fastapi import APIRouter, Body

from cyberwheel.server import actions_log, jobs, metrics, registry
from cyberwheel.server.options import env_config_params, validate_config_refs
from cyberwheel.server.paths import (
    ACTION_LOGS_DIR,
    GENERATED_CONFIG_DIR,
    GRAPHS_DIR,
    MODELS_DIR,
    RUNS_DIR,
)
from cyberwheel.server.validation import not_found, require

router = APIRouter(prefix="/api/runs", tags=["runs"])


def _decorate(record: dict) -> dict:
    out = dict(record)
    params = record.get("params", {})
    if record["kind"] == "train":
        total = params.get("total_timesteps") or 0
        if record["status"] == "running":
            step = metrics.last_step(record["experiment_name"]) or 0
            out["progress"] = min(1.0, step / total) if total else None
            out["last_global_step"] = step
            out["last_sps"] = metrics.last_value(record["experiment_name"], "charts/SPS")
        elif record["status"] == "succeeded":
            out["progress"] = 1.0
    elif record["kind"] == "evaluate":
        expected = (params.get("num_episodes") or 0) * (params.get("num_steps") or 0)
        if record["status"] == "running":
            rows = actions_log.row_count(record["graph_name"]) or 0
            out["progress"] = min(1.0, rows / expected) if expected else None
        elif record["status"] == "succeeded":
            out["progress"] = 1.0
    return out


def _get_record(run_id: str) -> dict:
    record = registry.load_run(run_id)
    if record is None:
        raise not_found(f"run {run_id!r} not found")
    return record


@router.get("")
def list_runs(kind: str | None = None, status: str | None = None) -> dict:
    runs = [
        _decorate(record)
        for record in registry.list_runs()
        if (kind is None or record.get("kind") == kind)
        and (status is None or record.get("status") == status)
    ]
    external = registry.external_models() if kind in (None, "external_model") else []
    return {"runs": runs, "external_models": external}


@router.post("/train", status_code=201)
def launch_train(body: dict = Body(...)) -> dict:
    display_name = body.get("display_name") or ""
    base_config = body.get("base_config") or ""
    params = dict(body.get("params") or {})
    require(display_name.strip(), "display_name is required")
    require(base_config, "base_config is required")

    run_id = jobs.make_run_id(display_name)
    params["experiment_name"] = run_id
    params.setdefault("track", False)
    merged = {**env_config_params(base_config), **params}
    require("total_timesteps" in merged, f"{base_config} is not a training config")
    validate_config_refs(merged)

    config_ref = jobs.generate_config(run_id, base_config, params)
    record = {
        "id": run_id,
        "kind": "train",
        "display_name": display_name.strip(),
        "status": "queued",
        "pid": None,
        "exit_code": None,
        "created_at": registry.now_iso(),
        "started_at": None,
        "ended_at": None,
        "base_config": base_config,
        "generated_config": config_ref,
        "params": merged,
        "experiment_name": run_id,
    }
    return _decorate(jobs.launch(record, "train", config_ref))


@router.post("/evaluate", status_code=201)
def launch_evaluate(body: dict = Body(...)) -> dict:
    display_name = body.get("display_name") or ""
    base_config = body.get("base_config") or ""
    source = body.get("source") or {}
    checkpoint = str(body.get("checkpoint") or "agent")
    params = dict(body.get("params") or {})
    require(display_name.strip(), "display_name is required")
    require(base_config, "base_config is required")

    source_run_id = source.get("run_id")
    if source_run_id:
        source_record = _get_record(source_run_id)
        require(source_record["kind"] == "train", f"{source_run_id} is not a training run")
        experiment_name = source_record["experiment_name"]
    else:
        experiment_name = source.get("experiment_name") or ""
        require(experiment_name, "source.run_id or source.experiment_name is required")

    available = registry.checkpoints_for(experiment_name)
    require(
        available["agents"],
        f"no model checkpoints found under models/{experiment_name}/",
        404,
    )
    require(
        checkpoint in available["checkpoints"],
        f"checkpoint {checkpoint!r} not available for every agent "
        f"(have: {available['checkpoints']})",
    )

    run_id = jobs.make_run_id(display_name)
    params.update(
        experiment_name=experiment_name,
        graph_name=run_id,
        checkpoint=checkpoint,
        download_model=False,
    )
    params.setdefault("visualize", True)
    merged = {**env_config_params(base_config), **params}
    require("checkpoint" in env_config_params(base_config), f"{base_config} is not an evaluation config")
    validate_config_refs(merged)

    config_ref = jobs.generate_config(run_id, base_config, params)
    record = {
        "id": run_id,
        "kind": "evaluate",
        "display_name": display_name.strip(),
        "status": "queued",
        "pid": None,
        "exit_code": None,
        "created_at": registry.now_iso(),
        "started_at": None,
        "ended_at": None,
        "base_config": base_config,
        "generated_config": config_ref,
        "params": merged,
        "experiment_name": experiment_name,
        "source_run_id": source_run_id,
        "graph_name": run_id,
        "checkpoint": checkpoint,
        "visualize": bool(merged.get("visualize")),
    }
    return _decorate(jobs.launch(record, "evaluate", config_ref))


@router.get("/{run_id}")
def get_run(run_id: str) -> dict:
    record = _decorate(_get_record(run_id))
    if record["kind"] == "train":
        record["artifacts"] = {
            "models": (MODELS_DIR / record["experiment_name"]).is_dir(),
            "metrics": (RUNS_DIR / record["experiment_name"]).is_dir(),
        }
    else:
        graph_name = record.get("graph_name", "")
        record["artifacts"] = {
            "actions": (ACTION_LOGS_DIR / f"{graph_name}.csv").is_file(),
            "viz": (GRAPHS_DIR / graph_name / "meta.json").is_file(),
        }
    return record


@router.post("/{run_id}/stop")
def stop_run(run_id: str) -> dict:
    return _decorate(jobs.stop(run_id))


@router.delete("/{run_id}")
def delete_run(run_id: str, artifacts: bool = False) -> dict:
    record = _get_record(run_id)
    require(record["status"] != "running", "stop the run before deleting it", 409)
    if artifacts:
        targets = [GENERATED_CONFIG_DIR / f"{run_id}.yaml"]
        if record["kind"] == "train":
            targets += [MODELS_DIR / run_id, RUNS_DIR / run_id]
        else:
            graph_name = record.get("graph_name", run_id)
            targets += [ACTION_LOGS_DIR / f"{graph_name}.csv", GRAPHS_DIR / graph_name]
        for target in targets:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.is_file():
                target.unlink(missing_ok=True)
    shutil.rmtree(registry.run_dir(run_id), ignore_errors=True)
    return {"deleted": run_id, "artifacts_removed": artifacts}


@router.get("/{run_id}/logs")
def get_logs(run_id: str, offset: int = 0) -> dict:
    _get_record(run_id)
    path = registry.run_dir(run_id) / "stdout.log"
    if not path.is_file():
        return {"offset_next": 0, "content": ""}
    size = path.stat().st_size
    offset = max(0, min(offset, size))
    with open(path, "rb") as f:
        f.seek(offset)
        content = f.read(512 * 1024)
    return {
        "offset_next": offset + len(content),
        "content": content.decode("utf-8", errors="replace"),
    }


@router.get("/{run_id}/metrics")
def get_metrics(run_id: str) -> dict:
    record = _get_record(run_id)
    require(record["kind"] == "train", "metrics exist only for training runs")
    out = metrics.summary(record["experiment_name"])
    out["total_timesteps"] = record.get("params", {}).get("total_timesteps")
    return out


@router.get("/{run_id}/metrics/scalars")
def get_scalars(
    run_id: str, tags: str = "", after_step: int = -1, max_points: int = 1000
) -> dict:
    record = _get_record(run_id)
    require(record["kind"] == "train", "metrics exist only for training runs")
    tag_list = [t for t in tags.split(",") if t]
    require(tag_list, "tags query parameter is required")
    return metrics.scalars(
        record["experiment_name"], tag_list, after_step, max(1, min(max_points, 5000))
    )


@router.get("/{run_id}/checkpoints")
def get_checkpoints(run_id: str) -> dict:
    if run_id.startswith("external:"):
        experiment_name = run_id.split(":", 1)[1]
    else:
        experiment_name = _get_record(run_id)["experiment_name"]
    return registry.checkpoints_for(experiment_name)
