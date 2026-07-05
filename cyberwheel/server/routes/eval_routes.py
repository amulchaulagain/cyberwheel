from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

from cyberwheel.server import actions_log, registry, report
from cyberwheel.server.paths import GRAPHS_DIR
from cyberwheel.server.validation import not_found, require

router = APIRouter(prefix="/api/runs", tags=["evaluation"])


def _graph_name(run_id: str) -> str:
    record = registry.load_run(run_id)
    if record is None:
        raise not_found(f"run {run_id!r} not found")
    require(record["kind"] == "evaluate", f"{run_id} is not an evaluation run")
    return record.get("graph_name") or run_id


@router.get("/{run_id}/actions")
def get_actions(run_id: str, episode: int | None = None) -> dict:
    return actions_log.actions(_graph_name(run_id), episode)


@router.get("/{run_id}/summary")
def get_summary(run_id: str) -> dict:
    return actions_log.summary(_graph_name(run_id))


@router.get("/{run_id}/report")
def get_report(run_id: str) -> HTMLResponse:
    record = registry.load_run(run_id)
    if record is None:
        raise not_found(f"run {run_id!r} not found")
    require(record["kind"] == "evaluate", f"{run_id} is not an evaluation run")
    graph_name = record.get("graph_name") or run_id
    html = report.build_report_html(record, graph_name, registry.now_iso())
    return HTMLResponse(
        content=html,
        headers={"Content-Disposition": f'inline; filename="{graph_name}_report.html"'},
    )


def _viz_file(run_id: str, filename: str) -> FileResponse:
    path = GRAPHS_DIR / _graph_name(run_id) / filename
    if not path.is_file():
        raise not_found(
            f"{filename} not available — the evaluation may still be running, "
            "or was launched with visualize disabled"
        )
    return FileResponse(path, media_type="application/json")


@router.get("/{run_id}/viz/meta")
def viz_meta(run_id: str) -> FileResponse:
    return _viz_file(run_id, "meta.json")


@router.get("/{run_id}/viz/layout")
def viz_layout(run_id: str) -> FileResponse:
    return _viz_file(run_id, "layout.json")


@router.get("/{run_id}/viz/episodes/{episode}")
def viz_episode(run_id: str, episode: int) -> FileResponse:
    return _viz_file(run_id, f"episode_{episode}.json")
