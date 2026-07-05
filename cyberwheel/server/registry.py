"""Run registry: one directory per launched run under ``data/frontend/runs/``.

``run.json`` is the single source of truth for a run: identity, status,
and the full resolved parameter snapshot (the models/ dirs only hold ``.pt``
weights, so the registry is what makes runs reproducible and lets the UI
prefill an evaluation from its source training run). Writes are atomic
(tmp + rename). Model directories with no registry entry — e.g. trained via
the bare CLI — surface as read-only "external" entries so they can still be
evaluated.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from cyberwheel.server.paths import MODELS_DIR, REGISTRY_DIR

CHECKPOINT_RE = re.compile(r"^(blue|red)_(agent|\d+)\.pt$")

# Set by jobs.py: callable(run_id) -> True while this server process owns a
# live Popen handle for the run. Its exit is reaped authoritatively there,
# so reconciliation must not race it (a just-exited child is a zombie whose
# /proc cmdline is empty — the pid probe alone would misread it as orphaned).
in_flight = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_dir(run_id: str) -> Path:
    return REGISTRY_DIR / run_id


def save_run(record: dict) -> None:
    directory = run_dir(record["id"])
    directory.mkdir(parents=True, exist_ok=True)
    # Unique tmp name: concurrent writers (stop() worker vs reaper) sharing
    # one tmp path would interleave writes and rename torn JSON into place.
    tmp = directory / f"run.json.{uuid4().hex}.tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2)
    os.replace(tmp, directory / "run.json")


def load_run(run_id: str) -> dict | None:
    path = run_dir(run_id) / "run.json"
    if not path.is_file():
        return None
    with open(path) as f:
        record = json.load(f)
    return _reconcile(record)


def list_runs() -> list[dict]:
    records = []
    if REGISTRY_DIR.is_dir():
        for entry in REGISTRY_DIR.iterdir():
            path = entry / "run.json"
            if not path.is_file():
                continue
            try:
                with open(path) as f:
                    records.append(_reconcile(json.load(f)))
            except (json.JSONDecodeError, OSError):
                continue
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return records


def _pid_alive(pid: int, run_id: str) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    # Guard against pid reuse: the process must actually be this run
    # (its argv carries the generated config named after the run id).
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return run_id.encode() in f.read()
    except OSError:
        return True  # non-procfs platform: trust the kill(0) probe


def _reconcile(record: dict) -> dict:
    """A 'running' record whose process is gone (server restart, crash,
    SIGKILL) becomes 'orphaned' — persisted so the answer is stable."""
    if record.get("status") != "running":
        return record
    if in_flight is not None and in_flight(record["id"]):
        return record
    if _pid_alive(record.get("pid"), record["id"]):
        return record
    # Dead pid, not owned here. Re-read before persisting: the reaper only
    # releases ownership AFTER saving the final status, so a fresh read that
    # still says "running" is a true orphan, not a reap in progress — without
    # this, marking orphaned here could overwrite a just-saved terminal
    # status (a succeeded run then stays 'orphaned' forever).
    path = run_dir(record["id"]) / "run.json"
    try:
        with open(path) as f:
            record = json.load(f)
    except (OSError, json.JSONDecodeError):
        return record
    if record.get("status") != "running":
        return record
    record["status"] = "orphaned"
    record["ended_at"] = record.get("ended_at") or now_iso()
    save_run(record)
    return record


def checkpoints_for(experiment_name: str) -> dict:
    """Scan a models dir: agents present and the checkpoint tags loadable
    for every agent (evaluation loads ``{agent}_{checkpoint}.pt`` for each
    RL agent, so only the intersection is safe to offer)."""
    directory = MODELS_DIR / experiment_name
    per_agent: dict[str, set] = {}
    if directory.is_dir():
        for entry in directory.iterdir():
            match = CHECKPOINT_RE.match(entry.name)
            if match:
                agent, tag = match.groups()
                per_agent.setdefault(agent, set()).add(tag)
    agents = sorted(per_agent)
    common = set.intersection(*per_agent.values()) if per_agent else set()
    tags = sorted((t for t in common if t != "agent"), key=int)
    checkpoints = (["agent"] if "agent" in common else []) + tags
    return {"agents": agents, "checkpoints": checkpoints}


def external_models() -> list[dict]:
    """Model dirs with no registry entry (e.g. CLI-trained runs)."""
    claimed = {
        record.get("experiment_name")
        for record in list_runs()
        if record.get("kind") == "train"
    }
    entries = []
    if MODELS_DIR.is_dir():
        for entry in sorted(MODELS_DIR.iterdir()):
            if not entry.is_dir() or entry.name in claimed:
                continue
            info = checkpoints_for(entry.name)
            if not info["agents"]:
                continue
            entries.append(
                {
                    "id": f"external:{entry.name}",
                    "kind": "external_model",
                    "experiment_name": entry.name,
                    **info,
                }
            )
    return entries
