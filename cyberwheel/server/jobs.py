"""Launching and controlling train/evaluate subprocesses.

Every run goes through the real CLI (``python -m cyberwheel <mode>
generated/<run_id>.yaml``) with a fully rendered env YAML and zero CLI
overrides: several YAML keys (the nested ``agents`` map, per-optimizer
learning rates, reward-function selection, multi-network lists, the
string-valued ``anneal_lr``) are unreachable or corrupted via override
flags, and the generated file doubles as the run's provenance snapshot.
Do not "optimize" this back to CLI flags.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import yaml

from cyberwheel.server import registry
from cyberwheel.server.options import env_config_params
from cyberwheel.server.paths import GENERATED_CONFIG_DIR, ensure_dirs
from cyberwheel.server.validation import ApiError, require, slugify

# Never let a spawned run phone home unless W&B tracking was explicitly
# requested; matches the test framework's subprocess environment.
def _spawn_env(track: bool) -> dict:
    env = dict(os.environ)
    env.update({"MPLBACKEND": "Agg", "TQDM_DISABLE": "1", "PYTHONUNBUFFERED": "1"})
    if not track:
        env.update({"WANDB_MODE": "disabled", "WANDB_SILENT": "true"})
    return env


_procs: dict[str, subprocess.Popen] = {}
_lock = threading.Lock()
_reaper_started = False


def _owns(run_id: str) -> bool:
    with _lock:
        return run_id in _procs


registry.in_flight = _owns


def make_run_id(display_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"{slugify(display_name)}-{stamp}"
    if registry.load_run(run_id) is not None:
        run_id = f"{run_id}-{os.getpid() % 1000}"
    require(registry.load_run(run_id) is None, f"run id {run_id} already exists", 409)
    return run_id


def _max_concurrency() -> int | None:
    raw = os.environ.get("CYBERWHEEL_FRONTEND_MAX_CONCURRENCY", "")
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


def generate_config(run_id: str, base_config: str, params: dict) -> str:
    """Render base env config + UI params to ``generated/<run_id>.yaml``.

    Returns the config reference to pass to the CLI (relative to the
    environment config root)."""
    ensure_dirs()
    merged = dict(env_config_params(base_config))
    merged.update(params)
    path = GENERATED_CONFIG_DIR / f"{run_id}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return f"generated/{run_id}.yaml"


def launch(record: dict, mode: str, config_ref: str) -> dict:
    limit = _max_concurrency()
    if limit is not None:
        active = sum(1 for r in registry.list_runs() if r.get("status") == "running")
        if active >= limit:
            raise ApiError(429, f"{active} runs already active (limit {limit})")

    directory = registry.run_dir(record["id"])
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "stdout.log"
    track = bool(record.get("params", {}).get("track"))
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "cyberwheel", mode, config_ref],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=_spawn_env(track),
        )
    record.update(
        status="running",
        pid=proc.pid,
        started_at=registry.now_iso(),
        exit_code=None,
    )
    registry.save_run(record)
    with _lock:
        _procs[record["id"]] = proc
    _ensure_reaper()
    return record


def stop(run_id: str) -> dict:
    record = registry.load_run(run_id)
    require(record is not None, f"run {run_id} not found", 404)
    if record["status"] != "running":
        return record
    pid = record.get("pid")
    with _lock:
        proc = _procs.get(run_id)
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    # Keep the Popen in _procs: the escalation/reaper threads must wait() it,
    # otherwise the child lingers as a zombie that still probes as alive.
    threading.Thread(target=_escalate_kill, args=(pid, proc), daemon=True).start()
    record["status"] = "stopped"
    record["ended_at"] = registry.now_iso()
    registry.save_run(record)
    return record


def _escalate_kill(pid: int, proc: subprocess.Popen | None, grace_s: float = 10.0) -> None:
    if proc is not None:
        try:
            proc.wait(grace_s)
            return
        except subprocess.TimeoutExpired:
            pass
    else:
        deadline = time.time() + grace_s
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.5)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    if proc is not None:
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            pass


def _ensure_reaper() -> None:
    global _reaper_started
    with _lock:
        if _reaper_started:
            return
        _reaper_started = True
    threading.Thread(target=_reap_loop, daemon=True).start()


def _reap_loop() -> None:
    while True:
        time.sleep(1.0)
        with _lock:
            finished = [
                (run_id, proc)
                for run_id, proc in _procs.items()
                if proc.poll() is not None
            ]
            for run_id, _ in finished:
                _procs.pop(run_id, None)
        for run_id, proc in finished:
            record = registry.load_run(run_id)
            # 'orphaned' can appear if a poll raced the exit; the owned
            # Popen handle has the authoritative exit code, so recover it.
            if record is None or record["status"] not in ("running", "orphaned"):
                continue
            record["status"] = "succeeded" if proc.returncode == 0 else "failed"
            record["exit_code"] = proc.returncode
            record["ended_at"] = registry.now_iso()
            registry.save_run(record)
