"""Filesystem anchors for the experimentation server.

All run artifacts live under the package's ``data/`` tree (same places the
CLI writes to); the server's own state lives in ``data/frontend/`` and the
per-run generated env configs in ``data/configs/environment/generated/``,
both git-ignored.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

DATA_ROOT = Path(str(files("cyberwheel.data")))
CONFIG_ROOT = DATA_ROOT / "configs"
ENV_CONFIG_DIR = CONFIG_ROOT / "environment"
GENERATED_CONFIG_DIR = ENV_CONFIG_DIR / "generated"

MODELS_DIR = DATA_ROOT / "models"
RUNS_DIR = DATA_ROOT / "runs"
GRAPHS_DIR = DATA_ROOT / "graphs"
ACTION_LOGS_DIR = DATA_ROOT / "action_logs"

FRONTEND_STATE_DIR = DATA_ROOT / "frontend"
REGISTRY_DIR = FRONTEND_STATE_DIR / "runs"

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Config subdirectory each file-reference field of an env YAML resolves in
# (mirrors the importlib.resources roots used across the runners).
CONFIG_FIELD_DIRS = {
    "network_config": "network",
    "decoy_config": "decoy_hosts",
    "host_config": "host_definitions",
    "detector_config": "detector",
    "red_agent": "red_agent",
    "blue_agent": "blue_agent",
}


def ensure_dirs() -> None:
    for path in (GENERATED_CONFIG_DIR, REGISTRY_DIR, MODELS_DIR, RUNS_DIR, GRAPHS_DIR, ACTION_LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
