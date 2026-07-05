"""Generate parameterized network configs (and preview their layout).

Both endpoints shell out to ``python -m cyberwheel.network.network_generation``
so the server process never imports the (torch-pulling) network stack. Generated
configs land in ``data/configs/network/`` and auto-appear in the options dropdowns.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

from fastapi import APIRouter, Body

from cyberwheel.server.options import _network_entry
from cyberwheel.server.paths import CONFIG_ROOT
from cyberwheel.server.validation import require

router = APIRouter(prefix="/api/networks", tags=["networks"])

_NAME_RE = re.compile(r"[a-zA-Z0-9_-]+")
_GEN_MODULE = "cyberwheel.network.network_generation"
# Matches the largest supported obs-size tier; bigger networks are unusable.
_MAX_HOSTS = 10_000


def _number(params: dict, key: str, cast, required: bool = False):
    """Fetch and convert a numeric param, turning bad input into a 400."""
    value = params.get(key)
    if value is None:
        require(not required, f"{key} is required")
        return None
    try:
        return cast(value)
    except (TypeError, ValueError):
        require(False, f"{key} must be a {cast.__name__}, got {value!r}")


def _params_to_argv(params: dict) -> list[str]:
    argv: list[str] = []
    seed = _number(params, "seed", int)
    if seed is not None:
        argv += ["--seed", str(seed)]
    num_hosts = _number(params, "num_hosts", int, required=True)
    require(
        1 <= num_hosts <= _MAX_HOSTS,
        f"num_hosts must be between 1 and {_MAX_HOSTS}",
    )
    argv += ["--num-hosts", str(num_hosts)]
    num_subnets = _number(params, "num_subnets", int)
    if num_subnets is not None:
        argv += ["--num-subnets", str(num_subnets)]
    server_ratio = _number(params, "server_ratio", float)
    if server_ratio is not None:
        argv += ["--server-ratio", str(server_ratio)]
    vuln_density = _number(params, "vuln_density", float)
    if vuln_density is not None:
        argv += ["--vuln-density", str(vuln_density)]
    if params.get("dedicated_server_subnets") is False:
        argv += ["--no-dedicated-server-subnets"]
    server_types = params.get("server_types")
    if server_types:
        require(
            isinstance(server_types, list)
            and all(isinstance(t, str) for t in server_types),
            "server_types must be a list of strings",
        )
        argv += ["--server-types", ",".join(server_types)]
    if params.get("size_tier"):
        argv += ["--size-tier", str(params["size_tier"])]
    return argv


def _run_generator(argv: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            [sys.executable, "-m", _GEN_MODULE, *argv],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        require(False, "network generation timed out after 120s")


@router.post("/generate", status_code=201)
def generate_network(body: dict = Body(...)) -> dict:
    name = str(body.get("name") or "").strip()
    params = dict(body.get("params") or {})
    require(bool(_NAME_RE.fullmatch(name)), "name must be letters, digits, '-' or '_'")
    target = CONFIG_ROOT / "network" / f"{name}.yaml"
    require(not target.exists(), f"network {name!r} already exists", 409)

    argv = ["--name", name, "--output", str(target), *_params_to_argv(params)]
    proc = _run_generator(argv)
    if proc.returncode != 0:
        require(False, f"generation failed: {(proc.stderr or proc.stdout)[-500:]}")
    return _network_entry(f"{name}.yaml")


@router.post("/preview")
def preview_network(body: dict = Body(...)) -> dict:
    params = dict(body.get("params") or {})
    argv = ["--name", "preview", "--layout-only", *_params_to_argv(params)]
    proc = _run_generator(argv)
    if proc.returncode != 0:
        require(False, f"preview failed: {(proc.stderr or proc.stdout)[-500:]}")
    try:
        layout = json.loads(proc.stdout)
    except json.JSONDecodeError:
        require(False, "preview did not return a valid layout")
    return {"layout": layout}
