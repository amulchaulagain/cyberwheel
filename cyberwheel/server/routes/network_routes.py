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


def _params_to_argv(params: dict) -> list[str]:
    argv: list[str] = []
    if params.get("seed") is not None:
        argv += ["--seed", str(int(params["seed"]))]
    require(params.get("num_hosts") is not None, "num_hosts is required")
    argv += ["--num-hosts", str(int(params["num_hosts"]))]
    if params.get("num_subnets") is not None:
        argv += ["--num-subnets", str(int(params["num_subnets"]))]
    if params.get("server_ratio") is not None:
        argv += ["--server-ratio", str(float(params["server_ratio"]))]
    if params.get("vuln_density") is not None:
        argv += ["--vuln-density", str(float(params["vuln_density"]))]
    if params.get("dedicated_server_subnets") is False:
        argv += ["--no-dedicated-server-subnets"]
    if params.get("server_types"):
        argv += ["--server-types", ",".join(params["server_types"])]
    if params.get("size_tier"):
        argv += ["--size-tier", str(params["size_tier"])]
    return argv


def _run_generator(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", _GEN_MODULE, *argv],
        capture_output=True,
        text=True,
        timeout=120,
    )


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
