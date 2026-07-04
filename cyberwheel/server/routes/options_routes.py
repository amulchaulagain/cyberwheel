from __future__ import annotations

from fastapi import APIRouter

from cyberwheel.server.options import all_options, env_config_params

router = APIRouter(prefix="/api/options", tags=["options"])


@router.get("")
def options() -> dict:
    return all_options()


@router.get("/env-config/{name}")
def env_config(name: str) -> dict:
    return {"name": name, "params": env_config_params(name)}
