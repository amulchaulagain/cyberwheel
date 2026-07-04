"""Enumerate the configuration universe for the frontend's dropdowns.

Everything is discovered from disk (the same ``data/configs/`` subdirs the
runners resolve bare filenames against) or introspected from the reward
modules, so the UI always reflects what is actually available.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from cyberwheel.server.paths import CONFIG_FIELD_DIRS, CONFIG_ROOT, ENV_CONFIG_DIR
from cyberwheel.server.validation import not_found, require

# (path, mtime) -> parsed YAML, so 1 MB network configs parse once.
_yaml_cache: dict[str, tuple[float, dict]] = {}


def load_yaml(path: Path) -> dict:
    key = str(path)
    mtime = path.stat().st_mtime
    cached = _yaml_cache.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    with open(path) as f:
        parsed = yaml.safe_load(f)
    _yaml_cache[key] = (mtime, parsed)
    return parsed


def _config_files(subdir: str) -> list[str]:
    base = CONFIG_ROOT / subdir
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.glob("*.yaml"))


def _classify_env_config(params: dict) -> str:
    if "total_timesteps" in params:
        return "train"
    if "checkpoint" in params:
        return "evaluate"
    return "run"


def env_config_params(name: str) -> dict:
    require("/" not in name and name.endswith(".yaml"), f"invalid config name {name!r}")
    path = ENV_CONFIG_DIR / name
    if not path.is_file():
        raise not_found(f"environment config {name!r} not found")
    return load_yaml(path)


def _reward_functions(module) -> list[str]:
    return sorted(
        name
        for name, fn in inspect.getmembers(module, inspect.isfunction)
        if fn.__module__ == module.__name__ and not name.startswith("_")
    )


def _network_entry(name: str) -> dict:
    parsed = load_yaml(CONFIG_ROOT / "network" / name)
    return {
        "file": name,
        "hosts": len(parsed.get("hosts") or {}),
        "subnets": len(parsed.get("subnets") or {}),
    }


def _check_config_ref(field: str, subdir: str, value) -> None:
    require(
        isinstance(value, str) and "/" not in value and value.endswith(".yaml"),
        f"{field}: {value!r} is not a config filename",
    )
    require(
        (CONFIG_ROOT / subdir / value).is_file(),
        f"{field}: {value!r} not found in configs/{subdir}/",
    )


def validate_config_refs(params: dict) -> None:
    """Every file-reference field of a merged env config must point at a
    real file — fail launch requests up front instead of mid-subprocess."""
    for field, subdir in CONFIG_FIELD_DIRS.items():
        if field not in params or params[field] is None:
            continue
        value = params[field]
        values = value if isinstance(value, list) else [value]
        require(bool(values), f"{field} must not be empty")
        for item in values:
            _check_config_ref(field, subdir, item)
    agents = params.get("agents")
    if isinstance(agents, dict):
        for agent, subdir in (("red", "red_agent"), ("blue", "blue_agent")):
            if agents.get(agent) is not None:
                _check_config_ref(f"agents.{agent}", subdir, agents[agent])


def all_options() -> dict:
    from cyberwheel.reward import blue_reward_functions, red_reward_functions

    env_configs: dict[str, list[str]] = {"train": [], "evaluate": [], "run": []}
    for name in _config_files("environment"):
        try:
            kind = _classify_env_config(load_yaml(ENV_CONFIG_DIR / name))
        except Exception:
            continue
        env_configs[kind].append(name)

    networks = []
    for name in _config_files("network"):
        try:
            networks.append(_network_entry(name))
        except Exception:
            networks.append({"file": name, "hosts": None, "subnets": None})

    try:
        import torch

        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    except Exception:
        devices = ["cpu"]

    return {
        "env_configs": env_configs,
        "network_configs": networks,
        "red_agents": _config_files("red_agent"),
        "blue_agents": _config_files("blue_agent"),
        "detector_configs": _config_files("detector"),
        "decoy_configs": _config_files("decoy_hosts"),
        "host_configs": _config_files("host_definitions"),
        "blue_reward_functions": _reward_functions(blue_reward_functions),
        "red_reward_functions": _reward_functions(red_reward_functions),
        "reward_functions": ["RLReward"],
        "environments": ["CyberwheelRL", "Cyberwheel"],
        "valid_targets": ["servers", "users", "all", "leader"],
        "devices": devices,
        "anneal_lr": [False, "cosine_restarts"],
        "network_size_compatibility": ["small", "medium", "large"],
    }
