"""Config suite: every YAML under ``cyberwheel/data/configs`` loads and its
code-consumed references resolve.

One test case is registered per discovered YAML file, so new configs are
covered automatically. Only references the code actually consumes are hard
failures; dangling decorative references (e.g. the baseline schema's
``red_agent`` key, which the runners never read) are reported as INFO.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from cyberwheel.tests.framework.core import (
    CONFIG_ROOT,
    Context,
    Outcome,
    Registry,
    Status,
    TestCase,
    check,
)

SUITE = "config"

# Shipped configs that are genuinely broken today. They register as
# known-issue cases: failing exactly as documented reports XFAIL (non-gating);
# if someone fixes them the runner flags XPASS_WARN so the entry gets removed.
KNOWN_BROKEN = {
    "environment/art_campaign_vs_rl_blue.yaml": "references network/emulator_15_host.yaml, which does not exist",
    "environment/rl_red_campaign_vs_rl_blue.yaml": "references network/emulator_15_host.yaml, which does not exist",
}


def _import_cyberwheel_utils() -> None:
    # cyberwheel modules have an import-order-sensitive cycle
    # (network.host -> utils.host_types -> utils/__init__ -> red_actions ->
    # detectors.alert -> network.host). Importing cyberwheel.utils first
    # resolves it, exactly as cyberwheel/__main__.py does.
    importlib.import_module("cyberwheel.utils")


# Environment YAMLs come in distinct schemas consumed by different runners.
_REQUIRED_KEYS = {
    "rl-train": [
        "experiment_name",
        "environment",
        "total_timesteps",
        "num_envs",
        "num_steps",
        "num_saves",
        "eval_episodes",
        "network_config",
        "host_config",
        "decoy_config",
        "detector_config",
        "track",
        "device",
        "async_env",
        "seed",
        "deterministic",
    ],
    "rl-eval": [
        "experiment_name",
        "environment",
        "checkpoint",
        "download_model",
        "num_episodes",
        "num_steps",
        "network_config",
        "host_config",
        "decoy_config",
        "detector_config",
        "seed",
        "deterministic",
    ],
    "rl-other": ["environment", "network_config"],
    "baseline": [
        "environment",
        "red_agent",
        "blue_agent",
        "network_config",
        "num_episodes",
        "num_steps",
        "host_config",
    ],
}


def _load(path: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f)
    check(data is not None, "YAML file is empty")
    return data


def _classify_environment(data: dict) -> str:
    if isinstance(data.get("agents"), dict):
        if "total_timesteps" in data:
            return "rl-train"
        if "checkpoint" in data:
            return "rl-eval"
        return "rl-other"
    return "baseline"


def _check_environment(path: Path) -> Outcome:
    data = _load(path)
    check(isinstance(data, dict), f"expected a mapping, got {type(data).__name__}")
    kind = _classify_environment(data)
    infos: list[str] = []

    missing = [k for k in _REQUIRED_KEYS[kind] if k not in data]
    check(not missing, f"missing required keys for {kind} schema: {missing}")

    # References the runners actually load (hard failures if unresolvable).
    refs: list[tuple[str, str]] = []
    network = data.get("network_config")
    for entry in network if isinstance(network, list) else [network]:
        if entry:
            refs.append(("network", entry))
    for key, subdir in (
        ("host_config", "host_definitions"),
        ("decoy_config", "decoy_hosts"),
        ("detector_config", "detector"),
    ):
        if data.get(key):
            refs.append((subdir, data[key]))
    if kind.startswith("rl"):
        agents = data["agents"]
        check(
            isinstance(agents, dict) and "red" in agents and "blue" in agents,
            f"'agents' must map both 'red' and 'blue', got: {agents!r}",
        )
        refs.append(("red_agent", agents["red"]))
        refs.append(("blue_agent", agents["blue"]))

    unresolved = [
        f"{subdir}/{name}"
        for subdir, name in refs
        if not (CONFIG_ROOT / subdir / name).is_file()
    ]
    check(not unresolved, f"unresolvable config references: {unresolved}")

    if kind == "baseline":
        # These keys exist in the YAML but no code path reads them
        # (baseline_runner hardcodes the inactive agents).
        for key, subdir in (("red_agent", "red_agent"), ("blue_agent", "blue_agent")):
            value = data.get(key)
            if value and not (CONFIG_ROOT / subdir / value).is_file():
                infos.append(f"dangling unconsumed reference {key}: {value!r}")

    if kind == "rl-eval" and isinstance(network, list):
        infos.append(
            "list network_config is load-only: rl_evaluator assumes a string "
            "(rl_evaluator.py:117 .split('.'))"
        )
    if data.get("environment") == "CyberwheelEmulator":
        infos.append(
            "emulator environment: validated load-only (FIREWHEEL not present)"
        )

    if infos:
        return Outcome(
            Status.INFO, f"{kind} schema OK; " + "; ".join(infos), {"kind": kind}
        )
    return Outcome(Status.PASS, f"{kind} schema OK", {"kind": kind})


def _check_network(path: Path) -> Outcome:
    _import_cyberwheel_utils()
    from cyberwheel.network.network_base import Network
    from cyberwheel.utils import get_service_map

    data = _load(path)
    for key in ("network", "routers", "subnets", "hosts"):
        check(key in data, f"missing top-level key {key!r}")

    network = Network.create_network_from_yaml(path)
    check(len(network.hosts) > 0, "built network has no hosts")
    expected_name = data["network"].get("name")
    check(
        network.name == expected_name,
        f"network name mismatch: built {network.name!r} != config {expected_name!r}",
    )
    service_map = get_service_map(network)
    unmapped = [h for h in network.hosts if h not in service_map]
    check(not unmapped, f"hosts missing from service map: {unmapped[:5]}")
    return Outcome(
        Status.PASS,
        f"{len(network.hosts)} hosts, service map complete",
        {"hosts": len(network.hosts)},
    )


def _check_host_definitions(path: Path) -> Outcome:
    data = _load(path)
    check("host_types" in data, "missing top-level key 'host_types'")
    host_types = data["host_types"]
    check(
        isinstance(host_types, dict) and host_types,
        "'host_types' must be a non-empty mapping",
    )
    services_doc = _services_catalog()
    infos = []
    for name, spec in host_types.items():
        check(isinstance(spec, dict), f"host type {name!r} must be a mapping")
        if "type" not in spec:
            # Legal: HostType.type defaults to UNKNOWN (network/host.py).
            infos.append(f"{name}: no 'type' (defaults to UNKNOWN)")
        services = spec.get("services") or []
        check(
            isinstance(services, list), f"host type {name!r}: 'services' must be a list"
        )
        # Services are either catalog references (str) or inline definitions.
        unknown = [s for s in services if isinstance(s, str) and s not in services_doc]
        if unknown:
            infos.append(f"{name}: services not in the services catalog: {unknown}")
        for inline in (s for s in services if isinstance(s, dict)):
            check(
                "name" in inline and "port" in inline,
                f"host type {name!r}: inline service needs 'name' and 'port': {inline}",
            )
    if infos:
        return Outcome(Status.INFO, "; ".join(infos))
    return Outcome(Status.PASS, f"{len(host_types)} host types OK")


_SERVICES_FILE = "windows_exploitable_services.yaml"


def _services_catalog() -> dict:
    with open(CONFIG_ROOT / "services" / _SERVICES_FILE) as f:
        return yaml.safe_load(f) or {}


def _check_services(path: Path) -> Outcome:
    data = _load(path)
    check(isinstance(data, dict) and data, "expected a non-empty mapping of services")
    for name, spec in data.items():
        check(isinstance(spec, dict), f"service {name!r} must be a mapping")
        check("port" in spec, f"service {name!r} missing 'port'")
        check(
            isinstance(spec.get("cve", []), list),
            f"service {name!r}: 'cve' must be a list",
        )
    return Outcome(Status.PASS, f"{len(data)} services OK")


def _check_decoys(path: Path) -> Outcome:
    data = _load(path)
    check(isinstance(data, dict) and data, "expected a non-empty mapping of decoys")
    with open(CONFIG_ROOT / "host_definitions" / "host_defs_services.yaml") as f:
        host_types = (yaml.safe_load(f) or {}).get("host_types", {})
    services_doc = _services_catalog()
    for name, spec in data.items():
        check(isinstance(spec, dict), f"decoy {name!r} must be a mapping")
        for key in ("type", "reward", "recurring_reward", "services"):
            check(key in spec, f"decoy {name!r} missing key {key!r}")
        check(
            spec["type"] in host_types,
            f"decoy {name!r}: host type {spec['type']!r} not in host_defs_services.yaml",
        )
        unknown = [s for s in spec["services"] if s not in services_doc]
        check(not unknown, f"decoy {name!r}: unknown services {unknown}")
    return Outcome(Status.PASS, f"{len(data)} decoys OK")


def _check_detector(path: Path) -> Outcome:
    data = _load(path)
    check(isinstance(data, dict) and data, "expected a non-empty mapping")
    if "adjacency_list" in data or "init_info" in data:
        # A detector-handler graph: DetectorHandler performs the full
        # validation (graph shape + detector import/instantiation).
        _import_cyberwheel_utils()
        from cyberwheel.detectors.handler import DetectorHandler

        check(
            "adjacency_list" in data and "init_info" in data,
            "handler config needs both 'adjacency_list' and 'init_info'",
        )
        DetectorHandler(str(path))
        return Outcome(
            Status.PASS,
            f"handler graph OK ({len(data['init_info'])} detectors)",
            {"kind": "handler"},
        )
    # Otherwise: a probability table (technique/action name -> probability).
    for key, value in data.items():
        probs = value if isinstance(value, list) else [value]
        for p in probs:
            check(
                isinstance(p, (int, float)) and 0.0 <= float(p) <= 1.0,
                f"entry {key!r}: probability {p!r} not in [0, 1]",
            )
    return Outcome(
        Status.PASS, f"probability table OK ({len(data)} entries)", {"kind": "table"}
    )


def _check_red_agent(path: Path) -> Outcome:
    data = _load(path)
    check(isinstance(data, dict), "expected a mapping")
    check("class" in data, "missing 'class'")
    _import_cyberwheel_utils()
    red_agents = importlib.import_module("cyberwheel.red_agents")
    check(
        hasattr(red_agents, data["class"]),
        f"agent class {data['class']!r} not exported by cyberwheel.red_agents",
    )
    if "actions" in data:
        red_actions = importlib.import_module("cyberwheel.red_actions.actions")
        for action, spec in data["actions"].items():
            check(
                isinstance(spec, dict) and "class" in spec,
                f"action {action!r} missing 'class'",
            )
            check(
                hasattr(red_actions, spec["class"]),
                f"action {action!r}: class {spec['class']!r} not in cyberwheel.red_actions.actions",
            )
            reward = spec.get("reward")
            check(
                isinstance(reward, dict) and "immediate" in reward,
                f"action {action!r}: reward must map at least 'immediate'",
            )
        return Outcome(
            Status.PASS, f"killchain agent OK ({len(data['actions'])} actions)"
        )
    if "campaign" in data:
        check(
            isinstance(data["campaign"], list) and data["campaign"],
            "'campaign' must be a non-empty list",
        )
        for i, entry in enumerate(data["campaign"]):
            check(isinstance(entry, dict), f"campaign step {i} must be a mapping")
            check(
                "technique_name" in entry or "lateral_movement_technique" in entry,
                f"campaign step {i} needs 'technique_name' or 'lateral_movement_technique'",
            )
        return Outcome(Status.PASS, f"campaign OK ({len(data['campaign'])} steps)")
    raise AssertionError  # unreachable: guarded below


def _check_red_agent_shape(path: Path) -> Outcome:
    data = _load(path)
    check(
        "actions" in data or "campaign" in data,
        "red agent config needs 'actions' (killchain) or 'campaign'",
    )
    return _check_red_agent(path)


def _check_blue_agent(path: Path) -> Outcome:
    data = _load(path)
    check(isinstance(data, dict), "expected a mapping")
    for key in ("class", "rl", "actions"):
        check(key in data, f"missing key {key!r}")
    _import_cyberwheel_utils()
    blue_agents = importlib.import_module("cyberwheel.blue_agents")
    check(
        hasattr(blue_agents, data["class"]),
        f"agent class {data['class']!r} not exported by cyberwheel.blue_agents",
    )
    blue_actions = importlib.import_module("cyberwheel.blue_actions.actions")
    for action, spec in data["actions"].items():
        check(
            isinstance(spec, dict) and "class" in spec,
            f"action {action!r} missing 'class'",
        )
        check(
            hasattr(blue_actions, spec["class"]),
            f"action {action!r}: class {spec['class']!r} not in cyberwheel.blue_actions.actions",
        )
        for subdir, fname in (spec.get("configs") or {}).items():
            check(
                (CONFIG_ROOT / subdir / fname).is_file(),
                f"action {action!r}: referenced config {subdir}/{fname} does not exist",
            )
    for name, spec in (data.get("shared_data") or {}).items():
        module = importlib.import_module(spec["module"])
        check(
            hasattr(module, spec["class"]),
            f"shared_data {name!r}: class {spec['class']!r} not in module {spec['module']!r}",
        )
    action_space = data.get("action_space")
    if action_space:
        module = importlib.import_module(
            f"cyberwheel.blue_agents.action_space.{action_space['module']}"
        )
        check(
            hasattr(module, action_space["class"]),
            f"action_space class {action_space['class']!r} not in module {action_space['module']!r}",
        )
    return Outcome(Status.PASS, f"{len(data['actions'])} actions OK")


def _network_size(fname: str) -> int:
    prefix = fname.split("-")[0]
    return int(prefix) if prefix.isdigit() else 0


def register(registry: Registry, ctx: Context) -> None:
    checks = {
        "environment": _check_environment,
        "network": _check_network,
        "host_definitions": _check_host_definitions,
        "services": _check_services,
        "decoy_hosts": _check_decoys,
        "detector": _check_detector,
        "red_agent": _check_red_agent_shape,
        "blue_agent": _check_blue_agent,
    }
    for subdir, fn in checks.items():
        for path in sorted((CONFIG_ROOT / subdir).glob("*.yaml")):
            quick_skip = False
            default_skip_reason = None
            if subdir == "network":
                size = _network_size(path.name)
                quick_skip = size >= 1000
                if size >= 10000:
                    default_skip_reason = (
                        "very large network; opt in with --filter 10000"
                    )
            registry.add(
                TestCase(
                    name=f"config:{subdir}/{path.name}",
                    suite=SUITE,
                    fn=(lambda p=path, f=fn: f(p)),
                    timeout_s=600.0,
                    quick_skip=quick_skip,
                    default_skip_reason=default_skip_reason,
                    known_issue=KNOWN_BROKEN.get(f"{subdir}/{path.name}"),
                )
            )
