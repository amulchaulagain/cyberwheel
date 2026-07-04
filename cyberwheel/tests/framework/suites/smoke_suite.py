"""Smoke suite: the environment, training, and evaluation work end-to-end.

Training and evaluation run as subprocesses through the real CLI
(``python -m cyberwheel <mode> <config> --overrides``) at tiny scale, so the
whole dispatch → config → runner path is exercised. The base environment is
constructed in-process (the ``run`` CLI mode is a known pre-existing issue,
covered by an xfail case).
"""

from __future__ import annotations

import csv
import json

import yaml

from cyberwheel.tests.framework.core import (
    CONFIG_ROOT,
    DATA_ROOT,
    Context,
    Outcome,
    Registry,
    Status,
    TestCase,
    check,
    check_file,
    run_cli,
)

SUITE = "smoke"

_NETWORK = "15-host-network.yaml"


def _build_base_env():
    from types import SimpleNamespace

    import cyberwheel.utils  # noqa: F401 -- resolves the import-order cycle
    from cyberwheel.cyberwheel_envs.cyberwheel import Cyberwheel
    from cyberwheel.network.network_base import Network
    from cyberwheel.utils import get_service_map

    network = Network.create_network_from_yaml(CONFIG_ROOT / "network" / _NETWORK)
    with open(CONFIG_ROOT / "red_agent" / "art_agent.yaml") as f:
        red_config = yaml.safe_load(f)
    args = SimpleNamespace(
        host_config="host_defs_services.yaml",
        num_steps=25,
        campaign=False,
        service_mapping={network.name: get_service_map(network)},
        agent_config={"red": red_config},
    )
    return Cyberwheel(args, network)


def _smoke_base_env() -> Outcome:
    """Build the base (inactive-agent) env directly and step it."""
    env = _build_base_env()
    for step in range(25):
        info = env.step()
        check(
            isinstance(info, dict),
            f"step() must return a dict, got {type(info).__name__}",
        )
        for key in ("red_agent_result", "blue_agent_result"):
            check(key in info, f"step() result missing {key!r} at step {step}")
        check(
            env.current_step == step + 1,
            f"current_step {env.current_step} != {step + 1}",
        )
    return Outcome(Status.PASS, "25 steps OK")


def _smoke_base_env_reset() -> Outcome:
    env = _build_base_env()
    env.step()
    # Known pre-existing issue: InactiveRedAgent.reset() calls
    # ARTAgent.reset() without its required (network, service_mapping) args
    # (inactive_red_agent.py:31-32 vs art_agent.py:383), so the base env
    # cannot reset. Success here means the bug got fixed (XPASS_WARN).
    env.reset()
    check(env.current_step == 0, f"current_step {env.current_step} != 0 after reset")
    for _ in range(5):
        env.step()
    return Outcome(Status.PASS, "reset + 5 steps OK")


def _smoke_train(run_id: str, async_env: bool = False) -> Outcome:
    # Async uses >1 env so AsyncVectorEnv actually spawns workers (the path
    # that must pickle the make_env closures to subprocesses).
    num_envs = "2" if async_env else "1"
    proc = run_cli(
        [
            "-m",
            "cyberwheel",
            "train",
            "train_rl_red_agent_vs_rl_blue.yaml",
            "--experiment-name",
            run_id,
            "--network-config",
            _NETWORK,
            "--total-timesteps",
            "32" if async_env else "16",
            "--num-steps",
            "8",
            "--num-envs",
            num_envs,
            "--num-saves",
            "1",
            "--num-minibatches",
            "2",
            "--update-epochs",
            "2",
            "--eval-episodes",
            "1",
            "--async-env",
            "true" if async_env else "false",
            "--track",
            "false",
            "--device",
            "cpu",
            "--seed",
            "1",
            "--deterministic",
            "true",
        ],
        timeout=900,
    )
    check(
        proc.returncode == 0,
        f"train exited {proc.returncode}; stderr tail: {proc.stderr[-800:]}",
    )
    model_dir = DATA_ROOT / "models" / run_id
    check_file(model_dir / "blue_agent.pt", min_size=1024)
    check_file(model_dir / "red_agent.pt", min_size=1024)
    runs_dir = DATA_ROOT / "runs" / run_id
    tfevents = list(runs_dir.glob("*tfevents*")) if runs_dir.is_dir() else []
    check(bool(tfevents), f"no tensorboard events written under {runs_dir}")
    check("SPS:" in proc.stdout, "training stdout did not report SPS")
    mode = "async" if async_env else "sync"
    return Outcome(Status.PASS, f"{mode} train OK; models saved; tfevents written")


def _smoke_evaluate(run_id: str) -> Outcome:
    graph_name = f"{run_id}_eval"
    num_steps = 10
    proc = run_cli(
        [
            "-m",
            "cyberwheel",
            "evaluate",
            "evaluate_rl_red_vs_rl_blue.yaml",
            "--experiment-name",
            run_id,
            "--network-config",
            _NETWORK,
            "--num-episodes",
            "1",
            "--num-steps",
            str(num_steps),
            "--graph-name",
            graph_name,
            "--download-model",
            "false",
            "--visualize",
            "true",
            "--seed",
            "1",
            "--deterministic",
            "true",
        ],
        timeout=600,
    )
    check(
        proc.returncode == 0,
        f"evaluate exited {proc.returncode}; stderr tail: {proc.stderr[-800:]}",
    )
    csv_path = DATA_ROOT / "action_logs" / f"{graph_name}.csv"
    check_file(csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    check(
        len(rows) == num_steps,
        f"expected {num_steps} action-log rows, got {len(rows)}",
    )
    for column in ("episode", "step"):
        check(
            column in rows[0], f"action log missing column {column!r}: {list(rows[0])}"
        )

    # Visualization artifacts (evaluate ran with --visualize true).
    viz_dir = DATA_ROOT / "graphs" / graph_name
    for artifact in ("meta.json", "layout.json", "episode_0.json"):
        check_file(viz_dir / artifact)
    with open(viz_dir / "layout.json") as f:
        layout = json.load(f)
    node_count = len(layout["nodes"])
    kinds = {node["kind"] for node in layout["nodes"]}
    check(
        kinds == {"router", "subnet", "host"},
        f"layout node kinds unexpected: {kinds}",
    )
    check(
        all(0 <= a < node_count and 0 <= b < node_count for a, b in layout["edges"]),
        "layout edge references out-of-range node ids",
    )
    check(bool(layout["decoy_slots"]), "layout has no decoy_slots")
    with open(viz_dir / "meta.json") as f:
        meta = json.load(f)
    check(
        meta["episodes_written"] == [0],
        f"meta episodes_written {meta['episodes_written']} != [0]",
    )
    with open(viz_dir / "episode_0.json") as f:
        episode = json.load(f)
    check(
        len(episode["steps"]) == num_steps,
        f"episode_0 has {len(episode['steps'])} frames, expected {num_steps}",
    )
    for frame in episode["steps"]:
        check("red" in frame and "blue" in frame, "frame missing red/blue action record")
    return Outcome(
        Status.PASS,
        f"1 episode x {num_steps} steps evaluated; action log has {len(rows)} rows; "
        f"viz artifacts written ({node_count} layout nodes)",
        {"columns": list(rows[0])},
    )


def _smoke_viz_layout_deterministic() -> Outcome:
    """compute_layout is deterministic and decoy slots are stable."""
    import cyberwheel.utils  # noqa: F401 -- resolves the import-order cycle
    from cyberwheel.network.network_base import Network
    from cyberwheel.visualization import compute_layout
    from cyberwheel.visualization.layout import decoy_slot_position

    layouts = []
    for _ in range(2):
        network = Network.create_network_from_yaml(CONFIG_ROOT / "network" / _NETWORK)
        layouts.append(compute_layout(network))

    def geometry(layout: dict) -> str:
        # Host IPs come from a randomized DHCP draw (seeded only in real
        # runs), so they are display metadata, not layout — strip them.
        stripped = dict(layout, nodes=[
            {k: v for k, v in node.items() if k != "ip"} for node in layout["nodes"]
        ])
        return json.dumps(stripped, sort_keys=True)

    check(
        geometry(layouts[0]) == geometry(layouts[1]),
        "layout geometry differs across identical builds",
    )

    layout = layouts[0]
    hosts = [n for n in layout["nodes"] if n["kind"] == "host"]
    check(len(hosts) == 15, f"expected 15 host nodes, got {len(hosts)}")
    subnet_name, slots = next(iter(layout["decoy_slots"].items()))
    a = decoy_slot_position(slots, 0)
    b = decoy_slot_position(slots, 0)
    check(a == b, "decoy_slot_position is not stable")
    check(
        a != decoy_slot_position(slots, 1),
        f"decoy slots 0 and 1 collide for subnet {subnet_name}",
    )
    return Outcome(
        Status.PASS,
        f"layout deterministic; {len(layout['nodes'])} nodes, "
        f"{len(layout['edges'])} edges",
    )


def _smoke_run_mode(run_id: str) -> Outcome:
    proc = run_cli(
        [
            "-m",
            "cyberwheel",
            "run",
            "cyberwheel.yaml",
            "--experiment-name",
            f"{run_id}_run",
            "--num-episodes",
            "1",
            "--num-steps",
            "5",
        ],
        timeout=300,
    )
    # Known pre-existing issue: baseline_runner sets a host-keyed
    # service_mapping and never sets agent_config, which ARTAgent requires
    # (baseline_runner.py:23 vs art_agent.py:101). A zero exit here means the
    # bug got fixed; the runner will flag it as XPASS_WARN.
    check(
        proc.returncode == 0,
        f"run mode exited {proc.returncode}; stderr tail: {proc.stderr[-400:]}",
    )
    return Outcome(Status.PASS, "run mode completed")


def register(registry: Registry, ctx: Context) -> None:
    registry.add(
        TestCase(
            name="smoke:base_env_step",
            suite=SUITE,
            fn=_smoke_base_env,
            timeout_s=300.0,
        )
    )
    registry.add(
        TestCase(
            name="smoke:base_env_reset_known_issue",
            suite=SUITE,
            fn=_smoke_base_env_reset,
            timeout_s=300.0,
            known_issue=(
                "base env reset crashes: InactiveRedAgent.reset() calls "
                "ARTAgent.reset() without its required arguments"
            ),
        )
    )
    registry.add(
        TestCase(
            name="smoke:train_e2e",
            suite=SUITE,
            fn=(lambda: _smoke_train(ctx.run_id)),
            timeout_s=900.0,
        )
    )
    registry.add(
        TestCase(
            name="smoke:train_async_e2e",
            suite=SUITE,
            fn=(lambda: _smoke_train(f"{ctx.run_id}_async", async_env=True)),
            timeout_s=900.0,
        )
    )
    registry.add(
        TestCase(
            name="smoke:evaluate_e2e",
            suite=SUITE,
            fn=(lambda: _smoke_evaluate(ctx.run_id)),
            timeout_s=600.0,
            depends_on="smoke:train_e2e",
        )
    )
    registry.add(
        TestCase(
            name="smoke:viz_layout_deterministic",
            suite=SUITE,
            fn=_smoke_viz_layout_deterministic,
            timeout_s=300.0,
        )
    )
    registry.add(
        TestCase(
            name="smoke:run_mode_known_issue",
            suite=SUITE,
            fn=(lambda: _smoke_run_mode(ctx.run_id)),
            timeout_s=300.0,
            known_issue=(
                "run mode crashes: baseline_runner.py passes a host-keyed "
                "service_mapping and no agent_config to ARTAgent"
            ),
        )
    )
