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
    for column in ("episode", "step", "seed"):
        check(
            column in rows[0], f"action log missing column {column!r}: {list(rows[0])}"
        )
    check(
        all(row["seed"] == "1" for row in rows),
        "seed column should be 1 on every row (--seed 1 --deterministic true)",
    )

    # Statistical summary artifact (written for every evaluate run).
    summary_path = DATA_ROOT / "action_logs" / f"{graph_name}.summary.json"
    check_file(summary_path)
    with open(summary_path) as f:
        summary = json.load(f)
    check(summary["seeds"] == [1], f"summary seeds {summary['seeds']} != [1]")
    check(summary["explicit_seeds"] is False, "single --seed run must not be explicit_seeds")
    check(
        len(summary["per_episode"]) == 1,
        f"summary per_episode has {len(summary['per_episode'])} entries, expected 1",
    )
    stat = summary["overall"]["total_reward"]
    check(stat["n"] == 1, f"overall total_reward n {stat['n']} != 1")
    check(
        stat["ci95_lo"] == stat["ci95_hi"] == stat["mean"],
        f"n=1 CI must collapse to the mean: {stat}",
    )
    csv_total = round(sum(float(row["reward"]) for row in rows), 4)
    check(
        abs(summary["per_episode"][0]["total_reward"] - csv_total) < 1e-3,
        f"summary episode total {summary['per_episode'][0]['total_reward']} "
        f"!= CSV reward sum {csv_total}",
    )
    reward_columns = {c for c in rows[0] if c.endswith("_reward")}
    check(
        set(summary["overall"]) == {"total_reward", *reward_columns},
        f"summary metrics {sorted(summary['overall'])} do not match CSV reward "
        f"columns {sorted(reward_columns)}",
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


def _smoke_summary_stats() -> Outcome:
    """The summary statistics helpers (t-table, CI math, aggregation) are correct."""
    import math

    from cyberwheel.utils.step_metrics import (
        build_evaluation_summary,
        mean_std_ci95,
        t_critical_95,
    )

    check(t_critical_95(1) == 12.706, f"t(df=1) {t_critical_95(1)} != 12.706")
    check(t_critical_95(30) == 2.042, f"t(df=30) {t_critical_95(30)} != 2.042")
    check(t_critical_95(31) == 1.96, f"t(df=31) {t_critical_95(31)} != 1.96")
    check(t_critical_95(200) == 1.96, f"t(df=200) {t_critical_95(200)} != 1.96")

    stat = mean_std_ci95([1.0, 2.0, 3.0])
    check(stat["mean"] == 2.0 and stat["std"] == 1.0, f"mean/std wrong: {stat}")
    half = 4.303 / math.sqrt(3)  # t(df=2) * std / sqrt(n)
    check(
        abs(stat["ci95_lo"] - (2.0 - half)) < 1e-3
        and abs(stat["ci95_hi"] - (2.0 + half)) < 1e-3,
        f"95% CI wrong: {stat}",
    )
    one = mean_std_ci95([5.0])
    check(
        one["std"] == 0.0 and one["ci95_lo"] == one["ci95_hi"] == one["mean"] == 5.0,
        f"n=1 stat block wrong: {one}",
    )
    empty = mean_std_ci95([])
    check(
        empty["n"] == 0 and empty["mean"] is None,
        f"n=0 stat block wrong: {empty}",
    )

    per_episode = [
        {"episode": i, "seed": seed, "steps": 10, "total_reward": float(i)}
        for i, seed in enumerate((7, 7, 9, 9))
    ]
    summary = build_evaluation_summary(
        seeds=[7, 9],
        explicit_seeds=True,
        deterministic=False,
        num_episodes=2,
        num_steps=10,
        per_episode=per_episode,
        metric_names=["total_reward"],
        graph_name="synthetic",
        experiment_name="synthetic",
    )
    check(
        len(summary["per_seed"]) == 2
        and [b["seed"] for b in summary["per_seed"]] == [7, 9],
        f"per_seed grouping wrong: {summary['per_seed']}",
    )
    check(
        summary["per_seed"][0]["metrics"]["total_reward"]["mean"] == 0.5
        and summary["per_seed"][1]["metrics"]["total_reward"]["mean"] == 2.5,
        f"per_seed means wrong: {summary['per_seed']}",
    )
    check(
        summary["overall"]["total_reward"]["n"] == 4
        and summary["total_episodes"] == 4,
        f"overall aggregation wrong: {summary['overall']}",
    )
    return Outcome(Status.PASS, "t-table, CI math, and summary aggregation OK")


def _smoke_evaluate_batch(run_id: str) -> Outcome:
    graph_name = f"{run_id}_eval_batch"
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
            # Explicit seeds must reseed per block even without --deterministic.
            "--deterministic",
            "false",
            "--seeds",
            "1,2",
        ],
        timeout=600,
    )
    check(
        proc.returncode == 0,
        f"batch evaluate exited {proc.returncode}; stderr tail: {proc.stderr[-800:]}",
    )
    csv_path = DATA_ROOT / "action_logs" / f"{graph_name}.csv"
    check_file(csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    check(
        len(rows) == 2 * num_steps,
        f"expected {2 * num_steps} action-log rows (2 seeds x 1 episode), got {len(rows)}",
    )
    check(
        sorted({row["episode"] for row in rows}) == ["0", "1"],
        "batch CSV must use a global episode index 0..1",
    )
    seeds_by_episode = {
        episode: {row["seed"] for row in rows if row["episode"] == episode}
        for episode in ("0", "1")
    }
    check(
        seeds_by_episode == {"0": {"1"}, "1": {"2"}},
        f"seed blocks wrong: {seeds_by_episode}",
    )

    summary_path = DATA_ROOT / "action_logs" / f"{graph_name}.summary.json"
    check_file(summary_path)
    with open(summary_path) as f:
        summary = json.load(f)
    check(summary["seeds"] == [1, 2], f"summary seeds {summary['seeds']} != [1, 2]")
    check(summary["explicit_seeds"] is True, "--seeds run must set explicit_seeds")
    check(
        summary["total_episodes"] == 2,
        f"total_episodes {summary['total_episodes']} != 2",
    )
    check(
        [b["seed"] for b in summary["per_seed"]] == [1, 2]
        and all(b["episodes"] == 1 for b in summary["per_seed"]),
        f"per_seed blocks wrong: {summary['per_seed']}",
    )
    stat = summary["overall"]["total_reward"]
    check(stat["n"] == 2, f"overall total_reward n {stat['n']} != 2")
    check(
        stat["min"] <= stat["mean"] <= stat["max"]
        and stat["ci95_lo"] <= stat["mean"] <= stat["ci95_hi"],
        f"overall stat block inconsistent: {stat}",
    )

    viz_dir = DATA_ROOT / "graphs" / graph_name
    with open(viz_dir / "meta.json") as f:
        meta = json.load(f)
    check(
        meta["episodes_written"] == [0, 1],
        f"meta episodes_written {meta['episodes_written']} != [0, 1]",
    )
    check(meta.get("seeds") == [1, 2], f"meta seeds {meta.get('seeds')} != [1, 2]")
    with open(viz_dir / "episode_1.json") as f:
        episode = json.load(f)
    check(
        len(episode["steps"]) == num_steps,
        f"episode_1 has {len(episode['steps'])} frames, expected {num_steps}",
    )
    return Outcome(
        Status.PASS,
        f"2-seed batch evaluated; {len(rows)} rows, global episodes [0, 1], "
        "summary + viz artifacts consistent",
    )


def _build_rl_env(blue_yaml: str):
    """Build a CyberwheelRL env on the 15-host net with the given blue agent."""
    import yaml

    import cyberwheel.utils  # noqa: F401 -- resolves the import-order cycle
    from cyberwheel.cyberwheel_envs.cyberwheel_rl import CyberwheelRL
    from cyberwheel.network.network_base import Network
    from cyberwheel.utils import YAMLConfig, get_service_map
    from cyberwheel.utils.set_seed import set_seed

    set_seed(1)
    args = YAMLConfig("train_rl_red_agent_vs_rl_blue.yaml")
    args.parse_config()
    args.network_config = _NETWORK
    args.seed = 1
    args.deterministic = True
    args.agents = {"red": "rl_red_agent.yaml", "blue": blue_yaml}
    network = Network.create_network_from_yaml(CONFIG_ROOT / "network" / _NETWORK)
    args.service_mapping = {network.name: get_service_map(network)}
    args.agent_config = {}
    for agent_type in args.agents:
        with open(CONFIG_ROOT / f"{agent_type}_agent" / args.agents[agent_type]) as f:
            args.agent_config[agent_type] = yaml.safe_load(f)
    env = CyberwheelRL(args, network=network, networks={network.name: network})
    env.reset(seed=1)
    return env, network, args


def _smoke_active_defense_actions() -> Outcome:
    """Quarantine/Restore/Patch mechanics + action-space size invariants."""
    import yaml

    from importlib.resources import files

    from cyberwheel.red_actions.actions.art_killchain_phase import ARTKillChainPhase
    from cyberwheel.red_agents.art_agent import ARTAgent

    env, net, args = _build_rl_env("active_defense_blue_agent.yaml")

    # Size invariants: new config uses the padded (host) size; the default
    # config keeps its historical size so old models still load.
    padded = env.blue_agent.action_space.padded_action_space_size(
        args.max_num_hosts, args.max_num_subnets
    )
    check(padded == 321, f"padded size {padded} != 321")
    check(env.blue_max_action_space_size == 321, f"new blue_max {env.blue_max_action_space_size} != 321")
    env2, _, _ = _build_rl_env("rl_blue_agent.yaml")
    check(
        env2.blue_max_action_space_size == 30,
        f"rl_blue_agent blue_max {env2.blue_max_action_space_size} != 30 (old models must still load)",
    )

    actions = {ac.name: ac.action for ac in env.blue_agent.action_space._action_checkers}
    red = env.red_agent
    real_hosts = [h for h in net.hosts.values() if not h.decoy]
    target = real_hosts[0]

    # Quarantine blocks red on that host.
    r = actions["quarantine_host"].execute(target)
    check(r.success and target.isolated, "quarantine did not isolate the host")
    check((target.name, target.subnet.name) in net.disconnected_nodes, "no disconnected edge")
    blocked = ARTKillChainPhase(red.current_host, target, valid_techniques=["Tx"]).sim_execute()
    check(blocked.attack_success is False, "isolated host was still attackable")
    check(actions["quarantine_host"].execute(target).success is False, "re-quarantine should fail")

    # Restore reconnects, cleans, and resets the RL red foothold.
    red.observation.add_host(target.name, escalated=1, impacted=1, on_host=1)
    red.current_host = target
    r = actions["restore_host"].execute(target)
    check(
        r.success and not target.isolated and not target.is_compromised and target.restored,
        "restore did not clear host state",
    )
    check(target.name in net.pending_restores, "restore did not queue a pending_restore")
    red.handle_host_state_changes()
    check(red.observation.obs[target.name]["escalated"] == 0, "obs escalated not cleared")
    check(red.observation.obs[target.name]["on_host"] == 0, "obs on_host not cleared")
    check(red.current_host == red._initial_host, "red not relocated to its entry host")
    check(not net.pending_restores, "pending_restores not drained")
    real_tid = next(t for lst in red.service_mapping[target.name].values() if lst for t in [lst[0]])
    healed = ARTKillChainPhase(red.current_host, target, valid_techniques=[real_tid]).sim_execute()
    check(healed.attack_success is True, "restored host should be attackable again")
    check(actions["restore_host"].execute(real_hosts[1]).success is False, "restore on clean host should fail")

    # ARTAgent (evaluate path) history-based foothold reset.
    args.agent_config["red"] = yaml.safe_load(
        open(files("cyberwheel.data.configs.red_agent").joinpath("art_agent.yaml"))
    )
    net.reset()
    art = ARTAgent(net, args)
    art_target = [h for h in net.hosts.values() if not h.decoy][2]
    khi_cls = type(next(iter(art.history.hosts.values())))
    khi = khi_cls(ip_address=art_target.ip_address)
    khi.escalated = True
    khi.impacted = True
    khi.on_host = True
    khi.last_step = 3
    khi.type = "Server"
    art.history.hosts[art_target.name] = khi
    art.current_host = art_target
    net.pending_restores.add(art_target.name)
    art.handle_host_state_changes()
    check(khi.escalated is False and khi.impacted is False and khi.last_step == -1, "ART khi not reset")
    check(art_target.name in art.unimpacted_hosts.data_set, "ART host not re-added to unimpacted")
    check(art_target.name in art.unimpacted_servers.data_set, "ART server not re-added")
    check(art.current_host == art._initial_host, "ART red not relocated")

    # Patch empties validity for one host without touching same-type siblings
    # or the shared service_mapping dict; reset restores the CVEs.
    from collections import defaultdict

    by_type = defaultdict(list)
    for host in real_hosts:
        if host.host_type and host.host_type.cve_list:
            by_type[host.host_type.name].append(host)
    pair = next(v for v in by_type.values() if len(v) >= 2)
    h0, h1 = pair[0], pair[1]
    orig = frozenset(h0.host_type.cve_list)
    r = actions["patch_host"].execute(h0)
    check(r.success and h0.patched and h0._pre_patch_host_type is not None, "patch did not apply")
    red.handle_host_state_changes()
    check(
        all(len(v) == 0 for v in red.service_mapping[h0.name].values()),
        "patched host still has valid techniques",
    )
    check(frozenset(h1.host_type.cve_list) == orig, "sibling CVEs changed (shared HostType leaked)")
    check(bool(args.service_mapping[net.name][h0.name]), "shared service_mapping dict was mutated")
    check(actions["patch_host"].execute(h0).success is False, "double patch should fail")
    net.reset()
    check(frozenset(h0.host_type.cve_list) == orig, "reset did not restore CVEs")
    check(h0.patched is False and h0._pre_patch_host_type is None, "reset did not clear patch state")

    return Outcome(Status.PASS, "quarantine/restore/patch mechanics + size invariants OK")


def _smoke_active_defense_e2e(run_id: str) -> Outcome:
    """Train + evaluate through the CLI with the active-defense blue agent."""
    exp = f"{run_id}_ad"
    train = run_cli(
        [
            "-m", "cyberwheel", "train", "train_rl_red_agent_vs_rl_blue.yaml",
            "--experiment-name", exp, "--network-config", _NETWORK,
            "--blue-agent", "active_defense_blue_agent.yaml",
            "--total-timesteps", "16", "--num-steps", "8", "--num-envs", "1",
            "--num-saves", "1", "--num-minibatches", "2", "--update-epochs", "2",
            "--eval-episodes", "1", "--track", "false", "--device", "cpu",
            "--seed", "1", "--deterministic", "true",
        ],
        timeout=900,
    )
    check(train.returncode == 0, f"train exited {train.returncode}; stderr: {train.stderr[-800:]}")
    model_dir = DATA_ROOT / "models" / exp
    check_file(model_dir / "blue_agent.pt", min_size=1024)
    check_file(model_dir / "red_agent.pt", min_size=1024)

    graph_name = f"{exp}_eval"
    eval_proc = run_cli(
        [
            "-m", "cyberwheel", "evaluate", "evaluate_rl_red_vs_rl_blue.yaml",
            "--experiment-name", exp, "--network-config", _NETWORK,
            "--blue-agent", "active_defense_blue_agent.yaml",
            "--num-episodes", "1", "--num-steps", "10", "--graph-name", graph_name,
            "--download-model", "false", "--visualize", "true",
            "--seed", "1", "--deterministic", "true",
        ],
        timeout=600,
    )
    check(eval_proc.returncode == 0, f"evaluate exited {eval_proc.returncode}; stderr: {eval_proc.stderr[-800:]}")
    csv_path = DATA_ROOT / "action_logs" / f"{graph_name}.csv"
    check_file(csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    check(len(rows) == 10, f"expected 10 action-log rows, got {len(rows)}")
    check("blue_action_success" in rows[0], f"missing blue_action_success column: {list(rows[0])}")
    check_file(DATA_ROOT / "action_logs" / f"{graph_name}.summary.json")
    check_file(DATA_ROOT / "graphs" / graph_name / "meta.json")
    return Outcome(Status.PASS, "active-defense train + evaluate e2e OK (RL train, ART evaluate)")


def _smoke_probabilistic_exploits() -> Outcome:
    """CVSS-weighted exploit gate: severity math + probability extremes."""
    import cyberwheel.utils  # noqa: F401 -- resolves the import-order cycle
    from cyberwheel.red_actions.actions.art_killchain_phase import ARTKillChainPhase
    from cyberwheel.utils.exploit_model import (
        ExploitModel,
        build_exploit_model,
        cve_severity,
    )

    # Deterministic per-CVE severity in [0, 1]; overrides win.
    s1 = cve_severity("CVE-2021-44228")
    check(s1 == cve_severity("CVE-2021-44228"), "cve_severity is not deterministic")
    check(0.0 <= s1 <= 1.0, f"cve_severity {s1} outside [0, 1]")
    check(cve_severity("CVE-2021-44228", {"CVE-2021-44228": 0.3}) == 0.3, "override ignored")

    # build_exploit_model gating: None unless the flag is on.
    class _Args:
        pass

    args = _Args()
    check(build_exploit_model(args) is None, "model should be None when flag absent")
    args.probabilistic_exploits = False
    check(build_exploit_model(args) is None, "model should be None when flag False")
    args.probabilistic_exploits = True
    model = build_exploit_model(args)
    check(model is not None and model.enabled, "model should be enabled when flag True")
    args.exploit_severity_config = "cvss_example.yaml"
    check(
        build_exploit_model(args).overrides.get("CVE-2021-44228") == 1.0,
        "severity override config did not load",
    )

    env, net, _ = _build_rl_env("rl_blue_agent.yaml")
    smap = env.red_agent.service_mapping
    target = tid = valid = None
    for host_name, phases in smap.items():
        for kcp, ids in phases.items():
            if ids and getattr(kcp, "__name__", "") not in ("ARTPingSweep", "ARTPortScan"):
                target, tid, valid = net.hosts[host_name], ids[0], ids
                break
        if target is not None:
            break
    check(target is not None, "no host with a valid killchain technique found")

    try:
        # Probability 0 fails even with valid techniques; probability 1 succeeds.
        ARTKillChainPhase.exploit_model = ExploitModel(True, 0.0, 0.0)
        r0 = ARTKillChainPhase(target, target, valid_techniques=valid).sim_execute()
        check(r0.attack_success is False, "p=0 exploit should fail")
        ARTKillChainPhase.exploit_model = ExploitModel(True, 1.0, 1.0)
        r1 = ARTKillChainPhase(target, target, valid_techniques=valid).sim_execute()
        check(r1.attack_success is True, "p=1 exploit should succeed")
        # Disabled model => the legacy binary path (always succeeds).
        ARTKillChainPhase.exploit_model = None
        rb = ARTKillChainPhase(target, target, valid_techniques=valid).sim_execute()
        check(rb.attack_success is True, "binary path should succeed on a valid technique")

        # Higher CVE severity => higher success probability.
        hi = ExploitModel(True, 0.0, 1.0, {c: 1.0 for c in target.host_type.cve_list})
        lo = ExploitModel(True, 0.0, 1.0, {c: 0.0 for c in target.host_type.cve_list})
        check(
            hi.success_probability(target, tid) > lo.success_probability(target, tid),
            "success probability is not monotonic in CVE severity",
        )
    finally:
        ARTKillChainPhase.exploit_model = None

    return Outcome(Status.PASS, "severity math, gating, and probability extremes OK")


def _smoke_probabilistic_exploits_e2e(run_id: str) -> Outcome:
    """Train + evaluate with probabilistic exploits enabled via CLI flag."""
    exp = f"{run_id}_pe"
    train = run_cli(
        [
            "-m", "cyberwheel", "train", "train_rl_red_agent_vs_rl_blue.yaml",
            "--experiment-name", exp, "--network-config", _NETWORK,
            "--probabilistic-exploits", "true",
            "--exploit-severity-config", "cvss_example.yaml",
            "--total-timesteps", "16", "--num-steps", "8", "--num-envs", "1",
            "--num-saves", "1", "--num-minibatches", "2", "--update-epochs", "2",
            "--eval-episodes", "1", "--track", "false", "--device", "cpu",
            "--seed", "1", "--deterministic", "true",
        ],
        timeout=900,
    )
    check(train.returncode == 0, f"train exited {train.returncode}; stderr: {train.stderr[-800:]}")
    check_file(DATA_ROOT / "models" / exp / "blue_agent.pt", min_size=1024)

    graph_name = f"{exp}_eval"
    eval_proc = run_cli(
        [
            "-m", "cyberwheel", "evaluate", "evaluate_rl_red_vs_rl_blue.yaml",
            "--experiment-name", exp, "--network-config", _NETWORK,
            "--probabilistic-exploits", "true",
            "--num-episodes", "1", "--num-steps", "10", "--graph-name", graph_name,
            "--download-model", "false", "--visualize", "true",
            "--seed", "1", "--deterministic", "true",
        ],
        timeout=600,
    )
    check(eval_proc.returncode == 0, f"evaluate exited {eval_proc.returncode}; stderr: {eval_proc.stderr[-800:]}")
    csv_path = DATA_ROOT / "action_logs" / f"{graph_name}.csv"
    check_file(csv_path)
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    check(len(rows) == 10, f"expected 10 action-log rows, got {len(rows)}")
    check_file(DATA_ROOT / "action_logs" / f"{graph_name}.summary.json")
    return Outcome(Status.PASS, "probabilistic-exploit train + evaluate e2e OK")


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
            name="smoke:summary_stats_math",
            suite=SUITE,
            fn=_smoke_summary_stats,
            timeout_s=60.0,
        )
    )
    registry.add(
        TestCase(
            name="smoke:evaluate_batch_e2e",
            suite=SUITE,
            fn=(lambda: _smoke_evaluate_batch(ctx.run_id)),
            timeout_s=600.0,
            depends_on="smoke:train_e2e",
        )
    )
    registry.add(
        TestCase(
            name="smoke:active_defense_actions",
            suite=SUITE,
            fn=_smoke_active_defense_actions,
            timeout_s=300.0,
        )
    )
    registry.add(
        TestCase(
            name="smoke:active_defense_e2e",
            suite=SUITE,
            fn=(lambda: _smoke_active_defense_e2e(ctx.run_id)),
            timeout_s=900.0,
        )
    )
    registry.add(
        TestCase(
            name="smoke:probabilistic_exploits",
            suite=SUITE,
            fn=_smoke_probabilistic_exploits,
            timeout_s=300.0,
        )
    )
    registry.add(
        TestCase(
            name="smoke:probabilistic_exploits_e2e",
            suite=SUITE,
            fn=(lambda: _smoke_probabilistic_exploits_e2e(ctx.run_id)),
            timeout_s=900.0,
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
