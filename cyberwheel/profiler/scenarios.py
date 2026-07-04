"""Profiling scenarios for the Cyberwheel environment.

Each scenario measures one representative workload three ways:

1. a plain timed loop (no instrumentation) that produces the gated metrics,
2. an instrumented loop (``PhaseAccumulator``) that attributes time to
   env phases (blue/red act, detector, observation, reward, ...),
3. an optional cProfile pass for function-level hotspots.

The passes are separate so the phase/hotspot observer overhead never skews
the gated numbers. Everything is seeded (``CYBERWHEEL_DETERMINISTIC``), so
runs on the same machine and commit are directly comparable.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from cyberwheel.profiler import hotspots as hotspots_mod
from cyberwheel.profiler.phase_timer import MethodInstrumenter, PhaseAccumulator

# Phases smaller than this are reported but not gated: at microsecond scale
# the relative tolerance would only ever trip on timer noise.
PHASE_GATE_FLOOR_MS = 0.02


@dataclass
class ScenarioOptions:
    env_config: str = "train_rl_red_agent_vs_rl_blue.yaml"
    network_config: str = "15-host-network.yaml"
    seed: int = 1
    quick: bool = False
    top: int = 12
    with_hotspots: bool = True
    rl_steps: int | None = None  # override for the rl-step sample size
    network_size: str | None = None  # override network_size_compatibility


@dataclass
class ScenarioResult:
    name: str
    params: dict
    metrics: dict = field(default_factory=dict)  # gated: name -> payload
    phase_metrics: dict = field(default_factory=dict)  # floor-gated: name -> payload
    phases: list = field(default_factory=list)  # display rows
    hotspots: dict | None = None
    notes: list = field(default_factory=list)

    def all_metrics(self) -> dict:
        return {**self.metrics, **self.phase_metrics}


def _metric(
    value: float, unit: str, samples=None, higher_is_better: bool = False
) -> dict:
    payload = {"value": value, "unit": unit, "higher_is_better": higher_is_better}
    if samples is not None:
        payload["samples"] = samples
    return payload


def _set_deterministic() -> None:
    # Must be set before any Network/HybridSetList construction: they read it
    # once at __init__ time.
    os.environ["CYBERWHEEL_DETERMINISTIC"] = "true"
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")


# --------------------------------------------------------------------------
# Environment builders
# --------------------------------------------------------------------------


def _network_config_path(network_config: str):
    return files("cyberwheel.data.configs.network").joinpath(network_config)


def build_rl_env(options: ScenarioOptions):
    """Construct a CyberwheelRL env the same way the training runner does."""
    _set_deterministic()
    import yaml

    import cyberwheel.utils  # noqa: F401 -- resolves the import-order cycle
    from cyberwheel.cyberwheel_envs.cyberwheel_rl import CyberwheelRL
    from cyberwheel.network.network_base import Network
    from cyberwheel.utils import YAMLConfig, get_service_map
    from cyberwheel.utils.set_seed import set_seed

    args = YAMLConfig(options.env_config)
    args.parse_config()
    args.network_config = options.network_config
    args.seed = options.seed
    args.deterministic = True
    if options.network_size:
        args.network_size_compatibility = options.network_size
    set_seed(options.seed)

    network = Network.create_network_from_yaml(
        _network_config_path(options.network_config)
    )
    args.service_mapping = {network.name: get_service_map(network)}
    args.agent_config = {}
    for agent_type in args.agents:
        agent_config = files(f"cyberwheel.data.configs.{agent_type}_agent").joinpath(
            args.agents[agent_type]
        )
        with open(agent_config) as f:
            args.agent_config[agent_type] = yaml.safe_load(f)

    env = CyberwheelRL(args, network=network, networks={network.name: network})
    env.reset(seed=options.seed)
    return env


def build_sim_env(options: ScenarioOptions):
    """Construct the base (inactive-agent) env, mirroring bench_sim_step."""
    _set_deterministic()
    from types import SimpleNamespace

    import yaml

    import cyberwheel.utils  # noqa: F401 -- resolves the import-order cycle
    from cyberwheel.cyberwheel_envs.cyberwheel import Cyberwheel
    from cyberwheel.network.network_base import Network
    from cyberwheel.utils import get_service_map
    from cyberwheel.utils.set_seed import set_seed

    set_seed(options.seed)
    network = Network.create_network_from_yaml(
        _network_config_path(options.network_config)
    )
    with open(
        files("cyberwheel.data.configs.red_agent").joinpath("art_agent.yaml")
    ) as f:
        red_config = yaml.safe_load(f)
    args = SimpleNamespace(
        host_config="host_defs_services.yaml",
        num_steps=100,
        campaign=False,
        service_mapping={network.name: get_service_map(network)},
        agent_config={"red": red_config},
    )
    return Cyberwheel(args, network)


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------


def run_rl_loop(
    env, steps: int, accumulator: PhaseAccumulator | None = None, action_seed: int = 7
) -> dict:
    """Step the RL env with mask-valid random actions; return timing totals."""
    rng = random.Random(action_seed)
    totals = {"step_ns": 0, "mask_ns": 0, "reset_ns": 0, "resets": 0}
    for _ in range(steps):
        if accumulator:
            accumulator.push("action_mask")
        start = time.perf_counter_ns()
        masks = env.action_mask
        totals["mask_ns"] += time.perf_counter_ns() - start
        if accumulator:
            accumulator.pop()

        actions = {}
        for agent, mask in masks.items():
            valid = [i for i, allowed in enumerate(mask) if allowed]
            actions[agent] = rng.choice(valid)

        if accumulator:
            accumulator.push("step")
        start = time.perf_counter_ns()
        _, _, done, _, _ = env.step(actions)
        totals["step_ns"] += time.perf_counter_ns() - start
        if accumulator:
            accumulator.pop()

        if done:
            if accumulator:
                accumulator.push("reset")
            start = time.perf_counter_ns()
            env.reset()
            totals["reset_ns"] += time.perf_counter_ns() - start
            if accumulator:
                accumulator.pop()
            totals["resets"] += 1
    return totals


def run_sim_loop(env, steps: int, accumulator: PhaseAccumulator | None = None) -> int:
    """Step the base env ``steps`` times; return elapsed ns. No resets: base
    env reset is broken with inactive agents (known issue) and it steps fine
    past num_steps."""
    if accumulator:
        start = time.perf_counter_ns()
        for _ in range(steps):
            accumulator.push("step")
            env.step()
            accumulator.pop()
        return time.perf_counter_ns() - start
    start = time.perf_counter_ns()
    for _ in range(steps):
        env.step()
    return time.perf_counter_ns() - start


def instrument_rl_env(instrumenter: MethodInstrumenter, env) -> None:
    """Attach phase timers to the moving parts of a CyberwheelRL env.

    Duck-typed on purpose: agents/detectors are pluggable, so wrap whatever
    of the known surface exists on this particular env.
    """
    from cyberwheel.red_actions.actions import (
        ARTDiscovery,
        ARTImpact,
        ARTLateralMovement,
        ARTPingSweep,
        ARTPortScan,
        ARTPrivilegeEscalation,
        Nothing,
    )

    instrumenter.wrap_if_present(env.blue_agent, "act", "blue.act")
    instrumenter.wrap_if_present(env.red_agent, "act", "red.act")
    instrumenter.wrap_if_present(
        env.blue_agent, "get_observation_space", "blue.observation"
    )
    detector = getattr(getattr(env.blue_agent, "observation", None), "detector", None)
    instrumenter.wrap_if_present(detector, "obs", "blue.detector")
    instrumenter.wrap_if_present(
        env.red_agent, "get_observation_space", "red.observation"
    )
    instrumenter.wrap_if_present(
        getattr(env, "reward_calculator", None), "calculate_reward", "reward"
    )
    instrumenter.wrap_if_present(
        env.red_agent, "handle_network_change", "red.network_change"
    )
    instrumenter.wrap_if_present(
        getattr(env.blue_agent, "action_space", None),
        "select_action",
        "blue.select_action",
    )
    instrumenter.wrap_if_present(
        getattr(env.red_agent, "action_space", None),
        "select_action",
        "red.select_action",
    )
    for action_class in (
        ARTDiscovery,
        ARTImpact,
        ARTLateralMovement,
        ARTPingSweep,
        ARTPortScan,
        ARTPrivilegeEscalation,
        Nothing,
    ):
        instrumenter.wrap(action_class, "sim_execute", "red.sim_execute")


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


def scenario_network_build(options: ScenarioOptions) -> ScenarioResult:
    _set_deterministic()
    import cyberwheel.utils  # noqa: F401 -- resolves the import-order cycle
    from cyberwheel.network.network_base import Network
    from cyberwheel.utils import get_service_map
    from cyberwheel.utils.set_seed import set_seed

    samples = 1 if options.quick else 3
    path = _network_config_path(options.network_config)
    build_samples, map_samples = [], []
    for _ in range(samples):
        set_seed(options.seed)
        start = time.perf_counter_ns()
        network = Network.create_network_from_yaml(path)
        built = time.perf_counter_ns()
        get_service_map(network)
        mapped = time.perf_counter_ns()
        build_samples.append((built - start) / 1e6)
        map_samples.append((mapped - built) / 1e6)

    result = ScenarioResult(
        name="network-build",
        params={"network_config": options.network_config, "samples": samples},
        metrics={
            "network_build/build_ms": _metric(
                statistics.median(build_samples), "ms", build_samples
            ),
            "network_build/service_map_ms": _metric(
                statistics.median(map_samples), "ms", map_samples
            ),
        },
    )
    if options.with_hotspots:
        set_seed(options.seed)
        result.hotspots = hotspots_mod.profile_callable(
            lambda: get_service_map(Network.create_network_from_yaml(path)),
            options.top,
        )
    return result


def scenario_sim_step(options: ScenarioOptions) -> ScenarioResult:
    warmup = 200
    sample_steps = 10_000 if options.quick else 50_000
    samples = 2 if options.quick else 3
    instrumented_steps = 5_000 if options.quick else 20_000

    env = build_sim_env(options)
    run_sim_loop(env, warmup)
    step_samples = [
        run_sim_loop(env, sample_steps) / sample_steps / 1e6 for _ in range(samples)
    ]

    accumulator = PhaseAccumulator()
    instrumenter = MethodInstrumenter(accumulator)
    instrumenter.wrap_if_present(env.blue_agent, "act", "blue.act")
    instrumenter.wrap_if_present(env.red_agent, "act", "red.act")
    try:
        run_sim_loop(env, instrumented_steps, accumulator)
    finally:
        instrumenter.restore()

    result = ScenarioResult(
        name="sim-step",
        params={
            "network_config": options.network_config,
            "steps_per_sample": sample_steps,
            "samples": samples,
        },
        metrics={
            "sim_step/step_ms": _metric(
                statistics.median(step_samples), "ms/step", step_samples
            )
        },
        phases=accumulator.rows(instrumented_steps, anchor="step"),
        notes=[
            "base env with inactive agents; phase table from a separate instrumented pass"
        ],
    )
    if options.with_hotspots:
        env2 = build_sim_env(options)
        run_sim_loop(env2, warmup)
        result.hotspots = hotspots_mod.profile_callable(
            lambda: run_sim_loop(env2, instrumented_steps), options.top
        )
    return result


def scenario_rl_step(options: ScenarioOptions) -> ScenarioResult:
    warmup = 100
    sample_steps = options.rl_steps or (300 if options.quick else 1000)
    samples = 2 if options.quick else 3
    instrumented_steps = 300 if options.quick else 1000

    env = build_rl_env(options)
    run_rl_loop(env, warmup)
    step_samples, mask_samples = [], []
    reset_ns, resets = 0, 0
    for _ in range(samples):
        totals = run_rl_loop(env, sample_steps)
        step_samples.append(totals["step_ns"] / sample_steps / 1e6)
        mask_samples.append(totals["mask_ns"] / sample_steps / 1e6)
        reset_ns += totals["reset_ns"]
        resets += totals["resets"]

    metrics = {
        "rl_step/step_ms": _metric(
            statistics.median(step_samples), "ms/step", step_samples
        ),
        "rl_step/action_mask_ms": _metric(
            statistics.median(mask_samples), "ms/step", mask_samples
        ),
    }
    if resets:
        metrics["rl_step/reset_ms"] = _metric(reset_ns / resets / 1e6, "ms/reset")

    # Separate instrumented pass on a fresh env: same seed, same trajectory.
    env2 = build_rl_env(options)
    run_rl_loop(env2, warmup)
    accumulator = PhaseAccumulator()
    instrumenter = MethodInstrumenter(accumulator)
    instrument_rl_env(instrumenter, env2)
    try:
        run_rl_loop(env2, instrumented_steps, accumulator)
    finally:
        instrumenter.restore()

    phase_rows = accumulator.rows(instrumented_steps, anchor="step")
    phase_metrics = {}
    for row in phase_rows:
        if row["phase"] in ("step", "reset", "action_mask"):
            continue  # already covered by the gated totals above
        phase_metrics[f"rl_step/phase/{row['phase']}"] = _metric(
            row["exclusive_ms_per_unit"], "ms/step"
        )

    result = ScenarioResult(
        name="rl-step",
        params={
            "env_config": options.env_config,
            "network_config": options.network_config,
            "steps_per_sample": sample_steps,
            "samples": samples,
        },
        metrics=metrics,
        phase_metrics=phase_metrics,
        phases=phase_rows,
        notes=[
            "full RL env step: mask-valid random actions, detector, obs and reward",
            "phase table from a separate instrumented pass (exclusive = minus nested phases)",
        ],
    )
    if options.with_hotspots:
        env3 = build_rl_env(options)
        run_rl_loop(env3, warmup)
        result.hotspots = hotspots_mod.profile_callable(
            lambda: run_rl_loop(env3, instrumented_steps), options.top
        )
    return result


def scenario_train(options: ScenarioOptions) -> ScenarioResult:
    """cProfile a tiny real training run through the actual CLI.

    Function-level only (no phase table: the run is a subprocess) and never
    gated: everything under cProfile is 2-4x slower, so the SPS reported here
    is informational. The gated training metric is the perf suite's
    ``train_sps_15host``.
    """
    _set_deterministic()
    import cyberwheel

    code_root = Path(cyberwheel.__file__).resolve().parents[1]
    data_root = Path(cyberwheel.__file__).resolve().parent / "data"
    experiment = f"TEST_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}_profile"
    total_timesteps = "128" if options.quick else "256"
    temp_dir = Path(tempfile.mkdtemp(prefix="cyberwheel-profiler-"))
    stats_path = temp_dir / "train.pstats"

    argv = [
        sys.executable,
        "-m",
        "cProfile",
        "-o",
        str(stats_path),
        "-m",
        "cyberwheel",
        "train",
        options.env_config,
        "--experiment-name",
        experiment,
        "--network-config",
        options.network_config,
        "--total-timesteps",
        total_timesteps,
        "--num-steps",
        "32",
        "--num-envs",
        "1",
        "--num-saves",
        "1",
        "--num-minibatches",
        "2",
        "--update-epochs",
        "2",
        "--eval-episodes",
        "1",
        "--async-env",
        "false",
        "--track",
        "false",
        "--device",
        "cpu",
        "--seed",
        str(options.seed),
        "--deterministic",
        "true",
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(code_root),
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"train scenario subprocess exited {proc.returncode}; "
                f"stderr tail: {proc.stderr[-800:]}"
            )
        sps_values = [
            int(m) for m in re.findall(r"^SPS:\s*(\d+)", proc.stdout, re.MULTILINE)
        ]
        result = ScenarioResult(
            name="train",
            params={
                "env_config": options.env_config,
                "network_config": options.network_config,
                "total_timesteps": int(total_timesteps),
            },
            hotspots=hotspots_mod.load_stats_file(stats_path, options.top),
            notes=[
                f"SPS under cProfile (informational only): {sps_values}",
                "not gated; the perf suite's train_sps_15host is the gated metric",
            ],
        )
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        for sub in ("models", "runs", "action_logs"):
            base = data_root / sub
            if base.is_dir():
                for entry in base.glob(f"{experiment}*"):
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink()


SCENARIOS = {
    "network-build": scenario_network_build,
    "sim-step": scenario_sim_step,
    "rl-step": scenario_rl_step,
    "train": scenario_train,
}
DEFAULT_SCENARIOS = ("network-build", "sim-step", "rl-step")
