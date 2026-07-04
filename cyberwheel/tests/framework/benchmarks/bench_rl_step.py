"""Benchmark: full RL env steps/second (CyberwheelRL) on the 15-host network.

Standalone script (see bench_network_build.py for the contract). Complements
bench_sim_step (inactive-agent base env) by exercising the complete RL step
path: action-mask computation, blue action execution, red ART action, the
detector stack, both observations, and the reward — without any PPO noise.
Actions are mask-valid random choices from a dedicated seeded RNG.

Deliberately self-contained (no cyberwheel.profiler import): the perf gate's
``--compare-rev`` runs this script against a parent-commit worktree that may
not contain the profiler package.
"""

import json
import os
import random
import statistics
import sys
import time
from importlib.resources import files


def build_env(seed: int):
    os.environ["CYBERWHEEL_DETERMINISTIC"] = "true"
    import yaml

    import cyberwheel.utils  # noqa: F401 -- resolves the import-order cycle
    from cyberwheel.cyberwheel_envs.cyberwheel_rl import CyberwheelRL
    from cyberwheel.network.network_base import Network
    from cyberwheel.utils import YAMLConfig, get_service_map
    from cyberwheel.utils.set_seed import set_seed

    args = YAMLConfig("train_rl_red_agent_vs_rl_blue.yaml")
    args.parse_config()
    args.network_config = "15-host-network.yaml"
    args.seed = seed
    args.deterministic = True
    set_seed(seed)

    network = Network.create_network_from_yaml(
        files("cyberwheel.data.configs.network").joinpath(args.network_config)
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
    env.reset(seed=seed)
    return env


def run_steps(env, steps: int, rng: random.Random) -> float:
    """Step the env ``steps`` times; return elapsed seconds of env work
    (masks + step + resets), excluding the driver's action sampling."""
    elapsed_ns = 0
    for _ in range(steps):
        start = time.perf_counter_ns()
        masks = env.action_mask
        elapsed_ns += time.perf_counter_ns() - start

        actions = {}
        for agent, mask in masks.items():
            valid = [i for i, allowed in enumerate(mask) if allowed]
            actions[agent] = rng.choice(valid)

        start = time.perf_counter_ns()
        _, _, done, _, _ = env.step(actions)
        if done:
            env.reset()
        elapsed_ns += time.perf_counter_ns() - start
    return elapsed_ns / 1e9


def main() -> int:
    quick = "--quick" in sys.argv
    env = build_env(seed=1)
    rng = random.Random(7)

    run_steps(env, 200, rng)  # warmup

    steps_per_run = 500 if quick else 2000
    samples = []
    for _ in range(2 if quick else 3):
        elapsed = run_steps(env, steps_per_run, rng)
        samples.append(steps_per_run / elapsed)

    print(
        json.dumps(
            {
                "metric": "rl_step_sps_15host",
                "value": statistics.median(samples),
                "samples": samples,
                "unit": "steps/s",
                "higher_is_better": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
