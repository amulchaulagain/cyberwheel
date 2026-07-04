"""Benchmark: simulation steps/second of the base env on the 15-host network.

Standalone script (see bench_network_build.py for the contract).
"""

import json
import statistics
import sys
import time
from importlib.resources import files
from types import SimpleNamespace

import yaml


def main() -> int:
    quick = "--quick" in sys.argv
    import cyberwheel.utils  # noqa: F401 -- resolves the import-order cycle
    from cyberwheel.cyberwheel_envs.cyberwheel import Cyberwheel
    from cyberwheel.network.network_base import Network
    from cyberwheel.utils import get_service_map
    from cyberwheel.utils.set_seed import set_seed

    set_seed(1)
    network = Network.create_network_from_yaml(
        files("cyberwheel.data.configs.network").joinpath("15-host-network.yaml")
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
    env = Cyberwheel(args, network)

    # Warmup (not measured), then fixed-size measured runs. No resets:
    # Cyberwheel.reset() is broken with inactive agents (known issue, see
    # smoke:base_env_reset_known_issue), and the base env steps fine past
    # num_steps.
    for _ in range(200):
        env.step()

    # The inactive-agent env steps at ~400k steps/s; size each sample to
    # hundreds of milliseconds so the timing is meaningful.
    steps_per_run = 25_000 if quick else 100_000
    samples = []
    for _ in range(2 if quick else 3):
        start = time.perf_counter()
        for _ in range(steps_per_run):
            env.step()
        samples.append(steps_per_run / (time.perf_counter() - start))

    print(
        json.dumps(
            {
                "metric": "sim_step_sps_15host",
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
