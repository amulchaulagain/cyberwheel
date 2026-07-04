"""Benchmark: wall time to build the 200-host network from YAML.

Standalone script: imports only ``cyberwheel`` + stdlib so it can measure any
checkout of the library via PYTHONPATH (the parent commit need not contain
this file). Prints one JSON object on the last stdout line.
"""

import json
import statistics
import sys
import time
from importlib.resources import files


def main() -> int:
    quick = "--quick" in sys.argv
    import cyberwheel.utils  # noqa: F401 -- resolves the import-order cycle
    from cyberwheel.network.network_base import Network

    config_dir = files("cyberwheel.data.configs.network")

    # Warm imports/IO with a small build that is not measured.
    Network.create_network_from_yaml(config_dir.joinpath("15-host-network.yaml"))

    samples = []
    for _ in range(2 if quick else 3):
        start = time.perf_counter()
        Network.create_network_from_yaml(config_dir.joinpath("200-host-network.yaml"))
        samples.append(time.perf_counter() - start)

    print(
        json.dumps(
            {
                "metric": "network_build_200host_s",
                "value": statistics.median(samples),
                "samples": samples,
                "unit": "s",
                "higher_is_better": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
