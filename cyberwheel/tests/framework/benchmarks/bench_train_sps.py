"""Benchmark: PPO training steps/second through the real train CLI.

Standalone script (see bench_network_build.py for the contract). Shells out
to ``python -m cyberwheel train`` so the measured path is the real one; the
``cyberwheel`` package that runs is whichever the inherited PYTHONPATH/cwd
resolves (this is how the perf gate measures a parent-commit worktree).
Cleans up its own TEST_* artifacts inside whatever tree it measured.
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    quick = "--quick" in sys.argv
    import cyberwheel

    code_root = Path(cyberwheel.__file__).resolve().parents[1]
    data_root = Path(cyberwheel.__file__).resolve().parent / "data"
    experiment = f"TEST_{time.strftime('%Y%m%d%H%M%S')}_{os.getpid()}_bench"

    total_timesteps = "128" if quick else "256"  # 32-step rollouts -> 4 or 8 updates
    argv = [
        sys.executable,
        "-m",
        "cyberwheel",
        "train",
        "train_rl_red_agent_vs_rl_blue.yaml",
        "--experiment-name",
        experiment,
        "--network-config",
        "15-host-network.yaml",
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
        "1",
        "--deterministic",
        "true",
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=1200,
            cwd=str(code_root),
            env=os.environ.copy(),
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr[-2000:])
            return 1
        sps_values = [
            int(m) for m in re.findall(r"^SPS:\s*(\d+)", proc.stdout, re.MULTILINE)
        ]
        # Drop the first updates: lazy imports and allocator warmup dominate.
        warm = sps_values[2:] if len(sps_values) > 2 else sps_values
        if not warm:
            sys.stderr.write("no SPS lines found in training output\n")
            return 1
        print(
            json.dumps(
                {
                    "metric": "train_sps_15host",
                    "value": statistics.median(warm),
                    "samples": warm,
                    "unit": "steps/s",
                    "higher_is_better": True,
                }
            )
        )
        return 0
    finally:
        for sub in ("models", "runs", "action_logs"):
            base = data_root / sub
            if base.is_dir():
                for entry in base.glob(f"{experiment}*"):
                    (
                        shutil.rmtree(entry, ignore_errors=True)
                        if entry.is_dir()
                        else entry.unlink()
                    )


if __name__ == "__main__":
    sys.exit(main())
