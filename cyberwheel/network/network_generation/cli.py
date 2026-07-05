"""CLI for the parameterized network generator.

    python -m cyberwheel.network.network_generation --name my_net --num-hosts 40 ...

Writes a network config into ``cyberwheel/data/configs/network/`` by default (so
it auto-appears in the web UI and config suite), or to ``--output PATH``. Use
``--stdout`` to print the YAML, or ``--layout-only`` to print the deterministic
canvas layout JSON (the web-UI preview seam) without writing a file.
"""

import argparse
import json
import os
import re
import sys
from importlib.resources import files

# Same charset the server's /api/networks/generate enforces.
_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")

import yaml

from cyberwheel.network.network_generation.parameterized_generator import (
    DEFAULT_SERVER_TYPES,
    generate_network_dict,
    validate_params,
)

EXIT_OK = 0
EXIT_ERROR = 3


def _build_params(args) -> dict:
    return {
        "name": args.name,
        "seed": args.seed,
        "num_hosts": args.num_hosts,
        "num_subnets": args.num_subnets if args.num_subnets is not None else max(1, args.num_hosts // 8),
        "server_ratio": args.server_ratio,
        "vuln_density": args.vuln_density,
        "dedicated_server_subnets": args.dedicated_server_subnets,
        "server_types": args.server_types.split(",") if args.server_types else DEFAULT_SERVER_TYPES,
        "size_tier": args.size_tier,
    }


def _parse_args(argv):
    p = argparse.ArgumentParser(prog="cyberwheel.network.network_generation")
    p.add_argument("--name", default="generated_network", help="network name (and default filename)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for deterministic generation")
    p.add_argument("--num-hosts", type=int, default=20, help="total host count")
    p.add_argument("--num-subnets", type=int, default=None, help="number of subnets (segmentation); default ~num_hosts/8")
    p.add_argument("--server-ratio", type=float, default=0.2, help="fraction of hosts that are servers / crown jewels [0,1]")
    p.add_argument("--vuln-density", type=float, default=1.0, help="fraction of non-server hosts that are vulnerable vs hardened [0,1]")
    p.add_argument("--dedicated-server-subnets", dest="dedicated_server_subnets", action="store_true", default=True, help="place servers in their own subnets (default)")
    p.add_argument("--no-dedicated-server-subnets", dest="dedicated_server_subnets", action="store_false", help="mix servers into user subnets")
    p.add_argument("--server-types", default="", help="comma-separated server host types (default all five *_server types)")
    p.add_argument("--size-tier", default="small", choices=["small", "medium", "large"], help="obs/action size tier the network must fit within")
    p.add_argument("--output", default=None, help="output path; default cyberwheel/data/configs/network/<name>.yaml")
    p.add_argument("--stdout", action="store_true", help="print the YAML to stdout instead of writing a file")
    p.add_argument("--layout-only", action="store_true", help="print the canvas layout JSON (no file written)")
    p.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = p.parse_args(argv)
    if not _NAME_RE.fullmatch(args.name):
        # The name doubles as the default filename inside the package config
        # dir — reject separators/traversal outright.
        p.error(f"--name must be letters, digits, '-' or '_', got {args.name!r}")
    return args


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    params = validate_params(_build_params(args))

    if args.layout_only:
        # Build the network in-process (imports the network stack) and emit its
        # deterministic layout for the UI preview. Build chatter is redirected to
        # stderr so stdout carries only the JSON the server parses.
        import contextlib

        from cyberwheel.network.network_generation.parameterized_generator import build_network
        from cyberwheel.visualization.layout import compute_layout

        with contextlib.redirect_stdout(sys.stderr):
            network = build_network(params)
            layout = compute_layout(network)
        json.dump(layout, sys.stdout)
        return EXIT_OK

    data = generate_network_dict(params)

    if args.stdout:
        yaml.safe_dump(data, sys.stdout)
        return EXIT_OK

    if args.output:
        path = args.output
    else:
        path = str(files("cyberwheel.data.configs.network").joinpath(f"{params['name']}.yaml"))
    if os.path.exists(path) and not args.force:
        print(f"error: {path} already exists (use --force to overwrite)", file=sys.stderr)
        return EXIT_ERROR
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    print(f"wrote {params['num_hosts']}-host network to {path}")
    return EXIT_OK
