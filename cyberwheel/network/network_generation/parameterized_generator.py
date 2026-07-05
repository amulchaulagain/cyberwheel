"""Knob-driven procedural network generator.

Produces a valid network config (the same YAML schema the loader consumes) from
security-posture knobs — size, segmentation, crown-jewel ratio & placement, and
vulnerability density — deterministically from a seed. The dict-building path is
torch-free (pure yaml/dict via ``NetworkYAMLGenerator``); only ``build_network``
imports the network stack, so the server can build configs without pulling torch.
"""

import random

from cyberwheel.network.network_generation.network_generator import NetworkYAMLGenerator

# Observation/action-space padding caps per size tier (hosts, subnets).
SIZE_TIERS = {"small": (100, 10), "medium": (1000, 100), "large": (10000, 1000)}

# Vulnerable host types (existing palette). Servers carry "server" in the name,
# which is how valid_targets:servers finds crown jewels.
DEFAULT_SERVER_TYPES = [
    "proxy_server",
    "web_server",
    "mail_server",
    "file_server",
    "ssh_jump_server",
]
WORKSTATION_TYPE = "workstation"      # vulnerable user host (full CVE surface)
HARDENED_TYPE = "generated_hardened"  # non-vulnerable user host (empty CVE surface)


def _defaults(params: dict) -> dict:
    p = dict(params)
    p.setdefault("name", "generated_network")
    p.setdefault("seed", 0)
    p.setdefault("num_hosts", 20)
    p.setdefault("num_subnets", max(1, int(p["num_hosts"]) // 8))
    p.setdefault("server_ratio", 0.2)
    p.setdefault("vuln_density", 1.0)
    p.setdefault("dedicated_server_subnets", True)
    p.setdefault("server_types", DEFAULT_SERVER_TYPES)
    p.setdefault("size_tier", "small")
    return p


def validate_params(params: dict) -> dict:
    """Return the defaulted params, raising ValueError on out-of-range knobs."""
    p = _defaults(params)
    tier = p["size_tier"]
    if tier not in SIZE_TIERS:
        raise ValueError(f"size_tier must be one of {sorted(SIZE_TIERS)}, got {tier!r}")
    max_hosts, max_subnets = SIZE_TIERS[tier]
    num_hosts, num_subnets = int(p["num_hosts"]), int(p["num_subnets"])
    if num_hosts < 1:
        raise ValueError("num_hosts must be >= 1")
    if num_subnets < 1:
        raise ValueError("num_subnets must be >= 1")
    if num_hosts > max_hosts:
        raise ValueError(f"num_hosts {num_hosts} exceeds the {tier} tier cap ({max_hosts})")
    if num_subnets > max_subnets:
        raise ValueError(f"num_subnets {num_subnets} exceeds the {tier} tier cap ({max_subnets})")
    if num_subnets > num_hosts:
        raise ValueError("num_subnets cannot exceed num_hosts")
    for key in ("server_ratio", "vuln_density"):
        if not 0.0 <= float(p[key]) <= 1.0:
            raise ValueError(f"{key} must be in [0, 1], got {p[key]}")
    if not p["server_types"]:
        raise ValueError("server_types must be non-empty")
    p["num_hosts"], p["num_subnets"] = num_hosts, num_subnets
    return p


def generate_network_dict(params: dict) -> dict:
    """Build the network config dict deterministically from the given knobs."""
    p = validate_params(params)
    rng = random.Random(p["seed"])
    num_hosts, num_subnets = p["num_hosts"], p["num_subnets"]
    num_servers = round(num_hosts * p["server_ratio"])

    gen = NetworkYAMLGenerator(network_name=p["name"], desc="parameterized generated network")
    gen.router("core_router")

    # Partition subnets into server vs user segments when segmentation is on.
    if p["dedicated_server_subnets"] and num_servers > 0:
        n_server_subnets = max(1, round(num_subnets * (num_servers / num_hosts)))
        n_server_subnets = min(n_server_subnets, num_subnets - 1) if num_subnets > 1 else num_subnets
        n_server_subnets = max(1, n_server_subnets)
    else:
        n_server_subnets = 0

    subnet_names = []
    for i in range(num_subnets):
        is_server_subnet = i < n_server_subnets
        prefix = "server_subnet" if is_server_subnet else "user_subnet"
        name = f"{prefix}{i}"
        gen.subnet(name, router_name="core_router", ip_range=f"10.{i}.0.0/24")
        subnet_names.append((name, is_server_subnet))

    server_subnets = [n for n, s in subnet_names if s]
    user_subnets = [n for n, s in subnet_names if not s]
    if not server_subnets:  # servers mix into user subnets
        server_subnets = [n for n, _ in subnet_names]
    if not user_subnets:
        user_subnets = [n for n, _ in subnet_names]

    # Assign host types: servers (crown jewels), then vulnerable/hardened users.
    for h in range(num_hosts):
        if h < num_servers:
            host_type = p["server_types"][h % len(p["server_types"])]
            subnet = server_subnets[h % len(server_subnets)]
            gen.host(f"server{h}", subnet, host_type)
        else:
            vulnerable = rng.random() < p["vuln_density"]
            host_type = WORKSTATION_TYPE if vulnerable else HARDENED_TYPE
            subnet = user_subnets[h % len(user_subnets)]
            gen.host(f"host{h}", subnet, host_type)

    gen._topology()
    return gen.data


def write_network_yaml(params: dict, path: str) -> dict:
    """Generate and write the config to ``path``; return the config dict."""
    import yaml

    data = generate_network_dict(params)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return data


def build_network(params: dict):
    """Build a Network object from the generated config (imports the network stack)."""
    import os
    import tempfile

    import yaml

    import cyberwheel.utils  # noqa: F401 -- resolves the import-order cycle
    from cyberwheel.network.network_base import Network

    data = generate_network_dict(params)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    try:
        yaml.safe_dump(data, tmp)
        tmp.close()
        return Network.create_network_from_yaml(tmp.name)
    finally:
        os.unlink(tmp.name)
