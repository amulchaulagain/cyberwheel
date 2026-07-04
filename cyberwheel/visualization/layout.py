"""Deterministic network layout without graphviz.

Exploits the strict router -> subnet -> host tree of cyberwheel networks:

* hosts of a subnet sit on a phyllotaxis (sunflower) disc around the subnet
  node, with extra slots reserved so decoys deployed mid-episode land inside
  the disc without moving anything;
* subnet discs are arranged around their router — on a ring while there are
  few of them, on a phyllotaxis disc of discs when there are many (10k-host
  networks have ~100 subnets; a ring would be a thin donut);
* multiple routers are placed on a ring around the canvas centre, each with
  its own subnet arrangement (the shipped configs have a single core router,
  which degenerates to router-at-centre).

Identical topology in, byte-identical layout out: all orderings are sorted,
rotations derive from crc32 of node names (the builtin ``hash`` is salted
per process), and there is no randomness.
"""

from __future__ import annotations

import math
import zlib

from cyberwheel.network.host import Host
from cyberwheel.network.router import Router
from cyberwheel.network.subnet import Subnet

GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

HOST_SPACING = 26.0
CLUSTER_PAD = 40.0
RING_BASE_RADIUS = 160.0
MARGIN = 80.0
# Above this many subnets per router, a ring wastes the canvas interior;
# pack the subnet discs on a phyllotaxis disc instead.
RING_TO_DISC_THRESHOLD = 16


def _rotation(name: str) -> float:
    return (zlib.crc32(name.encode()) % 3600) / 3600.0 * 2.0 * math.pi


def _phyllotaxis(index: int, spacing: float, rotation: float) -> tuple[float, float]:
    radius = spacing * math.sqrt(index + 1)
    theta = index * GOLDEN_ANGLE + rotation
    return radius * math.cos(theta), radius * math.sin(theta)


def cluster_radius(slots: int, spacing: float = HOST_SPACING) -> float:
    """Radius of a phyllotaxis disc holding ``slots`` points."""
    return spacing * math.sqrt(slots + 1) + spacing * 0.6


def _ring_positions(
    sizes: list[float], min_radius: float, pad: float, start_angle: float
) -> list[tuple[float, float]]:
    """Place discs of the given radii on a circle, each at the midpoint of an
    arc proportional to its diameter, so neighbours never overlap."""
    arcs = [2.0 * (r + pad) for r in sizes]
    total = sum(arcs)
    ring_radius = max(min_radius, total / (2.0 * math.pi), max(sizes) + pad)
    positions = []
    cumulative = 0.0
    for arc in arcs:
        angle = start_angle + (cumulative + arc / 2.0) / total * 2.0 * math.pi
        positions.append((ring_radius * math.cos(angle), ring_radius * math.sin(angle)))
        cumulative += arc
    return positions


def _disc_positions(sizes: list[float], pad: float, rotation: float) -> list[tuple[float, float]]:
    spacing = 2.0 * max(sizes) + pad
    return [_phyllotaxis(i, spacing, rotation) for i in range(len(sizes))]


def group_radius(offsets: list[tuple[float, float]], sizes: list[float]) -> float:
    return max(
        math.hypot(x, y) + r for (x, y), r in zip(offsets, sizes)
    )


def compute_layout(network, decoy_reserve: int = 10) -> dict:
    """Compute the static layout for a Network.

    ``decoy_reserve`` extra phyllotaxis slots are budgeted per subnet so
    dynamically deployed decoys stay inside their subnet's disc. Returns a
    JSON-serializable dict: ``bounds``, ``nodes`` (list index = node id),
    ``edges`` (id pairs), and per-subnet ``decoy_slots`` generator params.
    """
    graph = network.graph
    routers: list[str] = []
    subnets: list[str] = []
    hosts_by_subnet: dict[str, list[str]] = {}
    objects: dict[str, object] = {}

    for name, obj in sorted(graph.nodes(data="data")):
        objects[name] = obj
        if isinstance(obj, Router):
            routers.append(name)
        elif isinstance(obj, Subnet):
            subnets.append(name)
            hosts_by_subnet.setdefault(name, [])
    for name, obj in sorted(graph.nodes(data="data")):
        if isinstance(obj, Host):
            hosts_by_subnet.setdefault(obj.subnet.name, []).append(name)

    subnets_by_router: dict[str, list[str]] = {r: [] for r in routers}
    for subnet_name in subnets:
        router_name = objects[subnet_name].router.name
        subnets_by_router.setdefault(router_name, []).append(subnet_name)

    subnet_radii = {
        s: cluster_radius(len(hosts_by_subnet[s]) + decoy_reserve) for s in subnets
    }

    # Per-router local coordinates: router at local origin, subnets around it.
    subnet_centers: dict[str, tuple[float, float]] = {}
    router_group_radius: dict[str, float] = {}
    for router_name in routers:
        members = subnets_by_router[router_name]
        if not members:
            router_group_radius[router_name] = RING_BASE_RADIUS
            continue
        sizes = [subnet_radii[s] for s in members]
        if len(members) > RING_TO_DISC_THRESHOLD:
            offsets = _disc_positions(sizes, CLUSTER_PAD, _rotation(router_name))
        else:
            offsets = _ring_positions(
                sizes, RING_BASE_RADIUS, CLUSTER_PAD, _rotation(router_name)
            )
        for name, offset in zip(members, offsets):
            subnet_centers[name] = offset
        router_group_radius[router_name] = group_radius(offsets, sizes) + CLUSTER_PAD

    # Place router groups globally.
    router_centers: dict[str, tuple[float, float]] = {}
    if len(routers) == 1:
        router_centers[routers[0]] = (0.0, 0.0)
    elif routers:
        sizes = [router_group_radius[r] for r in routers]
        for name, position in zip(
            routers, _ring_positions(sizes, max(sizes), CLUSTER_PAD, -math.pi / 2.0)
        ):
            router_centers[name] = position

    positions: dict[str, tuple[float, float]] = dict(router_centers)
    for subnet_name in subnets:
        rx, ry = router_centers[objects[subnet_name].router.name]
        sx, sy = subnet_centers.get(subnet_name, (0.0, 0.0))
        positions[subnet_name] = (rx + sx, ry + sy)
    for subnet_name in subnets:
        cx, cy = positions[subnet_name]
        rotation = _rotation(subnet_name)
        for index, host_name in enumerate(hosts_by_subnet[subnet_name]):
            hx, hy = _phyllotaxis(index, HOST_SPACING, rotation)
            positions[host_name] = (cx + hx, cy + hy)

    min_x = min(x for x, _ in positions.values())
    min_y = min(y for _, y in positions.values())
    max_x = max(x for x, _ in positions.values())
    max_y = max(y for _, y in positions.values())

    def canvas(name: str) -> tuple[float, float]:
        x, y = positions[name]
        return round(x - min_x + MARGIN, 1), round(y - min_y + MARGIN, 1)

    nodes: list[dict] = []
    node_ids: dict[str, int] = {}
    for name in routers:
        x, y = canvas(name)
        node_ids[name] = len(nodes)
        nodes.append({"name": name, "kind": "router", "x": x, "y": y})
    for name in subnets:
        x, y = canvas(name)
        node_ids[name] = len(nodes)
        nodes.append(
            {
                "name": name,
                "kind": "subnet",
                "x": x,
                "y": y,
                "r": round(subnet_radii[name], 1),
            }
        )
    for subnet_name in subnets:
        for host_name in hosts_by_subnet[subnet_name]:
            host = objects[host_name]
            x, y = canvas(host_name)
            node_ids[host_name] = len(nodes)
            nodes.append(
                {
                    "name": host_name,
                    "kind": "host",
                    "subnet": subnet_name,
                    "type": host.host_type.name if host.host_type else "Unknown",
                    "ip": str(host.ip_address) if host.ip_address else None,
                    "x": x,
                    "y": y,
                }
            )

    edges = sorted(
        [node_ids[a], node_ids[b]]
        for a, b in graph.edges()
        if a in node_ids and b in node_ids
    )

    decoy_slots = {}
    for subnet_name in subnets:
        cx, cy = canvas(subnet_name)
        decoy_slots[subnet_name] = {
            "cx": cx,
            "cy": cy,
            "rot": round(_rotation(subnet_name), 6),
            "spacing": HOST_SPACING,
            "base": len(hosts_by_subnet[subnet_name]),
        }

    return {
        "bounds": {
            "w": round(max_x - min_x + 2 * MARGIN, 1),
            "h": round(max_y - min_y + 2 * MARGIN, 1),
        },
        "nodes": nodes,
        "edges": edges,
        "decoy_slots": decoy_slots,
    }


def decoy_slot_position(slots: dict, slot_index: int) -> tuple[float, float]:
    """Position of the ``slot_index``-th decoy of a subnet, from its
    ``decoy_slots`` generator params. Mirrored by the frontend renderer —
    keep the two implementations in sync."""
    index = slots["base"] + slot_index
    dx, dy = _phyllotaxis(index, slots["spacing"], slots["rot"])
    return round(slots["cx"] + dx, 1), round(slots["cy"] + dy, 1)
