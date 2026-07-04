"""Per-step JSON visualization artifacts for evaluations.

``VizWriter`` observes the live env after each evaluation step and writes,
per evaluation run directory:

* ``layout.json``  — static topology + positions (see ``layout.py``)
* ``meta.json``    — run parameters + which episodes have been flushed
* ``episode_<n>.json`` — per-step state deltas for one episode

Frames carry deltas only (the red agent's knowledge grows monotonically and
blue touches a handful of nodes per step), so files stay small even on
10k-host networks. Episode files are flushed atomically at episode end so a
frontend can replay finished episodes while later ones are still running.

Everything resets between episodes (decoys removed, isolation reconnected),
so frames never carry state across episode files; dynamically added decoys
get node ids past the static layout array, scoped to their episode.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from cyberwheel.visualization.layout import compute_layout
from cyberwheel.visualization.states import known_host_state

FORMAT_VERSION = 1


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


class VizWriter:
    def __init__(self, env, out_dir, meta: dict | None = None):
        self.env = env
        self.out_dir = Path(str(out_dir))
        self.out_dir.mkdir(parents=True, exist_ok=True)

        max_decoys = int(getattr(env.args, "max_decoys", 5) or 5)
        self.layout = compute_layout(env.network, decoy_reserve=max(8, 2 * max_decoys))
        self._static_ids = {n["name"]: i for i, n in enumerate(self.layout["nodes"])}
        _write_json(self.out_dir / "layout.json", self.layout)

        network = env.network
        self.meta = {
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "counts": {
                "hosts": len(network.hosts),
                "subnets": len(network.subnets),
                "nodes": len(self.layout["nodes"]),
            },
            "episodes_written": [],
            **(meta or {}),
        }
        _write_json(self.out_dir / "meta.json", self.meta)

        self._episode: int | None = None
        self._frames: list[dict] = []

    # -- per-episode lifecycle -------------------------------------------

    def start_episode(self, episode: int) -> None:
        """Call right after ``env.reset()``."""
        self._episode = episode
        self._frames = []
        self._ids = dict(self._static_ids)
        self._known_states: dict[int, int] = {}
        self._flags: dict[int, tuple[bool, bool, bool]] = {}
        self._edges_off: set[tuple[int, int]] = set()
        self._decoys: set[str] = set()
        self._slot_counters: dict[str, int] = {}
        self._red_position = self._node_id(self._red_host_name())
        self._totals = {"reward": 0.0, "blue": 0.0, "red": 0.0, "decoys_deployed": 0, "decoy_attacks": 0}
        self._initial = {"red_position": self._red_position}

    def record_step(self, episode: int, step: int, info: dict) -> None:
        """Call right after ``env.step()`` with its ``info`` dict."""
        if episode != self._episode:
            self.start_episode(episode)
        frame: dict = {"s": step}

        added, removed = self._diff_decoys()
        if added:
            frame["add"] = added
        if removed:
            frame["rm"] = removed

        edges_off, edges_on = self._diff_isolation()
        if edges_off:
            frame["eoff"] = edges_off
        if edges_on:
            frame["eon"] = edges_on

        known = self._diff_known_states()
        if known:
            frame["kh"] = known

        flags = self._diff_host_flags()
        if flags:
            frame["flags"] = flags

        position = self._node_id(self._red_host_name())
        if position != self._red_position:
            self._red_position = position
            frame["pos"] = position

        red_reward = float(info.get("red_reward", 0.0))
        blue_reward = float(info.get("blue_reward", 0.0))
        frame["red"] = {
            "a": info.get("red_action"),
            "src": self._node_id(info.get("red_action_src")),
            "dst": self._node_id(info.get("red_action_dst")),
            "ok": bool(info.get("red_action_success")),
            "r": round(red_reward, 4),
        }
        frame["blue"] = {
            "a": info.get("blue_action"),
            "tgt": info.get("blue_action_target"),
            "ok": bool(info.get("blue_action_success")),
            "r": round(blue_reward, 4),
        }
        if info.get("decoy_attacked"):
            frame["da"] = True
            self._totals["decoy_attacks"] += 1
        self._totals["reward"] += red_reward + blue_reward
        self._totals["blue"] += blue_reward
        self._totals["red"] += red_reward

        self._frames.append(frame)

    def end_episode(self) -> None:
        """Flush the current episode file and update ``meta.json``."""
        if self._episode is None:
            return
        totals = dict(self._totals)
        for key in ("reward", "blue", "red"):
            totals[key] = round(totals[key], 4)
        _write_json(
            self.out_dir / f"episode_{self._episode}.json",
            {
                "episode": self._episode,
                "initial": self._initial,
                "steps": self._frames,
                "totals": totals,
            },
        )
        if self._episode not in self.meta["episodes_written"]:
            self.meta["episodes_written"].append(self._episode)
        _write_json(self.out_dir / "meta.json", self.meta)
        self._episode = None
        self._frames = []

    # -- state diffing ----------------------------------------------------

    def _red_host_name(self) -> str | None:
        host = getattr(self.env.red_agent, "current_host", None)
        return host.name if host is not None else None

    def _node_id(self, name) -> int:
        if not isinstance(name, str):
            return -1
        return self._ids.get(name, -1)

    def _diff_decoys(self) -> tuple[list[dict], list[int]]:
        current = self.env.network.decoys
        added = []
        for name, host in current.items():
            if name in self._decoys:
                continue
            subnet = host.subnet.name
            slot = self._slot_counters.get(subnet, 0)
            self._slot_counters[subnet] = slot + 1
            node_id = len(self._ids)
            self._ids[name] = node_id
            self._decoys.add(name)
            self._totals["decoys_deployed"] += 1
            added.append(
                {
                    "id": node_id,
                    "name": name,
                    "subnet": subnet,
                    "type": host.host_type.name if host.host_type else "Unknown",
                    "slot": slot,
                }
            )
        removed = [
            self._ids[name] for name in sorted(self._decoys - set(current))
        ]
        for name in list(self._decoys):
            if name not in current:
                self._decoys.discard(name)
        return added, removed

    def _diff_isolation(self) -> tuple[list[list[int]], list[list[int]]]:
        current = set()
        for a, b in self.env.network.disconnected_nodes:
            ia, ib = self._node_id(a), self._node_id(b)
            if ia >= 0 and ib >= 0:
                current.add((ia, ib))
        edges_off = sorted(current - self._edges_off)
        edges_on = sorted(self._edges_off - current)
        self._edges_off = current
        return [list(e) for e in edges_off], [list(e) for e in edges_on]

    def _diff_known_states(self) -> dict[str, int]:
        changed: dict[str, int] = {}
        history = getattr(self.env.red_agent, "history", None)
        hosts = getattr(history, "hosts", None) if history is not None else None
        if not hosts:
            return changed
        for name, known in hosts.items():
            node_id = self._node_id(name)
            if node_id < 0:
                continue
            state = known_host_state(known)
            # Missing cache entry == SAFE: the client baseline is all-safe,
            # so a host entering history still-safe is not a visible change.
            if self._known_states.get(node_id, 0) != state:
                self._known_states[node_id] = state
                changed[str(node_id)] = state
        return changed

    def _diff_host_flags(self) -> dict[str, dict]:
        """Ground-truth flags for hosts the episode has touched (red history
        plus live decoys) — other hosts' flags cannot have changed."""
        changed: dict[str, dict] = {}
        names: set[str] = set(self._decoys)
        history = getattr(self.env.red_agent, "history", None)
        if history is not None and getattr(history, "hosts", None):
            names.update(history.hosts)
        network_hosts = self.env.network.hosts
        for name in names:
            host = network_hosts.get(name)
            node_id = self._node_id(name)
            if host is None or node_id < 0:
                continue
            flags = (bool(host.is_compromised), bool(host.isolated), bool(host.restored))
            # Missing cache entry == all-false (the client baseline).
            if self._flags.get(node_id, (False, False, False)) != flags:
                self._flags[node_id] = flags
                changed[str(node_id)] = {"c": flags[0], "i": flags[1], "r": flags[2]}
        return changed
