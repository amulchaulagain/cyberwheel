import random

from cyberwheel.detectors.alert import Alert
from cyberwheel.green_agents.green_agent_base import GreenAgent, GreenAgentResult
from cyberwheel.network.network_base import Network


class _Session:
    """One benign work session: a source host repeatedly doing one activity
    against one destination for a bounded number of steps."""

    __slots__ = ("src", "dst", "service", "technique", "steps_left")

    def __init__(self, src, dst, service, technique, steps_left):
        self.src = src
        self.dst = dst
        self.service = service
        self.technique = technique
        self.steps_left = steps_left


class ScriptedGreenAgent(GreenAgent):
    """Stochastic benign-user agent driven by multi-step sessions.

    Each step, new sessions start at a rate scaled by network size; every
    active session emits one benign event: an ``Alert`` from the session's
    source host to its destination, tagged with the activity's benign
    technique string. Sessions are deliberately bursty (multi-step, same
    host): i.i.d. per-step noise is a constant bias an RL blue agent learns
    to ignore, while sustained per-host activity resembles the repeated
    killchain steps that correlation detectors key on.

    Events whose source or destination host is isolated are blocked (counted,
    not emitted) — that count is the availability signal for a
    disruption-aware reward. Sessions never target decoys unless
    ``decoy_touch_probability`` rolls at session start; source and destination
    pools (``network.user_hosts`` / ``network.server_hosts``) exclude decoys
    by construction.

    All draws come straight off the global ``random`` stream (seeded by
    ``set_seed``); ``HybridSetList.get_random`` is avoided because it reseeds
    the global RNG in deterministic mode.
    """

    def __init__(self, network: Network, args) -> None:
        conf = args.agent_config["green"]
        self.rate_per_100 = float(conf.get("session_start_rate_per_100_hosts", 4.0))
        length = conf.get("session_length") or [2, 6]
        self.len_lo, self.len_hi = int(length[0]), int(length[1])
        self.max_sessions = int(conf.get("max_concurrent_sessions", 50))
        self.decoy_touch_probability = float(conf.get("decoy_touch_probability", 0.0))
        activities = conf.get("activities") or {
            "generic": {"weight": 1.0, "technique": "benign_generic"}
        }
        self._techniques = [
            str(spec.get("technique", f"benign_{name}"))
            for name, spec in activities.items()
        ]
        self._weights = [float(spec.get("weight", 1.0)) for spec in activities.values()]
        self.reset(network)

    def reset(self, network: Network) -> None:
        self.network = network
        self.sessions: list[_Session] = []

    def _pick_host(self, pool):
        """Uniform draw from a HybridSetList of real (non-decoy) host names."""
        if not pool.data_list:
            return None
        return self.network.hosts[random.choice(pool.data_list)]

    def _start_sessions(self) -> None:
        net = self.network
        expected = self.rate_per_100 * len(net.hosts) / 100.0
        count = int(expected)
        fraction = expected - count
        if fraction > 0.0 and random.random() < fraction:
            count += 1
        for _ in range(count):
            if len(self.sessions) >= self.max_sessions:
                break
            src = self._pick_host(net.user_hosts) or self._pick_host(net.server_hosts)
            if src is None:
                break
            dst = None
            if (
                self.decoy_touch_probability > 0.0
                and net.decoys
                and random.random() < self.decoy_touch_probability
            ):
                dst = net.decoys[random.choice(list(net.decoys))]
            if dst is None:
                dst = self._pick_host(net.server_hosts) or self._pick_host(net.user_hosts)
            if dst is None:
                break
            service = random.choice(dst.services) if dst.services else None
            technique = random.choices(self._techniques, weights=self._weights)[0]
            steps = random.randint(self.len_lo, self.len_hi)
            self.sessions.append(_Session(src, dst, service, technique, steps))

    def act(self) -> GreenAgentResult:
        self._start_sessions()
        alerts: list[Alert] = []
        emitted = blocked = decoy_touches = 0
        for session in self.sessions:
            session.steps_left -= 1
            if session.src.isolated or session.dst.isolated:
                blocked += 1
                continue
            alerts.append(
                Alert(
                    src_host=session.src,
                    techniques=[session.technique],
                    dst_hosts=[session.dst],
                    services=[session.service] if session.service else [],
                )
            )
            emitted += 1
            if session.dst.decoy:
                decoy_touches += 1
        self.sessions = [s for s in self.sessions if s.steps_left > 0]
        return GreenAgentResult(alerts, emitted, blocked, decoy_touches)
