"""Green (benign-user) agents.

A green agent generates benign background activity whose events flow through
the same ``Alert`` -> detector pipeline as red actions. Detectors decide
whether a benign event surfaces to the blue agent (a false positive), so
false-positive realism lives in detector config, not here. Green ground truth
(every event is knowably benign) is what makes availability costs and blue
precision measurable.
"""

from abc import ABC, abstractmethod
from typing import Iterable

from cyberwheel.detectors.alert import Alert
from cyberwheel.network.network_base import Network


class GreenAgentResult:
    """Outcome of one green step.

    - ``alerts``: benign Alerts to merge into this step's detector stream.
    - ``events_emitted``: benign events that actually happened this step.
    - ``events_blocked``: benign events prevented because the source or
      destination host is isolated/quarantined — the availability signal a
      disruption-aware blue reward can consume.
    - ``decoy_touches``: emitted events whose destination is a decoy host.
    """

    __slots__ = ("alerts", "events_emitted", "events_blocked", "decoy_touches")

    def __init__(
        self,
        alerts: Iterable[Alert] = (),
        events_emitted: int = 0,
        events_blocked: int = 0,
        decoy_touches: int = 0,
    ) -> None:
        self.alerts = alerts
        self.events_emitted = events_emitted
        self.events_blocked = events_blocked
        self.decoy_touches = decoy_touches


class GreenAgent(ABC):
    """Base class for green agents.

    Green agents are scripted/stochastic (``rl: false``), act autonomously
    once per env step, and must be RNG-neutral when configured off: an env
    without an ``agents: green:`` key must consume zero RNG draws for green.
    """

    @abstractmethod
    def act(self) -> GreenAgentResult:
        raise NotImplementedError

    @abstractmethod
    def reset(self, network: Network) -> None:
        raise NotImplementedError
