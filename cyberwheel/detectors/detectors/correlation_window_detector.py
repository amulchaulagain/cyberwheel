from typing import Iterable

from cyberwheel.detectors.alert import Alert
from cyberwheel.detectors.detector_base import Detector


class CorrelationWindowDetector(Detector):
    """SIEM-style correlation detector.

    A single alert is treated as noise. This detector tracks, per source host,
    how many alerts arrived within a sliding window of the most recent
    ``window`` steps, and only emits an alert once that count reaches
    ``threshold`` — modeling an analyst rule that escalates only on sustained
    activity from the same host.

    Stateful (the first stateful detector): it keeps a per-host history of the
    steps it fired on. That state is cleared each episode via
    ``DetectorHandler.reset_detectors()`` (``reset`` below), not per step.
    """

    name = "CorrelationWindowDetector"

    def __init__(self, config) -> None:
        config = config or {}
        self.window = int(config.get("window", 5))
        self.threshold = int(config.get("threshold", 3))
        self._step = 0
        self._events: dict[str, list[int]] = {}

    def obs(self, perfect_alerts: Iterable[Alert]) -> Iterable[Alert]:
        self._step += 1
        emitted = []
        for alert in perfect_alerts:
            host = getattr(alert, "src_host", None)
            if host is None:
                # Not correlatable (e.g. a no-op step); pass through unchanged.
                emitted.append(alert)
                continue
            key = host.name
            steps = self._events.setdefault(key, [])
            steps.append(self._step)
            # Drop occurrences that fell out of the sliding window.
            cutoff = self._step - self.window + 1
            steps[:] = [s for s in steps if s >= cutoff]
            if len(steps) >= self.threshold:
                emitted.append(alert)
        return emitted

    def reset(self) -> None:
        self._step = 0
        self._events = {}
