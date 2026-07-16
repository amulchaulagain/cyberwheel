import numpy as np

from collections import deque
from typing import Iterable

from cyberwheel.detectors.alert import Alert
from cyberwheel.detectors.handler import DetectorHandler
from cyberwheel.observation.blue_observation import BlueObservation


class WindowedBlueObservation(BlueObservation):
    """Blue observation with sliding-window per-host alert counts.

    Same vector size and layout as `BlueObservation` (current-step alert bits,
    then one slot per host, then standalone attributes), but the second half
    holds the number of alerts each host raised over the last `window` steps
    (clamped to `count_cap`) instead of a sticky ever-alerted bit. Counts let
    the blue agent separate a host that alerted once long ago from one that
    keeps alerting — the distinction that matters once benign green noise
    shares the detector stream with red activity.

    Opt-in via the blue agent YAML:

        observation:
          class: WindowedBlueObservation
          args:
            window: 10     # steps a hit stays visible
            count_cap: 10  # optional clamp; defaults to window
    """

    def __init__(self, args, network, detector: DetectorHandler,
                 window: int = 10, count_cap: int | None = None) -> None:
        self.window = int(window)
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.count_cap = self.window if count_cap is None else int(count_cap)
        if self.count_cap < 1:
            raise ValueError(f"count_cap must be >= 1, got {count_cap}")
        # The env raises the observation-space high bound to this when it
        # exceeds the legacy num_decoys-based ceiling.
        self.max_obs_value = self.count_cap
        super().__init__(args, network, detector)

    def _init_vars(self):
        super()._init_vars()
        self._step_hits: deque[list[int]] = deque()
        self._counts = np.zeros(len(self.mapping), dtype=np.int64)

    def create_obs_vector(self, alerts: Iterable[Alert], **kwargs) -> Iterable:
        barrier = self.len_alerts // 2

        # Refresh the current-step portion of the obs_vec
        self.obs_vec[:barrier] = 0

        hits = []
        for alert in alerts:
            alerted_host = alert.src_host
            if not alerted_host or alerted_host.name not in self.mapping:
                continue
            index = self.mapping[alerted_host.name]
            self.obs_vec[index] = 1
            hits.append(index)

        # Slide the window: admit this step's hits, expire the oldest step's.
        self._step_hits.append(hits)
        for index in hits:
            self._counts[index] += 1
        if len(self._step_hits) > self.window:
            for index in self._step_hits.popleft():
                self._counts[index] -= 1
        self.obs_vec[barrier : barrier + len(self._counts)] = np.minimum(
            self._counts, self.count_cap
        )

        self._write_standalone_attrs(kwargs)
        return self.obs_vec
