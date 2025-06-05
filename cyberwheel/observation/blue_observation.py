import numpy as np

from typing import Dict, Iterable
from importlib.resources import files

from cyberwheel.detectors.alert import Alert
from cyberwheel.network.host import Host
from cyberwheel.observation.observation import Observation
from cyberwheel.detectors.handler import DetectorHandler

import time

class BlueObservation(Observation):
    def __init__(self, shape: int, mapping: Dict[Host, int], detector_config: str) -> None:
        self.shape = shape
        self.mapping = mapping
        self.obs_vec = np.full(shape, -1)
        self.detector = DetectorHandler(files("cyberwheel.data.configs.detector").joinpath(detector_config))
        for i in range(len(self.mapping)):
            self.obs_vec[i] = 0
        #self.host_length = len(mapping)


    def create_obs_vector(self, alerts: Iterable[Alert]) -> Iterable:
        
        # Refresh the non-history portion of the obs_vec
        obs_length = len(self.mapping)
        barrier = obs_length // 2
        for i in range(barrier):
            self.obs_vec[i] = 0
        for alert in alerts:
            alerted_host = alert.src_host
            #print(f"Host {alerted_host.name} alerted")
            if alerted_host.name not in self.mapping:
                continue
            index = self.mapping[alerted_host.name]
            self.obs_vec[index] = 1
            self.obs_vec[index + barrier] = 1
        #print("OBS VEC:")
        #print(self.obs_vec)
        #print("-----------------------------------------------")
        #time.sleep(1)

        return self.obs_vec

    def reset(self, mapping) -> Iterable:
        self.mapping = mapping
        self.obs_vec = np.full(self.shape, -1, dtype=np.int64)
        for i in range(len(self.mapping)):
            self.obs_vec[i] = 0
        self.detector.reset()
        return self.obs_vec