from importlib.resources import files
from gym import spaces
import gym
from typing import Dict, List, Iterable
import yaml
import numpy as np
import importlib

from .cyberwheel import Cyberwheel
from cyberwheel.blue_agents import DynamicBlueAgent, InactiveBlueAgent
from cyberwheel.detectors.alert import Alert
from cyberwheel.network.network_base import Network
from cyberwheel.network.host import Host
from cyberwheel.red_agents import RLARTAgent, ARTAgent, ARTCampaign
from cyberwheel.utils import YAMLConfig
from cyberwheel.observation import HistoryObservation
from cyberwheel.detectors.handler import DetectorHandler


class CyberwheelEmulator(gym.Env, Cyberwheel):
    metadata = {"render.modes": ["human"]}

    def __init__(self, args: YAMLConfig, network: Network = None):
        network_conf_file = files("cyberwheel.resources.configs.network").joinpath(
            args.network_config
        )
        host_conf_file = files(
            "cyberwheel.resources.configs.host_definitions"
        ).joinpath(args.host_config)
        super().__init__(config_file_path=network_conf_file, network=network)
        self.total = 0
        self.current_step = 0

        with open(host_conf_file, "r") as f:
            self.host_defs = yaml.safe_load(f)["host_types"]

        self.service_mapping = args.service_mapping
        self.args = args
        self.max_steps = 10

        valid_targets = [h.name for h in self.network.get_all_hosts()]

        self.red_agent = ARTCampaign(self.network, args)

        self.blue_agent = DynamicBlueAgent(self.network, args)
        num_subnets = (
            self.network.get_num_subnets()
        )  # TODO: Do we need a less sim-y way to get this value?
        self.max_action_space_size = num_subnets * 2

        self.action_space = self.blue_agent.create_action_space(
            self.max_action_space_size
        )

        reward_function = args.reward_function
        rfm = importlib.import_module("cyberwheel.reward")

        self.reward_calculator = getattr(rfm, reward_function)(
            self.red_agent.get_reward_map(),
            self.blue_agent.get_reward_map(),
            valid_targets,
        )

        self.evaluation = args.evaluation

    def step(self, action):
        """
        Steps through environment.
        1. Blue agent runs action
        2. Red agent runs action
        3. Calculate reward based on red/blue actions and network state
        4. Convert Alerts from Detector into observation space
        5. Return obs and related metadata
        """

        action = 2  # TODO: Get action type -> dest subnet from int
        src_host = ""
        dst_host = ""
        action_name = ""

        blue_agent_result = self.blue_agent.act(
            action
        )  # TODO: Call Emulator Defender to take this action

        blue_action_name = (
            blue_agent_result.name
        )  # TODO: Get name by translating the int to
        blue_action_success = (
            blue_agent_result.success
        )  # TODO: Get action success from emulator

        # TODO: Use the following action metadata to execute the correct command in emulator
        red_action_name = self.red_agent.act().get_name()
        action_metadata = self.red_agent.history.history[-1]
        red_action_src = action_metadata["src_host"]
        red_action_dst = action_metadata["target_host"]
        red_action_success = action_metadata[
            "success"
        ]  # TODO: Get success from emulator

        obs_vec = self._get_obs(alerts)  # TODO: Get obs from emulator

        reward = self.reward_calculator.calculate_reward(
            red_action_name,
            blue_action_name,
            red_action_success,
            blue_action_success,
            self.network.get_node_from_name(red_action_dst),
        )  # TODO: Should be able to calculate reward from the values we have

        self.total += reward

        done = self.current_step >= self.max_steps

        self.current_step += 1

        # TODO: Reset Detector/Obs?

        return (
            obs_vec,
            reward,
            done,
            False,
            {},
        )

    def _get_obs(
        self, alerts: List[Alert]
    ) -> Iterable:  # TODO: implement function to get obs from emu
        return self.alert_converter.create_obs_vector(alerts)

    def _reset_obs(
        self,
    ) -> Iterable:  # TODO: Implement this function to also tell emu to reset
        return self.alert_converter.reset_obs_vector()

    def reset(self, seed=None, options=None):
        self.total = 0
        self.current_step = 0
        self.network.reset()

        self.red_agent.reset(
            self.network.get_random_user_host(),
            network=self.network,
        )

        self.blue_agent.reset()

        self.reward_calculator.reset()

        return self._reset_obs(), {}
