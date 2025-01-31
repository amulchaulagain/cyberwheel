from importlib.resources import files
from gym import spaces
import gym
from typing import Dict, List, Iterable
import yaml
import numpy as np
import importlib
import torch

from .cyberwheel import Cyberwheel
from cyberwheel.blue_agents import DynamicBlueAgent, InactiveBlueAgent
from cyberwheel.detectors.alert import Alert
from cyberwheel.network.network_base import Network
from cyberwheel.network.host import Host
from cyberwheel.red_agents import RLARTAgent, ARTAgent, ARTCampaign
from cyberwheel.utils import YAMLConfig
from cyberwheel.observation import HistoryObservation
from cyberwheel.detectors.handler import DetectorHandler
from cyberwheel.emulator.control import EmulatorControl


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
        self.observation_space = spaces.Box(0, 1, shape=(2 * self.network.size(),))
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

        # how to get subnet? I need to eventually remove it from EmulatorController.
        # self.emulator = EmulatorControl(network=network, subnet=?, network_config_name=args.network_config)

    def step(self, action):
        """
        Steps through environment.
        1. Blue agent runs action
        2. Red agent runs action
        3. Calculate reward based on red/blue actions and network state
        4. Convert Alerts from Detector into observation space
        5. Return obs and related metadata
        """

        blue_action_info = self.blue_agent.action_space.select_action(action)
        blue_action_name = blue_action_info.name
        print(blue_action_name)
        print(blue_action_info.args)
        blue_action_src = (
            blue_action_info.args[0] if blue_action_name != "nothing" else None
        )

        print(
            f"running blue action in emulator: {blue_action_name} from {blue_action_src}"
        )
        # blue_action_result = self.emulator.run_blue_action(
        #    blue_action_name, blue_action_src, id=self.current_step  # blue_action_src should be host name
        # )  # TODO

        blue_action_success = True  # blue_action_result.success

        # TODO: Use the following action metadata to execute the correct command in emulator
        red_action, red_action_result = (
            self.red_agent.get_next_action()
        )  # TODO: run act() on ARTCampaign to get next action
        red_action_name = red_action.get_name()
        red_action_src = red_action_result.src_host
        red_action_dst = red_action_result.target_host

        print(
            f"running red action in emulator: {red_action_name} from {red_action_src.name} to {red_action_dst.name}"
        )
        # red_action_result = self.emulator.run_red_action(
        #    red_action_name, red_action_src, red_action_dst, id=self.current_step
        # )  # TODO
        red_action_success = True  # red_action_result.success

        self.red_agent.resolve_action(
            red_action, red_action_result
        )  # TODO: Either pass success or pass emulator observation (this could just be red_agent.act() if it succeeds)

        # obs_vec = self.emulator.get_siem_obs()  # TODO
        obs_vec = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        reward = self.reward_calculator.calculate_reward(
            red_action_name,
            blue_action_name,
            red_action_success,
            blue_action_success,
            red_action_dst,
        )

        self.total += reward

        done = self.current_step >= self.max_steps

        self.current_step += 1

        # TODO: Reset Detector/Obs?

        return (
            obs_vec,
            reward,
            done,
            False,
            {
                "blue_action": blue_action_name,
                "blue_action_src": blue_action_src,
                "red_action": red_action_name,
                "red_action_src": red_action_src,
                "red_action_dst": red_action_dst,
                "blue_action_success": blue_action_success,
                "red_action_success": red_action_success,
            },
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

        return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), {}
