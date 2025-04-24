from importlib.resources import files
from gymnasium import spaces
import gymnasium as gym
from typing import Dict, List, Iterable
import yaml
import numpy as np
import importlib
import torch
import time

from .cyberwheel import Cyberwheel
from cyberwheel.blue_agents import DynamicBlueAgent, InactiveBlueAgent
from cyberwheel.detectors.alert import Alert
from cyberwheel.network.network_base import Network
from cyberwheel.network.host import Host
from cyberwheel.red_agents import RLARTAgent, ARTAgent, ARTCampaign
from cyberwheel.red_actions.red_base import RedActionResults
from cyberwheel.utils import YAMLConfig
from cyberwheel.observation import HistoryObservation
from cyberwheel.detectors.handler import DetectorHandler
from cyberwheel.emulator.control import EmulatorControl
from cyberwheel.cyberwheel_envs.cyberwheel_rl_red import host_to_index_mapping
from cyberwheel.red_agents import EmulatorRLRedCampaign
from cyberwheel.emulator.actions.red_actions import EmulatePing


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
        self.max_steps = 30

        valid_targets = [h.name for h in self.network.get_all_hosts() if "server" in h.host_type.name.lower()] + [
            "decoy01",
            "decoy02",
        ]

        self.emulator = EmulatorControl(
            network=network,
            network_config_name=args.network_config,
        )

        # get host IP addresses from emulator
        for h in self.network.get_all_hosts():
            print(f"retrieving emulator ip address for {h.name}")
            host_name = h.name.replace("_", "-")
            emu_host_ip = self.emulator.get_ip_address(host_name)
            h.set_ip_from_str(emu_host_ip)

        self.red_agent = EmulatorRLRedCampaign(self.network, args)

        self.blue_agent = DynamicBlueAgent(self.network, args)

        self.observation_space = spaces.Box(0, 1, shape=(2 * self.network.size(),))
        self.blue_alert_converter = HistoryObservation(self.observation_space.shape, host_to_index_mapping(self.network))
        self.blue_max_action_space_size = self.network.get_num_subnets() * 2
        self.action_space = self.blue_agent.create_action_space(self.blue_max_action_space_size)
        
        self.red_observation_space = spaces.Box(0, 2, shape=(len(self.red_agent.get_observation_space()),))
        self.red_max_action_space_size = self.network.size() * self.red_agent.action_space.num_actions * 2
        self.red_action_space = self.red_agent.action_space.create_action_space(self.red_max_action_space_size)

        reward_function = args.reward_function
        rfm = importlib.import_module("cyberwheel.reward")

        self.reward_calculator = getattr(rfm, reward_function)(
            self.red_agent.get_reward_map(),
            self.blue_agent.get_reward_map(),
            valid_targets,
        )

        self.evaluation = args.evaluation
        self.red_action = None


    def step(self, blue_action):
        """
        Steps through environment.
        1. Blue agent runs action
        2. Red agent runs action
        3. Calculate reward based on red/blue actions and network state
        4. Convert Alerts from Detector into observation space
        5. Return obs and related metadata
        """
        #print([h.name for h in self.network.get_all_hosts()])
        blue_action_info = self.blue_agent.action_space.select_action(blue_action)
        blue_action_name = blue_action_info.name

        blue_action_src = blue_action_info.args[0] if blue_action_name != "nothing" else None

        print(f"\n\nEmulator Blue Action: {blue_action_name} on {blue_action_src}")
        blue_action_result = self.emulator.run_blue_action(blue_action_name, blue_action_src, id=self.current_step)  # TODO

        blue_action_success = blue_action_result.success

        # TODO: Use the following action metadata to execute the correct command in emulator
        red_agent_result = self.red_agent.select_action(self.red_action)

        #red_action_result, red_action_type = self.red_agent.run_action(red_agent_result.target_host, red_agent_result.action)
        red_action_name = red_agent_result.action.get_name()
        red_action_src = red_agent_result.src_host
        red_action_dst = red_agent_result.target_host
        print(f"Validated Success: {red_agent_result.success}")
        if red_agent_result.success:
            red_action_result = self.emulator.run_red_action(red_action_name, red_action_src, red_action_dst, id=self.current_step)  # TODO
            red_action_success = red_action_result.attack_success
        else:
            red_action_result = RedActionResults(red_agent_result.src_host, red_agent_result.target_host)
            red_action_success = False
        
        if blue_action_success and blue_action_name == "deploy_decoy" and not (red_action_name == "Remote System Discovery" and red_action_success):
            ping_decoy = EmulatePing(src_host=red_action_src, target_host=red_action_dst, network=self.network)
            cmd = ping_decoy.build_emulator_cmd(str(blue_action_result.host.ip_address))
            result = ping_decoy.emulator_execute(cmd)
            self.red_agent.add_host(blue_action_result.host)
        
        red_action_result.action = red_agent_result.action
        
        red_obs_vec = self.red_agent.resolve_action(red_action_result)

        print(f"\n\nEmulator Red Action: {red_action_name} from {red_action_src.name} -> {red_action_dst.name} - {red_action_success}")

        blue_obs_vec = self.blue_alert_converter.create_obs_vector(
            self.emulator.get_siem_obs()
        )  # TODO
        #red_obs_vec = self.red_agent.get_observation_space()
        # obs_vec = [0] * (2 * len(self.network.get_all_hosts()))

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
            red_obs_vec,
            reward,
            done,
            False,
            {
                "blue_action": blue_action_name,
                "blue_action_src": blue_action_src,
                "red_action": red_action_name,
                "red_action_src": red_action_src.name,
                "red_action_dst": red_action_dst.name,
                "blue_action_success": blue_action_success,
                "red_action_success": red_action_success,
                "red_obs": red_obs_vec,
                "blue_obs": blue_obs_vec
            },
        )

    def _get_obs(
        self, alerts: List[Alert]
    ) -> Iterable:  # TODO: implement function to get obs from emu
        return self.blue_alert_converter.create_obs_vector(alerts)

    def _reset_obs(
        self,
    ) -> Iterable:  # TODO: Implement this function to also tell emu to reset
        return self.blue_alert_converter.reset_obs_vector()

    def reset(self, seed=None, options=None):
        self.total = 0
        self.current_step = 0
        self.network.reset()

        self.red_agent.reset(
            self.red_agent.entry_host,
            network=self.network,
        )

        self.blue_agent.reset()

        self.reward_calculator.reset()

        self.emulator.reset()

        return self._reset_obs(), {}
