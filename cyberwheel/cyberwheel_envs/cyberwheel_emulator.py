from importlib.resources import files
from gymnasium import spaces
import gymnasium as gym
from typing import Dict, List, Iterable
import yaml
import numpy as np
import importlib
import torch
import time
import os

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
        self.colors = {"blue": '\033[94m', 'red': '\033[91m', 'end': '\033[0m'}
        self.total = 0
        self.current_step = 0

        with open(host_conf_file, "r") as f:
            self.host_defs = yaml.safe_load(f)["host_types"]

        self.service_mapping = args.service_mapping
        self.args = args
        self.max_steps = 30

        valid_targets = [
            h.name
            for h in self.network.get_all_hosts()
            if "server" in h.host_type.name.lower()
        ] + [
            "decoy01",
            "decoy02",
        ]

        self.emulator = EmulatorControl(
            network=network,
            network_config_name=args.network_config,
        )
        print("Initializing Hosts...")

        self.initialize_ip_addresses()

        """
        Enroll emulator host's elastic agents to fleet.
        Will only enroll non-enrolled agents.
        """
        self.emulator.init_hosts()

        print("done")

        self.red_agent = EmulatorRLRedCampaign(self.network, args)

        self.blue_agent = DynamicBlueAgent(self.network, args)

        self.observation_space = spaces.MultiDiscrete(np.array([4] * 201)) #spaces.Box(0, 1, shape=(2 * self.network.size(),)) # TODO
        self.blue_alert_converter = HistoryObservation(
            self.observation_space.shape, host_to_index_mapping(self.network)
        )
        self.blue_max_action_space_size = self.blue_agent.action_space._action_space_size #self.network.get_num_subnets() * 2
        print(self.blue_max_action_space_size)
        self.action_space = self.blue_agent.create_action_space(
            self.blue_max_action_space_size
        )

        self.red_observation_space = spaces.MultiDiscrete(np.array([3] * self.red_agent.max_obs_size))
        #spaces.Box(
        #    0, 2, shape=(len(self.red_agent.get_observation_space()),)
        #) # TODO
        self.red_max_action_space_size = 100 * self.red_agent.action_space.num_actions * 2
        
        #(
        #    self.network.size() * self.red_agent.action_space.num_actions * 2
        #)
        self.red_action_space = self.red_agent.action_space.create_action_space(
            self.red_max_action_space_size
        )

        reward_function = args.reward_function
        rfm = importlib.import_module("cyberwheel.reward")

        self.reward_calculator = getattr(rfm, reward_function)(
            self.red_agent.get_reward_map(),
            self.blue_agent.get_reward_map(),
            valid_targets,
        )

        self.evaluation = args.evaluation
        self.red_action = None
    
    def initialize_ip_addresses(self):
        # get host IP addresses from emulator
        file_path = files("cyberwheel.emulator").joinpath(f"{self.network.name}_host_ips.yaml")
        initialize_ips = True
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                host_ips = yaml.safe_load(file)
            initialize_ips = host_ips == None
            host_ips = {} if initialize_ips else host_ips
        else:
            host_ips = {}

        if set(self.network.get_all_hostnames()) != set(host_ips.keys()):
            #print("YAML does not match Network, reinitializing IPs")
            initialize_ips = True
        
        for h in self.network.get_all_hosts():
            host_name = h.name
            if host_name in host_ips and not initialize_ips: # If host found in config
                emu_host_ip = host_ips[host_name]
            #elif not new_yaml and host_name not in host_ips: # If config doesn't match network
            #    raise Exception("Mismatch between firewheel network and host IP config file!")
            else: # Otherwise add host_name and IP to config
                emu_host_ip = self.emulator.get_ip_address(host_name.replace("_", "-"))
                host_ips[host_name] = emu_host_ip 
            h.set_ip_from_str(emu_host_ip)
            #print(f"Retried and saved emulator ip address for {h.name}.")
        
        if initialize_ips:
            with open(file_path, 'w') as f:
                yaml.dump(host_ips, f, default_flow_style=False)

    def step(self, blue_action):
        """
        Steps through environment.
        1. Blue agent runs action
        2. Red agent runs action
        3. Calculate reward based on red/blue actions and network state
        4. Convert Alerts from Detector into observation space
        5. Return obs and related metadata
        """
        print(f"---------------------------------------------------------------------------------------------------------\nStep {self.current_step}\n")
        # print([h.name for h in self.network.get_all_hosts()])
        blue_action_info = self.blue_agent.action_space.select_action(blue_action)
        blue_action_name = blue_action_info.name
        blue_action_src = (
            # blue_action_info.args[0] if blue_action_name != "nothing" else None
            blue_action_info.args[0].name if "nothing" not in blue_action_name else "nothing"
        )

        print(f"{self.colors['blue']}Running Blue Action: {blue_action_name} on {blue_action_src}...{self.colors['end']}")
        blue_action_result = self.emulator.run_blue_action(
            blue_action_name, blue_action_src, id=self.current_step
        )  # TODO
        print(f"{self.colors['blue']}Blue Action Success{self.colors['end']}") if blue_action_result.success else print(f"{self.colors['blue']}Blue Action Fail{self.colors['end']}")

        blue_action_success = blue_action_result.success

        # TODO: Use the following action metadata to execute the correct command in emulator
        #self.red_agent.handle_network_change()
        print(self.red_agent.observation.keys())
        red_agent_result = self.red_agent.select_action(self.red_action)

        # red_action_result, red_action_type = self.red_agent.run_action(red_agent_result.target_host, red_agent_result.action)
        red_action_name = red_agent_result.action.get_name()
        red_action_src = red_agent_result.src_host
        red_action_dst = red_agent_result.target_host
        #print(f"Validated Success: {red_agent_result.success}")
        print(f"{self.colors['red']}Running Red Action: {red_action_name} from {red_action_src.name} to {red_action_dst.name}...{self.colors['end']}")
        if red_agent_result.success:
            red_action_result = self.emulator.run_red_action(
                red_action_name, red_action_src, red_action_dst, id=self.current_step
            )  # TODO
            red_action_success = red_action_result.attack_success
        else:
            red_action_result = RedActionResults(
                red_agent_result.src_host, red_agent_result.target_host
            )
            red_action_success = False
        print(f"{self.colors['red']}Red Action Success{self.colors['end']}") if red_action_success else print(f"{self.colors['red']}Red Action Fail{self.colors['end']}")

        if (
            blue_action_success
            and blue_action_name == "deploy_decoy"
            and not (
                red_action_name == "Remote System Discovery" and red_action_success
            )
        ):
            ping_decoy = EmulatePing(
                src_host=red_action_src,
                target_host=blue_action_result.host,
                network=self.network,
            )
            cmd = ping_decoy.build_emulator_cmd()
            result = ping_decoy.emulator_execute(cmd)
            self.red_agent.add_host(blue_action_result.host)

        red_action_result.action = red_agent_result.action

        red_obs_vec = self.red_agent.resolve_action(red_action_result)

        #print(
        #    f"\n\nEmulator Red Action: {red_action_name} from {red_action_src.name} -> {red_action_dst.name} - {red_action_success}"
        #)
        decoys_deployed = self.network.num_decoys() # TODO
        blue_obs_vec = self.blue_alert_converter.create_obs_vector(
            self.emulator.get_siem_obs(), decoys_deployed=decoys_deployed
        )  # TODO
        # red_obs_vec = self.red_agent.get_observation_space()
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
                "blue_obs": blue_obs_vec,
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
