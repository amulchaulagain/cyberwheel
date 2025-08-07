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
from cyberwheel.blue_agents import RLBlueAgent, InactiveBlueAgent
from cyberwheel.detectors.alert import Alert
from cyberwheel.network.network_base import Network
from cyberwheel.network.host import Host
from cyberwheel.red_agents import RLARTAgent, ARTAgent, ARTCampaign
from cyberwheel.red_actions.red_base import RedActionResults
from cyberwheel.utils import YAMLConfig
from cyberwheel.observation.blue_observation import BlueObservation
from cyberwheel.detectors.handler import DetectorHandler
from cyberwheel.emulator.control import EmulatorControl
from cyberwheel.red_agents import EmulatorRLRedCampaign
from cyberwheel.emulator.actions.red_actions import EmulatePing
from cyberwheel.utils.set_seed import set_seed


class CyberwheelEmulator(gym.Env, Cyberwheel):
    metadata = {"render.modes": ["human"]}

    def __init__(self, args: YAMLConfig, network: Network = None, networks: dict = {}):
        super().__init__(args, network=network)
        self.colors = {"blue": '\033[94m', 'red': '\033[91m', 'end': '\033[0m'}
        self.networks = networks
        self.evaluation = args.evaluation

        reward_function = args.reward_function
        rfm = importlib.import_module("cyberwheel.reward")
        self.reward_calculator = getattr(rfm, reward_function)(
            args,
            red_agent=self.red_agent, 
            blue_agent=self.blue_agent,
            valid_targets=self.args.valid_targets,
            network=self.network)

        self.emulator = EmulatorControl(
            network=network,
            network_config_name=args.network_config,
        )

        self.initialize_network()
        self.initialize_agents()


    def initialize_network(self):
        print("Initializing Hosts...")

        # get host IP addresses from emulator
        file_path = files("cyberwheel.emulator.configs").joinpath(f"{self.network.name}_host_ips.yaml")
        initialize_ips = True
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                host_ips = yaml.safe_load(file)
            initialize_ips = host_ips == None
            host_ips = {} if initialize_ips else host_ips
        else:
            host_ips = {}

        if self.network.hosts.keys() != host_ips.keys():
            #print("YAML does not match Network, reinitializing IPs")
            initialize_ips = True
        
        for host_name, h in self.network.hosts.items():
            if host_name in host_ips and not initialize_ips: # If host found in config
                emu_host_ip = host_ips[host_name]
            else: # Otherwise add host_name and IP to config
                emu_host_ip = self.emulator.get_ip_address(host_name.replace("_", "-"))
                host_ips[host_name] = emu_host_ip 
            h.set_ip_from_str(emu_host_ip)
            #print(f"Retried and saved emulator ip address for {h.name}.")
        
        if initialize_ips:
            #print(file_path)
            with open(file_path, 'w') as f:
                yaml.dump(host_ips, f, default_flow_style=False)

        self.emulator.init_hosts()

        print("done")


    def initialize_agents(self):
        max_net = self.args.network_size_compatibility
        self.args.max_num_hosts = 100 if max_net == 'small' else 1000 if max_net == 'medium' else 10000 # if max_net == 'large'

        self.blue_agent = RLBlueAgent(self.network, self.args)
        self.red_agent = EmulatorRLRedCampaign(self.network, self.args)

        self.blue_max_action_space_size = self.blue_agent.action_space._action_space_size
        self.red_max_action_space_size = self.args.max_num_hosts * self.red_agent.action_space.num_actions * 2

        self.max_blue_attr_value = self.args.max_decoys + 2 # Max obs attribute is limited to when num_decoys_deployed exceeds max_decoys allowed
        self.max_red_attr_value = 4 # Max obs attribute is limited to the 'quadrant' attribute, which goes up to 4.

        self.observation_space = spaces.Dict({
            "blue": spaces.Box(
                low  = np.full(self.blue_agent.observation.max_size, -1, dtype=np.int32),
                high = np.full(self.blue_agent.observation.max_size, self.max_blue_attr_value, dtype=np.int32),
                dtype=np.int32
            ),
            "red": spaces.Box(
                low  = np.full(self.red_agent.observation.max_size, -1, dtype=np.int32),
                high = np.full(self.red_agent.observation.max_size,  self.max_red_attr_value, dtype=np.int32),
                dtype=np.int32
            )
        })

        self.action_space = spaces.Dict({
            "blue": self.blue_agent.create_action_space(self.blue_max_action_space_size),
            "red": self.red_agent.action_space.create_action_space(self.red_max_action_space_size)
        })

    def step(self, action):
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
        blue_action_info = self.blue_agent.action_space.select_action(action["blue"])
        blue_action_name = blue_action_info.name
        blue_action_src = (
            # blue_action_info.args[0] if blue_action_name != "nothing" else None
            blue_action_info.args[0].name if "nothing" not in blue_action_name else "nothing"
        )

        print(f"{self.colors['blue']}Running Blue Action: {blue_action_name} on {blue_action_src}...{self.colors['end']}")
        blue_agent_result = self.emulator.run_blue_action(
            blue_action_name, blue_action_src, id=self.current_step
        )  # TODO
        print(f"{self.colors['blue']}Blue Action Success{self.colors['end']}") if blue_agent_result.success else print(f"{self.colors['blue']}Blue Action Fail{self.colors['end']}")

        blue_action_success = blue_agent_result.success

        # TODO: Use the following action metadata to execute the correct command in emulator
        #self.red_agent.handle_network_change()
        print(self.red_agent.observation.obs.keys())
        red_agent_result = self.red_agent.select_action(action["red"])

        # red_action_result, red_action_type = self.red_agent.run_action(red_agent_result.target_host, red_agent_result.action)
        red_action_name = red_agent_result.action.get_name()
        red_action_src = red_agent_result.src_host
        red_action_dst = red_agent_result.target_host
        #print(f"Validated Success: {red_agent_result.success}")
        print(f"{self.colors['red']}Running Red Action: {red_action_name} from {red_action_src.name} to {red_action_dst.name}...{self.colors['end']}")
        if red_action_name == "nothing":
            red_action_result = RedActionResults(red_action_src, red_action_dst)
            red_action_result.attack_success = True
            red_action_success = True
        elif red_agent_result.success:
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
                target_host=blue_agent_result.target,
                network=self.network,
            )
            cmd = ping_decoy.build_emulator_cmd()
            result = ping_decoy.emulator_execute(cmd)
            self.red_agent.add_host(blue_agent_result.target)

        red_action_result.action = red_agent_result.action

        red_obs_vec = self.red_agent.resolve_action(red_action_result)

        #print(
        #    f"\n\nEmulator Red Action: {red_action_name} from {red_action_src.name} -> {red_action_dst.name} - {red_action_success}"
        #)
        decoys_deployed = len(self.network.decoys) # TODO
        blue_obs_vec = self.blue_agent.observation.create_obs_vector(
            self.emulator.get_siem_obs(), decoys_deployed=decoys_deployed
        )  # TODO
        # red_obs_vec = self.red_agent.get_observation_space()
        # obs_vec = [0] * (2 * len(self.network.get_all_hosts()))

        blue_reward, red_reward = self.reward_calculator.calculate_reward(
            blue_agent_result=blue_agent_result,
            red_agent_result=red_agent_result
        )

        done = self.current_step >= self.max_steps

        self.current_step += 1

        # TODO: Reset Detector/Obs?

        return (
            {
                "blue": blue_obs_vec,
                "red": red_obs_vec
            },
            blue_reward + red_reward,
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
                "red_reward": red_reward,
                "blue_reward": blue_reward
            },
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            set_seed(seed)
        self.current_step = 0
        self.network.reset()

        self.red_agent.reset(network=self.network, service_mapping=self.args.service_mapping[self.network.name])

        self.blue_agent.reset(self.network)

        self.reward_calculator.reset()

        self.emulator.reset()

        return {"blue": self.blue_agent.observation.obs_vec, "red": self.red_agent.observation.obs_vec}, {}

    @property
    def red_action_mask(self):
        return self.red_agent.action_space.get_action_mask(self.red_agent.current_host.name)

    @property
    def blue_action_mask(self):
        return self.blue_agent.action_space.get_action_mask()