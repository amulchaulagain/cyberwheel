import gymnasium as gym
import numpy as np
import importlib

from typing import Iterable, Any 
from gymnasium import spaces

from cyberwheel.cyberwheel_envs.cyberwheel import Cyberwheel
from cyberwheel.blue_agents import RLBlueAgent, InactiveBlueAgent, RandomBlueAgent
from cyberwheel.network.network_base import Network
from cyberwheel.red_agents import RLARTAgent, ARTAgent, ARTCampaign
from cyberwheel.red_agents.rl_red_campaign import RLRedCampaign
from cyberwheel.utils import YAMLConfig, HybridSetList
from cyberwheel.utils.set_seed import set_seed

import pandas as pd
import random

import time


class CyberwheelRL(gym.Env, Cyberwheel):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        args: YAMLConfig,
        network: Network = None,
        evaluation: bool = False,
        networks : dict = {},
        rank: int = 0
    ):
        """
        The CyberwheelRL class is used to define the Cyberwheel environment. It allows you to use a YAML
        file to configure the actions, rewards, and logic of the blue agent. Given various configurations, it
        will initiate the environment with the red agent, blue agent, reward functions, and network state.
        Important member variables:

        * `args`: required
            - YAMLConfig instance defining the environment state.

        * `network`: optional
            - The Network object to use throughout the environment. This prevents longer start-up times when training with multiple environments.
            - If not passed, it will build the network with the config file passed.
            - Default: None
        """
        super().__init__(args, network=network)
        self.networks = networks
        reward_function = args.reward_function
        rfm = importlib.import_module("cyberwheel.reward")

        self.reward_calculator = getattr(rfm, reward_function)(
            self.red_agent, 
            self.blue_agent,
            self.args.valid_targets,
            self.network)

        self.evaluation = evaluation
        self.total = 0

        #TODO: TEMP
        #self.action_usage = []
    
    def initialize_agents(self) -> None:
        args = self.args
        max_net = self.args.network_size_compatibility
        args.max_num_hosts = 100 if max_net == 'small' else 1000 if max_net == 'medium' else 10000 # if max_net == 'large'
        #max_num_subnets = max_num_hosts / 10 # 10 if max_net == 'small' else 100 if max_net == 'medium' else 1000 # if max_net == 'large'
        if args.train_red:
            self.red_agent = RLRedCampaign(self.network, args) if args.campaign else RLARTAgent(self.network, args)
            #self.red_agent = RLARTAgent(self.network, args)
            self.blue_agent = RandomBlueAgent(self.network, args)
            self.rl_agent = self.red_agent
            self.static_agent = self.blue_agent

            #self.observation_space = spaces.MultiDiscrete(np.array([6] * self.red_agent.observation.max_size))
            self.observation_space = spaces.Box(
                low  = np.full(self.red_agent.observation.max_size, -1, dtype=np.int32),
                high = np.full(self.red_agent.observation.max_size,  4, dtype=np.int32),
                dtype=np.int32
            )

            self.max_action_space_size = args.max_num_hosts * self.red_agent.action_space.num_actions * 2 # TODO: instead of hard-coding 500, make dependent on new arg (small/med/large networks - 100/1000/10000)
            self.action_space = self.red_agent.action_space.create_action_space(self.max_action_space_size)
            #print(self.action_space)
            #print(self.action_space.shape)
            #time.sleep(1)
            self.reward_sign = -1
        else:
            self.red_agent = ARTCampaign(self.network, args) if args.campaign else ARTAgent(self.network, args)
            self.blue_agent = RLBlueAgent(self.network, args)
            self.rl_agent = self.blue_agent
            self.static_agent = self.red_agent

            obs_shape = self.blue_agent.observation.shape
            #self.observation_space = spaces.MultiDiscrete(np.array([args.max_decoys + 2] * self.blue_agent.observation.shape))
            self.observation_space = spaces.Box(
                low  = np.full(self.blue_agent.observation.shape, -1, dtype=np.int32),
                high = np.full(self.blue_agent.observation.shape, args.max_decoys + 2, dtype=np.int32),
                dtype=np.int32
            )
            # self.observation_space = spaces.MultiBinary(self.blue_agent.observation.shape)
            self.max_action_space_size = self.blue_agent.action_space._action_space_size # TODO: instead of hard coding, just make it preset (small/med/large - 10/100/1000)
            self.action_space = self.blue_agent.create_action_space(self.max_action_space_size)
            
            self.reward_sign = 1

    def step(self, action: int) -> tuple[Iterable, int | float, bool, bool, dict[str, Any]]:
        """
        Steps through environment.
        1. Blue agent runs action
        2. Red agent runs action
        3. Calculate reward based on red/blue actions and network state
        4. Get obs from Red or Blue Observation
        5. Return obs and related metadata
        """
        #print("D-A")
        blue_agent_result = self.blue_agent.act(action)

        red_agent_result = self.red_agent.act(action)

        #print(blue_agent_result.name)
        #print(self.network.all_hosts.data_list)
        
        # TODO: cleanup
        #if action not in self.action_usage:
            #self.action_usage.append(action)
            #print(f"{action} was just used.")
            #print(f"So far, {len(self.action_usage)} actions have been used, {self.blue_agent.action_space._action_space_size} actions should be possible.")

        obs_vec = self.red_agent.get_observation_space() if self.args.train_red else self.blue_agent.get_observation_space(red_agent_result)
        #print("D-B")
        reward = self.reward_sign * self.reward_calculator.calculate_reward(
            red_agent_result.action.get_name(),
            blue_agent_result.name,
            red_agent_result.success,
            blue_agent_result.success,
            red_agent_result.target_host,
            blue_id=blue_agent_result.id,
            blue_recurring=blue_agent_result.recurring
        )
        #print("D-C")

        self.total += reward

        #done = self.current_step >= self.max_steps
        done = self.current_step == self.max_steps - 1

        #if done and self.total == 0:
        #    print("finished at 0 here?")
        #    print(f"{red_agent_result.action.get_name()} on {red_agent_result.target_host}, \nunknowns: {self.red_agent.unknowns.data_list}, \nunimpacted_servers: {self.red_agent.unimpacted_servers.data_list}")

        #print(f"{red_agent_result.action.get_name()} - {red_agent_result.src_host.name} -> {red_agent_result.target_host.name}")
        #if red_agent_result.success:
            #print(f"{red_agent_result.action.get_name()} - {red_agent_result.src_host.name} -> {red_agent_result.target_host.name}")
            #print(self.red_agent.observation.obs.keys())
        self.current_step += 1
        info = {}

        # TODO
        
        #if decoy_attacked:
        #    print("decoy was attacked in CyberwheelRL")
        
        if self.evaluation:
            decoy_attacked = red_agent_result.success and (red_agent_result.target_host.decoy or red_agent_result.src_host.decoy)
            #print(f"Red success: {red_agent_result.success}\nAction: {red_agent_result.action.get_name()}\nTarget: {red_agent_result.target_host.name}\nTarget is Decoy: {red_agent_result.target_host.decoy}\nSource: {red_agent_result.src_host.name}\nSource is Decoy: {red_agent_result.src_host.decoy}\n--------------------------------------")
            #print(f"Blue success: {blue_agent_result.success}\nBlue Action: {blue_agent_result.name}\nDecoy ID: {blue_agent_result.id}\n-------------------")

            #print(f"Red agent view: {list(self.red_agent.history.hosts.keys())}\n-----------------------------------\n--------------------------------\n-----------------------------")
            #if decoy_attacked:
            #    print("DECOY ATTACKED IS TRUE???????????????????????????????????????????????????????????")
            info = {
                "red_action": red_agent_result.action.get_name(),
                "red_action_src": red_agent_result.src_host.name,
                "red_action_dst": red_agent_result.target_host.name,
                "red_action_success": red_agent_result.success,
                "blue_action": blue_agent_result.name,
                "blue_action_id": blue_agent_result.id,
                "blue_action_target": blue_agent_result.target,
                "killchain": self.red_agent.killchain,
                "network": self.network,
                "history": self.red_agent.history,
                "commands": [], #red_agent_result.action_results.metadata.get("commands", []),
                "decoy_attacked": decoy_attacked,
            }
        #print("D-D")

        return obs_vec, reward, done, False, info

    def reset(self, seed=None, options=None) -> tuple[Iterable, dict]:
        if seed is not None:
            set_seed(seed)
        self.current_step = 0

        self.network.reset()
        network = random.choice(list(self.networks.values()))
        #print(f"Random Network: {network.name}")

        self.network = network

        self.red_agent.reset(network, self.args.service_mapping[network.name])
        self.blue_agent.reset(network)
        self.reward_calculator.reset()
        self.total = 0
        if self.args.train_red:
            return self.red_agent.observation.obs_vec, {}
        else:
            return self.blue_agent.observation.obs_vec, {} # TODO
        
    def close(self) -> None:
        pass

    @property
    def rl_agent_action_space_size(self):
        return self.rl_agent.action_space._action_space_size
    
    @property
    def action_mask(self):
        return self.red_agent.action_space.get_action_mask(self.red_agent.current_host.name)