import torch
import random
import gymnasium as gym
import time
import os
import importlib
import wandb
import numpy as np
import pandas as pd

from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch import optim, nn
from importlib.resources import files
from tqdm import tqdm

from cyberwheel.utils import RLAgent, get_service_map
from cyberwheel.network.network_base import Network
from cyberwheel.red_actions.actions import (
    ARTDiscovery,
    ARTLateralMovement,
    ARTPrivilegeEscalation,
    ARTImpact,
)
from cyberwheel.red_actions import art_techniques
from cyberwheel.red_agents import ARTAgent


def get_action_mask(action_space_size, action_masks):
    for i in range(len(action_masks)):
        if i < action_space_size:
            action_masks[i] = True
        else:
            action_masks[i] = False
    return action_masks


class EmulatorEvaluator:
    def __init__(self, args):
        self.args = args
        m = importlib.import_module("cyberwheel.cyberwheel_envs")
        self.env_class = getattr(m, args.environment)
        self.deterministic = os.getenv("CYBERWHEEL_DETERMINISTIC", "False").lower() in ('true', '1', 't')
        self.args.deterministic = self.deterministic
        self.seed = args.seed

    def make_env(self, rank):
        """
        Utility function for multiprocessed env.

        :param env_id: the environment ID
        :param num_env: the number of environments you wish to have in subprocesses
        :param seed: the inital seed for RNG
        :param rank: index of the subprocess
        """

        def _init():
            config_path = files("cyberwheel.data.configs.network").joinpath(
                self.args.network_config
            )
            net = Network.create_network_from_yaml(config_path)
            env = self.env_class(
                self.args, network=net, networks={net.name: net}
            )

            self.red_max_action_space_size = env.red_max_action_space_size
            self.blue_max_action_space_size = env.blue_max_action_space_size
            env.reset(
                seed=self.args.seed + rank
            )  # Reset the environment with a specific seed
            return env

        return _init

    def get_service_map(self, network: Network):
        """
        Class function to get the service mapping based on host attributes.
        """
        killchain = [
            ARTDiscovery,
            ARTPrivilegeEscalation,
            ARTImpact,
            ARTLateralMovement,
        ]
        service_mapping = {}
        for name, host in network.hosts.items():
            service_mapping[name] = {}
            for kcp in killchain:
                service_mapping[name][kcp] = []
                kcp_valid_techniques = kcp.validity_mapping[host.os][kcp.get_name()]
                for mid in kcp_valid_techniques:
                    technique = art_techniques.technique_mapping[mid]
                    if len(host.host_type.cve_list & technique.cve_list) > 0:
                        service_mapping[name][kcp].append(mid)
        return service_mapping

    def configure_evaluation(self):
        self.device = torch.device("cpu")
        print(f"Using device {self.device}")

        # Set up network and Host-Technique mapping outside of environment.
        # This keeps the time-consuming processes from running for each environment.
        # Load networks from yaml here
        network_configs = []
        if isinstance(self.args.network_config, str):
            network_configs.append(self.args.network_config)
        else:
            for config in self.args.network_config:
                network_configs.append(config)
        
        self.networks = {}
        self.args.service_mapping = {}
        for config in network_configs:
            network_config = files("cyberwheel.data.configs.network").joinpath(
                config
            )

            print(f"Building network: {config} ...")

            network = Network.create_network_from_yaml(network_config)
            network_name = network.name
            self.networks[network_name] = network

            print("Mapping attack validity to hosts...", end=" ")
            self.args.service_mapping[network_name] = get_service_map(network)
            print("done")
        self.env = self.make_env(0)()

        experiment_name = self.args.experiment

        agent_filename = f"{self.args.checkpoint}.pt"

        # If download from W&B, use API to get run data.
        #print(self.envs.envs[0].red_max_action_space_size)
        #print(self.envs.envs[0].red_agent.get_observation_space())
        if self.args.download_model:
            api = wandb.Api()
            blue_run = api.run(
                f"{self.args.wandb_entity}/{self.args.wandb_project_name}/runs/{self.args.blue_run}"
            )
            blue_model = blue_run.file(agent_filename)
            blue_model.download(
                files("cyberwheel.data.models").joinpath(self.args.blue_model), exist_ok=True
            )

            red_run = api.run(
                f"{self.args.wandb_entity}/{self.args.wandb_project_name}/runs/{self.args.red_run}"
            )
            red_model = red_run.file(agent_filename)
            red_model.download(
                files("cyberwheel.data.models").joinpath(self.args.red_model), exist_ok=True
            )

        self.blue_max_action_space_size = self.env.blue_max_action_space_size
        self.red_max_action_space_size = self.env.red_max_action_space_size

        blue_obs = self.env.observation_space
        red_obs = self.env.red_observation_space

        #print(f"B: {self.blue_obs.shape} | {self.blue_max_action_space_size}")
        #print(f"R: {self.red_obs.shape} | {self.red_max_action_space_size}")
        self.blue_agent = RLAgent(blue_obs.shape, self.blue_max_action_space_size).to(self.device)
        self.red_agent = RLAgent(red_obs.shape, self.red_max_action_space_size).to(self.device)

        self.red_obs = self.env.red_agent.get_observation_space()
        self.blue_obs = self.env.reset()

        # Load model from models/ directory
        self.blue_agent.load_state_dict(
            torch.load(
                files(f"cyberwheel.data.models.{self.args.blue_model}").joinpath(agent_filename),
                map_location=self.device,
            )
        )
        self.red_agent.load_state_dict(
            torch.load(
                files(f"cyberwheel.data.models.{self.args.red_model}").joinpath(agent_filename),
                map_location=self.device,
            )
        )

        self.blue_agent.eval()
        self.red_agent.eval()

        print("Resetting the environment...")

        self.episode_rewards = []
        self.total_reward = 0
        self.steps = 0

        print("Playing environment...")

        # Set up dirpath to store action logs CSV
        if self.args.graph_name != None:
            self.now_str = self.args.graph_name
        else:
            self.now_str = f"{experiment_name}_evaluate_{self.args.network_config.split('.')[0]}_{self.args.red_agent}_scaling{int(self.args.reward_scaling)}_{self.args.reward_function}reward"
        self.log_file = files("cyberwheel.data.action_logs").joinpath(f"{self.now_str}.csv")

        self.actions_df = pd.DataFrame()
        self.full_episodes = []
        self.full_steps = []
        self.full_red_action_type = []
        self.full_red_action_src = []
        self.full_red_action_dest = []
        self.full_red_action_success = []
        self.full_blue_actions = []
        self.full_rewards = []
        self.full_blue_action_successes = []


        self.blue_action_mask = [False] * self.blue_max_action_space_size
        self.blue_action_mask[0:3] = [True, True, True]
        self.red_action_mask = [False] * self.red_max_action_space_size

        with open(self.log_file, 'w') as f: # Create an empty CSV for new action logs, overwrite previous
            pass

    def evaluate(self):
        self.start_time = time.time()
        for episode in tqdm(range(self.args.num_episodes)):
            self.blue_obs = self.env.reset()
            self.red_obs = self.env.red_agent.get_observation_space()
            for step in range(self.args.num_steps):
                #print(self.blue_obs)
                #print(self.red_obs)
                if step == 0:
                    self.blue_obs = self.blue_obs[0]

                self.blue_obs = torch.Tensor(self.blue_obs).to(self.device)
                self.red_obs = torch.Tensor(self.red_obs).to(self.device)

                #blue_action_space_size = self.env.blue_agent.action_space._action_space_size
                red_action_space_size = self.env.red_agent.action_space._action_space_size

                #blue_action_mask = self.blue_action_mask # get_action_mask(blue_action_space_size, self.blue_action_mask)
                self.red_action_mask = get_action_mask(red_action_space_size, self.red_action_mask)

                blue_action_mask = torch.asarray(self.blue_action_mask)
                red_action_mask = torch.asarray(self.red_action_mask)

                blue_action, _, _, _ = self.blue_agent.get_action_and_value(
                    self.blue_obs, action_mask=blue_action_mask
                )
                red_action, _, _, _ = self.red_agent.get_action_and_value(
                    self.red_obs, action_mask=red_action_mask
                )
                self.env.red_action = red_action.cpu().numpy()

                #print(red_action.cpu().numpy())
                #print(blue_action.cpu().numpy())

                _, rew, done, _, info = self.env.step(blue_action.cpu().numpy())

                rew = rew
                done = done

                blue_action = info["blue_action"]
                red_action_type = info["red_action"]
                red_action_src = info["red_action_src"]
                red_action_dest = info["red_action_dst"]
                red_action_success = info["red_action_success"]
                blue_action_success = info["blue_action_success"]
                self.red_obs = info["red_obs"]
                self.blue_obs = info["blue_obs"]

                actions_df = pd.DataFrame(
                {
                    "episode": [episode],
                    "step": [step],
                    "red_action_success": [red_action_success],
                    "red_action_type": [red_action_type],
                    "red_action_src": [red_action_src],
                    "red_action_dest": [red_action_dest],
                    "blue_action": [blue_action],
                    "blue_success": [blue_action_success],
                    "reward": [rew],
                })
                actions_df.to_csv(self.log_file, mode='a', header = os.path.getsize(self.log_file) == 0, index=False)

                # If generating graphs for dash server view
                if self.args.visualize:
                    # visualize(net, episode, step, now_str, history, killchain)
                    pass

                self.total_reward += rew
                self.steps += 1

            self.steps = 0
            self.episode_rewards.append(self.total_reward)
            self.total_reward = 0



        # Save action metadata to CSV in action_logs/

        self.total_time = time.time() - self.start_time
        print("charts/SPS", int(2000 / self.total_time))
        self.total_reward = sum(self.episode_rewards)
        self.episodes = len(self.episode_rewards)
        if self.episodes == 0:
            print(f"Mean Episodic Reward: {float(self.total_reward)}")
        else:
            print(f"Mean Episodic Reward: {float(self.total_reward) / self.episodes}")

        print(f"Total Time Elapsed: {self.total_time}")
