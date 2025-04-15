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
from cyberwheel.red_agents.rl_red_campaign import RLRedCampaign
from cyberwheel.utils import YAMLConfig
from cyberwheel.observation import HistoryObservation
from cyberwheel.detectors.handler import DetectorHandler


def host_to_index_mapping(network: Network) -> Dict[Host, int]:
    """
    This will help with constructing the obs_vec.
    It will need to be called and save during __init__()
    because deploying decoy hosts may affect the order of
    the list network.get_non_decoy_hosts() returns.
    This might not be the case, but this will ensure the
    original indices are preserved.
    """
    mapping: Dict[Host, int] = {}
    i = 0
    for host in network.get_nondecoy_hosts():
        mapping[host.name] = i
        i += 1
    return mapping


def decoy_alerted(alerts: List[Alert]) -> bool:
    for alert in alerts:
        for dst_host in alert.dst_hosts:
            if dst_host.decoy:
                return True
    return False


class CyberwheelRedRL(gym.Env, Cyberwheel):
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
        self.max_steps = args.num_steps
        self.current_step = 0

        # Create action space. Decoy action for each decoy type for each subnet.
        # Length = num_decoy_host_types * num_subnets
        with open(host_conf_file, "r") as f:
            self.host_defs = yaml.safe_load(f)["host_types"]

        self.service_mapping = args.service_mapping
        self.args = args

        if args.valid_targets == "servers":
            #valid_targets = [h.name for h in self.network.get_all_server_hosts()] # TODO
            valid_targets = ["server01", "server02", "server03", "decoy01", "decoy02"]
        elif args.valid_targets == "users":
            valid_targets = [h.name for h in self.network.get_all_user_hosts()]
        elif type(args.valid_targets) is list:
            valid_targets = args.valid_targets
        elif type(args.valid_targets) is str:
            valid_targets = [args.valid_targets]
        else:
            valid_targets = [h.name for h in self.network.get_all_hosts()]

        if args.train_red:
            #self.red_agent = RLARTAgent(self.network, args) # TODO
            self.red_agent = RLRedCampaign(self.network, args)
            self.blue_agent = InactiveBlueAgent()
            self.rl_agent = self.red_agent
            self.static_agent = self.blue_agent
            self.observation_space = spaces.Box(
                0, 2, shape=(len(self.red_agent.get_observation_space()),)
            )
            self.max_action_space_size = (
                self.network.size()
                * self.red_agent.action_space.num_actions
                * 2
            )
            self.action_space = self.red_agent.action_space.create_action_space(
                self.max_action_space_size
            )
        else:
            if args.campaign:
                self.red_agent = ARTCampaign(self.network, args)
            else:
                self.red_agent = ARTAgent(self.network, args)
            self.blue_agent = DynamicBlueAgent(self.network, args)
            self.rl_agent = self.blue_agent
            self.static_agent = self.red_agent
            self.max_action_space_size = self.network.get_num_subnets() * 2
            self.action_space = self.blue_agent.create_action_space(
                self.max_action_space_size
            )

            detector_conf_file = files(
                "cyberwheel.resources.configs.detector"
            ).joinpath(args.detector_config)
            self.detector = DetectorHandler(detector_conf_file)
            self.observation_space = spaces.Box(0, 1, shape=(2 * self.network.size(),))
            self.alert_converter = HistoryObservation(
                self.observation_space.shape, host_to_index_mapping(self.network)
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
        red_id = -1
        red_recurring = 0
        blue_id = -1
        blue_recurring = 0

        #print(self.observation_space.shape)
        #print(self.max_action_space_size)

        # print(self.red_agent.history.hosts.keys())

        if self.args.train_red:
            blue_action_name = self.blue_agent.act()
            blue_action_success = True
            red_action_result = self.red_agent.act(action)
            red_action_name = red_action_result.action.get_name()
            red_action_src = red_action_result.src_host.name
            red_action_dst = red_action_result.target_host.name
            red_action_success = red_action_result.success

            obs_vec = self.red_agent.get_observation_space()
        else:
            blue_agent_result = self.blue_agent.act(action)
            blue_id = blue_agent_result.id
            blue_recurring = blue_agent_result.recurring
            blue_action_name = blue_agent_result.name
            blue_action_success = blue_agent_result.success

            red_action_name = self.red_agent.act().get_name()
            action_metadata = self.red_agent.history.history[-1]
            red_action_src = action_metadata["src_host"]
            red_action_dst = action_metadata["target_host"]
            red_action_success = action_metadata["success"]

            red_action_result = self.red_agent.history.recent_history()
            alerts = self.detector.obs([red_action_result.detector_alert])
            obs_vec = self._get_obs(alerts)

        reward = self.reward_calculator.calculate_reward(
            red_action_name,
            blue_action_name,
            red_action_success,
            blue_action_success,
            self.network.get_node_from_name(red_action_dst),
            red_id=red_id,
            red_recurring=red_recurring,
            blue_id=blue_id,
            blue_recurring=blue_recurring,
        )

        #print(f"{red_action_name} - {red_action_src} to {red_action_dst} | {reward}")

        self.total += reward

        done = self.current_step >= self.max_steps

        self.current_step += 1

        info = {}
        if self.evaluation:
            info = {
                "red_action": red_action_name,
                "red_action_src": red_action_src,
                "red_action_dst": red_action_dst,
                "red_action_success": red_action_success,
                "blue_action": blue_action_name,
                "network": self.red_agent.network,
                # "history": self.red_agent.history,
                # "killchain": self.red_agent.killchain,
            }
        if self.args.train_blue:
            self.detector.reset()
            self.detector.reset()
        #print(obs_vec.shape)
        return (
            obs_vec,
            reward,
            done,
            False,
            info,
        )

    def _get_obs(self, alerts: List[Alert]) -> Iterable:
        return self.alert_converter.create_obs_vector(alerts)

    def _reset_obs(self) -> Iterable:
        return self.alert_converter.reset_obs_vector()

    def reset(self, seed=None, options=None):
        self.total = 0
        self.current_step = 0
        self.network.reset()

        self.red_agent.reset(
            self.red_agent.entry_host,
            network=self.network,
            #leader=self.red_agent.leader, # TODO
        )

        self.blue_agent.reset()

        self.reward_calculator.reset()
        if self.args.train_red:
            return np.zeros((len(self.red_agent.get_observation_space()),)), {}
        else:
            self.observation_space = spaces.Box(0, 1, shape=(2 * self.network.size(),))
            self.alert_converter = HistoryObservation(
                self.observation_space.shape, host_to_index_mapping(self.network)
            )
            return self._reset_obs(), {}

    # if you open any other processes close them here
    def close(self):
        pass

    @property
    def red_agent_action_space_size(self):
        return self.red_agent.action_space._action_space_size

    @property
    def blue_agent_action_space_size(self):
        return self.blue_agent.action_space._action_space_size
