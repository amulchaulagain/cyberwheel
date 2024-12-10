import copy
from importlib.resources import files
from gym import spaces
import gym
from typing import Dict, List
import yaml
import numpy as np

from .cyberwheel import Cyberwheel
from cyberwheel.blue_agents import InactiveBlueAgent
from cyberwheel.detectors.alert import Alert
from cyberwheel.network.network_base import Network
from cyberwheel.network.host import Host
from cyberwheel.red_agents import RLARTAgent
from cyberwheel.reward import RLRedReward


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

    def __init__(
        self,
        network_config="15-host-network.yaml",
        host_def_file="host_definitions.yaml",
        reward_function="default",
        red_agent="rl_red_agent",
        evaluation=False,
        network=None,
        service_mapping={},
        **kwargs,
    ):
        """
        The DynamicCyberwheel class is used to define the Cyberwheel environment. It allows you to use a YAML
        file to configure the actions, rewards, and logic of the blue agent. Given various configurations, it
        will initiate the environment with the red agent, blue agent, reward functions, and network state.
        Important member variables:

        * `network_config`: optional
            - The name (not filepath) of the network configuration file.
            - Default: 15-host-network.yaml

        * `decoy_host_file`: optional
            - The name (not filepath) of the decoy configuration file.
            - Default: decoy_hosts.yaml

        * `host_def_file`: optional
            - The name (not filepath) of the host configuration file.
            - Default: host_definitions.yaml

        * `detector_config`: optional
            - The name (not filepath) of the detector configuration file.
            - Default: detector.yaml

        * `min_decoys`: optional
            - The minimum number of decoys the blue agent should deploy. This range is not used for the default reward function.
            - Default: 0

        * `max_decoys`: optional
            - The maximum number of decoys the blue agent should deploy. This range is not used for the default reward function.
            - Default: 1

        * `blue_reward_scaling`: optional
            - The scaling factor for the blue agent's rewards.
            - Default: 10

        * `reward_function`: optional
            - The reward function used in the environment. Options: 'default' | 'step_detected'
            - The default reward function uses the RecurringReward class.
            - Default: default

        * `red_agent`: optional
            - The red agent used in the environment. Currently only using the ART Agent
            - Default: 'art_agent'

        * `evaluation`: optional
            - boolean for if the environment should log information for evaluation script or not.
            - Default: False

        * `blue_config`: optional
            - The name (not filepath) of the blue agent configuration file.
            - Default: blue_agent_config.yaml

        * `network`: optional
            - The Network object to use throughout the environment. This prevents long start-up times when training with multiple environments.
            - If not passed, it will build the network with the config file passed.
            - Default: None

        * `service_mapping`: optional
            - The host -> valid_action mapping from the exploitable services on the Network.
            - If not passed, it will build the mapping when defining the red agent.
            - Default: {}
        """
        network_conf_file = files("cyberwheel.resources.configs.network").joinpath(
            network_config
        )
        host_conf_file = files(
            "cyberwheel.resources.configs.host_definitions"
        ).joinpath(host_def_file)
        super().__init__(config_file_path=network_conf_file, network=network)
        self.total = 0
        self.max_steps = kwargs.get("num_steps", 100)
        self.current_step = 0

        # Create action space. Decoy action for each decoy type for each subnet.
        # Length = num_decoy_host_types * num_subnets
        with open(host_conf_file, "r") as f:
            self.host_defs = yaml.safe_load(f)["host_types"]

        self.service_mapping = service_mapping

        self.red_agent = RLARTAgent(
            self.network,
            self.network.get_random_user_host(),
            service_mapping=service_mapping,
        )

        self.observation_space = spaces.Box(
            0, 2, shape=(len(self.red_agent.get_observation_space()),)
        )

        self.action_space = self.red_agent.create_action_space()

        self.blue_agent = InactiveBlueAgent()

        self.reward_function = reward_function

        self.reward_calculator = RLRedReward(
            self.red_agent.get_reward_map(), self.blue_agent.get_reward_map()
        )

        self.evaluation = evaluation

    def step(self, action):
        """
        Steps through environment.
        1. Blue agent runs action
        2. Red agent runs action
        3. Calculate reward based on red/blue actions and network state
        4. Convert Alerts from Detector into observation space
        5. Return obs and related metadata
        """
        blue_action = self.blue_agent.act()
        red_action_result = self.red_agent.act(action)

        red_action_name = red_action_result.action.get_name()
        red_action_src = red_action_result.src_host.name
        red_action_dst = red_action_result.target_host.name
        red_action_success = red_action_result.success

        # print(f"{red_action_name} from {red_action_src} to {red_action_dst} - {red_action_success}")
        # print(f"{self.red_agent.observation[red_action_dst].__dict__}")
        # for h in self.red_agent.observation:
        #    print(self.red_agent.observation[h].__dict__)
        # print(f"{self.red_agent.observation}")
        obs_vec = self.red_agent.get_observation_space()

        reward = self.reward_calculator.calculate_reward(
            red_action_name,
            blue_action,
            red_action_success,
            True,
            False,
        )

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
                "blue_action": blue_action,
                "network": self.red_agent.network,
                # "history": self.red_agent.history,
                "killchain": self.red_agent.killchain,
            }

        return (
            obs_vec,
            reward,
            done,
            False,
            info,
        )

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
        return np.zeros((len(self.red_agent.get_observation_space()),)), {}

    # if you open any other processes close them here
    def close(self):
        pass
