from cyberwheel.blue_agents.blue_agent import BlueAgent
from cyberwheel.reward import RewardMap
from importlib.resources import files
from cyberwheel.network.network_base import Network
from cyberwheel.blue_actions.actions import DeployDecoyHost, Nothing

import random
import yaml

class RandomBlueAgent(BlueAgent):
    """
    This agent does a random action to a random subnet.
    """

    def __init__(self, network: Network, args) -> None:
        super().__init__()
        self.config = files("cyberwheel.resources.configs.blue_agent").joinpath(
            args.blue_agent
        )
        self.network = network
        self.decoys_deployed = 0
        self.actions = []
        self.subnets = [s for s in self.network.get_all_subnets()]
    
    def _init_blue_actions(self) -> None:
        for action_class, action_info in self.actions:
            # Check configs and read them if they are new
            action_configs = {}
            for name, config in action_info.configs.items():
                # Skip configs that have already been seen
                if not config in self.configs:
                    conf_file = files(f"cyberwheel.resources.configs.{name}").joinpath(
                        config
                    )
                    with open(conf_file, "r") as f:
                        contents = yaml.safe_load(f)
                    self.configs[config] = contents
                    action_configs[name] = contents
                else:
                    action_configs[name] = self.configs[config]

            action_kwargs = {}
            for sd in action_info.shared_data:
                action_kwargs[sd] = self.shared_data[sd]
            action = action_class(self.network, action_configs, **action_kwargs)
            self.actions.append(action)

    def act(self) -> str:
        action = random.choice(self.actions)
        target = random.choice(self.subnets)
        action_result = action.execute(subnet=target)
        return action_result

    def get_reward_map(self) -> RewardMap:
        return {
            "nothing": (0, 0),
            "deploy_decoy": (0, 0)}

    def reset(self):
        return
