import time
import pandas as pd

from importlib.resources import files
from tqdm import tqdm
from cyberwheel.network.network_base import Network
from cyberwheel.red_actions.actions import (
    ARTDiscovery,
    ARTLateralMovement,
    ARTPrivilegeEscalation,
    ARTImpact,
)
from cyberwheel.red_actions import art_techniques
from cyberwheel.cyberwheel_envs import Cyberwheel

class Runner:
    def __init__(self, args):
        self.args = args

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
        for host in network.hosts.values():
            service_mapping[host.name] = {}
            for kcp in killchain:
                service_mapping[host.name][kcp] = []
                kcp_valid_techniques = kcp.validity_mapping[host.os][kcp.get_name()]
                for mid in kcp_valid_techniques:
                    technique = art_techniques.technique_mapping[mid]
                    if len(host.host_type.cve_list & technique.cve_list) > 0:
                        service_mapping[host.name][kcp].append(mid)
        return service_mapping

    def configure(self):
        network_config = files("cyberwheel.resources.configs.network").joinpath(
            self.args.network_config
        )
        network = Network.create_network_from_yaml(network_config)

        self.args.service_mapping = self.get_service_map(network)

        self.env = Cyberwheel(self.args, network)

        print("Resetting the environment...")

        self.steps = 0

        print("Playing environment...")

        self.log_file = files("cyberwheel.action_logs").joinpath(f"{self.args.experiment_name}.csv")

        self.actions_df = pd.DataFrame()
        self.full_episodes = []
        self.full_steps = []
        self.full_red_action_type = []
        self.full_red_action_src = []
        self.full_red_action_dest = []
        self.full_red_action_success = []
        self.full_blue_actions = []

    def run(self):
        self.start_time = time.time()
        for episode in tqdm(range(self.args.num_episodes)):
            for step in range(self.args.num_steps):
                info = self.env.step()
                red_agent_result = info['red_agent_result']
                blue_agent_result = info['blue_agent_result']

                blue_action = blue_agent_result.name
                red_action_type = red_agent_result.action.get_name()
                red_action_src = red_agent_result.src_host.name
                red_action_dest = red_agent_result.target_host.name
                red_action_success = red_agent_result.success

                self.full_episodes.append(episode)
                self.full_steps.append(step)
                self.full_red_action_type.append(red_action_type)
                self.full_red_action_src.append(red_action_src)
                self.full_red_action_dest.append(red_action_dest)
                self.full_red_action_success.append(red_action_success)
                self.full_blue_actions.append(blue_action)

                self.steps += 1
            self.steps = 0
            self.env.reset()

        self.actions_df = pd.DataFrame(
            {
                "episode": self.full_episodes,
                "step": self.full_steps,
                "red_action_success": self.full_red_action_success,
                "red_action_type": self.full_red_action_type,
                "red_action_src": self.full_red_action_src,
                "red_action_dest": self.full_red_action_dest,
                "blue_action": self.full_blue_actions,
            }
        )

        # Save action metadata to CSV in action_logs/
        self.actions_df.to_csv(self.log_file)

        self.total_time = time.time() - self.start_time
        print("charts/SPS", int((self.args.num_steps * self.args.num_episodes) / self.total_time))

        print(f"Total Time Elapsed: {self.total_time}")
    
    def close(self):
        pass
