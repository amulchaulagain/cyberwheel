import yaml
import importlib

from pathlib import PosixPath
from typing_extensions import Self, Tuple, Type
from importlib.resources import files

from typing import Iterable

from cyberwheel.red_agents import ARTAgent, RLARTAgent, ARTCampaign
import cyberwheel.red_agents.action_space as action_space
from cyberwheel.red_agents.rl_red_agent import RLARTAgentResult, HostView
from cyberwheel.red_agents.red_agent_base import KnownHostInfo, KnownSubnetInfo
from cyberwheel.network.network_base import Network, Host
from cyberwheel.red_actions.red_base import RedActionResults
from cyberwheel.red_actions.technique import Technique
from cyberwheel.red_agents.strategies import RedStrategy, BruteForce
from cyberwheel.red_actions.art_techniques import RemoteSystemDiscovery, NetworkServiceDiscovery, SudoandSudoCaching, DataEncryptedforImpact, LinuxLateralMovement
from cyberwheel.red_actions.atomic_test import AtomicTest
from cyberwheel.reward import RewardMap

import numpy as np

class RLRedCampaign(ARTCampaign):
    def __init__(self, network: Network, args) -> None:
        self.args = args
        super().__init__(network, args)

        self.observation = {}
        self.observation[self.current_host.name] = HostView(self.current_host.name, on_host=True)
        self.tracked_hosts = self.network.get_all_hostnames()
    
    def from_yaml(self) -> None:
        with open(self.config, "r") as f:
            config = yaml.safe_load(f)
        
        action_classes = [
            RemoteSystemDiscovery,
            NetworkServiceDiscovery,
            SudoandSudoCaching,
            DataEncryptedforImpact,
            LinuxLateralMovement
        ]
        self.atomic_test = {
            "Remote System Discovery": RemoteSystemDiscovery.get_atomic_test("96db2632-8417-4dbb-b8bb-a8b92ba391de"),
            "Network Service Discovery": NetworkServiceDiscovery.get_atomic_test("515942b0-a09f-4163-a7bb-22fefb6f185f"),
            "Sudo and Sudo Caching": SudoandSudoCaching.get_atomic_test("150c3a08-ee6e-48a6-aeaf-3659d24ceb4e"),
            "Data Encrypted for Impact": DataEncryptedforImpact.get_atomic_test("08cbf59f-85da-4369-a5f4-049cffd7709f"),
            "LinuxLateralMovement": LinuxLateralMovement.get_atomic_test("uuid")
        }
        self.entry_host: Host = self.network.get_random_user_host()
        self.current_host : Host = self.entry_host

        self.leader = []

        self.action_space = getattr(action_space, config["action_space"])(action_classes, self.current_host.name)

        self.killchain = []
        self.reward_map = {
            "Remote System Discovery": (0.0, 0.0),
            "Network Service Discovery": (10.0, 0.0),
            "Sudo and Sudo Caching": (0.0, 0.0),
            "Data Encrypted for Impact": (100.0, 0.0),
            "LinuxLateralMovement": (0.0, 0.0)
        }

    def act(self, action: int) -> RLARTAgentResult:
               # self.handle_network_change() TODO: Implement when developing static blue agent
        art_action, target_host_name = self.action_space.select_action(
            action
        )  # Selects ART Action, should include the action and target host (based on view?)
        source_host = self.current_host
        target_host = self.network.get_node_from_name(target_host_name)
        success = False
        if self.validate_action(art_action, target_host_name):
            action_results, action = self.run_action(target_host, art_action)
            success = action_results.attack_success
            self.handle_action(action_results)
        return RLARTAgentResult(
            art_action, 
            source_host, 
            target_host, 
            success, 
            self.get_observation_space()
        )  # Returns what ARTAgent act() should, probably. Or the observation space? 

    def validate_action(self, action, target_host: str) -> bool:
            host_view = self.observation[target_host]
            if action == RemoteSystemDiscovery:  # valid if host.sweeped == False
                return not host_view.sweeped
            elif (
                action == NetworkServiceDiscovery
            ):  # valid if host.scanned == False and host.sweeped == True
                return host_view.sweeped and not host_view.scanned
            elif (
                action == LinuxLateralMovement
            ):  # valid if host.scanned && host.sweeped && host.discovered && !host.on_target
                return (
                    host_view.sweeped
                    and host_view.scanned
                    and host_view.discovered
                    and not host_view.on_host
                )
            elif (
                action == SudoandSudoCaching
            ):  # valid if host.scanned && host.sweeped && host.discovered && host.on_target && !host.escalated
                return (
                    host_view.sweeped
                    and host_view.scanned
                    and host_view.discovered
                    and host_view.on_host
                    and not host_view.escalated
                )
            elif (
                action == DataEncryptedforImpact
            ):  # valid if host.scanned && host.sweeped && host.discovered && host.on_target && host.escalated
                return (
                    host_view.sweeped
                    and host_view.scanned
                    and host_view.discovered
                    and host_view.on_host
                    and host_view.escalated
                    and not host_view.impacted
                )
            else:
                return False
    
    def handle_action(self, result: RedActionResults) -> None:
        if not result.attack_success:
            return
        action = result.action
        src_host = result.src_host.name
        target_host = result.target_host.name
        if action == RemoteSystemDiscovery:  # Adds pingsweeped hosts to obs
            self.observation[target_host].sweeped = True
            hosts = [host.name for host in result.target_host.subnet.connected_hosts]
            #interfaced_hosts = result.metadata["interfaced_hosts"]
            for h in set(hosts) - self.observation.keys():
                self.observation[h] = HostView(h, sweeped=True)
                self.action_space.add_host(h)
            #for h in set(interfaced_hosts):
            #    self.observation[h] = HostView(h)
            #    self.action_space.add_host(h)
        elif action == NetworkServiceDiscovery:  # Scans target host
            self.observation[target_host].scanned = True
            self.observation[target_host].discovered = True
            self.observation[target_host].type = self.network.get_node_from_name(
                target_host
            ).host_type.name
        elif action == LinuxLateralMovement:  # Moves to target host
            self.observation[target_host].on_host = True
            self.observation[src_host].on_host = False
            self.current_host = result.target_host
        elif action == SudoandSudoCaching:
            self.observation[target_host].escalated = True
        elif action == DataEncryptedforImpact:
            self.observation[target_host].impacted = True

    def get_observation_space(self):
        """
        Takes red agent view of network and transforms it into the obs vector.
        """
        obs = []
        for view in self.observation.values():
            obs += [
                view.get_type(),
                int(view.sweeped),
                int(view.scanned),
                int(view.discovered),
                int(view.on_host),
                int(view.escalated),
                int(view.impacted),
            ]
        obs = obs + [-1] * (100 - len(obs))
        _obs = np.array(obs, dtype=np.float64)
        return _obs
    
    def reset(self, entry_host: Host, network: Network):
        self.network = network
        self.current_host = entry_host
        self.observation = {}
        self.observation[entry_host.name] = HostView(entry_host.name, on_host=True)
        self.action_space.reset(entry_host.name)

    def run_action(self, target_host: Host, art_action) -> Tuple[RedActionResults, Type[Technique]]:
        self.leader = self.network.get_all_hostnames()

        technique = art_action()
        atomic_test = self.atomic_test[art_action.get_name()]

        mitre_id = technique.mitre_id
        technique_name = technique.name

        action_results = RedActionResults(self.current_host, target_host)
        action_results.modify_alert(dst=target_host, src=self.current_host)

        # TODO: Checking if technique will work: OS match, CVE in cve_list, Killchain check
        action_results.add_successful_action()

        processes = []
        for dep in atomic_test.dependencies:
            processes.extend(dep.get_prerequisite_command)
            processes.extend(dep.prerequisite_command)
        if atomic_test.executor != None:
            processes.extend(atomic_test.executor.command)
            processes.extend(atomic_test.executor.cleanup_command)

        for p in processes:
            target_host.run_command(atomic_test.executor, p, "root")
        action_results.add_metadata(
            target_host.name,
            {
                "commands": processes,
                "mitre_id": mitre_id,
                "technique": technique_name,
            },
        )
        action_results.action = art_action



        return action_results, art_action

    def get_reward_map(self):
        return self.reward_map