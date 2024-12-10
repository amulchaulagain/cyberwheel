import builtins
import importlib
import yaml
import numpy as np

from importlib.resources import files
from typing import Dict, List, Iterable
from gym import Space

from cyberwheel.red_agents import ARTAgent, RedAgent
from cyberwheel.red_actions.red_base import RedActionResults
from cyberwheel.red_actions.actions.art_killchain_phases import (
    ARTKillChainPhase,
    ARTPingSweep,
    ARTPortScan,
    ARTDiscovery,
    ARTLateralMovement,
    ARTPrivilegeEscalation,
    ARTImpact,
)
from cyberwheel.reward.reward_base import RewardMap
from cyberwheel.network.network_base import Network, Host
from cyberwheel.blue_agents.action_space.discrete import RedDiscreteActionSpace


class RLARTAgentResult:
    def __init__(
        self,
        action: ARTKillChainPhase,
        src_host: Host,
        target_host: Host,
        success: bool,
        obs: Iterable[int],
    ):
        """
        - `name`: name of the red action executed
        - `success`: whether this action successfully executed or not
        """
        self.action = action
        self.src_host = src_host
        self.target_host = target_host
        self.success = success
        self.obs = obs


class HostView:
    def __init__(
        self,
        name: str,
        type: str = "unknown",
        sweeped: bool = False,
        scanned: bool = False,
        discovered: bool = False,
        on_host: bool = False,
        escalated: bool = False,
        impacted: bool = False,
    ):
        self.name = name
        self.type = type
        self.sweeped = sweeped
        self.scanned = scanned
        self.discovered = discovered
        self.on_host = on_host
        self.escalated = escalated
        self.impacted = impacted

    def get_type(self) -> int:
        if self.type.lower() == "workstation":
            return 1
        elif self.type.lower() == "server":
            return 2
        else:
            return 0


class RLARTAgent(ARTAgent):
    """
    Filler
    """

    def __init__(self, network: Network, entry_host: Host, service_mapping={}) -> None:
        super().__init__(entry_host=entry_host, service_mapping=service_mapping)
        self.network = network

        self.initialize_actions()

        self.reward_map = {
            ARTPingSweep.get_name(): (0, 0),
            ARTPortScan.get_name(): (0, 0),
            ARTDiscovery.get_name(): (0, 0),
            ARTLateralMovement.get_name(): (0, 0),
            ARTPrivilegeEscalation.get_name(): (0, 0),
            ARTImpact.get_name(): (400, 0),
        }
        self.observation = {}
        self.observation[entry_host.name] = HostView(entry_host.name, on_host=True)
        self.tracked_hosts = self.network.get_all_hostnames()

        # TODO: Delete this or comment out after debugging!!! Delete before committing.
        self.temp_step = 0
        self.temp_actions = []

    def initialize_actions(self) -> None:
        valid_actions = [
            ARTPingSweep,
            ARTPortScan,
            ARTDiscovery,
            ARTLateralMovement,
            ARTPrivilegeEscalation,
            ARTImpact,
        ]
        self.action_space: RedDiscreteActionSpace = RedDiscreteActionSpace(
            valid_actions, self.current_host.name
        )

    def act(self, action: int) -> RLARTAgentResult:
        # self.handle_network_change()
        # for h in self.observation:
        #    print(self.observation[h].__dict__)
        art_action, target_host_name = self.action_space.select_action(
            action
        )  # Selects ART Action, should include the action and target host (based on view?)
        source_host = self.current_host
        target_host = self.network.get_node_from_name(target_host_name)
        success = False
        if self.validate_action(art_action, target_host_name):
            if art_action == ARTPingSweep or art_action == ARTPortScan:
                result = art_action(
                    self.current_host, target_host
                ).sim_execute()  # Executes the ART Action, returns results
            else:
                result = art_action(
                    self.current_host,
                    target_host,
                    self.services_map[target_host_name][art_action],
                ).sim_execute()  # Executes the ART Action, returns results
            success = result.attack_success
            self.handle_action(result)
        return RLARTAgentResult(
            art_action, source_host, target_host, success, self.get_observation_space()
        )  # Returns what ARTAgent act() should, probably. Or the observation space?

    def handle_action(self, result: RedActionResults) -> None:
        if not result.attack_success:
            return
        action = result.action
        src_host = result.src_host.name
        target_host = result.target_host.name
        if action == ARTPingSweep:  # Adds pingsweeped hosts to obs
            self.observation[target_host].sweeped = True
            hosts = result.metadata["sweeped_hosts"]
            interfaced_hosts = result.metadata["interfaced_hosts"]
            for h in set(hosts) - self.observation.keys():
                self.observation[h] = HostView(h, sweeped=True)
                self.action_space.add_host(h)
            for h in set(
                interfaced_hosts
            ):  # - self.observation.keys(): TODO: Need to test if this works without set difference
                self.observation[h] = HostView(h)
                self.action_space.add_host(h)
        elif action == ARTPortScan:  # Scans target host
            self.observation[target_host].scanned = True
        elif action == ARTDiscovery:  # Discovers host type
            self.observation[target_host].discovered = True
            self.observation[target_host].type = self.network.get_node_from_name(
                target_host
            ).host_type.name
        elif action == ARTLateralMovement:  # Moves to target host
            self.observation[target_host].on_host = True
            self.observation[src_host].on_host = False
            self.current_host = result.target_host
        elif action == ARTPrivilegeEscalation:
            self.observation[target_host].escalated = True
        elif action == ARTImpact:
            self.observation[target_host].impacted = True

    def handle_network_change(self):
        current_hosts = set(self.network.get_all_hosts())
        new_hosts = current_hosts - self.tracked_hosts
        for h in new_hosts:
            self.services_map[h.name] = self.get_valid_techniques_by_host(
                h, self.all_kcps
            )
            self.observation[h.name] = HostView(h.name, sweeped=True)
            self.action_space.add_host(h.name)
        self.tracked_hosts = current_hosts

    def validate_action(self, action: ARTKillChainPhase, target_host: str) -> bool:
        host_view = self.observation[target_host]
        if action == ARTPingSweep:  # valid if host.sweeped == False
            return not host_view.sweeped
        elif (
            action == ARTPortScan
        ):  # valid if host.scanned == False and host.sweeped == True
            return host_view.sweeped and not host_view.scanned
        elif (
            action == ARTDiscovery
        ):  # valid if host.scanned && host.sweeped && !host.discovered
            return host_view.sweeped and host_view.scanned and not host_view.discovered
        elif (
            action == ARTLateralMovement
        ):  # valid if host.scanned && host.sweeped && host.discovered && !host.on_target
            return (
                host_view.sweeped
                and host_view.scanned
                and host_view.discovered
                and not host_view.on_host
            )
        elif (
            action == ARTPrivilegeEscalation
        ):  # valid if host.scanned && host.sweeped && host.discovered && host.on_target && !host.escalated
            return (
                host_view.sweeped
                and host_view.scanned
                and host_view.discovered
                and host_view.on_host
                and not host_view.escalated
            )
        elif (
            action == ARTImpact
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

    def get_reward_map(self) -> RewardMap:
        return self.reward_map

    def get_action_space_shape(self) -> tuple[int, ...]:
        return self.action_space.get_shape()

    def create_action_space(self) -> Space:
        return self.action_space.create_action_space()

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
        obs = obs + [-1] * (200 - len(obs))
        _obs = np.array(obs, dtype=np.float64)
        return _obs

    def reset(self, entry_host: Host, network: Network):
        self.network = network
        self.current_host = entry_host
        self.observation = {}
        self.observation[entry_host.name] = HostView(entry_host.name, on_host=True)
        self.initialize_actions()
