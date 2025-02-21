import importlib
import yaml
import numpy as np

from typing import Iterable

from cyberwheel.red_agents import ARTAgent
from cyberwheel.red_actions.red_base import RedActionResults
from cyberwheel.reward.reward_base import RewardMap
from cyberwheel.network.network_base import Network, Host
from cyberwheel.red_actions.actions.art_killchain_phases import (
    ARTKillChainPhase,
    ARTPingSweep,
    ARTPortScan,
    ARTDiscovery,
    ARTLateralMovement,
    ARTPrivilegeEscalation,
    ARTImpact,
)

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
        else: # Unknown
            return 0


class RLARTAgent(ARTAgent):
    """
    Filler
    """

    def __init__(self, network: Network, args) -> None:
        super().__init__(network, args, service_mapping=args.service_mapping)

        self.observation = {}
        self.observation[self.entry_host.name] = HostView(self.entry_host.name, on_host=True)
        self.obs_size = 7
        self.tracked_hosts = self.network.get_all_hostnames()
    
    def from_yaml(self) -> None:
        with open(self.config, "r") as f:
            contents = yaml.safe_load(f)

        # Get module import path
        action_classes = [
            ARTPingSweep,
            ARTPortScan,
            ARTDiscovery,
            ARTLateralMovement,
            ARTPrivilegeEscalation,
            ARTImpact,
        ]

        self.reward_map = {}

        self.entry_host: Host = self.network.get_node_from_name(contents["entry_host"]) if "entry_host" in contents and contents["entry_host"] else self.network.get_random_user_host()
        self.current_host : Host = self.entry_host

        # Initialize the action space
        as_class = contents['action_space']
        asm = importlib.import_module("cyberwheel.red_agents.action_space")
        self.action_space = getattr(asm, as_class)(action_classes, self.current_host.name)

        for k, v in contents['actions'].items():
            self.reward_map[k] = (v["reward"]["immediate"], v["reward"]["recurring"])

    def act(self, action: int) -> RLARTAgentResult:
        # self.handle_network_change() TODO: Implement when developing static blue agent
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
            art_action, 
            source_host, 
            target_host, 
            success, 
            self.get_observation_space()
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
            for h in set(interfaced_hosts):
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
        self.obs_size = len(obs)
        #TODO
        obs = obs + [0] * (200 - self.obs_size)
        _obs = np.array(obs, dtype=np.int64)
        #_obs = np.array(obs, dtype=np.float64) # TODO
        return _obs

    def reset(self, entry_host: Host, network: Network):
        self.network = network
        self.current_host = entry_host
        self.observation = {}
        self.observation[entry_host.name] = HostView(entry_host.name, on_host=True)
        self.obs_size = 7
        self.action_space.reset(entry_host.name)
