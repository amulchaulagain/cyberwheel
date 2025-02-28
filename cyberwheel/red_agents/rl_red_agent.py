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
from copy import copy

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


class RedObservation:
    def __init__(self, max_size: int):
        self.obs : dict[str, HostView] = {}
        self.max_size = max_size
        self.obs_vec : list[int] = [0] * max_size
        self.obs_index: dict[str, int] = {}
        self.size : int = 0
        #print(len(self.obs_vec))

    def add_host(
            self,
            host: str,
            type: str = "unknown",
            sweeped: bool = False,
            scanned: bool = False,
            discovered: bool = False,
            on_host: bool = False,
            escalated: bool = False,
            impacted: bool = False,
    ):
        self.obs[host] = HostView(name=host, type=type, sweeped=sweeped, scanned=scanned, discovered=discovered, on_host=on_host, escalated=escalated, impacted=impacted)
        self.obs_index[host] = self.size
        view = self.obs[host]
        #print(len(self.obs_vec))
        self.obs_vec[self.size:(self.size + 7)] = [
                view.get_type(),
                int(view.sweeped),
                int(view.scanned),
                int(view.discovered),
                int(view.on_host),
                int(view.escalated),
                int(view.impacted),
            ]
        self.size += 7
        #print(len(self.obs_vec))
        #print(list(self.obs.keys()))
        #print(self.obs_vec)
        pass

    def update_host(self, host: str, **kwargs):
        view = self.obs[host]
        view.type = kwargs.get("type", view.type)
        view.sweeped = kwargs.get("sweeped", view.sweeped)
        view.scanned = kwargs.get("scanned", view.scanned)
        view.discovered = kwargs.get("discovered", view.discovered)
        view.on_host = kwargs.get("on_host", view.on_host)
        view.escalated = kwargs.get("escalated", view.escalated)
        view.impacted = kwargs.get("impacted", view.impacted)

        host_index = self.obs_index[host]
        self.obs_vec[host_index:host_index+7] = self.get_view_obs(view)

    def get_view_obs(self, view: HostView) -> list[int]:
        view_obs = [
                view.get_type(),
                int(view.sweeped),
                int(view.scanned),
                int(view.discovered),
                int(view.on_host),
                int(view.escalated),
                int(view.impacted),
            ]
        return view_obs

    def reset(self, entry_host: str):
        self.obs = {}
        self.obs_index = {}
        self.obs_vec = [0] * self.max_size
        self.size = 0
        self.add_host(entry_host, on_host=True)


class RLARTAgent(ARTAgent):
    """
    Filler
    """

    def __init__(self, network: Network, args) -> None:
        super().__init__(network, args, service_mapping=args.service_mapping)
        self.tracked_hosts = self.network.hosts.keys()
        self.observation = RedObservation(len(self.network.hosts) * 15)
        self.observation.add_host(self.entry_host.name, on_host=True)
    
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

        self.entry_host: Host = self.network.hosts[contents["entry_host"]] if "entry_host" in contents and contents["entry_host"] else self.network.get_random_user_host()
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
        target_host = self.network.hosts[target_host_name]
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
            self.observation.update_host(target_host, sweeped=True)
            hosts = result.metadata["sweeped_hosts"]
            interfaced_hosts = result.metadata["interfaced_hosts"]
            for h in hosts:
                if h in self.observation.obs.keys():
                    continue
                self.observation.add_host(h, sweeped=True)
                self.action_space.add_host(h)
            for h in interfaced_hosts:
                if h in self.observation.obs.keys():
                    continue
                self.observation.add_host(h)
                self.action_space.add_host(h)
        elif action == ARTPortScan:  # Scans target host
            self.observation.update_host(target_host, scanned=True)
        elif action == ARTDiscovery:  # Discovers host type
            self.observation.update_host(target_host, discovered=True, type=result.target_host.host_type.name)
        elif action == ARTLateralMovement:  # Moves to target host
            self.observation.update_host(target_host, on_host=True)
            self.observation.update_host(src_host, on_host=False)
            self.current_host = result.target_host
        elif action == ARTPrivilegeEscalation:
            self.observation.update_host(target_host, escalated=True)
        elif action == ARTImpact:
            self.observation.update_host(target_host, impacted=True)

    def handle_network_change(self):
        current_hosts = self.network.hosts.keys()
        new_hosts = current_hosts - self.tracked_hosts
        for h in new_hosts:
            host = self.network.hosts[h]
            self.services_map[h] = self.get_valid_techniques_by_host(
                host, self.all_kcps
            )
            self.observation.add_host(h, sweeped=True)
            self.action_space.add_host(h)
        self.tracked_hosts = current_hosts

    def validate_action(self, action: ARTKillChainPhase, target_host: str) -> bool:
        host_view = self.observation.obs[target_host]
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
        _obs = np.array(self.observation.obs_vec, dtype=np.int64)
        #_obs = np.array(obs, dtype=np.float64) # TODO
        return _obs

    def reset(self, entry_host: Host, network: Network):
        self.network = network
        self.current_host = entry_host
        self.observation.reset(entry_host.name)
        self.action_space.reset(entry_host.name)
