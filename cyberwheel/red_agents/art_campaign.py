import yaml
from pathlib import PosixPath
from typing_extensions import Self, Tuple, Type

from cyberwheel.red_agents import ARTAgent
from cyberwheel.network.network_base import Network, Host
from cyberwheel.red_actions.red_base import RedActionResults
from cyberwheel.red_actions.technique import Technique
from cyberwheel.red_agents.strategies import RedStrategy


class ARTCampaign(ARTAgent):
    """
    Class defining an ART Campaign. Where the ART Agent performs logic checking to find valid
    ART Techniques to use on a host, ART Campaigns are defined with ART Techniques in the killchain.
    These are also
    """

    def __init__(
        self,
        name: str,
        entry_host_name: str,
        network: Network,
        killchain: list[tuple[Technique, str]],
        red_strategy: RedStrategy,
    ):
        super().__init__(
            name=name,
            entry_host=network.get_node_from_name(entry_host_name),
            network=network,
            killchain=killchain,
            red_strategy=red_strategy,
            campaign=True,
        )

    def run_action(self, target_host: Host) -> Tuple[RedActionResults, Type[Technique]]:
        step = self.history.hosts[target_host.name].get_next_step()
        if step > len(self.killchain) - 1:
            step = len(self.killchain) - 1

        technique_class = self.killchain[step][0]
        technique = technique_class()
        atomic_test = self.killchain[step][1]
        mitre_id = technique.mitre_id
        technique_name = technique.name

        for at in technique.atomic_tests:
            if atomic_test == at.auto_generated_guid:
                atomic_test = at
                break

        action_results = RedActionResults(self.current_host, target_host)
        action_results.modify_alert(dst=target_host)

        # TODO: Checking if technique will work: OS match, CVE in cve_list, Killchain check
        self.action_results.add_successful_action()

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
        # TODO: Add metadata dependeing on killchain phase
        return action_results, technique_class

    def act(self) -> type[Technique]:
        """
        This defines the red agent's action at each step of the simulation.
        It will
            *   handle any newly added hosts
            *   Select the next target
            *   Run an action on the target
            *   Handle any additional metadata and update history
        """
        self.handle_network_change()

        target_host = self.select_next_target()
        source_host = self.current_host
        action_results, action = self.run_action(target_host)
        action_obj = action()
        success = action_results.attack_success
        if success:
            self.history.hosts[target_host.name].update_killchain_step()
            for h_name in action_results.metadata.keys():
                self.add_host_info(h_name, action_results.metadata[h_name])
            if "impact" in action_obj.kill_chain_phases:  # If KCP was Impact
                self.history.hosts[target_host.name].impacted = True
                if self.history.hosts[target_host.name].type == "Server":
                    self.unimpacted_servers.remove(target_host.name)

        print(f"{action_obj.name} - from {source_host.name} to {target_host.name}")
        self.history.update_step(action, action_results)
        return action

    @classmethod
    def create_campaign_from_yaml(cls, campaign_config: PosixPath) -> Self:
        # Load the YAML config file
        with open(campaign_config, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        entry_host_name = config["entry_host"]
