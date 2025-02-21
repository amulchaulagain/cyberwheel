from cyberwheel.red_agents.strategies.red_strategy import RedStrategy
from cyberwheel.network.host import Host
from cyberwheel.red_actions.art_techniques import SudoandSudoCaching

"""
The Server Downtime strategy is to find and attack all of the Servers it can find in the network.
Once it finds a server, it will try to impact it. Once impacted, it will look for another server.
"""


class ServerDowntime(RedStrategy):
    @classmethod
    def select_target(cls, agent_obj) -> Host:
        current_host_type = agent_obj.history.hosts[agent_obj.current_host.name].type

        """
        It should continue impacting the current host if: it is Unknown or if it is a Server that has not been impacted yet. Otherwise it should move to another host.
        It should prioritize attacking other Servers that are unimpacted in its view. Then it should prioritize Unknown hosts in its view.
        If there are no unimpacted Servers or Unknown hosts in its view, it has succeeded. Maybe give this a very large cost to signify failure on the blue agent side.
        """

        target_host = agent_obj.current_host
        current_kc_step = agent_obj.history.hosts[target_host.name].get_next_step()
        print(f"CURRENT HOST TYPE: {current_host_type}")
        print(f"CURRENT HOST {target_host.name} KC STEP: {current_kc_step}")
        if (
            current_host_type == "Unknown"
            or agent_obj.unimpacted_servers.check_membership(
                agent_obj.current_host.name
            )
        ):
            target_host = agent_obj.current_host
        elif agent_obj.unimpacted_servers.length() > 0:
            target_host = agent_obj.history.mapping[
                agent_obj.unimpacted_servers.get_random()
            ]  # O(1)
        elif agent_obj.unknowns.length() > 0 and agent_obj.history.hosts[target_host.name].ports_scanned:
            target_host = agent_obj.history.mapping[
                agent_obj.unknowns.get_random()
            ]  # O(1)
        return target_host

    @classmethod
    def get_reward_map(cls) -> dict[str, tuple[int, int]]:
        return {
            "pingsweep": (-1, 0),
            "portscan": (-1, 0),
            "discovery": (-2, 0),
            "lateral-movement": (-4, 0),
            "privilege-escalation": (-6, 0),
            "impact": (-8, -4),
        }
