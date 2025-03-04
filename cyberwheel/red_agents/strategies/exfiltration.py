from cyberwheel.red_agents.strategies.red_strategy import RedStrategy
from cyberwheel.network.host import Host


class Exfiltration(RedStrategy):
    """
    The Server Downtime strategy is to find and attack all of the Servers it can find in the network.
    Once it finds a server, it will try to impact it. Once impacted, it will look for another server.
    """
    @classmethod
    def select_target(cls, agent_obj) -> Host:
        current_host_type = agent_obj.history.hosts[agent_obj.current_host.name].type
        """
        It should continue impacting the current host if: it is Unknown or if it is the Target. Otherwise it should move to another host.
        It should prioritize attacking other Servers that are unimpacted in its view. Then it should prioritize Unknown hosts in its view.
        """

        target_host = agent_obj.current_host
        if (
            current_host_type == "Unknown"
            or agent_obj.history.hosts[agent_obj.current_host.name].is_leader
        ):
            target_host = agent_obj.current_host
        elif agent_obj.unknowns.length() > 0:
            target_host = agent_obj.network.hosts[agent_obj.unknowns.get_random()]
        return target_host

    @classmethod
    def get_reward_map(cls) -> dict[str, tuple[int, int]]:
        return {
            "pingsweep": (-1, 0),
            "portscan": (-1, 0),
            "discovery": (-2, 0),
            "lateral-movement": (-4, 0),
            "privilege-escalation": (-20, 0),
            "impact": (-40, -4),
        }
