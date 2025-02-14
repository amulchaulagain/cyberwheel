from cyberwheel.red_agents.strategies.red_strategy import RedStrategy
from cyberwheel.network.host import Host

"""
The Server Downtime strategy is to find and attack all of the Servers it can find in the network.
Once it finds a server, it will try to impact it. Once impacted, it will look for another server.
"""


class Exfiltration(RedStrategy):
    @classmethod
    def select_target(cls, agent_obj) -> Host:
        current_host_type = agent_obj.history.hosts[agent_obj.current_host.name].type
        """
        It should continue impacting the current host if: it is Unknown or if it is the Target. Otherwise it should move to another host.
        It should prioritize attacking other Servers that are unimpacted in its view. Then it should prioritize Unknown hosts in its view.
        """

        target_host = agent_obj.current_host
        # print(agent_obj.history.hosts)
        # print(agent_obj.history.hosts[agent_obj.current_host.name].is_leader)

        # If current host is Unknown, or if it is known AND leader, keep attacking
        if (
            agent_obj.history.hosts[agent_obj.current_host.name].is_leader
            or agent_obj.history.hosts[agent_obj.current_host.name].type == "Unknown"
        ):
            target_host = agent_obj.current_host
        # If Leader is in view, and the type is known, it is the target (switch to it)
        elif (
            agent_obj.leader.name in agent_obj.history.hosts
            and agent_obj.history.hosts[agent_obj.leader.name].is_leader
        ):
            target_host = agent_obj.leader
        # If there are any unknown hosts attack them
        elif agent_obj.unknowns.length() > 0:
            target_host = agent_obj.history.mapping[agent_obj.unknowns.get_random()]
        # print(agent_obj.unknowns.data_list)
        # print(agent_obj.leader.name)
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
