from cyberwheel.red_agents.strategies.red_strategy import RedStrategy


class BruteForce(RedStrategy):
    """
    The Brute Force strategy is to attack the same host over and over.
    """
    @classmethod
    def select_target(cls, agent_obj):
        """
        Attack the host it's already on.
        """
        return agent_obj.current_host

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
