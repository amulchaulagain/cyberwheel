from cyberwheel.green_agents.green_agent_base import GreenAgent, GreenAgentResult


class InactiveGreenAgent(GreenAgent):
    """Default green agent when no ``agents: green:`` key is configured.

    Emits nothing and draws nothing from the RNG stream, so green-less runs
    stay byte-identical to pre-green behavior.
    """

    def __init__(self, network=None, args=None) -> None:
        pass

    def act(self) -> GreenAgentResult:
        return GreenAgentResult()

    def reset(self, network=None) -> None:
        return
