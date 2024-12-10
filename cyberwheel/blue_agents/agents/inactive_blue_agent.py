from cyberwheel.blue_agents.blue_agent import BlueAgent
from cyberwheel.reward import RewardMap


class InactiveBlueAgent(BlueAgent):
    """
    This agent does nothing.
    """

    def __init__(self) -> None:
        super().__init__()

    def act(self) -> str:
        return "nothing"

    def get_reward_map(self) -> RewardMap:
        return {"nothing": (0, 0)}

    def reset(self):
        return
