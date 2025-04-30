from cyberwheel.reward.reward_base import RewardMap
from cyberwheel.reward.rl_reward import RLReward
from cyberwheel.network.network_base import Network

import math

class StepAccomplishedReward(RLReward):
    def __init__(
        self,
        red_agent,
        blue_agent,
        valid_targets: list[str] | str,
        network: Network,
    ) -> None:
        """
        Reward is maximized if red agent is detected early by blue agent. The best reward it can get is
        one in which the blue agent immediately detects the red agent's actions. The worst reward it can get
        is one in which the blue agent detects the red agent at the final step of the episode.

        Reward Function: max_steps / n, where n is the number of steps

        TODO: Needs testing with recent reward changes.
        """
        super().__init__(
            red_agent=red_agent,
            blue_agent=blue_agent,
            valid_targets=valid_targets,
            network=network
        )
        self.finished = False

    def calculate_reward(
        self,
        red_action: str,
        blue_action: str,
        red_success: str,
        blue_success: bool,
        target_host,
        blue_id: str = -1,
        blue_recurring: int = 0,
    ) -> int | float:
        step = self.red_agent.history.step
        accomplished = len(self.red_agent.unknowns) == 0 and len(self.red_agent.unimpacted_servers) == 0 and not self.finished
        if accomplished:
            self.finished = True
            k = 0.01
            reward = 100 * math.exp(-k * step)

            return reward
        else:
            return 0

    def reset(
        self,
    ) -> None:
        self.finished = False