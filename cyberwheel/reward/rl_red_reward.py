from typing import List, Tuple

from cyberwheel.reward.reward_base import (
    Reward,
    RewardMap,
    RecurringAction,
    calc_quadratic,
)


class RLRedReward(Reward):
    def __init__(
        self,
        red_rewards: RewardMap,
        blue_rewards: RewardMap,
    ) -> None:
        super().__init__(red_rewards, blue_rewards)

    def calculate_reward(
        self,
        red_action: str,
        blue_action: str,
        red_success: str,
        blue_success: bool,
        decoy: bool,
    ) -> int | float:
        if red_success and not decoy:  # If red action succeeded on a real Host
            r = self.red_rewards[red_action][0]
        else:  # If red action did not succeed, or red action was on a decoy
            r = 0

        b = 0  # Hardcoding this for now
        # if blue_success:
        #    b = self.blue_rewards[blue_action][0]
        # else:
        #    b = -100
        #    print("Blue action failed - this shouldn't happen")

        return r + b

    def reset(self) -> None:
        self.blue_recurring_actions = []
        self.red_recurring_actions = []
