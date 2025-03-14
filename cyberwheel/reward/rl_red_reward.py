from cyberwheel.reward.reward_base import (
    Reward,
    RewardMap,
    RecurringAction,
)
from cyberwheel.network.host import Host


class RLRedReward(Reward):
    def __init__(
        self,
        red_rewards: RewardMap,
        blue_rewards: RewardMap,
        valid_targets: list[str] | str,
    ) -> None:
        super().__init__(red_rewards, blue_rewards)
        self.valid_targets = valid_targets

    def calculate_reward(
        self,
        red_action: str,
        blue_action: str,
        red_success: str,
        blue_success: bool,
        target_host: Host,
        red_id: str = -1,
        red_recurring: int = 0,
        blue_id: str = -1,
        blue_recurring: int = 0,
    ) -> int | float:
        target_host_name = target_host.name
        decoy = target_host.decoy
        if (
            red_success and not decoy and target_host_name in self.valid_targets
        ):  # If red action succeeded on a real Host
            if red_action == "LinuxLateralMovement":
                r = 0
            else:
                r = self.red_rewards[red_action][0]
        elif red_success and decoy:
            if red_action == "LinuxLateralMovement":
                r = -500
            else:
                r = -500  # -2 * self.red_rewards[red_action][0]
        else:
            r = 0

        if blue_success:
            b = self.blue_rewards[blue_action][0]
        else:
            b = 0  # -100?

        if len(self.blue_recurring_actions) < 1:
            b -= 0

        if red_recurring == -1:
            self.remove_recurring_red_action(red_id)
        elif red_recurring == 1:
            self.add_recurring_red_action(red_id, red_action, decoy)

        if blue_recurring == -1:
            self.remove_recurring_blue_action(blue_id)
        elif blue_recurring == 1:
            self.add_recurring_blue_action(blue_id, blue_action)

        return r + b + self.sum_recurring()

    def sum_recurring(self) -> int | float:
        sum = 0
        for ra in self.blue_recurring_actions:
            sum += self.blue_rewards[ra.action][1]
        for ra in self.red_recurring_actions:
            if ra[1]:
                sum -= self.red_rewards[ra[0].action][1] * 10
            else:
                sum += self.red_rewards[ra[0].action][1]
        return sum

    def add_recurring_blue_action(self, id: str, action: str) -> None:
        self.blue_recurring_actions.append(RecurringAction(id, action))

    def remove_recurring_blue_action(self, id: str) -> None:
        for i in range(len(self.blue_recurring_actions)):
            if self.blue_recurring_actions[i].id == id:
                self.blue_recurring_actions.pop(i)
                break

    def add_recurring_red_action(
        self, id: str, red_action: str, is_decoy: bool
    ) -> None:
        self.red_recurring_actions.append((RecurringAction(id, red_action), is_decoy))

    def remove_recurring_red_action(self, id: str) -> None:
        for i in range(len(self.red_recurring_actions)):
            if self.red_recurring_actions[i].id == id:
                self.red_recurring_actions.pop(i)
                break

    def reset(self) -> None:
        self.blue_recurring_actions = []
        self.red_recurring_actions = []
