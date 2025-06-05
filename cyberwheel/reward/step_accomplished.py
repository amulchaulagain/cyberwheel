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
        
        # TODO: remove
        self.run_num = 0

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
        valid_targets = self.get_valid_targets()
        step = self.red_agent.history.step
        #if self.valid_targets == "servers":
        #    accomplished = len(self.red_agent.unknowns) == 0 and len(self.red_agent.unimpacted_servers) == 0 and not self.finished
        #elif self.valid_targets
        #    accomplished = target_host.name == "server6"
        #print(valid_targets.data_list)
        agent_status = self.red_agent.history.hosts
        #uih = self.red_agent.unimpacted_hosts
        #uis = self.red_agent.unimpacted_servers
        #unh = self.red_agent.unknowns
        #print(f"Hosts:\t{uih.data_list}\nServers:\t{uis.data_list}\nUnknowns:{unh.data_list}")
        #print(valid_targets.data_list)
        #accomplished = False
        #for t in valid_targets:
        #    accomplished = agent_status[t].impacted 
        #accomplished = not any([t in uih or t in uis or t in unh or len(unh) > 0 for t in valid_targets])
        #print(valid_targets)
        accomplished = not any([t not in agent_status or not agent_status[t].impacted for t in valid_targets])
        #print(accomplished)
        #import time
        #time.sleep(5)
        k = 0.01
        #if target_host.decoy:
        #    print(f"ran {red_action} on {target_host.name}")
        #elif "server" in target_host.host_type.name and red_action == "impact":
        #    print(f"impacted {target_host.name}")

        #if self.run_num == 14 or True:
            #print(f"Run {self.run_num}\t-\t{red_action}\t->\t{target_host.name}")
        #    print(f"Env episode {self.run_num} - Step {step}")
        if accomplished and not self.finished:
            self.finished = True
            reward = step #math.exp(k * step)
            #print("accomplished")
            #print(f"RL Agent receives its reward here at step {step}")
            #print(f"Run {self.run_num}: Red agent finished its job at step {step}")
            #if reward == 0:
            #    print("finished at 0 here?")
            #    print(f"{red_action} on {target_host}, \nunknowns: {self.red_agent.unknowns.data_list}, \nunimpacted_servers: {self.red_agent.unimpacted_servers.data_list}")
            #print(step)
            return reward
        elif not self.finished and step in [self.red_agent.args.num_steps - 1, self.red_agent.args.num_steps]:
            #print("This is happening")
            return step * 2 #math.exp(k * self.red_agent.args.num_steps) * 2
        else:    
            return 0

    def reset(
        self,
    ) -> None:
        self.finished = False
        self.run_num += 1