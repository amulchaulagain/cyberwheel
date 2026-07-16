
import gymnasium as gym
import time
import importlib
import pandas as pd
import torch
import wandb
import os
import random
import yaml

from importlib.resources import files
from tqdm import tqdm

from cyberwheel.network.network_base import Network
from cyberwheel.utils import RLPolicy, get_service_map
from cyberwheel.runners.rl_trainer import RLTrainer
from cyberwheel.visualization import VizWriter
from cyberwheel.utils.set_seed import set_seed
from cyberwheel.utils.step_metrics import build_evaluation_summary, write_summary


class RLEvaluator(RLTrainer):
    def __init__(self, args):
        super().__init__(args)
        if self.args.download_model:
            self.api = wandb.Api()
            self.run = self.api.run(
                f"{self.args.wandb_entity}/{self.args.wandb_project_name}/runs/{self.args.run}"
            )

    def configure_evaluation(self):
        if self.args.deterministic:
            self.applied_seed = self.seed
            torch.backends.cudnn.deterministic = True
        else:
            self.applied_seed = random.randint(0, 999999999)
            torch.backends.cudnn.deterministic = False
        set_seed(self.applied_seed)
        self.eval_seeds, self.explicit_seeds = self._resolve_seeds()

        self.device = torch.device("cpu")
        print(f"Using device {self.device}")

        # Load networks from yaml here
        network_configs = []
        if isinstance(self.args.network_config, str):
            network_configs.append(self.args.network_config)
        else:
            for config in self.args.network_config:
                network_configs.append(config)
        
        self.networks = {}
        self.args.service_mapping = {}
        for config in network_configs:
            network_config = files("cyberwheel.data.configs.network").joinpath(
                config
            )

            print(f"Building network: {config} ...")

            network = Network.create_network_from_yaml(network_config)
            network_name = network.name
            self.networks[network_name] = [network]

            print("Mapping attack validity to hosts...", end=" ")
            self.args.service_mapping[network_name] = get_service_map(network)
            print("done")
        
        self.args.agent_config = {}
        #print(self.networks)

        for agent_type in self.args.agents:
            self.args.agent_config[agent_type] = {}
            agent_yaml = self.args.agents[agent_type]
            agent_config = files(f"cyberwheel.data.configs.{agent_type}_agent").joinpath(agent_yaml)
            with open(agent_config, "r") as yaml_file:
                self.args.agent_config[agent_type] = yaml.safe_load(yaml_file)
            if self.args.agent_config[agent_type]["rl"]:
                self.agents[agent_type] = None

        self.env = self.make_env(0, evaluation=True, net_name=list(self.networks.keys())[0])()
        self.policy = {}

    def _resolve_seeds(self):
        """Resolve the optional ``seeds`` config key to (seed list, explicit flag).

        Absent/empty means single-seed with today's semantics; the recorded seed
        is the one actually applied (a random draw when non-deterministic).
        """
        raw = getattr(self.args, "seeds", None)
        if raw in (None, "", []):
            return [self.applied_seed], False
        if isinstance(raw, int):
            return [raw], True
        if isinstance(raw, str):
            return [int(p) for p in raw.split(",") if p.strip()], True
        return [int(s) for s in raw], True

    def load_models(self):
        for agent in self.agents:
            self.policy[agent] = RLPolicy(self.agents[agent]["max_action_space_size"], self.agents[agent]["obs"].shape).to(self.device)
            agent_filename = f"{agent}_{self.args.checkpoint}.pt"

            # If download from W&B, use API to get run data.
            if self.args.download_model:
                model = self.run.file(agent_filename)
                model.download(
                    files("cyberwheel.data.models").joinpath(self.args.experiment_name), exist_ok=True
                )

            # Load model from models/ directory
            self.policy[agent].load_state_dict(
                torch.load(
                    files(f"cyberwheel.data.models.{self.args.experiment_name}").joinpath(agent_filename),
                    map_location=self.device,
                )
            )
            self.policy[agent].eval()

    def _initialize_environment(self):
        print("Resetting the environment...")

        self.episode_rewards = []
        self.total_reward = 0
        self.steps = 0
        self.obs = self.env.reset()

        print("Playing environment...")

        # Set up dirpath to store action logs CSV
        if self.args.graph_name != None:
            self.now_str = self.args.graph_name
        else:
            network_config = (
                self.args.network_config
                if isinstance(self.args.network_config, str)
                else self.args.network_config[0]
            )
            # Modern eval configs define agents as a red/blue map; legacy ones
            # use a flat red_agent key. Accept either for the fallback name.
            red_agent = getattr(self.args, "red_agent", None) or getattr(
                self.args, "agents", {}
            ).get("red", "red")
            self.now_str = f"{self.args.experiment_name}_evaluate_{network_config.split('.')[0]}_{red_agent.split('.')[0]}_{self.args.reward_function}reward"
        self.log_file = files("cyberwheel.data.action_logs").joinpath(f"{self.now_str}.csv")
        self.summary_file = files("cyberwheel.data.action_logs").joinpath(f"{self.now_str}.summary.json")

        self.viz = None
        if getattr(self.args, "visualize", False):
            self.viz = VizWriter(
                self.env,
                files("cyberwheel.data.graphs").joinpath(self.now_str),
                meta={
                    "experiment_name": self.args.experiment_name,
                    "graph_name": self.now_str,
                    "network_config": self.args.network_config,
                    "agents": getattr(self.args, "agents", None),
                    "num_episodes": self.args.num_episodes,
                    "num_steps": self.args.num_steps,
                    "seed": getattr(self.args, "seed", None),
                    "seeds": self.eval_seeds,
                },
            )

        self.actions_df = pd.DataFrame()
        data = {
                "episode": [],
                "step": [],
        }
        self.action_mask = {}
        self.rewards = {}
        
        """
        for agent in self.agents[agent]:
            data[agent]["action_name"] = []
            data[agent]["action_src"] = []
            data[agent]["action_dest"] = []
            data[agent]["action_success"] = []
            data[agent]["reward"] = []
        """

        with open(self.log_file, 'w'): # Create an empty CSV for new action logs, overwrite previous
            pass
    
    def mask_actions(self, new_action_mask, action_mask):
        new_mask = torch.tensor(
            new_action_mask,
            dtype=torch.bool,
            device=action_mask.device,
        )
        return new_mask

    def evaluate(self):
        for agent in self.agents:
            self.action_mask[agent] = torch.zeros(self.agents[agent]["max_action_space_size"], dtype=torch.bool).to(self.device)
            self.rewards[agent] = [0] * self.args.num_episodes

        reward_metrics = ["total_reward"] + [f"{agent}_reward" for agent in self.agents]
        # Per-step counters summed into per-episode totals. All zeros when the
        # run has no green agent / no host-isolating blue action, so the
        # summary schema is uniform across scenarios.
        count_metrics = [
            "green_events", "green_blocked",
            "blue_alerts", "blue_false_alerts",
            "blue_isolations", "blue_hostile_isolations",
        ]
        # Summary stat blocks: rewards, activity counts, and the two derived
        # precision ratios (episodes with an empty denominator contribute
        # nothing to a ratio's stats — see build_evaluation_summary).
        metric_names = reward_metrics + [
            "green_events", "green_blocked",
            "blue_alerts", "alert_precision",
            "blue_isolations", "blue_precision",
        ]
        per_episode = []
        global_episode = 0

        self.start_time = time.time()
        for seed in self.eval_seeds:
            for episode in range(self.args.num_episodes):
                # Explicit seeds reseed each block regardless of `deterministic`,
                # making every seed's episodes reproducible on their own.
                if self.explicit_seeds and episode == 0:
                    obs, _ = self.env.reset(seed=seed)
                else:
                    obs, _ = self.env.reset()
                if self.viz:
                    self.viz.start_episode(global_episode)
                episode_totals = {m: 0.0 for m in reward_metrics}
                episode_counts = {m: 0 for m in count_metrics}
                for step in range(self.args.num_steps):
                    action = None
                    actions = {}
                    action_masks = self.env.action_mask

                    for agent in self.agents:
                        agent_obs = torch.Tensor(obs[agent]).to(self.device)
                        tmp_mask = action_masks[agent]
                        self.action_mask[agent] = self.mask_actions(tmp_mask, self.action_mask[agent])
                        action, _, _, _ = self.policy[agent].get_action_and_value(agent_obs, action_mask=self.action_mask[agent])
                        actions[agent] = action

                    obs, rew, done, _, info = self.env.step(actions)

                    if self.viz:
                        self.viz.record_step(global_episode, step, info)

                    actions_df = {
                        "episode": global_episode,
                        "step": step,
                        "seed": seed,
                        "reward": rew,
                    }
                    self.total_reward += rew
                    episode_totals["total_reward"] += float(rew)
                    for agent in self.agents:
                        actions_df[f"{agent}_action_name"] = [info[f"{agent}_action"]]
                        actions_df[f"{agent}_action_success"] = [info[f"{agent}_action_success"]]
                        actions_df[f"{agent}_action_src"] = [info[f"{agent}_action_src"]]
                        actions_df[f"{agent}_action_dest"] = [info[f"{agent}_action_dst"]]
                        actions_df[f"{agent}_reward"] = [info[f"{agent}_reward"]]
                        episode_totals[f"{agent}_reward"] += float(info[f"{agent}_reward"])
                    # Green (benign user) activity + blue precision counters;
                    # all zeros when the scenario doesn't produce them.
                    for m in count_metrics:
                        value = int(info.get(m, 0))
                        episode_counts[m] += value
                        actions_df[m] = [value]

                    actions_df = pd.DataFrame(actions_df)
                    actions_df.to_csv(self.log_file, mode='a', header = os.path.getsize(self.log_file) == 0, index=False)

                if self.viz:
                    self.viz.end_episode()
                # Precision ratios: of what the detector surfaced, how much was
                # really red (alert_precision); of the hosts blue quarantined,
                # how many red had actually gained execution on (blue_precision).
                # None (not 0) when the episode had no alerts / no quarantines.
                alerts = episode_counts["blue_alerts"]
                isolations = episode_counts["blue_isolations"]
                per_episode.append({
                    "episode": global_episode,
                    "seed": seed,
                    "steps": self.args.num_steps,
                    **{m: round(v, 4) for m, v in episode_totals.items()},
                    **episode_counts,
                    "alert_precision": round(
                        (alerts - episode_counts["blue_false_alerts"]) / alerts, 4
                    ) if alerts else None,
                    "blue_precision": round(
                        episode_counts["blue_hostile_isolations"] / isolations, 4
                    ) if isolations else None,
                })
                global_episode += 1

        write_summary(self.summary_file, build_evaluation_summary(
            seeds=self.eval_seeds,
            explicit_seeds=self.explicit_seeds,
            deterministic=bool(self.args.deterministic),
            num_episodes=self.args.num_episodes,
            num_steps=self.args.num_steps,
            per_episode=per_episode,
            metric_names=metric_names,
            graph_name=self.now_str,
            experiment_name=self.args.experiment_name,
        ))

"""
    def evaluate(self):
        
        for episode in tqdm(range(self.args.num_episodes)):
            obs, _ = self.env.reset()
            for step in range(self.args.num_steps):
                self.blue_obs = torch.Tensor(obs["blue"]).to(self.device)
                self.red_obs = torch.Tensor(obs["red"]).to(self.device)

                tmp_blue_mask = self.env.blue_action_mask
                tmp_red_mask = self.env.red_action_mask

                self.blue_action_mask = self.mask_actions(tmp_blue_mask, self.blue_action_mask)
                self.red_action_mask = self.mask_actions(tmp_red_mask, self.red_action_mask)

                blue_action, _, _, _ = self.blue_agent.get_action_and_value(
                    self.blue_obs, action_mask=self.blue_action_mask
                )
                red_action, _, _, _ = self.red_agent.get_action_and_value(
                    self.red_obs, action_mask=self.red_action_mask
                )

                action = {"blue": blue_action, "red": red_action}

                obs, rew, done, _, info = self.env.step(action)

                blue_reward = info["blue_reward"]
                red_reward = info["red_reward"]

                blue_action = info["blue_action"]
                red_action_type = info["red_action"]
                red_action_src = info["red_action_src"]
                red_action_dest = info["red_action_dst"]
                red_action_success = info["red_action_success"]
                blue_action_success = info["blue_action_success"]
                self.red_obs = obs["red"]
                self.blue_obs = obs["blue"]

                actions_df = pd.DataFrame(
                {
                    "episode": [episode],
                    "step": [step],
                    "red_action_success": [red_action_success],
                    "red_action_type": [red_action_type],
                    "red_action_src": [red_action_src],
                    "red_action_dest": [red_action_dest],
                    "blue_action": [blue_action],
                    "blue_success": [blue_action_success],
                    "blue_reward": [blue_reward],
                    "red_reward": [red_reward],
                    "reward": [rew],
                })
                actions_df.to_csv(self.log_file, mode='a', header = os.path.getsize(self.log_file) == 0, index=False)

                # If generating graphs for dash server view
                if self.args.visualize:
                    # visualize(net, episode, step, now_str, history, killchain)
                    pass

                self.total_reward += rew
                self.steps += 1

            self.steps = 0
            self.episode_rewards.append(self.total_reward)
            self.total_reward = 0

        # Save action metadata to CSV in action_logs/

        self.total_time = time.time() - self.start_time
        print("charts/SPS", int(2000 / self.total_time))
        #self.total_reward = sum(self.episode_rewards)
        #self.episodes = len(self.episode_rewards)
        print(f"Total Time Elapsed: {self.total_time}")
"""