from torch import nn, optim

from cyberwheel.utils import RLPolicy
from gymnasium.vector import VectorEnv, AsyncVectorEnv
from gymnasium import spaces
from importlib.resources import files

import numpy as np
import torch
import os

class MultiAgentHandler:

    def __init__(self, envs: VectorEnv, args, blue_max_action_space_size, red_max_action_space_size, blue_max_obs_space_size, red_max_obs_space_size, blue_max_attrs, red_max_attrs):
        self.envs = envs
        self.args = args
        # Use a GPU if available. You can choose a specific GPU with CUDA, for example by setting 'device' to "cuda:0"
        self.device = self.args.device
        print(f"Using device '{self.device}'")

        self.red_obs = spaces.Box(
                low  = np.full(red_max_obs_space_size, -1, dtype=np.int32),
                high = np.full(red_max_obs_space_size,  red_max_attrs, dtype=np.int32),
                dtype=np.int32)
        self.blue_obs = spaces.Box(
                low  = np.full(blue_max_obs_space_size, -1, dtype=np.int32),
                high = np.full(blue_max_obs_space_size, blue_max_attrs, dtype=np.int32),
                dtype=np.int32)

        self.og_blue_shape = self.blue_obs.shape
        self.og_red_shape = self.red_obs.shape
        
        self.red_max_action_space_size = red_max_action_space_size
        self.blue_max_action_space_size = blue_max_action_space_size

        self.red_policy = RLPolicy(self.red_max_action_space_size, self.red_obs.shape).to(self.device)
        self.blue_policy = RLPolicy(self.blue_max_action_space_size, self.blue_obs.shape).to(self.device)

        red_actor_params = list(self.red_policy.actor.parameters())
        red_critic_params = list(self.red_policy.critic.parameters())

        blue_actor_params = list(self.blue_policy.actor.parameters())
        blue_critic_params = list(self.blue_policy.critic.parameters())

        self.red_optimizer = optim.Adam([
            { 'params': red_actor_params,  'lr': float(self.args.actor_lr),  'eps': 1e-5 },
            { 'params': red_critic_params, 'lr': float(self.args.critic_lr), 'eps': 1e-5 },
        ])
        self.blue_optimizer = optim.Adam([
            { 'params': blue_actor_params,  'lr': float(self.args.actor_lr),  'eps': 1e-5 },
            { 'params': blue_critic_params, 'lr': float(self.args.critic_lr), 'eps': 1e-5 },
        ])

        self.red_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.red_optimizer, T_0=int(self.args.restart_T0), T_mult=int(self.args.restart_Tmult), eta_min=float(self.args.min_lr), last_epoch=-1) if self.args.anneal_lr == 'cosine_restarts' else None
        self.blue_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.blue_optimizer, T_0=int(self.args.restart_T0), T_mult=int(self.args.restart_Tmult), eta_min=float(self.args.min_lr), last_epoch=-1) if self.args.anneal_lr == 'cosine_restarts' else None

    def define_multiagent_variables(self):
        self.blue_obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.blue_obs.shape).to(self.device)
        self.red_obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.red_obs.shape).to(self.device)

        self.blue_actions = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.red_actions = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        self.blue_logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.red_logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        self.blue_rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.red_rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        self.blue_values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.red_values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

        self.blue_action_masks = torch.zeros((self.args.num_steps, self.args.num_envs, self.blue_max_action_space_size), dtype=torch.bool).to(self.device)
        self.red_action_masks = torch.zeros((self.args.num_steps, self.args.num_envs, self.red_max_action_space_size), dtype=torch.bool).to(self.device)

        reset = self.envs.reset(seed=[i for i in range(self.args.num_envs)])[0]
        self.blue_resets = np.array(reset["blue"]) # TODO: Need to update with determinism
        self.red_resets = np.array(reset["red"])

        self.blue_next_obs = torch.Tensor(self.blue_resets).to(self.device)
        self.red_next_obs = torch.Tensor(self.red_resets).to(self.device) # TODO: get from env somehow

        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.next_done = torch.zeros(self.args.num_envs).to(self.device)

        self.global_step = 0

    def mask_actions(self, new_action_mask, action_mask):
        new_mask = torch.tensor(
            new_action_mask,
            dtype=torch.bool,
            device=action_mask.device,
        )
        return new_mask
    
    def update_action_masks(self, step: int):
        if self.args.async_env: #isinstance(self.envs, gym.vector.AsyncVectorEnv):
            blue_masks = self.envs.call("blue_action_mask")
            red_masks = self.envs.call("red_action_mask")
        else:
            blue_masks = [env.unwrapped.blue_action_mask for env in self.envs.envs]
            red_masks = [env.unwrapped.red_action_mask for env in self.envs.envs]
        
        for i in range(self.args.num_envs):
            self.blue_action_masks[step][i] = self.mask_actions(blue_masks[i], self.blue_action_masks[step][i])
            self.red_action_masks[step][i] = self.mask_actions(red_masks[i], self.red_action_masks[step][i])
        
    def step_multiagent(self, step: int):
        self.global_step += self.args.num_envs
        self.blue_obs[step] = self.blue_next_obs
        self.red_obs[step] = self.red_next_obs
        self.dones[step] = self.next_done

        with torch.no_grad():
            blue_action, blue_logprob, _, blue_value = self.blue_policy.get_action_and_value(self.blue_next_obs, action_mask=self.blue_action_masks[step])
            red_action, red_logprob, _, red_value = self.red_policy.get_action_and_value(self.red_next_obs, action_mask=self.red_action_masks[step])

            self.blue_values[step] = blue_value.flatten()
            self.red_values[step] = red_value.flatten()
            self.blue_actions[step] = blue_action
            self.red_actions[step] = red_action
            self.blue_logprobs[step] = blue_logprob
            self.red_logprobs[step] = red_logprob

        # Execute the selected action in the environment to collect experience for training.
        blue_policy_action = blue_action.cpu().numpy()
        red_policy_action = red_action.cpu().numpy()

        policy_action = {"blue": blue_policy_action, "red": red_policy_action}

        obs, reward, done, _, info = self.envs.step(policy_action)

        self.blue_next_obs = obs["blue"]
        self.red_next_obs = obs["red"]

        self.blue_rewards[step] = torch.tensor(info["blue_reward"]).to(self.device).view(-1)
        self.red_rewards[step] = torch.tensor(info["red_reward"]).to(self.device).view(-1)

        self.blue_next_obs = torch.Tensor(self.blue_next_obs).to(self.device)
        self.red_next_obs = torch.Tensor(self.red_next_obs).to(self.device)

        self.next_done = torch.Tensor(done).to(self.device)
    
    def log_stuff(self, writer, episodic_runtime, episodic_processing_time):
        blue_mean_rew = self.blue_rewards.sum(axis=0).mean()
        red_mean_rew = self.red_rewards.sum(axis=0).mean()
        print(f"global_step={self.global_step}, blue_episodic_return={blue_mean_rew}, red_episodic_return={red_mean_rew}")

        writer.add_scalar(f"charts/episodic_runtime", episodic_runtime, self.global_step)
        writer.add_scalar(f"charts/episodic_process_time", episodic_processing_time, self.global_step)

        writer.add_scalar("charts/blue_episodic_return", blue_mean_rew, self.global_step)
        writer.add_scalar("charts/red_episodic_return", red_mean_rew, self.global_step)
    
    def compute_gae(self):
        with torch.no_grad():
            blue_next_value = self.blue_policy.get_value(self.blue_next_obs).reshape(1, -1)
            red_next_value = self.red_policy.get_value(self.red_next_obs).reshape(1, -1)

            self.blue_advantages = torch.zeros_like(self.blue_rewards).to(self.device)
            self.red_advantages = torch.zeros_like(self.red_rewards).to(self.device)

            blue_lastgaelam = 0
            red_lastgaelam = 0

            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_done
                    blue_nextvalues = blue_next_value
                    red_nextvalues = red_next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    blue_nextvalues = self.blue_values[t + 1]
                    red_nextvalues = self.red_values[t + 1]
                
                blue_delta = self.blue_rewards[t] + self.args.gamma * blue_nextvalues * nextnonterminal - self.blue_values[t]
                red_delta = self.red_rewards[t] + self.args.gamma * red_nextvalues * nextnonterminal - self.red_values[t]

                self.blue_advantages[t] = blue_lastgaelam = blue_delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * blue_lastgaelam
                self.red_advantages[t] = red_lastgaelam = red_delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * red_lastgaelam
            self.blue_returns = self.blue_advantages + self.blue_values
            self.red_returns = self.red_advantages + self.red_values

    def flatten_batch(self):
        #print(self.blue_obs.shape)
        #obs_dims = self.blue_obs.shape[2:]
        self.batched_blue_obs = self.blue_obs.reshape((-1,) + self.og_blue_shape)
        self.batched_blue_logprobs = self.blue_logprobs.reshape(-1)
        self.batched_blue_actions = self.blue_actions.reshape(-1)
        self.batched_blue_advantages = self.blue_advantages.reshape(-1)
        self.batched_blue_returns = self.blue_returns.reshape(-1)
        self.batched_blue_values = self.blue_values.reshape(-1)
        self.batched_blue_action_masks = self.blue_action_masks.reshape(-1, self.blue_action_masks.shape[-1])

        self.batched_red_obs = self.red_obs.reshape((-1,) + self.og_red_shape)
        self.batched_red_logprobs = self.red_logprobs.reshape(-1)
        self.batched_red_actions = self.red_actions.reshape(-1)
        self.batched_red_advantages = self.red_advantages.reshape(-1)
        self.batched_red_returns = self.red_returns.reshape(-1)
        self.batched_red_values = self.red_values.reshape(-1)
        self.batched_red_action_masks = self.red_action_masks.reshape(-1, self.red_action_masks.shape[-1])

        self.blue_clipfracs = []
        self.red_clipfracs = []
    
    def update_blue_policy(self, mb_inds):
        _, blue_newlogprob, self.blue_entropy, self.blue_newvalue = self.blue_policy.get_action_and_value(
            self.batched_blue_obs[mb_inds],
            self.batched_blue_actions.long()[mb_inds],
            action_mask=self.batched_blue_action_masks[mb_inds],
        )
        blue_logratio = blue_newlogprob - self.batched_blue_logprobs[mb_inds]
        self.blue_ratio = blue_logratio.exp()

        # Calculate the difference between the old policy and the new policy to limit the size of the update using args.clip_coef.
        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            self.blue_old_approx_kl = (-blue_logratio).mean()
            self.blue_approx_kl = ((self.blue_ratio - 1) - blue_logratio).mean()
            self.blue_clipfracs += [
                ((self.blue_ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()
            ]

        self.blue_mb_advantages = self.batched_blue_advantages[mb_inds]
        if self.args.norm_adv:
            self.blue_mb_advantages = (self.blue_mb_advantages - self.blue_mb_advantages.mean()) / (self.blue_mb_advantages.std() + 1e-8)

    def update_red_policy(self, mb_inds):
        _, red_newlogprob, self.red_entropy, self.red_newvalue = self.red_policy.get_action_and_value(
            self.batched_red_obs[mb_inds],
            self.batched_red_actions.long()[mb_inds],
            action_mask=self.batched_red_action_masks[mb_inds],
        )
        red_logratio = red_newlogprob - self.batched_red_logprobs[mb_inds]
        self.red_ratio = red_logratio.exp()

        # Calculate the difference between the old policy and the new policy to limit the size of the update using args.clip_coef.
        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            self.red_old_approx_kl = (-red_logratio).mean()
            self.red_approx_kl = ((self.red_ratio - 1) - red_logratio).mean()
            self.red_clipfracs += [
                ((self.red_ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()
            ]

        self.red_mb_advantages = self.batched_red_advantages[mb_inds]
        if self.args.norm_adv:
            self.red_mb_advantages = (self.red_mb_advantages - self.red_mb_advantages.mean()) / (self.red_mb_advantages.std() + 1e-8)
    
    def calculate_blue_loss(self, mb_inds):
        # Policy loss using PPO's ration clipping
        pg_loss1 = -self.blue_mb_advantages * self.blue_ratio
        pg_loss2 = -self.blue_mb_advantages * torch.clamp(
            self.blue_ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
        )
        self.blue_policy_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = self.blue_newvalue.view(-1)
        # Calculate the MSE loss between the returns and the value predictions of the critic
        # Clipping V loss is often not necessary and arguably worse in practice
        if self.args.clip_vloss:
            v_loss_unclipped = (newvalue - self.batched_blue_returns[mb_inds]) ** 2
            v_clipped = self.batched_blue_values[mb_inds] + torch.clamp(
                newvalue - self.batched_blue_values[mb_inds],
                -self.args.clip_coef,
                self.args.clip_coef,
            )
            v_loss_clipped = (v_clipped - self.batched_blue_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            self.blue_value_loss = 0.5 * v_loss_max.mean()
        else:
            self.blue_value_loss = 0.5 * ((newvalue - self.batched_blue_returns[mb_inds]) ** 2).mean()

        # Add an entropy bonus to the loss
        self.blue_entropy_loss = self.blue_entropy.mean()
        self.blue_loss = self.blue_policy_loss - self.args.ent_coef * self.blue_entropy_loss + self.blue_value_loss * self.args.vf_coef
    
    def calculate_red_loss(self, mb_inds):
        # Policy loss using PPO's ration clipping
        pg_loss1 = -self.red_mb_advantages * self.red_ratio
        pg_loss2 = -self.red_mb_advantages * torch.clamp(
            self.red_ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef
        )
        self.red_policy_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = self.red_newvalue.view(-1)
        # Calculate the MSE loss between the returns and the value predictions of the critic
        # Clipping V loss is often not necessary and arguably worse in practice
        if self.args.clip_vloss:
            v_loss_unclipped = (newvalue - self.batched_red_returns[mb_inds]) ** 2
            v_clipped = self.batched_red_values[mb_inds] + torch.clamp(
                newvalue - self.batched_red_values[mb_inds],
                -self.args.clip_coef,
                self.args.clip_coef,
            )
            v_loss_clipped = (v_clipped - self.batched_red_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            self.red_value_loss = 0.5 * v_loss_max.mean()
        else:
            self.red_value_loss = 0.5 * ((newvalue - self.batched_red_returns[mb_inds]) ** 2).mean()

        # Add an entropy bonus to the loss
        self.red_entropy_loss = self.red_entropy.mean()
        self.red_loss = self.red_policy_loss - self.args.ent_coef * self.red_entropy_loss + self.red_value_loss * self.args.vf_coef

    def backpropagate(self, update):
        # Backpropagation
        self.blue_optimizer.zero_grad()
        self.blue_loss.backward()
        nn.utils.clip_grad_norm_(self.blue_policy.parameters(), self.args.max_grad_norm)
        self.blue_optimizer.step()

        self.red_optimizer.zero_grad()
        self.red_loss.backward()
        nn.utils.clip_grad_norm_(self.red_policy.parameters(), self.args.max_grad_norm)
        self.red_optimizer.step()

        if self.args.anneal_lr == 'cosine_restarts':
            self.blue_scheduler.step(update)
            self.red_scheduler.step(update)
    
    def calculate_explained_variance(self):
        blue_pred, blue_true = self.batched_blue_values.cpu().numpy(), self.batched_blue_returns.cpu().numpy()
        blue_var = np.var(blue_true)
        self.blue_explained_variance = np.nan if blue_var == 0 else 1 - np.var(blue_true - blue_pred) / blue_var

        red_pred, red_true = self.batched_red_values.cpu().numpy(), self.batched_red_returns.cpu().numpy()
        red_var = np.var(red_true)
        self.red_explained_variance = np.nan if red_var == 0 else 1 - np.var(red_true - red_pred) / red_var
    
    def save_models(self):
        run_path = files("cyberwheel.data.models").joinpath(self.args.experiment_name)
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        blue_agent_path = run_path.joinpath("blue_agent.pt")
        blue_globalstep_path = run_path.joinpath(f"blue_{self.global_step}.pt")
        red_agent_path = run_path.joinpath("red_agent.pt")
        red_globalstep_path = run_path.joinpath(f"red_{self.global_step}.pt")
        torch.save(self.blue_policy.state_dict(), blue_agent_path)
        torch.save(self.red_policy.state_dict(), red_agent_path)
        torch.save(self.blue_policy.state_dict(), blue_globalstep_path)
        torch.save(self.red_policy.state_dict(), red_globalstep_path)
        if self.args.track:
            import wandb
            wandb.save(blue_agent_path, base_path=run_path, policy="now")
            wandb.save(red_agent_path, base_path=run_path, policy="now")
            wandb.save(blue_globalstep_path, base_path=run_path, policy="now")
            wandb.save(red_globalstep_path, base_path=run_path, policy="now")
        return blue_agent_path, red_agent_path
    
    def log_training_metrics(self, writer):
        writer.add_scalar("charts/blue_actor_lr", self.blue_optimizer.param_groups[0]["lr"], self.global_step)
        writer.add_scalar("charts/blue_critic_lr", self.blue_optimizer.param_groups[1]["lr"], self.global_step)
        writer.add_scalar("losses/blue_value_loss", self.blue_value_loss.item(), self.global_step)
        writer.add_scalar("losses/blue_policy_loss", self.blue_policy_loss.item(), self.global_step)
        writer.add_scalar("losses/blue_entropy", self.blue_entropy_loss.item(), self.global_step)
        writer.add_scalar("losses/blue_old_approx_kl", self.blue_old_approx_kl.item(), self.global_step)
        writer.add_scalar("losses/blue_approx_kl", self.blue_approx_kl.item(), self.global_step)
        writer.add_scalar("losses/blue_clipfrac", np.mean(self.blue_clipfracs), self.global_step)
        writer.add_scalar("losses/blue_explained_variance", self.blue_explained_variance, self.global_step)

        writer.add_scalar("charts/red_actor_lr", self.red_optimizer.param_groups[0]["lr"], self.global_step)
        writer.add_scalar("charts/red_critic_lr", self.red_optimizer.param_groups[1]["lr"], self.global_step)
        writer.add_scalar("losses/red_value_loss", self.red_value_loss.item(), self.global_step)
        writer.add_scalar("losses/red_policy_loss", self.red_policy_loss.item(), self.global_step)
        writer.add_scalar("losses/red_entropy", self.red_entropy_loss.item(), self.global_step)
        writer.add_scalar("losses/red_old_approx_kl", self.red_old_approx_kl.item(), self.global_step)
        writer.add_scalar("losses/red_approx_kl", self.red_approx_kl.item(), self.global_step)
        writer.add_scalar("losses/red_clipfrac", np.mean(self.red_clipfracs), self.global_step)
        writer.add_scalar("losses/red_explained_variance", self.red_explained_variance, self.global_step)
    def reset(self):
        reset = self.envs.reset()[0]
        self.blue_resets = np.array(reset["blue"])
        self.red_resets = np.array(reset["red"])
        self.blue_next_obs = torch.Tensor(self.blue_resets).to(self.device)
        self.red_next_obs = torch.Tensor(self.red_resets).to(self.device)