import argparse
import sys

from distutils.util import strtobool

def parse_override_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    training_group = parser.add_argument_group("Training Parameters", description="parameters to handle training options")
    env_group = parser.add_argument_group("Environment Parameters", description="parameters to handle Cyberwheel environment options")
    rl_group = parser.add_argument_group("Reinforcement Learning Parameters", description="parameters to handle various RL options")

    # Training Parameters
    training_group.add_argument("--exp-name", type=str, help="the name of this experiment")
    training_group.add_argument("--seed", type=int, help="seed of the experiment")
    training_group.add_argument("--torch-not-deterministic", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    training_group.add_argument("--device", type=str, help="Choose the device used for optimization. Choose 'cuda', 'cpu', or specify a gpu with 'cuda:0'")
    training_group.add_argument("--async-env", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, uses AsyncVectorEnv instead of SyncVectorEnv")
    training_group.add_argument("--track", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, this experiment will be tracked with Weights and Biases")
    training_group.add_argument("--wandb-project-name", type=str, required = "--track" in sys.argv, help="the wandb's project name")
    training_group.add_argument("--wandb-entity", type=str, required = "--track" in sys.argv, help="the entity (team) of wandb's project")
    training_group.add_argument("--total-timesteps", type=int, help="total timesteps of the experiments")
    training_group.add_argument("--num-saves", type=int, help="the number of model saves and evaluations to run throughout training")
    training_group.add_argument("--num-envs", type=int, help="the number of parallel game environments")
    training_group.add_argument("--num-steps", type=int, help="the number of steps to run in each environment per policy rollout")
    training_group.add_argument('--eval-episodes', type=int, help='Number of evaluation episodes to run')
    training_group.add_argument('--resume', type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="Whether to resume a previous run. If so, exp_name must match.")
    training_group.add_argument('--training-config', type=str, help="Input the training config filename", default="train_blue.yaml")
    
    # Cyberwheel Environment Parameters
    env_group.add_argument("--red-agent", type=str, help="the red agent to train against. Current option: 'art_agent' | 'killchain_agent' (deprecated)")
    env_group.add_argument("--red-strategy", type=str, help="the red agent strategies to train against. Current options: 'server_downtime' | 'dfs_impact'")
    env_group.add_argument("--campaign", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, uses ARTCampaign as red agent")
    env_group.add_argument("--network-config", help="Input the network config filename", type=str)
    env_group.add_argument("--decoy-config", help="Input the decoy config filename", type=str)
    env_group.add_argument("--host-config", help="Input the host config filename", type=str)
    env_group.add_argument("--blue-config", help="Input the blue agent config filename", type=str)
    env_group.add_argument("--min-decoys", help="Minimum number of decoys that should be used", type=int)
    env_group.add_argument("--max-decoys", help="Maximum number of decoys that should be used", type=int)
    env_group.add_argument("--reward-function", help="Which reward function to use. Current options: default | step_detected", type=str)
    env_group.add_argument("--reward-scaling", help="Variable used to increase rewards", type=float)
    env_group.add_argument("--detector-config", help="Location of detector config file.", type=str)

    # RL Algorithm Parameters
    rl_group.add_argument("--env-id", type=str, help="the id of the environment")
    rl_group.add_argument("--learning-rate", type=float, help="the learning rate of the optimizer")
    rl_group.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="Toggle learning rate annealing for policy and value networks")
    rl_group.add_argument("--gamma", type=float, help="the discount factor gamma")
    rl_group.add_argument("--gae-lambda", type=float, help="the lambda for the general advantage estimation")
    rl_group.add_argument("--num-minibatches", type=int, help="the number of mini-batches")
    rl_group.add_argument("--update-epochs", type=int, help="the K epochs to update the policy")
    rl_group.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="Toggles advantages normalization")
    rl_group.add_argument("--clip-coef", type=float, help="the surrogate clipping coefficient")
    rl_group.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    rl_group.add_argument("--ent-coef", type=float, help="coefficient of the entropy")
    rl_group.add_argument("--vf-coef", type=float, help="coefficient of the value function")
    rl_group.add_argument("--max-grad-norm", type=float, help="the maximum norm for the gradient clipping")
    rl_group.add_argument("--target-kl", type=float, help="the target KL divergence threshold")

    return parser.parse_args()