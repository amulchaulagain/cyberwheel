import argparse
import os

from distutils.util import strtobool

from cyberwheel.utils import YAMLConfig


def _seed_list(raw: str):
    """argparse type for --seeds: comma-separated unique integers."""
    try:
        seeds = [int(p) for p in raw.split(",") if p.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"seeds must be comma-separated integers, got {raw!r}")
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique")
    return seeds


def parse(config, mode: str = 'train'):
    args = YAMLConfig(config)
    args.parse_config()
    args_dict = vars(args)

    override_args = parse_eval_override_args() if mode == 'evaluate' else parse_override_args() if mode == 'train' else parse_default_override_args() if mode == 'run' else args
    override_args_dict = vars(override_args)
    for arg in override_args_dict:
        if arg in args_dict and override_args_dict[arg] != None and override_args_dict[arg] != "":
            setattr(args, arg, override_args_dict[arg])

    # 'seeds' is optional in eval YAMLs, so the presence-gated loop above misses it.
    if mode == 'evaluate' and getattr(override_args, 'seeds', None):
        args.seeds = override_args.seeds

    # Probabilistic-exploit keys are optional (absent from base YAMLs), so the
    # presence-gated loop above skips them; apply them explicitly when set.
    for key in ("probabilistic_exploits", "exploit_success_floor",
                "exploit_success_ceiling", "exploit_severity_config"):
        value = getattr(override_args, key, None)
        if value is not None:
            setattr(args, key, value)

    # 'agents' is a nested map, so the presence-gated loop can't reach
    # agents.green (the same reason --blue-agent/--red-agent don't reach it).
    # Green is optional and off by default; apply the flag explicitly.
    green_agent = getattr(override_args, "green_agent", None)
    if green_agent and isinstance(getattr(args, "agents", None), dict):
        args.agents["green"] = green_agent

    if args.deterministic:
        os.environ["CYBERWHEEL_DETERMINISTIC"] = 'true'
    else:
        os.environ["CYBERWHEEL_DETERMINISTIC"] = 'false'

    return args


def parse_override_args(print_help: bool = False):
    """
    Parses through any command line arguments for training. These will override any args set in the YAML config.
    """
    parser = argparse.ArgumentParser()
    training_group = parser.add_argument_group("Training Parameters", description="parameters to handle training options")
    env_group = parser.add_argument_group("Environment Parameters", description="parameters to handle Cyberwheel environment options")
    rl_group = parser.add_argument_group("Reinforcement Learning Parameters", description="parameters to handle various RL options")

    # Training Parameters
    training_group.add_argument("--experiment-name", type=str, help="The name of the experiment. This will be the name the model is saved locally and on W&B.")
    training_group.add_argument("--seed", type=int, help="seed of the experiment")
    training_group.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=True`")
    training_group.add_argument("--device", type=str, help="Choose the device used for optimization. Choose 'cuda', 'cpu', or specify a gpu with 'cuda:0'")
    training_group.add_argument("--async-env", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, uses AsyncVectorEnv instead of SyncVectorEnv")
    training_group.add_argument("--track", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, this experiment will be tracked with Weights and Biases")
    training_group.add_argument("--wandb-project-name", type=str, help="the wandb's project name")
    training_group.add_argument("--wandb-entity", type=str, help="the entity (team) of wandb's project")
    training_group.add_argument("--total-timesteps", type=int, help="total timesteps of the experiments")
    training_group.add_argument("--num-saves", type=int, help="the number of model saves and evaluations to run throughout training")
    training_group.add_argument("--num-envs", type=int, help="the number of parallel game environments")
    training_group.add_argument("--num-steps", type=int, help="the number of steps to run in each environment per policy rollout")
    training_group.add_argument('--eval-episodes', type=int, help='Number of evaluation episodes to run')
    training_group.add_argument("--debug-mode", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if debug mode is on, sets num_envs to 1, doesn't track, and uses cpu")
    
    # Cyberwheel Environment Parameters
    env_group.add_argument("--environment", type=str, help="the environment class to use. Current options: CyberwheelRL | Cyberwheel")
    env_group.add_argument("--red-agent", type=str, help="the red agent config to train with. Current options: rl_red_agent.yaml | art_agent.yaml | inactive_red_agent.yaml")
    env_group.add_argument("--campaign", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, uses ARTCampaign as red agent")
    env_group.add_argument("--train-red", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="toggle if you want to train the red agent")
    env_group.add_argument("--valid-targets", type=str, help="Hosts to consider as valid targets for red agent. Can be: 'HOST_NAME' | 'servers' | 'all'")
    env_group.add_argument("--blue-agent", type=str, help="the blue agent config to train with. Current options: rl_blue_agent.yaml | inactive_blue_agent.yaml")
    env_group.add_argument("--green-agent", type=str, help="optional green (benign user) agent config from cyberwheel/data/configs/green_agent, e.g. scripted_green.yaml; omit for no green agent")
    env_group.add_argument("--train-blue", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="toggle if you want to train the blue agent")
    env_group.add_argument("--network-config", help="Input the network config filename", type=str)
    env_group.add_argument("--decoy-config", help="Input the decoy config filename", type=str)
    env_group.add_argument("--host-config", help="Input the host config filename", type=str)
    env_group.add_argument("--reward-function", help="Which reward function to use. Current option: 'RLReward'", type=str)
    env_group.add_argument("--detector-config", help="Location of detector config file.", type=str)
    env_group.add_argument("--probabilistic-exploits", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, red exploit success is a probability weighted by CVE severity instead of always succeeding on a valid technique")
    env_group.add_argument("--exploit-success-floor", type=float, help="minimum exploit success probability when probabilistic-exploits is on (default 0.1)")
    env_group.add_argument("--exploit-success-ceiling", type=float, help="maximum exploit success probability when probabilistic-exploits is on (default 0.95)")
    env_group.add_argument("--exploit-severity-config", type=str, help="optional YAML in cyberwheel/data/configs/exploit_severity mapping CVE id -> normalized CVSS score [0,1]")

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

    args = parser.parse_args()
    if print_help:
        parser.print_help()
        return
    return args

def parse_eval_override_args(print_help: bool = False):
    """
    Parses through any command line arguments for evaluating. These will override any args set in the YAML config.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--red-agent", type=str, help="the red agent config to evaluate with. Current options: rl_red_agent.yaml | art_agent.yaml | inactive_red_agent.yaml")
    parser.add_argument("--blue-agent", type=str, help="the blue agent config to evaluate with. Current options: rl_blue_agent.yaml | inactive_blue_agent.yaml")
    parser.add_argument("--green-agent", type=str, help="optional green (benign user) agent config from cyberwheel/data/configs/green_agent, e.g. scripted_green.yaml; omit for no green agent")
    parser.add_argument("--environment", type=str, help="the environment class to use. Current options: CyberwheelRL | Cyberwheel")
    parser.add_argument("--valid-targets", type=str, help="Hosts to consider as valid targets for red agent. Can be: 'HOST_NAME' | 'servers' | 'all'")
    parser.add_argument("--train-red", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="toggle if you want to train the red agent")
    parser.add_argument("--train-blue", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="toggle if you want to train the blue agent")
    parser.add_argument("--campaign", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, uses ARTCampaign as red agent")
    parser.add_argument("--seed", type=int, help="seed of the experiment")
    parser.add_argument("--seeds", type=_seed_list, help="comma-separated seeds for batch evaluation, e.g. 1,2,3; each seed reseeds the env for its block of episodes regardless of --deterministic")
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=True`")


    parser.add_argument("--download-model", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="Download agent model from WandB. If present, requires --run, --wandb-entity, --wandb-project-name flags.")
    parser.add_argument("--checkpoint", help="Which checkpoint of the model to evaluate. Defaults to 'agent' (latest).")
    parser.add_argument("--run", help="Run ID from WandB for pretrained blue agent to use. Required when downloading model from W&B")
    parser.add_argument("--experiment-name", help="Experiment name for storing/retrieving agent model")
    parser.add_argument("--visualize", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="Stores graphs of network state at each step/episode. Can be viewed in dash server.")
    parser.add_argument("--graph-name", help="Override naming convention of graph storage directory.")
    parser.add_argument("--network-config", help="Input the network config filename", type=str)
    parser.add_argument("--decoy-config", help="Input the decoy config filename", type=str)
    parser.add_argument("--host-config", help="Input the host config filename", type=str)
    parser.add_argument("--detector-config", help="Path to detector config file", type=str)
    parser.add_argument("--reward-function", help="Which reward function to use. Current option: 'RLReward' (default)", type=str)
    parser.add_argument("--probabilistic-exploits", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, red exploit success is a probability weighted by CVE severity instead of always succeeding on a valid technique")
    parser.add_argument("--exploit-success-floor", type=float, help="minimum exploit success probability when probabilistic-exploits is on (default 0.1)")
    parser.add_argument("--exploit-success-ceiling", type=float, help="maximum exploit success probability when probabilistic-exploits is on (default 0.95)")
    parser.add_argument("--exploit-severity-config", type=str, help="optional YAML in cyberwheel/data/configs/exploit_severity mapping CVE id -> normalized CVSS score [0,1]")
    parser.add_argument("--num-steps", help="Number of steps per episode for evaluation", type=int)
    parser.add_argument("--num-episodes", help="Number of episodes to evaluate", type=int)
    parser.add_argument("--wandb-entity", help="Username where W&B model is stored. Required when downloading model from W&B", type=str)
    parser.add_argument("--wandb-project-name", help="Project name where W&B model is stored. Required when downloading model from W&B", type=str)

    if print_help:
        parser.parse_args()
        parser.print_help()
    else:
        return parser.parse_args()

def parse_default_override_args(print_help: bool = False):
    """
    Parses through any command line arguments for setting up environment. These will override any args set in the YAML config.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", type=str, help="the environment class to use. Current options: CyberwheelRL | Cyberwheel")
    parser.add_argument("--experiment-name", help="Experiment name for storing/retrieving agent model")
    parser.add_argument("--network-config", help="Input the network config filename", type=str)
    parser.add_argument("--decoy-config", help="Input the decoy config filename", type=str)
    parser.add_argument("--host-config", help="Input the host config filename", type=str)
    parser.add_argument("--num-steps", help="Number of steps per episode for evaluation", type=int)
    parser.add_argument( "--num-episodes", help="Number of episodes to evaluate", type=int)
    parser.add_argument("--deterministic", type=lambda x: bool(strtobool(x)), nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=True`")


    if print_help:
        parser.parse_args()
        parser.print_help()
    else:
        return parser.parse_args()
