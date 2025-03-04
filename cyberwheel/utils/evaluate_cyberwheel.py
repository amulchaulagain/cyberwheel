from cyberwheel.utils import parse_eval_override_args, YAMLConfig, Evaluator


def evaluate_cyberwheel(evaluation_config: str):
    """
    This script will evaluate cyberwheel. Using the args from the config file passed, it will evaluate a pre-trained model and evaluate.
    Can fetch models from W&B, as well as use any stored in cyberwheel/data/models
    """
    # Allows using command line to override args in the YAML config
    args = YAMLConfig(evaluation_config)
    args.parse_config()
    args_dict = vars(args)

    override_args = parse_eval_override_args()
    override_args_dict = vars(override_args)
    for arg in override_args_dict:
        if arg in args_dict and override_args_dict[arg] != None and override_args_dict[arg] != "":
            setattr(args, arg, override_args_dict[arg])

    args.evaluation = True

    # Initialize the Evaluator object
    evaluator = Evaluator(args)

    # Configure training parameters and train
    evaluator.configure_evaluation()

    evaluator.evaluate()
