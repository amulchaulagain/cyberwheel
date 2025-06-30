from cyberwheel.utils import parse, YAMLConfig, Evaluator, EmulatorEvaluator


def evaluate_cyberwheel(args: YAMLConfig, emulate=False):
    """
    This script will evaluate cyberwheel. Using the args from the config file passed, it will evaluate a pre-trained model and evaluate.
    Can fetch models from W&B, as well as use any stored in cyberwheel/data/models
    """
    args.evaluation = True

    # Initialize the Evaluator object
    evaluator = EmulatorEvaluator(args) if emulate else Evaluator(args)

    # Configure training parameters and train
    evaluator.configure_evaluation()

    evaluator.evaluate()
