# CleanRL script for training Cage Challenge 2 agents. CleanRL documentation can be found at https://docs.cleanrl.dev/,
import os
import time

from cyberwheel.utils import (
    parse_override_args,
    parse_eval_override_args,
    YAMLConfig,
    Evaluator,
    EmulatorEvaluator,
)

"""
This script will train cyberwheel. Using the args from the config file passed, it will run an Actor-Critic RL algorithm and run training
with intermittent evaluations/saves. If tracking to W&B, this will be logged in your W&B project for each training run.
"""
# Allows using command line to override args in the YAML config
def evaluate_cyberwheel_emu(args: YAMLConfig, emulate=False):
    args.evaluation = True  # Should be set anyway?

    # Initialize the Evaluator object
    evaluator = EmulatorEvaluator(args) if emulate else Evaluator(args)
    # Configure training parameters and train
    evaluator.configure_evaluation()

    evaluator.evaluate()
