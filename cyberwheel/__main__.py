import sys
from cyberwheel.utils import train_cyberwheel, evaluate_cyberwheel, run_cyberwheel, run_visualization_server, parse_default_override_args, parse_eval_override_args, parse_override_args

def display_help():
    sys.argv = ['']
    print("---------------------------------------------------------------------------------------------------\nTraining Cyberwheel:\n\n")
    parse_override_args(print_help=True)
    print("---------------------------------------------------------------------------------------------------\nEvaluating Cyberwheel:\n\n")
    parse_eval_override_args(print_help=True)
    print("---------------------------------------------------------------------------------------------------\nRunning Cyberwheel:\n\n")
    parse_default_override_args(print_help=True)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        mode = sys.argv.pop(1)
        config = sys.argv.pop(1)
        if mode == 'train':
            train_cyberwheel(config)
        elif mode == 'evaluate':
            evaluate_cyberwheel(config)
        elif mode == 'run':
            run_cyberwheel(config)
        elif mode == 'visualizer':
            run_visualization_server(config)
        else:
            display_help()
    else:
        display_help()
    