import sys
from cyberwheel.utils import train_cyberwheel, evaluate_cyberwheel, run_cyberwheel, run_visualization_server

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
            print("Invalid arguments!")
    else:
        print("Invalid arguments!")
    