from cyberwheel.utils import YAMLConfig, parse_default_override_args, Runner

def run_cyberwheel(config: str):
    # Allows using command line to override args in the YAML config
    args = YAMLConfig(config)
    args.parse_config()
    args_dict = vars(args)

    override_args = parse_default_override_args()
    override_args_dict = vars(override_args)
    for arg in override_args_dict:
        if arg in args_dict and override_args_dict[arg]:
            setattr(args, arg, override_args_dict[arg])

    # Initialize the Evaluator object
    runner = Runner(args)

    # Configure training parameters and train
    runner.configure()

    runner.run()

    runner.close()
