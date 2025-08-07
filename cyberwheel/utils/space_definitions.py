from cyberwheel.network.network_base import Network

get_red_max_obs_space = lambda args: args.max_num_hosts * 7
get_red_max_action_space = lambda args, red_agent: args.max_num_hosts * red_agent.action_space.num_actions * 2
get_blue_max_obs_space = lambda args: args.max_num_hosts * 2 + 1
get_blue_max_action_space = lambda args: 0