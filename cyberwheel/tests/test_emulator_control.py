"""
Module to test the EmulatorSetup class.
"""

import unittest
from cyberwheel.emulator.control import EmulatorControl
from cyberwheel.network.network_base import Network
from importlib.resources import files

# from cyberwheel.network.router import Router
# from cyberwheel.network.subnet import Subnet

NETWORK_CONFIG = "example_config.yaml"

############# Create Network Topology from Config File ##################
config_path = files("cyberwheel.resources.configs.network").joinpath(NETWORK_CONFIG)
network = Network.create_network_from_yaml(config_path)

################# Manually Create Network Topology ######################
# network = Network(name="test")
# router = Router(name="core_router")
# subnet = Subnet(name="user_subnet", ip_range="192.168.0.0/24", router=router)


class TestEmulatorSetup(unittest.TestCase):
    """Unit tests for the the emulator red actions"""

    def test_init_hosts(self) -> None:
        """
        Test host setup sequence.
        """
        emulator = EmulatorControl(network=network, network_config_name=NETWORK_CONFIG)
        success_flag = emulator.init_hosts()

        self.assertTrue(success_flag)

    def test_reset(self) -> None:
        """
        Test reset sequence
        """
        emulator = EmulatorControl(network=network, network_config_name=NETWORK_CONFIG)
        success_flag = emulator.reset()

        self.assertTrue(success_flag)

    def test_get_ip_address(self) -> None:
        """
        Test retrieving IP address from emulator.
        """
        emulator = EmulatorControl(network=network, network_config_name=NETWORK_CONFIG)

        all_hosts = list(emulator.net_config["hosts"].keys())
        decoys = emulator.net_config["decoys"]
        user_hosts = [host for host in all_hosts if host not in decoys]
        print(user_hosts)

        for name in user_hosts:
            host_name = name.replace("_", "-")  # firewheel host names use hyphen
            ip = emulator.get_ip_address(host_name)
            print(f"{host_name} ip address: {ip}")
