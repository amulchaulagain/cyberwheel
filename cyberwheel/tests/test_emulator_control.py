"""
Module to test the EmulatorSetup class.
"""

import unittest
from cyberwheel.emulator.control import EmulatorControl
from cyberwheel.network.network_base import Network
from cyberwheel.network.router import Router
from cyberwheel.network.subnet import Subnet

NETWORK_CONFIG = "integration_config.yaml"

# TEST variables
network = Network(name="test")
router = Router(name="core_router")
subnet = Subnet(name="user_subnet", ip_range="192.168.0.0/24", router=router)


class TestEmulatorSetup(unittest.TestCase):
    """Unit tests for the the emulator red actions"""

    def test_init_hosts(self) -> None:
        """
        Test host setup sequence.
        """
        emulator = EmulatorControl(
            network=network, subnet=subnet, network_config_name=NETWORK_CONFIG
        )
        success_flag = emulator.init_hosts()

        self.assertTrue(success_flag)
