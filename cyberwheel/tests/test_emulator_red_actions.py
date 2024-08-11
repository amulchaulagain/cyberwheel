"""
Module to test the red actions in the emulator.
"""

import unittest
from cyberwheel.emulator.actions.red_actions import EmulatePingSweep
from cyberwheel.network.host import Host
from cyberwheel.network.router import Router
from cyberwheel.network.subnet import Subnet

# setup network for tests
router = Router(name="192.168.1.0")
subnet = Subnet(name="192.168.1.0", ip_range="192.168.1.0", router=router)
host = Host(name="user-subnet-host-0.cyberwheel.com", subnet=subnet, host_type=None)


class TestEmulatorRedActions(unittest.TestCase):
    """Unit tests for the the emulator red actions"""

    def test_ping_sweep(self) -> None:
        """
        Test ping sweep in emulator.
        """
        red_action = EmulatePingSweep(src_host=host, target_host=host)
        print(red_action.get_name())

        ping_sweep_cmd = red_action.build_emulator_cmd(
            start_host=1, end_host=6, subnet="192.168.0"
        )

        results = red_action.emulator_execute(ping_sweep_cmd)

        self.assertIsNotNone(results.attack_success)
