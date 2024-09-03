"""
Module to test the red actions in the emulator.
"""

import unittest
from cyberwheel.emulator.actions.red_actions import EmulatePingSweep, EmulatePortScan
from cyberwheel.network.host import Host
from cyberwheel.network.router import Router
from cyberwheel.network.subnet import Subnet

router = Router(name="192.168.1.0")
subnet = Subnet(name="192.168.1.0", ip_range="192.168.1.0", router=router)


class TestEmulatorRedActions(unittest.TestCase):
    """Unit tests for the the emulator red actions"""

    def test_ping_sweep(self) -> None:
        """Test ping sweep in emulator"""
        src_host = Host(name="user01", subnet=subnet, host_type=None)

        red_action = EmulatePingSweep(src_host, target_host=src_host)
        print(red_action.__class__.get_name())

        ping_sweep_cmd = red_action.build_emulator_cmd(
            start_host=3, end_host=6, subnet="192.168.0"
        )

        results = red_action.emulator_execute(ping_sweep_cmd)
        self.assertIsNotNone(results.attack_success)

    def test_port_scan(self) -> None:
        """Test port scan in emulator"""
        src_host = Host(name="user01", subnet=subnet, host_type=None)
        target_host = Host(name="user02", subnet=subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.3")

        red_action = EmulatePortScan(src_host, target_host)
        print(red_action.__class__.get_name())

        port_scan_cmd = red_action.build_emulator_cmd()
        results = red_action.emulator_execute(port_scan_cmd)
        self.assertIsNotNone(results.attack_success)
