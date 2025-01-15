"""
Module to test the EmulatorSetup class.
"""

import unittest
from cyberwheel.emulator.control import EmulatorControl

EMULATOR_CONFIG = "example_config.yaml"


class TestEmulatorSetup(unittest.TestCase):
    """Unit tests for the the emulator red actions"""

    def test_init_hosts(self) -> None:
        """
        Test host setup sequence.
        """
        emu_setup = EmulatorControl(EMULATOR_CONFIG)
        success_flag = emu_setup.init_hosts()

        self.assertTrue(success_flag)

    def test_get_ip_address(self) -> None:
        """
        Test retrieving IP address from firewheel.
        """
        host_name = "decoy01"
        ip_addr = EmulatorControl.get_ip_address(host_name)
        print(f"\n{host_name} emulator IP address: {ip_addr}")

        self.assertIsNotNone(ip_addr)
