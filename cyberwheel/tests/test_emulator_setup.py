"""
Module to test the EmulatorSetup class.
"""

import unittest
from cyberwheel.emulator.setup import EmulatorSetup

EMULATOR_CONFIG = "example_config.yaml"


class TestEmulatorSetup(unittest.TestCase):
    """Unit tests for the the emulator red actions"""

    def test_init_hosts(self) -> None:
        """
        Test host setup sequence.
        """
        emu_setup = EmulatorSetup(EMULATOR_CONFIG)
        success_flag = emu_setup.initHosts()

        self.assertTrue(success_flag)
