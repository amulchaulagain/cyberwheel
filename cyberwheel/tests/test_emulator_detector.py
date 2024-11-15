"""
Module to test the EmulatorDetector class.
"""

import unittest
from dotenv import load_dotenv
from cyberwheel.emulator.detectors import EmulatorDectector
from cyberwheel.emulator.siem import SiemQuery

load_dotenv()


class TestEmulatorDetector(unittest.TestCase):
    """Unit tests for the the EmulatorDetector class"""

    # def test_submit_obs_query(self) -> None:
    #     """Send test query SIEM to check if alive"""
    #     eoc = EmulatorDectector()

    #     response = eoc.submit_test_query("ubuntu", "siem", SiemQuery())
    #     self.assertIsNotNone(response)

    def test_submit_obs_query(self) -> None:
        """Test observiation query to SIEM"""
        eoc = EmulatorDectector()

        response = eoc.submit_obs_query(SiemQuery())
        self.assertIsNotNone(response)
