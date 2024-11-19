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

    def test_submit_test_query(self) -> None:
        """Send test query to elasticsearch to check if SIEM is running"""
        eoc = EmulatorDectector()

        print("\n")
        response = eoc.submit_test_query()
        self.assertIsNotNone(response)

    def test_submit_obs_query(self) -> None:
        """Submit query to elasticsearch"""
        eoc = EmulatorDectector()

        print("\n")
        response = eoc.submit_obs_query(query=SiemQuery(), size="1")
        self.assertIsNotNone(response)
