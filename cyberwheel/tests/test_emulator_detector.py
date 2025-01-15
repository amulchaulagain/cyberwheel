"""
Module to test the EmulatorDetector class.
"""

import unittest
from cyberwheel.emulator.detectors import EmulatorDectector
from cyberwheel.network.router import Router
from cyberwheel.network.subnet import Subnet
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

NETWORK_CONFIG = "example_config.yaml"

# Test variables
router = Router(name="192.168.0.0")
subnet = Subnet(
    name="192.168.0.0", ip_range="192.168.0.0/24", router=router
)  # user subnet


class TestEmulatorDetector(unittest.TestCase):
    """Unit tests for the the EmulatorDetector class"""

    def test_submit_test_query(self) -> None:
        """Send test query to elasticsearch to check if SIEM is running"""
        eoc = EmulatorDectector(network_config=NETWORK_CONFIG, subnet=subnet)

        print("\n")
        response = eoc.submit_test_query()
        self.assertIsNotNone(response)

    def test_submit_obs_query(self) -> None:
        """Submit query to elasticsearch"""
        eoc = EmulatorDectector(network_config=NETWORK_CONFIG, subnet=subnet)

        print("\n")
        response = eoc.submit_obs_query()
        self.assertIsNotNone(response)

    def test_parse_query_response(self) -> None:
        """Test the response parse function"""
        eoc = EmulatorDectector(network_config=NETWORK_CONFIG, subnet=subnet)

        print("\n")
        response = eoc.submit_obs_query()
        hits = eoc.parse_query_response(response)
        pprint(hits)
        self.assertIsNotNone(response)

    def test_create_alerts(self) -> None:
        """Test creating alerts from log hits"""
        eoc = EmulatorDectector(network_config=NETWORK_CONFIG, subnet=subnet)

        print("\n")
        response = eoc.submit_obs_query()
        hits = eoc.parse_query_response(response)
        alerts = eoc.create_alerts(hits)
        print(f"alert count: {len(list(alerts))}")
        self.assertIsNotNone(alerts)

    def test_obs(self) -> None:
        """Test detector full observation method."""
        eoc = EmulatorDectector(network_config=NETWORK_CONFIG, subnet=subnet)

        print("\n")
        alerts = eoc.obs()
        print(f"alert count: {len(list(alerts))}")
        self.assertIsNotNone(alerts)
