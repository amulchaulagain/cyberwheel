"""
Module to test the EmulatorController class and integration methods.

NOTES: ensure target decoy host is deployed (interface turned on) before running
red actions.
"""

import unittest
from cyberwheel.emulator.control import EmulatorControl
from cyberwheel.network.host import Host
from cyberwheel.network.network_base import Network
from cyberwheel.network.router import Router
from cyberwheel.network.subnet import Subnet

NETWORK_CONFIG = "example_config.yaml"

# TEST variables
network = Network(name="test")
router = Router(name="core_router")
subnet = Subnet(name="user_subnet", ip_range="192.168.0.0/24", router=router)


class TestEmulatorIntegration(unittest.TestCase):
    """Unit tests for the the emulator controller actions"""

    emulator = EmulatorControl(
        network=network, subnet=subnet, network_config_name=NETWORK_CONFIG
    )

    def test_run_deploy_decoy_host(self) -> None:
        """
        Test executing a blue action, deploy decoy host, in the emulator.
        """
        action_name = "deploy_decoy_host"
        src_host_name = "decoy01"

        blue_action_return = self.emulator.run_blue_action(action_name, src_host_name)
        self.assertTrue(blue_action_return.success)

    def test_run_remove_decoy_host(self) -> None:
        """
        Test executing a blue action, remove decoy host, in the emulator.
        """
        action_name = "remove_decoy_host"
        src_host_name = "decoy01"

        blue_action_return = self.emulator.run_blue_action(action_name, src_host_name)
        self.assertTrue(blue_action_return.success)

    def test_run_ping_sweep(self) -> None:
        """
        Test executing a red action, ping sweep, in the emulator.
        """
        action_name = "RemoteSystemDiscovery"
        src_host = Host(name="user01", subnet=subnet, host_type=None)
        options = {"start_host": 2, "end_host": 7}  # will go to 2-254 if not defined

        red_action_return = self.emulator.run_red_action(
            action_name, src_host=src_host, dst_host=src_host, options=options
        )
        self.assertTrue(red_action_return.attack_success)

    def test_run_port_scan(self) -> None:
        """
        Test executing a red action, port scan, in the emulator.
        """
        action_name = "RemoteServiceDiscovery"
        src_host = Host(name="user01", subnet=subnet, host_type=None)
        target_host = Host(name="decoy01", subnet=subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.5")

        red_action_return = self.emulator.run_red_action(
            action_name, src_host=src_host, dst_host=target_host
        )
        self.assertTrue(red_action_return.attack_success)

    def test_run_sudo_and_sudo_caching(self) -> None:
        """
        Test executing a red action, sudo and sudo caching, in the emulator.
        """
        action_name = "Sudo and Sudo Caching"
        src_host = Host(name="user01", subnet=subnet, host_type=None)
        target_host = Host(name="decoy01", subnet=subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.5")

        red_action_return = self.emulator.run_red_action(
            action_name, src_host=src_host, dst_host=target_host
        )
        self.assertTrue(red_action_return.attack_success)

    def test_run_data_encrypted_for_impact(self) -> None:
        """
        Test executing a red action, data encrypted for impact, in the emulator.
        """
        action_name = "DataEncryptedForImpact"
        src_host = Host(name="user01", subnet=subnet, host_type=None)
        target_host = Host(name="decoy01", subnet=subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.5")

        red_action_return = self.emulator.run_red_action(
            action_name, src_host=src_host, dst_host=target_host
        )
        self.assertTrue(red_action_return.attack_success)

    def test_run_lateral_movement(self) -> None:
        """
        Test executing a red action, lateral movement, in the emulator.
        """
        action_name = "lateral-movement"
        src_host = Host(name="user01", subnet=subnet, host_type=None)
        target_host = Host(name="decoy01", subnet=subnet, host_type=None)
        target_host.set_ip_from_str("192.168.0.5")

        red_action_return = self.emulator.run_red_action(
            action_name, src_host=src_host, dst_host=target_host
        )
        self.assertTrue(red_action_return.attack_success)

    def test_run_get_siem_obs(self) -> None:
        """
        Test executing querying the SIEM and converting hits to alerts.
        """
        alerts = self.emulator.get_siem_obs()
        self.assertIsNotNone(alerts)
