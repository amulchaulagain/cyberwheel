"""
Module defines the class to execute ping sweep action in the emulator.
"""

from __future__ import annotations
import os
from cyberwheel.emulator.actions import stdout_to_list
from .emulate_red_action_base import EmulateRedAction
from cyberwheel.network.network_base import Network

file_path = os.path.realpath(__file__)
dir_name = os.path.dirname(file_path)


class EmulatePing(EmulateRedAction):
    """
    Class to exeucte ping sweep in the emulator.
    """

    name = "Remote System Discovery"

    def __init__(self, src_host, target_host, network: Network):
        super().__init__(src_host, target_host)
        self.name = EmulatePing.name
        self.network = network
        self.host = None

    def build_emulator_cmd(
        self,
        decoy_ip: int = 2,
    ):
        """
        Construct shell command to execute ping sweep script on emulator host VM.

        Argruments:
            start_host - start ip range value (e.g. 1)
            end_host - end ip range value (e.g. start_host< value <=254)
            subnet - host subnet (e.g. 192.168.1)

        Returns:
            shell_cmd - full shell command that runs in a subprocess.
        """
        # TODO: update to parse network address
        action_cmd = f"ping -c 1 {decoy_ip} >/dev/null && echo {decoy_ip}"

        shell_cmd = self.prefix_emulator_cmd(action_cmd)
        return shell_cmd

    def emulator_execute(self, shell_cmd: str):
        """
        Execute ping sweep in the emulator. Discovered IP addresses are added to
        discovered hosts in RedActionResults.

        Argrument:
            shell_cmd - shell command to execute a ping sweep in emulator host VM.

        Returns:
            RedActionResults
        """

        # Execute ping sweep in emulator VM
        print(f"executing shell command: {shell_cmd}")
        result = self.run_cmd(shell_cmd)

        # Capture output after executing command
        if result.returncode != 0:
            self.action_results.attack_success = False
            print(result.stderr)
        else:
            self.action_results.attack_success = True
            discovered_ip = result.stdout.strip()
            print("discovered a new host: ", discovered_ip)
            for host in self.network.get_all_hosts():
                if str(host.ip_address) == discovered_ip:
                    self.action_results.add_host(host)

        print(
            f"added new discovered host to action results: {[host.name for host in self.action_results.discovered_hosts]}"
        )

        return self.action_results
