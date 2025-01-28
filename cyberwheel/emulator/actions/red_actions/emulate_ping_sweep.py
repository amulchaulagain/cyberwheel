"""
Module defines the class to execute ping sweep action in the emulator.
"""

from __future__ import annotations
import os
from cyberwheel.emulator.actions import stdout_to_list
from .emulate_red_action_base import EmulateRedAction

file_path = os.path.realpath(__file__)
dir_name = os.path.dirname(file_path)


class EmulatePingSweep(EmulateRedAction):
    """
    Class to exeucte ping sweep in the emulator.
    """

    name = "RemoteSystemDiscovery"

    def __init__(self, src_host, target_host):
        super().__init__(src_host, target_host)
        self.name = EmulatePingSweep.name

    def build_emulator_cmd(
        self,
        start_host: int = 2,
        end_host: int = 254,
        ip_range: str = "192.168.0.0/24",
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
        ip_split = ip_range.split(".")
        subnet = ".".join(ip_split[:-1])  # drop last element

        action_cmd_arr = [
            f"'for ip in $(seq {start_host} {end_host});",
            f"do ping -c 1 {subnet}.$ip;",
            f"[ $? -eq 0 ] && echo {subnet}.$ip UP || :;",
            r"done | grep UP | cut -d \" \" -f 1'",
        ]
        action_cmd = " ".join(action_cmd_arr)

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
            discovered_ips = stdout_to_list(result.stdout)
            print("discovered ips: ", discovered_ips)

        # TODO convert IPs to [Host] and add to action_results

        return self.action_results
