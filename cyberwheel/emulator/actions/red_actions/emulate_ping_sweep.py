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

    name = "Remote System Discovery - sweep"

    def __init__(self, src_host, target_host):
        super().__init__(src_host, target_host)
        self.name = EmulatePingSweep.name

    @property
    def shell_script_name(self):
        return "scripts/linux_ping_sweep.sh"

    def build_emulator_cmd(
        self, start_host: int = 1, end_host: int = 254, subnet: str = "192.168.1"
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
        # Uncomment to test command locally
        # command_arr = [
        #     f"{dir_name}/scripts/linux_ping_sweep.sh",
        #     f"{start_host}",
        #     f"{end_host}",
        #     f"{subnet}",
        # ]

        action_cmd_arr = [
            f"bash {self.shell_script_name}",
            f"{start_host}",
            f"{end_host}",
            f"{subnet}",
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
            print(result.stderr)
        else:
            output = result.stdout
            discovered_ips = stdout_to_list(output)
            print("discovered ips: ", discovered_ips)

        # TODO convert IPs to [Host] and add to action_results

        return self.action_results
