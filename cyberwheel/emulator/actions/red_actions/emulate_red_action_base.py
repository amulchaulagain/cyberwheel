"""
Module defines the class to execute actions in the emulator.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from cyberwheel.red_actions.red_base import ARTAction, RedActionResults
import subprocess
from subprocess import CompletedProcess
from typing import Any


class EmulateRedAction(ARTAction, ABC):
    """
    Abstract class to exeucte actions in the emulator.
    User needs to implement the following abstract methods:

        - shell_script_name() - the name of the shell script that contains the action.
        - build_emulator_cmd() - the shell command that ssh's into the emulator and calls the action script.
        - emulator_execute() - runs the shell command.
    """

    login_username = "ubuntu"  # set in the VM image
    sshpass_cmd = "sshpass -p ubuntu"
    emulator_ssh_cmd = f"firewheel ssh {login_username}"

    @property
    @abstractmethod
    def shell_script_name(self) -> str | type[NotImplementedError]:
        """
        Name of the shell script.

        Include the 'scripts' folder in the name, e.g. 'scripts/linux_ping_sweep.sh'
        """
        return NotImplementedError

    @abstractmethod
    def build_emulator_cmd(self) -> str | type[NotImplementedError]:
        """Construct the full emulator command."""
        raise NotImplementedError

    @abstractmethod
    def emulator_execute(
        self, shell_cmd: str
    ) -> RedActionResults | type[NotImplementedError]:
        """
        Execute red action in the emulator.

        Argrument:
            shell_cmd - shell command that executes shell script in emulator host VM.

        Returns
            RedActionResults
        """
        raise NotImplementedError

    def prefix_emulator_cmd(self, action_cmd: str) -> str:
        """
        Pre-fixes the 'sshpass -p <password>' and 'firewheel ssh' command.

        Argrument:
            action_cmd - action command to execute which includes the shell script.

        Returns:
            final_cmd - full shell command as a string - sshpass + firewheel + action command.
        """
        command_arr = [
            f"{EmulateRedAction.sshpass_cmd}",
            f"{EmulateRedAction.emulator_ssh_cmd}@{self.src_host.name}",
            action_cmd,
        ]
        final_cmd = " ".join(command_arr)
        return final_cmd

    def run_cmd(self, shell_cmd: str) -> CompletedProcess[Any]:
        """
        Run shell command that executes a script on emulator host VM.

        Argrument:
            shell_cmd: shell command to run in a subprocess.

        Returns:
           result: stdout or stderr from executing the shell command.
        """

        result = subprocess.run(
            shell_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
            # capture_output=True,
        )
        return result

    def sim_execute(self) -> type[NotImplementedError]:
        """Not used in emulator."""
        return NotImplementedError

    def perfect_alert(self):
        """Not used in emulator."""
