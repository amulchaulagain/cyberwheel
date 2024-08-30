"""
Module defines the class and functions to perform setup actions in the emulator before an experiment begins.
"""

from __future__ import annotations
import os
import subprocess
from typing import List
from cyberwheel.emulator.utils import read_config

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
NETWORK_CONFIGS_PATH = f"{DIR_PATH}/../../resources/configs/network"
EMULATOR_CONFIG_PATH = f"{DIR_PATH}/../"
EMULATOR_CONFIG = "emulator_config.yaml"


class EmulatorSetup:
    """
    Class setup emulator before an experiment begins.
    """

    emu_config = read_config(EMULATOR_CONFIG_PATH, EMULATOR_CONFIG)

    def __init__(self, config_file_name: str):
        self.net_config = read_config(NETWORK_CONFIGS_PATH, config_file_name)

    def get_host_names(self) -> List[str]:
        hosts = self.net_config["hosts"]

        host_names = []
        for name, _ in hosts.items():
            replace_underscore = lambda name: name.replace("_", "-")
            host_names.append(replace_underscore(name))

        return host_names

    def enroll_agent_to_fleet(self, host_name: str):
        """Enroll elastic agent to fleet server."""
        host_user = EmulatorSetup.emu_config["firewheel"]["host"]["username"]
        host_pwd = EmulatorSetup.emu_config["firewheel"]["host"]["password"]
        url = EmulatorSetup.emu_config["fleet"]["server-url"]
        token = EmulatorSetup.emu_config["fleet"]["enrollment-token"]

        cmd = f"""sshpass -p {host_pwd} firewheel ssh {host_user}@{host_name} \
        'echo {host_pwd} | sudo -S elastic-agent enroll -f \
        --url={url} \
        --enrollment-token={token} \
        --insecure'
        """

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
            # capture_output=True,
        )

        if result.returncode != 0:
            print(result.stderr)
            return False
        else:
            # print(result.stdout)
            print(f"Successfully enrolled {host_name} to fleet server.")
            return True

    def initHosts(self) -> bool:
        """Setup hosts and run scripts before an experiment begins"""

        host_names = self.get_host_names()
        for host_name in host_names:
            success = self.enroll_agent_to_fleet(host_name)

            if not success:
                print("Error with emulator initialization.")
                return False

        # Add more setup actions here

        print("Emulator host initialization is complete.")
        return True
