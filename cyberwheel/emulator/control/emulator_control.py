"""
Module defines the class and functions to perform setup actions in the emulator before an experiment begins.
"""

from __future__ import annotations
import pathlib
import subprocess
from typing import List
from cyberwheel.emulator.utils import read_config

DIR_PATH = pathlib.Path(__file__).parent.resolve()
NETWORK_CONFIGS_PATH = f"{DIR_PATH}/../../resources/configs/network"
EMULATOR_CONFIG_PATH = f"{DIR_PATH}/../"
EMULATOR_CONFIG = "emulator_config.yaml"


class EmulatorControl:
    """
    Class setup emulator before an experiment begins.
    """

    emu_config = read_config(EMULATOR_CONFIG_PATH, EMULATOR_CONFIG)

    def __init__(self, config_file_name: str):
        self.net_config = read_config(NETWORK_CONFIGS_PATH, config_file_name)

    def get_host_names(self) -> List[str]:
        hosts = self.net_config["hosts"]

        host_names = []

        # Firewheel host names cannot have "_" and are replaced with "-"
        # when creating the VM.
        for name, _ in hosts.items():
            replace_underscore = lambda name: name.replace("_", "-")
            host_names.append(replace_underscore(name))

        return host_names

    def get_decoy_host_names(self) -> List[str]:
        """Get all decoy host names."""
        decoys = self.net_config["decoys"]

        decoy_names = []

        # Firewheel host names cannot have "_" and are replaced with "-"
        # when creating the VM.
        for name in decoys:
            replace_underscore = lambda name: name.replace("_", "-")
            decoy_names.append(replace_underscore(name))

        return decoy_names

    def enroll_agent_to_fleet(self, host_name: str):
        """Enroll elastic agent to fleet server."""
        host_user = EmulatorControl.emu_config["firewheel"]["host"]["username"]
        host_pwd = EmulatorControl.emu_config["firewheel"]["host"]["password"]
        url = EmulatorControl.emu_config["fleet"]["server-url"]
        token = EmulatorControl.emu_config["fleet"]["enrollment-token"]

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

    def init_hosts(self) -> bool:
        """Setup hosts and run scripts before an experiment begins"""

        # Action #1: Enroll non-decoy hosts to fleet

        all_host_names = self.get_host_names()
        decoy_names = self.get_decoy_host_names()
        non_decoy_names = [name for name in all_host_names if name not in decoy_names]

        for host_name in non_decoy_names:
            success = self.enroll_agent_to_fleet(host_name)

            if not success:
                print("Error with emulator initialization.")
                return False

        # Add more setup actions here...

        print("Emulator host initialization is complete.")
        return True

    @classmethod
    def get_ip_address(cls, host_name: str) -> str:
        host_user = EmulatorControl.emu_config["firewheel"]["host"]["username"]
        host_pwd = EmulatorControl.emu_config["firewheel"]["host"]["password"]

        command_arr = [
            f"sshpass -p {host_pwd}",
            f"firewheel ssh {host_user}@{host_name}",
            "ip -4 -brief address show | grep ens2 | awk '{print $3}'",
        ]
        cmd = " ".join(command_arr)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
        )

        if result.returncode != 0:
            print(result.stderr)
            return ""
        elif result.stdout == "":
            return ""
        else:
            ip = result.stdout.split("/")[0]
            return ip
