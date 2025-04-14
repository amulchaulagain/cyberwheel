"""
Module defines the class and functions to perform setup actions in the emulator before an experiment begins.
"""

from __future__ import annotations
from cyberwheel.emulator.utils import read_config
from cyberwheel.emulator.actions.blue_actions import (
    EmulateDeployDecoyHost,
    EmulateRemoveDecoyHost,
)
from cyberwheel.emulator.actions.red_actions import (
    EmulatePingSweep,
    EmulatePortScan,
    EmulateSudoandSudoCaching,
    EmulateDataEncryptedForImpact,
    EmulateLateralMovement,
)
from cyberwheel.emulator.detectors import EmulatorDectector
from cyberwheel.blue_actions.blue_action import BlueActionReturn
from cyberwheel.red_actions.red_base import RedActionResults
from cyberwheel.detectors.alert import Alert
from cyberwheel.network.host import Host
from cyberwheel.network.network_base import Network
from typing import Any, Dict, List, Iterable
import pathlib
import subprocess
import random


DIR_PATH = pathlib.Path(__file__).parent.resolve()
NETWORK_CONFIGS_PATH = f"{DIR_PATH}/../../resources/configs/network"
EMULATOR_CONFIG_PATH = f"{DIR_PATH}/../"
EMULATOR_CONFIG = "emulator_config.yaml"


class EmulatorControl:
    """
    Class setup emulator before an experiment begins.
    """

    emu_config = read_config(EMULATOR_CONFIG_PATH, EMULATOR_CONFIG)

    def __init__(self, network: Network, network_config_name: str):
        self.network = network
        self.net_config_name = network_config_name
        self.net_config = read_config(NETWORK_CONFIGS_PATH, network_config_name)
        self.detector = EmulatorDectector(
            network_config=network_config_name, network=network
        )

    def init_hosts(self) -> bool:
        """Setup hosts and run scripts before an experiment begins."""

        # Action #1: Enroll non-decoy hosts to fleet

        all_host_names = self._get_host_names()
        decoy_names = self._get_decoy_host_names()
        non_decoy_names = [name for name in all_host_names if name not in decoy_names]

        for host_name in non_decoy_names:
            success = self._enroll_agent_to_fleet(host_name)

            if not success:
                print("Error with emulator initialization.")
                return False

        # Add more setup actions here...

        print("Emulator host initialization is complete.")
        return True

    def reset(self) -> bool:
        """Sequence of actions to reset the emulator for each episode"""

        print("removing decoys...")
        decoy_names = self._get_decoy_host_names()
        success = self._reset_decoys(decoy_names)

        return success

    def run_blue_action(
        self,
        action_name: str,
        src_host_name: str,
        id: str = "",
    ) -> BlueActionReturn:
        """Lookup and execute blue actions in the emulator."""

        shell_cmd = ""
        print(f"executing emulator blue action: {action_name}, id: {id}")

        match action_name:
            case "deploy_decoy":
                action = EmulateDeployDecoyHost(network=self.network, configs={})

                # random pick decoy within subnet
                decoy_names = self._get_decoy_host_names()
                random_int = random.randint(0, len(decoy_names) - 1)
                random_decoy_hostname = decoy_names[random_int]

                shell_cmd = action.build_emulator_cmd(random_decoy_hostname)
                return action.emulator_execute(shell_cmd)
            case "remove_decoy_host":
                action = EmulateRemoveDecoyHost(network=self.network, configs={})
                shell_cmd = action.build_emulator_cmd(src_host_name)
                return action.emulator_execute(shell_cmd)
            case "nothing":
                return BlueActionReturn(action_name, False, 0)
            case _:
                print("action does not exist.")
                return BlueActionReturn(action_name, False, 0)

    def run_red_action(
        self,
        action_name: str,
        src_host: Host,
        dst_host: Host,
        id: str = "",
        options: Dict[str, Any] = {},
    ) -> RedActionResults:
        """Lookup and execute red actions in the emulator."""

        shell_cmd = ""
        print(f"executing emulator red action: {action_name}, id: {id}")

        match action_name:
            case "Remote System Discovery":
                action = EmulatePingSweep(
                    src_host=src_host, target_host=dst_host, network=self.network
                )

                src_host_subnet = src_host.subnet
                ip_range = src_host_subnet.ip_range

                # limit ping sweep to the number of hosts on the subnet
                options = {
                    "start_host": 2,
                    "end_host": len(src_host_subnet.get_connected_hostnames()),
                }  # will go to 2-254 if not defined

                # NOTE: ip_range will come from src_host if not provided
                shell_cmd = action.build_emulator_cmd(
                    start_host=options["start_host"],
                    end_host=options["end_host"],
                    ip_range=ip_range,
                )
                return action.emulator_execute(shell_cmd)
            case "Network Service Discovery":
                action = EmulatePortScan(src_host=src_host, target_host=dst_host)
                shell_cmd = action.build_emulator_cmd()
                return action.emulator_execute(shell_cmd)
            case "Sudo and Sudo Caching":
                action = EmulateSudoandSudoCaching(
                    src_host=src_host, target_host=dst_host
                )
                shell_cmd = action.build_emulator_cmd()
                return action.emulator_execute(shell_cmd)
            case "Data Encrypted for Impact":
                action = EmulateDataEncryptedForImpact(
                    src_host=src_host, target_host=dst_host
                )
                shell_cmd = action.build_emulator_cmd()
                return action.emulator_execute(shell_cmd)
            case "LinuxLateralMovement":
                action = EmulateLateralMovement(src_host=src_host, target_host=dst_host)
                shell_cmd = action.build_emulator_cmd()
                return action.emulator_execute(shell_cmd)
            case _:
                print("Attack does not exist.")
                results = RedActionResults(src_host=src_host, target_host=dst_host)
                results.attack_success = False
                return results

    def get_siem_obs(self) -> Iterable[Alert]:
        """
        Returns alerts converted from SIEM logs.

        Queries the last 5 minutes of activity and filters for red action activity.
        Any action done to a decoy generates an alert.
        """

        print("\n")
        alerts = self.detector.obs()
        print(f"alert count: {len(list(alerts))}")
        return alerts

    def get_ip_address(self, host_name: str) -> str:
        """Returns emulator IP address."""
        host_user = self.emu_config["firewheel"]["host"]["username"]
        host_pwd = self.emu_config["firewheel"]["host"]["password"]

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

    def _get_host_names(self) -> List[str]:
        """Returns list of all host names."""

        hosts = self.net_config["hosts"]
        host_names = []

        # Firewheel host names cannot have "_" and are replaced with "-"
        # when creating the VM.
        for name, _ in hosts.items():
            replace_underscore = lambda name: name.replace("_", "-")
            host_names.append(replace_underscore(name))

        return host_names

    def _get_decoy_host_names(self) -> List[str]:
        """Returns list of all decoy host names."""

        decoys = self.net_config["decoys"]
        decoy_names = []

        # Firewheel host names cannot have "_" and are replaced with "-"
        # when creating the VM.
        for name in decoys:
            replace_underscore = lambda name: name.replace("_", "-")
            decoy_names.append(replace_underscore(name))

        return decoy_names

    def _enroll_agent_to_fleet(self, host_name: str):
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

    def _reset_decoys(self, hostnames: list[str]) -> bool:
        """Reset decoys by turning off network "ens2" interface"""

        for decoy_hostname in hostnames:
            action = EmulateRemoveDecoyHost(network=self.network, configs={})
            shell_cmd = action.build_emulator_cmd(decoy_hostname=decoy_hostname)
            result = action.emulator_execute(shell_cmd)
            if result.success:
                continue
            else:
                return False

        return True
