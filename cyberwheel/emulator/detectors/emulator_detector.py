"""
Module defines the Emulator Dectectory class.
"""

from __future__ import annotations
from subprocess import CompletedProcess
from typing import Any, Iterable
import json
import subprocess
import os
from cyberwheel.detectors.alert import Alert
from cyberwheel.detectors.detector_base import Detector
from cyberwheel.emulator.siem import SiemQuery
from cyberwheel.emulator.utils import read_config

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
EMULATOR_CONFIG_PATH = f"{DIR_PATH}/../"
EMULATOR_CONFIG = "emulator_config.yaml"


class EmulatorDectector(Detector):
    """
    Class to communicate with SIEM (elasticsearch) within the emulator.
    The detector, Sysmon, fowards information to the SIEM.
    """

    emu_config = read_config(EMULATOR_CONFIG_PATH, EMULATOR_CONFIG)

    def query_to_json(self, result: CompletedProcess[str]) -> Any | None:
        """Converts SIEM query reponse to JSON."""
        return json.loads(result.stdout)

    def submit_test_query(self) -> CompletedProcess[str] | None:
        """SSHs into the host with the SIEM and submits a query."""
        siem_pwd = EmulatorDectector.emu_config["firewheel"]["siem"]["password"]
        siem_user = EmulatorDectector.emu_config["firewheel"]["siem"]["username"]
        siem_hostname = EmulatorDectector.emu_config["firewheel"]["siem"]["hostname"]

        cmd_arr = [
            f"sshpass -p {siem_pwd} firewheel ssh {siem_user}@{siem_hostname}",
            f"curl -u elastic:elastic -X GET http://localhost:9200",
        ]
        cmd = " ".join(cmd_arr)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
        )

        if result.returncode != 0:
            error = {"error": result.stderr}
            print(json.dumps(error))
            return None
        else:
            print(result.stdout)

        return result

    def submit_obs_query(
        self, query: SiemQuery, size: str = "10"
    ) -> CompletedProcess[str] | None:
        """Shells into the SIEM VM and submits a query to get the oberstation state."""
        siem_pwd = EmulatorDectector.emu_config["firewheel"]["siem"]["password"]
        siem_user = EmulatorDectector.emu_config["firewheel"]["siem"]["username"]
        siem_hostname = EmulatorDectector.emu_config["firewheel"]["siem"]["hostname"]

        cmd_arr = [
            f"sshpass -p {siem_pwd} firewheel ssh {siem_user}@{siem_hostname}",
            "curl -u elastic:elastic",
            '-XGET -H "Content-Type: application/json"',
            f'"http://localhost:9200/logs-*/_search?size={size}"',
            # f"-d '{query.get_observation()}'",
        ]
        cmd = " ".join(cmd_arr)

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            shell=True,
        )

        if result.returncode != 0:
            error = {"error": result.stderr}
            print(json.dumps(error))
            return None
        else:
            print(result.stdout)

        return result

    def obs(self, perfect_alert: Alert) -> Iterable[Alert]:
        """
        Creates an array of alerts using information from the SIEM's query response.
        TODO: implement
        """
        alerts = []
        return alerts
