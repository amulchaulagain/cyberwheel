import yaml
from paramiko import SSHClient
from paramiko.channel import ChannelStdinFile, ChannelFile, ChannelStderrFile
from io import StringIO

import paramiko
import subprocess
import pandas as pd

def read_config(path: str, file_name: str):
    """Read network config from YAML file"""

    with open(f"{path}/{file_name}", "r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
        return data
    
class SSHPool():
    def __init__(self, emu_config):
        self.emu_config = emu_config
        self.host_connections : dict[str, SSHClient] = {}
        self.HOST_USERNAME = emu_config["firewheel"]["host"]["username"]
        self.HOST_PASSWORD = emu_config["firewheel"]["host"]["password"]

        cmd = f"""firewheel vm list ip --csv"""
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            check=True,
        )
        if result.returncode != 0:
            print(f"Error getting VM Control IPs: {result.stderr}")
            return

        vm_ips = pd.read_csv(StringIO(result.stdout))

        all_host_names = list(vm_ips.itertuples(index=False, name=None))

        for host_name, vm_ip in all_host_names:
            if 'siem' in host_name or 'subnet' in host_name:
                continue
            self.host_connections[host_name] = SSHClient()
            self.host_connections[host_name].set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.host_connections[host_name].connect(hostname=vm_ip, username=self.HOST_USERNAME, password=self.HOST_PASSWORD)
            print(vm_ip)

    def run_cmd(self, src = None, tgt = None, cmd = None) -> tuple[ChannelStdinFile, ChannelFile, ChannelStderrFile]:
        if not src or not cmd:
            return
        src_hostname = src.name.replace("_", "-")
        stdin, stdout, stderr = '', '', ''
        try:
            stdin, stdout, stderr = self.host_connections[src_hostname].exec_command(cmd, get_pty=True)
        except Exception as e:
            print(e)
            self.host_connections[src_hostname].close()
            return None, None, None # TODO
        #print(f"Input: {stdin}\nOutput: {stdout}\nError: {stderr}\n")
        
        #print(cmd)
        #print(output)
        #self.host_connections[src.name]
        return stdin, stdout, stderr
    
    def close(self):
        for _, s in self.host_connections.items():
            s.close()

