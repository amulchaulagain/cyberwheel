from typing import Dict

from cyberwheel.blue_actions.blue_action import BlueActionReturn, HostAction
from cyberwheel.network.host import Host
from cyberwheel.network.network_base import Network


class QuarantineHost(HostAction):
    """
    Network-isolate a real host. The host's link to its subnet is cut, which
    blocks all red killchain actions targeting it (and traps red if it sits on
    the host) until a RestoreHost. Returns recurring=1 so the action's YAML
    `recurring` value applies as a persistent per-step collateral cost while
    the host stays quarantined.
    """

    def __init__(self, network: Network, configs: Dict[str, any], **kwargs) -> None:
        super().__init__(network, configs)

    def execute(self, host: Host, **kwargs) -> BlueActionReturn:
        if host.isolated:
            return BlueActionReturn("", False, 0, target=host.name)
        self.network.isolate_host(host, host.subnet)
        self.network.isolated_hosts.append(host)
        return BlueActionReturn(host.name, True, 1, target=host.name)
