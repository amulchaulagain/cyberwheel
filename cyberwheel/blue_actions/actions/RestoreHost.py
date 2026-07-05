from typing import Dict

from cyberwheel.blue_actions.blue_action import BlueActionReturn, HostAction
from cyberwheel.network.host import Host
from cyberwheel.network.network_base import Network


class RestoreHost(HostAction):
    """
    Reimage a real host: undo a quarantine (reconnect it to its subnet) and
    clean any compromise. The red agent's foothold/killchain progress on the
    host is reset via network.pending_restores, drained by the red agent on
    its next act. Returns recurring=-1 to remove this host's quarantine
    per-step collateral cost (the reward ledger is keyed by host name).
    """

    def __init__(self, network: Network, configs: Dict[str, any], **kwargs) -> None:
        super().__init__(network, configs)

    def execute(self, host: Host, **kwargs) -> BlueActionReturn:
        if not host.isolated and not host.is_compromised:
            return BlueActionReturn("", False, 0, target=host.name)
        if host.isolated:
            self.network.reconnect_host(host, host.subnet)
        host.is_compromised = False
        host.restored = True
        self.network.pending_restores.add(host.name)
        return BlueActionReturn(host.name, True, -1, target=host.name)
