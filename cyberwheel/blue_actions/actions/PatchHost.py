from typing import Dict

from cyberwheel.blue_actions.blue_action import BlueActionReturn, HostAction
from cyberwheel.network.host import Host
from cyberwheel.network.network_base import Network


class PatchHost(HostAction):
    """
    Patch all vulnerabilities on a real host. Red technique validity derives
    from host.host_type.cve_list, so the host gets its own deep copy of the
    (cached, shared-across-hosts) HostType before the CVE list is emptied —
    sibling hosts of the same type keep their vulnerabilities. The red agent
    recomputes its service_mapping entry via network.pending_patches; the
    original host_type is restored on network.reset().
    """

    def __init__(self, network: Network, configs: Dict[str, any], **kwargs) -> None:
        super().__init__(network, configs)

    def execute(self, host: Host, **kwargs) -> BlueActionReturn:
        if host.patched or host.host_type is None:
            return BlueActionReturn("", False, 0, target=host.name)
        if host._pre_patch_host_type is None:
            host._pre_patch_host_type = host.host_type
        host.host_type = host.host_type.copy(deep=True)
        host.host_type.cve_list = set()
        host.patched = True
        self.network.pending_patches.add(host.name)
        return BlueActionReturn(host.name, True, 0, target=host.name)
