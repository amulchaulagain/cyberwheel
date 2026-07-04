from __future__ import annotations
from cyberwheel.red_actions import art_techniques
from cyberwheel.red_actions.actions import (
    ARTDiscovery,
    ARTLateralMovement,
    ARTPrivilegeEscalation,
    ARTImpact,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cyberwheel.network.host import Host
    from cyberwheel.network.network_base import Network

# Hosts share a small set of (os, cve_list) profiles (one per host type), so
# the technique-validity scan is memoized on that profile instead of being
# recomputed per host. Keyed content-wise (not by HostType identity) because
# the network builder creates a fresh HostType object per host.
_VALIDITY_CACHE: dict = {}


def get_valid_techniques_by_host(host: Host, kcps) -> dict:
    """
    Returns the service mapping for one host: which ART technique ids are
    valid for it, per killchain phase in ``kcps``.
    """
    key = (host.os, tuple(kcps), frozenset(host.host_type.cve_list))
    cached = _VALIDITY_CACHE.get(key)
    if cached is None:
        cached = {}
        for kcp in kcps:
            valid = []
            for mid in kcp.validity_mapping[host.os][kcp.get_name()]:
                technique = art_techniques.technique_mapping[mid]
                if host.host_type.cve_list & technique.cve_list:
                    valid.append(mid)
            cached[kcp] = valid
        _VALIDITY_CACHE[key] = cached
    # Fresh outer dict per host so callers can't cross-link hosts; the
    # per-phase lists are shared and treated as read-only by all consumers.
    return dict(cached)


def get_service_map(network: Network):
    """
    Function to get the service mapping based on host attributes.
    """
    killchain = [
        ARTDiscovery,
        ARTPrivilegeEscalation,
        ARTImpact,
        ARTLateralMovement,
    ]
    service_mapping = {}
    for host in network.hosts.values():
        service_mapping[host.name] = get_valid_techniques_by_host(host, killchain)
    return service_mapping
