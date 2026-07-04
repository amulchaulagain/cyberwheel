"""Node-state codes shared by the visualization writer and its consumers.

A host's rendered state is the red agent's deepest killchain progress on it
(``KnownHostInfo`` flags), reduced to a single integer so per-step frames
stay compact. Higher code = further along the killchain; when several flags
are set the highest one wins.
"""

from __future__ import annotations

SAFE = 0
SWEEPED = 1
SCANNED = 2
DISCOVERED = 3
ESCALATED = 4
IMPACTED = 5

STATE_NAMES = ("safe", "sweeped", "scanned", "discovered", "escalated", "impacted")


def known_host_state(known_host) -> int:
    """Reduce a ``KnownHostInfo`` to its highest-precedence state code."""
    if known_host.impacted:
        return IMPACTED
    if known_host.escalated:
        return ESCALATED
    if known_host.discovered:
        return DISCOVERED
    if known_host.scanned:
        return SCANNED
    if known_host.sweeped:
        return SWEEPED
    return SAFE
