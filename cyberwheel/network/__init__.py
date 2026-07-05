# Import utils first to resolve the import-order cycle (utils -> get_service_map
# -> red_actions -> detectors.alert -> network.host). A no-op on every path
# where utils is already loaded; only matters when this package is imported
# first (e.g. `python -m cyberwheel.network.network_generation`).
import cyberwheel.utils  # noqa: F401

from . import network_base, network_object, router, subnet, host, service
