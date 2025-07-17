"""This plugin emulates the Cage 2 Challenge Scenario"""

import yaml
from netaddr import IPNetwork
from base_objects import Switch


# from linux.ubuntu1604 import Ubuntu1604Server, Ubuntu1604Desktop
from linux.ubuntu1604cage import (
    Ubuntu1604Server,
    Ubuntu1604Desktop,
    Ubuntu1604DesktopSiem,
)
from vyos.helium118 import Helium118
from firewheel.control.experiment_graph import AbstractPlugin, Vertex

# Not working...
# from linux.ubuntu1804 import Ubuntu1804Server, Ubuntu1804Desktop
# from windows.windows_server_2008_r2 import WindowsServer2008R2


def read_config(config: str):
    """Read network config from YAML file"""

    with open(config, "r", encoding="utf-8") as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
        return data


class Plugin(AbstractPlugin):
    """cage.topology plugin documentation."""

    def run(self):
        """Run method documentation."""

        # must be in directory for executing experiment
        config = read_config("example_config.yaml")

        # Create an external-facing network
        # self.external_network = IPNetwork("1.0.0.0/24")
        # Create an external-facing network iterator
        # external_network_iter = self.external_network.iter_hosts()

        # Create an internal facing network
        core_router = config.get("routers").get("core_router")
        core_router_networks = core_router.get("routes")[0].get("dest")
        # internal_networks = IPNetwork("10.0.0.0/8")
        internal_networks = IPNetwork(core_router_networks)

        # Break the internal network in to various subnets
        self.internal_subnets = internal_networks.subnet(24)

        # Create an internal switch
        internal_switch = Vertex(self.g, name="CYBERWHEEL-INTERNAL")
        internal_switch.decorate(Switch)

        # Grab a subnet to use for connections to the internal switch
        internal_switch_network = next(self.internal_subnets)
        # Create a generator for the network
        internal_switch_network_iter = internal_switch_network.iter_hosts()

        # Build Subnets
        subnets = config.get("subnets")
        for name, values in subnets.items():
            name = name.replace("_", "-")

            # Create subnets
            subnet_router = self.build_subnet(
                name, next(self.internal_subnets), num_hosts=1
            )

            # Create subnet to internal switch
            subnet_router.ospf_connect(
                internal_switch,
                next(internal_switch_network_iter),
                internal_switch_network.netmask,
            )

        # Create User Hosts (Subnet 1)
        # user_subnet_router = self.build_user_subnet(
        #     "user", next(self.internal_subnets), num_hosts=5
        # )

        # Connect User Hosts to internal switch
        # user_subnet_router.ospf_connect(
        #     internal_switch,
        #     next(internal_switch_network_iter),
        #     internal_switch_network.netmask,
        # )

        # Create the Firewall and Enterprise Hosts subnet (Subnet 2)
        # enterprise_subnet_firewall = self.build_enterprise_subnet(
        #     "enterprise", next(self.internal_subnets), num_hosts=3
        # )

        # Create User Hosts (Subnet 3)
        # op_subnet_router = self.build_operational_subnet(
        #     "operation", next(self.internal_subnets), num_hosts=3
        # )

        # Connect firewall to internal switch
        # enterprise_subnet_firewall.ospf_connect(
        #     internal_switch,
        #     next(internal_switch_network_iter),
        #     internal_switch_network.netmask,
        # )

        # Connect Operation Hosts to internal switch
        # op_subnet_router.ospf_connect(
        #     internal_switch,
        #     next(internal_switch_network_iter),
        #     internal_switch_network.netmask,
        # )

    def build_subnet(self, name: str, network: IPNetwork, num_hosts: int = 1):
        """Build subnet

        Args:
            name (str): the name of the user hosts subnet.
            network (netaddr.IPNetwork): the subnet for the user hosts.
            num_hosts (int): the number of hosts the subnet should have.

        Returns:
        vyos.Helium118: The subnet router.
        """

        # Create subnet router
        subnet_router = Vertex(self.g, name=f"{name}.cyberwheel.com")
        subnet_router.decorate(Helium118)

        # Create hosts switch
        subnet_switch = Vertex(self.g, name=f"{name}-switch")
        subnet_switch.decorate(Switch)

        # Create a network generator
        network_iter = network.iter_hosts()

        # Connet the router to the switch
        subnet_router.connect(subnet_switch, next(network_iter), network.netmask)

        # Redistributes routes directly connected subnets to OSPF peers.
        subnet_router.redistribute_ospf_connected()

        # Create hosts
        for i in range(num_hosts):
            # Create a new host which are Ubuntu Desktops
            host = Vertex(self.g, name=f"{name}-host-{i}.cyberwheel.com")
            host.decorate(Ubuntu1604Desktop)

            # Connect the host ot the switch
            host.connect(subnet_switch, next(network_iter), network.netmask)

        return subnet_router

    def build_user_subnet(self, name: str, network: IPNetwork, num_hosts: int = 1):
        """Create the first subnet which contains User Hosts

        This is a single router with all of the hosts.
        This is what the topology looks like:

            firewall ---- router ---- switch --- user hosts
        (enterprise-fw1)

        Args:
            name (str): the name of the user hosts subnet.
            network (netaddr.IPNetwork): the subnet for the user hosts.
            num_hosts (int): the number of hosts the subnet should have.

        Returns:
           vyos.Helium118: The subnet router.
        """

        # Create the VyOS router which will connect subnet-1 router to enterprise firewall
        user_router = Vertex(self.g, name=f"{name}.cage.com")
        user_router.decorate(Helium118)

        # Create the user hosts switch
        user_switch = Vertex(self.g, name=f"{name}-switch")
        user_switch.decorate(Switch)

        # Create a generator for user hosts network
        user_network_iter = network.iter_hosts()

        # Connet the router to the switch
        user_router.connect(user_switch, next(user_network_iter), network.netmask)

        # This redistributes routes for directly connected subnets to OSPF peers.
        # That is, enables these peers to be discoverable by the rest of the OSPF
        # routing infrastructure.
        user_router.redistribute_ospf_connected()

        # Create the correct number of hosts
        for i in range(num_hosts):
            # Create a new host which are Ubuntu Desktops
            host = Vertex(self.g, name=f"{name}-host-{i}.cage.com")
            host.decorate(Ubuntu1604Desktop)

            # Connect the host ot the switch
            host.connect(user_switch, next(user_network_iter), network.netmask)

        return user_router

    def build_enterprise_subnet(
        self, name: str, network: IPNetwork, num_hosts: int = 1
    ):
        """Create the second subnet which contains Enterprise Hosts

        The enterprise topology looks like:

                      Enterpirse Hosts
                           |
                           |
           ---- fw ---- switch ---- fw ---- subnet 3
                           |
                           |
                        defender

        Args:
            name (str): the name of the user hosts subnet.
            network (netaddr.IPNetwork): the subnet for the user hosts.
            num_hosts (int): the number of hosts the subnet should have.

        Returns:
            vyos.Helium118: the firewall object
        """

        # Build the gateway switch?

        # Build the firewall
        firewall = Vertex(self.g, "firewall.cage.com")
        firewall.decorate(Helium118)

        # Create a network and a generator for the network between
        # the gateway and firewall?.

        # Create the enterprise switch
        enterprise_switch = Vertex(self.g, name=f"{name}-switch")
        enterprise_switch.decorate(Switch)

        # Create a generator for enterprise hosts network
        enterprise_network_iter = network.iter_hosts()

        # Connet the firewall to the switch
        firewall.connect(
            enterprise_switch, next(enterprise_network_iter), network.netmask
        )

        # This redistributes routes for directly connected subnets to OSPF peers.
        # That is, enables these peers to be discoverable by the rest of the OSPF
        # routing infrastructure.
        firewall.redistribute_ospf_connected()

        # Create the defender host
        defender = Vertex(self.g, name="defender-host.cage.com")
        defender.decorate(Ubuntu1604Server)

        # Connect the defender
        defender.connect(
            enterprise_switch, next(enterprise_network_iter), network.netmask
        )

        # Create the siem host
        siem = Vertex(self.g, name="siem-host.cage.com")
        siem.decorate(Ubuntu1604DesktopSiem)

        # Connect the siem
        siem.connect(
            enterprise_switch,
            next(enterprise_network_iter),
            network.netmask
            # enterprise_switch, "10.0.2.3", network.netmask,
        )

        # Create the correct number of enterprise hosts
        for i in range(num_hosts):
            # Create a new host which are Ubuntu Desktops
            host = Vertex(self.g, name=f"{name}-host-{i}.cage.com")
            # host.decorate(WindowsServer2008R2)
            host.decorate(Ubuntu1604Server)

            # Connect the host ot the switch
            host.connect(
                enterprise_switch, next(enterprise_network_iter), network.netmask
            )

        return firewall

    def build_operational_subnet(
        self, name: str, network: IPNetwork, num_hosts: int = 1
    ):
        """Create the third subnet which contains Operational Hosts

        This is a single router with all of the hosts.
        This is what the topology looks like:

            firewall ---- router ---- switch --- operational hosts
        (enterprise-fw1)

        Args:
            name (str): the name of the user hosts subnet.
            network (netaddr.IPNetwork): the subnet for the user hosts.
            num_hosts (int): the number of hosts the subnet should have.

        Returns:
           vyos.Helium118: The subnet router.
        """

        # Create the VyOS router which will connect subnet-1 router to enterprise firewall
        op_router = Vertex(self.g, name=f"{name}.cage.com")
        op_router.decorate(Helium118)

        # Create the user hosts switch
        op_switch = Vertex(self.g, name=f"{name}-switch")
        op_switch.decorate(Switch)

        # Create a generator for user hosts network
        op_network_iter = network.iter_hosts()

        # Connet the router to the switch
        op_router.connect(op_switch, next(op_network_iter), network.netmask)

        # This redistributes routes for directly connected subnets to OSPF peers.
        # That is, enables these peers to be discoverable by the rest of the OSPF
        # routing infrastructure.
        op_router.redistribute_ospf_connected()

        # Create the operational server
        op_server = Vertex(self.g, name=f"{name}-server.cage.com")
        op_server.decorate(Ubuntu1604Server)
        op_server.connect(op_switch, next(op_network_iter), network.netmask)

        # Create the correct number of hosts
        for i in range(num_hosts):
            # Create a new host which are Ubuntu Desktops
            host = Vertex(self.g, name=f"{name}-host-{i}.cage.com")
            host.decorate(Ubuntu1604Desktop)

            # Connect the host ot the switch
            host.connect(op_switch, next(op_network_iter), network.netmask)

        return op_router
