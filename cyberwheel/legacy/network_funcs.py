
# TODO - This method is not working properly
def find_path_between_hosts(self, source_host, target_host):
    if source_host not in self.graph or target_host not in self.graph:
        return None  # Source or target not found in the network

    try:
        return nx.shortest_path(self.graph, source=source_host, target=target_host)
        # shortest_path = nx.shortest_path(self.graph, source=source_host, target=target_host)
        ##shortest_path = [item for item in shortest_path if "Router" not in item]

        ## Replace subnet names with host names on those subnets
        # new_path = []

        # for node in shortest_path:
        #    if isinstance(self.graph.nodes[node]['data'], Subnet):
        #    #if node.startswith('Subnet'):
        #        subnet_name = node
        #        # Try to find a connected node that starts with 'Host'
        #        connected_host = None
        #        for neighbor in self.graph.neighbors(subnet_name):
        #            if neighbor.startswith('Host'):
        #                connected_host = neighbor
        #                break

        #        if connected_host:
        #            new_path.append(connected_host)  # Replace subnet with connected host
        #        else:
        #            new_path.append(node)  # If no connected host found, keep the subnet
        #    else:
        #        # Keep non-subnet nodes unchanged
        #        new_path.append(node)

        # return new_path
    except:
        return None
# TODO: still need to test this
def is_any_subnet_fully_compromised(self):
    all_subnets = self.get_all_subnets()
    for subnet in all_subnets:
        subnet_hosts = self.get_all_hosts_on_subnet(subnet)
        if all(host.is_compromised for host in subnet_hosts):
            return True
    return False

# TODO: still need to test this
def set_host_compromised(self, host_id: str, compromised: bool):
    host_to_modify = self.get_node_from_name(host_id)
    host_to_modify.is_compromised = compromised

# TODO: should this be defined in the red actions?
def scan_subnet(self, src: Host, subnet: Subnet) -> dict:
    """
    Scans a given subnet and returns found IPs and open ports

    """
    all_hosts = self.get_all_hosts_on_subnet(subnet)
    for host in all_hosts:
        pass
    found_hosts = {}
    return found_hosts

# TODO: should this be defined in the red actions?
def scan_host(self, src: Host, ip: str) -> list:
    """
    Scans a given host and returns open ports
    """
    open_ports = []
    return open_ports

def ping_sweep_subnet(self, src: Host, subnet: Subnet) -> list:
    """
    Attempts to ping all hosts on a subnet

    Hosts are only visible to ping if ICMP is allowed by the firewall(s).
    """
    subnet_hosts = self.get_all_hosts_on_subnet(subnet)
    found_ips = []
    for host in subnet_hosts:
        if self.is_traffic_allowed(src, host, None, "icmp"):
            found_ips.append(host.ip_address)
    return found_ips

def find_host_with_longest_path(self, source_host):
        all_hosts = self.get_all_hosts()

        all_hosts.remove(source_host)  # Remove the source host from the list
        if not all_hosts:
            return None  # No other hosts in the network

        longest_path_length = -1
        target_host = None

        for host in all_hosts:
            path = self.find_path_between_hosts(source_host, host)
            if path is not None and len(path) > longest_path_length:
                longest_path_length = len(path)
                target_host = host

        return target_host

    # def generate_observation_vector(self):
    #     all_hosts = self.get_all_hosts()
    #     num_hosts = len(all_hosts)
    #     observation_vector = np.zeros(num_hosts, dtype=np.int8)

    #     index = 0
    #     for data_object in all_hosts:
    #         is_compromised = data_object.is_compromised
    #         observation_vector[index] = 1 if is_compromised else 0
    #         index += 1

# def define_routing_rules(self, router, routes):
    #    if router.name in self.graph.nodes:
    #        data_object = self.graph.nodes[router.name]['data']
    #        if isinstance(data_object, Router):
    #            data_object.routes = routes

    # def define_firewall_rules(self, router, firewall_rules):
    #    if router.name in self.graph.nodes:
    #        data_object = self.graph.nodes[router.name]['data']
    #        if isinstance(data_object, Router):
    #            data_object.firewall_rules = firewall_rules

    # def define_host_firewall_rules(self, host, firewall_rules):
    #    if host.name in self.graph.nodes:
    #        data_object = self.graph.nodes[host.name]['data']
    #        if isinstance(data_object, Host):
    #            data_object.firewall_rules = firewall_rules