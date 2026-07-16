import importlib
import yaml
import matplotlib.pyplot as plt
import networkx as nx

from importlib.resources import files
from pathlib import Path
from typing import Iterator

from cyberwheel.detectors.detector_base import Detector
from cyberwheel.detectors.alert import Alert

class DetectorHandler:
    def __init__(self, config: str) -> None:
        """
        - `config`: file name of the detector handler config file. Currently only YAML is supported.
        """
        self.config = config
        self._from_config()

    def _create_graph(self):
        self.DG = nx.DiGraph()
    
    def _from_config(self):
        self._create_graph()
        with open(self.config, "r") as r:
            contents = yaml.safe_load(r)
        
        adjacency_list = contents["adjacency_list"]
        init_info = contents["init_info"]  

        for entry in adjacency_list:
            node = entry[0]
            detector = None
            self.DG.add_node(node, detector_output=[])
            if node != 'start' and node != 'end':
                if node not in init_info: 
                    raise KeyError(f'node {node} not defined in init_info')
                detector = import_detector(init_info[node]['module'], init_info[node]['class'], init_info[node].get('config', None))
            for child in entry[1:]:
                self.DG.add_edge(node, child, attr={"detector": detector})


        # Start should have no in-edges and End should have not out-edges/
        # All other nodes should have at least 1 of both.
        for node, in_degree in self.DG.in_degree():
            if node == 'start' and in_degree > 0:
                raise ValueError("'start' node must have an in-degree of 0")
            elif node != 'start' and in_degree == 0:
                raise ValueError(f"node '{node}' must have an in-degree > 0")
            
        for node, out_degree in self.DG.out_degree():
            if node == 'end' and out_degree > 0:
                raise ValueError("'end' node must have an out-degree of 0")
            elif node != 'end' and out_degree == 0:
                raise ValueError(f"node '{node}' must have an out-degree > 0")

        return self.DG

    def obs(self, perfect_alerts: Iterator[Alert]) -> Iterator[Alert]:
        """
        Traverses the detector graph and executing each detector's `obs()` method.

        - `perfect_alerts`: an iterable of Alerts produced by the red agent. Used as input to the detector graph.
        """
        # Every node's detector_output list exists (set by _from_config/reset);
        # appending to it in place updates the graph attribute directly.
        nodes = self.DG.nodes
        for edge in self.DG.edges:
            next_node_input = nodes[edge[1]]["detector_output"]
            if edge[0] == 'start':
                result = perfect_alerts
            else:
                input_alerts = nodes[edge[0]]["detector_output"]
                detector = self.DG.get_edge_data(*edge)['attr']['detector']
                result = detector.obs(input_alerts)
            for r in result:
                if r not in next_node_input:
                    next_node_input.append(r)
        return nodes['end']["detector_output"]

    def reset(self) -> None:
        for _, data in self.DG.nodes(data=True):
            data["detector_output"] = []

    def reset_detectors(self) -> None:
        """Clear each detector's per-episode state. Called once per episode
        (distinct from ``reset()``, which clears node buffers every step)."""
        seen = set()
        for edge in self.DG.edges:
            attr = self.DG.get_edge_data(*edge).get("attr", {})
            detector = attr.get("detector")
            if detector is not None and id(detector) not in seen:
                seen.add(id(detector))
                detector.reset()

    def draw(self, filename="detector.png"):
        """
        Draws the detector graph.
        - `filename`: file to save the drawing of the detector graph to
        """
        plt.clf()  # clear
        colors = []
        for node in list(self.DG):
            if node == 'start':
                colors.append('lightgreen')
            elif node == 'end':
                colors.append('red')
            else:
                colors.append('lightblue')
        nx.draw(
            self.DG,
            node_size=300,
            font_size=12,
            font_color="black",
            font_weight="bold",
            edge_color="black",
            with_labels=True,
            node_color=colors,
        )
        plt.savefig(filename)


def import_detector(module: str, class_: str, config: str | dict | None) -> Detector:
    """
    Imports the specifed detector.

    - `module`: The module this detector is located in.

    - `class_`: The detector's class name.

    - `config`: Detector-specific config: an inline mapping is passed through
      as-is; a string names a YAML file, and a bare filename that doesn't exist
      as given is resolved against the packaged detector config directory
      (``cyberwheel/data/configs/detector``), so handler YAMLs can reference
      sibling files like ``nids.yaml`` regardless of the working directory.
    """
    import_path = ".".join(["cyberwheel.detectors.detectors", module])
    m = importlib.import_module(import_path)
    detector_type = getattr(m, class_)
    if isinstance(config, str) and not Path(config).exists():
        config = str(files("cyberwheel.data.configs.detector").joinpath(config))
    return detector_type(config) if config else detector_type()