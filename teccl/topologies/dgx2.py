from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class DGX2(Topology):
    def __init__(self, topo_input: TopologyParams):
        super().__init__(topo_input)
        self.node_per_chassis = 16

    def construct_topology(self, topo_input: TopologyParams):
        switch_connections = [0] + [1] * 16
        gpu_connections = [1] + [0] * 16
        single_node = [gpu_connections for _ in range(16)]
        single_node.insert(0, switch_connections)

        chassis = 2
        total_nodes = 16 * chassis + chassis
        self.topology = []
        for i in range(chassis):
            for j in single_node:
                cap = [0] * total_nodes
                for k in range(17):
                    cap[17 * i + k] = j[k]
                self.topology.append(cap)
        link_capacity = 125 / self.chunk_size
        self.capacity = [list(map(lambda x: x * link_capacity, r))
                         for r in self.topology]
        self.alpha = [list(map(lambda x: x * 0.35 * pow(10, -6), r))
                      for r in self.topology]

        inter_node_link_capacity = 12.5 / self.chunk_size
        taccl = {"1": [0], "3": [2], "5": [
            4], "7": [6], "9": [8], "11": [10], "13": [12], "15": [14]}
        for i in range(chassis):
            for j in range(chassis):
                if i == j:
                    continue
                for s in taccl:
                    for r in taccl[s]:
                        src = int(s) + i * 17 + 1
                        dst = r + j * 17 + 1
                        self.topology[src][dst] = 1
                        self.capacity[src][dst] = inter_node_link_capacity
                        self.alpha[src][dst] = 2.6 * pow(10, -6)

    def set_switch_indicies(self) -> None:
        self.switch_indices = [17 * i for i in range(2)]
