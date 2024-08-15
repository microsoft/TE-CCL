from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class Mesh(Topology):
    def __init__(self, topo_input: TopologyParams):
        super().__init__(topo_input)

    def construct_topology(self, topo_input: TopologyParams):
        self.node_per_chassis = self.side_length ** 2

        speed = 100 / self.chunk_size
        # Each node is only connected to its neighbors
        self.capacity = []
        for i in range(self.node_per_chassis):
            row = []
            for j in range(self.node_per_chassis):
                if j == i - 1 or j == i + 1 or j == i - self.side_length or j == i + self.side_length:
                    row.append(speed)
                else:
                    row.append(0)
            self.capacity.append(row)

        self.alpha = []
        for r in self.capacity:
            row = []
            for i in r:
                if i:
                    row.append(topo_input.alpha[0])
                else:
                    row.append(-1)
            self.alpha.append(row)

    def set_switch_indicies(self) -> None:
        super().set_switch_indicies()
