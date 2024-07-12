from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class DGX1(Topology):
    def __init__(self, topo_input: TopologyParams):
        super().__init__(topo_input)

    def construct_topology(self, topo_input: TopologyParams):
        self.node_per_chassis = 8

        speed = 25 / self.chunk_size
        self.capacity = [
            # 0  1  2  3  4  5  6  7
            [0, 2 * speed, 1 * speed, 1 * speed, 2 * speed, 0, 0, 0],
            [2 * speed, 0, 1 * speed, 2 * speed, 0, 1 * speed, 0, 0],
            [1 * speed, 1 * speed, 0, 2 * speed, 0, 0, 2 * speed, 0],
            [1 * speed, 2 * speed, 2 * speed, 0, 0, 0, 0, 1 * speed],
            [2 * speed, 0, 0, 0, 0, 2 * speed, 1 * speed, 1 * speed],
            [0, 1 * speed, 0, 0, 2 * speed, 0, 1 * speed, 2 * speed],
            [0, 0, 2 * speed, 0, 1 * speed, 1 * speed, 0, 2 * speed],
            [0, 0, 0, 1 * speed, 1 * speed, 2 * speed, 2 * speed, 0]
        ]
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
