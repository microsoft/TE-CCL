from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class NDv2(Topology):
    def __init__(self, topo_input: TopologyParams):
        super().__init__(topo_input)
        self.node_per_chassis = 8

    def construct_topology(self, topo_input: TopologyParams):
        chassis = topo_input.chassis

        # 23 is beta in us/MB ==> link capacity is (10^-3 GB)/(23*(10^-6)s) = 43.47826 GB/s
        conversion_map = {}
        conversion_map[0] = 0
        conversion_map[23] = 43.47826 // self.chunk_size
        conversion_map[46] = 43.47826 // (2 * self.chunk_size)
        conversion_map[23] = 50 / self.chunk_size
        conversion_map[46] = 50 / (2 * self.chunk_size)
        # Beta is 105. 107 was used as an approximation of latency for routing stage.
        conversion_map[107] = 9.5238 // self.chunk_size
        conversion_map[107] = 12.5 / self.chunk_size
        self.switch_indices = []
        if chassis == 2:
            capacity = [
                [0, 23, 46, 46, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [23, 0, 46, 23, 0, 46, 0, 0, 107, 0, 0, 0, 0, 0, 0, 0],
                [46, 46, 0, 23, 0, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [46, 23, 23, 0, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 0],
                [23, 0, 0, 0, 0, 23, 46, 46, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 46, 0, 0, 23, 0, 46, 23, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 23, 0, 46, 46, 0, 23, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 46, 46, 23, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 46, 46, 23, 0, 0, 0],
                [107, 0, 0, 0, 0, 0, 0, 0, 23, 0, 46, 23, 0, 46, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 46, 46, 0, 23, 0, 0, 23, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 46, 23, 23, 0, 0, 0, 0, 46],
                [0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 0, 23, 46, 46],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 0, 0, 23, 0, 46, 23],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 0, 46, 46, 0, 23],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 46, 23, 23, 0]
            ]
        else:
            self.switch_indices = [0]
            single_capacity = [
                [0, 23, 46, 46, 23, 0, 0, 0],
                [23, 0, 46, 23, 0, 46, 0, 0],
                [46, 46, 0, 23, 0, 0, 23, 0],
                [46, 23, 23, 0, 0, 0, 0, 46],
                [23, 0, 0, 0, 0, 23, 46, 46],
                [0, 46, 0, 0, 23, 0, 46, 23],
                [0, 0, 23, 0, 46, 46, 0, 23],
                [0, 0, 0, 46, 46, 23, 23, 0]
            ]
            capacity = [[0] * 8 * chassis + [0]]
            # Row/column 0 is the switch
            for i in range(chassis):
                for j in single_capacity:
                    cap = [0] * 8 * chassis
                    for k in range(8):
                        cap[8 * i + k] = j[k]
                    capacity.append([0] + cap)
            for i in range(chassis):
                # 0 of every node is connected to switch
                capacity[0][i * 8 + 1] = 107
                # switch is connected to 1 of every node
                capacity[i * 8 + 2][0] = 107
        capacity = list(map(list, zip(*capacity)))
        self.capacity = [list(map(lambda x: conversion_map[x], r))
                          for r in capacity]
        self.topology = [list(map(lambda x: int(x > 0), r))
                          for r in self.capacity]
        self.alpha = []
        for r in capacity:
            row = []
            for i in r:
                if i:
                    if i == 107:
                        row.append(1.3 * pow(10, -6))
                    else:
                        row.append(0.7 * pow(10, -6))
                else:
                    row.append(-1)
            self.alpha.append(row)

    def set_switch_indicies(self) -> None:
        if self.chassis > 1:
            self.switch_indices = [0]
