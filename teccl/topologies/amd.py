from teccl.input_data import TopologyParams
from teccl.topologies.topology import Topology


class AMD(Topology):
    def __init__(self, topo_input: TopologyParams):
        super().__init__(topo_input)

    def construct_topology(self, topo_input: TopologyParams):
        self.node_per_chassis = 16

        gpu_link = 50 / self.chunk_size
        switch_link_capacity = 25.0 / self.chunk_size

        # 16 devices per chassis
        adjacency_list = {}
        adjacency_list[0] = {1: 4, 4: 2, 8: 1}
        adjacency_list[1] = {0: 4, 5: 1, 9: 1, 10: 1}
        adjacency_list[2] = {3: 4, 6: 1, 10: 1, 9: 1}
        adjacency_list[3] = {2: 4, 7: 2, 11: 1}
        adjacency_list[4] = {6: 1, 5: 4, 0: 2}
        adjacency_list[5] = {1: 1, 4: 4, 7: 1, 6: 1}
        adjacency_list[6] = {4: 1, 7: 4, 2: 1, 5: 1}
        adjacency_list[7] = {5: 1, 6: 4, 3: 2}
        adjacency_list[8] = {0: 1, 9: 4, 12: 2}
        adjacency_list[9] = {1: 1, 2: 1, 8: 4, 13: 1}
        adjacency_list[10] = {1: 1, 2: 1, 11: 4, 14: 1}
        adjacency_list[11] = {3: 1, 10: 4, 15: 2}
        adjacency_list[12] = {8: 2, 13: 4, 14: 1}
        adjacency_list[13] = {9: 1, 12: 4, 15: 1, 14: 1}
        adjacency_list[14] = {10: 1, 12: 1, 13: 1, 15: 4}
        adjacency_list[15] = {11: 2, 13: 1, 14: 4}

        devices_consider = 16 # Consider 16 devices in a chassis. If the number is less than 16, a subset of the devices will be considered
        single_capacity = [
            [0 for _ in range(devices_consider)] for _ in range(devices_consider)]
        for i in range(devices_consider):
            for j, w in adjacency_list[i].items():
                if j < devices_consider:
                    single_capacity[i][j] = w * gpu_link
        if topo_input.chassis == 1:
            self.capacity = single_capacity
            for r in self.capacity:
                row = []
                for i in r:
                    if i:
                        row.append(topo_input.alpha[0])
                    else:
                        row.append(-1)
                self.alpha.append(row)
            return

        #  for every two nodes there is a switch
        for i in range(8):
            adjacency_list[16 + i] = {2 * i: 1, 2 * i + 1: 1}
            adjacency_list[2 * i][16 + i] = 1
            adjacency_list[2 * i + 1][16 + i] = 1

        # Each chassis has 16 + 8 devices
        single_chassis_device_count = 24
        single_capacity = [[0 for _ in range(
            single_chassis_device_count)] for _ in range(single_chassis_device_count)]
        for i in range(single_chassis_device_count):
            for j, w in adjacency_list[i].items():
                if i > 16 or j > 16:
                    single_capacity[i][j] = w * switch_link_capacity
                else:
                    single_capacity[i][j] = w * gpu_link

        capacity = [[0.0] * single_chassis_device_count *
                    topo_input.chassis + [0.0]]
        # Row/column 0 is the switch interconnecting the chassis
        self.switch_indices = []
        for i in range(topo_input.chassis):
            for j in single_capacity:
                cap = [0.0] * single_chassis_device_count * topo_input.chassis
                for k in range(single_chassis_device_count):
                    cap[single_chassis_device_count * i + k] = j[k]
                capacity.append([0.0] + cap)
            for k in range(8):
                self.switch_indices.append(
                    single_chassis_device_count * i + 16 + k)

        # connect small switches to the main interconnecting switch
        for switch_index in self.switch_indices:
            capacity[switch_index][0] = switch_link_capacity
            capacity[0][switch_index] = switch_link_capacity

        # add the main switch to the list of switches
        self.switch_indices.append(0)

        self.capacity = capacity
        self.alpha = []
        for r in capacity:
            row = []
            for i in r:
                if i:
                    if i == switch_link_capacity:
                        row.append(topo_input.alpha[1])
                    else:
                        row.append(topo_input.alpha[0])
                else:
                    row.append(-1)
            self.alpha.append(row)

    def set_switch_indicies(self) -> None:
        # Implemented in construct_topology
        pass
