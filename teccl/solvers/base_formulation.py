from collections import defaultdict
import logging
import math
from abc import ABC, abstractmethod
from itertools import product
from typing import List

import gurobipy as gp
import numpy as np
from teccl.input_data import *
from teccl.topologies.topology import Topology


class BaseFormulation(ABC):
    @abstractmethod
    def __init__(self, user_input: UserInputParams, topology: Topology) -> None:
        """
            Base class for all the solvers which creates the model and the variables common across solvers.
            Sets the epoch duration and the number of epochs using the user input and the topology.
        """
        self.user_input = user_input
        self.solver_name = ""
        self.topology = topology
        self.model = gp.Model('Base')

        if user_input.instance.collective == Collective.ALLGATHER:
            self.all_gather_demand_generator()

        elif user_input.instance.collective == Collective.ALLTOALL:
            self.all_to_all_demand_generator()
        else:
            raise ValueError(
                f"Demand type {user_input.instance.collective} is not expected")

        self.num_nodes = len(self.topology.capacity)
        self.num_chunks = self.user_input.instance.num_chunks

        self.nodes = list(range(self.num_nodes))
        self.chunks = list(range(self.num_chunks))
        self.aux_var: List[dict] = []

        self.set_epoch_duration()
        self.set_num_epochs()

    def set_epoch_duration(self) -> None:
        """
            Sets the epoch duration based on the user input and the topology.
            If the user input is not -1, then the epoch duration is set to the user input.
            If the user input is -1, then the epoch duration is set to the fastest or slowest link in the topology,
                which means the time it takes for one chunk to go on the fastest or slowest link.
        """
        def alpha_check():
            """
                Checks if the alpha_epoch_duration_ratio_max is too large in which case the epoch duration is increased
                    to avoid large models. (Solution quality is not affected much as alpha dominates the collective finish time).           
            """
            min_alpha = self.topology.get_min_alpha()
            alpha_epoch_ratio = min_alpha / self.epoch_duration
            if alpha_epoch_ratio > self.user_input.instance.alpha_epoch_duration_ratio_max:
                logging.warning(f"alpha_epoch_ratio is too large. "
                                f"alpha_epoch_ratio = {alpha_epoch_ratio}. "
                                f"changing epoch_duration to alpha_epoch_ratio in the user input = {self.user_input.instance.alpha_epoch_duration_ratio_max}")
                self.epoch_duration = min_alpha / \
                    self.user_input.instance.alpha_epoch_duration_ratio_max
        if self.user_input.instance.epoch_duration != -1:
            self.epoch_duration = self.user_input.instance.epoch_duration
        elif self.user_input.instance.epoch_type == EpochType.FASTEST_LINK:
            self.epoch_duration = self.topology.epoch_duration_fast_link * \
                self.user_input.instance.epoch_multiplier
            assert self.epoch_duration > 0, "Epoch Multiplier in the user input is not positive"
        elif self.user_input.instance.epoch_type == EpochType.SLOWEST_LINK:
            self.epoch_duration = self.topology.epoch_duration_slow_link * \
                self.user_input.instance.epoch_multiplier
            assert self.epoch_duration > 0, "Epoch Multiplier in the user input is not positive"
        else:
            raise ValueError(
                f"Using epoch_type {EpochType.USER_INPUT} but epoch_duration is not set")
        
        self.expected_epoch_duration = self.epoch_duration
        alpha_check()

    def set_num_epochs(self, epochs=100) -> None:
        if self.user_input.instance.num_epochs != -1:
            assert self.user_input.instance.num_epochs > 0, "Number of epochs in the user input is not positive"
            self.num_epochs = self.user_input.instance.num_epochs
        else:
            self.num_epochs = epochs
        self.epochs = list(range(self.num_epochs))

    def get_alpha_num_back(self, i: int, j: int) -> int:
        """
            The number of epochs required for a chunk to travel on the link (i,j) taking
            into account the link propagation delay.
            If the link has a propogation delay of 1 us, and the epoch is 0.5 us, then the chunk
            takes 2 additional epochs to reach j.
        """
        link_alpha = self.topology.alpha[i][j]
        if (link_alpha / self.epoch_duration) > self.user_input.instance.alpha_threshold:
            return math.ceil(
                link_alpha / self.epoch_duration)
        else:
            return 0

    def get_beta_num_back(self, i: int, j: int) -> int:
        """
            The number of extra epochs required for a chunk to travel on the link (i,j).
            If the link capacity is >= 1 chunk/sec, then there is no beta_num_back as the chunk can travel in one epoch.
            If the link capacity is < 1 chunk/sec, then the chunk needs to travel in multiple epochs.
            For example, if the link capacity is 0.5 chunk/sec, then the chunk needs to travel in 2 epochs.
            Since we account for one epoch implicity, this function returns 2-1 = 1 as the extra epoch.
        """
        epoch_capacity = self.topology.capacity[i][j] * self.epoch_duration
        return max(0, math.ceil(1 / epoch_capacity) - 1)

    def compute_floyd_warshall(self) -> None:
        """
            Computes the shortest path between all pairs of nodes in the network,
            where the distance between two nodes is the number of epochs it
            takes for a chunk to travel between them.
        """
        INF = float("inf")
        # epoch_distance is the number of epochs it takes for a chunk to cross the link
        epoch_distance = []
        for i, row in enumerate(self.topology.capacity):
            dist_row = []
            for j, c in enumerate(row):
                if c > 0:
                    epochs_for_one_chunk = 1 / (c * self.epoch_duration)
                    alpha_epochs = 0
                    if (self.topology.alpha[i][j] / self.epoch_duration) > self.user_input.instance.alpha_threshold:
                        alpha_epochs = math.ceil(
                            self.topology.alpha[i][j] / self.epoch_duration)
                    total_epochs = epochs_for_one_chunk + alpha_epochs
                    dist_row.append(total_epochs)
                else:
                    dist_row.append(INF)
            epoch_distance.append(dist_row)
        n = len(self.topology.capacity)

        for k, i, j in product(range(n), repeat=3):
            epoch_distance[i][j] = min(
                epoch_distance[i][j], epoch_distance[i][k] + epoch_distance[k][j])

        self.floyd_warshall = epoch_distance

    class LinkType(Enum):
        GPU_GPU = 1
        SWITCH_GPU = 2
        GPU_SWITCH = 3
        SWITCH_SWITCH = 4

    def get_link_type(self, i, j) -> LinkType:
        if i not in self.topology.switch_indices and j not in self.topology.switch_indices:
            return self.LinkType.GPU_GPU
        elif i not in self.topology.switch_indices and j in self.topology.switch_indices:
            return self.LinkType.GPU_SWITCH
        elif i in self.topology.switch_indices and j in self.topology.switch_indices:
            return self.LinkType.SWITCH_SWITCH
        elif i in self.topology.switch_indices and j not in self.topology.switch_indices:
            return self.LinkType.SWITCH_GPU
        else:
            raise ValueError("Invalid link type")

    def all_gather_demand_generator(self) -> None:
        gpus = len(self.topology.capacity)
        chunks = self.user_input.instance.num_chunks
        self.demand = np.zeros((gpus, gpus, chunks), dtype=np.int32)
        for s in range(gpus):
            for t in range(gpus):
                for c in range(chunks):
                    if s == t:
                        continue
                    if s in self.topology.switch_indices or t in self.topology.switch_indices:
                        continue
                    self.demand[s][t][c] = 1

    def all_to_all_demand_generator(self) -> None:
        devices = len(self.topology.capacity)
        chunks_per_gpu = self.user_input.instance.num_chunks
        self.demand = np.zeros((devices, devices, chunks_per_gpu), dtype=np.int32)
        device_chunk_map = defaultdict(int)
        i = 0
        for d in range(devices):
            if d in self.topology.switch_indices:
                continue
            device_chunk_map[d] = i
            i += 1
        for s in range(devices):
            for t in range(devices):
                if s == t:
                    continue
                # the switch should not be sending/recieving chunks.
                if s in self.topology.switch_indices or t in self.topology.switch_indices:
                    continue
                gpus = devices - len(self.topology.switch_indices)
                for c in range(chunks_per_gpu // gpus):
                    if s != t:
                        self.demand[s][t][device_chunk_map[t] + c * gpus] = 1
        
    def set_gurobi_params(self) -> None:
        self.model.Params.OutputFlag = self.user_input.gurobi.output_flag
        self.model.Params.TimeLimit = self.user_input.gurobi.time_limit * 60 * 60
        self.model.Params.FeasibilityTol = self.user_input.gurobi.feasibility_tol
        self.model.Params.IntFeasTol = self.user_input.gurobi.intfeas_tol
        self.model.Params.OptimalityTol = self.user_input.gurobi.optimality_tol
        self.model.Params.MIPGap = self.user_input.gurobi.mip_gap
        self.model.Params.Crossover = self.user_input.gurobi.crossover
        self.model.Params.MIPFocus = self.user_input.gurobi.mip_focus
        self.model.Params.Method = self.user_input.gurobi.method
        self.model.Params.Heuristics = self.user_input.gurobi.heuristics
        self.model.Params.Presolve = self.user_input.gurobi.presolve
        self.model.Params.SolutionLimit = self.user_input.gurobi.solution_limit
        # self.model.Params.NoRelHeurTime = 1200
        self.model.Params.PreSOS1BigM = 1e6
        # self.model.Params.Cuts = 1
        # self.model_.Params.RINS = 5000
        # self.model_.Params.Threads = 80

    @abstractmethod
    def get_schedule(self) -> None:
        pass
