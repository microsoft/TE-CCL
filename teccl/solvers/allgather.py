from collections import defaultdict
import logging
import math
import time
from itertools import product
from typing import Dict, List, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from teccl.input_data import *
from teccl.solvers.base_formulation import BaseFormulation
from teccl.topologies.topology import Topology


class AllGatherFormulation(BaseFormulation):
    def __init__(self, user_input: UserInputParams, topology: Topology) -> None:
        super().__init__(user_input, topology)
        self.solver_name = "AllGather_MILP"
        self.required_flows = []
        self.flows_str_info = {}

    def initialize_variables(self) -> None:
        """
            flow (s, i, j, c, k) : Whether chunk (s,c) is going on link (i,j) in epoch k.
            buffer (s, i, c, k) : Whether the chunk (s, c) is in the buffer at node i at beginning of epoch k.
            total_demand_sat (s, i, c, k):  Whether the demand for chunk (s,c) is satisfied at node i by the end of epoch k.
        """
        logging.debug(f'Initializing flow variables')
        time_start = time.time()
        self.flow = np.zeros((self.num_nodes, self.num_nodes,
                              self.num_nodes, self.num_chunks, self.num_epochs)).tolist()

        for i, j, in product(self.nodes, self.nodes):
            if self.topology.capacity[i][j] <= 0:
                continue
            alpha_num_back = self.get_alpha_num_back(i, j)
            link_type = self.get_link_type(i, j)
            beta_num_back = self.get_beta_num_back(i, j)
            extra_epochs = alpha_num_back + beta_num_back
            if link_type == self.LinkType.GPU_SWITCH and not self.user_input.instance.switch_to_gpu_link_on:
                # For GPU-Switch link, we don't need to send the chunk in the one more extra epoch too as the switch to GPU link is instantaneous and we need another epoch to account for this case.
                extra_epochs += 1
            not_necessary_k = [self.num_epochs -
                               i - 1 for i in range(extra_epochs)]
            for s, c, k in product(self.nodes, self.chunks, self.epochs):
                if s == j:
                    # Node j never need to receive its chunks from others
                    continue
                if link_type == self.LinkType.GPU_SWITCH and k in not_necessary_k:
                    continue
                if link_type == self.LinkType.GPU_GPU and k in not_necessary_k:
                    # For GPU-GPU link, we don't need to send the chunk in the last epochs as the chunk will not reach the destination.
                    continue
                # We leave the SWITCH_GPU link as it is as it links to the buffers on the GPUs and can not be set arbitarily.
                self.flow[s][i][j][c][k] = self.model.addVar(
                    0, 1, vtype=GRB.INTEGER, name='flow_%d_%d_%d_%d_%d' % (s, i, j, c, k))
        logging.debug(
            f'Finished initializing flow variables in {time.time()-time_start}')

        logging.debug(f'Initializing buffer and demand_sat variables')
        time_start = time.time()
        self.buffer = np.zeros((self.num_nodes, self.num_nodes,
                                self.num_chunks, self.num_epochs)).tolist()

        self.total_demand_sat = np.zeros(
            (self.num_nodes, self.num_nodes, self.num_chunks, self.num_epochs)).tolist()

        for s, i, c, k in product(self.nodes, self.nodes, self.chunks, self.epochs):
            self.buffer[s][i][c][k] = self.model.addVar(
                0, 1, vtype=GRB.INTEGER, name='buffer_%d_%d_%d_%d' % (s, i, c, k))
            if self.demand[s][i][c]:
                self.total_demand_sat[s][i][c][k] = self.model.addVar(
                    0, 1, vtype=GRB.INTEGER, name='total_demand_%d_%d_%d_%d' % (s, i, c, k))
        logging.debug(
            f'Finished initializing buffer and demand_sat variables in {time.time()- time_start}')

    def destination_constraints(self, astar_multiple_rounds: bool = False) -> None:
        """
            Adds constraints to the model for calculating the demand satisfied at each node after each epoch.
            When astar_multiple_rounds is False, it adds constraints that demand must be satisfied by the end of the last epoch.
        """
        logging.debug(f'Adding destination constraints')
        start = time.time()
        for s, d, c, k in product(self.nodes, self.nodes, self.chunks, self.epochs):
            if not self.demand[s][d][c]:
                continue
            if k + 1 < self.num_epochs:
                # The demand satsified by the end of the kth epoch is equal to the
                #   buffer at the beginning of k+1th epoch
                self.model.addConstr(self.total_demand_sat[s][d][c][k] == self.buffer[
                    s][d][c][k + 1], name="dest_total_s_%d_d_%d_c_%d_k_%d" % (s, d, c, k))
            else:
                # k = self.num_epochs - 1
                # For the last epoch B[K+1] does not exist
                # We consider the buffer at the beginning of last round and the flows in the last round that add to the buffer
                dem_sat_constr = gp.LinExpr(0.0)
                dem_sat_constr.add(self.buffer[s][d][c][k])
                for i in self.nodes:
                    if self.topology.capacity[i][d] > 0:
                        alpha_num_back = self.get_alpha_num_back(i, d)
                        link_type = self.get_link_type(i, d)
                        if link_type != self.LinkType.SWITCH_GPU or self.user_input.instance.switch_to_gpu_link_on:
                            beta_num_back = self.get_beta_num_back(i, d)
                            if k - alpha_num_back - beta_num_back >= 0:
                                dem_sat_constr.add(
                                    self.flow[s][i][d][c][k - alpha_num_back - beta_num_back])
                self.model.addConstr(dem_sat_constr == self.total_demand_sat[
                    s][d][c][k], name="dest_total_last_s_%d_d_%d_c_%d_k_%d" % (s, d, c, k))
                if not astar_multiple_rounds:
                    # At the end of the last epoch, the demand must be satisfied if it is not an A* round
                    self.model.addConstr(self.total_demand_sat[
                        s][d][c][k] == self.demand[s][d][c], name="dest_final_s_%d_d_%d_c_%d_k_%d" % (s, d, c, k))
        logging.debug(
            f'Finished adding destination constraints in {time.time() - start}')

    def node_constraint_helper(self, i: int, s: int, c: int, k: int, previous_round_buffers: List[List[int]]) -> None:
        """
            Adds contraints to model how the buffer changes with each epoch.
            Adds constraints to model what flows can arise from a node given the buffers.
        """
        buffer_constr = gp.LinExpr(0.0)
        if k == 0 and i != s:
            # The node i is not the source and the epoch is the first epoch.
            buffer_constr.add(self.buffer[s][i][c][k])
            from_last_round = 0
            if i not in self.topology.switch_indices and previous_round_buffers:
                # If the node is not a switch then it can have the chunk from the previous round into the buffer.
                # For a switch, buffer is always zero; the previous round flows are accounted in the flow constraints below.
                from_last_round = previous_round_buffers[s][i][c][0]
            self.model.addConstr(
                buffer_constr == from_last_round, name="node_first_buffer_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
        elif k == 0 and i == s:
            # max request for chunk C that originates from node s.
            dem_c = max([self.demand[s][d][c] for d in self.nodes])
            buffer_constr.add(self.buffer[s][s][c][0])
            # we have the chunks that need to be sent at s.
            self.model.addConstr(
                buffer_constr == dem_c, name="node_initial_buffer_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
        elif i not in self.topology.switch_indices:
            # constraints on the buffer contents for non-switches for all epochs other than first (k > 0) epoch
            buffer_constr.add(self.buffer[s][i][c][k - 1])
            for j in self.nodes:
                if self.topology.capacity[j][i] > 0:
                    alpha_num_back = self.get_alpha_num_back(j, i)
                    beta_num_back = self.get_beta_num_back(j, i)
                    link_type = self.get_link_type(j, i)
                    if link_type != self.LinkType.SWITCH_GPU or self.user_input.instance.switch_to_gpu_link_on:
                        if k - alpha_num_back - 1 - beta_num_back >= 0:
                            # -1 is required as Buffer[i] stands for buffer at the "beginning" of epoch i
                            buffer_constr.add(
                                self.flow[s][j][i][c][k - alpha_num_back - 1 - beta_num_back])
                    else:
                        # switch second link is ignored to remove double accounting of transmission delay
                        if k - alpha_num_back >= 0:
                            buffer_constr.add(
                                self.flow[s][j][i][c][k - alpha_num_back])
            if previous_round_buffers and k < len(previous_round_buffers[s][i][c]):
                if previous_round_buffers[s][i][c][k - 1] == 0 and previous_round_buffers[s][i][c][k] == 1:
                    # The chunk sent in the previous round arrived by the beginning of epoch k
                    # If previous k-1 is positive, it means the chunk has arrived before.
                    buffer_constr.add(previous_round_buffers[s][i][c][k])
            self.model.addConstr(buffer_constr == self.buffer[
                s][i][c][k], name="node_buffer_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
        elif i in self.topology.switch_indices:
            buffer_constr.add(self.buffer[s][i][c][k])
            self.model.addConstr(
                buffer_constr == 0, name="node_switch_buffer_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))

        # constraint on what can get out of the node in the current epoch
        if i not in self.topology.switch_indices:
            # For non-switch nodes, the buffer already acounts for previous rounds and flows.
            # A node can send in an epoch what is in the buffer at the start of the epoch.
            # Since a node can (copy) send the chunk from its buffer to multiple neighbors, we account for that
            #  using the max_ function. The maximum that a node can send to any of its neighbors is the buffer.
            self.aux_var.append(self.model.addVar(
                0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='aux_var_%d_%d_%d_%d' % (s, i, c, k)))
            self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.max_(
                [self.flow[s][i][v][c][k] for v in range(self.num_nodes) if self.topology.capacity[i][v] > 0]), name="node_aux_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))

            node_constr = gp.LinExpr(0.0)
            node_constr.add(self.buffer[s][i][c][k])
            self.model.addConstr(node_constr >= self.aux_var[
                len(self.aux_var) - 1], name="node_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
        else:
            # Switch can send whatever it got in the last epochs and round
            switch_node_constr = gp.LinExpr(0.0)
            if k > 0:
                for j in range(self.num_nodes):
                    if self.topology.capacity[j][i] > 0:
                        alpha_num_back = self.get_alpha_num_back(j, i)
                        beta_num_back = self.get_beta_num_back(j, i)
                        if k - alpha_num_back - 1 - beta_num_back >= 0:
                            switch_node_constr.add(
                                self.flow[s][j][i][c][k - alpha_num_back - 1 - beta_num_back])

            if previous_round_buffers and k < len(previous_round_buffers[s][i][c]):
                switch_node_constr.add(previous_round_buffers[s][i][c][k])

            if not self.user_input.instance.switch_copy:
                '''
                    switch no-copy version. For every outgoing chunk there has to be a corresponding incoming chunk.
                '''
                switch_outgoing_links = gp.LinExpr(0.0)
                for v in range(self.num_nodes):
                    if self.topology.capacity[i][v] > 0:
                        switch_outgoing_links.add(self.flow[s][i][v][c][k])
                # We allow the switch to drop the chunks with ">=" as we found it to be faster.
                # If we use "==" then in allgather case, it will result in no unnecessary flows.
                self.model.addConstr(switch_node_constr >= switch_outgoing_links,
                                     name="switch_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
            else:
                '''
                    switch copy version. The switch can copy the incoming chunk to multiple outgoing links.
                '''
                self.aux_var.append(self.model.addVar(
                    0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='aux_var_%d_%d_%d_%d' % (s, i, c, k)))
                self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.max_(
                    [self.flow[s][i][v][c][k] for v in range(self.num_nodes) if self.topology.capacity[i][v] > 0]), name="node_switch_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
                # We allow the switch to drop the chunks with ">=" as we found it to be faster.
                # If we use "==" then in allgather case, it will result in no unnecessary flows, which is necessary for the symmetric case.
                if self.user_input.instance.symmetry:
                    self.model.addConstr(switch_node_constr == self.aux_var[len(self.aux_var) - 1],
                                         name="switch_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))
                else:
                    self.model.addConstr(switch_node_constr >= self.aux_var[len(self.aux_var) - 1],
                                         name="switch_flow_s_%d_i_%d_c_%d_k_%d" % (s, i, c, k))

    def node_constraints(self, previous_buffers=None) -> None:
        logging.debug(f'Adding node buffer and flow constraints')
        start = time.time()
        for i, s, c, k in product(self.nodes, self.nodes, self.chunks, self.epochs):
            self.node_constraint_helper(i, s, c, k, previous_buffers)
        logging.debug(
            f'Finished adding node buffer and flow constraints in {time.time() - start}')

    def capacity_constraints(self) -> None:
        logging.debug(f'Adding capacity constraints')
        start = time.time()
        for i, j, k in product(self.nodes, self.nodes, self.epochs):
            if self.topology.capacity[i][j] <= 0:
                continue
            cap_constr = gp.LinExpr(0.0)
            epoch_capacity = self.topology.capacity[i][j] * self.epoch_duration
            beta_num_back = max(0, math.ceil(1 / epoch_capacity) - 1)
            if k - beta_num_back < 0:
                continue
            for l in range(beta_num_back + 1):
                for s, c in product(self.nodes, self.chunks):
                    if k - l >= 0:
                        cap_constr.add(self.flow[s][i][j][c][k - l])
            self.model.addConstr(
                cap_constr <= ((beta_num_back + 1) * epoch_capacity), name="capacity_i_%d_j_%d_k_%d" % (i, j, k))
        logging.debug(
            f'Finished adding capacity constraints in {time.time() - start}')

    def use_one_less_epoch(self) -> None:
        """
           Constraints to find the solution using one less epoch than the input number of epochs.
           This is required for the case when the switch to gpu link is off (instantaneous transfer).
           We have flow_K to know which node n is receiving the chunk from the switch, but the demand at
           node n is taken to be satisifed in the previous epoch K-1.
           For the iterative search, the solver needs extra epochs to find the solution at K-1, but it can
           also find the solution at K epochs as we stop the search when the solution count is 1.
        """
        if self.user_input.instance.switch_to_gpu_link_on:
            return
        logging.debug(
            f'Adding demand sat for switch to gpu link constraints to limit max epochs to 1 less than input')
        start = time.time()
        self.aux_var.append(self.model.addVar(
            0, GRB.INFINITY, vtype=GRB.CONTINUOUS, name='aux_var_switch_gpu'))
        required_demand_sats = []
        for s, d, c in product(self.nodes, self.nodes, self.chunks):
            if not self.demand[s][d][c]:
                continue
            required_demand_sats.append(
                self.total_demand_sat[s][d][c][self.num_epochs - 2])

        self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.min_(
            required_demand_sats), name="min_total_demand_sat_switch_gpu")
        self.model.addConstr(1 == self.aux_var[len(
            self.aux_var) - 1], name="min_total_demand_sat_switch_gpu_1")
        logging.debug(
            f'Finished adding demand sat for switch to gpu link constraints in {time.time() - start}')

    def add_symmetry_constraints(self) -> None:
        """
            Constraints to make the solution symmetric.
            If node i and j are symmetric, then the total outflow /inflow from i and j should be the same.
        """
        logging.debug(f'Adding symmetry constraints')
        start = time.time()
        for symmetry_nodes in self.topology.equivalent_node_indices:
            i = symmetry_nodes[0]
            for j in symmetry_nodes[1:]:
                flow_equality_out_constr = gp.LinExpr(0.0)
                flow_equality_in_constr = gp.LinExpr(0.0)
                for s, l, c, k in product(self.nodes, self.nodes, self.chunks, self.epochs):
                    flow_equality_out_constr.add(self.flow[s][i][l][c][k])
                    flow_equality_out_constr.add(self.flow[s][j][l][c][k], -1)
                    flow_equality_in_constr.add(self.flow[s][l][i][c][k])
                    flow_equality_in_constr.add(self.flow[s][l][j][c][k], -1)
                self.model.addConstr(
                    flow_equality_out_constr == 0, name="symmetry_out_%d_%d" % (i, j))
                self.model.addConstr(
                    flow_equality_in_constr == 0, name="symmetry_in_%d_%d" % (i, j))
        logging.debug(
            f'Finished adding symmetry constraints in {time.time() - start}')

    def objective_formulation(self, objective_type: ObjectiveType = ObjectiveType.PAPER) -> gp.LinExpr:
        logging.debug(f'Adding objective')
        start = time.time()
        # Gurobi minimizes the objective
        objective = gp.LinExpr(0.0)
        mulitplier = 100
        self.epoch_used = []

        demand_total = 0
        for s, d, c in product(self.nodes, self.nodes, self.chunks):
            if not self.demand[s][d][c]:
                continue
            demand_total += 1

        if objective_type == ObjectiveType.BINARY_USED_EPOCHS:
            # Minimize the number of epochs used using a binary variable for each epoch
            for k in self.epochs:
                self.epoch_used.append(self.model.addVar(
                    vtype=GRB.BINARY, name=f'epoch_used_{k}'))
                if len(self.epoch_used) > 1:
                    # If epoch k is not used then all the subsequent epochs also should not be used.
                    self.model.addConstr(
                        self.epoch_used[-2] >= self.epoch_used[-1], name=f'epoch_used_{k}-{k-1}')
                tmp = gp.LinExpr(0.0)
                # An epoch is used if there is some flow in that epoch
                # "some flow" --> the flow variables could all be zero, but the epoch can be used if the flow started
                # in a prior epoch and is still going through
                for i, j in product(self.nodes, self.nodes):
                    if self.topology.capacity[i][j] <= 0 or i in self.topology.switch_indices:
                        continue
                    beta_num_back = self.get_beta_num_back(i, j)
                    if k - beta_num_back < 0:
                        continue
                    for l in range(beta_num_back + 1):
                        for s, c in product(self.nodes, self.chunks):
                            if k - l >= 0:
                                tmp.add(self.flow[s][i][j][c][k - l])
                self.model.addConstr(
                    self.epoch_used[-1] * demand_total >= tmp, name=f'flows_sum_{k}')
                objective.add(self.epoch_used[-1])

        if objective_type == ObjectiveType.TOTAL_DEMAND:
            # Objective based on total demand: Σ max((Σtotal_demand_k - (Demand-1)), 0)
            # aux_var_obj1 = tmp = Σtotal_demand_k - (Demand-1)
            # aux_var_obj2 = max(aux_var_obj1, 0)
            for k in self.epochs:
                demand_total = 0
                tmp = gp.LinExpr(0.0)
                for s, d, c in product(self.nodes, self.nodes, self.chunks):
                    if not self.demand[s][d][c]:
                        continue
                    demand_total += 1
                    tmp.add(self.total_demand_sat[s][d][c][k])
                self.aux_var.append(self.model.addVar(
                    -demand_total, 1, vtype=GRB.CONTINUOUS, name='aux_var_obj1_%d_%d_%d_%d' % (s, d, c, k)))
                demand_total = demand_total - 1
                tmp.add(demand_total, -1)
                self.model.addConstr(self.aux_var[len(
                    self.aux_var) - 1] == tmp, name="obj1_s_%d_i_%d_c_%d_k_%d" % (s, d, c, k))
                self.aux_var.append(self.model.addVar(
                    0, 1, vtype=GRB.CONTINUOUS, name='aux_var_obj2_%d_%d_%d_%d' % (s, d, c, k)))
                self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.max_(
                    [self.aux_var[len(self.aux_var) - 2], 0]), name="obj2_s_%d_i_%d_c_%d_k_%d" % (s, d, c, k))
                objective.add(
                    self.aux_var[len(self.aux_var) - 1], (-mulitplier))

        if objective_type == ObjectiveType.PAPER:
            # Objective based on incremental progress
            # sum_end = self.num_nodes * 2 + self.num_chunks + self.num_epochs
            for s, d, c, k in product(self.nodes, self.nodes, self.chunks, self.epochs):
                if not self.demand[s][d][c]:
                    continue
                # d + 1 to create priority
                # my_contribution = 1 + ((s + d + c + k) / sum_end)
                # objective.add(self.total_demand_sat[
                #     s][d][c][k], -mulitplier * my_contribution * (self.user_input.instance.epsilon / (k + 1)))
                objective.add(self.total_demand_sat[s][d][c][k], -mulitplier)
                """
                  To disallow unnecessary flows, we can give a positive contribution to the objective for each flow.
                  This however affects the solver time significantly and unnecessary flows are pruned in post processing.
                """
                # for i in self.nodes:
                #     if self.topology.capacity[i][d] <= 0:
                #         continue
                #     # my_gamma_cont = (s + d + c + k + i) / (sum_end + self.num_nodes)
                #     objective.add(self.flow[s][i][d][c][k], mulitplier * 0.2)
        logging.debug(f'Finished adding objective in {time.time() - start}')
        return objective

    def encode_problem(self, use_one_less_epoch: bool = False, previous_buffers: List[List[int]] = []) -> int:
        setup_start = time.time()
        self.model = gp.Model('AllGather_MILP')
        self.initialize_variables()
        self.destination_constraints()
        self.node_constraints(previous_buffers)
        self.capacity_constraints()
        if use_one_less_epoch:
            self.use_one_less_epoch()
        if self.user_input.instance.symmetry:
            self.add_symmetry_constraints()
        self.model.setObjective(self.objective_formulation(
            self.user_input.instance.objective_type))

        log_file = f'Logs/{self.solver_name}_{self.user_input.topology.name}_{self.num_nodes}-nodes_' \
            f'{self.num_chunks}-chunks_{self.num_epochs}-epochs_{self.epoch_duration}-epochduration_{self.user_input.instance.objective_type}'

        if self.user_input.gurobi.output_flag == 1 or self.user_input.instance.debug:
            if self.user_input.gurobi.log_file:
                log_file += self.user_input.gurobi.log_file
            self.model.setParam("LogFile", log_file + ".log")
            self.model.Params.LogToConsole = 0
            # if self.user_input.instance.debug:
            #     self.model.write(log_file + ".lp")

        self.set_gurobi_params()
        setup_end = time.time()

        logging.debug(f'Total time for setup {setup_end - setup_start}')
        logging.debug(f'Epoch duration {self.epoch_duration}')
        logging.debug(f'Starting model optimization {log_file}')

        # warm start option
        # self.model.update()
        # error = self.model.read("<path to .sol file>")

        if self.user_input.instance.warmstart and self.user_input.instance.solution_method == SolutionMethod.ONE_SHOT:
            self.model.update()
            self.model.read(self.user_input.instance.warmstart)

        solve_start = time.time()
        self.model.optimize()
        solve_end = time.time()
        logging.debug(
            f'Finished model optimization {log_file} in {solve_end - solve_start} ')

        if self.model.Status != GRB.OPTIMAL:
            logging.warning(
                f"Not_Optimal_{self.solver_name}_Status-{self.model.Status}_{self.user_input.topology.name}_"
                f"{self.num_nodes}-nodes_{self.num_chunks}-chunks_{self.num_epochs}-epochs_{self.epoch_duration}-epochduration")
            if self.user_input.instance.debug and self.model.SolCount > 0:
                logging.debug(
                    f"Epoch at the end of which all demands are satisfied: {self.find_demand_satisfied_k() + 1}")
                # self.model.write(log_file + '.sol')
            # compute an Irreducible Inconsistent Subsystem
            # https://www.gurobi.com/documentation/10.0/refman/py_model_computeiis.html
            # else:
            #     self.model.computeIIS()
            #     self.model.write(log_file + '_unsat.ilp')
            return self.model.Status

        if self.user_input.instance.debug:
            # self.model.write(log_file + '.sol')
            logging.debug(
                f"Epoch at the end of which all demands are satisfied: {self.find_demand_satisfied_k() + 1}")
        return self.model.Status

    def get_flows_buffer_demand(self) -> Tuple[List[Tuple[int, int, int, int, int]], Dict[Tuple[int, int, int], int], Dict[Tuple[int, int, int], int], int]:
        """
            After the model is solved, this function extracts the non-zero flows, non-zero buffers,
                the epoch at which each demand is met and the epoch in which final demand is met.
        """
        demand_met_epoch = {}
        list_of_sends = []
        buffers = {}
        for v in self.model.getVars():
            # Gurobi is not setting flows to be perfectly zero, so we check to ensure it is over 0.9.
            if 'flow' in v.varName and v.x > 0.9:
                if 'future' in v.varName:
                    continue
                components = v.varName.split('_')
                _, s_t, i, j, c, k = components
                list_of_sends.append(
                    (int(s_t), int(i), int(j), int(c), int(k)))
            if 'total_demand_' in v.varName and v.x > 0.9:
                components = v.varName.split('_')
                _, _, s, i, c, k = components
                s, i, c = int(s), int(i), int(c)
                if (s, i, c) in demand_met_epoch:
                    if demand_met_epoch[(s, i, c)] > int(k):
                        demand_met_epoch[(s, i, c)] = int(k)
                else:
                    demand_met_epoch[(s, i, c)] = int(k)
            if 'buffer_' in v.varName and v.x > 0.9:
                if 'buffer_ahead_' in v.varName:
                    continue
                components = v.varName.split('_')
                _, s, i, c, k = components
                s, i, c = int(s), int(i), int(c)
                if (s, i, c) in buffers:
                    if buffers[(s, i, c)] > int(k):
                        buffers[(s, i, c)] = int(k)
                else:
                    buffers[(s, i, c)] = int(k)
        correct_k = max(demand_met_epoch.values())
        return (list_of_sends, demand_met_epoch, buffers, correct_k)

    def find_flow(self, flows: Set[Tuple[int, int, int, int, int]], s: int, d: int, c: int, k: int, flow_memoization: Dict) -> Tuple[int, int, int, int, int]:
        """
            Find the flow (sending node) that causes the node d to have the chunk by epoch k.
        """
        if (s, d, c, k) in flow_memoization:
            return flow_memoization[(s, d, c, k)]
        viable_flows = []
        # Loop over the neighbors to find such a feasible flow.
        for n in self.nodes:
            if self.topology.capacity[n][d] <= 0:
                continue
            alpha_num_back = self.get_alpha_num_back(n, d)
            beta_num_back = -1
            link_type = self.get_link_type(n, d)
            if link_type != self.LinkType.SWITCH_GPU or self.user_input.instance.switch_to_gpu_link_on:
                beta_num_back = self.get_beta_num_back(n, d)
            expected_flow = (s, n, d, c, k - alpha_num_back - beta_num_back)
            if expected_flow in flows:
                viable_flows.append(expected_flow)
        if d not in self.topology.switch_indices:
            assert len(
                viable_flows) == 1, f"There should only be one viable flow for demand {s} {d} {c} but have {len(viable_flows)}"
        # If d is a switch, then solver can pick a solution where the switch receives the same chunk by same epoch from multiple nodes.
        #  We only need one such flow if switch is copying the chunk.
        closest_flow = min(viable_flows, key=lambda x: x[4])
        flow_memoization[(s, d, c, k)] = closest_flow
        if not self.user_input.instance.switch_copy and d in self.topology.switch_indices:
            # we remove the chosen flow from the set of flows so that we don't pick it again as there is no copy.
            flows.remove(closest_flow)
        return closest_flow

    def chunk_flow_path_to_string(self, chunk_path: List[Tuple[int, int, int, int, int]]) -> Tuple[int, List[str]]:
        """
            Convert the input flow path to a string merging the flows that go through the switches.
        """
        path = []
        start = 0
        next = 0
        chunk_path = chunk_path[::-1]
        while start < len(chunk_path):
            accumulated_flows = [chunk_path[start]]
            # Find the flows that goes through the switches.
            while chunk_path[next][2] in self.topology.switch_indices:
                next += 1
                accumulated_flows.append(chunk_path[next])
            start_node = accumulated_flows[0][1]
            end_node = accumulated_flows[-1][2]
            sending_epoch = accumulated_flows[0][4]
            switches = " -> ".join([str(x[2]) for x in accumulated_flows[:-1]])
            if switches:
                path.append(
                    (sending_epoch, f'{start_node}->{end_node} in epoch {sending_epoch} via switches {switches}'))
            else:
                path.append(
                    (sending_epoch, f'{start_node}->{end_node} in epoch {sending_epoch}'))
            start = next = next + 1
        return path

    def dfs_remove_unnecessary_flows(self, astar: bool) -> Tuple[List[Tuple[int, int, int, int, int]], Dict]:
        """
            This function removes the unnecessary flows from the list of flows.
            Traces the path of the chunk from the destination to the source accumulating the flows required to satisfy the demand.
            All the flows that are not required are removed.
        """
        flows, demand_met_epoch, buffers, _ = self.get_flows_buffer_demand()
        logging.debug(f'Number of flows before pruning: {len(flows)}')
        if astar:
            # In an A* round, we can't say whether a flow is required or not until the end of all the rounds as
            #  the node receiving the chunk may be an intermediate node in the path to the destination in the later rounds.
            flows.sort(key=lambda x: x[4])
            flows_str_info = {}
            flows_str_info["1-Epoch_Duration"] = self.epoch_duration
            flows_str_info["2-Expected_Epoch_Duration"] = self.expected_epoch_duration
            flows_str_info["3-Epochs_Required"] = self.find_demand_satisfied_k() + 1
            flows_str_info["4-Collective_Finish_Time"] = flows_str_info["1-Epoch_Duration"] * flows_str_info["3-Epochs_Required"]
            flows_str_info["5-Algo_Bandwidth"] = self.topology.node_per_chassis * self.topology.chunk_size * self.topology.chassis / flows_str_info["4-Collective_Finish_Time"]
            flows_str_info['7-Flows'] = [
                f"Chunk {c} from {s} traveled over {i}->{j} in epoch {k}" for s, i, j, c, k in flows]
            return flows, flows_str_info
        required_flows = set()
        required_flows_str = set()
        flows = set(flows)
        chunk_paths = {}
        demand_met_str = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float)))
        # (s, d, c, k) -> (s, i, j, c, k). This is used to memoize the flow that causes the node d to have the chunk by epoch k.
        flow_memoization = {}
        for s, d, c in product(self.nodes, self.nodes, self.chunks):
            if not self.demand[s][d][c]:
                # If the node d did not request the chunk s c, then we can skip it.
                # If d is an intermediate node in the path to the destination, then the flows to d are
                #   accounted when we consider the flows to the destination.
                continue
            if (s, d, c) not in demand_met_epoch:
                logging.warning(f"Demand not satisfied for s_{s}, d_{d}, c_{c}")
                continue
            demand_met_str[f"GPU {d}"][f"GPU {s}"][f"Chunk {c}"] = self.epoch_duration * (
                demand_met_epoch[(s, d, c)] + 1)
            my_path = list()
            # We use demand_met as the last step may not have a buffer variable.
            demand_met_k = k = demand_met_epoch[(s, d, c)]
            # Find the flow that causes the node d to have the chunk by epoch k.
            closest_flow = self.find_flow(flows, s, d, c, k, flow_memoization)
            required_flows.add(closest_flow)
            my_path.append(closest_flow)
            #  Trace the path to the source node.
            while True:
                sending_node = closest_flow[1]
                if sending_node == s:
                    # we reached the chunk source node
                    break
                if sending_node not in self.topology.switch_indices:
                    # If the sending node is not a switch, then the epoch in which it received the chunk is given by buffer.
                    if (s, sending_node, c) not in buffers:
                        logging.error(
                            f"Buffer not found for s_{s}, sending_node_{sending_node}, c_{c}")
                        break
                    k = buffers[(s, sending_node, c)] - 1
                else:
                    # If the sending node is a switch, then it should have received the chunk in the previous epoch.
                    k = closest_flow[4] - 1
                closest_flow = self.find_flow(
                    flows, s, sending_node, c, k, flow_memoization)
                required_flows.add(closest_flow)
                my_path.append(closest_flow)
            chunk_str_path = self.chunk_flow_path_to_string(my_path)
            chunk_paths[f"Demand at {d} for chunk {c} from {s} met by epoch {demand_met_k}"] = [
                x[1] for x in chunk_str_path]
            for epoch, cpath in chunk_str_path:
                required_flows_str.add(
                    (epoch, f"Chunk {c} from {s} traveled over {cpath}"))

        logging.debug(f'Number of flows after pruning: {len(required_flows)}')
        required_flows = list(required_flows)
        required_flows.sort(key=lambda x: x[4])
        required_flows_str = list(required_flows_str)
        required_flows_str.sort(key=lambda x: x[0])
        flows_str_info = {}
        flows_str_info["1-Epoch_Duration"] = self.epoch_duration
        flows_str_info["2-Expected_Epoch_Duration"] = self.expected_epoch_duration
        flows_str_info["3-Epochs_Required"] = self.find_demand_satisfied_k() + 1
        flows_str_info["4-Collective_Finish_Time"] = flows_str_info["1-Epoch_Duration"] * flows_str_info["3-Epochs_Required"]
        flows_str_info["5-Algo_Bandwidth"] = self.topology.node_per_chassis * self.topology.chunk_size * self.topology.chassis / flows_str_info["4-Collective_Finish_Time"]
        flows_str_info["6-Demand_Met"] = demand_met_str
        flows_str_info['7-Flows'] = [x[1] for x in required_flows_str]
        flows_str_info['8-Chunk paths'] = chunk_paths
        return required_flows, flows_str_info

    def find_demand_satisfied_k(self):
        satisfied_epochs = {}
        for v in self.model.getVars():
            if 'total_demand_' in v.varName and v.x > 0.0:
                components = v.varName.split('_')
                _, _, s, i, c, k = components
                if (s, i, c) in satisfied_epochs:
                    if satisfied_epochs[(s, i, c)] > int(k):
                        satisfied_epochs[(s, i, c)] = int(k)
                else:
                    satisfied_epochs[(s, i, c)] = int(k)
        for s, i, c in product(self.nodes, self.nodes, self.chunks):
            if self.demand[s][i][c]:
                if (str(s), str(i), str(c)) not in satisfied_epochs:
                    logging.warning(
                        f"Demand not satisfied for s_{s}, i_{i}, c_{c}")
        return max(satisfied_epochs.values())

    def get_schedule(self) -> Tuple[List[Tuple[int, int, int, int, int]], Dict]:
        if self.model.SolCount > 0:
            if not self.required_flows:
                self.required_flows, self.flows_str_info = self.dfs_remove_unnecessary_flows(
                    astar=False)
            return self.required_flows, self.flows_str_info
        else:
            return [], {}

    def write_schedule_to_file(self, schedule: List[Tuple[int, int, int, int, int]], filename: str):
        with open(filename, 'w') as f:
            for s, i, j, c, k in schedule:
                f.write(
                    f'Chunk {c} from {s} traveled over {i}->{j} at epoch {k}\n')
