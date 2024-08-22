import copy
import logging
import time
from collections import defaultdict
from itertools import product
from typing import Dict, List, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB
from teccl.input_data import *
from teccl.solvers.base_formulation import BaseFormulation
from teccl.topologies.topology import Topology


class AlltoAllFormulation(BaseFormulation):
    def __init__(self, user_input: UserInputParams, topology: Topology) -> None:
        super().__init__(user_input, topology)
        self.solver_name = "AllToAll_LP"

    def initialize_variables(self) -> None:
        """
          flow(s, i, j, k) : the fraction of data from source s that goes over link (i, j) in epoch k
          buffer(s, i, k) : the fraction of data from source s that is in the buffer at node i at epoch k
          total_demand_sat(s, i, k) : the amount of demand satisfied at node i till epoch k
          consumed_at_k(s, i, k) : fraction of demand coming from source s that is satisfied at node i in epoch k
        """

        logging.debug("starting to initialize the variables for alltoall")
        start_time = time.time()

        # compute the total demand across all nodes to set upper bound on flow variable.
        self.all_demand = 0
        for s, d in product(self.nodes, self.nodes):
            for c in range(len(self.demand[s][d])):
                self.all_demand += self.demand[s][d][c]

        # initialize flow variables
        self.flow = np.zeros(
            (self.num_nodes, self.num_nodes, self.num_nodes, self.num_epochs)).tolist()

        for i, j in product(self.nodes, self.nodes):
            if self.topology.capacity[i][j] < 0:
                continue
            for s in self.nodes:
                for k in self.epochs:
                    self.flow[s][i][j][k] = self.model.addVar(
                        0, self.all_demand, vtype=GRB.CONTINUOUS, name='f_%d_%d_%d_%d' % (s, i, j, k))
        logging.debug(
            f"Time for F (flows) initialization: {time.time() - start_time}")

        start_time = time.time()

        self.buffer = np.zeros(
            (self.num_nodes, self.num_nodes, self.num_epochs)).tolist()
        self.consumed_at_k = np.zeros(
            (self.num_nodes, self.num_nodes, self.num_epochs)).tolist()
        self.total_demand_sat = np.zeros(
            (self.num_nodes, self.num_nodes, self.num_epochs)).tolist()

        # calculate the demand at each node to set better variable limits.
        # total_demand_at_s : total volume of data that node s needs to send
        # demand_at_i : total volume of data node s needs to send to node i
        self.total_demand_at_s = defaultdict(float)
        self.demand_at_i = defaultdict(float)
        for s in self.nodes:
            node_demand = 0
            for d in self.nodes:
                for c in self.chunks:
                    node_demand += self.demand[s][d][c]
                    self.demand_at_i[(s, d)] += self.demand[s][d][c]
            self.total_demand_at_s[s] = node_demand
        logging.debug(
            f"Time for computing variable limits: {time.time() - start_time}")

        # initialize remaining variables
        start_time = time.time()
        for s, i, k in product(self.nodes, self.nodes, self.epochs):
            self.buffer[s][i][k] = self.model.addVar(
                0, self.total_demand_at_s[s], vtype=GRB.CONTINUOUS, name='B_%d_%d_%d' % (s, i, k))
            self.consumed_at_k[s][i][k] = self.model.addVar(
                0, self.demand_at_i[(s, i)], vtype=GRB.CONTINUOUS, name='T_%d_%d_%d' % (s, i, k))
            self.total_demand_sat[s][i][k] = self.model.addVar(
                0, self.demand_at_i[(s, i)], vtype=GRB.CONTINUOUS, name='t_%d_%d_%d' % (s, i, k))

    def destination_constraints(self) -> None:
        """
        add constraints to the model to account for how much of the demand is met at the end
        of each epoch.
        """
        for s, d, k in product(self.nodes, self.nodes, self.epochs):
            consume_constr = gp.LinExpr(0.0)
            # the demand met so far is the sum of all the chunk the node has consumed so far.
            if k > 0:
                consume_constr.add(self.total_demand_sat[s][d][k - 1])
            consume_constr.add(self.consumed_at_k[s][d][k])

            self.model.addConstr(
                consume_constr == self.total_demand_sat[s][d][k], name='total_demand_sat_%d_%d_%d' % (s, d, k))

            if k <  self.num_epochs - 1:
                # do not consume more than the demand.
                self.model.addConstr(consume_constr <= self.demand_at_i[(
                    s, d)], name='demand_constraint_%d_%d_%d' % (s, d, k))
            else:
                self.model.addConstr(consume_constr == self.demand_at_i[(
                    s, d)], name='full_demand_satisfiablility_%d_%d_%d' % (s, d, k))

    def node_constraint_helper(self, s: int, n: int, k: int) -> None:
        """
            Adds constraints on the buffers and adds flow conservation constraints.
            s : source node
            n : current node for which we are adding constraints
            k : epoch for which we are adding the constraints
        """

        # we need to account for the case where the number of epochs is 1 separately.
        if self.num_epochs > 1:
            if k == 0 and n != s:
                # nothing can go out of the node since there is nothing in the buffer of non-source nodes.
                buffer_constr = gp.LinExpr(0.0)
                buffer_constr.add(self.buffer[s][n][k])
                for j in self.nodes:
                    if self.topology.capacity[n][j] > 0:
                        buffer_constr.add(self.flow[s][n][j][k])
                self.model.addConstr(
                    buffer_constr == 0, name=f"Initial_flow_{s}-node_{n}-epoch_{k}")
            elif k == 0 and n == s:
                # the source node's buffer contains all the chunks s wants to send.
                # it can have flow at epoch 0 which is bounded by the contents of the buffer.
                buffer_constr = gp.LinExpr(0.0)
                buffer_constr.add(self.buffer[s][n][k])
                for j in self.nodes:
                    if self.topology.capacity[n][j] > 0:
                        buffer_constr.add(self.flow[s][n][j][k])
                self.model.addConstr(
                    buffer_constr == self.total_demand_at_s[n], name=f'EX_buffer-source_{s}-node_{n}')
        else:
            # if the Kmax is 1 then the middle constraints will interfere if we dont do this.
            if n != s:
                buffer_constr = gp.LinExpr(0.0)
                buffer_constr.add(self.buffer[s][n][k])
                self.model.addConstr(
                    buffer_constr == 0, f'Initial_flow_{s}-node_{n}-epoch_{k}')
            else:
                buffer_constr = gp.LinExpr(0.0)
                buffer_constr.add(self.buffer[s][n][k])
                self.model.addConstr(
                    buffer_constr == self.total_demand_at_s[n], name=f'Inital_flow_{s}-node_{n}')

        if n in self.topology.switch_indices:
            # if the node is a switch then we need to add a constraint that the buffer is 0.
            switch_buffer = gp.LinExpr(0.0)
            switch_buffer.add(self.buffer[s][n][k])
            self.model.addConstr(
                switch_buffer == 0, name=f'switch_constraint-source_{s}-node_{n}-epoch_{k}')

        # next we implement flow conservation constraints.
        if k + 1 < self.num_epochs and (n not in self.topology.switch_indices):
            flow_conservation = gp.LinExpr(0.0)
            flow_conservation.add(self.buffer[s][n][k])
            for j in self.nodes:
                if self.topology.capacity[j][n] > 0:
                    alpha_num_back = self.get_alpha_num_back(j, n)
                    if k - alpha_num_back >= 0:
                        flow_conservation.add(
                            self.flow[s][j][n][k - alpha_num_back])
                if self.topology.capacity[n][j] > 0:
                    flow_conservation.add(self.flow[s][n][j][k + 1], -1)
            flow_conservation.add(self.buffer[s][n][k + 1], -1)
            flow_conservation.add(self.consumed_at_k[s][n][k], -1)
            #  Buffer at the beginning of epoch k + flows that reach by end of epoch k >=
            #       buffer at the beginning of epoch (k + 1) + flows going out of the node in epoch (k + 1) + chunks portion consumed by the end of epoch k
            # TODO: this allows for flows getting dropped, we might want to just make it equality.
            self.model.addConstr(flow_conservation >= 0,
                                 name=f"midFC-epoch_{k}-node_{n}-source_{s}")

        elif k + 1 < self.num_epochs and n in self.topology.switch_indices:
            flow_conservation = gp.LinExpr(0.0)
            for j in self.nodes:
                if self.topology.capacity[j][n] > 0:
                    alpha_num_back = self.get_alpha_num_back(j, n)
                    if k - alpha_num_back >= 0:
                        flow_conservation.add(
                            self.flow[s][j][n][k - alpha_num_back])
                if self.topology.capacity[n][j] > 0:
                    flow_conservation.add(self.flow[s][n][j][k + 1], -1)
            # TODO: once again flow drop can happen here.
            self.model.addConstr(flow_conservation >= 0, name=f"midFC-switch-epoch_{k}-node_{n}-source_{s}")

        # last epoch flow conservation constraints.
        if k + 1 == self.num_epochs:
            if n not in self.topology.switch_indices and s != n:
                incoming = gp.LinExpr(0.0)
                for j in self.nodes:
                    if self.topology.capacity[j][n] > 0:
                        alpha_num_back = self.get_alpha_num_back(j, n)
                        if k - alpha_num_back >= 0:
                            incoming.add(self.flow[s][j]
                                         [n][k - alpha_num_back])
                incoming.add(self.consumed_at_k[s][n][k], -1)
                # All the flows that reach the node by the end of the last epoch should be consumed by the node as they can't be forwarded further.
                self.model.addConstr(
                    incoming >= 0, name=f'I_epoch_{k}-node_{n}-source_{s}')
            elif s == n or k == 0:
                flow_conservation = gp.LinExpr(0.0)
                flow_conservation.add(self.buffer[s][n][k])
                for j in self.nodes:
                    if self.topology.capacity[n][j] > 0:
                        flow_conservation.add(self.flow[s][n][j][k], -1)
                self.model.addConstr(flow_conservation >= 0,
                                     name=f"FC_epoch_{k}-node_{n}-source_{s}")

    def node_constraints(self) -> None:
        for n, s, k in product(self.nodes, self.nodes, self.epochs):
            self.node_constraint_helper(s, n, k)

    def capacity_constraints(self) -> None:
        """
        encodes capacity constraints.
        """

        # Uncomment this if you want to impose buffer limits.
        # if self.buffer_limit_ >= 0:
        #     for i in self.nodes_:
        #         for k in self.epochs_:
        #             buffer_constr = gp.LinExpr(0.0)
        #             for s in self.nodes_:
        #                 if s == i:
        #                     continue
        #                 buffer_constr.add(self.B_[s][i][k])
        #             self.model_.addConstr(
        #                 buffer_constr <= self.buffer_limit_, name=f"buffer_limit_constr_{i}_{k}")

        for i, j, k in product(self.nodes, self.nodes, self.epochs):
            if self.topology.capacity[i][j] <= 0:
                continue
            cap_constr = gp.LinExpr(0.0)
            for s in self.nodes:
                cap_constr.add(self.flow[s][i][j][k])
            self.model.addConstr(
                cap_constr <= (self.topology.capacity[i][j] * self.epoch_duration), name=f"cap_constr_link_{i}-{j}-{k}")

    def objective_formulation(self, objective_type: ObjectiveType = ObjectiveType.PAPER) -> gp.LinExpr:
        """
        returns the objective for the optimization.
        I've only implemented the paper and the total demand objective here.
        """
        objective = gp.LinExpr(0.0)
        multiplier = pow(10, 2)

        if objective_type == ObjectiveType.TOTAL_DEMAND:
            for k in self.epochs:
                tmp = gp.LinExpr(0.0)
                for s, d in product(self.nodes, self.nodes):
                    tmp.add(self.total_demand_sat[s][d][k])
 
                self.aux_var.append(self.model.addVar(-self.all_demand, 1,
                                    vtype=GRB.CONTINUOUS, name='aux_var_boj_%d'% k))
                tmp.add(self.all_demand - 1, -1)
                self.model.addConstr(self.aux_var[len(
                    self.aux_var) - 1] == tmp, "obj1_s_%d" % k)
                self.aux_var.append(self.model.addVar(
                    0, 1, vtype=GRB.CONTINUOUS, name='aux_var_obj2_%d' % k))
                self.model.addConstr(self.aux_var[len(self.aux_var) - 1] == gp.max_(
                    [self.aux_var[len(self.aux_var) - 2], 0]), name='obj2_k_%d' % k)

                objective.add(
                    self.aux_var[len(self.aux_var) - 1], (-multiplier))
                for i in self.nodes:
                    if self.topology.capacity[i][d] <= 0:
                        continue
                    objective.add(
                        self.flow[s][i][d][k], 10 * (pow(10, -1) / (self.num_epochs + 1)) * 2)
        # Experimental objective that has been removed.
        # elif objective_type == ObjectiveType.EXPERIMENTAL:
        #     objective = gp.LinExpr(0.0)
        #     for s, d, k in product(self.nodes, self.nodes, self.epochs):
        #         objective.add(
        #             self.total_demand_sat[s][d][k], -1 * multiplier * pow(0.1, (k + 1)))

        else:
            assert objective_type == ObjectiveType.PAPER, "wrong objective type"
            objective = gp.LinExpr(0.0)
            for s, d, k in product(self.nodes, self.nodes, self.epochs):
                objective.add(
                    self.total_demand_sat[s][d][k], -1 * multiplier * pow(10, -1) / (k + 1))
                for i in self.nodes:
                    if self.topology.capacity[i][d] <= 0:
                        continue
                    objective.add(
                        self.flow[s][i][d][k], 10 * (pow(10, -1) / (self.num_epochs + 1)) * 2)
        return objective

    def encode_problem(self, use_one_less_epoch=False) -> int:
        setup_start = time.time()
        self.model = gp.Model('AlltoAll_LP')
        self.initialize_variables()
        self.destination_constraints()
        self.node_constraints()
        self.capacity_constraints()
        self.model.setObjective(self.objective_formulation(self.user_input.instance.objective_type))

        log_file = f'Logs/{self.solver_name}_{self.user_input.topology.name}_{self.num_nodes}-nodes_' \
            f'{self.num_chunks}-chunks_{self.num_epochs}-epochs_{self.epoch_duration}-epoch_duration_{self.user_input.instance.objective_type}'

        if self.user_input.gurobi.output_flag == 1 or self.user_input.instance.debug:
            if self.user_input.gurobi.log_file:
                log_file += self.user_input.gurobi.log_file
            self.model.setParam("LogFile", log_file + ".log")
            self.model.Params.LogToConsole = 0
            # if self.user_input.instance.debug:
            #     self.model.write(log_file + '.lp')

        self.set_gurobi_params()
        setup_end = time.time()

        logging.debug(f'Total time for setup {setup_end - setup_start}')
        logging.debug(f'Epoch duration {self.epoch_duration}')
        logging.debug(f'Starting model optimization {log_file}')

        solve_start = time.time()
        self.model.optimize()
        solve_end = time.time()

        logging.debug(
            f'Finished model optimization {log_file} in {solve_end - solve_start}')

        if self.model.Status != GRB.OPTIMAL:
            logging.warning(
                f"Not_Optimal_{self.solver_name}_Status-{self.model.Status}_{self.user_input.topology.name}_"
                f"{self.num_nodes}-nodes_{self.num_chunks}-chunks_{self.num_epochs}-epochs_{self.epoch_duration}-epoch_duration"
            )
            if self.user_input.instance.debug and self.model.SolCount > 0:
                logging.debug(
                    f"Epoch at the end of which all demands are satisfied: {self.find_demand_satisfied_k() + 1}")
            # compute an Irreducible Inconsistent Subsystem
            # https://www.gurobi.com/documentation/10.0/refman/py_model_computeiis.html
            # else:
            # self.model.computeIIS()
            # self.model.write(log_file + '_unsat.ilp')
            return self.model.Status

        if self.user_input.instance.debug:
            logging.debug(
                f"Epoch at the end of which all demands are satisfied: {self.find_demand_satisfied_k() + 1}")
        return self.model.Status

    def get_flows_and_consumes(self) -> Tuple[List[Tuple[int, int, int, float, int]], Dict]:
        """
        this function returns the list of all the flows that the optimization assigned positive value to
        and the time-series for each destination of when it consumed part of its demand.
        """
        consumed = {}
        full_flow_list = []
        for v in self.model.getVars():
            if 'f_' in v.varName and v.x != 0.0:
                components = v.varName.split('_')
                _, s, i, j, k = components
                full_flow_list.append((int(s), int(i), int(j), round(v.x,6), int(k)))
            if 'T_' in v.varName and v.x > 0:
                components = v.varName.split('_')
                _, s, d, k = components
                if int(d) not in consumed:
                    consumed[int(d)] = []
                consumed[int(d)].append((int(s), int(k), round(v.x,6)))
        return (full_flow_list, consumed)

    def account_for_consume(self, consume: float, source: int, destination: int, i: int, j: int, k: int, paths: Dict) -> Dict:
        """
        converts the aggregated flow into per-flow form.
        """
        path = {}
        for c in self.chunks:
            if c not in path:
                path[c] = []
            if self.demand_copy[source][destination][c] > 0:
                if consume > self.demand_copy[source][destination][c]:
                    self.per_chunk_flows[source][i][j][c][k] = self.demand_copy[source][destination][c]
                    consume -= self.demand_copy[source][destination][c]
                    if c not in paths.keys():
                        path[c] = (source, i, j, c,
                                   self.demand_copy[source][destination][c], k)
                    else:
                        path[c] += (source, i, j, c,
                                    self.demand_copy[source][destination][c], k)
                    if source == i:
                        self.demand_copy[source][destination][c] = 0
                else:
                    assert consume > 0
                    self.per_chunk_flows[source][i][j][c][k] = consume
                    if c not in paths.keys():
                        path[c] = (source, i, j, c, consume, k)
                    else:
                        path[c] += (source, i, j, c, consume, k)
                    if source == i:
                        self.demand_copy[source][destination][c] -= consume
                    consume = 0
                    break
            if consume == 0:
                break
        for c in path.keys():
            if c not in paths:
                paths[c] = []
            paths[c] += [path[c]]
        consume = round(consume, 5)
        if consume != 0:
            print(f"source ={source}, destination={destination}, consume={consume}")
        if consume <= 1e-6:
            consume = 0
        assert consume == 0
        return paths

    def check_if_viable(self, hop: int, dest: int, step : int, instance: Tuple[int, int, int, int, float, int]) -> bool:
        """
        checks if the timing of the instnace is correct with respect to where we are at.
        """
        #enable the commented code if we make it such that the second link from the switch is instantaneous.
        if hop == dest: # or instance[1] in self.topology.switch_indicies:
            return (instance[-1] <= step - self.get_alpha_num_back(instance[1], hop))
        elif hop in self.topology.switch_indices:
            return (instance[-1] == step - 1 - self.get_alpha_num_back(instance[1], hop))
        else:
            return (instance[-1] <= step - 1 - self.get_alpha_num_back(instance[1], hop))
        

    def dig_to_source(self, hop: int, traffic: List[Tuple[int, int, int, int, float, int]],
                      consumed: Dict[int, Tuple[int, int, float]], source: int, step: int, dest: int,
                      volume: float, path: List[Tuple[int, int, int, float, int]], paths: Dict[int, List[Tuple[int, int, int, int, float, int]]] = {}) -> List[Tuple[int, int, int, int, float, int]]:
        """
        does DFS to trace back the path of a chunk to the source.
        """
        # find the nodes that sent the traffic to the node we are currently at.
        

        this_hop_consumed_volume = 0
        previous_hops = [x for x in traffic if x[2] == hop and (self.check_if_viable(hop, dest, step, x))]
        previous_hops = sorted(previous_hops, key=lambda x: x[-1], reverse = True)
        old_paths = paths
        for each_previous_hop in previous_hops:
            paths = old_paths
            if volume == 0:
                break
            found = False
            for i in range(len(traffic)):
                if traffic[i] == each_previous_hop:
                    consume = min(traffic[i][3], volume)
                    stored_traffic = traffic[i]
                    found = True
                    break
            assert found, "the previous hop not found in traffic list, must be a bug"
            
            step = each_previous_hop[-1]
            new_path = copy.deepcopy(path)
            new_path += [(source, each_previous_hop[1], hop, consume, step)]
            
            if consume <= 1e-5:
                continue
            traffic, consumed_volume = self.dig_to_source(
                    each_previous_hop[1], traffic, consumed, source, step, dest, consume, new_path, paths)
            
            for i in range(len(traffic)):
                if traffic[i][0] == stored_traffic[0]:
                    if traffic[i][1] == stored_traffic[1]:
                        if traffic[i][2] == stored_traffic[2]:
                                if traffic[i][4] == stored_traffic[4]:
                                    index = i
                                    break

            this_hop_consumed_volume += consumed_volume
            if consumed_volume < volume:
                volume = volume - consumed_volume
            else:
                volume = 0
           
            traffic[index] = (traffic[index][0], traffic[index][1], traffic[index]
                                  [2], traffic[index][3] - consumed_volume, traffic[index][4])
            if traffic[index][3] == 0:
                traffic = [x for x in traffic if x != traffic[index]]
        if len(previous_hops) == 0:
            assert source == hop
            this_hop_consumed_volume = min([x[3] for x in path])
            for i in range(len(path)):
                paths = self.account_for_consume(
                    this_hop_consumed_volume , source, dest, path[i][1], path[i][2], path[i][-1], paths)
            for c in paths.keys():
                if (source, dest, c) not in self.per_chunk_flow_paths.keys():
                    paths[c] = [x for x in paths[c] if len(x) > 0]
                    if len(paths[c]) == 0:
                        continue
                    self.per_chunk_flow_paths[(source, dest, c)] = [paths[c]]
                    paths[c] = []
                else:
                    self.per_chunk_flow_paths[(source, dest, c)] += [paths[c]]
                    paths[c] = []

        return traffic, this_hop_consumed_volume

    def get_per_chunk_flows(self) -> Dict:
        per_chunk_flow_list = {}
        for s, i, j, c, k in product(self.nodes, self.nodes, self.nodes, self.chunks, self.epochs):
            if self.per_chunk_flows[s][i][j][c][k] > 0:
                if k not in per_chunk_flow_list:
                    per_chunk_flow_list[k] = []
                per_chunk_flow_list[k].append((int(s), int(i), int(j), int(
                    c), self.per_chunk_flows[s][i][j][c][k], int(k)))
        return per_chunk_flow_list

    def chunk_flow_paths_to_string(self) -> Dict:
        """
        convert the flow paths in the per_chunk_flow_paths dictionary into their string version.
        """
        chunk_flow_paths = {}
        for s, d, c in self.per_chunk_flow_paths.keys():
            chunk_flow_paths[(s, d, c)] = []
            for each_path in self.per_chunk_flow_paths[(s, d, c)]:
                each_path = [x for x in each_path if len(x) != 0]
                if len(each_path) == 0:
                    continue
                each_path = sorted(each_path, key=lambda x: x[-1])
                path = []
                start = 0
                next = 0
                chunk_path = each_path[::-1]
                chunk_path = sorted(chunk_path, key=lambda x: x[-1])
                chunk_path = [x for x in chunk_path if round(x[4],6) >0]
                while start < len(chunk_path):
                    accumulated_flows = [chunk_path[start]]
                    while chunk_path[next][2] in self.topology.switch_indices:
                        next += 1
                        accumulated_flows.append(chunk_path[next])
                    start_node = accumulated_flows[0][1]
                    end_node = accumulated_flows[-1][2]
                    sending_epoch = accumulated_flows[0][5]
                    volume = accumulated_flows[0][4]
                    switches = "->".join([str(x[2])
                                         for x in accumulated_flows[:-1]])
                    if switches:
                        path.append(
                            (sending_epoch, f'{start_node}->{end_node} with volume {volume} in epoch {sending_epoch} via switches {switches}'))
                    else:
                        path.append(
                            (sending_epoch, f'{start_node}->{end_node} with volume {volume} in epoch {sending_epoch}'))
                    start = next = next + 1
                chunk_flow_paths[(s, d, c)].append(path)
        return chunk_flow_paths

    def get_flow_schedule(self) -> Tuple[List, Dict]:
        full_flow_list, consumed = self.get_flows_and_consumes()
        self.per_chunk_flows = np.zeros(
            (self.num_nodes, self.num_nodes, self.num_nodes, self.num_chunks, self.num_epochs))
        self.per_chunk_flow_paths = {}
        self.demand_copy = np.array(copy.deepcopy(self.demand), dtype=float)

        sources = set([x[0] for x in full_flow_list])
        destinations = self.nodes

        for each_source in sources:
            # traffic : flows sent on a link for a particular source
            traffic = [x for x in full_flow_list if x[0] == each_source]
            traffic = sorted(traffic, key=lambda x: x[-1])
            
            for each_dest in destinations:
                if self.demand_at_i[(each_source, each_dest)] == 0:
                    continue


                # all the steps in which a particular destination consumed hunks for a source
                steps = set(x[1]
                            for x in consumed[each_dest] if x[0] == each_source and x[2] > 1e-7)
                steps = sorted(list(steps))
                for each_step in steps:
                    # at a given step at a destination for a source there is only one
                    # consumed value so the volume list is a singleton
                    # volume represents the total amount consumed across all the incoming links
                    volume = [x[2] for x in consumed[each_dest]
                              if x[0] == each_source and x[1] == each_step][0]
                    hop = each_dest
                    traffic, _ = self.dig_to_source(
                        hop, traffic, consumed, each_source, each_step, each_dest, volume, [])
                    
            traffic = [x for x in traffic if x[3] > 1e-5]

            if len(traffic) != 0:
                logging.warning(
                    "there is a potential bug! there is traffic unaccounted for")
                print(traffic)
                assert 0, "potential bug, check code"

        per_chunk_flows = self.get_per_chunk_flows()
        chunk_str_paths = self.chunk_flow_paths_to_string()
        chunk_paths = {}
        flows_str = set()
        for (s, d, c) in self.per_chunk_flow_paths.keys():
            paths = self.per_chunk_flow_paths[(s, d, c)]
            k = max([x[-1] for x in [y[0] for y in paths if len(y[0]) > 0]])
            chunk_paths[f"Demand at {d} for chunk {c} from {s} met by epoch {k}"] = chunk_str_paths[(
                s, d, c)]
            for multipath in chunk_str_paths[(s, d, c)]:
                for epoch, path in multipath:
                    flows_str.add((epoch, f'Chunk {c} from {s} traveled over {path}'))

        required_flows = []
        Kmax = self.find_demand_satisfied_k()
        for k in range(Kmax):
            if k in per_chunk_flows.keys():
                required_flows += per_chunk_flows[k]
        flow_str_info = {}
        flow_str_info["1-Epoch_Duration"] = self.epoch_duration
        flow_str_info["2-Expected_Epoch_Duration"] = self.expected_epoch_duration
        flow_str_info["3-Epochs_Required"] = self.find_demand_satisfied_k() + 1
        flow_str_info["4-Collective_Finish_Time"] = flow_str_info["1-Epoch_Duration"] * flow_str_info["3-Epochs_Required"]
        flow_str_info["5-Algo_Bandwidth"] = self.topology.node_per_chassis * self.topology.chunk_size * self.topology.chassis / flow_str_info["4-Collective_Finish_Time"]
        flows_str = sorted(list(flows_str), key=lambda x: x[0])
        flow_str_info['7-Flows'] = [x[1] for x in flows_str]
        flow_str_info['8-Chunk paths'] = chunk_paths
        return required_flows, flow_str_info

    def find_demand_satisfied_k(self) -> int:
        """
        returns the total number of epochs we need to satisfy the demand.
        """
        satisfied_epochs = {}
        for v in self.model.getVars():
            if 'T_' in v.varName and v.x > 0:
                components = v.varName.split('_')
                _, s, i, k = components
                if (s, i) in satisfied_epochs:
                    if satisfied_epochs[(s, i)] < int(k):
                        satisfied_epochs[(s, i)] = int(k)
                else:
                    satisfied_epochs[(s, i)] = int(k)
        return max(satisfied_epochs.values())

    def get_schedule(self) -> Tuple[List, Dict]:
        if self.model.SolCount > 0:
            return self.get_flow_schedule()
        else:
            return [], {}
