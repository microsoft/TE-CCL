import copy
import json
import logging
import math
import pathlib
from time import time
from typing import Dict, Tuple, Union

from gurobipy import GRB
from teccl.input_data import *
from teccl.solvers.allgather import AllGatherFormulation
from teccl.solvers.allgather_astar import AStarFormulation
from teccl.solvers.alltoall import AlltoAllFormulation
from teccl.solvers.base_formulation import BaseFormulation
from teccl.topologies.dgx1 import DGX1
from teccl.topologies.dgx2 import DGX2
from teccl.topologies.ndv2 import NDv2
from teccl.topologies.amd import AMD
from teccl.topologies.mesh import Mesh
from teccl.topologies.topology import Topology


class TECCLSolver(object):
    def __init__(self, user_input: UserInputParams):
        self.user_input = user_input
        self.topology_obj = self.get_topology(user_input.topology)
        self.solver = self.get_solver(copy.deepcopy(user_input), self.topology_obj)

    
    def get_topology(self, topology_params: TopologyParams) -> Topology:
        if topology_params.name == "DGX1":
            return DGX1(topology_params)
        elif topology_params.name == "DGX2":
            return DGX2(topology_params)
        elif topology_params.name == "NDv2":
            return NDv2(topology_params)
        elif topology_params.name == "AMD":
            return AMD(topology_params)
        elif topology_params.name == "Mesh":
            return Mesh(topology_params)
        else:
            raise NotImplementedError(
                f"Input topology {topology_params.name} not implemented")


    def get_solver(self, user_input: UserInputParams, topology: Topology) -> BaseFormulation:
        if user_input.instance.collective == Collective.ALLGATHER:
            if user_input.instance.objective_type == ObjectiveType.ASTAR:
                return AStarFormulation(user_input, topology)
            return AllGatherFormulation(user_input, topology)
        elif user_input.instance.collective == Collective.ALLTOALL:
            user_input.instance.num_chunks = user_input.instance.num_chunks * \
                (len(topology.capacity) - len(topology.switch_indices))
            return AlltoAllFormulation(user_input, topology)
        else:
            raise NotImplementedError(
                f"Input collective {user_input.instance.collective} not implemented")

    def feasible_solution_search(self, user_input: UserInputParams, topology_obj: Topology, final_epoch_duration: float) -> Union[Tuple[float, int], ValueError]:
        """
            Finds a feasible time in which the collective can finish using large epochs with fewer of them.
        """
        if user_input.instance.collective == Collective.ALLTOALL:
            num_epochs = math.ceil(topology_obj.get_max_hop_distance() * 20)
            factor = 100
        else:
            num_epochs = math.ceil(topology_obj.get_max_hop_distance() * 3)
            factor = 1
        # 3 is a good factor to not have too many epochs, but also to take into account the alpha
        max_time_chunk = topology_obj.get_largest_time_chunk()

        collective_time_estimate = num_epochs * max_time_chunk * \
            user_input.instance.num_chunks * len(topology_obj.capacity) / 2

        # binary search on the time estimate
        lower_bound = 0

        upper_bound = collective_time_estimate * factor
        attempts = 0
        feasible_time = upper_bound

        while lower_bound <= upper_bound:
            mid = (upper_bound + lower_bound) / 2
            new_user_input = copy.deepcopy(user_input)
            new_user_input.gurobi = copy.deepcopy(user_input.gurobi)
            new_user_input.instance = copy.deepcopy(user_input.instance)
            # find some feasible solution
            new_user_input.gurobi.solution_limit = 1
            new_user_input.instance.num_epochs = num_epochs
            new_user_input.instance.epoch_duration = mid / num_epochs
            if new_user_input.instance.epoch_duration <= final_epoch_duration and feasible_time != collective_time_estimate * factor:
                break
            # user_input.instance.debug = True
            solver_inst = self.get_solver(new_user_input, topology_obj)
            result = solver_inst.encode_problem()
            if result != GRB.INFEASIBLE:
                epochs_taken = solver_inst.find_demand_satisfied_k() + 1
                time_taken = epochs_taken * solver_inst.epoch_duration
                upper_bound = time_taken
                feasible_time = min(feasible_time, time_taken)
            else:
                lower_bound = mid
            attempts += 1
            if attempts > 10:
                # Avoids trying too many times and spending time in the initial search.
                break
        if feasible_time != collective_time_estimate * factor:
            return feasible_time, num_epochs
        raise ValueError(
            "Unable to find a solution in the initial feasible search algorithm (try with factor > 1)")


    def get_schedules(self, initial_solver: BaseFormulation, user_input: UserInputParams, topology_obj: Topology) -> Dict:
        """
            Finds the optimal schedule for the collective either directly or iteratively.
            In the direct method, the solver is instantiated to find the optimal solution in the given number of epochs.
            In the iterative method, the solution is found using binary search and in each iteration the solver is instantiated
                to find some feasible solution.
        """
        epoch_result_schedule_solver = {}
        if user_input.instance.solution_method == SolutionMethod.ONE_SHOT:
            # One shot
            result = initial_solver.encode_problem()
            schedule, schedule_json = initial_solver.get_schedule()
            if schedule:
                epochs_taken = initial_solver.find_demand_satisfied_k() + 1
                epoch_result_schedule_solver[epochs_taken] = {"result": result,
                                                            "schedule": (schedule, schedule_json),
                                                            "solver": initial_solver}
        else:
            # Iterative
            user_input.gurobi.solution_limit = 1
            lower_bound = 0
            upper_bound = user_input.instance.num_epochs
            tried_epochs = set()
            while lower_bound <= upper_bound:
                mid = math.ceil((upper_bound + lower_bound) / 2)
                if mid in tried_epochs:
                    break
                tried_epochs.add(mid)
                new_user_input = copy.deepcopy(user_input)
                new_user_input.instance.num_epochs = mid
                solver_inst = self.get_solver(new_user_input, topology_obj)
                solver = solver_inst
                result = solver_inst.encode_problem(use_one_less_epoch=True)
                schedule, schedule_json = solver_inst.get_schedule()
                if schedule:
                    epochs_taken = solver_inst.find_demand_satisfied_k() + 1
                    logging.debug(
                        f"Found a feasible schedule in {epochs_taken} epochs")
                    upper_bound = epochs_taken
                    epoch_result_schedule_solver[epochs_taken] = {"result": result,
                                                                "schedule": (schedule, schedule_json),
                                                                "solver": solver_inst}
                else:
                    lower_bound = mid
        return epoch_result_schedule_solver
    
    def solve(self):
        """
            Main function that first finds a feasible collective finish time to estimate the number of epochs required
             if user did not provide it. It then finds the optimal solution and outputs the schedule to the file.
        """
        start = time()
        logs_dir = pathlib.Path("Logs")
        logs_dir.mkdir(exist_ok=True)
        user_input = self.user_input
        solver = self.solver
        if user_input.instance.debug:
            logging.basicConfig(format='%(asctime)s %(message)s',
                                datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG, filename=user_input.instance.debug_output_file)


        if user_input.instance.num_epochs == -1:
            # Search for a feasible time to estimate the number of epochs
            feasible_time, _ = self.feasible_solution_search(
                user_input, self.topology_obj, solver.epoch_duration)
            user_input.instance.num_epochs = math.ceil(
                feasible_time / solver.epoch_duration)
            solver.set_num_epochs(user_input.instance.num_epochs)

        epoch_result_schedule_solver = self.get_schedules(
            solver, user_input, self.topology_obj)
        timestamp = int(time())

        if epoch_result_schedule_solver:
            output_file = user_input.instance.schedule_output_file
            best_epochs = min(epoch_result_schedule_solver.keys())
            solver = epoch_result_schedule_solver[best_epochs]["solver"]
            if user_input.instance.schedule_output_file == "":
                output_file = f'{user_input.topology.name}_{solver.num_nodes}-nodes_{solver.num_chunks}-chunks_{user_input.topology.chunk_size}-chunksize_{solver.solver_name}_{timestamp}.json'
            epoch_result_schedule_solver[best_epochs]["schedule"][1]["Solver_Time"] = time(
            ) - start
            pathlib.Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w+') as f:
                json_obj = json.dumps(
                    epoch_result_schedule_solver[best_epochs]["schedule"][1], indent=2, sort_keys=True)
                f.write(json_obj)
            print(f'Schedule written to {output_file}')


        else:
            logging.error("No schedule found with the given parameters")


