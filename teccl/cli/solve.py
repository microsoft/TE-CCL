from argparse import SUPPRESS
from teccl.input_data import *
import pathlib
import json
from teccl.scheduler import *

def make_handle_solve(cmd_parsers):
    name = 'solve'
    cmd = cmd_parsers.add_parser(name)
    cmd.add_argument('-i', '--input_args', help='Path to the user input arguments JSON file (default: example_user_input.json)', default=SUPPRESS)

    def handler(args, command):
        if command != name:
            return False
        user_input = UserInputParams()
        input_path = pathlib.Path(args.input_args)
        with open(input_path, 'r') as jf:
            user_input_args = json.load(jf)
            for k, v in user_input_args['TopologyParams'].items():
                user_input.topology.__setattr__(k, v)
            for k, v in user_input_args['GurobiParams'].items():
                user_input.gurobi.__setattr__(k, v)
            for k, v in user_input_args['InstanceParams'].items():
                if k == 'objective_type':
                    user_input.instance.__setattr__(k, ObjectiveType(v))
                elif k == 'solution_method':
                    user_input.instance.__setattr__(k, SolutionMethod(v))
                elif k == 'collective':
                    user_input.instance.__setattr__(k, Collective(v))
                elif k == 'epoch_type':
                    user_input.instance.__setattr__(k, EpochType(v))
                else:
                    user_input.instance.__setattr__(k, v)

        solver = TECCLSolver(user_input)
        solver.solve()
        
        return True
    
    return handler

