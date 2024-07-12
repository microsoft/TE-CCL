# TE-CCL

TE-CCL is a tool to generate collective communication schedules for large topologies using a Traffic Engineering-based solver.

TE-CCL takes in a topology and collective (e.g. AllGather) and outputs a schedule (in JSON) detailing data transfer steps for each node that satisfies the demands specified by the collective. In a Traffic Engineering-based approach, TE-CCL encodes the collective communication process into *capacity constraints*, *flow conservation constraints*, and *destination constraints*, and solves a mixed-integer linear program (MILP), with options to convert to an linear program (LP) form or A* form for better scalibility.

> **Rethinking Machine Learning Collective Communication as a Multi-Commodity Flow Problem** <br/>
> Xuting Liu, Behnaz Arzani, Siva Kesava Reddy Kakarla, Liangyu Zhao, Vincent Liu, Miguel Castro, Srikanth Kandula, and Luke Marshall <br/>
> **SIGCOMM 2024** [https://doi.org/10.1145/3651890.3672249]

## Citing TE-CCL
[TO BE FILLED]

## Installation
### Prerequisites
- Install [Anaconda](https://www.anaconda.com/) and activate an anaconda environment.
- Obtain and install a [Gurobi license](https://www.gurobi.com/downloads/). After getting the license, following the steps to install and activate Gurobi.
```
conda config --add channels http://conda.anaconda.org/gurobi
conda install -c conda-forge gurobi -y
<command to install Gurobi license. e.g. grbgetkey xxxxxx>
```
### Install TE-CCL
In the anaconda environment with Gurobi installed, run
```
pip install .
```

## Usage
TE-CCL takes in a JSON user input file that specifies the topology, Gurobi settings, and model instance parameters. Please refer to [Input Data](#input-data) for details.
```
teccl solve --input_args <input.json>
```

### Example
To generate a schedule for AllGather in the NDv2 topology, run
```
teccl solve --input_args teccl/examples/sample_inputs/ndv2_input.json
```
This will generate the schedule file `teccl/examples/schedules/ndv2_schedule.json`

### Detailed Examples
For detailed examples, please refer to instructions in the [examples](teccl/examples/) directory.


## Input Data
A user input JSON file is consists of three parts: `TopologyParams`, `GurobiParams`, and `InstanceParams`. Detailed explainations of each argument is in [input_data.py](teccl/input_data.py).

### TopologyParams
This sepcifies the topology considered. Each topology is defined as a seperate Python file in teccl/topologies.

### GurobiParams
Parameters for the Gurobi solver.

### InstanceParams
Parameters for the model instance, including the collective, the choice of objective function and size of epochs.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
