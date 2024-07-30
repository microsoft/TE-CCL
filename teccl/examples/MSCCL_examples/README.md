# TECCL MSCCL example

This folder contains the following 3 files: `amd_32.xml`, `amd_32.txt`, and `run_teccl.sh`

## amd_32.xml
This is the input XML file to [MSCCL runtime](https://github.com/microsoft/msccl). More details on how we generated XML is discussed in [Generate XML](#generate-xml)

## run_teccl.sh
This is the script to run TECCL XML using MSCCL and reproduces the TECCL and RCCL performance results in Figure 8.

This script requires the following prerequisites (for multi-node experiments, all nodes should complete the exact prerequisites):

### OpenMPI
```
sudo apt-get install openmpi-bin
```
For multi-node experiments, a `hostfile` specifying the IP addresses of all nodes is required. For example, in our 2-node experiment, `hostfile` contains the following content:
```
<IP of the first node>:16
<IP of the second node>:16
``` 
where 16 means each node has 16 GPUs.

### ROCm
Our experiments used [ROCm](https://github.com/ROCm/ROCm) version 5.7.3.

### RCCL and rccl-tests
Build [RCCL](https://github.com/ROCm/rccl) first.

Then build [rccl-tests](https://github.com/ROCm/rccl-tests) with MPI enabled, which needs the path to the previously installed `OpenMPI` and `RCCL`.

### Move XML to MSCCL input folder for the second node
This script removes everything in `/rccl/build/release/msccl-algorithms/` directory and moves the TECCL XML there on the first node (the node executing this script). User needs to do the same thing manually on the second node so that the 2 nodes have the same schedule.

## amd_32.txt
This is the output of executing `run_teccl.sh`.

TECCL's performance in Figure 8 is shown in the column  `algbw` of `in-place`.

RCCL's performance in Figure 8 is shown in the column `algbw` of `out-of-place`.

`rccl-tests`'s `all_gather_perf` executes two variants of AllGather algorithms. `in-place` algorithms assign output buffer to be the same as the input buffer and for `out-of-place` input/output buffers are different. They would **not** cause significant differences performance. We make TECCL XML `in-place` and leave the `out-of-place` to be default RCCL, so that we can easily compare them.

## Generate XML
We adapted the`ncclize` script from [TACCL](https://github.com/microsoft/taccl/blob/main/taccl/ncclize.py) to convert our schedule JSONs into XML format suitable for our experiments. We basically modified TACCL code to inject our schedule into TACCL's format and run their ncclize code. Users have the flexibility to employ any other method to execute the schedule. However, TE-CCL presupposes that the hardware implementation can fully utilize the bandwidth of the link as defined by its beta value. As noted in the paper, achieving this in practice often requires determining the optimal number of channels and threadblocks for each schedule. MSCCL Lang provides a comprehensive discussion of these challenges. In our experiment for section 6.2, we tested various settings and selected the configuration that delivered the best performance. 