# Examples
The examples here reproduce the data in our paper.

## Generate input data
To prepare input data for experiments, run
```
python json_gen.py
```
to create `experiments` directory with input files inside:

```
experiments
├── DGX2
│   └── 2_chassis
│       ├── AllGather
│       │   ├── Fast
│       │   │   ├── 1KB.json
│       │   │   ├── <up to 1GB.json, 11 files in total>
│       │   ├── Fast_Early_Stop
│       │   └── Slow
│       └── AlltoAll
├── NDv2
│   ├── 2_chassis
│   │   ├── AllGather
│   │   └── AlltoAll
│   └── 4_chassis
│       ├── AllGather
│       └── AlltoAll
└── output
    ├── DGX2_output
    │   └── 2_chassis
    │       ├── AllGather
    │       └── AlltoAll
    └── NDv2_output
        ├── 2_chassis
        │   ├── AllGather
        │   └── AlltoAll
        └── 4_chassis
            ├── AllGather
            └── AlltoAll
```

Each of the `AllGather` directories contains **3** link types: `Fast`, `Fast_Early_Stop`, and `Slow`; `AlltoAll` contains `Fast` and `Slow`. Each link type directory contains **11** input config files for data sizes ranging from 1KB to 1GB. This totals to 11 * 3 (topologies i.e. DGX2 2 chassis, NDv2 2 chassis, NDv2 4 chassis) * 3 (link types) + 11 * 3 * 2 = 165 experiments to run.

## Run TE-CCL
To run TE-CCL on all input files we just created, run
```
./run_experiments.sh
```
Output are written to `experiments/output` directory. Running everything takes approximately 200 hours (8 days).

## Analyze Data

### Generate Tables
After getting all outputs from TE-CCL, run
```
python generate_tables.py
```
to generate tables from raw data. Tables are organized in `experiemts_output` directory as follows:
```
experiments_output
└── tables
    ├── comparison_tables
    └── individual_tables
```

Tables in `individual_tables` contains important metrics such as algorithm bandwidth for each experiment setup.

Tables in `comparison_tables` compares TE-CCL against [TACCL](#taccl). These are **Table 8** in the Appendix (see [Data Differences](#data-differences)).

### Generate Figures
After getting all the tables, run
```
python generate_figures.py
```
to generate figures in the paper, whcih are
```
experiments_output
└── figures
    ├── fig_10_a.pdf
    ├── fig_10_b.pdf
    ├── fig_5_a.pdf
    ├── fig_5_b.pdf
    ├── fig_6_a.pdf
    └── fig_6_b.pdf
```

Notice that we use the solver time data from our experiments to reproduce the figures, because solver time varies a lot due to physical capacities of different machines.

### Provided Output
We ran TE-CCL and provided its output in `experiments/output_provided`, which should have the same content in `experiments/output` after completing all experiments.

## Additional Notes
### TACCL [1]
We ran TACCL and provided the results in `individual_tables`. To run TACCL users should refer to the [TACCL repo](https://github.com/microsoft/taccl).

[1] Shah, A., Chidambaram, V., Cowan, M., Maleki, S., Musuvathi, M., Mytkowicz, T., Nelson, J. and Saarikivi, O., 2023. {TACCL}: Guiding Collective Algorithm Synthesis using Communication Sketches. In 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23) (pp. 593-612).

### Data Differences
We have been improving TE-CCL since publication so results here are similar to or better than the results in Appendix. Notably, for NDv2 2 chassis AllGather optimal epoch duration, TE-CCL took 7201.05s for 1GB case, as shown in the Appendix. Now TE-CCL takes around just 1 minute for the same setup.

### Proprietary Internal Topology
All the data and figures that involve our proprietary internal topology (int1 and int2 topology in the paper) are removed from the codebase to protect proprietary information.

### Python Version
`generate_figures.py` requires Python 3.7. TE-CCL was tested with Python 3.9.
