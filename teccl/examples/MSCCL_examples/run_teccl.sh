#!/bin/bash

# Remove any files in the msccl-algorithms directory
rm -rf <path-to-rccl>/rccl/build/release/msccl-algorithms/*

# Copy the TECCL schedule
cp amd_32.xml <path-to-rccl>/rccl/build/release/msccl-algorithms/

# !!! NEED TO REMOVE AND COPY SCHEDULE IN THE SECOND NODE MANUALLY!!! (for 2-chassis experiments)

# Run AllGather Perf

<path-to-openmpi>/bin/mpirun --allow-run-as-root -hostfile <path-to-hostfile>/hostfile -map-by ppr:16:node --bind-to numa -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_include eth0 -x PATH -x LD_LIBRARY_PATH=<path-to-rocm>/rocm-5.7.3/lib:$LD_LIBRARY_PATH -x NCCL_SOCKET_IFNAME=eth0 -x LD_PRELOAD=<path-to-rccl>/rccl/build/release/librccl.so:$LD_PRELOAD -x NCCL_DEBUG=INFO -x NCCL_MAX_NCHANNELS=2  -x NCCL_NET_GDR_LEVEL=3  -x RCCL_MSCCL_FORCE_ENABLE=1 <path-to-rccl-tests>/rccl-tests/build/all_gather_perf -b 256 -e 10G -f 2 -g 1 -z 0 -n 50 -w 50 -c 1 -a 2 > /amd_32.txt
