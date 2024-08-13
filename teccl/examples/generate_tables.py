import pathlib
from pathlib import Path
import json
import pandas as pd
import os

INPUT_DIR = Path("./experiments/output/")
OUTPUT_DIR = Path("./experiments_output/tables/")
DATA_SIZES = ["1GB", "256MB", "64MB", "16MB", "4MB", "1MB", "256KB", "64KB", "16KB", "4KB", "1KB"]
MICRO = 1e6

def generate_individual_table(topology: str, chassis: str, collective: str, epoch_type: str) -> None:
    data_dir = INPUT_DIR / Path(f"{topology}_output/{chassis}/{collective}/{epoch_type}/")
    data_dir.mkdir(parents=True, exist_ok=True)

    epoch_duration = []
    collective_time = []
    solver_time = []
    algorithm_bw = []

    for data_size in DATA_SIZES:
        try:
            with open(data_dir / Path(f"{data_size}.json"), 'r') as f:
                data = json.load(f)
            ed = data["2-Expected_Epoch_Duration"] * MICRO
            ct = data["4-Collective_Finish_Time"] * MICRO
            ct = str(round(ct, 2))
            st = data["Solver_Time"]
            st = str(round(st, 2))
            ab = data["5-Algo_Bandwidth"]
            ab = str(round(ab, 2))
        except FileNotFoundError:
            ed = -1
            ct = -1
            st = -1
            ab = -1
        epoch_duration.append(ed)
        collective_time.append(ct)
        solver_time.append(st)
        algorithm_bw.append(ab)
    
    table = pd.DataFrame({
        'Output Buffer Size': DATA_SIZES,
        'Epoch Duration (us)': epoch_duration,
        'Collective Finish Time (us)': collective_time,
        'Solver Time (s)': solver_time,
        'Algorithm Bandwidth (GB/s)': algorithm_bw
    })
    individual_table_path = OUTPUT_DIR / Path("individual_tables/")
    individual_table_path.mkdir(parents=True, exist_ok=True)
    table.to_csv(individual_table_path / Path(f"{topology}_{chassis}_{collective}_{epoch_type}.csv"), index=False)
    print(f"Individual table for {topology}_{chassis}_{collective}_{epoch_type} generated.", flush=True)

def generate_comparison_table(topology: str, chassis: str, collective: str, epoch_type: str, name_prefix: str = "") -> None:
    data_dir = OUTPUT_DIR / Path("individual_tables/")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    teccl_df = pd.read_csv(data_dir / Path(f"{topology}_{chassis}_{collective}_{epoch_type}.csv"))
    taccl_df = pd.read_csv(data_dir / Path(f"TACCL_{topology}_{chassis}_{collective}.csv"))
    comp_df = pd.concat([teccl_df, taccl_df], axis=1)

    improvements = []
    teccl_ab = comp_df["Algorithm Bandwidth (GB/s)"]
    taccl_ab = comp_df["TACCL Algo Bandwidth"]
    for teccl, taccl in zip(teccl_ab, taccl_ab):
        if teccl == -1 or taccl == -1:
            improvements.append(-1)
        else:
            improvement = (teccl - taccl) / taccl * 100
            improvement = str(round(improvement, 2))
            improvements.append(improvement)
    
    comparison_tables_dir = OUTPUT_DIR / Path("comparison_tables/")
    comparison_tables_dir.mkdir(parents=True, exist_ok=True)

    comp_df["Improvement (%)"] = improvements
    comp_df.to_csv(comparison_tables_dir / Path(f"{name_prefix}{topology}_{chassis}_{collective}_{epoch_type}.csv"), index=False)
    print(f"Comparison table for {topology}_{chassis}_{collective}_{epoch_type} generated.", flush=True)    



if __name__ == "__main__":
    for topology in ["NDv2", "DGX2"]:
        for collective in ["AllGather", "AlltoAll"]:
            for epoch_type in ["Fast", "Fast_Early_Stop", "Slow"]:
                if epoch_type == "Fast_Early_Stop" and collective == "AlltoAll":
                    # AlltoAll does not have Fast_Early_Stop
                    continue
                if topology == "NDv2":
                    for chassis in ["2_chassis", "4_chassis"]:
                        generate_individual_table(topology, chassis, collective, epoch_type)
                else:
                    chassis = "2_chassis"
                    generate_individual_table(topology, chassis, collective, epoch_type)
    
    generate_comparison_table("NDv2", "2_chassis", "AlltoAll", "Fast", "tab8_1_")
    generate_comparison_table("NDv2", "2_chassis", "AlltoAll", "Slow", "tab8_2_")
    generate_comparison_table("NDv2", "2_chassis", "AllGather", "Fast", "tab8_3_")
    generate_comparison_table("NDv2", "2_chassis", "AllGather", "Fast_Early_Stop", "tab8_4_")
    generate_comparison_table("NDv2", "2_chassis", "AllGather", "Slow", "tab8_5_")