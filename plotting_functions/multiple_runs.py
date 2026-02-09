#!/usr/bin/env python3
import numpy as np
import os
from datetime import datetime
import drn_config   # <--- import your training script as a module

def run_multiple(model="drn-xs", algorithm="EP", num_runs=10,
                 min_cond=0.1, max_cond=10.0,
                 config_file="config.json",
                 results_root="results"):

    diode_conductances = np.logspace(np.log10(min_cond),
                                     np.log10(max_cond),
                                     num_runs)

    # Use only date for sweep folder (YYYYMMDD)
    date_tag = datetime.now().strftime("%Y%m%d")
    sweep_root = os.path.join(
        results_root,
        f"{model}_{algorithm}_sweep_{date_tag}"
    )
    os.makedirs(sweep_root, exist_ok=True)

    successes, failures = 0, 0

    for i, cond in enumerate(diode_conductances, start=1):
        folder_name = f"cond_{cond:.3e}"
        run_path = os.path.join(sweep_root, folder_name)
        os.makedirs(run_path, exist_ok=True)

        print(f"\n=== Simulation {i}/{num_runs} ===")
        print(f"Model={model}, Algorithm={algorithm}, diode_conductance={cond:.6g}")
        print(f"Saving results to {run_path}\n")

        argv = [
            "--model", model,
            "--algorithm", algorithm,
            "--config", config_file,
            "--diode_conductance", str(float(cond)),
            "--base_path", run_path,
        ]

        try:
            drn_config.main(argv)   # <-- run directly, not via subprocess
            successes += 1
        except Exception as e:
            print(f"Simulation {i} failed with error: {e}")
            failures += 1

    print("\n=== SUMMARY ===")
    print(f"Results saved in: {sweep_root}")
    print(f"Total simulations: {num_runs}")
    print(f"Successful: {successes}")
    print(f"Failed: {failures}")
    print(f"Success rate: {100 * successes / num_runs:.1f}%")

if __name__ == "__main__":
    run_multiple(model="drn-xs", algorithm="EP", num_runs=10,
                 min_cond=1e-5, max_cond=1e-2,
                 config_file="config.json",
                 results_root="quadratic_diode_run")
