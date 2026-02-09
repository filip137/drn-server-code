import matplotlib.pyplot as plt
import os
from pathlib import Path
from plot_single_run_events import _collect_metrics
import argparse

def _plot_weight_dists(events_file):
    # Group by parameter name
    grouped = {}
    data = _collect_metrics(events_file)
    script_dir = Path(__file__).resolve().parent
    exp_dir = script_dir / "experimental_plots"
    exp_dir.mkdir(exist_ok=True)
    out_dir = exp_dir


    for tag, series in data.items():
        if not tag.startswith("WeightDist_"):
            continue
        # tag format: WeightDist_<stat>_<param>
        parts = tag.split("_", 2)
        if len(parts) < 3:
            continue
        _, stat, param = parts
        grouped.setdefault(param, {})[stat] = series
    for param, stats in grouped.items():
        plt.figure(figsize=(9, 5))
        for stat, s in sorted(stats.items()):
            if len(s["steps"]) == 0:
                continue
            plt.plot(s["steps"], s["values"], label=stat)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(f"Weight distribution stats for {param}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        os.makedirs(os.path.join(out_dir, "weight_stats"), exist_ok=True)
        plt.savefig(os.path.join(out_dir, "weight_stats", f"{param}_dist.png"), dpi=300, bbox_inches="tight")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot weight stats from a TensorBoard events file.")
    parser.add_argument("events_path", help="Path to events.out.tfevents* file or a run directory.") 
    #args = parser.parse_args()
    events_path = "/home/filip/velociraptor_remote/simulation_results/runs/voltage_amp_run/drn-conv/EP/voltage_amp_20251120_185909/001_voltage_amp_1.000e+00/events.out.tfevents.1763661550.velociraptor.243107.0"
    #_plot_weight_dists(args.events_path)
    _plot_weight_dists(events_path)
if __name__ == "__main__":
    main()