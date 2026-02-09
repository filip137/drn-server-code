#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    input_path = Path(
        "/home/filip/paper_maxicao_simulations/train_and_validate/timing_validate_cd/timing_validate_cd_results.json"
    )
    output_path = input_path.with_name("timing_validate_cd_plot.png")

    data = json.loads(input_path.read_text())
    by_layers = {}
    for row in data:
        layers = int(row["hidden_layers"])
        by_layers.setdefault(layers, []).append(row)

    plt.figure(figsize=(8, 5))
    for layers in sorted(by_layers):
        rows = sorted(by_layers[layers], key=lambda r: r["hidden_dim"])
        xs = [r["hidden_dim"] for r in rows]
        ys = [r["elapsed_seconds"] for r in rows]
        plt.plot(xs, ys, marker="o", label=f"{layers} hidden layer(s)")

    plt.xlabel("Hidden layer size")
    plt.ylabel("Elapsed time (s)")
    plt.title("validate_cd_results timing")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
