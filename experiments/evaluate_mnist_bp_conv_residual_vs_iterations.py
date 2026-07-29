#!/usr/bin/env python3
"""Estimate conv MNIST DRN residual current versus inference iterations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_mnist_bp_write_noise_sweep import (  # noqa: E402
    RunRow,
    _build_eval_context,
    _json_sanitize,
    _read_summary_rows,
    _scores,
    _select_shard,
)
from model.resistive.layer import NonlinearResistiveLayer  # noqa: E402
from model.variable.layer import layer_index  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "mnist_bp_conv2_best_lr_50epoch_64_128ch_s2_valid_iter6_seed0"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "mnist_bp_conv2_hardsigmoid_residual_vs_iterations_seed0"
)
RUN_ORDER = [
    "mnist_bp_amp_v1_c1",
    "mnist_bp_amp_v2_c1",
    "mnist_bp_amp_v4_c1",
    "mnist_bp_amp_v1_c2",
    "mnist_bp_amp_v1_c4",
]
RUN_LABELS = {
    "mnist_bp_amp_v1_c1": "v1/c1",
    "mnist_bp_amp_v2_c1": "v2/c1",
    "mnist_bp_amp_v4_c1": "v4/c1",
    "mnist_bp_amp_v1_c2": "v1/c2",
    "mnist_bp_amp_v1_c4": "v1/c4",
}
RESIDUAL_COLUMNS = [
    "model_label",
    "non_linearity",
    "residual_mode",
    "run_name",
    "seed",
    "voltage_amp",
    "current_amp",
    "checkpoint_path",
    "iteration_count",
    "layer_name",
    "layer_index",
    "layer_role",
    "num_examples",
    "accuracy",
    "loss",
    "mean",
    "median",
    "p90",
    "p99",
    "max",
]


def _write_rows(path: Path, columns: list[str], rows: list[dict], *, append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and append
    mode = "a" if append else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_sanitize(row.get(column, "")) for column in columns})


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, sort_keys=True) + "\n")


def _finite(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = _finite(values)
    if arr.size == 0:
        return {"mean": math.nan, "median": math.nan, "p90": math.nan, "p99": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def _layer_id(layer: object, fallback: int) -> int:
    try:
        return int(layer_index(layer))
    except ValueError:
        return int(fallback)


def _layer_role(index: int, total: int) -> str:
    if index == total - 1:
        return "output"
    return f"hidden_{index}"


def _perfect_diode_projected_kkt_residual(
    layer: object,
    grad: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    state = layer.state.detach()
    if not isinstance(layer, NonlinearResistiveLayer):
        return grad.abs()
    if state.ndim < 2 or state.shape[1] % 2 != 0:
        return grad.abs()

    half = state.shape[1] // 2
    residual = torch.empty_like(grad)

    exc_state = state[:, :half]
    exc_grad = grad[:, :half]
    exc_free = exc_state > eps
    residual[:, :half] = torch.where(exc_free, exc_grad.abs(), torch.relu(-exc_grad))

    inh_state = state[:, half:]
    inh_grad = grad[:, half:]
    inh_free = inh_state < -eps
    residual[:, half:] = torch.where(inh_free, inh_grad.abs(), torch.relu(inh_grad))
    return residual


def _residual_tensor(
    layer: object,
    grad: torch.Tensor,
    *,
    residual_mode: str,
    non_linearity: str,
    eps: float,
) -> tuple[torch.Tensor, str]:
    if residual_mode == "auto":
        mode = "projected_kkt" if non_linearity == "perfect_diode" else "raw"
    else:
        mode = residual_mode

    if mode == "projected_kkt":
        if isinstance(layer, NonlinearResistiveLayer):
            return _perfect_diode_projected_kkt_residual(layer, grad, eps=eps), "projected_kkt"
        return grad.abs(), "raw"
    if mode == "raw":
        return grad.abs(), "raw"
    raise ValueError(f"Unknown residual mode {residual_mode!r}.")


def _evaluate_context(
    context: dict,
    *,
    max_test_samples: int | None,
    residual_mode: str,
    kkt_eps: float,
) -> dict:
    network = context["network"]
    minimizer = context["minimizer"]
    fn = getattr(minimizer, "_fn", context["energy_fn"])
    cost_fn = context["cost_fn"]
    output_layer = context["output_layer"]
    test_loader = context["test_loader"]
    device = context["base_states"][0].device
    num_classes = context["num_classes"]
    free_layers = context["free_layers"]

    layer_values: list[list[float]] = [[] for _ in free_layers]
    layer_residual_modes: list[str] | None = None
    overall_values: list[float] = []
    loss_values: list[float] = []
    correct_values: list[bool] = []
    total_seen = 0

    for images, labels in test_loader:
        if max_test_samples is not None and total_seen >= max_test_samples:
            break
        images = images.to(device)
        labels = labels.to(device)

        network.set_input(images, reset=True)
        minimizer.compute_equilibrium()
        cost_fn.set_target(labels)

        scores = _scores(output_layer.state.detach(), num_classes)
        pred = torch.argmax(scores, dim=1)
        losses = cost_fn.eval().detach().reshape(-1)
        correct = pred.eq(labels).detach()

        grads = [fn.grad_layer_fn(layer)().detach() for layer in free_layers]
        residuals_and_modes = [
            _residual_tensor(
                layer,
                grad,
                residual_mode=residual_mode,
                non_linearity=context["model_cfg"]["non_linearity"],
                eps=kkt_eps,
            )
            for layer, grad in zip(free_layers, grads)
        ]
        if layer_residual_modes is None:
            layer_residual_modes = [mode for _, mode in residuals_and_modes]
        residuals = [residual for residual, _ in residuals_and_modes]
        norms = [
            residual.reshape(residual.shape[0], -1).amax(dim=1).detach()
            for residual in residuals
        ]
        batch_size = int(images.shape[0])
        take = batch_size
        if max_test_samples is not None:
            take = min(take, int(max_test_samples) - total_seen)
        if take <= 0:
            break

        stacked = torch.stack([norm[:take] for norm in norms], dim=0)
        for values, norm in zip(layer_values, norms):
            values.extend(norm[:take].cpu().numpy().astype(np.float64).tolist())
        overall_values.extend(stacked.amax(dim=0).cpu().numpy().astype(np.float64).tolist())
        loss_values.extend(losses[:take].cpu().numpy().astype(np.float64).tolist())
        correct_values.extend(correct[:take].cpu().numpy().astype(bool).tolist())
        total_seen += take

    if total_seen == 0:
        raise ValueError("Evaluation saw zero examples.")

    return {
        "num_examples": int(total_seen),
        "accuracy": float(np.mean(np.asarray(correct_values, dtype=bool))),
        "loss": float(np.mean(np.asarray(loss_values, dtype=np.float64))),
        "layer_values": layer_values,
        "layer_residual_modes": layer_residual_modes or [],
        "overall_values": overall_values,
    }


def _row_sort_key(row: dict) -> tuple:
    run_name = row.get("run_name", "")
    try:
        run_index = RUN_ORDER.index(run_name)
    except ValueError:
        run_index = len(RUN_ORDER)
    return (
        str(row.get("non_linearity", "")),
        run_index,
        int(row.get("seed", 0)),
        int(row.get("iteration_count", 0)),
        str(row.get("layer_role", "")),
        str(row.get("layer_name", "")),
    )


def _read_residual_rows(output_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(output_root.glob("raw_residual*.csv")):
        rows.extend(csv.DictReader(path.open()))
    summary = output_root / "residual_vs_iterations.csv"
    if not rows and summary.exists():
        rows.extend(csv.DictReader(summary.open()))
    return rows


def write_summaries_and_plots(output_root: Path, *, keep_raw: bool = True) -> None:
    rows = _read_residual_rows(output_root)
    if not rows:
        return
    rows = sorted(rows, key=_row_sort_key)
    summary_path = output_root / "residual_vs_iterations.csv"
    _write_rows(summary_path, RESIDUAL_COLUMNS, rows, append=False)

    gate_rows = [
        row
        for row in rows
        if row.get("layer_role") == "overall"
    ]
    by_run: dict[tuple[str, str, int], dict[int, dict]] = defaultdict(dict)
    for row in gate_rows:
        by_run[
            (str(row.get("non_linearity", "")), row["run_name"], int(row["seed"]))
        ][int(row["iteration_count"])] = row

    decisions = []
    recommend_iterations = None
    available_iterations = sorted(
        {int(row["iteration_count"]) for row in gate_rows if row.get("iteration_count", "") != ""}
    )
    reference_iteration = max(available_iterations) if available_iterations else 16
    candidate_iterations = [value for value in available_iterations if value <= reference_iteration]
    for (non_linearity, run_name, seed), values in sorted(by_run.items()):
        if reference_iteration not in values:
            continue
        reference_p90 = float(values[reference_iteration]["p90"])
        selected = reference_iteration
        for candidate in candidate_iterations:
            if candidate not in values:
                continue
            candidate_p90 = float(values[candidate]["p90"])
            ratio = candidate_p90 / reference_p90 if reference_p90 > 0 else math.inf
            if ratio <= 1.2:
                selected = candidate
                break
        recommend_iterations = max(recommend_iterations or selected, selected)
        p90_6 = float(values[6]["p90"]) if 6 in values else math.nan
        ratio_6 = p90_6 / reference_p90 if reference_p90 > 0 else math.inf
        decisions.append(
            {
                "run_name": run_name,
                "non_linearity": non_linearity,
                "seed": seed,
                "p90_at_6": p90_6,
                f"p90_at_{reference_iteration}": reference_p90,
                f"ratio_6_over_{reference_iteration}": ratio_6,
                "selected_num_iterations": selected,
            }
        )
    _write_json(
        output_root / "gate_decision.json",
        {
            "recommended_num_iterations": recommend_iterations,
            "reference_iteration": reference_iteration,
            "rule": "choose the smallest K whose p90_overall is within 20% of the largest evaluated K; top-level recommendation is the max across runs",
            "decisions": decisions,
        },
    )

    overall_rows = [row for row in rows if row.get("layer_role") == "overall"]
    if overall_rows:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
        for row in overall_rows:
            grouped[
                (str(row.get("non_linearity", "")), row["run_name"], int(row["seed"]))
            ].append(row)
        for (non_linearity, run_name, seed), group in sorted(
            grouped.items(),
            key=lambda item: (
                str(item[0][0]),
                RUN_ORDER.index(item[0][1]) if item[0][1] in RUN_ORDER else 999,
                item[0][2],
            ),
        ):
            group = sorted(group, key=lambda row: int(row["iteration_count"]))
            x = [int(row["iteration_count"]) for row in group]
            y = [float(row["p90"]) for row in group]
            label = f"{non_linearity} {RUN_LABELS.get(run_name, run_name)} seed {seed}"
            ax.plot(x, y, marker="o", label=label)
        ax.set_xlabel("Inference iterations")
        ax.set_ylabel("p90 residual current, overall")
        nonlinearities = sorted({str(row.get("non_linearity", "")) for row in overall_rows})
        ax.set_title(
            "Conv2 residual current vs iterations"
            if len(nonlinearities) != 1
            else f"{nonlinearities[0]} residual current vs iterations"
        )
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_root / "residual_vs_iterations_by_amp.png", dpi=180)
        plt.close(fig)

    if not keep_raw:
        for path in output_root.glob("raw_residual*.csv"):
            path.unlink()


def _filter_rows(rows: list[RunRow], *, non_linearities: set[str] | None) -> list[RunRow]:
    if non_linearities is None:
        return rows
    return [row for row in rows if row.non_linearity in non_linearities]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--model-label", default="conv2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    parser.add_argument("--run-name", action="append", choices=RUN_ORDER)
    parser.add_argument("--non-linearity", nargs="+", default=["hard_sigmoid"])
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--iteration-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10, 12, 16])
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-test-samples", type=int, default=1024)
    parser.add_argument("--residual-mode", choices=("auto", "raw", "projected_kkt"), default="auto")
    parser.add_argument("--kkt-eps", type=float, default=1.0e-8)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--raw-results-name", default=None)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        write_summaries_and_plots(output_root)
        print(f"[summary] wrote residual summaries under {output_root}")
        return

    if args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {args.device!r}, but CUDA is unavailable.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = _read_summary_rows(
        input_root,
        checkpoint_kind=args.checkpoint_kind,
        run_names=set(args.run_name) if args.run_name else None,
        training_seeds=set(args.training_seeds) if args.training_seeds else None,
    )
    rows = _filter_rows(rows, non_linearities=set(args.non_linearity) if args.non_linearity else None)
    rows = sorted(rows, key=lambda row: (RUN_ORDER.index(row.run_name), row.seed))
    selected = _select_shard(rows, args.num_shards, args.shard_index)
    raw_name = args.raw_results_name or f"raw_residual_shard_{args.shard_index}.csv"
    raw_path = output_root / raw_name

    config = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model_label": args.model_label,
        "device": str(device),
        "checkpoint_kind": args.checkpoint_kind,
        "run_name": args.run_name,
        "non_linearity": args.non_linearity,
        "training_seeds": args.training_seeds,
        "iteration_counts": args.iteration_counts,
        "eval_batch_size": args.eval_batch_size,
        "max_test_samples": args.max_test_samples,
        "residual_mode": args.residual_mode,
        "kkt_eps": args.kkt_eps,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "selected_runs": [f"{row.non_linearity}/{row.run_name}/seed_{row.seed}" for row in selected],
    }
    _write_json(output_root / f"config_shard_{args.shard_index}.json", config)

    print(f"[residual] input_root={input_root}")
    print(f"[residual] output_root={output_root}")
    print(f"[residual] selected_runs={len(selected)}/{len(rows)} shard={args.shard_index}/{args.num_shards}")
    if args.dry_run:
        for row in selected:
            print(f"[dry-run] {row.non_linearity} {row.run_name} seed={row.seed} checkpoint={row.checkpoint_path}")
        return

    if raw_path.exists():
        raw_path.unlink()

    for run in selected:
        for iteration_count in args.iteration_counts:
            print(f"[run] {run.non_linearity} {run.run_name} seed={run.seed} K={iteration_count}")
            context = _build_eval_context(
                run,
                device=device,
                eval_batch_size=args.eval_batch_size,
                no_download=args.no_download,
                inference_iterations_override=int(iteration_count),
            )
            metrics = _evaluate_context(
                context,
                max_test_samples=args.max_test_samples,
                residual_mode=args.residual_mode,
                kkt_eps=args.kkt_eps,
            )
            batch_rows = []
            for index, (layer, values) in enumerate(zip(context["free_layers"], metrics["layer_values"])):
                batch_rows.append(
                    {
                        "model_label": args.model_label,
                        "non_linearity": run.non_linearity,
                        "residual_mode": metrics["layer_residual_modes"][index],
                        "run_name": run.run_name,
                        "seed": run.seed,
                        "voltage_amp": run.voltage_amp,
                        "current_amp": run.current_amp,
                        "checkpoint_path": str(run.checkpoint_path),
                        "iteration_count": int(iteration_count),
                        "layer_name": getattr(layer, "name", f"free_layer_{index}"),
                        "layer_index": _layer_id(layer, index),
                        "layer_role": _layer_role(index, len(context["free_layers"])),
                        "num_examples": metrics["num_examples"],
                        "accuracy": metrics["accuracy"],
                        "loss": metrics["loss"],
                        **_stats(values),
                    }
                )
            batch_rows.append(
                {
                    "model_label": args.model_label,
                    "non_linearity": run.non_linearity,
                    "residual_mode": args.residual_mode,
                    "run_name": run.run_name,
                    "seed": run.seed,
                    "voltage_amp": run.voltage_amp,
                    "current_amp": run.current_amp,
                    "checkpoint_path": str(run.checkpoint_path),
                    "iteration_count": int(iteration_count),
                    "layer_name": "overall",
                    "layer_index": -1,
                    "layer_role": "overall",
                    "num_examples": metrics["num_examples"],
                    "accuracy": metrics["accuracy"],
                    "loss": metrics["loss"],
                    **_stats(metrics["overall_values"]),
                }
            )
            _write_rows(raw_path, RESIDUAL_COLUMNS, batch_rows)

    write_summaries_and_plots(output_root)
    print(f"[done] wrote {output_root / 'residual_vs_iterations.csv'}")


if __name__ == "__main__":
    main()
