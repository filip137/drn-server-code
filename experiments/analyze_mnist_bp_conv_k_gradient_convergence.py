#!/usr/bin/env python3
"""Compare Conv MNIST BP gradients across K against a high-K reference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
LABS_DIR = REPO_ROOT / "labs"
for path in (REPO_ROOT, LABS_DIR, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_mnist_bp_conv_good_run_saturation import (  # noqa: E402
    RunSpec,
    _build_context,
    _layer_saturation,
    _load_specs,
    _normalize_run_dir,
    _path_part,
    _target_from_label,
)
from labs.mnist_train import _build_tracking_minimizer, _json_sanitize  # noqa: E402
from training.sgd import Backprop  # noqa: E402


SUMMARY_COLUMNS = [
    "conv_depth",
    "run_name",
    "phase",
    "t",
    "k",
    "ref_t",
    "ref_k",
    "source_inference_k",
    "source_training_k",
    "num_batches",
    "num_samples",
    "input_gain",
    "v_off",
    "target_label",
    "target_saturation",
    "lr_reference",
    "best_test_accuracy",
    "final_test_accuracy",
    "all_grad_l2_mean",
    "all_ref_grad_l2_mean",
    "all_diff_l2_mean",
    "all_cosine_mean",
    "all_rel_error_mean",
    "all_rel_error_max",
    "all_norm_ratio_mean",
    "weight_grad_l2_mean",
    "weight_ref_grad_l2_mean",
    "weight_diff_l2_mean",
    "weight_cosine_mean",
    "weight_rel_error_mean",
    "weight_rel_error_max",
    "weight_norm_ratio_mean",
    "log_weight_grad_l2_mean",
    "log_weight_ref_grad_l2_mean",
    "log_weight_diff_l2_mean",
    "log_weight_cosine_mean",
    "log_weight_rel_error_mean",
    "log_weight_rel_error_max",
    "log_weight_norm_ratio_mean",
    "hidden_saturation_mean",
    "run_dir",
]

PARAM_COLUMNS = [
    "conv_depth",
    "run_name",
    "phase",
    "t",
    "k",
    "ref_t",
    "ref_k",
    "param_index",
    "param_name",
    "param_class",
    "is_weight",
    "num_batches",
    "grad_l2_mean",
    "ref_grad_l2_mean",
    "diff_l2_mean",
    "diff_l2_max",
    "cosine_mean",
    "rel_error_mean",
    "rel_error_max",
    "norm_ratio_mean",
    "grad_zero_fraction_mean",
    "run_dir",
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _parse_k_values(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("Expected at least one K value.")
    if any(value <= 0 for value in values):
        raise ValueError(f"Expected positive K values, got {raw!r}.")
    return values


def _mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def _max(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else math.nan


def _learning_rates(spec: RunSpec, num_params: int, phase: str) -> list[float]:
    raw = spec.source_config.get("lr", [])
    if isinstance(raw, (int, float)):
        values = [float(raw)]
    else:
        values = [float(value) for value in raw]
    if not values:
        values = [math.nan]
    if phase == "final":
        final = spec.metrics.get("final_learning_rate")
        if isinstance(final, (int, float)):
            values = [float(final)]
        elif final:
            values = [float(value) for value in final]
    if len(values) < num_params:
        values.extend([values[-1]] * (num_params - len(values)))
    return values[:num_params]


def _parameter_name(param: object, index: int) -> str:
    return str(getattr(param, "name", f"param_{index}")).strip()


def _is_weight(param: object) -> bool:
    return "Weight" in param.__class__.__name__ or "Weight" in _parameter_name(param, -1)


def _compare_vectors(vec: torch.Tensor, ref: torch.Tensor) -> dict:
    vec = vec.detach().flatten().to(dtype=torch.float64, device="cpu")
    ref = ref.detach().flatten().to(dtype=torch.float64, device="cpu")
    norm = float(torch.linalg.vector_norm(vec).item())
    ref_norm = float(torch.linalg.vector_norm(ref).item())
    diff_norm = float(torch.linalg.vector_norm(vec - ref).item())
    if norm > 0.0 and ref_norm > 0.0:
        cosine = float(torch.dot(vec, ref).item() / (norm * ref_norm))
    else:
        cosine = math.nan
    return {
        "grad_l2": norm,
        "ref_grad_l2": ref_norm,
        "diff_l2": diff_norm,
        "cosine": cosine,
        "rel_error": diff_norm / ref_norm if ref_norm > 0.0 else math.nan,
        "norm_ratio": norm / ref_norm if ref_norm > 0.0 else math.nan,
    }


def _flatten(
    params: list[object],
    grads: list[torch.Tensor],
    *,
    weights_only: bool = False,
    log_weights: bool = False,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for param, grad in zip(params, grads):
        if weights_only and not _is_weight(param):
            continue
        value = grad.detach()
        if log_weights:
            if not _is_weight(param):
                continue
            value = param.state.detach() * value
        pieces.append(value.flatten().to(dtype=torch.float64, device="cpu"))
    if not pieces:
        return torch.empty(0, dtype=torch.float64)
    return torch.cat(pieces)


def _hidden_saturation(context: dict) -> float:
    model_cfg = context["model_cfg"]
    non_linearity = str(model_cfg.get("non_linearity", ""))
    hard_sigmoid_param = dict(model_cfg.get("hard_sigmoid_param", {}))
    hidden_layers = context["energy_fn"].layers()[1:-1]
    saturated = 0.0
    total = 0
    for layer in hidden_layers:
        fraction, count = _layer_saturation(
            layer.state.detach(),
            non_linearity=non_linearity,
            hard_sigmoid_param=hard_sigmoid_param,
            perfect_eps=1.0e-8,
        )
        if math.isfinite(fraction) and count > 0:
            saturated += fraction * count
            total += count
    return saturated / total if total else math.nan


def _prefetch_batches(context: dict, *, split: str, max_samples: int) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
    loader = context["train_loader"] if split == "train" else context["test_loader"]
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    seen = 0
    for images, labels in loader:
        if seen >= max_samples:
            break
        remaining = max_samples - seen
        if int(images.shape[0]) > remaining:
            images = images[:remaining]
            labels = labels[:remaining]
        batches.append((images.detach().cpu().clone(), labels.detach().cpu().clone()))
        seen += int(images.shape[0])
    return batches, seen


def _build_phase_context(
    spec: RunSpec,
    *,
    phase: str,
    t: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    return _build_context(
        spec,
        device=device,
        load_final=(phase == "final"),
        batch_size=args.batch_size,
        no_download=args.no_download,
        dataset_root=args.dataset_root,
        inference_iterations_override=t,
    )


def _compute_for_k(
    spec: RunSpec,
    *,
    phase: str,
    t: int,
    k: int,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    context = _build_phase_context(spec, phase=phase, t=t, args=args, device=device)
    params = context["energy_fn"].params()
    base_states = [param.state.detach().clone() for param in params]
    minimizer_training = _build_tracking_minimizer(
        context["energy_fn"],
        context["free_layers"],
        context["model_cfg"],
        spec.source_config["energy_minimizer"]["mode"],
        num_iterations=k,
        voltage_amp=context["energy_fn"]._voltage_amp,
        current_amp=context["energy_fn"]._current_amp,
    )
    estimator = Backprop(params, context["free_layers"], context["cost_fn"], minimizer_training)

    records: list[dict] = []
    for batch_index, (images_cpu, labels_cpu) in enumerate(batches):
        for param, base in zip(params, base_states):
            param.state = base.detach().clone()
        images = images_cpu.to(device)
        labels = labels_cpu.to(device)
        context["network"].set_input(images, reset=True)
        with torch.no_grad():
            context["minimizer"].compute_equilibrium()
        context["cost_fn"].set_target(labels)
        hidden_saturation = _hidden_saturation(context)
        grads = estimator.compute_gradient()[: len(params)]
        param_records = []
        for index, (param, grad) in enumerate(zip(params, grads)):
            flat_grad = grad.detach().flatten().to(dtype=torch.float64, device="cpu")
            param_records.append(
                {
                    "index": index,
                    "name": _parameter_name(param, index),
                    "class": param.__class__.__name__,
                    "is_weight": _is_weight(param),
                    "vector": flat_grad,
                    "zero_fraction": float((grad.detach().abs() <= 1.0e-12).float().mean().item()),
                }
            )
        records.append(
            {
                "batch_index": batch_index,
                "num_samples": int(images_cpu.shape[0]),
                "hidden_saturation": hidden_saturation,
                "all": _flatten(params, grads),
                "weights": _flatten(params, grads, weights_only=True),
                "log_weights": _flatten(params, grads, log_weights=True),
                "params": param_records,
            }
        )
        del grads
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"records": records, "model_cfg": context["model_cfg"], "params": params}


def _aggregate_summary(
    spec: RunSpec,
    *,
    phase: str,
    t: int,
    k: int,
    ref_t: int,
    ref_k: int,
    current: dict,
    reference: dict,
    num_samples: int,
) -> dict:
    model_cfg = current["model_cfg"]
    hard_sigmoid_cfg = model_cfg.get("hard_sigmoid_param", {})
    v_off = hard_sigmoid_cfg.get("v_off", hard_sigmoid_cfg.get("v_max", math.nan))
    target_label = _path_part(spec.run_dir, "target_")
    lrs = _learning_rates(spec, len(current["params"]), phase)
    comparisons: dict[str, list[dict]] = {"all": [], "weights": [], "log_weights": []}
    hidden = []
    for row, ref in zip(current["records"], reference["records"]):
        hidden.append(float(row["hidden_saturation"]))
        for key in comparisons:
            comparisons[key].append(_compare_vectors(row[key], ref[key]))

    def values(key: str, field: str) -> list[float]:
        return [float(item[field]) for item in comparisons[key]]

    return {
        "conv_depth": len(model_cfg.get("conv_pipeline") or []),
        "run_name": spec.run_dir.parent.name,
        "phase": phase,
        "t": t,
        "k": k,
        "ref_t": ref_t,
        "ref_k": ref_k,
        "source_inference_k": int(model_cfg["num_iterations_inference"]),
        "source_training_k": int(
            model_cfg.get("num_iterations_training", model_cfg["num_iterations_inference"])
        ),
        "num_batches": len(current["records"]),
        "num_samples": num_samples,
        "input_gain": float(model_cfg.get("input_gain", math.nan)),
        "v_off": float(v_off),
        "target_label": target_label,
        "target_saturation": _target_from_label(target_label),
        "lr_reference": lrs[0] if lrs else math.nan,
        "best_test_accuracy": spec.metrics.get("best_test_accuracy", math.nan),
        "final_test_accuracy": spec.metrics.get("final_test_accuracy", math.nan),
        "all_grad_l2_mean": _mean(values("all", "grad_l2")),
        "all_ref_grad_l2_mean": _mean(values("all", "ref_grad_l2")),
        "all_diff_l2_mean": _mean(values("all", "diff_l2")),
        "all_cosine_mean": _mean(values("all", "cosine")),
        "all_rel_error_mean": _mean(values("all", "rel_error")),
        "all_rel_error_max": _max(values("all", "rel_error")),
        "all_norm_ratio_mean": _mean(values("all", "norm_ratio")),
        "weight_grad_l2_mean": _mean(values("weights", "grad_l2")),
        "weight_ref_grad_l2_mean": _mean(values("weights", "ref_grad_l2")),
        "weight_diff_l2_mean": _mean(values("weights", "diff_l2")),
        "weight_cosine_mean": _mean(values("weights", "cosine")),
        "weight_rel_error_mean": _mean(values("weights", "rel_error")),
        "weight_rel_error_max": _max(values("weights", "rel_error")),
        "weight_norm_ratio_mean": _mean(values("weights", "norm_ratio")),
        "log_weight_grad_l2_mean": _mean(values("log_weights", "grad_l2")),
        "log_weight_ref_grad_l2_mean": _mean(values("log_weights", "ref_grad_l2")),
        "log_weight_diff_l2_mean": _mean(values("log_weights", "diff_l2")),
        "log_weight_cosine_mean": _mean(values("log_weights", "cosine")),
        "log_weight_rel_error_mean": _mean(values("log_weights", "rel_error")),
        "log_weight_rel_error_max": _max(values("log_weights", "rel_error")),
        "log_weight_norm_ratio_mean": _mean(values("log_weights", "norm_ratio")),
        "hidden_saturation_mean": _mean(hidden),
        "run_dir": str(spec.run_dir),
    }


def _aggregate_params(
    spec: RunSpec,
    *,
    phase: str,
    t: int,
    k: int,
    ref_t: int,
    ref_k: int,
    current: dict,
    reference: dict,
) -> list[dict]:
    model_cfg = current["model_cfg"]
    grouped: dict[int, list[dict]] = {}
    for row, ref in zip(current["records"], reference["records"]):
        for param_row, ref_param_row in zip(row["params"], ref["params"]):
            comparison = _compare_vectors(param_row["vector"], ref_param_row["vector"])
            grouped.setdefault(int(param_row["index"]), []).append(
                {
                    **param_row,
                    **comparison,
                }
            )
    rows = []
    for index, values in sorted(grouped.items()):
        first = values[0]
        rows.append(
            {
                "conv_depth": len(model_cfg.get("conv_pipeline") or []),
                "run_name": spec.run_dir.parent.name,
                "phase": phase,
                "t": t,
                "k": k,
                "ref_t": ref_t,
                "ref_k": ref_k,
                "param_index": index,
                "param_name": first["name"],
                "param_class": first["class"],
                "is_weight": bool(first["is_weight"]),
                "num_batches": len(values),
                "grad_l2_mean": _mean([float(value["grad_l2"]) for value in values]),
                "ref_grad_l2_mean": _mean([float(value["ref_grad_l2"]) for value in values]),
                "diff_l2_mean": _mean([float(value["diff_l2"]) for value in values]),
                "diff_l2_max": _max([float(value["diff_l2"]) for value in values]),
                "cosine_mean": _mean([float(value["cosine"]) for value in values]),
                "rel_error_mean": _mean([float(value["rel_error"]) for value in values]),
                "rel_error_max": _max([float(value["rel_error"]) for value in values]),
                "norm_ratio_mean": _mean([float(value["norm_ratio"]) for value in values]),
                "grad_zero_fraction_mean": _mean(
                    [float(value["zero_fraction"]) for value in values]
                ),
                "run_dir": str(spec.run_dir),
            }
        )
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _json_sanitize(row.get(column, "")) for column in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--k-values", default="1,2,3,4,6,8,10,12,16,24,32,48,64")
    parser.add_argument(
        "--t-values",
        default=None,
        help=(
            "Comma-separated free-inference iteration values. If omitted, T is tied "
            "to K for backward-compatible paired sweeps."
        ),
    )
    parser.add_argument("--ref-k", type=int, default=64)
    parser.add_argument("--ref-t", type=int, default=None)
    parser.add_argument("--phase", choices=("init", "final", "both"), default="both")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--dataset-root", default=str(REPO_ROOT / "data"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = _parse_k_values(args.k_values)
    t_values = _parse_k_values(args.t_values) if args.t_values is not None else None
    ref_t = int(args.ref_t) if args.ref_t is not None else int(args.ref_k)
    if args.ref_k not in k_values:
        k_values.append(int(args.ref_k))
        k_values = sorted(k_values)
    if t_values is not None and ref_t not in t_values:
        t_values.append(ref_t)
        t_values = sorted(t_values)
    tk_pairs = (
        [(t, k) for t in t_values for k in k_values]
        if t_values is not None
        else [(k, k) for k in k_values]
    )
    phases = ["init", "final"] if args.phase == "both" else [args.phase]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dirs = [_normalize_run_dir(path) for path in args.run_dir]
    specs = _load_specs(run_dirs)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "diagnostic_config.json").write_text(
        json.dumps(
            _json_sanitize(
                {
                    "run_dirs": [str(path) for path in run_dirs],
                    "k_values": k_values,
                    "t_values": t_values,
                    "ref_k": args.ref_k,
                    "ref_t": ref_t,
                    "tk_pairs": tk_pairs,
                    "phases": phases,
                    "split": args.split,
                    "batch_size": args.batch_size,
                    "max_samples": args.max_samples,
                    "dataset_root": args.dataset_root,
                    "device": str(device),
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )

    summary_rows: list[dict] = []
    param_rows: list[dict] = []
    total = len(specs) * len(phases)
    count = 0
    for spec in specs:
        for phase in phases:
            count += 1
            conv_depth = len(spec.model_cfg.get("conv_pipeline") or [])
            print(
                f"[phase {count}/{total}] conv{conv_depth} {spec.run_dir.parent.name} "
                f"{phase} ref_t={ref_t} ref_k={args.ref_k}",
                flush=True,
            )
            data_context = _build_phase_context(
                spec, phase=phase, t=ref_t, args=args, device=device
            )
            batches, num_samples = _prefetch_batches(
                data_context, split=args.split, max_samples=args.max_samples
            )
            if not batches:
                raise RuntimeError(f"No batches available for {spec.run_dir}.")
            reference = _compute_for_k(
                spec,
                phase=phase,
                t=ref_t,
                k=args.ref_k,
                batches=batches,
                args=args,
                device=device,
            )
            by_tk = {(ref_t, args.ref_k): reference}
            for t, k in tk_pairs:
                if t == ref_t and k == args.ref_k:
                    current = reference
                else:
                    print(f"  [t,k] {t},{k}", flush=True)
                    current = _compute_for_k(
                        spec,
                        phase=phase,
                        t=t,
                        k=k,
                        batches=batches,
                        args=args,
                        device=device,
                    )
                    by_tk[(t, k)] = current
                summary_rows.append(
                    _aggregate_summary(
                        spec,
                        phase=phase,
                        t=t,
                        k=k,
                        ref_t=ref_t,
                        ref_k=args.ref_k,
                        current=current,
                        reference=reference,
                        num_samples=num_samples,
                    )
                )
                param_rows.extend(
                    _aggregate_params(
                        spec,
                        phase=phase,
                        t=t,
                        k=k,
                        ref_t=ref_t,
                        ref_k=args.ref_k,
                        current=current,
                        reference=reference,
                    )
                )
            del by_tk, reference
            if device.type == "cuda":
                torch.cuda.empty_cache()

    _write_csv(output_root / "summary.csv", SUMMARY_COLUMNS, summary_rows)
    _write_csv(output_root / "param_summary.csv", PARAM_COLUMNS, param_rows)
    print(f"[done] wrote {output_root}", flush=True)


if __name__ == "__main__":
    main()
