#!/usr/bin/env python3
"""Zero-shot projection from a trained ReLU MLP into a bounded sign-split DRN."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DEFAULT_TEACHER_CHECKPOINT = Path(
    "/home/filip/server_code/simulation_results/trained_models/"
    "mnist_relu_mlp_784_100_10_bp_10epochs_20260504-102349/model_final.pt"
)
DEFAULT_DATA_ROOT = Path("/home/filip/server_code/data")
DEFAULT_GMAX_VALUES = (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
DEFAULT_PROJECTION_RULES = (
    "global_max",
    "layer_max",
    "layer_pct_95",
    "layer_pct_98",
    "layer_pct_99",
    "layer_pct_99_5",
    "layer_l2",
)
CSV_FIELDNAMES = (
    "gmax",
    "projection_rule",
    "scale_w1",
    "scale_w2",
    "student_accuracy",
    "teacher_accuracy",
    "teacher_student_agreement",
    "student_ce",
    "teacher_ce",
    "teacher_student_kl",
    "logit_mse",
    "student_logit_std",
    "teacher_logit_std",
    "logit_std_ratio",
    "current_conductance_zero_fraction",
    "current_conductance_gmax_fraction",
    "hidden_to_output_zero_fraction",
    "hidden_to_output_gmax_fraction",
    "num_samples",
    "num_iterations",
    "seconds",
)


@dataclass(frozen=True)
class ReluTeacherWeights:
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor


@dataclass(frozen=True)
class BoundedProjection:
    rule: str
    gmax: float
    scale_w1: float
    scale_w2: float
    current_conductance: torch.Tensor
    current_bias: torch.Tensor
    hidden_to_output_conductance: torch.Tensor
    clipped_w1: torch.Tensor
    clipped_w2: torch.Tensor


def _safe_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _nonzero_scale(denominator: torch.Tensor, numerator: float) -> float:
    denom = _safe_float(denominator)
    if not math.isfinite(denom) or denom <= 0.0:
        return 0.0
    return float(numerator) / denom


def _max_abs_scale(tensor: torch.Tensor, gmax: float) -> float:
    return _nonzero_scale(tensor.abs().max(), gmax)


def _percentile_scale(tensor: torch.Tensor, gmax: float, percentile: float) -> float:
    flat = tensor.abs().reshape(-1)
    nonzero = flat[flat > 0]
    if nonzero.numel() == 0:
        return 0.0
    q = torch.quantile(nonzero, percentile / 100.0)
    return _nonzero_scale(q, gmax)


def _l2_scale(tensor: torch.Tensor, gmax: float) -> float:
    flat = tensor.abs().reshape(-1)
    if flat.numel() == 0:
        return 0.0
    rms = torch.linalg.vector_norm(flat) / math.sqrt(float(flat.numel()))
    return _nonzero_scale(rms, gmax)


def parse_projection_percentile(rule: str) -> float:
    if not rule.startswith("layer_pct_"):
        raise ValueError(f"Expected projection rule to start with 'layer_pct_', got {rule!r}.")
    suffix = rule.removeprefix("layer_pct_").replace("_", ".")
    try:
        return float(suffix)
    except ValueError as exc:
        raise ValueError(f"Expected percentile suffix like 99_5, got {rule!r}.") from exc


def projection_scales(rule: str, gmax: float, w1: torch.Tensor, w2: torch.Tensor) -> tuple[float, float]:
    if gmax <= 0.0 or not math.isfinite(gmax):
        raise ValueError(f"Expected gmax to be a positive finite float, got {gmax!r}.")
    if rule == "global_max":
        scale = _nonzero_scale(torch.maximum(w1.abs().max(), w2.abs().max()), gmax)
        return scale, scale
    if rule == "layer_max":
        return _max_abs_scale(w1, gmax), _max_abs_scale(w2, gmax)
    if rule.startswith("layer_pct_"):
        percentile = parse_projection_percentile(rule)
        return _percentile_scale(w1, gmax, percentile), _percentile_scale(w2, gmax, percentile)
    if rule == "layer_l2":
        return _l2_scale(w1, gmax), _l2_scale(w2, gmax)
    raise ValueError(
        "Expected projection rule to be one of "
        f"{', '.join(DEFAULT_PROJECTION_RULES)}, got {rule!r}."
    )


def _scaled_clipped_signed(weight: torch.Tensor, scale: float, gmax: float) -> torch.Tensor:
    return (weight * float(scale)).clamp(min=-float(gmax), max=float(gmax))


def build_sign_split_projection(
    weights: ReluTeacherWeights,
    *,
    rule: str,
    gmax: float,
) -> BoundedProjection:
    """Project signed ReLU weights to bounded nonnegative DRN conductances."""
    w1, b1, w2 = weights.w1, weights.b1, weights.w2
    input_dim, hidden_dim = w1.shape
    hidden_dim_w2, output_dim = w2.shape
    if hidden_dim_w2 != hidden_dim:
        raise ValueError(
            f"Expected w2 first dimension to equal hidden_dim {hidden_dim}, got {hidden_dim_w2}."
        )
    if b1.shape != (hidden_dim,):
        raise ValueError(f"Expected b1 shape ({hidden_dim},), got {tuple(b1.shape)}.")

    scale_w1, scale_w2 = projection_scales(rule, gmax, w1, w2)
    clipped_w1 = _scaled_clipped_signed(w1, scale_w1, gmax)
    clipped_w2 = _scaled_clipped_signed(w2, scale_w2, gmax)

    w1_pos = clipped_w1.clamp_min(0.0)
    w1_neg = (-clipped_w1).clamp_min(0.0)
    current_conductance = torch.zeros(
        (2 * input_dim, 2 * hidden_dim), dtype=clipped_w1.dtype, device=clipped_w1.device
    )
    current_conductance[:input_dim, :hidden_dim] = w1_pos
    current_conductance[input_dim:, :hidden_dim] = w1_neg
    current_conductance[:input_dim, hidden_dim:] = w1_neg
    current_conductance[input_dim:, hidden_dim:] = w1_pos

    current_bias = torch.cat((b1 * scale_w1, -b1 * scale_w1), dim=0)
    hidden_to_output_conductance = torch.cat(
        (clipped_w2.clamp_min(0.0), (-clipped_w2).clamp_min(0.0)),
        dim=0,
    )

    return BoundedProjection(
        rule=rule,
        gmax=float(gmax),
        scale_w1=float(scale_w1),
        scale_w2=float(scale_w2),
        current_conductance=current_conductance,
        current_bias=current_bias,
        hidden_to_output_conductance=hidden_to_output_conductance,
        clipped_w1=clipped_w1,
        clipped_w2=clipped_w2,
    )


def reconstruct_effective_weights(projection: BoundedProjection) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed effective W1/W2 represented by a sign-split projection."""
    input_dim2, hidden_dim2 = projection.current_conductance.shape
    input_dim = input_dim2 // 2
    hidden_dim = hidden_dim2 // 2
    w1_eff = (
        projection.current_conductance[:input_dim, :hidden_dim]
        - projection.current_conductance[input_dim:, :hidden_dim]
    )
    w2_eff = (
        projection.hidden_to_output_conductance[:hidden_dim, :]
        - projection.hidden_to_output_conductance[hidden_dim:, :]
    )
    return w1_eff, w2_eff


def tensor_stats(tensor: torch.Tensor, *, gmax: float | None = None) -> dict[str, float]:
    data = tensor.detach()
    if data.numel() == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "zero_fraction": float("nan"),
            "gmax_fraction": float("nan"),
        }
    stats = {
        "min": float(data.min().cpu().item()),
        "max": float(data.max().cpu().item()),
        "mean": float(data.mean().cpu().item()),
        "zero_fraction": float((data == 0).to(torch.float32).mean().cpu().item()),
    }
    if gmax is None:
        stats["gmax_fraction"] = float("nan")
    else:
        tol = max(abs(float(gmax)) * 1e-6, 1e-12)
        stats["gmax_fraction"] = float((data >= float(gmax) - tol).to(torch.float32).mean().cpu().item())
    return stats


def projection_stats(projection: BoundedProjection) -> dict[str, object]:
    w1_eff, w2_eff = reconstruct_effective_weights(projection)
    return {
        "rule": projection.rule,
        "gmax": projection.gmax,
        "scale_w1": projection.scale_w1,
        "scale_w2": projection.scale_w2,
        "current_conductance": tensor_stats(projection.current_conductance, gmax=projection.gmax),
        "hidden_to_output_conductance": tensor_stats(
            projection.hidden_to_output_conductance, gmax=projection.gmax
        ),
        "current_bias": tensor_stats(projection.current_bias, gmax=None),
        "effective_w1": tensor_stats(w1_eff, gmax=None),
        "effective_w2": tensor_stats(w2_eff, gmax=None),
    }


def load_relu_teacher_weights(path: Path) -> ReluTeacherWeights:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Expected teacher checkpoint to be a dict containing 'model_state_dict', "
            f"got {type(checkpoint).__name__} from {path}."
        )
    state = checkpoint["model_state_dict"]
    required = ("net.1.weight", "net.1.bias", "net.3.weight", "net.3.bias")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(
            "Expected teacher state_dict keys net.1.weight/net.1.bias/net.3.weight/net.3.bias, "
            f"missing {missing!r} from {path}."
        )
    return ReluTeacherWeights(
        w1=state["net.1.weight"].detach().to(torch.float32).T.contiguous(),
        b1=state["net.1.bias"].detach().to(torch.float32).contiguous(),
        w2=state["net.3.weight"].detach().to(torch.float32).T.contiguous(),
        b2=state["net.3.bias"].detach().to(torch.float32).contiguous(),
    )


def teacher_logits(x_flat: torch.Tensor, weights: ReluTeacherWeights) -> torch.Tensor:
    return torch.relu(x_flat @ weights.w1 + weights.b1) @ weights.w2 + weights.b2


def drn_logits(
    x_flat: torch.Tensor,
    projection: BoundedProjection,
    *,
    num_iterations: int,
) -> torch.Tensor:
    """Compute direct current-input perfect-diode DRN logits by quadratic updates."""
    if num_iterations <= 0:
        raise ValueError(f"Expected num_iterations to be positive, got {num_iterations}.")
    input_dim = x_flat.shape[1]
    hidden_dim2, output_dim = projection.hidden_to_output_conductance.shape
    hidden_dim = hidden_dim2 // 2

    current_conductance = projection.current_conductance.to(device=x_flat.device, dtype=x_flat.dtype)
    current_bias = projection.current_bias.to(device=x_flat.device, dtype=x_flat.dtype)
    hidden_to_output = projection.hidden_to_output_conductance.to(device=x_flat.device, dtype=x_flat.dtype)

    x_signed = torch.cat((x_flat, -x_flat), dim=1)
    if current_conductance.shape[0] != x_signed.shape[1]:
        raise ValueError(
            f"Expected current_conductance input dimension {x_signed.shape[1]}, "
            f"got {current_conductance.shape[0]}."
        )
    input_current = x_signed @ current_conductance + current_bias

    hidden = torch.zeros((x_flat.shape[0], hidden_dim2), device=x_flat.device, dtype=x_flat.dtype)
    logits = torch.zeros((x_flat.shape[0], output_dim), device=x_flat.device, dtype=x_flat.dtype)
    row_sum = hidden_to_output.sum(dim=1)
    col_sum = hidden_to_output.sum(dim=0)
    row_safe = torch.where(row_sum > 0, row_sum, torch.ones_like(row_sum))
    col_safe = torch.where(col_sum > 0, col_sum, torch.ones_like(col_sum))

    for _ in range(int(num_iterations)):
        pre_hidden = (input_current + logits @ hidden_to_output.T) / row_safe
        hidden = torch.cat(
            (pre_hidden[:, :hidden_dim].clamp_min(0.0), pre_hidden[:, hidden_dim:].clamp_max(0.0)),
            dim=1,
        )
        logits = (hidden @ hidden_to_output) / col_safe
    return logits


def make_mnist_loader(
    *,
    data_root: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    num_samples: int | None,
) -> DataLoader:
    if split not in {"train", "test"}:
        raise ValueError(f"Expected split to be 'train' or 'test', got {split!r}.")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        root=str(data_root),
        train=(split == "train"),
        download=False,
        transform=transform,
    )
    if num_samples is not None:
        if num_samples <= 0:
            raise ValueError(f"Expected num_samples to be positive, got {num_samples}.")
        count = min(int(num_samples), len(dataset))
        dataset = Subset(dataset, range(count))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def _std_from_sums(total: int, sum_value: float, sum_square: float) -> float:
    if total <= 1:
        return 0.0
    mean = sum_value / total
    var = max(sum_square / total - mean * mean, 0.0)
    return math.sqrt(var)


def evaluate_projection(
    projection: BoundedProjection,
    weights: ReluTeacherWeights,
    loader: DataLoader,
    *,
    device: torch.device,
    num_iterations: int,
) -> dict[str, float]:
    weights_device = ReluTeacherWeights(
        w1=weights.w1.to(device),
        b1=weights.b1.to(device),
        w2=weights.w2.to(device),
        b2=weights.b2.to(device),
    )
    total = 0
    correct_student = 0
    correct_teacher = 0
    agreement = 0
    student_ce = 0.0
    teacher_ce = 0.0
    kl_sum = 0.0
    mse_sum = 0.0
    student_sum = 0.0
    student_sum_sq = 0.0
    teacher_sum = 0.0
    teacher_sum_sq = 0.0
    logit_count = 0
    start = time.time()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32).flatten(start_dim=1)
            y = y.to(device=device)
            t_logits = teacher_logits(x, weights_device)
            s_logits = drn_logits(x, projection, num_iterations=num_iterations)

            t_pred = t_logits.argmax(dim=1)
            s_pred = s_logits.argmax(dim=1)
            batch_size = int(y.numel())
            total += batch_size
            correct_teacher += int((t_pred == y).sum().item())
            correct_student += int((s_pred == y).sum().item())
            agreement += int((s_pred == t_pred).sum().item())
            teacher_ce += float(F.cross_entropy(t_logits, y, reduction="sum").item())
            student_ce += float(F.cross_entropy(s_logits, y, reduction="sum").item())
            kl_sum += float(
                F.kl_div(
                    F.log_softmax(s_logits, dim=1),
                    F.softmax(t_logits, dim=1),
                    reduction="sum",
                ).item()
            )
            diff = s_logits - t_logits
            mse_sum += float((diff * diff).sum().item())
            student_sum += float(s_logits.sum().item())
            student_sum_sq += float((s_logits * s_logits).sum().item())
            teacher_sum += float(t_logits.sum().item())
            teacher_sum_sq += float((t_logits * t_logits).sum().item())
            logit_count += int(s_logits.numel())

    if total == 0:
        raise RuntimeError("Evaluation loader produced no samples.")
    student_std = _std_from_sums(logit_count, student_sum, student_sum_sq)
    teacher_std = _std_from_sums(logit_count, teacher_sum, teacher_sum_sq)
    return {
        "student_accuracy": correct_student / total,
        "teacher_accuracy": correct_teacher / total,
        "teacher_student_agreement": agreement / total,
        "student_ce": student_ce / total,
        "teacher_ce": teacher_ce / total,
        "teacher_student_kl": kl_sum / total,
        "logit_mse": mse_sum / logit_count,
        "student_logit_std": student_std,
        "teacher_logit_std": teacher_std,
        "logit_std_ratio": student_std / teacher_std if teacher_std > 0.0 else float("nan"),
        "num_samples": total,
        "seconds": time.time() - start,
    }


def flatten_row(
    *,
    projection: BoundedProjection,
    metrics: dict[str, float],
    stats: dict[str, object],
    num_iterations: int,
) -> dict[str, object]:
    current_stats = stats["current_conductance"]
    output_stats = stats["hidden_to_output_conductance"]
    row = {
        "gmax": projection.gmax,
        "projection_rule": projection.rule,
        "scale_w1": projection.scale_w1,
        "scale_w2": projection.scale_w2,
        "num_iterations": num_iterations,
        **metrics,
        "current_conductance_zero_fraction": current_stats["zero_fraction"],
        "current_conductance_gmax_fraction": current_stats["gmax_fraction"],
        "hidden_to_output_zero_fraction": output_stats["zero_fraction"],
        "hidden_to_output_gmax_fraction": output_stats["gmax_fraction"],
    }
    return row


def best_summary_row(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    return max(
        rows,
        key=lambda row: (
            float(row["student_accuracy"]),
            float(row["teacher_student_agreement"]),
            -float(row["current_conductance_gmax_fraction"])
            - float(row["hidden_to_output_gmax_fraction"]),
        ),
    )


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDNAMES})


def _jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
    return obj


def write_readme(path: Path, *, best_row: dict[str, object], summary_csv: Path, args: argparse.Namespace) -> None:
    content = f"""# Zero-Shot ReLU-To-Bounded-DRN Initialization Test

Teacher checkpoint: `{args.teacher_checkpoint}`

Best row by student accuracy:

- projection rule: `{best_row["projection_rule"]}`
- Gmax: `{best_row["gmax"]}`
- student test accuracy: `{float(best_row["student_accuracy"]):.6f}`
- teacher test accuracy: `{float(best_row["teacher_accuracy"]):.6f}`
- teacher/student agreement: `{float(best_row["teacher_student_agreement"]):.6f}`
- current conductance saturation: `{float(best_row["current_conductance_gmax_fraction"]):.6f}`
- hidden-to-output saturation: `{float(best_row["hidden_to_output_gmax_fraction"]):.6f}`

Full table: `{summary_csv.name}`

No optimizer steps, distillation, calibration, or data-selected scalar tuning were used.
"""
    path.write_text(content)


def run_debug_regression(
    *,
    weights: ReluTeacherWeights,
    data_root: Path,
    batch_size: int,
    device: torch.device,
    num_iterations: int,
    num_workers: int,
) -> dict[str, object]:
    loader = make_mnist_loader(
        data_root=data_root,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        num_samples=batch_size,
    )
    gmax = max(float(weights.w1.abs().max().item()), float(weights.w2.abs().max().item()))
    projection = build_sign_split_projection(weights, rule="layer_max", gmax=gmax)
    metrics = evaluate_projection(
        projection,
        weights,
        loader,
        device=device,
        num_iterations=num_iterations,
    )
    expected = 0.99609375
    tolerance = 0.03
    return {
        "expected_accuracy_about": expected,
        "expected_agreement_about": expected,
        "tolerance": tolerance,
        "metrics": metrics,
        "passed": (
            abs(metrics["student_accuracy"] - expected) <= tolerance
            and abs(metrics["teacher_student_agreement"] - expected) <= tolerance
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate zero-shot bounded sign-split DRN projections from a trained ReLU MLP."
    )
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gmax-values", type=float, nargs="+", default=list(DEFAULT_GMAX_VALUES))
    parser.add_argument("--projection-rules", nargs="+", default=list(DEFAULT_PROJECTION_RULES))
    parser.add_argument("--num-iterations", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-debug-regression", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Expected CUDA to be available for --device {args.device!r}, but it is not.")
    device = torch.device(args.device)

    teacher_path = args.teacher_checkpoint.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    if args.output_dir is None:
        date = datetime.now().strftime("%Y%m%d")
        output_dir = Path(
            f"/home/filip/server_code/simulation_results/experiments_labs/"
            f"relu_to_bounded_drn_zero_shot_gmax1e3_{date}"
        )
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = load_relu_teacher_weights(teacher_path)
    loader = make_mnist_loader(
        data_root=data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
    )

    rows: list[dict[str, object]] = []
    stats_by_key: dict[str, object] = {}
    for gmax in args.gmax_values:
        if gmax > 1e-3:
            raise ValueError(f"Expected all --gmax-values to be <= 1e-3, got {gmax}.")
        for rule in args.projection_rules:
            projection = build_sign_split_projection(weights, rule=rule, gmax=gmax)
            stats = projection_stats(projection)
            metrics = evaluate_projection(
                projection,
                weights,
                loader,
                device=device,
                num_iterations=args.num_iterations,
            )
            row = flatten_row(
                projection=projection,
                metrics=metrics,
                stats=stats,
                num_iterations=args.num_iterations,
            )
            rows.append(row)
            stats_by_key[f"gmax={gmax:g}/rule={rule}"] = stats
            print(
                f"[zero-shot] gmax={gmax:g} rule={rule} "
                f"acc={metrics['student_accuracy']:.4f} "
                f"agree={metrics['teacher_student_agreement']:.4f}"
            )

    best_row = best_summary_row(rows)
    debug_regression = None
    if args.run_debug_regression:
        debug_regression = run_debug_regression(
            weights=weights,
            data_root=data_root,
            batch_size=min(args.batch_size, 256),
            device=device,
            num_iterations=args.num_iterations,
            num_workers=args.num_workers,
        )
        print(f"[debug-regression] passed={debug_regression['passed']}")

    summary_csv = output_dir / "summary.csv"
    write_summary_csv(summary_csv, rows)
    (output_dir / "projection_stats.json").write_text(json.dumps(_jsonable(stats_by_key), indent=2))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "teacher_checkpoint": str(teacher_path),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "args": vars(args),
        "best_row": best_row,
        "debug_regression": debug_regression,
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))
    write_readme(output_dir / "README.md", best_row=best_row, summary_csv=summary_csv, args=args)
    print(f"[zero-shot] wrote {summary_csv}")
    print(
        "[zero-shot] best "
        f"gmax={best_row['gmax']} rule={best_row['projection_rule']} "
        f"acc={float(best_row['student_accuracy']):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
