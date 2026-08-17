#!/usr/bin/env python3
"""Diagnose why an affine finite conductance floor preserves MNIST accuracy.

This utility replays the checked-in one-hidden-layer perfect-diode DRN and
measures the mechanisms that can make a uniform conductance floor benign:
balanced input cancellation, differential-output common-mode rejection,
positive-homogeneous diode gates, and conductance-sum normalization.  It also
sweeps much more severe on/off ratios and deliberately breaks the uniform
floor assumptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from labs.tools.analyze_mnist_conductance_floor import (
    BIAS_KEY,
    W1_KEY,
    W2_KEY,
    _load_test_set,
    _load_weights,
)
from labs.tools.analyze_mnist_mapping_kl import predictive_divergence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze why a uniform affine conductance floor causes little "
            "functional drift in the MNIST DRN."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            ROOT
            / "examples/small_drn/mnist_ibm_devices/hwa_from_fp32.json"
        ),
    )
    parser.add_argument("--clean-weights", type=Path, required=True)
    parser.add_argument("--affine-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--g-min-us", type=float, default=9.0)
    parser.add_argument("--g-max-us", type=float, default=88.199997)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--replay-batch-size", type=int, default=512)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().to(torch.float64).flatten()
    quantiles = torch.quantile(
        flat,
        torch.tensor(
            [0.01, 0.5, 0.9, 0.99],
            dtype=flat.dtype,
            device=flat.device,
        ),
    )
    return {
        "mean": float(flat.mean()),
        "minimum": float(flat.min()),
        "p01": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "p99": float(quantiles[3]),
        "maximum": float(flat.max()),
    }


def _rms(values: torch.Tensor) -> float:
    return float(values.to(torch.float64).square().mean().sqrt())


def _accuracy(scores: torch.Tensor, targets: torch.Tensor) -> float:
    return float(
        (scores.argmax(1) == targets).to(torch.float32).mean()
    )


@torch.no_grad()
def trace_scores(
    inputs: torch.Tensor,
    *,
    numerator_w1: torch.Tensor,
    numerator_w2: torch.Tensor,
    bias: torch.Tensor,
    input_gain: float,
    iterations: int,
    batch_size: int,
    denominator_w1: torch.Tensor | None = None,
    denominator_w2: torch.Tensor | None = None,
    common_input_offset: float = 0.0,
) -> dict[str, Any]:
    """Replay the runtime update and retain per-iteration node states."""

    denominator_w1 = (
        numerator_w1 if denominator_w1 is None else denominator_w1
    )
    denominator_w2 = (
        numerator_w2 if denominator_w2 is None else denominator_w2
    )
    hidden_degree = denominator_w1.sum(0) + denominator_w2.sum(1)
    output_degree = denominator_w2.sum(0)
    if (hidden_degree <= 0.0).any() or (output_degree <= 0.0).any():
        raise ValueError(
            "Expected every replay conductance sum to be positive. Provided "
            "value contains a disconnected hidden or output node."
        )
    if numerator_w2.shape[1] % 2:
        raise ValueError(
            "Expected an even output width for paired differential scores. "
            f"Provided value: {numerator_w2.shape[1]!r}."
        )

    hidden_chunks: list[list[torch.Tensor]] = [
        [] for _ in range(iterations)
    ]
    preactivation_chunks: list[list[torch.Tensor]] = [
        [] for _ in range(iterations)
    ]
    output_chunks: list[list[torch.Tensor]] = [
        [] for _ in range(iterations)
    ]
    for start in range(0, inputs.shape[0], batch_size):
        logical = inputs[start : start + batch_size]
        physical_input = input_gain * torch.cat((logical, -logical), dim=1)
        if common_input_offset:
            physical_input = physical_input + (
                input_gain * float(common_input_offset)
            )
        hidden = torch.zeros(
            (logical.shape[0], numerator_w1.shape[1]),
            dtype=logical.dtype,
            device=logical.device,
        )
        output = torch.zeros(
            (logical.shape[0], numerator_w2.shape[1]),
            dtype=logical.dtype,
            device=logical.device,
        )
        for iteration in range(iterations):
            preactivation = (
                physical_input @ numerator_w1
                + output @ numerator_w2.transpose(0, 1)
                + bias
            ) / hidden_degree
            midpoint = preactivation.shape[1] // 2
            hidden = torch.cat(
                (
                    preactivation[:, :midpoint].clamp_min(0.0),
                    preactivation[:, midpoint:].clamp_max(0.0),
                ),
                dim=1,
            )
            output = hidden @ numerator_w2 / output_degree
            preactivation_chunks[iteration].append(preactivation)
            hidden_chunks[iteration].append(hidden)
            output_chunks[iteration].append(output)

    preactivations = [
        torch.cat(chunks, dim=0) for chunks in preactivation_chunks
    ]
    hidden_states = [
        torch.cat(chunks, dim=0) for chunks in hidden_chunks
    ]
    output_states = [
        torch.cat(chunks, dim=0) for chunks in output_chunks
    ]
    output = output_states[-1]
    paired = output.reshape(output.shape[0], output.shape[1] // 2, 2)
    return {
        "scores": paired[..., 0] - paired[..., 1],
        "preactivations": preactivations,
        "hidden_states": hidden_states,
        "output_states": output_states,
        "hidden_degree": hidden_degree,
        "output_degree": output_degree,
    }


def _centered_scores(scores: torch.Tensor) -> torch.Tensor:
    return scores - scores.mean(dim=1, keepdim=True)


def _normalized_margin(scores: torch.Tensor) -> torch.Tensor:
    centered = _centered_scores(scores)
    rms = centered.square().mean(dim=1, keepdim=True).sqrt()
    normalized = centered / rms.clamp_min(torch.finfo(scores.dtype).eps)
    top_two = torch.topk(normalized, 2, dim=1).values
    return top_two[:, 0] - top_two[:, 1]


def _absolute_margin(scores: torch.Tensor) -> torch.Tensor:
    top_two = torch.topk(scores, 2, dim=1).values
    return top_two[:, 0] - top_two[:, 1]


def _scores_from_output(output: torch.Tensor) -> torch.Tensor:
    paired = output.reshape(output.shape[0], output.shape[1] // 2, 2)
    return paired[..., 0] - paired[..., 1]


def _comparison(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    targets: torch.Tensor,
    include_states: bool,
) -> dict[str, Any]:
    reference_scores = reference["scores"]
    candidate_scores = candidate["scores"]
    centered_reference = _centered_scores(reference_scores)
    centered_candidate = _centered_scores(candidate_scores)
    cosine = torch.nn.functional.cosine_similarity(
        centered_reference,
        centered_candidate,
        dim=1,
        eps=torch.finfo(reference_scores.dtype).eps,
    )
    agreement = reference_scores.argmax(1) == candidate_scores.argmax(1)
    reference_margin = _normalized_margin(reference_scores)
    candidate_margin = _normalized_margin(candidate_scores)
    reference_absolute_margin = _absolute_margin(reference_scores)
    candidate_absolute_margin = _absolute_margin(candidate_scores)
    margin_quintile_edges = torch.quantile(
        reference_margin,
        torch.tensor(
            [0.2, 0.4, 0.6, 0.8],
            device=reference_margin.device,
            dtype=reference_margin.dtype,
        ),
    )
    margin_quintile = torch.bucketize(
        reference_margin,
        margin_quintile_edges,
    )
    result: dict[str, Any] = {
        "reference_accuracy": _accuracy(reference_scores, targets),
        "candidate_accuracy": _accuracy(candidate_scores, targets),
        "raw": predictive_divergence(
            reference_scores,
            candidate_scores,
            mode="raw",
        ),
        "per_example_rms": predictive_divergence(
            reference_scores,
            candidate_scores,
            mode="per_example_rms",
        ),
        "centered_score_cosine": _distribution(cosine),
        "reference_centered_score_rms": _rms(centered_reference),
        "candidate_centered_score_rms": _rms(centered_candidate),
        "reference_normalized_margin": _distribution(reference_margin),
        "candidate_normalized_margin": _distribution(candidate_margin),
        "reference_absolute_margin": _distribution(
            reference_absolute_margin
        ),
        "candidate_absolute_margin": _distribution(
            candidate_absolute_margin
        ),
        "disagreement_count": int((~agreement).sum()),
        "reference_margin_on_disagreements": (
            None
            if agreement.all()
            else _distribution(reference_margin[~agreement])
        ),
        "reference_absolute_margin_on_disagreements": (
            None
            if agreement.all()
            else _distribution(reference_absolute_margin[~agreement])
        ),
        "disagreements_by_reference_margin_quintile": [
            {
                "quintile": quintile + 1,
                "examples": int((margin_quintile == quintile).sum()),
                "disagreements": int(
                    (
                        (margin_quintile == quintile)
                        & (~agreement)
                    ).sum()
                ),
            }
            for quintile in range(5)
        ],
    }
    if include_states:
        gate_rows: list[dict[str, float | int]] = []
        for iteration, (reference_hidden, candidate_hidden) in enumerate(
            zip(
                reference["hidden_states"],
                candidate["hidden_states"],
            ),
            start=1,
        ):
            reference_active = reference_hidden != 0.0
            candidate_active = candidate_hidden != 0.0
            intersection = (reference_active & candidate_active).sum()
            union = (reference_active | candidate_active).sum()
            per_example = (
                reference_active == candidate_active
            ).to(torch.float32).mean(dim=1)
            gate_rows.append(
                {
                    "iteration": iteration,
                    "gate_agreement": float(per_example.mean()),
                    "examples_with_all_gates_equal_fraction": float(
                        (per_example == 1.0).to(torch.float32).mean()
                    ),
                    "reference_active_fraction": float(
                        reference_active.to(torch.float32).mean()
                    ),
                    "candidate_active_fraction": float(
                        candidate_active.to(torch.float32).mean()
                    ),
                    "active_set_jaccard": float(
                        intersection.to(torch.float64)
                        / union.clamp_min(1).to(torch.float64)
                    ),
                    "mean_gate_flips_per_example": float(
                        (reference_active != candidate_active)
                        .to(torch.float32)
                        .sum(dim=1)
                        .mean()
                    ),
                }
            )
        result["hidden_gate_agreement"] = gate_rows
    return result


def _degree_summary(values: torch.Tensor) -> dict[str, float]:
    values64 = values.to(torch.float64)
    mean = values64.mean()
    return {
        "minimum": float(values64.min()),
        "mean": float(mean),
        "maximum": float(values64.max()),
        "standard_deviation": float(values64.std(unbiased=False)),
        "coefficient_of_variation": float(
            values64.std(unbiased=False) / mean
        ),
    }


def _compact_comparison(comparison: dict[str, Any]) -> dict[str, float | int]:
    normalized = comparison["per_example_rms"]
    return {
        "accuracy": float(comparison["candidate_accuracy"]),
        "top1_agreement": float(normalized["top1_agreement"]),
        "symmetric_kl_nats": float(
            normalized["symmetric_kl_nats"]["mean"]
        ),
        "jensen_shannon_nats": float(
            normalized["jensen_shannon_nats"]["mean"]
        ),
        "centered_score_cosine_mean": float(
            comparison["centered_score_cosine"]["mean"]
        ),
        "centered_score_rms": float(
            comparison["candidate_centered_score_rms"]
        ),
        "disagreement_count": int(comparison["disagreement_count"]),
    }


def _component_rms(
    components: dict[str, torch.Tensor],
    *,
    total_name: str,
) -> dict[str, dict[str, float]]:
    total_rms = _rms(components[total_name])
    return {
        name: {
            "rms": _rms(values),
            "rms_over_total": (
                _rms(values) / total_rms if total_rms else math.inf
            ),
        }
        for name, values in components.items()
    }


def _plot_floor_sweep(
    path: Path,
    rows: list[dict[str, float | int]],
) -> None:
    ratios = [100.0 * float(row["floor_ratio"]) for row in rows]
    accuracies = [100.0 * float(row["accuracy"]) for row in rows]
    divergences = [float(row["symmetric_kl_nats"]) for row in rows]
    amplitudes = [
        float(row["centered_score_rms_ratio_to_clean"]) for row in rows
    ]
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.2), sharex=True)
    accuracy_axis = axes[0]
    divergence_axis = accuracy_axis.twinx()
    accuracy_axis.plot(
        ratios,
        accuracies,
        marker="o",
        color="#4C78A8",
        label="Accuracy",
    )
    divergence_axis.plot(
        ratios,
        divergences,
        marker="s",
        color="#E45756",
        label="Symmetric KL",
    )
    accuracy_axis.set_ylabel("Accuracy (%)", color="#4C78A8")
    divergence_axis.set_ylabel("Normalized KL (nats)", color="#E45756")
    accuracy_axis.grid(alpha=0.25)
    axes[1].plot(
        ratios,
        amplitudes,
        marker="o",
        color="#54A24B",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Minimum / maximum conductance (%)")
    axes[1].set_ylabel("Centered score RMS / clean")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_ablation(
    path: Path,
    scenarios: dict[str, dict[str, float | int]],
) -> None:
    names = list(scenarios)
    accuracies = [100.0 * float(scenarios[name]["accuracy"]) for name in names]
    fig, axis = plt.subplots(figsize=(9.0, 5.6))
    positions = list(range(len(names)))
    axis.barh(positions, accuracies, color="#4C78A8")
    axis.set_yticks(
        positions,
        labels=[name.replace("_", " ") for name in names],
    )
    axis.invert_yaxis()
    axis.set_xlabel("MNIST accuracy (%)")
    axis.set_xlim(max(0.0, min(accuracies) - 3.0), 100.0)
    axis.grid(axis="x", alpha=0.25)
    for position, value in zip(positions, accuracies):
        axis.text(value + 0.1, position, f"{value:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_readout_robustness(
    path: Path,
    *,
    noise_rows: list[dict[str, Any]],
    quantization_rows: list[dict[str, Any]],
) -> None:
    labels = {
        "clean": "Clean HWA",
        "affine_deterministic": "Affine deterministic",
        "affine_stored": "Affine stored",
    }
    colors = {
        "clean": "#4C78A8",
        "affine_deterministic": "#F58518",
        "affine_stored": "#54A24B",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    positive_noise = [
        row for row in noise_rows if float(row["node_noise_sigma"]) > 0.0
    ]
    for name, label in labels.items():
        axes[0].plot(
            [float(row["node_noise_sigma"]) for row in positive_noise],
            [
                100.0 * float(row["models"][name]["mean_accuracy"])
                for row in positive_noise
            ],
            marker="o",
            label=label,
            color=colors[name],
        )
        axes[1].plot(
            [
                float(row["node_quantization_step"])
                for row in quantization_rows
                if float(row["node_quantization_step"]) > 0.0
            ],
            [
                100.0 * float(row["models"][name]["accuracy"])
                for row in quantization_rows
                if float(row["node_quantization_step"]) > 0.0
            ],
            marker="o",
            label=label,
            color=colors[name],
        )
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlabel("Output-node Gaussian noise sigma")
    axes[1].set_xlabel("Output-node quantization step")
    axes[0].set_ylabel("MNIST accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = _parser().parse_args()
    if args.g_min_us <= 0.0 or args.g_max_us <= args.g_min_us:
        raise ValueError(
            "Expected 0 < --g-min-us < --g-max-us. Provided value: "
            f"g_min_us={args.g_min_us!r}, g_max_us={args.g_max_us!r}."
        )
    if args.replay_batch_size <= 0:
        raise ValueError(
            "Expected --replay-batch-size to be positive. Provided value: "
            f"{args.replay_batch_size!r}."
        )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    clean = _load_weights(args.clean_weights, device=device)
    stored = _load_weights(args.affine_weights, device=device)
    inputs, targets, input_gain, iterations = _load_test_set(
        args.config,
        device=device,
    )

    w1 = clean[W1_KEY]
    w2 = clean[W2_KEY]
    bias = clean[BIAS_KEY]
    floor = args.g_min_us / args.g_max_us
    span = 1.0 - floor
    affine_w1 = floor + span * w1
    affine_w2 = floor + span * w2

    clean_trace = trace_scores(
        inputs,
        numerator_w1=w1,
        numerator_w2=w2,
        bias=bias,
        input_gain=input_gain,
        iterations=iterations,
        batch_size=args.replay_batch_size,
    )
    affine_trace = trace_scores(
        inputs,
        numerator_w1=affine_w1,
        numerator_w2=affine_w2,
        bias=bias,
        input_gain=input_gain,
        iterations=iterations,
        batch_size=args.replay_batch_size,
    )
    stored_trace = trace_scores(
        inputs,
        numerator_w1=stored[W1_KEY],
        numerator_w2=stored[W2_KEY],
        bias=stored[BIAS_KEY],
        input_gain=input_gain,
        iterations=iterations,
        batch_size=args.replay_batch_size,
    )
    deterministic_comparison = _comparison(
        clean_trace,
        affine_trace,
        targets=targets,
        include_states=True,
    )
    stored_comparison = _comparison(
        clean_trace,
        stored_trace,
        targets=targets,
        include_states=True,
    )
    noise_comparison = _comparison(
        affine_trace,
        stored_trace,
        targets=targets,
        include_states=True,
    )

    ablation_definitions = {
        "clean": (w1, w2, None, None, bias),
        "affine_full": (
            affine_w1,
            affine_w2,
            None,
            None,
            bias,
        ),
        "affine_W1_only": (affine_w1, w2, None, None, bias),
        "affine_W2_only": (w1, affine_w2, None, None, bias),
        "remove_floor_from_numerators": (
            span * w1,
            span * w2,
            affine_w1,
            affine_w2,
            bias,
        ),
        "remove_floor_from_denominators": (
            affine_w1,
            affine_w2,
            span * w1,
            span * w2,
            bias,
        ),
        "full_signal_recovery": (
            span * w1,
            span * w2,
            span * w1,
            span * w2,
            span * bias,
        ),
    }
    ablations: dict[str, dict[str, float | int]] = {}
    for name, (
        numerator_w1,
        numerator_w2,
        denominator_w1,
        denominator_w2,
        scenario_bias,
    ) in ablation_definitions.items():
        scenario = trace_scores(
            inputs,
            numerator_w1=numerator_w1,
            numerator_w2=numerator_w2,
            denominator_w1=denominator_w1,
            denominator_w2=denominator_w2,
            bias=scenario_bias,
            input_gain=input_gain,
            iterations=iterations,
            batch_size=args.replay_batch_size,
        )
        ablations[name] = _compact_comparison(
            _comparison(
                clean_trace,
                scenario,
                targets=targets,
                include_states=False,
            )
        )

    floor_ratios = (
        0.0,
        0.01,
        0.03,
        0.05,
        0.1,
        floor,
        0.2,
        0.3,
        0.5,
        0.7,
        0.8,
        0.9,
        0.95,
        0.98,
        0.99,
        0.995,
        0.999,
    )
    floor_sweep: list[dict[str, float | int]] = []
    clean_score_rms = _rms(_centered_scores(clean_trace["scores"]))
    for ratio in sorted(set(floor_ratios)):
        ratio_span = 1.0 - ratio
        scenario = trace_scores(
            inputs,
            numerator_w1=ratio + ratio_span * w1,
            numerator_w2=ratio + ratio_span * w2,
            bias=bias,
            input_gain=input_gain,
            iterations=iterations,
            batch_size=args.replay_batch_size,
        )
        compact = _compact_comparison(
            _comparison(
                clean_trace,
                scenario,
                targets=targets,
                include_states=False,
            )
        )
        floor_sweep.append(
            {
                "floor_ratio": ratio,
                **compact,
                "centered_score_rms_ratio_to_clean": (
                    float(compact["centered_score_rms"]) / clean_score_rms
                ),
            }
        )

    common_offset_sweep: list[dict[str, float | int]] = []
    for offset in (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
        clean_offset = trace_scores(
            inputs,
            numerator_w1=w1,
            numerator_w2=w2,
            bias=bias,
            input_gain=input_gain,
            iterations=iterations,
            batch_size=args.replay_batch_size,
            common_input_offset=offset,
        )
        affine_offset = trace_scores(
            inputs,
            numerator_w1=affine_w1,
            numerator_w2=affine_w2,
            bias=bias,
            input_gain=input_gain,
            iterations=iterations,
            batch_size=args.replay_batch_size,
            common_input_offset=offset,
        )
        common_offset_sweep.append(
            {
                "common_logical_offset_per_input": offset,
                **_compact_comparison(
                    _comparison(
                        clean_offset,
                        affine_offset,
                        targets=targets,
                        include_states=False,
                    )
                ),
                "clean_accuracy": _accuracy(
                    clean_offset["scores"],
                    targets,
                ),
            }
        )

    nonuniform_floor_sweep: list[dict[str, Any]] = []
    for relative_std in (0.0, 0.01, 0.03, 0.1, 0.3):
        seed_rows: list[dict[str, float | int]] = []
        for seed in (17, 29, 41):
            generator = torch.Generator(device=device).manual_seed(seed)
            floor_w1 = (
                floor
                * (
                    1.0
                    + relative_std
                    * torch.randn(
                        w1.shape,
                        generator=generator,
                        device=device,
                        dtype=w1.dtype,
                    )
                )
            ).clamp(min=0.0, max=0.99)
            floor_w2 = (
                floor
                * (
                    1.0
                    + relative_std
                    * torch.randn(
                        w2.shape,
                        generator=generator,
                        device=device,
                        dtype=w2.dtype,
                    )
                )
            ).clamp(min=0.0, max=0.99)
            scenario = trace_scores(
                inputs,
                numerator_w1=floor_w1 + (1.0 - floor_w1) * w1,
                numerator_w2=floor_w2 + (1.0 - floor_w2) * w2,
                bias=bias,
                input_gain=input_gain,
                iterations=iterations,
                batch_size=args.replay_batch_size,
            )
            seed_rows.append(
                {
                    "seed": seed,
                    **_compact_comparison(
                        _comparison(
                            clean_trace,
                            scenario,
                            targets=targets,
                            include_states=False,
                        )
                    ),
                }
            )
        nonuniform_floor_sweep.append(
            {
                "relative_floor_standard_deviation": relative_std,
                "runs": seed_rows,
                "mean_accuracy": sum(
                    float(row["accuracy"]) for row in seed_rows
                )
                / len(seed_rows),
                "mean_symmetric_kl_nats": sum(
                    float(row["symmetric_kl_nats"]) for row in seed_rows
                )
                / len(seed_rows),
                "mean_top1_agreement": sum(
                    float(row["top1_agreement"]) for row in seed_rows
                )
                / len(seed_rows),
            }
        )

    traces = {
        "clean": clean_trace,
        "affine_deterministic": affine_trace,
        "affine_stored": stored_trace,
    }
    iteration_stability: dict[str, list[dict[str, float | int]]] = {}
    for name, trace in traces.items():
        final_prediction = trace["scores"].argmax(1)
        iteration_stability[name] = []
        for iteration, output_state in enumerate(
            trace["output_states"],
            start=1,
        ):
            iteration_scores = _scores_from_output(output_state)
            clean_iteration_scores = _scores_from_output(
                clean_trace["output_states"][iteration - 1]
            )
            previous_output = (
                torch.zeros_like(output_state)
                if iteration == 1
                else trace["output_states"][iteration - 2]
            )
            iteration_stability[name].append(
                {
                    "iteration": iteration,
                    "accuracy": _accuracy(iteration_scores, targets),
                    "agreement_with_final_iteration": float(
                        (
                            iteration_scores.argmax(1)
                            == final_prediction
                        )
                        .to(torch.float32)
                        .mean()
                    ),
                    "centered_score_rms": _rms(
                        _centered_scores(iteration_scores)
                    ),
                    "agreement_with_clean_same_iteration": float(
                        (
                            iteration_scores.argmax(1)
                            == clean_iteration_scores.argmax(1)
                        )
                        .to(torch.float32)
                        .mean()
                    ),
                    "output_state_delta_rms": _rms(
                        output_state - previous_output
                    ),
                }
            )

    readout_noise_rows: list[dict[str, Any]] = []
    for sigma in (
        0.0,
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
    ):
        model_accuracies: dict[str, list[float]] = {
            name: [] for name in traces
        }
        model_agreements: dict[str, list[float]] = {
            name: [] for name in traces
        }
        seeds = (0,) if sigma == 0.0 else tuple(range(32))
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(
                2000 + seed
            )
            noise = sigma * torch.randn(
                clean_trace["output_states"][-1].shape,
                generator=generator,
                device=device,
                dtype=inputs.dtype,
            )
            for name, trace in traces.items():
                noisy_scores = _scores_from_output(
                    trace["output_states"][-1] + noise
                )
                model_accuracies[name].append(
                    _accuracy(noisy_scores, targets)
                )
                model_agreements[name].append(
                    float(
                        (
                            noisy_scores.argmax(1)
                            == trace["scores"].argmax(1)
                        )
                        .to(torch.float32)
                        .mean()
                    )
                )
        readout_noise_rows.append(
            {
                "node_noise_sigma": sigma,
                "draws": len(seeds),
                "seed_formula": "2000 + draw_index",
                "common_random_numbers_across_models": True,
                "models": {
                    name: {
                        "mean_accuracy": sum(values) / len(values),
                        "standard_deviation": float(
                            torch.tensor(values, dtype=torch.float64).std(
                                unbiased=False
                            )
                        ),
                        "minimum_accuracy": min(values),
                        "maximum_accuracy": max(values),
                        "mean_agreement_with_noiseless_top1": (
                            sum(model_agreements[name])
                            / len(model_agreements[name])
                        ),
                    }
                    for name, values in model_accuracies.items()
                },
            }
        )

    readout_quantization_rows: list[dict[str, Any]] = []
    for step in (0.0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        model_rows: dict[str, dict[str, float]] = {}
        for name, trace in traces.items():
            output_state = trace["output_states"][-1]
            quantized = (
                output_state
                if step == 0.0
                else torch.round(output_state / step) * step
            )
            model_rows[name] = {
                "accuracy": _accuracy(
                    _scores_from_output(quantized),
                    targets,
                ),
                "agreement_with_unquantized_top1": float(
                    (
                        _scores_from_output(quantized).argmax(1)
                        == trace["scores"].argmax(1)
                    )
                    .to(torch.float32)
                    .mean()
                ),
                "changed_node_fraction": (
                    0.0
                    if step == 0.0
                    else float(
                        (quantized != output_state)
                        .to(torch.float32)
                        .mean()
                    )
                ),
                "global_output_level_count": int(
                    torch.unique(quantized).numel()
                ),
            }
        readout_quantization_rows.append(
            {
                "node_quantization_step": step,
                "rounding": "nearest_multiple_ties_to_even",
                "models": model_rows,
            }
        )

    physical_input = input_gain * torch.cat((inputs, -inputs), dim=1)
    affine_hidden_degree = affine_w1.sum(0) + affine_w2.sum(1)
    affine_output_degree = affine_w2.sum(0)
    previous_output = (
        torch.zeros_like(affine_trace["output_states"][0])
        if iterations == 1
        else affine_trace["output_states"][-2]
    )
    final_hidden = affine_trace["hidden_states"][-1]
    hidden_components = {
        "learned_input": (
            span * (physical_input @ w1) / affine_hidden_degree
        ),
        "learned_recurrent": (
            span
            * (previous_output @ w2.transpose(0, 1))
            / affine_hidden_degree
        ),
        "floor_recurrent": (
            floor
            * previous_output.sum(dim=1, keepdim=True)
            / affine_hidden_degree
        ),
        "bias": bias / affine_hidden_degree,
        "total_preactivation": affine_trace["preactivations"][-1],
    }
    learned_output = (
        span * (final_hidden @ w2) / affine_output_degree
    )
    floor_output = (
        floor
        * final_hidden.sum(dim=1, keepdim=True)
        / affine_output_degree
    )
    learned_output_pairs = learned_output.reshape(
        learned_output.shape[0],
        learned_output.shape[1] // 2,
        2,
    )
    floor_output_pairs = floor_output.reshape(
        floor_output.shape[0],
        floor_output.shape[1] // 2,
        2,
    )
    output_score_components = {
        "learned_score": (
            learned_output_pairs[..., 0] - learned_output_pairs[..., 1]
        ),
        "floor_common_mode_leakage": (
            floor_output_pairs[..., 0] - floor_output_pairs[..., 1]
        ),
        "total_score": affine_trace["scores"],
    }

    clean_hidden_degree = w1.sum(0) + w2.sum(1)
    clean_output_degree = w2.sum(0)
    clean_output_pairs = clean_output_degree.reshape(-1, 2)
    affine_output_pairs = affine_output_degree.reshape(-1, 2)
    degree_statistics = {
        "clean_hidden": _degree_summary(clean_hidden_degree),
        "affine_hidden": _degree_summary(affine_hidden_degree),
        "clean_output": _degree_summary(clean_output_degree),
        "affine_output": _degree_summary(affine_output_degree),
        "clean_paired_output_relative_mismatch": _distribution(
            (clean_output_pairs[:, 0] - clean_output_pairs[:, 1]).abs()
            / clean_output_pairs.mean(dim=1)
        ),
        "affine_paired_output_relative_mismatch": _distribution(
            (affine_output_pairs[:, 0] - affine_output_pairs[:, 1]).abs()
            / affine_output_pairs.mean(dim=1)
        ),
    }
    input_sums = physical_input.sum(dim=1)
    input_midpoint = physical_input.shape[1] // 2
    paired_input_residual = (
        physical_input[:, :input_midpoint]
        + physical_input[:, input_midpoint:]
    )
    exact_cancellation = {
        "paired_input_residual_abs_max": float(
            paired_input_residual.abs().max()
        ),
        "balanced_input_sum_abs_max": float(input_sums.abs().max()),
        "uniform_floor_input_current_abs_max": float(
            (floor * input_sums).abs().max()
        ),
        "per_crossbar_uniform_offset_rank": 1,
    }

    weight_statistics = {
        key: {
            "minimum": float(clean[key].min()),
            "maximum": float(clean[key].max()),
            "fraction_above_deployment_ceiling_1": float(
                (clean[key] > 1.0).to(torch.float32).mean()
            ),
        }
        for key in (W1_KEY, W2_KEY)
    }

    source_paths = {
        "analysis_script": Path(__file__).resolve(),
        "config": args.config.expanduser().resolve(),
        "clean_weights": args.clean_weights.expanduser().resolve(),
        "affine_weights": args.affine_weights.expanduser().resolve(),
    }
    summary = {
        "schema": "mnist-affine-floor-mechanism-analysis",
        "schema_version": 1,
        "device": device.type,
        "torch_version": str(torch.__version__),
        "examples": int(targets.numel()),
        "iterations": iterations,
        "input_gain": input_gain,
        "g_min_us": args.g_min_us,
        "g_max_us": args.g_max_us,
        "floor_ratio": floor,
        "affine_span": span,
        "weight_statistics": weight_statistics,
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        },
        "exact_common_mode_cancellation": exact_cancellation,
        "degree_statistics": degree_statistics,
        "comparisons": {
            "clean_vs_affine_deterministic": deterministic_comparison,
            "clean_vs_affine_stored": stored_comparison,
            "affine_deterministic_vs_stored_noise": noise_comparison,
        },
        "mechanism_ablations": ablations,
        "final_iteration_component_rms": {
            "hidden_update": _component_rms(
                hidden_components,
                total_name="total_preactivation",
            ),
            "differential_output_score": _component_rms(
                output_score_components,
                total_name="total_score",
            ),
        },
        "floor_ratio_sweep": floor_sweep,
        "common_input_offset_sweep": common_offset_sweep,
        "nonuniform_floor_sweep": nonuniform_floor_sweep,
        "iteration_stability": iteration_stability,
        "synthetic_readout_robustness": {
            "gaussian_output_node_noise": readout_noise_rows,
            "output_node_quantization": readout_quantization_rows,
            "caveat": (
                "These absolute output-voltage perturbations are mechanism "
                "controls, not calibrated CMO or ADC models."
            ),
        },
        "interpretation": {
            "scope": (
                "One HWA checkpoint and the canonical 10,000-example MNIST "
                "test split."
            ),
            "kl": (
                "Primary functional comparisons use per-example centered, "
                "RMS-normalized score softmax and natural-log nats."
            ),
            "expressivity": (
                "Low empirical divergence for one target function does not "
                "establish equality of bounded and loose-range hypothesis "
                "classes."
            ),
            "extreme_floor_numerics": (
                "At floor ratios at or above about 0.98, score amplitudes "
                "approach float32 cancellation and the normalized KL/cosine "
                "metrics are not physically meaningful. The surviving "
                "top-1 ordering instead demonstrates the fragility of ideal "
                "floating-point argmax on vanishing margins."
            ),
            "synthetic_ablations": (
                "The common-input-offset, nonuniform-floor, output-noise, "
                "and quantization sweeps are mechanism controls and are not "
                "calibrated device or converter models."
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_floor_sweep(output_dir / "floor_ratio_sweep.png", floor_sweep)
    _plot_ablation(output_dir / "mechanism_ablations.png", ablations)
    _plot_readout_robustness(
        output_dir / "synthetic_readout_robustness.png",
        noise_rows=readout_noise_rows,
        quantization_rows=readout_quantization_rows,
    )
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
