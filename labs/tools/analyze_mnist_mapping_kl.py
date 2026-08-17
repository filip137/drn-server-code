#!/usr/bin/env python3
"""Compare literal and affine CMO mappings with KL-style divergences.

The DRN readout produces paired voltage differences rather than calibrated
probabilities. Predictive KL is therefore reported both on the raw-score
softmax and after a deterministic per-example RMS normalization that removes
the large mapping-dependent voltage scale. Conductance KL is a smoothed
histogram estimate and is reported at several resolutions because it is not a
coordinate-free quantity.
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure predictive and conductance divergences between literal "
            "and affine CMO mappings on the matched MNIST DRN."
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
    parser.add_argument("--literal-weights", type=Path, required=True)
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


@torch.no_grad()
def paired_scores(
    inputs: torch.Tensor,
    *,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias: torch.Tensor,
    input_gain: float,
    iterations: int,
    batch_size: int,
) -> torch.Tensor:
    """Return the paired class-voltage differences used for prediction."""

    hidden_degree = w1.sum(0) + w2.sum(1)
    output_degree = w2.sum(0)
    if (hidden_degree <= 0.0).any() or (output_degree <= 0.0).any():
        raise ValueError(
            "Expected every replay conductance sum to be positive. Provided "
            "value contains a disconnected hidden or output node."
        )
    if w2.shape[1] % 2:
        raise ValueError(
            "Expected an even output width for paired differential scores. "
            f"Provided value: {w2.shape[1]!r}."
        )

    chunks: list[torch.Tensor] = []
    for start in range(0, inputs.shape[0], batch_size):
        logical = inputs[start : start + batch_size]
        physical_input = input_gain * torch.cat((logical, -logical), dim=1)
        hidden = torch.zeros(
            (logical.shape[0], w1.shape[1]),
            dtype=logical.dtype,
            device=logical.device,
        )
        output = torch.zeros(
            (logical.shape[0], w2.shape[1]),
            dtype=logical.dtype,
            device=logical.device,
        )
        for _ in range(iterations):
            hidden = (
                physical_input @ w1
                + output @ w2.transpose(0, 1)
                + bias
            ) / hidden_degree
            midpoint = hidden.shape[1] // 2
            hidden = torch.cat(
                (
                    hidden[:, :midpoint].clamp_min(0.0),
                    hidden[:, midpoint:].clamp_max(0.0),
                ),
                dim=1,
            )
            output = hidden @ w2 / output_degree
        paired = output.reshape(
            output.shape[0],
            output.shape[1] // 2,
            2,
        )
        chunks.append(paired[..., 0] - paired[..., 1])
    return torch.cat(chunks, dim=0)


def _distribution_summary(values: torch.Tensor) -> dict[str, float]:
    quantiles = torch.quantile(
        values,
        torch.tensor(
            [0.5, 0.9, 0.99],
            dtype=values.dtype,
            device=values.device,
        ),
    )
    return {
        "mean": float(values.mean()),
        "median": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "maximum": float(values.max()),
    }


def _probabilities(
    scores: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if mode == "raw":
        logits = scores
    elif mode == "per_example_rms":
        centered = scores - scores.mean(dim=1, keepdim=True)
        rms = torch.sqrt(centered.square().mean(dim=1, keepdim=True))
        logits = centered / rms.clamp_min(torch.finfo(scores.dtype).eps)
    else:
        raise ValueError(
            "Expected probability mode to be 'raw' or 'per_example_rms'. "
            f"Provided value: {mode!r}."
        )
    return torch.softmax(logits.to(torch.float64), dim=1)


def predictive_divergence(
    first_scores: torch.Tensor,
    second_scores: torch.Tensor,
    *,
    mode: str,
) -> dict[str, Any]:
    """Return directional KL and symmetric JS over matched examples."""

    first = _probabilities(first_scores, mode=mode)
    second = _probabilities(second_scores, mode=mode)
    midpoint = 0.5 * (first + second)
    first_to_second = (
        first * (torch.log(first) - torch.log(second))
    ).sum(dim=1)
    second_to_first = (
        second * (torch.log(second) - torch.log(first))
    ).sum(dim=1)
    js = 0.5 * (
        (first * (torch.log(first) - torch.log(midpoint))).sum(dim=1)
        + (second * (torch.log(second) - torch.log(midpoint))).sum(dim=1)
    )
    return {
        "probability_mode": mode,
        "first_to_second_kl_nats": _distribution_summary(first_to_second),
        "second_to_first_kl_nats": _distribution_summary(second_to_first),
        "symmetric_kl_nats": _distribution_summary(
            0.5 * (first_to_second + second_to_first)
        ),
        "jensen_shannon_nats": _distribution_summary(js),
        "top1_agreement": float(
            (first_scores.argmax(1) == second_scores.argmax(1))
            .to(torch.float32)
            .mean()
        ),
    }


def histogram_divergence(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    bins: int,
    minimum: float,
    maximum: float,
    pseudocount: float = 0.5,
) -> dict[str, float | int]:
    """Return smoothed histogram KL and JS in nats."""

    first_count = torch.histc(
        first.to(torch.float64),
        bins=bins,
        min=minimum,
        max=maximum,
    )
    second_count = torch.histc(
        second.to(torch.float64),
        bins=bins,
        min=minimum,
        max=maximum,
    )
    first_prob = (first_count + pseudocount) / (
        first_count.sum() + pseudocount * bins
    )
    second_prob = (second_count + pseudocount) / (
        second_count.sum() + pseudocount * bins
    )
    midpoint = 0.5 * (first_prob + second_prob)
    first_to_second = (
        first_prob * (torch.log(first_prob) - torch.log(second_prob))
    ).sum()
    second_to_first = (
        second_prob * (torch.log(second_prob) - torch.log(first_prob))
    ).sum()
    js = 0.5 * (
        (
            first_prob
            * (torch.log(first_prob) - torch.log(midpoint))
        ).sum()
        + (
            second_prob
            * (torch.log(second_prob) - torch.log(midpoint))
        ).sum()
    )
    return {
        "bins": bins,
        "pseudocount_per_bin": pseudocount,
        "first_to_second_kl_nats": float(first_to_second),
        "second_to_first_kl_nats": float(second_to_first),
        "symmetric_kl_nats": float(
            0.5 * (first_to_second + second_to_first)
        ),
        "jensen_shannon_nats": float(js),
    }


def _accuracy(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    return float((scores.argmax(1) == targets).to(torch.float32).mean())


def _plot_predictive_kl(
    path: Path,
    comparisons: dict[str, dict[str, Any]],
) -> None:
    names: list[str] = []
    raw: list[float] = []
    normalized: list[float] = []
    for name, modes in comparisons.items():
        names.append(name.replace("_", " "))
        raw.append(
            float(modes["raw"]["symmetric_kl_nats"]["mean"])
        )
        normalized.append(
            float(
                modes["per_example_rms"]["symmetric_kl_nats"]["mean"]
            )
        )
    positions = torch.arange(len(names), dtype=torch.float64).numpy()
    width = 0.36
    fig, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.bar(
        positions - width / 2.0,
        raw,
        width,
        label="Raw-score softmax",
        color="#4C78A8",
    )
    axis.bar(
        positions + width / 2.0,
        normalized,
        width,
        label="Per-example RMS-normalized",
        color="#F58518",
    )
    axis.set_xticks(positions, labels=names, rotation=18, ha="right")
    axis.set_ylabel("Mean symmetric KL (nats)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_conductance_histograms(
    path: Path,
    *,
    literal: dict[str, torch.Tensor],
    affine: dict[str, torch.Tensor],
    minimum: float,
    maximum: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharey=True)
    for axis, key, title in zip(
        axes,
        (W1_KEY, W2_KEY),
        ("W1", "W2"),
    ):
        axis.hist(
            literal[key].detach().cpu().numpy().ravel(),
            bins=128,
            range=(minimum, maximum),
            density=True,
            histtype="step",
            linewidth=1.7,
            label="Literal",
            color="#E45756",
        )
        axis.hist(
            affine[key].detach().cpu().numpy().ravel(),
            bins=128,
            range=(minimum, maximum),
            density=True,
            histtype="step",
            linewidth=1.7,
            label="Affine",
            color="#4C78A8",
        )
        axis.set_title(title)
        axis.set_xlabel("Effective DRN conductance")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Histogram density (log scale)")
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
    literal = _load_weights(args.literal_weights, device=device)
    affine = _load_weights(args.affine_weights, device=device)
    inputs, targets, input_gain, iterations = _load_test_set(
        args.config,
        device=device,
    )
    floor = args.g_min_us / args.g_max_us
    span = 1.0 - floor
    deterministic_literal = {
        W1_KEY: clean[W1_KEY].clamp_min(floor),
        W2_KEY: clean[W2_KEY].clamp_min(floor),
        BIAS_KEY: clean[BIAS_KEY],
    }
    deterministic_affine = {
        W1_KEY: floor + span * clean[W1_KEY],
        W2_KEY: floor + span * clean[W2_KEY],
        BIAS_KEY: clean[BIAS_KEY],
    }
    models = {
        "clean": clean,
        "literal_stored": literal,
        "affine_stored": affine,
        "literal_deterministic": deterministic_literal,
        "affine_deterministic": deterministic_affine,
    }
    scores = {
        name: paired_scores(
            inputs,
            w1=weights[W1_KEY],
            w2=weights[W2_KEY],
            bias=weights[BIAS_KEY],
            input_gain=input_gain,
            iterations=iterations,
            batch_size=args.replay_batch_size,
        )
        for name, weights in models.items()
    }
    accuracies = {
        name: _accuracy(value, targets) for name, value in scores.items()
    }

    comparison_pairs = {
        "literal_stored_vs_affine_stored": (
            "literal_stored",
            "affine_stored",
        ),
        "literal_deterministic_vs_affine_deterministic": (
            "literal_deterministic",
            "affine_deterministic",
        ),
        "clean_vs_literal_stored": ("clean", "literal_stored"),
        "clean_vs_affine_stored": ("clean", "affine_stored"),
        "clean_vs_literal_deterministic": (
            "clean",
            "literal_deterministic",
        ),
        "clean_vs_affine_deterministic": (
            "clean",
            "affine_deterministic",
        ),
    }
    predictive: dict[str, dict[str, Any]] = {}
    for name, (first, second) in comparison_pairs.items():
        predictive[name] = {
            mode: predictive_divergence(
                scores[first],
                scores[second],
                mode=mode,
            )
            for mode in ("raw", "per_example_rms")
        }

    conductance_pairs = {
        "literal_stored_vs_affine_stored": (literal, affine),
        "literal_deterministic_vs_affine_deterministic": (
            deterministic_literal,
            deterministic_affine,
        ),
    }
    conductance: dict[str, Any] = {}
    for name, (first, second) in conductance_pairs.items():
        conductance[name] = {}
        for layer, first_values, second_values in (
            (W1_KEY, first[W1_KEY], second[W1_KEY]),
            (W2_KEY, first[W2_KEY], second[W2_KEY]),
            (
                "combined_dense",
                torch.cat((first[W1_KEY].flatten(), first[W2_KEY].flatten())),
                torch.cat(
                    (second[W1_KEY].flatten(), second[W2_KEY].flatten())
                ),
            ),
        ):
            conductance[name][layer] = {
                "matched_element_mean_abs_difference": float(
                    (first_values - second_values).abs().mean()
                ),
                "matched_element_rmse": float(
                    torch.sqrt((first_values - second_values).square().mean())
                ),
                "histogram_divergence": [
                    histogram_divergence(
                        first_values,
                        second_values,
                        bins=bins,
                        minimum=0.0,
                        maximum=1.0,
                    )
                    for bins in (64, 128, 256)
                ],
            }

    source_paths = {
        "analysis_script": Path(__file__).resolve(),
        "config": args.config.expanduser().resolve(),
        "clean_weights": args.clean_weights.expanduser().resolve(),
        "literal_weights": args.literal_weights.expanduser().resolve(),
        "affine_weights": args.affine_weights.expanduser().resolve(),
    }
    summary = {
        "schema": "mnist-mapping-kl-analysis",
        "schema_version": 1,
        "device": device.type,
        "torch_version": str(torch.__version__),
        "g_min_us": args.g_min_us,
        "g_max_us": args.g_max_us,
        "floor_drn_units": floor,
        "input_gain": input_gain,
        "iterations": iterations,
        "examples": int(targets.numel()),
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in source_paths.items()
        },
        "accuracy": accuracies,
        "predictive_divergence": predictive,
        "conductance_divergence": conductance,
        "interpretation": {
            "predictive_kl_units": "natural-log nats",
            "raw_probability_caveat": (
                "DRN paired voltage differences are not calibrated logits; "
                "raw-softmax KL is strongly affected by voltage attenuation."
            ),
            "normalized_probability_caveat": (
                "Per-example RMS normalization removes score amplitude but "
                "introduces an explicit, deterministic temperature choice."
            ),
            "conductance_kl_caveat": (
                "Conductance KL is a Jeffreys-smoothed histogram estimate and "
                "depends on bin count; JS and multiple resolutions are "
                "reported for sensitivity."
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_predictive_kl(
        output_dir / "predictive_kl.png",
        predictive,
    )
    _plot_conductance_histograms(
        output_dir / "conductance_histograms.png",
        literal=literal,
        affine=affine,
        minimum=0.0,
        maximum=1.0,
    )
    print(output_dir / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
