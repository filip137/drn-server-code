#!/usr/bin/env python3
"""Validate and summarize the Conv2 centered-EqProp read-noise exploration."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import atomic_write_json, validate_run


STUDY_ID = (
    "perfectdiode-conv2-centered-float64-eqprop-read-noise-training-"
    "10ep-seed0-20260812-v1"
)
SCHEMES = ("baseline", "ours", "legacy")
SIGMAS = (0.0, 1.0e-6, 1.0e-5, 1.0e-4)
EXPECTED_INJECTED_BETA = {"baseline": 1000.0, "ours": 100.0, "legacy": 0.3}
EXPECTED_BASE_BETA = {
    "baseline": 1000.0,
    "ours": 6.25,
    "legacy": 0.001171875,
}
EXPECTED_ENDPOINT_NOISE_DRAWS = 10 * math.ceil(55_000 / 16) * 2 * 3


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _float_equal(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-15)


def _case_contract(manifest: Mapping[str, Any]) -> tuple[str, float, Mapping[str, Any]]:
    config = manifest.get("configuration", {}).get("resolved")
    if not isinstance(config, Mapping):
        raise ValueError("Manifest is missing its resolved scientific config.")
    arm_id = str(config.get("arm_id", ""))
    matches = [scheme for scheme in SCHEMES if arm_id.startswith(f"conv2_{scheme}_")]
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve amplification scheme from arm_id={arm_id!r}.")
    eqprop = config.get("eqprop")
    if not isinstance(eqprop, Mapping):
        raise ValueError(f"Missing EqProp contract for {arm_id}.")
    return matches[0], float(eqprop["endpoint_read_noise_std"]), config


def _load_array(run_dir: Path, name: str, epochs: int) -> np.ndarray:
    path = run_dir / name
    if not path.is_file():
        raise ValueError(f"Missing history array {path}.")
    values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if values.size != epochs or not np.all(np.isfinite(values)):
        raise ValueError(
            f"Expected {epochs} finite values in {path}; got shape={values.shape}."
        )
    return values


def load_and_validate(production_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    production_root = production_root.expanduser().resolve()
    run_dirs = sorted(
        path
        for path in production_root.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    if len(run_dirs) != 12:
        raise ValueError(f"Expected twelve canonical runs in {production_root}; found {len(run_dirs)}.")

    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    identities: set[tuple[str, float]] = set()
    initial_hashes: set[str] = set()
    train_index_hashes: set[str] = set()
    validation_index_hashes: set[str] = set()
    batch_order_hashes: set[tuple[str, ...]] = set()
    config_indices: set[int] = set()

    for run_dir in run_dirs:
        bundle_errors = validate_run(run_dir)
        errors.extend(f"{run_dir.name}: {error}" for error in bundle_errors)
        manifest = _read_json(run_dir / "manifest.json")
        status = _read_json(run_dir / "status.json")
        metrics = _read_json(run_dir / "metrics.json")
        exact = _read_json(run_dir / "exact_run.json")
        scheme, sigma, config = _case_contract(manifest)
        identity = (scheme, sigma)
        if identity in identities:
            errors.append(f"duplicate case {identity}")
        identities.add(identity)
        index = int(exact.get("index", -1))
        config_indices.add(index)

        eqprop = config.get("eqprop", {})
        metric_eqprop = metrics.get("eqprop", {})
        epochs = int(manifest.get("configuration", {}).get("epochs", -1))
        dataset = manifest.get("dataset", {})
        provenance = metrics.get("dataset_provenance", {})
        expected_draws = 0 if sigma == 0.0 else EXPECTED_ENDPOINT_NOISE_DRAWS
        checks = {
            "study_id": manifest.get("study_id") == STUDY_ID,
            "complete": status.get("state") == "complete",
            "not_smoke": manifest.get("smoke") is False,
            "evidence_class": manifest.get("evidence_class") == "ordinary_mnist_exploratory",
            "epochs": epochs == 10,
            "runtime_dtype": metrics.get("runtime_dtype") == "float64",
            "algorithm": metrics.get("training_algorithm") == "EP",
            "variant": metric_eqprop.get("variant") == "centered",
            "nudging_mode": metric_eqprop.get("nudging_mode") == "current",
            "normalized_current": metric_eqprop.get("normalize_current_scale") is True,
            "input_noise_disabled": metric_eqprop.get("input_read_noise") is False,
            "noise_sigma": _float_equal(metric_eqprop.get("endpoint_read_noise_std"), sigma),
            "noise_seed": int(metric_eqprop.get("endpoint_read_noise_seed", -1)) == 2026081201,
            "injected_beta": _float_equal(
                eqprop.get("injected_beta_B"), EXPECTED_INJECTED_BETA[scheme]
            ),
            "base_beta": _float_equal(config.get("beta"), EXPECTED_BASE_BETA[scheme]),
            "T": int(config.get("model_base", {}).get("num_iterations_inference", -1)) == 8,
            "K": int(config.get("model_base", {}).get("num_iterations_training", -1)) == 8,
            "official_test_manifest": dataset.get("official_test_read") is False,
            "official_test_examples": int(metrics.get("official_test_examples", -1)) == 0,
            "official_test_evaluations": int(metrics.get("official_test_evaluations", -1)) == 0,
            "optimizer_steps": metrics.get("optimizer_steps_applied") is True,
            "noise_draws": int(metrics.get("eqprop_endpoint_read_noise_draw_count", -1))
            == expected_draws,
        }
        errors.extend(
            f"{run_dir.name}: failed guard {name}"
            for name, passed in checks.items()
            if not passed
        )

        initial_hashes.add(str(metrics.get("initial_parameter_state_sha256")))
        train_index_hashes.add(str(provenance.get("train_indices_sha256")))
        validation_index_hashes.add(str(provenance.get("validation_indices_sha256")))
        batch_order = tuple(str(value) for value in provenance.get("train_batch_order_sha256", []))
        batch_order_hashes.add(batch_order)

        train_loss = _load_array(run_dir, "loss_train.npy", 10)
        validation_loss = _load_array(run_dir, "loss_test.npy", 10)
        train_accuracy = _load_array(run_dir, "accuracy_train.npy", 10)
        validation_accuracy = _load_array(run_dir, "accuracy_test.npy", 10)
        best_epoch = int(np.argmax(validation_accuracy)) + 1
        if not _float_equal(metrics.get("best_validation_accuracy"), np.max(validation_accuracy)):
            errors.append(f"{run_dir.name}: best validation accuracy/history mismatch")
        if int(metrics.get("best_epoch", -1)) != best_epoch:
            errors.append(f"{run_dir.name}: best epoch/history mismatch")
        if not _float_equal(metrics.get("final_validation_accuracy"), validation_accuracy[-1]):
            errors.append(f"{run_dir.name}: final validation accuracy/history mismatch")

        cases.append(
            {
                "index": index,
                "scheme": scheme,
                "sigma": sigma,
                "run_dir": str(run_dir),
                "injected_beta_B": float(eqprop["injected_beta_B"]),
                "base_beta": float(config["beta"]),
                "learning_rates": [float(value) for value in config["lr"]],
                "initial_parameter_state_sha256": metrics.get(
                    "initial_parameter_state_sha256"
                ),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": validation_accuracy,
                "best_epoch": best_epoch,
            }
        )

    expected_identities = {(scheme, sigma) for scheme in SCHEMES for sigma in SIGMAS}
    if identities != expected_identities:
        errors.append(f"case coverage mismatch: {sorted(identities)!r}")
    if config_indices != set(range(12)):
        errors.append(f"global exact-run indices mismatch: {sorted(config_indices)!r}")
    for label, values in (
        ("initial parameter SHA", initial_hashes),
        ("train indices SHA", train_index_hashes),
        ("validation indices SHA", validation_index_hashes),
        ("epoch batch-order SHA sequence", batch_order_hashes),
    ):
        if len(values) != 1:
            errors.append(f"{label} is not identical across all cases: {values!r}")
    if errors:
        raise ValueError("Production validation failed:\n- " + "\n- ".join(errors))

    cases.sort(key=lambda row: row["index"])
    verification = {
        "schema_version": "conv2-eqprop-read-noise-analysis-verification/v1",
        "study_id": STUDY_ID,
        "status": "pass",
        "canonical_run_count": len(cases),
        "expected_case_count": 12,
        "indices": sorted(config_indices),
        "initial_parameter_state_sha256": next(iter(initial_hashes)),
        "train_indices_sha256": next(iter(train_index_hashes)),
        "validation_indices_sha256": next(iter(validation_index_hashes)),
        "train_batch_order_sha256": list(next(iter(batch_order_hashes))),
        "expected_nonzero_noise_draws_per_case": EXPECTED_ENDPOINT_NOISE_DRAWS,
        "official_test_read": False,
        "runtime_dtype": "float64",
        "T": 8,
        "K": 8,
    }
    return cases, verification


def summarize(cases: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clean = {row["scheme"]: row for row in cases if float(row["sigma"]) == 0.0}
    summary: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for case in cases:
        scheme = str(case["scheme"])
        anchor = clean[scheme]
        validation_accuracy = np.asarray(case["validation_accuracy"])
        clean_validation_accuracy = np.asarray(anchor["validation_accuracy"])
        train_accuracy = np.asarray(case["train_accuracy"])
        clean_train_accuracy = np.asarray(anchor["train_accuracy"])
        validation_loss = np.asarray(case["validation_loss"])
        clean_validation_loss = np.asarray(anchor["validation_loss"])
        sigma = float(case["sigma"])
        summary.append(
            {
                "index": case["index"],
                "scheme": scheme,
                "sigma": f"{sigma:.12g}",
                "injected_beta_B": f"{float(case['injected_beta_B']):.12g}",
                "base_beta": f"{float(case['base_beta']):.12g}",
                "best_epoch": case["best_epoch"],
                "best_validation_accuracy": float(np.max(validation_accuracy)),
                "final_validation_accuracy": float(validation_accuracy[-1]),
                "final_train_accuracy": float(train_accuracy[-1]),
                "final_validation_loss": float(validation_loss[-1]),
                "clean_best_validation_accuracy": float(
                    np.max(clean_validation_accuracy)
                ),
                "clean_final_validation_accuracy": float(
                    clean_validation_accuracy[-1]
                ),
                "best_validation_drop_pp_vs_clean": 100.0
                * (np.max(clean_validation_accuracy) - np.max(validation_accuracy)),
                "final_validation_drop_pp_vs_clean": 100.0
                * (clean_validation_accuracy[-1] - validation_accuracy[-1]),
                "final_train_drop_pp_vs_clean": 100.0
                * (clean_train_accuracy[-1] - train_accuracy[-1]),
                "mean_validation_drop_pp_vs_clean": 100.0
                * float(np.mean(clean_validation_accuracy - validation_accuracy)),
                "final_validation_loss_delta_vs_clean": float(
                    validation_loss[-1] - clean_validation_loss[-1]
                ),
                "run_dir": case["run_dir"],
                "initial_parameter_state_sha256": case[
                    "initial_parameter_state_sha256"
                ],
            }
        )
        for epoch, values in enumerate(
            zip(
                train_accuracy,
                validation_accuracy,
                validation_loss,
                clean_validation_accuracy,
                strict=True,
            ),
            start=1,
        ):
            train_acc, val_acc, val_loss, clean_val_acc = values
            curves.append(
                {
                    "scheme": scheme,
                    "sigma": f"{sigma:.12g}",
                    "epoch": epoch,
                    "train_accuracy": float(train_acc),
                    "validation_accuracy": float(val_acc),
                    "validation_loss": float(val_loss),
                    "validation_drop_pp_vs_clean": 100.0
                    * float(clean_val_acc - val_acc),
                }
            )
    return summary, curves


def _sigma_label(sigma: float) -> str:
    return "0" if sigma == 0.0 else f"{sigma:.0e}"


def make_plot(cases: Sequence[Mapping[str, Any]], path: Path) -> None:
    colors = {0.0: "#222222", 1e-6: "#4c78a8", 1e-5: "#f58518", 1e-4: "#d62728"}
    markers = {0.0: "o", 1e-6: "s", 1e-5: "^", 1e-4: "D"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
    by_identity = {(row["scheme"], float(row["sigma"])): row for row in cases}
    epochs = np.arange(1, 11)
    for axis, scheme in zip(axes, SCHEMES, strict=True):
        for sigma in SIGMAS:
            case = by_identity[(scheme, sigma)]
            axis.plot(
                epochs,
                100.0 * np.asarray(case["validation_accuracy"]),
                color=colors[sigma],
                marker=markers[sigma],
                markersize=4.0,
                linewidth=1.7,
                label=rf"$\sigma_v={_sigma_label(sigma)}$",
            )
        axis.set_title(
            f"{scheme.capitalize()}  (injected β={EXPECTED_INJECTED_BETA[scheme]:g})"
        )
        axis.set_xticks([1, 2, 4, 6, 8, 10])
        axis.grid(True, alpha=0.25)
        axis.set_xlabel("Epoch")
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[-1].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle(
        "Conv2 float64 centered EqProp: noiseless validation under endpoint read noise",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: Sequence[Mapping[str, Any]], path: Path) -> None:
    lines = [
        "# Conv2 centered-float64 EqProp read-noise exploration",
        "",
        "This is a ten-epoch ordinary-MNIST exploratory comparison, not paper-facing accuracy evidence. Validation is noiseless; independent Gaussian noise is applied only when the two training-phase endpoint states are read for the local EqProp gradient.",
        "",
        "| Scheme | σ | β injected | Best val. | Final val. | Final drop vs clean | Mean curve drop vs clean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {scheme} | {sigma} | {beta} | {best:.2f}% | {final:.2f}% | {drop:+.2f} pp | {mean:+.2f} pp |".format(
                scheme=row["scheme"],
                sigma=_sigma_label(float(row["sigma"])),
                beta=float(row["injected_beta_B"]),
                best=100.0 * float(row["best_validation_accuracy"]),
                final=100.0 * float(row["final_validation_accuracy"]),
                drop=-float(row["final_validation_drop_pp_vs_clean"]),
                mean=-float(row["mean_validation_drop_pp_vs_clean"]),
            )
        )
    lines.extend(
        [
            "",
            "Positive degradation values in the CSV mean the noisy run is worse than its own scheme-matched σ=0 control; the report table displays the signed accuracy change, so negative values indicate damage.",
            "",
            "All twelve bundles, initialization/cohort/order hashes, float64 and T/K contracts, noise draw counts, and official-test guards must pass before this report is written.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases, verification = load_and_validate(args.production_root)
    summary, curves = summarize(cases)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = list(summary[0])
    curve_fields = list(curves[0])
    _write_csv(output_dir / "summary.csv", summary, summary_fields)
    _write_csv(output_dir / "epoch_curves.csv", curves, curve_fields)
    atomic_write_json(output_dir / "verification.json", verification)
    make_plot(cases, output_dir / "validation_accuracy_side_by_side.png")
    write_report(summary, output_dir / "report.md")
    print(json.dumps(verification, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
