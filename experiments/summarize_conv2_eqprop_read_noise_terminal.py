#!/usr/bin/env python3
"""Summarize terminal Conv2 EqProp noise cases at two lower-beta tiers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import atomic_write_json, validate_run


STUDY_ID = (
    "perfectdiode-conv2-centered-float64-eqprop-read-noise-beta-down-"
    "1to2decades-10ep-seed0-20260812-v1"
)
SCHEMES = ("baseline", "ours", "legacy")
BETA_TIERS = ("one_decade_lower", "two_decades_lower")
# The user stopped the exploratory grid at 1e-5 after the clean and 1e-4
# anchors had already completed.  The six materialized 1e-6 configs are
# intentionally outside the final scientific coverage.
SIGMAS = (0.0, 1.0e-5, 1.0e-4)
EXPECTED_INDICES = frozenset(
    (0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22, 23)
)
INJECTED_BETA = {
    "one_decade_lower": {"baseline": 100.0, "ours": 10.0, "legacy": 0.03},
    "two_decades_lower": {"baseline": 10.0, "ours": 1.0, "legacy": 0.003},
}
BASE_BETA = {
    "one_decade_lower": {
        "baseline": 100.0,
        "ours": 0.625,
        "legacy": 0.0001171875,
    },
    "two_decades_lower": {
        "baseline": 10.0,
        "ours": 0.0625,
        "legacy": 0.00001171875,
    },
}
BATCHES_PER_EPOCH = math.ceil(55_000 / 16)
TOTAL_STEPS = 10 * BATCHES_PER_EPOCH
FAILURE_RE = re.compile(
    r"NonFiniteTrainingError:.*?epoch=(?P<epoch>\d+), batch=(?P<batch>\d+).*?"
    r"(?P<count>\d+) non-finite element",
    re.DOTALL,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _same(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=1.0e-12, abs_tol=1.0e-15)
    except (TypeError, ValueError):
        return False


def _epoch_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") != "epoch":
            continue
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"Invalid epoch record at {path}:{line_number}.")
        rows.append(
            {
                "epoch": int(row["epoch"]),
                "train_accuracy": float(metrics["train_accuracy"]),
                "train_loss": float(metrics["train_loss"]),
                "validation_accuracy": float(metrics["validation_accuracy"]),
                "validation_loss": float(metrics["validation_loss"]),
            }
        )
    expected_epochs = list(range(1, len(rows) + 1))
    if [row["epoch"] for row in rows] != expected_epochs:
        raise ValueError(f"Non-contiguous epoch records in {path}.")
    return rows


def _parse_failure_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = FAILURE_RE.search(text)
    if match is None:
        tail = [line for line in text.splitlines() if line.strip()][-1:]
        return {
            "failure_epoch": None,
            "failure_batch": None,
            "nonfinite_element_count": None,
            "failure_detail": tail[0] if tail else "missing failure detail",
        }
    return {
        "failure_epoch": int(match.group("epoch")),
        "failure_batch": int(match.group("batch")),
        "nonfinite_element_count": int(match.group("count")),
        "failure_detail": match.group(0).splitlines()[-1],
    }


def _case_identity(config: Mapping[str, Any]) -> tuple[str, str, float]:
    arm_id = str(config.get("arm_id", ""))
    matches = [scheme for scheme in SCHEMES if arm_id.startswith(f"conv2_{scheme}_")]
    if len(matches) != 1:
        raise ValueError(f"Cannot parse scheme from arm_id={arm_id!r}.")
    eqprop = config.get("eqprop", {})
    beta_tier = str(eqprop.get("beta_tier", ""))
    if beta_tier not in BETA_TIERS:
        raise ValueError(f"Cannot parse beta tier from arm_id={arm_id!r}.")
    return beta_tier, matches[0], float(eqprop["endpoint_read_noise_std"])


def load_terminal_cases(
    production_root: Path, logs_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    production_root = production_root.expanduser().resolve()
    logs_root = logs_root.expanduser().resolve()
    run_dirs = sorted(
        path
        for path in production_root.iterdir()
        if path.is_dir() and (path / "manifest.json").is_file()
    )
    if len(run_dirs) != len(EXPECTED_INDICES):
        raise ValueError(
            f"Expected {len(EXPECTED_INDICES)} run bundles; found {len(run_dirs)}."
        )

    cases: list[dict[str, Any]] = []
    errors: list[str] = []
    identities: set[tuple[str, str, float]] = set()
    indices: set[int] = set()
    initial_hashes: set[str] = set()
    train_index_hashes: set[str] = set()
    validation_index_hashes: set[str] = set()
    batch_orders: set[tuple[str, ...]] = set()

    for run_dir in run_dirs:
        errors.extend(f"{run_dir.name}: {error}" for error in validate_run(run_dir))
        manifest = _read_json(run_dir / "manifest.json")
        status = _read_json(run_dir / "status.json")
        exact = _read_json(run_dir / "exact_run.json")
        run_config = _read_json(run_dir / "config.json")
        config = manifest.get("configuration", {}).get("resolved", {})
        beta_tier, scheme, sigma = _case_identity(config)
        identity = (beta_tier, scheme, sigma)
        index = int(exact.get("index", -1))
        identities.add(identity)
        indices.add(index)
        state = str(status.get("state"))
        if state not in {"complete", "failed"}:
            errors.append(f"{run_dir.name}: non-terminal state {state!r}")

        training = run_config.get("training", {})
        provenance = run_config.get("dataset", {}).get("provenance", {})
        architecture = run_config.get("architecture", {})
        metric_rows = _epoch_records(run_dir / "metrics.jsonl")
        completed_epochs = len(metric_rows)
        log_path = logs_root / f"case_{index:02d}.log"
        if not log_path.is_file():
            errors.append(f"{run_dir.name}: missing transport log {log_path}")
        failure = (
            _parse_failure_log(log_path)
            if state == "failed" and log_path.is_file()
            else {
                "failure_epoch": None,
                "failure_batch": None,
                "nonfinite_element_count": None,
                "failure_detail": None,
            }
        )
        if state == "complete" and completed_epochs != 10:
            errors.append(f"{run_dir.name}: complete with {completed_epochs}/10 epochs")
        if state == "failed" and completed_epochs >= 10:
            errors.append(f"{run_dir.name}: failed despite ten completed epochs")
        if state == "failed" and failure["failure_epoch"] is None:
            errors.append(f"{run_dir.name}: failure log lacks a non-finite location")

        checks = {
            "study": manifest.get("study_id") == STUDY_ID,
            "evidence": manifest.get("evidence_class")
            == "ordinary_mnist_eqprop_read_noise_training_diagnostic",
            "not_smoke": manifest.get("smoke") is False,
            "official_test": manifest.get("dataset", {}).get("official_test_read") is False,
            "float64": training.get("runtime_dtype") == "float64",
            "algorithm": run_config.get("training_algorithm") == "EP",
            "epochs": int(training.get("epochs", -1)) == 10,
            "T": int(training.get("num_iterations_inference", -1)) == 8,
            "K": int(training.get("num_iterations_training", -1)) == 8,
            "base_beta": _same(training.get("beta"), BASE_BETA[beta_tier][scheme]),
            "injected_beta": _same(
                config.get("eqprop", {}).get("injected_beta_B"),
                INJECTED_BETA[beta_tier][scheme],
            ),
            "noise_sigma": _same(
                training.get("eqprop", {}).get("endpoint_read_noise_std"), sigma
            ),
            "noise_seed": int(
                training.get("eqprop", {}).get("endpoint_read_noise_seed", -1)
            )
            == 2026081201,
            "centered": training.get("eqprop", {}).get("variant") == "centered",
            "current": training.get("eqprop", {}).get("nudging_mode") == "current",
            "normalized": training.get("eqprop", {}).get("normalize_current_scale")
            is True,
            "no_input_noise": training.get("eqprop", {}).get("input_read_noise")
            is False,
        }
        errors.extend(
            f"{run_dir.name}: failed guard {name}"
            for name, passed in checks.items()
            if not passed
        )

        initial_hash = str(architecture.get("initial_parameter_state_sha256"))
        initial_hashes.add(initial_hash)
        train_index_hashes.add(str(provenance.get("train_indices_sha256")))
        validation_index_hashes.add(str(provenance.get("validation_indices_sha256")))
        batch_orders.add(
            tuple(str(value) for value in provenance.get("train_batch_order_sha256", []))
        )
        failure_epoch = failure["failure_epoch"]
        failure_batch = failure["failure_batch"]
        if state == "complete":
            terminal_steps = TOTAL_STEPS
            step_is_lower_bound = False
        elif failure_epoch is not None and failure_batch is not None:
            terminal_steps = (int(failure_epoch) - 1) * BATCHES_PER_EPOCH + int(
                failure_batch
            )
            step_is_lower_bound = False
        else:
            terminal_steps = completed_epochs * BATCHES_PER_EPOCH
            step_is_lower_bound = True

        cases.append(
            {
                "index": index,
                "beta_tier": beta_tier,
                "scheme": scheme,
                "sigma": sigma,
                "state": state,
                "completed_epochs": completed_epochs,
                "terminal_steps": terminal_steps,
                "terminal_step_is_lower_bound": step_is_lower_bound,
                "failure_epoch": failure_epoch,
                "failure_batch": failure_batch,
                "nonfinite_element_count": failure["nonfinite_element_count"],
                "failure_detail": failure["failure_detail"],
                "injected_beta_B": float(config["eqprop"]["injected_beta_B"]),
                "base_beta": float(training["beta"]),
                "initial_parameter_state_sha256": initial_hash,
                "epochs": metric_rows,
                "run_dir": str(run_dir),
                "log_path": str(log_path),
            }
        )

    expected = {
        (beta_tier, scheme, sigma)
        for beta_tier in BETA_TIERS
        for scheme in SCHEMES
        for sigma in SIGMAS
    }
    if identities != expected:
        errors.append(f"case coverage mismatch: {sorted(identities)!r}")
    if indices != EXPECTED_INDICES:
        errors.append(f"global index coverage mismatch: {sorted(indices)!r}")
    for label, values in (
        ("initialization", initial_hashes),
        ("train indices", train_index_hashes),
        ("validation indices", validation_index_hashes),
        ("batch orders", batch_orders),
    ):
        if len(values) != 1:
            errors.append(f"{label} differs across cases: {values!r}")
    if errors:
        raise ValueError("Terminal coverage validation failed:\n- " + "\n- ".join(errors))

    cases.sort(key=lambda row: row["index"])
    verification = {
        "schema_version": "conv2-eqprop-noise-terminal-verification/v1",
        "study_id": STUDY_ID,
        "status": "pass",
        "canonical_run_count": len(cases),
        "complete_run_count": sum(row["state"] == "complete" for row in cases),
        "failed_run_count": sum(row["state"] == "failed" for row in cases),
        "indices": sorted(indices),
        "initial_parameter_state_sha256": next(iter(initial_hashes)),
        "train_indices_sha256": next(iter(train_index_hashes)),
        "validation_indices_sha256": next(iter(validation_index_hashes)),
        "train_batch_order_sha256": list(next(iter(batch_orders))),
        "runtime_dtype": "float64",
        "T": 8,
        "K": 8,
        "official_test_read": False,
    }
    return cases, verification


def summarize(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    clean = {
        (row["beta_tier"], row["scheme"]): row
        for row in cases
        if float(row["sigma"]) == 0.0
    }
    rows: list[dict[str, Any]] = []
    for case in cases:
        anchor = clean[(case["beta_tier"], case["scheme"])]
        epoch_rows = list(case["epochs"])
        anchor_rows = list(anchor["epochs"])
        common = min(len(epoch_rows), len(anchor_rows))
        best = max((row["validation_accuracy"] for row in epoch_rows), default=None)
        best_epoch = next(
            (
                row["epoch"]
                for row in epoch_rows
                if best is not None and row["validation_accuracy"] == best
            ),
            None,
        )
        last = epoch_rows[-1]["validation_accuracy"] if epoch_rows else None
        if common:
            common_delta = 100.0 * (
                epoch_rows[common - 1]["validation_accuracy"]
                - anchor_rows[common - 1]["validation_accuracy"]
            )
            best_prefix_delta = 100.0 * (
                max(row["validation_accuracy"] for row in epoch_rows[:common])
                - max(row["validation_accuracy"] for row in anchor_rows[:common])
            )
        else:
            common_delta = None
            best_prefix_delta = None
        rows.append(
            {
                "index": case["index"],
                "beta_tier": case["beta_tier"],
                "scheme": case["scheme"],
                "sigma": f"{float(case['sigma']):.12g}",
                "injected_beta_B": f"{float(case['injected_beta_B']):.12g}",
                "base_beta": f"{float(case['base_beta']):.12g}",
                "state": case["state"],
                "completed_epochs": case["completed_epochs"],
                "terminal_steps": case["terminal_steps"],
                "survival_fraction": float(case["terminal_steps"]) / TOTAL_STEPS,
                "survival_step_delta_vs_clean": int(case["terminal_steps"])
                - int(anchor["terminal_steps"]),
                "failure_epoch": case["failure_epoch"],
                "failure_batch": case["failure_batch"],
                "nonfinite_element_count": case["nonfinite_element_count"],
                "best_observed_validation_accuracy": best,
                "best_observed_epoch": best_epoch,
                "last_observed_validation_accuracy": last,
                "common_completed_epochs_vs_clean": common,
                "validation_change_pp_at_last_common_epoch": common_delta,
                "best_prefix_change_pp_vs_clean": best_prefix_delta,
                "run_dir": case["run_dir"],
                "log_path": case["log_path"],
            }
        )
    return rows


def _sigma_label(value: float) -> str:
    return "0" if value == 0.0 else f"{value:.0e}"


def make_plot(cases: Sequence[Mapping[str, Any]], path: Path) -> None:
    colors = {0.0: "#222222", 1e-5: "#f58518", 1e-4: "#d62728"}
    markers = {0.0: "o", 1e-5: "^", 1e-4: "D"}
    by_identity = {
        (row["beta_tier"], row["scheme"], float(row["sigma"])): row
        for row in cases
    }
    fig, axes = plt.subplots(4, 3, figsize=(13.2, 13.2), sharex=False)
    for tier_index, beta_tier in enumerate(BETA_TIERS):
        for column, scheme in enumerate(SCHEMES):
            top = axes[2 * tier_index, column]
            bottom = axes[2 * tier_index + 1, column]
            for position, sigma in enumerate(SIGMAS):
                case = by_identity[(beta_tier, scheme, sigma)]
                records = list(case["epochs"])
                x = [row["epoch"] for row in records]
                y = [100.0 * row["validation_accuracy"] for row in records]
                top.plot(
                    x,
                    y,
                    color=colors[sigma],
                    marker=markers[sigma],
                    linewidth=1.7,
                    markersize=4,
                    label=rf"$\sigma_v={_sigma_label(sigma)}$",
                )
                if case["state"] == "failed" and y:
                    top.scatter(
                        x[-1],
                        y[-1],
                        color=colors[sigma],
                        marker="x",
                        s=55,
                        zorder=5,
                    )
                bottom.bar(
                    position,
                    100.0 * float(case["terminal_steps"]) / TOTAL_STEPS,
                    color=colors[sigma],
                    width=0.72,
                )
            tier_title = "1 decade lower" if tier_index == 0 else "2 decades lower"
            top.set_title(
                f"{scheme.capitalize()}, {tier_title}  "
                f"(injected β={INJECTED_BETA[beta_tier][scheme]:g})"
            )
            top.grid(True, alpha=0.25)
            top.set_xlim(0.5, 10.5)
            top.set_xticks([1, 2, 4, 6, 8, 10])
            top.set_xlabel("Epoch")
            bottom.set_xticks(
                range(len(SIGMAS)), [_sigma_label(value) for value in SIGMAS]
            )
            bottom.set_ylim(0, 105)
            bottom.grid(True, axis="y", alpha=0.25)
            bottom.set_xlabel(r"Endpoint read-noise $\sigma_v$")
    axes[0, 0].set_ylabel("Validation accuracy (%)")
    axes[1, 0].set_ylabel("Training steps survived (%)")
    axes[2, 0].set_ylabel("Validation accuracy (%)")
    axes[3, 0].set_ylabel("Training steps survived (%)")
    axes[0, -1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Conv2 float64 centered EqProp: lower-beta training under read noise")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_penalty_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    by_identity = {
        (row["beta_tier"], row["scheme"], float(row["sigma"])): row
        for row in rows
    }
    nonzero_sigmas = tuple(sigma for sigma in SIGMAS if sigma > 0.0)
    colors = {1.0e-5: "#f58518", 1.0e-4: "#d62728"}
    x = np.arange(len(SCHEMES), dtype=float)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for axis, beta_tier in zip(axes, BETA_TIERS, strict=True):
        for sigma_index, sigma in enumerate(nonzero_sigmas):
            values = [
                float(
                    by_identity[(beta_tier, scheme, sigma)][
                        "validation_change_pp_at_last_common_epoch"
                    ]
                )
                for scheme in SCHEMES
            ]
            offsets = x + (sigma_index - (len(nonzero_sigmas) - 1) / 2.0) * width
            bars = axis.bar(
                offsets,
                values,
                width=width,
                color=colors[sigma],
                label=rf"$\sigma_v={_sigma_label(sigma)}$",
            )
            axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        tier_title = (
            "Injected beta: 100 / 10 / 0.03"
            if beta_tier == "one_decade_lower"
            else "Injected beta: 10 / 1 / 0.003"
        )
        axis.set_title(tier_title)
        axis.set_xticks(x, [scheme.capitalize() for scheme in SCHEMES])
        axis.axhline(0.0, color="#222222", linewidth=0.9)
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_xlabel("Amplification scheme")
    axes[0].set_ylabel("Validation change vs matched clean control (pp)\n(last common epoch)")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Conv2 float64 centered EqProp: endpoint-read-noise penalty")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_accuracy(value: Any) -> str:
    return "--" if value is None else f"{100.0 * float(value):.2f}%"


def _format_pp(value: Any) -> str:
    return "--" if value is None else f"{float(value):+.2f}"


def write_report(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    lines = [
        "# Conv2 lower-beta endpoint-read-noise terminal report",
        "",
        "This is ordinary-MNIST exploratory evidence. A failed run is retained as a scientific outcome; it is not converted into an accuracy result or retried unchanged.",
        "",
        "| β tier | Scheme | σ | Injected β | State | Epochs complete | Failure location | Best val. | Last val. | Δ val. vs clean (pp) | Δ best prefix (pp) | Steps survived |",
        "|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        failure = (
            "--"
            if row["failure_epoch"] is None
            else f"e{row['failure_epoch']} b{row['failure_batch']}"
        )
        lines.append(
            "| {tier} | {scheme} | {sigma} | {beta} | {state} | {epochs}/10 | {failure} | {best} | {last} | {common_delta} | {best_delta} | {survival:.1f}% |".format(
                tier=(
                    "1 decade lower"
                    if row["beta_tier"] == "one_decade_lower"
                    else "2 decades lower"
                ),
                scheme=row["scheme"],
                sigma=_sigma_label(float(row["sigma"])),
                beta=float(row["injected_beta_B"]),
                state=row["state"],
                epochs=row["completed_epochs"],
                failure=failure,
                best=_format_accuracy(row["best_observed_validation_accuracy"]),
                last=_format_accuracy(row["last_observed_validation_accuracy"]),
                common_delta=_format_pp(
                    row["validation_change_pp_at_last_common_epoch"]
                ),
                best_delta=_format_pp(row["best_prefix_change_pp_vs_clean"]),
                survival=100.0 * float(row["survival_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "The σ=0 row within each beta tier and amplification scheme is the survival and validation anchor. Accuracy deltas are only compared over epochs completed by both members of a pair.",
            "",
            "All 18 retained cases use identical initialization, MNIST split, minibatch order, float64 arithmetic, T=K=8, fixed Adam learning-rate vectors, and noiseless validation; the official test set is never read. The six sigma=1e-6 cases materialized in the original configuration were intentionally omitted when the user stopped the grid at sigma=1e-5.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases, verification = load_terminal_cases(args.production_root, args.logs_root)
    rows = summarize(cases)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "terminal_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(output_dir / "verification.json", verification)
    make_plot(cases, output_dir / "terminal_training_side_by_side.png")
    make_penalty_plot(rows, output_dir / "noise_penalty_by_beta_tier.png")
    write_report(rows, output_dir / "report.md")
    print(json.dumps(verification, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
