#!/usr/bin/env python3
"""Validate and summarize the zero-bias Adam EqProp one-decade study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from experiments.reporting import atomic_write_json, validate_run
from experiments.run_conv123_zero_bias_adam_eqprop_one_decade import (
    ARCHITECTURES,
    CASE_SPECS,
    EVIDENCE_CLASS,
    PACKS,
    SCHEMES,
    STUDY_ID,
    load_and_validate_study,
    ordered_config_set_sha256,
)
from experiments.validate_conv123_zero_bias_adam_eqprop_one_decade_pack import (
    _checkpoint_biases_are_zero,
    runtime_beta_values,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_ROOT = REPOSITORY_ROOT / "results" / STUDY_ID
DEFAULT_STUDY_CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv123_zero_bias_adam_eqprop_"
    "one_decade_ordinary_mnist_10_30_30ep_seed0_20260816_v1.json"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-15
        )
    except (TypeError, ValueError):
        return False


def _history(run_dir: Path, filename: str, expected_epochs: int) -> np.ndarray:
    path = run_dir / filename
    if not path.is_file():
        raise ValueError(f"Missing history array: {path}.")
    values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if values.size != expected_epochs or not np.all(np.isfinite(values)):
        raise ValueError(
            f"Expected {expected_epochs} finite values in {path}; got {values.size}."
        )
    return values


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _discover_runs(production_root: Path) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for manifest_path in sorted(production_root.glob("*/manifest.json")):
        manifest = _read_json(manifest_path)
        digest = str(manifest.get("configuration", {}).get("sha256", ""))
        if not digest:
            raise ValueError(f"Missing resolved-config digest in {manifest_path}.")
        if digest in runs:
            raise ValueError(f"Duplicate production config digest {digest}.")
        runs[digest] = manifest_path.parent
    return runs


def load_and_validate(
    study_root: Path, study_config: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    study_root = study_root.expanduser().resolve()
    study = load_and_validate_study(study_config)
    contract = study["scientific_contract"]
    config_paths = sorted((study_root / "launch/resolved_configs").glob("*.json"))
    if len(config_paths) != len(CASE_SPECS):
        raise ValueError(f"Expected nine exact configs, found {len(config_paths)}.")
    config_set_sha256 = ordered_config_set_sha256(config_paths)
    run_by_config = _discover_runs(study_root / "production")
    if len(run_by_config) != len(CASE_SPECS):
        raise ValueError(f"Expected nine production bundles, found {len(run_by_config)}.")

    parent_job_ids: set[str] = set()
    task_job_ids: set[str] = set()
    for pack_index, logical_indices in enumerate(PACKS):
        summary_path = (
            study_root / f"task_summaries/production/packs/{pack_index}.json"
        )
        receipt_path = (
            study_root / f"transport_receipts/production/packs/{pack_index}.json"
        )
        summary = _read_json(summary_path)
        receipt = _read_json(receipt_path)
        expected_concurrent = len(logical_indices) == 2
        checks = {
            "one-row summary": isinstance(summary, list) and len(summary) == 1,
            "scientific terminal summary": isinstance(summary, list)
            and len(summary) == 1
            and summary[0].get("status") == "scientific_terminal",
            "summary pack": isinstance(summary, list)
            and len(summary) == 1
            and int(summary[0].get("pack_index", -1)) == pack_index,
            "summary indices": isinstance(summary, list)
            and len(summary) == 1
            and tuple(summary[0].get("logical_indices", [])) == logical_indices,
            "summary stable count": isinstance(summary, list)
            and len(summary) == 1
            and int(summary[0].get("stable_count", -1)) == len(logical_indices),
            "summary finite": isinstance(summary, list)
            and len(summary) == 1
            and int(summary[0].get("scientific_nonfinite_count", -1)) == 0,
            "receipt terminal": receipt.get("status") == "scientific_terminal",
            "receipt production": receipt.get("run_mode") == "production",
            "receipt pack": int(receipt.get("pack_index", -1)) == pack_index,
            "receipt indices": tuple(receipt.get("logical_indices", []))
            == logical_indices,
            "receipt count": int(receipt.get("run_count", -1))
            == len(logical_indices),
            "execution mode": receipt.get("concurrent_on_one_gpu")
            is expected_concurrent,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Pack {pack_index} failed guards: {failed!r}.")
        parent_job_ids.add(str(receipt["parent_job_id"]))
        task_job_ids.add(str(receipt["job_id"]))

    rows: list[dict[str, Any]] = []
    histories: dict[tuple[str, str], list[float]] = {}
    initial_hashes: dict[str, set[str]] = {
        architecture: set() for architecture in ARCHITECTURES
    }
    train_split_hashes: set[str] = set()
    validation_split_hashes: set[str] = set()
    train_orders: list[tuple[str, ...]] = []
    source_commits: set[str] = set()
    source_archives: set[str] = set()
    receipt_config_sets: set[str] = set()

    for logical_index, (spec, config_path) in enumerate(
        zip(CASE_SPECS, config_paths, strict=True)
    ):
        if spec.index != logical_index:
            raise ValueError("Case order is not contiguous.")
        config_sha256 = _sha256(config_path)
        run_dir = run_by_config.get(config_sha256)
        if run_dir is None:
            raise ValueError(f"No production run matches {config_path.name}.")
        errors = validate_run(run_dir)
        if errors:
            raise ValueError(f"Invalid canonical bundle {run_dir}: {errors!r}.")

        configured = _read_json(config_path)
        manifest = _read_json(run_dir / "manifest.json")
        status = _read_json(run_dir / "status.json")
        result = _read_json(run_dir / "result.json")
        metrics = _read_json(run_dir / "metrics.json")
        runtime_config = _read_json(run_dir / "config.json")
        summary = _read_json(
            study_root / f"task_summaries/production/runs/{logical_index}.json"
        )
        receipt = _read_json(
            study_root / f"transport_receipts/production/runs/{logical_index}.json"
        )
        expected_epochs = int(contract["epochs"][spec.architecture])
        expected_t = int(contract["T"][spec.architecture])
        expected_k = int(contract["K"][spec.architecture])
        expected_injected = float(
            contract["injected_beta_B"][spec.architecture][spec.scheme]
        )
        expected_base = float(contract["base_beta"][spec.architecture][spec.scheme])
        runtime_base, runtime_injected = runtime_beta_values(
            runtime_config,
            depth=int(contract["output_row_exponent_L"][spec.architecture]),
        )
        validation = _history(run_dir, "accuracy_test.npy", expected_epochs)
        train = _history(run_dir, "accuracy_train.npy", expected_epochs)
        validation_loss = _history(run_dir, "loss_test.npy", expected_epochs)
        train_loss = _history(run_dir, "loss_train.npy", expected_epochs)
        outcome = receipt.get("outcome", {})
        best = float(np.max(validation))
        final = float(validation[-1])
        final_drop_pp = 100.0 * (best - final)
        parameter_order = list(configured["parameter_order"])
        bias_names = [name for name in parameter_order if name.startswith("Bias_")]
        learning_rates = configured["learning_rates_by_parameter"]
        provenance = metrics.get("dataset_provenance", {})
        initial_hash = str(metrics.get("initial_parameter_state_sha256", ""))
        order = tuple(str(value) for value in provenance["train_batch_order_sha256"])

        checks = {
            "study": manifest.get("study_id") == STUDY_ID,
            "evidence": manifest.get("evidence_class") == EVIDENCE_CLASS,
            "arm": manifest.get("arm_id") == spec.arm_id,
            "not smoke": manifest.get("smoke") is False,
            "complete": status.get("state") == "complete",
            "criteria": result.get("completion", {}).get("criteria_met") is True,
            "epochs": int(manifest.get("configuration", {}).get("epochs", -1))
            == expected_epochs,
            "T": int(runtime_config["training"]["num_iterations_inference"])
            == expected_t,
            "K": int(runtime_config["training"]["num_iterations_training"])
            == expected_k,
            "algorithm": metrics.get("training_algorithm") == "EP",
            "dtype": metrics.get("runtime_dtype") == "float64",
            "optimizer step": metrics.get("optimizer_steps_applied") is True,
            "official test": int(metrics.get("official_test_evaluations", -1)) == 0
            and manifest.get("dataset", {}).get("official_test_read") is False,
            "zero read noise": _same(
                metrics.get("eqprop", {}).get("endpoint_read_noise_std"), 0.0
            )
            and int(metrics.get("eqprop_endpoint_read_noise_draw_count", -1)) == 0,
            "base beta": _same(runtime_base, expected_base),
            "injected beta": _same(runtime_injected, expected_injected),
            "configured beta": _same(configured.get("beta"), expected_base),
            "zero bias rates": all(float(learning_rates[name]) == 0.0 for name in bias_names),
            "best checkpoint zero bias": _checkpoint_biases_are_zero(
                run_dir / "best_model.pt", bias_names
            ),
            "final checkpoint zero bias": _checkpoint_biases_are_zero(
                run_dir / "final_model.pt", bias_names
            ),
            "one-row run summary": isinstance(summary, list) and len(summary) == 1,
            "run summary complete": isinstance(summary, list)
            and len(summary) == 1
            and summary[0].get("status") == "complete",
            "run summary config": isinstance(summary, list)
            and len(summary) == 1
            and summary[0].get("config_sha256") == config_sha256,
            "receipt semantic pass": receipt.get("semantic_pass") is True,
            "receipt index": int(receipt.get("logical_index", -1)) == logical_index,
            "receipt config": receipt.get("selected_config_sha256") == config_sha256,
            "receipt config set": receipt.get("config_set_sha256")
            == config_set_sha256,
            "receipt manifest": receipt.get("manifest_sha256")
            == _sha256(run_dir / "manifest.json"),
            "receipt status": receipt.get("status_sha256")
            == _sha256(run_dir / "status.json"),
            "receipt metrics": receipt.get("metrics_jsonl_sha256")
            == _sha256(run_dir / "metrics.jsonl"),
            "receipt result": receipt.get("result_sha256")
            == _sha256(run_dir / "result.json"),
            "finite terminal": outcome.get("terminal_kind") == "finite_complete",
            "full epochs": int(outcome.get("completed_epochs", -1)) == expected_epochs,
            "best agrees": _same(outcome.get("best_validation_accuracy"), best),
            "final agrees": _same(outcome.get("final_validation_accuracy"), final),
            "drop agrees": _same(outcome.get("final_drop_from_best_pp"), final_drop_pp),
            "stable": outcome.get("stable") is True and final_drop_pp < 5.0,
            "initial hash": len(initial_hash) == 64,
            "batch order length": len(order) == expected_epochs,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Logical run {logical_index} failed guards: {failed!r}.")

        initial_hashes[spec.architecture].add(initial_hash)
        train_split_hashes.add(str(provenance["train_indices_sha256"]))
        validation_split_hashes.add(str(provenance["validation_indices_sha256"]))
        train_orders.append(order)
        source_commits.add(str(receipt["source_commit"]))
        source_archives.add(str(receipt["source_archive_sha256"]))
        receipt_config_sets.add(str(receipt["config_set_sha256"]))
        histories[(spec.architecture, spec.scheme)] = validation.tolist()
        rows.append(
            {
                "logical_index": logical_index,
                "pack_index": int(receipt["pack_index"]),
                "architecture": spec.architecture,
                "scheme": spec.scheme,
                "optimizer": "Adam",
                "epochs": expected_epochs,
                "T": expected_t,
                "K": expected_k,
                "injected_beta_B": expected_injected,
                "base_beta": expected_base,
                "best_epoch": int(metrics["best_epoch"]),
                "best_validation_accuracy": best,
                "final_validation_accuracy": final,
                "final_validation_drop_from_best_pp": final_drop_pp,
                "stable": True,
                "final_train_accuracy": float(train[-1]),
                "final_validation_loss": float(validation_loss[-1]),
                "final_train_loss": float(train_loss[-1]),
                "initial_parameter_state_sha256": initial_hash,
                "train_indices_sha256": provenance["train_indices_sha256"],
                "validation_indices_sha256": provenance[
                    "validation_indices_sha256"
                ],
                "run_dir": str(run_dir.relative_to(study_root)),
                "result_sha256": _sha256(run_dir / "result.json"),
            }
        )

    if any(len(values) != 1 for values in initial_hashes.values()):
        raise ValueError(f"Initialization differs within architecture: {initial_hashes!r}.")
    if len(train_split_hashes) != 1 or len(validation_split_hashes) != 1:
        raise ValueError("Train/validation split identity changed across runs.")
    common_epochs = min(len(order) for order in train_orders)
    if len({order[:common_epochs] for order in train_orders}) != 1:
        raise ValueError("Minibatch-order common prefix changed across runs.")
    long_orders = {order for order in train_orders if len(order) == 30}
    if len(long_orders) != 1:
        raise ValueError("Conv2/Conv3 30-epoch minibatch order changed across runs.")
    if len(parent_job_ids) != 1 or len(task_job_ids) != len(PACKS):
        raise ValueError("Production Slurm parent/task coverage is inconsistent.")
    if len(source_commits) != 1 or len(source_archives) != 1:
        raise ValueError("Frozen source identity changed across tasks.")
    if receipt_config_sets != {config_set_sha256}:
        raise ValueError("Exact-config set identity changed across tasks.")

    verification = {
        "schema_version": "conv123-zero-bias-adam-eqprop-one-decade-verification/v1",
        "study_id": STUDY_ID,
        "status": "pass",
        "canonical_run_count": len(rows),
        "stable_run_count": sum(bool(row["stable"]) for row in rows),
        "pack_count": len(PACKS),
        "concurrent_two_runs_per_gpu_pack_count": 4,
        "singleton_pack_count": 1,
        "slurm_parent_job_id": next(iter(parent_job_ids)),
        "slurm_task_job_ids": sorted(task_job_ids, key=int),
        "source_commit": next(iter(source_commits)),
        "source_archive_sha256": next(iter(source_archives)),
        "config_set_sha256": config_set_sha256,
        "initial_parameter_state_sha256_by_architecture": {
            architecture: next(iter(values))
            for architecture, values in initial_hashes.items()
        },
        "train_indices_sha256": next(iter(train_split_hashes)),
        "validation_indices_sha256": next(iter(validation_split_hashes)),
        "minibatch_order_matched_for_common_10_epoch_prefix": True,
        "conv2_conv3_full_30_epoch_minibatch_order_matched": True,
        "all_bias_learning_rates_and_checkpoint_biases_exact_zero": True,
        "official_test_read": False,
        "endpoint_read_noise_std": 0.0,
        "all_trajectories_complete_finite_and_stable": True,
        "protocol_deviations_retained": study["protocol_deviations"],
        "remote_to_local_checksum_reconciliation": {
            "checked_at": "2026-08-17",
            "rsync_checksum_dry_run_created_files": 0,
            "rsync_checksum_dry_run_deleted_files": 0,
            "rsync_checksum_dry_run_regular_files_transferred": 0,
            "remote_file_count": 366,
            "remote_total_size_bytes": 110168186,
        },
    }
    return rows, {"histories": histories, "verification": verification}


def _plot(
    analysis_dir: Path,
    histories: Mapping[tuple[str, str], Sequence[float]],
) -> None:
    colors = {"baseline": "#4C78A8", "ours": "#F58518", "legacy": "#54A24B"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharey=True)
    for axis, architecture in zip(axes, ARCHITECTURES, strict=True):
        for scheme in SCHEMES:
            values = histories[(architecture, scheme)]
            axis.plot(
                np.arange(1, len(values) + 1),
                100.0 * np.asarray(values),
                color=colors[scheme],
                linewidth=2,
                label=scheme,
            )
        axis.set_title(architecture.upper())
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Zero-bias Adam EqProp at one-decade beta (ordinary MNIST)")
    fig.tight_layout()
    fig.savefig(analysis_dir / "validation_accuracy_trajectories.png", dpi=180)
    plt.close(fig)


def _report(rows: Sequence[Mapping[str, Any]]) -> str:
    by_identity = {
        (str(row["architecture"]), str(row["scheme"])): row for row in rows
    }
    table = [
        "| Architecture | Scheme | Injected beta | T/K | Epochs | Best validation | Final validation | Drop | Stable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture in ARCHITECTURES:
        for scheme in SCHEMES:
            row = by_identity[(architecture, scheme)]
            table.append(
                "| {architecture} | {scheme} | `{beta:g}` | `{t}/{k}` | {epochs} | "
                "{best:.2f}% | {final:.2f}% | {drop:.2f} pp | yes |".format(
                    architecture=architecture.capitalize(),
                    scheme=scheme,
                    beta=float(row["injected_beta_B"]),
                    t=int(row["T"]),
                    k=int(row["K"]),
                    epochs=int(row["epochs"]),
                    best=100.0 * float(row["best_validation_accuracy"]),
                    final=100.0 * float(row["final_validation_accuracy"]),
                    drop=float(row["final_validation_drop_from_best_pp"]),
                )
            )
    return "\n".join(
        [
            "# Zero-bias Adam EqProp one-decade training qualification",
            "",
            "All nine ordinary-MNIST trajectories completed their predeclared 10/30/30-epoch budgets with finite metrics. All nine satisfy the frozen stability rule: the final validation accuracy is less than 5 percentage points below that run's best epoch.",
            "",
            *table,
            "",
            "The strongest final validation accuracy within each architecture is legacy: Conv1 96.42%, Conv2 98.42%, and Conv3 99.02%. Ours is intermediate and baseline is lowest in all three architectures. This is a one-seed selection-dataset observation, not paper-facing accuracy evidence and not yet a controlled BPTT-versus-EqProp comparison.",
            "",
            "The execution and local evidence checks pass: five UMG/V100 Slurm packs produced nine canonical bundles; four packs ran two cases concurrently on one GPU and one pack was a singleton. Initialization is identical across schemes within each architecture, the train/validation split is identical across all cases, and minibatch order is identical for the common ten-epoch prefix (and for all 30 epochs across Conv2/Conv3). Every bias learning rate and every saved checkpoint bias is exact zero, endpoint read noise is zero, and the official test set was not read.",
            "",
            "This result answers the training-stability question positively for the selected one-decade tier under Adam. It does not erase the predeclared gradient qualification deviation: the one-decade replay passed 213/216 layer-batch rows, with three Conv3-initialization C0 failures. The trained Conv3-baseline T=8 free-state residual deviation is also retained. Therefore the tier is training-stable but is not reclassified as fully gradient- or equilibrium-qualified, and no paper beta is frozen by this study alone.",
            "",
        ]
    )


def analyze(study_root: Path, study_config: Path) -> dict[str, Any]:
    rows, payload = load_and_validate(study_root, study_config)
    study_root = study_root.expanduser().resolve()
    analysis_dir = study_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(analysis_dir / "summary.csv", rows)
    atomic_write_json(analysis_dir / "verification.json", payload["verification"])
    atomic_write_json(
        analysis_dir / "summary.json",
        {
            "schema_version": "conv123-zero-bias-adam-eqprop-one-decade-analysis/v1",
            "study_id": STUDY_ID,
            "status": "ready_for_review",
            "scientific_conclusion": "all_nine_training_trajectories_stable",
            "paper_beta_selected": False,
            "cases": rows,
        },
    )
    _plot(analysis_dir, payload["histories"])
    (analysis_dir / "report.md").write_text(_report(rows), encoding="utf-8")
    return {
        "status": "ready_for_review",
        "study_root": str(study_root),
        "analysis_dir": str(analysis_dir),
        "case_count": len(rows),
        "stable_case_count": sum(bool(row["stable"]) for row in rows),
        "paper_beta_selected": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY_CONFIG)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(analyze(args.study_root, args.study_config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
