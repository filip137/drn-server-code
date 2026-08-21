#!/usr/bin/env python3
"""Validate and summarize the Conv1/Conv3 clean-beta qualification study."""

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
from experiments.run_conv13_centered_eqprop_beta_qualification import (
    ARCHITECTURES,
    CASE_SPECS,
    EVIDENCE_CLASS,
    PACKS,
    SCHEMES,
    STUDY_ID,
    load_and_validate_study,
    ordered_config_set_sha256,
)
from experiments.validate_conv13_eqprop_beta_pack import runtime_beta_values


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY_ROOT = REPOSITORY_ROOT / "results" / STUDY_ID


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1.0e-12, abs_tol=1.0e-15)
    except (TypeError, ValueError):
        return False


def _history(run_dir: Path, filename: str) -> np.ndarray:
    path = run_dir / filename
    if not path.is_file():
        raise ValueError(f"Missing history array: {path}.")
    values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if values.size != 10 or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected ten finite values in {path}.")
    return values


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _discover_run_by_config(production_root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for manifest_path in sorted(production_root.glob("*/manifest.json")):
        manifest = _read_json(manifest_path)
        digest = str(manifest.get("configuration", {}).get("sha256", ""))
        if not digest:
            raise ValueError(f"Missing config digest in {manifest_path}.")
        if digest in mapping:
            raise ValueError(f"Duplicate production config digest {digest}.")
        mapping[digest] = manifest_path.parent
    return mapping


def load_and_validate(
    study_root: Path, study_config: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    study_root = study_root.expanduser().resolve()
    study = load_and_validate_study(study_config)
    config_root = study_root / "launch/resolved_configs"
    config_paths = sorted(config_root.glob("*.json"))
    if len(config_paths) != len(CASE_SPECS):
        raise ValueError(f"Expected twelve exact configs, found {len(config_paths)}.")
    expected_config_set_sha = ordered_config_set_sha256(config_paths)
    run_by_config = _discover_run_by_config(study_root / "production")
    if len(run_by_config) != len(CASE_SPECS):
        raise ValueError(f"Expected twelve production bundles, found {len(run_by_config)}.")

    pack_summaries = sorted((study_root / "task_summaries/production/packs").glob("*.json"))
    pack_receipts = sorted((study_root / "transport_receipts/production/packs").glob("*.json"))
    if len(pack_summaries) != len(PACKS) or len(pack_receipts) != len(PACKS):
        raise ValueError(
            "Expected six pack summaries and receipts; got "
            f"summaries={len(pack_summaries)}, receipts={len(pack_receipts)}."
        )

    parent_job_ids: set[str] = set()
    task_job_ids: set[str] = set()
    for pack_index, logical_indices in enumerate(PACKS):
        semantic = _read_json(study_root / f"task_summaries/production/packs/{pack_index}.json")
        receipt = _read_json(study_root / f"transport_receipts/production/packs/{pack_index}.json")
        checks = {
            "one-row semantic summary": isinstance(semantic, list) and len(semantic) == 1,
            "semantic complete": isinstance(semantic, list)
            and len(semantic) == 1
            and semantic[0].get("status") == "complete",
            "semantic pack": isinstance(semantic, list)
            and len(semantic) == 1
            and int(semantic[0].get("pack_index", -1)) == pack_index,
            "semantic logical indices": isinstance(semantic, list)
            and len(semantic) == 1
            and tuple(semantic[0].get("logical_indices", [])) == logical_indices,
            "pack receipt complete": receipt.get("status") == "complete",
            "pack receipt mode": receipt.get("run_mode") == "production",
            "pack receipt index": int(receipt.get("pack_index", -1)) == pack_index,
            "pack receipt logical indices": tuple(receipt.get("logical_indices", []))
            == logical_indices,
            "two concurrent runs": receipt.get("concurrent_on_one_gpu") is True
            and int(receipt.get("run_count", -1)) == 2,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Pack {pack_index} failed guards: {failed!r}.")
        parent_job_ids.add(str(receipt.get("parent_job_id")))
        task_job_ids.add(str(receipt.get("job_id")))

    rows: list[dict[str, Any]] = []
    histories: dict[int, dict[str, list[float]]] = {}
    initial_hashes: dict[str, set[str]] = {architecture: set() for architecture in ARCHITECTURES}
    train_split_hashes: set[str] = set()
    validation_split_hashes: set[str] = set()
    batch_orders: set[tuple[str, ...]] = set()
    source_commits: set[str] = set()
    source_archives: set[str] = set()
    config_set_hashes: set[str] = set()

    for logical_index, (spec, config_path) in enumerate(
        zip(CASE_SPECS, config_paths, strict=True)
    ):
        config_sha = _sha256(config_path)
        run_dir = run_by_config.get(config_sha)
        if run_dir is None:
            raise ValueError(f"No production run matches config {config_path.name}.")
        errors = validate_run(run_dir)
        if errors:
            raise ValueError(f"Invalid canonical bundle {run_dir}: {errors!r}.")
        config = _read_json(config_path)
        manifest = _read_json(run_dir / "manifest.json")
        status = _read_json(run_dir / "status.json")
        result = _read_json(run_dir / "result.json")
        metrics = _read_json(run_dir / "metrics.json")
        run_config = _read_json(run_dir / "config.json")
        summary = _read_json(
            study_root / f"task_summaries/production/runs/{logical_index}.json"
        )
        receipt = _read_json(
            study_root / f"transport_receipts/production/runs/{logical_index}.json"
        )
        eqprop = metrics.get("eqprop", {})
        configured_eqprop = config.get("eqprop", {})
        provenance = metrics.get("dataset_provenance", {})
        runtime_base_beta, runtime_injected_beta = runtime_beta_values(
            run_config,
            depth=int(study["architectures"][spec.architecture]["depth"]),
        )
        train_accuracy = _history(run_dir, "accuracy_train.npy")
        validation_accuracy = _history(run_dir, "accuracy_test.npy")
        train_loss = _history(run_dir, "loss_train.npy")
        validation_loss = _history(run_dir, "loss_test.npy")
        checks = {
            "study": manifest.get("study_id") == STUDY_ID,
            "evidence": manifest.get("evidence_class") == EVIDENCE_CLASS,
            "arm": manifest.get("arm_id") == spec.arm_id,
            "complete": status.get("state") == "complete",
            "criteria": result.get("completion", {}).get("criteria_met") is True,
            "not smoke": manifest.get("smoke") is False,
            "ten epochs": int(manifest.get("configuration", {}).get("epochs", -1)) == 10,
            "algorithm": metrics.get("training_algorithm") == "EP",
            "dtype": metrics.get("runtime_dtype") == "float64",
            "optimizer step": metrics.get("optimizer_steps_applied") is True,
            "official test count": int(metrics.get("official_test_evaluations", -1)) == 0,
            "official test manifest": manifest.get("dataset", {}).get("official_test_read")
            is False,
            "centered": eqprop.get("variant") == "centered",
            "current": eqprop.get("nudging_mode") == "current",
            "normalized current": eqprop.get("normalize_current_scale") is True,
            "zero read noise": _same(eqprop.get("endpoint_read_noise_std"), 0.0),
            "zero read-noise draws": int(
                metrics.get("eqprop_endpoint_read_noise_draw_count", -1)
            )
            == 0,
            "input noise disabled": eqprop.get("input_read_noise") is False,
            "base beta": _same(config.get("beta"), runtime_base_beta),
            "injected beta": _same(
                configured_eqprop.get("injected_beta_B"),
                runtime_injected_beta,
            ),
            "beta tier": configured_eqprop.get("beta_tier") == spec.beta_tier,
            "summary one row": isinstance(summary, list) and len(summary) == 1,
            "summary complete": isinstance(summary, list)
            and len(summary) == 1
            and summary[0].get("status") == "complete",
            "summary config hash": isinstance(summary, list)
            and len(summary) == 1
            and summary[0].get("config_sha256") == config_sha,
            "receipt logical index": int(receipt.get("logical_index", -1))
            == logical_index,
            "receipt pack": int(receipt.get("pack_index", -1)) == logical_index // 2,
            "receipt semantic pass": receipt.get("semantic_pass") is True,
            "receipt source config": receipt.get("selected_config_sha256") == config_sha,
            "receipt config set": receipt.get("config_set_sha256")
            == expected_config_set_sha,
            "receipt manifest": receipt.get("manifest_sha256")
            == _sha256(run_dir / "manifest.json"),
            "receipt status": receipt.get("status_sha256")
            == _sha256(run_dir / "status.json"),
            "receipt metrics": receipt.get("metrics_sha256")
            == _sha256(run_dir / "metrics.jsonl"),
            "receipt result": receipt.get("result_sha256")
            == _sha256(run_dir / "result.json"),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Logical run {logical_index} failed guards: {failed!r}.")

        initial_hash = str(metrics.get("initial_parameter_state_sha256", ""))
        train_hash = str(provenance.get("train_indices_sha256", ""))
        validation_hash = str(provenance.get("validation_indices_sha256", ""))
        batch_order = tuple(
            str(value) for value in provenance.get("train_batch_order_sha256", [])
        )
        if not initial_hash or not train_hash or not validation_hash or len(batch_order) != 10:
            raise ValueError(f"Missing matched-cohort provenance in logical run {logical_index}.")
        initial_hashes[spec.architecture].add(initial_hash)
        train_split_hashes.add(train_hash)
        validation_split_hashes.add(validation_hash)
        batch_orders.add(batch_order)
        source_commits.add(str(receipt.get("source_commit")))
        source_archives.add(str(receipt.get("source_archive_sha256")))
        config_set_hashes.add(str(receipt.get("config_set_sha256")))
        histories[logical_index] = {
            "train_accuracy": train_accuracy.tolist(),
            "validation_accuracy": validation_accuracy.tolist(),
            "train_loss": train_loss.tolist(),
            "validation_loss": validation_loss.tolist(),
        }
        rows.append(
            {
                "logical_index": logical_index,
                "pack_index": logical_index // 2,
                "architecture": spec.architecture,
                "scheme": spec.scheme,
                "beta_tier": spec.beta_tier,
                "boundary_status": configured_eqprop.get("beta_boundary_status"),
                "boundary_injected_beta_B": configured_eqprop.get(
                    "beta_boundary_injected_B"
                ),
                "injected_beta_B": configured_eqprop.get("injected_beta_B"),
                "base_beta": config.get("beta"),
                "amplification_factor": configured_eqprop.get("amplification_factor"),
                "trajectory_status": "complete_finite_10_epochs",
                "best_epoch": int(metrics["best_epoch"]),
                "best_validation_accuracy": float(metrics["best_validation_accuracy"]),
                "final_validation_accuracy": float(validation_accuracy[-1]),
                "final_validation_loss": float(validation_loss[-1]),
                "final_train_accuracy": float(train_accuracy[-1]),
                "final_train_loss": float(train_loss[-1]),
                "minimum_validation_accuracy": float(np.min(validation_accuracy)),
                "maximum_validation_loss": float(np.max(validation_loss)),
                "initial_parameter_state_sha256": initial_hash,
                "train_indices_sha256": train_hash,
                "validation_indices_sha256": validation_hash,
                "run_dir": str(run_dir),
                "result_sha256": _sha256(run_dir / "result.json"),
            }
        )

    if any(len(values) != 1 for values in initial_hashes.values()):
        raise ValueError(f"Initialization was not matched within architecture: {initial_hashes!r}.")
    if len(train_split_hashes) != 1 or len(validation_split_hashes) != 1:
        raise ValueError("Train/validation split identity changed across cases.")
    if len(batch_orders) != 1:
        raise ValueError("Minibatch order changed across cases.")
    if len(parent_job_ids) != 1 or len(task_job_ids) != len(PACKS):
        raise ValueError("Production Slurm parent/task coverage is inconsistent.")
    if len(source_commits) != 1 or len(source_archives) != 1:
        raise ValueError("Frozen source identity changed across tasks.")
    if config_set_hashes != {expected_config_set_sha}:
        raise ValueError("Exact-config set identity changed across tasks.")

    comparisons: list[dict[str, Any]] = []
    by_identity = {
        (row["architecture"], row["scheme"], row["beta_tier"]): row for row in rows
    }
    for architecture in ARCHITECTURES:
        for scheme in SCHEMES:
            one = by_identity[(architecture, scheme, "one_decade_lower")]
            two = by_identity[(architecture, scheme, "two_decades_lower")]
            comparisons.append(
                {
                    "architecture": architecture,
                    "scheme": scheme,
                    "one_decade_injected_beta_B": one["injected_beta_B"],
                    "two_decades_injected_beta_B": two["injected_beta_B"],
                    "one_decade_best_validation_accuracy": one[
                        "best_validation_accuracy"
                    ],
                    "two_decades_best_validation_accuracy": two[
                        "best_validation_accuracy"
                    ],
                    "two_minus_one_best_validation_accuracy": two[
                        "best_validation_accuracy"
                    ]
                    - one["best_validation_accuracy"],
                    "one_decade_final_validation_accuracy": one[
                        "final_validation_accuracy"
                    ],
                    "two_decades_final_validation_accuracy": two[
                        "final_validation_accuracy"
                    ],
                    "two_minus_one_final_validation_accuracy": two[
                        "final_validation_accuracy"
                    ]
                    - one["final_validation_accuracy"],
                }
            )

    verification = {
        "schema_version": "conv13-eqprop-beta-qualification-verification/v1",
        "study_id": STUDY_ID,
        "status": "pass",
        "canonical_run_count": len(rows),
        "pack_count": len(PACKS),
        "runs_per_gpu": 2,
        "slurm_parent_job_id": next(iter(parent_job_ids)),
        "slurm_task_job_ids": sorted(task_job_ids),
        "source_commit": next(iter(source_commits)),
        "source_archive_sha256": next(iter(source_archives)),
        "config_set_sha256": expected_config_set_sha,
        "initial_parameter_state_sha256_by_architecture": {
            architecture: next(iter(values))
            for architecture, values in initial_hashes.items()
        },
        "train_indices_sha256": next(iter(train_split_hashes)),
        "validation_indices_sha256": next(iter(validation_split_hashes)),
        "train_batch_order_sha256": list(next(iter(batch_orders))),
        "official_test_read": False,
        "endpoint_read_noise_std": 0.0,
        "all_trajectories_complete_and_finite": True,
        "conv3_baseline_tk8_limitation": study["scientific_contract"][
            "conv3_tk8_residual_deviation"
        ],
    }
    return rows, {
        "verification": verification,
        "comparisons": comparisons,
        "histories": histories,
    }


def _plot(analysis_dir: Path, histories: Mapping[int, Mapping[str, Sequence[float]]]) -> None:
    tier_styles = {"one_decade_lower": "-", "two_decades_lower": "--"}
    colors = {"baseline": "#4C78A8", "ours": "#F58518", "legacy": "#54A24B"}
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True, sharey=True)
    for row_index, architecture in enumerate(ARCHITECTURES):
        for column_index, scheme in enumerate(SCHEMES):
            axis = axes[row_index, column_index]
            for spec in CASE_SPECS:
                if spec.architecture != architecture or spec.scheme != scheme:
                    continue
                values = histories[spec.index]["validation_accuracy"]
                axis.plot(
                    np.arange(1, 11),
                    values,
                    tier_styles[spec.beta_tier],
                    color=colors[scheme],
                    linewidth=2,
                    label=spec.beta_tier.replace("_", " "),
                )
            axis.set_title(f"{architecture.upper()} — {scheme}")
            axis.grid(alpha=0.25)
            if row_index == 1:
                axis.set_xlabel("Epoch")
            if column_index == 0:
                axis.set_ylabel("Validation accuracy")
            axis.legend(fontsize=8)
    fig.suptitle("Clean centered-float64 EqProp beta qualification (ordinary MNIST)")
    fig.tight_layout()
    fig.savefig(analysis_dir / "validation_accuracy_trajectories.png", dpi=180)
    plt.close(fig)


def analyze(study_root: Path, study_config: Path) -> dict[str, Any]:
    rows, payload = load_and_validate(study_root, study_config)
    study_root = study_root.expanduser().resolve()
    analysis_dir = study_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(analysis_dir / "summary.csv", rows)
    _write_csv(analysis_dir / "one_vs_two_decades.csv", payload["comparisons"])
    atomic_write_json(analysis_dir / "verification.json", payload["verification"])
    atomic_write_json(
        analysis_dir / "summary.json",
        {
            "schema_version": "conv13-eqprop-beta-qualification-analysis/v1",
            "study_id": STUDY_ID,
            "selection_status": "awaiting_user_review",
            "selection_performed": False,
            "cases": rows,
            "one_vs_two_decades": payload["comparisons"],
        },
    )
    _plot(analysis_dir, payload["histories"])

    table_lines = [
        "| Architecture | Scheme | Best val. (1 decade) | Best val. (2 decades) | 2−1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["comparisons"]:
        table_lines.append(
            "| {architecture} | {scheme} | {one:.6f} | {two:.6f} | {delta:+.6f} |".format(
                architecture=row["architecture"],
                scheme=row["scheme"],
                one=row["one_decade_best_validation_accuracy"],
                two=row["two_decades_best_validation_accuracy"],
                delta=row["two_minus_one_best_validation_accuracy"],
            )
        )
    report = "\n".join(
        [
            "# Conv1/Conv3 clean-beta qualification",
            "",
            "All twelve ordinary-MNIST diagnostic trajectories completed ten finite epochs with no official-test access and no endpoint read noise.",
            "",
            *table_lines,
            "",
            "No beta has been selected automatically. These measurements are staged for user review before any read-noise follow-up.",
            "",
            "The Conv3 baseline boundary remains cosine-only at T=K=8 because the shared free state was under-relaxed; interpret both baseline candidates with that declared limitation.",
            "",
        ]
    )
    (analysis_dir / "report.md").write_text(report, encoding="utf-8")
    return {
        "status": "complete",
        "study_root": str(study_root),
        "analysis_dir": str(analysis_dir),
        "case_count": len(rows),
        "selection_performed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT)
    parser.add_argument(
        "--study-config",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "configs/conv/perfectdiode_conv13_centered_float64_eqprop_"
            "beta_qualification_1to2decades_10ep_seed0_20260814_v1.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(analyze(args.study_root, args.study_config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
