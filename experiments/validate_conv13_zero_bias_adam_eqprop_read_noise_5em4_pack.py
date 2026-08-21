#!/usr/bin/env python3
"""Validate one packed Conv1/Conv3 Adam EqProp sigma-5e-4 task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.reporting import validate_run
from experiments.run_conv13_zero_bias_adam_eqprop_read_noise_5em4 import (
    CASE_SPECS,
    EVIDENCE_CLASS,
    NOISE_STD,
    PACKS,
    STUDY_ID,
    load_and_validate_study,
)
from experiments.validate_conv123_zero_bias_adam_eqprop_one_decade_pack import (
    _checkpoint_biases_are_zero,
    _epoch_rows,
    runtime_beta_values,
    scientific_outcome,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=1e-12, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def validate_pack(args: argparse.Namespace) -> dict[str, Any]:
    study = load_and_validate_study(args.study)
    contract = study["scientific_contract"]
    pack_index = int(args.pack_index)
    if not 0 <= pack_index < len(PACKS):
        raise ValueError(f"Pack index {pack_index} is outside [0,{len(PACKS) - 1}].")
    logical_indices = tuple(int(value) for value in args.logical_indices)
    if logical_indices != PACKS[pack_index]:
        raise ValueError(
            f"Pack {pack_index} must contain {PACKS[pack_index]}, got {logical_indices}."
        )
    if args.execution_mode != "concurrent":
        raise ValueError("Every Conv1/Conv3 pack must execute concurrently on one GPU.")

    expected_smoke = args.run_mode == "smoke"
    config_paths = sorted(args.config_root.expanduser().resolve().glob("*.json"))
    if len(config_paths) != len(CASE_SPECS):
        raise ValueError(f"Expected six configs, found {len(config_paths)}.")

    run_root = args.run_root.expanduser().resolve()
    summary_root = args.summary_root.expanduser().resolve()
    log_root = args.log_root.expanduser().resolve()
    receipt_root = args.receipt_root.expanduser().resolve()
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []

    for logical_index in logical_indices:
        spec = CASE_SPECS[logical_index]
        expected_epochs = (
            1 if expected_smoke else int(contract["epochs"][spec.architecture])
        )
        configured_epochs = int(contract["epochs"][spec.architecture])
        expected_t = int(contract["T"][spec.architecture])
        expected_k = int(contract["K"][spec.architecture])
        expected_draws = int(
            contract["expected_endpoint_read_noise_draws"][args.run_mode][
                spec.architecture
            ]
        )
        depth = int(contract["output_row_exponent_L"][spec.architecture])

        config_path = config_paths[logical_index]
        config = _read_json(config_path)
        summary_path = summary_root / f"{logical_index}.json"
        summary = _read_json(summary_path)
        if not isinstance(summary, list) or len(summary) != 1:
            raise ValueError(f"Expected one summary row in {summary_path}.")
        summary_row = summary[0]
        if int(summary_row.get("index", -1)) != logical_index:
            raise ValueError(f"Global exact-run index mismatch for {logical_index}.")
        if summary_row.get("config_sha256") != _sha256(config_path):
            raise ValueError(f"Config hash mismatch for logical run {logical_index}.")
        run_dir = Path(summary_row["output_dir"]).resolve()
        if not run_dir.is_relative_to(run_root):
            raise ValueError(f"Run {logical_index} escaped the declared result root.")
        errors = validate_run(run_dir)
        if errors:
            raise ValueError(f"Canonical validation failed for {run_dir}: {errors!r}")

        manifest = _read_json(run_dir / "manifest.json")
        status = _read_json(run_dir / "status.json")
        exact = _read_json(run_dir / "exact_run.json")
        run_config = _read_json(run_dir / "config.json")
        log_path = log_root / f"{logical_index}.log"
        if not log_path.is_file():
            raise ValueError(f"Missing packed run log {log_path}.")
        epoch_rows = _epoch_rows(run_dir / "metrics.jsonl")
        outcome = scientific_outcome(
            state=str(status.get("state")),
            summary_status=str(summary_row.get("status")),
            exact_returncode=exact.get("returncode"),
            log_path=log_path,
            epoch_rows=epoch_rows,
            expected_epochs=expected_epochs,
        )
        if expected_smoke and outcome["terminal_kind"] != "finite_complete":
            raise ValueError(
                f"Canary logical run {logical_index} did not complete finitely."
            )

        configured_eqprop = config.get("eqprop", {})
        runtime_base_beta, runtime_injected_beta = runtime_beta_values(
            run_config, depth=depth
        )
        training = run_config.get("training", {})
        runtime_eqprop = training.get("eqprop", {})
        dataset = run_config.get("dataset", {})
        provenance = dataset.get("provenance", {})
        architecture = run_config.get("architecture", {})
        initial_hash = str(architecture.get("initial_parameter_state_sha256", ""))
        train_orders = tuple(
            str(value) for value in provenance.get("train_batch_order_sha256", [])
        )
        parameter_order = list(config["parameter_order"])
        bias_names = [name for name in parameter_order if name.startswith("Bias_")]
        named_rates = config["learning_rates_by_parameter"]
        resolved_config = manifest.get("configuration", {}).get("resolved", {})
        resolved_rates = resolved_config.get("learning_rates_by_parameter", {})
        qualification = config.get("qualification_source", {})
        checks: dict[str, bool] = {
            "study": manifest.get("study_id") == STUDY_ID,
            "evidence": manifest.get("evidence_class") == EVIDENCE_CLASS,
            "arm": manifest.get("arm_id") == spec.arm_id,
            "smoke": bool(manifest.get("smoke")) == expected_smoke,
            "configured epochs": int(
                manifest.get("configuration", {}).get("configured_epochs", -1)
            )
            == configured_epochs,
            "runtime epochs": int(training.get("epochs", -1)) == expected_epochs,
            "algorithm": runtime_eqprop.get("variant") == "centered",
            "current nudge": runtime_eqprop.get("nudging_mode") == "current",
            "normalized current": runtime_eqprop.get("normalize_current_scale")
            is True,
            "dtype": training.get("runtime_dtype") == "float64",
            "optimizer step": training.get("optimizer_steps_applied") is True,
            "read noise config": _same(
                configured_eqprop.get("endpoint_read_noise_std"), NOISE_STD
            ),
            "read noise runtime": _same(
                runtime_eqprop.get("endpoint_read_noise_std"), NOISE_STD
            ),
            "noise seed config": int(
                configured_eqprop.get("endpoint_read_noise_seed", -1)
            )
            == int(contract["endpoint_read_noise_seed"]),
            "noise seed runtime": int(
                runtime_eqprop.get("endpoint_read_noise_seed", -1)
            )
            == int(contract["endpoint_read_noise_seed"]),
            "input noise disabled": runtime_eqprop.get("input_read_noise") is False,
            "independent phase noise": configured_eqprop.get(
                "phase_noise_correlation"
            )
            == "independent",
            "independent layer noise": configured_eqprop.get(
                "layer_noise_correlation"
            )
            == "independent",
            "injected beta": _same(
                configured_eqprop.get("injected_beta_B"), runtime_injected_beta
            ),
            "base beta": _same(config.get("beta"), runtime_base_beta),
            "beta tier": configured_eqprop.get("beta_tier")
            == "one_decade_lower",
            "optimizer": config.get("optimizer", {}).get("name") == "Adam",
            "T": int(training.get("num_iterations_inference", -1)) == expected_t,
            "K": int(training.get("num_iterations_training", -1)) == expected_k,
            "ordinary MNIST": dataset.get("factory")
            == "labs.datasets.MnistTrainValidationDataset",
            "official test": manifest.get("dataset", {}).get("official_test_read")
            is False,
            "initial hash": len(initial_hash) == 64,
            "train split hash": len(str(provenance.get("train_indices_sha256", "")))
            == 64,
            "validation split hash": len(
                str(provenance.get("validation_indices_sha256", ""))
            )
            == 64,
            "batch order": len(train_orders) == expected_epochs,
            "configured zero bias": all(
                float(named_rates[name]) == 0.0 for name in bias_names
            ),
            "resolved zero bias": all(
                float(resolved_rates.get(name, math.nan)) == 0.0
                for name in bias_names
            ),
            "qualification non-paper": qualification.get("paper_ready") is False,
            "gradient deviation retained": qualification.get(
                "gradient_gate_passed_rows"
            )
            == 213,
            "clean control id": qualification.get("clean_control_study_id")
            == contract["clean_control_study_id"],
            "clean control digest": qualification.get(
                "clean_control_result_sha256"
            )
            == study["source"]["clean_controls"]["result_sha256"][
                spec.architecture
            ][spec.scheme],
        }
        if outcome["terminal_kind"] == "finite_complete":
            metrics = _read_json(run_dir / "metrics.json")
            metric_eqprop = metrics.get("eqprop", {})
            checks.update(
                {
                    "result exists": (run_dir / "result.json").is_file(),
                    "metrics algorithm": metrics.get("training_algorithm") == "EP",
                    "metrics dtype": metrics.get("runtime_dtype") == "float64",
                    "metrics optimizer step": metrics.get(
                        "optimizer_steps_applied"
                    )
                    is True,
                    "metrics official test": int(
                        metrics.get("official_test_evaluations", -1)
                    )
                    == 0,
                    "metrics noise sigma": _same(
                        metric_eqprop.get("endpoint_read_noise_std"), NOISE_STD
                    ),
                    "metrics noise seed": int(
                        metric_eqprop.get("endpoint_read_noise_seed", -1)
                    )
                    == int(contract["endpoint_read_noise_seed"]),
                    "noise draws": int(
                        metrics.get("eqprop_endpoint_read_noise_draw_count", -1)
                    )
                    == expected_draws,
                    "best checkpoint zero bias": _checkpoint_biases_are_zero(
                        run_dir / "best_model.pt", bias_names
                    ),
                    "final checkpoint zero bias": _checkpoint_biases_are_zero(
                        run_dir / "final_model.pt", bias_names
                    ),
                }
            )
        else:
            checks.update(
                {
                    "failed no result": not (run_dir / "result.json").exists(),
                    "failed status": status.get("state") == "failed",
                }
            )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Run {logical_index} failed guards: {failed!r}")

        provenance_rows.append(
            {
                "architecture": spec.architecture,
                "initial_hash": initial_hash,
                "train_indices": str(provenance["train_indices_sha256"]),
                "validation_indices": str(provenance["validation_indices_sha256"]),
                "first_epoch": str(provenance["first_epoch_batch_order_sha256"]),
                "train_orders": train_orders,
            }
        )
        receipt = {
            "schema_version": "conv13-zero-bias-adam-eqprop-read-noise-run-receipt/v1",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "study_id": STUDY_ID,
            "evidence_class": EVIDENCE_CLASS,
            "run_mode": args.run_mode,
            "pack_index": pack_index,
            "logical_index": logical_index,
            "architecture": spec.architecture,
            "scheme": spec.scheme,
            "optimizer": "Adam",
            "endpoint_read_noise_std": NOISE_STD,
            "endpoint_read_noise_seed": int(contract["endpoint_read_noise_seed"]),
            "expected_endpoint_read_noise_draws": expected_draws,
            "job_id": args.job_id,
            "parent_job_id": args.parent_job_id,
            "environment_id": args.environment_id,
            "source_commit": args.source_commit,
            "source_archive_sha256": args.source_archive_sha256,
            "config_set_sha256": args.config_set_sha256,
            "selected_config": str(config_path),
            "selected_config_sha256": _sha256(config_path),
            "run_dir": str(run_dir),
            "manifest_sha256": _sha256(run_dir / "manifest.json"),
            "status_sha256": _sha256(run_dir / "status.json"),
            "metrics_jsonl_sha256": _sha256(run_dir / "metrics.jsonl"),
            "result_sha256": (
                _sha256(run_dir / "result.json")
                if (run_dir / "result.json").is_file()
                else None
            ),
            "initial_parameter_state_sha256": initial_hash,
            "train_indices_sha256": provenance["train_indices_sha256"],
            "validation_indices_sha256": provenance["validation_indices_sha256"],
            "train_batch_order_sha256": list(train_orders),
            "outcome": outcome,
            "semantic_pass": True,
            "official_test_read": False,
            "protocol_deviations_retained": study["protocol_deviations"],
        }
        receipt_path = receipt_root / f"{logical_index}.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        receipts.append(receipt)

    for key in ("train_indices", "validation_indices", "first_epoch"):
        if len({row[key] for row in provenance_rows}) != 1:
            raise ValueError(f"Dataset cohort mismatch within pack for {key}.")
    common_epoch_count = min(len(row["train_orders"]) for row in provenance_rows)
    if len(
        {tuple(row["train_orders"][:common_epoch_count]) for row in provenance_rows}
    ) != 1:
        raise ValueError("Minibatch-order common prefix differs within the pack.")

    summary_status = "complete" if expected_smoke else "scientific_terminal"
    pack_receipt = {
        "schema_version": "conv13-zero-bias-adam-eqprop-read-noise-pack-receipt/v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "study_id": STUDY_ID,
        "status": summary_status,
        "run_mode": args.run_mode,
        "pack_index": pack_index,
        "logical_indices": list(logical_indices),
        "run_count": len(receipts),
        "execution_mode": args.execution_mode,
        "concurrent_on_one_gpu": True,
        "job_id": args.job_id,
        "parent_job_id": args.parent_job_id,
        "runs": receipts,
    }
    pack_path = args.pack_receipt.expanduser().resolve()
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path.write_text(
        json.dumps(pack_receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    semantic_summary = [
        {
            "status": summary_status,
            "study_id": STUDY_ID,
            "run_mode": args.run_mode,
            "pack_index": pack_index,
            "logical_indices": list(logical_indices),
            "run_count": len(receipts),
            "stable_count": sum(row["outcome"]["stable"] for row in receipts),
            "scientific_nonfinite_count": sum(
                row["outcome"]["terminal_kind"] == "nonfinite_training"
                for row in receipts
            ),
            "execution_mode": args.execution_mode,
            "concurrent_on_one_gpu": True,
            "pack_receipt": str(pack_path),
            "pack_receipt_sha256": _sha256(pack_path),
        }
    ]
    semantic_path = args.semantic_summary.expanduser().resolve()
    semantic_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_path.write_text(
        json.dumps(semantic_summary, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return semantic_summary[0]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--pack-receipt", type=Path, required=True)
    parser.add_argument("--semantic-summary", type=Path, required=True)
    parser.add_argument("--pack-index", type=int, required=True)
    parser.add_argument("--logical-indices", type=int, nargs="+", required=True)
    parser.add_argument("--run-mode", choices=("smoke", "production"), required=True)
    parser.add_argument("--execution-mode", choices=("concurrent",), required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--parent-job-id", required=True)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--config-set-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = validate_pack(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
