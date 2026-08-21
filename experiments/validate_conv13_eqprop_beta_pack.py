#!/usr/bin/env python3
"""Validate a packed Conv1/Conv3 Jean Zay task and write its receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.reporting import validate_run
from experiments.run_conv13_centered_eqprop_beta_qualification import (
    CASE_SPECS,
    EVIDENCE_CLASS,
    PACKS,
    STUDY_ID,
    load_and_validate_study,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _same(left: Any, right: float) -> bool:
    try:
        return math.isclose(float(left), right, rel_tol=1e-12, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def runtime_beta_values(
    run_config: Mapping[str, Any], *, depth: int
) -> tuple[float, float]:
    """Recover the base and injected beta from the trainer's runtime config."""
    training = run_config.get("training", {})
    amplification = run_config.get("amplification", {})
    base_beta = float(training["beta"])
    voltage_amp = float(amplification["voltage_amp"])
    current_amp = float(amplification["current_amp"])
    if current_amp == 0.0:
        raise ValueError("Runtime current amplification must be nonzero.")
    return base_beta, base_beta * (voltage_amp / current_amp) ** int(depth)


def validate_pack(args: argparse.Namespace) -> dict[str, Any]:
    study = load_and_validate_study(args.study)
    pack_index = int(args.pack_index)
    if not 0 <= pack_index < len(PACKS):
        raise ValueError(f"Pack index {pack_index} is outside [0,{len(PACKS) - 1}].")
    logical_indices = tuple(int(value) for value in args.logical_indices)
    if logical_indices != PACKS[pack_index]:
        raise ValueError(
            f"Pack {pack_index} must contain {PACKS[pack_index]}, got {logical_indices}."
        )
    expected_epochs = 1 if args.run_mode == "canary" else 10
    expected_smoke = args.run_mode == "canary"
    config_paths = sorted(args.config_root.expanduser().resolve().glob("*.json"))
    if len(config_paths) != len(CASE_SPECS):
        raise ValueError(f"Expected twelve configs, found {len(config_paths)}.")

    run_root = args.run_root.expanduser().resolve()
    summary_root = args.summary_root.expanduser().resolve()
    receipt_root = args.receipt_root.expanduser().resolve()
    receipt_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for logical_index in logical_indices:
        spec = CASE_SPECS[logical_index]
        config_path = config_paths[logical_index]
        config = _read_json(config_path)
        summary_path = summary_root / f"{logical_index}.json"
        summary = _read_json(summary_path)
        if not isinstance(summary, list) or len(summary) != 1:
            raise ValueError(f"Expected one summary row in {summary_path}.")
        summary_row = summary[0]
        if summary_row.get("status") != "complete":
            raise ValueError(f"Run {logical_index} did not complete: {summary_row!r}")
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
        result = _read_json(run_dir / "result.json")
        metrics = _read_json(run_dir / "metrics.json")
        run_config = _read_json(run_dir / "config.json")
        eqprop = metrics.get("eqprop", {})
        configured_eqprop = config.get("eqprop", {})
        runtime_base_beta, runtime_injected_beta = runtime_beta_values(
            run_config,
            depth=int(study["architectures"][spec.architecture]["depth"]),
        )
        checks: dict[str, bool] = {
            "study": manifest.get("study_id") == STUDY_ID,
            "evidence": manifest.get("evidence_class") == EVIDENCE_CLASS,
            "arm": manifest.get("arm_id") == spec.arm_id,
            "complete": status.get("state") == "complete",
            "criteria": result.get("completion", {}).get("criteria_met") is True,
            "smoke": bool(manifest.get("smoke")) == expected_smoke,
            "epochs": int(manifest.get("configuration", {}).get("epochs", -1))
            == expected_epochs,
            "algorithm": metrics.get("training_algorithm") == "EP",
            "dtype": metrics.get("runtime_dtype") == "float64",
            "optimizer step": metrics.get("optimizer_steps_applied") is True,
            "official test": int(metrics.get("official_test_evaluations", -1)) == 0,
            "manifest official test": manifest.get("dataset", {}).get(
                "official_test_read"
            )
            is False,
            "centered": eqprop.get("variant") == "centered",
            "current nudge": eqprop.get("nudging_mode") == "current",
            "normalized current": eqprop.get("normalize_current_scale") is True,
            "zero read noise": _same(eqprop.get("endpoint_read_noise_std"), 0.0),
            "zero read-noise draws": int(
                metrics.get("eqprop_endpoint_read_noise_draw_count", -1)
            )
            == 0,
            "input noise disabled": eqprop.get("input_read_noise") is False,
            "injected beta": _same(
                configured_eqprop.get("injected_beta_B"),
                runtime_injected_beta,
            ),
            "base beta": _same(config.get("beta"), runtime_base_beta),
            "beta tier": configured_eqprop.get("beta_tier") == spec.beta_tier,
            "T": int(config.get("model_base", {}).get("num_iterations_inference", -1))
            == 8,
            "K": int(config.get("model_base", {}).get("num_iterations_training", -1))
            == 8,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f"Run {logical_index} failed guards: {failed!r}")

        receipt = {
            "schema_version": "conv13-eqprop-beta-jean-zay-run-receipt/v1",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "study_id": STUDY_ID,
            "evidence_class": EVIDENCE_CLASS,
            "run_mode": args.run_mode,
            "pack_index": pack_index,
            "logical_index": logical_index,
            "architecture": spec.architecture,
            "scheme": spec.scheme,
            "beta_tier": spec.beta_tier,
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
            "metrics_sha256": _sha256(run_dir / "metrics.jsonl"),
            "result_sha256": _sha256(run_dir / "result.json"),
            "semantic_pass": True,
            "official_test_read": False,
        }
        receipt_path = receipt_root / f"{logical_index}.json"
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        rows.append(receipt)

    pack_receipt = {
        "schema_version": "conv13-eqprop-beta-jean-zay-pack-receipt/v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "study_id": STUDY_ID,
        "status": "complete",
        "run_mode": args.run_mode,
        "pack_index": pack_index,
        "logical_indices": list(logical_indices),
        "run_count": len(rows),
        "concurrent_on_one_gpu": True,
        "job_id": args.job_id,
        "parent_job_id": args.parent_job_id,
        "runs": rows,
    }
    args.pack_receipt.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.pack_receipt.expanduser().resolve().write_text(
        json.dumps(pack_receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    semantic_summary = [
        {
            "status": "complete",
            "study_id": STUDY_ID,
            "run_mode": args.run_mode,
            "pack_index": pack_index,
            "logical_indices": list(logical_indices),
            "run_count": len(rows),
            "concurrent_on_one_gpu": True,
            "pack_receipt": str(args.pack_receipt.expanduser().resolve()),
            "pack_receipt_sha256": _sha256(args.pack_receipt.expanduser().resolve()),
        }
    ]
    args.semantic_summary.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.semantic_summary.expanduser().resolve().write_text(
        json.dumps(semantic_summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"pack": pack_receipt, "semantic_summary": semantic_summary}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--pack-receipt", type=Path, required=True)
    parser.add_argument("--semantic-summary", type=Path, required=True)
    parser.add_argument("--pack-index", type=int, required=True)
    parser.add_argument("--logical-indices", type=int, nargs=2, required=True)
    parser.add_argument("--run-mode", choices=("canary", "production"), required=True)
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
