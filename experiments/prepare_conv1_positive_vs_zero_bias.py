#!/usr/bin/env python3
"""Materialize the matched Conv1 positive-only versus zero-bias configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = (
    REPO_ROOT
    / "configs/conv/perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1"
    / "conv1/00_baseline_sgd_bias_zero_seed0.json"
)
STUDY_ID = (
    "perfectdiode-conv1-positive-vs-zero-bias-ordinary-mnist-"
    "seed0-20260815-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_bias_ablation"
OUTPUT_ROOT = REPO_ROOT / "configs/conv" / STUDY_ID
PARAMETER_ORDER = ("ConvWeight_0", "DenseWeight_0", "Bias_0")
WEIGHT_RATES = {"ConvWeight_0": 0.142696, "DenseWeight_0": 0.0120238}
PRACTICAL_SIMILARITY_THRESHOLD_PP = 0.5


@dataclass(frozen=True)
class Case:
    role: str
    filename: str
    arm_id: str
    bias_learning_rate: float


CASES = (
    Case(
        role="positive_only_bias",
        filename="00_baseline_sgd_positive_bias_seed0.json",
        arm_id="conv1_baseline_sgd_positive_only_bias_seed0",
        bias_learning_rate=0.142696,
    ),
    Case(
        role="zero_bias",
        filename="01_baseline_sgd_zero_bias_seed0.json",
        arm_id="conv1_baseline_sgd_zero_bias_seed0",
        bias_learning_rate=0.0,
    ),
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_source(config: dict[str, Any]) -> None:
    if tuple(config.get("parameter_order", ())) != PARAMETER_ORDER:
        raise ValueError("Unexpected Conv1 parameter order in source config.")
    if config.get("training_algorithm") != "BP":
        raise ValueError("Expected BPTT/BP training in source config.")
    if config.get("optimizer", {}).get("name") != "SGD":
        raise ValueError("Expected plain SGD in source config.")
    if config.get("seed") != 0 or config.get("lab", {}).get("epochs") != 10:
        raise ValueError("Expected the accepted seed-0 ten-epoch Conv1 config.")
    model = config.get("model_base", {})
    if (
        model.get("non_linearity") != "perfect_diode"
        or float(model.get("voltage_amp")) != 1.0
        or float(model.get("current_amp")) != 1.0
        or int(model.get("num_iterations_inference")) != 4
        or int(model.get("num_iterations_training")) != 4
    ):
        raise ValueError("Unexpected perfect-diode baseline operating point.")
    if config.get("evaluation", {}).get("official_test", {}).get("policy") != "disabled":
        raise ValueError("Source config must disable the official test split.")
    rates = config.get("learning_rates_by_parameter", {})
    if any(float(rates.get(name, -1.0)) != value for name, value in WEIGHT_RATES.items()):
        raise ValueError("Source weight learning rates changed.")
    if float(rates.get("Bias_0", -1.0)) != 0.0:
        raise ValueError("Expected the source Bias_0 learning rate to be zero.")


def _case_payload(source: dict[str, Any], case: Case) -> bytes:
    config = deepcopy(source)
    rates = {**WEIGHT_RATES, "Bias_0": case.bias_learning_rate}
    ordered_rates = [rates[name] for name in PARAMETER_ORDER]

    config["arm_id"] = case.arm_id
    config["study_id"] = STUDY_ID
    config["lr"] = ordered_rates
    config["learning_rates_by_parameter"] = rates
    config["optimizer"]["learning_rate"] = ordered_rates
    config["datasets"]["mnist"]["params"]["root"] = "/home/filiposana/data"
    config["bias_contract"] = {
        "implementation": "historical_positive_only_nonnegative_projection",
        "initialization": "exact_zero",
        "learning_rate": case.bias_learning_rate,
        "role": case.role,
    }
    config["evaluation"]["inclusion_rule"] = (
        "Include this predeclared seed-0 ordinary-MNIST bias-ablation arm only "
        "if it completes all 10 epochs with finite metrics, initialization and "
        "cohort/order guards match its paired arm, every Bias_0 checkpoint is "
        + (
            "nonnegative and at least one selected/final Bias_0 value is positive"
            if case.bias_learning_rate > 0.0
            else "exactly zero"
        )
        + ", its canonical bundle validates, and the official test is not read."
    )
    config["handoff"]["diagnostic_bias_ablation"] = {
        "all_nonbias_scientific_fields_matched": True,
        "bias_initialization": "zero",
        "bias_learning_rate": case.bias_learning_rate,
        "comparison_role": case.role,
        "dataset_variant": "ordinary_mnist_train_validation",
        "parent_config": str(SOURCE_CONFIG.relative_to(REPO_ROOT)),
        "parent_config_sha256": _sha256(SOURCE_CONFIG.read_bytes()),
        "practical_similarity_threshold_pp": PRACTICAL_SIMILARITY_THRESHOLD_PP,
        "scientific_question": (
            "Does the current positive-only hidden-bias implementation materially "
            "change seed-0 baseline Conv1 ordinary-MNIST validation performance "
            "relative to an otherwise identical run with Bias_0 fixed at zero?"
        ),
    }
    config["reporting"] = {
        "arm_id": case.arm_id,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
        "study_id": STUDY_ID,
    }
    return _json_bytes(config)


def _readme_bytes() -> bytes:
    return (
        "# Conv1 positive-only versus zero-bias diagnostic\n\n"
        "Two matched seed-0 baseline perfect-diode Conv1 SGD arms run for ten "
        "epochs on ordinary MNIST. Both start from the same zero bias and use "
        "the accepted Conv/Dense learning rates; only `Bias_0` learning rate "
        "differs (`0.142696` versus `0`). The official test is disabled.\n\n"
        "The predeclared practical-similarity threshold is an absolute best-"
        "validation-accuracy difference of at most `0.5` percentage points. "
        "This is a single-seed diagnostic, not paper-facing accuracy evidence.\n"
    ).encode("utf-8")


def _materialized_files() -> dict[Path, bytes]:
    source_bytes = SOURCE_CONFIG.read_bytes()
    source = json.loads(source_bytes)
    _validate_source(source)

    files: dict[Path, bytes] = {}
    rows = []
    for case in CASES:
        payload = _case_payload(source, case)
        path = OUTPUT_ROOT / case.filename
        files[path] = payload
        rows.append(
            {
                "arm_id": case.arm_id,
                "bias_learning_rate": case.bias_learning_rate,
                "config": str(path.relative_to(REPO_ROOT)),
                "config_sha256": _sha256(payload),
                "role": case.role,
            }
        )

    manifest = {
        "schema_version": "conv1-positive-vs-zero-bias-study/v1",
        "study_id": STUDY_ID,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
        "expected_run_count": 2,
        "source_config": str(SOURCE_CONFIG.relative_to(REPO_ROOT)),
        "source_config_sha256": _sha256(source_bytes),
        "scientific_contract": {
            "architecture": "conv1",
            "scheme": "baseline",
            "optimizer": "SGD",
            "seed": 0,
            "dataset": "ordinary_mnist_55000_train_5000_validation",
            "epochs": 10,
            "voltage_amp": 1.0,
            "current_amp": 1.0,
            "inference_iterations": 4,
            "training_iterations": 4,
            "input_gain": 40.0,
            "weight_interval": [0.0, 100.0],
            "parameter_order": list(PARAMETER_ORDER),
            "weight_learning_rates": WEIGHT_RATES,
            "official_test_policy": "disabled",
            "practical_similarity_threshold_pp": PRACTICAL_SIMILARITY_THRESHOLD_PP,
        },
        "runs": rows,
    }
    files[OUTPUT_ROOT / "study_manifest.json"] = _json_bytes(manifest)
    files[OUTPUT_ROOT / "README.md"] = _readme_bytes()
    return files


def prepare(*, check: bool = False) -> dict[str, str]:
    files = _materialized_files()
    for path, payload in files.items():
        if check:
            if not path.is_file() or path.read_bytes() != payload:
                raise ValueError(f"Materialized file is missing or stale: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    return {str(path.relative_to(REPO_ROOT)): _sha256(payload) for path, payload in files.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(prepare(check=args.check), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
