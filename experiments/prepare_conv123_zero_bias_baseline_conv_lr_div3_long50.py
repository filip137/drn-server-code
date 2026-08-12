from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = (
    REPO_ROOT
    / "configs/conv/"
    "perfectdiode_conv123_zero_bias_lr_counterfactual_ordinary_mnist_"
    "seed0_20260807_v1"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "configs/conv/"
    "perfectdiode_conv123_zero_bias_baseline_conv_lr_div3_long50_"
    "ordinary_mnist_seed0_20260808_v1"
)
STUDY_ID = (
    "perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-long50-"
    "ordinary-mnist-seed0-20260808-v1"
)
EVIDENCE_CLASS = "ordinary_mnist_lr_duration_counterfactual"
EPOCHS = 50
CONV_LR_DIVISOR = 3.0
ARCHITECTURES = ("conv1", "conv2", "conv3")
PARENT_FILENAME = "00_baseline_sgd_current.json"
OPERATING_POINTS = {
    "conv1": {"input_gain": 40.0, "T": 4, "K": 4},
    "conv2": {"input_gain": 100.0, "T": 6, "K": 6},
    "conv3": {"input_gain": 360.0, "T": 8, "K": 8},
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _arm_id(architecture: str) -> str:
    return (
        f"{architecture}_baseline_sgd_bias_zero_conv_lr_div3_"
        "long50_seed0"
    )


def _build_config(
    architecture: str,
) -> tuple[bytes, dict[str, Any]]:
    parent_path = PARENT_ROOT / architecture / PARENT_FILENAME
    parent_bytes = parent_path.read_bytes()
    parent = json.loads(parent_bytes)
    config = deepcopy(parent)
    parameter_order = config["parameter_order"]
    parent_rates = config["learning_rates_by_parameter"]

    target_rates = {
        name: (
            float(parent_rates[name]) / CONV_LR_DIVISOR
            if name.startswith("ConvWeight_")
            else float(parent_rates[name])
        )
        for name in parameter_order
    }
    conv_parameters = [
        name for name in parameter_order if name.startswith("ConvWeight_")
    ]
    dense_parameters = [
        name for name in parameter_order if name.startswith("DenseWeight_")
    ]
    bias_parameters = [
        name for name in parameter_order if name.startswith("Bias_")
    ]
    arm_id = _arm_id(architecture)

    config["arm_id"] = arm_id
    config["study_id"] = STUDY_ID
    config["lab"]["epochs"] = EPOCHS
    config["learning_rates_by_parameter"] = target_rates
    config["lr"] = [target_rates[name] for name in parameter_order]
    config["optimizer"]["learning_rate"] = list(config["lr"])
    config["reporting"] = {
        "arm_id": arm_id,
        "evidence_class": EVIDENCE_CLASS,
        "paper_facing": False,
        "study_id": STUDY_ID,
    }
    config["evaluation"]["inclusion_rule"] = (
        "Include this predeclared ordinary-MNIST LR-duration counterfactual "
        "arm only if it completes all 50 epochs with finite metrics, all 51 "
        "model and optimizer checkpoints and all 250 traced optimizer "
        "transitions are complete, every hidden-bias learning rate and "
        "checkpoint tensor is exactly zero, the result bundle validates, "
        "and the official test is not read."
    )

    diagnostic = config["handoff"]["diagnostic_bias_ablation"]
    diagnostic["all_weight_learning_rates_unchanged"] = False
    diagnostic["changed_weight_parameters_csv"] = ",".join(conv_parameters)
    diagnostic["unchanged_weight_parameters_csv"] = ",".join(dense_parameters)
    diagnostic["scientific_question"] = (
        "Does baseline merely need smaller convolutional learning rates and "
        "more optimization time? Compare the current-rate trajectory with a "
        "three-times-smaller Conv-LR, dense-rate-matched 50-epoch run."
    )
    config["handoff"].pop("lr_counterfactual", None)
    config["handoff"]["lr_duration_counterfactual"] = {
        "architecture": architecture,
        "parent_config": _relative(parent_path),
        "parent_config_sha256": _sha256(parent_bytes),
        "role": "baseline_conv_lr_div3_long50",
        "rate_intervention": {
            "conv_learning_rate_divisor": CONV_LR_DIVISOR,
            "formula": "eta_conv_target = eta_conv_parent / 3",
            "changed_parameters": conv_parameters,
            "unchanged_parameters": dense_parameters + bias_parameters,
            "parent_rates_by_parameter": {
                name: float(parent_rates[name]) for name in parameter_order
            },
            "target_rates_by_parameter": target_rates,
        },
        "duration_intervention": {
            "parent_epochs": int(parent["lab"]["epochs"]),
            "target_epochs": EPOCHS,
        },
        "rates_selected_post_hoc_from_this_study": False,
        "weight_rates_match_parent": False,
    }

    payload = _json_bytes(config)
    row = {
        "architecture": architecture,
        "arm_id": arm_id,
        "config": (
            f"{_relative(OUTPUT_ROOT)}/{architecture}/"
            "00_baseline_sgd_conv_lr_div3_long50.json"
        ),
        "config_sha256": _sha256(payload),
        "epoch_budget": EPOCHS,
        "optimizer": "SGD",
        "parent_config": _relative(parent_path),
        "parent_config_sha256": _sha256(parent_bytes),
        "parent_rates_by_parameter": {
            name: float(parent_rates[name]) for name in parameter_order
        },
        "parameter_order": parameter_order,
        "rate_intervention": {
            "conv_learning_rate_divisor": CONV_LR_DIVISOR,
            "dense_learning_rates_unchanged": True,
            "bias_learning_rates": {
                name: target_rates[name] for name in bias_parameters
            },
        },
        "role": "baseline_conv_lr_div3_long50",
        "scheme": "baseline",
        "target_rates_by_parameter": target_rates,
    }
    return payload, row


def _readme_bytes() -> bytes:
    text = f"""# Conv1/2/3 zero-bias baseline Conv-LR /3 long follow-up

This directory contains three exact ordinary-MNIST diagnostic configs:
one baseline-SGD arm for each of Conv1, Conv2, and Conv3.

The intervention mirrors the earlier Conv-LR x3 stress arm in the opposite
direction:

- divide every `ConvWeight_*` learning rate by exactly `3`;
- keep `DenseWeight_0` at its current baseline learning rate;
- keep every `Bias_*` learning rate exactly `0`; and
- increase the training budget from 30 to {EPOCHS} epochs.

All seeds, data split and order, architecture, amplification, initialization,
preprocessing, weight projection, optimizer, and accepted operating points
remain unchanged. The official test split is disabled. These runs are
ordinary-MNIST optimization diagnostics, not paper-facing evidence.

Materialize or verify the deterministic files with:

```bash
python -m experiments.prepare_conv123_zero_bias_baseline_conv_lr_div3_long50
python -m experiments.prepare_conv123_zero_bias_baseline_conv_lr_div3_long50 --check
```

Launch transport is
`experiments/run_conv123_zero_bias_baseline_conv_lr_div3_long50_jeanzay.slurm`.
"""
    return text.encode("utf-8")


def _build_outputs() -> dict[Path, bytes]:
    outputs: dict[Path, bytes] = {}
    rows: list[dict[str, Any]] = []
    config_digests: list[str] = []
    for index, architecture in enumerate(ARCHITECTURES):
        payload, row = _build_config(architecture)
        row["global_index"] = index
        path = (
            OUTPUT_ROOT
            / architecture
            / "00_baseline_sgd_conv_lr_div3_long50.json"
        )
        outputs[path] = payload
        rows.append(row)
        config_digests.append(row["config_sha256"])

    digest_payload = "".join(f"{digest}\n" for digest in config_digests)
    manifest = {
        "axes": {
            "architectures": list(ARCHITECTURES),
            "optimizer": ["SGD"],
            "scheme": ["baseline"],
            "seed": [0],
        },
        "bias_contract": {
            "initialization": "zero",
            "learning_rate": 0.0,
            "required_checkpoint_value": 0.0,
        },
        "configs": rows,
        "created_at": "2026-08-08",
        "dataset": {
            "affine_corruption": False,
            "name": "MNIST",
            "official_test_read": False,
            "source_split": "train",
            "train_validation_split": "deterministic_stratified_55000_5000",
            "variant": "ordinary",
        },
        "diagnostics": {
            "checkpoint_every_epoch": True,
            "expected_checkpoint_epochs": list(range(EPOCHS + 1)),
            "expected_gradient_trace_transitions": EPOCHS * 5,
            "gradient_trace_samples_per_epoch": 5,
        },
        "epoch_budget": {architecture: EPOCHS for architecture in ARCHITECTURES},
        "evidence_class": EVIDENCE_CLASS,
        "launch_plan": {
            "account": "fmu@v100",
            "array": "0-2%3",
            "constraint": "v100-16g",
            "cpus_per_task": 10,
            "expected_compute_wall_hours_at_concurrency_three": 7,
            "expected_gpu_hours_by_architecture": {
                "conv1": 2,
                "conv2": 4,
                "conv3": 7,
            },
            "expected_gpu_hours_total": 13,
            "gpus_per_task": 1,
            "module": "pytorch-gpu/py3/2.5.0",
            "partition": "gpu_p13",
            "qos": "qos_gpu-t3",
            "remote_result_pattern": (
                "/lustre/fsn1/projects/rech/fmu/$USER/server_code/results/"
                + STUDY_ID
            ),
            "status": "prepared_not_submitted",
            "target": "jean-zay",
            "time_limit": "10:00:00",
            "wrapper": (
                "experiments/"
                "run_conv123_zero_bias_baseline_conv_lr_div3_long50_"
                "jeanzay.slurm"
            ),
        },
        "operating_points": OPERATING_POINTS,
        "ordered_config_set_sha256": _sha256(digest_payload.encode("utf-8")),
        "paper_facing": False,
        "rate_intervention": {
            "conv_learning_rate_divisor": CONV_LR_DIVISOR,
            "dense_learning_rates_unchanged": True,
            "bias_learning_rates_unchanged_at_zero": True,
        },
        "schema_version": (
            "perfectdiode-conv123-zero-bias-baseline-conv-lr-div3-"
            "long50-study/v1"
        ),
        "scientific_question": (
            "Can zero-bias baseline SGD improve materially when convolutional "
            "learning rates are three times smaller and the optimization "
            "budget is extended to 50 epochs?"
        ),
        "study_id": STUDY_ID,
        "weight_contract": {"projection": [0.0, 100.0]},
    }
    outputs[OUTPUT_ROOT / "study_manifest.json"] = _json_bytes(manifest)
    outputs[OUTPUT_ROOT / "README.md"] = _readme_bytes()
    return outputs


def prepare(*, check: bool) -> None:
    outputs = _build_outputs()
    if check:
        mismatches = [
            _relative(path)
            for path, expected in outputs.items()
            if not path.is_file() or path.read_bytes() != expected
        ]
        if mismatches:
            raise SystemExit(
                "Materialized Conv1/2/3 Conv-LR /3 long50 files are stale or "
                "missing: " + ", ".join(mismatches)
            )
        return

    for path, payload in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the exact Conv1/2/3 zero-bias baseline Conv-LR /3 "
            "50-epoch follow-up."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated files differ from the deterministic build.",
    )
    args = parser.parse_args()
    prepare(check=args.check)


if __name__ == "__main__":
    main()
