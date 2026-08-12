#!/usr/bin/env python3
"""Run focused bounded perfect-diode Conv1/Conv2/Conv3 rho studies.

The runner preserves the historical focused baseline/ours studies and also
supports the all-depth bounded-uniform, zero-bias study. It materializes shared
initialization checkpoints, runs the optimizer-independent fixed-T/K security
checks, then drives each surface's declared core, optional one-wave expansion,
and terminal selector through :mod:`experiments.rho_search`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from experiments.rho_search import (
    _float_token,
    _git_state,
    _run_trainer,
    _write_json,
    linear_quantile,
    run as run_rho_search,
    verify_zero_bias_checkpoint,
    verify_zero_bias_run_checkpoints,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv12_bounded_rho_baseline_ours_20260729_v1.json"
)
DEFAULT_MINIMIZER = REPO_ROOT / "labs" / "configs" / "mnist_minimizer_fixed_iterations.json"
CONV12_STUDY_SCHEMA = "perfectdiode-conv12-bounded-rho-study/v1"
CONV3_STUDY_SCHEMA = "perfectdiode-conv3-bounded-rho-study/v1"
CONV123_ZERO_BIAS_STUDY_SCHEMA = (
    "perfectdiode-conv123-bounded-uniform-zero-bias-rho-study/v1"
)
CONV3_ZERO_BIAS_STUDY_SCHEMA = (
    "perfectdiode-conv3-bounded-uniform-zero-bias-rho-study/v1"
)
CONV3_ZERO_BIAS_INITIALIZER_SHA256 = (
    "8ebe9dd916e1e9299c31053e2bb37b4f5121f8322f26af7746a92875e555438f"
)
CONV3_ZERO_BIAS_PARENT_STUDY_ID = (
    "perfectdiode-bounded-uniform-zero-bias-lr-search-conv123-seed0-20260810-v1"
)


def _study_variant(study: Mapping[str, Any]) -> str:
    schema = study.get("schema_version")
    if schema == CONV12_STUDY_SCHEMA:
        return "conv12"
    if schema == CONV3_STUDY_SCHEMA:
        return "conv3"
    if schema == CONV123_ZERO_BIAS_STUDY_SCHEMA:
        return "conv123_zero_bias"
    if schema == CONV3_ZERO_BIAS_STUDY_SCHEMA:
        return "conv3_zero_bias"
    raise ValueError(f"Unexpected study schema: {schema!r}.")


def _is_zero_bias_study(study: Mapping[str, Any]) -> bool:
    return _study_variant(study) in {"conv123_zero_bias", "conv3_zero_bias"}


def _artifact_schema(study: Mapping[str, Any], artifact: str) -> str:
    variant = _study_variant(study)
    if variant == "conv123_zero_bias":
        return (
            "perfectdiode-conv123-bounded-uniform-zero-bias-rho-"
            f"{artifact}/v1"
        )
    if variant == "conv3_zero_bias":
        return (
            "perfectdiode-conv3-bounded-uniform-zero-bias-rho-"
            f"{artifact}/v1"
        )
    if variant == "conv12":
        legacy = {
            "bounded-init-asset": "perfectdiode-conv12-bounded-init-asset/v1",
            "fixed-tk-security": "perfectdiode-conv12-fixed-tk-security/v1",
        }
        if artifact in legacy:
            return legacy[artifact]
    return f"perfectdiode-{variant}-bounded-rho-{artifact}/v1"


def _surface_search(
    study: Mapping[str, Any], surface: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve optional core and safety policy overrides for one surface."""

    search = copy.deepcopy(dict(study["rho_search"]))
    policies = search.pop("core_policy_by_surface", None)
    if policies is not None:
        key = f"{surface['architecture']}__{surface['scheme']}"
        policy = policies.get(key)
        if not isinstance(policy, Mapping):
            raise ValueError(f"Missing rho-search core policy for {key!r}.")
        search.update(copy.deepcopy(dict(policy)))

    safety_by_architecture = search.pop("safety_by_architecture", None)
    if safety_by_architecture is not None:
        override = safety_by_architecture.get(surface["architecture"], {})
        if not isinstance(override, Mapping):
            raise ValueError(
                "Expected a safety mapping for architecture "
                f"{surface['architecture']!r}."
            )
        search["safety"] = {
            **copy.deepcopy(dict(search["safety"])),
            **copy.deepcopy(dict(override)),
        }
        if "bound_occupancy_increase_maximum" in override:
            search["safety"]["bound_occupancy"] = (
                "reject_persistent_increase"
            )
        if "projection_efficiency_minimum" in override:
            search["safety"]["projection_efficiency"] = (
                "reject_persistent_low_efficiency"
            )
    return search


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _expected_initializer_sha256(
    study: Mapping[str, Any], architecture: str
) -> str | None:
    reference = study.get("initialization_reference")
    if not isinstance(reference, Mapping):
        return None
    by_architecture = reference.get("checkpoint_sha256_by_architecture")
    if not isinstance(by_architecture, Mapping):
        return None
    value = by_architecture.get(architecture)
    return str(value) if value is not None else None


def load_study(path: str | Path = DEFAULT_STUDY) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    study = json.loads(source.read_text(encoding="utf-8"))
    variant = _study_variant(study)
    scope = study["scope"]
    if variant == "conv12":
        expected = {
            "architectures": ["conv1", "conv2"],
            "schemes": ["baseline", "ours"],
            "optimizers": ["SGD", "Adam"],
            "initializers": ["bounded_uniform", "bounded_kaiming_uniform"],
        }
        expected_excluded = {"legacy", "conv3"}
    elif variant == "conv3":
        expected = {
            "architectures": ["conv3"],
            "schemes": ["baseline", "ours"],
            "optimizers": ["SGD", "Adam"],
            "initializers": ["bounded_uniform", "bounded_kaiming_uniform"],
        }
        expected_excluded = {"legacy", "conv1", "conv2"}
    elif variant == "conv3_zero_bias":
        expected = {
            "architectures": ["conv3"],
            "schemes": ["baseline", "ours", "legacy"],
            "optimizers": ["SGD", "Adam"],
            "initializers": ["bounded_uniform"],
        }
        expected_excluded = {"conv1", "conv2"}
    else:
        expected = {
            "architectures": ["conv1", "conv2", "conv3"],
            "schemes": ["baseline", "ours", "legacy"],
            "optimizers": ["SGD", "Adam"],
            "initializers": ["bounded_uniform"],
        }
        expected_excluded = set()
    for key, value in expected.items():
        if scope.get(key) != value:
            raise ValueError(
                f"Expected scope.{key}={value!r}. Provided value: {scope.get(key)!r}."
            )
    if set(scope.get("excluded", ())) != expected_excluded:
        raise ValueError(
            f"The focused {variant} study must explicitly exclude "
            f"{sorted(expected_excluded)!r}."
        )
    if study["dataset"].get("official_test_read") is not False:
        raise ValueError("The ordinary-MNIST selector must exclude the official test split.")
    model = study["model"]
    for key in (
        "quadratic_diode_param",
        "exponential_diode_param",
        "hard_sigmoid_param",
    ):
        if not isinstance(model.get(key), dict) or not model[key]:
            raise ValueError(f"Expected an explicit non-empty model.{key} dictionary.")
    if [float(model["weight_min"]), float(model["weight_max"])] != [1e-5, 1e-4]:
        raise ValueError("The bounded conductance interval must be [1e-5, 1e-4].")
    if variant == "conv3":
        architecture = model["architectures"]["conv3"]
        if [int(architecture["T"]), int(architecture["K"])] != [8, 8]:
            raise ValueError("The focused Conv3 study must use T=8 and K=8.")
        search = study["rho_search"]
        if search.get("core_mode") != "fixed_grid":
            raise ValueError("The focused Conv3 study must use the fixed high grid.")
        expected_core = {
            "rho_conv": [0.009, 0.027, 0.081],
            "rho_dense": [0.03, 0.09, 0.27],
        }
        for axis, expected_values in expected_core.items():
            actual = [float(value) for value in search["fixed_core"][axis]]
            if actual != expected_values:
                raise ValueError(
                    f"Expected Conv3 {axis}={expected_values!r}. "
                    f"Provided value: {actual!r}."
                )
    search = study["rho_search"]
    safety = search["safety"]
    if _is_zero_bias_study(study):
        expected_bias_contract = {
            "initialization": "default_zero",
            "learning_rate": 0.0,
            "conductance_projection": False,
        }
        if study.get("bias_contract") != expected_bias_contract:
            raise ValueError(
                "The zero-bias study requires bias_contract="
                f"{expected_bias_contract!r}."
            )
        if search.get("bias_policy") != "zero":
            raise ValueError("The zero-bias study requires rho_search.bias_policy='zero'.")
        if search.get("select_best_safe_below_accuracy") is not False:
            raise ValueError(
                "The all-scheme zero-bias study must enforce the generic 90% "
                "accuracy gate without a below-accuracy fallback."
            )
        expected_policy_keys = {
            f"{architecture}__{scheme}"
            for architecture in expected["architectures"]
            for scheme in expected["schemes"]
        }
        policies = search.get("core_policy_by_surface")
        if not isinstance(policies, Mapping) or set(policies) != expected_policy_keys:
            raise ValueError(
                "Expected one explicit core_policy_by_surface row for every "
                f"architecture/scheme pair: {sorted(expected_policy_keys)!r}."
            )
        fixed_high_keys = {"conv3__baseline", "conv3__ours"}
        for key in sorted(expected_policy_keys):
            policy = policies[key]
            expected_mode = (
                "fixed_grid" if key in fixed_high_keys else "adaptive_safe_center"
            )
            if policy.get("core_mode") != expected_mode:
                raise ValueError(
                    f"Expected {key} core_mode={expected_mode!r}, got "
                    f"{policy.get('core_mode')!r}."
                )
            if expected_mode == "fixed_grid":
                fixed = policy.get("fixed_core", {})
                if [float(value) for value in fixed.get("rho_conv", ())] != [
                    0.009,
                    0.027,
                    0.081,
                ] or [float(value) for value in fixed.get("rho_dense", ())] != [
                    0.03,
                    0.09,
                    0.27,
                ]:
                    raise ValueError(f"Unexpected fixed high rho grid for {key}.")
        safety_by_architecture = search.get("safety_by_architecture")
        expected_conv3_safety = {
            **(
                {"bound_occupancy_increase_maximum": 0.20}
                if variant == "conv123_zero_bias"
                else {}
            ),
            "projection_efficiency_minimum": 0.50,
            "zero_proposal_epsilon": 1e-12,
            "boundary_persistence_steps": 16,
        }
        if safety_by_architecture != {"conv3": expected_conv3_safety}:
            raise ValueError(
                "Expected only the active Conv3 persistent boundary gates: "
                f"{expected_conv3_safety!r}."
            )
        if variant == "conv3_zero_bias":
            expected_initialization_reference = {
                "source_study_id": CONV3_ZERO_BIAS_PARENT_STUDY_ID,
                "checkpoint_sha256_by_architecture": {
                    "conv3": CONV3_ZERO_BIAS_INITIALIZER_SHA256,
                },
            }
            if study.get("initialization_reference") != expected_initialization_reference:
                raise ValueError(
                    "The Conv3 occupancy-report-only repeat requires the exact "
                    "parent initializer reference "
                    f"{expected_initialization_reference!r}."
                )
    else:
        if search.get("select_best_safe_below_accuracy") is not True:
            raise ValueError(
                "The focused bounded study must select the best safety-clean "
                "candidate when the 90% reporting threshold is not reached."
            )
        suspicious_floor = search.get("suspicious_validation_accuracy_floor")
        if (
            suspicious_floor is None
            or not 0.0 < float(suspicious_floor) < float(
                search["minimum_validation_accuracy"]
            )
        ):
            raise ValueError(
                "Expected rho_search.suspicious_validation_accuracy_floor strictly "
                "between zero and the reporting accuracy threshold."
            )
    if safety.get("bound_occupancy") != "report_only":
        raise ValueError("Bound occupancy must be report-only.")
    if safety.get("projection_efficiency") != "report_only":
        raise ValueError("Projection efficiency must be report-only.")
    return source, study


def surface_specs(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for initializer in study["scope"]["initializers"]:
        for architecture in study["scope"]["architectures"]:
            for scheme in study["scope"]["schemes"]:
                for optimizer in study["scope"]["optimizers"]:
                    result.append(
                        {
                            "index": len(result),
                            "initializer": initializer,
                            "architecture": architecture,
                            "scheme": scheme,
                            "optimizer": optimizer,
                            "surface_id": (
                                f"{initializer}__{architecture}__{scheme}__"
                                f"{optimizer.lower()}"
                            ),
                        }
                    )
    return result


def _expected_bias_names(
    study: Mapping[str, Any], architecture: str
) -> list[str]:
    channels = study["model"]["architectures"][architecture]["channels"]
    return [f"Bias_{index}" for index in range(len(channels))]


def _layer_shapes_and_pipeline(
    study: Mapping[str, Any], architecture: str
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    model = study["model"]
    spec = model["architectures"][architecture]
    height = width = 28
    shapes: list[list[int]] = [[2, height, width]]
    pipeline = []
    for channels, stride in zip(spec["channels"], spec["strides"]):
        kernel = int(model["kernel_size"])
        padding = int(model["padding"])
        height = (height + 2 * padding - kernel) // int(stride) + 1
        width = (width + 2 * padding - kernel) // int(stride) + 1
        shapes.append([int(channels), height, width])
        pipeline.append(
            {
                "kernel": [kernel, kernel],
                "stride": int(stride),
                "padding": padding,
                "mode": "convolution",
            }
        )
    shapes.append([int(model["output_dim"])])
    return shapes, pipeline


def build_source_config(
    study: Mapping[str, Any],
    *,
    initializer: str,
    architecture: str,
    scheme: str,
    optimizer: str,
    init_checkpoint_path: Path | None,
    batch_size: int | None = None,
    T: int | None = None,
    K: int | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    dataset = study["dataset"]
    model = study["model"]
    architecture_spec = model["architectures"][architecture]
    scheme_spec = model["schemes"][scheme]
    shapes, pipeline = _layer_shapes_and_pipeline(study, architecture)
    parameter_count = 2 * len(architecture_spec["channels"]) + 1
    if _is_zero_bias_study(study):
        weight_count = len(architecture_spec["channels"]) + 1
        rates = [1.0] * weight_count + [0.0] * len(
            architecture_spec["channels"]
        )
    else:
        rates = [1.0] * parameter_count
    resolved_dataset_root = (
        Path(dataset["root"]).expanduser()
        if dataset_root is None
        else Path(dataset_root).expanduser()
    )
    minimizer = json.loads(DEFAULT_MINIMIZER.read_text(encoding="utf-8"))
    config: dict[str, Any] = {
        "lab": {
            "model_key": "mnist_bp_conv_amp",
            "dataset_key": "mnist",
            "epochs": int(study["rho_search"]["candidate_epochs"]),
            "plots_dir": (
                "plots/perfectdiode_conv12_bounded_rho"
                if _study_variant(study) == "conv12"
                else "plots/perfectdiode_conv3_bounded_rho"
            ),
        },
        "input_mode": "train",
        "training_algorithm": "BP",
        "seed": int(dataset["model_seed"]),
        "lr": rates,
        "beta": 1.0,
        "lr_decay": 1.0,
        "optimizer": {
            "name": optimizer,
            "learning_rate": rates,
            "lr_decay": 1.0,
            "momentum": 0.0,
            "weight_decay": 0.0,
            **(
                {"betas": [0.9, 0.999], "eps": 1e-8}
                if optimizer == "Adam"
                else {}
            ),
        },
        "log_interval": 500,
        "max_batches": None,
        "max_test_batches": None,
        "batch_state_policy": "reset_each_batch",
        "datasets": {
            "mnist": {
                "factory": "labs.datasets.MnistTrainValidationDataset",
                "params": {
                    "name": "mnist",
                    "batch_size": int(batch_size or dataset["batch_size"]),
                    "validation_batch_size": int(dataset["validation_batch_size"]),
                    "root": str(resolved_dataset_root),
                    "train": True,
                    "download": False,
                    "normalize": True,
                    "normalize_mean": float(dataset["normalize_mean"]),
                    "normalize_std": float(dataset["normalize_std"]),
                    "normalize_scale": float(dataset["normalize_scale"]),
                    "split_seed": int(dataset["split_seed"]),
                    "shuffle_seed": int(dataset["shuffle_seed"]),
                },
            }
        },
        "model_base": {
            "weight_min": float(model["weight_min"]),
            "weight_max": float(model["weight_max"]),
            "weight_init_mode": initializer,
            "input_gain": float(architecture_spec["input_gain"]),
            "voltage_amp": float(scheme_spec["voltage_amp"]),
            "current_amp": float(scheme_spec["current_amp"]),
            "trainable_amplification": False,
            "amplification_min": 1e-6,
            "amplification_max": None,
            "non_linearity": model["non_linearity"],
            "quadratic_diode_param": copy.deepcopy(model["quadratic_diode_param"]),
            "exponential_diode_param": copy.deepcopy(model["exponential_diode_param"]),
            "hard_sigmoid_param": copy.deepcopy(model["hard_sigmoid_param"]),
            "num_iterations_inference": int(
                architecture_spec["T"] if T is None else T
            ),
            "num_iterations_training": int(
                architecture_spec["K"] if K is None else K
            ),
            "minimizer": minimizer,
        },
        "model_overrides": {
            "mnist_bp_conv_amp": {
                "layer_shapes": shapes,
                "conv_pipeline": pipeline,
                "weight_gains": [float(model["weight_gain"])]
                * (len(architecture_spec["channels"]) + 1),
            }
        },
        "energy_minimizer": {"mode": "asynchronous"},
    }
    if init_checkpoint_path is not None:
        config["init_checkpoint_path"] = str(init_checkpoint_path.resolve())
    if _is_zero_bias_study(study):
        config["bias_contract"] = copy.deepcopy(study["bias_contract"])
    return config


def materialize(
    study_path: Path,
    study: Mapping[str, Any],
    output_root: Path,
    *,
    target: str | None = None,
    dataset_root: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    surfaces = surface_specs(study)
    resolved = {
        "schema_version": _artifact_schema(study, "resolved"),
        "study_id": study["study_id"],
        "study_config": str(study_path),
        "study_config_sha256": _sha256_file(study_path),
        "code": _git_state(),
        "surface_count": len(surfaces),
        "fixed_tk_gate_count": len(surfaces) // len(study["scope"]["optimizers"]),
        "surfaces": surfaces,
        "target": target or study["execution"]["target"],
        "device": device or study["execution"]["device"],
        "official_test_read": False,
    }
    if _is_zero_bias_study(study):
        resolved.update(
            {
                "configured_target": study["execution"]["target"],
                "dataset_root": str(
                    Path(
                        study["dataset"]["root"]
                        if dataset_root is None
                        else dataset_root
                    ).expanduser()
                ),
                "configured_dataset_root": str(study["dataset"]["root"]),
                "configured_device": study["execution"]["device"],
                "bias_contract": copy.deepcopy(study["bias_contract"]),
            }
        )
    resolved_path = output_root / "study.resolved.json"
    if resolved_path.exists():
        existing = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing != resolved:
            raise RuntimeError(
                f"Existing resolved study does not match this launch: {resolved_path}."
            )
    else:
        _write_json(resolved_path, resolved)
    return resolved


class GradientCollector:
    def __init__(self, expected_batches: int):
        self.expected_batches = int(expected_batches)
        self.by_parameter: dict[str, list[torch.Tensor]] = {}
        self.batch_count = 0

    def __call__(self, batch: Mapping[str, Any]) -> bool:
        for parameter, gradient in zip(batch["parameters"], batch["gradients"]):
            name = str(getattr(parameter, "name", "")).strip()
            if name.startswith("ConvWeight_"):
                self.by_parameter.setdefault(name, []).append(
                    gradient.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
                )
        self.batch_count += 1
        return self.batch_count >= self.expected_batches


def compare_fixed_tk_gradients(
    operational: Mapping[str, Sequence[torch.Tensor]],
    reference: Mapping[str, Sequence[torch.Tensor]],
    contract: Mapping[str, Any],
    *,
    operational_t: int,
    operational_k: int,
) -> dict[str, Any]:
    if set(operational) != set(reference) or not operational:
        raise ValueError("Expected identical non-empty ConvWeight gradient sets.")
    zero_epsilon = float(contract["gradient_zero_epsilon"])
    records = []
    for name in sorted(operational):
        left = tuple(operational[name])
        right = tuple(reference[name])
        if len(left) != len(right) or not left:
            raise ValueError(f"Expected matched gradient batches for {name}.")
        batches = []
        for batch_index, (current, target) in enumerate(zip(left, right)):
            if current.shape != target.shape:
                raise ValueError(f"Gradient shape mismatch for {name}, batch {batch_index}.")
            current_norm = float(torch.linalg.vector_norm(current).item())
            target_norm = float(torch.linalg.vector_norm(target).item())
            norm_delta = abs(current_norm - target_norm) / max(
                current_norm, target_norm, 1e-30
            )
            current_zero = float((current.abs() <= zero_epsilon).to(torch.float64).mean())
            target_zero = float((target.abs() <= zero_epsilon).to(torch.float64).mean())
            denominator = current_norm * target_norm
            cosine = (
                float(torch.dot(current, target).item() / denominator)
                if denominator > 0.0
                else None
            )
            batches.append(
                {
                    "batch_index": batch_index,
                    "gradient_l2": current_norm,
                    "reference_gradient_l2": target_norm,
                    "relative_gradient_l2_norm_delta": norm_delta,
                    "gradient_zero_fraction": current_zero,
                    "reference_gradient_zero_fraction": target_zero,
                    "absolute_zero_fraction_delta": abs(current_zero - target_zero),
                    "gradient_vector_cosine": cosine,
                }
            )
        mean_norm = sum(item["gradient_l2"] for item in batches) / len(batches)
        mean_reference = sum(item["reference_gradient_l2"] for item in batches) / len(
            batches
        )
        mean_zero = sum(item["gradient_zero_fraction"] for item in batches) / len(
            batches
        )
        mean_reference_zero = sum(
            item["reference_gradient_zero_fraction"] for item in batches
        ) / len(batches)
        cosines = [item["gradient_vector_cosine"] for item in batches]
        mean_cosine = (
            sum(float(value) for value in cosines) / len(cosines)
            if all(value is not None and math.isfinite(float(value)) for value in cosines)
            else None
        )
        norm_delta = abs(mean_norm - mean_reference) / max(
            mean_norm, mean_reference, 1e-30
        )
        zero_delta = abs(mean_zero - mean_reference_zero)
        passed = (
            mean_cosine is not None
            and norm_delta
            <= float(contract["relative_gradient_l2_norm_delta_maximum"])
            and zero_delta
            <= float(contract["absolute_zero_fraction_delta_maximum"])
            and mean_cosine >= float(contract["gradient_vector_cosine_minimum"])
        )
        records.append(
            {
                "parameter": name,
                "batch_count": len(batches),
                "gradient_l2_mean": mean_norm,
                "reference_gradient_l2_mean": mean_reference,
                "relative_gradient_l2_norm_delta": norm_delta,
                "gradient_zero_fraction_mean": mean_zero,
                "reference_gradient_zero_fraction_mean": mean_reference_zero,
                "absolute_zero_fraction_delta": zero_delta,
                "gradient_vector_cosine_mean": mean_cosine,
                "passed": passed,
                "batches": batches,
            }
        )
    passed = all(record["passed"] for record in records)
    return {
        "schema_version": "perfectdiode-conv12-fixed-tk-security/v1",
        "status": "complete" if passed else "unresolved_fixed_tk_gradient_mismatch",
        "security_passed": passed,
        "operational": {"T": operational_t, "K": operational_k},
        "reference": {
            "T": int(contract["reference_T"]),
            "K": int(contract["reference_K"]),
        },
        "gates": {
            key: contract[key]
            for key in (
                "relative_gradient_l2_norm_delta_maximum",
                "absolute_zero_fraction_delta_maximum",
                "gradient_vector_cosine_minimum",
                "gradient_zero_epsilon",
            )
        },
        "parameter_diagnostics": records,
        "official_test_read": False,
    }


def ensure_asset(
    study: Mapping[str, Any],
    output_root: Path,
    *,
    initializer: str,
    architecture: str,
    device: str,
    dataset_root: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    asset_dir = output_root / "assets" / initializer / architecture
    checkpoint = asset_dir / "final_model.pt"
    record_path = asset_dir / "asset.json"
    expected_bias_names = _expected_bias_names(study, architecture)
    zero_bias_study = _is_zero_bias_study(study)
    expected_checkpoint_sha256 = _expected_initializer_sha256(
        study, architecture
    )
    resolved_dataset_root = str(
        Path(
            study["dataset"]["root"]
            if dataset_root is None
            else dataset_root
        ).expanduser()
    )
    if checkpoint.exists():
        observed_checkpoint_sha256 = _sha256_file(checkpoint)
        if (
            expected_checkpoint_sha256 is not None
            and observed_checkpoint_sha256 != expected_checkpoint_sha256
        ):
            raise RuntimeError(
                "Pinned initializer SHA-256 mismatch for "
                f"{initializer}/{architecture}: expected "
                f"{expected_checkpoint_sha256}, observed "
                f"{observed_checkpoint_sha256}."
            )
    if checkpoint.exists() and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        reusable = record.get("checkpoint_sha256") == observed_checkpoint_sha256
        if expected_checkpoint_sha256 is not None:
            reusable = (
                reusable
                and record.get("expected_checkpoint_sha256")
                == expected_checkpoint_sha256
            )
        if zero_bias_study and reusable:
            verification = verify_zero_bias_checkpoint(
                checkpoint, expected_bias_names
            )
            reusable = (
                record.get("schema_version")
                == _artifact_schema(study, "bounded-init-asset")
                and record.get("initializer") == initializer
                and record.get("architecture") == architecture
                and record.get("dataset_root") == resolved_dataset_root
                and record.get("zero_bias_checkpoint_verification")
                == verification
            )
        if reusable:
            return checkpoint, record

    config = build_source_config(
        study,
        initializer=initializer,
        architecture=architecture,
        scheme="baseline",
        optimizer="SGD",
        init_checkpoint_path=None,
        dataset_root=dataset_root,
    )
    rates = [0.0] * len(config["lr"])
    config["lr"] = rates
    config["optimizer"]["learning_rate"] = rates
    config["lab"]["epochs"] = 1
    config["max_batches"] = 1
    config["max_test_batches"] = 1
    config_path = asset_dir / "source_config.json"
    _write_json(config_path, config)
    _run_trainer(
        config_path,
        asset_dir,
        device=device,
        apply_optimizer_steps=False,
    )
    checkpoint_sha256 = _sha256_file(checkpoint)
    if (
        expected_checkpoint_sha256 is not None
        and checkpoint_sha256 != expected_checkpoint_sha256
    ):
        raise RuntimeError(
            "Pinned initializer SHA-256 mismatch for "
            f"{initializer}/{architecture}: expected "
            f"{expected_checkpoint_sha256}, observed {checkpoint_sha256}."
        )
    zero_bias_verification = (
        verify_zero_bias_checkpoint(checkpoint, expected_bias_names)
        if zero_bias_study
        else None
    )
    record = {
        "schema_version": _artifact_schema(study, "bounded-init-asset"),
        "initializer": initializer,
        "architecture": architecture,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "source_config_sha256": _sha256_file(config_path),
        "official_test_read": False,
    }
    if zero_bias_study:
        record.update(
            {
                "dataset_root": resolved_dataset_root,
                "zero_bias_checkpoint_verification": zero_bias_verification,
            }
        )
    if expected_checkpoint_sha256 is not None:
        record["expected_checkpoint_sha256"] = expected_checkpoint_sha256
    _write_json(record_path, record)
    return checkpoint, record


def run_fixed_tk_gate(
    study: Mapping[str, Any],
    output_root: Path,
    surface: Mapping[str, Any],
    checkpoint: Path,
    *,
    device: str,
    smoke: bool,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    gate_dir = (
        output_root
        / ("smoke" if smoke else "fixed_tk")
        / surface["initializer"]
        / surface["architecture"]
        / surface["scheme"]
    )
    result_path = gate_dir / "result.json"
    contract = study["fixed_tk_security"]
    expected_batches = (
        1
        if smoke
        else int(contract["cohort_examples"]) // int(contract["batch_size"])
    )
    architecture = study["model"]["architectures"][surface["architecture"]]
    configs = {}
    collectors = {}
    for label, T, K in (
        ("operational", architecture["T"], architecture["K"]),
        ("reference", contract["reference_T"], contract["reference_K"]),
    ):
        config = build_source_config(
            study,
            initializer=surface["initializer"],
            architecture=surface["architecture"],
            scheme=surface["scheme"],
            optimizer="SGD",
            init_checkpoint_path=checkpoint,
            batch_size=int(contract["batch_size"]),
            T=int(T),
            K=int(K),
            dataset_root=dataset_root,
        )
        rates = [0.0] * len(config["lr"])
        config["lr"] = rates
        config["optimizer"]["learning_rate"] = rates
        config["lab"]["epochs"] = 1
        config["max_batches"] = expected_batches
        config["max_test_batches"] = 1
        config_path = gate_dir / label / "source_config.json"
        _write_json(config_path, config)
        configs[label] = config_path
    signature = _sha256_json(
        {
            "operational_config": _sha256_file(configs["operational"]),
            "reference_config": _sha256_file(configs["reference"]),
            "checkpoint": _sha256_file(checkpoint),
            "expected_batches": expected_batches,
            "contract": contract,
        }
    )
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature:
            if _is_zero_bias_study(study):
                expected_bias_names = _expected_bias_names(
                    study, surface["architecture"]
                )
                verification = {
                    label: verify_zero_bias_run_checkpoints(
                        gate_dir / label,
                        expected_bias_names,
                    )
                    for label in ("operational", "reference")
                }
                existing["zero_bias_checkpoint_verification"] = verification
                _write_json(result_path, existing)
            return existing
    for label in ("operational", "reference"):
        collector = GradientCollector(expected_batches)
        collectors[label] = collector
        _run_trainer(
            configs[label],
            gate_dir / label,
            device=device,
            gradient_callback=collector,
            apply_optimizer_steps=False,
        )
        if collector.batch_count != expected_batches:
            raise RuntimeError(
                f"Expected {expected_batches} {label} batches, got {collector.batch_count}."
            )
    result = compare_fixed_tk_gradients(
        collectors["operational"].by_parameter,
        collectors["reference"].by_parameter,
        contract,
        operational_t=int(architecture["T"]),
        operational_k=int(architecture["K"]),
    )
    result["schema_version"] = _artifact_schema(study, "fixed-tk-security")
    result["signature"] = signature
    result["initializer"] = surface["initializer"]
    result["architecture"] = surface["architecture"]
    result["scheme"] = surface["scheme"]
    result["smoke"] = smoke
    if _is_zero_bias_study(study):
        expected_bias_names = _expected_bias_names(
            study, surface["architecture"]
        )
        result["zero_bias_checkpoint_verification"] = {
            label: verify_zero_bias_run_checkpoints(
                gate_dir / label,
                expected_bias_names,
            )
            for label in ("operational", "reference")
        }
    _write_json(result_path, result)
    return result


def run_conv3_tk_operating_point_gate(
    study: Mapping[str, Any],
    output_root: Path,
    surface: Mapping[str, Any],
    checkpoint: Path,
    *,
    device: str,
    smoke: bool,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run or reuse the full Conv3 T-residual and fixed-T K audit."""

    from experiments.conv3_operating_point_gate import (
        run_conv3_operating_point_gate,
    )

    if not _is_zero_bias_study(study):
        raise ValueError("The full Conv3 gate is bound to the new zero-bias study.")
    if surface["architecture"] != "conv3":
        raise ValueError("The full Conv3 gate requires a Conv3 surface.")

    gate_dir = (
        output_root
        / ("smoke" if smoke else "fixed_tk")
        / surface["initializer"]
        / surface["architecture"]
        / surface["scheme"]
    )
    source_path = gate_dir / "source_config.json"
    result_path = gate_dir / "result.json"
    source_config = build_source_config(
        study,
        initializer=surface["initializer"],
        architecture=surface["architecture"],
        scheme=surface["scheme"],
        optimizer="SGD",
        init_checkpoint_path=checkpoint,
        dataset_root=dataset_root,
    )
    rates = [0.0] * len(source_config["lr"])
    source_config["lr"] = rates
    source_config["optimizer"]["learning_rate"] = rates
    _write_json(source_path, source_config)

    source_sha256 = _sha256_file(source_path)
    checkpoint_sha256 = _sha256_file(checkpoint)
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("source_config_sha256") == source_sha256
            and existing.get("checkpoint_sha256") == checkpoint_sha256
            and existing.get("smoke") is smoke
        ):
            return existing

    result = run_conv3_operating_point_gate(
        source_path,
        checkpoint,
        device=device,
        output_path=result_path,
        smoke=smoke,
    )
    result["initializer"] = surface["initializer"]
    result["architecture"] = surface["architecture"]
    result["scheme"] = surface["scheme"]
    _write_json(result_path, result)
    return result


def _rho_axes(
    study: Mapping[str, Any],
    surface: Mapping[str, Any] | None = None,
) -> tuple[list[float], list[float]]:
    if surface is None:
        if "core_policy_by_surface" in study["rho_search"]:
            raise ValueError(
                "A surface is required to resolve per-surface rho axes."
            )
        search = study["rho_search"]
    else:
        search = _surface_search(study, surface)
    if search.get("core_mode", "adaptive_safe_center") == "fixed_grid":
        core_conv = [float(value) for value in search["fixed_core"]["rho_conv"]]
        core_dense = [float(value) for value in search["fixed_core"]["rho_dense"]]
        return (
            sorted({min(core_conv) / 3.0, *core_conv, max(core_conv) * 3.0}),
            sorted({min(core_dense) / 3.0, *core_dense, max(core_dense) * 3.0}),
        )
    center = search["center"]
    # Covers six downward center attempts, either side of every core, and the
    # single permitted expansion wave.
    exponents = range(-7, 3)
    return (
        sorted(float(center["rho_conv"]) * (3.0**exponent) for exponent in exponents),
        sorted(float(center["rho_dense"]) * (3.0**exponent) for exponent in exponents),
    )


def _rho_index(
    rho_conv_axis: Sequence[float],
    rho_dense_axis: Sequence[float],
    rho_conv: float,
    rho_dense: float,
) -> int:
    conv_index = min(
        range(len(rho_conv_axis)),
        key=lambda index: abs(rho_conv_axis[index] - rho_conv),
    )
    dense_index = min(
        range(len(rho_dense_axis)),
        key=lambda index: abs(rho_dense_axis[index] - rho_dense),
    )
    for requested, observed, label in (
        (rho_conv, rho_conv_axis[conv_index], "rho_conv"),
        (rho_dense, rho_dense_axis[dense_index], "rho_dense"),
    ):
        if abs(requested - observed) > 1e-12 * max(abs(requested), abs(observed), 1.0):
            raise ValueError(f"{label}={requested!r} is not on the resolved rho axis.")
    return conv_index * len(rho_dense_axis) + dense_index


def _rho_args(
    study: Mapping[str, Any],
    surface: Mapping[str, Any],
    source_config: Path,
    rho_root: Path,
    rho_conv_axis: Sequence[float],
    rho_dense_axis: Sequence[float],
    *,
    device: str,
    target: str | None = None,
    index: int | None,
    probe_only: bool = False,
    canary_only: bool = False,
    smoke: bool = False,
) -> Namespace:
    search = _surface_search(study, surface)
    safety = search["safety"]
    conv3_post_training_tk = bool(
        _is_zero_bias_study(study)
        and surface["architecture"] == "conv3"
    )
    return Namespace(
        config=str(source_config),
        output_root=str(rho_root),
        rho_conv=list(rho_conv_axis),
        rho_dense=list(rho_dense_axis),
        optimizer=surface["optimizer"],
        bias_policy=search["bias_policy"],
        probe_batches=[2] if smoke else list(search["probe_batches"]),
        stability_tolerance=1e9 if smoke else search["probe_stability_tolerance"],
        epochs=1 if smoke else int(search["candidate_epochs"]),
        checkpoint_every_epoch=conv3_post_training_tk,
        post_candidate_gate_name=(
            "conv3_epochwise_k64_viability" if conv3_post_training_tk else None
        ),
        max_batches=1 if smoke else None,
        max_validation_batches=1 if smoke else None,
        validation_batch_size=int(study["dataset"]["validation_batch_size"]),
        split_seed=int(study["dataset"]["split_seed"]),
        shuffle_seed=int(study["dataset"]["shuffle_seed"]),
        device=device,
        study_id=study["study_id"] + ("-smoke" if smoke else ""),
        target=target or study["execution"]["target"],
        probe_only=probe_only,
        canary_only=canary_only,
        canary_steps=1 if smoke else int(search["canary_steps"]),
        expected_candidate_steps=1
        if smoke
        else int(search["expected_candidate_steps"]),
        minimum_validation_accuracy=0.0
        if smoke
        else float(search["minimum_validation_accuracy"]),
        safety_warmup_steps=int(safety["warmup_steps"]),
        safety_ema_decay=float(safety["loss_ema_decay"]),
        safety_loss_factor=float(safety["loss_ema_factor"]),
        safety_gradient_factor=float(safety["gradient_rms_factor"]),
        safety_persistence=int(safety["persistence_steps"]),
        safety_bound_occupancy_increase_maximum=safety.get(
            "bound_occupancy_increase_maximum"
        ),
        safety_projection_efficiency_minimum=safety.get(
            "projection_efficiency_minimum"
        ),
        safety_zero_proposal_epsilon=float(
            safety.get("zero_proposal_epsilon", 1e-30)
        ),
        safety_boundary_persistence=int(
            safety.get("boundary_persistence_steps", 16)
        ),
        index=index,
        collect_only=False,
        force=False,
        dry_run=False,
        smoke=smoke,
        reporting_command={
            "module": "experiments.run_conv12_bounded_rho",
            "surface_index": surface["index"],
            "cell_index": index,
        },
    )


def _cell(
    rho_root: Path,
    index: int,
    rho_conv: float,
    rho_dense: float,
) -> tuple[Path, dict[str, Any]]:
    name = f"{index:03d}_rc_{_float_token(rho_conv)}_rd_{_float_token(rho_dense)}"
    cell_dir = rho_root / "cells" / name
    return cell_dir, json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))


def _run_rho_cell(
    study: Mapping[str, Any],
    surface: Mapping[str, Any],
    source_config: Path,
    rho_root: Path,
    rho_conv_axis: Sequence[float],
    rho_dense_axis: Sequence[float],
    rho_conv: float,
    rho_dense: float,
    *,
    device: str,
    target: str | None = None,
    canary_only: bool,
    smoke: bool = False,
    operating_point_gate: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    index = _rho_index(rho_conv_axis, rho_dense_axis, rho_conv, rho_dense)
    conv_index, dense_index = divmod(index, len(rho_dense_axis))
    rho_conv = rho_conv_axis[conv_index]
    rho_dense = rho_dense_axis[dense_index]
    checkpoint_epochs = bool(
        _is_zero_bias_study(study)
        and surface["architecture"] == "conv3"
        and not canary_only
    )
    rho_args = _rho_args(
        study,
        surface,
        source_config,
        rho_root,
        rho_conv_axis,
        rho_dense_axis,
        device=device,
        target=target,
        index=index,
        canary_only=canary_only,
        smoke=smoke,
    )
    if checkpoint_epochs:
        def post_candidate_gate(
            cell_dir: Path,
            cell_record: Mapping[str, Any],
        ) -> dict[str, Any]:
            return ensure_conv3_candidate_epochwise_viability(
                study,
                surface,
                cell_dir,
                cell_record,
                device=device,
                smoke=smoke,
                operating_point_gate=operating_point_gate,
            )

        rho_args.post_candidate_callback = post_candidate_gate
    run_rho_search(rho_args)
    cell_dir, cell = _cell(rho_root, index, rho_conv, rho_dense)
    if checkpoint_epochs and cell.get("status") in {
        "complete",
        "candidate_rejected_post_training_tk",
    }:
        if operating_point_gate is None:
            raise RuntimeError(
                "A completed Conv3 candidate requires its initialization "
                "operating-point gate for strict evidence validation."
            )
        _validate_conv3_candidate_epochwise_summary(
            study,
            surface,
            cell_dir,
            cell,
            operating_point_gate,
            smoke=smoke,
            require_canonical_result=True,
        )
    return cell_dir, cell


def _has_clean_canary(cell_dir: Path, cell: Mapping[str, Any]) -> bool:
    if cell.get("status") == "canary_clean":
        return True
    if cell.get("status") != "complete":
        return False
    canary_path = cell_dir / "canary.json"
    if not canary_path.is_file():
        return False
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    return (
        canary.get("status") == "clean"
        and int(canary.get("completed_steps", -1))
        == int(canary.get("requested_steps", -2))
    )


def _conv3_gradient_cohort_binding(result: Mapping[str, Any]) -> dict[str, Any]:
    cohort = result.get("dataset_cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError("A Conv3 T/K gate is missing its dataset cohort.")
    keys = (
        "source_split",
        "split_seed",
        "shuffle_seed",
        "train_indices_sha256",
        "validation_indices_sha256",
        "gradient_prefix_examples",
        "gradient_batch_size",
        "gradient_prefix_original_indices_sha256",
        "gradient_prefix_tensor_sha256",
    )
    missing = [key for key in keys if cohort.get(key) is None]
    if missing:
        raise ValueError(f"A Conv3 T/K gate has incomplete cohort binding: {missing!r}.")
    return {key: cohort[key] for key in keys}


def _conv3_full_cohort_binding(result: Mapping[str, Any]) -> dict[str, Any]:
    cohort = result.get("dataset_cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError("A Conv3 T/K gate is missing its dataset cohort.")
    binding = _conv3_gradient_cohort_binding(result)
    keys = (
        "examples",
        "batch_size",
        "cohort_original_indices_sha256",
        "cohort_tensor_sha256",
    )
    missing = [key for key in keys if cohort.get(key) is None]
    if missing:
        raise ValueError(
            f"A Conv3 residual gate has incomplete cohort binding: {missing!r}."
        )
    binding.update({key: cohort[key] for key in keys})
    return binding


def _artifact_beneath(root: Path, relative: str, *, label: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"{label} escapes its candidate directory: {relative!r}.")
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}.")
    return path


def _candidate_epoch_records(
    cell_dir: Path,
    *,
    expected_epochs: int,
) -> list[dict[str, Any]]:
    """Load and hash-verify the trainer's authoritative epoch index and metrics."""

    index_path = cell_dir / "epoch_checkpoint_index.jsonl"
    metrics_path = cell_dir / "metrics.json"
    if not index_path.is_file() or not metrics_path.is_file():
        raise RuntimeError(
            "Conv3 post-training gates require metrics.json and "
            "epoch_checkpoint_index.jsonl."
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("checkpoint_every_epoch") is not True:
        raise RuntimeError("Conv3 candidates must enable checkpoint_every_epoch.")
    if int(metrics.get("epoch_checkpoint_count", -1)) != expected_epochs + 1:
        raise RuntimeError(
            "Unexpected Conv3 epoch checkpoint count: "
            f"expected={expected_epochs + 1}, "
            f"observed={metrics.get('epoch_checkpoint_count')!r}."
        )

    rows: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(
        index_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        epoch = int(row["epoch"])
        if epoch in rows:
            raise RuntimeError(f"Duplicate epoch {epoch} in {index_path}.")
        model_path = _artifact_beneath(
            cell_dir, str(row["model_path"]), label=f"epoch {epoch} model checkpoint"
        )
        optimizer_path = _artifact_beneath(
            cell_dir,
            str(row["optimizer_path"]),
            label=f"epoch {epoch} optimizer checkpoint",
        )
        for path, prefix in ((model_path, "model"), (optimizer_path, "optimizer")):
            expected_sha256 = str(row[f"{prefix}_sha256"])
            observed_sha256 = _sha256_file(path)
            if observed_sha256 != expected_sha256:
                raise RuntimeError(
                    f"Epoch {epoch} {prefix} checkpoint hash mismatch at line "
                    f"{line_number}: {path}."
                )
            if int(row[f"{prefix}_size_bytes"]) != path.stat().st_size:
                raise RuntimeError(
                    f"Epoch {epoch} {prefix} checkpoint size mismatch: {path}."
                )
        rows[epoch] = {
            "epoch": epoch,
            "checkpoint": str(model_path),
            "checkpoint_relative_path": str(row["model_path"]),
            "checkpoint_sha256": str(row["model_sha256"]),
            "optimizer_checkpoint": str(optimizer_path),
            "optimizer_checkpoint_relative_path": str(row["optimizer_path"]),
            "optimizer_checkpoint_sha256": str(row["optimizer_sha256"]),
        }
    expected = set(range(expected_epochs + 1))
    if set(rows) != expected:
        raise RuntimeError(
            f"Expected epoch checkpoint set {sorted(expected)!r}, got {sorted(rows)!r}."
        )

    loss_path = cell_dir / "loss_test.npy"
    accuracy_path = cell_dir / "accuracy_test.npy"
    if not loss_path.is_file() or not accuracy_path.is_file():
        raise RuntimeError("Missing Conv3 epochwise validation metric arrays.")
    losses = np.load(loss_path, allow_pickle=False)
    accuracies = np.load(accuracy_path, allow_pickle=False)
    if losses.shape != (expected_epochs,) or accuracies.shape != (expected_epochs,):
        raise RuntimeError(
            "Unexpected Conv3 epochwise validation metric shapes: "
            f"loss={losses.shape!r}, accuracy={accuracies.shape!r}."
        )
    records = []
    for epoch in range(1, expected_epochs + 1):
        record = dict(rows[epoch])
        record.update(
            {
                "validation_loss": float(losses[epoch - 1]),
                "validation_accuracy": float(accuracies[epoch - 1]),
            }
        )
        records.append(record)
    return records


def _validate_conv3_k64_gate_receipt(
    gate: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    source_config_sha256: str,
    expected_cohort: Mapping[str, Any],
    smoke: bool,
) -> tuple[bool, dict[str, Any]]:
    expected_schema = "perfectdiode-conv3-bounded-uniform-k64-gradient-viability/v1"
    if gate.get("schema_version") != expected_schema:
        raise RuntimeError(f"Unexpected Conv3 K64 receipt schema: {gate.get('schema_version')!r}.")
    if gate.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
        raise RuntimeError("Conv3 K64 receipt/checkpoint hash mismatch.")
    if gate.get("checkpoint_sha256_after_replay") != checkpoint["checkpoint_sha256"]:
        raise RuntimeError("Conv3 K64 receipt lacks a matching post-replay checkpoint hash.")
    if gate.get("source_config_sha256") != source_config_sha256:
        raise RuntimeError("Conv3 K64 receipt/source-config hash mismatch.")
    if gate.get("checkpoint_unchanged") is not True:
        raise RuntimeError("Conv3 K64 replay did not prove the checkpoint unchanged.")
    if gate.get("smoke") is not smoke:
        raise RuntimeError("Conv3 K64 receipt smoke/production mismatch.")
    if not smoke and gate.get("scientifically_complete") is not True:
        raise RuntimeError("Conv3 K64 production receipt is scientifically incomplete.")
    observed_cohort = _conv3_gradient_cohort_binding(gate)
    if observed_cohort != dict(expected_cohort):
        raise RuntimeError("Conv3 K64 receipt uses a different frozen gradient cohort.")
    status = gate.get("status")
    viability = gate.get("viability_passed")
    if status == "complete" and viability is True:
        passed = True
    elif status == "unresolved_tk_gradient_viability" and viability is False:
        passed = False
    else:
        raise RuntimeError(
            "Inconsistent Conv3 K64 scientific status/viability fields: "
            f"status={status!r}, viability_passed={viability!r}."
        )
    if gate.get("official_test_read") is not False:
        raise RuntimeError("Conv3 K64 receipt violates official-test isolation.")
    return passed, observed_cohort


def _validate_conv3_full_gate_receipt(
    gate: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    source_config_sha256: str,
    expected_cohort: Mapping[str, Any],
    smoke: bool,
) -> tuple[bool, dict[str, Any]]:
    expected_schema = "perfectdiode-conv3-bounded-uniform-operating-point-gate/v1"
    if gate.get("schema_version") != expected_schema:
        raise RuntimeError(f"Unexpected Conv3 full T/K receipt schema: {gate.get('schema_version')!r}.")
    if gate.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]:
        raise RuntimeError("Conv3 full T/K receipt/checkpoint hash mismatch.")
    if gate.get("checkpoint_sha256_after_replay") != checkpoint["checkpoint_sha256"]:
        raise RuntimeError(
            "Conv3 full T/K receipt lacks a matching post-replay checkpoint hash."
        )
    if gate.get("source_config_sha256") != source_config_sha256:
        raise RuntimeError("Conv3 full T/K receipt/source-config hash mismatch.")
    if gate.get("checkpoint_unchanged") is not True:
        raise RuntimeError("Conv3 full T/K replay did not prove the checkpoint unchanged.")
    if gate.get("smoke") is not smoke:
        raise RuntimeError("Conv3 full T/K receipt smoke/production mismatch.")
    if not smoke and gate.get("scientifically_complete") is not True:
        raise RuntimeError("Conv3 full T/K production receipt is scientifically incomplete.")
    observed_cohort = _conv3_full_cohort_binding(gate)
    if observed_cohort != dict(expected_cohort):
        raise RuntimeError("Conv3 full T/K receipt uses different frozen cohorts.")
    status = gate.get("status")
    security_passed = gate.get("security_passed")
    negative_statuses = {
        "unresolved_tk_residual",
        "unresolved_tk_gradient_viability",
        "unresolved_fixed_tk_gradient_mismatch",
    }
    if status == "complete" and security_passed is True:
        passed = True
    elif status in negative_statuses and security_passed is False:
        passed = False
    else:
        raise RuntimeError(
            "Inconsistent Conv3 full T/K scientific status/security fields: "
            f"status={status!r}, security_passed={security_passed!r}."
        )
    if gate.get("official_test_read") is not False:
        raise RuntimeError("Conv3 full T/K receipt violates official-test isolation.")
    return passed, observed_cohort


def _validate_conv3_initial_operating_point_gate(
    gate: Mapping[str, Any], *, smoke: bool
) -> None:
    """Validate the initialization gate that freezes all later replay cohorts."""

    if gate.get("schema_version") != (
        "perfectdiode-conv3-bounded-uniform-operating-point-gate/v1"
    ):
        raise RuntimeError("Unexpected Conv3 initialization operating-point schema.")
    if gate.get("status") != "complete" or gate.get("security_passed") is not True:
        raise RuntimeError("Conv3 post-training replay requires a passing initialization gate.")
    if gate.get("checkpoint_unchanged") is not True:
        raise RuntimeError("The Conv3 initialization gate did not preserve its checkpoint.")
    if gate.get("checkpoint_sha256_after_replay") != gate.get("checkpoint_sha256"):
        raise RuntimeError(
            "The Conv3 initialization gate has inconsistent checkpoint hashes."
        )
    if gate.get("smoke") is not smoke:
        raise RuntimeError("Conv3 initialization gate smoke/production mismatch.")
    if not smoke and gate.get("scientifically_complete") is not True:
        raise RuntimeError("The Conv3 initialization gate is scientifically incomplete.")
    if gate.get("official_test_read") is not False:
        raise RuntimeError("The Conv3 initialization gate violates official-test isolation.")
    _conv3_full_cohort_binding(gate)


def _conv3_candidate_gate_context(
    study: Mapping[str, Any],
    surface: Mapping[str, Any],
    cell_dir: Path,
    cell: Mapping[str, Any],
    operating_point_gate: Mapping[str, Any],
    *,
    smoke: bool,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Resolve the current, hash-bound inputs to an epochwise K64 gate."""

    _validate_conv3_initial_operating_point_gate(operating_point_gate, smoke=smoke)
    expected_epochs = 1 if smoke else int(
        _surface_search(study, surface)["candidate_epochs"]
    )
    source_path = cell_dir / "source_config.json"
    if not source_path.is_file():
        raise RuntimeError(f"Missing candidate source config: {source_path}.")
    checkpoints = _candidate_epoch_records(cell_dir, expected_epochs=expected_epochs)
    expected_cohort = _conv3_gradient_cohort_binding(operating_point_gate)
    signature = {
        "cell_signature": cell.get("signature"),
        "source_config_sha256": _sha256_file(source_path),
        "checkpoint_index_sha256": _sha256_file(
            cell_dir / "epoch_checkpoint_index.jsonl"
        ),
        "metrics_sha256": _sha256_file(cell_dir / "metrics.json"),
        "loss_test_sha256": _sha256_file(cell_dir / "loss_test.npy"),
        "accuracy_test_sha256": _sha256_file(cell_dir / "accuracy_test.npy"),
        "expected_epochs": expected_epochs,
        "smoke": smoke,
        "expected_gradient_cohort": expected_cohort,
    }
    return source_path, checkpoints, expected_cohort, signature


def _validate_conv3_candidate_epochwise_summary(
    study: Mapping[str, Any],
    surface: Mapping[str, Any],
    cell_dir: Path,
    cell: Mapping[str, Any],
    operating_point_gate: Mapping[str, Any],
    *,
    smoke: bool,
    require_canonical_result: bool,
) -> dict[str, Any]:
    """Strictly validate one candidate's flat K64 summary and detailed receipts."""

    from experiments.reporting import validate_run

    summary_path = cell_dir / "epochwise_k64_viability.json"
    if not summary_path.is_file():
        raise RuntimeError(f"Missing Conv3 epochwise K64 summary: {summary_path}.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise RuntimeError("Conv3 epochwise K64 summary must be a JSON object.")
    source_path, checkpoints, expected_cohort, signature = (
        _conv3_candidate_gate_context(
            study,
            surface,
            cell_dir,
            cell,
            operating_point_gate,
            smoke=smoke,
        )
    )
    if summary.get("schema_version") != (
        "perfectdiode-conv3-bounded-uniform-epochwise-k64-viability/v1"
    ):
        raise RuntimeError("Unexpected Conv3 epochwise K64 summary schema.")
    if summary.get("signature") != signature:
        raise RuntimeError("Conv3 epochwise K64 summary signature is stale or corrupt.")
    if summary.get("official_test_read") is not False:
        raise RuntimeError("Conv3 epochwise K64 summary violates official-test isolation.")
    expected_epochs = int(signature["expected_epochs"])
    if summary.get("expected_epochs") != expected_epochs:
        raise RuntimeError("Conv3 epochwise K64 summary epoch-count mismatch.")
    if summary.get("expected_gradient_cohort") != expected_cohort:
        raise RuntimeError("Conv3 epochwise K64 summary cohort binding mismatch.")
    if summary.get("candidate_metrics_sha256") != signature["metrics_sha256"]:
        raise RuntimeError("Conv3 epochwise K64 summary metrics hash mismatch.")
    if (
        summary.get("epoch_checkpoint_index_sha256")
        != signature["checkpoint_index_sha256"]
    ):
        raise RuntimeError("Conv3 epochwise K64 summary checkpoint-index hash mismatch.")

    rows = summary.get("epochs")
    if not isinstance(rows, list) or len(rows) != expected_epochs:
        raise RuntimeError("Conv3 epochwise K64 summary has incomplete epoch coverage.")
    rows_by_epoch: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("epoch"), int):
            raise RuntimeError("Conv3 epochwise K64 summary has an invalid epoch row.")
        epoch = int(row["epoch"])
        if epoch in rows_by_epoch:
            raise RuntimeError(f"Duplicate Conv3 K64 summary epoch {epoch}.")
        rows_by_epoch[epoch] = row
    expected_epoch_set = set(range(1, expected_epochs + 1))
    if set(rows_by_epoch) != expected_epoch_set:
        raise RuntimeError("Conv3 epochwise K64 summary has the wrong epoch set.")

    observed_passes: list[bool] = []
    for checkpoint in checkpoints:
        epoch = int(checkpoint["epoch"])
        row = rows_by_epoch[epoch]
        for key in (
            "checkpoint_relative_path",
            "checkpoint_sha256",
            "optimizer_checkpoint_relative_path",
            "optimizer_checkpoint_sha256",
            "validation_loss",
            "validation_accuracy",
        ):
            if row.get(key) != checkpoint[key]:
                raise RuntimeError(
                    f"Conv3 K64 epoch {epoch} has a stale {key} binding."
                )
        expected_relative = (
            f"post_training_tk/epochwise_k64_viability/epoch_{epoch:03d}.json"
        )
        if row.get("gate_artifact_relative_path") != expected_relative:
            raise RuntimeError(
                f"Conv3 K64 epoch {epoch} has an unexpected receipt path."
            )
        receipt_path = _artifact_beneath(
            cell_dir,
            expected_relative,
            label=f"epoch {epoch} Conv3 K64 receipt",
        )
        if row.get("gate_artifact_sha256") != _sha256_file(receipt_path):
            raise RuntimeError(f"Conv3 K64 epoch {epoch} receipt hash mismatch.")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, Mapping):
            raise RuntimeError(f"Conv3 K64 epoch {epoch} receipt is not a JSON object.")
        receipt_passed, observed_cohort = _validate_conv3_k64_gate_receipt(
            receipt,
            checkpoint,
            source_config_sha256=_sha256_file(source_path),
            expected_cohort=expected_cohort,
            smoke=smoke,
        )
        if row.get("gate_status") != receipt.get("status"):
            raise RuntimeError(f"Conv3 K64 epoch {epoch} status binding mismatch.")
        if row.get("gradient_cohort") != observed_cohort:
            raise RuntimeError(f"Conv3 K64 epoch {epoch} cohort record mismatch.")
        if row.get("cohort_matches_operating_point") is not True:
            raise RuntimeError(f"Conv3 K64 epoch {epoch} lacks its cohort proof.")
        if row.get("passed") is not receipt_passed or row.get("error") is not None:
            raise RuntimeError(f"Conv3 K64 epoch {epoch} outcome record mismatch.")
        observed_passes.append(receipt_passed)

    passed = bool(observed_passes and all(observed_passes))
    expected_status = "complete" if passed else "unresolved_tk_gradient_viability"
    if summary.get("passed") is not passed or summary.get("status") != expected_status:
        raise RuntimeError("Conv3 epochwise K64 aggregate outcome is inconsistent.")

    if require_canonical_result:
        cell_path = cell_dir / "cell.json"
        result_path = cell_dir / "result.json"
        status_path = cell_dir / "status.json"
        if not cell_path.is_file() or not result_path.is_file() or not status_path.is_file():
            raise RuntimeError("Conv3 candidate is missing its canonical completion bundle.")
        canonical_cell = json.loads(cell_path.read_text(encoding="utf-8"))
        if canonical_cell.get("signature") != signature["cell_signature"]:
            raise RuntimeError("Conv3 candidate cell signature changed after the K64 gate.")
        expected_cell_status = (
            "complete" if passed else "candidate_rejected_post_training_tk"
        )
        if canonical_cell.get("status") != expected_cell_status:
            raise RuntimeError("Conv3 candidate cell status disagrees with its K64 gate.")
        gate_record = canonical_cell.get("post_candidate_gate")
        if not isinstance(gate_record, Mapping):
            raise RuntimeError("Conv3 candidate cell lacks its post-candidate gate record.")
        if (
            gate_record.get("name") != "conv3_epochwise_k64_viability"
            or gate_record.get("status") != expected_status
            or gate_record.get("passed") is not passed
            or gate_record.get("artifact_sha256") != _sha256_file(summary_path)
        ):
            raise RuntimeError("Conv3 candidate post-candidate gate record is inconsistent.")
        if canonical_cell.get("selection_eligible") is not bool(
            canonical_cell.get("training_accuracy_selection_eligible") and passed
        ):
            raise RuntimeError("Conv3 candidate eligibility disagrees with its K64 gate.")
        validation_errors = validate_run(cell_dir)
        if validation_errors:
            raise RuntimeError(
                "Invalid canonical Conv3 candidate run: " + "; ".join(validation_errors)
            )
        canonical_status = json.loads(status_path.read_text(encoding="utf-8"))
        canonical_result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            canonical_status.get("state") != "complete"
            or canonical_status.get("result_sha256") != _sha256_file(result_path)
        ):
            raise RuntimeError("Conv3 candidate canonical status/result binding failed.")
        completion = canonical_result.get("completion")
        if not isinstance(completion, Mapping):
            raise RuntimeError("Conv3 candidate result lacks completion evidence.")
        if (
            completion.get("post_candidate_gate") != gate_record
            or completion.get("post_training_tk_admissible") is not passed
            or completion.get("criteria_met")
            is not canonical_cell.get("selection_eligible")
            or completion.get("official_test_read") is not False
        ):
            raise RuntimeError("Conv3 candidate canonical completion is inconsistent.")
        indexed = [
            artifact
            for artifact in canonical_result.get("artifacts", ())
            if artifact.get("path") == "artifacts/epochwise_k64_viability.json"
        ]
        if len(indexed) != 1 or indexed[0].get("sha256") != _sha256_file(summary_path):
            raise RuntimeError(
                "Conv3 candidate result does not uniquely index its flat K64 summary."
            )

    return {
        **dict(summary),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256_file(summary_path),
    }


def ensure_conv3_candidate_epochwise_viability(
    study: Mapping[str, Any],
    surface: Mapping[str, Any],
    cell_dir: Path,
    cell: Mapping[str, Any],
    *,
    device: str,
    smoke: bool,
    operating_point_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require K64 Conv-gradient viability after every promoted epoch."""

    from experiments.conv3_operating_point_gate import (
        run_conv3_k64_gradient_viability_gate,
    )

    if (
        not _is_zero_bias_study(study)
        or surface["architecture"] != "conv3"
    ):
        raise ValueError("Epochwise K64 viability is bound to the new Conv3 study.")
    if operating_point_gate is None:
        raise RuntimeError(
            "Conv3 epochwise replay requires the initialization operating-point gate."
        )
    source_path, checkpoints, expected_cohort, signature = (
        _conv3_candidate_gate_context(
            study,
            surface,
            cell_dir,
            cell,
            operating_point_gate,
            smoke=smoke,
        )
    )
    expected_epochs = int(signature["expected_epochs"])
    gate_root = cell_dir / "post_training_tk" / "epochwise_k64_viability"
    # Reporting materializes only flat root artifacts, so this SHA-binding
    # summary must live at the candidate root even though detailed receipts do not.
    summary_path = cell_dir / "epochwise_k64_viability.json"
    if summary_path.is_file():
        return _validate_conv3_candidate_epochwise_summary(
            study,
            surface,
            cell_dir,
            cell,
            operating_point_gate,
            smoke=smoke,
            require_canonical_result=False,
        )

    epoch_results = []
    for checkpoint in checkpoints:
        epoch = int(checkpoint["epoch"])
        output_path = gate_root / f"epoch_{epoch:03d}.json"
        returned_gate = run_conv3_k64_gradient_viability_gate(
            source_path,
            checkpoint["checkpoint"],
            device=device,
            output_path=output_path,
            smoke=smoke,
        )
        if not output_path.is_file():
            raise RuntimeError(
                f"Conv3 K64 gate did not write its epoch {epoch} receipt."
            )
        gate = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(returned_gate, Mapping) or any(
            returned_gate.get(key) != gate.get(key)
            for key in (
                "schema_version",
                "status",
                "viability_passed",
                "checkpoint_sha256",
                "source_config_sha256",
                "smoke",
                "official_test_read",
            )
        ):
            raise RuntimeError(
                f"Conv3 K64 epoch {epoch} returned/written receipt mismatch."
            )
        passed, observed_cohort = _validate_conv3_k64_gate_receipt(
            gate,
            checkpoint,
            source_config_sha256=signature["source_config_sha256"],
            expected_cohort=expected_cohort,
            smoke=smoke,
        )
        epoch_results.append(
            {
                **checkpoint,
                "gate_artifact": str(output_path.resolve()),
                "gate_artifact_relative_path": str(output_path.relative_to(cell_dir)),
                "gate_artifact_sha256": _sha256_file(output_path),
                "gate_status": gate.get("status"),
                "gradient_cohort": observed_cohort,
                "cohort_matches_operating_point": True,
                "passed": passed,
                "error": None,
            }
        )

    passed = bool(
        len(epoch_results) == expected_epochs
        and all(row["passed"] for row in epoch_results)
    )
    summary = {
        "schema_version": (
            "perfectdiode-conv3-bounded-uniform-epochwise-k64-viability/v1"
        ),
        "status": "complete" if passed else "unresolved_tk_gradient_viability",
        "passed": passed,
        "signature": signature,
        "candidate_cell": str(cell_dir.resolve()),
        "candidate_metrics": str((cell_dir / "metrics.json").resolve()),
        "candidate_metrics_sha256": _sha256_file(cell_dir / "metrics.json"),
        "epoch_checkpoint_index": str(
            (cell_dir / "epoch_checkpoint_index.jsonl").resolve()
        ),
        "epoch_checkpoint_index_sha256": _sha256_file(
            cell_dir / "epoch_checkpoint_index.jsonl"
        ),
        "expected_epochs": expected_epochs,
        "expected_gradient_cohort": expected_cohort,
        "epochs": epoch_results,
        "official_test_read": False,
    }
    _write_json(summary_path, summary)
    return _validate_conv3_candidate_epochwise_summary(
        study,
        surface,
        cell_dir,
        cell,
        operating_point_gate,
        smoke=smoke,
        require_canonical_result=False,
    )


def _validate_conv3_selected_post_summary(
    study: Mapping[str, Any],
    surface: Mapping[str, Any],
    selected: Mapping[str, Any],
    operating_point_gate: Mapping[str, Any],
    *,
    output_path: Path,
    smoke: bool,
    cell_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate a selected-candidate replay without rerunning any computation."""

    if cell_dir is None:
        cell_dir = Path(str(selected["path"])).expanduser().resolve()
    else:
        cell_dir = cell_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not output_path.is_file():
        raise RuntimeError(f"Missing selected Conv3 post-training summary: {output_path}.")
    cell_path = cell_dir / "cell.json"
    source_path = cell_dir / "source_config.json"
    if not cell_path.is_file() or not source_path.is_file():
        raise RuntimeError("Selected Conv3 candidate is missing cell/source evidence.")
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    if selected.get("cell_id") != cell_dir.name:
        raise RuntimeError("Selected Conv3 cell id does not match its candidate directory.")
    viability = _validate_conv3_candidate_epochwise_summary(
        study,
        surface,
        cell_dir,
        cell,
        operating_point_gate,
        smoke=smoke,
        require_canonical_result=True,
    )
    if viability.get("passed") is not True:
        raise RuntimeError("A Conv3 candidate with a failed K64 gate cannot be selected.")
    viability_path = cell_dir / "epochwise_k64_viability.json"
    expected_epochs = 1 if smoke else int(
        _surface_search(study, surface)["candidate_epochs"]
    )
    checkpoints = _candidate_epoch_records(cell_dir, expected_epochs=expected_epochs)
    expected_full_cohort = _conv3_full_cohort_binding(operating_point_gate)
    expected_gradient_cohort = _conv3_gradient_cohort_binding(operating_point_gate)
    viability_epochs = {
        int(row["epoch"]): row for row in viability.get("epochs", ())
    }
    signature = {
        "surface_id": surface["surface_id"],
        "selected_cell_id": selected["cell_id"],
        "cell_sha256": _sha256_file(cell_path),
        "source_config_sha256": _sha256_file(source_path),
        "epoch_checkpoint_index_sha256": _sha256_file(
            cell_dir / "epoch_checkpoint_index.jsonl"
        ),
        "epochwise_k64_viability_sha256": _sha256_file(viability_path),
        "operating_point_gate_sha256": _sha256_json(operating_point_gate),
        "expected_full_cohort": expected_full_cohort,
        "expected_epochs": expected_epochs,
        "smoke": smoke,
    }
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise RuntimeError("Selected Conv3 post-training summary must be a JSON object.")
    if summary.get("schema_version") != (
        "perfectdiode-conv3-bounded-uniform-selected-post-training-tk/v1"
    ):
        raise RuntimeError("Unexpected selected Conv3 post-training summary schema.")
    if summary.get("signature") != signature:
        raise RuntimeError("Selected Conv3 post-training summary signature is stale or corrupt.")
    if summary.get("surface_id") != surface["surface_id"]:
        raise RuntimeError("Selected Conv3 summary surface binding mismatch.")
    if summary.get("selected_cell_id") != selected["cell_id"]:
        raise RuntimeError("Selected Conv3 summary candidate binding mismatch.")
    if summary.get("selected_candidate_cell_sha256") != signature["cell_sha256"]:
        raise RuntimeError("Selected Conv3 summary cell hash mismatch.")
    if summary.get("source_config_sha256") != signature["source_config_sha256"]:
        raise RuntimeError("Selected Conv3 summary source-config hash mismatch.")
    if (
        summary.get("candidate_epochwise_k64_viability_sha256")
        != signature["epochwise_k64_viability_sha256"]
        or summary.get("candidate_epochwise_k64_viability_bound") is not True
    ):
        raise RuntimeError("Selected Conv3 summary is not bound to its K64 evidence.")
    if summary.get("operating_point_gate_checkpoint_sha256") != (
        operating_point_gate.get("checkpoint_sha256")
    ):
        raise RuntimeError("Selected Conv3 summary initialization checkpoint mismatch.")
    if summary.get("expected_full_cohort") != expected_full_cohort:
        raise RuntimeError("Selected Conv3 summary full-cohort binding mismatch.")
    if summary.get("expected_epochs") != expected_epochs:
        raise RuntimeError("Selected Conv3 summary epoch-count mismatch.")
    if summary.get("official_test_read") is not False:
        raise RuntimeError("Selected Conv3 summary violates official-test isolation.")

    rows = summary.get("epochs")
    if not isinstance(rows, list) or len(rows) != expected_epochs:
        raise RuntimeError("Selected Conv3 summary has incomplete epoch coverage.")
    rows_by_epoch: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("epoch"), int):
            raise RuntimeError("Selected Conv3 summary has an invalid epoch row.")
        epoch = int(row["epoch"])
        if epoch in rows_by_epoch:
            raise RuntimeError(f"Duplicate selected Conv3 replay epoch {epoch}.")
        rows_by_epoch[epoch] = row
    expected_epoch_set = set(range(1, expected_epochs + 1))
    if set(rows_by_epoch) != expected_epoch_set:
        raise RuntimeError("Selected Conv3 summary has the wrong epoch set.")

    observed_passes: list[bool] = []
    for checkpoint in checkpoints:
        epoch = int(checkpoint["epoch"])
        row = rows_by_epoch[epoch]
        for key in (
            "checkpoint_relative_path",
            "checkpoint_sha256",
            "optimizer_checkpoint_relative_path",
            "optimizer_checkpoint_sha256",
            "validation_loss",
            "validation_accuracy",
        ):
            if row.get(key) != checkpoint[key]:
                raise RuntimeError(
                    f"Selected Conv3 epoch {epoch} has a stale {key} binding."
                )
        expected_relative = f"post_training_tk/selected_epochs/epoch_{epoch:03d}.json"
        if row.get("gate_artifact_relative_path") != expected_relative:
            raise RuntimeError(
                f"Selected Conv3 epoch {epoch} has an unexpected receipt path."
            )
        receipt_path = _artifact_beneath(
            output_path.parent,
            expected_relative,
            label=f"selected epoch {epoch} Conv3 full T/K receipt",
        )
        if row.get("gate_artifact_sha256") != _sha256_file(receipt_path):
            raise RuntimeError(f"Selected Conv3 epoch {epoch} receipt hash mismatch.")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, Mapping):
            raise RuntimeError(
                f"Selected Conv3 epoch {epoch} receipt is not a JSON object."
            )
        receipt_passed, observed_cohort = _validate_conv3_full_gate_receipt(
            receipt,
            checkpoint,
            source_config_sha256=signature["source_config_sha256"],
            expected_cohort=expected_full_cohort,
            smoke=smoke,
        )
        k64_epoch = viability_epochs.get(epoch)
        if not isinstance(k64_epoch, Mapping):
            raise RuntimeError(f"Selected Conv3 epoch {epoch} lacks K64 evidence.")
        if (
            k64_epoch.get("passed") is not True
            or k64_epoch.get("checkpoint_sha256") != checkpoint["checkpoint_sha256"]
            or k64_epoch.get("gradient_cohort") != expected_gradient_cohort
        ):
            raise RuntimeError(f"Selected Conv3 epoch {epoch} K64 binding failed.")
        if row.get("gate_status") != receipt.get("status"):
            raise RuntimeError(f"Selected Conv3 epoch {epoch} status binding mismatch.")
        if row.get("full_cohort") != observed_cohort:
            raise RuntimeError(f"Selected Conv3 epoch {epoch} cohort record mismatch.")
        if row.get("cohort_matches_operating_point") is not True:
            raise RuntimeError(f"Selected Conv3 epoch {epoch} lacks its cohort proof.")
        if row.get("epochwise_k64_receipt_bound") is not True:
            raise RuntimeError(f"Selected Conv3 epoch {epoch} lacks its K64 receipt proof.")
        if row.get("passed") is not receipt_passed or row.get("error") is not None:
            raise RuntimeError(f"Selected Conv3 epoch {epoch} outcome record mismatch.")
        observed_passes.append(receipt_passed)

    passed = bool(observed_passes and all(observed_passes))
    expected_status = "complete" if passed else "unresolved_post_training_tk"
    if summary.get("passed") is not passed or summary.get("status") != expected_status:
        raise RuntimeError("Selected Conv3 aggregate post-training outcome is inconsistent.")
    return {
        **dict(summary),
        "summary_path": str(output_path),
        "summary_sha256": _sha256_file(output_path),
    }


def run_conv3_selected_post_training_tk(
    study: Mapping[str, Any],
    surface: Mapping[str, Any],
    selected: Mapping[str, Any],
    operating_point_gate: Mapping[str, Any],
    *,
    output_path: Path,
    device: str,
    smoke: bool = False,
) -> dict[str, Any]:
    """Replay every selected candidate epoch at T8/T64 and K8/K64."""

    from experiments.conv3_operating_point_gate import (
        run_conv3_operating_point_gate,
    )

    if (
        not _is_zero_bias_study(study)
        or surface["architecture"] != "conv3"
    ):
        raise ValueError("Selected post-training T/K replay is Conv3-only.")
    _validate_conv3_initial_operating_point_gate(operating_point_gate, smoke=smoke)
    cell_dir = Path(str(selected["path"])).expanduser().resolve()
    cell_path = cell_dir / "cell.json"
    source_path = cell_dir / "source_config.json"
    viability_path = cell_dir / "epochwise_k64_viability.json"
    if not cell_path.is_file() or not source_path.is_file() or not viability_path.is_file():
        raise RuntimeError(
            "Selected Conv3 replay requires the candidate cell, source config, "
            "and epochwise K64 viability summary."
        )
    cell = json.loads(cell_path.read_text(encoding="utf-8"))
    viability = _validate_conv3_candidate_epochwise_summary(
        study,
        surface,
        cell_dir,
        cell,
        operating_point_gate,
        smoke=smoke,
        require_canonical_result=True,
    )
    if viability.get("passed") is not True:
        raise RuntimeError("A Conv3 candidate with a failed K64 gate cannot be selected.")
    expected_epochs = 1 if smoke else int(_surface_search(study, surface)["candidate_epochs"])
    checkpoints = _candidate_epoch_records(cell_dir, expected_epochs=expected_epochs)
    expected_full_cohort = _conv3_full_cohort_binding(operating_point_gate)
    expected_gradient_cohort = _conv3_gradient_cohort_binding(
        operating_point_gate
    )
    viability_epochs = {
        int(row["epoch"]): row for row in viability.get("epochs", ())
    }
    candidate_viability_bound = True
    signature = {
        "surface_id": surface["surface_id"],
        "selected_cell_id": selected["cell_id"],
        "cell_sha256": _sha256_file(cell_path),
        "source_config_sha256": _sha256_file(source_path),
        "epoch_checkpoint_index_sha256": _sha256_file(
            cell_dir / "epoch_checkpoint_index.jsonl"
        ),
        "epochwise_k64_viability_sha256": _sha256_file(viability_path),
        "operating_point_gate_sha256": _sha256_json(operating_point_gate),
        "expected_full_cohort": expected_full_cohort,
        "expected_epochs": expected_epochs,
        "smoke": smoke,
    }
    if output_path.is_file():
        return _validate_conv3_selected_post_summary(
            study,
            surface,
            selected,
            operating_point_gate,
            output_path=output_path,
            smoke=smoke,
            cell_dir=cell_dir,
        )

    receipt_root = output_path.parent / "post_training_tk" / "selected_epochs"
    epoch_results = []
    for checkpoint in checkpoints:
        epoch = int(checkpoint["epoch"])
        receipt_path = receipt_root / f"epoch_{epoch:03d}.json"
        returned_gate = run_conv3_operating_point_gate(
            source_path,
            checkpoint["checkpoint"],
            device=device,
            output_path=receipt_path,
            smoke=smoke,
        )
        if not receipt_path.is_file():
            raise RuntimeError(
                f"Selected Conv3 gate did not write its epoch {epoch} receipt."
            )
        gate = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(returned_gate, Mapping) or any(
            returned_gate.get(key) != gate.get(key)
            for key in (
                "schema_version",
                "status",
                "security_passed",
                "checkpoint_sha256",
                "source_config_sha256",
                "smoke",
                "official_test_read",
            )
        ):
            raise RuntimeError(
                f"Selected Conv3 epoch {epoch} returned/written receipt mismatch."
            )
        passed, observed_full_cohort = _validate_conv3_full_gate_receipt(
            gate,
            checkpoint,
            source_config_sha256=signature["source_config_sha256"],
            expected_cohort=expected_full_cohort,
            smoke=smoke,
        )
        k64_epoch = viability_epochs.get(epoch)
        if not isinstance(k64_epoch, Mapping):
            raise RuntimeError(f"Selected Conv3 epoch {epoch} lacks K64 evidence.")
        if (
            k64_epoch.get("passed") is not True
            or k64_epoch.get("checkpoint_sha256")
            != checkpoint["checkpoint_sha256"]
            or k64_epoch.get("gradient_cohort") != expected_gradient_cohort
        ):
            raise RuntimeError(f"Selected Conv3 epoch {epoch} K64 binding failed.")
        epoch_results.append(
            {
                **checkpoint,
                "gate_artifact": str(receipt_path.resolve()),
                "gate_artifact_relative_path": str(
                    receipt_path.relative_to(output_path.parent)
                ),
                "gate_artifact_sha256": _sha256_file(receipt_path),
                "gate_status": gate.get("status"),
                "full_cohort": observed_full_cohort,
                "cohort_matches_operating_point": True,
                "epochwise_k64_receipt_bound": True,
                "passed": passed,
                "error": None,
            }
        )
    passed = bool(
        candidate_viability_bound
        and len(epoch_results) == expected_epochs
        and all(row["passed"] for row in epoch_results)
    )
    summary = {
        "schema_version": (
            "perfectdiode-conv3-bounded-uniform-selected-post-training-tk/v1"
        ),
        "status": "complete" if passed else "unresolved_post_training_tk",
        "passed": passed,
        "signature": signature,
        "surface_id": surface["surface_id"],
        "selected_cell_id": selected["cell_id"],
        "selected_candidate": str(cell_dir),
        "selected_candidate_cell_sha256": _sha256_file(cell_path),
        "source_config": str(source_path.resolve()),
        "source_config_sha256": _sha256_file(source_path),
        "candidate_epochwise_k64_viability": str(viability_path.resolve()),
        "candidate_epochwise_k64_viability_sha256": _sha256_file(
            viability_path
        ),
        "candidate_epochwise_k64_viability_bound": candidate_viability_bound,
        "operating_point_gate_checkpoint_sha256": operating_point_gate.get(
            "checkpoint_sha256"
        ),
        "expected_full_cohort": expected_full_cohort,
        "expected_epochs": expected_epochs,
        "epochs": epoch_results,
        "official_test_read": False,
    }
    _write_json(output_path, summary)
    return _validate_conv3_selected_post_summary(
        study,
        surface,
        selected,
        operating_point_gate,
        output_path=output_path,
        smoke=smoke,
        cell_dir=cell_dir,
    )


def _validate_conv3_selection_post_evidence(
    study: Mapping[str, Any],
    surface: Mapping[str, Any],
    selection_path: Path,
    operating_point_gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate selection, candidate, and both layers of post-training evidence."""

    selection_path = selection_path.expanduser().resolve()
    if not selection_path.is_file():
        raise RuntimeError(f"Missing Conv3 surface selection: {selection_path}.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, Mapping):
        raise RuntimeError("Conv3 surface selection must be a JSON object.")
    if selection.get("schema_version") != _artifact_schema(study, "surface"):
        raise RuntimeError("Unexpected Conv3 surface-selection schema.")
    for key in ("surface_id", "initializer", "architecture", "scheme", "optimizer"):
        if selection.get(key) != surface.get(key):
            raise RuntimeError(f"Conv3 selection {key} binding mismatch.")
    if selection.get("official_test_read") is not False:
        raise RuntimeError("Conv3 selection violates official-test isolation.")
    _validate_conv3_initial_operating_point_gate(
        operating_point_gate, smoke=False
    )

    post_record = selection.get("post_training_tk")
    if not isinstance(post_record, Mapping):
        raise RuntimeError("Conv3 post-training selection lacks its summary record.")
    status = selection.get("status")
    passed = post_record.get("passed") is True
    selector = selection.get("selection")
    if not isinstance(selector, Mapping):
        raise RuntimeError("Conv3 post-training selection lacks selector evidence.")
    if passed:
        if status not in {
            "complete",
            "complete_below_accuracy_bracketed",
            "complete_below_accuracy_range_bounded",
        }:
            raise RuntimeError("Passing Conv3 post-training evidence has invalid status.")
        selected = selection.get("selected")
        provisional = selection.get("pre_post_training_tk_selected")
        if not isinstance(selected, Mapping) or provisional != selected:
            raise RuntimeError("Passing Conv3 selection lost its selected candidate binding.")
        if selector.get("selected") != selected:
            raise RuntimeError("Passing Conv3 selector disagrees with the selected candidate.")
        if selector.get("post_training_tk_passed") is not True:
            raise RuntimeError("Passing Conv3 selector lacks its post-training proof.")
    else:
        if status != "unresolved_post_training_tk":
            raise RuntimeError("Failed Conv3 post-training evidence has invalid status.")
        if selection.get("selected") is not None or selector.get("selected") is not None:
            raise RuntimeError("Failed Conv3 post-training evidence retained a selection.")
        provisional = selection.get("pre_post_training_tk_selected")
        if not isinstance(provisional, Mapping):
            raise RuntimeError("Failed Conv3 post-training evidence lost its candidate binding.")
        if selector.get("pre_post_training_tk_selected") != provisional:
            raise RuntimeError("Failed Conv3 selector has a mismatched provisional candidate.")
        if selector.get("post_training_tk_passed") is not False:
            raise RuntimeError("Failed Conv3 selector lacks its post-training outcome.")
        selected = provisional

    cell_id = str(selected.get("cell_id", ""))
    if not cell_id or Path(cell_id).name != cell_id:
        raise RuntimeError("Conv3 selection contains an invalid candidate cell id.")
    cell_dir = selection_path.parent / "rho" / "cells" / cell_id
    recorded_cell_path = Path(str(selected.get("path", ""))).expanduser().resolve()
    if recorded_cell_path != cell_dir.resolve():
        raise RuntimeError(
            "Conv3 selected candidate path does not match the trusted surface cell."
        )
    output_path = selection_path.parent / "selected_post_training_tk.json"
    summary = _validate_conv3_selected_post_summary(
        study,
        surface,
        selected,
        operating_point_gate,
        output_path=output_path,
        smoke=False,
        cell_dir=cell_dir,
    )
    if summary.get("passed") is not passed:
        raise RuntimeError("Conv3 selection and selected post-training summary disagree.")
    summary_document = json.loads(output_path.read_text(encoding="utf-8"))
    expected_post_keys = set(summary_document) | {"summary_path", "summary_sha256"}
    if set(post_record) != expected_post_keys:
        raise RuntimeError("Conv3 embedded post-training summary has unexpected fields.")
    for key, value in summary_document.items():
        if post_record.get(key) != value:
            raise RuntimeError(
                f"Conv3 embedded post-training summary differs at {key!r}."
            )
    recorded_path = Path(str(post_record.get("summary_path", "")))
    if recorded_path.name != output_path.name:
        raise RuntimeError("Conv3 embedded post-training summary path is inconsistent.")
    if post_record.get("summary_sha256") != _sha256_file(output_path):
        raise RuntimeError("Conv3 embedded post-training summary hash mismatch.")
    if (
        selected.get("status") != "complete"
        or selected.get("selection_eligible") is not True
    ):
        raise RuntimeError("Conv3 provisional candidate is not complete and eligible.")
    canonical_cell = json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))
    reconstructed_selected = _candidate_record(cell_dir, canonical_cell)
    if dict(selected) != reconstructed_selected:
        raise RuntimeError(
            "Conv3 selected candidate record does not match canonical candidate evidence."
        )
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or sum(
        candidate == selected for candidate in candidates
    ) != 1:
        raise RuntimeError(
            "Conv3 selected candidate is not uniquely present in the surface candidates."
        )

    return {
        "schema_version": (
            "perfectdiode-conv3-selected-post-training-tk-validation/v1"
        ),
        "status": "valid",
        "surface_id": surface["surface_id"],
        "post_training_tk_passed": passed,
        "selected_cell_id": cell_id,
        "selected_run_dir": str(cell_dir.resolve()),
        "epoch_count": len(summary["epochs"]),
        "selection_sha256": _sha256_file(selection_path),
        "selected_post_training_tk_sha256": _sha256_file(output_path),
        "epochwise_k64_viability_sha256": summary[
            "candidate_epochwise_k64_viability_sha256"
        ],
        "canonical_candidate_result_sha256": _sha256_file(
            cell_dir / "result.json"
        ),
        "operating_point_gate_sha256": _sha256_json(operating_point_gate),
        "official_test_read": False,
    }


def validate_conv3_selected_post_training_tk_evidence(
    study_path: str | Path,
    selection_path: str | Path,
    operating_point_gate_path: str | Path,
) -> dict[str, Any]:
    """Public read-only transport validator for terminal Conv3 post-T/K evidence."""

    _resolved_study_path, study = load_study(study_path)
    resolved_selection_path = Path(selection_path).expanduser().resolve()
    selection = json.loads(resolved_selection_path.read_text(encoding="utf-8"))
    surface_id = selection.get("surface_id")
    matches = [
        surface for surface in surface_specs(study) if surface["surface_id"] == surface_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Unknown or ambiguous Conv3 surface id: {surface_id!r}.")
    surface = matches[0]
    if surface["architecture"] != "conv3":
        raise ValueError("The Conv3 post-training validator received a non-Conv3 surface.")
    gate_path = Path(operating_point_gate_path).expanduser().resolve()
    if not gate_path.is_file():
        raise RuntimeError(f"Missing Conv3 initialization operating-point gate: {gate_path}.")
    operating_point_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not isinstance(operating_point_gate, Mapping):
        raise RuntimeError("Conv3 initialization operating-point gate must be a JSON object.")
    return _validate_conv3_selection_post_evidence(
        study,
        surface,
        resolved_selection_path,
        operating_point_gate,
    )


def _candidate_record(cell_dir: Path, cell: Mapping[str, Any]) -> dict[str, Any]:
    record = {
        "cell_id": cell_dir.name,
        "index": int(cell["index"]),
        "rho_conv": float(cell["rho_conv"]),
        "rho_dense": float(cell["rho_dense"]),
        "status": cell["status"],
        "selection_eligible": bool(cell.get("selection_eligible", False)),
        "final_validation_loss": None,
        "final_validation_accuracy": None,
        "median_projection_efficiency": None,
        "path": str(cell_dir),
    }
    metrics_path = cell_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        record["final_validation_loss"] = metrics.get("final_test_loss")
        record["final_validation_accuracy"] = metrics.get("final_test_accuracy")
    diagnostics_path = cell_dir / "safety_diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        record["median_projection_efficiency"] = diagnostics.get(
            "median_projection_efficiency"
        )
    viability_path = (
        cell_dir / "epochwise_k64_viability.json"
    )
    if viability_path.is_file():
        viability = json.loads(viability_path.read_text(encoding="utf-8"))
        record["training_accuracy_selection_eligible"] = bool(
            cell.get(
                "training_accuracy_selection_eligible",
                record["selection_eligible"],
            )
        )
        record["epochwise_conv_gradient_viability"] = {
            "path": str(viability_path.resolve()),
            "sha256": _sha256_file(viability_path),
            "status": viability.get("status"),
            "passed": viability.get("passed") is True,
            "expected_epochs": viability.get("expected_epochs"),
            "checked_epochs": [row.get("epoch") for row in viability.get("epochs", ())],
        }
        record["selection_eligible"] = bool(
            record["selection_eligible"] and viability.get("passed") is True
        )
    return record


def select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    plateau_fraction: float,
    *,
    select_best_safe_below_accuracy: bool = False,
) -> dict[str, Any]:
    accuracy_eligible = [
        dict(candidate)
        for candidate in candidates
        if candidate.get("selection_eligible")
        and candidate.get("final_validation_loss") is not None
    ]
    eligible = accuracy_eligible
    selection_basis = "accuracy_gate"
    if not eligible and select_best_safe_below_accuracy:
        eligible = [
            dict(candidate)
            for candidate in candidates
            if candidate.get("status") == "complete"
            and candidate.get("final_validation_loss") is not None
            and candidate.get("final_validation_accuracy") is not None
        ]
        selection_basis = "best_safe_below_accuracy"
    if not eligible:
        return {
            "eligible": [],
            "plateau": [],
            "selected": None,
            "selection_basis": "none",
            "accuracy_gate_met": False,
        }
    best_loss = min(float(candidate["final_validation_loss"]) for candidate in eligible)
    plateau = [
        candidate
        for candidate in eligible
        if float(candidate["final_validation_loss"])
        <= best_loss * (1.0 + float(plateau_fraction))
    ]

    def key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        efficiency = candidate.get("median_projection_efficiency")
        return (
            -float(candidate["final_validation_accuracy"]),
            -float(efficiency) if efficiency is not None else math.inf,
            max(float(candidate["rho_conv"]), float(candidate["rho_dense"])),
            float(candidate["rho_conv"]) + float(candidate["rho_dense"]),
            float(candidate["rho_conv"]),
            float(candidate["rho_dense"]),
            str(candidate["cell_id"]),
        )

    return {
        "eligible": eligible,
        "minimum_final_validation_loss": best_loss,
        "plateau": plateau,
        "selected": min(plateau, key=key),
        "selection_basis": selection_basis,
        "accuracy_gate_met": bool(accuracy_eligible),
    }


def _confined_edge(
    plateau: Sequence[Mapping[str, Any]],
    axis: str,
    values: Sequence[float],
) -> str | None:
    observed = [float(candidate[axis]) for candidate in plateau]
    if observed and all(_same_rho(value, min(values)) for value in observed):
        return "lower"
    if observed and all(_same_rho(value, max(values)) for value in observed):
        return "upper"
    return None


def _same_rho(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-15)


def _selection_edge(
    selection: Mapping[str, Any],
    axis: str,
    values: Sequence[float],
) -> str | None:
    if selection.get("selection_basis") != "best_safe_below_accuracy":
        return _confined_edge(selection["plateau"], axis, values)
    selected = float(selection["selected"][axis])
    if _same_rho(selected, min(values)):
        return "lower"
    if _same_rho(selected, max(values)):
        return "upper"
    return None


def _maximum_safe_accuracy(candidates: Sequence[Mapping[str, Any]]) -> float | None:
    accuracies = [
        float(candidate["final_validation_accuracy"])
        for candidate in candidates
        if candidate.get("status") == "complete"
        and candidate.get("final_validation_accuracy") is not None
    ]
    return max(accuracies) if accuracies else None


def _accuracy_trend_edge(
    candidates: Sequence[Mapping[str, Any]],
    axis: str,
    values: Sequence[float],
) -> str | None:
    """Choose one outward direction from safety-clean edge accuracy evidence."""

    edge_accuracy: dict[str, float] = {}
    for edge, value in (("lower", min(values)), ("upper", max(values))):
        accuracies = [
            float(candidate["final_validation_accuracy"])
            for candidate in candidates
            if candidate.get("status") == "complete"
            and candidate.get("final_validation_accuracy") is not None
            and _same_rho(float(candidate[axis]), float(value))
        ]
        if accuracies:
            edge_accuracy[edge] = max(accuracies)
    if not edge_accuracy:
        return None
    if len(edge_accuracy) == 1:
        return next(iter(edge_accuracy))
    if edge_accuracy["upper"] > edge_accuracy["lower"]:
        return "upper"
    return "lower"


def run_rho_surface(
    study: Mapping[str, Any],
    output_root: Path,
    surface: Mapping[str, Any],
    source_config: Path,
    *,
    device: str,
    target: str | None = None,
    operating_point_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    surface_dir = output_root / "surfaces" / surface["surface_id"]
    selection_path = surface_dir / "selection.json"
    if selection_path.exists():
        existing = json.loads(selection_path.read_text(encoding="utf-8"))
        conv3_post_training_surface = bool(
            _is_zero_bias_study(study)
            and surface["architecture"] == "conv3"
        )
        post_record = existing.get("post_training_tk")
        if not conv3_post_training_surface:
            return existing
        if isinstance(post_record, Mapping):
            if operating_point_gate is None:
                raise RuntimeError(
                    "Resuming a Conv3 post-training selection requires its "
                    "initialization operating-point gate."
                )
            _validate_conv3_selection_post_evidence(
                study,
                surface,
                selection_path,
                operating_point_gate,
            )
            return existing
        elif existing.get("selected") is None:
            if existing.get("status") == "unresolved_post_training_tk":
                raise RuntimeError(
                    "Conv3 unresolved_post_training_tk selection is missing its evidence."
                )
            return existing
        else:
            raise RuntimeError(
                "Existing Conv3 selection retained a candidate without post-training evidence."
            )
    rho_root = surface_dir / "rho"
    search = _surface_search(study, surface)
    rho_conv_axis, rho_dense_axis = _rho_axes(study, surface)
    probe_result = run_rho_search(
        _rho_args(
            study,
            surface,
            source_config,
            rho_root,
            rho_conv_axis,
            rho_dense_axis,
            device=device,
            target=target,
            index=None,
            probe_only=True,
        )
    )
    if probe_result.get("status") != "complete":
        result = {
            "schema_version": _artifact_schema(study, "surface"),
            **dict(surface),
            "status": "unresolved_probe",
            "probe": probe_result,
            "candidates": [],
            "selected": None,
            "official_test_read": False,
        }
        _write_json(selection_path, result)
        return result

    core_mode = search.get("core_mode", "adaptive_safe_center")
    safe_center = None
    center_attempts: list[dict[str, Any]] = []
    if core_mode == "adaptive_safe_center":
        center_conv = float(search["center"]["rho_conv"])
        center_dense = float(search["center"]["rho_dense"])
        for attempt in range(int(search["maximum_center_attempts"])):
            rho_conv = center_conv / (3.0**attempt)
            rho_dense = center_dense / (3.0**attempt)
            cell_kwargs: dict[str, Any] = {
                "device": device,
                "canary_only": True,
            }
            if operating_point_gate is not None:
                cell_kwargs["operating_point_gate"] = operating_point_gate
            if target is not None:
                cell_kwargs["target"] = target
            cell_dir, cell = _run_rho_cell(
                study,
                surface,
                source_config,
                rho_root,
                rho_conv_axis,
                rho_dense_axis,
                rho_conv,
                rho_dense,
                **cell_kwargs,
            )
            center_attempts.append(
                {
                    "attempt": attempt + 1,
                    "rho_conv": float(cell["rho_conv"]),
                    "rho_dense": float(cell["rho_dense"]),
                    "status": cell["status"],
                    "path": str(cell_dir),
                }
            )
            if _has_clean_canary(cell_dir, cell):
                safe_center = (float(cell["rho_conv"]), float(cell["rho_dense"]))
                break
        if safe_center is None:
            result = {
                "schema_version": _artifact_schema(study, "surface"),
                **dict(surface),
                "core_mode": core_mode,
                "center_attempts": center_attempts,
                "safe_center": None,
                "status": "unresolved_no_safe_center",
                "candidates": [],
                "selected": None,
                "official_test_read": False,
            }
            _write_json(selection_path, result)
            return result
        core_conv = sorted(
            rho_conv_axis[
                _rho_index(
                    rho_conv_axis,
                    rho_dense_axis,
                    safe_center[0] * factor,
                    safe_center[1],
                )
                // len(rho_dense_axis)
            ]
            for factor in search["core_factors"]
        )
        core_dense = sorted(
            rho_dense_axis[
                _rho_index(
                    rho_conv_axis,
                    rho_dense_axis,
                    safe_center[0],
                    safe_center[1] * factor,
                )
                % len(rho_dense_axis)
            ]
            for factor in search["core_factors"]
        )
    elif core_mode == "fixed_grid":
        core_conv = sorted(float(value) for value in search["fixed_core"]["rho_conv"])
        core_dense = sorted(float(value) for value in search["fixed_core"]["rho_dense"])
    else:
        raise ValueError(f"Unsupported rho core mode: {core_mode!r}.")

    result: dict[str, Any] = {
        "schema_version": _artifact_schema(study, "surface"),
        **dict(surface),
        "core_mode": core_mode,
        "center_attempts": center_attempts,
        "safe_center": (
            {"rho_conv": safe_center[0], "rho_dense": safe_center[1]}
            if safe_center is not None
            else None
        ),
        "official_test_read": False,
    }
    candidate_by_pair: dict[tuple[float, float], dict[str, Any]] = {}
    for rho_conv in core_conv:
        for rho_dense in core_dense:
            cell_kwargs = {
                "device": device,
                "canary_only": False,
            }
            if operating_point_gate is not None:
                cell_kwargs["operating_point_gate"] = operating_point_gate
            if target is not None:
                cell_kwargs["target"] = target
            cell_dir, cell = _run_rho_cell(
                study,
                surface,
                source_config,
                rho_root,
                rho_conv_axis,
                rho_dense_axis,
                rho_conv,
                rho_dense,
                **cell_kwargs,
            )
            candidate_by_pair[(rho_conv, rho_dense)] = _candidate_record(cell_dir, cell)

    selection = select_candidates(
        list(candidate_by_pair.values()),
        float(search["inclusive_loss_plateau"]),
        select_best_safe_below_accuracy=bool(
            search.get("select_best_safe_below_accuracy", False)
        ),
    )
    if selection["selected"] is None:
        result.update(
            status="unresolved_no_safe_completed_core_candidate",
            candidates=list(candidate_by_pair.values()),
            selection=selection,
            selected=None,
        )
        _write_json(selection_path, result)
        return result

    conv_edge = _selection_edge(selection, "rho_conv", core_conv)
    dense_edge = _selection_edge(selection, "rho_dense", core_dense)
    core_maximum_safe_accuracy = _maximum_safe_accuracy(
        list(candidate_by_pair.values())
    )
    suspicious_floor_value = search.get("suspicious_validation_accuracy_floor")
    suspicious_floor = (
        None
        if suspicious_floor_value is None
        else float(suspicious_floor_value)
    )
    suspicious_policy_enabled = bool(
        search.get("select_best_safe_below_accuracy", False)
        and suspicious_floor is not None
    )
    suspicious_accuracy_triggered = bool(
        suspicious_policy_enabled
        and core_maximum_safe_accuracy is not None
        and core_maximum_safe_accuracy < suspicious_floor
    )
    if suspicious_accuracy_triggered:
        if conv_edge is None:
            conv_edge = _accuracy_trend_edge(
                list(candidate_by_pair.values()), "rho_conv", core_conv
            )
        if dense_edge is None:
            dense_edge = _accuracy_trend_edge(
                list(candidate_by_pair.values()), "rho_dense", core_dense
            )
    expanded_conv = list(core_conv)
    expanded_dense = list(core_dense)
    new_pairs: set[tuple[float, float]] = set()
    if conv_edge is not None:
        new_conv = (
            min(core_conv) / 3.0
            if conv_edge == "lower"
            else max(core_conv) * 3.0
        )
        new_conv = rho_conv_axis[
            _rho_index(
                rho_conv_axis,
                rho_dense_axis,
                new_conv,
                core_dense[len(core_dense) // 2],
            )
            // len(rho_dense_axis)
        ]
        expanded_conv.append(new_conv)
        expanded_conv.sort()
        new_pairs.update((new_conv, value) for value in expanded_dense)
    if dense_edge is not None:
        new_dense = (
            min(core_dense) / 3.0
            if dense_edge == "lower"
            else max(core_dense) * 3.0
        )
        new_dense = rho_dense_axis[
            _rho_index(
                rho_conv_axis,
                rho_dense_axis,
                core_conv[len(core_conv) // 2],
                new_dense,
            )
            % len(rho_dense_axis)
        ]
        expanded_dense.append(new_dense)
        expanded_dense.sort()
        new_pairs.update((value, new_dense) for value in expanded_conv)

    for rho_conv, rho_dense in sorted(new_pairs):
        cell_kwargs = {
            "device": device,
            "canary_only": False,
        }
        if operating_point_gate is not None:
            cell_kwargs["operating_point_gate"] = operating_point_gate
        if target is not None:
            cell_kwargs["target"] = target
        cell_dir, cell = _run_rho_cell(
            study,
            surface,
            source_config,
            rho_root,
            rho_conv_axis,
            rho_dense_axis,
            rho_conv,
            rho_dense,
            **cell_kwargs,
        )
        candidate_by_pair[(rho_conv, rho_dense)] = _candidate_record(cell_dir, cell)

    final_selection = select_candidates(
        list(candidate_by_pair.values()),
        float(search["inclusive_loss_plateau"]),
        select_best_safe_below_accuracy=bool(
            search.get("select_best_safe_below_accuracy", False)
        ),
    )
    outer_conv_edge = _selection_edge(
        final_selection, "rho_conv", expanded_conv
    )
    outer_dense_edge = _selection_edge(
        final_selection, "rho_dense", expanded_dense
    )
    range_bounded = bool(
        outer_conv_edge is not None or outer_dense_edge is not None
    )
    unresolved_boundary = bool(new_pairs) and range_bounded
    below_accuracy_fallback = (
        final_selection["selection_basis"] == "best_safe_below_accuracy"
    )
    final_maximum_safe_accuracy = _maximum_safe_accuracy(
        list(candidate_by_pair.values())
    )
    if below_accuracy_fallback:
        status = (
            "complete_below_accuracy_range_bounded"
            if range_bounded
            else "complete_below_accuracy_bracketed"
        )
        selected = final_selection["selected"]
    else:
        status = (
            "unresolved_after_boundary_expansion"
            if unresolved_boundary
            else "complete"
        )
        selected = None if unresolved_boundary else final_selection["selected"]
    post_training_tk = None
    pre_post_training_tk_selected = None
    if (
        selected is not None
        and _is_zero_bias_study(study)
        and surface["architecture"] == "conv3"
    ):
        pre_post_training_tk_selected = copy.deepcopy(selected)
        post_path = surface_dir / "selected_post_training_tk.json"
        if operating_point_gate is None:
            raise RuntimeError(
                "Selected Conv3 replay requires the initialization operating-point gate."
            )
        post_training_tk = run_conv3_selected_post_training_tk(
            study,
            surface,
            selected,
            operating_point_gate,
            output_path=post_path,
            device=device,
            smoke=False,
        )
        if post_training_tk.get("passed") is not True:
            status = "unresolved_post_training_tk"
            selected = None
            final_selection = copy.deepcopy(final_selection)
            final_selection["pre_post_training_tk_selected"] = copy.deepcopy(
                final_selection.get("selected")
            )
            final_selection["selected"] = None
            final_selection["post_training_tk_passed"] = False
        else:
            final_selection = copy.deepcopy(final_selection)
            final_selection["post_training_tk_passed"] = True
    result.update(
        status=status,
        core_axes={"rho_conv": core_conv, "rho_dense": core_dense},
        expansion={
            "triggered": bool(new_pairs),
            "trigger": (
                "suspicious_low_accuracy"
                if suspicious_accuracy_triggered
                else "selected_boundary"
                if new_pairs
                else None
            ),
            "rho_conv_edge": conv_edge,
            "rho_dense_edge": dense_edge,
            "expanded_axes": {
                "rho_conv": expanded_conv,
                "rho_dense": expanded_dense,
            },
            "new_cell_count": len(new_pairs),
            "plateau_on_outer_rho_conv_edge": outer_conv_edge,
            "plateau_on_outer_rho_dense_edge": outer_dense_edge,
        },
        rho_range_status={
            "classification": "bounded" if range_bounded else "unbounded",
            "bracketed": not range_bounded,
            "rho_conv_edge": outer_conv_edge,
            "rho_dense_edge": outer_dense_edge,
        },
        suspicious_accuracy_status=(
            {
                "floor": suspicious_floor,
                "triggered": suspicious_accuracy_triggered,
                "core_maximum_safe_accuracy": core_maximum_safe_accuracy,
                "final_maximum_safe_accuracy": final_maximum_safe_accuracy,
                "remains_below_floor": bool(
                    final_maximum_safe_accuracy is not None
                    and final_maximum_safe_accuracy < suspicious_floor
                ),
                "direction_policy": "better_safety_clean_core_edge_per_axis",
            }
            if suspicious_policy_enabled
            else None
        ),
        candidates=sorted(
            candidate_by_pair.values(),
            key=lambda item: (item["rho_conv"], item["rho_dense"]),
        ),
        selection=final_selection,
        selected=selected,
    )
    if post_training_tk is not None:
        result["pre_post_training_tk_selected"] = pre_post_training_tk_selected
        result["post_training_tk"] = post_training_tk
    _write_json(selection_path, result)
    return result


def run_surface(
    study_path: Path,
    study: Mapping[str, Any],
    output_root: Path,
    surface: Mapping[str, Any],
    *,
    device: str,
    smoke: bool,
    target: str | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    materialize(
        study_path,
        study,
        output_root,
        target=target,
        dataset_root=dataset_root,
        device=device,
    )
    checkpoint, asset = ensure_asset(
        study,
        output_root,
        initializer=surface["initializer"],
        architecture=surface["architecture"],
        device=device,
        dataset_root=dataset_root,
    )
    source_config = build_source_config(
        study,
        initializer=surface["initializer"],
        architecture=surface["architecture"],
        scheme=surface["scheme"],
        optimizer=surface["optimizer"],
        init_checkpoint_path=checkpoint,
        dataset_root=dataset_root,
    )
    if smoke:
        smoke_root = output_root / "smoke" / "rho" / surface["surface_id"]
        source_path = smoke_root / "source_config.json"
    else:
        source_path = (
            output_root / "surfaces" / surface["surface_id"] / "source_config.json"
        )
    _write_json(source_path, source_config)
    if (
        _is_zero_bias_study(study)
        and surface["architecture"] == "conv3"
    ):
        gate = run_conv3_tk_operating_point_gate(
            study,
            output_root,
            surface,
            checkpoint,
            device=device,
            smoke=smoke,
            dataset_root=dataset_root,
        )
    else:
        gate = run_fixed_tk_gate(
            study,
            output_root,
            surface,
            checkpoint,
            device=device,
            smoke=smoke,
            dataset_root=dataset_root,
        )
    if smoke:
        search = _surface_search(study, surface)
        rho_conv_axis, rho_dense_axis = _rho_axes(study, surface)
        probe_result = run_rho_search(
            _rho_args(
                study,
                surface,
                source_path,
                smoke_root,
                rho_conv_axis,
                rho_dense_axis,
                device=device,
                target=target,
                index=None,
                probe_only=True,
                smoke=True,
            )
        )
        if probe_result.get("status") != "complete":
            result = {
                "schema_version": _artifact_schema(study, "smoke"),
                "status": "unresolved_probe",
                "surface": dict(surface),
                "asset": asset,
                "fixed_tk_path_exercised": True,
                "fixed_tk_diagnostics": gate,
                "probe": probe_result,
                "smoke": True,
                "official_test_read": False,
            }
            _write_json(output_root / "smoke" / "result.json", result)
            return result
        if search.get("core_mode", "adaptive_safe_center") == "fixed_grid":
            fixed = search["fixed_core"]
            representative = {
                "rho_conv": fixed["rho_conv"][len(fixed["rho_conv"]) // 2],
                "rho_dense": fixed["rho_dense"][len(fixed["rho_dense"]) // 2],
            }
        else:
            representative = search["center"]
        cell_kwargs: dict[str, Any] = {
            "device": device,
            "canary_only": False,
            "smoke": True,
            "operating_point_gate": gate,
        }
        if target is not None:
            cell_kwargs["target"] = target
        cell_dir, cell = _run_rho_cell(
            study,
            surface,
            source_path,
            smoke_root,
            rho_conv_axis,
            rho_dense_axis,
            float(representative["rho_conv"]),
            float(representative["rho_dense"]),
            **cell_kwargs,
        )
        result = {
            "schema_version": _artifact_schema(study, "smoke"),
            "status": "complete" if cell["status"] == "complete" else cell["status"],
            "surface": dict(surface),
            "asset": asset,
            "fixed_tk_path_exercised": True,
            "fixed_tk_diagnostics": gate,
            "rho_cell": _candidate_record(cell_dir, cell),
            "smoke": True,
            "official_test_read": False,
        }
        _write_json(output_root / "smoke" / "result.json", result)
        return result
    gate_scientifically_complete = (
        gate.get("scientifically_complete") is True
        if (
            _is_zero_bias_study(study)
            and surface["architecture"] == "conv3"
        )
        else True
    )
    if gate["security_passed"] is not True or not gate_scientifically_complete:
        gate_status = str(
            gate.get("status", "unresolved_fixed_tk_gradient_mismatch")
        )
        result = {
            "schema_version": _artifact_schema(study, "surface"),
            **dict(surface),
            "status": gate_status,
            "fixed_tk": gate,
            "selected": None,
            "official_test_read": False,
        }
        _write_json(
            output_root / "surfaces" / surface["surface_id"] / "selection.json",
            result,
        )
        return result
    return run_rho_surface(
        study,
        output_root,
        surface,
        source_path,
        device=device,
        target=target,
        operating_point_gate=gate,
    )


def _reported_range_status(record: Mapping[str, Any]) -> dict[str, Any] | None:
    existing = record.get("rho_range_status")
    if isinstance(existing, Mapping):
        return dict(existing)
    expansion = record.get("expansion")
    if not isinstance(expansion, Mapping):
        return None
    conv_edge = expansion.get("plateau_on_outer_rho_conv_edge")
    dense_edge = expansion.get("plateau_on_outer_rho_dense_edge")
    bounded = conv_edge is not None or dense_edge is not None
    return {
        "classification": "bounded" if bounded else "unbounded",
        "bracketed": not bounded,
        "rho_conv_edge": conv_edge,
        "rho_dense_edge": dense_edge,
        "derived_from_legacy_expansion_record": True,
    }


def _reported_accuracy_gate_met(record: Mapping[str, Any]) -> bool | None:
    selection = record.get("selection")
    if isinstance(selection, Mapping):
        explicit = selection.get("accuracy_gate_met")
        if isinstance(explicit, bool):
            return explicit
    selected = record.get("selected")
    if isinstance(selected, Mapping):
        eligible = selected.get("selection_eligible")
        if isinstance(eligible, bool):
            return eligible
    return None


def collect(study: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    records = []
    for surface in surface_specs(study):
        path = output_root / "surfaces" / surface["surface_id"] / "selection.json"
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    **surface,
                    "status": record["status"],
                    "selected": record.get("selected"),
                    "accuracy_gate_met": _reported_accuracy_gate_met(record),
                    "rho_range_status": _reported_range_status(record),
                    "suspicious_accuracy_status": record.get(
                        "suspicious_accuracy_status"
                    ),
                    "path": str(path),
                }
            )
        else:
            records.append(
                {
                    **surface,
                    "status": "pending",
                    "selected": None,
                    "accuracy_gate_met": None,
                    "rho_range_status": None,
                    "suspicious_accuracy_status": None,
                    "path": str(path),
                }
            )
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    successful_terminal_statuses = {"complete"}
    if not _is_zero_bias_study(study):
        successful_terminal_statuses.update(
            {
                "complete_below_accuracy_range_bounded",
                "complete_below_accuracy_bracketed",
            }
        )
    study_complete = all(
        record["status"] in successful_terminal_statuses for record in records
    )
    summary = {
        "schema_version": _artifact_schema(study, "summary"),
        "study_id": study["study_id"],
        "status": "complete" if study_complete else "partial",
        "execution_status": (
            "terminal" if counts.get("pending", 0) == 0 else "in_progress"
        ),
        "scope_is_partial_all_depth_initializer_selector": (
            _study_variant(study) != "conv123_zero_bias"
        ),
        "global_initializer_selection_allowed": False,
        "counts": counts,
        "surfaces": records,
        "official_test_read": False,
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plan", "smoke", "run-surface", "run-all", "status", "collect"),
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--surface-index", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument(
        "--target",
        help="Actual execution target recorded in runtime provenance.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Transport-only replacement for the configured MNIST root.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    study_path, study = load_study(args.study)
    configured_root = Path(study["execution"]["result_root"])
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (REPO_ROOT / configured_root).resolve()
    )
    device = args.device or study["execution"]["device"]
    target = args.target or study["execution"]["target"]
    dataset_root = Path(
        study["dataset"]["root"]
        if args.dataset_root is None
        else args.dataset_root
    ).expanduser().resolve()
    surfaces = surface_specs(study)
    if args.surface_index < 0 or args.surface_index >= len(surfaces):
        raise ValueError(
            f"Expected --surface-index in [0, {len(surfaces) - 1}], "
            f"got {args.surface_index}."
        )
    if (
        _is_zero_bias_study(study)
        and args.command in {"smoke", "run-surface", "run-all"}
        and args.target is None
        and study["execution"]["target"] == "multi-target"
    ):
        raise ValueError(
            "The multi-target zero-bias study requires --target for every "
            "executing command so runtime provenance names the actual host."
        )

    if args.command == "plan":
        result = {
            "study_id": study["study_id"],
            "study": str(study_path),
            "output_root": str(output_root),
            "target": target,
            "configured_target": study["execution"]["target"],
            "dataset_root": str(dataset_root),
            "configured_dataset_root": study["dataset"]["root"],
            "device": device,
            "surface_count": len(surfaces),
            "fixed_tk_gate_count": len(surfaces)
            // len(study["scope"]["optimizers"]),
            "surfaces": surfaces,
            "rho": study["rho_search"],
            "rho_by_architecture_scheme": {
                f"{surface['architecture']}__{surface['scheme']}": _surface_search(
                    study, surface
                )
                for surface in surfaces
                if surface["optimizer"] == study["scope"]["optimizers"][0]
                and surface["initializer"] == study["scope"]["initializers"][0]
            },
        }
    elif args.command == "smoke":
        result = run_surface(
            study_path,
            study,
            output_root,
            surfaces[args.surface_index],
            device=device,
            smoke=True,
            target=target,
            dataset_root=dataset_root,
        )
    elif args.command == "run-surface":
        result = run_surface(
            study_path,
            study,
            output_root,
            surfaces[args.surface_index],
            device=device,
            smoke=False,
            target=target,
            dataset_root=dataset_root,
        )
    elif args.command == "run-all":
        completed = []
        for surface in surfaces:
            completed.append(
                run_surface(
                    study_path,
                    study,
                    output_root,
                    surface,
                    device=device,
                    smoke=False,
                    target=target,
                    dataset_root=dataset_root,
                )
            )
        result = collect(study, output_root)
        result["executed_surface_count"] = len(completed)
    elif args.command in {"status", "collect"}:
        result = collect(study, output_root)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
