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


def _study_variant(study: Mapping[str, Any]) -> str:
    schema = study.get("schema_version")
    if schema == CONV12_STUDY_SCHEMA:
        return "conv12"
    if schema == CONV3_STUDY_SCHEMA:
        return "conv3"
    if schema == CONV123_ZERO_BIAS_STUDY_SCHEMA:
        return "conv123_zero_bias"
    raise ValueError(f"Unexpected study schema: {schema!r}.")


def _artifact_schema(study: Mapping[str, Any], artifact: str) -> str:
    variant = _study_variant(study)
    if variant == "conv123_zero_bias":
        return (
            "perfectdiode-conv123-bounded-uniform-zero-bias-rho-"
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
    if variant == "conv123_zero_bias":
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
            "bound_occupancy_increase_maximum": 0.20,
            "projection_efficiency_minimum": 0.50,
            "boundary_persistence_steps": 16,
        }
        if safety_by_architecture != {"conv3": expected_conv3_safety}:
            raise ValueError(
                "Expected only the active Conv3 persistent boundary gates: "
                f"{expected_conv3_safety!r}."
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
    if _study_variant(study) == "conv123_zero_bias":
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
    if _study_variant(study) == "conv123_zero_bias":
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
    if _study_variant(study) == "conv123_zero_bias":
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
    zero_bias_study = _study_variant(study) == "conv123_zero_bias"
    resolved_dataset_root = str(
        Path(
            study["dataset"]["root"]
            if dataset_root is None
            else dataset_root
        ).expanduser()
    )
    if checkpoint.exists() and record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        reusable = record.get("checkpoint_sha256") == _sha256_file(checkpoint)
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
        "checkpoint_sha256": _sha256_file(checkpoint),
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
            if _study_variant(study) == "conv123_zero_bias":
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
    if _study_variant(study) == "conv123_zero_bias":
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

    if _study_variant(study) != "conv123_zero_bias":
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
        safety_boundary_persistence=int(
            safety.get("boundary_persistence_steps", 16)
        ),
        index=index,
        collect_only=False,
        force=False,
        dry_run=False,
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
) -> tuple[Path, dict[str, Any]]:
    index = _rho_index(rho_conv_axis, rho_dense_axis, rho_conv, rho_dense)
    conv_index, dense_index = divmod(index, len(rho_dense_axis))
    rho_conv = rho_conv_axis[conv_index]
    rho_dense = rho_dense_axis[dense_index]
    run_rho_search(
        _rho_args(
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
    )
    return _cell(rho_root, index, rho_conv, rho_dense)


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
) -> dict[str, Any]:
    surface_dir = output_root / "surfaces" / surface["surface_id"]
    selection_path = surface_dir / "selection.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    rho_root = surface_dir / "rho"
    search = _surface_search(study, surface)
    rho_conv_axis, rho_dense_axis = _rho_axes(study, surface)
    run_rho_search(
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
            cell_kwargs = {"device": device, "canary_only": False}
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
        cell_kwargs = {"device": device, "canary_only": False}
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
        _study_variant(study) == "conv123_zero_bias"
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
        run_rho_search(
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
            "official_test_read": False,
        }
        _write_json(output_root / "smoke" / "result.json", result)
        return result
    gate_scientifically_complete = (
        gate.get("scientifically_complete") is True
        if (
            _study_variant(study) == "conv123_zero_bias"
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
    if _study_variant(study) != "conv123_zero_bias":
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
        _study_variant(study) == "conv123_zero_bias"
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
