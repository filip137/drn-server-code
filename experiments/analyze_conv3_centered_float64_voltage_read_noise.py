#!/usr/bin/env python3
"""Replay centered float64 Conv3 EqProp with endpoint-voltage read noise.

This exploratory ordinary-MNIST diagnostic first reproduces the accepted
clean centered-EqProp tensors.  It then adds independent Gaussian noise to
each negative and positive *free-layer* endpoint voltage after equilibrium
and before the unchanged local squared-drop energy-gradient evaluation.  The
clamped input is kept exact.  Biases remain active in the dynamics but only
ConvWeight and DenseWeight gradients are scored.  No optimizer is built and
the official MNIST test split is never read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import socket
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONFIG = (
    REPOSITORY_ROOT
    / "results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase/config.resolved.json"
)
DEFAULT_STUDY_CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv3_centered_float64_voltage_read_noise_20260811_v1.json"
)
DEFAULT_SOURCE_RUN = (
    REPOSITORY_ROOT
    / "results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "results/perfectdiode-conv3-centered-float64-voltage-read-noise-20260811-v1"
)
FLOAT64_AUDIT = REPOSITORY_ROOT / "experiments/audit_eqprop_float64_shadow.py"
SCHEMA = "perfectdiode-conv3-centered-float64-voltage-read-noise/v1"
NORM_EPSILON = 1.0e-30
LAYER_LABELS = {
    "ConvWeight_0": "C0",
    "ConvWeight_1": "C1",
    "ConvWeight_2": "C2",
    "DenseWeight_0": "Dense",
}


_IMPORT_PATH_SNAPSHOT = tuple(sys.path)
_IMPORT_MODULE_SNAPSHOT = dict(sys.modules)


def _load_audit() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_eqprop_float64_audit_for_voltage_read_noise", FLOAT64_AUDIT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {FLOAT64_AUDIT}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit()
extended = audit.extended
base = audit.base
np = base.np
torch = base.torch
plt = base.plt


def _restore_helper_import_state() -> None:
    """Contain the frozen-runtime bootstrap when this CLI is imported."""

    sys.path[:] = _IMPORT_PATH_SNAPSHOT
    runtime_root = Path(base.RUNTIME_SOURCE_ROOT).expanduser().resolve()
    external_modules: dict[str, Any] = {}
    for name, module in tuple(sys.modules.items()):
        if module is None:
            continue
        origins: list[Path] = []
        module_file = getattr(module, "__file__", None)
        if module_file:
            origins.append(Path(module_file).expanduser().resolve())
        module_path = getattr(module, "__path__", None)
        if isinstance(module_path, (list, tuple)) or type(module_path).__name__ == (
            "_NamespacePath"
        ):
            for value in module_path:
                origins.append(Path(value).expanduser().resolve())
        if any(origin.is_relative_to(runtime_root) for origin in origins):
            external_modules[name] = module

    for name in external_modules:
        if name in _IMPORT_MODULE_SNAPSHOT:
            sys.modules[name] = _IMPORT_MODULE_SNAPSHOT[name]
        else:
            sys.modules.pop(name, None)

    for name, external_module in external_modules.items():
        parent_name, separator, child_name = name.rpartition(".")
        if not separator:
            continue
        parent = sys.modules.get(parent_name)
        if parent is None or getattr(parent, child_name, None) is not external_module:
            continue
        replacement = sys.modules.get(name)
        if replacement is None:
            delattr(parent, child_name)
        else:
            setattr(parent, child_name, replacement)


if __name__ != "__main__":
    _restore_helper_import_state()


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def _validate_study(
    source_config: Mapping[str, Any], study: Mapping[str, Any]
) -> None:
    if source_config.get("schema_version") != (
        "perfectdiode-conv-eqprop-bptt-beta-tk-displacement-replay/v1"
    ):
        raise ValueError("Unexpected source beta-config schema.")
    if study.get("schema_version") != (
        "perfectdiode-conv3-centered-float64-voltage-read-noise-study/v1"
    ):
        raise ValueError("Unexpected voltage-read-noise study schema.")
    if _semantic_sha256(source_config) != study["source"][
        "beta_config_semantic_sha256"
    ]:
        raise ValueError("Source beta config changed semantically.")
    replay = study["replay"]
    expected = {
        "architecture": "conv3",
        "T": 64,
        "K": 64,
        "batch_index": 0,
        "batch_size": 16,
        "runtime_dtype": "float64",
        "eqprop_variant": "centered",
        "nudging_mode": "current",
        "input_read_noise": False,
        "biases_active_in_dynamics": True,
        "bias_gradients_excluded": True,
        "optimizer_steps_applied": False,
        "official_test_read": False,
    }
    if any(replay.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen replay contract changed.")
    cases = list(study["cases"])
    if {(row["scheme"], row["checkpoint_role"]) for row in cases} != {
        (scheme, role)
        for scheme in ("baseline", "ours", "legacy")
        for role in ("reconstructed_initialization", "best_validation")
    }:
        raise ValueError("Study must declare all six scheme/checkpoint cases.")
    for row in cases:
        if not math.isclose(
            float(row["actual_beta"]) * float(row["base_beta_divisor"]),
            float(row["injected_beta_B"]),
            rel_tol=1.0e-15,
            abs_tol=0.0,
        ):
            raise ValueError(f"Beta scaling mismatch for {row}.")
    sigma = [float(value) for value in study["noise"]["sigma_values"]]
    if sigma != sorted(set(sigma)) or not sigma or sigma[0] != 0.0:
        raise ValueError("Noise sigma grid must be unique, sorted, and zero-anchored.")
    if int(study["noise"]["trial_count"]) <= 0:
        raise ValueError("Noise trial count must be positive.")


def _write_source_snapshots(
    run_dir: Path, sources: Mapping[str, Path]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, source in sources.items():
        source = source.resolve()
        destination = run_dir / f"source_snapshot__{label}{source.suffix}"
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_bytes(source.read_bytes())
        rows.append(
            {
                "label": label,
                "source_path": str(source),
                "source_sha256": base.sha256_file(source),
                "snapshot_path": destination.name,
                "snapshot_sha256": base.sha256_file(destination),
                "matches": base.sha256_file(source)
                == base.sha256_file(destination),
            }
        )
    if not rows or not all(bool(row["matches"]) for row in rows):
        raise RuntimeError("Analysis source snapshot guard failed.")
    return rows


def _materialize_batch(
    *,
    source_run: Path,
    dataset_config: Mapping[str, Any],
    dataset_root: Path,
    study: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_cohort = base._read_json(source_run / "cohort.json")
    batches, materialized = base._build_validation_cohort(
        dataset_config,
        data_root=dataset_root,
        batch_size=int(source_cohort["batch_size"]),
        example_count=int(source_cohort["example_count"]),
    )
    keys = ("batch_index", "source_indices_sha256", "payload_sha256")
    expected = [
        {key: row[key] for key in keys} for row in source_cohort["batches"]
    ]
    observed = [{key: row[key] for key in keys} for row in batches]
    exact = bool(
        expected == observed
        and materialized["validation_indices_sha256"]
        == source_cohort["validation_indices_sha256"]
        and materialized["cohort_sha256"] == source_cohort["cohort_sha256"]
    )
    if not exact:
        raise ValueError("Materialized cohort differs from the completed source run.")
    batch = batches[int(study["replay"]["batch_index"])]
    if (
        batch["payload_sha256"] != study["replay"]["batch_payload_sha256"]
        or batch["source_indices_sha256"]
        != study["replay"]["batch_source_indices_sha256"]
        or int(batch["images"].shape[0]) != int(study["replay"]["batch_size"])
    ):
        raise ValueError("Selected batch differs from the frozen study declaration.")
    return batch, materialized, source_cohort


def _gradient_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    cosine_minimum: float,
    symmetric_norm_delta_maximum: float,
) -> dict[str, Any]:
    left = candidate.detach().cpu().to(torch.float64).reshape(-1)
    right = reference.detach().cpu().to(torch.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Gradient shape mismatch: {left.shape} != {right.shape}.")
    base._finite(left, "candidate read-noise gradient")
    base._finite(right, "reference gradient")
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    difference_norm = float(torch.linalg.vector_norm(left - right))
    denominator = left_norm * right_norm
    cosine = (
        float(torch.dot(left, right)) / denominator
        if denominator > NORM_EPSILON
        else None
    )
    symmetric = 2.0 * abs(left_norm - right_norm) / max(
        left_norm + right_norm, NORM_EPSILON
    )
    return {
        "candidate_l2": left_norm,
        "reference_l2": right_norm,
        "difference_l2": difference_norm,
        "relative_l2": (
            difference_norm / right_norm if right_norm > NORM_EPSILON else None
        ),
        "cosine": cosine,
        "norm_ratio": (
            left_norm / right_norm if right_norm > NORM_EPSILON else None
        ),
        "symmetric_norm_delta": symmetric,
        "gate_passed": bool(
            cosine is not None
            and cosine >= cosine_minimum
            and symmetric <= symmetric_norm_delta_maximum
        ),
    }


def _comparison(
    candidate: torch.Tensor, reference: np.ndarray
) -> dict[str, Any]:
    left = candidate.detach().cpu().to(torch.float64).numpy()
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"Accepted-reference shape mismatch: {left.shape} != {right.shape}.")
    difference = np.linalg.norm((left - right).reshape(-1))
    reference_norm = np.linalg.norm(right.reshape(-1))
    return {
        "exact_equal": bool(np.array_equal(left, right)),
        "relative_l2": float(difference / max(reference_norm, NORM_EPSILON)),
        "maximum_absolute_error": float(np.max(np.abs(left - right))),
    }


def _energy_gradients_at_states(
    runtime: Mapping[str, Any],
    *,
    input_state: torch.Tensor,
    states: Sequence[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    input_layer = runtime["energy_fn"].layers()[0]
    input_layer.state = input_state.detach().to(
        device=device, dtype=torch.float64
    ).clone()
    restored = [
        value.detach().to(device=device, dtype=torch.float64).clone()
        for value in states
    ]
    base._restore_states(runtime["free_layers"], restored)
    return [
        value.detach().cpu().to(torch.float64).clone()
        for value in base._energy_gradients(runtime)
    ]


def _weight_dict(
    runtime: Mapping[str, Any], gradients: Sequence[torch.Tensor]
) -> dict[str, torch.Tensor]:
    return {
        str(runtime["parameters"][index].name).strip(): gradients[index]
        .detach()
        .cpu()
        .to(torch.float64)
        .clone()
        for index in runtime["weight_indices"]
    }


def _noised_states(
    states: Sequence[torch.Tensor],
    *,
    sigma: float,
    base_seed: int,
    trial_index: int,
    phase_index: int,
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    output: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for layer_index, state in enumerate(states):
        seed = int(base_seed + 1000 * trial_index + 100 * phase_index + layer_index)
        if sigma == 0.0:
            noise = torch.zeros_like(state, dtype=torch.float64, device="cpu")
        else:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            noise = torch.randn(
                tuple(state.shape), generator=generator, dtype=torch.float64
            ) * float(sigma)
        exact = state.detach().cpu().to(torch.float64)
        output.append(exact + noise)
        rows.append(
            {
                "state_layer_index": layer_index,
                "seed": seed,
                "element_count": int(noise.numel()),
                "empirical_noise_mean": float(noise.mean()),
                "empirical_noise_rms": float(noise.square().mean().sqrt()),
                "empirical_noise_maximum_absolute": float(noise.abs().max()),
            }
        )
    return output, rows


def _capture_case(
    *,
    inventory_case: Mapping[str, Any],
    case: Mapping[str, Any],
    batch: Mapping[str, Any],
    device: torch.device,
    residual_threshold: float,
    T: int = 64,
    K: int = 64,
) -> dict[str, Any]:
    selected = {
        "architecture": "conv3",
        "scheme": case["scheme"],
        "checkpoint_role": case["checkpoint_role"],
        "T": int(T),
        "K": int(K),
        "source_native_T": int(inventory_case["native_T"]),
        "source_native_K": int(inventory_case["native_K"]),
        "native_context": False,
        "tk_source": "fixed_read_noise_replay",
        "actual_beta": float(case["actual_beta"]),
    }
    output = audit._run_precision(
        case=inventory_case,
        selected=selected,
        batch=batch,
        precision="float64",
        device=device,
        residual_threshold=residual_threshold,
        amplification_exponent=3,
        nudging_mode="current",
        eqprop_variant="centered",
        capture_endpoint_states=True,
    )
    runtime_scale = float(output["guard"]["runtime_nudging_current_scale"])
    if not math.isclose(
        runtime_scale,
        float(case["base_beta_divisor"]),
        rel_tol=1.0e-15,
        abs_tol=0.0,
    ):
        raise RuntimeError("Runtime current scale differs from the declared beta divisor.")
    return output


def _accepted_reference_rows(
    *,
    case: Mapping[str, Any],
    output: Mapping[str, Any],
    accepted_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    reference_run = accepted_root / str(case["accepted_reference_run"])
    errors = base.validate_run(reference_run)
    if errors:
        raise ValueError("Invalid accepted reference bundle: " + "; ".join(errors))
    result_path = reference_run / "result.json"
    if base.sha256_file(result_path) != case["accepted_reference_result_sha256"]:
        raise ValueError("Accepted reference result changed.")
    archive = (
        reference_run
        / "artifacts/eqprop_gradients"
        / f"conv3__{case['scheme']}__{case['checkpoint_role']}.npz"
    )
    if base.sha256_file(archive) != case["accepted_reference_archive_sha256"]:
        raise ValueError("Accepted reference gradient archive changed.")
    rows: list[dict[str, Any]] = []
    with np.load(archive, allow_pickle=False) as loaded:
        for name in output["gradients"]:
            tensors = {
                "eqprop": output["gradients"][name],
                "bptt": output["bptt_gradients"][name],
                "negative_endpoint": output["endpoint_gradients"]["negative"][name],
                "positive_endpoint": output["endpoint_gradients"]["positive"][name],
            }
            keys = {
                "eqprop": f"eqprop_float64__{name}",
                "bptt": f"bptt_float64__{name}",
                "negative_endpoint": f"endpoint_negative_float64__{name}",
                "positive_endpoint": f"endpoint_positive_float64__{name}",
            }
            for quantity, tensor in tensors.items():
                comparison = _comparison(tensor, loaded[keys[quantity]])
                rows.append(
                    {
                        "schema": SCHEMA,
                        "scheme": case["scheme"],
                        "checkpoint_role": case["checkpoint_role"],
                        "parameter_name": name,
                        "quantity": quantity,
                        **comparison,
                        "guard_passed": comparison["relative_l2"] <= 1.0e-12,
                    }
                )
    if not rows or not all(bool(row["guard_passed"]) for row in rows):
        raise RuntimeError("Clean replay failed accepted-reference compatibility.")
    return rows, {
        "reference_result": str(result_path),
        "reference_result_sha256": base.sha256_file(result_path),
        "reference_archive": str(archive),
        "reference_archive_sha256": base.sha256_file(archive),
    }


def _clean_state_rows(
    case: Mapping[str, Any], output: Mapping[str, Any]
) -> list[dict[str, Any]]:
    runtime = output["captured_runtime"]
    negative = output["captured_phase_states"]["negative"]
    positive = output["captured_phase_states"]["positive"]
    rows = extended._state_displacement_rows(
        runtime["free_layers"],
        positive,
        negative,
        reference_kind="negative_phase",
        reference_norm_epsilon=NORM_EPSILON,
    )
    return [
        {
            "schema": SCHEMA,
            "scheme": case["scheme"],
            "checkpoint_role": case["checkpoint_role"],
            "injected_beta_B": case["injected_beta_B"],
            "actual_beta": case["actual_beta"],
            **row,
        }
        for row in rows
    ]


def _evaluate_case_noise(
    *,
    case: Mapping[str, Any],
    output: Mapping[str, Any],
    sigma_values: Sequence[float],
    trial_count: int,
    base_seed: int,
    device: torch.device,
    cosine_minimum: float,
    norm_delta_maximum: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    runtime = output["captured_runtime"]
    input_state = output["captured_input_state"]
    input_sha256 = base._tensor_sha256(input_state)
    negative_exact = output["captured_phase_states"]["negative"]
    positive_exact = output["captured_phase_states"]["positive"]
    clean = output["gradients"]
    bptt = output["bptt_gradients"]
    denominator_scale = float(runtime["nudging_current_scale"])
    gradient_rows: list[dict[str, Any]] = []
    state_noise_rows: list[dict[str, Any]] = []
    zero_identity_passed = True
    for sigma in sigma_values:
        trials = 1 if sigma == 0.0 else int(trial_count)
        for trial_index in range(trials):
            negative, negative_noise = _noised_states(
                negative_exact,
                sigma=float(sigma),
                base_seed=base_seed,
                trial_index=trial_index,
                phase_index=1,
            )
            positive, positive_noise = _noised_states(
                positive_exact,
                sigma=float(sigma),
                base_seed=base_seed,
                trial_index=trial_index,
                phase_index=2,
            )
            negative_all = _energy_gradients_at_states(
                runtime,
                input_state=input_state,
                states=negative,
                device=device,
            )
            positive_all = _energy_gradients_at_states(
                runtime,
                input_state=input_state,
                states=positive,
                device=device,
            )
            negative_weights = _weight_dict(runtime, negative_all)
            positive_weights = _weight_dict(runtime, positive_all)
            for phase, noise_rows in (
                ("negative", negative_noise),
                ("positive", positive_noise),
            ):
                for row in noise_rows:
                    state_noise_rows.append(
                        {
                            "schema": SCHEMA,
                            "scheme": case["scheme"],
                            "checkpoint_role": case["checkpoint_role"],
                            "injected_beta_B": case["injected_beta_B"],
                            "sigma_voltage": sigma,
                            "trial_index": trial_index,
                            "phase": phase,
                            **row,
                        }
                    )
            for name in clean:
                _, noisy = audit._native_eqprop_estimate(
                    eqprop_variant="centered",
                    beta=float(case["actual_beta"]),
                    denominator_scale=denominator_scale,
                    zero=output["endpoint_gradients"]["zero"][name],
                    negative=negative_weights[name],
                    positive=positive_weights[name],
                )
                acquisition = _gradient_metrics(
                    noisy,
                    clean[name],
                    cosine_minimum=cosine_minimum,
                    symmetric_norm_delta_maximum=norm_delta_maximum,
                )
                task = _gradient_metrics(
                    noisy,
                    bptt[name],
                    cosine_minimum=cosine_minimum,
                    symmetric_norm_delta_maximum=norm_delta_maximum,
                )
                if sigma == 0.0:
                    zero_identity_passed = bool(
                        zero_identity_passed
                        and acquisition["relative_l2"] is not None
                        and acquisition["relative_l2"] <= 1.0e-15
                    )
                gradient_rows.append(
                    {
                        "schema": SCHEMA,
                        "scheme": case["scheme"],
                        "checkpoint_role": case["checkpoint_role"],
                        "clean_beta_selection_status": case[
                            "clean_beta_selection_status"
                        ],
                        "injected_beta_B": case["injected_beta_B"],
                        "actual_beta": case["actual_beta"],
                        "base_beta_divisor": case["base_beta_divisor"],
                        "sigma_voltage": sigma,
                        "trial_index": trial_index,
                        "parameter_name": name,
                        "layer": LAYER_LABELS[name],
                        **{
                            f"acquisition_{key}": value
                            for key, value in acquisition.items()
                        },
                        **{f"task_{key}": value for key, value in task.items()},
                        "usable_gate_passed": bool(
                            acquisition["gate_passed"] and task["gate_passed"]
                        ),
                    }
                )
    final_input_sha256 = base._tensor_sha256(runtime["energy_fn"].layers()[0].state)
    return gradient_rows, state_noise_rows, {
        "noise_zero_clean_identity_passed": zero_identity_passed,
        "input_state_sha256_before": input_sha256,
        "input_state_sha256_after": final_input_sha256,
        "input_remained_exact": final_input_sha256 == input_sha256,
    }


def _quantile(values: Sequence[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not bool(np.isfinite(array).all()):
        raise ValueError("Expected finite non-empty values for quantile.")
    return float(np.quantile(array, q))


def _aggregate_trials(
    rows: Sequence[Mapping[str, Any]],
    *,
    cosine_minimum: float,
    norm_delta_maximum: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["scheme"],
                row["checkpoint_role"],
                float(row["sigma_voltage"]),
                row["parameter_name"],
            )
        ].append(row)
    layer_rows: list[dict[str, Any]] = []
    for (scheme, role, sigma, name), values in sorted(grouped.items()):
        acquisition_cosines = [float(row["acquisition_cosine"]) for row in values]
        acquisition_norm = [
            float(row["acquisition_symmetric_norm_delta"]) for row in values
        ]
        task_cosines = [float(row["task_cosine"]) for row in values]
        task_norm = [float(row["task_symmetric_norm_delta"]) for row in values]
        acquisition_pass = bool(
            _quantile(acquisition_cosines, 0.05) >= cosine_minimum
            and _quantile(acquisition_norm, 0.95) <= norm_delta_maximum
        )
        task_pass = bool(
            _quantile(task_cosines, 0.05) >= cosine_minimum
            and _quantile(task_norm, 0.95) <= norm_delta_maximum
        )
        layer_rows.append(
            {
                "schema": SCHEMA,
                "scheme": scheme,
                "checkpoint_role": role,
                "sigma_voltage": sigma,
                "parameter_name": name,
                "layer": LAYER_LABELS[name],
                "clean_beta_selection_status": values[0][
                    "clean_beta_selection_status"
                ],
                "trial_count": len(values),
                "acquisition_cosine_p05": _quantile(acquisition_cosines, 0.05),
                "acquisition_cosine_median": _quantile(acquisition_cosines, 0.5),
                "acquisition_cosine_minimum": min(acquisition_cosines),
                "acquisition_symmetric_norm_delta_p95": _quantile(
                    acquisition_norm, 0.95
                ),
                "task_cosine_p05": _quantile(task_cosines, 0.05),
                "task_cosine_median": _quantile(task_cosines, 0.5),
                "task_cosine_minimum": min(task_cosines),
                "task_symmetric_norm_delta_p95": _quantile(task_norm, 0.95),
                "acquisition_gate_passed": acquisition_pass,
                "task_gate_passed": task_pass,
                "usable_gate_passed": acquisition_pass and task_pass,
                "all_trials_acquisition_gate_passed": all(
                    bool(row["acquisition_gate_passed"]) for row in values
                ),
                "all_trials_task_gate_passed": all(
                    bool(row["task_gate_passed"]) for row in values
                ),
                "all_trials_usable_gate_passed": all(
                    bool(row["usable_gate_passed"]) for row in values
                ),
            }
        )
    by_config: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        by_config[
            (row["scheme"], row["checkpoint_role"], row["sigma_voltage"])
        ].append(row)
    config_rows: list[dict[str, Any]] = []
    for (scheme, role, sigma), values in sorted(by_config.items()):
        config_rows.append(
            {
                "schema": SCHEMA,
                "scheme": scheme,
                "checkpoint_role": role,
                "sigma_voltage": sigma,
                "clean_beta_selection_status": values[0][
                    "clean_beta_selection_status"
                ],
                "layer_count": len(values),
                "minimum_layer_acquisition_cosine_p05": min(
                    float(row["acquisition_cosine_p05"]) for row in values
                ),
                "maximum_layer_acquisition_symmetric_norm_delta_p95": max(
                    float(row["acquisition_symmetric_norm_delta_p95"])
                    for row in values
                ),
                "minimum_layer_task_cosine_p05": min(
                    float(row["task_cosine_p05"]) for row in values
                ),
                "maximum_layer_task_symmetric_norm_delta_p95": max(
                    float(row["task_symmetric_norm_delta_p95"]) for row in values
                ),
                "all_layers_acquisition_gate_passed": all(
                    bool(row["acquisition_gate_passed"]) for row in values
                ),
                "all_layers_task_gate_passed": all(
                    bool(row["task_gate_passed"]) for row in values
                ),
                "all_layers_usable_gate_passed": all(
                    bool(row["usable_gate_passed"]) for row in values
                ),
                "all_layers_all_trials_usable_gate_passed": all(
                    bool(row["all_trials_usable_gate_passed"]) for row in values
                ),
            }
        )
    return layer_rows, config_rows


def _threshold_rows(
    configuration_rows: Sequence[Mapping[str, Any]],
    clean_state_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in configuration_rows:
        grouped[(str(row["scheme"]), str(row["checkpoint_role"]))].append(row)
    clean_minimum = {
        (str(row["scheme"]), str(row["checkpoint_role"])): min(
            float(value["delta_rms"])
            for value in clean_state_rows
            if value["scheme"] == row["scheme"]
            and value["checkpoint_role"] == row["checkpoint_role"]
        )
        for row in clean_state_rows
    }
    cases = {
        (str(row["scheme"]), str(row["checkpoint_role"])): row
        for row in configuration_rows
    }
    result: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: float(row["sigma_voltage"]))
        thresholds: dict[str, float | None] = {}
        for label, field in (
            ("acquisition_only", "all_layers_acquisition_gate_passed"),
            ("usable", "all_layers_usable_gate_passed"),
        ):
            passing: list[float] = []
            for row in ordered:
                if not bool(row[field]):
                    break
                passing.append(float(row["sigma_voltage"]))
            thresholds[label] = max(passing) if passing else None
        first_acquisition_fail = next(
            (
                float(row["sigma_voltage"])
                for row in ordered
                if not bool(row["all_layers_acquisition_gate_passed"])
            ),
            None,
        )
        first_usable_fail = next(
            (
                float(row["sigma_voltage"])
                for row in ordered
                if not bool(row["all_layers_usable_gate_passed"])
            ),
            None,
        )
        minimum_delta = clean_minimum[key]
        acquisition = thresholds["acquisition_only"]
        usable = thresholds["usable"]
        result.append(
            {
                "schema": SCHEMA,
                "scheme": key[0],
                "checkpoint_role": key[1],
                "acquisition_only_maximum_sustained_sigma": acquisition,
                "acquisition_only_first_failing_sigma": first_acquisition_fail,
                "usable_maximum_sustained_sigma": usable,
                "usable_first_failing_sigma": first_usable_fail,
                "minimum_clean_positive_minus_negative_state_delta_rms": minimum_delta,
                "usable_sigma_over_minimum_state_delta_rms": (
                    float(usable) / minimum_delta if usable is not None else None
                ),
                "usable_phase_difference_noise_rms_over_minimum_state_delta_rms": (
                    math.sqrt(2.0) * float(usable) / minimum_delta
                    if usable is not None
                    else None
                ),
                "noise_threshold_bracketed": first_usable_fail is not None,
                "clean_beta_selection_status": cases[key].get(
                    "clean_beta_selection_status"
                ),
            }
        )
    return result


def _plot_summary(
    path: Path,
    configuration_rows: Sequence[Mapping[str, Any]],
    *,
    cosine_minimum: float,
    norm_delta_maximum: float,
) -> None:
    colors = {"baseline": "#4C78A8", "ours": "#F58518", "legacy": "#E45756"}
    styles = {"reconstructed_initialization": "--", "best_validation": "-"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for scheme in ("baseline", "ours", "legacy"):
        for role in ("reconstructed_initialization", "best_validation"):
            rows = sorted(
                [
                    row
                    for row in configuration_rows
                    if row["scheme"] == scheme
                    and row["checkpoint_role"] == role
                    and float(row["sigma_voltage"]) > 0.0
                ],
                key=lambda row: float(row["sigma_voltage"]),
            )
            label = f"{scheme}, {'init' if role.startswith('reconstructed') else 'best'}"
            x = [float(row["sigma_voltage"]) for row in rows]
            axes[0].plot(
                x,
                [float(row["minimum_layer_task_cosine_p05"]) for row in rows],
                marker="o",
                markersize=3,
                color=colors[scheme],
                linestyle=styles[role],
                label=label,
            )
            axes[1].plot(
                x,
                [
                    float(row["maximum_layer_task_symmetric_norm_delta_p95"])
                    for row in rows
                ],
                marker="o",
                markersize=3,
                color=colors[scheme],
                linestyle=styles[role],
                label=label,
            )
    for axis in axes:
        axis.set_xscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.set_xlabel(r"Endpoint read-noise $\sigma_v$ (normalized voltage)")
    axes[0].axhline(cosine_minimum, color="black", linewidth=1, linestyle=":")
    axes[0].set_ylabel("Worst-layer p05 cosine vs BPTT")
    axes[0].set_ylim(-0.05, 1.02)
    axes[1].axhline(norm_delta_maximum, color="black", linewidth=1, linestyle=":")
    axes[1].set_ylabel("Worst-layer p95 symmetric norm delta vs BPTT")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Conv3 centered float64 EqProp: independent endpoint-voltage read noise")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_layerwise(
    path: Path,
    layer_rows: Sequence[Mapping[str, Any]],
    *,
    cosine_minimum: float,
) -> None:
    colors = {"C0": "#4C78A8", "C1": "#F58518", "C2": "#54A24B", "Dense": "#B279A2"}
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, sharey=True, constrained_layout=True)
    for row_index, scheme in enumerate(("baseline", "ours", "legacy")):
        for column_index, role in enumerate(("reconstructed_initialization", "best_validation")):
            axis = axes[row_index, column_index]
            for layer in ("C0", "C1", "C2", "Dense"):
                rows = sorted(
                    [
                        row
                        for row in layer_rows
                        if row["scheme"] == scheme
                        and row["checkpoint_role"] == role
                        and row["layer"] == layer
                        and float(row["sigma_voltage"]) > 0.0
                    ],
                    key=lambda row: float(row["sigma_voltage"]),
                )
                axis.plot(
                    [float(row["sigma_voltage"]) for row in rows],
                    [float(row["task_cosine_p05"]) for row in rows],
                    marker="o",
                    markersize=2.8,
                    label=layer,
                    color=colors[layer],
                )
            axis.axhline(cosine_minimum, color="black", linewidth=1, linestyle=":")
            axis.set_xscale("log")
            axis.grid(True, which="both", alpha=0.25)
            axis.set_title(
                f"{scheme} — {'initialization' if role.startswith('reconstructed') else 'best epoch'}"
            )
            if column_index == 0:
                axis.set_ylabel("p05 cosine vs BPTT")
            if row_index == 2:
                axis.set_xlabel(r"$\sigma_v$ (normalized voltage)")
            if row_index == 0 and column_index == 1:
                axis.legend(fontsize=8)
    fig.suptitle("Layerwise read-noise fidelity (32 fixed trials per nonzero sigma)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    thresholds: Sequence[Mapping[str, Any]],
    smoke: bool,
) -> None:
    lines = [
        "# Conv3 centered float64 endpoint-voltage read noise",
        "",
        "Evidence class: exploratory ordinary-MNIST learning-algorithm diagnostic; not paper-facing.",
        "",
        "Independent zero-mean Gaussian noise is added after equilibrium to every negative and positive free-layer endpoint state. The clamped input remains exact; the unchanged local squared-drop weight statistic is then evaluated and centrally differenced in float64. The reported sigma is in normalized model-voltage units, not physical volts or hardware ENOB.",
        "",
        "The 32 fixed trials are a coarse empirical tail sample (p05 cosine, p95 symmetric norm delta), not formal tail-probability evidence. Acquisition-only thresholds require noisy-vs-clean EqProp fidelity. Usable thresholds additionally require noisy-vs-BPTT fidelity, both layer-by-layer.",
        "",
        "Baseline B=100 is the largest *tested* clean point and remains on an open upper beta-grid edge; ours and legacy clean operating betas are bracketed by the next tested beta.",
        "",
        "| scheme | checkpoint | acquisition-only max sigma | usable max sigma | first usable fail | min clean phase delta RMS | sqrt(2)*sigma/min delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in thresholds:
        lines.append(
            "| {scheme} | {role} | {acq} | {usable} | {fail} | {delta:.6g} | {ratio} |".format(
                scheme=row["scheme"],
                role=row["checkpoint_role"],
                acq=row["acquisition_only_maximum_sustained_sigma"],
                usable=row["usable_maximum_sustained_sigma"],
                fail=row["usable_first_failing_sigma"],
                delta=float(row["minimum_clean_positive_minus_negative_state_delta_rms"]),
                ratio=row[
                    "usable_phase_difference_noise_rms_over_minimum_state_delta_rms"
                ],
            )
        )
    if smoke:
        lines.extend(["", "Smoke scope only; no scientific threshold conclusion."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source_config_path = args.config.expanduser().resolve()
    study_config_path = args.study_config.expanduser().resolve()
    source_config = base._read_json(source_config_path)
    study = base._read_json(study_config_path)
    _validate_study(source_config, study)
    if source_config_path != _resolve_repository_path(study["source"]["beta_config"]):
        raise ValueError("--config differs from the frozen study source config.")
    if base.sha256_file(source_config_path) != study["source"]["beta_config_sha256"]:
        raise ValueError("Source beta config bytes changed.")
    source_run = args.source_run.expanduser().resolve()
    if source_run != _resolve_repository_path(study["source"]["native_beta_run"]):
        raise ValueError("--source-run differs from the frozen study source run.")
    source_errors = base.validate_run(source_run)
    if source_errors:
        raise ValueError("Invalid source bundle: " + "; ".join(source_errors))
    if base.sha256_file(source_run / "result.json") != study["source"][
        "native_beta_result_sha256"
    ]:
        raise ValueError("Source result changed.")
    if base.sha256_file(source_run / "cohort.json") != study["source"][
        "cohort_file_sha256"
    ]:
        raise ValueError("Source cohort changed.")

    runtime_source = base._validate_runtime_source(source_config)
    inventory, source_hashes_before = extended._source_inventory(source_config)
    inventory_by_scheme = {
        str(row["scheme"]): row
        for row in inventory
        if row["architecture"] == "conv3"
    }
    if set(inventory_by_scheme) != {"baseline", "ours", "legacy"}:
        raise ValueError("Conv3 inventory is incomplete.")
    batch, materialized_cohort, source_cohort = _materialize_batch(
        source_run=source_run,
        dataset_config=inventory[0]["source_config"],
        dataset_root=args.dataset_root.expanduser().resolve(),
        study=study,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    cases = list(study["cases"][:1] if args.smoke else study["cases"])
    sigma_values = (
        [0.0, 1.0e-7]
        if args.smoke
        else [float(value) for value in study["noise"]["sigma_values"]]
    )
    trial_count = 2 if args.smoke else int(study["noise"]["trial_count"])
    gate = study["gradient_fidelity_gate"]
    cosine_minimum = float(gate["cosine_minimum"])
    norm_delta_maximum = float(gate["symmetric_norm_delta_maximum"])
    residual_threshold = float(
        source_config["equilibrium_residual_contract"].get(
            "per_layer_sample_max_p90_threshold",
            source_config["equilibrium_residual_contract"].get("p90_threshold", 1.0e-2),
        )
    )
    run_id = args.run_id or ("smoke" if args.smoke else "production")
    run_dir = args.output_root.expanduser().resolve() / run_id
    sources = {
        "analyzer": Path(__file__).resolve(),
        "float64_audit": Path(audit.__file__).resolve(),
        "extended_eqprop_analysis": Path(extended.__file__).resolve(),
        "base_eqprop_analysis": Path(base.__file__).resolve(),
    }
    manifest = {
        "study_id": study["study_id"],
        "run_id": run_id,
        "arm_id": "conv3_centered_float64_endpoint_voltage_read_noise",
        "evidence_class": study["evidence_class"],
        "dataset": source_config["dataset"]["name"],
        "smoke": bool(args.smoke),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "configuration": {
            "study_config_path": str(study_config_path),
            "study_config_sha256": base.sha256_file(study_config_path),
            "source_config_path": str(source_config_path),
            "source_config_sha256": base.sha256_file(source_config_path),
            "resolved": {
                "cases": cases,
                "sigma_values": sigma_values,
                "trial_count_per_nonzero_sigma": trial_count,
                "T": 64,
                "K": 64,
                "batch_index": 0,
                "runtime_dtype": "float64",
                "eqprop_variant": "centered",
                "noise_model": study["noise"],
                "gradient_fidelity_gate": gate,
            },
        },
        "runtime": {
            **base.runtime_context(target=args.target),
            "device": str(device),
            "hostname": socket.gethostname(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "runtime_source": runtime_source,
        },
        "source": {
            "native_beta_run": str(source_run),
            "native_beta_result_sha256": base.sha256_file(source_run / "result.json"),
            "cohort_file_sha256": base.sha256_file(source_run / "cohort.json"),
            "accepted_sweep_root": str(
                _resolve_repository_path(study["source"]["accepted_sweep_root"])
            ),
        },
        "replay": {
            "endpoint_noise_location": "after_equilibrium_before_local_squared_drop_gradient",
            "input_read_noise": False,
            "negative_positive_noise_independent": True,
            "matched_standard_normal_draws_across_cases": True,
            "biases_active_in_dynamics": True,
            "bias_gradients_excluded": True,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "official_test_read": False,
            "trial_interpretation": study["noise"]["trial_interpretation"],
        },
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.sha256_file(Path(__file__).resolve()),
            "repository_head": base._git_output(REPOSITORY_ROOT, "rev-parse", "HEAD"),
            "source_files": {
                label: {"path": str(path), "sha256": base.sha256_file(path)}
                for label, path in sources.items()
            },
        },
    }
    base.start_run(run_dir, manifest)
    try:
        snapshots = _write_source_snapshots(run_dir, sources)
        base._write_json(run_dir / "study_config.resolved.json", study)
        base._write_json(run_dir / "source_config.resolved.json", source_config)
        base._write_json(run_dir / "source_cohort.json", source_cohort)
        base._write_json(run_dir / "materialized_cohort.json", materialized_cohort)
        gradient_rows: list[dict[str, Any]] = []
        state_noise_rows: list[dict[str, Any]] = []
        clean_state_rows: list[dict[str, Any]] = []
        residual_rows: list[dict[str, Any]] = []
        reference_rows: list[dict[str, Any]] = []
        reference_files: list[dict[str, Any]] = []
        case_guards: list[dict[str, Any]] = []
        accepted_root = _resolve_repository_path(study["source"]["accepted_sweep_root"])
        for case_index, case in enumerate(cases):
            base.update_status_progress(
                run_dir,
                {
                    "stage": "case_replay",
                    "case_index": case_index,
                    "case_count": len(cases),
                    "scheme": case["scheme"],
                    "checkpoint_role": case["checkpoint_role"],
                },
            )
            output = _capture_case(
                inventory_case=inventory_by_scheme[str(case["scheme"])],
                case=case,
                batch=batch,
                device=device,
                residual_threshold=residual_threshold,
            )
            case_reference_rows, files = _accepted_reference_rows(
                case=case, output=output, accepted_root=accepted_root
            )
            reference_rows.extend(case_reference_rows)
            reference_files.append({**case, **files})
            clean_state_rows.extend(_clean_state_rows(case, output))
            residual_rows.extend(output["residual_rows"])
            parameter_hash_before_noise = base._parameter_state_sha256(
                output["captured_runtime"]["parameters"]
            )
            rows, noise_rows, noise_guards = _evaluate_case_noise(
                case=case,
                output=output,
                sigma_values=sigma_values,
                trial_count=trial_count,
                base_seed=int(study["noise"]["base_seed"]),
                device=device,
                cosine_minimum=cosine_minimum,
                norm_delta_maximum=norm_delta_maximum,
            )
            gradient_rows.extend(rows)
            state_noise_rows.extend(noise_rows)
            parameter_hash_after_noise = base._parameter_state_sha256(
                output["captured_runtime"]["parameters"]
            )
            clean_task_gate = all(
                bool(row["task_gate_passed"])
                for row in rows
                if float(row["sigma_voltage"]) == 0.0
            )
            case_guards.append(
                {
                    "scheme": case["scheme"],
                    "checkpoint_role": case["checkpoint_role"],
                    "accepted_reference_reproduced": all(
                        bool(row["guard_passed"]) for row in case_reference_rows
                    ),
                    "clean_all_layer_eqprop_vs_bptt_gate_passed": clean_task_gate,
                    "all_residual_gates_passed": all(
                        bool(row["gate_passed"]) for row in output["residual_rows"]
                    ),
                    "parameter_state_sha256_before_noise": parameter_hash_before_noise,
                    "parameter_state_sha256_after_noise": parameter_hash_after_noise,
                    "parameters_unchanged_during_noise_evaluation": parameter_hash_before_noise
                    == parameter_hash_after_noise,
                    **noise_guards,
                }
            )
            del output
            if device.type == "cuda":
                torch.cuda.empty_cache()

        layer_rows, configuration_rows = _aggregate_trials(
            gradient_rows,
            cosine_minimum=cosine_minimum,
            norm_delta_maximum=norm_delta_maximum,
        )
        thresholds = _threshold_rows(configuration_rows, clean_state_rows)
        source_hashes_after = extended._verify_source_hashes(
            inventory, source_hashes_before
        )
        source_hashes_before_json = {
            "/".join(key): value for key, value in source_hashes_before.items()
        }
        guards = {
            "case_guards": case_guards,
            "source_hashes_before": source_hashes_before_json,
            "source_hashes_after": source_hashes_after,
            "source_bundle_files_unchanged": source_hashes_after
            == source_hashes_before_json,
            "all_analysis_source_snapshots_match": all(
                bool(row["matches"]) for row in snapshots
            ),
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        base._write_csv(run_dir / "gradient_trials.csv", gradient_rows)
        base._write_csv(run_dir / "state_noise_trials.csv", state_noise_rows)
        base._write_csv(run_dir / "clean_state_displacement.csv", clean_state_rows)
        base._write_csv(run_dir / "layer_quantiles.csv", layer_rows)
        base._write_csv(run_dir / "configuration_summary.csv", configuration_rows)
        base._write_csv(run_dir / "noise_thresholds.csv", thresholds)
        base._write_csv(run_dir / "phase_residuals.csv", residual_rows)
        base._write_csv(run_dir / "accepted_reference_compatibility.csv", reference_rows)
        base._write_json(run_dir / "accepted_reference_files.json", reference_files)
        base._write_json(run_dir / "read_only_guards.json", guards)
        _plot_summary(
            run_dir / "noise_fidelity_vs_sigma.png",
            configuration_rows,
            cosine_minimum=cosine_minimum,
            norm_delta_maximum=norm_delta_maximum,
        )
        _plot_layerwise(
            run_dir / "layerwise_cosine_vs_sigma.png",
            layer_rows,
            cosine_minimum=cosine_minimum,
        )
        _write_report(run_dir / "report.md", thresholds=thresholds, smoke=bool(args.smoke))
        for row in thresholds:
            base.append_metric(
                run_dir / "metrics.jsonl",
                {
                    "stage": "voltage_read_noise_threshold",
                    "split": "ordinary_mnist_validation",
                    **row,
                    "official_test_read": False,
                },
            )

        expected_gradient_rows = len(cases) * (
            1 + (len(sigma_values) - 1) * trial_count
        ) * 4
        expected_state_rows = len(cases) * (
            1 + (len(sigma_values) - 1) * trial_count
        ) * 2 * 4
        completion = {
            "declared_case_coverage_complete": len(case_guards) == len(cases),
            "gradient_trial_coverage_complete": len(gradient_rows)
            == expected_gradient_rows,
            "state_noise_coverage_complete": len(state_noise_rows)
            == expected_state_rows,
            "all_accepted_references_reproduced": all(
                bool(row["accepted_reference_reproduced"]) for row in case_guards
            ),
            "all_clean_eqprop_vs_bptt_gates_passed": all(
                bool(row["clean_all_layer_eqprop_vs_bptt_gate_passed"])
                for row in case_guards
            ),
            "all_noise_zero_identity_guards_passed": all(
                bool(row["noise_zero_clean_identity_passed"]) for row in case_guards
            ),
            "all_inputs_remained_exact": all(
                bool(row["input_remained_exact"]) for row in case_guards
            ),
            "all_residual_gates_passed": all(
                bool(row["all_residual_gates_passed"]) for row in case_guards
            ),
            "all_parameters_unchanged": all(
                bool(row["parameters_unchanged_during_noise_evaluation"])
                for row in case_guards
            ),
            "source_bundle_files_unchanged": guards["source_bundle_files_unchanged"],
            "analysis_sources_archived_exactly": guards[
                "all_analysis_source_snapshots_match"
            ],
            "all_production_noise_thresholds_bracketed": bool(
                args.smoke or all(bool(row["noise_threshold_bracketed"]) for row in thresholds)
            ),
            "float64_endpoint_and_subtraction_arithmetic": True,
            "noise_applied_after_equilibrium_before_local_squared_drop_evaluation": True,
            "negative_positive_endpoint_noise_independent": True,
            "input_read_noise_excluded": True,
            "bias_gradients_excluded": True,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        required = [
            key
            for key in completion
            if key
            not in {
                "optimizer_steps_applied",
                "official_test_read",
            }
        ]
        completion["criteria_met"] = bool(
            all(bool(completion[key]) for key in required)
            and completion["optimizer_steps_applied"] is False
            and completion["official_test_read"] is False
        )
        if not completion["criteria_met"]:
            failed = [key for key in required if not bool(completion[key])]
            raise RuntimeError("Completion guards failed: " + ", ".join(failed))
        terminal_metrics = {
            "case_count": len(cases),
            "sigma_count": len(sigma_values),
            "trial_count_per_nonzero_sigma": trial_count,
            "gradient_trial_row_count": len(gradient_rows),
            "state_noise_trial_row_count": len(state_noise_rows),
            "threshold_count": len(thresholds),
            "minimum_usable_sigma": min(
                float(row["usable_maximum_sustained_sigma"]) for row in thresholds
            ),
            "maximum_usable_sigma": max(
                float(row["usable_maximum_sustained_sigma"]) for row in thresholds
            ),
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        running_errors = base.validate_run(run_dir)
        if running_errors:
            raise RuntimeError("Running bundle validation failed: " + "; ".join(running_errors))
        result = base.complete_run(
            run_dir, terminal_metrics=terminal_metrics, completion=completion
        )
        errors = base.validate_run(run_dir)
        if errors:
            raise RuntimeError("Completed bundle validation failed: " + "; ".join(errors))
        return result
    except BaseException as error:
        status_path = run_dir / "status.json"
        if status_path.is_file():
            status = base._read_json(status_path)
            result_path = run_dir / "result.json"
            if result_path.is_file() and status.get("state") == "running":
                result_path.unlink()
            if status.get("state") == "running":
                base.fail_run(run_dir, error=error)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SOURCE_CONFIG)
    parser.add_argument("--study-config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/filip/datasets/mnist"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target", default="local:RTX3090")
    parser.add_argument(
        "--runtime-source-root", type=Path, default=base.RUNTIME_SOURCE_ROOT
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runtime_source_root.expanduser().resolve() != base.RUNTIME_SOURCE_ROOT:
        raise ValueError(
            "--runtime-source-root must match the source bootstrapped from --config."
        )
    result = _run(args)
    print(
        json.dumps(
            {
                "state": "complete",
                "terminal_metrics": result["terminal_metrics"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
