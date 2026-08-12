#!/usr/bin/env python3
"""Measure node-readout ADC requirements for positive one-sided DRN EqProp.

The analog dynamics are replayed in true float64.  The matched beta=0 and
positive-beta endpoint states are then passed through an explicit node-level
ADC model *before* the local squared-voltage-drop weight statistics are
evaluated.  This preserves node sharing and the nonlinear local observable;
it is not a perturbation of already formed gradients.  The two endpoint
statistics are subtracted in float64, so the primary result measures the
combined ideal-ADC and endpoint-arithmetic path rather than ADC error alone.

Two acquisition contracts are reported:

``absolute_endpoint_shared_adc``
    Quantize both endpoint states independently with the same per-layer range,
    zero code, and transfer function.  The range is calibrated once from the
    matched-zero cohort and frozen across phases, betas, and minibatches.

``analog_delta_ideal_common``
    An optimistic control that subtracts the endpoint states in analog, applies
    a narrow-range ADC to the difference, and retains the common state exactly.
    It isolates the benefit of differential acquisition and is not a complete
    hardware readout proposal.

This runner is read-only: biases remain active in the dynamics, only weight
gradients are scored, no optimizer is constructed, and the official MNIST test
split is never read.
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
from typing import Any, Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv123_positive_onesided_small_beta_native_seed0_20260810_v1.json"
)
DEFAULT_STUDY_CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv3_legacy_eqprop_adc_readout_precision_20260811_v1.json"
)
DEFAULT_SOURCE_RUN = (
    REPOSITORY_ROOT
    / "results/perfectdiode-conv123-fc123-positive-eqprop-small-beta-seed0-20260810-v1/conv-current-commonbase"
)
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "results/perfectdiode-conv3-legacy-eqprop-adc-readout-precision-20260811-v1"
)
FLOAT64_AUDIT = REPOSITORY_ROOT / "experiments/audit_eqprop_float64_shadow.py"
SCHEMA = "perfectdiode-eqprop-adc-readout-precision/v1"
MODELS = ("absolute_endpoint_shared_adc", "analog_delta_ideal_common")
NORM_EPSILON = 1.0e-30
DEFAULT_COSINE_MINIMUM = 0.99
DEFAULT_SYMMETRIC_NORM_DELTA_MAXIMUM = 0.10


def _load_float64_audit() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_eqprop_float64_audit_for_adc_precision", FLOAT64_AUDIT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {FLOAT64_AUDIT}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_float64_audit()
extended = audit.extended
base = audit.base
np = base.np
torch = base.torch
plt = base.plt


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


def _write_source_snapshots(
    run_dir: Path, sources: Mapping[str, Path]
) -> list[dict[str, Any]]:
    """Archive the exact local analysis sources used by this run."""

    rows: list[dict[str, Any]] = []
    for label, source in sources.items():
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = run_dir / f"source_snapshot__{label}{source.suffix}"
        if destination.exists():
            raise FileExistsError(destination)
        destination.write_bytes(source.read_bytes())
        source_sha256 = base.sha256_file(source)
        snapshot_sha256 = base.sha256_file(destination)
        matches = source_sha256 == snapshot_sha256
        rows.append(
            {
                "label": str(label),
                "source_path": str(source),
                "source_sha256": source_sha256,
                "snapshot_path": destination.name,
                "snapshot_sha256": snapshot_sha256,
                "snapshot_matches_source": matches,
            }
        )
    if not rows or not all(bool(row["snapshot_matches_source"]) for row in rows):
        raise RuntimeError("At least one analysis source snapshot is incomplete.")
    return rows


def _quantize_symmetric_midtread(
    tensor: torch.Tensor,
    bits: int,
    full_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Quantize to a signed, symmetric, zero-preserving mid-tread codebook."""

    bits = int(bits)
    full_scale = float(full_scale)
    if bits < 2 or bits > 52:
        raise ValueError(f"ADC bits must be in [2,52], got {bits}.")
    if not math.isfinite(full_scale) or full_scale <= 0.0:
        raise ValueError("ADC full scale must be finite and positive.")
    if not tensor.dtype.is_floating_point:
        raise ValueError("ADC input tensor must be floating point.")
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError("ADC input contains a non-finite value.")

    qmax = (1 << (bits - 1)) - 1
    step = full_scale / float(qmax)
    clipped = tensor.clamp(min=-full_scale, max=full_scale)
    codes = torch.round(clipped / step).to(torch.int64)
    codes = codes.clamp(min=-qmax, max=qmax)
    quantized = codes.to(dtype=tensor.dtype) * step
    clipping = tensor.abs() > full_scale
    metadata = {
        "bits": bits,
        "qmax": qmax,
        "full_scale": full_scale,
        "step": step,
        "clipping_count": int(clipping.sum().detach().cpu()),
        "clipping_fraction": float(
            clipping.to(torch.float64).mean().detach().cpu()
        ),
        "input_minimum": float(tensor.min().detach().cpu()),
        "input_maximum": float(tensor.max().detach().cpu()),
        "input_maximum_absolute": float(tensor.abs().max().detach().cpu()),
        "nonzero_code_fraction": float(
            (codes != 0).to(torch.float64).mean().detach().cpu()
        ),
    }
    return quantized, codes, metadata


def _gradient_metrics(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    cosine_minimum: float = DEFAULT_COSINE_MINIMUM,
    symmetric_norm_delta_maximum: float = DEFAULT_SYMMETRIC_NORM_DELTA_MAXIMUM,
) -> dict[str, Any]:
    """Return direction, scale, error, sparsity, and the declared fidelity gate."""

    left = candidate.detach().cpu().to(torch.float64).reshape(-1)
    right = reference.detach().cpu().to(torch.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(
            f"Gradient shape mismatch: {tuple(left.shape)} != {tuple(right.shape)}."
        )
    base._finite(left, "candidate ADC gradient")
    base._finite(right, "reference gradient")
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    difference = left - right
    difference_norm = float(torch.linalg.vector_norm(difference))
    dot = float(torch.dot(left, right))
    denominator = left_norm * right_norm
    cosine = dot / denominator if denominator > NORM_EPSILON else None
    relative_l2 = (
        difference_norm / right_norm if right_norm > NORM_EPSILON else None
    )
    norm_ratio = left_norm / right_norm if right_norm > NORM_EPSILON else None
    symmetric_norm_delta = 2.0 * abs(left_norm - right_norm) / max(
        left_norm + right_norm, NORM_EPSILON
    )
    reference_nonzero = right != 0.0
    if bool(reference_nonzero.any()):
        sign_agreement = float(
            (
                torch.sign(left[reference_nonzero])
                == torch.sign(right[reference_nonzero])
            )
            .to(torch.float64)
            .mean()
        )
    else:
        sign_agreement = None
    if difference_norm == 0.0 and right_norm > NORM_EPSILON:
        snr_db = math.inf
    elif difference_norm > 0.0 and right_norm > NORM_EPSILON:
        snr_db = 20.0 * math.log10(right_norm / difference_norm)
    else:
        snr_db = None
    gate = bool(
        cosine is not None
        and cosine >= float(cosine_minimum)
        and symmetric_norm_delta <= float(symmetric_norm_delta_maximum)
    )
    return {
        "element_count": int(left.numel()),
        "candidate_l2": left_norm,
        "reference_l2": right_norm,
        "difference_l2": difference_norm,
        "dot_product": dot,
        "cosine": cosine,
        "relative_l2": relative_l2,
        "norm_ratio": norm_ratio,
        "symmetric_norm_delta": symmetric_norm_delta,
        "exact_zero_fraction": float((left == 0.0).to(torch.float64).mean()),
        "reference_exact_zero_fraction": float(
            (right == 0.0).to(torch.float64).mean()
        ),
        "active_entry_sign_agreement": sign_agreement,
        "snr_db": snr_db,
        "cosine_minimum": float(cosine_minimum),
        "symmetric_norm_delta_maximum": float(
            symmetric_norm_delta_maximum
        ),
        "gate_passed": gate,
    }


def _refinement_bits(
    coarse_bits: Sequence[int], pass_by_bit: Mapping[int, bool]
) -> list[int]:
    """Fill the stable coarse transition and all higher gaps one bit at a time.

    Quantizer code alignment can make an isolated bit depth pass even when the
    next one fails.  We therefore locate the first coarse point from which all
    higher coarse points pass, then test every missing integer through the
    declared maximum.  If the lowest coarse point already begins a sustained
    pass, the refinement extends down to the quantizer's two-bit model limit.
    """

    ordered = sorted({int(value) for value in coarse_bits})
    if not ordered:
        return []
    flags = [bool(pass_by_bit.get(bit, False)) for bit in ordered]
    sustained = [
        bit for index, bit in enumerate(ordered) if all(flags[index:])
    ]
    if not sustained:
        return []
    first_sustained = min(sustained)
    lower_tested = [bit for bit in ordered if bit < first_sustained]
    start = (max(lower_tested) + 1) if lower_tested else 2
    declared = set(ordered)
    return [
        bit
        for bit in range(start, max(ordered) + 1)
        if bit not in declared
    ]


def _resolve_beta_rows(
    source_config: Mapping[str, Any],
    case: Mapping[str, Any],
    requested_hats: Sequence[float],
) -> list[dict[str, Any]]:
    """Resolve requested common-base beta hats through the frozen source rule."""

    requested = [float(value) for value in requested_hats]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Requested base beta hats must be non-empty and unique.")
    available = extended._beta_grid_for_case(
        source_config["gradient_contract"], case
    )
    rows: list[dict[str, Any]] = []
    for value in requested:
        matches = [
            row
            for row in available
            if math.isclose(
                float(row["beta_hat_requested"]),
                value,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Requested base beta hat {value} has {len(matches)} source-grid matches."
            )
        rows.append(dict(matches[0]))
    return rows


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().cpu().to(torch.float64).reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise ValueError("Expected a non-empty finite tensor for quantiles.")
    return {
        "q50": float(torch.quantile(flat, 0.50)),
        "q90": float(torch.quantile(flat, 0.90)),
        "q99": float(torch.quantile(flat, 0.99)),
        "maximum": float(flat.max()),
    }


def _prefixed(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _mean_tensors(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("Cannot average an empty tensor collection.")
    return torch.stack(
        [value.detach().cpu().to(torch.float64) for value in values], dim=0
    ).mean(dim=0)


def _validate_configs(
    source_config: Mapping[str, Any], study_config: Mapping[str, Any]
) -> None:
    if source_config.get("schema_version") != (
        "perfectdiode-conv-eqprop-bptt-beta-tk-displacement-replay/v1"
    ):
        raise ValueError("Unexpected source beta-config schema.")
    if study_config.get("schema_version") != (
        "perfectdiode-eqprop-adc-readout-precision-study/v1"
    ):
        raise ValueError("Unexpected ADC study-config schema.")
    expected_semantic = study_config["source"]["beta_config_semantic_sha256"]
    if _semantic_sha256(source_config) != expected_semantic:
        raise ValueError("Source beta config changed semantically.")
    if extended._eqprop_variant(source_config["gradient_contract"]) != (
        "positive_one_sided"
    ):
        raise ValueError("ADC replay requires positive one-sided EqProp.")
    if source_config["gradient_contract"].get("nudging_mode") != "current":
        raise ValueError("ADC replay requires the literal frozen-current contract.")
    adc = study_config["adc"]
    if tuple(adc["models"]) != MODELS:
        raise ValueError(f"ADC models must be exactly {MODELS}.")
    if study_config["dac_scope"].get("included") is not False:
        raise ValueError("This first-stage runner must not silently add DAC effects.")


def _materialize_fixed_cohort(
    *,
    source_run: Path,
    source_config_for_dataset: Mapping[str, Any],
    dataset_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source_cohort = base._read_json(source_run / "cohort.json")
    batches, materialized = base._build_validation_cohort(
        source_config_for_dataset,
        data_root=dataset_root,
        batch_size=int(source_cohort["batch_size"]),
        example_count=int(source_cohort["example_count"]),
    )
    keys = ("batch_index", "source_indices_sha256", "payload_sha256")
    expected = [
        {key: (int(row[key]) if key == "batch_index" else row[key]) for key in keys}
        for row in source_cohort["batches"]
    ]
    observed = [
        {key: (int(row[key]) if key == "batch_index" else row[key]) for key in keys}
        for row in batches
    ]
    exact = bool(
        expected == observed
        and materialized["validation_indices_sha256"]
        == source_cohort["validation_indices_sha256"]
        and materialized["cohort_sha256"] == source_cohort["cohort_sha256"]
        and int(materialized["example_count"])
        == int(source_cohort["example_count"])
        and int(materialized["batch_count"])
        == int(source_cohort["batch_count"])
    )
    if not exact:
        raise ValueError("Materialized cohort differs from the completed source run.")
    return batches, materialized, source_cohort


def _case_inventory(
    source_config: Mapping[str, Any], study_config: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[Any, Any]]:
    inventory, source_hashes = extended._source_inventory(source_config)
    requested = study_config["case"]
    matches = [
        row
        for row in inventory
        if row["architecture"] == requested["architecture"]
        and row["scheme"] == requested["scheme"]
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one source case, found {len(matches)}.")
    case = matches[0]
    native = (int(case["native_T"]), int(case["native_K"]))
    declared = (int(requested["native_T"]), int(requested["native_K"]))
    if native != declared:
        raise ValueError(f"Native T/K changed: source={native}, declared={declared}.")
    return case, inventory, source_hashes


def _set_record_input(
    runtime: Mapping[str, Any], input_state: torch.Tensor, device: torch.device
) -> None:
    input_layer = runtime["energy_fn"].layers()[0]
    input_layer.state = input_state.detach().to(device=device, dtype=torch.float64).clone()


def _collect_ideal_endpoints(
    *,
    case: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    beta_rows: Sequence[Mapping[str, Any]],
    device: torch.device,
    T: int,
    K: int,
    checkpoint_role: str,
    residual_threshold: float,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Capture ideal float64 endpoints and references without retaining graphs."""

    runtime, checkpoint_guard = audit._checkpoint_runtime(
        case,
        role=checkpoint_role,
        device=device,
        gradient_iterations=K,
        nudging_mode="current",
    )
    dtype_proof = audit._convert_runtime_dtype(runtime, torch.float64)
    parameters_hash_before = base._parameter_state_sha256(runtime["parameters"])
    parameter_numeric_hash_before = audit._numeric_parameter_sha256(
        runtime["parameters"]
    )
    denominator_scale = float(runtime["nudging_current_scale"])
    if not math.isfinite(denominator_scale) or denominator_scale <= 0.0:
        raise ValueError("Finite-difference denominator scale must be positive.")
    beta_scale_rows: list[dict[str, Any]] = []
    for beta_row in beta_rows:
        beta = float(beta_row["beta"])
        declared_effective = float(beta_row["beta_effective"])
        declared_amplification = float(beta_row["amplification_factor"])
        computed_effective = beta * denominator_scale
        effective_matches = math.isclose(
            declared_effective,
            computed_effective,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
        amplification_matches = math.isclose(
            declared_amplification,
            denominator_scale,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
        beta_scale_rows.append(
            {
                "beta_hat_base_requested": float(
                    beta_row["beta_hat_requested"]
                ),
                "actual_beta": beta,
                "declared_effective_beta": declared_effective,
                "computed_effective_beta": computed_effective,
                "declared_amplification_factor": declared_amplification,
                "runtime_denominator_scale": denominator_scale,
                "effective_beta_matches_runtime": effective_matches,
                "amplification_matches_runtime": amplification_matches,
            }
        )
    if not beta_scale_rows or not all(
        bool(row["effective_beta_matches_runtime"])
        and bool(row["amplification_matches_runtime"])
        for row in beta_scale_rows
    ):
        raise RuntimeError(
            "Resolved beta metadata does not match the runtime current scale."
        )

    residuals = extended._ResidualAccumulator()
    records: list[dict[str, Any]] = []
    phase_guards: list[dict[str, Any]] = []
    names = list(runtime["weight_names"])
    for batch_number, batch in enumerate(batches):
        images = batch["images"].to(device=device, dtype=torch.float64)
        labels = batch["labels"].to(device=device)
        reset_proof = audit._set_input_and_reset(
            runtime, images, dtype=torch.float64
        )
        runtime["minimizer_inference"].num_iterations = int(T)
        runtime["inference_iterations"] = int(T)
        runtime["gradient_iterations"] = int(K)
        runtime["minimizer_inference"].compute_equilibrium()
        runtime["cost_fn"].set_target(labels)
        post_t_states = base._clone_states(runtime["free_layers"])
        post_t_hash = base._layer_state_sha256(runtime["free_layers"])
        residuals.add_endpoint(
            runtime,
            gradient_source=runtime["energy_fn"],
            phase="post_T_free",
            beta=0.0,
            source_indices=batch["source_indices"],
        )

        bptt, bptt_guard = audit._native_bptt_weight_gradients(
            runtime,
            post_t_states=post_t_states,
            post_t_hash=post_t_hash,
            expected_k=K,
        )

        base._restore_states(runtime["free_layers"], post_t_states)
        zero_start_hash = base._layer_state_sha256(runtime["free_layers"])
        runtime["augmented_fn"].prepare_nudging()
        runtime["augmented_fn"].nudging = torch.as_tensor(
            0.0, dtype=torch.float64, device=device
        )
        runtime["minimizer_augmented"].compute_equilibrium()
        zero_all = base._energy_gradients(runtime)
        zero_states = base._clone_states(runtime["free_layers"])
        zero_hash = base._layer_state_sha256(runtime["free_layers"])
        residuals.add_endpoint(
            runtime,
            gradient_source=runtime["augmented_fn"],
            phase="zero",
            beta=0.0,
            source_indices=batch["source_indices"],
        )

        positives: dict[float, dict[str, Any]] = {}
        for beta_row in beta_rows:
            beta = float(beta_row["beta"])
            base._restore_states(runtime["free_layers"], post_t_states)
            positive_start_hash = base._layer_state_sha256(runtime["free_layers"])
            if positive_start_hash != post_t_hash:
                raise RuntimeError("Positive phase did not restore common post-T state.")
            runtime["augmented_fn"].prepare_nudging()
            runtime["augmented_fn"].nudging = torch.as_tensor(
                beta, dtype=torch.float64, device=device
            )
            runtime["minimizer_augmented"].compute_equilibrium()
            positive_all = base._energy_gradients(runtime)
            positive_states = base._clone_states(runtime["free_layers"])
            positive_hash = base._layer_state_sha256(runtime["free_layers"])
            residuals.add_endpoint(
                runtime,
                gradient_source=runtime["augmented_fn"],
                phase="positive",
                beta=beta,
                source_indices=batch["source_indices"],
            )
            ideal: dict[str, torch.Tensor] = {}
            for index in runtime["weight_indices"]:
                name = str(runtime["parameters"][index].name).strip()
                _, estimate = audit._native_one_sided_estimate(
                    beta=beta,
                    denominator_scale=denominator_scale,
                    zero=zero_all[index],
                    positive=positive_all[index],
                )
                ideal[name] = estimate.detach().cpu().clone()
            positives[beta] = {
                "states": [value.detach().cpu().clone() for value in positive_states],
                "all_gradients": [
                    value.detach().cpu().clone() for value in positive_all
                ],
                "ideal_gradients": ideal,
                "state_sha256": positive_hash,
            }
            phase_guards.append(
                {
                    "batch_index": int(batch["batch_index"]),
                    "phase": "positive",
                    "actual_beta": beta,
                    "phase_start_state_sha256": positive_start_hash,
                    "common_post_T_state_sha256": post_t_hash,
                    "phase_end_state_sha256": positive_hash,
                    "start_matches_common_post_T": positive_start_hash
                    == post_t_hash,
                }
            )

        records.append(
            {
                "batch_index": int(batch["batch_index"]),
                "payload_sha256": batch["payload_sha256"],
                "source_indices_sha256": batch["source_indices_sha256"],
                "source_indices": tuple(int(value) for value in batch["source_indices"]),
                "input_state": runtime["energy_fn"]
                .layers()[0]
                .state.detach()
                .cpu()
                .clone(),
                "post_t_states": [value.detach().cpu().clone() for value in post_t_states],
                "post_t_state_sha256": post_t_hash,
                "zero_states": [value.detach().cpu().clone() for value in zero_states],
                "zero_state_sha256": zero_hash,
                "zero_all_gradients": [value.detach().cpu().clone() for value in zero_all],
                "bptt_gradients": {
                    name: value.detach().cpu().clone() for name, value in bptt.items()
                },
                "positives": positives,
            }
        )
        phase_guards.append(
            {
                "batch_index": int(batch["batch_index"]),
                "phase": "zero",
                "actual_beta": 0.0,
                "phase_start_state_sha256": zero_start_hash,
                "common_post_T_state_sha256": post_t_hash,
                "phase_end_state_sha256": zero_hash,
                "start_matches_common_post_T": zero_start_hash == post_t_hash,
                **bptt_guard,
            }
        )

    residual_rows = residuals.summary_rows(
        context={
            "architecture": case["architecture"],
            "scheme": case["scheme"],
            "checkpoint_role": checkpoint_role,
            "T": int(T),
            "K": int(K),
            "runtime_dtype": "float64",
        },
        expected_examples=sum(len(record["source_indices"]) for record in records),
        p90_threshold=float(residual_threshold),
    )
    parameters_hash_after = base._parameter_state_sha256(runtime["parameters"])
    parameter_numeric_hash_after = audit._numeric_parameter_sha256(
        runtime["parameters"]
    )
    guards = {
        **checkpoint_guard,
        "dtype_proof": dtype_proof,
        "runtime_weight_names": names,
        "runtime_weight_types": list(runtime["weight_types"]),
        "finite_difference_denominator_scale": denominator_scale,
        "beta_scale_rows": beta_scale_rows,
        "all_beta_effective_and_amplification_guards_passed": True,
        "parameter_state_sha256_before": parameters_hash_before,
        "parameter_state_sha256_after": parameters_hash_after,
        "parameter_numeric_sha256_before": parameter_numeric_hash_before,
        "parameter_numeric_sha256_after": parameter_numeric_hash_after,
        "parameter_tensors_unchanged": parameters_hash_after
        == parameters_hash_before,
        "parameter_values_unchanged": parameter_numeric_hash_after
        == parameter_numeric_hash_before,
        "all_phase_starts_match_common_post_T": all(
            bool(row["start_matches_common_post_T"]) for row in phase_guards
        ),
        "all_residual_gates_passed": all(
            bool(row["gate_passed"]) for row in residual_rows
        ),
        "optimizer_constructed": False,
        "optimizer_steps_applied": False,
        "official_test_read": False,
    }
    if not guards["parameter_tensors_unchanged"]:
        raise RuntimeError("Parameters changed during the read-only endpoint capture.")
    if not guards["all_phase_starts_match_common_post_T"]:
        raise RuntimeError("At least one phase start differs from common post-T.")
    return runtime, records, residual_rows, {"guards": guards, "phases": phase_guards}


def _absolute_ranges(
    records: Sequence[Mapping[str, Any]],
    layer_names: Sequence[str],
    *,
    headroom: float,
) -> dict[str, float]:
    if not math.isfinite(float(headroom)) or float(headroom) <= 1.0:
        raise ValueError("Absolute ADC headroom must be finite and > 1.")
    ranges: dict[str, float] = {}
    for index, name in enumerate(layer_names):
        maximum = max(
            float(record["zero_states"][index].abs().max()) for record in records
        )
        if maximum <= 0.0 or not math.isfinite(maximum):
            raise ValueError(f"Invalid zero-calibrated full scale for {name}: {maximum}.")
        ranges[str(name)] = float(headroom) * maximum
    return ranges


def _accepted_reference_compatibility(
    *,
    runtime: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    beta_rows: Sequence[Mapping[str, Any]],
    primary_beta_hat: float,
    reference_archive: Path,
    expected_T: int,
    expected_K: int,
    relative_l2_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    """Fail closed unless the current clean replay reproduces the accepted tensors."""

    if not reference_archive.is_file():
        raise FileNotFoundError(reference_archive)
    primary_rows = [
        row
        for row in beta_rows
        if math.isclose(
            float(row["beta_hat_requested"]),
            float(primary_beta_hat),
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
    ]
    batch_zero_records = [
        record for record in records if int(record["batch_index"]) == 0
    ]
    if len(primary_rows) != 1 or len(batch_zero_records) != 1:
        raise RuntimeError(
            "Accepted-reference replay requires exactly one primary beta row and batch 0."
        )
    beta_row = primary_rows[0]
    beta = float(beta_row["beta"])
    record = batch_zero_records[0]
    if beta not in record["positives"]:
        raise RuntimeError("Primary beta is absent from the batch-0 replay.")
    positive = record["positives"][beta]

    comparison_rows: list[dict[str, Any]] = []
    with np.load(reference_archive, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        metadata_guards = {
            "architecture_matches": metadata.get("architecture") == "conv3",
            "scheme_matches": metadata.get("scheme") == "legacy",
            "checkpoint_role_matches": metadata.get("checkpoint_role")
            == "best_validation",
            "T_matches": int(metadata.get("T", -1)) == int(expected_T),
            "K_matches": int(metadata.get("K", -1)) == int(expected_K),
            "actual_beta_matches": math.isclose(
                float(metadata.get("actual_beta", math.nan)),
                beta,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            ),
            "effective_beta_matches": math.isclose(
                float(metadata.get("effective_beta", math.nan)),
                float(beta_row["beta_effective"]),
                rel_tol=1.0e-12,
                abs_tol=0.0,
            ),
            "amplification_factor_matches": math.isclose(
                float(metadata.get("amplification_factor", math.nan)),
                float(beta_row["amplification_factor"]),
                rel_tol=1.0e-12,
                abs_tol=0.0,
            ),
        }
        if not all(metadata_guards.values()):
            raise RuntimeError(
                "Accepted float64 reference metadata does not match the current replay."
            )

        for index in runtime["weight_indices"]:
            name = str(runtime["parameters"][index].name).strip()
            current_by_quantity = {
                "eqprop": positive["ideal_gradients"][name],
                "bptt": record["bptt_gradients"][name],
                "numerator": positive["all_gradients"][index]
                - record["zero_all_gradients"][index],
            }
            for quantity, current in current_by_quantity.items():
                key = f"{quantity}_float64__{name}"
                if key not in archive.files:
                    raise KeyError(f"Accepted reference archive is missing {key}.")
                accepted = torch.from_numpy(np.asarray(archive[key])).to(
                    torch.float64
                )
                current = current.detach().cpu().to(torch.float64)
                if current.shape != accepted.shape:
                    raise RuntimeError(
                        f"Accepted {key} shape {tuple(accepted.shape)} differs from "
                        f"current {tuple(current.shape)}."
                    )
                difference = current - accepted
                difference_l2 = float(torch.linalg.vector_norm(difference.reshape(-1)))
                accepted_l2 = float(torch.linalg.vector_norm(accepted.reshape(-1)))
                relative_l2 = difference_l2 / max(accepted_l2, NORM_EPSILON)
                numeric_matches = bool(relative_l2 <= relative_l2_tolerance)
                comparison_rows.append(
                    {
                        "schema": SCHEMA,
                        "parameter_name": name,
                        "quantity": quantity,
                        "element_count": int(current.numel()),
                        "accepted_l2": accepted_l2,
                        "difference_l2": difference_l2,
                        "relative_l2": relative_l2,
                        "maximum_absolute_difference": float(difference.abs().max()),
                        "exactly_equal": bool(torch.equal(current, accepted)),
                        "relative_l2_tolerance": float(relative_l2_tolerance),
                        "numeric_matches": numeric_matches,
                    }
                )

    all_numeric_matches = bool(
        comparison_rows
        and all(bool(row["numeric_matches"]) for row in comparison_rows)
    )
    if not all_numeric_matches:
        worst = max(float(row["relative_l2"]) for row in comparison_rows)
        raise RuntimeError(
            "Current float64 replay differs from the accepted tensor archive; "
            f"worst relative L2={worst:.6g}."
        )
    return {
        "reference_archive": str(reference_archive),
        "reference_archive_sha256": base.sha256_file(reference_archive),
        "metadata_guards": metadata_guards,
        "relative_l2_tolerance": float(relative_l2_tolerance),
        "all_numeric_matches": all_numeric_matches,
        "maximum_relative_l2": max(
            float(row["relative_l2"]) for row in comparison_rows
        ),
        "comparison_rows": comparison_rows,
    }


def _differential_ranges(
    records: Sequence[Mapping[str, Any]],
    layer_names: Sequence[str],
    beta_rows: Sequence[Mapping[str, Any]],
    *,
    headroom: float,
) -> dict[float, dict[str, float]]:
    if not math.isfinite(float(headroom)) or float(headroom) <= 1.0:
        raise ValueError("Differential ADC headroom must be finite and > 1.")
    output: dict[float, dict[str, float]] = {}
    for beta_row in beta_rows:
        beta = float(beta_row["beta"])
        output[beta] = {}
        for index, name in enumerate(layer_names):
            maximum = max(
                float(
                    (
                        record["positives"][beta]["states"][index]
                        - record["zero_states"][index]
                    )
                    .abs()
                    .max()
                )
                for record in records
            )
            if maximum <= 0.0 or not math.isfinite(maximum):
                raise ValueError(
                    f"Invalid differential full scale for beta={beta}/{name}: {maximum}."
                )
            output[beta][str(name)] = float(headroom) * maximum
    return output


def _states_on_device(
    states: Sequence[torch.Tensor], device: torch.device
) -> list[torch.Tensor]:
    return [
        value.detach().to(device=device, dtype=torch.float64).clone()
        for value in states
    ]


def _energy_gradients_at_states(
    runtime: Mapping[str, Any],
    *,
    input_state: torch.Tensor,
    states: Sequence[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    _set_record_input(runtime, input_state, device)
    base._restore_states(runtime["free_layers"], _states_on_device(states, device))
    return [
        value.detach().cpu().to(torch.float64).clone()
        for value in base._energy_gradients(runtime)
    ]


def _state_adc_row(
    *,
    model: str,
    bits: int,
    beta_row: Mapping[str, Any],
    batch_index: int,
    layer_index: int,
    layer_name: str,
    exact_zero: torch.Tensor,
    exact_positive: torch.Tensor,
    quantized_zero: torch.Tensor,
    quantized_positive: torch.Tensor,
    zero_codes: torch.Tensor | None,
    positive_codes: torch.Tensor,
    zero_metadata: Mapping[str, Any] | None,
    positive_metadata: Mapping[str, Any],
    range_calibration: str,
) -> dict[str, Any]:
    exact_delta = exact_positive - exact_zero
    quantized_delta = quantized_positive - quantized_zero
    delta_error = quantized_delta - exact_delta
    step = float(positive_metadata["step"])
    displacement_over_lsb = _quantiles(exact_delta.abs() / step)
    exact_delta_l2 = float(torch.linalg.vector_norm(exact_delta.reshape(-1)))
    quantized_delta_l2 = float(
        torch.linalg.vector_norm(quantized_delta.reshape(-1))
    )
    delta_error_l2 = float(torch.linalg.vector_norm(delta_error.reshape(-1)))
    if zero_codes is None:
        code_changed = positive_codes != 0
    else:
        code_changed = positive_codes != zero_codes
    return {
        "schema": SCHEMA,
        "acquisition_model": model,
        "adc_bits": int(bits),
        "batch_index": int(batch_index),
        "layer_index": int(layer_index),
        "state_layer_name": str(layer_name),
        "beta_hat_base_requested": float(beta_row["beta_hat_requested"]),
        "actual_beta": float(beta_row["beta"]),
        "effective_beta": float(beta_row["beta_effective"]),
        "amplification_factor": float(beta_row["amplification_factor"]),
        "range_calibration": range_calibration,
        "full_scale": float(positive_metadata["full_scale"]),
        "adc_lsb": step,
        "exact_delta_l2": exact_delta_l2,
        "exact_delta_rms": float(exact_delta.square().mean().sqrt()),
        "exact_delta_maximum_absolute": float(exact_delta.abs().max()),
        "quantized_delta_l2": quantized_delta_l2,
        "quantized_delta_rms": float(quantized_delta.square().mean().sqrt()),
        "delta_error_l2": delta_error_l2,
        "delta_relative_l2_error": delta_error_l2
        / max(exact_delta_l2, NORM_EPSILON),
        "endpoint_code_change_fraction": float(
            code_changed.to(torch.float64).mean()
        ),
        "zero_endpoint_clipping_fraction": (
            None
            if zero_metadata is None
            else float(zero_metadata["clipping_fraction"])
        ),
        "positive_or_delta_clipping_fraction": float(
            positive_metadata["clipping_fraction"]
        ),
        "delta_abs_over_lsb_q50": displacement_over_lsb["q50"],
        "delta_abs_over_lsb_q90": displacement_over_lsb["q90"],
        "delta_abs_over_lsb_q99": displacement_over_lsb["q99"],
        "delta_abs_over_lsb_maximum": displacement_over_lsb["maximum"],
    }


def _evaluate_adc_bits(
    *,
    runtime: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    beta_rows: Sequence[Mapping[str, Any]],
    model: str,
    bits_values: Sequence[int],
    absolute_ranges: Mapping[str, float],
    differential_ranges: Mapping[float, Mapping[str, float]],
    device: torch.device,
    cosine_minimum: float,
    symmetric_norm_delta_maximum: float,
    batch_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    gradient_sets: dict[tuple[Any, ...], dict[str, list[Any]]],
) -> None:
    if model not in MODELS:
        raise ValueError(f"Unknown acquisition model {model!r}.")
    layer_names = [str(layer.name) for layer in runtime["free_layers"]]
    beta_by_actual = {float(row["beta"]): row for row in beta_rows}
    for bits in sorted({int(value) for value in bits_values}):
        for record in records:
            absolute_zero_states: list[torch.Tensor] | None = None
            absolute_zero_codes: list[torch.Tensor] | None = None
            absolute_zero_metadata: list[dict[str, Any]] | None = None
            absolute_zero_gradients: list[torch.Tensor] | None = None
            if model == "absolute_endpoint_shared_adc":
                absolute_zero_states = []
                absolute_zero_codes = []
                absolute_zero_metadata = []
                for layer_index, name in enumerate(layer_names):
                    quantized, codes, metadata = _quantize_symmetric_midtread(
                        record["zero_states"][layer_index],
                        bits,
                        absolute_ranges[name],
                    )
                    absolute_zero_states.append(quantized)
                    absolute_zero_codes.append(codes)
                    absolute_zero_metadata.append(metadata)
                absolute_zero_gradients = _energy_gradients_at_states(
                    runtime,
                    input_state=record["input_state"],
                    states=absolute_zero_states,
                    device=device,
                )

            for beta, positive in record["positives"].items():
                beta = float(beta)
                beta_row = beta_by_actual[beta]
                quantized_positive_states: list[torch.Tensor] = []
                positive_codes: list[torch.Tensor] = []
                positive_metadata: list[dict[str, Any]] = []
                quantized_zero_states: list[torch.Tensor] = []
                zero_codes: list[torch.Tensor | None] = []
                zero_metadata: list[dict[str, Any] | None] = []

                if model == "absolute_endpoint_shared_adc":
                    assert absolute_zero_states is not None
                    assert absolute_zero_codes is not None
                    assert absolute_zero_metadata is not None
                    quantized_zero_states = absolute_zero_states
                    zero_codes = list(absolute_zero_codes)
                    zero_metadata = list(absolute_zero_metadata)
                    for layer_index, name in enumerate(layer_names):
                        quantized, codes, metadata = _quantize_symmetric_midtread(
                            positive["states"][layer_index],
                            bits,
                            absolute_ranges[name],
                        )
                        quantized_positive_states.append(quantized)
                        positive_codes.append(codes)
                        positive_metadata.append(metadata)
                    assert absolute_zero_gradients is not None
                    zero_all = absolute_zero_gradients
                    range_label = "zero_cohort_layerwise_frozen_all_phases_betas_batches"
                else:
                    quantized_zero_states = [
                        value.detach().clone() for value in record["zero_states"]
                    ]
                    zero_codes = [None] * len(layer_names)
                    zero_metadata = [None] * len(layer_names)
                    for layer_index, name in enumerate(layer_names):
                        exact_delta = (
                            positive["states"][layer_index]
                            - record["zero_states"][layer_index]
                        )
                        quantized_delta, codes, metadata = (
                            _quantize_symmetric_midtread(
                                exact_delta,
                                bits,
                                differential_ranges[beta][name],
                            )
                        )
                        quantized_positive_states.append(
                            record["zero_states"][layer_index] + quantized_delta
                        )
                        positive_codes.append(codes)
                        positive_metadata.append(metadata)
                    zero_all = record["zero_all_gradients"]
                    range_label = "oracle_delta_cohort_layerwise_per_beta_ideal_common"

                positive_all = _energy_gradients_at_states(
                    runtime,
                    input_state=record["input_state"],
                    states=quantized_positive_states,
                    device=device,
                )
                for layer_index, name in enumerate(layer_names):
                    state_rows.append(
                        _state_adc_row(
                            model=model,
                            bits=bits,
                            beta_row=beta_row,
                            batch_index=int(record["batch_index"]),
                            layer_index=layer_index,
                            layer_name=name,
                            exact_zero=record["zero_states"][layer_index],
                            exact_positive=positive["states"][layer_index],
                            quantized_zero=quantized_zero_states[layer_index],
                            quantized_positive=quantized_positive_states[layer_index],
                            zero_codes=zero_codes[layer_index],
                            positive_codes=positive_codes[layer_index],
                            zero_metadata=zero_metadata[layer_index],
                            positive_metadata=positive_metadata[layer_index],
                            range_calibration=range_label,
                        )
                    )

                for index in runtime["weight_indices"]:
                    parameter = runtime["parameters"][index]
                    name = str(parameter.name).strip()
                    _, candidate = audit._native_one_sided_estimate(
                        beta=beta,
                        denominator_scale=float(runtime["nudging_current_scale"]),
                        zero=zero_all[index],
                        positive=positive_all[index],
                    )
                    candidate = candidate.detach().cpu().to(torch.float64).clone()
                    ideal = positive["ideal_gradients"][name]
                    bptt = record["bptt_gradients"][name]
                    quantized_vs_ideal = _gradient_metrics(
                        candidate,
                        ideal,
                        cosine_minimum=cosine_minimum,
                        symmetric_norm_delta_maximum=symmetric_norm_delta_maximum,
                    )
                    quantized_vs_bptt = _gradient_metrics(
                        candidate,
                        bptt,
                        cosine_minimum=cosine_minimum,
                        symmetric_norm_delta_maximum=symmetric_norm_delta_maximum,
                    )
                    ideal_vs_bptt = _gradient_metrics(
                        ideal,
                        bptt,
                        cosine_minimum=cosine_minimum,
                        symmetric_norm_delta_maximum=symmetric_norm_delta_maximum,
                    )
                    key = (model, beta, bits, name)
                    bucket = gradient_sets.setdefault(
                        key,
                        {
                            "candidate": [],
                            "ideal": [],
                            "bptt": [],
                            "batch_quantized_vs_ideal": [],
                        },
                    )
                    bucket["candidate"].append(candidate)
                    bucket["ideal"].append(ideal)
                    bucket["bptt"].append(bptt)
                    bucket["batch_quantized_vs_ideal"].append(
                        quantized_vs_ideal
                    )
                    batch_rows.append(
                        {
                            "schema": SCHEMA,
                            "acquisition_model": model,
                            "adc_bits": bits,
                            "batch_index": int(record["batch_index"]),
                            "batch_payload_sha256": record["payload_sha256"],
                            "parameter_name": name,
                            "parameter_type": parameter.__class__.__name__,
                            "beta_hat_base_requested": float(
                                beta_row["beta_hat_requested"]
                            ),
                            "actual_beta": beta,
                            "effective_beta": float(beta_row["beta_effective"]),
                            "amplification_factor": float(
                                beta_row["amplification_factor"]
                            ),
                            "bias_excluded": True,
                            **_prefixed(
                                "quantized_vs_ideal_eqprop", quantized_vs_ideal
                            ),
                            **_prefixed(
                                "quantized_vs_bptt", quantized_vs_bptt
                            ),
                            **_prefixed("ideal_eqprop_vs_bptt", ideal_vs_bptt),
                            "adc_fidelity_gate_passed": bool(
                                quantized_vs_ideal["gate_passed"]
                            ),
                        }
                    )


def _cohort_gradient_rows(
    *,
    gradient_sets: Mapping[tuple[Any, ...], Mapping[str, Sequence[Any]]],
    beta_rows: Sequence[Mapping[str, Any]],
    cosine_minimum: float,
    symmetric_norm_delta_maximum: float,
) -> list[dict[str, Any]]:
    beta_by_actual = {float(row["beta"]): row for row in beta_rows}
    rows: list[dict[str, Any]] = []
    for key, values in sorted(gradient_sets.items(), key=lambda item: item[0]):
        model, beta, bits, name = key
        candidate = _mean_tensors(values["candidate"])
        ideal = _mean_tensors(values["ideal"])
        bptt = _mean_tensors(values["bptt"])
        quantized_vs_ideal = _gradient_metrics(
            candidate,
            ideal,
            cosine_minimum=cosine_minimum,
            symmetric_norm_delta_maximum=symmetric_norm_delta_maximum,
        )
        quantized_vs_bptt = _gradient_metrics(
            candidate,
            bptt,
            cosine_minimum=cosine_minimum,
            symmetric_norm_delta_maximum=symmetric_norm_delta_maximum,
        )
        ideal_vs_bptt = _gradient_metrics(
            ideal,
            bptt,
            cosine_minimum=cosine_minimum,
            symmetric_norm_delta_maximum=symmetric_norm_delta_maximum,
        )
        batch_metrics = list(values["batch_quantized_vs_ideal"])
        finite_cosines = [
            float(row["cosine"])
            for row in batch_metrics
            if row["cosine"] is not None
        ]
        all_batches_pass = bool(
            batch_metrics and all(bool(row["gate_passed"]) for row in batch_metrics)
        )
        beta_row = beta_by_actual[float(beta)]
        rows.append(
            {
                "schema": SCHEMA,
                "acquisition_model": model,
                "adc_bits": int(bits),
                "parameter_name": name,
                "beta_hat_base_requested": float(
                    beta_row["beta_hat_requested"]
                ),
                "actual_beta": float(beta),
                "effective_beta": float(beta_row["beta_effective"]),
                "amplification_factor": float(beta_row["amplification_factor"]),
                "batch_count": len(batch_metrics),
                **_prefixed("cohort_quantized_vs_ideal_eqprop", quantized_vs_ideal),
                **_prefixed("cohort_quantized_vs_bptt", quantized_vs_bptt),
                **_prefixed("cohort_ideal_eqprop_vs_bptt", ideal_vs_bptt),
                "minimum_batch_quantized_vs_ideal_cosine": (
                    min(finite_cosines) if finite_cosines else None
                ),
                "maximum_batch_quantized_vs_ideal_symmetric_norm_delta": max(
                    float(row["symmetric_norm_delta"]) for row in batch_metrics
                ),
                "all_minibatches_adc_gate_passed": all_batches_pass,
                "cohort_adc_gate_passed": bool(quantized_vs_ideal["gate_passed"]),
                "parameter_adc_gate_passed": bool(
                    all_batches_pass and quantized_vs_ideal["gate_passed"]
                ),
            }
        )
    return rows


def _configuration_rows(
    cohort_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in cohort_rows:
        groups[
            (
                row["acquisition_model"],
                float(row["actual_beta"]),
                int(row["adc_bits"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (model, beta, bits), rows in sorted(groups.items(), key=lambda item: item[0]):
        finite_batch_cosines = [
            float(row["minimum_batch_quantized_vs_ideal_cosine"])
            for row in rows
            if row["minimum_batch_quantized_vs_ideal_cosine"] is not None
        ]
        finite_cohort_cosines = [
            float(row["cohort_quantized_vs_ideal_eqprop_cosine"])
            for row in rows
            if row["cohort_quantized_vs_ideal_eqprop_cosine"] is not None
        ]
        finite_ideal_cosines = [
            float(row["cohort_ideal_eqprop_vs_bptt_cosine"])
            for row in rows
            if row["cohort_ideal_eqprop_vs_bptt_cosine"] is not None
        ]
        output.append(
            {
                "schema": SCHEMA,
                "acquisition_model": model,
                "adc_bits": int(bits),
                "beta_hat_base_requested": float(
                    rows[0]["beta_hat_base_requested"]
                ),
                "actual_beta": float(beta),
                "effective_beta": float(rows[0]["effective_beta"]),
                "scored_weight_count": len(rows),
                "passing_weight_count": sum(
                    bool(row["parameter_adc_gate_passed"]) for row in rows
                ),
                "minimum_cohort_quantized_vs_ideal_cosine": (
                    min(finite_cohort_cosines) if finite_cohort_cosines else None
                ),
                "minimum_batch_quantized_vs_ideal_cosine": (
                    min(finite_batch_cosines) if finite_batch_cosines else None
                ),
                "maximum_cohort_quantized_vs_ideal_symmetric_norm_delta": max(
                    float(
                        row[
                            "cohort_quantized_vs_ideal_eqprop_symmetric_norm_delta"
                        ]
                    )
                    for row in rows
                ),
                "maximum_batch_quantized_vs_ideal_symmetric_norm_delta": max(
                    float(
                        row[
                            "maximum_batch_quantized_vs_ideal_symmetric_norm_delta"
                        ]
                    )
                    for row in rows
                ),
                "minimum_ideal_eqprop_vs_bptt_cosine": (
                    min(finite_ideal_cosines) if finite_ideal_cosines else None
                ),
                "all_weight_layers_and_minibatches_passed": bool(
                    rows
                    and all(
                        bool(row["parameter_adc_gate_passed"])
                        for row in rows
                    )
                ),
            }
        )
    return output


def _minimum_passing_rows(
    configuration_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in configuration_rows:
        grouped[(str(row["acquisition_model"]), float(row["actual_beta"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, beta), rows in sorted(grouped.items()):
        ordered_rows = sorted(rows, key=lambda row: int(row["adc_bits"]))
        tested = [int(row["adc_bits"]) for row in ordered_rows]
        flags = [
            bool(row["all_weight_layers_and_minibatches_passed"])
            for row in ordered_rows
        ]
        passing = [bit for bit, passed in zip(tested, flags, strict=True) if passed]
        sustained = [
            bit
            for index, bit in enumerate(tested)
            if all(flags[index:])
        ]
        sustained_from = min(sustained) if sustained else None
        lower_failure = (
            max(
                bit
                for bit, passed in zip(tested, flags, strict=True)
                if sustained_from is not None and bit < sustained_from and not passed
            )
            if sustained_from is not None
            and any(
                bit < sustained_from and not passed
                for bit, passed in zip(tested, flags, strict=True)
            )
            else None
        )
        pass_to_fail_reversals = [
            [tested[index - 1], tested[index]]
            for index in range(1, len(tested))
            if flags[index - 1] and not flags[index]
        ]
        tested_model_minimum = min(tested) == 2
        threshold_resolved = bool(
            sustained_from is not None
            and (lower_failure is not None or tested_model_minimum)
        )
        if sustained_from is None:
            interpretation = "no_sustained_passing_region"
        elif lower_failure is None and not tested_model_minimum:
            interpretation = "at_or_below_tested_minimum"
        elif sustained_from == 2:
            interpretation = "sustained_from_quantizer_model_minimum"
        else:
            interpretation = "bracketed_sustained_minimum"
        finite_ideal_cosines = [
            float(row["minimum_ideal_eqprop_vs_bptt_cosine"])
            for row in ordered_rows
            if row["minimum_ideal_eqprop_vs_bptt_cosine"] is not None
        ]
        output.append(
            {
                "schema": SCHEMA,
                "acquisition_model": model,
                "beta_hat_base_requested": float(rows[0]["beta_hat_base_requested"]),
                "actual_beta": beta,
                "effective_beta": float(rows[0]["effective_beta"]),
                "first_isolated_passing_adc_bits": min(passing) if passing else None,
                "sustained_passing_from_adc_bits": sustained_from,
                "minimum_passing_adc_bits": (
                    sustained_from if threshold_resolved else None
                ),
                "passing_upper_bound_adc_bits": (
                    sustained_from
                    if sustained_from is not None and not threshold_resolved
                    else None
                ),
                "tested_bit_minimum": min(tested),
                "tested_bit_maximum": max(tested),
                "tested_bit_count": len(tested),
                "passing_point_count": len(passing),
                "lower_failing_adc_bits": lower_failure,
                "pass_to_fail_reversal_count": len(pass_to_fail_reversals),
                "pass_to_fail_reversals": pass_to_fail_reversals,
                "threshold_interpretation": interpretation,
                "gate_resolved": threshold_resolved,
                "minimum_ideal_eqprop_vs_bptt_cosine": (
                    min(finite_ideal_cosines) if finite_ideal_cosines else None
                ),
            }
        )
    return output


def _coverage_guards(
    *,
    records: Sequence[Mapping[str, Any]],
    beta_rows: Sequence[Mapping[str, Any]],
    bits_by_model: Mapping[str, Sequence[int]],
    weight_names: Sequence[str],
    layer_names: Sequence[str],
    batch_rows: Sequence[Mapping[str, Any]],
    cohort_rows: Sequence[Mapping[str, Any]],
    configuration_rows: Sequence[Mapping[str, Any]],
    minimum_rows: Sequence[Mapping[str, Any]],
    state_rows: Sequence[Mapping[str, Any]],
    residual_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove exact row-key coverage, including duplicate and residual checks."""

    batch_indices = [int(record["batch_index"]) for record in records]
    betas = [float(row["beta"]) for row in beta_rows]
    weights = [str(name) for name in weight_names]
    layers = [str(name) for name in layer_names]
    normalized_bits = {
        str(model): sorted({int(bit) for bit in bits})
        for model, bits in bits_by_model.items()
    }
    declared_models_exact = set(normalized_bits) == set(MODELS)

    expected_batch = {
        (model, beta, bit, batch, weight)
        for model in MODELS
        for beta in betas
        for bit in normalized_bits.get(model, [])
        for batch in batch_indices
        for weight in weights
    }
    observed_batch_list = [
        (
            str(row["acquisition_model"]),
            float(row["actual_beta"]),
            int(row["adc_bits"]),
            int(row["batch_index"]),
            str(row["parameter_name"]),
        )
        for row in batch_rows
    ]
    expected_cohort = {
        (model, beta, bit, weight)
        for model in MODELS
        for beta in betas
        for bit in normalized_bits.get(model, [])
        for weight in weights
    }
    observed_cohort_list = [
        (
            str(row["acquisition_model"]),
            float(row["actual_beta"]),
            int(row["adc_bits"]),
            str(row["parameter_name"]),
        )
        for row in cohort_rows
    ]
    expected_configuration = {
        (model, beta, bit)
        for model in MODELS
        for beta in betas
        for bit in normalized_bits.get(model, [])
    }
    observed_configuration_list = [
        (
            str(row["acquisition_model"]),
            float(row["actual_beta"]),
            int(row["adc_bits"]),
        )
        for row in configuration_rows
    ]
    expected_minimum = {(model, beta) for model in MODELS for beta in betas}
    observed_minimum_list = [
        (str(row["acquisition_model"]), float(row["actual_beta"]))
        for row in minimum_rows
    ]
    expected_state = {
        (model, beta, bit, batch, layer)
        for model in MODELS
        for beta in betas
        for bit in normalized_bits.get(model, [])
        for batch in batch_indices
        for layer in layers
    }
    observed_state_list = [
        (
            str(row["acquisition_model"]),
            float(row["actual_beta"]),
            int(row["adc_bits"]),
            int(row["batch_index"]),
            str(row["state_layer_name"]),
        )
        for row in state_rows
    ]
    expected_residual = {
        (phase, beta, layer)
        for phase, phase_betas in (
            ("post_T_free", [0.0]),
            ("zero", [0.0]),
            ("positive", betas),
        )
        for beta in phase_betas
        for layer in layers
    }
    observed_residual_list = [
        (
            str(row["phase"]),
            float(row["beta"]),
            str(row["state_layer_name"]),
        )
        for row in residual_rows
    ]

    def exact(expected: set[Any], observed: Sequence[Any]) -> bool:
        return bool(
            expected
            and len(observed) == len(expected)
            and len(set(observed)) == len(observed)
            and set(observed) == expected
        )

    expected_examples = sum(len(record["source_indices"]) for record in records)
    residual_semantics = bool(
        residual_rows
        and all(
            int(row["example_count"]) == expected_examples
            and int(row["expected_example_count"]) == expected_examples
            and int(row["unique_source_index_count"]) == expected_examples
            and bool(row["coverage_complete"])
            and bool(row["gate_passed"])
            for row in residual_rows
        )
    )
    guards = {
        "nonempty_records_betas_weights_layers_and_bits": bool(
            records
            and beta_rows
            and weights
            and layers
            and declared_models_exact
            and all(normalized_bits[model] for model in MODELS)
        ),
        "declared_acquisition_models_exact": declared_models_exact,
        "batch_gradient_key_coverage_exact": exact(
            expected_batch, observed_batch_list
        ),
        "cohort_gradient_key_coverage_exact": exact(
            expected_cohort, observed_cohort_list
        ),
        "configuration_key_coverage_exact": exact(
            expected_configuration, observed_configuration_list
        ),
        "minimum_summary_key_coverage_exact": exact(
            expected_minimum, observed_minimum_list
        ),
        "state_adc_key_coverage_exact": exact(expected_state, observed_state_list),
        "residual_key_coverage_exact": exact(
            expected_residual, observed_residual_list
        ),
        "residual_semantics_and_gates_pass": residual_semantics,
        "expected_counts": {
            "batch_gradient_rows": len(expected_batch),
            "cohort_gradient_rows": len(expected_cohort),
            "configuration_rows": len(expected_configuration),
            "minimum_rows": len(expected_minimum),
            "state_rows": len(expected_state),
            "residual_rows": len(expected_residual),
        },
        "observed_counts": {
            "batch_gradient_rows": len(observed_batch_list),
            "cohort_gradient_rows": len(observed_cohort_list),
            "configuration_rows": len(observed_configuration_list),
            "minimum_rows": len(observed_minimum_list),
            "state_rows": len(observed_state_list),
            "residual_rows": len(observed_residual_list),
        },
    }
    guards["all_output_and_residual_coverage_guards_passed"] = bool(
        all(value for value in guards.values() if isinstance(value, bool))
    )
    return guards


def _plot_gradient_fidelity(
    path: Path,
    configuration_rows: Sequence[Mapping[str, Any]],
    *,
    cosine_minimum: float,
) -> None:
    if not configuration_rows:
        raise ValueError("Cannot plot an empty ADC configuration summary.")
    figure, axes = plt.subplots(1, len(MODELS), figsize=(12.0, 4.6), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]
    for axis, model in zip(axes, MODELS, strict=True):
        model_rows = [
            row for row in configuration_rows if row["acquisition_model"] == model
        ]
        if not model_rows:
            raise ValueError(f"No configuration rows for acquisition model {model}.")
        beta_hats = sorted(
            {float(row["beta_hat_base_requested"]) for row in model_rows}
        )
        for beta_hat in beta_hats:
            rows = sorted(
                (
                    row
                    for row in model_rows
                    if float(row["beta_hat_base_requested"]) == beta_hat
                ),
                key=lambda row: int(row["adc_bits"]),
            )
            axis.plot(
                [int(row["adc_bits"]) for row in rows],
                [
                    (
                        float(row["minimum_batch_quantized_vs_ideal_cosine"])
                        if row["minimum_batch_quantized_vs_ideal_cosine"] is not None
                        else np.nan
                    )
                    for row in rows
                ],
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                label=rf"$\hat{{\beta}}_{{base}}={beta_hat:g}$",
            )
        axis.axhline(
            float(cosine_minimum), color="black", linestyle="--", linewidth=1.0
        )
        axis.set_xlabel("nominal ideal ADC bits")
        axis.set_title(
            "absolute endpoint ADC"
            if model == "absolute_endpoint_shared_adc"
            else "analog-delta control"
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("worst layer/minibatch cosine to clean EqProp")
    axes[0].set_ylim(-0.08, 1.015)
    figure.suptitle("Conv3 legacy EqProp gradient fidelity after node readout")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_code_change_fraction(
    path: Path,
    state_rows: Sequence[Mapping[str, Any]],
    *,
    primary_beta_hat: float,
) -> None:
    selected = [
        row
        for row in state_rows
        if row["acquisition_model"] == "absolute_endpoint_shared_adc"
        and math.isclose(
            float(row["beta_hat_base_requested"]),
            float(primary_beta_hat),
            rel_tol=1.0e-12,
            abs_tol=0.0,
        )
    ]
    groups: defaultdict[tuple[str, int], list[float]] = defaultdict(list)
    for row in selected:
        groups[(str(row["state_layer_name"]), int(row["adc_bits"]))].append(
            float(row["endpoint_code_change_fraction"])
        )
    if not groups:
        raise ValueError("No primary-beta absolute ADC state rows to plot.")
    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    for layer_name in sorted({key[0] for key in groups}):
        points = sorted(
            (
                bits,
                float(np.mean(groups[(layer_name, bits)])),
            )
            for name, bits in groups
            if name == layer_name
        )
        axis.plot(
            [value[0] for value in points],
            [value[1] for value in points],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=layer_name,
        )
    axis.set_xlabel("nominal ideal ADC bits")
    axis.set_ylabel("mean fraction of endpoint codes that change")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title(rf"Absolute ADC code changes at $\hat{{\beta}}_{{base}}={primary_beta_hat:g}$")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    path: Path,
    *,
    study_config: Mapping[str, Any],
    minimum_rows: Sequence[Mapping[str, Any]],
    configuration_rows: Sequence[Mapping[str, Any]],
    absolute_ranges: Mapping[str, float],
    smoke: bool,
) -> None:
    table_rows = []
    for row in minimum_rows:
        if row["minimum_passing_adc_bits"] is not None:
            threshold = str(int(row["minimum_passing_adc_bits"]))
        elif row["passing_upper_bound_adc_bits"] is not None:
            threshold = f"<={int(row['passing_upper_bound_adc_bits'])} (lower edge open)"
        else:
            threshold = "unresolved"
        ideal_cosine = row["minimum_ideal_eqprop_vs_bptt_cosine"]
        table_rows.append(
            (
                row["acquisition_model"],
                f"{float(row['beta_hat_base_requested']):.3g}",
                f"{float(row['actual_beta']):.7g}",
                f"{float(row['effective_beta']):.7g}",
                threshold,
                (
                    "undefined"
                    if ideal_cosine is None
                    else f"{float(ideal_cosine):.6f}"
                ),
                str(int(row["pass_to_fail_reversal_count"])),
            )
        )
    primary = float(study_config["case"]["primary_base_beta_hat"])
    primary_absolute = next(
        (
            row
            for row in minimum_rows
            if row["acquisition_model"] == "absolute_endpoint_shared_adc"
            and math.isclose(
                float(row["beta_hat_base_requested"]),
                primary,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        ),
        None,
    )
    lines = [
        "# Conv3 legacy EqProp ADC readout precision",
        "",
        f"Run class: `{'smoke' if smoke else 'production'}`. Evidence class: `{study_config['evidence_class']}`; diagnostic and not paper-facing.",
        "",
        "The analog zero and positive phases are solved in true float64. Node states are then quantized before the local squared-voltage-drop statistics are evaluated. The two endpoint statistics are subtracted in float64, matching the implemented digital EqProp path. Consequently, the primary result combines ideal ADC quantization with float64 endpoint-statistic cancellation; it is not an ADC-only lower bound. The primary comparison is against clean float64 EqProp at the identical beta; BPTT is reported separately so readout loss is not confused with finite-beta estimator bias.",
        "",
        base._markdown_table(
            (
                "Acquisition",
                "Base beta hat",
                "Actual beta",
                "Injected beta",
                "Sustained nominal ADC-bit threshold",
                "Worst clean EqProp/BPTT cosine",
                "Pass-to-fail reversals",
            ),
            table_rows,
        ),
        "",
        f"The fidelity gate requires every scored weight and every minibatch to have cosine >= {float(study_config['gradient_fidelity_gate']['cosine_minimum']):.2f} and symmetric norm delta <= {float(study_config['gradient_fidelity_gate']['symmetric_norm_delta_maximum']):.2f} relative to clean EqProp.",
        "",
        f"`absolute_endpoint_shared_adc` uses one signed symmetric mid-tread transfer function per state layer. Its full scale is calibrated on the same selected matched-zero cohort being scored ({16 if smoke else 64} examples), multiplied by {float(study_config['adc']['absolute_range_headroom']):g}, and then frozen across phases, beta values, and minibatches. This in-sample range calibration is optimistic, as are the exclusions of temporal read noise, gain/offset error, and converter mismatch. The bit count is the nominal resolution of this ideal transfer function, not hardware ENOB.",
        "",
        "`analog_delta_ideal_common` subtracts the endpoint states before conversion, uses an oracle per-beta differential range, and retains the common state exactly. It is an architectural control, not a complete hardware claim.",
        "",
        "Zero-calibrated absolute full scales: "
        + ", ".join(f"`{name}={value:.7g}`" for name, value in absolute_ranges.items())
        + ".",
        "",
        (
            "Primary absolute-ADC result: unresolved in the tested range."
            if primary_absolute is None
            or primary_absolute["sustained_passing_from_adc_bits"] is None
            else (
                "Primary absolute-ADC result: the sustained threshold is at or below "
                f"{int(primary_absolute['passing_upper_bound_adc_bits'])} nominal bits; the lower edge is open."
                if primary_absolute["minimum_passing_adc_bits"] is None
                else "Primary absolute-ADC result: the first point that passes and remains passing at every higher tested nominal bit depth is "
                f"{int(primary_absolute['minimum_passing_adc_bits'])} bits."
            )
        ),
        "",
        "DAC and conductance programming are intentionally excluded. They require a declared optimizer proposal, conductance range, write transfer function, and accumulation policy, and cannot restore a gradient already lost at acquisition.",
        "",
        "Artifacts: `accepted_reference_compatibility.csv`, `batch_gradient_fidelity.csv`, `cohort_gradient_fidelity.csv`, `configuration_summary.csv`, `minimum_passing_bits.csv`, `state_adc_metrics.csv`, `phase_residuals.csv`, `read_only_guards.json`, the four exact local analysis-source snapshots, and the two PNG plots.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source_config_path = args.config.expanduser().resolve()
    study_config_path = args.study_config.expanduser().resolve()
    source_config = base._read_json(source_config_path)
    study_config = base._read_json(study_config_path)
    _validate_configs(source_config, study_config)
    declared_source_config = _resolve_repository_path(
        study_config["source"]["beta_config"]
    )
    if declared_source_config != source_config_path:
        raise ValueError(
            f"--config {source_config_path} differs from study source {declared_source_config}."
        )

    source_run = args.source_run.expanduser().resolve()
    declared_source_run = _resolve_repository_path(
        study_config["source"]["native_beta_run"]
    )
    if source_run != declared_source_run:
        raise ValueError("--source-run differs from the frozen study config.")
    source_errors = base.validate_run(source_run)
    if source_errors:
        raise ValueError("Invalid source run: " + "; ".join(source_errors))
    if base.sha256_file(source_run / "result.json") != study_config["source"][
        "native_beta_result_sha256"
    ]:
        raise ValueError("Native beta source result changed.")
    source_run_config = base._read_json(source_run / "config.resolved.json")
    if _semantic_sha256(source_run_config) != _semantic_sha256(source_config):
        raise ValueError("Source run resolved config differs from --config.")

    reference_run = _resolve_repository_path(
        study_config["source"]["accepted_float64_reference_run"]
    )
    reference_errors = base.validate_run(reference_run)
    if reference_errors:
        raise ValueError("Invalid accepted float64 reference: " + "; ".join(reference_errors))
    if base.sha256_file(reference_run / "result.json") != study_config["source"][
        "accepted_float64_reference_result_sha256"
    ]:
        raise ValueError("Accepted float64 reference result changed.")
    reference_archive = (
        reference_run
        / "artifacts/eqprop_gradients/conv3__legacy__best_validation.npz"
    )
    if not reference_archive.is_file():
        raise FileNotFoundError(reference_archive)

    runtime_source = base._validate_runtime_source(source_config)
    case, inventory, source_hashes_before = _case_inventory(
        source_config, study_config
    )
    beta_rows = _resolve_beta_rows(
        source_config,
        case,
        study_config["case"]["base_beta_hat_requested"],
    )
    if args.smoke:
        primary = float(study_config["case"]["primary_base_beta_hat"])
        beta_rows = [
            row
            for row in beta_rows
            if math.isclose(
                float(row["beta_hat_requested"]),
                primary,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        ]
    batches, materialized_cohort, source_cohort = _materialize_fixed_cohort(
        source_run=source_run,
        source_config_for_dataset=inventory[0]["source_config"],
        dataset_root=args.dataset_root.expanduser().resolve(),
    )
    if base.sha256_file(source_run / "cohort.json") != study_config["source"][
        "cohort_sha256"
    ]:
        raise ValueError("Frozen cohort file changed.")
    selected_batches = batches[:1] if args.smoke else batches

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    output_root = args.output_root.expanduser().resolve()
    run_id = args.run_id or ("smoke" if args.smoke else "production")
    if not run_id or "/" in run_id or run_id in {".", ".."}:
        raise ValueError(f"Invalid run id {run_id!r}.")
    run_dir = output_root / run_id
    absolute_coarse = [
        int(value) for value in study_config["adc"]["absolute_coarse_bits"]
    ]
    differential_coarse = [
        int(value) for value in study_config["adc"]["differential_coarse_bits"]
    ]
    if args.smoke:
        absolute_coarse = [16, 40, 52]
        differential_coarse = [8, 12, 16]
    gate = study_config["gradient_fidelity_gate"]
    cosine_minimum = float(gate["cosine_minimum"])
    norm_delta_maximum = float(gate["symmetric_norm_delta_maximum"])
    residual_threshold = float(
        source_config["equilibrium_residual_contract"].get(
            "per_layer_sample_max_p90_threshold",
            source_config["equilibrium_residual_contract"].get(
                "p90_threshold", 1.0e-2
            ),
        )
    )
    analysis_sources = {
        "analyzer": Path(__file__).resolve(),
        "float64_audit": Path(audit.__file__).resolve(),
        "extended_eqprop_analysis": Path(extended.__file__).resolve(),
        "base_eqprop_analysis": Path(base.__file__).resolve(),
    }
    analysis_source_declarations = [
        {
            "label": label,
            "source_path": str(path),
            "source_sha256": base.sha256_file(path),
            "snapshot_name": f"source_snapshot__{label}{path.suffix}",
        }
        for label, path in analysis_sources.items()
    ]

    manifest = {
        "study_id": study_config["study_id"],
        "run_id": run_id,
        "arm_id": "conv3_legacy_positive_eqprop_node_adc_precision",
        "evidence_class": study_config["evidence_class"],
        "dataset": source_config["dataset"]["name"],
        "smoke": bool(args.smoke),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "configuration": {
            "study_config_path": str(study_config_path),
            "study_config_sha256": base.sha256_file(study_config_path),
            "source_beta_config_path": str(source_config_path),
            "source_beta_config_sha256": base.sha256_file(source_config_path),
            "source_beta_config_semantic_sha256": _semantic_sha256(source_config),
            "resolved": {
                "case": {
                    "architecture": case["architecture"],
                    "scheme": case["scheme"],
                    "checkpoint_role": study_config["case"]["checkpoint_role"],
                    "T": int(study_config["case"]["native_T"]),
                    "K": int(study_config["case"]["native_K"]),
                },
                "beta_rows": beta_rows,
                "absolute_coarse_bits": absolute_coarse,
                "differential_coarse_bits": differential_coarse,
                "smoke": bool(args.smoke),
                "selected_batch_count": len(selected_batches),
                "adc": study_config["adc"],
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
            "native_beta_result_sha256": base.sha256_file(
                source_run / "result.json"
            ),
            "accepted_float64_reference_run": str(reference_run),
            "accepted_float64_reference_result_sha256": base.sha256_file(
                reference_run / "result.json"
            ),
            "accepted_float64_reference_archive": str(reference_archive),
            "accepted_float64_reference_archive_sha256": base.sha256_file(
                reference_archive
            ),
            "cohort_sha256": base.sha256_file(source_run / "cohort.json"),
        },
        "replay": {
            "runtime_dtype": "float64",
            "endpoint_acquisition_models": list(MODELS),
            "input_layer_quantized": False,
            "analog_noise_included": False,
            "gain_offset_mismatch_included": False,
            "dac_or_weight_programming_included": False,
            "biases_active_in_dynamics": True,
            "bias_gradients_excluded": True,
            "optimizer_constructed": False,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        },
        "analyzer": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.sha256_file(Path(__file__).resolve()),
            "float64_audit_path": str(FLOAT64_AUDIT),
            "float64_audit_sha256": base.sha256_file(FLOAT64_AUDIT),
            "local_analysis_source_snapshots": analysis_source_declarations,
            "repository_head": base._git_output(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        },
    }

    base.start_run(run_dir, manifest)
    try:
        analysis_source_snapshots = _write_source_snapshots(
            run_dir, analysis_sources
        )
        base._write_json(run_dir / "study_config.resolved.json", study_config)
        base._write_json(run_dir / "source_beta_config.resolved.json", source_config)
        base._write_json(run_dir / "source_cohort.json", source_cohort)
        base._write_json(run_dir / "materialized_cohort.json", materialized_cohort)
        base.update_status_progress(
            run_dir,
            {
                "stage": "capturing_float64_endpoints",
                "batch_count": len(selected_batches),
                "beta_count": len(beta_rows),
            },
        )
        runtime, records, residual_rows, read_only = _collect_ideal_endpoints(
            case=case,
            batches=selected_batches,
            beta_rows=beta_rows,
            device=device,
            T=int(study_config["case"]["native_T"]),
            K=int(study_config["case"]["native_K"]),
            checkpoint_role=str(study_config["case"]["checkpoint_role"]),
            residual_threshold=residual_threshold,
        )
        accepted_reference = _accepted_reference_compatibility(
            runtime=runtime,
            records=records,
            beta_rows=beta_rows,
            primary_beta_hat=float(study_config["case"]["primary_base_beta_hat"]),
            reference_archive=reference_archive,
            expected_T=int(study_config["case"]["native_T"]),
            expected_K=int(study_config["case"]["native_K"]),
        )
        read_only["accepted_reference_compatibility"] = {
            key: value
            for key, value in accepted_reference.items()
            if key != "comparison_rows"
        }
        read_only["guards"]["accepted_reference_reproduced"] = bool(
            accepted_reference["all_numeric_matches"]
        )
        read_only["analysis_source_snapshots"] = analysis_source_snapshots
        read_only["guards"]["all_analysis_source_snapshots_match"] = bool(
            analysis_source_snapshots
            and all(
                bool(row["snapshot_matches_source"])
                for row in analysis_source_snapshots
            )
        )
        layer_names = [str(layer.name) for layer in runtime["free_layers"]]
        absolute_ranges = _absolute_ranges(
            records,
            layer_names,
            headroom=float(study_config["adc"]["absolute_range_headroom"]),
        )
        differential_ranges = _differential_ranges(
            records,
            layer_names,
            beta_rows,
            headroom=float(study_config["adc"]["differential_range_headroom"]),
        )
        base._write_json(
            run_dir / "adc_ranges.json",
            {
                "absolute_ranges": absolute_ranges,
                "differential_ranges_by_actual_beta": {
                    str(beta): values for beta, values in differential_ranges.items()
                },
                "layer_names": layer_names,
            },
        )

        batch_rows: list[dict[str, Any]] = []
        state_rows: list[dict[str, Any]] = []
        gradient_sets: dict[tuple[Any, ...], dict[str, list[Any]]] = {}
        coarse_by_model = {
            "absolute_endpoint_shared_adc": absolute_coarse,
            "analog_delta_ideal_common": differential_coarse,
        }
        evaluated_bits_by_model: dict[str, set[int]] = {
            model: set() for model in MODELS
        }
        for model in MODELS:
            base.update_status_progress(
                run_dir,
                {
                    "stage": "adc_coarse_sweep",
                    "acquisition_model": model,
                    "bits": coarse_by_model[model],
                },
            )
            _evaluate_adc_bits(
                runtime=runtime,
                records=records,
                beta_rows=beta_rows,
                model=model,
                bits_values=coarse_by_model[model],
                absolute_ranges=absolute_ranges,
                differential_ranges=differential_ranges,
                device=device,
                cosine_minimum=cosine_minimum,
                symmetric_norm_delta_maximum=norm_delta_maximum,
                batch_rows=batch_rows,
                state_rows=state_rows,
                gradient_sets=gradient_sets,
            )
            evaluated_bits_by_model[model].update(coarse_by_model[model])
            if bool(study_config["adc"]["refine_transition_to_single_bit"]) and not args.smoke:
                interim_cohort = _cohort_gradient_rows(
                    gradient_sets=gradient_sets,
                    beta_rows=beta_rows,
                    cosine_minimum=cosine_minimum,
                    symmetric_norm_delta_maximum=norm_delta_maximum,
                )
                interim_configurations = _configuration_rows(interim_cohort)
                refinement: set[int] = set()
                for beta_row in beta_rows:
                    beta = float(beta_row["beta"])
                    candidates = [
                        row
                        for row in interim_configurations
                        if row["acquisition_model"] == model
                        and float(row["actual_beta"]) == beta
                    ]
                    pass_by_bit = {
                        int(row["adc_bits"]): bool(
                            row["all_weight_layers_and_minibatches_passed"]
                        )
                        for row in candidates
                    }
                    refinement.update(
                        _refinement_bits(coarse_by_model[model], pass_by_bit)
                    )
                if refinement:
                    refined = sorted(refinement)
                    base.update_status_progress(
                        run_dir,
                        {
                            "stage": "adc_transition_refinement",
                            "acquisition_model": model,
                            "bits": refined,
                        },
                    )
                    _evaluate_adc_bits(
                        runtime=runtime,
                        records=records,
                        beta_rows=beta_rows,
                        model=model,
                        bits_values=refined,
                        absolute_ranges=absolute_ranges,
                        differential_ranges=differential_ranges,
                        device=device,
                        cosine_minimum=cosine_minimum,
                        symmetric_norm_delta_maximum=norm_delta_maximum,
                        batch_rows=batch_rows,
                        state_rows=state_rows,
                        gradient_sets=gradient_sets,
                    )
                    evaluated_bits_by_model[model].update(refined)

        cohort_rows = _cohort_gradient_rows(
            gradient_sets=gradient_sets,
            beta_rows=beta_rows,
            cosine_minimum=cosine_minimum,
            symmetric_norm_delta_maximum=norm_delta_maximum,
        )
        configuration_rows = _configuration_rows(cohort_rows)
        minimum_rows = _minimum_passing_rows(configuration_rows)
        coverage = _coverage_guards(
            records=records,
            beta_rows=beta_rows,
            bits_by_model={
                model: sorted(bits)
                for model, bits in evaluated_bits_by_model.items()
            },
            weight_names=runtime["weight_names"],
            layer_names=layer_names,
            batch_rows=batch_rows,
            cohort_rows=cohort_rows,
            configuration_rows=configuration_rows,
            minimum_rows=minimum_rows,
            state_rows=state_rows,
            residual_rows=residual_rows,
        )
        read_only["coverage_guards"] = coverage
        read_only["guards"]["all_output_and_residual_coverage_guards_passed"] = bool(
            coverage["all_output_and_residual_coverage_guards_passed"]
        )

        source_hashes_before_json = {
            "/".join(key): value for key, value in source_hashes_before.items()
        }
        source_hashes_after = extended._verify_source_hashes(
            inventory, source_hashes_before
        )
        read_only["guards"].update(
            {
                "source_hashes_before": source_hashes_before_json,
                "source_hashes_after": source_hashes_after,
                "source_bundle_files_unchanged": source_hashes_after
                == source_hashes_before_json,
                "parameter_state_sha256_after_adc_evaluation": base._parameter_state_sha256(
                    runtime["parameters"]
                ),
            }
        )
        read_only["guards"]["parameters_unchanged_after_adc_evaluation"] = (
            read_only["guards"]["parameter_state_sha256_after_adc_evaluation"]
            == read_only["guards"]["parameter_state_sha256_before"]
        )
        if not read_only["guards"]["source_bundle_files_unchanged"]:
            raise RuntimeError("Source bundle files changed during ADC replay.")
        if not read_only["guards"]["parameters_unchanged_after_adc_evaluation"]:
            raise RuntimeError("Parameters changed during ADC evaluation.")

        base._write_csv(run_dir / "batch_gradient_fidelity.csv", batch_rows)
        base._write_csv(run_dir / "cohort_gradient_fidelity.csv", cohort_rows)
        base._write_csv(run_dir / "configuration_summary.csv", configuration_rows)
        base._write_csv(run_dir / "minimum_passing_bits.csv", minimum_rows)
        base._write_csv(run_dir / "state_adc_metrics.csv", state_rows)
        base._write_csv(run_dir / "phase_residuals.csv", residual_rows)
        base._write_csv(
            run_dir / "accepted_reference_compatibility.csv",
            accepted_reference["comparison_rows"],
        )
        base._write_json(run_dir / "read_only_guards.json", read_only)
        _plot_gradient_fidelity(
            run_dir / "gradient_fidelity_vs_adc_bits.png",
            configuration_rows,
            cosine_minimum=cosine_minimum,
        )
        _plot_code_change_fraction(
            run_dir / "absolute_adc_code_change_fraction.png",
            state_rows,
            primary_beta_hat=float(study_config["case"]["primary_base_beta_hat"]),
        )
        _write_report(
            run_dir / "report.md",
            study_config=study_config,
            minimum_rows=minimum_rows,
            configuration_rows=configuration_rows,
            absolute_ranges=absolute_ranges,
            smoke=bool(args.smoke),
        )

        for row in minimum_rows:
            base.append_metric(
                run_dir / "metrics.jsonl",
                {
                    "stage": "adc_precision_selection",
                    "split": "ordinary_mnist_validation",
                    "acquisition_model": row["acquisition_model"],
                    "beta_hat_base_requested": row[
                        "beta_hat_base_requested"
                    ],
                    "actual_beta": row["actual_beta"],
                    "effective_beta": row["effective_beta"],
                    "minimum_passing_adc_bits": row[
                        "minimum_passing_adc_bits"
                    ],
                    "sustained_passing_from_adc_bits": row[
                        "sustained_passing_from_adc_bits"
                    ],
                    "passing_upper_bound_adc_bits": row[
                        "passing_upper_bound_adc_bits"
                    ],
                    "pass_to_fail_reversal_count": row[
                        "pass_to_fail_reversal_count"
                    ],
                    "threshold_interpretation": row[
                        "threshold_interpretation"
                    ],
                    "gate_resolved": row["gate_resolved"],
                    "official_test_read": False,
                },
            )

        primary_beta_hat = float(study_config["case"]["primary_base_beta_hat"])
        primary_absolute = next(
            row
            for row in minimum_rows
            if row["acquisition_model"] == "absolute_endpoint_shared_adc"
            and math.isclose(
                float(row["beta_hat_base_requested"]),
                primary_beta_hat,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        )
        primary_differential = next(
            row
            for row in minimum_rows
            if row["acquisition_model"] == "analog_delta_ideal_common"
            and math.isclose(
                float(row["beta_hat_base_requested"]),
                primary_beta_hat,
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        )
        terminal_metrics = {
            "checkpoint_cases": 1,
            "batch_count": len(records),
            "example_count": sum(len(row["source_indices"]) for row in records),
            "beta_count": len(beta_rows),
            "acquisition_model_count": len(MODELS),
            "scored_weight_count": len(runtime["weight_names"]),
            "batch_gradient_comparison_count": len(batch_rows),
            "state_adc_measurement_count": len(state_rows),
            "primary_beta_hat_base_requested": primary_beta_hat,
            "primary_absolute_minimum_passing_adc_bits": primary_absolute[
                "minimum_passing_adc_bits"
            ],
            "primary_absolute_sustained_passing_from_adc_bits": primary_absolute[
                "sustained_passing_from_adc_bits"
            ],
            "primary_absolute_passing_upper_bound_adc_bits": primary_absolute[
                "passing_upper_bound_adc_bits"
            ],
            "primary_absolute_pass_to_fail_reversal_count": primary_absolute[
                "pass_to_fail_reversal_count"
            ],
            "primary_differential_minimum_passing_adc_bits": primary_differential[
                "minimum_passing_adc_bits"
            ],
            "primary_differential_sustained_passing_from_adc_bits": primary_differential[
                "sustained_passing_from_adc_bits"
            ],
            "primary_differential_passing_upper_bound_adc_bits": primary_differential[
                "passing_upper_bound_adc_bits"
            ],
            "primary_differential_pass_to_fail_reversal_count": primary_differential[
                "pass_to_fail_reversal_count"
            ],
            "all_residual_gates_passed": bool(
                read_only["guards"]["all_residual_gates_passed"]
            ),
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        selected_betas = {float(row["beta"]) for row in beta_rows}
        selected_scope_captured = bool(
            records
            and len(records) == len(selected_batches)
            and {int(row["batch_index"]) for row in records}
            == {int(row["batch_index"]) for row in selected_batches}
            and all(
                {float(value) for value in record["positives"]}
                == selected_betas
                for record in records
            )
        )
        production_declared_scope_captured = bool(
            not args.smoke
            and len(records) == int(study_config["cohort"]["batch_count"])
            and len(beta_rows)
            == len(study_config["case"]["base_beta_hat_requested"])
            and sum(len(row["source_indices"]) for row in records)
            == int(study_config["cohort"]["example_count"])
        )
        completion = {
            "selected_smoke_or_production_scope_captured": selected_scope_captured,
            "production_declared_scope_captured": production_declared_scope_captured,
            "absolute_and_differential_models_reported": set(
                row["acquisition_model"] for row in minimum_rows
            )
            == set(MODELS)
            and bool(minimum_rows),
            "all_scored_weight_layers_reported": bool(
                configuration_rows
                and all(
                    int(row["scored_weight_count"])
                    == len(runtime["weight_names"])
                    for row in configuration_rows
                )
            ),
            "all_output_and_residual_coverage_guards_passed": bool(
                coverage["all_output_and_residual_coverage_guards_passed"]
            ),
            "all_residual_gates_passed": bool(
                read_only["guards"]["all_residual_gates_passed"]
                and coverage["residual_semantics_and_gates_pass"]
            ),
            "source_and_parameter_hash_guards_pass": bool(
                read_only["guards"]["source_bundle_files_unchanged"]
                and read_only["guards"]["parameters_unchanged_after_adc_evaluation"]
            ),
            "common_post_T_phase_start_guards_pass": bool(
                read_only["guards"]["all_phase_starts_match_common_post_T"]
            ),
            "accepted_float64_reference_reproduced": bool(
                read_only["guards"]["accepted_reference_reproduced"]
            ),
            "analysis_sources_archived_exactly": bool(
                read_only["guards"]["all_analysis_source_snapshots_match"]
            ),
            "beta_effective_and_amplification_guards_pass": bool(
                read_only["guards"][
                    "all_beta_effective_and_amplification_guards_passed"
                ]
            ),
            "float64_dynamics_and_endpoint_arithmetic": True,
            "adc_applied_before_local_weight_gradient_evaluation": True,
            "clean_eqprop_and_bptt_references_reported_separately": True,
            "endpoint_statistic_subtraction_labeled_as_float64_cancellation_inclusive": True,
            "dac_or_weight_programming_excluded": True,
            "bias_gradients_excluded": True,
            "optimizer_steps_applied": False,
            "official_test_read": False,
        }
        required_completion_keys = (
            "selected_smoke_or_production_scope_captured",
            "absolute_and_differential_models_reported",
            "all_scored_weight_layers_reported",
            "all_output_and_residual_coverage_guards_passed",
            "all_residual_gates_passed",
            "source_and_parameter_hash_guards_pass",
            "common_post_T_phase_start_guards_pass",
            "accepted_float64_reference_reproduced",
            "analysis_sources_archived_exactly",
            "beta_effective_and_amplification_guards_pass",
        )
        if not args.smoke:
            required_completion_keys = (
                *required_completion_keys,
                "production_declared_scope_captured",
            )
        completion["criteria_met"] = bool(
            all(bool(completion[key]) for key in required_completion_keys)
            and completion["optimizer_steps_applied"] is False
            and completion["official_test_read"] is False
        )
        if not completion["criteria_met"]:
            failed = [
                key for key in required_completion_keys if not completion[key]
            ]
            raise RuntimeError(
                "ADC replay completion guards failed: " + ", ".join(failed)
            )

        required_artifacts = (
            "accepted_reference_compatibility.csv",
            "adc_ranges.json",
            "batch_gradient_fidelity.csv",
            "cohort_gradient_fidelity.csv",
            "configuration_summary.csv",
            "minimum_passing_bits.csv",
            "state_adc_metrics.csv",
            "phase_residuals.csv",
            "read_only_guards.json",
            "gradient_fidelity_vs_adc_bits.png",
            "absolute_adc_code_change_fraction.png",
            "report.md",
            "study_config.resolved.json",
            "source_beta_config.resolved.json",
            "source_cohort.json",
            "materialized_cohort.json",
            "source_snapshot__analyzer.py",
            "source_snapshot__float64_audit.py",
            "source_snapshot__extended_eqprop_analysis.py",
            "source_snapshot__base_eqprop_analysis.py",
        )
        missing_artifacts = [
            name for name in required_artifacts if not (run_dir / name).is_file()
        ]
        if missing_artifacts:
            raise RuntimeError(
                "Missing required ADC replay artifacts: "
                + ", ".join(missing_artifacts)
            )
        precompletion_errors = base.validate_run(run_dir)
        if precompletion_errors:
            raise RuntimeError(
                "Running ADC bundle failed pre-completion validation: "
                + "; ".join(precompletion_errors)
            )
        result = base.complete_run(
            run_dir,
            terminal_metrics=terminal_metrics,
            completion=completion,
        )
        errors = base.validate_run(run_dir)
        if errors:
            validation_error = RuntimeError(
                "Completed ADC bundle failed validation: " + "; ".join(errors)
            )
            result_path = run_dir / "result.json"
            if result_path.is_file():
                result_path.unlink()
            base.fail_run(run_dir, error=validation_error)
            raise validation_error
        return result
    except BaseException as error:
        status_path = run_dir / "status.json"
        if status_path.is_file():
            status = base._read_json(status_path)
            result_path = run_dir / "result.json"
            if result_path.is_file() and status.get("state") in {"running", "complete"}:
                result_path.unlink()
            if status.get("state") in {"running", "complete"} and not result_path.exists():
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
    parser.add_argument(
        "--runtime-source-root", type=Path, default=base.RUNTIME_SOURCE_ROOT
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/filip/datasets/mnist")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target", default="local:RTX3090")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runtime_source_root.expanduser().resolve() != base.RUNTIME_SOURCE_ROOT:
        raise ValueError(
            "--runtime-source-root must be supplied at process bootstrap and match the exact source contract."
        )
    result = _run(args)
    run_id = args.run_id or ("smoke" if args.smoke else "production")
    result_path = args.output_root.expanduser().resolve() / run_id / "result.json"
    print(
        json.dumps(
            {
                "state": "complete",
                "result_sha256": base.sha256_file(result_path),
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
