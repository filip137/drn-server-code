"""Run direct initialization, endpoint-noise HWA, and pulse recovery.

This is an exploratory, matched DRN experiment for the synthetic Figure-6
HfO2 endpoint model.  It is intentionally a direct executable rather than a
new public ``ebl`` schema while the endpoint assumptions are still being
developed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.config import (
    parse_student_config,
    resolve_student_spec,
)
from experiments.mnist_relu_drn.hfo2_figure6_drn import (
    ENDPOINT_LAYOUTS,
    HFO2_CYCLE_NOISE_STD,
    HFO2_DW_MIN_DTOD_LOG_STD,
    HFO2_NOMINAL_DW_MIN_PROGRESS,
    HFO2_NOMINAL_DW_MIN_RAW_A,
    HFO2_UP_DOWN_DTOD,
    HFO2_WRITE_NOISE_STD,
    EndpointField,
    HfO2Figure6PulsePlant,
    PersistentHfO2PulseAdam,
    endpoint_field_from_population,
    lift_endpoint_gradients,
    map_masters_to_conductance,
    master_to_progress,
    rotate_endpoint_field,
)
from experiments.mnist_relu_drn.hfo2_figure6_endpoint_regimes import (
    ENDPOINT_REGIMES,
    INDEPENDENT_ENDPOINTS,
    RANK_MATCHED_ENDPOINTS,
    sample_hfo2_figure6_endpoint_population,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    apply_full_conductance_targets,
)
from experiments.mnist_relu_drn.ibm_om_global_positive_g import (
    map_global_positive_g,
)
from experiments.mnist_relu_drn.runtime import _evaluate, _load_teacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import RunMode
from training.checkpoint import atomic_torch_save


_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "mnist_hfo2_figure6_direct_hwa_recovery.v1"
EXPECTED_TEACHER_SHA256 = (
    "9a961a77628e365b54fdf59304ec7fddf46f1f4834c4c03cecea9c204f837f52"
)
FIXED_LOGIT_GAIN = 14.12537544622754
CONDUCTANCE_CEILING = 6.0
HWA_LEARNING_RATES = (0.004411914893617021, 0.000011974808510638298)
OM_RECOVERY_LEARNING_RATE_PROGRESS = 3.0e-5
OM_NOMINAL_DELTA_PROGRESS = 0.04745
HFO2_RECOVERY_LEARNING_RATE_PROGRESS = (
    OM_RECOVERY_LEARNING_RATE_PROGRESS
    * HFO2_NOMINAL_DW_MIN_PROGRESS
    / OM_NOMINAL_DELTA_PROGRESS
)

PROFILE_SETTINGS: Mapping[str, Mapping[str, Any]] = {
    "smoke": {
        "batch_size": 16,
        "training_points": 64,
        "validation_points": 64,
        "validation_sample_limit": 64,
        "test_sample_limit": 64,
        "hwa_epochs": 1,
        "hwa_bank_size": 1,
        "hwa_assignment_rotations": 1,
        "maximum_batches_per_epoch": 1,
        "recovery_epochs": 1,
        "recovery_pulse_cap": 4,
        "progress_log_interval": 1,
    },
    "pilot": {
        "batch_size": 16,
        "training_points": 8192,
        "validation_points": 2048,
        "validation_sample_limit": 2048,
        "test_sample_limit": None,
        "hwa_epochs": 5,
        "hwa_bank_size": 4,
        "hwa_assignment_rotations": 8,
        "maximum_batches_per_epoch": None,
        "recovery_epochs": 3,
        "recovery_pulse_cap": 64,
        "progress_log_interval": 128,
    },
    "full": {
        "batch_size": 16,
        "training_points": None,
        "validation_points": 5000,
        "validation_sample_limit": None,
        "test_sample_limit": None,
        "hwa_epochs": 10,
        "hwa_bank_size": 8,
        "hwa_assignment_rotations": 8,
        "maximum_batches_per_epoch": None,
        "recovery_epochs": 3,
        "recovery_pulse_cap": 64,
        "progress_log_interval": 500,
    },
}


def _student_payload(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": "mnist_relu_drn_kd.v1",
        "runtime": {
            "seed": 42,
            "data_seed": 42,
            "device": "cuda",
            "dtype": "float32",
        },
        "data": {
            "batch_size": int(profile["batch_size"]),
            "validation_points": int(profile["validation_points"]),
            "num_points": profile["training_points"],
            "shuffle": True,
        },
        "teacher": {
            "type": "bias_free_relu",
            "initialization": "signed_weight_mapping",
        },
        "model": {
            "dims": [1568, 100, 20],
            "input_gain": 100.0,
            "conductance_min": 0.0,
            "conductance_max": CONDUCTANCE_CEILING,
            "voltage_amp": 4.0,
            "current_amp": 0.25,
            "encoding": "single",
            "include_biases": False,
            "non_linearity": {
                "type": "perfect_diode",
                "quadratic_diode_param": {},
                "exponential_diode_param": {},
                "hard_sigmoid_param": {},
            },
        },
        "solver": {
            "inference_iterations": 4,
            "training_iterations": 4,
            "mode": "asynchronous",
            "overrelaxation_factor": 1.1,
        },
        "mapping": {
            "scale_fractions": [1.0],
            "scale_fraction_pairs": [[1.0, 1.0]],
            "range_placement": "lower",
            "calibration_examples": min(
                1024, int(profile["training_points"] or 1024)
            ),
            "calibration_batch_size": 128,
            "logit_gain_min": 0.001,
            "logit_gain_max": 1000.0,
            "logit_gain_steps": 121,
        },
        "modes": {
            "train": {
                "num_epochs": int(profile["hwa_epochs"]),
                "learning_rates": list(HWA_LEARNING_RATES),
                "temperature": 1.0,
                "log_every": 1,
                "max_batches": profile["maximum_batches_per_epoch"],
                "max_validation_batches": None,
                "minimum_relative_kl_improvement": 0.0,
                "weight_modifier": {"type": "none", "parameters": {}},
                "update_backend": {"type": "ideal", "parameters": {}},
                "selection_evaluation": "clean",
                "selection_metric": "student_accuracy",
                "selection_noise_repeats": 1,
                "selection_weight_modifier": {"type": "none", "parameters": {}},
            }
        },
    }


def _resolved_contract(profile_name: str) -> dict[str, Any]:
    profile = dict(PROFILE_SETTINGS[profile_name])
    return {
        "schema": EXPERIMENT_ID,
        "schema_version": 1,
        "evidence_tier": "exploratory_noncanonical",
        "profile": profile_name,
        "profile_settings": profile,
        "teacher": {
            "path": "data/mnist_relu_teacher_fixed_init_20260816.pt",
            "sha256": EXPECTED_TEACHER_SHA256,
        },
        "topology": {
            "logical_teacher_shapes": [[784, 50], [50, 10]],
            "physical_drn_shapes": [[1568, 100], [100, 20]],
            "layouts": list(ENDPOINT_LAYOUTS),
            "full_positive_conductances_enter_kcl": True,
        },
        "endpoint_model": {
            "regimes": list(ENDPOINT_REGIMES),
            "embedding": "G=full_tile_RESET+p*(full_tile_SET-full_tile_RESET)",
            "target_candidate_seed_start": 91001,
            "hwa_bank_candidate_seed_start": 92001,
            "invalid_whole_population_policy": (
                "reject_candidate_seed_for_both_regimes;no_cell_clipping_or_redraw"
            ),
        },
        "direct_initialization": {
            "logical_master": "teacher_weight/layer_absmax",
            "signed_bounds": [-1.0, 1.0],
            "inactive_cells": "p=0=full_tile_RESET",
            "unit_magnitude_cells": "p=1=full_tile_SET",
            "programming_simulated": False,
        },
        "hwa": {
            "type": "direct_endpoint_population_noise_injection",
            "epochs": profile["hwa_epochs"],
            "bank_size": profile["hwa_bank_size"],
            "assignment_rotations_per_population": profile[
                "hwa_assignment_rotations"
            ],
            "effective_assignment_bank_size": (
                int(profile["hwa_bank_size"])
                * int(profile["hwa_assignment_rotations"])
            ),
            "population_schedule": (
                "round_robin_joint_endpoint_pair_identity_rotations_per_minibatch"
            ),
            "same_draw_forward_and_backward": True,
            "optimizer": "digital_Adam_on_clean_signed_master",
            "learning_rates": list(HWA_LEARNING_RATES),
            "gradient": "exact_chain_rule_through_per_cell_endpoint_window",
            "noise_strength": 1.0,
            "programming_noise_beyond_endpoint_variation": False,
        },
        "deployment": {
            "target_population": "one_fixed_evaluation_endpoint_assignment",
            "target_is_sealed_across_exploratory_method_development": False,
            "mapping": "continuous_direct_endpoint_affine",
            "programming_error": False,
            "inference_read_noise": False,
            "fixed_logit_gain": FIXED_LOGIT_GAIN,
            "simulator_conductance_ceiling_only_not_SET_clip": CONDUCTANCE_CEILING,
        },
        "recovery": {
            "claim_label": "hardware-in-loop pulse-mediated recovery",
            "start_state": "exact_selected_HWA_target_deployment",
            "weight_authority": "persistent_normalized_cell_state_only",
            "optimizer": "digital_BPTT_Adam_to_Bernoulli_one_pulse_commands",
            "learning_rate_progress": HFO2_RECOVERY_LEARNING_RATE_PROGRESS,
            "learning_rate_source": (
                "OM_3e-5_scaled_by_HfO2_to_OM_nominal_progress_step_ratio"
            ),
            "nominal_delta_progress": HFO2_NOMINAL_DW_MIN_PROGRESS,
            "pulse_cap_per_cell": profile["recovery_pulse_cap"],
            "epochs": profile["recovery_epochs"],
            "verify_reads_during_updates": 0,
            "pulse_dynamics": {
                "nominal_dw_min_raw_a": HFO2_NOMINAL_DW_MIN_RAW_A,
                "dw_min_dtod_log_std": HFO2_DW_MIN_DTOD_LOG_STD,
                "up_down_dtod": HFO2_UP_DOWN_DTOD,
                "cycle_noise_std": HFO2_CYCLE_NOISE_STD,
                "write_noise_std": HFO2_WRITE_NOISE_STD,
                "endpoint_bounds_replaced_by_figure6_model": True,
            },
            "qualification": (
                "digital_gradient_and_optimizer_state;not_autonomous_on_chip_learning"
            ),
        },
        "selection": {
            "hwa": "best_target_validation_accuracy_then_KL_among_trained_epochs",
            "recovery": "best_target_validation_accuracy_then_KL_among_recovery_epochs",
            "test_used_for_selection": False,
            "qualification": (
                "within_run_only;development_pilot_not_sealed_confirmatory_evidence"
            ),
        },
        "matching": {
            "same_teacher_topology_solver_data_and_gain": True,
            "same_finite_endpoint_marginals_across_pairing_regimes": True,
            "same_HWA_bank_seed_schedule": True,
            "same_recovery_pulse_parameter_noise_and_selection_seeds": True,
        },
    }


def _verify_aihwkit_pulse_constants(executable: Path) -> Mapping[str, Any]:
    source = executable.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Expected an AIHWKit Python executable: {source}.")
    code = """
import json
import aihwkit
from aihwkit.simulator.presets.devices import ReRamArrayHfO2PresetDevice
d=ReRamArrayHfO2PresetDevice()
print(json.dumps({
  'version': aihwkit.__version__,
  'dw_min': d.dw_min,
  'dw_min_dtod': d.dw_min_dtod,
  'dw_min_dtod_log_normal': d.dw_min_dtod_log_normal,
  'up_down_dtod': d.up_down_dtod,
  'dw_min_std': d.dw_min_std,
  'write_noise_std': d.write_noise_std,
  'mult_noise': d.mult_noise,
}))
"""
    completed = subprocess.run(
        (str(source), "-c", code),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"AIHWKit pulse-constant probe failed: {completed.stderr}")
    observed = json.loads(completed.stdout)
    expected = {
        "version": "1.1.0",
        "dw_min": HFO2_NOMINAL_DW_MIN_RAW_A,
        "dw_min_dtod": HFO2_DW_MIN_DTOD_LOG_STD,
        "dw_min_dtod_log_normal": True,
        "up_down_dtod": HFO2_UP_DOWN_DTOD,
        "dw_min_std": HFO2_CYCLE_NOISE_STD,
        "write_noise_std": HFO2_WRITE_NOISE_STD,
        "mult_noise": False,
    }
    if observed != expected:
        raise RuntimeError(
            f"Pinned HfO2 pulse constants changed: expected={expected}, observed={observed}."
        )
    return {"executable": str(source), **observed}


def _apply_full_g(stack: Any, values: Sequence[torch.Tensor]) -> None:
    apply_full_conductance_targets(
        stack.bundle.catalog,
        tuple(values),
        conductance_min=0.0,
        conductance_max=CONDUCTANCE_CEILING,
    )


def _initial_masters(teacher: Any, device: torch.device) -> tuple[torch.Tensor, ...]:
    result = []
    for value in teacher.parameters():
        source = value.detach().to(device=device, dtype=torch.float32)
        maximum = source.abs().max()
        if not bool(torch.isfinite(maximum)) or float(maximum.item()) <= 0.0:
            raise RuntimeError("Teacher layer has an invalid absolute maximum.")
        result.append(source / maximum)
    if len(result) != 2:
        raise RuntimeError("Expected exactly two frozen ReLU teacher layers.")
    return tuple(result)


def _snapshot(values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().cpu().to(torch.float32).clone() for value in values)


def _evaluate_full_g(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    full_g: Sequence[torch.Tensor],
    sample_limit: int | None,
) -> Mapping[str, Any]:
    _apply_full_g(stack, full_g)
    return _evaluate(stack, teacher, loader, sample_limit=sample_limit)


def _one_forward_gradient(
    *,
    stack: Any,
    teacher: Any,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    full_g: Sequence[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, float | int]]:
    _apply_full_g(stack, full_g)
    with torch.no_grad():
        teacher_logits = teacher.logits(inputs)
    stack.network.set_input(inputs, reset=True)
    stack.minimizer.compute_equilibrium()
    stack.cost.set_teacher(teacher_logits, labels)
    with torch.no_grad():
        per_example = stack.cost.eval()
        prediction = stack.cost.student_logits().argmax(dim=1)
        teacher_prediction = teacher_logits.argmax(dim=1)
        metrics: Mapping[str, float | int] = {
            "examples": int(labels.numel()),
            "kl_sum": float(per_example.to(torch.float64).sum().item()),
            "correct": int(prediction.eq(labels).sum().item()),
            "teacher_agreement": int(
                prediction.eq(teacher_prediction).sum().item()
            ),
        }
    gradients = tuple(stack.differentiator.compute_gradient())
    if len(gradients) != 2 or any(
        not bool(torch.isfinite(value).all()) for value in gradients
    ):
        raise FloatingPointError("Physical DRN gradient is invalid.")
    return gradients, metrics


def _metric_key(metrics: Mapping[str, Any], epoch: int) -> tuple[float, float, int]:
    return (
        float(metrics["student_accuracy"]),
        -float(metrics["kl_teacher_student"]),
        -int(epoch),
    )


def _sample_joint_valid_fields(
    *,
    count: int,
    candidate_start: int,
    devices: int,
    shapes: Sequence[Sequence[int]],
    device: torch.device,
) -> tuple[Mapping[str, tuple[EndpointField, ...]], Mapping[str, Any]]:
    fields: dict[str, list[EndpointField]] = {
        INDEPENDENT_ENDPOINTS: [],
        RANK_MATCHED_ENDPOINTS: [],
    }
    accepted = []
    rejected = []
    candidate = int(candidate_start)
    while len(accepted) < count:
        populations = {}
        errors = {}
        for regime in ENDPOINT_REGIMES:
            try:
                populations[regime] = sample_hfo2_figure6_endpoint_population(
                    devices=devices,
                    assignment_seed=candidate,
                    regime=regime,
                )
            except ValueError as error:
                errors[regime] = str(error)
        if errors:
            rejected.append({"candidate_seed": candidate, "errors": errors})
        else:
            accepted.append(candidate)
            for regime, population in populations.items():
                field = endpoint_field_from_population(
                    population,
                    shapes=shapes,
                    device=device,
                )
                observed_maximum = max(
                    float(value.max().item()) for value in field.set
                )
                if observed_maximum > CONDUCTANCE_CEILING:
                    raise RuntimeError(
                        "Unclipped SET draw exceeds the declared simulator ceiling; "
                        f"seed={candidate}, maximum={observed_maximum}."
                    )
                fields[regime].append(field)
        candidate += 1
        if candidate - candidate_start > 10_000:
            raise RuntimeError("Could not obtain a valid whole-array endpoint draw.")
    return (
        {name: tuple(values) for name, values in fields.items()},
        {
            "candidate_start": int(candidate_start),
            "accepted_seeds": accepted,
            "rejected_candidates": rejected,
            "whole_population_rejections": len(rejected),
            "cellwise_redraws": 0,
            "cellwise_SET_clips": 0,
        },
    )


def _fresh_loaders(spec: Any) -> Any:
    return build_mnist_loaders(
        spec.data,
        data_seed=spec.runtime.data_seed,
        calibration_examples=spec.mapping.calibration_examples,
        calibration_batch_size=spec.mapping.calibration_batch_size,
    )


def _augment_assignment_bank(
    base_fields: Sequence[EndpointField], *, rotations: int
) -> tuple[EndpointField, ...]:
    if isinstance(rotations, bool) or not isinstance(rotations, int) or rotations < 1:
        raise ValueError("Expected at least one endpoint-assignment rotation.")
    result = []
    for rotation_index in range(rotations):
        for base_index, field in enumerate(base_fields):
            offset = (
                rotation_index * 104_729 + base_index * 65_537
            ) % field.devices
            result.append(rotate_endpoint_field(field, offset))
    return tuple(result)


def _train_hwa(
    *,
    regime: str,
    initial_masters: Sequence[torch.Tensor],
    bank: Sequence[EndpointField],
    target_field: EndpointField,
    stack: Any,
    teacher: Any,
    loaders: Any,
    profile: Mapping[str, Any],
    store: RunStore,
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, Any]]:
    masters = tuple(
        torch.nn.Parameter(value.detach().clone().to(stack.device))
        for value in initial_masters
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [master], "lr": float(rate)}
            for master, rate in zip(masters, HWA_LEARNING_RATES, strict=True)
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    history = []
    selected_key = None
    selected_epoch = None
    selected_masters = None
    global_batch = 0
    epochs = int(profile["hwa_epochs"])
    maximum_batches = profile["maximum_batches_per_epoch"]
    interval = int(profile["progress_log_interval"])
    for epoch in range(1, epochs + 1):
        totals = {"examples": 0, "batches": 0, "kl": 0.0, "correct": 0, "agreement": 0}
        gradient_square = [0.0, 0.0]
        gradient_values = [0, 0]
        bank_counts = [0 for _ in bank]
        for batch_index, (inputs, labels) in enumerate(
            limited(loaders.train, maximum_batches), start=1
        ):
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            bank_index = global_batch % len(bank)
            field = bank[bank_index]
            full_g = map_masters_to_conductance(masters, field)
            physical, metrics = _one_forward_gradient(
                stack=stack,
                teacher=teacher,
                inputs=inputs,
                labels=labels,
                full_g=full_g,
            )
            logical = lift_endpoint_gradients(masters, physical, field)
            optimizer.zero_grad(set_to_none=True)
            for layer, (master, gradient) in enumerate(
                zip(masters, logical, strict=True)
            ):
                master.grad = gradient
                gradient_square[layer] += float(
                    gradient.detach().to(torch.float64).square().sum().item()
                )
                gradient_values[layer] += gradient.numel()
            optimizer.step()
            with torch.no_grad():
                for master in masters:
                    master.clamp_(-1.0, 1.0)
            count = int(metrics["examples"])
            totals["examples"] += count
            totals["batches"] += 1
            totals["kl"] += float(metrics["kl_sum"])
            totals["correct"] += int(metrics["correct"])
            totals["agreement"] += int(metrics["teacher_agreement"])
            bank_counts[bank_index] += 1
            global_batch += 1
            if batch_index % interval == 0:
                print(
                    f"HWA {regime} epoch={epoch}/{epochs} batch={batch_index}",
                    flush=True,
                )
        examples = int(totals["examples"])
        if examples < 1:
            raise RuntimeError("Endpoint-noise HWA processed no examples.")
        validation = _evaluate_full_g(
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            full_g=map_masters_to_conductance(masters, target_field),
            sample_limit=profile["validation_sample_limit"],
        )
        record = {
            "mode": "endpoint_noise_hwa",
            "regime": regime,
            "epoch": epoch,
            "train": {
                "examples": examples,
                "batches": int(totals["batches"]),
                "kl_teacher_student": float(totals["kl"]) / examples,
                "student_accuracy": int(totals["correct"]) / examples,
                "teacher_agreement": int(totals["agreement"]) / examples,
                "logical_gradient_rms": [
                    math.sqrt(total / count)
                    for total, count in zip(
                        gradient_square, gradient_values, strict=True
                    )
                ],
                "bank_draw_counts": bank_counts,
            },
            "fixed_target_validation": validation,
        }
        history.append(record)
        store.append_metric(record)
        key = _metric_key(validation, epoch)
        if selected_key is None or key > selected_key:
            selected_key = key
            selected_epoch = epoch
            selected_masters = _snapshot(masters)
        print(
            f"HWA {regime} epoch={epoch}/{epochs} "
            f"target_val={100.0*float(validation['student_accuracy']):.2f}%",
            flush=True,
        )
    if selected_masters is None or selected_epoch is None:
        raise RuntimeError("HWA produced no selectable trained epoch.")
    return selected_masters, {
        "regime": regime,
        "selected_epoch": selected_epoch,
        "selection_used_test": False,
        "history": history,
        "selected_master": selected_masters,
        "optimizer_state_at_final_epoch": optimizer.state_dict(),
    }


def _recover(
    *,
    regime: str,
    hwa_masters: Sequence[torch.Tensor],
    target_field: EndpointField,
    stack: Any,
    teacher: Any,
    loaders: Any,
    profile: Mapping[str, Any],
    store: RunStore,
) -> tuple[HfO2Figure6PulsePlant, PersistentHfO2PulseAdam, Mapping[str, Any]]:
    progress = master_to_progress(
        tuple(value.to(stack.device) for value in hwa_masters), target_field
    )
    plant = HfO2Figure6PulsePlant(
        target_field,
        progress,
        pulse_parameter_seed=93001,
        pulse_noise_seed=93002,
    )
    optimizer = PersistentHfO2PulseAdam(
        plant,
        learning_rate_progress=HFO2_RECOVERY_LEARNING_RATE_PROGRESS,
        pulse_cap=int(profile["recovery_pulse_cap"]),
        pulse_selection_seed=93003,
    )
    initial_state = plant.state_dict()
    history = []
    selected_key = None
    selected_epoch = None
    selected_plant_state = None
    selected_optimizer_state = None
    epochs = int(profile["recovery_epochs"])
    maximum_batches = profile["maximum_batches_per_epoch"]
    interval = int(profile["progress_log_interval"])
    for epoch in range(1, epochs + 1):
        totals = {
            "examples": 0,
            "batches": 0,
            "kl": 0.0,
            "correct": 0,
            "agreement": 0,
            "requested": 0,
            "pulsed": 0,
            "capped": 0,
            "up": 0,
            "down": 0,
            "changed": 0,
            "probability_sum": 0.0,
        }
        for batch_index, (inputs, labels) in enumerate(
            limited(loaders.train, maximum_batches), start=1
        ):
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            physical, metrics = _one_forward_gradient(
                stack=stack,
                teacher=teacher,
                inputs=inputs,
                labels=labels,
                full_g=plant.full_conductance,
            )
            pulse = optimizer.step(physical)
            count = int(metrics["examples"])
            totals["examples"] += count
            totals["batches"] += 1
            totals["kl"] += float(metrics["kl_sum"])
            totals["correct"] += int(metrics["correct"])
            totals["agreement"] += int(metrics["teacher_agreement"])
            totals["requested"] += pulse.requested_cells
            totals["pulsed"] += pulse.pulsed_cells
            totals["capped"] += pulse.capped_cells
            totals["up"] += pulse.upward_pulses
            totals["down"] += pulse.downward_pulses
            totals["changed"] += pulse.effective_state_changes
            totals["probability_sum"] += pulse.mean_probability
            if batch_index % interval == 0:
                print(
                    f"recovery {regime} epoch={epoch}/{epochs} "
                    f"batch={batch_index} pulses={totals['pulsed']}",
                    flush=True,
                )
        examples = int(totals["examples"])
        batches = int(totals["batches"])
        if examples < 1 or batches < 1:
            raise RuntimeError("Persistent HfO2 recovery processed no examples.")
        validation = _evaluate_full_g(
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            full_g=plant.full_conductance,
            sample_limit=profile["validation_sample_limit"],
        )
        record = {
            "mode": "persistent_hfo2_pulse_recovery",
            "regime": regime,
            "epoch": epoch,
            "train": {
                "examples": examples,
                "batches": batches,
                "kl_teacher_student": float(totals["kl"]) / examples,
                "student_accuracy": int(totals["correct"]) / examples,
                "teacher_agreement": int(totals["agreement"]) / examples,
            },
            "pulses": {
                "requested": int(totals["requested"]),
                "applied": int(totals["pulsed"]),
                "capped": int(totals["capped"]),
                "upward": int(totals["up"]),
                "downward": int(totals["down"]),
                "effective_state_changes": int(totals["changed"]),
                "mean_selection_probability": (
                    float(totals["probability_sum"]) / batches
                ),
                "verify_reads": 0,
            },
            "fixed_target_validation": validation,
            "plant": dict(plant.report()),
        }
        history.append(record)
        store.append_metric(record)
        key = _metric_key(validation, epoch)
        if selected_key is None or key > selected_key:
            selected_key = key
            selected_epoch = epoch
            selected_plant_state = plant.state_dict()
            selected_optimizer_state = optimizer.state_dict()
        print(
            f"recovery {regime} epoch={epoch}/{epochs} "
            f"target_val={100.0*float(validation['student_accuracy']):.2f}% "
            f"cumulative_pulses={plant.report()['total_pulses']}",
            flush=True,
        )
    if (
        selected_epoch is None
        or selected_plant_state is None
        or selected_optimizer_state is None
    ):
        raise RuntimeError("Recovery produced no selectable epoch.")
    plant.load_state_dict(selected_plant_state)
    optimizer.load_state_dict(selected_optimizer_state)
    return plant, optimizer, {
        "regime": regime,
        "selected_epoch": selected_epoch,
        "selection_used_test": False,
        "history": history,
        "initial_plant_state": initial_state,
        "selected_plant_state": selected_plant_state,
        "selected_optimizer_state": selected_optimizer_state,
    }


def _without_tensor_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        if key in {
            "selected_master",
            "optimizer_state_at_final_epoch",
            "initial_plant_state",
            "selected_plant_state",
            "selected_optimizer_state",
        }:
            continue
        result[key] = item
    return result


def run(args: argparse.Namespace) -> Path:
    if args.profile not in PROFILE_SETTINGS:
        raise ValueError(f"Expected profile in {tuple(PROFILE_SETTINGS)!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("The Figure-6 DRN experiment requires CUDA.")
    profile = dict(PROFILE_SETTINGS[args.profile])
    contract = _resolved_contract(args.profile)
    aihwkit = _verify_aihwkit_pulse_constants(args.aihwkit_python)
    contract["aihwkit_pulse_source"] = dict(aihwkit)
    teacher_path = args.teacher_weights.expanduser().resolve()
    if sha256_file(teacher_path) != EXPECTED_TEACHER_SHA256:
        raise ValueError("Frozen ReLU teacher SHA-256 mismatch.")
    store = RunStore.create(
        output_root=args.output_dir,
        experiment_id=EXPERIMENT_ID,
        resolved_config=contract,
        command=sys.argv,
        repo_root=_ROOT,
        input_artifacts=(
            {
                "role": "frozen_relu_teacher",
                "path": str(teacher_path),
                "sha256": sha256_file(teacher_path),
            },
            {
                "role": "pinned_aihwkit_python",
                "path": str(args.aihwkit_python.expanduser().resolve()),
                "version": aihwkit["version"],
            },
        ),
        resume_capability="unsupported",
    )
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        device = torch.device("cuda")
        student = resolve_student_spec(
            parse_student_config(_student_payload(profile)), RunMode.TRAIN
        )
        loaders = _fresh_loaders(student)
        teacher, teacher_metadata = _load_teacher(
            teacher_path, device=device, spec=student
        )
        stack = build_student_stack(student, enable_measured=False)
        stack.cost.gain = FIXED_LOGIT_GAIN
        masters = _initial_masters(teacher, device)
        shapes = tuple(
            tuple(int(item) for item in binding.state.shape)
            for binding in stack.bundle.catalog.trainable
        )
        devices = sum(math.prod(shape) for shape in shapes)
        if shapes != ((1568, 100), (100, 20)) or devices != 158_800:
            raise RuntimeError("Frozen DRN physical topology changed.")

        target_fields, target_sampling = _sample_joint_valid_fields(
            count=1,
            candidate_start=91001,
            devices=devices,
            shapes=shapes,
            device=device,
        )
        bank_fields, bank_sampling = _sample_joint_valid_fields(
            count=int(profile["hwa_bank_size"]),
            candidate_start=92001,
            devices=devices,
            shapes=shapes,
            device=device,
        )
        augmented_banks = {
            regime: _augment_assignment_bank(
                bank_fields[regime],
                rotations=int(profile["hwa_assignment_rotations"]),
            )
            for regime in ENDPOINT_REGIMES
        }
        print(
            "endpoint populations ready "
            f"target={target_sampling['accepted_seeds']} "
            f"hwa_bank={bank_sampling['accepted_seeds']} "
            f"effective_assignments={len(augmented_banks[INDEPENDENT_ENDPOINTS])}",
            flush=True,
        )

        ideal = map_global_positive_g(masters).full_conductance
        ideal_test = _evaluate_full_g(
            stack=stack,
            teacher=teacher,
            loader=loaders.test,
            full_g=ideal,
            sample_limit=profile["test_sample_limit"],
        )
        regime_reports = {}
        checkpoint_paths = []
        for regime in ENDPOINT_REGIMES:
            target_field = target_fields[regime][0]
            direct_test = _evaluate_full_g(
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                full_g=map_masters_to_conductance(masters, target_field),
                sample_limit=profile["test_sample_limit"],
            )
            print(
                f"direct {regime} test="
                f"{100.0*float(direct_test['student_accuracy']):.2f}%",
                flush=True,
            )

            hwa_loaders = _fresh_loaders(student)
            hwa_masters, hwa = _train_hwa(
                regime=regime,
                initial_masters=masters,
                bank=augmented_banks[regime],
                target_field=target_field,
                stack=stack,
                teacher=teacher,
                loaders=hwa_loaders,
                profile=profile,
                store=store,
            )
            hwa_device = tuple(value.to(device) for value in hwa_masters)
            hwa_test = _evaluate_full_g(
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                full_g=map_masters_to_conductance(hwa_device, target_field),
                sample_limit=profile["test_sample_limit"],
            )
            print(
                f"HWA-selected {regime} test="
                f"{100.0*float(hwa_test['student_accuracy']):.2f}%",
                flush=True,
            )

            recovery_loaders = _fresh_loaders(student)
            plant, pulse_optimizer, recovery = _recover(
                regime=regime,
                hwa_masters=hwa_masters,
                target_field=target_field,
                stack=stack,
                teacher=teacher,
                loaders=recovery_loaders,
                profile=profile,
                store=store,
            )
            recovered_test = _evaluate_full_g(
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                full_g=plant.full_conductance,
                sample_limit=profile["test_sample_limit"],
            )
            print(
                f"recovered {regime} test="
                f"{100.0*float(recovered_test['student_accuracy']):.2f}%",
                flush=True,
            )

            checkpoint_payload = {
                "schema": "ebl.hfo2_figure6_drn_arm_checkpoint",
                "schema_version": 1,
                "regime": regime,
                "profile": args.profile,
                "target_endpoint": {
                    "reset": tuple(value.detach().cpu() for value in target_field.reset),
                    "set": tuple(value.detach().cpu() for value in target_field.set),
                    "report": target_field.report(),
                },
                "HWA": hwa,
                "recovery": recovery,
                "selected_recovery_plant": plant.state_dict(),
                "selected_recovery_optimizer": pulse_optimizer.state_dict(),
                "test": {
                    "direct_initialization": dict(direct_test),
                    "selected_HWA": dict(hwa_test),
                    "selected_recovery": dict(recovered_test),
                },
            }
            checkpoint = atomic_torch_save(
                checkpoint_payload,
                store.run_dir / "checkpoints" / f"{regime}.pt",
            )
            checkpoint_paths.append(checkpoint)
            regime_reports[regime] = {
                "target_endpoint": target_field.report(),
                "direct_initialization_test": dict(direct_test),
                "HWA": _without_tensor_payload(hwa),
                "selected_HWA_test": dict(hwa_test),
                "recovery": _without_tensor_payload(recovery),
                "selected_recovery_test": dict(recovered_test),
                "selected_recovery_plant": dict(plant.report()),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            }

        summary = {
            "schema": EXPERIMENT_ID,
            "schema_version": 1,
            "status": "complete",
            "evidence_tier": "exploratory_noncanonical",
            "profile": args.profile,
            "teacher": {
                "path": str(teacher_path),
                "sha256": sha256_file(teacher_path),
                "metadata": teacher_metadata,
            },
            "ideal_global_test": dict(ideal_test),
            "endpoint_sampling": {
                "target": target_sampling,
                "HWA_bank": bank_sampling,
            },
            "regimes": regime_reports,
            "claims": {
                "direct_initialization_is_exact_endpoint_affine_mapping": True,
                "HWA_noise_is_endpoint_population_variation_only": True,
                "recovery_starts_from_exact_saved_deployed_state": True,
                "recovery_weight_mutations_are_pulses_only": True,
                "recovery_is_autonomous_on_chip_learning": False,
                "physical_power_claim_supported": False,
            },
        }
        summary_path = store.run_dir / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        result = store.complete(
            metrics=summary,
            artifacts=(
                store.artifact_record(summary_path, kind="scientific_summary"),
                *(
                    store.artifact_record(path, kind="exact_arm_checkpoint")
                    for path in checkpoint_paths
                ),
            ),
        )
        print(f"complete result={result}", flush=True)
        return result
    except BaseException as error:
        store.fail(error)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=tuple(PROFILE_SETTINGS), default="pilot"
    )
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=_ROOT / "data" / "mnist_relu_teacher_fixed_init_20260816.pt",
    )
    parser.add_argument(
        "--aihwkit-python",
        type=Path,
        default=Path("/home/filip/miniconda3/envs/aihwkit/bin/python"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "simulation_results" / "hfo2_figure6_drn",
    )
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
