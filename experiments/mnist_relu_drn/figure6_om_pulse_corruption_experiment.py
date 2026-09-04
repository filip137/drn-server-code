"""Run the Figure-6 endpoint DRN with IBM-OM pulses and paired corruption.

For each Figure-6 endpoint pairing and each literal OM assignment view this
exploratory experiment evaluates:

* direct logical initialization followed by apparent-feedback P&V;
* endpoint-distribution HWA followed by the same P&V;
* open-loop stochastic OM-pulse recovery from the direct P&V endpoint; and
* open-loop stochastic OM-pulse recovery from the HWA P&V endpoint.

The endpoint distributions are unchanged between arms.  The published and
counterfactual-repaired OM views originate from one sampled assignment; repair
replaces only the identities that AIHWKit marked corrupt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.config import parse_student_config, resolve_student_spec
from experiments.mnist_relu_drn.figure6_om_pulse import (
    CORRUPTION_POLICIES,
    OM_CYCLE_NOISE_STD,
    OM_NOMINAL_DW_MIN_PROGRESS,
    OM_NOMINAL_DW_MIN_RAW_A,
    OM_PROGRAM_VERIFY_MAXIMUM_PULSES,
    OM_PROGRAM_VERIFY_TOLERANCE_PROGRESS,
    OM_WRITE_NOISE_STD,
    Figure6OmPulsePlant,
    PersistentFigure6OmPulseAdam,
    corruption_constrained_progress,
    program_masters_with_verify,
    validate_paired_om_populations,
)
from experiments.mnist_relu_drn.hfo2_figure6_drn import (
    EndpointField,
    flatten_physical,
    map_masters_to_conductance,
    master_to_progress,
    progress_to_conductance,
)
from experiments.mnist_relu_drn.hfo2_figure6_drn_experiment import (
    EXPECTED_TEACHER_SHA256,
    FIXED_LOGIT_GAIN,
    OM_RECOVERY_LEARNING_RATE_PROGRESS,
    PROFILE_SETTINGS,
    _augment_assignment_bank,
    _evaluate_full_g,
    _fresh_loaders,
    _initial_masters,
    _one_forward_gradient,
    _sample_joint_valid_fields,
    _student_payload,
    _train_hwa,
    _without_tensor_payload,
)
from experiments.mnist_relu_drn.hfo2_figure6_endpoint_regimes import (
    ENDPOINT_REGIMES,
    INDEPENDENT_ENDPOINTS,
    RESET_STATE_FLOOR,
)
from experiments.mnist_relu_drn.hfo2_figure6_program_verify_experiment import (
    TARGET_KINDS,
    _program_verify_tensor_payload,
    _recovery_summary,
    _residual_summary,
    _tensor_sha256,
    _tensor_summary,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import limited
from experiments.schema import RunMode
from training.checkpoint import atomic_torch_save
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    load_om_array_population,
    sample_om_array_population_external,
)
from training.ibm_reram_program_verify import PUBLISHED_CORRUPT_PROBABILITY


_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "mnist_figure6_endpoints_om_pulse_corruption_ladder.v1"
OM_ASSIGNMENT_SEED = 97001
PROGRAM_PULSE_NOISE_SEED = 97002
RECOVERY_PULSE_NOISE_SEED = 97003
RECOVERY_PULSE_SELECTION_SEED = 97004


def _verify_om_pulse_constants(executable: Path) -> Mapping[str, Any]:
    source = executable.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Expected an AIHWKit Python executable: {source}.")
    code = """
import json
import aihwkit
from aihwkit.simulator.presets.devices import ReRamArrayOMPresetDevice
d=ReRamArrayOMPresetDevice()
print(json.dumps({
  'version': aihwkit.__version__,
  'dw_min': d.dw_min,
  'dw_min_dtod': d.dw_min_dtod,
  'dw_min_dtod_log_normal': d.dw_min_dtod_log_normal,
  'up_down_dtod': d.up_down_dtod,
  'dw_min_std': d.dw_min_std,
  'write_noise_std': d.write_noise_std,
  'mult_noise': d.mult_noise,
  'w_min': d.w_min,
  'w_max': d.w_max,
  'corrupt_devices_range': d.corrupt_devices_range,
}))
"""
    completed = subprocess.run(
        (str(source), "-c", code),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"AIHWKit OM pulse-constant probe failed: {completed.stderr}")
    observed = json.loads(completed.stdout)
    expected = {
        "version": "1.1.0",
        "dw_min": OM_NOMINAL_DW_MIN_RAW_A,
        "dw_min_dtod": 0.7829,
        "dw_min_dtod_log_normal": True,
        "up_down_dtod": 0.01,
        "dw_min_std": OM_CYCLE_NOISE_STD,
        "write_noise_std": OM_WRITE_NOISE_STD,
        "mult_noise": False,
        "w_min": -1.0,
        "w_max": 1.0,
        "corrupt_devices_range": 0.01,
    }
    if observed != expected:
        raise RuntimeError(
            f"Pinned OM pulse constants changed: expected={expected}, observed={observed}."
        )
    return {
        "executable": str(source),
        **observed,
        "published_corrupt_devices_prob": PUBLISHED_CORRUPT_PROBABILITY[
            "reram_array_om"
        ],
    }


def _load_om_probe_input(source_dir: Path) -> Mapping[str, Any]:
    path = source_dir.expanduser().resolve() / "om_probe.json"
    if not path.is_file():
        raise FileNotFoundError(f"Expected a pre-sampled OM probe at {path}.")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "version": "1.1.0",
        "dw_min": OM_NOMINAL_DW_MIN_RAW_A,
        "dw_min_dtod": 0.7829,
        "dw_min_dtod_log_normal": True,
        "up_down_dtod": 0.01,
        "dw_min_std": OM_CYCLE_NOISE_STD,
        "write_noise_std": OM_WRITE_NOISE_STD,
        "mult_noise": False,
        "w_min": -1.0,
        "w_max": 1.0,
        "corrupt_devices_range": 0.01,
        "published_corrupt_devices_prob": PUBLISHED_CORRUPT_PROBABILITY[
            "reram_array_om"
        ],
    }
    if not isinstance(value, Mapping) or any(
        value.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError("Pre-sampled OM constant probe does not match the pinned model.")
    return {
        **dict(value),
        "verification": "hash_tracked_input_from_pinned_local_AIHWKit",
        "probe_path": str(path),
        "probe_sha256": sha256_file(path),
    }


def _resolved_contract(
    profile_name: str,
    *,
    om_probe: Mapping[str, Any],
    population_source_dir: Path | None,
) -> Mapping[str, Any]:
    profile = dict(PROFILE_SETTINGS[profile_name])
    return {
        "schema": EXPERIMENT_ID,
        "schema_version": 1,
        "evidence_tier": "exploratory_noncanonical",
        "profile": profile_name,
        "profile_settings": profile,
        "endpoint_model": {
            "source": "Figure-6-informed synthetic RESET and SET distributions",
            "regimes": list(ENDPOINT_REGIMES),
            "embedding": "G=full_tile_RESET+p*(full_tile_SET-full_tile_RESET)",
            "RESET_lower_clip": RESET_STATE_FLOOR,
            "RESET_upper_clip": None,
            "SET_clipping": None,
            "changed_for_OM_pulse_follow_up": False,
        },
        "OM_population": {
            "assignment_seed": OM_ASSIGNMENT_SEED,
            "policies": list(CORRUPTION_POLICIES),
            "published_corrupt_probability": PUBLISHED_CORRUPT_PROBABILITY[
                "reram_array_om"
            ],
            "pairing": "one_literal_draw;repair_only_same_corrupt_coordinates",
            "healthy_identity_redraw_between_policies": False,
            "sampling": (
                "pre_sampled_hash_checked_pinned_AIHWKit_1.1.0_artifacts"
                if population_source_dir is not None
                else "pinned_external_AIHWKit_1.1.0_array_tiles"
            ),
            "input_directory": (
                str(population_source_dir.expanduser().resolve())
                if population_source_dir is not None
                else None
            ),
        },
        "pulse_dynamics": {
            "source": dict(om_probe),
            "per_cell_dwmin_up_down": "literal_sampled_OM_hidden_parameters",
            "soft_bounds_coordinate": "raw_a=2*p-1_with_bounds_[-1,1]",
            "synthetic_endpoint_role": "replaces_OM_bound_locations_only",
            "OM_cycle_noise_retained": True,
            "OM_write_noise_retained": True,
            "OM_reference_used": False,
            "native_OM_bound_dtod_used_as_endpoint_distribution": False,
            "published_corrupt_state": (
                "native_AIHWKit_singleton_raw_a_with_zero_SET_and_RESET_steps"
            ),
        },
        "targets": {
            "direct": "teacher_weight/layer_absmax",
            "endpoint_noise_HWA": "selected_in_run_HWA_logical_master",
            "HWA_training_in_this_follow_up": True,
            "HWA_noise": "Figure-6_RESET_and_SET_endpoint_population",
            "HWA_includes_OM_programming_noise": False,
            "HWA_includes_corrupt_devices": False,
            "same_logical_targets_across_corruption_policies": True,
        },
        "program_and_verify": {
            "initial_healthy_persistent_state": "p=0=full_tile_RESET",
            "initial_published_corrupt_state": "immutable_native_stuck_progress",
            "initial_apparent_verify": "one_noisy_OM_write_observation",
            "exact_healthy_RESET_targets": "preaccepted_without_programming_pulses",
            "exact_corrupt_RESET_targets": "not_preaccepted",
            "controller": "one_stochastic_OM_pulse_then_apparent_verify",
            "controller_access": "target_apparent_progress_and_history_only",
            "tolerance_progress": OM_PROGRAM_VERIFY_TOLERANCE_PROGRESS,
            "tolerance_definition": "half_nominal_OM_central_progress_step",
            "maximum_target_pulses_per_cell": OM_PROGRAM_VERIFY_MAXIMUM_PULSES,
            "pulse_noise_seed": PROGRAM_PULSE_NOISE_SEED,
            "persistent_endpoint_used_for_continuation": True,
        },
        "network_evaluation": {
            "primary": "held_apparent_progress_projected_to_[0,1]",
            "secondary": "persistent_progress",
            "literal_apparent_state_saved": True,
            "projection_reason": "positive_passive_DRN_conductance",
        },
        "recovery": {
            "start_state": "bitwise_exact_named_persistent_PV_endpoint",
            "P&V_pulse_accounting_carried_into_recovery_cap": False,
            "pulse_noise_seed": RECOVERY_PULSE_NOISE_SEED,
            "pulse_selection_seed": RECOVERY_PULSE_SELECTION_SEED,
            "optimizer": "digital_BPTT_Adam_to_Bernoulli_one_OM_pulse_commands",
            "learning_rate_progress": OM_RECOVERY_LEARNING_RATE_PROGRESS,
            "nominal_delta_progress": OM_NOMINAL_DW_MIN_PROGRESS,
            "pulse_cap_per_cell": profile["recovery_pulse_cap"],
            "epochs": profile["recovery_epochs"],
            "verify_reads_during_updates": 0,
            "autonomous_on_chip_learning": False,
        },
        "matching": {
            "same_teacher_topology_solver_data_gain_endpoint_and_targets": True,
            "same_healthy_OM_identities_between_corruption_policies": True,
            "same_programming_and_recovery_random_seeds": True,
            "same_minibatch_order_and_recovery_budget": True,
        },
    }


def _fraction(mask: torch.Tensor, within: torch.Tensor | None = None) -> float | None:
    selected = mask if within is None else mask & within
    count = int(mask.sum().item())
    return float(selected.sum().item()) / count if count else None


def _program_verify_report(
    *,
    target_progress: Sequence[torch.Tensor],
    plant: Figure6OmPulsePlant,
    result: Any,
) -> Mapping[str, Any]:
    target = flatten_physical(tuple(target_progress))
    persistent = flatten_physical(plant.progress)
    apparent = flatten_physical(plant.apparent_progress_unprojected)
    projected = apparent.clamp(0.0, 1.0)
    tolerance = OM_PROGRAM_VERIFY_TOLERANCE_PROGRESS
    corrupt = plant.corrupt
    healthy = ~corrupt
    reset_target = target == 0.0
    nonreset_target = ~reset_target
    preaccepted = healthy & reset_target
    apparent_within = (apparent - target).abs() <= tolerance
    persistent_within = (persistent - target).abs() <= tolerance
    if not torch.equal(result.apparent_endpoint, apparent):
        raise RuntimeError("OM P&V apparent endpoint reporting mismatch.")
    if not torch.equal(result.set_count + result.reset_count, result.total_pulses):
        raise RuntimeError("OM P&V directional counts do not sum to total pulses.")
    if not bool(torch.all(result.accepted | result.nonfinite | result.budget_exhausted)):
        raise RuntimeError("OM P&V left a cell without a terminal outcome.")
    if bool(torch.any(result.total_pulses[preaccepted] != 0)):
        raise RuntimeError("A healthy exact-RESET target received a programming pulse.")
    return {
        "cells": plant.size,
        "healthy_cells": int(healthy.sum().item()),
        "corrupt_cells": int(corrupt.sum().item()),
        "target_progress": _tensor_summary(target),
        "target_progress_sha256": _tensor_sha256(target),
        "persistent_progress": _tensor_summary(persistent),
        "persistent_progress_sha256": _tensor_sha256(persistent),
        "literal_apparent_progress": _tensor_summary(apparent),
        "literal_apparent_progress_sha256": _tensor_sha256(apparent),
        "projected_apparent_progress": _tensor_summary(projected),
        "persistent_minus_target": _residual_summary(persistent - target),
        "literal_apparent_minus_target": _residual_summary(apparent - target),
        "controller_completed": int(result.accepted.sum().item()),
        "controller_completion_fraction": float(result.accepted.float().mean().item()),
        "healthy_completion_fraction": _fraction(healthy, result.accepted),
        "corrupt_completion_fraction": _fraction(corrupt, result.accepted),
        "healthy_persistent_within_tolerance_fraction": _fraction(
            healthy, persistent_within
        ),
        "corrupt_persistent_within_tolerance_fraction": _fraction(
            corrupt, persistent_within
        ),
        "nonRESET_apparent_acceptance_fraction": _fraction(
            nonreset_target, result.accepted
        ),
        "healthy_exact_RESET_targets_preaccepted": int(preaccepted.sum().item()),
        "corrupt_exact_RESET_targets_not_preaccepted": int(
            (corrupt & reset_target).sum().item()
        ),
        "budget_exhausted": int(result.budget_exhausted.sum().item()),
        "healthy_budget_exhausted": int((healthy & result.budget_exhausted).sum().item()),
        "corrupt_budget_exhausted": int((corrupt & result.budget_exhausted).sum().item()),
        "nonfinite": int(result.nonfinite.sum().item()),
        "all_targets_literal_apparent_within_tolerance": int(apparent_within.sum().item()),
        "all_targets_persistent_within_tolerance": int(persistent_within.sum().item()),
        "apparent_below_RESET": int((apparent < 0.0).sum().item()),
        "apparent_above_SET": int((apparent > 1.0).sum().item()),
        "pulses": {
            "total": int(result.total_pulses.sum().item()),
            "healthy": int(result.total_pulses[healthy].sum().item()),
            "corrupt_attempts": int(result.total_pulses[corrupt].sum().item()),
            "SET": int(result.set_count.sum().item()),
            "RESET": int(result.reset_count.sum().item()),
            "pulsed_cells": int((result.total_pulses > 0).sum().item()),
            "maximum_per_cell": int(result.total_pulses.max().item()),
            "reversals": int(result.reversals.sum().item()),
            "verifies_including_initial": int(result.verify_count.sum().item()),
        },
        "controller": {
            "kind": "one_pulse_apparent_feedback",
            "tolerance_progress": tolerance,
            "maximum_pulses_per_cell": OM_PROGRAM_VERIFY_MAXIMUM_PULSES,
            "initial_reset_verify_is_noisy": True,
            "exact_healthy_RESET_targets_are_preaccepted": True,
            "corrupt_mask_exposed_to_controller": False,
        },
    }


def _evaluate_plant(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    plant: Figure6OmPulsePlant,
    sample_limit: int | None,
) -> Mapping[str, Any]:
    apparent = _evaluate_full_g(
        stack=stack,
        teacher=teacher,
        loader=loader,
        full_g=plant.apparent_full_conductance,
        sample_limit=sample_limit,
    )
    persistent = _evaluate_full_g(
        stack=stack,
        teacher=teacher,
        loader=loader,
        full_g=plant.full_conductance,
        sample_limit=sample_limit,
    )
    literal = flatten_physical(plant.apparent_progress_unprojected)
    return {
        "primary_state": "held_apparent_progress_projected_to_[0,1]",
        "held_apparent_projected": dict(apparent),
        "persistent": dict(persistent),
        "apparent_projection": {
            "below_RESET": int((literal < 0.0).sum().item()),
            "above_SET": int((literal > 1.0).sum().item()),
            "projected_cells": int(((literal < 0.0) | (literal > 1.0)).sum().item()),
        },
    }


def _metric_key(evaluation: Mapping[str, Any], epoch: int) -> tuple[float, float, int]:
    primary = evaluation["held_apparent_projected"]
    return (
        float(primary["student_accuracy"]),
        -float(primary["kl_teacher_student"]),
        -int(epoch),
    )


def _recover_from_programmed(
    *,
    corruption_policy: str,
    regime: str,
    target_kind: str,
    programmed: Figure6OmPulsePlant,
    stack: Any,
    teacher: Any,
    loaders: Any,
    profile: Mapping[str, Any],
    store: RunStore,
) -> tuple[Figure6OmPulsePlant, PersistentFigure6OmPulseAdam, Mapping[str, Any]]:
    plant = programmed.clone_for_new_pulse_phase(
        pulse_noise_seed=RECOVERY_PULSE_NOISE_SEED,
        retain_apparent_observation=True,
    )
    if not torch.equal(plant.raw_a, programmed.raw_a):
        raise RuntimeError("OM recovery did not clone the P&V persistent state exactly.")
    if bool(torch.any(plant.pulse_count != 0)):
        raise RuntimeError("OM recovery pulse accounting did not start at zero.")
    optimizer = PersistentFigure6OmPulseAdam(
        plant,
        learning_rate_progress=OM_RECOVERY_LEARNING_RATE_PROGRESS,
        pulse_cap=int(profile["recovery_pulse_cap"]),
        pulse_selection_seed=RECOVERY_PULSE_SELECTION_SEED,
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
                    f"recovery {corruption_policy}/{regime}/{target_kind} "
                    f"epoch={epoch}/{epochs} batch={batch_index} "
                    f"pulses={totals['pulsed']}",
                    flush=True,
                )
        examples = int(totals["examples"])
        batches = int(totals["batches"])
        if examples < 1 or batches < 1:
            raise RuntimeError("OM open-loop recovery processed no examples.")
        validation = _evaluate_plant(
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            plant=plant,
            sample_limit=profile["validation_sample_limit"],
        )
        record = {
            "mode": "open_loop_persistent_OM_pulse_recovery",
            "corruption_policy": corruption_policy,
            "regime": regime,
            "target_kind": target_kind,
            "epoch": epoch,
            "train_persistent_state": {
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
                "SET": int(totals["up"]),
                "RESET": int(totals["down"]),
                "effective_state_changes": int(totals["changed"]),
                "mean_selection_probability": float(totals["probability_sum"]) / batches,
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
        primary = validation["held_apparent_projected"]
        print(
            f"recovery {corruption_policy}/{regime}/{target_kind} "
            f"epoch={epoch}/{epochs} "
            f"apparent_val={100.0*float(primary['student_accuracy']):.2f}% "
            f"cumulative_pulses={plant.report()['total_pulses']}",
            flush=True,
        )
    if selected_epoch is None or selected_plant_state is None or selected_optimizer_state is None:
        raise RuntimeError("OM recovery produced no selectable epoch.")
    plant.load_state_dict(selected_plant_state)
    optimizer.load_state_dict(selected_optimizer_state)
    return plant, optimizer, {
        "corruption_policy": corruption_policy,
        "regime": regime,
        "target_kind": target_kind,
        "selected_epoch": selected_epoch,
        "selection_state": "held_apparent_projected",
        "selection_used_test": False,
        "history": history,
        "initial_plant_state": initial_state,
        "selected_plant_state": selected_plant_state,
        "selected_optimizer_state": selected_optimizer_state,
    }


def _population_paths(store: RunStore, policy: str) -> tuple[Path, Path]:
    return (
        store.run_dir / "artifacts" / f"om_population__{policy}.npz",
        store.run_dir / "artifacts" / f"om_population__{policy}.receipt.json",
    )


def _population_input_paths(source_dir: Path, policy: str) -> tuple[Path, Path]:
    root = source_dir.expanduser().resolve()
    return (
        root / f"om_population__{policy}.npz",
        root / f"om_population__{policy}.receipt.json",
    )


def _load_population_input(
    *,
    source_dir: Path,
    policy: str,
    destination_population: Path,
    destination_receipt: Path,
) -> tuple[IbmReramArrayPopulation, Mapping[str, Any]]:
    source_population, source_receipt = _population_input_paths(source_dir, policy)
    if not source_population.is_file() or not source_receipt.is_file():
        raise FileNotFoundError(
            f"Expected pre-sampled OM population and receipt for {policy!r}."
        )
    receipt = json.loads(source_receipt.read_text(encoding="utf-8"))
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("population_sha256") != sha256_file(source_population)
        or receipt.get("population_fingerprint") is None
    ):
        raise ValueError(f"Invalid pre-sampled OM receipt for {policy!r}.")
    population = load_om_array_population(source_population)
    if (
        population.assignment_seed != OM_ASSIGNMENT_SEED
        or population.corruption_policy != policy
        or population.fingerprint != receipt.get("population_fingerprint")
        or population.aihwkit_version != "1.1.0"
    ):
        raise ValueError(f"Pre-sampled OM population does not match {policy!r}.")
    shutil.copy2(source_population, destination_population)
    shutil.copy2(source_receipt, destination_receipt)
    return population, dict(receipt)


def run(args: argparse.Namespace) -> Path:
    if args.profile not in PROFILE_SETTINGS:
        raise ValueError(f"Expected profile in {tuple(PROFILE_SETTINGS)!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("The Figure-6 OM pulse experiment requires CUDA.")
    profile = dict(PROFILE_SETTINGS[args.profile])
    population_source_dir = (
        args.population_source_dir.expanduser().resolve()
        if args.population_source_dir is not None
        else None
    )
    probe = (
        _load_om_probe_input(population_source_dir)
        if population_source_dir is not None
        else _verify_om_pulse_constants(args.aihwkit_python)
    )
    contract = _resolved_contract(
        args.profile,
        om_probe=probe,
        population_source_dir=population_source_dir,
    )
    teacher_path = args.teacher_weights.expanduser().resolve()
    if sha256_file(teacher_path) != EXPECTED_TEACHER_SHA256:
        raise ValueError("Frozen ReLU teacher SHA-256 mismatch.")
    population_inputs = []
    if population_source_dir is not None:
        probe_path = population_source_dir / "om_probe.json"
        population_inputs.append(
            {
                "role": "pinned_OM_constant_probe",
                "path": str(probe_path),
                "sha256": sha256_file(probe_path),
            }
        )
        for policy in CORRUPTION_POLICIES:
            population_path, receipt_path = _population_input_paths(
                population_source_dir, policy
            )
            population_inputs.extend(
                (
                    {
                        "role": f"OM_population_{policy}",
                        "path": str(population_path),
                        "sha256": sha256_file(population_path),
                    },
                    {
                        "role": f"OM_population_receipt_{policy}",
                        "path": str(receipt_path),
                        "sha256": sha256_file(receipt_path),
                    },
                )
            )
    else:
        population_inputs.append(
            {
                "role": "pinned_aihwkit_python",
                "path": str(args.aihwkit_python.expanduser().resolve()),
                "version": probe["version"],
            }
        )
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
            *population_inputs,
        ),
        resume_capability="unsupported",
    )
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        device = torch.device("cuda")
        student = resolve_student_spec(
            parse_student_config(_student_payload(profile)),
            RunMode.TRAIN,
        )
        loaders = _fresh_loaders(student)
        teacher, teacher_metadata = _load_teacher(teacher_path, device=device, spec=student)
        stack = build_student_stack(student, enable_measured=False)
        stack.cost.gain = FIXED_LOGIT_GAIN
        bindings = tuple(stack.bundle.catalog.trainable)
        shapes = tuple(tuple(int(item) for item in binding.state.shape) for binding in bindings)
        if shapes != ((1568, 100), (100, 20)):
            raise RuntimeError("Frozen DRN physical topology changed.")

        populations: dict[str, IbmReramArrayPopulation] = {}
        receipts = {}
        population_paths = []
        for policy in CORRUPTION_POLICIES:
            population_path, receipt_path = _population_paths(store, policy)
            if population_source_dir is None:
                population, receipt = sample_om_array_population_external(
                    bindings,
                    assignment_seed=OM_ASSIGNMENT_SEED,
                    corruption_policy=policy,
                    aihwkit_python=args.aihwkit_python,
                    population_path=population_path,
                    receipt_path=receipt_path,
                )
            else:
                population, receipt = _load_population_input(
                    source_dir=population_source_dir,
                    policy=policy,
                    destination_population=population_path,
                    destination_receipt=receipt_path,
                )
            if population.binding_keys != tuple(binding.key for binding in bindings):
                raise RuntimeError("OM population binding keys changed.")
            if population.binding_shapes != shapes:
                raise RuntimeError("OM population binding shapes changed.")
            populations[policy] = population
            receipts[policy] = receipt
            population_paths.extend((population_path, receipt_path))
        pairing = validate_paired_om_populations(
            populations["published"],
            populations["counterfactual_repaired"],
        )

        direct_masters = _initial_masters(teacher, device)
        devices = sum(value.numel() for value in direct_masters) * 4
        if devices != 158_800:
            raise RuntimeError("Frozen four-cell DRN device count changed.")
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
            "Figure-6 endpoints ready "
            f"target={target_sampling['accepted_seeds']} "
            f"hwa_bank={bank_sampling['accepted_seeds']} "
            f"effective_assignments={len(augmented_banks[INDEPENDENT_ENDPOINTS])}",
            flush=True,
        )
        hwa_records = {}
        for regime in ENDPOINT_REGIMES:
            target_field = target_fields[regime][0]
            reset_report = target_field.population_report.get("full_tile_RESET")
            if (
                not isinstance(reset_report, Mapping)
                or float(reset_report.get("lower_clip", -1.0)) != RESET_STATE_FLOOR
            ):
                raise RuntimeError("Generated endpoint field uses the wrong RESET floor.")
            hwa_loaders = _fresh_loaders(student)
            selected_masters, hwa = _train_hwa(
                regime=regime,
                initial_masters=direct_masters,
                bank=augmented_banks[regime],
                target_field=target_field,
                stack=stack,
                teacher=teacher,
                loaders=hwa_loaders,
                profile=profile,
                store=store,
            )
            hwa_records[regime] = {
                "masters": tuple(
                    value.to(device=device, dtype=torch.float32)
                    for value in selected_masters
                ),
                "report": hwa,
            }
        hwa_checkpoint = atomic_torch_save(
            {
                "schema": "ebl.figure6_endpoint_hwa_targets",
                "schema_version": 1,
                "profile": args.profile,
                "target_sampling": target_sampling,
                "bank_sampling": bank_sampling,
                "regimes": {
                    regime: {
                        "target_endpoint": {
                            "reset": tuple(
                                value.detach().cpu()
                                for value in target_fields[regime][0].reset
                            ),
                            "set": tuple(
                                value.detach().cpu()
                                for value in target_fields[regime][0].set
                            ),
                            "report": target_fields[regime][0].report(),
                        },
                        "HWA": hwa_records[regime]["report"],
                    }
                    for regime in ENDPOINT_REGIMES
                },
            },
            store.run_dir / "checkpoints" / "figure6_endpoint_hwa_targets.pt",
        )
        policy_reports = {}
        checkpoint_paths = []
        for policy in CORRUPTION_POLICIES:
            population = populations[policy]
            regime_reports = {}
            for regime in ENDPOINT_REGIMES:
                field = target_fields[regime][0]
                hwa_masters = hwa_records[regime]["masters"]
                targets = {
                    "direct": direct_masters,
                    "endpoint_noise_HWA": hwa_masters,
                }
                arm_reports = {}
                for target_kind in TARGET_KINDS:
                    masters = targets[target_kind]
                    target_progress = master_to_progress(masters, field)
                    ideal_test = _evaluate_full_g(
                        stack=stack,
                        teacher=teacher,
                        loader=loaders.test,
                        full_g=map_masters_to_conductance(masters, field),
                        sample_limit=profile["test_sample_limit"],
                    )
                    constrained_progress = corruption_constrained_progress(
                        target_progress,
                        field,
                        population,
                    )
                    corruption_constrained_test = _evaluate_full_g(
                        stack=stack,
                        teacher=teacher,
                        loader=loaders.test,
                        full_g=progress_to_conductance(constrained_progress, field),
                        sample_limit=profile["test_sample_limit"],
                    )
                    programmed, pv_result = program_masters_with_verify(
                        masters,
                        field,
                        population,
                        pulse_noise_seed=PROGRAM_PULSE_NOISE_SEED,
                        tolerance_progress=OM_PROGRAM_VERIFY_TOLERANCE_PROGRESS,
                        maximum_pulses=OM_PROGRAM_VERIFY_MAXIMUM_PULSES,
                        noisy_initial_reset_verify=True,
                        preaccept_exact_healthy_reset_targets=True,
                    )
                    programming = _program_verify_report(
                        target_progress=target_progress,
                        plant=programmed,
                        result=pv_result,
                    )
                    programmed_test = _evaluate_plant(
                        stack=stack,
                        teacher=teacher,
                        loader=loaders.test,
                        plant=programmed,
                        sample_limit=profile["test_sample_limit"],
                    )
                    primary = programmed_test["held_apparent_projected"]
                    print(
                        f"P&V {policy}/{regime}/{target_kind} "
                        f"apparent_test={100.0*float(primary['student_accuracy']):.2f}% "
                        f"persistent_test={100.0*float(programmed_test['persistent']['student_accuracy']):.2f}% "
                        f"accept={100.0*float(programming['controller_completion_fraction']):.2f}%",
                        flush=True,
                    )

                    recovery_loaders = _fresh_loaders(student)
                    recovered, pulse_optimizer, recovery = _recover_from_programmed(
                        corruption_policy=policy,
                        regime=regime,
                        target_kind=target_kind,
                        programmed=programmed,
                        stack=stack,
                        teacher=teacher,
                        loaders=recovery_loaders,
                        profile=profile,
                        store=store,
                    )
                    recovered_test = _evaluate_plant(
                        stack=stack,
                        teacher=teacher,
                        loader=loaders.test,
                        plant=recovered,
                        sample_limit=profile["test_sample_limit"],
                    )
                    recovered_primary = recovered_test["held_apparent_projected"]
                    print(
                        f"recovered {policy}/{regime}/{target_kind} "
                        f"apparent_test={100.0*float(recovered_primary['student_accuracy']):.2f}% "
                        f"persistent_test={100.0*float(recovered_test['persistent']['student_accuracy']):.2f}%",
                        flush=True,
                    )

                    checkpoint_payload = {
                        "schema": "ebl.figure6_om_pulse_corruption_arm",
                        "schema_version": 1,
                        "profile": args.profile,
                        "corruption_policy": policy,
                        "population_fingerprint": population.fingerprint,
                        "population_artifact": {
                            "path": str(_population_paths(store, policy)[0]),
                            "sha256": receipts[policy]["population_sha256"],
                        },
                        "regime": regime,
                        "target_kind": target_kind,
                        "HWA_source": {
                            "checkpoint": str(hwa_checkpoint),
                            "checkpoint_sha256": sha256_file(hwa_checkpoint),
                            "selected_epoch": hwa_records[regime]["report"][
                                "selected_epoch"
                            ],
                            "trained_in_this_run": True,
                        },
                        "target_endpoint": {
                            "reset": tuple(value.detach().cpu() for value in field.reset),
                            "set": tuple(value.detach().cpu() for value in field.set),
                            "report": field.report(),
                        },
                        "logical_master": tuple(
                            value.detach().cpu().clone() for value in masters
                        ),
                        "target_progress": tuple(
                            value.detach().cpu().clone() for value in target_progress
                        ),
                        "corruption_constrained_target_progress": tuple(
                            value.detach().cpu().clone() for value in constrained_progress
                        ),
                        "program_verify": {
                            "result": _program_verify_tensor_payload(pv_result),
                            "report": programming,
                            "plant": programmed.state_dict(),
                        },
                        "recovery": recovery,
                        "selected_recovery_plant": recovered.state_dict(),
                        "selected_recovery_optimizer": pulse_optimizer.state_dict(),
                        "test": {
                            "ideal_requested_target": dict(ideal_test),
                            "corruption_constrained_exact_target": dict(
                                corruption_constrained_test
                            ),
                            "program_verify": programmed_test,
                            "selected_open_loop_recovery": recovered_test,
                        },
                    }
                    checkpoint = atomic_torch_save(
                        checkpoint_payload,
                        store.run_dir
                        / "checkpoints"
                        / f"{policy}__{regime}__{target_kind}.pt",
                    )
                    checkpoint_paths.append(checkpoint)
                    arm_reports[target_kind] = {
                        "ideal_requested_target_test": dict(ideal_test),
                        "corruption_constrained_exact_target_test": dict(
                            corruption_constrained_test
                        ),
                        "program_verify": programming,
                        "program_verify_test": programmed_test,
                        "recovery": _recovery_summary(recovery),
                        "selected_recovery_test": recovered_test,
                        "selected_recovery_plant": dict(recovered.report()),
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256_file(checkpoint),
                    }
                regime_reports[regime] = {
                    "HWA_selected_epoch": hwa_records[regime]["report"][
                        "selected_epoch"
                    ],
                    "target_endpoint": field.report(),
                    "arms": arm_reports,
                }
            policy_reports[policy] = {
                "population_fingerprint": population.fingerprint,
                "population_receipt": receipts[policy],
                "corrupt_cells": int(population.corrupt.sum().item()),
                "regimes": regime_reports,
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
            "HWA": {
                "trained_in_this_run": True,
                "target_sampling": target_sampling,
                "bank_sampling": bank_sampling,
                "checkpoint": str(hwa_checkpoint),
                "checkpoint_sha256": sha256_file(hwa_checkpoint),
                "regimes": {
                    regime: _without_tensor_payload(hwa_records[regime]["report"])
                    for regime in ENDPOINT_REGIMES
                },
            },
            "OM_probe": dict(probe),
            "population_pairing": pairing,
            "corruption_policies": policy_reports,
            "claims": {
                "RESET_lower_clip": RESET_STATE_FLOOR,
                "RESET_upper_clip": None,
                "SET_clipping": None,
                "RESET_and_SET_endpoint_distributions_match_source_Figure6_model": True,
                "pulse_dynamics_are_OM_not_HfO2": True,
                "published_and_repaired_healthy_identities_are_bitwise_equal": True,
                "published_corrupt_singletons_are_immutable": True,
                "programming_uses_apparent_feedback": True,
                "recovery_starts_from_exact_persistent_PV_endpoint": True,
                "recovery_mutations_are_open_loop_stochastic_OM_pulses_only": True,
                "recovery_is_autonomous_on_chip_learning": False,
                "HWA_includes_PV_or_corruption_noise_during_training": False,
                "physical_power_claim_supported": False,
            },
        }
        summary_path = store.run_dir / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        artifacts = [
            store.artifact_record(summary_path, kind="scientific_summary"),
            store.artifact_record(hwa_checkpoint, kind="Figure6_endpoint_HWA_targets"),
            *(store.artifact_record(path, kind="exact_arm_checkpoint") for path in checkpoint_paths),
        ]
        for path in population_paths:
            artifacts.append(
                store.artifact_record(
                    path,
                    kind=("OM_population" if path.suffix == ".npz" else "OM_population_receipt"),
                )
            )
        result = store.complete(metrics=summary, artifacts=artifacts)
        print(f"complete result={result}", flush=True)
        return result
    except BaseException as error:
        store.fail(error)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_SETTINGS), default="pilot")
    parser.add_argument(
        "--population-source-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing paired OM NPZ/receipt artifacts and "
            "om_probe.json; otherwise sample with --aihwkit-python."
        ),
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
        default=_ROOT / "simulation_results" / "figure6_om_pulse_corruption",
    )
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
