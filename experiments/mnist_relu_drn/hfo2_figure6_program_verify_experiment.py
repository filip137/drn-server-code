"""Program-and-verify deployment and open-loop recovery for Figure-6 HfO2.

This exploratory follow-up freezes the direct and endpoint-noise-HWA logical
targets selected by ``hfo2_figure6_drn_experiment.py``.  For each endpoint
pairing regime it evaluates four distinct arms:

* direct target programmed from full RESET with apparent-feedback P&V;
* endpoint-noise-HWA target programmed by the same P&V controller;
* open-loop stochastic-pulse recovery from the direct P&V endpoint; and
* open-loop stochastic-pulse recovery from the HWA P&V endpoint.

P&V and recovery never share pulse accounting.  Recovery clones the exact
persistent P&V endpoint, regenerates the same device-to-device pulse
parameters, and starts a separately seeded stochastic pulse phase.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
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
    HFO2_PROGRAM_VERIFY_MAXIMUM_PULSES,
    HFO2_PROGRAM_VERIFY_TOLERANCE_PROGRESS,
    EndpointField,
    HfO2Figure6PulsePlant,
    PersistentHfO2PulseAdam,
    flatten_physical,
    master_to_progress,
    map_masters_to_conductance,
    program_masters_with_verify,
)
from experiments.mnist_relu_drn.hfo2_figure6_drn_experiment import (
    EXPECTED_TEACHER_SHA256,
    FIXED_LOGIT_GAIN,
    HFO2_RECOVERY_LEARNING_RATE_PROGRESS,
    PROFILE_SETTINGS,
    _evaluate_full_g,
    _fresh_loaders,
    _initial_masters,
    _one_forward_gradient,
    _student_payload,
    _verify_aihwkit_pulse_constants,
)
from experiments.mnist_relu_drn.hfo2_figure6_endpoint_regimes import (
    ENDPOINT_REGIMES,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import limited
from experiments.schema import RunMode
from training.checkpoint import atomic_torch_save


_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "mnist_hfo2_figure6_program_verify_recovery.v2"
SOURCE_EXPERIMENT_ID = "mnist_hfo2_figure6_direct_hwa_recovery.v1"
TARGET_KINDS = ("direct", "endpoint_noise_HWA")

PROGRAM_PULSE_PARAMETER_SEED = 94001
PROGRAM_PULSE_NOISE_SEED = 94002
RECOVERY_PULSE_NOISE_SEED = 95002
RECOVERY_PULSE_SELECTION_SEED = 95003

DEFAULT_SOURCE_HWA_RUN = (
    _ROOT
    / "simulation_results"
    / "hfo2_figure6_drn"
    / "20260903T162705.930811Z-8c10965a-7f934fe4"
)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    digest = sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_summary(value: torch.Tensor) -> Mapping[str, float]:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if tensor.numel() < 1 or not bool(torch.isfinite(tensor).all()):
        raise ValueError("Expected a non-empty finite tensor summary input.")
    return {
        "minimum": float(tensor.min().item()),
        "median": float(torch.quantile(tensor, 0.5).item()),
        "mean": float(tensor.mean().item()),
        "standard_deviation": (
            float(tensor.std(unbiased=True).item()) if tensor.numel() > 1 else 0.0
        ),
        "maximum": float(tensor.max().item()),
    }


def _residual_summary(residual: torch.Tensor) -> Mapping[str, float]:
    value = residual.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    absolute = value.abs()
    return {
        "mean": float(value.mean().item()),
        "standard_deviation": (
            float(value.std(unbiased=True).item()) if value.numel() > 1 else 0.0
        ),
        "mean_absolute": float(absolute.mean().item()),
        "rms": float(value.square().mean().sqrt().item()),
        "p90_absolute": float(torch.quantile(absolute, 0.9).item()),
        "p99_absolute": float(torch.quantile(absolute, 0.99).item()),
        "maximum_absolute": float(absolute.max().item()),
    }


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _load_source_hwa_run(
    source_dir: Path,
) -> tuple[Mapping[str, Mapping[str, Any]], Mapping[str, Any]]:
    source = source_dir.expanduser().resolve()
    result_path = source / "result.json"
    summary_path = source / "scientific_summary.json"
    if not result_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Expected a completed source HWA run bundle.")
    result = _load_json(result_path)
    summary = _load_json(summary_path)
    if result.get("status") != "complete" or summary.get("status") != "complete":
        raise ValueError("The source HWA run is not semantically complete.")
    if summary.get("schema") != SOURCE_EXPERIMENT_ID:
        raise ValueError("Unexpected source HWA experiment schema.")
    source_regimes = summary.get("regimes")
    if not isinstance(source_regimes, Mapping) or set(source_regimes) != set(
        ENDPOINT_REGIMES
    ):
        raise ValueError("The source HWA run does not cover both endpoint regimes.")

    loaded: dict[str, Mapping[str, Any]] = {}
    checkpoint_records = {}
    for regime in ENDPOINT_REGIMES:
        report = source_regimes[regime]
        if not isinstance(report, Mapping):
            raise ValueError(f"Invalid source report for {regime}.")
        checkpoint = source / "checkpoints" / f"{regime}.pt"
        observed_hash = sha256_file(checkpoint)
        if observed_hash != report.get("checkpoint_sha256"):
            raise ValueError(f"Source HWA checkpoint hash mismatch for {regime}.")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != "ebl.hfo2_figure6_drn_arm_checkpoint"
            or payload.get("schema_version") != 1
            or payload.get("regime") != regime
        ):
            raise ValueError(f"Invalid source HWA checkpoint for {regime}.")
        endpoint = payload.get("target_endpoint")
        hwa = payload.get("HWA")
        if not isinstance(endpoint, Mapping) or not isinstance(hwa, Mapping):
            raise ValueError("Source checkpoint is missing endpoint or HWA state.")
        reset = endpoint.get("reset")
        set_state = endpoint.get("set")
        endpoint_report = endpoint.get("report")
        masters = hwa.get("selected_master")
        if (
            not isinstance(reset, (tuple, list))
            or not isinstance(set_state, (tuple, list))
            or not isinstance(endpoint_report, Mapping)
            or not isinstance(masters, (tuple, list))
            or len(reset) != 2
            or len(set_state) != 2
            or len(masters) != 2
        ):
            raise ValueError("Malformed source endpoint/HWA tensor payload.")
        shapes = tuple(tuple(int(item) for item in value.shape) for value in reset)
        field = EndpointField(
            reset=tuple(value.detach().cpu().to(torch.float32) for value in reset),
            set=tuple(value.detach().cpu().to(torch.float32) for value in set_state),
            shapes=shapes,  # type: ignore[arg-type]
            layouts=ENDPOINT_LAYOUTS,
            population_report=dict(endpoint_report.get("population", {})),
        )
        current_report = field.report()
        if (
            current_report["reset_sha256"] != endpoint_report.get("reset_sha256")
            or current_report["set_sha256"] != endpoint_report.get("set_sha256")
        ):
            raise ValueError("Source endpoint tensors do not match their report.")
        selected_masters = tuple(
            value.detach().cpu().to(torch.float32).clone() for value in masters
        )
        master_to_progress(selected_masters, field)
        loaded[regime] = {
            "field": field,
            "HWA_masters": selected_masters,
            "HWA_selected_epoch": int(hwa["selected_epoch"]),
            "source_test": dict(payload["test"]),
        }
        checkpoint_records[regime] = {
            "path": str(checkpoint),
            "sha256": observed_hash,
            "selected_HWA_epoch": int(hwa["selected_epoch"]),
        }
    source_info = {
        "path": str(source),
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "scientific_summary_path": str(summary_path),
        "scientific_summary_sha256": sha256_file(summary_path),
        "profile": summary.get("profile"),
        "checkpoints": checkpoint_records,
    }
    return loaded, source_info


def _resolved_contract(
    profile_name: str,
    *,
    source_info: Mapping[str, Any],
) -> Mapping[str, Any]:
    profile = dict(PROFILE_SETTINGS[profile_name])
    return {
        "schema": EXPERIMENT_ID,
        "schema_version": 1,
        "evidence_tier": "exploratory_noncanonical",
        "profile": profile_name,
        "profile_settings": profile,
        "source_HWA_run": dict(source_info),
        "target_states": {
            "direct": "teacher_weight/layer_absmax",
            "endpoint_noise_HWA": (
                "frozen selected logical master from source HWA checkpoint"
            ),
            "HWA_training_in_this_follow_up": False,
            "HWA_source_noise": "endpoint_population_variation_only",
        },
        "program_and_verify": {
            "initial_persistent_state": "p=0=full_tile_RESET_for_every_cell",
            "initial_apparent_verify": "one_noisy_RESET_observation",
            "exact_RESET_targets": (
                "preaccepted_from_known_full_RESET_initialization_without_"
                "target_programming_pulses"
            ),
            "target": "continuous_normalized_progress",
            "controller": "one_stochastic_pulse_then_apparent_verify",
            "controller_access": "target_apparent_progress_and_pulse_history_only",
            "tolerance_progress": HFO2_PROGRAM_VERIFY_TOLERANCE_PROGRESS,
            "tolerance_definition": "half_nominal_HfO2_central_progress_step",
            "maximum_target_pulses_per_cell": HFO2_PROGRAM_VERIFY_MAXIMUM_PULSES,
            "pulse_parameter_seed": PROGRAM_PULSE_PARAMETER_SEED,
            "pulse_noise_seed": PROGRAM_PULSE_NOISE_SEED,
            "same_seeds_all_target_kinds_and_regimes": True,
            "persistent_endpoint_used_for_continuation": True,
            "apparent_endpoint_substituted_for_persistent_state": False,
        },
        "network_evaluation": {
            "primary": "held_apparent_progress_projected_to_[0,1]",
            "secondary": "persistent_progress",
            "literal_apparent_state_saved": True,
            "projection_reason": "positive_passive_DRN_conductance",
            "inference_read_noise_model": None,
        },
        "recovery": {
            "arms": [
                "direct_PV_then_open_loop_recovery",
                "endpoint_noise_HWA_PV_then_open_loop_recovery",
            ],
            "start_state": "exact_named_persistent_PV_endpoint",
            "persistent_state_clone_is_bitwise_exact": True,
            "P&V_pulse_accounting_carried_into_recovery_cap": False,
            "pulse_parameter_seed": PROGRAM_PULSE_PARAMETER_SEED,
            "pulse_noise_seed": RECOVERY_PULSE_NOISE_SEED,
            "pulse_selection_seed": RECOVERY_PULSE_SELECTION_SEED,
            "optimizer": "digital_BPTT_Adam_to_Bernoulli_one_pulse_commands",
            "learning_rate_progress": HFO2_RECOVERY_LEARNING_RATE_PROGRESS,
            "pulse_cap_per_cell": profile["recovery_pulse_cap"],
            "epochs": profile["recovery_epochs"],
            "verify_reads_during_updates": 0,
            "closed_loop_target_controller_during_recovery": False,
            "claim_label": "hardware-in-loop open-loop stochastic-pulse recovery",
            "autonomous_on_chip_learning": False,
        },
        "matching": {
            "same_teacher_topology_solver_data_gain_and_target_endpoint_field": True,
            "same_device_pulse_parameters_across_direct_and_HWA": True,
            "common_random_programming_and_recovery_seeds": True,
            "same_minibatch_order_and_recovery_budget": True,
        },
        "selection": {
            "recovery": (
                "best held-apparent projected validation accuracy then KL"
            ),
            "test_used_for_selection": False,
            "target_assignment_was_not_sealed_during_prior_HWA_development": True,
        },
    }


def _program_verify_tensor_payload(result: Any) -> Mapping[str, torch.Tensor]:
    return {
        name: getattr(result, name).detach().cpu().clone()
        for name in (
            "accepted",
            "nonfinite",
            "budget_exhausted",
            "apparent_endpoint",
            "set_count",
            "reset_count",
            "total_pulses",
            "verify_count",
            "reversals",
        )
    }


def _program_verify_report(
    *,
    target_progress: Sequence[torch.Tensor],
    plant: HfO2Figure6PulsePlant,
    result: Any,
) -> Mapping[str, Any]:
    target = flatten_physical(tuple(target_progress))
    persistent = flatten_physical(plant.progress)
    apparent = flatten_physical(plant.apparent_progress_unprojected)
    projected = apparent.clamp(0.0, 1.0)
    tolerance = HFO2_PROGRAM_VERIFY_TOLERANCE_PROGRESS
    reset_target = target == 0.0
    programmed_target = ~reset_target
    apparent_within_tolerance = (apparent - target).abs() <= tolerance
    persistent_within_tolerance = (persistent - target).abs() <= tolerance
    if not torch.equal(result.apparent_endpoint, apparent):
        raise RuntimeError("P&V apparent endpoint reporting mismatch.")
    if not torch.equal(result.set_count + result.reset_count, result.total_pulses):
        raise RuntimeError("P&V directional pulse counts do not sum to total.")
    terminal = result.accepted | result.nonfinite | result.budget_exhausted
    if not bool(torch.all(terminal)):
        raise RuntimeError("P&V left a cell without a terminal outcome.")
    return {
        "cells": plant.size,
        "target_progress": _tensor_summary(target),
        "target_progress_sha256": _tensor_sha256(target),
        "persistent_progress": _tensor_summary(persistent),
        "persistent_progress_sha256": _tensor_sha256(persistent),
        "literal_apparent_progress": _tensor_summary(apparent),
        "literal_apparent_progress_sha256": _tensor_sha256(apparent),
        "projected_apparent_progress": _tensor_summary(projected),
        "projected_apparent_progress_sha256": _tensor_sha256(projected),
        "persistent_minus_target": _residual_summary(persistent - target),
        "literal_apparent_minus_target": _residual_summary(apparent - target),
        "projected_apparent_minus_target": _residual_summary(projected - target),
        "controller_completed": int(result.accepted.sum().item()),
        "controller_completion_fraction": float(result.accepted.float().mean().item()),
        "RESET_targets_preaccepted_without_programming": int(
            reset_target.sum().item()
        ),
        "programmed_nonRESET_targets": int(programmed_target.sum().item()),
        "programmed_nonRESET_apparent_accepted": int(
            (result.accepted & programmed_target).sum().item()
        ),
        "programmed_nonRESET_apparent_acceptance_fraction": float(
            result.accepted[programmed_target].float().mean().item()
        ),
        "all_targets_literal_apparent_within_tolerance": int(
            apparent_within_tolerance.sum().item()
        ),
        "persistent_within_tolerance": int(
            persistent_within_tolerance.sum().item()
        ),
        "persistent_within_tolerance_fraction": float(
            persistent_within_tolerance.float().mean().item()
        ),
        "programmed_nonRESET_persistent_within_tolerance": int(
            (persistent_within_tolerance & programmed_target).sum().item()
        ),
        "programmed_nonRESET_persistent_within_tolerance_fraction": float(
            persistent_within_tolerance[programmed_target].float().mean().item()
        ),
        "budget_exhausted": int(result.budget_exhausted.sum().item()),
        "nonfinite": int(result.nonfinite.sum().item()),
        "target_at_RESET": int((target == 0.0).sum().item()),
        "target_at_SET": int((target == 1.0).sum().item()),
        "persistent_at_RESET": int((persistent == 0.0).sum().item()),
        "persistent_at_SET": int((persistent == 1.0).sum().item()),
        "apparent_below_RESET": int((apparent < 0.0).sum().item()),
        "apparent_above_SET": int((apparent > 1.0).sum().item()),
        "pulses": {
            "total": int(result.total_pulses.sum().item()),
            "SET": int(result.set_count.sum().item()),
            "RESET": int(result.reset_count.sum().item()),
            "pulsed_cells": int((result.total_pulses > 0).sum().item()),
            "mean_per_cell": float(result.total_pulses.float().mean().item()),
            "maximum_per_cell": int(result.total_pulses.max().item()),
            "reversals": int(result.reversals.sum().item()),
            "maximum_reversals_per_cell": int(result.reversals.max().item()),
            "verifies_including_initial": int(result.verify_count.sum().item()),
            "maximum_verifies_per_cell": int(result.verify_count.max().item()),
        },
        "controller": {
            "kind": "one_pulse_apparent_feedback",
            "tolerance_progress": tolerance,
            "maximum_pulses_per_cell": HFO2_PROGRAM_VERIFY_MAXIMUM_PULSES,
            "initial_reset_verify_is_noisy": True,
            "exact_RESET_targets_are_preaccepted": True,
            "fresh_read_correctness": None,
            "fresh_read_reason": "no_independent_inference_read_noise_model",
        },
    }


def _evaluate_plant(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    plant: HfO2Figure6PulsePlant,
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
    regime: str,
    target_kind: str,
    programmed: HfO2Figure6PulsePlant,
    stack: Any,
    teacher: Any,
    loaders: Any,
    profile: Mapping[str, Any],
    store: RunStore,
) -> tuple[HfO2Figure6PulsePlant, PersistentHfO2PulseAdam, Mapping[str, Any]]:
    plant = programmed.clone_for_new_pulse_phase(
        pulse_noise_seed=RECOVERY_PULSE_NOISE_SEED,
        retain_apparent_observation=True,
    )
    if not torch.equal(plant.raw_a, programmed.raw_a):
        raise RuntimeError("Recovery did not clone the persistent P&V state exactly.")
    if bool(torch.any(plant.pulse_count != 0)):
        raise RuntimeError("Recovery pulse accounting did not start at zero.")
    optimizer = PersistentHfO2PulseAdam(
        plant,
        learning_rate_progress=HFO2_RECOVERY_LEARNING_RATE_PROGRESS,
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
                    f"recovery {regime}/{target_kind} epoch={epoch}/{epochs} "
                    f"batch={batch_index} pulses={totals['pulsed']}",
                    flush=True,
                )
        examples = int(totals["examples"])
        batches = int(totals["batches"])
        if examples < 1 or batches < 1:
            raise RuntimeError("Open-loop recovery processed no examples.")
        validation = _evaluate_plant(
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            plant=plant,
            sample_limit=profile["validation_sample_limit"],
        )
        record = {
            "mode": "open_loop_persistent_hfo2_pulse_recovery",
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
        primary = validation["held_apparent_projected"]
        print(
            f"recovery {regime}/{target_kind} epoch={epoch}/{epochs} "
            f"apparent_val={100.0*float(primary['student_accuracy']):.2f}% "
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
        "target_kind": target_kind,
        "selected_epoch": selected_epoch,
        "selection_state": "held_apparent_projected",
        "selection_used_test": False,
        "history": history,
        "initial_plant_state": initial_state,
        "selected_plant_state": selected_plant_state,
        "selected_optimizer_state": selected_optimizer_state,
    }


def _recovery_summary(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key
        not in {
            "initial_plant_state",
            "selected_plant_state",
            "selected_optimizer_state",
        }
    }


def run(args: argparse.Namespace) -> Path:
    if args.profile not in PROFILE_SETTINGS:
        raise ValueError(f"Expected profile in {tuple(PROFILE_SETTINGS)!r}.")
    if not torch.cuda.is_available():
        raise RuntimeError("The Figure-6 P&V experiment requires CUDA.")
    profile = dict(PROFILE_SETTINGS[args.profile])
    source_arms, source_info = _load_source_hwa_run(args.source_hwa_run)
    aihwkit = _verify_aihwkit_pulse_constants(args.aihwkit_python)
    contract = dict(_resolved_contract(args.profile, source_info=source_info))
    contract["aihwkit_pulse_source"] = dict(aihwkit)
    teacher_path = args.teacher_weights.expanduser().resolve()
    if sha256_file(teacher_path) != EXPECTED_TEACHER_SHA256:
        raise ValueError("Frozen ReLU teacher SHA-256 mismatch.")
    source_inputs = tuple(
        {
            "role": f"source_HWA_checkpoint_{regime}",
            "path": source_info["checkpoints"][regime]["path"],
            "sha256": source_info["checkpoints"][regime]["sha256"],
        }
        for regime in ENDPOINT_REGIMES
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
            *source_inputs,
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
        direct_masters = _initial_masters(teacher, device)
        shapes = tuple(
            tuple(int(item) for item in binding.state.shape)
            for binding in stack.bundle.catalog.trainable
        )
        if shapes != ((1568, 100), (100, 20)):
            raise RuntimeError("Frozen DRN physical topology changed.")

        regime_reports = {}
        checkpoint_paths = []
        for regime in ENDPOINT_REGIMES:
            source_arm = source_arms[regime]
            source_field = source_arm["field"]
            if source_field.shapes != shapes:
                raise RuntimeError("Source HWA endpoint topology changed.")
            field = source_field.to(device)
            hwa_masters = tuple(
                value.to(device=device, dtype=torch.float32)
                for value in source_arm["HWA_masters"]
            )
            targets = {
                "direct": direct_masters,
                "endpoint_noise_HWA": hwa_masters,
            }
            arm_reports = {}
            for target_kind in TARGET_KINDS:
                masters = targets[target_kind]
                target_progress = master_to_progress(masters, field)
                exact_test = _evaluate_full_g(
                    stack=stack,
                    teacher=teacher,
                    loader=loaders.test,
                    full_g=map_masters_to_conductance(masters, field),
                    sample_limit=profile["test_sample_limit"],
                )
                programmed, pv_result = program_masters_with_verify(
                    masters,
                    field,
                    pulse_parameter_seed=PROGRAM_PULSE_PARAMETER_SEED,
                    pulse_noise_seed=PROGRAM_PULSE_NOISE_SEED,
                    tolerance_progress=HFO2_PROGRAM_VERIFY_TOLERANCE_PROGRESS,
                    maximum_pulses=HFO2_PROGRAM_VERIFY_MAXIMUM_PULSES,
                    noisy_initial_reset_verify=True,
                    preaccept_exact_reset_targets=True,
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
                    f"P&V {regime}/{target_kind} "
                    f"apparent_test={100.0*float(primary['student_accuracy']):.2f}% "
                    f"persistent_test={100.0*float(programmed_test['persistent']['student_accuracy']):.2f}% "
                    f"accept={100.0*float(programming['programmed_nonRESET_apparent_acceptance_fraction']):.2f}%",
                    flush=True,
                )

                recovery_loaders = _fresh_loaders(student)
                recovered, pulse_optimizer, recovery = _recover_from_programmed(
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
                    f"recovered {regime}/{target_kind} "
                    f"apparent_test={100.0*float(recovered_primary['student_accuracy']):.2f}% "
                    f"persistent_test={100.0*float(recovered_test['persistent']['student_accuracy']):.2f}%",
                    flush=True,
                )

                checkpoint_payload = {
                    "schema": "ebl.hfo2_figure6_program_verify_recovery_arm",
                    "schema_version": 2,
                    "regime": regime,
                    "target_kind": target_kind,
                    "profile": args.profile,
                    "source_HWA": dict(source_info["checkpoints"][regime]),
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
                    "program_verify": {
                        "result": _program_verify_tensor_payload(pv_result),
                        "report": programming,
                        "plant": programmed.state_dict(),
                    },
                    "recovery": recovery,
                    "selected_recovery_plant": recovered.state_dict(),
                    "selected_recovery_optimizer": pulse_optimizer.state_dict(),
                    "test": {
                        "exact_target": dict(exact_test),
                        "program_verify": programmed_test,
                        "selected_open_loop_recovery": recovered_test,
                    },
                }
                checkpoint = atomic_torch_save(
                    checkpoint_payload,
                    store.run_dir
                    / "checkpoints"
                    / f"{regime}__{target_kind}.pt",
                )
                checkpoint_paths.append(checkpoint)
                arm_reports[target_kind] = {
                    "exact_target_test": dict(exact_test),
                    "program_verify": programming,
                    "program_verify_test": programmed_test,
                    "recovery": _recovery_summary(recovery),
                    "selected_recovery_test": recovered_test,
                    "selected_recovery_plant": dict(recovered.report()),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            regime_reports[regime] = {
                "source_HWA_selected_epoch": source_arm["HWA_selected_epoch"],
                "target_endpoint": field.report(),
                "arms": arm_reports,
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
            "source_HWA_run": dict(source_info),
            "regimes": regime_reports,
            "claims": {
                "all_programming_starts_from_full_tile_RESET": True,
                "exact_RESET_targets_receive_target_programming_pulses": False,
                "programming_uses_apparent_feedback": True,
                "P&V_persistent_and_apparent_endpoints_are_distinct": True,
                "recovery_starts_from_exact_persistent_PV_endpoint": True,
                "recovery_verify_reads_during_updates": 0,
                "recovery_weight_mutations_are_open_loop_stochastic_pulses_only": True,
                "recovery_is_autonomous_on_chip_learning": False,
                "HWA_includes_PV_noise_during_training": False,
                "HWA_target_receives_PV_at_deployment": True,
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
        "--source-hwa-run",
        type=Path,
        default=DEFAULT_SOURCE_HWA_RUN,
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
        default=_ROOT / "simulation_results" / "hfo2_figure6_program_verify",
    )
    return parser


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
