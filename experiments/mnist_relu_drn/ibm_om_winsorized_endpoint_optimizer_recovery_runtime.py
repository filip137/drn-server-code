"""CUDA runtime for matched optimizer recovery on saved target-87004 endpoints."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    _artifact_records,
    _evaluate_detailed,
    _input,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam import (
    ColumnSerialOpenLoopAdam,
    open_loop_pulse_port,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam_runtime import (
    DEPLOYMENT_SCHEMA,
    DEPLOYMENT_SCHEMA_VERSION,
    _physical_drift,
    _restore_deployment_plant,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_endpoint_optimizer_recovery import (
    ColumnSerialOpenLoopSgd,
    OmPlantTikiTaka,
    PulseApplication,
    initialize_auxiliary_at_symmetry,
    intrinsic_symmetry_raw_a,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_endpoint_optimizer_recovery_config import (
    FIXED_LOGIT_GAIN,
    EndpointArtifactContract,
    EndpointOptimizerRecoveryTrainSpec,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import (
    sync_catalog_from_persistent,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_runtime import (
    _load_winsorized_population,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save
from training.ibm_reram_program_verify import IbmReramRawActivePlant
from training.ibm_reram_raw_active_program_verify import (
    array_population_as_pulse_population,
    matched_trajectory_seeds,
)


if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_winsorized_endpoint_optimizer_recovery_exploratory"
)
SUMMARY_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_winsorized_endpoint_optimizer_recovery_state"
)
CHECKPOINT_SCHEMA_VERSION = 1
PREDECESSOR_SUMMARY_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam_exploratory"
)
OPTIMIZER_NAMES = ("open_loop_sgd", "open_loop_adam", "tt_v1", "tt_v2")


def _predecessor_run_dir(protocol: Any) -> Path:
    return (
        _ROOT
        / "results"
        / protocol.predecessor.result_id
        / protocol.predecessor.run_relative_path
    ).resolve()


def _expected_summary_path(protocol: Any) -> Path:
    return (
        _predecessor_run_dir(protocol) / protocol.predecessor.summary_relative_path
    ).resolve()


def _load_predecessor_summary(path: Path, protocol: Any) -> Mapping[str, Any]:
    expected = _expected_summary_path(protocol)
    if path.resolve() != expected:
        raise ValueError(
            "Expected --weights to name the exact predecessor scientific summary."
        )
    if not path.is_file() or sha256_file(path) != protocol.predecessor.summary_sha256:
        raise ValueError("Predecessor scientific-summary SHA-256 mismatch.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Expected a readable predecessor scientific summary.") from error
    target = payload.get("target")
    deployments = payload.get("deployments")
    if (
        payload.get("schema") != PREDECESSOR_SUMMARY_SCHEMA
        or payload.get("schema_version") != 1
        or payload.get("evidence_tier") != "exploratory_noncanonical"
        or not isinstance(target, Mapping)
        or target.get("assignment_seed") != protocol.target_assignment_seed
        or not isinstance(deployments, Mapping)
        or deployments.get("endpoint_seeds")
        != [item.endpoint_seed for item in protocol.predecessor.endpoints]
        or deployments.get("maximum_random_draws")
        != protocol.optimizers.c_maximum_random_draws
    ):
        raise ValueError("Predecessor summary semantic contract mismatch.")
    repeats = deployments.get("repeats")
    if not isinstance(repeats, list) or len(repeats) != len(protocol.predecessor.endpoints):
        raise ValueError("Expected all five predecessor deployment reports.")
    by_seed = {
        item.get("endpoint_seed"): item
        for item in repeats
        if isinstance(item, Mapping)
    }
    for contract in protocol.predecessor.endpoints:
        report = by_seed.get(contract.endpoint_seed)
        if (
            not isinstance(report, Mapping)
            or report.get("sha256") != contract.sha256
            or report.get("target_kind")
            != "target_baseline_plus_source_signed_contrast"
        ):
            raise ValueError("Predecessor deployment report/hash mismatch.")
    return payload


def _load_endpoint(
    path: Path,
    contract: EndpointArtifactContract,
    *,
    protocol: Any,
    population: Any,
) -> Mapping[str, Any]:
    if not path.is_file() or sha256_file(path) != contract.sha256:
        raise ValueError(f"Endpoint {contract.endpoint_seed} SHA-256 mismatch.")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Expected a readable weights-only deployment endpoint.") from error
    expected_fields = {
        "schema",
        "schema_version",
        "evidence_tier",
        "assignment_seed",
        "endpoint_seed",
        "target_kind",
        "target_clipping",
        "maximum_program_pulses",
        "maximum_random_draws",
        "binding_keys",
        "binding_shapes",
        "target_raw_x",
        "programming",
        "continuation_state",
        "test",
        "test_prediction_sha256",
        "persistent_endpoint_applied_to_drn",
        "apparent_endpoint_applied_to_drn",
        "inference_read_noise",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("Deployment endpoint fields differ from the frozen schema.")
    if (
        payload["schema"] != DEPLOYMENT_SCHEMA
        or payload["schema_version"] != DEPLOYMENT_SCHEMA_VERSION
        or payload["evidence_tier"] != "exploratory_noncanonical"
        or payload["assignment_seed"] != protocol.target_assignment_seed
        or payload["endpoint_seed"] != contract.endpoint_seed
        or payload["target_kind"]
        != "target_baseline_plus_source_signed_contrast"
        or payload["target_clipping"] is not False
        or payload["maximum_program_pulses"] != 128
        or payload["maximum_random_draws"]
        != protocol.optimizers.c_maximum_random_draws
        or payload["binding_keys"] != list(population.binding_keys)
        or payload["binding_shapes"]
        != [list(shape) for shape in population.binding_shapes]
        or payload["persistent_endpoint_applied_to_drn"] is not True
        or payload["apparent_endpoint_applied_to_drn"] is not False
        or payload["inference_read_noise"] is not False
        or payload["test_prediction_sha256"]
        != contract.historical_test_prediction_sha256
        or payload["test"].get("student_correct")
        != contract.historical_test_correct
        or payload["test"].get("examples") != 10_000
    ):
        raise ValueError("Deployment endpoint semantic contract mismatch.")
    size = population.size
    target = payload["target_raw_x"]
    state = payload["continuation_state"]
    if (
        not isinstance(target, torch.Tensor)
        or target.dtype != torch.float32
        or target.shape != (size,)
        or not bool(torch.all(torch.isfinite(target)))
        or not isinstance(state, Mapping)
        or state.get("schema_version") != 2
        or state.get("maximum_random_draws")
        != protocol.optimizers.c_maximum_random_draws
        or not isinstance(state.get("persistent"), torch.Tensor)
        or state["persistent"].dtype != torch.float32
        or state["persistent"].shape != (size,)
        or not isinstance(state.get("draw_indices"), torch.Tensor)
        or state["draw_indices"].dtype != torch.int64
        or state["draw_indices"].shape != (size,)
        or int(state["draw_indices"].max().item()) > 257
    ):
        raise ValueError("Deployment endpoint continuation state is invalid.")
    return payload


def _load_predecessor_bundle(protocol: Any) -> tuple[Any, Any, dict[int, tuple[Path, Mapping[str, Any]]]]:
    run_dir = _predecessor_run_dir(protocol)
    population_path = (run_dir / protocol.predecessor.population_relative_path).resolve()
    receipt_path = (
        run_dir / protocol.predecessor.population_receipt_relative_path
    ).resolve()
    if sha256_file(receipt_path) != protocol.predecessor.population_receipt_sha256:
        raise ValueError("Winsorized population receipt SHA-256 mismatch.")
    population = _load_winsorized_population(
        population_path,
        receipt_path,
        expected_sha256=protocol.predecessor.population_sha256,
        expected_fingerprint=protocol.predecessor.population_fingerprint,
        expected_assignment_seed=protocol.target_assignment_seed,
        source_joint_hardware_instance_id=(
            protocol.predecessor.source_joint_hardware_instance_id
        ),
    )
    pulse_population = array_population_as_pulse_population(population)
    endpoints = {}
    for contract in protocol.predecessor.endpoints:
        path = (run_dir / contract.relative_path).resolve()
        endpoints[contract.endpoint_seed] = (
            path,
            _load_endpoint(
                path, contract, protocol=protocol, population=population
            ),
        )
    return population, pulse_population, endpoints


def _seed_map(pairs: Sequence[Sequence[int]], endpoint_seed: int) -> int:
    mapping = {int(source): int(target) for source, target in pairs}
    if endpoint_seed not in mapping:
        raise ValueError("Endpoint seed is absent from a frozen RNG map.")
    return mapping[endpoint_seed]


def _tensor_summary(value: torch.Tensor) -> Mapping[str, float]:
    flat = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return {
        "minimum": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "rms": float(flat.square().mean().sqrt().item()),
        "maximum": float(flat.max().item()),
    }


def _counts_by_layer(
    counts: torch.Tensor, binding_shapes: Sequence[Sequence[int]]
) -> list[int]:
    cpu = counts.detach().cpu()
    result = []
    offset = 0
    for shape in binding_shapes:
        size = math.prod(tuple(shape))
        result.append(int(cpu[offset : offset + size].sum().item()))
        offset += size
    return result


def _maximum_by_layer(
    values: torch.Tensor, binding_shapes: Sequence[Sequence[int]]
) -> list[int]:
    cpu = values.detach().cpu()
    result = []
    offset = 0
    for shape in binding_shapes:
        size = math.prod(tuple(shape))
        result.append(int(cpu[offset : offset + size].max().item()))
        offset += size
    return result


def _summaries_by_layer(
    values: torch.Tensor, binding_shapes: Sequence[Sequence[int]]
) -> list[Mapping[str, float]]:
    result = []
    offset = 0
    for shape in binding_shapes:
        size = math.prod(tuple(shape))
        selected = values[offset : offset + size]
        summary = dict(_tensor_summary(selected))
        summary["absolute_maximum"] = float(selected.abs().max().item())
        result.append(summary)
        offset += size
    return result


def _a_diagnostic(
    plant: IbmReramRawActivePlant,
    symmetry_raw_a: torch.Tensor,
    *,
    binding_shapes: Sequence[Sequence[int]],
    pulse_count: torch.Tensor,
    pulse_cap: int,
) -> Mapping[str, Any]:
    symmetry = symmetry_raw_a.detach().to(plant.device)
    persistent_x = (plant.persistent - symmetry) / 2.0
    apparent_x = (plant.apparent - symmetry) / 2.0
    held_noise = apparent_x - persistent_x
    lower = plant.population.min_bound.to(plant.device)
    upper = plant.population.max_bound.to(plant.device)
    return {
        "persistent_raw_x_displacement": _tensor_summary(persistent_x),
        "held_apparent_raw_x_displacement": _tensor_summary(apparent_x),
        "held_apparent_minus_persistent_raw_x": _tensor_summary(held_noise),
        "apparent_semantics": (
            "held_post_write_sample_not_fresh_independent_transfer_read_noise"
        ),
        "persistent_saturated_lower": int(
            (torch.abs(plant.persistent - lower) <= 2e-6).sum().item()
        ),
        "persistent_saturated_upper": int(
            (torch.abs(plant.persistent - upper) <= 2e-6).sum().item()
        ),
        "pulses_by_layer": _counts_by_layer(pulse_count, binding_shapes),
        "cells_at_pulse_cap": int((pulse_count >= pulse_cap).sum().item()),
        "maximum_pulses_per_cell": int(pulse_count.max().item()),
        "algorithmic_transfer_read_noise": False,
    }


def _empty_pulse_totals() -> dict[str, Any]:
    return {
        "requested": 0,
        "applied": 0,
        "capped": 0,
        "upward": 0,
        "downward": 0,
        "probability_clipped": 0,
        "mean_probability_sum": 0.0,
        "maximum_probability": 0.0,
        "calls": 0,
    }


def _add_pulse(total: dict[str, Any], pulse: PulseApplication) -> None:
    for name in ("requested", "applied", "capped", "upward", "downward"):
        total[name] += int(getattr(pulse, name))
    total["probability_clipped"] += int(pulse.probability_clipped)
    total["mean_probability_sum"] += float(pulse.mean_probability)
    total["maximum_probability"] = max(
        float(total["maximum_probability"]), float(pulse.maximum_probability)
    )
    total["calls"] += 1


def _finalize_pulse(total: Mapping[str, Any]) -> Mapping[str, Any]:
    calls = int(total["calls"])
    return {
        "requested": int(total["requested"]),
        "applied": int(total["applied"]),
        "capped": int(total["capped"]),
        "upward": int(total["upward"]),
        "downward": int(total["downward"]),
        "probability_clipped_cells_across_steps": int(
            total["probability_clipped"]
        ),
        "mean_probability_across_calls": (
            float(total["mean_probability_sum"]) / calls if calls else 0.0
        ),
        "maximum_probability": float(total["maximum_probability"]),
        "calls": calls,
        "verify_reads": 0,
    }


def _restore_c(
    endpoint: Mapping[str, Any],
    *,
    pulse_population: Any,
    protocol: Any,
    device: torch.device,
) -> IbmReramRawActivePlant:
    plant = _restore_deployment_plant(
        endpoint,
        pulse_population=pulse_population,
        maximum_random_draws=protocol.optimizers.c_maximum_random_draws,
        device=device,
    )
    expected = endpoint["continuation_state"]["persistent"].to(device)
    if not torch.equal(plant.persistent, expected):
        raise RuntimeError("Restored slow C differs from its exact saved P0.")
    return plant


def _build_optimizer(
    *,
    optimizer_name: str,
    candidate_scale: float,
    endpoint_seed: int,
    c_plant: IbmReramRawActivePlant,
    population: Any,
    pulse_population: Any,
    symmetry: Any,
    protocol: Any,
    device: torch.device,
    tt_effective_transfer_lambdas: Sequence[float] | None = None,
) -> tuple[Any, IbmReramRawActivePlant | None, Mapping[str, Any]]:
    shapes = population.binding_shapes
    slow_seed = _seed_map(protocol.slow_pulse_selection_seed_by_p0, endpoint_seed)
    base_rates = protocol.optimizers.qat_layer_learning_rates
    if optimizer_name == "open_loop_sgd":
        if tt_effective_transfer_lambdas is not None:
            raise ValueError("Direct SGD does not accept TT transfer lambdas.")
        rates = tuple(candidate_scale * value for value in base_rates)
        return (
            ColumnSerialOpenLoopSgd(
                open_loop_pulse_port(c_plant),
                device=device,
                binding_shapes=shapes,
                layer_learning_rates_raw_x=rates,
                nominal_delta_x=protocol.optimizers.nominal_delta_x,
                pulse_cap=protocol.optimizers.pulse_cap_per_cell,
                pulse_selection_seed=slow_seed,
            ),
            None,
            {"candidate_scale": candidate_scale, "layer_learning_rates_raw_x": list(rates)},
        )
    if optimizer_name == "open_loop_adam":
        if tt_effective_transfer_lambdas is not None:
            raise ValueError("Direct Adam does not accept TT transfer lambdas.")
        if candidate_scale != 1.0:
            raise ValueError("Fixed Adam does not accept a candidate scale.")
        return (
            ColumnSerialOpenLoopAdam(
                open_loop_pulse_port(c_plant),
                device=device,
                binding_shapes=shapes,
                learning_rate_raw_x=protocol.optimizers.adam_learning_rate_raw_x,
                nominal_delta_x=protocol.optimizers.nominal_delta_x,
                pulse_cap=protocol.optimizers.pulse_cap_per_cell,
                pulse_selection_seed=slow_seed,
                beta1=protocol.optimizers.adam_beta1,
                beta2=protocol.optimizers.adam_beta2,
                epsilon=protocol.optimizers.adam_epsilon,
            ),
            None,
            {
                "candidate_scale": 1.0,
                "learning_rate_raw_x": protocol.optimizers.adam_learning_rate_raw_x,
                "selection_source": "frozen_predecessor_endpoint_89401_validation_screen",
            },
        )
    if optimizer_name not in {"tt_v1", "tt_v2"}:
        raise ValueError(f"Unexpected optimizer: {optimizer_name!r}.")
    auxiliary_endpoint_seed = _seed_map(
        protocol.auxiliary_endpoint_seed_by_p0, endpoint_seed
    )
    a_plant = IbmReramRawActivePlant(
        pulse_population,
        seeds=matched_trajectory_seeds(
            population, endpoint_seed=auxiliary_endpoint_seed
        ),
        device=device,
        maximum_random_draws=protocol.optimizers.tt_a_maximum_random_draws,
    )
    initialize_auxiliary_at_symmetry(a_plant, symmetry.raw_a)
    lambdas = (
        tuple(candidate_scale * value for value in base_rates)
        if tt_effective_transfer_lambdas is None
        else tuple(float(value) for value in tt_effective_transfer_lambdas)
    )
    if len(lambdas) != 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in lambdas
    ):
        raise ValueError("Expected two positive finite effective TT lambdas.")
    optimizer = OmPlantTikiTaka(
        open_loop_pulse_port(c_plant),
        a_plant,
        variant=optimizer_name,
        symmetry_raw_a=symmetry.raw_a,
        device=device,
        binding_shapes=shapes,
        effective_transfer_lambdas=lambdas,
        fast_learning_rate_raw_x=(
            protocol.optimizers.tt_fast_learning_rate_raw_x
        ),
        nominal_delta_x=protocol.optimizers.nominal_delta_x,
        pulse_cap=protocol.optimizers.pulse_cap_per_cell,
        fast_pulse_selection_seed=_seed_map(
            protocol.auxiliary_pulse_selection_seed_by_p0, endpoint_seed
        ),
        slow_pulse_selection_seed=slow_seed,
    )
    return (
        optimizer,
        a_plant,
        {
            "candidate_scale": candidate_scale,
            "effective_transfer_lambdas": list(lambdas),
            "scale_transfer_lr": False,
            "fast_learning_rate_raw_x": (
                protocol.optimizers.tt_fast_learning_rate_raw_x
            ),
            "auxiliary_endpoint_seed": auxiliary_endpoint_seed,
        },
    )


def _optimizer_pulse_counts(
    optimizer_name: str, optimizer: Any
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if optimizer_name == "open_loop_sgd":
        return optimizer.writer.pulse_count, None
    if optimizer_name == "open_loop_adam":
        return optimizer.pulse_count, None
    return optimizer.slow_writer.pulse_count, optimizer.fast_writer.pulse_count


def _run_arm(
    *,
    phase: str,
    optimizer_name: str,
    candidate_scale: float,
    maximum_batches: int | None,
    endpoint_path: Path,
    endpoint: Mapping[str, Any],
    endpoint_seed: int,
    population: Any,
    pulse_population: Any,
    symmetry: Any,
    stack: Any,
    teacher: Any,
    loaders: Any,
    train_generator_state: torch.Tensor,
    protocol: Any,
    store: RunStore,
    before_validation: Mapping[str, Any],
    before_test: Mapping[str, Any] | None,
    tt_effective_transfer_lambdas: Sequence[float] | None = None,
) -> Mapping[str, Any]:
    device = stack.device
    c_plant = _restore_c(
        endpoint,
        pulse_population=pulse_population,
        protocol=protocol,
        device=device,
    )
    c_before = c_plant.persistent.detach().cpu().clone()
    optimizer, a_plant, hyperparameters = _build_optimizer(
        optimizer_name=optimizer_name,
        candidate_scale=candidate_scale,
        endpoint_seed=endpoint_seed,
        c_plant=c_plant,
        population=population,
        pulse_population=pulse_population,
        symmetry=symmetry,
        protocol=protocol,
        device=device,
        tt_effective_transfer_lambdas=tt_effective_transfer_lambdas,
    )
    loaders.train_generator.set_state(train_generator_state.clone())
    train_examples = 0
    train_batches = 0
    train_kl = 0.0
    train_correct = 0
    c_total = _empty_pulse_totals()
    a_total = _empty_pulse_totals()
    conceptual_column_phases = 0
    a_read_events = 0
    a_values_read = 0
    h_crossings = [0, 0]
    h_cap_debt = [0, 0]
    for batch_index, (inputs, labels) in enumerate(
        limited(loaders.train, maximum_batches)
    ):
        sync_catalog_from_persistent(
            stack.bundle.catalog,
            c_plant,
            binding_shapes=population.binding_shapes,
        )
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        with torch.no_grad():
            teacher_logits = teacher.logits(inputs)
        stack.network.set_input(inputs, reset=True)
        stack.minimizer.compute_equilibrium()
        stack.cost.set_teacher(teacher_logits, labels)
        with torch.no_grad():
            train_kl += float(stack.cost.eval().sum().item())
            train_correct += int(
                stack.cost.student_logits().argmax(dim=1).eq(labels).sum().item()
            )
        gradients = tuple(stack.differentiator.compute_gradient())
        step = optimizer.step(gradients)
        if optimizer_name == "open_loop_sgd":
            _add_pulse(c_total, step.pulse)
            conceptual_column_phases += int(step.conceptual_column_phases)
        elif optimizer_name == "open_loop_adam":
            c_total["requested"] += int(step.requested_cell_coincidences)
            c_total["applied"] += int(step.applied_cell_pulses)
            c_total["capped"] += int(step.capped_cell_requests)
            c_total["upward"] += int(step.upward_pulses)
            c_total["downward"] += int(step.downward_pulses)
            c_total["probability_clipped"] += int(step.probability_clipped_cells)
            c_total["mean_probability_sum"] += float(step.mean_probability)
            c_total["maximum_probability"] = max(
                float(c_total["maximum_probability"]),
                float(step.maximum_probability),
            )
            c_total["calls"] += 1
            conceptual_column_phases += int(step.conceptual_column_phases)
        else:
            _add_pulse(a_total, step.fast_a_pulse)
            _add_pulse(c_total, step.slow_c_pulse)
            a_read_events += int(step.a_read_events)
            a_values_read += int(step.a_values_read)
            for layer_index in range(2):
                h_crossings[layer_index] += int(
                    step.h_threshold_crossings[layer_index]
                )
                h_cap_debt[layer_index] += int(
                    step.h_cap_debt_retained[layer_index]
                )
        train_examples += int(labels.shape[0])
        train_batches = batch_index + 1
    if train_batches < 1:
        raise RuntimeError("Expected every optimizer arm to process a minibatch.")
    sync_catalog_from_persistent(
        stack.bundle.catalog,
        c_plant,
        binding_shapes=population.binding_shapes,
    )
    after_validation, after_validation_prediction = _evaluate_detailed(
        stack, teacher, loaders.validation, sample_limit=None
    )
    after_test = None
    after_test_prediction_sha256 = None
    if before_test is not None:
        after_test, prediction = _evaluate_detailed(
            stack, teacher, loaders.test, sample_limit=None
        )
        after_test_prediction_sha256 = _tensor_sha256(prediction.to(torch.int64))
    c_counts, a_counts = _optimizer_pulse_counts(optimizer_name, optimizer)
    c_drift = _physical_drift(
        c_before,
        c_plant.persistent,
        binding_shapes=population.binding_shapes,
        pulse_population=pulse_population,
    )
    a_report = None
    if a_plant is not None:
        assert a_counts is not None
        a_report = _a_diagnostic(
            a_plant,
            symmetry.raw_a,
            binding_shapes=population.binding_shapes,
            pulse_count=a_counts,
            pulse_cap=protocol.optimizers.pulse_cap_per_cell,
        )
    optimizer_state = optimizer.state_dict()
    checkpoint_payload = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "evidence_tier": protocol.execution.evidence_tier,
        "phase": phase,
        "optimizer": optimizer_name,
        "endpoint_seed": endpoint_seed,
        "target_assignment_seed": protocol.target_assignment_seed,
        "source_p0_path": str(endpoint_path),
        "source_p0_sha256": sha256_file(endpoint_path),
        "hyperparameters": dict(hyperparameters),
        "binding_shapes": [list(shape) for shape in population.binding_shapes],
        "slow_c_continuation_state": c_plant.state_dict(),
        "fast_a_continuation_state": (
            None if a_plant is None else a_plant.state_dict()
        ),
        "intrinsic_symmetry_raw_a": symmetry.raw_a.detach().cpu().clone(),
        "optimizer_state": optimizer_state,
        "train_generator_start_state": train_generator_state.detach().cpu().clone(),
        "train_generator_end_state": loaders.train_generator.get_state().cpu().clone(),
        "before_validation": dict(before_validation),
        "after_validation": dict(after_validation),
        "before_test": None if before_test is None else dict(before_test),
        "after_test": None if after_test is None else dict(after_test),
        "full_conductance_forward": protocol.execution.full_conductance_forward,
        "verify_reads_during_training": 0,
        "authoritative_weight_shadow": None,
    }
    scale_token = str(candidate_scale).replace(".", "p")
    checkpoint_path = atomic_torch_save(
        checkpoint_payload,
        store.run_dir
        / "checkpoints"
        / f"{phase}_seed_{endpoint_seed}_{optimizer_name}_scale_{scale_token}.pt",
    )
    # Cheap strict round-trip catches accidental omission of the physical A/C
    # continuation before the exploratory run fans out over 16 evaluation arms.
    round_trip = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        round_trip.get("schema") != CHECKPOINT_SCHEMA
        or round_trip.get("optimizer") != optimizer_name
        or round_trip.get("source_p0_sha256") != sha256_file(endpoint_path)
        or not torch.equal(
            round_trip["slow_c_continuation_state"]["persistent"],
            c_plant.persistent.detach().cpu(),
        )
        or (a_plant is None) != (round_trip["fast_a_continuation_state"] is None)
    ):
        raise RuntimeError("Optimizer checkpoint failed strict physical-state round-trip.")
    report = {
        "phase": phase,
        "optimizer": optimizer_name,
        "endpoint_seed": endpoint_seed,
        "source_p0_path": str(endpoint_path),
        "source_p0_sha256": sha256_file(endpoint_path),
        "same_exact_p0_for_every_matched_arm": True,
        "hyperparameters": dict(hyperparameters),
        "train": {
            "examples": train_examples,
            "batches": train_batches,
            "kl_teacher_student": train_kl / train_examples,
            "student_accuracy": train_correct / train_examples,
        },
        "before_validation": dict(before_validation),
        "after_validation": dict(after_validation),
        "after_validation_prediction_sha256": _tensor_sha256(
            after_validation_prediction.to(torch.int64)
        ),
        "before_test": None if before_test is None else dict(before_test),
        "after_test": None if after_test is None else dict(after_test),
        "after_test_prediction_sha256": after_test_prediction_sha256,
        "slow_c_pulse": {
            **_finalize_pulse(c_total),
            "by_layer": _counts_by_layer(c_counts, population.binding_shapes),
            "cells_at_cap_by_layer": _counts_by_layer(
                (c_counts >= protocol.optimizers.pulse_cap_per_cell).to(torch.int64),
                population.binding_shapes,
            ),
            "maximum_per_cell_by_layer": _maximum_by_layer(
                c_counts, population.binding_shapes
            ),
            "cells_at_cap": int(
                (c_counts >= protocol.optimizers.pulse_cap_per_cell).sum().item()
            ),
            "maximum_per_cell": int(c_counts.max().item()),
            "conceptual_column_phases": conceptual_column_phases,
        },
        "fast_a_pulse": (
            None
            if a_plant is None
            else {
                **_finalize_pulse(a_total),
                "by_layer": _counts_by_layer(a_counts, population.binding_shapes),
                "cells_at_cap_by_layer": _counts_by_layer(
                    (
                        a_counts
                        >= protocol.optimizers.pulse_cap_per_cell
                    ).to(torch.int64),
                    population.binding_shapes,
                ),
                "maximum_per_cell_by_layer": _maximum_by_layer(
                    a_counts, population.binding_shapes
                ),
                "cells_at_cap": int(
                    (a_counts >= protocol.optimizers.pulse_cap_per_cell).sum().item()
                ),
                "maximum_per_cell": int(a_counts.max().item()),
            }
        ),
        "algorithmic_a_read": (
            None
            if a_plant is None
            else {
                "events": a_read_events,
                "values": a_values_read,
                "events_per_minibatch": 2,
                "values_per_minibatch": sum(
                    2 * shape[1] for shape in population.binding_shapes
                ),
                "state": (
                    "held_apparent_post_write_raw_x_displacement;"
                    "not_fresh_independent_read_noise"
                ),
                "verify_reads": 0,
            }
        ),
        "h": {
            "threshold_crossings_by_layer": h_crossings,
            "cap_blocked_debt_retained_by_layer_across_steps": h_cap_debt,
            "final_pulse_unit_state_by_layer": _summaries_by_layer(
                optimizer.h, population.binding_shapes
            ),
        }
        if optimizer_name == "tt_v2"
        else None,
        "slow_c_physical_drift": c_drift,
        "fast_a_physical_state": a_report,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_round_trip_validated": True,
        "test_used_for_selection": False,
        "verify_reads_during_training": 0,
    }
    store.append_metric({"mode": "endpoint_optimizer_arm", **report})
    after_label = "sealed" if after_test is None else (
        f"{100.0*float(after_test['student_accuracy']):.2f}%"
    )
    print(
        f"{phase} seed={endpoint_seed} optimizer={optimizer_name} "
        f"scale={candidate_scale:g} val="
        f"{100.0*float(after_validation['student_accuracy']):.2f}% "
        f"test={after_label} C_pulses={int(c_total['applied'])} "
        f"A_pulses={int(a_total['applied'])} verify=0",
        flush=True,
    )
    del optimizer, a_plant, c_plant, round_trip
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _selection_key(report: Mapping[str, Any]) -> tuple[int, float, int, float]:
    validation = report["after_validation"]
    return (
        int(validation["student_correct"]),
        -float(validation["kl_teacher_student"]),
        -int(report["slow_c_pulse"]["applied"]),
        -float(report["hyperparameters"]["candidate_scale"]),
    )


def _evaluate_endpoint_p0(
    *,
    endpoint: Mapping[str, Any],
    endpoint_seed: int,
    contract: EndpointArtifactContract,
    pulse_population: Any,
    population: Any,
    stack: Any,
    teacher: Any,
    loaders: Any,
    protocol: Any,
    open_test: bool,
) -> Mapping[str, Any]:
    plant = _restore_c(
        endpoint,
        pulse_population=pulse_population,
        protocol=protocol,
        device=stack.device,
    )
    sync_catalog_from_persistent(
        stack.bundle.catalog, plant, binding_shapes=population.binding_shapes
    )
    validation, validation_prediction = _evaluate_detailed(
        stack, teacher, loaders.validation, sample_limit=None
    )
    result: dict[str, Any] = {
        "endpoint_seed": endpoint_seed,
        "validation": dict(validation),
        "validation_prediction_sha256": _tensor_sha256(
            validation_prediction.to(torch.int64)
        ),
        "test": None,
        "test_prediction_sha256": None,
    }
    if open_test:
        test, prediction = _evaluate_detailed(
            stack, teacher, loaders.test, sample_limit=None
        )
        prediction_sha = _tensor_sha256(prediction.to(torch.int64))
        if (
            test["student_correct"] != contract.historical_test_correct
            or test["examples"] != 10_000
            or prediction_sha != contract.historical_test_prediction_sha256
        ):
            raise RuntimeError(
                f"Endpoint {endpoint_seed} P0 replay changed before optimizer evaluation."
            )
        result["test"] = dict(test)
        result["test_prediction_sha256"] = prediction_sha
    del plant
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_train(request: "TrainRequest") -> int:
    spec = request.spec
    if not isinstance(spec, EndpointOptimizerRecoveryTrainSpec):
        raise TypeError("Expected the dedicated endpoint optimizer recovery spec.")
    if request.weights is None or request.teacher_weights is None:
        raise ValueError(
            "Expected --weights predecessor scientific summary and --teacher-weights."
        )
    if (
        request.base_weights is not None
        or request.resume is not None
        or request.device_data is not None
        or request.device_model is not None
    ):
        raise ValueError("Endpoint optimizer recovery rejects base/resume/device inputs.")
    protocol = spec.protocol
    summary_path = request.weights.expanduser().resolve()
    teacher_path = request.teacher_weights.expanduser().resolve()
    _load_predecessor_summary(summary_path, protocol)
    if (
        teacher_path
        != (_ROOT / protocol.expected_teacher_weights_path).resolve()
        or sha256_file(teacher_path) != protocol.expected_teacher_weights_sha256
    ):
        raise ValueError("Frozen teacher path/SHA-256 mismatch.")
    population, pulse_population, endpoints = _load_predecessor_bundle(protocol)
    input_artifacts = [
        _input("teacher_weights", teacher_path),
        _input("predecessor_scientific_summary", summary_path),
        _input(
            "target_87004_winsorized_population",
            _predecessor_run_dir(protocol)
            / protocol.predecessor.population_relative_path,
        ),
        _input(
            "target_87004_winsorized_population_receipt",
            _predecessor_run_dir(protocol)
            / protocol.predecessor.population_receipt_relative_path,
        ),
    ]
    for endpoint_seed, (path, _payload) in endpoints.items():
        input_artifacts.append(_input(f"target_87004_endpoint_{endpoint_seed}", path))
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=tuple(input_artifacts),
        resume_capability="unsupported",
    )
    try:
        if spec.student.runtime.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Matched endpoint optimizer recovery requires CUDA.")
        torch.manual_seed(spec.student.runtime.seed)
        torch.cuda.manual_seed_all(spec.student.runtime.seed)
        device = torch.device("cuda")
        loaders = build_mnist_loaders(
            spec.student.data,
            data_seed=spec.student.runtime.data_seed,
            calibration_examples=spec.student.mapping.calibration_examples,
            calibration_batch_size=spec.student.mapping.calibration_batch_size,
        )
        teacher, _teacher_metadata = _load_teacher(
            teacher_path, device=device, spec=spec.student
        )
        stack = build_student_stack(spec.student, enable_measured=False)
        stack.cost.gain = FIXED_LOGIT_GAIN
        if tuple(population.binding_shapes) != ((1568, 100), (100, 20)):
            raise RuntimeError("Saved endpoint bindings differ from the frozen DRN.")
        symmetry = intrinsic_symmetry_raw_a(pulse_population)
        train_generator_state = loaders.train_generator.get_state().clone()

        development_seed = protocol.development.endpoint_seed
        development_contract = next(
            item
            for item in protocol.predecessor.endpoints
            if item.endpoint_seed == development_seed
        )
        development_path, development_endpoint = endpoints[development_seed]
        development_p0 = _evaluate_endpoint_p0(
            endpoint=development_endpoint,
            endpoint_seed=development_seed,
            contract=development_contract,
            pulse_population=pulse_population,
            population=population,
            stack=stack,
            teacher=teacher,
            loaders=loaders,
            protocol=protocol,
            open_test=False,
        )
        development_reports: dict[str, list[Mapping[str, Any]]] = {}
        grids = {
            "open_loop_sgd": protocol.development.sgd_scale_grid,
            "tt_v1": protocol.development.tt_v1_transfer_scale_grid,
            "tt_v2": protocol.development.tt_v2_transfer_scale_grid,
        }
        selected: dict[str, Mapping[str, Any]] = {}
        for optimizer_name, grid in grids.items():
            reports = []
            for scale in grid:
                reports.append(
                    _run_arm(
                        phase="development",
                        optimizer_name=optimizer_name,
                        candidate_scale=float(scale),
                        maximum_batches=protocol.development.maximum_batches,
                        endpoint_path=development_path,
                        endpoint=development_endpoint,
                        endpoint_seed=development_seed,
                        population=population,
                        pulse_population=pulse_population,
                        symmetry=symmetry,
                        stack=stack,
                        teacher=teacher,
                        loaders=loaders,
                        train_generator_state=train_generator_state,
                        protocol=protocol,
                        store=store,
                        before_validation=development_p0["validation"],
                        before_test=None,
                    )
                )
            development_reports[optimizer_name] = reports
            selected[optimizer_name] = max(reports, key=_selection_key)
        frozen_settings = {
            "open_loop_sgd": dict(selected["open_loop_sgd"]["hyperparameters"]),
            "open_loop_adam": {
                "candidate_scale": 1.0,
                "learning_rate_raw_x": protocol.optimizers.adam_learning_rate_raw_x,
                "selection_source": (
                    "already_selected_on_endpoint_89401_in_frozen_predecessor"
                ),
            },
            "tt_v1": dict(selected["tt_v1"]["hyperparameters"]),
            "tt_v2": dict(selected["tt_v2"]["hyperparameters"]),
        }
        # No evaluation test loader is touched until every optimizer-specific
        # setting above has been frozen from development validation alone.
        evaluation_reports = []
        artifacts: list[Mapping[str, Any]] = []
        for endpoint_seed in protocol.evaluation.endpoint_seeds:
            contract = next(
                item
                for item in protocol.predecessor.endpoints
                if item.endpoint_seed == endpoint_seed
            )
            endpoint_path, endpoint = endpoints[endpoint_seed]
            p0 = _evaluate_endpoint_p0(
                endpoint=endpoint,
                endpoint_seed=endpoint_seed,
                contract=contract,
                pulse_population=pulse_population,
                population=population,
                stack=stack,
                teacher=teacher,
                loaders=loaders,
                protocol=protocol,
                open_test=True,
            )
            arms = []
            for optimizer_name in OPTIMIZER_NAMES:
                scale = float(frozen_settings[optimizer_name]["candidate_scale"])
                report = _run_arm(
                    phase="evaluation",
                    optimizer_name=optimizer_name,
                    candidate_scale=scale,
                    maximum_batches=None,
                    endpoint_path=endpoint_path,
                    endpoint=endpoint,
                    endpoint_seed=endpoint_seed,
                    population=population,
                    pulse_population=pulse_population,
                    symmetry=symmetry,
                    stack=stack,
                    teacher=teacher,
                    loaders=loaders,
                    train_generator_state=train_generator_state,
                    protocol=protocol,
                    store=store,
                    before_validation=p0["validation"],
                    before_test=p0["test"],
                )
                arms.append(report)
                artifacts.append(
                    {"path": report["checkpoint"], "kind": "optimizer_recovery_state"}
                )
            evaluation_reports.append(
                {
                    "endpoint_seed": endpoint_seed,
                    "semantics": "P&V_endpoint_realization_not_independent_array",
                    "p0": p0,
                    "arms": arms,
                }
            )
        aggregate = {}
        for optimizer_name in OPTIMIZER_NAMES:
            matching = [
                arm
                for endpoint in evaluation_reports
                for arm in endpoint["arms"]
                if arm["optimizer"] == optimizer_name
            ]
            before = [float(item["before_test"]["student_accuracy"]) for item in matching]
            after = [float(item["after_test"]["student_accuracy"]) for item in matching]
            aggregate[optimizer_name] = {
                "endpoints": len(matching),
                "mean_before_test_accuracy": sum(before) / len(before),
                "mean_after_test_accuracy": sum(after) / len(after),
                "mean_test_accuracy_delta": sum(
                    after_value - before_value
                    for before_value, after_value in zip(before, after)
                )
                / len(after),
                "minimum_after_test_accuracy": min(after),
                "maximum_after_test_accuracy": max(after),
                "total_slow_c_pulses": sum(
                    int(item["slow_c_pulse"]["applied"]) for item in matching
                ),
                "total_fast_a_pulses": sum(
                    0
                    if item["fast_a_pulse"] is None
                    else int(item["fast_a_pulse"]["applied"])
                    for item in matching
                ),
            }
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "evidence_tier": protocol.execution.evidence_tier,
            "target_assignment_seed": protocol.target_assignment_seed,
            "endpoint_semantics": protocol.evaluation.endpoint_semantics,
            "independent_population_count": 1,
            "development": {
                "endpoint_seed": development_seed,
                "role": protocol.development.role,
                "p0_validation": development_p0,
                "maximum_batches_per_candidate": (
                    protocol.development.maximum_batches
                ),
                "candidate_reports": development_reports,
                "selected_by_optimizer": {
                    name: {
                        "hyperparameters": dict(report["hyperparameters"]),
                        "validation": dict(report["after_validation"]),
                        "slow_c_pulses": report["slow_c_pulse"]["applied"],
                    }
                    for name, report in selected.items()
                },
                "adam_selection": frozen_settings["open_loop_adam"],
                "test_opened": False,
            },
            "frozen_settings_before_evaluation_test": frozen_settings,
            "evaluation": {
                "endpoint_seeds": list(protocol.evaluation.endpoint_seeds),
                "endpoint_realizations": evaluation_reports,
                "aggregate": aggregate,
                "test_used_for_selection": False,
                "validation_used_after_freeze_for_selection": False,
            },
            "intrinsic_symmetry_initialization": dict(symmetry.report),
            "optimizer_semantics": {
                "open_loop_sgd": (
                    "digital_layerwise_SGD_command_to_column_serial_Bernoulli_C_pulses"
                ),
                "open_loop_adam": (
                    "digital_Adam_to_column_serial_Bernoulli_C_pulses"
                ),
                "tt_v1": (
                    "qualified_OM_plant_TT_v1_minibatch_equation_emulator"
                ),
                "tt_v2": (
                    "qualified_unchopped_OM_plant_TTv2_minibatch_equation_emulator"
                ),
            },
            "full_conductance_forward": protocol.execution.full_conductance_forward,
            "verify_reads_during_training": 0,
            "limitations": [
                "exploratory_noncanonical_one_target_87004_device_population",
                "89402_to_89405_are_PV_endpoint_realizations_not_independent_arrays",
                "model_based_winsorized_IBM_OM_not_a_fabricated_chip",
                "digital_teacher_BPTT_Adam_moments_and_TTv2_H",
                "TT_arms_are_minibatch_equation_emulators_not_native_parallel_tiles",
                "fast_A_uses_hidden_model_oracle_intrinsic_symmetry_initialization",
                "fast_A_transfer_uses_held_apparent_write_sample_not_fresh_MVM_read_noise",
                "A_and_C_each_have_a_64_pulse_per_cell_training_cap",
                "one_epoch_recovery_only",
            ],
        }
        summary_path_out = store.run_dir / "artifacts/scientific_summary.json"
        atomic_write_json(summary_path_out, summary)
        artifacts.append({"path": str(summary_path_out), "kind": "scientific_summary"})
        terminal = {
            "evidence_tier": protocol.execution.evidence_tier,
            "target_assignment_seed": protocol.target_assignment_seed,
            "independent_population_count": 1,
            "evaluation_endpoint_count": len(protocol.evaluation.endpoint_seeds),
            "evaluation_arm_count": sum(
                len(item["arms"]) for item in evaluation_reports
            ),
            "mean_after_test_accuracy_by_optimizer": {
                name: value["mean_after_test_accuracy"]
                for name, value in aggregate.items()
            },
            "verify_reads_during_training": 0,
            "test_used_for_selection": False,
        }
        store.complete(metrics=terminal, artifacts=_artifact_records(store, artifacts))
        print(f"run_dir={store.run_dir}", flush=True)
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "_load_endpoint",
    "_load_predecessor_summary",
    "_selection_key",
    "run_train",
]
