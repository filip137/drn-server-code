"""Exploratory persistent-endpoint ensemble QAT for Winsorized IBM OM.

The clean trainable object is one normalized logical master per layer.  Each
training forward first constructs the exact deterministic device code, then
optionally replaces that code by a persistent endpoint drawn from a finite,
pulse-resolved codebook ensemble.  Apparent verify values never enter the DRN
and every physical forward uses the complete conductance ``G=2*x``.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    _atomic_save_npz,
    _evaluate_detailed,
    _input,
)
from experiments.mnist_relu_drn.ibm_om_baseline_spacing_pv import (
    apply_full_conductance_targets,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_multi_assignment_qat_config import (
    SOURCE_EXPLORATORY_RESULT_ID,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_multi_assignment_qat_runtime import (
    FrozenAssignmentBundle,
    _bundle_inputs,
    _initial_logical_masters,
    _validate_epoch0_parity,
    assignment_seed_for_minibatch,
    pre_final_assignment_seeds,
    resolve_frozen_assignment_bundle,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat import (
    PersistentEndpointCodebookEnsemble,
    build_persistent_endpoint_codebook_ensemble,
    gradient_mixture_weights,
    mix_logical_gradients,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat import (
    WinsorizedQatLayerTemplate,
    load_winsorized_qat_templates,
    logical_master_gradients,
    quantize_winsorized_logical_master,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_runtime import (
    MAXIMUM_PROGRAM_PULSES,
    NOMINAL_DELTA_X,
    VERIFY_TOLERANCE_DELTA_X_RATIO,
    _evaluate_deployment,
    _load_winsorized_population,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save


if TYPE_CHECKING:
    from ebl.cli import TrainRequest


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "ebl.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat"
SUMMARY_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat_logical_master"
)
CHECKPOINT_SCHEMA_VERSION = 1
ENSEMBLE_SCHEMA = "ebl.ibm_om.winsorized_persistent_endpoint_codebook_ensemble"
ENSEMBLE_SCHEMA_VERSION = 1


def _load_population(
    bundle: FrozenAssignmentBundle,
) -> Any:
    return _load_winsorized_population(
        bundle.population_path,
        bundle.population_receipt_path,
        expected_sha256=bundle.artifact_sha256["population"],
        expected_fingerprint=bundle.population_fingerprint,
        expected_assignment_seed=bundle.assignment_seed,
        source_joint_hardware_instance_id=bundle.source_hardware_instance_id,
    )


def _sample_indices(arm: str, global_minibatch_ordinal: int) -> tuple[int, ...]:
    """Return the frozen balanced subset of the four-sample training bank."""

    ordinal = int(global_minibatch_ordinal)
    if ordinal < 0:
        raise ValueError("Expected a nonnegative global minibatch ordinal.")
    if arm == "deterministic":
        return ()
    if arm == "mean2":
        start = ordinal % 4
        return (start, (start + 1) % 4)
    if arm == "tail4":
        return (0, 1, 2, 3)
    raise ValueError(f"Unsupported persistent-ensemble QAT arm: {arm!r}.")


def _objective_from_losses(
    arm: str,
    *,
    ideal_loss: float,
    persistent_losses: Sequence[float],
) -> float:
    """Evaluate the same declared scalar mixture used for the STE gradient."""

    mixture = gradient_mixture_weights(
        arm,
        persistent_losses,
        ideal_loss=ideal_loss,
    )
    objective = mixture.ideal_weight * float(ideal_loss) + sum(
        weight * float(loss)
        for weight, loss in zip(
            mixture.persistent_weights, persistent_losses, strict=True
        )
    )
    if not math.isfinite(objective):
        raise FloatingPointError("Persistent-ensemble objective became non-finite.")
    return objective


def _apply_full_g(stack: Any, values: Sequence[torch.Tensor]) -> None:
    apply_full_conductance_targets(
        stack.bundle.catalog,
        tuple(values),
        conductance_min=0.0,
        conductance_max=2.0,
    )


def _one_forward_gradient(
    *,
    stack: Any,
    inputs: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
    full_conductance: Sequence[torch.Tensor],
) -> tuple[float, tuple[torch.Tensor, ...], Mapping[str, float | int]]:
    """Evaluate one physical state and return its identity-STE gradient."""

    _apply_full_g(stack, full_conductance)
    stack.network.set_input(inputs, reset=True)
    stack.minimizer.compute_equilibrium()
    stack.cost.set_teacher(teacher_logits, labels)
    with torch.no_grad():
        per_example_kl = stack.cost.eval()
        prediction = stack.cost.student_logits().argmax(dim=1)
        teacher_prediction = teacher_logits.argmax(dim=1)
        loss = float(per_example_kl.to(torch.float64).mean().item())
        metrics: Mapping[str, float | int] = {
            "examples": int(labels.numel()),
            "kl_teacher_student": loss,
            "correct": int(prediction.eq(labels).sum().item()),
            "teacher_agreement": int(
                prediction.eq(teacher_prediction).sum().item()
            ),
        }
    physical = tuple(stack.differentiator.compute_gradient())
    if any(not bool(torch.isfinite(value).all()) for value in physical):
        raise FloatingPointError("A physical ensemble-QAT gradient is non-finite.")
    logical = logical_master_gradients(masters, physical, templates)
    return loss, logical, metrics


def _train_epoch(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.nn.Parameter],
    templates_by_assignment: Mapping[int, Sequence[WinsorizedQatLayerTemplate]],
    ensembles_by_assignment: Mapping[int, PersistentEndpointCodebookEnsemble],
    assignment_cycle: Sequence[int],
    optimizer: torch.optim.Optimizer,
    arm: str,
    global_minibatch_ordinal: int,
    maximum_batches: int | None,
) -> tuple[Mapping[str, Any], int]:
    totals: dict[int, dict[str, Any]] = {
        int(seed): {
            "examples": 0,
            "batches": 0,
            "objective_kl_sum": 0.0,
            "ideal_kl_sum": 0.0,
            "ideal_correct": 0,
            "persistent_kl_sum": 0.0,
            "persistent_correct": 0,
            "persistent_views": 0,
            "worst_persistent_kl_sum": 0.0,
        }
        for seed in assignment_cycle
    }
    gradient_squared = [0.0, 0.0]
    gradient_values = [0, 0]
    start_ordinal = int(global_minibatch_ordinal)
    for inputs, labels in limited(loader, maximum_batches):
        assignment_seed = assignment_seed_for_minibatch(
            assignment_cycle, global_minibatch_ordinal
        )
        templates = tuple(templates_by_assignment[assignment_seed])
        inputs = inputs.to(stack.device, dtype=torch.float32)
        labels = labels.to(stack.device, dtype=torch.long)
        with torch.no_grad():
            teacher_logits = teacher.logits(inputs)
            ideal_view = quantize_winsorized_logical_master(
                masters, templates, include_report=False
            )

        ideal_loss, ideal_gradients, ideal_metrics = _one_forward_gradient(
            stack=stack,
            inputs=inputs,
            teacher_logits=teacher_logits,
            labels=labels,
            masters=masters,
            templates=templates,
            full_conductance=ideal_view.full_conductance_targets,
        )
        sample_indices = _sample_indices(arm, global_minibatch_ordinal)
        persistent_losses: list[float] = []
        persistent_gradients: list[tuple[torch.Tensor, ...]] = []
        persistent_metrics: list[Mapping[str, float | int]] = []
        if sample_indices:
            ensemble = ensembles_by_assignment[assignment_seed]
            views = ensemble.full_conductance_views(
                ideal_view.requested_index,
                sample_indices=sample_indices,
            )
            for full_conductance in views:
                loss, gradients, metrics = _one_forward_gradient(
                    stack=stack,
                    inputs=inputs,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    masters=masters,
                    templates=templates,
                    full_conductance=full_conductance,
                )
                persistent_losses.append(loss)
                persistent_gradients.append(gradients)
                persistent_metrics.append(metrics)
        mixed = mix_logical_gradients(
            arm,
            persistent_losses,
            persistent_gradients,
            ideal_loss=ideal_loss,
            ideal_gradients=ideal_gradients,
        )
        objective = _objective_from_losses(
            arm,
            ideal_loss=ideal_loss,
            persistent_losses=persistent_losses,
        )
        optimizer.zero_grad(set_to_none=True)
        for layer, (master, gradient) in enumerate(zip(masters, mixed)):
            master.grad = gradient
            gradient_squared[layer] += float(
                gradient.detach().to(torch.float64).square().sum().item()
            )
            gradient_values[layer] += int(gradient.numel())
        optimizer.step()
        with torch.no_grad():
            for master in masters:
                master.clamp_(-1.0, 1.0)

        count = int(labels.numel())
        selected = totals[assignment_seed]
        selected["examples"] += count
        selected["batches"] += 1
        selected["objective_kl_sum"] += objective * count
        selected["ideal_kl_sum"] += ideal_loss * count
        selected["ideal_correct"] += int(ideal_metrics["correct"])
        if persistent_metrics:
            selected["persistent_views"] += len(persistent_metrics)
            selected["persistent_kl_sum"] += sum(persistent_losses) * count
            selected["persistent_correct"] += sum(
                int(item["correct"]) for item in persistent_metrics
            )
            selected["worst_persistent_kl_sum"] += max(persistent_losses) * count
        global_minibatch_ordinal += 1

    if global_minibatch_ordinal == start_ordinal:
        raise RuntimeError("Expected persistent-ensemble QAT to process a batch.")
    by_assignment: dict[str, Mapping[str, Any]] = {}
    for assignment_seed in assignment_cycle:
        selected = totals[int(assignment_seed)]
        examples = int(selected["examples"])
        batches = int(selected["batches"])
        if examples < 1 or batches < 1:
            raise RuntimeError("Every training assignment must receive minibatches.")
        views = int(selected["persistent_views"])
        by_assignment[str(assignment_seed)] = {
            "examples": examples,
            "batches": batches,
            "objective_kl_teacher_student": selected["objective_kl_sum"] / examples,
            "ideal_kl_teacher_student": selected["ideal_kl_sum"] / examples,
            "ideal_student_accuracy": selected["ideal_correct"] / examples,
            "persistent_views_per_batch": views / batches,
            "persistent_mean_kl_teacher_student": (
                selected["persistent_kl_sum"] / (examples * views / batches)
                if views
                else None
            ),
            "persistent_mean_student_accuracy": (
                selected["persistent_correct"] / (examples * views / batches)
                if views
                else None
            ),
            "persistent_batch_worst_kl_teacher_student": (
                selected["worst_persistent_kl_sum"] / examples if views else None
            ),
        }
    return {
        "arm": arm,
        "global_minibatch_ordinal_start": start_ordinal,
        "global_minibatch_ordinal_end": global_minibatch_ordinal,
        "assignment_cycle": list(map(int, assignment_cycle)),
        "by_assignment": by_assignment,
        "logical_gradient_rms": [
            math.sqrt(total / count)
            for total, count in zip(gradient_squared, gradient_values)
        ],
    }, global_minibatch_ordinal


def _evaluate_ensemble(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
    ensemble: PersistentEndpointCodebookEnsemble,
) -> Mapping[str, Any]:
    view = quantize_winsorized_logical_master(masters, templates)
    _apply_full_g(stack, view.full_conductance_targets)
    ideal, _ = _evaluate_detailed(stack, teacher, loader, sample_limit=None)
    repeats = []
    full_views = ensemble.full_conductance_views(view.requested_index)
    for endpoint_seed, full_conductance in zip(
        ensemble.endpoint_seeds, full_views
    ):
        _apply_full_g(stack, full_conductance)
        metrics, _ = _evaluate_detailed(
            stack, teacher, loader, sample_limit=None
        )
        repeats.append({"endpoint_seed": int(endpoint_seed), "metrics": metrics})
    accuracies = [float(item["metrics"]["student_accuracy"]) for item in repeats]
    divergences = [
        float(item["metrics"]["kl_teacher_student"]) for item in repeats
    ]
    return {
        "assignment_seed": int(ensemble.assignment_seed),
        "population_fingerprint": ensemble.population_fingerprint,
        "ideal": ideal,
        "persistent_repeats": repeats,
        "persistent_mean_accuracy": fmean(accuracies),
        "persistent_minimum_accuracy": min(accuracies),
        "persistent_population_std_accuracy": pstdev(accuracies),
        "persistent_mean_kl_teacher_student": fmean(divergences),
        "persistent_maximum_kl_teacher_student": max(divergences),
        "apparent_endpoint_applied_to_drn": False,
        "persistent_projection_count": 0,
    }


def _development_key(report: Mapping[str, Any], epoch: int) -> tuple[float, float, float, int]:
    return (
        float(report["persistent_mean_accuracy"]),
        float(report["persistent_minimum_accuracy"]),
        -float(report["persistent_mean_kl_teacher_student"]),
        -int(epoch),
    )


def _save_ensemble(
    path: Path,
    ensemble: PersistentEndpointCodebookEnsemble,
) -> Path:
    _atomic_save_npz(
        path,
        {
            "schema": np.asarray(ENSEMBLE_SCHEMA),
            "schema_version": np.asarray(ENSEMBLE_SCHEMA_VERSION, dtype=np.int64),
            "assignment_seed": np.asarray(ensemble.assignment_seed, dtype=np.int64),
            "population_fingerprint": np.asarray(ensemble.population_fingerprint),
            "binding_shapes_json": np.asarray(json.dumps(ensemble.binding_shapes)),
            "endpoint_seeds": np.asarray(ensemble.endpoint_seeds, dtype=np.int64),
            "spacing_raw_x": np.asarray(ensemble.spacing_raw_x, dtype=np.float64),
            "tolerance_raw_x": np.asarray(ensemble.tolerance_raw_x, dtype=np.float64),
            "maximum_program_pulses": np.asarray(
                ensemble.maximum_program_pulses, dtype=np.int64
            ),
            "baseline_raw_x": ensemble.baseline_raw_x.detach().cpu().numpy(),
            "maximum_index": ensemble.maximum_index.detach().cpu().numpy(),
            "persistent_raw_x": ensemble.persistent_raw_x.detach().cpu().numpy(),
        },
    )
    return path


def _ensemble_report(ensemble: PersistentEndpointCodebookEnsemble) -> Mapping[str, Any]:
    values = ensemble.persistent_raw_x.detach().cpu().to(torch.float64)
    finite = torch.isfinite(values)
    selected = values[finite]
    return {
        "assignment_seed": int(ensemble.assignment_seed),
        "population_fingerprint": ensemble.population_fingerprint,
        "endpoint_seeds": list(map(int, ensemble.endpoint_seeds)),
        "samples": len(ensemble.endpoint_seeds),
        "levels": int(values.shape[-1]),
        "cells": int(values.shape[1]),
        "valid_entries": int(finite.sum().item()),
        "invalid_entries_are_nan_and_unselectable": int((~finite).sum().item()),
        "persistent_raw_x_minimum": float(selected.min().item()),
        "persistent_raw_x_maximum": float(selected.max().item()),
        "full_conductance_formula": "G=2*x_persistent",
        "apparent_endpoint_applied_to_drn": False,
        "endpoint_projection_count": 0,
        "controller_build": ensemble.build_report,
    }


def _checkpoint_payload(
    *,
    epoch: int,
    arm: str,
    masters: Sequence[torch.Tensor],
    source_absmax: Sequence[float],
    optimizer: torch.optim.Optimizer,
    development: Mapping[str, Any],
    protocol: Any,
    teacher_sha256: str,
    global_minibatch_ordinal: int,
) -> Mapping[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "evidence_tier": "exploratory_noncanonical",
        "arm": arm,
        "epoch": int(epoch),
        "teacher_sha256": teacher_sha256,
        "spacing_delta_x_multiplier": int(
            protocol.mapping.spacing_delta_x_multiplier
        ),
        "fixed_logit_gain": float(protocol.mapping.fixed_logit_gain),
        "source_absmax": list(map(float, source_absmax)),
        "logical_master": tuple(
            value.detach().cpu().to(torch.float32).clone() for value in masters
        ),
        "logical_master_sha256": tuple(_tensor_sha256(value) for value in masters),
        "optimizer_state": optimizer.state_dict(),
        "development": dict(development),
        "global_minibatch_ordinal": int(global_minibatch_ordinal),
    }


def run_train(request: "TrainRequest") -> int:
    from experiments.mnist_relu_drn.ibm_om_winsorized_pv_ensemble_qat_config import (
        WinsorizedPvEnsembleQatTrainSpec,
    )

    spec = request.spec
    if not isinstance(spec, WinsorizedPvEnsembleQatTrainSpec):
        raise TypeError("Expected the dedicated persistent-ensemble QAT train spec.")
    if request.teacher_weights is None:
        raise ValueError("Expected --teacher-weights for persistent-ensemble QAT.")
    for name in ("weights", "base_weights", "resume", "device_data", "device_model"):
        if getattr(request, name) is not None:
            raise ValueError(
                "Persistent-ensemble QAT accepts only --teacher-weights; "
                f"unexpected {name}."
            )
    protocol = spec.protocol
    student = spec.student
    teacher_path = request.teacher_weights.expanduser().resolve()
    expected_teacher_sha = protocol.source["expected_teacher_weights_sha256"]
    if sha256_file(teacher_path) != expected_teacher_sha:
        raise ValueError("Frozen teacher/source checkpoint hash mismatch.")

    bundles = {
        contract.assignment_seed: resolve_frozen_assignment_bundle(
            contract,
            source_exploratory_result_id=SOURCE_EXPLORATORY_RESULT_ID,
            spacing_delta_x_multiplier=protocol.mapping.spacing_delta_x_multiplier,
        )
        for contract in protocol.predecessors
    }
    expected_assignments = {
        *protocol.assignments.training_assignment_seeds,
        protocol.assignments.development_assignment_seed,
        protocol.assignments.final_assignment_seed,
    }
    if set(bundles) != expected_assignments or len(bundles) != 4:
        raise RuntimeError("Expected four distinct frozen assignment roles.")
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("teacher_weights", teacher_path),
            *(
                item
                for seed in sorted(bundles)
                for item in _bundle_inputs(bundles[seed])
            ),
        ),
        resume_capability="unsupported",
    )
    try:
        torch.manual_seed(student.runtime.seed)
        torch.cuda.manual_seed_all(student.runtime.seed)
        device = torch.device(student.runtime.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Exploratory persistent-ensemble QAT requires CUDA.")
        loaders = build_mnist_loaders(
            student.data,
            data_seed=student.runtime.data_seed,
            calibration_examples=student.mapping.calibration_examples,
            calibration_batch_size=student.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            teacher_path, device=device, spec=student
        )
        stack = build_student_stack(student, enable_measured=False)
        stack.cost.gain = float(protocol.mapping.fixed_logit_gain)
        spacing = int(protocol.mapping.spacing_delta_x_multiplier) * NOMINAL_DELTA_X
        pre_final = pre_final_assignment_seeds(
            protocol.assignments.training_assignment_seeds,
            protocol.assignments.development_assignment_seed,
            protocol.assignments.final_assignment_seed,
        )
        templates_cpu = {
            seed: load_winsorized_qat_templates(
                bundles[seed].mapping_path,
                level_spacing_raw_x=spacing,
            )
            for seed in pre_final
        }
        templates = {
            seed: tuple(template.to(device) for template in values)
            for seed, values in templates_cpu.items()
        }
        masters, source_absmax = _initial_logical_masters(
            teacher, templates_cpu, protocol, device=device
        )
        initial_master = tuple(value.detach().clone() for value in masters)
        epoch0_parity = {
            str(seed): list(
                _validate_epoch0_parity(
                    masters, templates[seed], bundles[seed].mapping_path
                )
            )
            for seed in sorted(pre_final)
        }

        arm = str(protocol.training.arm)
        table_assignments = {protocol.assignments.development_assignment_seed}
        if arm != "deterministic":
            table_assignments.update(protocol.assignments.training_assignment_seeds)
        ensembles: dict[int, PersistentEndpointCodebookEnsemble] = {}
        ensemble_artifacts: list[Path] = []
        ensemble_reports: dict[str, Mapping[str, Any]] = {}
        for assignment_seed in sorted(table_assignments):
            population = _load_population(bundles[assignment_seed])
            endpoint_seeds = (
                protocol.assignments.development_endpoint_seeds
                if assignment_seed == protocol.assignments.development_assignment_seed
                else protocol.assignments.training_endpoint_seeds
            )
            print(
                "building exact persistent endpoint codebook ensemble "
                f"arm={arm} assignment={assignment_seed} seeds={list(endpoint_seeds)}",
                flush=True,
            )
            ensemble_cpu = build_persistent_endpoint_codebook_ensemble(
                population,
                templates_cpu[assignment_seed],
                endpoint_seeds=endpoint_seeds,
                tolerance_raw_x=(
                    VERIFY_TOLERANCE_DELTA_X_RATIO * NOMINAL_DELTA_X
                ),
                maximum_program_pulses=MAXIMUM_PROGRAM_PULSES,
                device=device,
            )
            artifact = _save_ensemble(
                store.run_dir
                / "artifacts/endpoint_ensembles"
                / f"assignment_{assignment_seed}.npz",
                ensemble_cpu,
            )
            ensemble_artifacts.append(artifact)
            ensemble_reports[str(assignment_seed)] = _ensemble_report(ensemble_cpu)
            ensembles[assignment_seed] = ensemble_cpu.to(device)
            print(
                "prepared exact persistent endpoint codebook ensemble "
                f"assignment={assignment_seed}",
                flush=True,
            )

        optimizer = torch.optim.SGD(
            [
                {"params": [master], "lr": float(rate)}
                for master, rate in zip(masters, protocol.training.learning_rates)
            ],
            momentum=float(protocol.training.momentum),
            weight_decay=float(protocol.training.weight_decay),
        )
        development_seed = protocol.assignments.development_assignment_seed
        initial_development = _evaluate_ensemble(
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            masters=masters,
            templates=templates[development_seed],
            ensemble=ensembles[development_seed],
        )
        store.append_metric(
            {
                "mode": "train",
                "arm": arm,
                "epoch": 0,
                "development": initial_development,
            }
        )
        best_epoch = 0
        best_development = initial_development
        history = []
        global_ordinal = 0
        assignment_cycle = protocol.assignments.training_assignment_seeds
        for epoch in range(1, int(protocol.training.epochs) + 1):
            train_report, global_ordinal = _train_epoch(
                stack=stack,
                teacher=teacher,
                loader=loaders.train,
                masters=masters,
                templates_by_assignment=templates,
                ensembles_by_assignment=ensembles,
                assignment_cycle=assignment_cycle,
                optimizer=optimizer,
                arm=arm,
                global_minibatch_ordinal=global_ordinal,
                maximum_batches=student.settings.max_batches,
            )
            development = _evaluate_ensemble(
                stack=stack,
                teacher=teacher,
                loader=loaders.validation,
                masters=masters,
                templates=templates[development_seed],
                ensemble=ensembles[development_seed],
            )
            record = {
                "mode": "train",
                "arm": arm,
                "epoch": epoch,
                "train": train_report,
                "development": development,
            }
            history.append(record)
            store.append_metric(record)
            if _development_key(development, epoch) > _development_key(
                best_development, best_epoch
            ):
                best_epoch = epoch
                best_development = development
            print(
                "persistent-ensemble QAT "
                f"arm={arm} epoch={epoch}/{protocol.training.epochs} "
                f"dev_pv_mean={100*development['persistent_mean_accuracy']:.2f}% "
                f"dev_pv_min={100*development['persistent_minimum_accuracy']:.2f}%",
                flush=True,
            )

        final_development = history[-1]["development"]
        checkpoint_payload = _checkpoint_payload(
            epoch=int(protocol.training.epochs),
            arm=arm,
            masters=masters,
            source_absmax=source_absmax,
            optimizer=optimizer,
            development=final_development,
            protocol=protocol,
            teacher_sha256=sha256_file(teacher_path),
            global_minibatch_ordinal=global_ordinal,
        )
        checkpoint_path = atomic_torch_save(
            checkpoint_payload,
            store.run_dir
            / "checkpoints"
            / f"epoch_{protocol.training.epochs}_{arm}_logical_master.pt",
        )

        # Final hardware is not loaded or evaluated until every training and
        # development action, including diagnostic best-epoch selection, ends.
        final_seed = protocol.assignments.final_assignment_seed
        final_templates_cpu = load_winsorized_qat_templates(
            bundles[final_seed].mapping_path,
            level_spacing_raw_x=spacing,
        )
        final_templates = tuple(value.to(device) for value in final_templates_cpu)
        epoch0_parity[str(final_seed)] = list(
            _validate_epoch0_parity(
                initial_master, final_templates, bundles[final_seed].mapping_path
            )
        )
        final_population = _load_population(bundles[final_seed])
        final_evaluations = {
            "epoch_0": _evaluate_deployment(
                role="previously_inspected_final_assignment_87003_exploratory",
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                masters=initial_master,
                templates=final_templates,
                population=final_population,
                endpoint_seeds=protocol.assignments.final_endpoint_seeds,
                device=device,
            ),
            "fixed_final_epoch": _evaluate_deployment(
                role="previously_inspected_final_assignment_87003_exploratory",
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                masters=masters,
                templates=final_templates,
                population=final_population,
                endpoint_seeds=protocol.assignments.final_endpoint_seeds,
                device=device,
            ),
        }
        summary = {
            "schema": SUMMARY_SCHEMA,
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "complete",
            "evidence_tier": "exploratory_noncanonical",
            "claim_boundary": (
                "Finite exact persistent endpoint-codebook ensemble QAT on the "
                "analyst-Winsorized AIHWKit OM model. Endpoint tables are pulse-"
                "resolved counterfactual codebooks, not measured-device traces or "
                "fresh resampling each minibatch. Full persistent G=2*x is used; "
                "apparent verify is never deployed. Assignment 87003 was already "
                "inspected historically, so this is a mechanism screen only."
            ),
            "arm": arm,
            "loss": {
                "ideal_weight": float(protocol.training.ideal_loss_weight),
                "mean_persistent_weight": float(
                    protocol.training.mean_pv_loss_weight
                ),
                "batch_worst_persistent_weight": float(
                    protocol.training.tail_pv_loss_weight
                ),
                "persistent_samples_per_minibatch": int(
                    protocol.training.persistent_samples_per_minibatch
                ),
            },
            "teacher": teacher_metadata,
            "source_revision": store.manifest.get("source", {}),
            "assignments": {
                "training": list(map(int, assignment_cycle)),
                "development": int(development_seed),
                "final": int(final_seed),
                "training_endpoint_seeds": list(
                    map(int, protocol.assignments.training_endpoint_seeds)
                ),
                "development_endpoint_seeds": list(
                    map(int, protocol.assignments.development_endpoint_seeds)
                ),
                "final_endpoint_seeds": list(
                    map(int, protocol.assignments.final_endpoint_seeds)
                ),
            },
            "epoch0_mapping_parity_sha256": epoch0_parity,
            "endpoint_ensembles": ensemble_reports,
            "training": {
                "epochs": int(protocol.training.epochs),
                "fixed_headline_epoch": int(protocol.training.epochs),
                "best_development_epoch_diagnostic": int(best_epoch),
                "initial_development": initial_development,
                "final_development": final_development,
                "history": history,
            },
            "final_87003": final_evaluations,
        }
        summary_path = store.run_dir / "artifacts/scientific_summary.json"
        atomic_write_json(summary_path, summary)
        epoch0_final = final_evaluations["epoch_0"]
        trained_final = final_evaluations["fixed_final_epoch"]
        terminal = {
            "evidence_tier": "exploratory_noncanonical",
            "arm": arm,
            "epochs": int(protocol.training.epochs),
            "development_epoch0_persistent_mean_accuracy": float(
                initial_development["persistent_mean_accuracy"]
            ),
            "development_final_persistent_mean_accuracy": float(
                final_development["persistent_mean_accuracy"]
            ),
            "final_87003_epoch0_ideal_accuracy": float(
                epoch0_final["ideal_quantized"]["student_accuracy"]
            ),
            "final_87003_epoch0_pv_mean_accuracy": float(
                epoch0_final["persistent_mean_accuracy"]
            ),
            "final_87003_trained_ideal_accuracy": float(
                trained_final["ideal_quantized"]["student_accuracy"]
            ),
            "final_87003_trained_pv_mean_accuracy": float(
                trained_final["persistent_mean_accuracy"]
            ),
        }
        store.append_metric({"mode": "train_terminal", **terminal})
        artifacts = [
            store.artifact_record(checkpoint_path, kind="logical_master"),
            store.artifact_record(summary_path, kind="scientific_summary"),
        ]
        artifacts.extend(
            store.artifact_record(path, kind="persistent_endpoint_codebook_ensemble")
            for path in ensemble_artifacts
        )
        store.complete(metrics=terminal, artifacts=tuple(artifacts))
        del final_population
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "_development_key",
    "_evaluate_ensemble",
    "_objective_from_losses",
    "_sample_indices",
    "_train_epoch",
    "run_train",
]
