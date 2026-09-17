"""Matched RESET-only / hidden-SET Figure-6 and IBM-OM screen.

This exploratory runner reuses the completed 784-256-10 ladder's teacher,
seeds, OM identities, write streams, solver, gain, and HWA learning rates.  It
changes the target-information contract: per-cell RESET endpoints are known,
while true per-cell SET endpoints remain plant-only state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.figure6_om_256_ladder import (
    _apply_full_g,
    _augmented_hwa_bank,
    _evaluate_plant,
    _field,
    _initial_masters,
    _loaders,
    _master_report,
    _one_forward_gradient,
    _require_cuda,
    _runtime,
    _snapshot,
    _tensor_sha256,
    _tensor_summary,
)
from experiments.mnist_relu_drn.figure6_om_256_ladder_config import (
    load_ladder_config,
)
from experiments.mnist_relu_drn.figure6_om_pulse import (
    Figure6OmPulsePlant,
    validate_paired_om_populations,
)
from experiments.mnist_relu_drn.figure6_om_reset_only import (
    ResetOnlyMapping,
    apparent_conductance_unprojected,
    lift_reset_only_gradients,
    map_masters_from_reset_only,
    mapping_contract,
    program_conductance_with_verify,
)
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _evaluate_detailed,
)
from experiments.mnist_shared import limited
from training.checkpoint import atomic_torch_save
from training.ibm_reram_hwa import load_om_array_population


EXPERIMENT_ID = "mnist_figure6_om_784_256_10_reset_only_shared_quad.v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT
    / "examples/mnist_relu_drn/figure6_om_784_256_10_reset_only/"
    "shared_quad_nominal_set.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "simulation_results/"
    "exploratory_noncanonical_figure6_om_784_256_10_reset_only"
)


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Expected readable JSON at {path}.") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _resolve_input(record: Mapping[str, Any], *, label: str) -> Path:
    path_value = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str):
        raise ValueError(f"Expected path and SHA-256 for {label}.")
    path = (ROOT / path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen {label}: {path}.")
    observed = sha256_file(path)
    if observed != digest:
        raise ValueError(
            f"Frozen {label} digest changed: expected={digest}, observed={observed}."
        )
    return path


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("evidence_tier") != "exploratory_noncanonical"
    ):
        raise ValueError("RESET-only screen config identity changed.")
    mapping = config.get("mapping")
    hwa = config.get("hwa")
    pv = config.get("program_verify")
    replication = config.get("replication")
    evaluation = config.get("evaluation")
    if not all(
        isinstance(value, Mapping)
        for value in (mapping, hwa, pv, replication, evaluation)
    ):
        raise ValueError("RESET-only config sections must be objects.")
    assert isinstance(mapping, Mapping)
    assert isinstance(hwa, Mapping)
    assert isinstance(pv, Mapping)
    assert isinstance(replication, Mapping)
    assert isinstance(evaluation, Mapping)
    expected = {
        "known_per_cell": "exact_fixed_Figure6_RESET_endpoint",
        "hidden_plant_only": "true_fixed_Figure6_SET_endpoint",
        "baseline": "maximum_known_RESET_shared_by_all_four_cells_in_each_logical_quad",
        "nominal_set": 2.0,
        "scale_fractions": [1.0, 1.0],
        "true_set_consumed_by_target_mapper": False,
    }
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise ValueError(f"RESET-only mapping field {key!r} changed.")
    if (
        hwa.get("learning_rates")
        != [0.0014706382978723403, 0.00011974808510638298]
        or hwa.get("epochs") != 10
        or hwa.get("gradient_lift")
        != "nominal_span_STE_without_true_SET_or_saturation_mask"
        or pv.get("target_coordinate") != "absolute_conductance_G"
        or pv.get("verify_coordinate")
        != "literal_held_apparent_absolute_conductance_G"
        or pv.get("true_set_used_to_normalize_verify") is not False
        or pv.get("tolerance_conductance") != 0.0450775
        or pv.get("maximum_pulses_per_cell") != 128
        or replication.get("endpoint_seeds") != [104001, 104002, 104003]
        or replication.get("om_assignment_seeds") != [104101, 104102, 104103]
        or replication.get("arrays") != 3
        or replication.get("writes_per_array") != 3
        or evaluation.get("primary") != "current_held_apparent_state"
        or evaluation.get("fixed_logit_gain") != 3.1622776601683795
    ):
        raise ValueError("RESET-only frozen execution contract changed.")


def _inputs(config_path: Path, config: Mapping[str, Any]) -> Mapping[str, Path]:
    frozen = config.get("frozen_inputs")
    if not isinstance(frozen, Mapping):
        raise ValueError("Expected frozen input records.")
    result = {
        "config": config_path,
        "base_protocol_config": _resolve_input(
            frozen["base_protocol_config"], label="base protocol config"
        ),
        "teacher_checkpoint": _resolve_input(
            frozen["teacher_checkpoint"], label="teacher checkpoint"
        ),
        "oracle_endpoint_summary": _resolve_input(
            frozen["oracle_endpoint_summary"], label="oracle endpoint summary"
        ),
        "oracle_hwa_rate_receipt": _resolve_input(
            frozen["oracle_hwa_rate_receipt"], label="oracle HWA rate receipt"
        ),
    }
    populations = frozen.get("heldout_om_populations")
    if not isinstance(populations, list) or len(populations) != 3:
        raise ValueError("Expected exactly three held-out OM population pairs.")
    for expected_index, record in enumerate(populations, start=1):
        if not isinstance(record, Mapping) or record.get("array_index") != expected_index:
            raise ValueError("Held-out OM population ordering changed.")
        for policy in ("published", "repaired"):
            selected = {
                "path": record.get(f"{policy}_path"),
                "sha256": record.get(f"{policy}_sha256"),
            }
            result[f"array_{expected_index}_{policy}"] = _resolve_input(
                selected, label=f"array {expected_index} {policy} OM population"
            )
    return result


def _rmse(value: torch.Tensor) -> float:
    return float(value.detach().to(torch.float64).square().mean().sqrt().item())


def _maximum_absolute(value: torch.Tensor) -> float:
    return float(value.detach().abs().max().item())


def _fraction(mask: torch.Tensor, selected: torch.Tensor) -> float | None:
    count = int(mask.sum().item())
    if count == 0:
        return None
    return float(selected[mask].to(torch.float64).mean().item())


def _physical_report(
    full_g: Sequence[torch.Tensor],
    requested_g: Sequence[torch.Tensor],
    masters: Sequence[torch.Tensor],
    field: Any,
) -> Mapping[str, Any]:
    layers = []
    for layer_index, (actual, requested, master, layout) in enumerate(
        zip(full_g, requested_g, masters, field.layouts, strict=True)
    ):
        actual_quad = quad_stack(actual, layout=layout)
        requested_quad = quad_stack(requested, layout=layout)
        requested_diagonal_residual = requested_quad[..., 0] - requested_quad[..., 3]
        requested_off_diagonal_residual = (
            requested_quad[..., 1] - requested_quad[..., 2]
        )
        if (
            bool(torch.any(requested_diagonal_residual != 0.0))
            or bool(torch.any(requested_off_diagonal_residual != 0.0))
        ):
            raise RuntimeError("RESET-only requested physical pair equality broke.")
        actual_diagonal_residual = actual_quad[..., 0] - actual_quad[..., 3]
        actual_off_diagonal_residual = actual_quad[..., 1] - actual_quad[..., 2]
        requested_contrast = (
            requested_quad[..., 0]
            - requested_quad[..., 1]
            - requested_quad[..., 2]
            + requested_quad[..., 3]
        ) / 2.0
        actual_contrast = (
            actual_quad[..., 0]
            - actual_quad[..., 1]
            - actual_quad[..., 2]
            + actual_quad[..., 3]
        ) / 2.0
        nonzero = master != 0.0
        layers.append(
            {
                "layer": layer_index,
                "layout": layout,
                "full_conductance": _tensor_summary(actual),
                "requested_conductance": _tensor_summary(requested),
                "requested_Gpp_minus_Gmm_maximum_absolute": _maximum_absolute(
                    requested_diagonal_residual
                ),
                "requested_Gpm_minus_Gmp_maximum_absolute": _maximum_absolute(
                    requested_off_diagonal_residual
                ),
                "realized_Gpp_minus_Gmm_rms": _rmse(actual_diagonal_residual),
                "realized_Gpp_minus_Gmm_maximum_absolute": _maximum_absolute(
                    actual_diagonal_residual
                ),
                "realized_Gpm_minus_Gmp_rms": _rmse(
                    actual_off_diagonal_residual
                ),
                "realized_Gpm_minus_Gmp_maximum_absolute": _maximum_absolute(
                    actual_off_diagonal_residual
                ),
                "requested_signed_contrast": _tensor_summary(requested_contrast),
                "realized_signed_contrast": _tensor_summary(actual_contrast),
                "signed_contrast_rmse_to_request": _rmse(
                    actual_contrast - requested_contrast
                ),
                "signed_contrast_sign_flips_nonzero": int(
                    (
                        nonzero
                        & (torch.sign(actual_contrast) != torch.sign(master))
                    )
                    .sum()
                    .item()
                ),
                "quad_loading": _tensor_summary(actual_quad.sum(dim=-1)),
            }
        )
    return {"layers": layers}


def _mapping_report(
    mapping: ResetOnlyMapping,
    masters: Sequence[torch.Tensor],
    field: Any,
) -> Mapping[str, Any]:
    layers = []
    for layer_index, (
        requested,
        realized,
        baseline,
        span,
        unreachable,
        true_set,
        shape,
        layout,
    ) in enumerate(
        zip(
            mapping.requested_g,
            mapping.plant_limited_g,
            mapping.baselines,
            mapping.nominal_spans,
            mapping.requested_above_true_set,
            field.set,
            field.shapes,
            field.layouts,
            strict=True,
        )
    ):
        baseline_full = scatter_quads(
            baseline.unsqueeze(-1).expand(*baseline.shape, 4),
            shape=shape,
            layout=layout,
        )
        baseline_unreachable = baseline_full > true_set
        layers.append(
            {
                "layer": layer_index,
                "known_RESET_shared_baseline": _tensor_summary(baseline),
                "nominal_span": _tensor_summary(span),
                "requested_target": _tensor_summary(requested),
                "plant_limited_target": _tensor_summary(realized),
                "requested_above_hidden_true_SET_cells": int(
                    unreachable.sum().item()
                ),
                "requested_above_hidden_true_SET_fraction": float(
                    unreachable.to(torch.float64).mean().item()
                ),
                "shared_baseline_above_hidden_true_SET_cells": int(
                    baseline_unreachable.sum().item()
                ),
                "requested_target_sha256": _tensor_sha256(requested),
                "hidden_true_SET_sha256_analysis_only": _tensor_sha256(true_set),
            }
        )
    return {
        "information_boundary": dict(mapping_contract()),
        "true_SET_used_only_for_plant_and_analysis": True,
        "layers": layers,
        "plant_limited_physical": _physical_report(
            mapping.plant_limited_g, mapping.requested_g, masters, field
        ),
    }


def _programming_report(
    *,
    mapping: ResetOnlyMapping,
    plant: Figure6OmPulsePlant,
    result: Any,
    masters: Sequence[torch.Tensor],
    tolerance: float,
) -> Mapping[str, Any]:
    target = torch.cat([value.reshape(-1) for value in mapping.requested_g])
    reset = torch.cat([value.reshape(-1) for value in plant.field.reset])
    true_set = torch.cat([value.reshape(-1) for value in plant.field.set])
    persistent = torch.cat([value.reshape(-1) for value in plant.full_conductance])
    apparent_literal = apparent_conductance_unprojected(plant)
    apparent_forward = torch.cat(
        [value.reshape(-1) for value in plant.apparent_full_conductance]
    )
    unreachable = target > true_set
    preaccepted = (target == reset) & ~plant.corrupt
    persistent_within = (persistent - target).abs() <= tolerance
    apparent_within = (apparent_literal - target).abs() <= tolerance
    if not torch.equal(result.apparent_endpoint, apparent_literal):
        raise RuntimeError("Absolute-G controller endpoint does not match plant.")
    if not bool(
        torch.all(result.accepted | result.nonfinite | result.budget_exhausted)
    ):
        raise RuntimeError("Absolute-G P&V left non-terminal cells.")
    return {
        "coordinate": "absolute_conductance_G",
        "information_boundary": dict(mapping_contract()),
        "cells": plant.size,
        "requested_target": _tensor_summary(target),
        "persistent_conductance": _tensor_summary(persistent),
        "literal_apparent_verify_conductance": _tensor_summary(apparent_literal),
        "projected_apparent_forward_conductance": _tensor_summary(apparent_forward),
        "requested_above_hidden_true_SET_cells": int(unreachable.sum().item()),
        "requested_above_hidden_true_SET_fraction": float(
            unreachable.to(torch.float64).mean().item()
        ),
        "unreachable_apparent_acceptance_fraction": _fraction(
            unreachable, result.accepted
        ),
        "unreachable_budget_exhaustion_fraction": _fraction(
            unreachable, result.budget_exhausted
        ),
        "controller_completed": int(result.accepted.sum().item()),
        "controller_completion_fraction": float(
            result.accepted.to(torch.float64).mean().item()
        ),
        "persistent_within_tolerance_fraction": float(
            persistent_within.to(torch.float64).mean().item()
        ),
        "literal_apparent_within_tolerance_fraction": float(
            apparent_within.to(torch.float64).mean().item()
        ),
        "healthy_exact_RESET_targets_preaccepted": int(preaccepted.sum().item()),
        "budget_exhausted": int(result.budget_exhausted.sum().item()),
        "nonfinite": int(result.nonfinite.sum().item()),
        "persistent_minus_target_rmse": _rmse(persistent - target),
        "literal_apparent_minus_target_rmse": _rmse(apparent_literal - target),
        "pulses": {
            "total": int(result.total_pulses.sum().item()),
            "SET": int(result.set_count.sum().item()),
            "RESET": int(result.reset_count.sum().item()),
            "pulsed_cells": int((result.total_pulses > 0).sum().item()),
            "maximum_per_cell": int(result.total_pulses.max().item()),
            "reversals": int(result.reversals.sum().item()),
            "verifies_including_initial": int(result.verify_count.sum().item()),
        },
        "apparent_forward_physical": _physical_report(
            plant.apparent_full_conductance,
            mapping.requested_g,
            masters,
            plant.field,
        ),
        "persistent_physical_secondary": _physical_report(
            plant.full_conductance,
            mapping.requested_g,
            masters,
            plant.field,
        ),
        "controller": {
            "kind": "one_pulse_apparent_absolute_G_feedback",
            "tolerance_conductance": tolerance,
            "maximum_pulses_per_cell": 128,
            "true_SET_used_to_normalize_verify": False,
            "persistent_state_exposed": False,
        },
    }


def _evaluate_static(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    full_g: Sequence[torch.Tensor],
    sample_limit: int | None,
) -> Mapping[str, Any]:
    _apply_full_g(stack, full_g)
    metrics, _ = _evaluate_detailed(
        stack, teacher, loader, sample_limit=sample_limit
    )
    return metrics


def _macro_evaluate(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.Tensor],
    fields: Sequence[Any],
    nominal_set: float,
    scale_fractions: Sequence[float],
    sample_limit: int | None,
) -> Mapping[str, Any]:
    evaluations = []
    for field in fields:
        mapping = map_masters_from_reset_only(
            masters,
            field,
            nominal_set=nominal_set,
            scale_fractions=scale_fractions,
        )
        evaluations.append(
            _evaluate_static(
                stack=stack,
                teacher=teacher,
                loader=loader,
                full_g=mapping.plant_limited_g,
                sample_limit=sample_limit,
            )
        )
    return {
        "primary_state": "held_plant_limited_apparent_conductance",
        "endpoint_evaluations": evaluations,
        "macro_student_accuracy": sum(
            float(item["student_accuracy"]) for item in evaluations
        )
        / len(evaluations),
        "macro_kl_teacher_student": sum(
            float(item["kl_teacher_student"]) for item in evaluations
        )
        / len(evaluations),
        "macro_teacher_agreement": sum(
            float(item["teacher_agreement"]) for item in evaluations
        )
        / len(evaluations),
    }


def _train_hwa(
    *,
    store: RunStore,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    stack: Any,
    teacher: Any,
    spec: Any,
    initial_masters: Sequence[torch.Tensor],
    smoke: bool,
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, Any]]:
    hwa = config["hwa"]
    mapping_config = config["mapping"]
    nominal_set = float(mapping_config["nominal_set"])
    scale_fractions = tuple(float(value) for value in mapping_config["scale_fractions"])
    rates = tuple(float(value) for value in hwa["learning_rates"])
    epochs = 1 if smoke else int(hwa["epochs"])
    masters = tuple(
        torch.nn.Parameter(value.detach().clone().to(stack.device))
        for value in initial_masters
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [master], "lr": rate}
            for master, rate in zip(masters, rates, strict=True)
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    loaders = _loaders(spec, data_seed=42)
    bank = _augmented_hwa_bank(base_config, device=stack.device, smoke=smoke)
    development_seeds = list(hwa["development_endpoint_seeds"])
    if smoke:
        development_seeds = development_seeds[:1]
    development = tuple(
        _field(int(seed), device=stack.device) for seed in development_seeds
    )
    selected_key = None
    selected_epoch = None
    selected_masters = None
    history = []
    global_batch = 0
    for epoch in range(1, epochs + 1):
        totals = {"examples": 0, "batches": 0, "kl": 0.0, "correct": 0, "agreement": 0}
        bank_counts = [0 for _ in bank]
        for batch_index, (inputs, labels) in enumerate(
            limited(loaders.train, 1 if smoke else None), start=1
        ):
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            bank_index = global_batch % len(bank)
            field = bank[bank_index]
            mapped = map_masters_from_reset_only(
                masters,
                field,
                nominal_set=nominal_set,
                scale_fractions=scale_fractions,
            )
            physical, metrics = _one_forward_gradient(
                stack=stack,
                teacher=teacher,
                inputs=inputs,
                labels=labels,
                full_g=mapped.plant_limited_g,
            )
            logical = lift_reset_only_gradients(
                masters,
                physical,
                field,
                nominal_set=nominal_set,
                scale_fractions=scale_fractions,
            )
            optimizer.zero_grad(set_to_none=True)
            for master, gradient in zip(masters, logical, strict=True):
                master.grad = gradient
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
            if batch_index % (1 if smoke else 500) == 0:
                print(
                    f"RESET-only HWA epoch={epoch}/{epochs} batch={batch_index}",
                    flush=True,
                )
        examples = int(totals["examples"])
        if not smoke and examples != 55_000:
            raise RuntimeError(f"Expected 55,000 HWA examples; observed {examples}.")
        validation = _macro_evaluate(
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            masters=masters,
            fields=development,
            nominal_set=nominal_set,
            scale_fractions=scale_fractions,
            sample_limit=64 if smoke else None,
        )
        record = {
            "phase": "HWA",
            "epoch": epoch,
            "rates": list(rates),
            "forward_state": "held_plant_limited_apparent_conductance",
            "true_SET_gradient_access": False,
            "digital_master_updated": True,
            "persistent_device_updated": False,
            "train": {
                "examples": examples,
                "batches": int(totals["batches"]),
                "kl_teacher_student": float(totals["kl"]) / examples,
                "student_accuracy": int(totals["correct"]) / examples,
                "teacher_agreement": int(totals["agreement"]) / examples,
                "bank_draw_counts": bank_counts,
            },
            "development_validation": validation,
        }
        history.append(record)
        store.append_metric(record)
        key = (
            float(validation["macro_student_accuracy"]),
            -float(validation["macro_kl_teacher_student"]),
            -epoch,
        )
        if selected_key is None or key > selected_key:
            selected_key = key
            selected_epoch = epoch
            selected_masters = _snapshot(masters)
        print(
            f"RESET-only HWA epoch={epoch}/{epochs} macro_val="
            f"{100.0*float(validation['macro_student_accuracy']):.2f}%",
            flush=True,
        )
    if selected_masters is None or selected_epoch is None:
        raise RuntimeError("RESET-only HWA produced no selected checkpoint.")
    return selected_masters, {
        "selected_epoch": selected_epoch,
        "selected_rates": list(rates),
        "selection": "macro_apparent_accuracy_then_KL_then_earlier_epoch",
        "selected_validation": history[selected_epoch - 1]["development_validation"],
        "history": history,
        "test_opened": False,
    }


def _aggregate(values: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(values) != 9:
        raise ValueError("Expected nine records per P&V arm.")
    result = {}
    metrics = {
        "apparent_student_accuracy": ("apparent", "student_accuracy"),
        "apparent_kl_teacher_student": ("apparent", "kl_teacher_student"),
        "apparent_teacher_agreement": ("apparent", "teacher_agreement"),
        "persistent_student_accuracy": (
            "persistent_secondary_diagnostic",
            "student_accuracy",
        ),
        "persistent_kl_teacher_student": (
            "persistent_secondary_diagnostic",
            "kl_teacher_student",
        ),
        "persistent_teacher_agreement": (
            "persistent_secondary_diagnostic",
            "teacher_agreement",
        ),
    }
    for name, path in metrics.items():
        array_means = []
        for array_index in (1, 2, 3):
            selected = [
                item for item in values if int(item["array_index"]) == array_index
            ]
            observed = [
                float(item["evaluation"][path[0]][path[1]]) for item in selected
            ]
            array_means.append(sum(observed) / len(observed))
        result[name] = {
            "array_means": array_means,
            "mean_across_arrays": sum(array_means) / len(array_means),
            "full_range_across_array_means": [min(array_means), max(array_means)],
        }
    return result


def _paired_delta(
    by_replica: Mapping[tuple[int, int, str], Mapping[str, Any]],
    *,
    left: str,
    right: str,
) -> Mapping[str, float]:
    accuracy = []
    kl = []
    for array_index in (1, 2, 3):
        for write_index in (1, 2, 3):
            lvalue = by_replica[(array_index, write_index, left)]["evaluation"]["apparent"]
            rvalue = by_replica[(array_index, write_index, right)]["evaluation"]["apparent"]
            accuracy.append(
                float(rvalue["student_accuracy"]) - float(lvalue["student_accuracy"])
            )
            kl.append(
                float(rvalue["kl_teacher_student"])
                - float(lvalue["kl_teacher_student"])
            )
    return {
        "accuracy_delta_mean": sum(accuracy) / len(accuracy),
        "kl_delta_mean": sum(kl) / len(kl),
    }


def run(args: argparse.Namespace) -> Path:
    config_path = args.config.expanduser().resolve()
    config = _load_json(config_path)
    _validate_config(config)
    inputs = _inputs(config_path, config)
    base_config = load_ladder_config(inputs["base_protocol_config"])
    if list(config["hwa"]["endpoint_train_seeds"]) != list(
        base_config["endpoint_model"]["train_seeds"]
    ):
        raise ValueError("HWA endpoint training seeds are not matched to the oracle run.")
    if list(config["replication"]["endpoint_seeds"]) != list(
        base_config["endpoint_model"]["heldout_seeds"]
    ):
        raise ValueError("Held-out endpoint seeds are not matched to the oracle run.")

    _require_cuda()
    input_records = [
        {"role": name, "path": str(path), "sha256": sha256_file(path)}
        for name, path in inputs.items()
    ]
    store = RunStore.create(
        output_root=args.output_root.expanduser().resolve(),
        experiment_id=EXPERIMENT_ID,
        resolved_config={
            "schema": EXPERIMENT_ID,
            "schema_version": 1,
            "evidence_tier": "exploratory_noncanonical",
            "smoke_override": bool(args.smoke),
            "primary_forward_state": "current_held_apparent_state",
            "persistent_state_role": "write_authority_and_secondary_diagnostic_only",
            "config": dict(config),
        },
        command=sys.argv,
        repo_root=ROOT,
        input_artifacts=input_records,
        resume_capability="unsupported",
    )
    try:
        spec, teacher, runtime = _runtime(
            base_config,
            teacher_path=inputs["teacher_checkpoint"],
            gain=float(config["evaluation"]["fixed_logit_gain"]),
            smoke=args.smoke,
        )
        stack = runtime["stack"]
        direct = _initial_masters(teacher, stack.device)
        selected_hwa, hwa_report = _train_hwa(
            store=store,
            config=config,
            base_config=base_config,
            stack=stack,
            teacher=teacher,
            spec=spec,
            initial_masters=direct,
            smoke=args.smoke,
        )
        hwa_checkpoint = atomic_torch_save(
            {
                "schema": "ebl.figure6_om_reset_only_hwa_master",
                "schema_version": 1,
                "information_boundary": dict(mapping_contract()),
                "selected_master": selected_hwa,
                "master_report": _master_report(selected_hwa),
                "report": hwa_report,
            },
            store.run_dir / "checkpoints" / "selected_hwa_master.pt",
        )
        hwa_device = tuple(value.to(stack.device) for value in selected_hwa)
        masters = {"direct": direct, "hwa": hwa_device}
        loaders = _loaders(spec)
        mapping_config = config["mapping"]
        nominal_set = float(mapping_config["nominal_set"])
        scale_fractions = tuple(
            float(value) for value in mapping_config["scale_fractions"]
        )
        sample_limit = 64 if args.smoke else None
        array_indices = (1,) if args.smoke else (1, 2, 3)
        write_indices = (1,) if args.smoke else (1, 2, 3)
        ideal_records = []
        pv_records = []
        replication = config["replication"]
        pv_config = config["program_verify"]
        population_pairs = {}
        for array_index in array_indices:
            published = load_om_array_population(
                inputs[f"array_{array_index}_published"]
            )
            repaired = load_om_array_population(
                inputs[f"array_{array_index}_repaired"]
            )
            pairing = validate_paired_om_populations(published, repaired)
            expected_assignment = int(
                replication["om_assignment_seeds"][array_index - 1]
            )
            if repaired.assignment_seed != expected_assignment:
                raise ValueError("Held-out OM assignment seed changed.")
            population_pairs[array_index] = (published, repaired, pairing)

        # Test is opened only after the HWA epoch is fixed.
        fields = {
            array_index: _field(
                int(replication["endpoint_seeds"][array_index - 1]),
                device=stack.device,
            )
            for array_index in array_indices
        }
        mappings = {}
        for array_index, field in fields.items():
            for family, master in masters.items():
                mapped = map_masters_from_reset_only(
                    master,
                    field,
                    nominal_set=nominal_set,
                    scale_fractions=scale_fractions,
                )
                mappings[(array_index, family)] = mapped
                evaluation = _evaluate_static(
                    stack=stack,
                    teacher=teacher,
                    loader=loaders.test,
                    full_g=mapped.plant_limited_g,
                    sample_limit=sample_limit,
                )
                ideal_record = {
                    "array_index": array_index,
                    "family": family,
                    "endpoint_seed": int(
                        replication["endpoint_seeds"][array_index - 1]
                    ),
                    "evaluation": evaluation,
                    "mapping": _mapping_report(mapped, master, field),
                }
                ideal_records.append(ideal_record)
                store.append_metric(
                    {
                        "phase": "ideal_plant_limited",
                        "array_index": array_index,
                        "family": family,
                        "student_accuracy": evaluation["student_accuracy"],
                        "kl_teacher_student": evaluation["kl_teacher_student"],
                    }
                )
                print(
                    f"RESET-only ideal array={array_index} family={family} "
                    f"accuracy={100.0*float(evaluation['student_accuracy']):.2f}%",
                    flush=True,
                )

        for array_index in array_indices:
            field = fields[array_index]
            published, repaired, pairing = population_pairs[array_index]
            for write_index in write_indices:
                suffix = 100 * array_index + write_index
                pv_seed = int(replication["program_verify_seed_base"]) + suffix
                fault_seed = int(replication["fault_observation_seed_base"]) + suffix
                for family, master in masters.items():
                    mapped = mappings[(array_index, family)]
                    plant, pv_result = program_conductance_with_verify(
                        field,
                        mapped.requested_g,
                        repaired,
                        pulse_noise_seed=pv_seed,
                        tolerance_conductance=float(
                            pv_config["tolerance_conductance"]
                        ),
                        maximum_pulses=int(
                            pv_config["maximum_pulses_per_cell"]
                        ),
                        noisy_initial_reset_verify=True,
                        preaccept_exact_healthy_reset_targets=True,
                    )
                    programming = _programming_report(
                        mapping=mapped,
                        plant=plant,
                        result=pv_result,
                        masters=master,
                        tolerance=float(pv_config["tolerance_conductance"]),
                    )
                    clean_evaluation = _evaluate_plant(
                        stack=stack,
                        teacher=teacher,
                        loader=loaders.test,
                        plant=plant,
                        sample_limit=sample_limit,
                    )
                    faulted = plant.clone_for_new_pulse_phase(
                        pulse_noise_seed=pv_seed + 1_000_000,
                        retain_apparent_observation=True,
                    )
                    fault_receipt = faulted.inject_post_pv_reset_stuck_faults(
                        published.published_corrupt,
                        observation_seed=fault_seed,
                    )
                    faulted_evaluation = _evaluate_plant(
                        stack=stack,
                        teacher=teacher,
                        loader=loaders.test,
                        plant=faulted,
                        sample_limit=sample_limit,
                    )
                    common = {
                        "array_index": array_index,
                        "write_index": write_index,
                        "family": family,
                        "endpoint_seed": int(
                            replication["endpoint_seeds"][array_index - 1]
                        ),
                        "om_assignment_seed": repaired.assignment_seed,
                        "program_verify_seed": pv_seed,
                        "fault_observation_seed": fault_seed,
                        "om_pairing": pairing,
                        "program_verify": programming,
                    }
                    pv_records.extend(
                        (
                            {
                                **common,
                                "arm": f"{family}_pv",
                                "evaluation": clean_evaluation,
                            },
                            {
                                **common,
                                "arm": f"{family}_pv_reset_stuck",
                                "fault_receipt": dict(fault_receipt),
                                "faulted_apparent_physical": _physical_report(
                                    faulted.apparent_full_conductance,
                                    mapped.requested_g,
                                    master,
                                    field,
                                ),
                                "evaluation": faulted_evaluation,
                            },
                        )
                    )
                    store.append_metric(
                        {
                            "phase": "program_verify",
                            "array_index": array_index,
                            "write_index": write_index,
                            "family": family,
                            "clean_apparent_accuracy": clean_evaluation["apparent"][
                                "student_accuracy"
                            ],
                            "faulted_apparent_accuracy": faulted_evaluation[
                                "apparent"
                            ]["student_accuracy"],
                            "controller_completion_fraction": programming[
                                "controller_completion_fraction"
                            ],
                            "requested_above_hidden_true_SET_fraction": programming[
                                "requested_above_hidden_true_SET_fraction"
                            ],
                        }
                    )
                    print(
                        f"RESET-only P&V array={array_index} write={write_index} "
                        f"family={family} clean/fault="
                        f"{100.0*float(clean_evaluation['apparent']['student_accuracy']):.2f}%/"
                        f"{100.0*float(faulted_evaluation['apparent']['student_accuracy']):.2f}%",
                        flush=True,
                    )

        raw_path = store.run_dir / "artifacts" / "raw_records.json"
        atomic_write_json(
            raw_path,
            {"ideal_records": ideal_records, "program_verify_records": pv_records},
        )
        if args.smoke:
            arms = {}
            comparisons = {}
        else:
            arm_names = (
                "direct_pv",
                "direct_pv_reset_stuck",
                "hwa_pv",
                "hwa_pv_reset_stuck",
            )
            by_arm = {
                arm: [record for record in pv_records if record["arm"] == arm]
                for arm in arm_names
            }
            arms = {arm: _aggregate(values) for arm, values in by_arm.items()}
            by_replica = {
                (
                    int(record["array_index"]),
                    int(record["write_index"]),
                    str(record["arm"]),
                ): record
                for record in pv_records
            }
            comparisons = {
                "HWA_minus_direct_clean": _paired_delta(
                    by_replica, left="direct_pv", right="hwa_pv"
                ),
                "HWA_minus_direct_reset_stuck": _paired_delta(
                    by_replica,
                    left="direct_pv_reset_stuck",
                    right="hwa_pv_reset_stuck",
                ),
                "direct_add_post_PV_fault": _paired_delta(
                    by_replica,
                    left="direct_pv",
                    right="direct_pv_reset_stuck",
                ),
                "HWA_add_post_PV_fault": _paired_delta(
                    by_replica,
                    left="hwa_pv",
                    right="hwa_pv_reset_stuck",
                ),
            }
        oracle = _load_json(inputs["oracle_endpoint_summary"])
        summary = {
            "status": "complete",
            "experiment_id": EXPERIMENT_ID,
            "evidence_tier": "exploratory_noncanonical",
            "smoke_override": bool(args.smoke),
            "question": config["question"],
            "information_boundary": dict(mapping_contract()),
            "primary_state": "current_held_apparent_state",
            "persistent_state": "secondary_diagnostic_only",
            "HWA": hwa_report,
            "ideal_plant_limited_records": ideal_records,
            "program_verify_arms": arms,
            "paired_comparisons": comparisons,
            "oracle_exact_endpoint_reference": {
                arm: oracle.get("arms", {}).get(arm)
                for arm in (
                    "direct_pv",
                    "direct_pv_reset_stuck",
                    "hwa_pv",
                    "hwa_pv_reset_stuck",
                )
            },
            "coverage": {
                "arrays": len(array_indices),
                "writes_per_array": len(write_indices),
                "HWA_epochs": 1 if args.smoke else int(config["hwa"]["epochs"]),
                "test_opened_after_HWA_selection": True,
            },
            "limitations": list(config["limitations"]),
        }
        summary_path = store.run_dir / "scientific_summary.json"
        atomic_write_json(summary_path, summary)
        result_path = store.complete(
            metrics=summary,
            artifacts=(
                store.artifact_record(summary_path, kind="scientific_summary"),
                store.artifact_record(raw_path, kind="raw_records"),
                store.artifact_record(hwa_checkpoint, kind="selected_HWA_master"),
            ),
        )
        print(f"complete result={result_path}", flush=True)
        return result_path
    except BaseException as error:
        store.fail(error)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one batch, epoch, array, write, and 64-example evaluations.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
