"""Staged 784-256-10 DRN deployment/HWA/post-P&V-fault ladder.

This is an exploratory composition root.  Every physical forward uses the
current held apparent state.  Persistent state is write authority and a
secondary diagnostic only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.config import parse_student_config, resolve_student_spec
from experiments.mnist_relu_drn.figure6_om_256_ladder_config import (
    EXPERIMENT_ID,
    load_ladder_config,
)
from experiments.mnist_relu_drn.figure6_om_pulse import (
    OM_PROGRAM_VERIFY_MAXIMUM_PULSES,
    OM_PROGRAM_VERIFY_TOLERANCE_PROGRESS,
    Figure6OmPulsePlant,
    PersistentFigure6OmPulseAdam,
    program_masters_with_verify,
    validate_paired_om_populations,
)
from experiments.mnist_relu_drn.figure6_om_pulse_corruption_experiment import (
    _program_verify_report,
)
from experiments.mnist_relu_drn.hfo2_figure6_drn import (
    ENDPOINT_LAYOUTS,
    EndpointField,
    endpoint_field_from_population,
    flatten_physical,
    lift_endpoint_gradients,
    map_masters_to_conductance,
    master_to_progress,
    rotate_endpoint_field,
)
from experiments.mnist_relu_drn.hfo2_figure6_drn_experiment import (
    _apply_full_g,
    _evaluate_full_g,
    _initial_masters,
    _one_forward_gradient,
)
from experiments.mnist_relu_drn.hfo2_figure6_endpoint_regimes import (
    CONDITIONAL_REDRAW_INVALID_PAIRS,
    INDEPENDENT_ENDPOINTS,
    sample_hfo2_figure6_endpoint_population,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _evaluate_detailed,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import RunMode
from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from training.checkpoint import atomic_torch_save
from training.ibm_reram_hwa import (
    IbmReramArrayPopulation,
    load_om_array_population,
    sample_om_array_population_external,
)
from training.ibm_reram_program_verify import PUBLISHED_CORRUPT_PROBABILITY


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = (
    _ROOT
    / "examples/mnist_relu_drn/figure6_om_784_256_10_postpv_fault/full.json"
)
_DEFAULT_OUTPUT = (
    _ROOT
    / "simulation_results/"
    "exploratory_noncanonical_figure6_om_784_256_10_postpv_fault"
)
_SHAPES = ((1568, 512), (512, 20))
_CELLS = 813_056
_FAMILIES = ("hwa", "scratch")
_FAULTS = ("clean", "reset_stuck")


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The Figure-6 OM 256 ladder requires CUDA; CPU fallback is forbidden."
        )
    device = torch.device("cuda")
    probe = torch.arange(4096, device=device, dtype=torch.float32).reshape(64, 64)
    _ = probe @ probe.T
    torch.cuda.synchronize()
    return device


def _student_spec(
    config: Mapping[str, Any],
    *,
    smoke: bool,
) -> Any:
    student = config["student"]
    payload = {
        "schema_version": 1,
        "experiment_id": "mnist_relu_drn_kd.v1",
        "runtime": {
            "seed": 42,
            "data_seed": 42,
            "device": "cuda",
            "dtype": "float32",
        },
        "data": {
            "batch_size": int(config["adam"]["batch_size"]),
            "validation_points": 64 if smoke else 5000,
            "num_points": 64 if smoke else None,
            "shuffle": True,
        },
        "teacher": {
            "type": "bias_free_relu",
            "initialization": "signed_weight_mapping",
        },
        "model": {
            "dims": list(student["physical_dims"]),
            "input_gain": float(student["input_gain"]),
            "conductance_min": float(student["conductance_min"]),
            "conductance_max": float(student["conductance_max"]),
            "voltage_amp": float(student["voltage_amp"]),
            "current_amp": float(student["current_amp"]),
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
            "inference_iterations": int(student["solver_iterations"]),
            "training_iterations": int(student["solver_iterations"]),
            "mode": student["solver"],
            "overrelaxation_factor": float(student["overrelaxation_factor"]),
        },
        "mapping": {
            "scale_fractions": [1.0],
            "scale_fraction_pairs": [[1.0, 1.0]],
            "range_placement": "lower",
            "calibration_examples": 32 if smoke else int(config["gain_calibration"]["examples"]),
            "calibration_batch_size": 16 if smoke else 128,
            "logit_gain_min": float(config["gain_calibration"]["minimum"]),
            "logit_gain_max": float(config["gain_calibration"]["maximum"]),
            "logit_gain_steps": int(config["gain_calibration"]["steps"]),
        },
        "modes": {
            "train": {
                "num_epochs": int(config["hwa"]["main_epochs"]),
                "learning_rates": list(config["hwa"]["base_learning_rates"]),
                "temperature": 1.0,
                "log_every": 1,
                "max_batches": None,
                "max_validation_batches": None,
                "minimum_relative_kl_improvement": 0.0,
                "weight_modifier": {"type": "none", "parameters": {}},
                "selection_weight_modifier": {"type": "none", "parameters": {}},
                "update_backend": {"type": "ideal", "parameters": {}},
                "selection_evaluation": "clean",
                "selection_metric": "student_accuracy",
                "selection_noise_repeats": 1,
            }
        },
    }
    return resolve_student_spec(parse_student_config(payload), RunMode.TRAIN)


def _loaders(spec: Any, *, data_seed: int | None = None) -> Any:
    return build_mnist_loaders(
        spec.data,
        data_seed=spec.runtime.data_seed if data_seed is None else int(data_seed),
        calibration_examples=spec.mapping.calibration_examples,
        calibration_batch_size=spec.mapping.calibration_batch_size,
    )


def _runtime(
    config: Mapping[str, Any],
    *,
    teacher_path: Path,
    gain: float | None,
    smoke: bool,
) -> tuple[Any, Any, Any]:
    device = _require_cuda()
    spec = _student_spec(config, smoke=smoke)
    teacher, metadata = _load_teacher(teacher_path, device=device, spec=spec)
    stack = build_student_stack(spec, enable_measured=False)
    if gain is not None:
        stack.cost.gain = float(gain)
    shapes = tuple(tuple(binding.state.shape) for binding in stack.bundle.catalog.trainable)
    if shapes != _SHAPES:
        raise RuntimeError(f"Expected physical shapes {_SHAPES!r}; observed {shapes!r}.")
    if sum(math.prod(shape) for shape in shapes) != _CELLS:
        raise RuntimeError("The four-cell 784-256-10 DRN cell count changed.")
    return spec, teacher, {"stack": stack, "teacher_metadata": metadata}


def _field(
    seed: int,
    *,
    device: torch.device,
) -> EndpointField:
    population = sample_hfo2_figure6_endpoint_population(
        devices=_CELLS,
        assignment_seed=int(seed),
        regime=INDEPENDENT_ENDPOINTS,
        invalid_pair_policy=CONDITIONAL_REDRAW_INVALID_PAIRS,
    )
    return endpoint_field_from_population(
        population,
        shapes=_SHAPES,
        layouts=ENDPOINT_LAYOUTS,
        device=device,
        dtype=torch.float32,
    )


def _augmented_hwa_bank(
    config: Mapping[str, Any],
    *,
    device: torch.device,
    smoke: bool,
) -> tuple[EndpointField, ...]:
    seeds = list(config["endpoint_model"]["train_seeds"])
    rotations = int(config["hwa"]["rotations_per_population"])
    if smoke:
        seeds = seeds[:1]
        rotations = 1
    base = tuple(_field(seed, device=device) for seed in seeds)
    result = []
    for rotation_index in range(rotations):
        for base_index, field in enumerate(base):
            offset = (rotation_index * 104_729 + base_index * 65_537) % field.devices
            result.append(rotate_endpoint_field(field, offset))
    return tuple(result)


def _snapshot(values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().cpu().to(torch.float32).clone() for value in values)


def _tensor_sha256(value: torch.Tensor) -> str:
    import hashlib

    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _master_report(masters: Sequence[torch.Tensor]) -> Mapping[str, Any]:
    return {
        "layers": [
            {
                "shape": list(value.shape),
                "minimum": float(value.min().item()),
                "maximum": float(value.max().item()),
                "mean": float(value.to(torch.float64).mean().item()),
                "rms": float(value.to(torch.float64).square().mean().sqrt().item()),
                "sha256": _tensor_sha256(value),
            }
            for value in masters
        ]
    }


def _completed_run(campaign_root: Path, stage: str, job: str) -> Path:
    root = campaign_root / "stages" / stage / job
    candidates = sorted(
        root.glob("*/result.json") if root.is_dir() else (),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for result_path in candidates:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") == "complete" and result.get("experiment_id") == EXPERIMENT_ID:
            return result_path.parent
    raise FileNotFoundError(f"No complete {stage}/{job} run exists below {root}.")


def _artifact(campaign_root: Path, stage: str, job: str, relative: str) -> Path:
    path = _completed_run(campaign_root, stage, job) / relative
    if not path.is_file():
        raise FileNotFoundError(f"Expected completed artifact {path}.")
    return path


def _input(role: str, path: Path) -> Mapping[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Expected {role} input at {source}.")
    return {"role": role, "path": str(source), "sha256": sha256_file(source)}


def _store(
    args: argparse.Namespace,
    *,
    stage: str,
    job: str,
    extra: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
) -> RunStore:
    contract = {
        "schema": EXPERIMENT_ID,
        "schema_version": 1,
        "evidence_tier": "exploratory_noncanonical",
        "stage": stage,
        "job": job,
        "smoke_override": bool(args.smoke),
        "primary_forward_state": "current_held_apparent_state",
        "persistent_state_role": "write_authority_and_secondary_diagnostic_only",
        "config": dict(args.config_payload),
        **dict(extra),
    }
    return RunStore.create(
        output_root=args.campaign_root / "stages" / stage / job,
        experiment_id=EXPERIMENT_ID,
        resolved_config=contract,
        command=sys.argv,
        repo_root=_ROOT,
        input_artifacts=inputs,
        resume_capability="unsupported",
    )


def _finish(
    store: RunStore,
    *,
    summary: Mapping[str, Any],
    artifacts: Sequence[tuple[Path, str]],
) -> Path:
    summary_path = store.run_dir / "scientific_summary.json"
    atomic_write_json(summary_path, summary)
    result = store.complete(
        metrics=summary,
        artifacts=(
            store.artifact_record(summary_path, kind="scientific_summary"),
            *(store.artifact_record(path, kind=kind) for path, kind in artifacts),
        ),
    )
    print(f"complete result={result}", flush=True)
    return result


def _gain_screen(
    *,
    config: Mapping[str, Any],
    stack: Any,
    teacher: Any,
    calibration: Iterable,
    direct_masters: Sequence[torch.Tensor],
    device: torch.device,
    smoke: bool,
) -> tuple[float, list[Mapping[str, Any]]]:
    settings = config["gain_calibration"]
    gains = torch.logspace(
        math.log10(float(settings["minimum"])),
        math.log10(float(settings["maximum"])),
        steps=int(settings["steps"]),
        dtype=torch.float64,
    ).tolist()
    seeds = list(settings["endpoint_seeds"])
    if smoke:
        gains = [0.1, 1.0, 10.0]
        seeds = seeds[:1]
    fields = tuple(_field(seed, device=device) for seed in seeds)
    records = []
    for gain in gains:
        stack.cost.gain = float(gain)
        evaluations = [
            _evaluate_full_g(
                stack=stack,
                teacher=teacher,
                loader=calibration,
                full_g=map_masters_to_conductance(direct_masters, field),
                sample_limit=32 if smoke else int(settings["examples"]),
            )
            for field in fields
        ]
        record = {
            "gain": float(gain),
            "macro_kl_teacher_student": sum(
                float(item["kl_teacher_student"]) for item in evaluations
            )
            / len(evaluations),
            "macro_student_accuracy": sum(
                float(item["student_accuracy"]) for item in evaluations
            )
            / len(evaluations),
            "endpoint_seeds": seeds,
        }
        records.append(record)
    selected = min(records, key=lambda item: (item["macro_kl_teacher_student"], item["gain"]))
    stack.cost.gain = float(selected["gain"])
    return float(selected["gain"]), records


def _population_paths(store: RunStore, tag: str, policy: str) -> tuple[Path, Path]:
    stem = f"om_{tag}_{policy}"
    return (
        store.run_dir / "artifacts" / f"{stem}.npz",
        store.run_dir / "artifacts" / f"{stem}.receipt.json",
    )


def run_prepare(args: argparse.Namespace) -> Path:
    teacher_path = args.teacher_weights.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    store = _store(
        args,
        stage="prepare",
        job="main",
        extra={"mode": "prepare_frozen_inputs_and_gain"},
        inputs=(
            _input("strict_config", config_path),
            _input("teacher", teacher_path),
            _input("pinned_AIHWKit_interpreter", args.aihwkit_python),
        ),
    )
    try:
        config = args.config_payload
        expected_corrupt_probability = PUBLISHED_CORRUPT_PROBABILITY[
            "reram_array_om"
        ]
        if (
            float(config["om_devices"]["published_corrupt_probability"])
            != expected_corrupt_probability
        ):
            raise RuntimeError("Configured OM corruption probability changed.")
        spec, teacher, runtime = _runtime(
            config,
            teacher_path=teacher_path,
            gain=None,
            smoke=args.smoke,
        )
        stack = runtime["stack"]
        device = stack.device
        loaders = _loaders(spec)
        direct = _initial_masters(teacher, device)
        torch.manual_seed(int(config["scratch"]["seed"]))
        torch.cuda.manual_seed_all(int(config["scratch"]["seed"]))
        scratch_teacher = BiasFreeReluTeacher(
            device=device,
            dims=tuple(config["teacher"]["dims"]),
        )
        scratch = _initial_masters(scratch_teacher, device)
        gain, gain_records = _gain_screen(
            config=config,
            stack=stack,
            teacher=teacher,
            calibration=loaders.calibration,
            direct_masters=direct,
            device=device,
            smoke=args.smoke,
        )

        endpoint_seeds = sorted(
            set(
                config["endpoint_model"]["train_seeds"]
                + config["endpoint_model"]["development_seeds"]
                + [config["endpoint_model"]["adam_development_seed"]]
                + config["endpoint_model"]["heldout_seeds"]
            )
        )
        if args.smoke:
            endpoint_seeds = endpoint_seeds[:1]
        endpoint_reports = {
            str(seed): _field(seed, device=device).report() for seed in endpoint_seeds
        }

        assignments = [
            ("development", int(config["om_devices"]["development_assignment_seed"])),
            *(
                (f"heldout_array_{index}", int(seed))
                for index, seed in enumerate(
                    config["om_devices"]["heldout_assignment_seeds"], start=1
                )
            ),
        ]
        population_artifacts = []
        pairing_reports = {}
        for tag, seed in assignments:
            populations = {}
            for policy in ("published", "counterfactual_repaired"):
                population_path, receipt_path = _population_paths(store, tag, policy)
                population, receipt = sample_om_array_population_external(
                    stack.bundle.catalog.trainable,
                    assignment_seed=seed,
                    corruption_policy=policy,
                    aihwkit_python=args.aihwkit_python,
                    population_path=population_path,
                    receipt_path=receipt_path,
                )
                if population.size != _CELLS or population.binding_shapes != _SHAPES:
                    raise RuntimeError("Sampled OM population has the wrong topology.")
                populations[policy] = population
                population_artifacts.extend(
                    ((population_path, "OM_population"), (receipt_path, "OM_population_receipt"))
                )
            pairing_reports[tag] = validate_paired_om_populations(
                populations["published"], populations["counterfactual_repaired"]
            )

        prepared = {
            "schema": "ebl.figure6_om_256_ladder_prepared",
            "schema_version": 1,
            "evidence_tier": "exploratory_noncanonical",
            "teacher": {
                "path": str(teacher_path),
                "sha256": sha256_file(teacher_path),
                "metadata": runtime["teacher_metadata"],
            },
            "physical_shapes": _SHAPES,
            "cells": _CELLS,
            "fixed_logit_gain": gain,
            "gain_screen": gain_records,
            "direct_master": _snapshot(direct),
            "direct_master_report": _master_report(direct),
            "scratch_master": _snapshot(scratch),
            "scratch_master_report": _master_report(scratch),
            "endpoint_reports": endpoint_reports,
            "OM_pairing": pairing_reports,
            "apparent_forward_contract": True,
        }
        prepared_path = atomic_torch_save(
            prepared, store.run_dir / "artifacts" / "prepared.pt"
        )
        summary = {
            "status": "complete",
            "stage": "prepare",
            "fixed_logit_gain": gain,
            "cells": _CELLS,
            "teacher_sha256": sha256_file(teacher_path),
            "OM_assignments": pairing_reports,
            "endpoint_seeds_validated": endpoint_seeds,
            "primary_forward_state": "held_apparent",
        }
        return _finish(
            store,
            summary=summary,
            artifacts=((prepared_path, "prepared_inputs"), *population_artifacts),
        )
    except BaseException as error:
        store.fail(error)
        raise


def _load_torch(path: Path, *, schema: str) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"Could not load checkpoint {path}.") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != schema:
        raise ValueError(f"Checkpoint {path} does not have schema {schema!r}.")
    return payload


def _prepared(args: argparse.Namespace) -> tuple[Path, Mapping[str, Any]]:
    path = _artifact(args.campaign_root, "prepare", "main", "artifacts/prepared.pt")
    return path, _load_torch(path, schema="ebl.figure6_om_256_ladder_prepared")


def _masters_to_device(
    values: Sequence[torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, ...]:
    result = tuple(value.detach().to(device=device, dtype=torch.float32) for value in values)
    expected = ((784, 256), (256, 10))
    if tuple(tuple(value.shape) for value in result) != expected:
        raise ValueError(f"Expected logical master shapes {expected!r}.")
    if any(
        not bool(torch.isfinite(value).all())
        or bool(torch.any(value < -1.0))
        or bool(torch.any(value > 1.0))
        for value in result
    ):
        raise ValueError("Logical master is not finite in [-1,1].")
    return result


def _macro_evaluate_masters(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.Tensor],
    fields: Sequence[EndpointField],
    sample_limit: int | None,
) -> Mapping[str, Any]:
    evaluations = [
        _evaluate_full_g(
            stack=stack,
            teacher=teacher,
            loader=loader,
            full_g=map_masters_to_conductance(masters, field),
            sample_limit=sample_limit,
        )
        for field in fields
    ]
    return {
        "primary_state": "held_apparent_endpoint_view",
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


def _hwa_epoch_key(evaluation: Mapping[str, Any], epoch: int) -> tuple[float, float, int]:
    return (
        float(evaluation["macro_student_accuracy"]),
        -float(evaluation["macro_kl_teacher_student"]),
        -int(epoch),
    )


def _train_hwa(
    *,
    config: Mapping[str, Any],
    stack: Any,
    teacher: Any,
    spec: Any,
    initial_masters: Sequence[torch.Tensor],
    rates: tuple[float, float],
    epochs: int,
    smoke: bool,
    metric_sink: Any,
) -> tuple[tuple[torch.Tensor, ...], Mapping[str, Any]]:
    masters = tuple(
        torch.nn.Parameter(value.detach().clone().to(stack.device))
        for value in initial_masters
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [master], "lr": float(rate)}
            for master, rate in zip(masters, rates, strict=True)
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    loaders = _loaders(spec, data_seed=42)
    bank = _augmented_hwa_bank(config, device=stack.device, smoke=smoke)
    development_seeds = list(config["endpoint_model"]["development_seeds"])
    if smoke:
        development_seeds = development_seeds[:1]
        epochs = 1
    development = tuple(_field(seed, device=stack.device) for seed in development_seeds)
    global_batch = 0
    history = []
    selected_key = None
    selected_epoch = None
    selected_masters = None
    for epoch in range(1, int(epochs) + 1):
        totals = {
            "examples": 0,
            "batches": 0,
            "kl": 0.0,
            "correct": 0,
            "agreement": 0,
        }
        bank_counts = [0 for _ in bank]
        for batch_index, (inputs, labels) in enumerate(
            limited(loaders.train, 1 if smoke else None), start=1
        ):
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            bank_index = global_batch % len(bank)
            field = bank[bank_index]
            # This is the sampled held apparent hardware view.  Only the
            # digital logical master is updated during off-chip HWA.
            physical, metrics = _one_forward_gradient(
                stack=stack,
                teacher=teacher,
                inputs=inputs,
                labels=labels,
                full_g=map_masters_to_conductance(masters, field),
            )
            logical = lift_endpoint_gradients(masters, physical, field)
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
                    f"HWA epoch={epoch}/{epochs} batch={batch_index} rates={rates}",
                    flush=True,
                )
        examples = int(totals["examples"])
        if examples < 1:
            raise RuntimeError("HWA processed no examples.")
        if not smoke and examples != 55_000:
            raise RuntimeError(
                f"Expected one full 55,000-example HWA epoch; observed {examples}."
            )
        validation = _macro_evaluate_masters(
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            masters=masters,
            fields=development,
            sample_limit=64 if smoke else None,
        )
        record = {
            "epoch": epoch,
            "rates": list(rates),
            "forward_state": "held_apparent_endpoint_view",
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
        metric_sink(record)
        key = _hwa_epoch_key(validation, epoch)
        if selected_key is None or key > selected_key:
            selected_key = key
            selected_epoch = epoch
            selected_masters = _snapshot(masters)
        print(
            f"HWA epoch={epoch}/{epochs} apparent_macro_val="
            f"{100.0*float(validation['macro_student_accuracy']):.2f}%",
            flush=True,
        )
    if selected_masters is None or selected_epoch is None:
        raise RuntimeError("HWA produced no selected epoch.")
    return selected_masters, {
        "rates": list(rates),
        "selected_epoch": selected_epoch,
        "selection": "apparent_macro_accuracy_then_KL_then_earlier_epoch",
        "selected_validation": history[selected_epoch - 1]["development_validation"],
        "history": history,
        "final_optimizer_state": optimizer.state_dict(),
        "test_opened": False,
    }


def _multiplier_token(value: float) -> str:
    return format(float(value), ".12g").replace(".", "p").replace("-", "m")


def _hwa_candidate_job(input_multiplier: float, output_multiplier: float) -> str:
    return (
        f"input_{_multiplier_token(input_multiplier)}__"
        f"output_{_multiplier_token(output_multiplier)}"
    )


def run_hwa_screen_candidate(args: argparse.Namespace) -> Path:
    config = args.config_payload
    input_multiplier = float(args.input_multiplier)
    output_multiplier = float(args.output_multiplier)
    if input_multiplier not in config["hwa"]["input_multipliers"]:
        raise ValueError("Input multiplier is outside the strict HWA grid.")
    if output_multiplier not in config["hwa"]["output_multipliers"]:
        raise ValueError("Output multiplier is outside the strict HWA grid.")
    job = _hwa_candidate_job(input_multiplier, output_multiplier)
    prepared_path, prepared = _prepared(args)
    teacher_path = Path(prepared["teacher"]["path"])
    if sha256_file(teacher_path) != prepared["teacher"]["sha256"]:
        raise ValueError("Prepared teacher hash changed.")
    store = _store(
        args,
        stage="hwa-screen",
        job=job,
        extra={
            "mode": "HWA_rate_candidate",
            "input_multiplier": input_multiplier,
            "output_multiplier": output_multiplier,
        },
        inputs=(_input("prepared", prepared_path), _input("teacher", teacher_path)),
    )
    try:
        spec, teacher, runtime = _runtime(
            config,
            teacher_path=teacher_path,
            gain=float(prepared["fixed_logit_gain"]),
            smoke=args.smoke,
        )
        initial = _masters_to_device(prepared["direct_master"], runtime["stack"].device)
        base = config["hwa"]["base_learning_rates"]
        rates = (float(base[0]) * input_multiplier, float(base[1]) * output_multiplier)
        selected, report = _train_hwa(
            config=config,
            stack=runtime["stack"],
            teacher=teacher,
            spec=spec,
            initial_masters=initial,
            rates=rates,
            epochs=int(config["hwa"]["screen_epochs"]),
            smoke=args.smoke,
            metric_sink=store.append_metric,
        )
        checkpoint = atomic_torch_save(
            {
                "schema": "ebl.figure6_om_256_hwa_screen_candidate",
                "schema_version": 1,
                "input_multiplier": input_multiplier,
                "output_multiplier": output_multiplier,
                "selected_master": selected,
                "report": report,
            },
            store.run_dir / "checkpoints" / "candidate.pt",
        )
        summary = {
            "status": "complete",
            "stage": "hwa-screen",
            "job": job,
            "input_multiplier": input_multiplier,
            "output_multiplier": output_multiplier,
            "rates": list(rates),
            "selected_epoch": report["selected_epoch"],
            "selected_validation": report["selected_validation"],
            "test_opened": False,
        }
        return _finish(store, summary=summary, artifacts=((checkpoint, "HWA_candidate"),))
    except BaseException as error:
        store.fail(error)
        raise


def run_hwa_screen_select(args: argparse.Namespace) -> Path:
    config = args.config_payload
    candidate_paths = []
    candidates = []
    for input_multiplier in config["hwa"]["input_multipliers"]:
        for output_multiplier in config["hwa"]["output_multipliers"]:
            job = _hwa_candidate_job(input_multiplier, output_multiplier)
            path = _artifact(args.campaign_root, "hwa-screen", job, "scientific_summary.json")
            candidate_paths.append(path)
            candidates.append(json.loads(path.read_text(encoding="utf-8")))
    selected = max(
        candidates,
        key=lambda item: (
            float(item["selected_validation"]["macro_student_accuracy"]),
            -float(item["selected_validation"]["macro_kl_teacher_student"]),
            -int(item["selected_epoch"]),
            -float(item["input_multiplier"]),
            -float(item["output_multiplier"]),
        ),
    )
    store = _store(
        args,
        stage="hwa-screen",
        job="selection",
        extra={"mode": "HWA_rate_selection"},
        inputs=tuple(_input("HWA_candidate", path) for path in candidate_paths),
    )
    try:
        receipt = {
            "schema": "ebl.figure6_om_256_hwa_rate_receipt",
            "schema_version": 1,
            "selection_uses": "held_apparent_development_endpoint_views",
            "test_opened": False,
            "selected": selected,
            "candidates": candidates,
        }
        receipt_path = store.run_dir / "artifacts" / "rate_receipt.json"
        atomic_write_json(receipt_path, receipt)
        summary = {
            "status": "complete",
            "stage": "hwa-screen",
            "job": "selection",
            "selected_rates": selected["rates"],
            "selected_input_multiplier": selected["input_multiplier"],
            "selected_output_multiplier": selected["output_multiplier"],
            "selected_validation": selected["selected_validation"],
            "test_opened": False,
        }
        return _finish(store, summary=summary, artifacts=((receipt_path, "HWA_rate_receipt"),))
    except BaseException as error:
        store.fail(error)
        raise


def run_hwa_train(args: argparse.Namespace) -> Path:
    config = args.config_payload
    prepared_path, prepared = _prepared(args)
    receipt_path = _artifact(
        args.campaign_root, "hwa-screen", "selection", "artifacts/rate_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "ebl.figure6_om_256_hwa_rate_receipt":
        raise ValueError("HWA rate receipt schema changed.")
    rates = tuple(float(value) for value in receipt["selected"]["rates"])
    teacher_path = Path(prepared["teacher"]["path"])
    store = _store(
        args,
        stage="hwa-train",
        job="main",
        extra={"mode": "main_HWA", "selected_rates": list(rates)},
        inputs=(
            _input("prepared", prepared_path),
            _input("HWA_rate_receipt", receipt_path),
            _input("teacher", teacher_path),
        ),
    )
    try:
        spec, teacher, runtime = _runtime(
            config,
            teacher_path=teacher_path,
            gain=float(prepared["fixed_logit_gain"]),
            smoke=args.smoke,
        )
        initial = _masters_to_device(prepared["direct_master"], runtime["stack"].device)
        selected, report = _train_hwa(
            config=config,
            stack=runtime["stack"],
            teacher=teacher,
            spec=spec,
            initial_masters=initial,
            rates=rates,  # type: ignore[arg-type]
            epochs=int(config["hwa"]["main_epochs"]),
            smoke=args.smoke,
            metric_sink=store.append_metric,
        )
        checkpoint = atomic_torch_save(
            {
                "schema": "ebl.figure6_om_256_hwa_master",
                "schema_version": 1,
                "selected_master": selected,
                "master_report": _master_report(selected),
                "fixed_logit_gain": float(prepared["fixed_logit_gain"]),
                "rate_receipt": {
                    "path": str(receipt_path),
                    "sha256": sha256_file(receipt_path),
                },
                "report": report,
            },
            store.run_dir / "checkpoints" / "selected_hwa_master.pt",
        )
        summary = {
            "status": "complete",
            "stage": "hwa-train",
            "selected_epoch": report["selected_epoch"],
            "selected_rates": list(rates),
            "selected_validation": report["selected_validation"],
            "test_opened": False,
            "primary_forward_state": "held_apparent_endpoint_view",
        }
        return _finish(store, summary=summary, artifacts=((checkpoint, "selected_HWA_master"),))
    except BaseException as error:
        store.fail(error)
        raise


def _paired_om_populations(
    args: argparse.Namespace, tag: str
) -> tuple[IbmReramArrayPopulation, IbmReramArrayPopulation, Mapping[str, Any]]:
    prepare_dir = _completed_run(args.campaign_root, "prepare", "main")
    published_path = prepare_dir / "artifacts" / f"om_{tag}_published.npz"
    repaired_path = prepare_dir / "artifacts" / f"om_{tag}_counterfactual_repaired.npz"
    published = load_om_array_population(published_path)
    repaired = load_om_array_population(repaired_path)
    report = validate_paired_om_populations(published, repaired)
    return published, repaired, {
        "pairing": report,
        "published_path": str(published_path),
        "published_sha256": sha256_file(published_path),
        "repaired_path": str(repaired_path),
        "repaired_sha256": sha256_file(repaired_path),
    }


def _tensor_summary(value: torch.Tensor) -> Mapping[str, Any]:
    tensor = value.detach().to(torch.float64)
    return {
        "values": tensor.numel(),
        "minimum": float(tensor.min().item()),
        "mean": float(tensor.mean().item()),
        "rms": float(tensor.square().mean().sqrt().item()),
        "standard_deviation": float(tensor.std(unbiased=False).item()),
        "maximum": float(tensor.max().item()),
        "sha256": _tensor_sha256(value),
    }


def _state_diagnostics(
    plant: Figure6OmPulsePlant,
    *,
    target_master: Sequence[torch.Tensor],
) -> Mapping[str, Any]:
    literal = plant.apparent_progress_unprojected
    apparent = plant.apparent_progress_projected
    persistent = plant.progress
    apparent_g = plant.apparent_full_conductance
    persistent_g = plant.full_conductance
    layers = []
    for index, (target, layout) in enumerate(
        zip(target_master, ENDPOINT_LAYOUTS, strict=True)
    ):
        apparent_quad = quad_stack(apparent[index], layout=layout)
        persistent_quad = quad_stack(persistent[index], layout=layout)
        apparent_contrast = (
            apparent_quad[..., 0]
            - apparent_quad[..., 1]
            - apparent_quad[..., 2]
            + apparent_quad[..., 3]
        ) / 2.0
        persistent_contrast = (
            persistent_quad[..., 0]
            - persistent_quad[..., 1]
            - persistent_quad[..., 2]
            + persistent_quad[..., 3]
        ) / 2.0
        target_on_device = target.to(apparent_contrast)
        nonzero = target_on_device != 0.0
        layers.append(
            {
                "layer": index,
                "layout": layout,
                "literal_apparent_progress": _tensor_summary(literal[index]),
                "projected_apparent_progress": _tensor_summary(apparent[index]),
                "persistent_progress": _tensor_summary(persistent[index]),
                "apparent_full_conductance": _tensor_summary(apparent_g[index]),
                "persistent_full_conductance": _tensor_summary(persistent_g[index]),
                "apparent_logical_contrast": _tensor_summary(apparent_contrast),
                "persistent_logical_contrast": _tensor_summary(persistent_contrast),
                "apparent_contrast_rmse_to_target": float(
                    (apparent_contrast - target_on_device)
                    .to(torch.float64)
                    .square()
                    .mean()
                    .sqrt()
                    .item()
                ),
                "apparent_sign_flips_nonzero": int(
                    (
                        nonzero
                        & (torch.sign(apparent_contrast) != torch.sign(target_on_device))
                    )
                    .sum()
                    .item()
                ),
            }
        )
    return {
        "layers": layers,
        "apparent_projection": {
            "below_RESET": int(
                sum((value < 0.0).sum().item() for value in literal)
            ),
            "above_SET": int(
                sum((value > 1.0).sum().item() for value in literal)
            ),
        },
        "plant": dict(plant.report()),
    }


def _evaluate_plant(
    *,
    stack: Any,
    teacher: Any,
    loader: Iterable,
    plant: Figure6OmPulsePlant,
    sample_limit: int | None,
) -> Mapping[str, Any]:
    # The apparent evaluation is always the primary readout and is performed
    # first.  Persistent evaluation is an explicitly named, read-only ablation
    # on the identical loader/examples.
    _apply_full_g(stack, plant.apparent_full_conductance)
    apparent, apparent_prediction = _evaluate_detailed(
        stack, teacher, loader, sample_limit=sample_limit
    )
    _apply_full_g(stack, plant.full_conductance)
    persistent, persistent_prediction = _evaluate_detailed(
        stack, teacher, loader, sample_limit=sample_limit
    )
    return {
        "primary_state": "held_apparent_projected_progress",
        "apparent": apparent,
        "persistent_secondary_diagnostic": persistent,
        "same_examples_and_settings": True,
        "prediction_flips_apparent_vs_persistent": int(
            (apparent_prediction != persistent_prediction).sum().item()
        ),
    }


def _field_payload(field: EndpointField) -> Mapping[str, Any]:
    return {
        "reset": _snapshot(field.reset),
        "set": _snapshot(field.set),
        "report": field.report(),
    }


def _field_from_payload(
    payload: Mapping[str, Any], *, device: torch.device
) -> EndpointField:
    return EndpointField(
        reset=tuple(value.to(device=device, dtype=torch.float32) for value in payload["reset"]),
        set=tuple(value.to(device=device, dtype=torch.float32) for value in payload["set"]),
        shapes=_SHAPES,
        layouts=ENDPOINT_LAYOUTS,
        population_report=payload["report"]["population"],
    )


def _plant_from_state(
    *,
    field: EndpointField,
    population: IbmReramArrayPopulation,
    state: Mapping[str, Any],
) -> Figure6OmPulsePlant:
    seed = state.get("pulse_noise_seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Saved plant lacks an integer pulse-noise seed.")
    plant = Figure6OmPulsePlant(
        field,
        tuple(torch.zeros(shape, device=field.reset[0].device) for shape in _SHAPES),
        population,
        pulse_noise_seed=seed,
    )
    plant.load_state_dict(state)
    return plant


def _replica_suffix(array_index: int, write_index: int) -> int:
    if array_index not in {1, 2, 3} or write_index not in {1, 2, 3}:
        raise ValueError("Array and write indices must each be in {1,2,3}.")
    return 100 * array_index + write_index


def _replica_job(array_index: int, write_index: int) -> str:
    return f"array_{array_index}__write_{write_index}"


def _load_hwa_master(args: argparse.Namespace) -> tuple[Path, Mapping[str, Any]]:
    path = _artifact(
        args.campaign_root,
        "hwa-train",
        "main",
        "checkpoints/selected_hwa_master.pt",
    )
    return path, _load_torch(path, schema="ebl.figure6_om_256_hwa_master")


def run_deploy(args: argparse.Namespace) -> Path:
    config = args.config_payload
    array_index = int(args.array_index)
    write_index = int(args.write_index)
    suffix = _replica_suffix(array_index, write_index)
    job = _replica_job(array_index, write_index)
    prepared_path, prepared = _prepared(args)
    hwa_path, hwa = _load_hwa_master(args)
    teacher_path = Path(prepared["teacher"]["path"])
    published, repaired, population_report = _paired_om_populations(
        args, f"heldout_array_{array_index}"
    )
    store = _store(
        args,
        stage="deploy",
        job=job,
        extra={
            "mode": "clean_PV_then_exact_clone_and_post_PV_fault",
            "array_index": array_index,
            "write_index": write_index,
            "seed_suffix": suffix,
        },
        inputs=(
            _input("prepared", prepared_path),
            _input("HWA_master", hwa_path),
            _input("teacher", teacher_path),
            _input("OM_published", Path(population_report["published_path"])),
            _input("OM_repaired", Path(population_report["repaired_path"])),
        ),
    )
    try:
        spec, teacher, runtime = _runtime(
            config,
            teacher_path=teacher_path,
            gain=float(prepared["fixed_logit_gain"]),
            smoke=args.smoke,
        )
        stack = runtime["stack"]
        loaders = _loaders(spec)
        endpoint_seed = int(config["endpoint_model"]["heldout_seeds"][array_index - 1])
        field = _field(endpoint_seed, device=stack.device)
        masters = {
            "direct": _masters_to_device(prepared["direct_master"], stack.device),
            "hwa": _masters_to_device(hwa["selected_master"], stack.device),
            "scratch": _masters_to_device(prepared["scratch_master"], stack.device),
        }
        seeds = config["replication"]
        pv_seed = int(seeds["program_verify_seed_base"]) + suffix
        fault_seed = int(seeds["fault_observation_seed_base"]) + suffix
        adam_noise_seed = int(seeds["adam_pulse_noise_seed_base"]) + suffix
        target_payloads = {}
        for target_kind, master in masters.items():
            target_progress = master_to_progress(master, field)
            programmed, pv_result = program_masters_with_verify(
                master,
                field,
                repaired,
                pulse_noise_seed=pv_seed,
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
            clean_start = programmed.clone_for_new_pulse_phase(
                pulse_noise_seed=adam_noise_seed,
                retain_apparent_observation=True,
            )
            corrupt_start = programmed.clone_for_new_pulse_phase(
                pulse_noise_seed=adam_noise_seed,
                retain_apparent_observation=True,
            )
            if (
                not torch.equal(clean_start.raw_a, corrupt_start.raw_a)
                or not torch.equal(
                    clean_start.apparent_raw_a, corrupt_start.apparent_raw_a
                )
            ):
                raise RuntimeError("Clean and corrupt branches did not clone exact P&V state.")
            fault_receipt = corrupt_start.inject_post_pv_reset_stuck_faults(
                published.published_corrupt,
                observation_seed=fault_seed,
            )
            sample_limit = 64 if args.smoke else None
            clean_validation = _evaluate_plant(
                stack=stack,
                teacher=teacher,
                loader=loaders.validation,
                plant=clean_start,
                sample_limit=sample_limit,
            )
            corrupt_validation = _evaluate_plant(
                stack=stack,
                teacher=teacher,
                loader=loaders.validation,
                plant=corrupt_start,
                sample_limit=sample_limit,
            )
            open_test = target_kind in {"direct", "hwa"}
            clean_test = (
                _evaluate_plant(
                    stack=stack,
                    teacher=teacher,
                    loader=loaders.test,
                    plant=clean_start,
                    sample_limit=sample_limit,
                )
                if open_test
                else None
            )
            corrupt_test = (
                _evaluate_plant(
                    stack=stack,
                    teacher=teacher,
                    loader=loaders.test,
                    plant=corrupt_start,
                    sample_limit=sample_limit,
                )
                if open_test
                else None
            )
            target_payloads[target_kind] = {
                "logical_master": _snapshot(master),
                "logical_master_report": _master_report(master),
                "target_progress": _snapshot(target_progress),
                "program_verify": programming,
                "programmed_state": programmed.state_dict(),
                "clean_adam_start": clean_start.state_dict(),
                "corrupt_adam_start": corrupt_start.state_dict(),
                "fault_receipt": dict(fault_receipt),
                "clean_validation": clean_validation,
                "corrupt_validation": corrupt_validation,
                "clean_test": clean_test,
                "corrupt_test": corrupt_test,
                "clean_diagnostics": _state_diagnostics(
                    clean_start, target_master=master
                ),
                "corrupt_diagnostics": _state_diagnostics(
                    corrupt_start, target_master=master
                ),
            }
            print(
                f"deploy {job} {target_kind} apparent clean/corrupt val="
                f"{100.0*clean_validation['apparent']['student_accuracy']:.2f}%/"
                f"{100.0*corrupt_validation['apparent']['student_accuracy']:.2f}%",
                flush=True,
            )
        checkpoint_payload = {
            "schema": "ebl.figure6_om_256_deployment_replica",
            "schema_version": 1,
            "array_index": array_index,
            "write_index": write_index,
            "seed_suffix": suffix,
            "seeds": {
                "endpoint": endpoint_seed,
                "OM_assignment": repaired.assignment_seed,
                "program_verify": pv_seed,
                "fault_observation": fault_seed,
                "adam_pulse_noise": adam_noise_seed,
                "adam_selection": int(seeds["adam_selection_seed_base"]) + suffix,
                "data_order": int(seeds["data_order_seed_base"]) + suffix,
            },
            "fixed_logit_gain": float(prepared["fixed_logit_gain"]),
            "endpoint_field": _field_payload(field),
            "OM_population": population_report,
            "targets": target_payloads,
            "branching_proof": {
                "clean_PV_first": True,
                "clean_and_corrupt_PV_state_bitwise_identical_before_fault": True,
                "fault_injected_after_PV": True,
            },
        }
        checkpoint = atomic_torch_save(
            checkpoint_payload,
            store.run_dir / "checkpoints" / "deployment.pt",
        )
        summary = {
            "status": "complete",
            "stage": "deploy",
            "array_index": array_index,
            "write_index": write_index,
            "seed_suffix": suffix,
            "endpoint_seed": endpoint_seed,
            "OM_assignment_seed": repaired.assignment_seed,
            "fault_cells": int(published.published_corrupt.sum().item()),
            "fault_fraction": float(
                published.published_corrupt.to(torch.float64).mean().item()
            ),
            "arms": {
                "direct_pv": target_payloads["direct"]["clean_test"],
                "direct_pv_reset_stuck": target_payloads["direct"]["corrupt_test"],
                "hwa_pv": target_payloads["hwa"]["clean_test"],
                "hwa_pv_reset_stuck": target_payloads["hwa"]["corrupt_test"],
            },
            "scratch_test_opened": False,
            "primary_forward_state": "held_apparent",
            "persistent_forward_role": "secondary_diagnostic",
        }
        return _finish(store, summary=summary, artifacts=((checkpoint, "deployment_replica"),))
    except BaseException as error:
        store.fail(error)
        raise


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _adam_epoch_key(evaluation: Mapping[str, Any], epoch: int) -> tuple[float, float, int]:
    apparent = evaluation["apparent"]
    return (
        float(apparent["student_accuracy"]),
        -float(apparent["kl_teacher_student"]),
        -int(epoch),
    )


def _train_pulse_adam(
    *,
    config: Mapping[str, Any],
    plant: Figure6OmPulsePlant,
    target_master: Sequence[torch.Tensor],
    stack: Any,
    teacher: Any,
    loaders: Any,
    learning_rate: float,
    pulse_selection_seed: int,
    epochs: int,
    maximum_batches: int | None,
    evaluation_sample_limit: int | None,
    include_test: bool,
    metric_sink: Any,
    label: str,
) -> Mapping[str, Any]:
    optimizer = PersistentFigure6OmPulseAdam(
        plant,
        learning_rate_progress=float(learning_rate),
        pulse_cap=int(config["adam"]["pulse_cap_per_cell"]),
        pulse_selection_seed=int(pulse_selection_seed),
        beta1=float(config["adam"]["beta1"]),
        beta2=float(config["adam"]["beta2"]),
        epsilon=float(config["adam"]["epsilon"]),
    )
    initial_persistent = plant.raw_a.detach().clone()
    initial_apparent = plant.apparent_raw_a.detach().clone()
    before_validation = _evaluate_plant(
        stack=stack,
        teacher=teacher,
        loader=loaders.validation,
        plant=plant,
        sample_limit=evaluation_sample_limit,
    )
    history = []
    selected_key = None
    selected_epoch = None
    selected_state = None
    selected_optimizer = None
    selected_validation = None
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
        "probability_clipped": 0,
    }
    for epoch in range(1, int(epochs) + 1):
        epoch_start = dict(totals)
        epoch_batches = 0
        for batch_index, (inputs, labels) in enumerate(
            limited(loaders.train, maximum_batches), start=1
        ):
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            # Mandatory boundary: the gradient-producing DRN forward consumes
            # the held apparent state.  The hidden persistent state is never
            # supplied here.
            physical, metrics = _one_forward_gradient(
                stack=stack,
                teacher=teacher,
                inputs=inputs,
                labels=labels,
                full_g=plant.apparent_full_conductance,
            )
            clipped_before = optimizer.total_probability_clipped_cells
            pulse = optimizer.step(physical)
            totals["probability_clipped"] += (
                optimizer.total_probability_clipped_cells - clipped_before
            )
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
            epoch_batches += 1
            if batch_index % (1 if maximum_batches == 1 else 500) == 0:
                print(
                    f"{label} epoch={epoch}/{epochs} batch={batch_index} "
                    f"pulses={totals['pulsed']}",
                    flush=True,
                )
        if epoch_batches < 1:
            raise RuntimeError("Pulse Adam processed no minibatches.")
        epoch_examples = int(totals["examples"] - epoch_start["examples"])
        if maximum_batches is None and epoch_examples != 55_000:
            raise RuntimeError(
                "Expected one full 55,000-example pulse-Adam epoch; "
                f"observed {epoch_examples}."
            )
        if maximum_batches is not None and epoch_batches != maximum_batches:
            raise RuntimeError(
                f"Expected {maximum_batches} screened minibatches; "
                f"observed {epoch_batches}."
            )
        validation = _evaluate_plant(
            stack=stack,
            teacher=teacher,
            loader=loaders.validation,
            plant=plant,
            sample_limit=evaluation_sample_limit,
        )
        record = {
            "mode": "open_loop_OM_pulse_Adam",
            "label": label,
            "epoch": epoch,
            "forward_state": "current_held_apparent_state",
            "persistent_forward_used_for_gradient": False,
            "verify_reads_during_updates": 0,
            "examples": epoch_examples,
            "batches": epoch_batches,
            "kl_teacher_student": (
                float(totals["kl"] - epoch_start["kl"]) / epoch_examples
            ),
            "student_accuracy": (
                int(totals["correct"] - epoch_start["correct"]) / epoch_examples
            ),
            "teacher_agreement": (
                int(totals["agreement"] - epoch_start["agreement"]) / epoch_examples
            ),
            "applied_pulses": int(totals["pulsed"] - epoch_start["pulsed"]),
            "cumulative_applied_pulses": int(totals["pulsed"]),
            "validation": validation,
        }
        history.append(record)
        metric_sink(record)
        key = _adam_epoch_key(validation, epoch)
        if selected_key is None or key > selected_key:
            selected_key = key
            selected_epoch = epoch
            selected_state = _cpu_tree(plant.state_dict())
            selected_optimizer = _cpu_tree(optimizer.state_dict())
            selected_validation = validation
        print(
            f"{label} epoch={epoch}/{epochs} apparent_val="
            f"{100.0*float(validation['apparent']['student_accuracy']):.2f}% "
            f"persistent_diag_val="
            f"{100.0*float(validation['persistent_secondary_diagnostic']['student_accuracy']):.2f}%",
            flush=True,
        )
    if selected_state is None or selected_optimizer is None or selected_epoch is None:
        raise RuntimeError("Pulse Adam produced no selected apparent-state epoch.")
    final_state = _cpu_tree(plant.state_dict())
    final_validation = history[-1]["validation"]
    plant.load_state_dict(selected_state)
    selected_test = (
        _evaluate_plant(
            stack=stack,
            teacher=teacher,
            loader=loaders.test,
            plant=plant,
            sample_limit=evaluation_sample_limit,
        )
        if include_test
        else None
    )
    selected_pulse_count = int(plant.pulse_count.sum().item())
    if selected_pulse_count > plant.size * int(config["adam"]["pulse_cap_per_cell"]):
        raise RuntimeError("Pulse Adam exceeded its per-cell cap.")
    corrupt_initial = plant.corrupt & (initial_persistent == -1.0)
    if bool(torch.any(corrupt_initial)) and not torch.equal(
        plant.raw_a[corrupt_initial], initial_persistent[corrupt_initial]
    ):
        raise RuntimeError("A post-P&V reset-stuck cell moved persistently.")
    return {
        "learning_rate_progress": float(learning_rate),
        "pulse_selection_seed": int(pulse_selection_seed),
        "forward_contract": {
            "gradient_and_selection": "held_apparent_state",
            "persistent": "write_authority_and_secondary_diagnostic_only",
            "apparent_redraw_per_example": False,
            "apparent_refresh": "only_after_touched_device_write",
            "verify_reads_during_updates": 0,
        },
        "initial_persistent_sha256": _tensor_sha256(initial_persistent),
        "initial_apparent_sha256": _tensor_sha256(initial_apparent),
        "before_validation": before_validation,
        "history": history,
        "selected_epoch": selected_epoch,
        "selected_validation": selected_validation,
        "selected_test": selected_test,
        "selected_plant_state": selected_state,
        "selected_optimizer_state": selected_optimizer,
        "selected_state_diagnostics": _state_diagnostics(
            plant, target_master=target_master
        ),
        "selected_applied_pulses": selected_pulse_count,
        "final_plant_state": final_state,
        "final_validation": final_validation,
        "training_totals": totals,
        "test_opened_only_after_selection": include_test,
    }


def run_adam_screen_prepare(args: argparse.Namespace) -> Path:
    config = args.config_payload
    prepared_path, prepared = _prepared(args)
    hwa_path, hwa = _load_hwa_master(args)
    teacher_path = Path(prepared["teacher"]["path"])
    published, repaired, population_report = _paired_om_populations(args, "development")
    store = _store(
        args,
        stage="adam-screen",
        job="p0",
        extra={"mode": "prepare_matched_development_P0_states"},
        inputs=(
            _input("prepared", prepared_path),
            _input("HWA_master", hwa_path),
            _input("OM_published", Path(population_report["published_path"])),
            _input("OM_repaired", Path(population_report["repaired_path"])),
        ),
    )
    try:
        spec, teacher, runtime = _runtime(
            config,
            teacher_path=teacher_path,
            gain=float(prepared["fixed_logit_gain"]),
            smoke=args.smoke,
        )
        stack = runtime["stack"]
        field = _field(
            int(config["endpoint_model"]["adam_development_seed"]),
            device=stack.device,
        )
        masters = {
            "hwa": _masters_to_device(hwa["selected_master"], stack.device),
            "scratch": _masters_to_device(prepared["scratch_master"], stack.device),
        }
        adam = config["adam"]
        payload = {
            "schema": "ebl.figure6_om_256_adam_screen_p0",
            "schema_version": 1,
            "endpoint_field": _field_payload(field),
            "OM_population": population_report,
            "families": {},
        }
        for family, master in masters.items():
            streams = {}
            for write_index in range(1, int(adam["screen_write_streams"]) + 1):
                pv_seed = int(adam["development_program_verify_seed_base"]) + write_index
                programmed, result = program_masters_with_verify(
                    master,
                    field,
                    repaired,
                    pulse_noise_seed=pv_seed,
                    tolerance_progress=OM_PROGRAM_VERIFY_TOLERANCE_PROGRESS,
                    maximum_pulses=OM_PROGRAM_VERIFY_MAXIMUM_PULSES,
                    noisy_initial_reset_verify=True,
                    preaccept_exact_healthy_reset_targets=True,
                )
                clean = programmed.clone_for_new_pulse_phase(
                    pulse_noise_seed=int(adam["development_pulse_noise_seed_base"])
                    + write_index,
                    retain_apparent_observation=True,
                )
                corrupt = programmed.clone_for_new_pulse_phase(
                    pulse_noise_seed=int(adam["development_pulse_noise_seed_base"])
                    + write_index,
                    retain_apparent_observation=True,
                )
                if not torch.equal(clean.raw_a, corrupt.raw_a) or not torch.equal(
                    clean.apparent_raw_a, corrupt.apparent_raw_a
                ):
                    raise RuntimeError("Adam-screen branches did not clone exact P&V state.")
                fault_receipt = corrupt.inject_post_pv_reset_stuck_faults(
                    published.published_corrupt,
                    observation_seed=int(adam["development_fault_observation_seed_base"])
                    + write_index,
                )
                streams[str(write_index)] = {
                    "program_verify": _program_verify_report(
                        target_progress=master_to_progress(master, field),
                        plant=programmed,
                        result=result,
                    ),
                    "clean_state": clean.state_dict(),
                    "corrupt_state": corrupt.state_dict(),
                    "fault_receipt": dict(fault_receipt),
                }
            payload["families"][family] = {
                "logical_master": _snapshot(master),
                "streams": streams,
            }
        checkpoint = atomic_torch_save(
            payload, store.run_dir / "checkpoints" / "screen_p0.pt"
        )
        summary = {
            "status": "complete",
            "stage": "adam-screen",
            "job": "p0",
            "families": list(_FAMILIES),
            "write_streams": int(adam["screen_write_streams"]),
            "clean_PV_before_fault": True,
            "test_opened": False,
        }
        return _finish(store, summary=summary, artifacts=((checkpoint, "Adam_screen_P0"),))
    except BaseException as error:
        store.fail(error)
        raise


def _rate_token(rate: float) -> str:
    return format(float(rate), ".0e").replace("-", "m").replace("+", "p")


def _adam_screen_candidate_job(family: str, rate: float) -> str:
    return f"{family}__lr_{_rate_token(rate)}"


def run_adam_screen_candidate(args: argparse.Namespace) -> Path:
    config = args.config_payload
    family = args.family
    rate = float(args.learning_rate)
    if family not in _FAMILIES:
        raise ValueError(f"Expected family in {_FAMILIES!r}.")
    if rate not in config["adam"]["learning_rate_grid"]:
        raise ValueError("Adam learning rate is outside the strict grid.")
    prepared_path, prepared = _prepared(args)
    p0_path = _artifact(
        args.campaign_root, "adam-screen", "p0", "checkpoints/screen_p0.pt"
    )
    p0 = _load_torch(p0_path, schema="ebl.figure6_om_256_adam_screen_p0")
    teacher_path = Path(prepared["teacher"]["path"])
    _published, repaired, population_report = _paired_om_populations(args, "development")
    job = _adam_screen_candidate_job(family, rate)
    store = _store(
        args,
        stage="adam-screen",
        job=job,
        extra={"mode": "Adam_rate_candidate", "family": family, "learning_rate": rate},
        inputs=(
            _input("prepared", prepared_path),
            _input("Adam_screen_P0", p0_path),
            _input("teacher", teacher_path),
            _input("OM_repaired", Path(population_report["repaired_path"])),
        ),
    )
    try:
        spec, teacher, runtime = _runtime(
            config,
            teacher_path=teacher_path,
            gain=float(prepared["fixed_logit_gain"]),
            smoke=args.smoke,
        )
        field = _field_from_payload(p0["endpoint_field"], device=runtime["stack"].device)
        master = _masters_to_device(
            p0["families"][family]["logical_master"], runtime["stack"].device
        )
        reports = []
        for write_index in range(1, int(config["adam"]["screen_write_streams"]) + 1):
            stream = p0["families"][family]["streams"][str(write_index)]
            for fault in _FAULTS:
                state = stream["clean_state"] if fault == "clean" else stream["corrupt_state"]
                plant = _plant_from_state(field=field, population=repaired, state=state)
                loaders = _loaders(
                    spec,
                    data_seed=int(config["adam"]["development_data_order_seed_base"])
                    + write_index,
                )
                report = _train_pulse_adam(
                    config=config,
                    plant=plant,
                    target_master=master,
                    stack=runtime["stack"],
                    teacher=teacher,
                    loaders=loaders,
                    learning_rate=rate,
                    pulse_selection_seed=int(
                        config["adam"]["development_selection_seed_base"]
                    )
                    + write_index,
                    epochs=1,
                    maximum_batches=1 if args.smoke else int(config["adam"]["screen_batches"]),
                    evaluation_sample_limit=64 if args.smoke else None,
                    include_test=False,
                    metric_sink=store.append_metric,
                    label=f"screen/{family}/{fault}/write_{write_index}/lr_{rate}",
                )
                reports.append(
                    {
                        "write_index": write_index,
                        "fault": fault,
                        "after_validation": report["selected_validation"],
                        "applied_pulses": report["selected_applied_pulses"],
                    }
                )
                del plant, loaders, report
                gc.collect()
                torch.cuda.empty_cache()
        macro_accuracy = sum(
            float(item["after_validation"]["apparent"]["student_accuracy"])
            for item in reports
        ) / len(reports)
        macro_kl = sum(
            float(item["after_validation"]["apparent"]["kl_teacher_student"])
            for item in reports
        ) / len(reports)
        total_pulses = sum(int(item["applied_pulses"]) for item in reports)
        candidate = {
            "family": family,
            "learning_rate": rate,
            "macro_apparent_validation_accuracy": macro_accuracy,
            "macro_apparent_validation_KL": macro_kl,
            "total_selected_pulses": total_pulses,
            "replicas": reports,
            "test_opened": False,
        }
        candidate_path = store.run_dir / "artifacts" / "candidate.json"
        atomic_write_json(candidate_path, candidate)
        summary = {"status": "complete", "stage": "adam-screen", "job": job, **candidate}
        return _finish(store, summary=summary, artifacts=((candidate_path, "Adam_rate_candidate"),))
    except BaseException as error:
        store.fail(error)
        raise


def run_adam_screen_select(args: argparse.Namespace) -> Path:
    family = args.family
    if family not in _FAMILIES:
        raise ValueError(f"Expected family in {_FAMILIES!r}.")
    paths = []
    candidates = []
    for rate in args.config_payload["adam"]["learning_rate_grid"]:
        job = _adam_screen_candidate_job(family, float(rate))
        path = _artifact(args.campaign_root, "adam-screen", job, "artifacts/candidate.json")
        paths.append(path)
        candidates.append(json.loads(path.read_text(encoding="utf-8")))
    selected = max(
        candidates,
        key=lambda item: (
            float(item["macro_apparent_validation_accuracy"]),
            -float(item["macro_apparent_validation_KL"]),
            -int(item["total_selected_pulses"]),
            -float(item["learning_rate"]),
        ),
    )
    store = _store(
        args,
        stage="adam-screen",
        job=f"selection_{family}",
        extra={"mode": "Adam_rate_selection", "family": family},
        inputs=tuple(_input("Adam_rate_candidate", path) for path in paths),
    )
    try:
        receipt = {
            "schema": "ebl.figure6_om_256_adam_rate_receipt",
            "schema_version": 1,
            "family": family,
            "selection_state": "held_apparent_validation",
            "selected": selected,
            "candidates": candidates,
            "test_opened": False,
        }
        path = store.run_dir / "artifacts" / "rate_receipt.json"
        atomic_write_json(path, receipt)
        summary = {
            "status": "complete",
            "stage": "adam-screen",
            "job": f"selection_{family}",
            "family": family,
            "selected_learning_rate": selected["learning_rate"],
            "selection": selected,
            "test_opened": False,
        }
        return _finish(store, summary=summary, artifacts=((path, "Adam_rate_receipt"),))
    except BaseException as error:
        store.fail(error)
        raise


def _adam_arm_job(
    family: str, fault: str, array_index: int, write_index: int
) -> str:
    return f"{family}__{fault}__{_replica_job(array_index, write_index)}"


def run_adam_arm(args: argparse.Namespace) -> Path:
    config = args.config_payload
    family = args.family
    fault = args.fault
    array_index = int(args.array_index)
    write_index = int(args.write_index)
    if family not in _FAMILIES or fault not in _FAULTS:
        raise ValueError("Invalid Adam family or fault branch.")
    _replica_suffix(array_index, write_index)
    prepared_path, prepared = _prepared(args)
    deploy_path = _artifact(
        args.campaign_root,
        "deploy",
        _replica_job(array_index, write_index),
        "checkpoints/deployment.pt",
    )
    deployment = _load_torch(
        deploy_path, schema="ebl.figure6_om_256_deployment_replica"
    )
    receipt_path = _artifact(
        args.campaign_root,
        "adam-screen",
        f"selection_{family}",
        "artifacts/rate_receipt.json",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "ebl.figure6_om_256_adam_rate_receipt"
        or receipt.get("family") != family
    ):
        raise ValueError("Adam rate receipt does not match the requested family.")
    learning_rate = float(receipt["selected"]["learning_rate"])
    teacher_path = Path(prepared["teacher"]["path"])
    _published, repaired, population_report = _paired_om_populations(
        args, f"heldout_array_{array_index}"
    )
    job = _adam_arm_job(family, fault, array_index, write_index)
    store = _store(
        args,
        stage="adam-arm",
        job=job,
        extra={
            "mode": "main_10_epoch_open_loop_OM_pulse_Adam",
            "family": family,
            "fault": fault,
            "array_index": array_index,
            "write_index": write_index,
            "learning_rate": learning_rate,
        },
        inputs=(
            _input("prepared", prepared_path),
            _input("deployment", deploy_path),
            _input("Adam_rate_receipt", receipt_path),
            _input("teacher", teacher_path),
            _input("OM_repaired", Path(population_report["repaired_path"])),
        ),
    )
    try:
        spec, teacher, runtime = _runtime(
            config,
            teacher_path=teacher_path,
            gain=float(prepared["fixed_logit_gain"]),
            smoke=args.smoke,
        )
        target = deployment["targets"][family]
        master = _masters_to_device(target["logical_master"], runtime["stack"].device)
        field = _field_from_payload(
            deployment["endpoint_field"], device=runtime["stack"].device
        )
        state = (
            target["clean_adam_start"]
            if fault == "clean"
            else target["corrupt_adam_start"]
        )
        plant = _plant_from_state(field=field, population=repaired, state=state)
        loaders = _loaders(spec, data_seed=int(deployment["seeds"]["data_order"]))
        report = _train_pulse_adam(
            config=config,
            plant=plant,
            target_master=master,
            stack=runtime["stack"],
            teacher=teacher,
            loaders=loaders,
            learning_rate=learning_rate,
            pulse_selection_seed=int(deployment["seeds"]["adam_selection"]),
            epochs=1 if args.smoke else int(config["adam"]["epochs"]),
            maximum_batches=1 if args.smoke else None,
            evaluation_sample_limit=64 if args.smoke else None,
            include_test=True,
            metric_sink=store.append_metric,
            label=job,
        )
        checkpoint = atomic_torch_save(
            {
                "schema": "ebl.figure6_om_256_adam_arm",
                "schema_version": 1,
                "family": family,
                "fault": fault,
                "array_index": array_index,
                "write_index": write_index,
                "source_deployment": {
                    "path": str(deploy_path),
                    "sha256": sha256_file(deploy_path),
                },
                "learning_rate_receipt": {
                    "path": str(receipt_path),
                    "sha256": sha256_file(receipt_path),
                },
                "report": report,
            },
            store.run_dir / "checkpoints" / "selected_adam_state.pt",
        )
        selected = report["selected_validation"]["apparent"]
        summary = {
            "status": "complete",
            "stage": "adam-arm",
            "job": job,
            "family": family,
            "fault": fault,
            "array_index": array_index,
            "write_index": write_index,
            "learning_rate": learning_rate,
            "selected_epoch": report["selected_epoch"],
            "selected_apparent_validation": selected,
            "selected_test": report["selected_test"],
            "selected_applied_pulses": report["selected_applied_pulses"],
            "primary_forward_state": "held_apparent",
            "persistent_forward_role": "secondary_diagnostic",
            "test_opened_only_after_selection": True,
        }
        return _finish(store, summary=summary, artifacts=((checkpoint, "selected_Adam_arm"),))
    except BaseException as error:
        store.fail(error)
        raise


def _evaluation_metric(
    evaluation: Mapping[str, Any], state: str, metric: str
) -> float:
    key = "apparent" if state == "apparent" else "persistent_secondary_diagnostic"
    return float(evaluation[key][metric])


def _aggregate_values(records: Sequence[Mapping[str, Any]], value_key: str) -> Mapping[str, Any]:
    array_means = []
    for array_index in (1, 2, 3):
        values = [
            float(record[value_key])
            for record in records
            if int(record["array_index"]) == array_index
        ]
        if len(values) != 3:
            raise RuntimeError(
                f"Expected three writes for array {array_index}; observed {len(values)}."
            )
        array_means.append(sum(values) / len(values))
    return {
        "array_means": array_means,
        "mean_across_arrays": sum(array_means) / len(array_means),
        "full_range_across_array_means": [min(array_means), max(array_means)],
    }


def _aggregate_arm(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    enriched = []
    for record in records:
        evaluation = record["evaluation"]
        item = dict(record)
        for state in ("apparent", "persistent"):
            for metric in (
                "student_accuracy",
                "teacher_agreement",
                "kl_teacher_student",
            ):
                item[f"{state}_{metric}"] = _evaluation_metric(
                    evaluation, state, metric
                )
        enriched.append(item)
    return {
        key: _aggregate_values(enriched, key)
        for key in (
            "apparent_student_accuracy",
            "apparent_teacher_agreement",
            "apparent_kl_teacher_student",
            "persistent_student_accuracy",
            "persistent_teacher_agreement",
            "persistent_kl_teacher_student",
        )
    }


def _paired_delta(
    by_replica: Mapping[tuple[int, int, str], Mapping[str, Any]],
    *,
    left: str,
    right: str,
) -> Mapping[str, Any]:
    records = []
    for array_index in (1, 2, 3):
        for write_index in (1, 2, 3):
            left_eval = by_replica[(array_index, write_index, left)]["evaluation"]
            right_eval = by_replica[(array_index, write_index, right)]["evaluation"]
            records.append(
                {
                    "array_index": array_index,
                    "write_index": write_index,
                    "accuracy_delta_right_minus_left": (
                        _evaluation_metric(right_eval, "apparent", "student_accuracy")
                        - _evaluation_metric(left_eval, "apparent", "student_accuracy")
                    ),
                    "KL_delta_right_minus_left": (
                        _evaluation_metric(right_eval, "apparent", "kl_teacher_student")
                        - _evaluation_metric(left_eval, "apparent", "kl_teacher_student")
                    ),
                }
            )
    return {
        "left": left,
        "right": right,
        "apparent_accuracy_delta_right_minus_left": _aggregate_values(
            records, "accuracy_delta_right_minus_left"
        ),
        "apparent_KL_delta_right_minus_left": _aggregate_values(
            records, "KL_delta_right_minus_left"
        ),
    }


def run_summarize(args: argparse.Namespace) -> Path:
    input_paths = []
    records = []
    for array_index in (1, 2, 3):
        for write_index in (1, 2, 3):
            deploy_summary_path = _artifact(
                args.campaign_root,
                "deploy",
                _replica_job(array_index, write_index),
                "scientific_summary.json",
            )
            input_paths.append(deploy_summary_path)
            deployment = json.loads(deploy_summary_path.read_text(encoding="utf-8"))
            for arm in (
                "direct_pv",
                "direct_pv_reset_stuck",
                "hwa_pv",
                "hwa_pv_reset_stuck",
            ):
                records.append(
                    {
                        "array_index": array_index,
                        "write_index": write_index,
                        "arm": arm,
                        "evaluation": deployment["arms"][arm],
                        "learning_rate": None,
                        "selected_epoch": None,
                    }
                )
            for family in _FAMILIES:
                for fault in _FAULTS:
                    job = _adam_arm_job(
                        family, fault, array_index, write_index
                    )
                    path = _artifact(
                        args.campaign_root,
                        "adam-arm",
                        job,
                        "scientific_summary.json",
                    )
                    input_paths.append(path)
                    summary = json.loads(path.read_text(encoding="utf-8"))
                    arm = f"{family}_pv_"
                    if fault == "reset_stuck":
                        arm += "reset_stuck_"
                    arm += "adam"
                    records.append(
                        {
                            "array_index": array_index,
                            "write_index": write_index,
                            "arm": arm,
                            "evaluation": summary["selected_test"],
                            "learning_rate": summary["learning_rate"],
                            "selected_epoch": summary["selected_epoch"],
                            "selected_applied_pulses": summary[
                                "selected_applied_pulses"
                            ],
                        }
                    )
    expected_arms = tuple(args.config_payload["arms"])
    by_arm = {
        arm: [record for record in records if record["arm"] == arm]
        for arm in expected_arms
    }
    if any(len(values) != 9 for values in by_arm.values()):
        raise RuntimeError("Summary does not contain nine records for every arm.")
    by_replica = {
        (int(record["array_index"]), int(record["write_index"]), str(record["arm"])): record
        for record in records
    }
    comparisons = {
        "direct_post_pv_fault": _paired_delta(
            by_replica, left="direct_pv", right="direct_pv_reset_stuck"
        ),
        "hwa_post_pv_fault": _paired_delta(
            by_replica, left="hwa_pv", right="hwa_pv_reset_stuck"
        ),
        "hwa_vs_direct_clean": _paired_delta(
            by_replica, left="direct_pv", right="hwa_pv"
        ),
        "hwa_vs_direct_faulted": _paired_delta(
            by_replica,
            left="direct_pv_reset_stuck",
            right="hwa_pv_reset_stuck",
        ),
        "hwa_recovery_clean": _paired_delta(
            by_replica, left="hwa_pv", right="hwa_pv_adam"
        ),
        "hwa_recovery_faulted": _paired_delta(
            by_replica,
            left="hwa_pv_reset_stuck",
            right="hwa_pv_reset_stuck_adam",
        ),
        "hwa_init_vs_scratch_clean_after_10_epochs": _paired_delta(
            by_replica, left="scratch_pv_adam", right="hwa_pv_adam"
        ),
        "hwa_init_vs_scratch_faulted_after_10_epochs": _paired_delta(
            by_replica,
            left="scratch_pv_reset_stuck_adam",
            right="hwa_pv_reset_stuck_adam",
        ),
        "residual_corruption_after_hwa_recovery": _paired_delta(
            by_replica,
            left="hwa_pv_adam",
            right="hwa_pv_reset_stuck_adam",
        ),
        "residual_corruption_after_scratch_training": _paired_delta(
            by_replica,
            left="scratch_pv_adam",
            right="scratch_pv_reset_stuck_adam",
        ),
    }
    rate_receipts = {}
    for family in _FAMILIES:
        path = _artifact(
            args.campaign_root,
            "adam-screen",
            f"selection_{family}",
            "artifacts/rate_receipt.json",
        )
        input_paths.append(path)
        rate_receipts[family] = json.loads(path.read_text(encoding="utf-8"))[
            "selected"
        ]["learning_rate"]
    store = _store(
        args,
        stage="summarize",
        job="main",
        extra={"mode": "paired_hierarchical_analysis"},
        inputs=tuple(_input("completed_arm_or_receipt", path) for path in input_paths),
    )
    try:
        records_path = store.run_dir / "artifacts" / "raw_records.json"
        atomic_write_json(records_path, records)
        summary = {
            "status": "complete",
            "stage": "summarize",
            "evidence_tier": "exploratory_noncanonical",
            "claim_scope": "model_based_Figure6_endpoint_plus_IBM_OM_pulse_hybrid",
            "records": len(records),
            "replication": {
                "arrays": 3,
                "writes_per_array": 3,
                "raw_records_per_arm": 9,
                "aggregation": (
                    "average_three_writes_within_array_then_mean_and_full_range_"
                    "across_three_array_means"
                ),
            },
            "primary_state": "held_apparent",
            "persistent_state": "secondary_diagnostic_only",
            "arms": {
                arm: _aggregate_arm(values) for arm, values in by_arm.items()
            },
            "paired_comparisons": comparisons,
            "selected_Adam_learning_rates": rate_receipts,
            "learning_rate_comparison_qualification": (
                "HWA and scratch are each best-tuned; their comparison is not a "
                "same-learning-rate causal contrast"
            ),
            "limitations": [
                "exploratory_noncanonical",
                "Figure-6 endpoint marginals are analyst-defined model inputs",
                "IBM OM pulse equations are simulated rather than measured array runs",
                "post-PV reset-stuck faults are a synthetic intervention using the paired published-OM mask",
                "digital gradients and Adam moments are computed off-chip",
                "no physical-power or autonomous-on-chip-learning claim is supported",
            ],
        }
        return _finish(store, summary=summary, artifacts=((records_path, "raw_paired_records"),))
    except BaseException as error:
        store.fail(error)
        raise


def _has_complete(campaign_root: Path, stage: str, job: str) -> bool:
    try:
        _completed_run(campaign_root, stage, job)
        return True
    except FileNotFoundError:
        return False


def _teacher_checkpoint(campaign_root: Path) -> Path:
    root = campaign_root / "teacher"
    candidates = sorted(
        root.glob("*/result.json") if root.is_dir() else (),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for result_path in candidates:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        selected = result.get("metrics", {}).get("selected", {})
        gate = result.get("metrics", {}).get("acceptance_gate", {})
        checkpoint = result_path.parent / "checkpoints" / "weights.pt"
        if (
            result.get("status") == "complete"
            and bool(gate.get("passed"))
            and float(selected.get("accuracy", 0.0)) >= 0.97
            and checkpoint.is_file()
        ):
            return checkpoint
    raise FileNotFoundError("No accepted 784-256-10 teacher checkpoint exists.")


def _source_is_clean() -> bool:
    completed = subprocess.run(
        ("git", "-C", str(_ROOT), "status", "--porcelain=v1", "--untracked-files=all"),
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _launcher_status(path: Path, **values: Any) -> None:
    atomic_write_json(
        path,
        {
            "schema": "ebl.figure6_om_256_ladder_launcher_status",
            "schema_version": 1,
            "pid": os.getpid(),
            **values,
        },
    )


def run_launch(args: argparse.Namespace) -> Path:
    if not args.smoke and not _source_is_clean():
        raise RuntimeError(
            "Full launch requires a clean frozen source revision. Commit or stash "
            "the implementation before launch."
        )
    args.campaign_root.mkdir(parents=True, exist_ok=True)
    status_path = args.campaign_root / "launcher_status.json"
    revision = subprocess.check_output(
        ("git", "-C", str(_ROOT), "rev-parse", "HEAD"), text=True
    ).strip()
    monitoring_contract = {
        "schema": "ebl.figure6_om_256_ladder_monitoring_contract",
        "schema_version": 1,
        "evidence_tier": "exploratory_noncanonical",
        "target": "local_RTX_3090_CUDA_only",
        "launcher": "one_sequential_persistent_parent_process",
        "launcher_pid": os.getpid(),
        "source_revision": revision,
        "source_clean_at_full_launch": _source_is_clean(),
        "config_path": str(args.config),
        "config_sha256": sha256_file(args.config),
        "coverage": {
            "teacher": 1,
            "HWA_rate_candidates": 9,
            "HWA_main_runs": 1,
            "deployment_replicas": 9,
            "deployment_targets_per_replica": 3,
            "Adam_rate_candidates": 10,
            "Adam_main_arms": 36,
            "final_raw_records": 72,
        },
        "paths": {
            "launcher_status": str(status_path),
            "stage_root": str(args.campaign_root / "stages"),
            "terminal_summary_glob": str(
                args.campaign_root / "stages/summarize/main/*/scientific_summary.json"
            ),
        },
        "semantic_progress": (
            "RunStore status/result files, per-epoch metrics.jsonl growth, "
            "stage receipts, and launcher_status.current transitions"
        ),
        "expected_progress_interval": "less_than_30_minutes_during_GPU_work",
        "stale_threshold_minutes": 45,
        "expected_runtime": "approximately_one_day_subject_to_measured_throughput",
        "safe_retry": (
            "relaunch exact command; completed immutable stage jobs are skipped "
            "and failed attempts are preserved"
        ),
    }
    atomic_write_json(
        args.campaign_root / "monitoring_contract.json", monitoring_contract
    )
    teacher_path = args.teacher_weights
    if teacher_path is None:
        try:
            teacher_path = _teacher_checkpoint(args.campaign_root)
        except FileNotFoundError:
            if args.smoke:
                raise RuntimeError(
                    "Smoke launch requires --teacher-weights or an accepted campaign teacher."
                )
            teacher_config = _ROOT / str(args.config_payload["teacher"]["config"])
            command = [
                str(args.python),
                "-m",
                "ebl",
                "train",
                "--config",
                str(teacher_config),
                "--output-dir",
                str(args.campaign_root / "teacher"),
            ]
            _launcher_status(
                status_path,
                status="running",
                current="teacher",
                command=command,
            )
            subprocess.run(command, cwd=_ROOT, check=True)
            teacher_path = _teacher_checkpoint(args.campaign_root)
    teacher_path = teacher_path.expanduser().resolve()

    base = [
        str(args.python),
        "-m",
        "experiments.mnist_relu_drn.figure6_om_256_ladder",
        "--config",
        str(args.config),
        "--campaign-root",
        str(args.campaign_root),
    ]
    if args.smoke:
        base.append("--smoke")

    def invoke(stage: str, job: str, tail: Sequence[str]) -> None:
        if _has_complete(args.campaign_root, stage, job):
            print(f"skip complete {stage}/{job}", flush=True)
            return
        command = [*base, *tail]
        _launcher_status(
            status_path,
            status="running",
            current=f"{stage}/{job}",
            command=command,
        )
        subprocess.run(command, cwd=_ROOT, check=True)

    try:
        invoke(
            "prepare",
            "main",
            (
                "prepare",
                "--teacher-weights",
                str(teacher_path),
                "--aihwkit-python",
                str(args.aihwkit_python),
            ),
        )
        for input_multiplier in args.config_payload["hwa"]["input_multipliers"]:
            for output_multiplier in args.config_payload["hwa"]["output_multipliers"]:
                invoke(
                    "hwa-screen",
                    _hwa_candidate_job(input_multiplier, output_multiplier),
                    (
                        "hwa-screen",
                        "--input-multiplier",
                        str(input_multiplier),
                        "--output-multiplier",
                        str(output_multiplier),
                    ),
                )
        invoke("hwa-screen", "selection", ("hwa-screen", "--select"))
        invoke("hwa-train", "main", ("hwa-train",))
        for array_index in (1, 2, 3):
            for write_index in (1, 2, 3):
                invoke(
                    "deploy",
                    _replica_job(array_index, write_index),
                    (
                        "deploy",
                        "--array-index",
                        str(array_index),
                        "--write-index",
                        str(write_index),
                    ),
                )
        invoke("adam-screen", "p0", ("adam-screen", "--action", "prepare-p0"))
        for family in _FAMILIES:
            for rate in args.config_payload["adam"]["learning_rate_grid"]:
                invoke(
                    "adam-screen",
                    _adam_screen_candidate_job(family, rate),
                    (
                        "adam-screen",
                        "--action",
                        "candidate",
                        "--family",
                        family,
                        "--learning-rate",
                        str(rate),
                    ),
                )
            invoke(
                "adam-screen",
                f"selection_{family}",
                ("adam-screen", "--action", "select", "--family", family),
            )
        for family in _FAMILIES:
            for fault in _FAULTS:
                for array_index in (1, 2, 3):
                    for write_index in (1, 2, 3):
                        invoke(
                            "adam-arm",
                            _adam_arm_job(
                                family, fault, array_index, write_index
                            ),
                            (
                                "adam-arm",
                                "--family",
                                family,
                                "--fault",
                                fault,
                                "--array-index",
                                str(array_index),
                                "--write-index",
                                str(write_index),
                            ),
                        )
        invoke("summarize", "main", ("summarize",))
        summary_path = _artifact(
            args.campaign_root, "summarize", "main", "scientific_summary.json"
        )
        _launcher_status(
            status_path,
            status="complete",
            current=None,
            summary=str(summary_path),
        )
        return summary_path
    except BaseException as error:
        _launcher_status(
            status_path,
            status="failed",
            current=json.loads(status_path.read_text(encoding="utf-8")).get("current"),
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--campaign-root", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a reduced exploratory_noncanonical numerical canary.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--teacher-weights", type=Path, required=True)
    prepare.add_argument(
        "--aihwkit-python",
        type=Path,
        default=Path("/home/filip/miniconda3/envs/aihwkit/bin/python"),
    )

    hwa_screen = subparsers.add_parser("hwa-screen")
    hwa_screen.add_argument("--input-multiplier", type=float)
    hwa_screen.add_argument("--output-multiplier", type=float)
    hwa_screen.add_argument("--select", action="store_true")

    subparsers.add_parser("hwa-train")

    deploy = subparsers.add_parser("deploy")
    deploy.add_argument("--array-index", type=int, required=True)
    deploy.add_argument("--write-index", type=int, required=True)

    adam_screen = subparsers.add_parser("adam-screen")
    adam_screen.add_argument(
        "--action", choices=("prepare-p0", "candidate", "select"), required=True
    )
    adam_screen.add_argument("--family", choices=_FAMILIES)
    adam_screen.add_argument("--learning-rate", type=float)

    adam_arm = subparsers.add_parser("adam-arm")
    adam_arm.add_argument("--family", choices=_FAMILIES, required=True)
    adam_arm.add_argument("--fault", choices=_FAULTS, required=True)
    adam_arm.add_argument("--array-index", type=int, required=True)
    adam_arm.add_argument("--write-index", type=int, required=True)

    subparsers.add_parser("summarize")

    launch = subparsers.add_parser("launch")
    launch.add_argument("--teacher-weights", type=Path)
    launch.add_argument(
        "--aihwkit-python",
        type=Path,
        default=Path("/home/filip/miniconda3/envs/aihwkit/bin/python"),
    )
    launch.add_argument(
        "--python",
        type=Path,
        default=Path("/home/filip/miniconda3/envs/py312/bin/python"),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.config = args.config.expanduser().resolve()
    args.campaign_root = args.campaign_root.expanduser().resolve()
    args.config_payload = load_ladder_config(args.config)
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "hwa-screen":
        if args.select:
            if args.input_multiplier is not None or args.output_multiplier is not None:
                raise ValueError("HWA selection does not accept candidate multipliers.")
            run_hwa_screen_select(args)
        else:
            if args.input_multiplier is None or args.output_multiplier is None:
                raise ValueError("HWA candidate requires both multipliers.")
            run_hwa_screen_candidate(args)
    elif args.command == "hwa-train":
        run_hwa_train(args)
    elif args.command == "deploy":
        run_deploy(args)
    elif args.command == "adam-screen":
        if args.action == "prepare-p0":
            if args.family is not None or args.learning_rate is not None:
                raise ValueError("Adam P0 preparation accepts no family or rate.")
            run_adam_screen_prepare(args)
        elif args.action == "candidate":
            if args.family is None or args.learning_rate is None:
                raise ValueError("Adam candidate requires family and learning rate.")
            run_adam_screen_candidate(args)
        else:
            if args.family is None or args.learning_rate is not None:
                raise ValueError("Adam rate selection requires only family.")
            run_adam_screen_select(args)
    elif args.command == "adam-arm":
        run_adam_arm(args)
    elif args.command == "summarize":
        run_summarize(args)
    elif args.command == "launch":
        run_launch(args)
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
