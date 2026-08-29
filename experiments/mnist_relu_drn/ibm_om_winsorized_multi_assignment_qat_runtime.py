"""CUDA runtime for deterministic multi-assignment Winsorized IBM OM QAT."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np
import torch

from experiments.artifacts import RunStore, atomic_write_json, sha256_file
from experiments.mnist_relu_drn.components import build_student_stack
from experiments.mnist_relu_drn.ibm_om_baseline_selection_runtime import (
    _evaluate_detailed,
    _input,
)
from experiments.mnist_relu_drn.ibm_om_bounded_codebook_scheme_screen import (
    _tensor_sha256,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat import (
    WinsorizedQatLayerTemplate,
    load_winsorized_qat_templates,
    logical_master_gradients,
    quantize_winsorized_logical_master,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_qat_runtime import (
    NOMINAL_DELTA_X,
    _apply_view,
    _evaluate_deployment,
    _load_winsorized_population,
)
from experiments.mnist_relu_drn.runtime import _load_teacher
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from training.checkpoint import atomic_torch_save


if TYPE_CHECKING:
    from ebl.cli import TrainRequest
    from experiments.mnist_relu_drn.ibm_om_winsorized_multi_assignment_qat_config import (
        PredecessorBundleContract,
        WinsorizedMultiAssignmentQatProtocol,
    )


_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_winsorized_multi_assignment_qat_exploratory"
)
SUMMARY_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_winsorized_multi_assignment_qat_logical_master"
)
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FrozenAssignmentBundle:
    assignment_seed: int
    role: str
    component: str
    run_dir: Path
    result_path: Path
    summary_path: Path
    mapping_path: Path
    population_path: Path
    population_receipt_path: Path
    commissioning_path: Path
    joint_path: Path
    calibration_path: Path | None
    artifact_sha256: Mapping[str, str]
    source_hardware_instance_id: str
    population_fingerprint: str
    commissioning_seed: int


def assignment_seed_for_minibatch(
    assignment_seeds: Sequence[int], global_minibatch_ordinal: int
) -> int:
    """Return the frozen assignment for one global minibatch ordinal."""

    seeds = tuple(int(value) for value in assignment_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Expected a nonempty unique assignment cycle.")
    if (
        isinstance(global_minibatch_ordinal, bool)
        or not isinstance(global_minibatch_ordinal, int)
        or global_minibatch_ordinal < 0
    ):
        raise ValueError("Expected a nonnegative integer global minibatch ordinal.")
    return seeds[global_minibatch_ordinal % len(seeds)]


def pre_final_assignment_seeds(
    training_assignment_seeds: Sequence[int],
    development_assignment_seed: int,
    final_assignment_seed: int,
) -> tuple[int, ...]:
    """Validate role separation and return assignments allowed before final eval."""

    training = tuple(int(value) for value in training_assignment_seeds)
    development = int(development_assignment_seed)
    final = int(final_assignment_seed)
    if (
        len(training) != 2
        or len(set(training)) != 2
        or development in training
        or final in training
        or final == development
    ):
        raise ValueError("Expected two training, one development, and one final role.")
    return (*training, development)


def _registered_artifacts(result: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("Expected predecessor result artifacts.")
    records: dict[str, Mapping[str, Any]] = {}
    for item in artifacts:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not isinstance(item.get("sha256"), str)
            or isinstance(item.get("size_bytes"), bool)
            or not isinstance(item.get("size_bytes"), int)
            or item["size_bytes"] < 0
            or item["path"] in records
        ):
            raise RuntimeError("Expected unique strict predecessor artifact records.")
        records[item["path"]] = item
    return records


def _verified_registered_path(
    run_dir: Path,
    records: Mapping[str, Mapping[str, Any]],
    relative: str,
) -> tuple[Path, str]:
    record = records.get(relative)
    if record is None:
        raise RuntimeError(f"Missing predecessor artifact record {relative!r}.")
    path = (run_dir / relative).resolve()
    if not path.is_relative_to(run_dir.resolve()) or not path.is_file():
        raise RuntimeError(f"Missing safe predecessor artifact {relative!r}.")
    if path.stat().st_size != record["size_bytes"]:
        raise RuntimeError(f"Predecessor artifact size mismatch for {relative!r}.")
    observed = sha256_file(path)
    if observed != record["sha256"]:
        raise RuntimeError(f"Predecessor artifact SHA-256 mismatch for {relative!r}.")
    return path, observed


def resolve_frozen_assignment_bundle(
    contract: "PredecessorBundleContract",
    *,
    source_exploratory_result_id: str,
    spacing_delta_x_multiplier: int,
) -> FrozenAssignmentBundle:
    """Resolve one exact result-hash-pinned predecessor assignment bundle."""

    runs_root = (
        _ROOT
        / "results"
        / source_exploratory_result_id
        / "runs"
    ).resolve()
    run_dir = (runs_root / contract.run_relative_path).resolve()
    if not run_dir.is_relative_to(runs_root):
        raise RuntimeError("Predecessor run path escaped the frozen result root.")
    result_path = run_dir / "result.json"
    status_path = run_dir / "status.json"
    if (
        not result_path.is_file()
        or sha256_file(result_path) != contract.result_sha256
        or not status_path.is_file()
    ):
        raise RuntimeError("Frozen predecessor result path or SHA-256 mismatch.")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Expected readable predecessor control files.") from error
    if result.get("status") != "complete" or status.get("status") != "complete":
        raise RuntimeError("Expected one complete predecessor run.")
    records = _registered_artifacts(result)

    if contract.component == "development":
        prefix = "artifacts/development"
        mapping_relative = f"{prefix}/selected_physical_mapping.npz"
        calibration_relative = f"{prefix}/calibration.json"
    elif contract.component == "heldout":
        prefix = "artifacts/heldout"
        mapping_relative = f"{prefix}/ideal_physical_mapping.npz"
        calibration_relative = None
    else:
        raise RuntimeError("Expected development or heldout predecessor component.")
    required = {
        "mapping": mapping_relative,
        "population": f"{prefix}/winsorized_identity/winsorized_population.npz",
        "population_receipt": (
            f"{prefix}/winsorized_identity/winsorized_population.receipt.json"
        ),
        "commissioning": (
            f"{prefix}/winsorized_identity/winsorized_reset_commissioning.npz"
        ),
        "joint": f"{prefix}/joint_assignment.npz",
        "summary": "artifacts/scientific_summary.json",
    }
    if calibration_relative is not None:
        required["calibration"] = calibration_relative
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {"result": contract.result_sha256}
    for name, relative in required.items():
        paths[name], hashes[name] = _verified_registered_path(
            run_dir, records, relative
        )

    try:
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Expected a readable frozen predecessor summary.") from error
    if summary.get("status") != "complete":
        raise RuntimeError("Expected complete predecessor scientific summary.")
    role_summary = summary.get(contract.component)
    if not isinstance(role_summary, dict):
        raise RuntimeError("Missing predecessor assignment role summary.")
    hardware = role_summary.get("hardware")
    native = role_summary.get("native_coordinate")
    mapping = role_summary.get("mapping")
    if not all(isinstance(value, dict) for value in (hardware, native, mapping)):
        raise RuntimeError("Missing strict predecessor hardware/mapping metadata.")
    assert isinstance(hardware, dict) and isinstance(native, dict)
    assert isinstance(mapping, dict)
    winsorization = native.get("winsorization")
    commissioning = native.get("commissioning")
    if not isinstance(winsorization, dict) or not isinstance(commissioning, dict):
        raise RuntimeError("Missing predecessor Winsorization metadata.")
    expected_spacing = int(spacing_delta_x_multiplier) * NOMINAL_DELTA_X
    if (
        hardware.get("assignment_seed") != contract.assignment_seed
        or role_summary.get("assignment_seed", contract.assignment_seed)
        != contract.assignment_seed
        or mapping.get("baseline_position_fraction") != 0.0
        or not math.isclose(
            float(mapping.get("level_spacing_unit", math.nan)),
            expected_spacing,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or winsorization.get("policy") != "nominal_bound_winsorization"
        or winsorization.get("raw_a_interval") != [-1.0, 1.0]
        or winsorization.get("identity_resampled") is not False
        or commissioning.get("assignment_seed") != contract.assignment_seed
    ):
        raise RuntimeError("Frozen predecessor assignment contract mismatch.")
    source_hardware = winsorization.get("source_population_fingerprint")
    population_fingerprint = winsorization.get(
        "winsorized_population_fingerprint"
    )
    commissioning_seed = commissioning.get("commissioning_seed")
    if (
        not isinstance(source_hardware, str)
        or len(source_hardware) != 64
        or not isinstance(population_fingerprint, str)
        or len(population_fingerprint) != 64
        or isinstance(commissioning_seed, bool)
        or not isinstance(commissioning_seed, int)
    ):
        raise RuntimeError("Invalid predecessor assignment identity metadata.")
    return FrozenAssignmentBundle(
        assignment_seed=contract.assignment_seed,
        role=contract.role,
        component=contract.component,
        run_dir=run_dir,
        result_path=result_path,
        summary_path=paths["summary"],
        mapping_path=paths["mapping"],
        population_path=paths["population"],
        population_receipt_path=paths["population_receipt"],
        commissioning_path=paths["commissioning"],
        joint_path=paths["joint"],
        calibration_path=paths.get("calibration"),
        artifact_sha256=hashes,
        source_hardware_instance_id=source_hardware,
        population_fingerprint=population_fingerprint,
        commissioning_seed=commissioning_seed,
    )


def _bundle_inputs(bundle: FrozenAssignmentBundle) -> tuple[Mapping[str, Any], ...]:
    prefix = f"predecessor_{bundle.role}_{bundle.assignment_seed}"
    paths = {
        "result": bundle.result_path,
        "summary": bundle.summary_path,
        "mapping": bundle.mapping_path,
        "population": bundle.population_path,
        "population_receipt": bundle.population_receipt_path,
        "commissioning": bundle.commissioning_path,
        "joint": bundle.joint_path,
    }
    if bundle.calibration_path is not None:
        paths["calibration"] = bundle.calibration_path
    return tuple(_input(f"{prefix}_{name}", path) for name, path in paths.items())


def _initial_logical_masters(
    teacher: Any,
    templates_by_assignment: Mapping[int, Sequence[WinsorizedQatLayerTemplate]],
    protocol: "WinsorizedMultiAssignmentQatProtocol",
    *,
    device: torch.device,
) -> tuple[tuple[torch.nn.Parameter, ...], tuple[float, ...]]:
    source = tuple(
        value.detach().cpu().to(torch.float32) for value in teacher.parameters()
    )
    expected_hashes = tuple(
        protocol.initialization["source_weight_sha256_by_layer"]
    )
    if tuple(_tensor_sha256(value) for value in source) != expected_hashes:
        raise RuntimeError("Frozen ReLU source tensor SHA-256 mismatch.")
    maxima = tuple(float(value.abs().max().item()) for value in source)
    normalized = tuple(
        (value.to(torch.float64) / maximum).to(torch.float32)
        for value, maximum in zip(source, maxima)
    )
    for assignment_seed, templates in templates_by_assignment.items():
        if len(templates) != 2:
            raise RuntimeError("Expected two templates per frozen assignment.")
        for layer, (value, template) in enumerate(zip(normalized, templates)):
            difference = float(
                (value - template.initial_normalized_weight.cpu())
                .abs()
                .max()
                .item()
            )
            if difference > 1e-7:
                raise RuntimeError(
                    "Assignment template logical initializer mismatch: "
                    f"assignment={assignment_seed}, layer={layer}, max={difference}."
                )
    masters = tuple(
        torch.nn.Parameter(value.to(device=device)) for value in normalized
    )
    return masters, maxima


def _validate_epoch0_parity(
    masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
    mapping_path: Path,
) -> tuple[str, str]:
    view = quantize_winsorized_logical_master(masters, templates)
    observed = tuple(
        _tensor_sha256(value) for value in view.full_conductance_targets
    )
    expected = []
    with np.load(mapping_path, allow_pickle=False) as mapping:
        for layer in range(2):
            value = torch.from_numpy(
                np.array(mapping[f"layer_{layer}_quantized_conductance"], copy=True)
            ).to(torch.float32)
            expected.append(_tensor_sha256(value))
    if observed != tuple(expected):
        raise RuntimeError("Epoch-0 multi-assignment QAT parity mismatch.")
    return observed[0], observed[1]


def _evaluate_quantized(
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.Tensor],
    templates: Sequence[WinsorizedQatLayerTemplate],
) -> tuple[Mapping[str, Any], Any]:
    view = quantize_winsorized_logical_master(masters, templates)
    _apply_view(stack, view)
    metrics, _prediction = _evaluate_detailed(
        stack,
        teacher,
        loader,
        sample_limit=None,
    )
    return metrics, view


def _train_epoch(
    stack: Any,
    teacher: Any,
    loader: Iterable,
    masters: Sequence[torch.nn.Parameter],
    templates_by_assignment: Mapping[int, Sequence[WinsorizedQatLayerTemplate]],
    assignment_cycle: Sequence[int],
    optimizer: torch.optim.Optimizer,
    *,
    global_minibatch_ordinal: int,
    maximum_batches: int | None,
) -> tuple[Mapping[str, Any], int]:
    totals = {
        int(seed): {"examples": 0, "batches": 0, "kl": 0.0, "correct": 0, "agreement": 0}
        for seed in assignment_cycle
    }
    gradient_squared = [0.0, 0.0]
    gradient_values = [0, 0]
    start_ordinal = global_minibatch_ordinal
    for inputs, labels in limited(loader, maximum_batches):
        assignment_seed = assignment_seed_for_minibatch(
            assignment_cycle, global_minibatch_ordinal
        )
        templates = templates_by_assignment[assignment_seed]
        inputs = inputs.to(stack.device, dtype=torch.float32)
        labels = labels.to(stack.device, dtype=torch.long)
        with torch.no_grad():
            teacher_logits = teacher.logits(inputs)
            view = quantize_winsorized_logical_master(
                masters, templates, include_report=False
            )
            _apply_view(stack, view)
        optimizer.zero_grad(set_to_none=True)
        stack.network.set_input(inputs, reset=True)
        stack.minimizer.compute_equilibrium()
        stack.cost.set_teacher(teacher_logits, labels)
        with torch.no_grad():
            batch_kl = stack.cost.eval()
            student_prediction = stack.cost.student_logits().argmax(dim=1)
            teacher_prediction = teacher_logits.argmax(dim=1)
        physical_gradients = tuple(stack.differentiator.compute_gradient())
        logical_gradients = logical_master_gradients(
            masters, physical_gradients, templates
        )
        for layer, (master, gradient) in enumerate(zip(masters, logical_gradients)):
            master.grad = gradient
            gradient_squared[layer] += float(
                gradient.detach().to(torch.float64).square().sum().item()
            )
            gradient_values[layer] += int(gradient.numel())
        optimizer.step()
        with torch.no_grad():
            for master in masters:
                master.clamp_(-1.0, 1.0)

        count = int(labels.shape[0])
        selected = totals[assignment_seed]
        selected["examples"] += count
        selected["batches"] += 1
        selected["kl"] += float(batch_kl.sum().item())
        selected["correct"] += int(student_prediction.eq(labels).sum().item())
        selected["agreement"] += int(
            student_prediction.eq(teacher_prediction).sum().item()
        )
        global_minibatch_ordinal += 1
    batches = global_minibatch_ordinal - start_ordinal
    if batches <= 0:
        raise RuntimeError("Expected multi-assignment QAT to process a minibatch.")
    assignment_metrics = {}
    for seed in assignment_cycle:
        selected = totals[int(seed)]
        examples = int(selected["examples"])
        if examples <= 0:
            raise RuntimeError("Every training assignment must receive minibatches.")
        assignment_metrics[str(seed)] = {
            "examples": examples,
            "batches": int(selected["batches"]),
            "kl_teacher_student": float(selected["kl"]) / examples,
            "student_accuracy": int(selected["correct"]) / examples,
            "teacher_agreement": int(selected["agreement"]) / examples,
        }
    total_examples = sum(value["examples"] for value in totals.values())
    return (
        {
            "global_minibatch_ordinal_start": start_ordinal,
            "global_minibatch_ordinal_stop": global_minibatch_ordinal,
            "batches": batches,
            "examples": total_examples,
            "assignment_cycle": list(map(int, assignment_cycle)),
            "by_assignment": assignment_metrics,
            "logical_gradient_rms": [
                math.sqrt(total / count)
                for total, count in zip(gradient_squared, gradient_values)
            ],
        },
        global_minibatch_ordinal,
    )


def development_candidate_key(
    validation: Mapping[str, Any], epoch: int
) -> tuple[int, float, int]:
    return (
        int(validation["student_correct"]),
        -float(validation["kl_teacher_student"]),
        -int(epoch),
    )


def _checkpoint_payload(
    *,
    epoch: int,
    masters: Sequence[torch.Tensor],
    source_absmax: Sequence[float],
    optimizer: torch.optim.Optimizer,
    development_validation: Mapping[str, Any],
    protocol: "WinsorizedMultiAssignmentQatProtocol",
    bundles: Mapping[int, FrozenAssignmentBundle],
    teacher_sha256: str,
    global_minibatch_ordinal: int,
) -> Mapping[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": int(epoch),
        "evidence_tier": "exploratory_noncanonical",
        "teacher_sha256": teacher_sha256,
        "spacing_delta_x_multiplier": int(
            protocol.mapping.spacing_delta_x_multiplier
        ),
        "fixed_logit_gain": float(protocol.mapping.fixed_logit_gain),
        "source_absmax": list(map(float, source_absmax)),
        "training_assignment_seeds": list(
            protocol.assignments.training_assignment_seeds
        ),
        "training_population_fingerprints": {
            str(seed): bundles[seed].population_fingerprint
            for seed in protocol.assignments.training_assignment_seeds
        },
        "development_assignment_seed": int(
            protocol.assignments.development_assignment_seed
        ),
        "development_population_fingerprint": bundles[
            protocol.assignments.development_assignment_seed
        ].population_fingerprint,
        "assignment_cycle": protocol.assignments.training_cycle,
        "global_minibatch_ordinal": int(global_minibatch_ordinal),
        "logical_master": tuple(
            value.detach().cpu().to(torch.float32).clone() for value in masters
        ),
        "logical_master_sha256": tuple(_tensor_sha256(value) for value in masters),
        "optimizer_state": optimizer.state_dict(),
        "development_validation": dict(development_validation),
    }


def _bundle_summary(bundle: FrozenAssignmentBundle) -> Mapping[str, Any]:
    return {
        "assignment_seed": bundle.assignment_seed,
        "role": bundle.role,
        "component": bundle.component,
        "run_dir": str(bundle.run_dir),
        "source_hardware_instance_id": bundle.source_hardware_instance_id,
        "winsorized_population_fingerprint": bundle.population_fingerprint,
        "commissioning_seed": bundle.commissioning_seed,
        "artifacts": {
            "result": {"path": str(bundle.result_path), "sha256": bundle.artifact_sha256["result"]},
            "summary": {"path": str(bundle.summary_path), "sha256": bundle.artifact_sha256["summary"]},
            "mapping": {"path": str(bundle.mapping_path), "sha256": bundle.artifact_sha256["mapping"]},
            "population": {"path": str(bundle.population_path), "sha256": bundle.artifact_sha256["population"]},
            "population_receipt": {"path": str(bundle.population_receipt_path), "sha256": bundle.artifact_sha256["population_receipt"]},
            "commissioning": {"path": str(bundle.commissioning_path), "sha256": bundle.artifact_sha256["commissioning"]},
            "joint": {"path": str(bundle.joint_path), "sha256": bundle.artifact_sha256["joint"]},
            **(
                {"calibration": {"path": str(bundle.calibration_path), "sha256": bundle.artifact_sha256["calibration"]}}
                if bundle.calibration_path is not None
                else {}
            ),
        },
    }


def run_train(request: "TrainRequest") -> int:
    from experiments.mnist_relu_drn.ibm_om_winsorized_multi_assignment_qat_config import (
        SOURCE_EXPLORATORY_RESULT_ID,
        WinsorizedMultiAssignmentQatTrainSpec,
    )

    spec = request.spec
    if not isinstance(spec, WinsorizedMultiAssignmentQatTrainSpec):
        raise TypeError("Expected the dedicated multi-assignment QAT train spec.")
    if request.teacher_weights is None:
        raise ValueError("Expected --teacher-weights for multi-assignment QAT.")
    for name in (
        "weights",
        "base_weights",
        "resume",
        "device_data",
        "device_model",
    ):
        if getattr(request, name) is not None:
            raise ValueError(
                "Multi-assignment QAT accepts only --teacher-weights; "
                f"unexpected {name}."
            )
    protocol = spec.protocol
    student = spec.student
    teacher_path = request.teacher_weights.expanduser().resolve()
    if sha256_file(teacher_path) != protocol.source["expected_teacher_weights_sha256"]:
        raise ValueError("Frozen teacher/source checkpoint hash mismatch.")

    bundles = {
        contract.assignment_seed: resolve_frozen_assignment_bundle(
            contract,
            source_exploratory_result_id=SOURCE_EXPLORATORY_RESULT_ID,
            spacing_delta_x_multiplier=protocol.mapping.spacing_delta_x_multiplier,
        )
        for contract in protocol.predecessors
    }
    if set(bundles) != {86001, 87001, 87002, 87003}:
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
            raise RuntimeError("Exploratory multi-assignment QAT requires CUDA.")
        loaders = build_mnist_loaders(
            student.data,
            data_seed=student.runtime.data_seed,
            calibration_examples=student.mapping.calibration_examples,
            calibration_batch_size=student.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            teacher_path,
            device=device,
            spec=student,
        )
        stack = build_student_stack(student, enable_measured=False)
        stack.cost.gain = float(protocol.mapping.fixed_logit_gain)
        spacing = int(protocol.mapping.spacing_delta_x_multiplier) * NOMINAL_DELTA_X
        pre_final_seeds = pre_final_assignment_seeds(
            protocol.assignments.training_assignment_seeds,
            protocol.assignments.development_assignment_seed,
            protocol.assignments.final_assignment_seed,
        )
        templates_cpu = {
            seed: load_winsorized_qat_templates(
                bundle.mapping_path,
                level_spacing_raw_x=spacing,
            )
            for seed, bundle in bundles.items()
            if seed in pre_final_seeds
        }
        templates = {
            seed: tuple(template.to(device) for template in values)
            for seed, values in templates_cpu.items()
        }
        masters, source_absmax = _initial_logical_masters(
            teacher,
            templates_cpu,
            protocol,
            device=device,
        )
        initial_master = tuple(value.detach().clone() for value in masters)
        epoch0_parity = {
            str(seed): list(
                _validate_epoch0_parity(
                    masters,
                    templates[seed],
                    bundles[seed].mapping_path,
                )
            )
            for seed in sorted(pre_final_seeds)
        }
        optimizer = torch.optim.SGD(
            [
                {"params": [master], "lr": float(rate)}
                for master, rate in zip(
                    masters, protocol.training.learning_rates
                )
            ],
            momentum=protocol.training.momentum,
            weight_decay=protocol.training.weight_decay,
        )

        development_seed = protocol.assignments.development_assignment_seed
        initial_development, initial_development_view = _evaluate_quantized(
            stack,
            teacher,
            loaders.validation,
            masters,
            templates[development_seed],
        )
        store.append_metric(
            {
                "mode": "train",
                "epoch": 0,
                "development_assignment_seed": development_seed,
                "development_validation": initial_development,
                "development_quantizer": initial_development_view.report,
            }
        )
        best_epoch = 0
        best_development = dict(initial_development)
        global_minibatch_ordinal = 0
        best_payload = _checkpoint_payload(
            epoch=0,
            masters=masters,
            source_absmax=source_absmax,
            optimizer=optimizer,
            development_validation=initial_development,
            protocol=protocol,
            bundles=bundles,
            teacher_sha256=sha256_file(teacher_path),
            global_minibatch_ordinal=global_minibatch_ordinal,
        )
        history = []
        training_cycle = protocol.assignments.training_assignment_seeds
        for epoch in range(1, int(protocol.training.epochs) + 1):
            train_metrics, global_minibatch_ordinal = _train_epoch(
                stack,
                teacher,
                loaders.train,
                masters,
                templates,
                training_cycle,
                optimizer,
                global_minibatch_ordinal=global_minibatch_ordinal,
                maximum_batches=student.settings.max_batches,
            )
            development, development_view = _evaluate_quantized(
                stack,
                teacher,
                loaders.validation,
                masters,
                templates[development_seed],
            )
            record = {
                "mode": "train",
                "epoch": epoch,
                "spacing_delta_x_multiplier": int(
                    protocol.mapping.spacing_delta_x_multiplier
                ),
                "train": train_metrics,
                "development_assignment_seed": development_seed,
                "development_validation": development,
                "development_quantizer": development_view.report,
            }
            history.append(record)
            store.append_metric(record)
            if development_candidate_key(
                development, epoch
            ) > development_candidate_key(best_development, best_epoch):
                best_epoch = epoch
                best_development = dict(development)
                best_payload = _checkpoint_payload(
                    epoch=epoch,
                    masters=masters,
                    source_absmax=source_absmax,
                    optimizer=optimizer,
                    development_validation=development,
                    protocol=protocol,
                    bundles=bundles,
                    teacher_sha256=sha256_file(teacher_path),
                    global_minibatch_ordinal=global_minibatch_ordinal,
                )
            print(
                "winsorized multi-assignment QAT "
                f"h={protocol.mapping.spacing_delta_x_multiplier} "
                f"epoch={epoch}/10 development87002="
                f"{100*development['student_accuracy']:.2f}%",
                flush=True,
            )

        epoch10_development = dict(history[-1]["development_validation"])
        epoch10_payload = _checkpoint_payload(
            epoch=10,
            masters=masters,
            source_absmax=source_absmax,
            optimizer=optimizer,
            development_validation=epoch10_development,
            protocol=protocol,
            bundles=bundles,
            teacher_sha256=sha256_file(teacher_path),
            global_minibatch_ordinal=global_minibatch_ordinal,
        )
        epoch10_checkpoint = atomic_torch_save(
            epoch10_payload,
            store.run_dir / "checkpoints/epoch_10_logical_master.pt",
        )
        best_checkpoint = atomic_torch_save(
            best_payload,
            store.run_dir / "checkpoints/best_development_87002_logical_master.pt",
        )

        final_seed = protocol.assignments.final_assignment_seed
        final_bundle = bundles[final_seed]
        final_templates_cpu = load_winsorized_qat_templates(
            final_bundle.mapping_path,
            level_spacing_raw_x=spacing,
        )
        final_templates = tuple(
            template.to(device) for template in final_templates_cpu
        )
        epoch0_parity[str(final_seed)] = list(
            _validate_epoch0_parity(
                initial_master,
                final_templates,
                final_bundle.mapping_path,
            )
        )
        final_population = _load_winsorized_population(
            final_bundle.population_path,
            final_bundle.population_receipt_path,
            expected_sha256=final_bundle.artifact_sha256["population"],
            expected_fingerprint=final_bundle.population_fingerprint,
            expected_assignment_seed=final_seed,
            source_joint_hardware_instance_id=(
                final_bundle.source_hardware_instance_id
            ),
        )
        final_evaluations = {
            "epoch_0": _evaluate_deployment(
                role="final_heldout_from_qat_87003_previously_inspected",
                stack=stack,
                teacher=teacher,
                loader=loaders.test,
                masters=initial_master,
                templates=final_templates,
                population=final_population,
                endpoint_seeds=protocol.assignments.final_endpoint_seeds,
                device=device,
            ),
            "epoch_10": _evaluate_deployment(
                role="final_heldout_from_qat_87003_previously_inspected",
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
                "Ten-epoch deterministic codebook QAT cycles frozen model-based "
                "Winsorized assignments 86001 and 87001 per global minibatch. "
                "No P&V or read noise is present during training. Assignment "
                "87002 is selection-only; exact epoch 10 remains the headline. "
                "Assignment 87003 is held out from this QAT and selection but was "
                "inspected in the predecessor screen, so it is not a canonical "
                "untouched target. Persistent one-pulse P&V is deployment-only. "
                "No fabricated-device or on-chip-training claim."
            ),
            "source": {
                "teacher_path": str(teacher_path),
                "teacher_sha256": sha256_file(teacher_path),
                "teacher_metadata": teacher_metadata,
                "source_absmax": list(source_absmax),
            },
            "protocol": to_plain_data(protocol),
            "predecessor_bundles": {
                str(seed): _bundle_summary(bundle)
                for seed, bundle in sorted(bundles.items())
            },
            "initialization": {
                "logical_master_sha256": [
                    _tensor_sha256(value) for value in initial_master
                ],
                "epoch_0_exact_mapping_parity_sha256": epoch0_parity,
                "development_87002_validation": initial_development,
            },
            "training": {
                "optimizer": "sgd",
                "epochs": 10,
                "learning_rates": list(protocol.training.learning_rates),
                "assignment_cycle": list(training_cycle),
                "global_minibatches": global_minibatch_ordinal,
                "program_verify": False,
                "read_noise": False,
                "fixed_headline_epoch": 10,
                "epoch_10_development_87002": epoch10_development,
                "best_development_epoch_diagnostic": best_epoch,
                "best_development_87002_diagnostic": best_development,
                "history": history,
            },
            "final_heldout_87003": final_evaluations,
        }
        summary_path = store.run_dir / "artifacts/scientific_summary.json"
        atomic_write_json(summary_path, summary)
        epoch0_final = final_evaluations["epoch_0"]
        epoch10_final = final_evaluations["epoch_10"]
        terminal = {
            "evidence_tier": "exploratory_noncanonical",
            "spacing_delta_x_multiplier": int(
                protocol.mapping.spacing_delta_x_multiplier
            ),
            "epochs": 10,
            "fixed_headline_epoch": 10,
            "best_development_epoch_diagnostic": best_epoch,
            "epoch_0_development_87002_accuracy": float(
                initial_development["student_accuracy"]
            ),
            "epoch_10_development_87002_accuracy": float(
                epoch10_development["student_accuracy"]
            ),
            "epoch_0_final_heldout_87003_ideal_accuracy": float(
                epoch0_final["ideal_quantized"]["student_accuracy"]
            ),
            "epoch_10_final_heldout_87003_ideal_accuracy": float(
                epoch10_final["ideal_quantized"]["student_accuracy"]
            ),
            "epoch_0_final_heldout_87003_pv_accuracy": float(
                epoch0_final["persistent_mean_accuracy"]
            ),
            "epoch_10_final_heldout_87003_pv_accuracy": float(
                epoch10_final["persistent_mean_accuracy"]
            ),
        }
        store.append_metric({"mode": "train_terminal", **terminal})
        store.complete(
            metrics=terminal,
            artifacts=(
                store.artifact_record(
                    epoch10_checkpoint, kind="epoch_10_logical_master"
                ),
                store.artifact_record(
                    best_checkpoint, kind="best_development_87002_logical_master"
                ),
                store.artifact_record(summary_path, kind="scientific_summary"),
            ),
        )
        del final_population
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = [
    "FrozenAssignmentBundle",
    "assignment_seed_for_minibatch",
    "development_candidate_key",
    "pre_final_assignment_seeds",
    "resolve_frozen_assignment_bundle",
    "run_train",
]
