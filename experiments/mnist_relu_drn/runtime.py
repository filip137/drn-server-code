"""Application runtime for teacher-mapped DRN KL distillation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

import torch
import torch.nn.functional as F

from experiments.artifacts import RunStore, sha256_file
from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_relu_drn.components import (
    StudentStack,
    build_student_stack,
    collect_calibration,
    conductance_statistics,
    fit_positive_logit_gain,
    measured_optimizer_type,
    select_mapping_and_gain,
)
from experiments.mnist_relu_drn.config import (
    MEASURED_BACKENDS,
    MEASURED_COHORT_A_BACKENDS,
    MEASURED_COHORT_B_BACKENDS,
    StudentTrainSpec,
    StudentValidateSpec,
)
from experiments.mnist_shared import build_mnist_loaders, limited
from experiments.schema import to_plain_data
from model.resistive.interaction import DenseResistive, SignedDenseResistive
from training.checkpoint import (
    encode_named_weights,
    load_epoch_boundary_checkpoint,
    load_named_weights,
    save_encoded_named_weights,
    save_epoch_boundary_checkpoint,
)
from training.measured_trace import (
    MeasuredCohortBOptimizer,
    MeasuredTraceOptimizer,
)

if TYPE_CHECKING:
    from ebl.cli import TrainRequest, ValidateRequest


_ROOT = Path(__file__).resolve().parents[2]


def _amplification_index_report(stack: StudentStack) -> list[dict[str, Any]]:
    """Record model-local stage semantics independently of generated names."""

    report = []
    for interaction in stack.bundle.energy._interactions:
        if not isinstance(
            interaction, (DenseResistive, SignedDenseResistive)
        ):
            continue
        item = {
            "pre_layer": interaction._layer_pre.name,
            "post_layer": interaction._layer_post.name,
            "resolved_pre_index": interaction._logical_pre_index,
            "resolved_post_index": interaction._logical_post_index,
        }
        if isinstance(interaction, SignedDenseResistive):
            item.update(
                {
                    "interaction": "differential_pair",
                    "voltage_amp": interaction._voltage_amp,
                    "current_amp": interaction._current_amp,
                    "forward_gain": interaction._forward_gain(),
                    "post_layer_metric": interaction._post_metric(),
                }
            )
        else:
            logical_pre_index = interaction._logical_pre_index
            item.update(
                {
                    "interaction": "single_conductance",
                    "voltage_amp": interaction._voltage_amp,
                    "current_amp": interaction._current_amp,
                    "forward_gain": (
                        1.0
                        if logical_pre_index == 0
                        else float(interaction._voltage_amp)
                    ),
                    "post_layer_metric": (
                        float(interaction._current_amp)
                        / float(interaction._voltage_amp)
                    )
                    ** logical_pre_index,
                }
            )
        report.append(item)
    return report


def _differential_topology_signature(
    report: Any,
) -> tuple[tuple[int, int, float, float], ...] | None:
    """Project recorded topology onto its model-local numerical semantics."""

    if not isinstance(report, (list, tuple)):
        return None
    try:
        return tuple(
            (
                int(item["resolved_pre_index"]),
                int(item["resolved_post_index"]),
                float(item["forward_gain"]),
                float(item["post_layer_metric"]),
            )
            for item in report
            if item.get("interaction") == "differential_pair"
        )
    except (KeyError, TypeError, ValueError):
        return None


def _single_topology_signature(
    report: Any,
    *,
    voltage_amp: float,
    current_amp: float,
) -> tuple[tuple[int, int, float, float], ...] | None:
    """Project a one-conductance stack onto model-local stage semantics.

    Older named checkpoints recorded only the explicit logical indices. The
    single-conductance schema fixes the amplifier magnitudes, so missing stage
    scales can be reconstructed without consulting generated layer names.
    New checkpoints record the resolved scales directly and are checked
    exactly.
    """

    if not isinstance(report, (list, tuple)):
        return None
    try:
        signature = []
        for item in report:
            interaction = item.get("interaction")
            if interaction not in {None, "single_conductance"}:
                return None
            pre_index = int(item["resolved_pre_index"])
            post_index = int(item["resolved_post_index"])
            forward_gain = float(
                item.get("forward_gain", 1.0 if pre_index == 0 else voltage_amp)
            )
            post_metric = float(
                item.get(
                    "post_layer_metric",
                    (current_amp / voltage_amp) ** pre_index,
                )
            )
            signature.append(
                (pre_index, post_index, forward_gain, post_metric)
            )
        return tuple(signature)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _model_checkpoint_metadata(
    stack: StudentStack,
    *,
    spec: StudentTrainSpec,
    teacher_sha256: str,
    mapping: dict[str, Any] | None,
    deployment_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "experiment_id": spec.experiment_id,
        "encoding": spec.model.encoding,
        "include_biases": spec.model.include_biases,
        "amplification_indexing": spec.model.amplification_indexing,
        "amplification_indices": _amplification_index_report(stack),
        "objective": "teacher_kl",
        "initialization": "teacher_mapped",
        "fixed_logit_gain": stack.cost.gain,
        "temperature": spec.settings.temperature,
        "conductance_bounds_s": [
            spec.model.conductance_min,
            spec.model.conductance_max,
        ],
        "mapping_range_placement": spec.mapping.range_placement,
        "mapping_scale_fraction_pairs": (
            None
            if spec.mapping.scale_fraction_pairs is None
            else [list(pair) for pair in spec.mapping.scale_fraction_pairs]
        ),
        "teacher_sha256": teacher_sha256,
        "mapping": mapping,
        "update_backend": spec.settings.update_backend.type,
        "deployment_source": deployment_source,
    }
    if isinstance(stack.optimizer, MeasuredTraceOptimizer):
        data_report = stack.optimizer.data_report
        metadata.update(
            {
                "initial_target_mapping": spec.settings.update_backend.parameters[
                    "initial_target_mapping"
                ],
                "dual_rail_layout_by_parameter": (
                    None
                    if spec.settings.update_backend.parameters.get(
                        "dual_rail_layout_by_parameter"
                    )
                    is None
                    else dict(
                        spec.settings.update_backend.parameters[
                            "dual_rail_layout_by_parameter"
                        ]
                    )
                ),
                "positive_gradient_threshold_by_parameter": (
                    data_report.get(
                        "positive_gradient_threshold_by_parameter"
                    )
                ),
                "device_data_sha256": data_report["source_sha256"],
                "device_assignment_sha256_by_parameter": data_report[
                    "assignment_sha256_by_parameter"
                ],
            }
        )
    else:
        metadata["initial_target_mapping"] = "ideal_teacher_mapping"
    return metadata


def _validate_checkpoint_metadata(
    metadata: Any,
    *,
    spec: StudentValidateSpec | StudentTrainSpec,
    teacher_sha256: str,
    expected_amplification_indices: Any,
) -> None:
    provided = metadata if isinstance(metadata, dict) else {}
    expected = {
        "experiment_id": spec.experiment_id,
        "encoding": spec.model.encoding,
        "include_biases": spec.model.include_biases,
        "amplification_indexing": spec.model.amplification_indexing,
        "objective": "teacher_kl",
        "initialization": "teacher_mapped",
        "temperature": 1.0,
        "teacher_sha256": teacher_sha256,
    }
    if (
        spec.mapping.range_placement != "lower"
        or spec.mapping.scale_fraction_pairs is not None
    ):
        expected.update(
            {
                "conductance_bounds_s": [
                    spec.model.conductance_min,
                    spec.model.conductance_max,
                ],
                "mapping_range_placement": spec.mapping.range_placement,
                "mapping_scale_fraction_pairs": (
                    None
                    if spec.mapping.scale_fraction_pairs is None
                    else [
                        list(pair)
                        for pair in spec.mapping.scale_fraction_pairs
                    ]
                ),
            }
        )
    train_settings = (
        spec.settings
        if isinstance(spec, StudentTrainSpec)
        else None
    )
    if (
        train_settings is not None
        and train_settings.update_backend.type in MEASURED_COHORT_A_BACKENDS
        and train_settings.update_backend.parameters["initial_target_mapping"]
        in {
            "dual_rail_pairwise_common_window",
            "dual_rail_quad_common_window",
        }
    ):
        expected.update(
            {
                "initial_target_mapping": train_settings.update_backend.parameters[
                    "initial_target_mapping"
                ],
                "dual_rail_layout_by_parameter": dict(
                    train_settings.update_backend.parameters[
                        "dual_rail_layout_by_parameter"
                    ]
                ),
            }
        )
    if (
        train_settings is not None
        and train_settings.update_backend.type
        == "measured_cohort_a_one_pulse_down"
        and train_settings.update_backend.parameters.get(
            "positive_gradient_threshold_by_parameter"
        )
        is not None
    ):
        expected["positive_gradient_threshold_by_parameter"] = dict(
            train_settings.update_backend.parameters[
                "positive_gradient_threshold_by_parameter"
            ]
        )
    mismatches = {
        name: {"expected": value, "provided": provided.get(name)}
        for name, value in expected.items()
        if provided.get(name) != value
    }
    if spec.model.encoding == "differential":
        expected_signature = _differential_topology_signature(
            expected_amplification_indices
        )
        provided_signature = _differential_topology_signature(
            provided.get("amplification_indices")
        )
        if provided_signature != expected_signature:
            mismatches["amplification_indices"] = {
                "expected": expected_signature,
                "provided": provided_signature,
            }
    else:
        expected_signature = _single_topology_signature(
            expected_amplification_indices,
            voltage_amp=spec.model.voltage_amp,
            current_amp=spec.model.current_amp,
        )
        provided_signature = _single_topology_signature(
            provided.get("amplification_indices"),
            voltage_amp=spec.model.voltage_amp,
            current_amp=spec.model.current_amp,
        )
        if provided_signature != expected_signature:
            mismatches["amplification_indices"] = {
                "expected": expected_signature,
                "provided": provided_signature,
            }
    if mismatches:
        raise ValueError(
            "Expected KD checkpoint metadata to match the requested model, "
            f"topology, and teacher. Provided value: {mismatches!r}."
        )


def _validate_cohort_b_source_metadata(
    metadata: Any,
    *,
    spec: StudentTrainSpec,
    device_data_sha256: str,
) -> None:
    """Require cohort-B deployment to originate from the matched A protocol."""

    provided = metadata if isinstance(metadata, dict) else {}
    source_mapping = (
        "dual_rail_quad_common_window"
        if spec.model.encoding == "single"
        else "paired_affine_common_window"
    )
    expected = {
        "encoding": spec.model.encoding,
        "initialization": "teacher_mapped",
        "update_backend": "measured_cohort_a",
        "initial_target_mapping": source_mapping,
        "device_data_sha256": device_data_sha256,
    }
    if spec.model.encoding == "single":
        expected["dual_rail_layout_by_parameter"] = dict(
            spec.settings.update_backend.parameters[
                "dual_rail_layout_by_parameter"
            ]
        )
    mismatches = {
        name: {"expected": value, "provided": provided.get(name)}
        for name, value in expected.items()
        if provided.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "Expected cohort-B deployment weights to come from the matched "
            "differential cohort-A common-window protocol. Provided value: "
            f"{mismatches!r}."
        )


def _validate_resume_backend_metadata(
    metadata: Any,
    *,
    spec: StudentTrainSpec,
    device_data_sha256: str | None,
) -> None:
    provided = metadata if isinstance(metadata, dict) else {}
    backend = spec.settings.update_backend.type
    expected = {"update_backend": backend}
    if backend in MEASURED_BACKENDS:
        expected.update(
            {
                "initial_target_mapping": spec.settings.update_backend.parameters[
                    "initial_target_mapping"
                ],
                "device_data_sha256": device_data_sha256,
            }
        )
        if (
            spec.settings.update_backend.parameters["initial_target_mapping"]
            in {
                "dual_rail_pairwise_common_window",
                "dual_rail_quad_common_window",
            }
        ):
            expected["dual_rail_layout_by_parameter"] = dict(
                spec.settings.update_backend.parameters[
                    "dual_rail_layout_by_parameter"
                ]
            )
        if (
            backend == "measured_cohort_a_one_pulse_down"
            and spec.settings.update_backend.parameters.get(
                "positive_gradient_threshold_by_parameter"
            )
            is not None
        ):
            expected["positive_gradient_threshold_by_parameter"] = dict(
                spec.settings.update_backend.parameters[
                    "positive_gradient_threshold_by_parameter"
                ]
            )
    if (
        spec.mapping.range_placement != "lower"
        or spec.mapping.scale_fraction_pairs is not None
    ):
        expected.update(
            {
                "conductance_bounds_s": [
                    spec.model.conductance_min,
                    spec.model.conductance_max,
                ],
                "mapping_range_placement": spec.mapping.range_placement,
                "mapping_scale_fraction_pairs": (
                    None
                    if spec.mapping.scale_fraction_pairs is None
                    else [
                        list(pair)
                        for pair in spec.mapping.scale_fraction_pairs
                    ]
                ),
            }
        )
    mismatches = {
        name: {"expected": value, "provided": provided.get(name)}
        for name, value in expected.items()
        if provided.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "Expected resume checkpoint backend provenance to match the "
            f"requested training protocol. Provided value: {mismatches!r}."
        )


def _input(role: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Expected --{role.replace('_', '-')} to name an existing file. "
            f"Provided value: {str(resolved)!r}."
        )
    return {"role": role, "path": str(resolved), "sha256": sha256_file(resolved)}


def _validate_train_request(request: Any, spec: StudentTrainSpec) -> None:
    if request.teacher_weights is None:
        raise ValueError(
            "Expected mnist_relu_drn_kd.v1 training to receive "
            "--teacher-weights. Provided value: None."
        )
    if request.base_weights is not None:
        raise ValueError(
            "Expected mnist_relu_drn_kd.v1 not to use --base-weights. "
            f"Provided value: {str(request.base_weights)!r}."
        )
    selected = [request.weights is not None, request.resume is not None]
    if sum(selected) > 1:
        raise ValueError(
            "Expected at most one of --weights or --resume. Provided value: both."
        )
    backend = spec.settings.update_backend.type
    measured = backend in MEASURED_BACKENDS
    if measured != (request.device_data is not None):
        expected = "an explicit --device-data" if measured else "no --device-data"
        raise ValueError(
            f"Expected update backend {spec.settings.update_backend.type!r} to use {expected}. "
            f"Provided value: {request.device_data!r}."
        )
    if (
        backend in MEASURED_COHORT_B_BACKENDS
        and request.weights is None
        and request.resume is None
    ):
        raise ValueError(
            "Expected measured_cohort_b training to receive cohort-A "
            "--weights or an exact --resume checkpoint. Provided value: "
            "weights=None, resume=None."
        )


def _load_teacher(path: Path, *, device: torch.device) -> tuple[BiasFreeReluTeacher, dict[str, Any]]:
    teacher = BiasFreeReluTeacher(device=device)
    loaded = load_named_weights(path, teacher.catalog)
    architecture = loaded.metadata.get("architecture")
    if architecture != "bias_free_relu_784_50_10":
        raise ValueError(
            "Expected --teacher-weights metadata architecture to equal "
            "'bias_free_relu_784_50_10'. "
            f"Provided value: {architecture!r}."
        )
    return teacher, loaded.metadata


def _evaluate(
    stack: StudentStack,
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    *,
    maximum_batches: int | None = None,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    totals = {
        "kl": 0.0,
        "raw_kl": 0.0,
        "student_correct": 0,
        "teacher_correct": 0,
        "agreement": 0,
        "raw_score_squared": 0.0,
        "calibrated_score_squared": 0.0,
        "teacher_score_squared": 0.0,
    }
    examples = 0
    with torch.no_grad():
        for inputs, labels in limited(loader, maximum_batches):
            if sample_limit is not None:
                remaining = sample_limit - examples
                if remaining <= 0:
                    break
                inputs = inputs[:remaining]
                labels = labels[:remaining]
            inputs = inputs.to(stack.device, dtype=torch.float32)
            labels = labels.to(stack.device, dtype=torch.long)
            teacher_logits = teacher.logits(inputs)
            stack.network.set_input(inputs, reset=True)
            stack.minimizer.compute_equilibrium()
            stack.cost.set_teacher(teacher_logits, labels)
            raw_scores = stack.cost.student_logits() / stack.cost.gain
            student_logits = raw_scores * stack.cost.gain
            teacher_log_prob = F.log_softmax(teacher_logits, dim=1)
            teacher_prob = teacher_log_prob.exp()
            student_log_prob = F.log_softmax(student_logits, dim=1)
            raw_student_log_prob = F.log_softmax(raw_scores, dim=1)
            totals["kl"] += float(
                (teacher_prob * (teacher_log_prob - student_log_prob)).sum().item()
            )
            totals["raw_kl"] += float(
                (teacher_prob * (teacher_log_prob - raw_student_log_prob)).sum().item()
            )
            student_prediction = student_logits.argmax(dim=1)
            teacher_prediction = teacher_logits.argmax(dim=1)
            totals["student_correct"] += int(student_prediction.eq(labels).sum().item())
            totals["teacher_correct"] += int(teacher_prediction.eq(labels).sum().item())
            totals["agreement"] += int(student_prediction.eq(teacher_prediction).sum().item())
            totals["raw_score_squared"] += float(raw_scores.square().sum().item())
            totals["calibrated_score_squared"] += float(student_logits.square().sum().item())
            totals["teacher_score_squared"] += float(teacher_logits.square().sum().item())
            examples += int(labels.shape[0])
    if examples == 0:
        raise ValueError("Expected KD evaluation to process examples. Provided value: 0.")
    score_values = examples * 10
    return {
        "examples": examples,
        "kl_teacher_student": totals["kl"] / examples,
        "raw_kl_teacher_student": totals["raw_kl"] / examples,
        "student_accuracy": totals["student_correct"] / examples,
        "teacher_accuracy": totals["teacher_correct"] / examples,
        "teacher_agreement": totals["agreement"] / examples,
        "raw_score_rms": (totals["raw_score_squared"] / score_values) ** 0.5,
        "calibrated_score_rms": (totals["calibrated_score_squared"] / score_values) ** 0.5,
        "teacher_logit_rms": (totals["teacher_score_squared"] / score_values) ** 0.5,
        "fixed_logit_gain": stack.cost.gain,
    }


def _train_epoch(
    stack: StudentStack,
    teacher: BiasFreeReluTeacher,
    loader: Iterable,
    *,
    maximum_batches: int | None,
) -> tuple[dict[str, Any], int]:
    kl_sum = 0.0
    student_correct = 0
    agreement = 0
    examples = 0
    batches = 0
    for batch_index, (inputs, labels) in enumerate(limited(loader, maximum_batches)):
        inputs = inputs.to(stack.device, dtype=torch.float32)
        labels = labels.to(stack.device, dtype=torch.long)
        with torch.no_grad():
            teacher_logits = teacher.logits(inputs)
        stack.network.set_input(inputs, reset=True)
        stack.minimizer.compute_equilibrium()
        stack.cost.set_teacher(teacher_logits, labels)
        with torch.no_grad():
            batch_kl = stack.cost.eval()
            student_prediction = stack.cost.student_logits().argmax(dim=1)
            teacher_prediction = teacher_logits.argmax(dim=1)
            kl_sum += float(batch_kl.sum().item())
            student_correct += int(student_prediction.eq(labels).sum().item())
            agreement += int(student_prediction.eq(teacher_prediction).sum().item())
        stack.optimizer.zero_grad(set_to_none=True)
        gradients = tuple(stack.differentiator.compute_gradient())
        parameters = tuple(stack.bundle.energy.params())
        if len(gradients) != len(parameters):
            raise RuntimeError(
                "Expected one KL gradient per trainable conductance tensor. "
                f"Provided value: gradients={len(gradients)}, parameters={len(parameters)}."
            )
        for binding, parameter, gradient in zip(
            stack.bundle.catalog.trainable,
            parameters,
            gradients,
        ):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    "Expected finite KD gradients. "
                    f"Provided value: parameter={binding.key!r}, batch={batch_index}."
                )
            parameter.state.grad = gradient
        stack.optimizer.step()
        for parameter in parameters:
            parameter.clamp_()
        examples += int(labels.shape[0])
        batches = batch_index + 1
    if examples == 0:
        raise ValueError("Expected KD training to process examples. Provided value: 0.")
    return (
        {
            "examples": examples,
            "kl_teacher_student": kl_sum / examples,
            "student_accuracy": student_correct / examples,
            "teacher_agreement": agreement / examples,
        },
        batches,
    )


def _selected_payload(
    stack: StudentStack,
    *,
    spec: StudentTrainSpec,
    teacher_path: Path,
    teacher_sha256: str,
    mapping: dict[str, Any] | None,
    deployment_source: dict[str, Any] | None,
    epoch: int,
    validation: dict[str, Any],
) -> dict[str, Any]:
    metadata = _model_checkpoint_metadata(
        stack,
        spec=spec,
        teacher_sha256=teacher_sha256,
        mapping=mapping,
        deployment_source=deployment_source,
    )
    metadata.update(
        {
            "teacher_path": str(teacher_path.expanduser().resolve()),
            "selection_metric": "validation.kl_teacher_student",
            "selection_value": validation["kl_teacher_student"],
            "selection_epoch": epoch,
            "selection_student_accuracy": validation["student_accuracy"],
            "selection_teacher_agreement": validation["teacher_agreement"],
        }
    )
    return encode_named_weights(
        stack.bundle.catalog,
        metadata=metadata,
    )


def _adaptation_summary(
    baseline: dict[str, Any],
    selected: dict[str, Any],
    last: dict[str, Any],
    *,
    selected_epoch: int,
) -> dict[str, Any]:
    baseline_kl = float(baseline["kl_teacher_student"])

    def comparison(candidate: dict[str, Any]) -> dict[str, Any]:
        kl_reduction = baseline_kl - float(candidate["kl_teacher_student"])
        return {
            "student_accuracy_change": (
                float(candidate["student_accuracy"])
                - float(baseline["student_accuracy"])
            ),
            "teacher_agreement_change": (
                float(candidate["teacher_agreement"])
                - float(baseline["teacher_agreement"])
            ),
            "kl_reduction": kl_reduction,
            "relative_kl_reduction": (
                kl_reduction / baseline_kl if baseline_kl > 0.0 else 0.0
            ),
        }

    return {
        "baseline": "post_deployment_calibrated_validation",
        "selected_epoch": selected_epoch,
        "selected": comparison(selected),
        "after_ten_epochs": comparison(last),
    }


def run_train(request: "TrainRequest") -> int:
    spec = request.spec
    if not isinstance(spec, StudentTrainSpec):
        raise TypeError(
            "Expected mnist_relu_drn_kd.v1 train to resolve StudentTrainSpec. "
            f"Provided value: {type(spec).__name__}."
        )
    _validate_train_request(request, spec)
    input_artifacts = [_input("teacher_weights", request.teacher_weights)]
    for role in ("weights", "resume", "device_data"):
        value = getattr(request, role)
        if value is not None:
            input_artifacts.append(_input(role, value))
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=input_artifacts,
    )
    try:
        torch.manual_seed(spec.runtime.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(spec.runtime.seed)
        device = torch.device(spec.runtime.device)
        data = build_mnist_loaders(
            spec.data,
            data_seed=spec.runtime.data_seed,
            calibration_examples=spec.mapping.calibration_examples,
            calibration_batch_size=spec.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(
            request.teacher_weights,
            device=device,
        )
        teacher_sha = sha256_file(request.teacher_weights)
        backend = spec.settings.update_backend.type
        measured = backend in MEASURED_BACKENDS
        cohort_b = backend in MEASURED_COHORT_B_BACKENDS
        measured_optimizer = (
            measured_optimizer_type(backend) if measured else None
        )
        device_data_sha = (
            sha256_file(request.device_data)
            if request.device_data is not None
            else None
        )
        stack = build_student_stack(
            spec,
            device_data_path=request.device_data,
            enable_measured=request.resume is not None,
        )
        weights_path = store.run_dir / "checkpoints" / "weights.pt"
        resume_path = store.run_dir / "checkpoints" / "resume.pt"
        start_epoch = 0
        global_step = 0
        mapping_report: dict[str, Any] | None = None
        programming_report = None
        selected_weights = None
        selected_epoch = -1
        selected_validation = None
        deployment_source: dict[str, Any] | None = None
        deployment_metrics: dict[str, Any] | None = None
        initial: dict[str, Any] | None = None
        initial_conductances: dict[str, Any] | None = None
        resumed_last_train: dict[str, Any] | None = None
        resumed_last_validation: dict[str, Any] | None = None
        resumed_run = request.resume is not None

        if request.resume is not None:
            resumed = load_epoch_boundary_checkpoint(
                request.resume,
                catalog=stack.bundle.catalog,
                optimizer=stack.optimizer,
                dataloader_generators={"train": data.train_generator},
            )
            _validate_checkpoint_metadata(
                resumed.metadata,
                spec=spec,
                teacher_sha256=teacher_sha,
                expected_amplification_indices=(
                    _amplification_index_report(stack)
                ),
            )
            _validate_resume_backend_metadata(
                resumed.metadata,
                spec=spec,
                device_data_sha256=device_data_sha,
            )
            start_epoch = resumed.epoch
            global_step = resumed.global_step
            stack.cost.gain = float(resumed.progress_state["fixed_logit_gain"])
            selected_epoch = int(resumed.progress_state["selected_epoch"])
            selected_validation = dict(resumed.progress_state["selected_validation"])
            mapping_report = deepcopy(resumed.progress_state.get("mapping"))
            deployment_source = deepcopy(
                resumed.progress_state.get("deployment_source")
            )
            deployment_metrics = deepcopy(
                resumed.progress_state.get("deployment")
            )
            initial = dict(resumed.progress_state["initial_validation"])
            initial_conductances = deepcopy(
                resumed.progress_state.get("initial_conductances")
            )
            resumed_last_train = deepcopy(
                resumed.progress_state.get("last_train")
            )
            resumed_last_validation = deepcopy(
                resumed.progress_state.get("last_validation")
            )
            selected_weights = resumed.selected_weights
            if selected_weights is not None:
                save_encoded_named_weights(weights_path, selected_weights, catalog=stack.bundle.catalog)
        elif request.weights is not None:
            loaded = load_named_weights(request.weights, stack.bundle.catalog)
            _validate_checkpoint_metadata(
                loaded.metadata,
                spec=spec,
                teacher_sha256=teacher_sha,
                expected_amplification_indices=(
                    _amplification_index_report(stack)
                ),
            )
            stack.cost.gain = float(loaded.metadata["fixed_logit_gain"])
            mapping_report = deepcopy(loaded.metadata.get("mapping"))
            if cohort_b:
                if device_data_sha is None:
                    raise RuntimeError(
                        "Expected cohort-B device data hash after request "
                        "validation. Provided value: None."
                    )
                _validate_cohort_b_source_metadata(
                    loaded.metadata,
                    spec=spec,
                    device_data_sha256=device_data_sha,
                )
                source_gain = stack.cost.gain
                pre_deployment = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                pre_deployment_conductances = conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                )
                deployment_source = {
                    "path": str(request.weights.expanduser().resolve()),
                    "sha256": sha256_file(request.weights),
                    "encoding": loaded.metadata.get("encoding"),
                    "update_backend": loaded.metadata.get("update_backend"),
                    "initial_target_mapping": loaded.metadata.get(
                        "initial_target_mapping"
                    ),
                    "dual_rail_layout_by_parameter": deepcopy(
                        loaded.metadata.get("dual_rail_layout_by_parameter")
                    ),
                    "device_data_sha256": loaded.metadata.get(
                        "device_data_sha256"
                    ),
                    "device_assignment_sha256_by_parameter": deepcopy(
                        loaded.metadata.get(
                            "device_assignment_sha256_by_parameter"
                        )
                    ),
                    "selection_epoch": loaded.metadata.get("selection_epoch"),
                    "fixed_logit_gain": source_gain,
                }
                optimizer = MeasuredCohortBOptimizer(
                    stack.optimizer,
                    stack.bundle.catalog,
                    spec.settings.update_backend.parameters,
                    request.device_data,
                )
                stack = replace(stack, optimizer=optimizer)
                programming_report = optimizer.initialize_from_loaded_targets()
                post_deployment_source_gain = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                post_deployment_conductances = conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                )
                raw_scores, teacher_logits = collect_calibration(
                    stack,
                    teacher,
                    data.calibration,
                )
                deployment_calibration = fit_positive_logit_gain(
                    raw_scores,
                    teacher_logits,
                    gain_min=spec.mapping.logit_gain_min,
                    gain_max=spec.mapping.logit_gain_max,
                    steps=spec.mapping.logit_gain_steps,
                )
                stack.cost.gain = float(deployment_calibration["gain"])
                post_deployment_calibrated = _evaluate(
                    stack,
                    teacher,
                    data.validation,
                    maximum_batches=spec.settings.max_validation_batches,
                )
                if mapping_report is None:
                    mapping_report = {}
                mapping_report["cohort_b_deployment"] = {
                    "initial_target_mapping": spec.settings.update_backend.parameters[
                        "initial_target_mapping"
                    ],
                    "source_fixed_logit_gain": source_gain,
                    "post_deployment_calibration": deployment_calibration,
                }
                deployment_metrics = {
                    "source": deployment_source,
                    "initial_target_mapping": spec.settings.update_backend.parameters[
                        "initial_target_mapping"
                    ],
                    "pre_deployment": {
                        "validation": pre_deployment,
                        "conductances": pre_deployment_conductances,
                    },
                    "post_deployment_source_gain": {
                        "validation": post_deployment_source_gain,
                        "conductances": post_deployment_conductances,
                    },
                    "post_deployment_calibrated": {
                        "calibration": deployment_calibration,
                        "validation": post_deployment_calibrated,
                        "conductances": post_deployment_conductances,
                    },
                }
                initial = dict(post_deployment_calibrated)
                initial_conductances = post_deployment_conductances
            elif measured:
                if measured_optimizer is None:  # pragma: no cover - guarded above
                    raise AssertionError("measured optimizer type is missing")
                optimizer = measured_optimizer(
                    stack.optimizer,
                    stack.bundle.catalog,
                    spec.settings.update_backend.parameters,
                    request.device_data,
                )
                stack = replace(stack, optimizer=optimizer)
                programming_report = optimizer.initialize_from_reset_targets()
        else:
            measured_candidate_projector = None
            if (
                measured
                and spec.settings.update_backend.parameters[
                    "initial_target_mapping"
                ]
                in {
                    "dual_rail_pairwise_common_window",
                    "dual_rail_quad_common_window",
                }
            ):
                if measured_optimizer is None:  # pragma: no cover - guarded above
                    raise AssertionError("measured optimizer type is missing")
                measured_candidate_projector = measured_optimizer(
                    stack.optimizer,
                    stack.bundle.catalog,
                    spec.settings.update_backend.parameters,
                    request.device_data,
                )
                stack = replace(
                    stack,
                    optimizer=measured_candidate_projector,
                )
            mapping_report, _targets = select_mapping_and_gain(
                stack,
                teacher,
                data.calibration,
                spec,
                measured_candidate_projector=(
                    measured_candidate_projector
                ),
            )
            stack.cost.gain = float(mapping_report["selected"]["calibration"]["gain"])
            if measured:
                if measured_candidate_projector is None:
                    if measured_optimizer is None:  # pragma: no cover
                        raise AssertionError("measured optimizer type is missing")
                    optimizer = measured_optimizer(
                        stack.optimizer,
                        stack.bundle.catalog,
                        spec.settings.update_backend.parameters,
                        request.device_data,
                    )
                    stack = replace(stack, optimizer=optimizer)
                else:
                    optimizer = measured_candidate_projector
                programming_report = (
                    optimizer.initialize_from_reset_targets()
                )
                raw_scores, teacher_logits = collect_calibration(
                    stack,
                    teacher,
                    data.calibration,
                )
                actual_calibration = fit_positive_logit_gain(
                    raw_scores,
                    teacher_logits,
                    gain_min=spec.mapping.logit_gain_min,
                    gain_max=spec.mapping.logit_gain_max,
                    steps=spec.mapping.logit_gain_steps,
                )
                mapping_report["post_programming_calibration"] = actual_calibration
                stack.cost.gain = float(actual_calibration["gain"])

        if initial is None:
            initial = _evaluate(
                stack,
                teacher,
                data.validation,
                maximum_batches=spec.settings.max_validation_batches,
            )
        if initial_conductances is None:
            initial_conductances = conductance_statistics(
                stack.bundle.catalog,
                encoding=spec.model.encoding,
            )
        if deployment_metrics is not None and not resumed_run:
            for mode in (
                "pre_deployment",
                "post_deployment_source_gain",
                "post_deployment_calibrated",
            ):
                store.append_metric(
                    {
                        "mode": mode,
                        **deployment_metrics[mode],
                    }
                )
        store.append_metric(
            {
                "mode": "initialization",
                "mapping": mapping_report,
                "programming": programming_report,
                "validation": initial,
                "conductances": initial_conductances,
            }
        )
        if selected_validation is None:
            selected_validation = dict(initial)
            selected_epoch = -1
            selected_weights = _selected_payload(
                stack,
                spec=spec,
                teacher_path=request.teacher_weights,
                teacher_sha256=teacher_sha,
                mapping=mapping_report,
                deployment_source=deployment_source,
                epoch=-1,
                validation=initial,
            )
            save_encoded_named_weights(weights_path, selected_weights, catalog=stack.bundle.catalog)

        last_train = resumed_last_train
        last_validation = (
            resumed_last_validation
            if resumed_last_validation is not None
            else initial
        )
        for epoch in range(start_epoch, spec.settings.num_epochs):
            last_train, batches = _train_epoch(
                stack,
                teacher,
                data.train,
                maximum_batches=spec.settings.max_batches,
            )
            global_step += batches
            last_validation = _evaluate(
                stack,
                teacher,
                data.validation,
                maximum_batches=spec.settings.max_validation_batches,
            )
            improved = (
                last_validation["kl_teacher_student"]
                < selected_validation["kl_teacher_student"]
            )
            if improved:
                selected_epoch = epoch
                selected_validation = dict(last_validation)
                selected_weights = _selected_payload(
                    stack,
                    spec=spec,
                    teacher_path=request.teacher_weights,
                    teacher_sha256=teacher_sha,
                    mapping=mapping_report,
                    deployment_source=deployment_source,
                    epoch=epoch,
                    validation=last_validation,
                )
                save_encoded_named_weights(weights_path, selected_weights, catalog=stack.bundle.catalog)
            if selected_weights is None:
                raise RuntimeError("Expected selected KD weights at every epoch boundary.")
            progress = {
                "fixed_logit_gain": stack.cost.gain,
                "selected_epoch": selected_epoch,
                "selected_validation": selected_validation,
                "mapping": mapping_report,
                "initial_validation": initial,
                "initial_conductances": initial_conductances,
                "deployment_source": deployment_source,
                "deployment": deployment_metrics,
                "last_train": last_train,
                "last_validation": last_validation,
            }
            save_epoch_boundary_checkpoint(
                resume_path,
                catalog=stack.bundle.catalog,
                epoch=epoch + 1,
                global_step=global_step,
                optimizer=stack.optimizer,
                progress_state=progress,
                selected_weights=selected_weights,
                dataloader_generators={"train": data.train_generator},
                metadata=_model_checkpoint_metadata(
                    stack,
                    spec=spec,
                    teacher_sha256=teacher_sha,
                    mapping=mapping_report,
                    deployment_source=deployment_source,
                ),
            )
            if (epoch + 1) % spec.settings.log_every == 0 or epoch + 1 == spec.settings.num_epochs:
                metric = {
                    "mode": "train",
                    "epoch": epoch,
                    "completed_epochs": epoch + 1,
                    "global_step": global_step,
                    "train": last_train,
                    "validation": last_validation,
                    "selected": improved,
                    "selected_epoch": selected_epoch,
                    "selected_kl": selected_validation["kl_teacher_student"],
                    "conductances": conductance_statistics(
                        stack.bundle.catalog,
                        encoding=spec.model.encoding,
                    ),
                }
                if isinstance(stack.optimizer, MeasuredTraceOptimizer):
                    metric["measured_projection"] = stack.optimizer.programming_report
                store.append_metric(metric)

        if selected_weights is None or selected_validation is None:
            raise RuntimeError("Expected training to provide selected KD weights.")
        if not resume_path.exists():
            save_epoch_boundary_checkpoint(
                resume_path,
                catalog=stack.bundle.catalog,
                epoch=start_epoch,
                global_step=global_step,
                optimizer=stack.optimizer,
                progress_state={
                    "fixed_logit_gain": stack.cost.gain,
                    "selected_epoch": selected_epoch,
                    "selected_validation": selected_validation,
                    "mapping": mapping_report,
                    "initial_validation": initial,
                    "initial_conductances": initial_conductances,
                    "deployment_source": deployment_source,
                    "deployment": deployment_metrics,
                    "last_train": last_train,
                    "last_validation": last_validation,
                },
                selected_weights=selected_weights,
                dataloader_generators={"train": data.train_generator},
                metadata=_model_checkpoint_metadata(
                    stack,
                    spec=spec,
                    teacher_sha256=teacher_sha,
                    mapping=mapping_report,
                    deployment_source=deployment_source,
                ),
            )
        relative_improvement = (
            initial["kl_teacher_student"] - selected_validation["kl_teacher_student"]
        ) / initial["kl_teacher_student"] if initial["kl_teacher_student"] > 0.0 else 0.0
        final_programming = (
            stack.optimizer.programming_report
            if isinstance(stack.optimizer, MeasuredTraceOptimizer)
            else None
        )
        adaptation = (
            _adaptation_summary(
                initial,
                selected_validation,
                last_validation,
                selected_epoch=selected_epoch,
            )
            if cohort_b
            else None
        )
        store.complete(
            metrics={
                "teacher": {"sha256": teacher_sha, "metadata": teacher_metadata},
                "encoding": spec.model.encoding,
                "include_biases": spec.model.include_biases,
                "amplification_indexing": spec.model.amplification_indexing,
                "amplification_indices": _amplification_index_report(stack),
                "objective": "teacher_kl",
                "initialization": "teacher_mapped",
                "fixed_logit_gain": stack.cost.gain,
                "temperature": spec.settings.temperature,
                "update_backend": spec.settings.update_backend.type,
                "mapping": mapping_report,
                "deployment": deployment_metrics,
                "analog_adaptation": adaptation,
                "initial_validation": initial,
                "last_train": last_train,
                "last_validation": last_validation,
                "selected": {"epoch": selected_epoch, **selected_validation},
                "relative_kl_recovery_from_initialization": relative_improvement,
                "acceptance_gate": {
                    "minimum_relative_kl_improvement": spec.settings.minimum_relative_kl_improvement,
                    "passed": relative_improvement >= spec.settings.minimum_relative_kl_improvement,
                },
                "initial_conductances": initial_conductances,
                "final_conductances": conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                ),
                "measured_programming": final_programming,
            },
            artifacts=(
                store.artifact_record(weights_path, kind="selected_named_weights"),
                store.artifact_record(resume_path, kind="epoch_boundary_resume"),
            ),
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


def run_validate(request: "ValidateRequest") -> int:
    spec = request.spec
    if not isinstance(spec, StudentValidateSpec):
        raise TypeError(
            "Expected mnist_relu_drn_kd.v1 validate to resolve StudentValidateSpec. "
            f"Provided value: {type(spec).__name__}."
        )
    if request.teacher_weights is None:
        raise ValueError(
            "Expected mnist_relu_drn_kd.v1 validation to receive "
            "--teacher-weights. Provided value: None."
        )
    store = RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=to_plain_data(spec),
        command=request.command,
        repo_root=_ROOT,
        input_artifacts=(
            _input("weights", request.weights),
            _input("teacher_weights", request.teacher_weights),
        ),
    )
    try:
        torch.manual_seed(spec.runtime.seed)
        device = torch.device(spec.runtime.device)
        data = build_mnist_loaders(
            spec.data,
            data_seed=spec.runtime.data_seed,
            calibration_examples=spec.mapping.calibration_examples,
            calibration_batch_size=spec.mapping.calibration_batch_size,
        )
        teacher, teacher_metadata = _load_teacher(request.teacher_weights, device=device)
        teacher_sha = sha256_file(request.teacher_weights)
        stack = build_student_stack(spec, enable_measured=False)
        loaded = load_named_weights(request.weights, stack.bundle.catalog)
        _validate_checkpoint_metadata(
            loaded.metadata,
            spec=spec,
            teacher_sha256=teacher_sha,
            expected_amplification_indices=_amplification_index_report(stack),
        )
        stack.cost.gain = float(loaded.metadata["fixed_logit_gain"])
        loader = data.validation if spec.settings.split == "validation" else data.test
        metrics = _evaluate(
            stack,
            teacher,
            loader,
            sample_limit=spec.settings.sample_limit,
        )
        store.append_metric({"mode": "validate", "split": spec.settings.split, **metrics})
        store.complete(
            metrics={
                "split": spec.settings.split,
                **metrics,
                "encoding": spec.model.encoding,
                "include_biases": spec.model.include_biases,
                "amplification_indexing": spec.model.amplification_indexing,
                "amplification_indices": _amplification_index_report(stack),
                "objective": "teacher_kl",
                "initialization": "teacher_mapped",
                "teacher": {"sha256": teacher_sha, "metadata": teacher_metadata},
                "checkpoint_metadata": loaded.metadata,
                "conductances": conductance_statistics(
                    stack.bundle.catalog,
                    encoding=spec.model.encoding,
                ),
            }
        )
        return 0
    except BaseException as error:
        store.fail(error)
        raise


__all__ = ["run_train", "run_validate"]
