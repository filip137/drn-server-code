"""Application handlers for ``small_drn.v1``.

The public functions in this module own one command/run lifecycle each.  They
compose the numerical objects from :mod:`components`, invoke the shared
training/evaluation engine, and persist only through the versioned RunStore
and checkpoint contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.artifacts import (
    ArtifactRecord,
    RunStore,
    content_hash,
    sha256_file,
)
from experiments.schema import to_plain_data
from experiments.small_network.components import (
    EvaluationRuntime,
    TrainRuntime,
    ValidationRuntime,
    build_evaluation_runtime,
    build_model_stack,
    build_train_runtime,
    build_validation_runtime,
    configured_resume_capability,
    seed_runtime,
)
from experiments.small_network.config import (
    LinspaceSpec,
    SmallDrnConfig,
    TrainSpec,
    ValidateSpec,
)
from labs.datasets import MoonsDataset
from training.checkpoint import (
    LEGACY_BASE_ONLY,
    LEGACY_FULL,
    encode_named_weights,
    load_epoch_boundary_checkpoint,
    load_legacy_positional_weights,
    load_named_weights,
    save_encoded_named_weights,
    save_epoch_boundary_checkpoint,
    save_named_weights,
)
from training.engine import (
    FreePhaseEvent,
    GradientsReadyEvent,
    evaluate,
    train_epoch,
)
from training.probes import (
    MeanCostProbe,
    MeanErrorProbe,
    ResidualInfinityNormProbe,
    SettledLayerStatesProbe,
    SolverIterationCountsProbe,
)

if TYPE_CHECKING:
    from ebl.cli import (
        ImportLegacyCheckpointRequest,
        LinspaceRequest,
        TrainRequest,
        ValidateRequest,
    )


@dataclass(frozen=True)
class TrainingEpochReport:
    """Typed, immutable measurements available after one completed epoch.

    Error and accuracy values are fractions in ``[0, 1]``.  Selection fields
    always describe the global clean-validation best known at this boundary,
    including a best restored from a resume checkpoint.
    """

    epoch_index: int
    completed_epochs: int
    global_step: int
    train_examples: int
    train_mean_cost: float | None
    train_error_fraction: float | None
    train_accuracy: float | None
    validation_examples: int
    validation_mean_cost: float
    validation_error_fraction: float
    validation_accuracy: float
    selected_this_epoch: bool
    selected_epoch_index: int
    selected_validation_cost: float
    selected_validation_error_fraction: float | None
    selected_validation_accuracy: float | None


@dataclass(frozen=True)
class TrainingOutcome:
    """Typed completion record for programmatic training callers."""

    run_dir: Path
    result_path: Path
    weights_path: Path
    resume_path: Path
    completed_epochs: int
    global_step: int
    selected_epoch_index: int
    selected_validation_cost: float
    selected_validation_error_fraction: float | None
    selected_validation_accuracy: float | None
    last_epoch_report: TrainingEpochReport | None
    resume_capability: str


TrainingObserver = Callable[[TrainingEpochReport], None]


def run_train(request: "TrainRequest") -> int:
    """CLI-compatible wrapper around :func:`execute_train`."""

    execute_train(request)
    return 0


def execute_train(
    request: "TrainRequest",
    observers: Iterable[TrainingObserver] = (),
) -> TrainingOutcome:
    """Train one run and return typed results to programmatic callers.

    Observers run exactly once after every completed epoch, after clean
    validation, global-best selection, resume persistence, and normal metric
    logging for that boundary.  Their cadence is independent of
    ``settings.log_every``.  An observer exception is recorded as the run's
    terminal failure and then re-raised unchanged.
    """

    spec = _expect_spec(request.spec, TrainSpec, mode="train")
    active_observers = _validated_observers(observers)
    resume_capability = configured_resume_capability(spec)
    store = _create_run_store(
        request,
        spec,
        input_artifacts=_training_input_artifacts(request),
        resume_capability=resume_capability,
        runtime_protocol={
            "selection_evaluation": "clean",
            "validation_requested": "validation",
            "validation_effective": "held_out_test",
            "checkpoint_weights": "checkpoints/weights.pt",
            "checkpoint_resume": "checkpoints/resume.pt",
        },
    )
    try:
        seed_runtime(spec.common.runtime.seed)
        runtime = build_train_runtime(spec)
        if runtime.resume_capability != resume_capability:
            raise RuntimeError(
                "Expected preflight and numerical resume capabilities to "
                "match. Provided value: "
                f"preflight={resume_capability!r}, "
                f"numerical={runtime.resume_capability!r}."
            )
        metrics, artifacts, last_epoch_report = _execute_training(
            request,
            spec,
            runtime,
            store,
            observers=active_observers,
        )
        result_path = store.complete(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        store.fail(exc)
        raise
    selected = metrics["selected"]
    return TrainingOutcome(
        run_dir=store.run_dir,
        result_path=result_path,
        weights_path=store.run_dir / "checkpoints" / "weights.pt",
        resume_path=store.run_dir / "checkpoints" / "resume.pt",
        completed_epochs=int(metrics["completed_epochs"]),
        global_step=int(metrics["global_step"]),
        selected_epoch_index=int(selected["epoch"]),
        selected_validation_cost=float(selected["value"]),
        selected_validation_error_fraction=_optional_float(
            selected.get("error_fraction")
        ),
        selected_validation_accuracy=_optional_float(
            selected.get("accuracy")
        ),
        last_epoch_report=last_epoch_report,
        resume_capability=str(metrics["resume_capability"]),
    )


def run_linspace(request: "LinspaceRequest") -> int:
    """Evaluate explicit named weights on the configured two-dimensional grid."""

    spec = _expect_spec(request.spec, LinspaceSpec, mode="linspace")
    store = _create_run_store(
        request,
        spec,
        input_artifacts=(_input_artifact("weights", request.weights),),
        resume_capability="unsupported",
        runtime_protocol={
            "input_assignment": "standard_train_mode",
            "input_gain": 1.0,
            "legacy_test_mode_effect": "no_op",
        },
    )
    try:
        seed_runtime(spec.common.runtime.seed)
        runtime = build_evaluation_runtime(spec.common)
        load_named_weights(request.weights, runtime.stack.bundle.catalog)
        metrics, artifacts = _execute_linspace(spec, runtime, store)
        store.append_metric({"mode": "linspace", **metrics})
        store.complete(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        store.fail(exc)
        raise
    return 0


def run_validate(request: "ValidateRequest") -> int:
    """Evaluate explicit named weights on one configured dataset split."""

    spec = _expect_spec(request.spec, ValidateSpec, mode="validate")
    effective_split = (
        "held_out_test"
        if spec.settings.split in {"validation", "test"}
        else "train"
    )
    store = _create_run_store(
        request,
        spec,
        input_artifacts=(_input_artifact("weights", request.weights),),
        resume_capability="unsupported",
        runtime_protocol={
            "split_requested": spec.settings.split,
            "split_effective": effective_split,
            "evaluation": "clean",
        },
    )
    try:
        seed_runtime(spec.common.runtime.seed)
        runtime = build_validation_runtime(spec.common)
        load_named_weights(request.weights, runtime.stack.bundle.catalog)
        metrics, artifacts = _execute_validation(spec, runtime, store)
        store.append_metric({"mode": "validate", **metrics})
        store.complete(metrics=metrics, artifacts=artifacts)
    except Exception as exc:
        store.fail(exc)
        raise
    return 0


def import_legacy_checkpoint(
    request: "ImportLegacyCheckpointRequest",
) -> int:
    """Convert one explicitly profiled positional checkpoint to named weights."""

    if not isinstance(request.document, SmallDrnConfig):
        raise TypeError(
            "Expected checkpoint import document to be a SmallDrnConfig. "
            f"Provided value: {type(request.document).__name__}."
        )
    source = Path(request.source).expanduser().resolve()
    output = Path(request.output).expanduser().resolve()
    if source == output:
        raise ValueError(
            "Expected checkpoint import output to differ from its source. "
            f"Provided value: {str(output)!r}."
        )
    seed_runtime(request.document.common.runtime.seed)
    stack = build_model_stack(request.document.common)
    profile = LEGACY_FULL if request.kind == "full" else LEGACY_BASE_ONLY
    loaded = load_legacy_positional_weights(
        source,
        stack.bundle.catalog,
        profile=profile,
    )
    save_named_weights(
        output,
        stack.bundle.catalog,
        metadata={
            "experiment_id": request.definition.experiment_id,
            "legacy_profile": profile.name,
            "legacy_schema": loaded.schema,
            "source_path": str(source),
            "source_sha256": sha256_file(source),
            "config_sha256": content_hash(to_plain_data(request.document)),
        },
    )
    return 0


def _execute_training(
    request: Any,
    spec: TrainSpec,
    runtime: TrainRuntime,
    store: RunStore,
    *,
    observers: tuple[TrainingObserver, ...],
) -> tuple[
    dict[str, Any],
    tuple[ArtifactRecord, ...],
    TrainingEpochReport | None,
]:
    catalog = runtime.stack.bundle.catalog
    weights_path = store.run_dir / "checkpoints" / "weights.pt"
    resume_path = store.run_dir / "checkpoints" / "resume.pt"
    generators = runtime.data.dataloader_generators

    start_epoch = 0
    global_step = 0
    selected_cost: float | None = None
    selected_epoch: int | None = None
    selected_error: float | None = None
    selected_accuracy: float | None = None
    selected_weights: Mapping[str, Any] | None = None
    last_epoch_report: TrainingEpochReport | None = None

    if request.resume is not None:
        resumed = load_epoch_boundary_checkpoint(
            request.resume,
            catalog=catalog,
            optimizer=runtime.optimizer,
            modifier=None,
            scheduler=None,
            runtime_state=runtime.runtime_state,
            dataloader_generators=generators,
        )
        if resumed.resume_capability != runtime.resume_capability:
            raise ValueError(
                "Expected resume checkpoint capability to match this runtime. "
                f"Provided value: checkpoint={resumed.resume_capability!r}, "
                f"runtime={runtime.resume_capability!r}."
            )
        expected_resume_config = _resume_config_sha256(spec)
        provided_resume_config = resumed.metadata.get(
            "resume_config_sha256"
        )
        if provided_resume_config != expected_resume_config:
            raise ValueError(
                "Expected the resume checkpoint numerical configuration to "
                "match the requested training configuration except for "
                "modes.train.num_epochs. "
                f"Provided value: checkpoint={provided_resume_config!r}, "
                f"requested={expected_resume_config!r}."
            )
        if resumed.epoch > spec.settings.num_epochs:
            raise ValueError(
                "Expected modes.train.num_epochs to be greater than or equal "
                "to the completed epoch stored in the resume checkpoint. "
                f"Provided value: configured={spec.settings.num_epochs}, "
                f"checkpoint={resumed.epoch}."
            )
        if resumed.selected_weights is None:
            raise ValueError(
                "Expected an epoch-boundary resume checkpoint to include its "
                "clean-evaluation selected_weights snapshot. "
                "Provided value: null."
            )
        selected_weights = resumed.selected_weights
        (
            selected_cost,
            selected_epoch,
            selected_error,
            selected_accuracy,
        ) = _selection_from_progress(resumed.progress_state)
        save_encoded_named_weights(
            weights_path,
            selected_weights,
            catalog=catalog,
        )
        start_epoch = resumed.epoch
        global_step = resumed.global_step
    elif request.weights is not None:
        load_named_weights(request.weights, catalog)
    elif request.base_weights is not None:
        # In the foundation runtime every checkpointed parameter is in the
        # base group.  The future adapter runtime can narrow this explicitly.
        load_named_weights(request.base_weights, catalog)

    last_validation: dict[str, Any] | None = None
    completed_epoch = start_epoch
    for epoch in range(start_epoch, spec.settings.num_epochs):
        train_metrics = _TrainingMetrics()
        training_loader = _limit_batches(
            runtime.data.train_loader,
            spec.settings.max_batches,
        )
        trained = train_epoch(
            runtime.training_components,
            training_loader,
            modifier=None,
            event_handlers=(train_metrics, _FiniteGradientGuard()),
            epoch=epoch,
            start_global_step=global_step,
            reset_input=False,
        )
        global_step = trained.next_global_step
        completed_epoch = epoch + 1

        validation = evaluate(
            runtime.evaluation_components,
            _limit_batches(
                runtime.data.held_out_loader,
                spec.settings.max_validation_batches,
            ),
            modifier=None,
            probes=(MeanCostProbe(), MeanErrorProbe()),
            epoch=epoch,
            split="validation",
            reset_input=True,
        )
        last_validation = _classification_metrics(validation)
        validation_cost = last_validation["mean_cost"]
        if validation_cost is None:
            raise ValueError(
                "Expected clean validation to evaluate at least one example. "
                f"Provided value: {validation.example_count} examples."
            )

        improved = selected_cost is None or validation_cost < selected_cost
        if improved:
            selected_cost = validation_cost
            selected_epoch = epoch
            selected_error = last_validation["mean_error"]
            selected_accuracy = last_validation["accuracy"]
            selected_weights = encode_named_weights(
                catalog,
                metadata={
                    "experiment_id": spec.experiment_id,
                    "selection_metric": "validation.mean_cost",
                    "selection_value": selected_cost,
                    "selection_epoch": selected_epoch,
                    "selection_error_fraction": selected_error,
                    "selection_accuracy": selected_accuracy,
                },
            )
            save_encoded_named_weights(
                weights_path,
                selected_weights,
                catalog=catalog,
            )

        if selected_weights is None or selected_epoch is None:
            raise RuntimeError(
                "Expected clean validation to select a named-weights snapshot. "
                "Provided value: no selected snapshot."
            )
        progress_state = _selection_progress(
            cost=selected_cost,
            epoch=selected_epoch,
            error_fraction=selected_error,
            accuracy=selected_accuracy,
        )
        save_epoch_boundary_checkpoint(
            resume_path,
            catalog=catalog,
            epoch=completed_epoch,
            global_step=global_step,
            optimizer=runtime.optimizer,
            modifier=None,
            scheduler=None,
            runtime_state=runtime.runtime_state,
            progress_state=progress_state,
            selected_weights=selected_weights,
            dataloader_generators=generators,
            resume_capability=runtime.resume_capability,
            metadata=_resume_metadata(spec),
        )

        should_log = (
            completed_epoch % spec.settings.log_every == 0
            or completed_epoch == spec.settings.num_epochs
        )
        train_values = train_metrics.result()
        last_epoch_report = _epoch_report(
            epoch=epoch,
            completed_epoch=completed_epoch,
            global_step=global_step,
            train=train_values,
            validation=last_validation,
            improved=improved,
            selected_epoch=selected_epoch,
            selected_cost=selected_cost,
            selected_error=selected_error,
            selected_accuracy=selected_accuracy,
        )
        if should_log:
            store.append_metric(
                {
                    "mode": "train",
                    "epoch": epoch,
                    "completed_epochs": completed_epoch,
                    "global_step": global_step,
                    "train": train_values,
                    "validation": last_validation,
                    "selected": improved,
                    "selected_epoch": selected_epoch,
                    "selected_cost": selected_cost,
                }
            )
        for observer in observers:
            observer(last_epoch_report)

    if selected_weights is None:
        raise ValueError(
            "Expected training or its resume checkpoint to provide selected "
            "clean-evaluation weights. Provided value: null."
        )

    # A continuation whose configured target epoch is already complete still
    # owns a fresh, self-contained resume artifact.
    if not resume_path.exists():
        save_epoch_boundary_checkpoint(
            resume_path,
            catalog=catalog,
            epoch=completed_epoch,
            global_step=global_step,
            optimizer=runtime.optimizer,
            modifier=None,
            scheduler=None,
            runtime_state=runtime.runtime_state,
            progress_state=_selection_progress(
                cost=selected_cost,
                epoch=selected_epoch,
                error_fraction=selected_error,
                accuracy=selected_accuracy,
            ),
            selected_weights=selected_weights,
            dataloader_generators=generators,
            resume_capability=runtime.resume_capability,
            metadata=_resume_metadata(spec),
        )

    metrics = {
        "completed_epochs": completed_epoch,
        "global_step": global_step,
        "selected": {
            "metric": "validation.mean_cost",
            "value": selected_cost,
            "epoch": selected_epoch,
            "error_fraction": selected_error,
            "accuracy": selected_accuracy,
            "evaluation": "clean",
        },
        "last_validation": last_validation,
        "evaluation_protocol": {
            "validation_requested": "validation",
            "validation_effective": "held_out_test",
            "parameter_modifier": "none",
        },
        "resume_capability": runtime.resume_capability,
    }
    artifacts = (
        store.artifact_record(weights_path, kind="weights"),
        store.artifact_record(resume_path, kind="resume"),
    )
    return metrics, artifacts, last_epoch_report


def _execute_linspace(
    spec: LinspaceSpec,
    runtime: EvaluationRuntime,
    store: RunStore,
) -> tuple[dict[str, Any], tuple[ArtifactRecord, ...]]:
    raw_points, model_points = _linspace_points(spec, runtime)
    labels = torch.zeros(
        model_points.shape[0],
        dtype=torch.long,
        device=runtime.stack.device,
    )
    loader = DataLoader(
        TensorDataset(model_points, labels),
        batch_size=spec.common.data.batch_size,
        shuffle=False,
    )
    probes: list[Any] = [
        ResidualInfinityNormProbe(),
        SolverIterationCountsProbe(),
    ]
    if spec.settings.record_states:
        probes.append(SettledLayerStatesProbe())

    input_layer = runtime.stack.network.layers()[0]
    original_gain = getattr(input_layer, "_gain", None)
    if original_gain is None:
        raise ValueError(
            "Expected the small-DRN input layer to expose its gain. "
            f"Provided value: {input_layer!r}."
        )
    input_layer._gain = 1.0
    try:
        result = evaluate(
            runtime.components,
            loader,
            probes=tuple(probes),
            epoch=0,
            split="linspace",
            reset_input=True,
        )
    finally:
        input_layer._gain = original_gain

    inputs_path = store.run_dir / "artifacts" / "linspace_inputs.npz"
    residuals_path = (
        store.run_dir / "artifacts" / "linspace_residual_currents.npz"
    )
    iterations_path = (
        store.run_dir / "artifacts" / "linspace_iteration_counts.npz"
    )
    _atomic_npz(inputs_path, inputs=raw_points.detach().cpu().numpy())
    residuals = result.probe_value("residual_inf_norm")
    _atomic_npz(residuals_path, **residuals)
    iteration_counts = result.probe_value("solver_iteration_counts")
    _atomic_npz(
        iterations_path,
        iteration_counts=iteration_counts,
        iteration_grid=iteration_counts.reshape(
            spec.settings.samples,
            spec.settings.samples,
        ),
        linspace_samples=np.asarray(spec.settings.samples),
    )

    artifacts = [
        store.artifact_record(inputs_path, kind="linspace_inputs"),
        store.artifact_record(residuals_path, kind="residuals"),
        store.artifact_record(iterations_path, kind="solver_iterations"),
    ]
    if spec.settings.record_states:
        states_path = store.run_dir / "artifacts" / "linspace_states.npz"
        _atomic_npz(
            states_path,
            **_numpy_mapping(result.probe_value("settled_layer_states")),
        )
        artifacts.append(store.artifact_record(states_path, kind="states"))

    residual_metrics = _residual_metrics(
        residuals,
        tolerance=spec.common.solver.tolerances.residual_current,
    )
    metrics = {
        "examples": result.example_count,
        "batches": result.batch_count,
        "grid": {
            "minimum": spec.settings.minimum,
            "maximum": spec.settings.maximum,
            "samples_per_axis": spec.settings.samples,
        },
        "residuals": residual_metrics,
        "solver_iterations": _numeric_summary(iteration_counts),
        "input_protocol": {
            "configured_gain": spec.common.model.input_gain,
            "effective_gain": 1.0,
            "assignment_mode": "train",
            "legacy_test_assignment_effect": "no_op",
        },
    }
    return metrics, tuple(artifacts)


def _execute_validation(
    spec: ValidateSpec,
    runtime: ValidationRuntime,
    store: RunStore,
) -> tuple[dict[str, Any], tuple[ArtifactRecord, ...]]:
    if spec.settings.split == "train":
        loader = runtime.data.train_loader
        effective_split = "train"
    else:
        loader = runtime.data.held_out_loader
        effective_split = "held_out_test"
    limited = _limit_examples(loader, spec.settings.sample_limit)
    capture = _BatchCapture()
    probes: list[Any] = [
        MeanCostProbe(),
        MeanErrorProbe(),
        ResidualInfinityNormProbe(),
        SolverIterationCountsProbe(),
    ]
    if spec.settings.record_states:
        probes.append(SettledLayerStatesProbe())
    result = evaluate(
        runtime.components,
        limited,
        probes=tuple(probes),
        event_handlers=(capture,),
        epoch=0,
        split=spec.settings.split,
        reset_input=True,
    )
    metrics = {
        **_classification_metrics(result),
        "residuals": _residual_metrics(
            result.probe_value("residual_inf_norm"),
            tolerance=spec.common.solver.tolerances.residual_current,
        ),
        "solver_iterations": _numeric_summary(
            result.probe_value("solver_iteration_counts")
        ),
        "split_protocol": {
            "requested": spec.settings.split,
            "effective": effective_split,
        },
        "evaluation": "clean",
    }

    capture_values = capture.result()
    inputs_path = store.run_dir / "artifacts" / "validation_inputs.npz"
    _atomic_npz(inputs_path, **capture_values)
    residuals_path = (
        store.run_dir / "artifacts" / "validation_residual_currents.npz"
    )
    _atomic_npz(
        residuals_path,
        **result.probe_value("residual_inf_norm"),
    )
    iterations_path = (
        store.run_dir / "artifacts" / "validation_iteration_counts.npz"
    )
    _atomic_npz(
        iterations_path,
        iteration_counts=result.probe_value("solver_iteration_counts"),
    )
    artifacts = [
        store.artifact_record(inputs_path, kind="validation_inputs"),
        store.artifact_record(residuals_path, kind="residuals"),
        store.artifact_record(iterations_path, kind="solver_iterations"),
    ]
    if spec.settings.record_states:
        states_path = store.run_dir / "artifacts" / "validation_states.npz"
        _atomic_npz(
            states_path,
            **_numpy_mapping(result.probe_value("settled_layer_states")),
        )
        artifacts.append(store.artifact_record(states_path, kind="states"))
    return metrics, tuple(artifacts)


def _linspace_points(
    spec: LinspaceSpec,
    runtime: EvaluationRuntime,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(
        spec.settings.minimum,
        spec.settings.maximum,
        spec.settings.samples,
        dtype=runtime.stack.dtype,
        device=runtime.stack.device,
    )
    x_axis, y_axis = torch.meshgrid(axis, axis, indexing="ij")
    raw = torch.stack((x_axis.flatten(), y_axis.flatten()), dim=1)
    logical_width = runtime.stack.logical_input_dim
    if logical_width == 2:
        return raw, raw
    if spec.common.data.dataset == "moons":
        return raw, MoonsDataset._expand_xy_features(raw, logical_width)
    raise ValueError(
        "Expected linspace evaluation to use a two-dimensional logical input "
        "or the legacy expanded-moons representation. "
        f"Provided value: dataset={spec.common.data.dataset!r}, "
        f"logical_input_dim={logical_width!r}."
    )


def _classification_metrics(result: Any) -> dict[str, Any]:
    cost = result.probe_value("mean_cost")
    error = result.probe_value("mean_error")
    mean_error = error["mean"]
    return {
        "examples": result.example_count,
        "batches": result.batch_count,
        "mean_cost": cost["mean"],
        "mean_error": mean_error,
        "accuracy": None if mean_error is None else 1.0 - mean_error,
    }


@dataclass
class _TrainingMetrics:
    cost_sum: float = 0.0
    error_sum: float = 0.0
    count: int = 0

    def __call__(self, event: Any) -> None:
        if not isinstance(event, FreePhaseEvent):
            return
        cost = event.components.cost_fn.eval()
        error = event.components.cost_fn.error_fn()
        count = int(cost.numel())
        self.cost_sum += float(cost.sum().item())
        self.error_sum += float(error.sum().item())
        self.count += count

    def result(self) -> dict[str, Any]:
        return {
            "examples": self.count,
            "mean_cost": self.cost_sum / self.count if self.count else None,
            "mean_error": self.error_sum / self.count if self.count else None,
            "accuracy": (
                1.0 - self.error_sum / self.count if self.count else None
            ),
        }


class _FiniteGradientGuard:
    def __call__(self, event: Any) -> None:
        if not isinstance(event, GradientsReadyEvent):
            return
        for index, gradient in enumerate(event.gradients):
            if not isinstance(gradient, torch.Tensor) or not torch.isfinite(
                gradient
            ).all():
                raise FloatingPointError(
                    "Expected every computed gradient to be a finite tensor. "
                    f"Provided value: gradient index {index}, "
                    f"type={type(gradient).__name__}."
                )


class _BatchCapture:
    def __init__(self) -> None:
        self._inputs: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._indices: list[np.ndarray] = []

    def __call__(self, event: Any) -> None:
        self._inputs.append(_as_numpy(event.batch.inputs))
        self._labels.append(_as_numpy(event.batch.targets))
        if event.batch.indices is not None:
            self._indices.append(_as_numpy(event.batch.indices))

    def result(self) -> dict[str, np.ndarray]:
        values = {
            "inputs": _concatenate(self._inputs),
            "labels": _concatenate(self._labels),
        }
        if self._indices:
            values["indices"] = _concatenate(self._indices)
        return values


def _selection_from_progress(
    progress: Mapping[str, Any],
) -> tuple[float, int, float | None, float | None]:
    metric = progress.get("selected_metric")
    value = progress.get("selected_value")
    epoch = progress.get("selected_epoch")
    valid_value = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )
    valid_epoch = (
        not isinstance(epoch, bool) and isinstance(epoch, int) and epoch >= 0
    )
    if metric != "validation.mean_cost" or not valid_value or not valid_epoch:
        raise ValueError(
            "Expected resume progress to describe a finite clean-validation "
            "selection with keys selected_metric, selected_value, and "
            f"selected_epoch. Provided value: {dict(progress)!r}."
        )
    error_fraction = _optional_fraction(
        progress.get("selected_error_fraction"),
        name="resume progress selected_error_fraction",
    )
    accuracy = _optional_fraction(
        progress.get("selected_accuracy"),
        name="resume progress selected_accuracy",
    )
    return float(value), epoch, error_fraction, accuracy


def _selection_progress(
    *,
    cost: float | None,
    epoch: int | None,
    error_fraction: float | None,
    accuracy: float | None,
) -> dict[str, Any]:
    if cost is None or epoch is None:
        raise RuntimeError(
            "Expected selected clean-validation cost and epoch before "
            "persisting progress. "
            f"Provided value: cost={cost!r}, epoch={epoch!r}."
        )
    return {
        "selected_metric": "validation.mean_cost",
        "selected_value": float(cost),
        "selected_epoch": int(epoch),
        "selected_error_fraction": error_fraction,
        "selected_accuracy": accuracy,
    }


def _epoch_report(
    *,
    epoch: int,
    completed_epoch: int,
    global_step: int,
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    improved: bool,
    selected_epoch: int | None,
    selected_cost: float | None,
    selected_error: float | None,
    selected_accuracy: float | None,
) -> TrainingEpochReport:
    if selected_epoch is None or selected_cost is None:
        raise RuntimeError(
            "Expected clean validation to define selection fields before "
            "reporting an epoch. "
            f"Provided value: epoch={selected_epoch!r}, cost={selected_cost!r}."
        )
    validation_cost = validation.get("mean_cost")
    validation_error = validation.get("mean_error")
    validation_accuracy = validation.get("accuracy")
    if (
        validation_cost is None
        or validation_error is None
        or validation_accuracy is None
    ):
        raise RuntimeError(
            "Expected completed clean validation metrics to include cost, "
            "error fraction, and accuracy. "
            f"Provided value: {dict(validation)!r}."
        )
    return TrainingEpochReport(
        epoch_index=epoch,
        completed_epochs=completed_epoch,
        global_step=global_step,
        train_examples=int(train["examples"]),
        train_mean_cost=_optional_float(train.get("mean_cost")),
        train_error_fraction=_optional_fraction(
            train.get("mean_error"),
            name="training error fraction",
        ),
        train_accuracy=_optional_fraction(
            train.get("accuracy"),
            name="training accuracy",
        ),
        validation_examples=int(validation["examples"]),
        validation_mean_cost=float(validation_cost),
        validation_error_fraction=float(validation_error),
        validation_accuracy=float(validation_accuracy),
        selected_this_epoch=improved,
        selected_epoch_index=selected_epoch,
        selected_validation_cost=float(selected_cost),
        selected_validation_error_fraction=selected_error,
        selected_validation_accuracy=selected_accuracy,
    )


def _validated_observers(
    observers: Iterable[TrainingObserver],
) -> tuple[TrainingObserver, ...]:
    try:
        resolved = tuple(observers)
    except TypeError as exc:
        raise TypeError(
            "Expected observers to be an iterable of callables accepting one "
            f"TrainingEpochReport. Provided value: {observers!r}."
        ) from exc
    for index, observer in enumerate(resolved):
        if not callable(observer):
            raise TypeError(
                "Expected every observer to be callable with one "
                f"TrainingEpochReport. Provided value: observers[{index}]="
                f"{observer!r}."
            )
    return resolved


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_fraction(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    valid = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )
    if not valid:
        raise ValueError(
            f"Expected {name} to be a finite fraction in [0, 1] or null. "
            f"Provided value: {value!r}."
        )
    return float(value)


def _resume_metadata(spec: TrainSpec) -> dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "schema_version": spec.schema_version,
        "resume_config_sha256": _resume_config_sha256(spec),
    }


def _resume_config_sha256(spec: TrainSpec) -> str:
    """Fingerprint every numerical setting except the target epoch horizon."""

    value = to_plain_data(spec)
    settings = value.get("settings")
    if not isinstance(settings, dict) or "num_epochs" not in settings:
        raise TypeError(
            "Expected a plain TrainSpec to contain settings.num_epochs. "
            f"Provided value: {value!r}."
        )
    settings = dict(settings)
    del settings["num_epochs"]
    value["settings"] = settings
    return content_hash(value)


def _limit_batches(
    loader: Iterable[Any],
    maximum: int | None,
) -> Iterable[Any]:
    return loader if maximum is None else islice(loader, maximum)


def _limit_examples(
    loader: Iterable[Any],
    maximum: int | None,
) -> Iterable[Any]:
    if maximum is None:
        return loader

    def limited():
        remaining = maximum
        for raw_batch in loader:
            if remaining <= 0:
                return
            if not isinstance(raw_batch, (tuple, list)) or len(raw_batch) not in (
                2,
                3,
            ):
                raise ValueError(
                    "Expected validation loader batches to be pairs or "
                    f"triples. Provided value: {raw_batch!r}."
                )
            batch_size = int(raw_batch[0].shape[0])
            take = min(batch_size, remaining)
            if take == batch_size:
                yield raw_batch
            else:
                yield tuple(item[:take] for item in raw_batch)
            remaining -= take

    return limited()


def _residual_metrics(
    values: Mapping[str, np.ndarray],
    *,
    tolerance: float | None,
) -> dict[str, Any]:
    per_layer = {
        name: _numeric_summary(array, tolerance=tolerance)
        for name, array in values.items()
    }
    # The first ordered layer is the clamped input and is not an equilibrium
    # residual.  Do not rely on legacy process-global layer-name counters.
    comparable = [
        np.asarray(array).reshape(-1)
        for index, array in enumerate(values.values())
        if index != 0 and np.asarray(array).size
    ]
    overall = (
        _numeric_summary(
            np.concatenate(comparable),
            tolerance=tolerance,
        )
        if comparable
        else None
    )
    return {"layers": per_layer, "overall": overall}


def _numeric_summary(
    value: Any,
    *,
    tolerance: float | None = None,
) -> dict[str, Any]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size and not np.isfinite(array).all():
        raise FloatingPointError(
            "Expected diagnostic values to be finite. "
            f"Provided value: {array!r}."
        )
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "max": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }
    result: dict[str, Any] = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
    }
    if tolerance is not None:
        result.update(
            {
                "tolerance": float(tolerance),
                "pass_rate": float(np.mean(array <= tolerance)),
                "pass_all": bool(np.all(array <= tolerance)),
            }
        )
    return result


def _training_input_artifacts(request: Any) -> tuple[Mapping[str, Any], ...]:
    records = []
    for role in ("weights", "base_weights", "resume"):
        path = getattr(request, role)
        if path is not None:
            records.append(_input_artifact(role, path))
    return tuple(records)


def _input_artifact(role: str, path: Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            "Expected input artifact path to name an existing file. "
            f"Provided value: {str(source)!r}."
        )
    return {
        "role": role,
        "path": str(source),
        "sha256": sha256_file(source),
    }


def _create_run_store(
    request: Any,
    spec: Any,
    *,
    input_artifacts: Iterable[Mapping[str, Any]],
    resume_capability: str,
    runtime_protocol: Mapping[str, Any],
) -> RunStore:
    resolved = to_plain_data(spec)
    resolved["runtime_protocol"] = dict(runtime_protocol)
    return RunStore.create(
        output_root=request.output_dir,
        experiment_id=spec.experiment_id,
        resolved_config=resolved,
        command=request.command,
        repo_root=Path(__file__).resolve().parents[2],
        input_artifacts=input_artifacts,
        resume_capability=resume_capability,
    )


def _expect_spec(value: Any, expected: type, *, mode: str) -> Any:
    if not isinstance(value, expected):
        raise TypeError(
            f"Expected {mode} request spec to be {expected.__name__}. "
            f"Provided value: {type(value).__name__}."
        )
    return value


def _atomic_npz(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _numpy_mapping(values: Mapping[str, Any]) -> dict[str, np.ndarray]:
    return {name: _as_numpy(value) for name, value in values.items()}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _concatenate(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(values, axis=0)


__all__ = [
    "TrainingEpochReport",
    "TrainingObserver",
    "TrainingOutcome",
    "execute_train",
    "import_legacy_checkpoint",
    "run_linspace",
    "run_train",
    "run_validate",
]
