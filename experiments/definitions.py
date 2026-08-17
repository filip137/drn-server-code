"""The explicit registry of executable experiment definitions.

Registration is deliberately a plain dictionary.  There is no import-time
discovery, decorator side effect, entry-point scan, or dynamic module path in
the config file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Union

from experiments.schema import (
    ConfigError,
    ExperimentDefinition,
    ExtensionSelection,
    RunMode,
    ValidatedCombination,
    config_error,
    to_plain_data,
)
from experiments.small_network.config import (
    EXPERIMENT_ID,
    SCHEMA_VERSION,
    SmallDrnConfig,
    SmallDrnSpec,
    TrainSpec,
    parse_small_drn_config,
    resolve_small_drn_spec,
)
from experiments.mnist_relu.config import (
    EXPERIMENT_ID as MNIST_RELU_EXPERIMENT_ID,
    SCHEMA_VERSION as MNIST_RELU_SCHEMA_VERSION,
    parse_teacher_config,
    resolve_teacher_spec,
)
from experiments.mnist_relu_drn.config import (
    EXPERIMENT_ID as MNIST_RELU_DRN_EXPERIMENT_ID,
    SCHEMA_VERSION as MNIST_RELU_DRN_SCHEMA_VERSION,
    StudentTrainSpec,
    parse_student_config,
    resolve_student_spec,
)


_SMALL_DRN_COMBINATIONS: Tuple[ValidatedCombination, ...] = (
    ValidatedCombination(
        ExtensionSelection("none", "none", "direct", "ep"),
        "validated",
        "Reference equilibrium-propagation path.",
    ),
    ValidatedCombination(
        ExtensionSelection("none", "none", "direct", "backprop"),
        "validated",
        "Reference backpropagation path.",
    ),
    ValidatedCombination(
        ExtensionSelection("none", "none", "tiki_taka", "ep"),
        "experimental",
        "Tiki-taka backend with either ideal or AIHWKit parameters.",
    ),
    ValidatedCombination(
        ExtensionSelection("none", "none", "tiki_taka", "backprop"),
        "experimental",
        "Tiki-taka backend with backpropagation.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "passive_low_rank",
            "none",
            "direct",
            "ep",
        ),
        "experimental",
        "Passive low-rank factors trained by direct EP updates.",
    ),
    ValidatedCombination(
        ExtensionSelection(
            "passive_low_rank",
            "none",
            "tiki_taka",
            "ep",
        ),
        "experimental",
        "Passive low-rank factors trained by ideal-tensor Tiki-Taka.",
    ),
    ValidatedCombination(
        ExtensionSelection("none", "add_normal", "direct", "backprop"),
        "experimental",
        "Additive Gaussian weight noise with backpropagation.",
    ),
)


def _resolve_small_drn(
    document: SmallDrnConfig,
    mode: RunMode,
) -> SmallDrnSpec:
    spec = resolve_small_drn_spec(document, mode)
    if isinstance(spec, TrainSpec):
        combination = next(
            (
                item
                for item in _SMALL_DRN_COMBINATIONS
                if item.selection == spec.extensions
            ),
            None,
        )
        if combination is None:
            raise config_error(
                "the train extension combination "
                "(model.adapter, weight_modifier, update_backend, algorithm)",
                "to be listed explicitly by the 'small_drn.v1' definition",
                (
                    spec.extensions.model_adapter,
                    spec.extensions.weight_modifier,
                    spec.extensions.update_backend,
                    spec.extensions.algorithm,
                ),
            )
        if (
            spec.extensions.model_adapter == "passive_low_rank"
            and spec.extensions.update_backend == "tiki_taka"
            and spec.settings.update_backend.parameters.get(
                "aihwkit_preset"
            )
            is not None
        ):
            raise config_error(
                "config.modes.train.update_backend.parameters.aihwkit_preset",
                "to be null or omitted for passive_low_rank because only the "
                "ideal-tensor Tiki-Taka backend is validated",
                spec.settings.update_backend.parameters.get(
                    "aihwkit_preset"
                ),
            )
    return spec


SMALL_DRN_V1 = ExperimentDefinition(
    experiment_id=EXPERIMENT_ID,
    schema_version=SCHEMA_VERSION,
    description=(
        "Small dissipative-resistive-network training, linspace analysis, "
        "and checkpoint validation."
    ),
    supported_modes=(
        RunMode.TRAIN,
        RunMode.LINSPACE,
        RunMode.VALIDATE,
    ),
    parser=parse_small_drn_config,
    resolver=_resolve_small_drn,
    combinations=_SMALL_DRN_COMBINATIONS,
    legacy_names=("labs.small_network",),
)


MNIST_RELU_V1 = ExperimentDefinition(
    experiment_id=MNIST_RELU_EXPERIMENT_ID,
    schema_version=MNIST_RELU_SCHEMA_VERSION,
    description=(
        "Bias-free 784-50-10 ReLU MNIST teacher training and validation."
    ),
    supported_modes=(RunMode.TRAIN, RunMode.VALIDATE),
    parser=parse_teacher_config,
    resolver=resolve_teacher_spec,
    combinations=(
        ValidatedCombination(
            ExtensionSelection(
                "relu_teacher",
                "none",
                "adam",
                "cross_entropy",
            ),
            "validated",
            "Bias-free digital teacher selected by validation cross-entropy.",
        ),
    ),
)


_MNIST_RELU_DRN_COMBINATIONS: Tuple[ValidatedCombination, ...] = tuple(
    ValidatedCombination(
        ExtensionSelection(encoding, "none", backend, "teacher_kl"),
        "experimental",
        (
            "Teacher-mapped DRN with pure KL distillation and "
            + (
                "one device per physical edge."
                if encoding == "single"
                else (
                    "a differential G+/G- pair per physical edge with "
                    "finite positive voltage/current amplifier magnitudes."
                )
            )
        ),
    )
    for encoding in ("single", "differential")
    for backend in ("ideal", "measured_cohort_a")
)


def _resolve_mnist_relu_drn(document, mode: RunMode):
    spec = resolve_student_spec(document, mode)
    if isinstance(spec, StudentTrainSpec):
        selection = ExtensionSelection(
            spec.model.encoding,
            "none",
            spec.settings.update_backend.type,
            "teacher_kl",
        )
        if not any(
            item.selection == selection
            for item in _MNIST_RELU_DRN_COMBINATIONS
        ):
            raise config_error(
                "the train extension combination",
                "to be listed explicitly by the "
                "'mnist_relu_drn_kd.v1' definition",
                to_plain_data(selection),
            )
    return spec


MNIST_RELU_DRN_KD_V1 = ExperimentDefinition(
    experiment_id=MNIST_RELU_DRN_EXPERIMENT_ID,
    schema_version=MNIST_RELU_DRN_SCHEMA_VERSION,
    description=(
        "Teacher-initialized MNIST DRN trained with pure teacher-to-student KL."
    ),
    supported_modes=(RunMode.TRAIN, RunMode.VALIDATE),
    parser=parse_student_config,
    resolver=_resolve_mnist_relu_drn,
    combinations=_MNIST_RELU_DRN_COMBINATIONS,
)


# This dictionary is the complete registration mechanism.
EXPERIMENT_REGISTRY: Dict[str, ExperimentDefinition] = {
    SMALL_DRN_V1.experiment_id: SMALL_DRN_V1,
    MNIST_RELU_V1.experiment_id: MNIST_RELU_V1,
    MNIST_RELU_DRN_KD_V1.experiment_id: MNIST_RELU_DRN_KD_V1,
}


def list_definitions() -> Tuple[ExperimentDefinition, ...]:
    """Return definitions in stable identifier order."""

    return tuple(
        EXPERIMENT_REGISTRY[key]
        for key in sorted(EXPERIMENT_REGISTRY)
    )


def get_definition(experiment_id: str) -> ExperimentDefinition:
    """Return one registered definition or raise a user-facing config error."""

    try:
        return EXPERIMENT_REGISTRY[experiment_id]
    except KeyError as error:
        raise config_error(
            "config.experiment_id",
            "to be one of "
            + ", ".join(repr(key) for key in sorted(EXPERIMENT_REGISTRY)),
            experiment_id,
        ) from error


def parse_experiment_config(
    payload: Mapping[str, Any],
) -> Tuple[ExperimentDefinition, Any]:
    """Select the definition from the document and parse it strictly."""

    if not isinstance(payload, Mapping):
        raise config_error("config", "to be a JSON object", payload)
    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise config_error(
            "config.experiment_id",
            "to be a non-empty registered identifier",
            experiment_id,
        )
    definition = get_definition(experiment_id)
    return definition, definition.parse(payload)


def load_experiment_config(
    path: Union[str, Path],
) -> Tuple[ExperimentDefinition, Any]:
    """Read and parse one versioned JSON experiment document."""

    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(
            "Expected --config to reference a readable JSON file. "
            f"Provided value: {str(config_path)!r}. {error}"
        ) from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError(
            "Expected --config to contain valid JSON. "
            f"Provided value: {str(config_path)!r} "
            f"(line {error.lineno}, column {error.colno}: {error.msg})."
        ) from error
    return parse_experiment_config(payload)


def resolve_experiment_config(
    path: Union[str, Path],
    mode: Union[str, RunMode],
) -> Tuple[ExperimentDefinition, Any]:
    """Load a config and resolve the selected mode-specific immutable spec."""

    try:
        run_mode = mode if isinstance(mode, RunMode) else RunMode(mode)
    except ValueError as error:
        raise config_error(
            "the requested run mode",
            "to be 'train', 'linspace', or 'validate'",
            mode,
        ) from error
    definition, document = load_experiment_config(path)
    return definition, definition.resolve(document, run_mode)
