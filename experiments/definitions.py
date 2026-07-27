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


# This dictionary is the complete registration mechanism.
EXPERIMENT_REGISTRY: Dict[str, ExperimentDefinition] = {
    SMALL_DRN_V1.experiment_id: SMALL_DRN_V1,
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
