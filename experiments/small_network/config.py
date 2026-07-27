"""Strict, immutable configuration for the ``small_drn.v1`` experiment.

This module describes scientific intent only.  Paths to input checkpoints,
output directories, resume snapshots, and other operational concerns belong
to the CLI request, not this document.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from experiments.schema import (
    ConfigError,
    ExtensionSelection,
    RunMode,
    config_error,
    freeze_json,
)


EXPERIMENT_ID = "small_drn.v1"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RuntimeSettings:
    seed: Optional[int]
    data_seed: Optional[int]
    device: str
    dtype: str


@dataclass(frozen=True)
class DataSettings:
    dataset: str
    batch_size: int
    num_points: Optional[int]
    shuffle: bool


@dataclass(frozen=True)
class NonLinearitySettings:
    type: str
    quadratic_diode_param: Mapping[str, Union[int, float]]
    exponential_diode_param: Mapping[str, Union[int, float]]
    hard_sigmoid_param: Mapping[str, Union[int, float]]
    iv_data_path: Optional[str]


@dataclass(frozen=True)
class ComponentSettings:
    """A named extension and its recursively immutable parameter object."""

    type: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class ModelSettings:
    dims: Tuple[int, ...]
    input_gain: float
    weight_gains: Tuple[float, ...]
    weight_min: float
    weight_max: float
    voltage_amp: float
    current_amp: float
    non_linearity: NonLinearitySettings
    adapter: ComponentSettings


@dataclass(frozen=True)
class UpdaterSettings:
    double_diode: Optional[str]
    single_diode: Optional[str]


@dataclass(frozen=True)
class ToleranceSettings:
    relative: float
    voltage: float
    residual_current: Optional[float]


@dataclass(frozen=True)
class PolishSettings:
    enabled: bool
    dynamic: bool
    max_newton_iterations: int
    z_threshold: float
    exponential_clip: float


@dataclass(frozen=True)
class AndersonSettings:
    memory: int
    omega: float
    tolerance_floor: float
    regularization: float


@dataclass(frozen=True)
class OverrelaxationSettings:
    factor: float
    reject_steps: bool
    reject_max_tries: int
    reject_shrink: float
    reject_epsilon: float


@dataclass(frozen=True)
class ExperimentalExponentialSettings:
    damping: float
    newton_max_steps: int
    progressive_tolerance: bool
    tolerance_start: float
    tolerance_end: float
    tolerance_switch_high: float
    tolerance_switch_low: float


@dataclass(frozen=True)
class SolverSettings:
    inference_iterations: int
    training_iterations: int
    minimizer_impl: str
    minimizer_mode: str
    adaptive_equilibrium: bool
    random_initialization: bool
    updaters: UpdaterSettings
    tolerances: ToleranceSettings
    polish: PolishSettings
    anderson: AndersonSettings
    overrelaxation: OverrelaxationSettings
    experimental_exponential: ExperimentalExponentialSettings


@dataclass(frozen=True)
class CommonSettings:
    runtime: RuntimeSettings
    data: DataSettings
    model: ModelSettings
    solver: SolverSettings


@dataclass(frozen=True)
class TrainSettings:
    num_epochs: int
    algorithm: str
    learning_rates: Tuple[float, ...]
    bias_learning_rates: Tuple[float, ...]
    nudging: float
    log_every: int
    max_batches: Optional[int]
    max_validation_batches: Optional[int]
    weight_modifier: ComponentSettings
    update_backend: ComponentSettings


@dataclass(frozen=True)
class LinspaceSettings:
    minimum: float
    maximum: float
    samples: int
    record_states: bool


@dataclass(frozen=True)
class ValidateSettings:
    split: str
    sample_limit: Optional[int]
    record_states: bool


@dataclass(frozen=True)
class SmallDrnConfig:
    schema_version: int
    experiment_id: str
    common: CommonSettings
    train: Optional[TrainSettings]
    linspace: Optional[LinspaceSettings]
    validate: Optional[ValidateSettings]


@dataclass(frozen=True)
class TrainSpec:
    schema_version: int
    experiment_id: str
    common: CommonSettings
    settings: TrainSettings

    @property
    def extensions(self) -> ExtensionSelection:
        return ExtensionSelection(
            model_adapter=self.common.model.adapter.type,
            weight_modifier=self.settings.weight_modifier.type,
            update_backend=self.settings.update_backend.type,
            algorithm=self.settings.algorithm,
        )


@dataclass(frozen=True)
class LinspaceSpec:
    schema_version: int
    experiment_id: str
    common: CommonSettings
    settings: LinspaceSettings


@dataclass(frozen=True)
class ValidateSpec:
    schema_version: int
    experiment_id: str
    common: CommonSettings
    settings: ValidateSettings


SmallDrnSpec = Union[TrainSpec, LinspaceSpec, ValidateSpec]


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise config_error(path, "to be a JSON object", value)
    for key in value:
        if not isinstance(key, str):
            raise config_error(
                path,
                "to be a JSON object with string keys",
                value,
            )
    return value


def _check_keys(
    value: Mapping[str, Any],
    path: str,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise config_error(
            path,
            "to define the required keys "
            + ", ".join(repr(item) for item in required),
            dict(value),
        )
    allowed = set(required) | set(optional)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise config_error(
            path,
            "to contain only the keys "
            + ", ".join(repr(item) for item in sorted(allowed)),
            unknown,
        )


def _string(
    value: Any,
    path: str,
    *,
    choices: Optional[Sequence[str]] = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise config_error(path, "to be a non-empty string", value)
    if choices is not None and value not in choices:
        raise config_error(
            path,
            "to be one of " + ", ".join(repr(item) for item in choices),
            value,
        )
    return value


def _optional_string(value: Any, path: str) -> Optional[str]:
    if value is None:
        return None
    return _string(value, path)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise config_error(path, "to be a boolean", value)
    return value


def _integer(
    value: Any,
    path: str,
    *,
    minimum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise config_error(path, "to be an integer", value)
    if minimum is not None and value < minimum:
        raise config_error(path, f"to be an integer >= {minimum}", value)
    return value


def _optional_integer(
    value: Any,
    path: str,
    *,
    minimum: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return None
    return _integer(value, path, minimum=minimum)


def _number(
    value: Any,
    path: str,
    *,
    minimum: Optional[float] = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise config_error(path, "to be a finite number", value)
    parsed = float(value)
    if not math.isfinite(parsed):
        raise config_error(path, "to be a finite number", value)
    if strictly_positive and parsed <= 0.0:
        raise config_error(path, "to be a finite number > 0", value)
    if minimum is not None and parsed < minimum:
        raise config_error(
            path,
            f"to be a finite number >= {minimum}",
            value,
        )
    return parsed


def _optional_number(
    value: Any,
    path: str,
    *,
    minimum: Optional[float] = None,
) -> Optional[float]:
    if value is None:
        return None
    return _number(value, path, minimum=minimum)


def _number_tuple(
    value: Any,
    path: str,
    *,
    nonempty: bool = False,
    minimum: Optional[float] = None,
) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise config_error(path, "to be a JSON array of finite numbers", value)
    if nonempty and not value:
        raise config_error(
            path,
            "to be a non-empty JSON array of finite numbers",
            value,
        )
    return tuple(
        _number(item, f"{path}[{index}]", minimum=minimum)
        for index, item in enumerate(value)
    )


def _numeric_parameter_object(
    value: Any,
    path: str,
) -> Mapping[str, Union[int, float]]:
    """Require an explicit diode dictionary and freeze its numeric values."""

    parsed = _object(value, path)
    numeric = {}
    for key, item in parsed.items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise config_error(
                f"{path}.{key}",
                "to be a finite number",
                item,
            )
        if not math.isfinite(float(item)):
            raise config_error(
                f"{path}.{key}",
                "to be a finite number",
                item,
            )
        numeric[key] = item
    return freeze_json(numeric, path=path)


def _component(
    value: Any,
    path: str,
    *,
    allowed_types: Sequence[str],
) -> ComponentSettings:
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=("type", "parameters"),
    )
    component_type = _string(
        parsed["type"],
        f"{path}.type",
        choices=allowed_types,
    )
    parameters = _object(parsed["parameters"], f"{path}.parameters")
    if component_type == "none" and parameters:
        raise config_error(
            f"{path}.parameters",
            "to be an empty object when type is 'none'",
            dict(parameters),
        )
    return ComponentSettings(
        type=component_type,
        parameters=freeze_json(
            parameters,
            path=f"{path}.parameters",
        ),
    )


def _parse_runtime(value: Any) -> RuntimeSettings:
    path = "config.runtime"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=("seed", "data_seed", "device", "dtype"),
    )
    return RuntimeSettings(
        seed=_optional_integer(parsed["seed"], f"{path}.seed", minimum=0),
        data_seed=_optional_integer(
            parsed["data_seed"],
            f"{path}.data_seed",
            minimum=0,
        ),
        device=_string(parsed["device"], f"{path}.device"),
        dtype=_string(
            parsed["dtype"],
            f"{path}.dtype",
            choices=("float32", "float64"),
        ),
    )


def _parse_data(value: Any) -> DataSettings:
    path = "config.data"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=("dataset", "batch_size", "num_points", "shuffle"),
    )
    return DataSettings(
        dataset=_string(
            parsed["dataset"],
            f"{path}.dataset",
            choices=("moons", "yinyang", "digits"),
        ),
        batch_size=_integer(
            parsed["batch_size"],
            f"{path}.batch_size",
            minimum=1,
        ),
        num_points=_optional_integer(
            parsed["num_points"],
            f"{path}.num_points",
            minimum=1,
        ),
        shuffle=_boolean(parsed["shuffle"], f"{path}.shuffle"),
    )


def _parse_non_linearity(value: Any) -> NonLinearitySettings:
    path = "config.model.non_linearity"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=(
            "type",
            "quadratic_diode_param",
            "exponential_diode_param",
            "hard_sigmoid_param",
            "iv_data_path",
        ),
    )
    iv_data_path = parsed["iv_data_path"]
    if iv_data_path is not None:
        iv_data_path = _string(iv_data_path, f"{path}.iv_data_path")
    return NonLinearitySettings(
        type=_string(
            parsed["type"],
            f"{path}.type",
            choices=(
                "perfect_diode",
                "lpw_diode",
                "double_diode",
                "double_diode_quadratic",
                "double_diode_exponential",
                "single_diode_exponential",
                "hard_sigmoid",
                "linear",
                "experimental",
            ),
        ),
        quadratic_diode_param=_numeric_parameter_object(
            parsed["quadratic_diode_param"],
            f"{path}.quadratic_diode_param",
        ),
        exponential_diode_param=_numeric_parameter_object(
            parsed["exponential_diode_param"],
            f"{path}.exponential_diode_param",
        ),
        hard_sigmoid_param=_numeric_parameter_object(
            parsed["hard_sigmoid_param"],
            f"{path}.hard_sigmoid_param",
        ),
        iv_data_path=iv_data_path,
    )


def _parse_model(value: Any) -> ModelSettings:
    path = "config.model"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=(
            "dims",
            "input_gain",
            "weight_gains",
            "weight_min",
            "weight_max",
            "voltage_amp",
            "current_amp",
            "non_linearity",
            "adapter",
        ),
    )

    dims_value = parsed["dims"]
    if not isinstance(dims_value, (list, tuple)) or len(dims_value) < 2:
        raise config_error(
            f"{path}.dims",
            "to be an array of at least two positive integers",
            dims_value,
        )
    dims = tuple(
        _integer(item, f"{path}.dims[{index}]", minimum=1)
        for index, item in enumerate(dims_value)
    )
    weight_gains = _number_tuple(
        parsed["weight_gains"],
        f"{path}.weight_gains",
        nonempty=True,
        minimum=0.0,
    )
    expected_layers = len(dims) - 1
    if len(weight_gains) != expected_layers:
        raise config_error(
            f"{path}.weight_gains",
            f"to contain exactly {expected_layers} values "
            "(one per weight layer)",
            list(weight_gains),
        )

    weight_min = _number(parsed["weight_min"], f"{path}.weight_min")
    weight_max = _number(parsed["weight_max"], f"{path}.weight_max")
    if weight_min > weight_max:
        raise config_error(
            f"{path}.weight_min and {path}.weight_max",
            "to satisfy weight_min <= weight_max",
            {"weight_min": weight_min, "weight_max": weight_max},
        )

    return ModelSettings(
        dims=dims,
        input_gain=_number(
            parsed["input_gain"],
            f"{path}.input_gain",
            strictly_positive=True,
        ),
        weight_gains=weight_gains,
        weight_min=weight_min,
        weight_max=weight_max,
        voltage_amp=_number(
            parsed["voltage_amp"],
            f"{path}.voltage_amp",
            strictly_positive=True,
        ),
        current_amp=_number(
            parsed["current_amp"],
            f"{path}.current_amp",
            strictly_positive=True,
        ),
        non_linearity=_parse_non_linearity(parsed["non_linearity"]),
        adapter=_component(
            parsed["adapter"],
            f"{path}.adapter",
            allowed_types=("none", "passive_low_rank"),
        ),
    )


def _parse_solver(value: Any) -> SolverSettings:
    path = "config.solver"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=(
            "inference_iterations",
            "training_iterations",
            "minimizer_impl",
            "minimizer_mode",
            "adaptive_equilibrium",
            "random_initialization",
            "updaters",
            "tolerances",
            "polish",
            "anderson",
            "overrelaxation",
            "experimental_exponential",
        ),
    )

    updaters_path = f"{path}.updaters"
    updaters = _object(parsed["updaters"], updaters_path)
    _check_keys(
        updaters,
        updaters_path,
        required=("double_diode", "single_diode"),
    )

    tolerances_path = f"{path}.tolerances"
    tolerances = _object(parsed["tolerances"], tolerances_path)
    _check_keys(
        tolerances,
        tolerances_path,
        required=("relative", "voltage", "residual_current"),
    )

    polish_path = f"{path}.polish"
    polish = _object(parsed["polish"], polish_path)
    _check_keys(
        polish,
        polish_path,
        required=(
            "enabled",
            "dynamic",
            "max_newton_iterations",
            "z_threshold",
            "exponential_clip",
        ),
    )

    anderson_path = f"{path}.anderson"
    anderson = _object(parsed["anderson"], anderson_path)
    _check_keys(
        anderson,
        anderson_path,
        required=(
            "memory",
            "omega",
            "tolerance_floor",
            "regularization",
        ),
    )

    overrelaxation_path = f"{path}.overrelaxation"
    overrelaxation = _object(
        parsed["overrelaxation"],
        overrelaxation_path,
    )
    _check_keys(
        overrelaxation,
        overrelaxation_path,
        required=(
            "factor",
            "reject_steps",
            "reject_max_tries",
            "reject_shrink",
            "reject_epsilon",
        ),
    )

    experimental_path = f"{path}.experimental_exponential"
    experimental = _object(
        parsed["experimental_exponential"],
        experimental_path,
    )
    _check_keys(
        experimental,
        experimental_path,
        required=(
            "damping",
            "newton_max_steps",
            "progressive_tolerance",
            "tolerance_start",
            "tolerance_end",
            "tolerance_switch_high",
            "tolerance_switch_low",
        ),
    )

    settings = SolverSettings(
        inference_iterations=_integer(
            parsed["inference_iterations"],
            f"{path}.inference_iterations",
            minimum=1,
        ),
        training_iterations=_integer(
            parsed["training_iterations"],
            f"{path}.training_iterations",
            minimum=1,
        ),
        minimizer_impl=_string(
            parsed["minimizer_impl"],
            f"{path}.minimizer_impl",
            choices=("custom", "anderson"),
        ),
        minimizer_mode=_string(
            parsed["minimizer_mode"],
            f"{path}.minimizer_mode",
            choices=(
                "forward",
                "backward",
                "synchronous",
                "asynchronous",
            ),
        ),
        adaptive_equilibrium=_boolean(
            parsed["adaptive_equilibrium"],
            f"{path}.adaptive_equilibrium",
        ),
        random_initialization=_boolean(
            parsed["random_initialization"],
            f"{path}.random_initialization",
        ),
        updaters=UpdaterSettings(
            double_diode=_optional_string(
                updaters["double_diode"],
                f"{updaters_path}.double_diode",
            ),
            single_diode=_optional_string(
                updaters["single_diode"],
                f"{updaters_path}.single_diode",
            ),
        ),
        tolerances=ToleranceSettings(
            relative=_number(
                tolerances["relative"],
                f"{tolerances_path}.relative",
                minimum=0.0,
            ),
            voltage=_number(
                tolerances["voltage"],
                f"{tolerances_path}.voltage",
                minimum=0.0,
            ),
            residual_current=_optional_number(
                tolerances["residual_current"],
                f"{tolerances_path}.residual_current",
                minimum=0.0,
            ),
        ),
        polish=PolishSettings(
            enabled=_boolean(
                polish["enabled"],
                f"{polish_path}.enabled",
            ),
            dynamic=_boolean(
                polish["dynamic"],
                f"{polish_path}.dynamic",
            ),
            max_newton_iterations=_integer(
                polish["max_newton_iterations"],
                f"{polish_path}.max_newton_iterations",
                minimum=1,
            ),
            z_threshold=_number(
                polish["z_threshold"],
                f"{polish_path}.z_threshold",
                strictly_positive=True,
            ),
            exponential_clip=_number(
                polish["exponential_clip"],
                f"{polish_path}.exponential_clip",
                strictly_positive=True,
            ),
        ),
        anderson=AndersonSettings(
            memory=_integer(
                anderson["memory"],
                f"{anderson_path}.memory",
                minimum=1,
            ),
            omega=_number(
                anderson["omega"],
                f"{anderson_path}.omega",
                strictly_positive=True,
            ),
            tolerance_floor=_number(
                anderson["tolerance_floor"],
                f"{anderson_path}.tolerance_floor",
                minimum=0.0,
            ),
            regularization=_number(
                anderson["regularization"],
                f"{anderson_path}.regularization",
                minimum=0.0,
            ),
        ),
        overrelaxation=OverrelaxationSettings(
            factor=_number(
                overrelaxation["factor"],
                f"{overrelaxation_path}.factor",
                strictly_positive=True,
            ),
            reject_steps=_boolean(
                overrelaxation["reject_steps"],
                f"{overrelaxation_path}.reject_steps",
            ),
            reject_max_tries=_integer(
                overrelaxation["reject_max_tries"],
                f"{overrelaxation_path}.reject_max_tries",
                minimum=1,
            ),
            reject_shrink=_number(
                overrelaxation["reject_shrink"],
                f"{overrelaxation_path}.reject_shrink",
                strictly_positive=True,
            ),
            reject_epsilon=_number(
                overrelaxation["reject_epsilon"],
                f"{overrelaxation_path}.reject_epsilon",
                minimum=0.0,
            ),
        ),
        experimental_exponential=ExperimentalExponentialSettings(
            damping=_number(
                experimental["damping"],
                f"{experimental_path}.damping",
                strictly_positive=True,
            ),
            newton_max_steps=_integer(
                experimental["newton_max_steps"],
                f"{experimental_path}.newton_max_steps",
                minimum=1,
            ),
            progressive_tolerance=_boolean(
                experimental["progressive_tolerance"],
                f"{experimental_path}.progressive_tolerance",
            ),
            tolerance_start=_number(
                experimental["tolerance_start"],
                f"{experimental_path}.tolerance_start",
                strictly_positive=True,
            ),
            tolerance_end=_number(
                experimental["tolerance_end"],
                f"{experimental_path}.tolerance_end",
                strictly_positive=True,
            ),
            tolerance_switch_high=_number(
                experimental["tolerance_switch_high"],
                f"{experimental_path}.tolerance_switch_high",
                strictly_positive=True,
            ),
            tolerance_switch_low=_number(
                experimental["tolerance_switch_low"],
                f"{experimental_path}.tolerance_switch_low",
                strictly_positive=True,
            ),
        ),
    )
    if settings.overrelaxation.reject_shrink >= 1.0:
        raise config_error(
            f"{overrelaxation_path}.reject_shrink",
            "to be smaller than 1.0",
            settings.overrelaxation.reject_shrink,
        )
    if (
        settings.experimental_exponential.tolerance_switch_high
        <= settings.experimental_exponential.tolerance_switch_low
    ):
        raise config_error(
            f"{experimental_path}.tolerance_switch_high and "
            f"{experimental_path}.tolerance_switch_low",
            "to satisfy tolerance_switch_high > tolerance_switch_low > 0",
            {
                "tolerance_switch_high": (
                    settings.experimental_exponential.tolerance_switch_high
                ),
                "tolerance_switch_low": (
                    settings.experimental_exponential.tolerance_switch_low
                ),
            },
        )
    return settings


def _parse_train(value: Any, *, layer_count: int) -> TrainSettings:
    path = "config.modes.train"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=(
            "num_epochs",
            "algorithm",
            "learning_rates",
            "bias_learning_rates",
            "nudging",
            "log_every",
            "max_batches",
            "max_validation_batches",
            "weight_modifier",
            "update_backend",
        ),
    )
    learning_rates = _number_tuple(
        parsed["learning_rates"],
        f"{path}.learning_rates",
        nonempty=True,
        minimum=0.0,
    )
    if len(learning_rates) != layer_count:
        raise config_error(
            f"{path}.learning_rates",
            f"to contain exactly {layer_count} values "
            "(one per weight layer)",
            list(learning_rates),
        )
    bias_learning_rates = _number_tuple(
        parsed["bias_learning_rates"],
        f"{path}.bias_learning_rates",
        minimum=0.0,
    )
    expected_biases = max(0, layer_count - 1)
    if len(bias_learning_rates) != expected_biases:
        raise config_error(
            f"{path}.bias_learning_rates",
            f"to contain exactly {expected_biases} values "
            "(one per hidden-layer bias; use 0.0 to freeze one)",
            list(bias_learning_rates),
        )
    algorithm = _string(
        parsed["algorithm"],
        f"{path}.algorithm",
        choices=("ep", "backprop"),
    )
    nudging = _number(parsed["nudging"], f"{path}.nudging")
    if algorithm == "ep" and nudging == 0.0:
        raise config_error(
            f"{path}.nudging",
            "to be non-zero for equilibrium propagation",
            nudging,
        )
    return TrainSettings(
        num_epochs=_integer(
            parsed["num_epochs"],
            f"{path}.num_epochs",
            minimum=1,
        ),
        algorithm=algorithm,
        learning_rates=learning_rates,
        bias_learning_rates=bias_learning_rates,
        nudging=nudging,
        log_every=_integer(
            parsed["log_every"],
            f"{path}.log_every",
            minimum=1,
        ),
        max_batches=_optional_integer(
            parsed["max_batches"],
            f"{path}.max_batches",
            minimum=1,
        ),
        max_validation_batches=_optional_integer(
            parsed["max_validation_batches"],
            f"{path}.max_validation_batches",
            minimum=1,
        ),
        weight_modifier=_component(
            parsed["weight_modifier"],
            f"{path}.weight_modifier",
            allowed_types=("none", "add_normal"),
        ),
        update_backend=_component(
            parsed["update_backend"],
            f"{path}.update_backend",
            allowed_types=("direct", "tiki_taka"),
        ),
    )


def _parse_linspace(value: Any) -> LinspaceSettings:
    path = "config.modes.linspace"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=("minimum", "maximum", "samples", "record_states"),
    )
    minimum = _number(parsed["minimum"], f"{path}.minimum")
    maximum = _number(parsed["maximum"], f"{path}.maximum")
    if minimum >= maximum:
        raise config_error(
            f"{path}.minimum and {path}.maximum",
            "to satisfy minimum < maximum",
            {"minimum": minimum, "maximum": maximum},
        )
    return LinspaceSettings(
        minimum=minimum,
        maximum=maximum,
        samples=_integer(
            parsed["samples"],
            f"{path}.samples",
            minimum=2,
        ),
        record_states=_boolean(
            parsed["record_states"],
            f"{path}.record_states",
        ),
    )


def _parse_validate(value: Any) -> ValidateSettings:
    path = "config.modes.validate"
    parsed = _object(value, path)
    _check_keys(
        parsed,
        path,
        required=("split", "sample_limit", "record_states"),
    )
    return ValidateSettings(
        split=_string(
            parsed["split"],
            f"{path}.split",
            choices=("train", "validation", "test"),
        ),
        sample_limit=_optional_integer(
            parsed["sample_limit"],
            f"{path}.sample_limit",
            minimum=1,
        ),
        record_states=_boolean(
            parsed["record_states"],
            f"{path}.record_states",
        ),
    )


def parse_small_drn_config(payload: Mapping[str, Any]) -> SmallDrnConfig:
    """Parse one complete ``small_drn.v1`` document.

    Every supplied mode is validated, while a document may intentionally
    contain only the modes used by a particular campaign.
    """

    root = _object(payload, "config")
    _check_keys(
        root,
        "config",
        required=(
            "schema_version",
            "experiment_id",
            "runtime",
            "data",
            "model",
            "solver",
            "modes",
        ),
    )
    schema_version = _integer(
        root["schema_version"],
        "config.schema_version",
        minimum=1,
    )
    if schema_version != SCHEMA_VERSION:
        raise config_error(
            "config.schema_version",
            f"to equal {SCHEMA_VERSION} for {EXPERIMENT_ID!r}",
            schema_version,
        )
    experiment_id = _string(
        root["experiment_id"],
        "config.experiment_id",
    )
    if experiment_id != EXPERIMENT_ID:
        raise config_error(
            "config.experiment_id",
            f"to equal {EXPERIMENT_ID!r}",
            experiment_id,
        )

    model = _parse_model(root["model"])
    runtime = _parse_runtime(root["runtime"])
    data = _parse_data(root["data"])
    if model.dims[0] % 2:
        raise config_error(
            "config.model.dims[0]",
            "to be even because inputs use positive/negative paired nodes",
            model.dims[0],
        )
    logical_input = model.dims[0] // 2
    expected_inputs = {"yinyang": 2, "digits": 64}
    if data.dataset in expected_inputs and logical_input != expected_inputs[
        data.dataset
    ]:
        raise config_error(
            "config.model.dims[0]",
            f"to equal {2 * expected_inputs[data.dataset]} for "
            f"dataset {data.dataset!r}",
            model.dims[0],
        )
    class_count = {"moons": 2, "yinyang": 3, "digits": 10}[data.dataset]
    if model.dims[-1] not in (class_count, 2 * class_count):
        raise config_error(
            "config.model.dims[-1]",
            f"to equal {class_count} (standard output) or "
            f"{2 * class_count} (differential output) for "
            f"dataset {data.dataset!r}",
            model.dims[-1],
        )
    if data.dataset in {"moons", "yinyang"} and data.num_points is None:
        raise config_error(
            "config.data.num_points",
            f"to be an integer for dataset {data.dataset!r}",
            data.num_points,
        )
    common = CommonSettings(
        runtime=runtime,
        data=data,
        model=model,
        solver=_parse_solver(root["solver"]),
    )

    modes = _object(root["modes"], "config.modes")
    _check_keys(
        modes,
        "config.modes",
        required=(),
        optional=("train", "linspace", "validate"),
    )
    if not modes:
        raise config_error(
            "config.modes",
            "to define at least one of 'train', 'linspace', or 'validate'",
            dict(modes),
        )
    layer_count = len(model.dims) - 1
    return SmallDrnConfig(
        schema_version=schema_version,
        experiment_id=experiment_id,
        common=common,
        train=(
            _parse_train(modes["train"], layer_count=layer_count)
            if "train" in modes
            else None
        ),
        linspace=(
            _parse_linspace(modes["linspace"])
            if "linspace" in modes
            else None
        ),
        validate=(
            _parse_validate(modes["validate"])
            if "validate" in modes
            else None
        ),
    )


def resolve_small_drn_spec(
    document: SmallDrnConfig,
    mode: RunMode,
) -> SmallDrnSpec:
    """Resolve a parsed document to the selected immutable mode spec."""

    if mode is RunMode.TRAIN:
        if document.train is None:
            raise config_error(
                "config.modes.train",
                "to be present for the 'train' command",
                None,
            )
        return TrainSpec(
            schema_version=document.schema_version,
            experiment_id=document.experiment_id,
            common=document.common,
            settings=document.train,
        )
    if mode is RunMode.LINSPACE:
        if document.linspace is None:
            raise config_error(
                "config.modes.linspace",
                "to be present for the 'linspace' command",
                None,
            )
        return LinspaceSpec(
            schema_version=document.schema_version,
            experiment_id=document.experiment_id,
            common=document.common,
            settings=document.linspace,
        )
    if mode is RunMode.VALIDATE:
        if document.validate is None:
            raise config_error(
                "config.modes.validate",
                "to be present for the 'validate' command",
                None,
            )
        return ValidateSpec(
            schema_version=document.schema_version,
            experiment_id=document.experiment_id,
            common=document.common,
            settings=document.validate,
        )
    raise ConfigError(
        "Expected the requested run mode to be 'train', 'linspace', or "
        f"'validate'. Provided value: {mode!r}."
    )
