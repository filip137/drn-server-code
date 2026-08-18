"""Pure, strict schema for ``mnist_relu_drn_kd.v1``."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from experiments.mnist_relu.config import (
    DataSettings,
    RuntimeSettings,
    _integer,
    _keys,
    _number,
    _object,
    _optional_batches,
    _parse_data,
    _parse_runtime,
)
from experiments.schema import RunMode, config_error, freeze_json


EXPERIMENT_ID = "mnist_relu_drn_kd.v1"
SCHEMA_VERSION = 1
_MEASURED_KEYS = {
    "curve_preprocessing",
    "split_seed",
    "assignment_seed",
    "formed_resistance_max_ohm",
    "cohort_fraction",
    "cohort",
    "source_traces_per_cell",
    "initial_pulse_index",
    "projection",
    "expected_trace_length",
    "initial_target_mapping",
    "programming_deadband_mode",
    "programming_deadband_relative",
    "probabilistic_write_mode",
    "probabilistic_write_probability",
    "probabilistic_write_scale_relative",
    "probabilistic_write_seed",
}
_MEASURED_OPTIONAL_KEYS = {
    "dual_rail_layout_by_parameter",
}


@dataclass(frozen=True)
class StudentModelSettings:
    dims: tuple[int, int, int]
    input_gain: float
    conductance_min: float
    conductance_max: float
    voltage_amp: float
    current_amp: float
    encoding: str
    include_biases: bool
    non_linearity: str
    quadratic_diode_param: Mapping[str, Any]
    exponential_diode_param: Mapping[str, Any]
    hard_sigmoid_param: Mapping[str, Any]
    amplification_indexing: str = "logical"


@dataclass(frozen=True)
class SolverSettings:
    inference_iterations: int
    training_iterations: int
    mode: str
    overrelaxation_factor: float


@dataclass(frozen=True)
class MappingSettings:
    scale_fractions: tuple[float, ...]
    scale_fraction_pairs: tuple[tuple[float, float], ...] | None
    range_placement: str
    calibration_examples: int
    calibration_batch_size: int
    logit_gain_min: float
    logit_gain_max: float
    logit_gain_steps: int


@dataclass(frozen=True)
class UpdateBackendSettings:
    type: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class StudentTrainSettings:
    num_epochs: int
    learning_rates: tuple[float, float]
    temperature: float
    log_every: int
    max_batches: int | None
    max_validation_batches: int | None
    minimum_relative_kl_improvement: float
    update_backend: UpdateBackendSettings


@dataclass(frozen=True)
class StudentValidateSettings:
    split: str
    sample_limit: int | None


@dataclass(frozen=True)
class StudentConfig:
    schema_version: int
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: StudentModelSettings
    solver: SolverSettings
    mapping: MappingSettings
    modes: Mapping[str, Any]


@dataclass(frozen=True)
class StudentTrainSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: StudentModelSettings
    solver: SolverSettings
    mapping: MappingSettings
    settings: StudentTrainSettings


@dataclass(frozen=True)
class StudentValidateSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    model: StudentModelSettings
    solver: SolverSettings
    mapping: MappingSettings
    settings: StudentValidateSettings


def _empty_object(value: Any, path: str) -> Mapping[str, Any]:
    raw = _object(value, path)
    if raw:
        raise config_error(path, "to be an explicit empty JSON object", dict(raw))
    return MappingProxyType({})


def _parse_model(value: Any) -> StudentModelSettings:
    path = "config.model"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "dims",
            "input_gain",
            "conductance_min",
            "conductance_max",
            "voltage_amp",
            "current_amp",
            "encoding",
            "include_biases",
            "non_linearity",
        },
    )
    if not isinstance(raw["dims"], (list, tuple)) or tuple(raw["dims"]) != (1568, 100, 20):
        raise config_error(f"{path}.dims", "to equal [1568, 100, 20]", raw["dims"])
    if raw["encoding"] not in {"single", "differential"}:
        raise config_error(f"{path}.encoding", "to be 'single' or 'differential'", raw["encoding"])
    if raw["include_biases"] is not False:
        raise config_error(f"{path}.include_biases", "to be false", raw["include_biases"])
    if _number(raw["input_gain"], f"{path}.input_gain") != 100.0:
        raise config_error(f"{path}.input_gain", "to equal 100.0", raw["input_gain"])
    voltage_amp = _number(raw["voltage_amp"], f"{path}.voltage_amp")
    if voltage_amp <= 0.0:
        raise config_error(
            f"{path}.voltage_amp",
            "to be a finite number > 0",
            raw["voltage_amp"],
        )
    current_amp = _number(raw["current_amp"], f"{path}.current_amp")
    if current_amp <= 0.0:
        raise config_error(
            f"{path}.current_amp",
            "to be a finite number > 0",
            raw["current_amp"],
        )
    if raw["encoding"] == "single" and voltage_amp != 4.0:
        raise config_error(
            f"{path}.voltage_amp",
            "to equal 4.0 for legacy single-device encoding",
            raw["voltage_amp"],
        )
    if raw["encoding"] == "single" and current_amp != 0.25:
        raise config_error(
            f"{path}.current_amp",
            "to equal 0.25 for legacy single-device encoding",
            raw["current_amp"],
        )
    lower = _number(raw["conductance_min"], f"{path}.conductance_min")
    upper = _number(raw["conductance_max"], f"{path}.conductance_max")
    if lower >= upper:
        raise config_error(path, "to have conductance_min < conductance_max", dict(raw))
    non_linearity = _object(raw["non_linearity"], f"{path}.non_linearity")
    _keys(
        non_linearity,
        f"{path}.non_linearity",
        {"type", "quadratic_diode_param", "exponential_diode_param", "hard_sigmoid_param"},
    )
    if non_linearity["type"] != "perfect_diode":
        raise config_error(f"{path}.non_linearity.type", "to equal 'perfect_diode'", non_linearity["type"])
    return StudentModelSettings(
        dims=(1568, 100, 20),
        input_gain=100.0,
        conductance_min=lower,
        conductance_max=upper,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        encoding=raw["encoding"],
        include_biases=False,
        non_linearity="perfect_diode",
        quadratic_diode_param=_empty_object(non_linearity["quadratic_diode_param"], f"{path}.non_linearity.quadratic_diode_param"),
        exponential_diode_param=_empty_object(non_linearity["exponential_diode_param"], f"{path}.non_linearity.exponential_diode_param"),
        hard_sigmoid_param=_empty_object(non_linearity["hard_sigmoid_param"], f"{path}.non_linearity.hard_sigmoid_param"),
    )


def _parse_solver(value: Any) -> SolverSettings:
    path = "config.solver"
    raw = _object(value, path)
    _keys(raw, path, {"inference_iterations", "training_iterations", "mode", "overrelaxation_factor"})
    if raw["mode"] != "asynchronous":
        raise config_error(f"{path}.mode", "to equal 'asynchronous'", raw["mode"])
    return SolverSettings(
        inference_iterations=_integer(raw["inference_iterations"], f"{path}.inference_iterations", minimum=1),
        training_iterations=_integer(raw["training_iterations"], f"{path}.training_iterations", minimum=1),
        mode="asynchronous",
        overrelaxation_factor=_number(raw["overrelaxation_factor"], f"{path}.overrelaxation_factor"),
    )


def _parse_mapping(value: Any) -> MappingSettings:
    path = "config.mapping"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "scale_fractions",
            "calibration_examples",
            "calibration_batch_size",
            "logit_gain_min",
            "logit_gain_max",
            "logit_gain_steps",
        },
        {"scale_fraction_pairs", "range_placement"},
    )
    fractions = raw["scale_fractions"]
    if not isinstance(fractions, (list, tuple)) or not fractions:
        raise config_error(f"{path}.scale_fractions", "to be a non-empty list of values in (0, 1]", fractions)
    parsed = tuple(_number(item, f"{path}.scale_fractions[{index}]") for index, item in enumerate(fractions))
    if any(item <= 0.0 or item > 1.0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise config_error(f"{path}.scale_fractions", "to contain unique values in (0, 1]", fractions)
    raw_pairs = raw.get("scale_fraction_pairs")
    parsed_pairs = None
    if raw_pairs is not None:
        if not isinstance(raw_pairs, (list, tuple)) or not raw_pairs:
            raise config_error(
                f"{path}.scale_fraction_pairs",
                "to be null or a non-empty list of unique two-value lists in (0, 1]",
                raw_pairs,
            )
        candidate_pairs = []
        for index, pair in enumerate(raw_pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise config_error(
                    f"{path}.scale_fraction_pairs[{index}]",
                    "to contain exactly two values in (0, 1]",
                    pair,
                )
            candidate = tuple(
                _number(item, f"{path}.scale_fraction_pairs[{index}][{offset}]")
                for offset, item in enumerate(pair)
            )
            if any(item <= 0.0 or item > 1.0 for item in candidate):
                raise config_error(
                    f"{path}.scale_fraction_pairs[{index}]",
                    "to contain exactly two values in (0, 1]",
                    pair,
                )
            candidate_pairs.append((candidate[0], candidate[1]))
        if len(set(candidate_pairs)) != len(candidate_pairs):
            raise config_error(
                f"{path}.scale_fraction_pairs",
                "to contain unique two-value lists",
                raw_pairs,
            )
        parsed_pairs = tuple(candidate_pairs)
    range_placement = raw.get("range_placement", "lower")
    if range_placement not in {"lower", "centered"}:
        raise config_error(
            f"{path}.range_placement",
            "to be 'lower' or 'centered'",
            range_placement,
        )
    gain_min = _number(raw["logit_gain_min"], f"{path}.logit_gain_min")
    gain_max = _number(raw["logit_gain_max"], f"{path}.logit_gain_max")
    if gain_min <= 0.0 or gain_min >= gain_max:
        raise config_error(path, "to have 0 < logit_gain_min < logit_gain_max", dict(raw))
    return MappingSettings(
        scale_fractions=parsed,
        scale_fraction_pairs=parsed_pairs,
        range_placement=range_placement,
        calibration_examples=_integer(raw["calibration_examples"], f"{path}.calibration_examples", minimum=1),
        calibration_batch_size=_integer(raw["calibration_batch_size"], f"{path}.calibration_batch_size", minimum=1),
        logit_gain_min=gain_min,
        logit_gain_max=gain_max,
        logit_gain_steps=_integer(raw["logit_gain_steps"], f"{path}.logit_gain_steps", minimum=3),
    )


def _parse_measured(
    value: Any,
    path: str,
    *,
    backend_type: str,
) -> Mapping[str, Any]:
    raw = _object(value, path)
    _keys(raw, path, _MEASURED_KEYS, _MEASURED_OPTIONAL_KEYS)
    if raw["curve_preprocessing"] not in {"raw", "isotonic_nonincreasing"}:
        raise config_error(f"{path}.curve_preprocessing", "to be 'raw' or 'isotonic_nonincreasing'", raw["curve_preprocessing"])
    if raw["initial_target_mapping"] not in {
        "literal",
        "per_device_affine",
        "paired_affine_common_window",
        "dual_rail_pairwise_common_window",
    }:
        raise config_error(
            f"{path}.initial_target_mapping",
            "to be 'literal', 'per_device_affine', "
            "'paired_affine_common_window', or "
            "'dual_rail_pairwise_common_window'",
            raw["initial_target_mapping"],
        )
    expected_cohort = {
        "measured_cohort_a": "A",
        "measured_cohort_b": "B",
    }[backend_type]
    if (
        expected_cohort == "B"
        and raw["initial_target_mapping"]
        in {"per_device_affine", "dual_rail_pairwise_common_window"}
    ):
        raise config_error(
            f"{path}.initial_target_mapping",
            "to be 'literal' or 'paired_affine_common_window' for cohort B",
            raw["initial_target_mapping"],
        )
    layouts = raw.get("dual_rail_layout_by_parameter")
    if raw["initial_target_mapping"] == "dual_rail_pairwise_common_window":
        if not isinstance(layouts, Mapping) or not layouts:
            raise config_error(
                f"{path}.dual_rail_layout_by_parameter",
                "to map stable single-conductance parameter keys to "
                "'halves' or 'paired' for dual-rail common-window mapping",
                layouts,
            )
        invalid = {
            key: layout
            for key, layout in layouts.items()
            if not isinstance(key, str)
            or layout not in {"halves", "paired"}
        }
        if invalid:
            raise config_error(
                f"{path}.dual_rail_layout_by_parameter",
                "to map stable parameter keys to 'halves' or 'paired'",
                layouts,
            )
    elif layouts is not None:
        raise config_error(
            f"{path}.dual_rail_layout_by_parameter",
            "to be omitted or null unless initial_target_mapping is "
            "'dual_rail_pairwise_common_window'",
            layouts,
        )
    exact = {
        "cohort": expected_cohort,
        "cohort_fraction": 0.5,
        "source_traces_per_cell": 2,
        "initial_pulse_index": 0,
        "projection": "global_nearest",
        "programming_deadband_mode": "none",
        "programming_deadband_relative": 0.0,
        "probabilistic_write_mode": "none",
        "probabilistic_write_probability": 1.0,
        "probabilistic_write_scale_relative": 0.0,
        "probabilistic_write_seed": 0,
    }
    for name, expected in exact.items():
        if raw[name] != expected:
            raise config_error(f"{path}.{name}", f"to equal {expected!r}", raw[name])
    for name in ("split_seed", "assignment_seed"):
        _integer(raw[name], f"{path}.{name}")
    _integer(raw["expected_trace_length"], f"{path}.expected_trace_length", minimum=2)
    if _number(raw["formed_resistance_max_ohm"], f"{path}.formed_resistance_max_ohm") <= 0.0:
        raise config_error(f"{path}.formed_resistance_max_ohm", "to be positive", raw["formed_resistance_max_ohm"])
    normalized = dict(raw)
    normalized.setdefault("dual_rail_layout_by_parameter", None)
    return freeze_json(normalized, path=path)


def _parse_train(value: Any) -> StudentTrainSettings:
    path = "config.modes.train"
    raw = _object(value, path)
    _keys(raw, path, {"num_epochs", "learning_rates", "temperature", "log_every", "max_batches", "max_validation_batches", "minimum_relative_kl_improvement", "update_backend"})
    rates = raw["learning_rates"]
    if not isinstance(rates, (list, tuple)) or len(rates) != 2:
        raise config_error(f"{path}.learning_rates", "to contain exactly two non-negative rates", rates)
    parsed_rates = tuple(_number(item, f"{path}.learning_rates[{index}]") for index, item in enumerate(rates))
    if _number(raw["temperature"], f"{path}.temperature") != 1.0:
        raise config_error(f"{path}.temperature", "to equal 1.0", raw["temperature"])
    relative = _number(raw["minimum_relative_kl_improvement"], f"{path}.minimum_relative_kl_improvement")
    if relative > 1.0:
        raise config_error(f"{path}.minimum_relative_kl_improvement", "to be in [0, 1]", relative)
    backend = _object(raw["update_backend"], f"{path}.update_backend")
    _keys(backend, f"{path}.update_backend", {"type", "parameters"})
    if backend["type"] not in {
        "ideal",
        "measured_cohort_a",
        "measured_cohort_b",
    }:
        raise config_error(
            f"{path}.update_backend.type",
            "to be 'ideal', 'measured_cohort_a', or 'measured_cohort_b'",
            backend["type"],
        )
    if backend["type"] == "ideal":
        parameters = _empty_object(backend["parameters"], f"{path}.update_backend.parameters")
    else:
        parameters = _parse_measured(
            backend["parameters"],
            f"{path}.update_backend.parameters",
            backend_type=backend["type"],
        )
    return StudentTrainSettings(
        num_epochs=_integer(raw["num_epochs"], f"{path}.num_epochs", minimum=1),
        learning_rates=(parsed_rates[0], parsed_rates[1]),
        temperature=1.0,
        log_every=_integer(raw["log_every"], f"{path}.log_every", minimum=1),
        max_batches=_optional_batches(raw["max_batches"], f"{path}.max_batches"),
        max_validation_batches=_optional_batches(raw["max_validation_batches"], f"{path}.max_validation_batches"),
        minimum_relative_kl_improvement=relative,
        update_backend=UpdateBackendSettings(type=backend["type"], parameters=parameters),
    )


def _parse_validate(value: Any) -> StudentValidateSettings:
    path = "config.modes.validate"
    raw = _object(value, path)
    _keys(raw, path, {"split", "sample_limit"})
    if raw["split"] not in {"validation", "test"}:
        raise config_error(f"{path}.split", "to be 'validation' or 'test'", raw["split"])
    return StudentValidateSettings(split=raw["split"], sample_limit=_optional_batches(raw["sample_limit"], f"{path}.sample_limit"))


def parse_student_config(payload: Mapping[str, Any]) -> StudentConfig:
    raw = _object(payload, "config")
    _keys(raw, "config", {"schema_version", "experiment_id", "runtime", "data", "model", "solver", "mapping", "modes"})
    if raw["schema_version"] != 1:
        raise config_error("config.schema_version", "to equal 1", raw["schema_version"])
    if raw["experiment_id"] != EXPERIMENT_ID:
        raise config_error("config.experiment_id", f"to equal {EXPERIMENT_ID!r}", raw["experiment_id"])
    modes_raw = _object(raw["modes"], "config.modes")
    if not modes_raw or set(modes_raw) - {"train", "validate"}:
        raise config_error("config.modes", "to contain one or both of 'train' and 'validate'", dict(modes_raw))
    modes: dict[str, Any] = {}
    if "train" in modes_raw:
        modes["train"] = _parse_train(modes_raw["train"])
    if "validate" in modes_raw:
        modes["validate"] = _parse_validate(modes_raw["validate"])
    model = _parse_model(raw["model"])
    train = modes.get("train")
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type
        in {"measured_cohort_a", "measured_cohort_b"}
        and train.update_backend.parameters["initial_target_mapping"]
        == "paired_affine_common_window"
        and model.encoding != "differential"
    ):
        raise config_error(
            "config.modes.train.update_backend.parameters.initial_target_mapping",
            "to select differential model encoding when using "
            "'paired_affine_common_window'",
            train.update_backend.parameters["initial_target_mapping"],
        )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type
        in {"measured_cohort_a", "measured_cohort_b"}
        and train.update_backend.parameters["initial_target_mapping"]
        == "dual_rail_pairwise_common_window"
        and model.encoding != "single"
    ):
        raise config_error(
            "config.modes.train.update_backend.parameters.initial_target_mapping",
            "to select single model encoding when using "
            "'dual_rail_pairwise_common_window'",
            train.update_backend.parameters["initial_target_mapping"],
        )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type == "measured_cohort_a"
        and train.update_backend.parameters["initial_target_mapping"]
        == "dual_rail_pairwise_common_window"
    ):
        layouts = train.update_backend.parameters[
            "dual_rail_layout_by_parameter"
        ]
        expected_layouts = {
            "base.dense_weight.0": "halves",
            "base.dense_weight.1": "paired",
        }
        if dict(layouts) != expected_layouts:
            raise config_error(
                "config.modes.train.update_backend.parameters."
                "dual_rail_layout_by_parameter",
                "to equal the model-local dual-rail layouts "
                f"{expected_layouts!r}",
                dict(layouts),
            )
    mapping = _parse_mapping(raw["mapping"])
    if mapping.range_placement == "centered" and model.encoding != "single":
        raise config_error(
            "config.mapping.range_placement",
            "to select single model encoding when using 'centered'",
            mapping.range_placement,
        )
    return StudentConfig(
        schema_version=1,
        experiment_id=EXPERIMENT_ID,
        runtime=_parse_runtime(raw["runtime"]),
        data=_parse_data(raw["data"]),
        model=model,
        solver=_parse_solver(raw["solver"]),
        mapping=mapping,
        modes=MappingProxyType(modes),
    )


def resolve_student_spec(document: StudentConfig, mode: RunMode) -> Any:
    key = mode.value
    if key not in document.modes:
        raise config_error("config.modes", f"to define requested mode {key!r}", tuple(document.modes))
    common = {
        "experiment_id": document.experiment_id,
        "runtime": document.runtime,
        "data": document.data,
        "model": document.model,
        "solver": document.solver,
        "mapping": document.mapping,
        "settings": document.modes[key],
    }
    if mode is RunMode.TRAIN:
        return StudentTrainSpec(**common)
    if mode is RunMode.VALIDATE:
        return StudentValidateSpec(**common)
    raise config_error("the requested run mode", "to be 'train' or 'validate'", key)


__all__ = [
    "EXPERIMENT_ID",
    "SCHEMA_VERSION",
    "StudentConfig",
    "StudentTrainSpec",
    "StudentValidateSpec",
    "parse_student_config",
    "resolve_student_spec",
]
