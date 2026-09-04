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
from model.resistive.device_config import (
    AIHWKIT_RERAM_CMO,
    IBM_AFM2025_PCM,
    WAN2022_PHYSICAL,
    device_programming_to_mapping,
    parse_device_programming_config,
)


EXPERIMENT_ID = "mnist_relu_drn_kd.v1"
SCHEMA_VERSION = 1
MEASURED_COHORT_A_BACKENDS = frozenset(
    {
        "measured_cohort_a",
        "measured_cohort_a_sign_sgd",
        "measured_cohort_a_one_pulse_down",
    }
)
MEASURED_COHORT_B_BACKENDS = frozenset({"measured_cohort_b"})
MEASURED_BACKENDS = MEASURED_COHORT_A_BACKENDS | MEASURED_COHORT_B_BACKENDS
_IBM_ARRAY_SPECIFIC_TARGET_MAPPINGS = frozenset(
    {
        "dual_rail_quad_common_window",
        "differential_pair_common_window",
        "cell_aware_exact_bounds_quad",
    }
)
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
    "positive_gradient_threshold_by_parameter",
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
class TeacherSettings:
    type: str
    initialization: str
    conductance_bounds: tuple[float, float] | None


@dataclass(frozen=True)
class StudentTrainSettings:
    num_epochs: int
    learning_rates: tuple[float, float]
    temperature: float
    log_every: int
    max_batches: int | None
    max_validation_batches: int | None
    minimum_relative_kl_improvement: float
    selection_evaluation: str
    selection_metric: str
    selection_noise_repeats: int
    weight_modifier: UpdateBackendSettings
    selection_weight_modifier: UpdateBackendSettings
    update_backend: UpdateBackendSettings


@dataclass(frozen=True)
class StudentValidateSettings:
    split: str
    sample_limit: int | None
    weight_modifier: UpdateBackendSettings
    noise_repeats: int


@dataclass(frozen=True)
class StudentConfig:
    schema_version: int
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    teacher: TeacherSettings
    model: StudentModelSettings
    solver: SolverSettings
    mapping: MappingSettings
    modes: Mapping[str, Any]


@dataclass(frozen=True)
class StudentTrainSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    teacher: TeacherSettings
    model: StudentModelSettings
    solver: SolverSettings
    mapping: MappingSettings
    settings: StudentTrainSettings


@dataclass(frozen=True)
class StudentValidateSpec:
    experiment_id: str
    runtime: RuntimeSettings
    data: DataSettings
    teacher: TeacherSettings
    model: StudentModelSettings
    solver: SolverSettings
    mapping: MappingSettings
    settings: StudentValidateSettings


def _empty_object(value: Any, path: str) -> Mapping[str, Any]:
    raw = _object(value, path)
    if raw:
        raise config_error(path, "to be an explicit empty JSON object", dict(raw))
    return MappingProxyType({})


def _parse_teacher(value: Any) -> TeacherSettings:
    path = "config.teacher"
    raw = _object(value, path)
    if "type" not in raw:
        raise config_error(path, "to contain 'type'", dict(raw))
    teacher_type = raw["type"]
    if teacher_type not in {"bias_free_relu", "bounded_drn"}:
        raise config_error(
            f"{path}.type",
            "to be 'bias_free_relu' or 'bounded_drn'",
            teacher_type,
        )
    expected_keys = {"type", "initialization"}
    if teacher_type == "bounded_drn":
        expected_keys.add("conductance_bounds")
    _keys(raw, path, expected_keys)
    expected_initialization = {
        "bias_free_relu": "signed_weight_mapping",
        "bounded_drn": "literal_named_weight_copy",
    }[teacher_type]
    if raw["initialization"] != expected_initialization:
        raise config_error(
            f"{path}.initialization",
            f"to equal {expected_initialization!r} for teacher type "
            f"{teacher_type!r}",
            raw["initialization"],
        )
    conductance_bounds = None
    if teacher_type == "bounded_drn":
        raw_bounds = raw["conductance_bounds"]
        if not isinstance(raw_bounds, (list, tuple)) or len(raw_bounds) != 2:
            raise config_error(
                f"{path}.conductance_bounds",
                "to be [minimum, maximum]",
                raw_bounds,
            )
        lower = _number(
            raw_bounds[0],
            f"{path}.conductance_bounds[0]",
            minimum=0.0,
        )
        upper = _number(
            raw_bounds[1],
            f"{path}.conductance_bounds[1]",
            minimum=0.0,
        )
        if lower >= upper:
            raise config_error(
                f"{path}.conductance_bounds",
                "to satisfy 0 <= minimum < maximum",
                raw_bounds,
            )
        conductance_bounds = (lower, upper)
    return TeacherSettings(
        type=teacher_type,
        initialization=expected_initialization,
        conductance_bounds=conductance_bounds,
    )


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
    allowed_dims = {(1568, 100, 20), (1568, 512, 20)}
    if not isinstance(raw["dims"], (list, tuple)) or tuple(raw["dims"]) not in allowed_dims:
        raise config_error(
            f"{path}.dims",
            "to equal [1568, 100, 20] or [1568, 512, 20]",
            raw["dims"],
        )
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
        dims=tuple(raw["dims"]),
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
        "dual_rail_quad_common_window",
    }:
        raise config_error(
            f"{path}.initial_target_mapping",
            "to be 'literal', 'per_device_affine', "
            "'paired_affine_common_window', or "
            "a dual-rail common-window mapping "
            "('dual_rail_pairwise_common_window' or "
            "'dual_rail_quad_common_window')",
            raw["initial_target_mapping"],
        )
    expected_cohort = {
        "measured_cohort_a": "A",
        "measured_cohort_a_sign_sgd": "A",
        "measured_cohort_a_one_pulse_down": "A",
        "measured_cohort_b": "B",
    }[backend_type]
    if (
        expected_cohort == "B"
        and raw["initial_target_mapping"]
        in {
            "per_device_affine",
            "dual_rail_pairwise_common_window",
        }
    ):
        raise config_error(
            f"{path}.initial_target_mapping",
            "to be 'literal' or a matched common-window mapping "
            "('paired_affine_common_window' or "
            "'dual_rail_quad_common_window') for cohort B",
            raw["initial_target_mapping"],
        )
    layouts = raw.get("dual_rail_layout_by_parameter")
    if raw["initial_target_mapping"] in {
        "dual_rail_pairwise_common_window",
        "dual_rail_quad_common_window",
    }:
        if not isinstance(layouts, Mapping) or not layouts:
            raise config_error(
                f"{path}.dual_rail_layout_by_parameter",
                "to map stable single-conductance parameter keys to "
                "'halves' or 'paired' for a dual-rail common-window mapping",
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
            "a dual-rail common-window mapping",
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
    raw_thresholds = raw.get("positive_gradient_threshold_by_parameter")
    if backend_type != "measured_cohort_a_one_pulse_down":
        if raw_thresholds is not None:
            raise config_error(
                f"{path}.positive_gradient_threshold_by_parameter",
                "to be omitted or null unless update_backend.type is "
                "'measured_cohort_a_one_pulse_down'",
                raw_thresholds,
            )
        normalized["positive_gradient_threshold_by_parameter"] = None
    elif raw_thresholds is None:
        # Omission is the exact zero-threshold historical control.
        normalized["positive_gradient_threshold_by_parameter"] = None
    else:
        mapping = raw["initial_target_mapping"]
        expected_threshold_keys = (
            {
                "base.conductance_plus.0",
                "base.conductance_minus.0",
                "base.conductance_plus.1",
                "base.conductance_minus.1",
            }
            if mapping == "paired_affine_common_window"
            else {
                "base.dense_weight.0",
                "base.dense_weight.1",
            }
        )
        if not isinstance(raw_thresholds, Mapping) or (
            set(raw_thresholds) != expected_threshold_keys
        ):
            raise config_error(
                f"{path}.positive_gradient_threshold_by_parameter",
                "to map exactly the stable parameter keys "
                f"{sorted(expected_threshold_keys)!r} to finite "
                "non-negative raw-gradient thresholds",
                raw_thresholds,
            )
        parsed_thresholds = {
            key: _number(
                raw_thresholds[key],
                f"{path}.positive_gradient_threshold_by_parameter.{key}",
            )
            for key in sorted(expected_threshold_keys)
        }
        if any(value < 0.0 for value in parsed_thresholds.values()):
            raise config_error(
                f"{path}.positive_gradient_threshold_by_parameter",
                "to map exactly the stable parameter keys "
                f"{sorted(expected_threshold_keys)!r} to finite "
                "non-negative raw-gradient thresholds",
                raw_thresholds,
            )
        normalized["positive_gradient_threshold_by_parameter"] = (
            parsed_thresholds
        )
    if backend_type == "measured_cohort_a_one_pulse_down":
        if normalized["curve_preprocessing"] != "isotonic_nonincreasing":
            raise config_error(
                f"{path}.curve_preprocessing",
                "to equal 'isotonic_nonincreasing' for "
                "measured_cohort_a_one_pulse_down",
                normalized["curve_preprocessing"],
            )
        allowed_mappings = {
            "paired_affine_common_window",
            "dual_rail_quad_common_window",
        }
        if normalized["initial_target_mapping"] not in allowed_mappings:
            raise config_error(
                f"{path}.initial_target_mapping",
                "to be 'paired_affine_common_window' or "
                "'dual_rail_quad_common_window' for "
                "measured_cohort_a_one_pulse_down",
                normalized["initial_target_mapping"],
            )
    return freeze_json(normalized, path=path)


def _parse_weight_modifier(value: Any, path: str) -> UpdateBackendSettings:
    raw = _object(value, path)
    _keys(raw, path, {"type", "parameters"})
    modifier_type = raw["type"]
    parameters_path = f"{path}.parameters"
    parameters = _object(raw["parameters"], parameters_path)
    if modifier_type == "none":
        return UpdateBackendSettings(
            type="none",
            parameters=_empty_object(parameters, parameters_path),
        )
    if modifier_type == "ibm_reram_om_program_verify":
        required = {
            "execution",
            "assignment_seed",
            "endpoint_seed",
            "corruption_policy",
            "noisy_evaluation",
            "endpoint_policy",
            "target_out_of_support",
            "preset",
            "controller",
            "start_protocol",
            "tolerance_step_ratio",
            "maximum_program_pulses",
        }
        mapping_fields = {
            "target_mapping",
            "dual_rail_layout_by_parameter",
            "common_window_margin_fraction",
        }
        cell_aware_fields = {
            "cell_aware_mode",
            "cell_aware_signed_levels",
        }
        _keys(
            parameters,
            parameters_path,
            required,
            mapping_fields | cell_aware_fields | {"forward_logit_gain"},
        )
        provided_mapping_fields = mapping_fields & set(parameters)
        if provided_mapping_fields and provided_mapping_fields != mapping_fields:
            raise config_error(
                parameters_path,
                "to provide all three target-mapping fields together: "
                "target_mapping, dual_rail_layout_by_parameter, and "
                "common_window_margin_fraction",
                dict(parameters),
            )
        normalized_parameters = dict(parameters)
        if not provided_mapping_fields:
            # Legacy pilot configs retain their literal global-coordinate path.
            normalized_parameters.update(
                {
                    "target_mapping": "literal_global",
                    "dual_rail_layout_by_parameter": None,
                    "common_window_margin_fraction": 0.0,
                }
            )
        target_mapping = normalized_parameters["target_mapping"]
        if target_mapping not in {
            "literal_global",
            "dual_rail_quad_common_window",
            "differential_pair_common_window",
            "cell_aware_exact_bounds_quad",
        }:
            raise config_error(
                f"{parameters_path}.target_mapping",
                "to be 'literal_global' or "
                "an array-specific common-window mapping "
                "('dual_rail_quad_common_window', "
                "'differential_pair_common_window', or "
                "'cell_aware_exact_bounds_quad')",
                target_mapping,
            )
        margin = _number(
            normalized_parameters["common_window_margin_fraction"],
            f"{parameters_path}.common_window_margin_fraction",
        )
        if margin < 0.0 or margin >= 0.5:
            raise config_error(
                f"{parameters_path}.common_window_margin_fraction",
                "to be a finite number in [0, 0.5)",
                margin,
            )
        normalized_parameters["common_window_margin_fraction"] = margin
        raw_layouts = normalized_parameters["dual_rail_layout_by_parameter"]
        if target_mapping in {
            "dual_rail_quad_common_window",
            "cell_aware_exact_bounds_quad",
        }:
            expected_layouts = {
                "base.dense_weight.0": "halves",
                "base.dense_weight.1": "paired",
            }
            if not isinstance(raw_layouts, Mapping) or dict(raw_layouts) != expected_layouts:
                raise config_error(
                    f"{parameters_path}.dual_rail_layout_by_parameter",
                    "to equal the canonical W1/W2 layout "
                    f"{expected_layouts!r}",
                    raw_layouts,
                )
            normalized_parameters["dual_rail_layout_by_parameter"] = dict(
                raw_layouts
            )
        if target_mapping == "cell_aware_exact_bounds_quad":
            if margin != 0.0:
                raise config_error(
                    f"{parameters_path}.common_window_margin_fraction",
                    "to equal 0.0 for cell_aware_exact_bounds_quad",
                    margin,
                )
            if not (
                cell_aware_fields | {"forward_logit_gain"}
            ) <= set(parameters):
                raise config_error(
                    parameters_path,
                    "to provide cell_aware_mode and "
                    "cell_aware_signed_levels plus forward_logit_gain for "
                    "cell_aware_exact_bounds_quad",
                    dict(parameters),
                )
            mode = normalized_parameters["cell_aware_mode"]
            if mode not in {"continuous", "quantized_9_level"}:
                raise config_error(
                    f"{parameters_path}.cell_aware_mode",
                    "to be 'continuous' or 'quantized_9_level'",
                    mode,
                )
            levels = _integer(
                normalized_parameters["cell_aware_signed_levels"],
                f"{parameters_path}.cell_aware_signed_levels",
                minimum=1,
            )
            if levels != 9:
                raise config_error(
                    f"{parameters_path}.cell_aware_signed_levels",
                    "to equal 9",
                    levels,
                )
            normalized_parameters["cell_aware_signed_levels"] = levels
        elif cell_aware_fields & set(parameters):
            raise config_error(
                parameters_path,
                "to omit cell-aware fields unless target_mapping is "
                "cell_aware_exact_bounds_quad",
                dict(parameters),
            )
        elif target_mapping == "differential_pair_common_window":
            if raw_layouts is not None:
                raise config_error(
                    f"{parameters_path}.dual_rail_layout_by_parameter",
                    "to be null for differential_pair_common_window; the "
                    "core mapper enforces the canonical adjacent plus/minus "
                    "catalog",
                    raw_layouts,
                )
        elif target_mapping == "literal_global" and (
            raw_layouts is not None or margin != 0.0
        ):
            raise config_error(
                parameters_path,
                "to use null dual_rail_layout_by_parameter and zero "
                "common_window_margin_fraction for literal_global",
                dict(normalized_parameters),
            )
        if "forward_logit_gain" in normalized_parameters:
            raw_gain = normalized_parameters["forward_logit_gain"]
            if raw_gain is not None:
                gain = _number(
                    raw_gain,
                    f"{parameters_path}.forward_logit_gain",
                )
                if gain <= 0.0:
                    raise config_error(
                        f"{parameters_path}.forward_logit_gain",
                        "to be null or positive",
                        gain,
                    )
                normalized_parameters["forward_logit_gain"] = gain
        if (
            target_mapping == "cell_aware_exact_bounds_quad"
            and normalized_parameters.get("forward_logit_gain") is None
        ):
            raise config_error(
                f"{parameters_path}.forward_logit_gain",
                "to be a positive frozen development-set gain for the "
                "cell-aware device forward",
                normalized_parameters.get("forward_logit_gain"),
            )
        exact = {
            "preset": "reram_array_om",
            "controller": "adaptive",
            "start_protocol": "lower_to_target",
            "tolerance_step_ratio": 0.5,
            "maximum_program_pulses": 128,
            "endpoint_policy": "clip_0_1",
            "target_out_of_support": "error",
        }
        for name, expected in exact.items():
            if normalized_parameters[name] != expected:
                raise config_error(
                    f"{parameters_path}.{name}",
                    f"to equal {expected!r}",
                    normalized_parameters[name],
                )
        if normalized_parameters["execution"] not in {
            "compact_endpoint",
            "pulse_resolved",
        }:
            raise config_error(
                f"{parameters_path}.execution",
                "to be 'compact_endpoint' or 'pulse_resolved'",
                normalized_parameters["execution"],
            )
        if normalized_parameters["corruption_policy"] not in {
            "counterfactual_repaired",
            "published",
        }:
            raise config_error(
                f"{parameters_path}.corruption_policy",
                "to be 'counterfactual_repaired' or 'published'",
                normalized_parameters["corruption_policy"],
            )
        for name in ("assignment_seed", "endpoint_seed"):
            seed = _integer(
                normalized_parameters[name], f"{parameters_path}.{name}"
            )
            if seed >= 2**63:
                raise config_error(
                    f"{parameters_path}.{name}",
                    "to be an integer in [0, 2**63)",
                    seed,
                )
        if not isinstance(normalized_parameters["noisy_evaluation"], bool):
            raise config_error(
                f"{parameters_path}.noisy_evaluation",
                "to be a boolean",
                normalized_parameters["noisy_evaluation"],
            )
        return UpdateBackendSettings(
            type=modifier_type,
            parameters=freeze_json(
                normalized_parameters,
                path=parameters_path,
            ),
        )
    if modifier_type != "add_normal":
        raise config_error(
            f"{path}.type",
            "to be 'none', 'add_normal', or "
            "'ibm_reram_om_program_verify'",
            modifier_type,
        )
    _keys(
        parameters,
        parameters_path,
        {"std_dev", "seed", "noisy_evaluation", "scale_mode"},
    )
    std_dev = _number(parameters["std_dev"], f"{parameters_path}.std_dev")
    seed = _integer(parameters["seed"], f"{parameters_path}.seed")
    if seed > 2**64 - 1:
        raise config_error(
            f"{parameters_path}.seed",
            "to be an integer in [0, 2**64 - 1]",
            seed,
        )
    if not isinstance(parameters["noisy_evaluation"], bool):
        raise config_error(
            f"{parameters_path}.noisy_evaluation",
            "to be a boolean",
            parameters["noisy_evaluation"],
        )
    scale_mode = parameters["scale_mode"]
    if scale_mode not in {"tensor_abs_max", "output_channel_abs_max"}:
        raise config_error(
            f"{parameters_path}.scale_mode",
            "to be 'tensor_abs_max' or 'output_channel_abs_max'",
            scale_mode,
        )
    return UpdateBackendSettings(
        type="add_normal",
        parameters=freeze_json(
            {
                "std_dev": std_dev,
                "seed": seed,
                "noisy_evaluation": parameters["noisy_evaluation"],
                "scale_mode": scale_mode,
            },
            path=parameters_path,
        ),
    )


def _parse_program_verify(value: Any, path: str) -> Mapping[str, Any]:
    raw = _object(value, path)
    _keys(raw, path, {"device"})
    try:
        device = parse_device_programming_config(
            raw["device"],
            path=f"{path}.device",
        )
    except ValueError as error:
        raise config_error(
            f"{path}.device",
            "to contain a valid repeated-write endpoint device model",
            raw["device"],
        ) from error
    if device.type not in {
        IBM_AFM2025_PCM,
        WAN2022_PHYSICAL,
        AIHWKIT_RERAM_CMO,
    }:
        raise config_error(
            f"{path}.device.type",
            "to select a device model with repeated-write support",
            device.type,
        )
    return freeze_json(
        {"device": device_programming_to_mapping(device)},
        path=path,
    )


def _parse_train(value: Any) -> StudentTrainSettings:
    path = "config.modes.train"
    raw = _object(value, path)
    _keys(
        raw,
        path,
        {
            "num_epochs",
            "learning_rates",
            "temperature",
            "log_every",
            "max_batches",
            "max_validation_batches",
            "minimum_relative_kl_improvement",
            "update_backend",
        },
        {
            "weight_modifier",
            "selection_weight_modifier",
            "selection_evaluation",
            "selection_metric",
            "selection_noise_repeats",
        },
    )
    rates = raw["learning_rates"]
    if not isinstance(rates, (list, tuple)) or len(rates) != 2:
        raise config_error(f"{path}.learning_rates", "to contain exactly two non-negative rates", rates)
    parsed_rates = tuple(_number(item, f"{path}.learning_rates[{index}]") for index, item in enumerate(rates))
    if any(rate < 0.0 for rate in parsed_rates):
        raise config_error(
            f"{path}.learning_rates",
            "to contain exactly two non-negative rates",
            rates,
        )
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
        "measured_cohort_a_sign_sgd",
        "measured_cohort_a_one_pulse_down",
        "measured_cohort_b",
        "program_verify",
    }:
        raise config_error(
            f"{path}.update_backend.type",
            "to be 'ideal', 'measured_cohort_a', "
            "'measured_cohort_a_sign_sgd', "
            "'measured_cohort_a_one_pulse_down', 'measured_cohort_b', "
            "or 'program_verify'",
            backend["type"],
        )
    if backend["type"] == "ideal":
        parameters = _empty_object(backend["parameters"], f"{path}.update_backend.parameters")
    elif backend["type"] == "program_verify":
        parameters = _parse_program_verify(
            backend["parameters"],
            f"{path}.update_backend.parameters",
        )
    else:
        parameters = _parse_measured(
            backend["parameters"],
            f"{path}.update_backend.parameters",
            backend_type=backend["type"],
        )
    if (
        backend["type"] == "measured_cohort_a_one_pulse_down"
        and parsed_rates != (0.0, 0.0)
    ):
        raise config_error(
            f"{path}.learning_rates",
            "to equal [0.0, 0.0] because "
            "measured_cohort_a_one_pulse_down ignores learning-rate magnitude",
            rates,
        )
    weight_modifier = _parse_weight_modifier(
        raw.get(
            "weight_modifier",
            {"type": "none", "parameters": {}},
        ),
        f"{path}.weight_modifier",
    )
    selection_weight_modifier = _parse_weight_modifier(
        raw.get(
            "selection_weight_modifier",
            {"type": "none", "parameters": {}},
        ),
        f"{path}.selection_weight_modifier",
    )
    selection_evaluation = raw.get("selection_evaluation", "clean")
    if selection_evaluation not in {"clean", "modifier"}:
        raise config_error(
            f"{path}.selection_evaluation",
            "to be 'clean' or 'modifier'",
            selection_evaluation,
        )
    selection_metric = raw.get("selection_metric", "kl_teacher_student")
    if selection_metric not in {"kl_teacher_student", "student_accuracy"}:
        raise config_error(
            f"{path}.selection_metric",
            "to be 'kl_teacher_student' or 'student_accuracy'",
            selection_metric,
        )
    selection_noise_repeats = _integer(
        raw.get("selection_noise_repeats", 1),
        f"{path}.selection_noise_repeats",
        minimum=1,
    )
    if selection_evaluation == "modifier":
        selected_modifier = (
            selection_weight_modifier
            if selection_weight_modifier.type != "none"
            else weight_modifier
        )
        valid_modifier = bool(
            selected_modifier.parameters.get("noisy_evaluation", False)
        )
        if not valid_modifier:
            raise config_error(
                f"{path}.selection_evaluation",
                "to select a weight modifier with "
                "noisy_evaluation=true when set to 'modifier'",
                selection_evaluation,
            )
    elif selection_noise_repeats != 1:
        raise config_error(
            f"{path}.selection_noise_repeats",
            "to equal 1 when selection_evaluation is 'clean'",
            selection_noise_repeats,
        )
    if (
        selection_weight_modifier.type != "none"
        and selection_evaluation != "modifier"
    ):
        raise config_error(
            f"{path}.selection_weight_modifier",
            "to be 'none' unless selection_evaluation is 'modifier'",
            selection_weight_modifier.type,
        )
    if weight_modifier.type == "ibm_reram_om_program_verify" and (
        weight_modifier.parameters["execution"] != "compact_endpoint"
        or weight_modifier.parameters["noisy_evaluation"]
    ):
        raise config_error(
            f"{path}.weight_modifier",
            "to use compact_endpoint with noisy_evaluation=false for "
            "off-chip HWA minibatches",
            dict(weight_modifier.parameters),
        )
    if selection_weight_modifier.type == "ibm_reram_om_program_verify":
        selection_parameters = selection_weight_modifier.parameters
        selection_is_array_specific = (
            selection_parameters["target_mapping"]
            in _IBM_ARRAY_SPECIFIC_TARGET_MAPPINGS
        )
        expected_selection_corruption = (
            "counterfactual_repaired"
            if selection_is_array_specific
            else "published"
        )
        if (
            selection_parameters["execution"] != "pulse_resolved"
            or selection_parameters["corruption_policy"]
            != expected_selection_corruption
            or not selection_parameters["noisy_evaluation"]
        ):
            raise config_error(
                f"{path}.selection_weight_modifier",
                "to use pulse_resolved, noisy_evaluation=true, and "
                f"{expected_selection_corruption} corruption for its "
                "declared target mapping",
                dict(selection_parameters),
            )
    if (
        weight_modifier.type == "ibm_reram_om_program_verify"
        and weight_modifier.parameters["target_mapping"]
        in _IBM_ARRAY_SPECIFIC_TARGET_MAPPINGS
    ):
        if selection_weight_modifier.type != "ibm_reram_om_program_verify":
            raise config_error(
                f"{path}.selection_weight_modifier",
                "to be an IBM OM modifier matched to the array-specific "
                "training modifier",
                selection_weight_modifier.type,
            )
        matched_fields = (
            "assignment_seed",
            "corruption_policy",
            "target_mapping",
            "dual_rail_layout_by_parameter",
            "common_window_margin_fraction",
        )
        if (
            weight_modifier.parameters["target_mapping"]
            == "cell_aware_exact_bounds_quad"
        ):
            matched_fields += (
                "cell_aware_mode",
                "cell_aware_signed_levels",
                "forward_logit_gain",
            )
        mismatches = {
            name: {
                "training": weight_modifier.parameters[name],
                "selection": selection_weight_modifier.parameters[name],
            }
            for name in matched_fields
            if weight_modifier.parameters[name]
            != selection_weight_modifier.parameters[name]
        }
        if mismatches:
            raise config_error(
                f"{path}.selection_weight_modifier",
                "to match the array-specific training modifier's fixed "
                "array, corruption policy, mapping, layout, and "
                "inner-window margin",
                mismatches,
            )
    return StudentTrainSettings(
        num_epochs=_integer(raw["num_epochs"], f"{path}.num_epochs", minimum=0),
        learning_rates=(parsed_rates[0], parsed_rates[1]),
        temperature=1.0,
        log_every=_integer(raw["log_every"], f"{path}.log_every", minimum=1),
        max_batches=_optional_batches(raw["max_batches"], f"{path}.max_batches"),
        max_validation_batches=_optional_batches(raw["max_validation_batches"], f"{path}.max_validation_batches"),
        minimum_relative_kl_improvement=relative,
        selection_evaluation=selection_evaluation,
        selection_metric=selection_metric,
        selection_noise_repeats=selection_noise_repeats,
        weight_modifier=weight_modifier,
        selection_weight_modifier=selection_weight_modifier,
        update_backend=UpdateBackendSettings(type=backend["type"], parameters=parameters),
    )


def _parse_validate(value: Any) -> StudentValidateSettings:
    path = "config.modes.validate"
    raw = _object(value, path)
    _keys(raw, path, {"split", "sample_limit"}, {"weight_modifier", "noise_repeats"})
    if raw["split"] not in {"validation", "test"}:
        raise config_error(f"{path}.split", "to be 'validation' or 'test'", raw["split"])
    modifier = _parse_weight_modifier(
        raw.get("weight_modifier", {"type": "none", "parameters": {}}),
        f"{path}.weight_modifier",
    )
    repeats = _integer(
        raw.get("noise_repeats", 1),
        f"{path}.noise_repeats",
        minimum=1,
    )
    if modifier.type == "none" and repeats != 1:
        raise config_error(
            f"{path}.noise_repeats",
            "to equal 1 when weight_modifier is 'none'",
            repeats,
        )
    if modifier.type != "none" and not bool(
        modifier.parameters.get("noisy_evaluation", False)
    ):
        raise config_error(
            f"{path}.weight_modifier",
            "to set noisy_evaluation=true for validation",
            dict(modifier.parameters),
        )
    if modifier.type == "ibm_reram_om_program_verify":
        array_specific = (
            modifier.parameters["target_mapping"]
            in _IBM_ARRAY_SPECIFIC_TARGET_MAPPINGS
        )
        expected_corruption = (
            "counterfactual_repaired" if array_specific else "published"
        )
        if (
            modifier.parameters["execution"] != "pulse_resolved"
            or modifier.parameters["corruption_policy"]
            != expected_corruption
        ):
            raise config_error(
                f"{path}.weight_modifier",
                "to use pulse_resolved with "
                f"{expected_corruption} corruption for its declared target "
                "mapping during physical deployment validation",
                dict(modifier.parameters),
            )
    return StudentValidateSettings(
        split=raw["split"],
        sample_limit=_optional_batches(raw["sample_limit"], f"{path}.sample_limit"),
        weight_modifier=modifier,
        noise_repeats=repeats,
    )


def parse_student_config(payload: Mapping[str, Any]) -> StudentConfig:
    raw = _object(payload, "config")
    _keys(
        raw,
        "config",
        {
            "schema_version",
            "experiment_id",
            "runtime",
            "data",
            "model",
            "solver",
            "mapping",
            "modes",
        },
        {"teacher"},
    )
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
    teacher = _parse_teacher(
        raw.get(
            "teacher",
            {
                "type": "bias_free_relu",
                "initialization": "signed_weight_mapping",
            },
        )
    )
    model = _parse_model(raw["model"])
    ibm_modifiers: list[tuple[str, UpdateBackendSettings]] = []
    if isinstance(modes.get("train"), StudentTrainSettings):
        train_settings = modes["train"]
        ibm_modifiers.extend(
            (
                (
                    "config.modes.train.weight_modifier",
                    train_settings.weight_modifier,
                ),
                (
                    "config.modes.train.selection_weight_modifier",
                    train_settings.selection_weight_modifier,
                ),
            )
        )
    if isinstance(modes.get("validate"), StudentValidateSettings):
        ibm_modifiers.append(
            (
                "config.modes.validate.weight_modifier",
                modes["validate"].weight_modifier,
            )
        )
    for modifier_path, modifier in ibm_modifiers:
        if modifier.type != "ibm_reram_om_program_verify":
            continue
        target_mapping = modifier.parameters["target_mapping"]
        if (
            target_mapping
            in {
                "dual_rail_quad_common_window",
                "cell_aware_exact_bounds_quad",
            }
            and model.encoding != "single"
        ):
            raise config_error(
                f"{modifier_path}.parameters.target_mapping",
                f"to select model.encoding='single' for {target_mapping}",
                target_mapping,
            )
        if (
            target_mapping == "differential_pair_common_window"
            and model.encoding != "differential"
        ):
            raise config_error(
                f"{modifier_path}.parameters.target_mapping",
                "to select model.encoding='differential' for "
                "differential_pair_common_window",
                target_mapping,
            )
    if teacher.type == "bounded_drn":
        teacher_bounds = teacher.conductance_bounds
        if teacher_bounds is None:  # pragma: no cover - parser guarantees it
            raise RuntimeError("Expected bounded_drn teacher bounds after parsing.")
        if (
            model.conductance_min > teacher_bounds[0]
            or model.conductance_max < teacher_bounds[1]
        ):
            raise config_error(
                "config.model",
                "to contain the bounded_drn teacher conductance interval "
                f"{list(teacher_bounds)!r}",
                {
                    "conductance_min": model.conductance_min,
                    "conductance_max": model.conductance_max,
                },
            )
    train = modes.get("train")
    if (
        teacher.type == "bounded_drn"
        and isinstance(train, StudentTrainSettings)
        and train.update_backend.type in MEASURED_BACKENDS
    ):
        raise config_error(
            "config.modes.train.update_backend.type",
            "to be 'ideal' or 'program_verify' for a bounded_drn teacher",
            train.update_backend.type,
        )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type
        in MEASURED_BACKENDS
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
        in MEASURED_BACKENDS
        and train.update_backend.parameters["initial_target_mapping"]
        in {
            "dual_rail_pairwise_common_window",
            "dual_rail_quad_common_window",
        }
        and model.encoding != "single"
    ):
        raise config_error(
            "config.modes.train.update_backend.parameters.initial_target_mapping",
            "to select single model encoding when using "
            "a dual-rail common-window mapping",
            train.update_backend.parameters["initial_target_mapping"],
        )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type
        in MEASURED_BACKENDS
        and train.update_backend.parameters["initial_target_mapping"]
        in {
            "dual_rail_pairwise_common_window",
            "dual_rail_quad_common_window",
        }
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
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type == "measured_cohort_b"
        and model.encoding == "single"
        and train.update_backend.parameters["initial_target_mapping"]
        != "dual_rail_quad_common_window"
    ):
        raise config_error(
            "config.modes.train.update_backend.parameters."
            "initial_target_mapping",
            "to equal 'dual_rail_quad_common_window' for a four-device "
            "single-encoding cohort-B deployment",
            train.update_backend.parameters["initial_target_mapping"],
        )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type == "measured_cohort_a_sign_sgd"
        and (
            model.encoding != "single"
            or train.update_backend.parameters["initial_target_mapping"]
            != "dual_rail_quad_common_window"
        )
    ):
        raise config_error(
            "config.modes.train.update_backend.type",
            "to select the single-encoding four-device "
            "dual_rail_quad_common_window protocol when using "
            "'measured_cohort_a_sign_sgd'",
            train.update_backend.type,
        )
    if (
        isinstance(train, StudentTrainSettings)
        and train.update_backend.type
        == "measured_cohort_a_one_pulse_down"
    ):
        expected_mapping = (
            "dual_rail_quad_common_window"
            if model.encoding == "single"
            else "paired_affine_common_window"
        )
        provided_mapping = train.update_backend.parameters[
            "initial_target_mapping"
        ]
        if provided_mapping != expected_mapping:
            raise config_error(
                "config.modes.train.update_backend.parameters."
                "initial_target_mapping",
                f"to equal {expected_mapping!r} for {model.encoding!r} "
                "encoding with measured_cohort_a_one_pulse_down",
                provided_mapping,
            )
    mapping = _parse_mapping(raw["mapping"])
    if teacher.type == "bounded_drn":
        constraints = {
            "config.model.encoding": (model.encoding, "single"),
            "config.mapping.scale_fractions": (
                mapping.scale_fractions,
                (1.0,),
            ),
            "config.mapping.scale_fraction_pairs": (
                mapping.scale_fraction_pairs,
                None,
            ),
            "config.mapping.range_placement": (
                mapping.range_placement,
                "lower",
            ),
        }
        for constraint_path, (provided, expected) in constraints.items():
            if provided != expected:
                raise config_error(
                    constraint_path,
                    f"to equal {expected!r} for a bounded_drn teacher",
                    provided,
                )
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
        teacher=teacher,
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
        "teacher": document.teacher,
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
    "MEASURED_BACKENDS",
    "MEASURED_COHORT_A_BACKENDS",
    "MEASURED_COHORT_B_BACKENDS",
    "SCHEMA_VERSION",
    "StudentConfig",
    "StudentTrainSpec",
    "StudentValidateSpec",
    "TeacherSettings",
    "parse_student_config",
    "resolve_student_spec",
]
