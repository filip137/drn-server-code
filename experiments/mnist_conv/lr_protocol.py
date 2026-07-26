"""Dependency-light primitives for the Conv1/Conv2/Conv3 learning-rate study.

This module deliberately contains no training-backend or array-library imports.  It
is the shared numerical contract used by the probe, range-test, and candidate
selection stages; orchestration and artifact I/O live elsewhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


LR_STUDY_SCHEMA_VERSION = "mnist-conv-lr-study/v1"
LR_RUN_SCHEMA_VERSION = "mnist-conv-run/v2"

MODEL_SEED = 0
TRAIN_SHUFFLE_SEED = 0
AFFINE_SEED = 1729
VALIDATION_PER_CLASS = 500
VALIDATION_SIZE = 5_000
TRAIN_SIZE = 55_000
TRAIN_BATCH_SIZE = 16
VALIDATION_BATCH_SIZE = 128
STEPS_PER_EPOCH = 3_438
CANDIDATE_EPOCHS = 5
CANDIDATE_TOTAL_STEPS = 17_190
CANDIDATE_WARMUP_STEPS = 860

WEIGHT_MIN = 0.0
WEIGHT_MAX = 100.0
WEIGHT_SPAN = WEIGHT_MAX - WEIGHT_MIN
CALIBRATION_SETTLING_ITERATIONS = 64
TARGET_INITIAL_SATURATION = 0.30
HARD_SIGMOID_G_ON = 100.0
HARD_SIGMOID_G_OFF = 0.0
HARD_SIGMOID_V_OFF = 4.0
BIAS_NORMALIZATION_EPSILON = 1e-8
ZERO_UPDATE_EPSILON = 1e-30

PROBE_BATCHES = 32
RHO_TARGETS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
V6_RHO_CONV_GRID = (5e-4, 1e-3, 3e-3, 1e-2)
V6_RHO_DENSE_GRID = (3e-3, 1e-2, 3e-2, 1e-1)
MAIN_RANGE_STEPS = 600
RANGE_EXTENSION_STEPS = 128
RANGE_RHO_START = 1e-5
RANGE_RHO_END = 1e-2
RANGE_EXTENSION_RHO_END = 3e-2
RANGE_EMA_DECAY = 0.98
TARGET_SUMMARY_STEPS = 8


class LRProtocolValidationError(ValueError):
    """Raised when a value drifts from the frozen LR-study contract."""


def _error(path: str, expected: str, provided: Any) -> LRProtocolValidationError:
    return LRProtocolValidationError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _finite_number(
    value: Any,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(path, "a finite number", value)
    result = float(value)
    if not math.isfinite(result):
        raise _error(path, "a finite number", value)
    if positive and result <= 0.0:
        raise _error(path, "a positive finite number", value)
    if nonnegative and result < 0.0:
        raise _error(path, "a non-negative finite number", value)
    return result


def _positive_integer(value: Any, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _error(path, f"an integer >= {minimum}", value)
    return value


def _number_sequence(values: Iterable[float], path: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise _error(path, "a non-empty sequence of finite numbers", values)
    try:
        result = tuple(_finite_number(value, f"{path}[{index}]") for index, value in enumerate(values))
    except TypeError as exc:
        raise _error(path, "a non-empty sequence of finite numbers", values) from exc
    if not result:
        raise _error(path, "a non-empty sequence of finite numbers", values)
    return result


@dataclass(frozen=True)
class FrozenStudyRow:
    architecture: str
    scheme: str
    run_name: str
    voltage_amp: float
    current_amp: float
    input_gain: float
    settling_iterations: int
    training_iterations: int

    @property
    def key(self) -> tuple[str, str]:
        return self.architecture, self.scheme

    @property
    def T(self) -> int:
        return self.settling_iterations

    @property
    def K(self) -> int:
        return self.training_iterations


FROZEN_STUDY_ROWS = (
    FrozenStudyRow("conv1", "baseline", "mnist_bp_amp_v1_c1", 1.0, 1.0, 75.6030807495, 4, 4),
    FrozenStudyRow("conv1", "ours", "mnist_bp_amp_v4_c1", 4.0, 1.0, 84.8402175903, 4, 4),
    FrozenStudyRow("conv1", "legacy", "mnist_bp_amp_v4_c0p25", 4.0, 0.25, 31.8188591003, 4, 4),
    FrozenStudyRow("conv2", "baseline", "mnist_bp_amp_v1_c1", 1.0, 1.0, 253.3022308350, 16, 6),
    FrozenStudyRow("conv2", "ours", "mnist_bp_amp_v4_c1", 4.0, 1.0, 716.3439331055, 24, 6),
    FrozenStudyRow("conv2", "legacy", "mnist_bp_amp_v4_c0p25", 4.0, 0.25, 661.4369506836, 8, 4),
)

FROZEN_STUDY_ROWS_BY_KEY = MappingProxyType({row.key: row for row in FROZEN_STUDY_ROWS})

FROZEN_STUDY_SETTINGS = MappingProxyType(
    {
        "model_seed": MODEL_SEED,
        "train_shuffle_seed": TRAIN_SHUFFLE_SEED,
        "affine_seed": AFFINE_SEED,
        "validation_per_class": VALIDATION_PER_CLASS,
        "validation_size": VALIDATION_SIZE,
        "train_size": TRAIN_SIZE,
        "batch_size": TRAIN_BATCH_SIZE,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "weight_min": WEIGHT_MIN,
        "weight_max": WEIGHT_MAX,
        "weight_init_mode": "kaiming_uniform",
        "weight_gain": 1.0,
        "calibration_settling_iterations": CALIBRATION_SETTLING_ITERATIONS,
        "target_initial_saturation": TARGET_INITIAL_SATURATION,
        "non_linearity": "hard_sigmoid",
        "hard_sigmoid_g_on": HARD_SIGMOID_G_ON,
        "hard_sigmoid_g_off": HARD_SIGMOID_G_OFF,
        "hard_sigmoid_v_off": HARD_SIGMOID_V_OFF,
        "output_dim": 20,
        "fixed_amplification": True,
        "adaptive_equilibrium": False,
        "reset_state_each_batch": True,
        "optimizer_name": "SGD",
        "momentum": 0.0,
        "weight_decay": 0.0,
    }
)


def frozen_study_row(architecture: str, scheme: str) -> FrozenStudyRow:
    """Return one frozen row, rejecting unknown architecture/scheme pairs."""

    key = architecture, scheme
    try:
        return FROZEN_STUDY_ROWS_BY_KEY[key]
    except KeyError as exc:
        raise _error(
            "row key",
            f"one of {sorted(FROZEN_STUDY_ROWS_BY_KEY)}",
            key,
        ) from exc


def _row_from_mapping(value: Mapping[str, Any], path: str) -> FrozenStudyRow:
    fields = (
        "architecture",
        "scheme",
        "run_name",
        "voltage_amp",
        "current_amp",
        "input_gain",
        "settling_iterations",
        "training_iterations",
    )
    missing = [field for field in fields if field not in value]
    if missing:
        raise _error(path, f"a row containing fields {list(fields)!r}", {"missing": missing})
    architecture = value["architecture"]
    scheme = value["scheme"]
    run_name = value["run_name"]
    for field, item in (
        ("architecture", architecture),
        ("scheme", scheme),
        ("run_name", run_name),
    ):
        if not isinstance(item, str) or not item:
            raise _error(f"{path}.{field}", "a non-empty string", item)
    return FrozenStudyRow(
        architecture=architecture,
        scheme=scheme,
        run_name=run_name,
        voltage_amp=_finite_number(value["voltage_amp"], f"{path}.voltage_amp", positive=True),
        current_amp=_finite_number(value["current_amp"], f"{path}.current_amp", positive=True),
        input_gain=_finite_number(value["input_gain"], f"{path}.input_gain", positive=True),
        settling_iterations=_positive_integer(
            value["settling_iterations"], f"{path}.settling_iterations"
        ),
        training_iterations=_positive_integer(
            value["training_iterations"], f"{path}.training_iterations"
        ),
    )


def validate_frozen_study_row(
    value: FrozenStudyRow | Mapping[str, Any],
    *,
    path: str = "row",
) -> FrozenStudyRow:
    """Validate one row against its architecture/scheme frozen handoff."""

    if isinstance(value, FrozenStudyRow):
        row = value
    elif isinstance(value, Mapping):
        row = _row_from_mapping(value, path)
    else:
        raise _error(path, "a FrozenStudyRow or row mapping", value)
    expected = frozen_study_row(*row.key)
    if row != expected:
        raise _error(path, f"the frozen row {expected!r}", row)
    return expected


def validate_frozen_study_rows(
    rows: Iterable[FrozenStudyRow | Mapping[str, Any]],
) -> tuple[FrozenStudyRow, ...]:
    """Validate that ``rows`` are exactly the six frozen Conv1/Conv2 cases."""

    normalized: list[FrozenStudyRow] = []
    for index, value in enumerate(rows):
        if isinstance(value, FrozenStudyRow):
            row = value
        elif isinstance(value, Mapping):
            row = _row_from_mapping(value, f"rows[{index}]")
        else:
            raise _error(f"rows[{index}]", "a FrozenStudyRow or row mapping", value)
        normalized.append(row)

    keys = [row.key for row in normalized]
    expected_keys = set(FROZEN_STUDY_ROWS_BY_KEY)
    if len(keys) != len(set(keys)):
        raise _error("rows", "six rows with unique architecture/scheme keys", keys)
    if set(keys) != expected_keys:
        raise _error(
            "rows",
            f"exactly the frozen keys {sorted(expected_keys)!r}",
            sorted(keys),
        )
    for row in normalized:
        validate_frozen_study_row(row, path=f"row {row.key!r}")
    return FROZEN_STUDY_ROWS


def validate_frozen_study_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Require every fixed study setting while allowing unrelated config fields."""

    if not isinstance(settings, Mapping):
        raise _error("study settings", "a mapping", settings)
    missing = [name for name in FROZEN_STUDY_SETTINGS if name not in settings]
    if missing:
        raise _error(
            "study settings",
            f"a mapping containing {list(FROZEN_STUDY_SETTINGS)!r}",
            {"missing": missing},
        )
    for name, expected in FROZEN_STUDY_SETTINGS.items():
        provided = settings[name]
        if isinstance(expected, bool):
            matches = isinstance(provided, bool) and provided is expected
        elif isinstance(expected, float):
            matches = (
                not isinstance(provided, bool)
                and isinstance(provided, (int, float))
                and math.isfinite(float(provided))
                and float(provided) == expected
            )
        elif isinstance(expected, int):
            matches = isinstance(provided, int) and not isinstance(provided, bool) and provided == expected
        else:
            matches = provided == expected and type(provided) is type(expected)
        if not matches:
            raise _error(f"study settings.{name}", repr(expected), provided)
    return {name: settings[name] for name in FROZEN_STUDY_SETTINGS}


def rms(values: Iterable[float]) -> float:
    """Return root-mean-square with finite, non-empty input validation."""

    numbers = _number_sequence(values, "values")
    return math.sqrt(math.fsum(value * value for value in numbers) / len(numbers))


def normalized_update_from_rms(
    proposed_update_rms: float,
    *,
    bound_span: float = WEIGHT_SPAN,
) -> float:
    update = _finite_number(
        proposed_update_rms, "proposed_update_rms", nonnegative=True
    )
    span = _finite_number(bound_span, "bound_span", positive=True)
    return update / span


def normalized_update_rho(
    proposed_update: Iterable[float],
    *,
    bound_span: float = WEIGHT_SPAN,
) -> float:
    """Compute the plan's scale-aware bounded-weight update ``RMS(dW)/100``."""

    return normalized_update_from_rms(rms(proposed_update), bound_span=bound_span)


def normalized_bias_update(
    proposed_update: Iterable[float],
    parameter_value: Iterable[float],
    *,
    epsilon_s: float = BIAS_NORMALIZATION_EPSILON,
) -> float:
    """Normalize a bias update for reporting; this value never participates in gates."""

    epsilon = _finite_number(epsilon_s, "epsilon_s", positive=True)
    return rms(proposed_update) / max(rms(parameter_value), epsilon)


def projection_efficiency(
    proposed_update: Iterable[float],
    applied_update: Iterable[float],
    *,
    zero_epsilon: float = ZERO_UPDATE_EPSILON,
) -> float | None:
    """Return ``RMS(applied)/RMS(proposed)`` or ``None`` for a zero proposal."""

    proposed = _number_sequence(proposed_update, "proposed_update")
    applied = _number_sequence(applied_update, "applied_update")
    if len(proposed) != len(applied):
        raise _error(
            "applied_update",
            f"a sequence of length {len(proposed)}",
            len(applied),
        )
    epsilon = _finite_number(zero_epsilon, "zero_epsilon", nonnegative=True)
    proposed_rms = rms(proposed)
    if proposed_rms <= epsilon:
        return None
    return rms(applied) / proposed_rms


def linear_quantile(values: Iterable[float], quantile: float) -> float:
    """Return the unweighted quantile using standard linear interpolation."""

    numbers = sorted(_number_sequence(values, "values"))
    q = _finite_number(quantile, "quantile")
    if not 0.0 <= q <= 1.0:
        raise _error("quantile", "a finite number in [0, 1]", quantile)
    if len(numbers) == 1:
        return numbers[0]
    position = (len(numbers) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return numbers[lower]
    fraction = position - lower
    return numbers[lower] + fraction * (numbers[upper] - numbers[lower])


def probe_rho_unit(normalized_updates: Iterable[float]) -> float:
    """Aggregate tensor-by-minibatch normalized updates with linear Q90."""

    value = linear_quantile(normalized_updates, 0.9)
    if value <= 0.0:
        raise _error("rho_unit", "a positive Q90 normalized update", value)
    return value


def parameter_relative_update_from_rms(
    proposed_update_rms: float,
    *,
    initial_parameter_rms: float,
) -> float:
    """Normalize a proposal by the frozen RMS of its initialization tensor."""

    update = _finite_number(
        proposed_update_rms, "proposed_update_rms", nonnegative=True
    )
    scale = _finite_number(
        initial_parameter_rms, "initial_parameter_rms", positive=True
    )
    return update / scale


def probe_parameter_relative_rho_unit(
    normalized_updates_by_parameter: Mapping[str, Iterable[float]],
) -> tuple[float, dict[str, float]]:
    """Return ``max_parameter(Q90_batch(RMS(dW)/RMS(W0)))``.

    Taking the batch quantile independently for every bounded tensor and only
    then taking the maximum gives the aggregate the same interpretation for
    architectures with different numbers of trainable tensors: at nominal
    learning rate one, ``rho_unit`` is the largest robust relative proposal
    scale among bounded tensors.
    """

    if not isinstance(normalized_updates_by_parameter, Mapping):
        raise _error(
            "normalized_updates_by_parameter",
            "a non-empty mapping from parameter name to updates",
            normalized_updates_by_parameter,
        )
    if not normalized_updates_by_parameter:
        raise _error(
            "normalized_updates_by_parameter",
            "a non-empty mapping from parameter name to updates",
            normalized_updates_by_parameter,
        )
    per_parameter: dict[str, float] = {}
    for raw_name, updates in normalized_updates_by_parameter.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise _error(
                "normalized_updates_by_parameter",
                "non-empty string parameter names",
                raw_name,
            )
        name = raw_name.strip()
        if name in per_parameter:
            raise _error(
                "normalized_updates_by_parameter",
                "unique normalized parameter names",
                tuple(normalized_updates_by_parameter),
            )
        value = linear_quantile(updates, 0.9)
        if value <= 0.0:
            raise _error(
                f"rho_unit_by_parameter[{name!r}]",
                "a positive Q90 parameter-relative update",
                value,
            )
        per_parameter[name] = value
    return max(per_parameter.values()), per_parameter


def probe_median_parameter_relative_units(
    normalized_updates_by_weight: Mapping[str, Iterable[float]],
) -> dict[str, float]:
    """Return the per-weight batch median used by the v5 diagnostic.

    Values are nominal-LR relative proposals, ``RMS(delta W) / RMS(W0)``.
    Biases are deliberately rejected here: they inherit their convolutional
    weight rate after calibration and never define a normalization unit.
    """

    if not isinstance(normalized_updates_by_weight, Mapping) or not normalized_updates_by_weight:
        raise _error(
            "normalized_updates_by_weight",
            "a non-empty mapping containing only bounded weights",
            normalized_updates_by_weight,
        )
    units: dict[str, float] = {}
    for raw_name, updates in normalized_updates_by_weight.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise _error(
                "normalized_updates_by_weight",
                "non-empty canonical bounded-weight names",
                raw_name,
            )
        name = raw_name.strip()
        if not (name.startswith("ConvWeight_") or name.startswith("DenseWeight_")):
            raise _error(
                "normalized_updates_by_weight",
                "a mapping containing only bounded weights",
                name,
            )
        value = linear_quantile(updates, 0.5)
        if value <= 0.0:
            raise _error(
                f"median_relative_unit[{name!r}]",
                "a positive median parameter-relative proposal",
                value,
            )
        units[name] = value
    return units


def layerwise_parameter_groups(
    parameter_names: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Return the frozen Conv/Dense weight-to-parameter LR groups.

    Hidden biases are tied by their numeric suffix to the corresponding
    convolution weight.  The paired-output DenseWeight has no bias in this
    model.  The topology is validated from names so optimizer ordering can
    never silently change the association.
    """

    raw_names = tuple(parameter_names)
    names: list[str] = []
    for index, raw_name in enumerate(raw_names):
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise _error(
                f"parameter_names[{index}]", "a non-empty string", raw_name
            )
        names.append(raw_name.strip())
    if not names or len(names) != len(set(names)):
        raise _error(
            "parameter_names",
            "a non-empty sequence of unique canonical parameter names",
            raw_names,
        )

    def indexed(prefix: str) -> dict[int, str]:
        result: dict[int, str] = {}
        for name in names:
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if not suffix.isdigit():
                raise _error(
                    "parameter_names",
                    f"canonical names of the form {prefix}<nonnegative integer>",
                    name,
                )
            value = int(suffix)
            if value in result:
                raise _error(
                    "parameter_names", f"unique {prefix} indices", raw_names
                )
            result[value] = name
        return result

    conv = indexed("ConvWeight_")
    dense = indexed("DenseWeight_")
    biases = indexed("Bias_")
    expected_conv_indices = set(range(len(conv)))
    if set(conv) != expected_conv_indices or len(conv) not in (1, 2, 3):
        raise _error(
            "parameter_names",
            "one, two, or three contiguous ConvWeight indices starting at zero",
            sorted(conv),
        )
    if set(dense) != {0}:
        raise _error(
            "parameter_names", "exactly DenseWeight_0", sorted(dense)
        )
    if set(biases) != expected_conv_indices:
        raise _error(
            "parameter_names",
            "exactly one Bias_i for every ConvWeight_i and no Dense output bias",
            sorted(biases),
        )
    recognized = set(conv.values()) | set(dense.values()) | set(biases.values())
    if recognized != set(names):
        raise _error(
            "parameter_names",
            "only ConvWeight_i, DenseWeight_0, and Bias_i trainable parameters",
            sorted(set(names) - recognized),
        )

    groups: dict[str, tuple[str, ...]] = {}
    for index in sorted(conv):
        groups[conv[index]] = (conv[index], biases[index])
    groups[dense[0]] = (dense[0],)
    return groups


def layerwise_target_learning_rates(
    rho_unit_by_weight: Mapping[str, float],
    rho_target: float,
    parameter_names: Iterable[str],
) -> dict[str, float]:
    """Convert one common relative-rho target to a complete LR vector."""

    groups = layerwise_parameter_groups(parameter_names)
    if not isinstance(rho_unit_by_weight, Mapping) or set(rho_unit_by_weight) != set(
        groups
    ):
        raise _error(
            "rho_unit_by_weight",
            f"exactly the bounded weights {sorted(groups)!r}",
            sorted(rho_unit_by_weight) if isinstance(rho_unit_by_weight, Mapping) else rho_unit_by_weight,
        )
    target = _finite_number(rho_target, "rho_target", positive=True)
    rates: dict[str, float] = {}
    for weight_name, members in groups.items():
        unit = _finite_number(
            rho_unit_by_weight[weight_name],
            f"rho_unit_by_weight[{weight_name!r}]",
            positive=True,
        )
        rate = target / unit
        for member in members:
            rates[member] = rate
    return rates


def architecture_relative_learning_rates(
    median_units_by_weight: Mapping[str, float],
    alpha: float,
    parameter_names: Iterable[str],
    *,
    target_multipliers: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build a v5 constant LR vector from one architecture-level ``alpha``.

    ``target_multipliers`` is one for strict equalization.  The historical
    profile keeps convolutional multipliers at one and assigns the documented
    dense multiplier.  Every ``Bias_i`` is tied exactly to ``ConvWeight_i``.
    """

    groups = layerwise_parameter_groups(parameter_names)
    expected_weights = set(groups)
    if not isinstance(median_units_by_weight, Mapping) or set(median_units_by_weight) != expected_weights:
        raise _error(
            "median_units_by_weight",
            f"exactly the bounded weights {sorted(expected_weights)!r}",
            (
                sorted(median_units_by_weight)
                if isinstance(median_units_by_weight, Mapping)
                else median_units_by_weight
            ),
        )
    base_target = _finite_number(alpha, "alpha", positive=True)
    multipliers: Mapping[str, float]
    if target_multipliers is None:
        multipliers = {name: 1.0 for name in expected_weights}
    else:
        if not isinstance(target_multipliers, Mapping) or set(target_multipliers) != expected_weights:
            raise _error(
                "target_multipliers",
                f"exactly the bounded weights {sorted(expected_weights)!r}",
                (
                    sorted(target_multipliers)
                    if isinstance(target_multipliers, Mapping)
                    else target_multipliers
                ),
            )
        multipliers = target_multipliers

    rates: dict[str, float] = {}
    for weight_name, members in groups.items():
        unit = _finite_number(
            median_units_by_weight[weight_name],
            f"median_units_by_weight[{weight_name!r}]",
            positive=True,
        )
        multiplier = _finite_number(
            multipliers[weight_name],
            f"target_multipliers[{weight_name!r}]",
            positive=True,
        )
        rate = base_target * multiplier / unit
        for member in members:
            rates[member] = rate
    return rates


def two_rho_learning_rates(
    median_units_by_weight: Mapping[str, float],
    rho_conv: float,
    rho_dense: float,
    parameter_names: Iterable[str] | None = None,
    *,
    bias_weight_lr_groups: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, float]:
    """Build a constant LR vector from direct Conv and Dense targets.

    Each bounded weight keeps its independently measured median unit.  All
    convolutional weights target ``rho_conv`` and the output weight targets
    ``rho_dense``; consequently convolutional raw rates need not be equal.
    Hidden biases inherit the raw rate of their associated convolutional
    weight and never enter either normalization unit.
    """

    if (parameter_names is None) == (bias_weight_lr_groups is None):
        raise _error(
            "parameter topology",
            "exactly one of parameter_names or bias_weight_lr_groups",
            {
                "parameter_names": parameter_names,
                "bias_weight_lr_groups": bias_weight_lr_groups,
            },
        )
    if bias_weight_lr_groups is None:
        groups = layerwise_parameter_groups(parameter_names or ())
    else:
        if not isinstance(bias_weight_lr_groups, Mapping):
            raise _error(
                "bias_weight_lr_groups",
                "a canonical weight-to-parameter mapping",
                bias_weight_lr_groups,
            )
        provided_groups: dict[str, tuple[str, ...]] = {}
        flattened: list[str] = []
        for raw_weight, raw_members in bias_weight_lr_groups.items():
            if not isinstance(raw_weight, str):
                raise _error(
                    "bias_weight_lr_groups",
                    "string bounded-weight keys",
                    raw_weight,
                )
            members = tuple(raw_members)
            provided_groups[raw_weight] = members
            flattened.extend(members)
        groups = layerwise_parameter_groups(flattened)
        if provided_groups != groups:
            raise _error(
                "bias_weight_lr_groups",
                f"the canonical mapping {groups!r}",
                provided_groups,
            )
    expected_weights = set(groups)
    if not isinstance(median_units_by_weight, Mapping) or set(
        median_units_by_weight
    ) != expected_weights:
        raise _error(
            "median_units_by_weight",
            f"exactly the bounded weights {sorted(expected_weights)!r}",
            (
                sorted(median_units_by_weight)
                if isinstance(median_units_by_weight, Mapping)
                else median_units_by_weight
            ),
        )
    conv_target = _finite_number(rho_conv, "rho_conv", positive=True)
    dense_target = _finite_number(rho_dense, "rho_dense", positive=True)
    rates: dict[str, float] = {}
    for weight_name, members in groups.items():
        unit = _finite_number(
            median_units_by_weight[weight_name],
            f"median_units_by_weight[{weight_name!r}]",
            positive=True,
        )
        target = conv_target if weight_name.startswith("ConvWeight_") else dense_target
        rate = target / unit
        for member in members:
            rates[member] = rate
    return rates


def two_rho_candidate_grid(
    rho_conv_values: Iterable[float] = V6_RHO_CONV_GRID,
    rho_dense_values: Iterable[float] = V6_RHO_DENSE_GRID,
) -> tuple[tuple[float, float], ...]:
    """Return a deterministic Conv-outer, Dense-inner Cartesian target grid."""

    conv_values = tuple(rho_conv_values)
    dense_values = tuple(rho_dense_values)
    if not conv_values:
        raise _error("rho_conv_values", "a non-empty sequence", conv_values)
    if not dense_values:
        raise _error("rho_dense_values", "a non-empty sequence", dense_values)

    def normalized(values: tuple[float, ...], path: str) -> tuple[float, ...]:
        result = tuple(
            _finite_number(value, f"{path}[{index}]", positive=True)
            for index, value in enumerate(values)
        )
        if len(result) != len(set(result)):
            raise _error(path, "unique positive finite values", values)
        if tuple(sorted(result)) != result:
            raise _error(path, "strictly increasing values", values)
        return result

    conv = normalized(conv_values, "rho_conv_values")
    dense = normalized(dense_values, "rho_dense_values")
    return tuple((rho_conv, rho_dense) for rho_conv in conv for rho_dense in dense)


def alpha_candidate_grid(center: float) -> tuple[float, float, float]:
    """Return the frozen architecture/arm grid ``{center/3, center, 3*center}``."""

    value = _finite_number(center, "center", positive=True)
    return value / 3.0, value, value * 3.0


def layerwise_learning_rate_report(
    learning_rates: Mapping[str, float],
    *,
    thresholds: Iterable[float] = (1.0, 10.0),
    bias_learning_rate_overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Summarize raw weight rates; thresholds are reporting-only flags.

    The default path retains the frozen bias-to-weight equality contract.
    A separately content-addressed controlled diagnostic may declare exact
    bias-rate overrides; undeclared mismatches remain invalid.
    """

    groups = layerwise_parameter_groups(learning_rates)
    weights = {
        name: _finite_number(
            learning_rates[name], f"learning_rates[{name!r}]", positive=True
        )
        for name in groups
    }
    canonical_biases = {
        member
        for members in groups.values()
        for member in members
        if member.startswith("Bias_")
    }
    overrides: dict[str, float] = {}
    if bias_learning_rate_overrides is not None:
        if (
            not isinstance(bias_learning_rate_overrides, Mapping)
            or not bias_learning_rate_overrides
        ):
            raise _error(
                "bias_learning_rate_overrides",
                "a non-empty mapping of canonical Bias_i names to positive finite rates",
                bias_learning_rate_overrides,
            )
        if not set(bias_learning_rate_overrides).issubset(canonical_biases):
            raise _error(
                "bias_learning_rate_overrides",
                f"a subset of the canonical biases {sorted(canonical_biases)!r}",
                sorted(bias_learning_rate_overrides),
            )
        overrides = {
            name: _finite_number(
                value,
                f"bias_learning_rate_overrides[{name!r}]",
                positive=True,
            )
            for name, value in bias_learning_rate_overrides.items()
        }
    for weight_name, members in groups.items():
        weight_rate = weights[weight_name]
        for member in members:
            expected = overrides.get(member, weight_rate)
            observed = _finite_number(
                learning_rates[member],
                f"learning_rates[{member!r}]",
                positive=True,
            )
            if observed != expected:
                raise _error(
                    f"learning_rates[{member!r}]",
                    (
                        f"exactly its declared controlled override {expected!r}"
                        if member in overrides
                        else f"exactly its associated weight rate {expected!r}"
                    ),
                    observed,
                )
    threshold_values = _number_sequence(thresholds, "thresholds")
    if any(value <= 0.0 for value in threshold_values):
        raise _error("thresholds", "positive finite values", threshold_values)
    minimum = min(weights.values())
    maximum = max(weights.values())
    report = {
        "weight_learning_rates": weights,
        "minimum_weight_learning_rate": minimum,
        "maximum_weight_learning_rate": maximum,
        "maximum_to_minimum_weight_lr_ratio": maximum / minimum,
        "reporting_only_thresholds": {
            str(value): sorted(
                name for name, rate in weights.items() if rate >= value
            )
            for value in threshold_values
        },
        "raw_lr_is_gate": False,
        "raw_lr_is_capped": False,
    }
    if overrides:
        report["bias_learning_rate_overrides"] = overrides
    return report


def target_learning_rates(
    rho_unit: float,
    targets: Iterable[float] = RHO_TARGETS,
) -> dict[float, float]:
    """Convert normalized-update targets to raw SGD rates via ``eta=rho/rho_unit``."""

    unit = _finite_number(rho_unit, "rho_unit", positive=True)
    target_values = tuple(targets)
    result: dict[float, float] = {}
    for index, value in enumerate(target_values):
        target = _finite_number(value, f"targets[{index}]", positive=True)
        if target in result:
            raise _error("targets", "unique positive finite values", target_values)
        result[target] = target / unit
    if not result:
        raise _error(
            "targets", "a non-empty sequence of positive finite values", target_values
        )
    return result


def geometric_schedule(start: float, end: float, steps: int) -> tuple[float, ...]:
    """Return a positive geometric schedule containing both exact endpoints."""

    first = _finite_number(start, "start", positive=True)
    last = _finite_number(end, "end", positive=True)
    count = _positive_integer(steps, "steps", minimum=2)
    log_first = math.log(first)
    log_delta = math.log(last / first)
    values = [math.exp(log_first + log_delta * index / (count - 1)) for index in range(count)]
    values[0] = first
    values[-1] = last
    return tuple(values)


def range_normalized_update_schedule(
    *,
    include_extension: bool = False,
    main_steps: int = MAIN_RANGE_STEPS,
    start_rho: float = RANGE_RHO_START,
    end_rho: float = RANGE_RHO_END,
    exact_tail_start_rho: float | None = None,
    exact_tail_steps: int | None = None,
    extension_steps: int = RANGE_EXTENSION_STEPS,
    extension_start_rho: float | None = None,
    extension_end_rho: float = RANGE_EXTENSION_RHO_END,
) -> tuple[float, ...]:
    """Return a configurable geometric range schedule with exact endpoints.

    With no overrides this is the frozen 600-step v1 schedule.  A caller that
    must prepend a warm-in while preserving an existing schedule *bit for bit*
    can provide ``exact_tail_start_rho`` and ``exact_tail_steps``.  The prefix
    and tail are then constructed independently and their shared endpoint is
    stored once.  For example, ``main_steps=1199``, ``start_rho=1e-8``,
    ``exact_tail_start_rho=1e-5``, and ``exact_tail_steps=600`` makes steps
    600--1199 exactly equal to the legacy 600-step ``1e-5``--``1e-2`` schedule.

    The optional extension deliberately retains its repeated boundary value so
    that the steps following the final main-range target are available for
    target-window summarization.
    """

    if not isinstance(include_extension, bool):
        raise _error("include_extension", "a boolean", include_extension)
    main_count = _positive_integer(main_steps, "main_steps", minimum=2)
    first = _finite_number(start_rho, "start_rho", positive=True)
    last = _finite_number(end_rho, "end_rho", positive=True)
    if first >= last:
        raise _error("start_rho", f"a positive number < end_rho ({last})", start_rho)

    if exact_tail_start_rho is None and exact_tail_steps is None:
        main = geometric_schedule(first, last, main_count)
    elif exact_tail_start_rho is None or exact_tail_steps is None:
        raise _error(
            "exact tail",
            "both exact_tail_start_rho and exact_tail_steps, or neither",
            {
                "exact_tail_start_rho": exact_tail_start_rho,
                "exact_tail_steps": exact_tail_steps,
            },
        )
    else:
        seam = _finite_number(
            exact_tail_start_rho, "exact_tail_start_rho", positive=True
        )
        tail_count = _positive_integer(
            exact_tail_steps, "exact_tail_steps", minimum=2
        )
        prefix_count = main_count - tail_count + 1
        if prefix_count < 2:
            raise _error(
                "exact_tail_steps",
                f"an integer <= main_steps - 1 ({main_count - 1})",
                exact_tail_steps,
            )
        if not first < seam < last:
            raise _error(
                "exact_tail_start_rho",
                f"a number strictly between start_rho ({first}) and end_rho ({last})",
                exact_tail_start_rho,
            )
        prefix_log_step = math.log(seam / first) / (prefix_count - 1)
        tail_log_step = math.log(last / seam) / (tail_count - 1)
        if not math.isclose(
            prefix_log_step,
            tail_log_step,
            rel_tol=1e-14,
            abs_tol=1e-15,
        ):
            raise _error(
                "exact tail geometric ratio",
                "the same per-step log ratio in the prefix and exact tail",
                {
                    "prefix_log_step": prefix_log_step,
                    "tail_log_step": tail_log_step,
                },
            )
        prefix = geometric_schedule(first, seam, prefix_count)
        tail = geometric_schedule(seam, last, tail_count)
        main = prefix[:-1] + tail

    if not include_extension:
        return main
    extension_count = _positive_integer(
        extension_steps, "extension_steps", minimum=2
    )
    extension_first = (
        last
        if extension_start_rho is None
        else _finite_number(
            extension_start_rho, "extension_start_rho", positive=True
        )
    )
    if extension_first != last:
        raise _error(
            "extension_start_rho",
            f"exactly end_rho ({last})",
            extension_start_rho,
        )
    extension_last = _finite_number(
        extension_end_rho, "extension_end_rho", positive=True
    )
    if extension_last <= extension_first:
        raise _error(
            "extension_end_rho",
            f"a number > extension_start_rho ({extension_first})",
            extension_end_rho,
        )
    extension = geometric_schedule(
        extension_first, extension_last, extension_count
    )
    return main + extension


def range_learning_rate_schedule(
    rho_unit: float,
    *,
    include_extension: bool = False,
    main_steps: int = MAIN_RANGE_STEPS,
    start_rho: float = RANGE_RHO_START,
    end_rho: float = RANGE_RHO_END,
    exact_tail_start_rho: float | None = None,
    exact_tail_steps: int | None = None,
    extension_steps: int = RANGE_EXTENSION_STEPS,
    extension_start_rho: float | None = None,
    extension_end_rho: float = RANGE_EXTENSION_RHO_END,
) -> tuple[float, ...]:
    unit = _finite_number(rho_unit, "rho_unit", positive=True)
    return tuple(
        rho / unit
        for rho in range_normalized_update_schedule(
            include_extension=include_extension,
            main_steps=main_steps,
            start_rho=start_rho,
            end_rho=end_rho,
            exact_tail_start_rho=exact_tail_start_rho,
            exact_tail_steps=exact_tail_steps,
            extension_steps=extension_steps,
            extension_start_rho=extension_start_rho,
            extension_end_rho=extension_end_rho,
        )
    )


def candidate_learning_rate_at_step(
    peak_learning_rate: float,
    step: int,
    *,
    total_steps: int = CANDIDATE_TOTAL_STEPS,
    warmup_steps: int = CANDIDATE_WARMUP_STEPS,
) -> float:
    """Return the 1-indexed linear-warmup/cosine-decay learning rate."""

    peak = _finite_number(peak_learning_rate, "peak_learning_rate", positive=True)
    total = _positive_integer(total_steps, "total_steps")
    warmup = _positive_integer(warmup_steps, "warmup_steps")
    if warmup >= total:
        raise _error("warmup_steps", f"an integer < total_steps ({total})", warmup_steps)
    current = _positive_integer(step, "step")
    if current > total:
        raise _error("step", f"an integer in [1, {total}]", step)
    if current <= warmup:
        return peak * current / warmup
    progress = (current - warmup) / (total - warmup)
    if current == total:
        return 0.0
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def candidate_learning_rate_schedule(
    peak_learning_rate: float,
    *,
    total_steps: int = CANDIDATE_TOTAL_STEPS,
    warmup_steps: int = CANDIDATE_WARMUP_STEPS,
) -> tuple[float, ...]:
    total = _positive_integer(total_steps, "total_steps")
    return tuple(
        candidate_learning_rate_at_step(
            peak_learning_rate,
            step,
            total_steps=total,
            warmup_steps=warmup_steps,
        )
        for step in range(1, total + 1)
    )


def exponential_moving_average(
    values: Iterable[float],
    *,
    decay: float = RANGE_EMA_DECAY,
) -> tuple[float, ...]:
    numbers = _number_sequence(values, "values")
    alpha = _finite_number(decay, "decay")
    if not 0.0 <= alpha < 1.0:
        raise _error("decay", "a finite number in [0, 1)", decay)
    result = [numbers[0]]
    for value in numbers[1:]:
        result.append(alpha * result[-1] + (1.0 - alpha) * value)
    return tuple(result)


@dataclass(frozen=True)
class RangeTensorRecord:
    name: str
    bounded: bool
    gradient_rms: float
    proposed_update_rms: float
    normalized_update: float
    lower_bound_occupancy: float = 0.0
    upper_bound_occupancy: float = 0.0
    projection_efficiency: float | None = None
    proposed_bound_crossing_fraction: float = 0.0

    @property
    def bound_occupancy(self) -> float:
        return self.lower_bound_occupancy + self.upper_bound_occupancy


@dataclass(frozen=True)
class RangeStepRecord:
    step: int
    learning_rate: float
    rho: float
    loss: float
    tensors: tuple[RangeTensorRecord, ...]
    state_finite: bool = True
    diagnostic_finite: bool = True


@dataclass(frozen=True)
class RangeFailure:
    kind: str
    onset_step: int
    confirmed_step: int
    tensor_name: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class RangeGateResult:
    failure: RangeFailure | None
    maximum_admissible_step: int
    maximum_admissible_learning_rate: float | None
    maximum_admissible_rho: float | None
    last_processed_step: int
    extension_allowed: bool
    active_gates_at_main_end: tuple[str, ...] = ()


def _validate_range_structure(records: Sequence[RangeStepRecord], reference_steps: int) -> tuple[str, ...]:
    if len(records) < reference_steps:
        raise _error("records", f"at least {reference_steps} step records", len(records))
    first_names: tuple[str, ...] | None = None
    bounded_by_name: dict[str, bool] | None = None
    for index, record in enumerate(records):
        if not isinstance(record, RangeStepRecord):
            raise _error(f"records[{index}]", "a RangeStepRecord", record)
        expected_step = index + 1
        if record.step != expected_step:
            raise _error(f"records[{index}].step", str(expected_step), record.step)
        if not isinstance(record.tensors, tuple) or not record.tensors:
            raise _error(f"records[{index}].tensors", "a non-empty tuple", record.tensors)
        for tensor_index, tensor in enumerate(record.tensors):
            if not isinstance(tensor, RangeTensorRecord):
                raise _error(
                    f"records[{index}].tensors[{tensor_index}]",
                    "a RangeTensorRecord",
                    tensor,
                )
        names = tuple(tensor.name for tensor in record.tensors)
        if any(not isinstance(name, str) or not name for name in names):
            raise _error(f"records[{index}].tensors", "non-empty tensor names", names)
        if len(names) != len(set(names)):
            raise _error(f"records[{index}].tensors", "unique tensor names", names)
        if first_names is None:
            first_names = names
            bounded_by_name = {tensor.name: tensor.bounded for tensor in record.tensors}
        elif set(names) != set(first_names):
            raise _error(
                f"records[{index}].tensors",
                f"the tensor-name set {sorted(first_names)!r}",
                sorted(names),
            )
        else:
            assert bounded_by_name is not None
            provided_flags = {tensor.name: tensor.bounded for tensor in record.tensors}
            if provided_flags != bounded_by_name:
                raise _error(
                    f"records[{index}].tensors bounded flags",
                    repr(bounded_by_name),
                    provided_flags,
                )
    assert first_names is not None
    return first_names


def _immediate_failure(record: RangeStepRecord, zero_epsilon: float) -> RangeFailure | None:
    if record.state_finite is not True:
        return RangeFailure("non_finite", record.step, record.step, detail="state_finite")
    if record.diagnostic_finite is not True:
        return RangeFailure("non_finite", record.step, record.step, detail="diagnostic_finite")
    for field, value in (
        ("learning_rate", record.learning_rate),
        ("rho", record.rho),
        ("loss", record.loss),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return RangeFailure("non_finite", record.step, record.step, detail=field)
    for tensor in record.tensors:
        values = (
            ("gradient_rms", tensor.gradient_rms),
            ("proposed_update_rms", tensor.proposed_update_rms),
            ("normalized_update", tensor.normalized_update),
            ("lower_bound_occupancy", tensor.lower_bound_occupancy),
            ("upper_bound_occupancy", tensor.upper_bound_occupancy),
            ("proposed_bound_crossing_fraction", tensor.proposed_bound_crossing_fraction),
        )
        for field, value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return RangeFailure(
                    "non_finite", record.step, record.step, tensor.name, field
                )
        if tensor.projection_efficiency is None:
            if float(tensor.proposed_update_rms) > zero_epsilon:
                return RangeFailure(
                    "non_finite",
                    record.step,
                    record.step,
                    tensor.name,
                    "projection_efficiency",
                )
        elif (
            not isinstance(tensor.projection_efficiency, (int, float))
            or isinstance(tensor.projection_efficiency, bool)
            or not math.isfinite(float(tensor.projection_efficiency))
        ):
            return RangeFailure(
                "non_finite",
                record.step,
                record.step,
                tensor.name,
                "projection_efficiency",
            )
        if (
            tensor.lower_bound_occupancy < 0.0
            or tensor.upper_bound_occupancy < 0.0
            or tensor.bound_occupancy > 1.0 + 1e-12
            or tensor.proposed_bound_crossing_fraction < 0.0
            or tensor.proposed_bound_crossing_fraction > 1.0
        ):
            return RangeFailure(
                "non_finite", record.step, record.step, tensor.name, "diagnostic_range"
            )
    return None


def _failure_result(
    records: Sequence[RangeStepRecord],
    failure: RangeFailure,
    *,
    main_range_steps: int = MAIN_RANGE_STEPS,
    active_at_main_end: tuple[str, ...] = (),
) -> RangeGateResult:
    main_count = _positive_integer(
        main_range_steps, "main_range_steps", minimum=2
    )
    maximum_step = max(0, failure.onset_step - 1)
    record = records[maximum_step - 1] if maximum_step else None
    return RangeGateResult(
        failure=failure,
        maximum_admissible_step=maximum_step,
        maximum_admissible_learning_rate=None if record is None else float(record.learning_rate),
        maximum_admissible_rho=None if record is None else float(record.rho),
        last_processed_step=failure.confirmed_step,
        extension_allowed=(
            failure.confirmed_step > main_count and not active_at_main_end
        ),
        active_gates_at_main_end=active_at_main_end,
    )


def analyze_range_stop_gates(
    records: Sequence[RangeStepRecord],
    initial_bound_occupancy: Mapping[str, float],
    *,
    reference_steps: int = PROBE_BATCHES,
    main_range_steps: int = MAIN_RANGE_STEPS,
    ema_decay: float = RANGE_EMA_DECAY,
    zero_update_epsilon: float = ZERO_UPDATE_EPSILON,
) -> RangeGateResult:
    """Apply the range-test gates in online order and backdate sustained onset.

    A zero proposed update neither increments nor clears a projection-efficiency
    streak.  Every other false condition clears its corresponding streak.
    """

    reference_count = _positive_integer(reference_steps, "reference_steps")
    main_count = _positive_integer(
        main_range_steps, "main_range_steps", minimum=2
    )
    if main_count <= reference_count:
        raise _error(
            "main_range_steps",
            f"an integer > reference_steps ({reference_count})",
            main_range_steps,
        )
    decay = _finite_number(ema_decay, "ema_decay")
    if not 0.0 <= decay < 1.0:
        raise _error("ema_decay", "a finite number in [0, 1)", ema_decay)
    zero_epsilon = _finite_number(
        zero_update_epsilon, "zero_update_epsilon", nonnegative=True
    )
    _validate_range_structure(records, reference_count)
    bounded_names = {
        tensor.name for tensor in records[0].tensors if tensor.bounded
    }
    if not bounded_names:
        raise _error("records[0].tensors", "at least one bounded tensor", records[0].tensors)
    if set(initial_bound_occupancy) != bounded_names:
        raise _error(
            "initial_bound_occupancy",
            f"exactly the bounded tensor names {sorted(bounded_names)!r}",
            sorted(initial_bound_occupancy),
        )
    initial_occupancy: dict[str, float] = {}
    for name, value in initial_bound_occupancy.items():
        occupancy = _finite_number(value, f"initial_bound_occupancy[{name!r}]")
        if not 0.0 <= occupancy <= 1.0:
            raise _error(
                f"initial_bound_occupancy[{name!r}]", "a fraction in [0, 1]", value
            )
        initial_occupancy[name] = occupancy

    ema_values: list[float] = []
    for record in records[:reference_count]:
        immediate = _immediate_failure(record, zero_epsilon)
        if immediate is not None:
            return _failure_result(
                records, immediate, main_range_steps=main_count
            )
        current_ema = (
            float(record.loss)
            if not ema_values
            else decay * ema_values[-1] + (1.0 - decay) * float(record.loss)
        )
        ema_values.append(current_ema)

    gradient_reference: dict[str, float] = {}
    for name in bounded_names:
        gradients = [
            next(tensor for tensor in record.tensors if tensor.name == name).gradient_rms
            for record in records[:reference_count]
        ]
        gradient_reference[name] = linear_quantile(gradients, 0.5)

    prior_loss_minimum = min(ema_values)
    loss_streak = 0
    loss_start: int | None = None
    gradient_streak = {name: 0 for name in bounded_names}
    gradient_start: dict[str, int | None] = {name: None for name in bounded_names}
    occupancy_streak = {name: 0 for name in bounded_names}
    occupancy_start: dict[str, int | None] = {name: None for name in bounded_names}
    projection_streak = {name: 0 for name in bounded_names}
    projection_start: dict[str, int | None] = {name: None for name in bounded_names}
    active_at_main_end: tuple[str, ...] = ()

    for record in records[reference_count:]:
        immediate = _immediate_failure(record, zero_epsilon)
        if immediate is not None:
            return _failure_result(
                records,
                immediate,
                main_range_steps=main_count,
                active_at_main_end=active_at_main_end,
            )

        current_ema = decay * ema_values[-1] + (1.0 - decay) * float(record.loss)
        ema_values.append(current_ema)
        newly_confirmed: list[RangeFailure] = []

        if current_ema > 4.0 * prior_loss_minimum:
            if loss_streak == 0:
                loss_start = record.step
            loss_streak += 1
            if loss_streak == 8:
                assert loss_start is not None
                newly_confirmed.append(
                    RangeFailure("loss_ema_explosion", loss_start, record.step)
                )
        else:
            loss_streak, loss_start = 0, None
        prior_loss_minimum = min(prior_loss_minimum, current_ema)

        tensors = {tensor.name: tensor for tensor in record.tensors}
        for name in bounded_names:
            tensor = tensors[name]
            if tensor.gradient_rms > 100.0 * gradient_reference[name]:
                if gradient_streak[name] == 0:
                    gradient_start[name] = record.step
                gradient_streak[name] += 1
                if gradient_streak[name] == 8:
                    assert gradient_start[name] is not None
                    newly_confirmed.append(
                        RangeFailure(
                            "gradient_rms_explosion",
                            gradient_start[name],
                            record.step,
                            name,
                        )
                    )
            else:
                gradient_streak[name], gradient_start[name] = 0, None

            if tensor.bound_occupancy > initial_occupancy[name] + 0.20:
                if occupancy_streak[name] == 0:
                    occupancy_start[name] = record.step
                occupancy_streak[name] += 1
                if occupancy_streak[name] == 16:
                    assert occupancy_start[name] is not None
                    newly_confirmed.append(
                        RangeFailure(
                            "bound_occupancy_increase",
                            occupancy_start[name],
                            record.step,
                            name,
                        )
                    )
            else:
                occupancy_streak[name], occupancy_start[name] = 0, None

            if tensor.proposed_update_rms > zero_epsilon:
                assert tensor.projection_efficiency is not None
                if tensor.projection_efficiency < 0.50:
                    if projection_streak[name] == 0:
                        projection_start[name] = record.step
                    projection_streak[name] += 1
                    if projection_streak[name] == 16:
                        assert projection_start[name] is not None
                        newly_confirmed.append(
                            RangeFailure(
                                "projection_efficiency",
                                projection_start[name],
                                record.step,
                                name,
                            )
                        )
                else:
                    projection_streak[name], projection_start[name] = 0, None

        if record.step == main_count:
            active: list[str] = []
            if loss_streak:
                active.append("loss_ema_explosion")
            active.extend(
                f"gradient_rms_explosion:{name}"
                for name in sorted(bounded_names)
                if gradient_streak[name]
            )
            active.extend(
                f"bound_occupancy_increase:{name}"
                for name in sorted(bounded_names)
                if occupancy_streak[name]
            )
            active.extend(
                f"projection_efficiency:{name}"
                for name in sorted(bounded_names)
                if projection_streak[name]
            )
            active_at_main_end = tuple(active)

        if newly_confirmed:
            failure = min(
                newly_confirmed,
                key=lambda item: (item.onset_step, item.kind, item.tensor_name or ""),
            )
            return _failure_result(
                records,
                failure,
                main_range_steps=main_count,
                active_at_main_end=active_at_main_end,
            )

    last = records[-1]
    extension_allowed = (
        last.step >= main_count and not active_at_main_end
    )
    return RangeGateResult(
        failure=None,
        maximum_admissible_step=last.step,
        maximum_admissible_learning_rate=float(last.learning_rate),
        maximum_admissible_rho=float(last.rho),
        last_processed_step=last.step,
        extension_allowed=extension_allowed,
        active_gates_at_main_end=active_at_main_end,
    )


@dataclass(frozen=True)
class TargetWindowSummary:
    rho_target: float
    crossing_step: int
    window_start_step: int
    window_end_step: int
    median_smoothed_loss: float


@dataclass(frozen=True)
class RangeCandidateSet:
    status: str
    reason: str | None
    fast: TargetWindowSummary | None
    middle: TargetWindowSummary | None
    high: TargetWindowSummary | None
    stable_targets: tuple[TargetWindowSummary, ...]

    @property
    def selected_targets(self) -> tuple[float, ...]:
        if self.status != "resolved":
            return ()
        assert self.fast is not None and self.middle is not None and self.high is not None
        return self.fast.rho_target, self.middle.rho_target, self.high.rho_target


def _nearest_unused_target(
    summaries: Sequence[TargetWindowSummary],
    desired: float,
    used: set[float],
) -> TargetWindowSummary:
    available = [summary for summary in summaries if summary.rho_target not in used]
    if not available:
        raise _error("stable targets", "an unused target", tuple(sorted(used)))
    return min(
        available,
        key=lambda summary: (
            abs(math.log(summary.rho_target) - math.log(desired)),
            summary.rho_target,
        ),
    )


def extract_range_candidates(
    records: Sequence[RangeStepRecord],
    *,
    failure_onset_step: int | None = None,
    targets: Iterable[float] = RHO_TARGETS,
    reference_steps: int = PROBE_BATCHES,
    summary_steps: int = TARGET_SUMMARY_STEPS,
    ema_decay: float = RANGE_EMA_DECAY,
) -> RangeCandidateSet:
    """Extract deterministic fast/middle/high targets from a range-test trace.

    A target window is the *next* eight steps after its first crossing.  Thus the
    optional extension is needed to make the main step-600 ``1e-2`` target stable.
    Duplicate role choices are replaced in deterministic priority order:
    ``fast``, then ``high``, then ``middle``.
    """

    reference_count = _positive_integer(reference_steps, "reference_steps")
    window_count = _positive_integer(summary_steps, "summary_steps")
    if failure_onset_step is not None:
        failure_step = _positive_integer(failure_onset_step, "failure_onset_step")
    else:
        failure_step = None
    if len(records) < reference_count:
        return RangeCandidateSet(
            "unresolved", "incomplete_reference_window", None, None, None, ()
        )
    for index, record in enumerate(records):
        if record.step != index + 1:
            raise _error(f"records[{index}].step", str(index + 1), record.step)
        _finite_number(record.rho, f"records[{index}].rho", nonnegative=True)
    safe_records = [
        record for record in records if failure_step is None or record.step < failure_step
    ]
    if len(safe_records) < reference_count:
        return RangeCandidateSet(
            "unresolved", "range_failed_during_reference", None, None, None, ()
        )
    try:
        smoothed = exponential_moving_average(
            (record.loss for record in safe_records), decay=ema_decay
        )
    except LRProtocolValidationError:
        return RangeCandidateSet(
            "unresolved", "non_finite_safe_loss", None, None, None, ()
        )
    reference_median = linear_quantile(smoothed[:reference_count], 0.5)
    target_values = tuple(
        _finite_number(value, f"targets[{index}]", positive=True)
        for index, value in enumerate(targets)
    )
    if not target_values or len(target_values) != len(set(target_values)):
        raise _error("targets", "unique positive finite values", target_values)
    if tuple(sorted(target_values)) != target_values:
        raise _error("targets", "strictly increasing values", target_values)

    stable: list[TargetWindowSummary] = []
    for target in target_values:
        crossing_index = next(
            (
                index
                for index, record in enumerate(safe_records)
                if record.rho >= target
            ),
            None,
        )
        if crossing_index is None:
            continue
        start = crossing_index + 1
        stop = start + window_count
        if stop > len(safe_records):
            continue
        window = safe_records[start:stop]
        if failure_step is not None and any(record.step >= failure_step for record in window):
            continue
        stable.append(
            TargetWindowSummary(
                rho_target=target,
                crossing_step=safe_records[crossing_index].step,
                window_start_step=window[0].step,
                window_end_step=window[-1].step,
                median_smoothed_loss=linear_quantile(smoothed[start:stop], 0.5),
            )
        )

    stable_tuple = tuple(stable)
    if len(stable) < 3:
        return RangeCandidateSet(
            "unresolved", "fewer_than_three_stable_targets", None, None, None, stable_tuple
        )
    fast = next(
        (
            summary
            for summary in stable
            if summary.median_smoothed_loss <= 0.90 * reference_median
        ),
        None,
    )
    if fast is None:
        return RangeCandidateSet(
            "unresolved", "no_decreasing_loss_region", None, None, None, stable_tuple
        )
    high = stable[-1]
    used = {fast.rho_target}
    if high.rho_target in used:
        high = _nearest_unused_target(stable, high.rho_target, used)
    used.add(high.rho_target)
    geometric_middle = math.sqrt(fast.rho_target * high.rho_target)
    middle = _nearest_unused_target(stable, geometric_middle, used)
    return RangeCandidateSet(
        "resolved", None, fast, middle, high, stable_tuple
    )


@dataclass(frozen=True)
class CandidateRunResult:
    candidate_id: str
    peak_learning_rate: float
    admissible: bool
    final_validation_loss: float | None
    final_validation_accuracy: float | None
    median_projection_efficiency: float | None
    inadmissible_reason: str | None = None
    selection_coordinate: float | None = None


@dataclass(frozen=True)
class FinalCandidateSelection:
    status: str
    reason: str | None
    selected: CandidateRunResult | None
    plateau: tuple[CandidateRunResult, ...]
    minimum_final_validation_loss: float | None


def _validate_candidate_result(value: CandidateRunResult, index: int) -> None:
    path = f"candidates[{index}]"
    if not isinstance(value, CandidateRunResult):
        raise _error(path, "a CandidateRunResult", value)
    if not isinstance(value.candidate_id, str) or not value.candidate_id:
        raise _error(f"{path}.candidate_id", "a non-empty string", value.candidate_id)
    _finite_number(
        value.peak_learning_rate, f"{path}.peak_learning_rate", positive=True
    )
    if value.selection_coordinate is not None:
        _finite_number(
            value.selection_coordinate,
            f"{path}.selection_coordinate",
            positive=True,
        )
    if not isinstance(value.admissible, bool):
        raise _error(f"{path}.admissible", "a boolean", value.admissible)
    if value.admissible:
        _finite_number(
            value.final_validation_loss,
            f"{path}.final_validation_loss",
            nonnegative=True,
        )
        accuracy = _finite_number(
            value.final_validation_accuracy,
            f"{path}.final_validation_accuracy",
        )
        if not 0.0 <= accuracy <= 1.0:
            raise _error(
                f"{path}.final_validation_accuracy", "a fraction in [0, 1]", accuracy
            )
        if value.median_projection_efficiency is not None:
            _finite_number(
                value.median_projection_efficiency,
                f"{path}.median_projection_efficiency",
                nonnegative=True,
            )
        if value.inadmissible_reason is not None:
            raise _error(
                f"{path}.inadmissible_reason", "null for an admissible candidate", value.inadmissible_reason
            )
    elif not isinstance(value.inadmissible_reason, str) or not value.inadmissible_reason:
        raise _error(
            f"{path}.inadmissible_reason",
            "a non-empty string for an inadmissible candidate",
            value.inadmissible_reason,
        )


def select_final_candidate(
    candidates: Sequence[CandidateRunResult],
    *,
    plateau_relative_tolerance: float = 0.02,
) -> FinalCandidateSelection:
    """Apply the final-loss plateau and deterministic tie-breaking rule."""

    tolerance = _finite_number(
        plateau_relative_tolerance,
        "plateau_relative_tolerance",
        nonnegative=True,
    )
    if not candidates:
        raise _error("candidates", "a non-empty sequence", candidates)
    for index, candidate in enumerate(candidates):
        _validate_candidate_result(candidate, index)
    ids = [candidate.candidate_id for candidate in candidates]
    rates = [candidate.peak_learning_rate for candidate in candidates]
    coordinates = [
        candidate.peak_learning_rate
        if candidate.selection_coordinate is None
        else candidate.selection_coordinate
        for candidate in candidates
    ]
    if len(ids) != len(set(ids)):
        raise _error("candidates", "unique candidate_id values", ids)
    if len(rates) != len(set(rates)):
        raise _error("candidates", "unique peak_learning_rate values", rates)
    if len(coordinates) != len(set(coordinates)):
        raise _error(
            "candidates", "unique selection-coordinate values", coordinates
        )

    admissible = [candidate for candidate in candidates if candidate.admissible]
    if not admissible:
        return FinalCandidateSelection(
            "unresolved", "no_admissible_candidates", None, (), None
        )
    minimum_loss = min(float(candidate.final_validation_loss) for candidate in admissible)
    plateau = tuple(
        candidate
        for candidate in admissible
        if float(candidate.final_validation_loss) <= minimum_loss * (1.0 + tolerance)
    )
    def coordinate(candidate: CandidateRunResult) -> float:
        return (
            candidate.peak_learning_rate
            if candidate.selection_coordinate is None
            else candidate.selection_coordinate
        )

    lower_rate = min(coordinate(candidate) for candidate in plateau)
    upper_rate = max(coordinate(candidate) for candidate in plateau)
    log_center = 0.5 * (math.log(lower_rate) + math.log(upper_rate))

    def selection_key(candidate: CandidateRunResult) -> tuple[float, float, float, float]:
        efficiency = (
            float(candidate.median_projection_efficiency)
            if candidate.median_projection_efficiency is not None
            else -math.inf
        )
        return (
            abs(math.log(coordinate(candidate)) - log_center),
            -float(candidate.final_validation_accuracy),
            -efficiency,
            coordinate(candidate),
        )

    ranked = [(selection_key(candidate), candidate) for candidate in plateau]
    minimum_distance = min(key[0] for key, _candidate in ranked)
    centered = [
        (key, candidate)
        for key, candidate in ranked
        if math.isclose(key[0], minimum_distance, rel_tol=1e-12, abs_tol=1e-15)
    ]
    selected = min(centered, key=lambda item: item[0][1:])[1]
    return FinalCandidateSelection(
        "frozen_seed0_screen", None, selected, plateau, minimum_loss
    )


def two_rho_upper_boundary_axes(
    plateau: Sequence[Mapping[str, Any]],
    *,
    rho_conv_values: Iterable[float],
    rho_dense_values: Iterable[float],
) -> tuple[str, ...]:
    """Return the upper target axes implicated by a two-rho plateau.

    The established v6 boundary rule treats a plateau as unbracketed when
    every point lies on the union of the upper Conv and Dense edges.  This
    helper preserves that rule while identifying the smallest useful set of
    axes to extend:

    * a plateau wholly on one edge implicates only that axis;
    * a plateau split across both edges implicates both axes;
    * a corner-only plateau implicates both axes; and
    * any plateau containing an interior point implicates neither axis.
    """

    if not isinstance(plateau, Sequence) or isinstance(plateau, (str, bytes)):
        raise _error("plateau", "a sequence of candidate mappings", plateau)
    grid = two_rho_candidate_grid(rho_conv_values, rho_dense_values)
    if not plateau:
        return ()
    expected_pairs = set(grid)
    pairs: list[tuple[float, float]] = []
    for candidate_index, candidate in enumerate(plateau):
        path = f"plateau[{candidate_index}]"
        if not isinstance(candidate, Mapping):
            raise _error(path, "a candidate mapping", candidate)
        pair = (
            _finite_number(
                candidate.get("rho_conv"), f"{path}.rho_conv", positive=True
            ),
            _finite_number(
                candidate.get("rho_dense"), f"{path}.rho_dense", positive=True
            ),
        )
        if pair not in expected_pairs:
            raise _error(path, f"a candidate on the configured grid {grid!r}", pair)
        pairs.append(pair)
    if len(pairs) != len(set(pairs)):
        raise _error("plateau", "unique rho target pairs", pairs)

    maximum_conv = max(pair[0] for pair in grid)
    maximum_dense = max(pair[1] for pair in grid)
    on_conv_edge = tuple(rho_conv == maximum_conv for rho_conv, _ in pairs)
    on_dense_edge = tuple(rho_dense == maximum_dense for _, rho_dense in pairs)
    if not all(
        conv_edge or dense_edge
        for conv_edge, dense_edge in zip(on_conv_edge, on_dense_edge)
    ):
        return ()

    all_conv = all(on_conv_edge)
    all_dense = all(on_dense_edge)
    if all_conv and not all_dense:
        return ("rho_conv",)
    if all_dense and not all_conv:
        return ("rho_dense",)
    return ("rho_conv", "rho_dense")


def select_two_rho_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rho_conv_values: Iterable[float] = V6_RHO_CONV_GRID,
    rho_dense_values: Iterable[float] = V6_RHO_DENSE_GRID,
    minimum_accuracy: float | None = None,
    plateau_relative_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Select one candidate from a complete direct two-rho grid.

    Safety-gate failures never enter the loss plateau.  When
    ``minimum_accuracy`` is not ``None``, final accuracies below its inclusive
    floor are also excluded.  The plateau tie breakers are, in order, higher
    accuracy, higher projection efficiency, lower maximum target, lower
    target sum, then lower Conv and Dense targets for total determinism.  A
    plateau confined to either upper grid edge is reported as unbracketed and
    deliberately returns no frozen selection.
    """

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise _error(
            "candidates", "a non-empty sequence of candidate mappings", candidates
        )
    if not candidates:
        raise _error(
            "candidates", "a non-empty sequence of candidate mappings", candidates
        )
    grid = two_rho_candidate_grid(rho_conv_values, rho_dense_values)
    expected_pairs = set(grid)
    accuracy_floor: float | None
    if minimum_accuracy is None:
        accuracy_floor = None
    else:
        accuracy_floor = _finite_number(minimum_accuracy, "minimum_accuracy")
        if not 0.0 <= accuracy_floor <= 1.0:
            raise _error(
                "minimum_accuracy", "a fraction in [0, 1]", minimum_accuracy
            )
    tolerance = _finite_number(
        plateau_relative_tolerance,
        "plateau_relative_tolerance",
        nonnegative=True,
    )

    by_pair: dict[tuple[float, float], dict[str, Any]] = {}
    candidate_ids: set[str] = set()
    for candidate_index, raw_candidate in enumerate(candidates):
        path = f"candidates[{candidate_index}]"
        if not isinstance(raw_candidate, Mapping):
            raise _error(path, "a candidate mapping", raw_candidate)
        rho_conv = _finite_number(
            raw_candidate.get("rho_conv"), f"{path}.rho_conv", positive=True
        )
        rho_dense = _finite_number(
            raw_candidate.get("rho_dense"), f"{path}.rho_dense", positive=True
        )
        pair = (rho_conv, rho_dense)
        if pair not in expected_pairs:
            raise _error(
                path,
                f"a candidate on the configured grid {grid!r}",
                pair,
            )
        if pair in by_pair:
            raise _error("candidates", "unique rho target pairs", pair)
        identifier_raw = raw_candidate.get(
            "candidate_id", f"rho_conv={rho_conv:g}--rho_dense={rho_dense:g}"
        )
        if not isinstance(identifier_raw, str) or not identifier_raw.strip():
            raise _error(f"{path}.candidate_id", "a non-empty string", identifier_raw)
        identifier = identifier_raw.strip()
        if identifier in candidate_ids:
            raise _error("candidates", "unique candidate_id values", identifier)
        candidate_ids.add(identifier)
        admissible = raw_candidate.get("admissible")
        if not isinstance(admissible, bool):
            raise _error(f"{path}.admissible", "a boolean", admissible)

        normalized: dict[str, Any] = {
            "candidate_id": identifier,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "admissible": admissible,
        }
        if admissible:
            loss = _finite_number(
                raw_candidate.get("final_validation_loss"),
                f"{path}.final_validation_loss",
                nonnegative=True,
            )
            accuracy = _finite_number(
                raw_candidate.get("final_validation_accuracy"),
                f"{path}.final_validation_accuracy",
            )
            if not 0.0 <= accuracy <= 1.0:
                raise _error(
                    f"{path}.final_validation_accuracy",
                    "a fraction in [0, 1]",
                    accuracy,
                )
            efficiency_raw = raw_candidate.get("median_projection_efficiency")
            efficiency = (
                None
                if efficiency_raw is None
                else _finite_number(
                    efficiency_raw,
                    f"{path}.median_projection_efficiency",
                    nonnegative=True,
                )
            )
            passes = accuracy_floor is None or accuracy >= accuracy_floor
            normalized.update(
                {
                    "final_validation_loss": loss,
                    "final_validation_accuracy": accuracy,
                    "median_projection_efficiency": efficiency,
                    "passes": passes,
                    "failure_reason": (
                        None
                        if passes
                        else f"final_validation_accuracy_below_{accuracy_floor:g}"
                    ),
                }
            )
        else:
            reason = raw_candidate.get("inadmissible_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise _error(
                    f"{path}.inadmissible_reason",
                    "a non-empty safety-gate reason",
                    reason,
                )
            normalized.update(
                {
                    "final_validation_loss": None,
                    "final_validation_accuracy": None,
                    "median_projection_efficiency": None,
                    "passes": False,
                    "failure_reason": reason.strip(),
                }
            )
        by_pair[pair] = normalized

    provided_pairs = set(by_pair)
    if provided_pairs != expected_pairs:
        raise _error(
            "candidates",
            "exactly one result for every configured rho target pair",
            {
                "missing": sorted(expected_pairs - provided_pairs),
                "extra": sorted(provided_pairs - expected_pairs),
            },
        )
    evaluated = [by_pair[pair] for pair in grid]
    passing = [candidate for candidate in evaluated if candidate["passes"]]
    if not passing:
        return {
            "status": "failed",
            "reason": "no_passing_candidate",
            "selected": None,
            "diagnostic_best": None,
            "selected_rho_conv": None,
            "selected_rho_dense": None,
            "minimum_final_validation_loss": None,
            "plateau": [],
            "evaluated_candidates": evaluated,
        }

    minimum_loss = min(
        float(candidate["final_validation_loss"]) for candidate in passing
    )
    plateau = [
        candidate
        for candidate in passing
        if float(candidate["final_validation_loss"])
        <= minimum_loss * (1.0 + tolerance)
    ]

    def selection_key(candidate: Mapping[str, Any]) -> tuple[float, ...]:
        efficiency = candidate["median_projection_efficiency"]
        rho_conv = float(candidate["rho_conv"])
        rho_dense = float(candidate["rho_dense"])
        return (
            -float(candidate["final_validation_accuracy"]),
            -(float(efficiency) if efficiency is not None else -math.inf),
            max(rho_conv, rho_dense),
            rho_conv + rho_dense,
            rho_conv,
            rho_dense,
        )

    diagnostic_best = min(plateau, key=selection_key)
    conv_grid_values = tuple(dict.fromkeys(pair[0] for pair in grid))
    dense_grid_values = tuple(dict.fromkeys(pair[1] for pair in grid))
    upper_boundary_axes = two_rho_upper_boundary_axes(
        plateau,
        rho_conv_values=conv_grid_values,
        rho_dense_values=dense_grid_values,
    )
    if upper_boundary_axes:
        return {
            "status": "unbracketed",
            "reason": "passing_plateau_confined_to_outer_boundary",
            "selected": None,
            "diagnostic_best": diagnostic_best,
            "selected_rho_conv": None,
            "selected_rho_dense": None,
            "minimum_final_validation_loss": minimum_loss,
            "plateau": plateau,
            "evaluated_candidates": evaluated,
        }
    return {
        "status": "selected",
        "reason": None,
        "selected": diagnostic_best,
        "diagnostic_best": diagnostic_best,
        "selected_rho_conv": float(diagnostic_best["rho_conv"]),
        "selected_rho_dense": float(diagnostic_best["rho_dense"]),
        "minimum_final_validation_loss": minimum_loss,
        "plateau": plateau,
        "evaluated_candidates": evaluated,
    }


def select_v6_conv2_baseline(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rho_conv_values: Iterable[float] = V6_RHO_CONV_GRID,
    rho_dense_values: Iterable[float] = V6_RHO_DENSE_GRID,
    minimum_accuracy: float = 0.90,
    plateau_relative_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Select the baseline Conv2 pair from a complete direct two-rho grid.

    Safety-gate failures and final accuracies below ``minimum_accuracy`` do
    not enter the loss plateau.  The plateau tie breakers are, in order,
    higher accuracy, higher projection efficiency, lower maximum target,
    lower target sum, then lower Conv and Dense targets for total
    determinism.  A plateau confined to either upper grid edge is reported as
    unbracketed and deliberately returns no frozen selection.
    """

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise _error(
            "candidates", "a non-empty sequence of candidate mappings", candidates
        )
    if not candidates:
        raise _error(
            "candidates", "a non-empty sequence of candidate mappings", candidates
        )
    conv_values = tuple(rho_conv_values)
    dense_values = tuple(rho_dense_values)
    two_rho_candidate_grid(conv_values, dense_values)
    accuracy_floor = _finite_number(minimum_accuracy, "minimum_accuracy")
    if not 0.0 <= accuracy_floor <= 1.0:
        raise _error(
            "minimum_accuracy", "a fraction in [0, 1]", minimum_accuracy
        )
    tolerance = _finite_number(
        plateau_relative_tolerance,
        "plateau_relative_tolerance",
        nonnegative=True,
    )
    selection = select_two_rho_candidates(
        candidates,
        rho_conv_values=conv_values,
        rho_dense_values=dense_values,
        minimum_accuracy=accuracy_floor,
        plateau_relative_tolerance=tolerance,
    )
    if selection["reason"] == "no_passing_candidate":
        selection["reason"] = "no_passing_baseline_candidate"
    return selection


def select_v5_architecture(
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimum_accuracy: float = 0.90,
    plateau_relative_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Select a v5 alpha and arm jointly across the three amplification rows.

    An alpha is viable only when baseline, ours, and legacy are all admissible
    and each reaches the final-epoch accuracy threshold.  Selection minimizes
    the worst row's final loss.  Values on the configured loss plateau use the
    frozen accuracy, projection-efficiency, and lower-alpha tie breakers.
    """

    accuracy_floor = _finite_number(minimum_accuracy, "minimum_accuracy")
    if not 0.0 <= accuracy_floor <= 1.0:
        raise _error("minimum_accuracy", "a fraction in [0, 1]", minimum_accuracy)
    tolerance = _finite_number(
        plateau_relative_tolerance,
        "plateau_relative_tolerance",
        nonnegative=True,
    )
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
        raise _error("candidates", "a non-empty sequence of candidate mappings", candidates)

    required_schemes = {"baseline", "ours", "legacy"}
    allowed_arms = {"strict_equal", "historical_profile"}
    evaluated: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, float]] = set()
    for candidate_index, raw_candidate in enumerate(candidates):
        path = f"candidates[{candidate_index}]"
        if not isinstance(raw_candidate, Mapping):
            raise _error(path, "a candidate mapping", raw_candidate)
        arm = raw_candidate.get("arm")
        if arm not in allowed_arms:
            raise _error(f"{path}.arm", f"one of {sorted(allowed_arms)!r}", arm)
        alpha = _finite_number(raw_candidate.get("alpha"), f"{path}.alpha", positive=True)
        key = str(arm), alpha
        if key in seen_keys:
            raise _error("candidates", "unique (arm, alpha) pairs", key)
        seen_keys.add(key)
        rows = raw_candidate.get("rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise _error(f"{path}.rows", "three row mappings", rows)
        normalized_rows: list[dict[str, Any]] = []
        schemes: list[str] = []
        failure_reasons: list[str] = []
        for row_index, raw_row in enumerate(rows):
            row_path = f"{path}.rows[{row_index}]"
            if not isinstance(raw_row, Mapping):
                raise _error(row_path, "a row mapping", raw_row)
            scheme = raw_row.get("scheme")
            if scheme not in required_schemes:
                raise _error(
                    f"{row_path}.scheme",
                    f"one of {sorted(required_schemes)!r}",
                    scheme,
                )
            schemes.append(str(scheme))
            admissible = raw_row.get("admissible")
            if not isinstance(admissible, bool):
                raise _error(f"{row_path}.admissible", "a boolean", admissible)
            normalized: dict[str, Any] = {
                "scheme": scheme,
                "admissible": admissible,
                "inadmissible_reason": raw_row.get("inadmissible_reason"),
            }
            if admissible:
                loss = _finite_number(
                    raw_row.get("final_validation_loss"),
                    f"{row_path}.final_validation_loss",
                    nonnegative=True,
                )
                accuracy = _finite_number(
                    raw_row.get("final_validation_accuracy"),
                    f"{row_path}.final_validation_accuracy",
                )
                if not 0.0 <= accuracy <= 1.0:
                    raise _error(
                        f"{row_path}.final_validation_accuracy",
                        "a fraction in [0, 1]",
                        accuracy,
                    )
                efficiency_raw = raw_row.get("median_projection_efficiency")
                efficiency = (
                    None
                    if efficiency_raw is None
                    else _finite_number(
                        efficiency_raw,
                        f"{row_path}.median_projection_efficiency",
                        nonnegative=True,
                    )
                )
                normalized.update(
                    {
                        "final_validation_loss": loss,
                        "final_validation_accuracy": accuracy,
                        "median_projection_efficiency": efficiency,
                    }
                )
                if accuracy < accuracy_floor:
                    failure_reasons.append(
                        f"{scheme}:final_validation_accuracy_below_{accuracy_floor:g}"
                    )
            else:
                reason = raw_row.get("inadmissible_reason")
                if not isinstance(reason, str) or not reason:
                    raise _error(
                        f"{row_path}.inadmissible_reason",
                        "a non-empty safety-gate reason",
                        reason,
                    )
                normalized.update(
                    {
                        "final_validation_loss": None,
                        "final_validation_accuracy": None,
                        "median_projection_efficiency": None,
                    }
                )
                failure_reasons.append(f"{scheme}:{reason}")
            normalized_rows.append(normalized)
        if len(schemes) != 3 or set(schemes) != required_schemes or len(set(schemes)) != 3:
            raise _error(
                f"{path}.rows",
                "exactly one baseline, ours, and legacy row",
                schemes,
            )
        viable = not failure_reasons
        passing_rows = normalized_rows if viable else []
        efficiencies = [
            float(row["median_projection_efficiency"])
            for row in passing_rows
            if row["median_projection_efficiency"] is not None
        ]
        evaluated.append(
            {
                "arm": arm,
                "alpha": alpha,
                "rows": normalized_rows,
                "viable": viable,
                "failure_reasons": failure_reasons,
                "worst_row_final_validation_loss": (
                    max(float(row["final_validation_loss"]) for row in passing_rows)
                    if viable
                    else None
                ),
                "worst_row_final_validation_accuracy": (
                    min(float(row["final_validation_accuracy"]) for row in passing_rows)
                    if viable
                    else None
                ),
                "median_projection_efficiency": (
                    linear_quantile(efficiencies, 0.5)
                    if viable and len(efficiencies) == 3
                    else None
                ),
            }
        )

    arm_selections: dict[str, dict[str, Any]] = {}
    for arm in ("strict_equal", "historical_profile"):
        arm_candidates = [item for item in evaluated if item["arm"] == arm]
        viable = [item for item in arm_candidates if item["viable"]]
        if not viable:
            arm_selections[arm] = {
                "status": "failed",
                "reason": "no_common_passing_alpha",
                "viable_alphas": [],
                "selected_alpha": None,
                "selected": None,
                "plateau_alphas": [],
                "candidates": arm_candidates,
            }
            continue
        minimum_loss = min(
            float(item["worst_row_final_validation_loss"]) for item in viable
        )
        plateau = [
            item
            for item in viable
            if float(item["worst_row_final_validation_loss"])
            <= minimum_loss * (1.0 + tolerance)
        ]

        def within_arm_key(item: Mapping[str, Any]) -> tuple[float, float, float]:
            efficiency = item["median_projection_efficiency"]
            return (
                -float(item["worst_row_final_validation_accuracy"]),
                -(float(efficiency) if efficiency is not None else -math.inf),
                float(item["alpha"]),
            )

        selected = min(plateau, key=within_arm_key)
        arm_selections[arm] = {
            "status": "selected",
            "reason": None,
            "viable_alphas": [float(item["alpha"]) for item in viable],
            "selected_alpha": float(selected["alpha"]),
            "selected": selected,
            "minimum_worst_row_final_validation_loss": minimum_loss,
            "plateau_alphas": [float(item["alpha"]) for item in plateau],
            "candidates": arm_candidates,
        }

    winners = [
        value
        for value in arm_selections.values()
        if value["status"] == "selected"
    ]
    if not winners:
        return {
            "status": "failed",
            "reason": "no_common_passing_alpha",
            "selected_arm": None,
            "selected_alpha": None,
            "selected": None,
            "arm_selections": arm_selections,
        }
    best_loss = min(
        float(value["selected"]["worst_row_final_validation_loss"])
        for value in winners
    )
    arm_plateau = [
        value
        for value in winners
        if float(value["selected"]["worst_row_final_validation_loss"])
        <= best_loss * (1.0 + tolerance)
    ]
    strict = arm_selections["strict_equal"]
    if strict in arm_plateau:
        winner = strict
        winner_arm = "strict_equal"
    else:
        winner = min(
            arm_plateau,
            key=lambda value: (
                -float(value["selected"]["worst_row_final_validation_accuracy"]),
                -(
                    float(value["selected"]["median_projection_efficiency"])
                    if value["selected"]["median_projection_efficiency"] is not None
                    else -math.inf
                ),
                float(value["selected"]["alpha"]),
            ),
        )
        winner_arm = str(winner["selected"]["arm"])
    return {
        "status": "selected",
        "reason": None,
        "selected_arm": winner_arm,
        "selected_alpha": float(winner["selected"]["alpha"]),
        "selected": winner["selected"],
        "arm_selections": arm_selections,
    }


__all__ = [
    "AFFINE_SEED",
    "BIAS_NORMALIZATION_EPSILON",
    "CANDIDATE_EPOCHS",
    "CANDIDATE_TOTAL_STEPS",
    "CANDIDATE_WARMUP_STEPS",
    "CandidateRunResult",
    "FROZEN_STUDY_ROWS",
    "FROZEN_STUDY_ROWS_BY_KEY",
    "FROZEN_STUDY_SETTINGS",
    "FinalCandidateSelection",
    "FrozenStudyRow",
    "LRProtocolValidationError",
    "LR_RUN_SCHEMA_VERSION",
    "LR_STUDY_SCHEMA_VERSION",
    "MAIN_RANGE_STEPS",
    "PROBE_BATCHES",
    "RANGE_EXTENSION_STEPS",
    "RHO_TARGETS",
    "RangeCandidateSet",
    "RangeFailure",
    "RangeGateResult",
    "RangeStepRecord",
    "RangeTensorRecord",
    "STEPS_PER_EPOCH",
    "TARGET_SUMMARY_STEPS",
    "TRAIN_BATCH_SIZE",
    "VALIDATION_BATCH_SIZE",
    "V6_RHO_CONV_GRID",
    "V6_RHO_DENSE_GRID",
    "analyze_range_stop_gates",
    "alpha_candidate_grid",
    "architecture_relative_learning_rates",
    "candidate_learning_rate_at_step",
    "candidate_learning_rate_schedule",
    "exponential_moving_average",
    "extract_range_candidates",
    "frozen_study_row",
    "geometric_schedule",
    "linear_quantile",
    "layerwise_learning_rate_report",
    "layerwise_parameter_groups",
    "layerwise_target_learning_rates",
    "normalized_bias_update",
    "parameter_relative_update_from_rms",
    "normalized_update_from_rms",
    "normalized_update_rho",
    "probe_rho_unit",
    "probe_parameter_relative_rho_unit",
    "probe_median_parameter_relative_units",
    "projection_efficiency",
    "range_learning_rate_schedule",
    "range_normalized_update_schedule",
    "rms",
    "select_final_candidate",
    "select_two_rho_candidates",
    "select_v5_architecture",
    "select_v6_conv2_baseline",
    "target_learning_rates",
    "two_rho_candidate_grid",
    "two_rho_learning_rates",
    "two_rho_upper_boundary_axes",
    "validate_frozen_study_rows",
    "validate_frozen_study_row",
    "validate_frozen_study_settings",
]
