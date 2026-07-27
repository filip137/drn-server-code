"""Immutable contract and pure selectors for the Conv3 perfect-diode T/K study.

This module is intentionally separate from the completed Conv1/Conv2
perfect-diode LR v1 study.  It defines an ordinary-MNIST, initialization-only
diagnostic whose output can be bound by a later LR study without changing any
historical study identity.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import sha256_json


TK_STUDY_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-study/v1"
TK_RUN_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-run/v1"
TK_MANIFEST_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-manifest/v1"
TK_ENTRY_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-tk-entry-completion/v1"
)
TK_SELECTION_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-selection/v1"
TK_PROTOCOL_ID = "conv3-perfectdiode-tk-ordinary-mnist-gain360-seed0-v1"
TK_STUDY_ID_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-study-id/v1"
TK_ENTRY_ID_SCHEMA_VERSION = "mnist-conv-perfectdiode-tk-entry-id/v1"

SCHEME_ORDER = ("baseline", "ours", "legacy")
SCHEME_CONTRACT = (
    ("baseline", "mnist_bp_amp_v1_c1", 1.0, 1.0),
    ("ours", "mnist_bp_amp_v4_c1", 4.0, 1.0),
    ("legacy", "mnist_bp_amp_v4_c0p25", 4.0, 0.25),
)
CONV3_PARAMETER_ORDER = (
    "ConvWeight_0",
    "Bias_0",
    "ConvWeight_1",
    "Bias_1",
    "ConvWeight_2",
    "Bias_2",
    "DenseWeight_0",
)
CONV3_CONV_WEIGHTS = ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2")
T_LAYER_CONTRACT = (
    ("hidden_0", "projected_kkt"),
    ("hidden_1", "projected_kkt"),
    ("hidden_2", "projected_kkt"),
    ("output", "raw"),
)

T_CORE_GRID = (4, 6, 8, 10, 16, 24, 32, 48, 64)
T_EXTENSION_GRID = (96, 128, 192, 256)
T_REFERENCE = 64
T_RESIDUAL_THRESHOLD = 1.0e-2
T_EXAMPLES = 1_024
T_BATCH_SIZE = 64

K_GRID = (4, 6, 8, 16, 32, 64)
K_REFERENCE = 64
K_BOUNDARY_SENTINEL = 128
K_EXAMPLES = 256
K_BATCH_SIZE = 32
K_NORM_DELTA_MAXIMUM = 0.10
K_ZERO_FRACTION_DELTA_MAXIMUM = 0.02
K_COSINE_MINIMUM = 0.90
GRADIENT_ZERO_EPSILON = 1.0e-12
REFERENCE_MEDIAN_GRADIENT_RMS_MINIMUM = 1.0e-12
REFERENCE_Q90_ZERO_FRACTION_MAXIMUM = 0.99
_FROZEN_CONFIG_SHA256 = (
    "bd5cc81c63ed4362aea3506a38e98b2f4a600ed6e75fbdc1d9b63ceb2b8a955e"
)


class PerfectDiodeTKValidationError(ValueError):
    """Raised when a T/K config or selection artifact violates the contract."""


def _error(path: str, expected: str, provided: Any) -> PerfectDiodeTKValidationError:
    return PerfectDiodeTKValidationError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "an object", value)
    return copy.deepcopy(dict(value))


def _sequence(value: Any, path: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _error(path, "an array", value)
    return list(value)


def _exact(value: Any, expected: Any, path: str) -> None:
    if value != expected:
        raise _error(path, f"exactly {expected!r}", value)


def _finite(value: Any, path: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise _error(path, "a finite number", value)
    result = float(value)
    if positive and result <= 0.0:
        raise _error(path, "a positive finite number", value)
    return result


def _strict_json_loads(text: str, source: Path) -> Any:
    def reject_constant(value: str) -> Any:
        raise _error(str(source), "strict JSON with finite numbers", value)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _error(str(source), "JSON objects with unique keys", key)
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except json.JSONDecodeError as exc:
        raise PerfectDiodeTKValidationError(
            f"Expected {source} to contain valid strict JSON. Provided error: {exc}."
        ) from exc


def _reject_non_finite(value: Any, path: str = "study") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _error(path, "a finite JSON value", value)
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def _validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(value, "study")
    _reject_non_finite(data)
    _exact(data.get("schema_version"), TK_STUDY_SCHEMA_VERSION, "study.schema_version")
    _exact(
        data.get("run_schema_version"),
        TK_RUN_SCHEMA_VERSION,
        "study.run_schema_version",
    )
    _exact(data.get("protocol_id"), TK_PROTOCOL_ID, "study.protocol_id")
    if not isinstance(data.get("name"), str) or not data["name"].strip():
        raise _error("study.name", "a non-empty string", data.get("name"))

    status = _mapping(data.get("status"), "study.status")
    _exact(
        status.get("study_role"),
        "ordinary_mnist_tk_diagnostic",
        "study.status.study_role",
    )
    _exact(
        status.get("medium_affine_paper_handoff"),
        "unresolved",
        "study.status.medium_affine_paper_handoff",
    )
    _exact(
        status.get("final_paper_training_authorized"),
        False,
        "study.status.final_paper_training_authorized",
    )

    dataset = _mapping(data.get("dataset"), "study.dataset")
    _exact(dataset.get("name"), "mnist", "study.dataset.name")
    _exact(dataset.get("variant"), "ordinary", "study.dataset.variant")
    _exact(dataset.get("input_shape"), [2, 28, 28], "study.dataset.input_shape")
    normalization = _mapping(
        dataset.get("normalization"), "study.dataset.normalization"
    )
    for key, expected in (("mean", 0.1307), ("std", 0.3081), ("scale", 0.3)):
        _exact(normalization.get(key), expected, f"study.dataset.normalization.{key}")
    affine = _mapping(dataset.get("affine"), "study.dataset.affine")
    _exact(affine.get("enabled"), False, "study.dataset.affine.enabled")
    train = _mapping(dataset.get("train"), "study.dataset.train")
    for key, expected in (
        ("source", "mnist_train"),
        ("size", 55_000),
        ("batch_size", 16),
        ("drop_last", False),
        ("shuffle_seed", 0),
    ):
        _exact(train.get(key), expected, f"study.dataset.train.{key}")
    validation = _mapping(dataset.get("validation"), "study.dataset.validation")
    for key, expected in (
        ("source", "mnist_train"),
        ("size", 5_000),
        ("samples_per_class", 500),
        ("split_seed", 0),
        ("batch_size", T_BATCH_SIZE),
    ):
        _exact(validation.get(key), expected, f"study.dataset.validation.{key}")
    official = _mapping(dataset.get("official_test"), "study.dataset.official_test")
    _exact(official.get("enabled"), False, "study.dataset.official_test.enabled")
    _exact(
        official.get("read_allowed"),
        False,
        "study.dataset.official_test.read_allowed",
    )

    model = _mapping(data.get("model"), "study.model")
    _exact(model.get("model_seed"), 0, "study.model.model_seed")
    architectures = _mapping(
        model.get("architectures"), "study.model.architectures"
    )
    _exact(tuple(architectures), ("conv3",), "study.model.architectures keys")
    conv3 = _mapping(architectures["conv3"], "study.model.architectures.conv3")
    for key, expected in (
        ("channels", [64, 128, 256]),
        ("kernel_sizes", [3, 3, 3]),
        ("strides", [2, 2, 1]),
        ("paddings", [1, 1, 1]),
        ("pooling", "none"),
        ("output_dim", 20),
        ("bounded_weight_order", list(CONV3_CONV_WEIGHTS) + ["DenseWeight_0"]),
        ("trainable_parameter_order", list(CONV3_PARAMETER_ORDER)),
    ):
        _exact(conv3.get(key), expected, f"study.model.architectures.conv3.{key}")
    _exact(model.get("non_linearity"), "perfect_diode", "study.model.non_linearity")
    _exact(
        model.get("quadratic_diode_param"),
        {},
        "study.model.quadratic_diode_param",
    )
    _exact(
        model.get("exponential_diode_param"),
        {},
        "study.model.exponential_diode_param",
    )
    _exact(model.get("hard_sigmoid"), {}, "study.model.hard_sigmoid")
    perfect = _mapping(model.get("perfect_diode"), "study.model.perfect_diode")
    _exact(perfect.get("clamp_epsilon"), 1.0e-8, "study.model.perfect_diode.clamp_epsilon")
    _exact(model.get("conductance_bounds"), [0.0, 100.0], "study.model.conductance_bounds")
    _exact(model.get("weight_initialization"), "kaiming_uniform", "study.model.weight_initialization")
    _exact(model.get("weight_gains"), 1.0, "study.model.weight_gains")
    _exact(model.get("paired_output_loss"), "squared_error_paired_20", "study.model.paired_output_loss")
    gain = _mapping(model.get("input_gain"), "study.model.input_gain")
    _exact(gain.get("conv3"), 360.0, "study.model.input_gain.conv3")
    _exact(
        gain.get("shared_across_schemes"),
        True,
        "study.model.input_gain.shared_across_schemes",
    )
    _exact(gain.get("recalibrate"), False, "study.model.input_gain.recalibrate")
    _exact(
        model.get("reset_global_name_counters_before_build"),
        True,
        "study.model.reset_global_name_counters_before_build",
    )

    solver = _mapping(data.get("solver"), "study.solver")
    _exact(solver.get("fixed_step_minimization"), True, "study.solver.fixed_step_minimization")
    _exact(solver.get("batch_state_policy"), "reset_each_batch", "study.solver.batch_state_policy")
    minimizer = _mapping(solver.get("minimizer"), "study.solver.minimizer")
    _exact(minimizer.get("adaptive_equilibrium"), False, "study.solver.minimizer.adaptive_equilibrium")

    optimizer = _mapping(data.get("optimizer"), "study.optimizer")
    for key, expected in (
        ("name", "SGD"),
        ("momentum", 0.0),
        ("weight_decay", 0.0),
        ("scalar_lr_for_all_trainable_parameters", True),
        ("bounded_weights_set_gates", True),
        ("biases_reported_separately", True),
    ):
        _exact(optimizer.get(key), expected, f"study.optimizer.{key}")

    rows = _sequence(data.get("rows"), "study.rows")
    if len(rows) != 3:
        raise _error("study.rows", "exactly three Conv3 scheme rows", rows)
    for index, (raw, expected) in enumerate(zip(rows, SCHEME_CONTRACT)):
        row = _mapping(raw, f"study.rows[{index}]")
        scheme, run_name, voltage_amp, current_amp = expected
        for key, expected_value in (
            ("row_id", f"conv3_{scheme}_{run_name.removeprefix('mnist_bp_amp_')}"),
            ("architecture", "conv3"),
            ("scheme", scheme),
            ("run_name", run_name),
            ("voltage_amp", voltage_amp),
            ("current_amp", current_amp),
            ("input_gain", 360.0),
        ):
            _exact(row.get(key), expected_value, f"study.rows[{index}].{key}")

    t_selection = _mapping(data.get("t_selection"), "study.t_selection")
    for key, expected in (
        ("cohort_source", "mnist_train_55000_subset"),
        ("cohort_examples", T_EXAMPLES),
        ("batch_size", T_BATCH_SIZE),
        ("candidate_grid", list(T_CORE_GRID)),
        ("reference_sentinel", T_REFERENCE),
        ("extension_grid", list(T_EXTENSION_GRID)),
        ("threshold", T_RESIDUAL_THRESHOLD),
        ("comparison", "strictly_less_than"),
        ("hidden_residual", "projected_kkt"),
        ("output_residual", "raw"),
    ):
        _exact(t_selection.get(key), expected, f"study.t_selection.{key}")

    k_selection = _mapping(data.get("k_selection"), "study.k_selection")
    for key, expected in (
        ("cohort_source", "mnist_train_55000_subset"),
        ("cohort_examples", K_EXAMPLES),
        ("batch_size", K_BATCH_SIZE),
        ("candidate_grid", list(K_GRID)),
        ("reference", K_REFERENCE),
        ("boundary_sentinel", K_BOUNDARY_SENTINEL),
        ("parameter_scope", list(CONV3_CONV_WEIGHTS)),
        ("relative_gradient_l2_norm_delta_maximum", K_NORM_DELTA_MAXIMUM),
        ("absolute_zero_fraction_delta_maximum", K_ZERO_FRACTION_DELTA_MAXIMUM),
        ("gradient_vector_cosine_minimum", K_COSINE_MINIMUM),
        ("reference_median_gradient_rms_minimum", REFERENCE_MEDIAN_GRADIENT_RMS_MINIMUM),
        ("reference_q90_zero_fraction_maximum", REFERENCE_Q90_ZERO_FRACTION_MAXIMUM),
    ):
        _exact(k_selection.get(key), expected, f"study.k_selection.{key}")
    return data


@dataclass(frozen=True)
class PerfectDiodeTKStudySpec:
    """Validated canonical Conv3 perfect-diode T/K config."""

    data: dict[str, Any]
    config_sha256: str
    study_id: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PerfectDiodeTKStudySpec":
        data = _validate_config(value)
        config_sha256 = sha256_json(data)
        if config_sha256 != _FROZEN_CONFIG_SHA256:
            raise PerfectDiodeTKValidationError(
                "Expected the canonical Conv3 perfect-diode T/K v1 config SHA-256 "
                f"to be {_FROZEN_CONFIG_SHA256!r}. Provided value: "
                f"{config_sha256!r}."
            )
        study_id = "tkstudy_" + sha256_json(
            {
                "schema_version": TK_STUDY_ID_SCHEMA_VERSION,
                "config_sha256": config_sha256,
            }
        )
        return cls(data=data, config_sha256=config_sha256, study_id=study_id)

    @classmethod
    def from_path(cls, path: str | Path) -> "PerfectDiodeTKStudySpec":
        source = Path(path).expanduser().resolve()
        try:
            value = _strict_json_loads(source.read_text(), source)
        except OSError as exc:
            raise PerfectDiodeTKValidationError(
                f"Expected the T/K config to be readable. Provided value: {source}."
            ) from exc
        if not isinstance(value, Mapping):
            raise _error(str(source), "a JSON object", value)
        return cls.from_dict(value)

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self.data["rows"]))


def tk_entry_id(study_id: str, row: Mapping[str, Any]) -> str:
    """Content-address one independently executable scheme row."""

    return "tkentry_" + sha256_json(
        {
            "schema_version": TK_ENTRY_ID_SCHEMA_VERSION,
            "study_id": study_id,
            "row_id": row["row_id"],
            "scheme": row["scheme"],
        }
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_iteration_mapping(
    measurements: Any,
) -> dict[int, Any] | None:
    if not isinstance(measurements, Mapping):
        return None
    result: dict[int, Any] = {}
    for key, value in measurements.items():
        if isinstance(key, bool) or not isinstance(key, int):
            return None
        if key in result:
            return None
        result[key] = value
    return result


def _linear_quantile_numbers(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _candidate_t_pass(
    record: Any,
    *,
    expected_iteration: int,
) -> tuple[bool, list[str], bool]:
    structural: list[str] = []
    gates: list[str] = []
    if not isinstance(record, Mapping):
        return False, ["missing_measurement"], False
    if record.get("iteration_count") != expected_iteration:
        structural.append("wrong_iteration_count")
    if record.get("num_examples") != T_EXAMPLES:
        structural.append("wrong_num_examples")
    if record.get("batch_size") != T_BATCH_SIZE:
        structural.append("wrong_batch_size")
    for field in (
        "cohort_source_indices_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
    ):
        if not _is_sha256(record.get(field)):
            structural.append(f"wrong_{field}")
    if record.get("official_test_read") is not False:
        structural.append("wrong_official_test_read")
    if record.get("fixed_step_minimization") is not True:
        structural.append("wrong_fixed_step_minimization")
    if record.get("batch_state_policy") != "reset_each_batch":
        structural.append("wrong_batch_state_policy")
    layers = record.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, (str, bytes)):
        return False, structural + ["missing_layers"], False
    roles: list[str] = []
    by_role: dict[str, Mapping[str, Any]] = {}
    for raw in layers:
        if isinstance(raw, Mapping) and isinstance(raw.get("role"), str):
            role = str(raw["role"])
            roles.append(role)
            if role not in by_role:
                by_role[role] = raw
    expected_roles = [role for role, _mode in T_LAYER_CONTRACT]
    if roles != expected_roles or len(by_role) != len(expected_roles):
        structural.append("layers_not_exact_unique_contract_order")
    for role, expected_mode in T_LAYER_CONTRACT:
        layer = by_role.get(role)
        if layer is None:
            structural.append(f"{role}:missing")
            continue
        if layer.get("selection_residual_mode") != expected_mode:
            structural.append(f"{role}:wrong_residual_mode")
            continue
        p90 = layer.get("selection_residual_p90")
        if (
            isinstance(p90, bool)
            or not isinstance(p90, (int, float))
            or not math.isfinite(float(p90))
        ):
            structural.append(f"{role}:nonfinite_or_missing_p90")
        for statistic_name in ("selection_residual", "raw_residual"):
            statistic = layer.get(statistic_name)
            if not isinstance(statistic, Mapping) or set(statistic) != {
                "mean",
                "median",
                "p90",
                "p99",
                "max",
            }:
                structural.append(f"{role}:{statistic_name}:wrong_fields")
                continue
            values: dict[str, float] = {}
            for field in ("mean", "median", "p90", "p99", "max"):
                raw = statistic.get(field)
                if (
                    isinstance(raw, bool)
                    or not isinstance(raw, (int, float))
                    or not math.isfinite(float(raw))
                    or float(raw) < 0.0
                ):
                    structural.append(
                        f"{role}:{statistic_name}:{field}:invalid"
                    )
                else:
                    values[field] = float(raw)
            if len(values) == 5 and not (
                values["median"]
                <= values["p90"]
                <= values["p99"]
                <= values["max"]
            ):
                structural.append(f"{role}:{statistic_name}:quantile_order")
            if (
                statistic_name == "selection_residual"
                and len(values) == 5
                and isinstance(p90, (int, float))
                and not isinstance(p90, bool)
                and not math.isclose(
                    float(p90), values["p90"], rel_tol=0.0, abs_tol=0.0
                )
            ):
                structural.append(f"{role}:selection_p90_inconsistent")
        if role != "output":
            occupancy = layer.get("clamped_occupancy")
            if not isinstance(occupancy, Mapping):
                structural.append(f"{role}:missing_clamped_occupancy")
            else:
                for field in (
                    "fraction",
                    "excitation_fraction",
                    "inhibition_fraction",
                ):
                    raw = occupancy.get(field)
                    if (
                        isinstance(raw, bool)
                        or not isinstance(raw, (int, float))
                        or not math.isfinite(float(raw))
                        or not 0.0 <= float(raw) <= 1.0
                    ):
                        structural.append(f"{role}:{field}:invalid")
        if (
            not structural
            and isinstance(p90, (int, float))
            and not float(p90) < T_RESIDUAL_THRESHOLD
        ):
            gates.append(f"{role}:threshold")
    reasons = structural + gates
    return not reasons, reasons, not structural


def select_t_measurements(
    measurements: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select T, request the fixed extension, or return a terminal failure."""

    normalized = _strict_iteration_mapping(measurements)
    if normalized is None:
        return {
            "status": "unresolved_t_incomplete_grid",
            "selected_t": None,
            "extension_used": False,
            "core_sentinel_passed": False,
            "extension_sentinel_passed": None,
            "candidates": [],
        }
    keys = set(normalized)
    core_keys = set(T_CORE_GRID)
    complete_keys = core_keys | set(T_EXTENSION_GRID)
    if not core_keys.issubset(keys) or not keys.issubset(complete_keys):
        return {
            "status": "unresolved_t_incomplete_grid",
            "selected_t": None,
            "extension_used": bool(keys & set(T_EXTENSION_GRID)),
            "core_sentinel_passed": False,
            "extension_sentinel_passed": None,
            "candidates": [],
        }
    extension_present = bool(keys & set(T_EXTENSION_GRID))
    if extension_present and keys != complete_keys:
        return {
            "status": "unresolved_t_incomplete_grid",
            "selected_t": None,
            "extension_used": True,
            "core_sentinel_passed": False,
            "extension_sentinel_passed": None,
            "candidates": [],
        }
    core = []
    for iteration in T_CORE_GRID:
        passed, reasons, complete = _candidate_t_pass(
            normalized.get(iteration), expected_iteration=iteration
        )
        core.append(
            {
                "iteration_count": iteration,
                "passed": passed,
                "complete": complete,
                "reasons": reasons,
            }
        )
    if not all(item["complete"] for item in core):
        return {
            "status": "unresolved_t_incomplete_measurement",
            "selected_t": None,
            "extension_used": extension_present,
            "core_sentinel_passed": False,
            "extension_sentinel_passed": None,
            "candidates": core,
        }
    context_fields = (
        "cohort_source_indices_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
    )
    context_values = {
        tuple(normalized[count].get(field) for field in context_fields)
        for count in T_CORE_GRID
    }
    if len(context_values) != 1:
        return {
            "status": "unresolved_t_mixed_context",
            "selected_t": None,
            "extension_used": extension_present,
            "core_sentinel_passed": False,
            "extension_sentinel_passed": None,
            "candidates": core,
        }
    core_sentinel_passed = core[-1]["passed"]
    core_passing = [item["iteration_count"] for item in core if item["passed"]]
    if core_passing and core_sentinel_passed and not extension_present:
        return {
            "status": "selected",
            "selected_t": min(core_passing),
            "extension_used": False,
            "core_sentinel_passed": True,
            "extension_sentinel_passed": None,
            "candidates": core,
        }
    if not extension_present:
        return {
            "status": "needs_extension",
            "selected_t": None,
            "extension_used": False,
            "core_sentinel_passed": bool(core_sentinel_passed),
            "extension_sentinel_passed": None,
            "candidates": core,
        }

    extension = []
    for iteration in T_EXTENSION_GRID:
        passed, reasons, complete = _candidate_t_pass(
            normalized.get(iteration), expected_iteration=iteration
        )
        extension.append(
            {
                "iteration_count": iteration,
                "passed": passed,
                "complete": complete,
                "reasons": reasons,
            }
        )
    if not all(item["complete"] for item in extension):
        return {
            "status": "unresolved_t_incomplete_measurement",
            "selected_t": None,
            "extension_used": True,
            "core_sentinel_passed": bool(core_sentinel_passed),
            "extension_sentinel_passed": False,
            "candidates": core + extension,
        }
    extension_context_values = {
        tuple(normalized[count].get(field) for field in context_fields)
        for count in T_EXTENSION_GRID
    }
    if extension_context_values != context_values:
        return {
            "status": "unresolved_t_mixed_context",
            "selected_t": None,
            "extension_used": True,
            "core_sentinel_passed": bool(core_sentinel_passed),
            "extension_sentinel_passed": False,
            "candidates": core + extension,
        }
    extension_sentinel_passed = extension[-1]["passed"]
    combined = core + extension
    if not extension_sentinel_passed:
        return {
            "status": "unresolved_t_extension_sentinel",
            "selected_t": None,
            "extension_used": True,
            "core_sentinel_passed": bool(core_sentinel_passed),
            "extension_sentinel_passed": False,
            "candidates": combined,
        }
    passing = [item["iteration_count"] for item in combined if item["passed"]]
    if not passing:
        return {
            "status": "unresolved_t_no_pass",
            "selected_t": None,
            "extension_used": True,
            "core_sentinel_passed": bool(core_sentinel_passed),
            "extension_sentinel_passed": True,
            "candidates": combined,
        }
    return {
        "status": "selected",
        "selected_t": min(passing),
        "extension_used": True,
        "core_sentinel_passed": bool(core_sentinel_passed),
        "extension_sentinel_passed": True,
        "candidates": combined,
    }


def _candidate_k_pass(
    record: Any,
    expected_parameters: Sequence[str],
) -> tuple[bool, list[str], bool]:
    if not isinstance(record, Mapping):
        return False, ["missing_measurement"], False
    structural: list[str] = []
    gates: list[str] = []
    if record.get("batch_count") != K_EXAMPLES // K_BATCH_SIZE:
        structural.append("wrong_batch_count")
    if record.get("cohort_examples") != K_EXAMPLES:
        structural.append("wrong_cohort_examples")
    if record.get("batch_size") != K_BATCH_SIZE:
        structural.append("wrong_batch_size")
    if record.get("official_test_read") is not False:
        structural.append("wrong_official_test_read")
    if record.get("fixed_step_minimization") is not True:
        structural.append("wrong_fixed_step_minimization")
    if record.get("batch_state_policy") != "reset_each_batch":
        structural.append("wrong_batch_state_policy")
    if (
        isinstance(record.get("selected_t"), bool)
        or not isinstance(record.get("selected_t"), int)
        or int(record["selected_t"]) < 1
    ):
        structural.append("wrong_selected_t")
    for field in (
        "cohort_source_indices_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
    ):
        if not _is_sha256(record.get(field)):
            structural.append(f"wrong_{field}")
    equilibrium_hashes = record.get("free_equilibrium_sha256_by_batch")
    if (
        not isinstance(equilibrium_hashes, Sequence)
        or isinstance(equilibrium_hashes, (str, bytes))
        or len(equilibrium_hashes) != K_EXAMPLES // K_BATCH_SIZE
        or any(not _is_sha256(value) for value in equilibrium_hashes)
    ):
        structural.append("wrong_free_equilibrium_hash_coverage")
    diagnostics = record.get("parameter_diagnostics")
    if not isinstance(diagnostics, Sequence) or isinstance(diagnostics, (str, bytes)):
        return False, structural + ["missing_parameter_diagnostics"], False
    names = [
        str(item.get("parameter"))
        for item in diagnostics
        if isinstance(item, Mapping) and isinstance(item.get("parameter"), str)
    ]
    if names != list(expected_parameters) or len(set(names)) != len(names):
        structural.append("parameters_not_exact_unique_contract_order")
    by_name: dict[str, Mapping[str, Any]] = {}
    for item in diagnostics:
        if isinstance(item, Mapping) and isinstance(item.get("parameter"), str):
            name = str(item["parameter"])
            if name not in by_name:
                by_name[name] = item
    for name in expected_parameters:
        item = by_name.get(name)
        if item is None:
            structural.append(f"{name}:missing")
            continue
        parameter_structural_start = len(structural)
        if item.get("batch_count") != K_EXAMPLES // K_BATCH_SIZE:
            structural.append(f"{name}:wrong_batch_count")
        batches = item.get("batches")
        if (
            not isinstance(batches, Sequence)
            or isinstance(batches, (str, bytes))
            or len(batches) != K_EXAMPLES // K_BATCH_SIZE
        ):
            structural.append(f"{name}:wrong_batch_diagnostics")
            continue
        batch_values: list[Mapping[str, Any]] = []
        for batch_index, batch in enumerate(batches):
            if not isinstance(batch, Mapping) or batch.get("batch_index") != batch_index:
                structural.append(f"{name}:batch_{batch_index}:wrong_identity")
                continue
            batch_values.append(batch)
            for field in (
                "gradient_l2",
                "reference_gradient_l2",
                "relative_gradient_l2_norm_delta",
                "gradient_vector_relative_error",
                "gradient_zero_fraction",
                "reference_gradient_zero_fraction",
                "absolute_zero_fraction_delta",
                "reference_gradient_rms",
            ):
                raw = batch.get(field)
                if (
                    isinstance(raw, bool)
                    or not isinstance(raw, (int, float))
                    or not math.isfinite(float(raw))
                    or float(raw) < 0.0
                ):
                    structural.append(
                        f"{name}:batch_{batch_index}:{field}:invalid"
                    )
            for field in (
                "gradient_zero_fraction",
                "reference_gradient_zero_fraction",
                "absolute_zero_fraction_delta",
                "relative_gradient_l2_norm_delta",
            ):
                raw = batch.get(field)
                if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                    if not 0.0 <= float(raw) <= 1.0:
                        structural.append(
                            f"{name}:batch_{batch_index}:{field}:out_of_range"
                        )
            cosine_value = batch.get("gradient_vector_cosine")
            if cosine_value is not None and (
                isinstance(cosine_value, bool)
                or not isinstance(cosine_value, (int, float))
                or not math.isfinite(float(cosine_value))
                or not -1.0 <= float(cosine_value) <= 1.0
            ):
                structural.append(
                    f"{name}:batch_{batch_index}:gradient_vector_cosine:invalid"
                )
        aggregate_nonnegative = (
            "gradient_l2",
            "reference_gradient_l2",
            "relative_gradient_l2_norm_delta",
            "gradient_vector_relative_error",
            "gradient_zero_fraction",
            "reference_gradient_zero_fraction",
            "absolute_zero_fraction_delta",
            "reference_median_gradient_rms",
            "reference_q90_zero_fraction",
            "initial_weight_rms",
            "reference_nominal_update_unit",
        )
        for field in aggregate_nonnegative:
            value = item.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                structural.append(f"{name}:{field}:invalid")
        for field in (
            "relative_gradient_l2_norm_delta",
            "gradient_zero_fraction",
            "reference_gradient_zero_fraction",
            "absolute_zero_fraction_delta",
            "reference_q90_zero_fraction",
        ):
            raw = item.get(field)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                if not 0.0 <= float(raw) <= 1.0:
                    structural.append(f"{name}:{field}:out_of_range")
        cosine = item.get("gradient_vector_cosine")
        if cosine is not None and (
            isinstance(cosine, bool)
            or not isinstance(cosine, (int, float))
            or not math.isfinite(float(cosine))
            or not -1.0 <= float(cosine) <= 1.0
        ):
            structural.append(f"{name}:gradient_vector_cosine:invalid")
        if (
            len(batch_values) == K_EXAMPLES // K_BATCH_SIZE
            and len(structural) == parameter_structural_start
        ):
            mean_gradient = sum(
                float(batch["gradient_l2"]) for batch in batch_values
            ) / len(batch_values)
            mean_reference = sum(
                float(batch["reference_gradient_l2"]) for batch in batch_values
            ) / len(batch_values)
            expected_norm_delta = abs(mean_gradient - mean_reference) / max(
                mean_gradient, mean_reference, 1.0e-30
            )
            mean_zero = sum(
                float(batch["gradient_zero_fraction"]) for batch in batch_values
            ) / len(batch_values)
            mean_reference_zero = sum(
                float(batch["reference_gradient_zero_fraction"])
                for batch in batch_values
            ) / len(batch_values)
            expected_zero_delta = abs(mean_zero - mean_reference_zero)
            batch_cosines = [
                batch.get("gradient_vector_cosine") for batch in batch_values
            ]
            expected_cosine = (
                sum(float(value) for value in batch_cosines) / len(batch_cosines)
                if all(value is not None for value in batch_cosines)
                else None
            )
            expected_values = {
                "gradient_l2": mean_gradient,
                "reference_gradient_l2": mean_reference,
                "relative_gradient_l2_norm_delta": expected_norm_delta,
                "gradient_vector_relative_error": sum(
                    float(batch["gradient_vector_relative_error"])
                    for batch in batch_values
                )
                / len(batch_values),
                "gradient_zero_fraction": mean_zero,
                "reference_gradient_zero_fraction": mean_reference_zero,
                "absolute_zero_fraction_delta": expected_zero_delta,
                "gradient_vector_cosine": expected_cosine,
            }
            for field, expected in expected_values.items():
                observed = item.get(field)
                if expected is None:
                    if observed is not None:
                        structural.append(f"{name}:{field}:aggregate_mismatch")
                elif not isinstance(observed, (int, float)) or isinstance(
                    observed, bool
                ) or not math.isclose(
                    float(observed), expected, rel_tol=1.0e-12, abs_tol=1.0e-15
                ):
                    structural.append(f"{name}:{field}:aggregate_mismatch")
            expected_median_rms = _linear_quantile_numbers(
                [
                    float(batch["reference_gradient_rms"])
                    for batch in batch_values
                ],
                0.5,
            )
            expected_q90_zero = _linear_quantile_numbers(
                [
                    float(batch["reference_gradient_zero_fraction"])
                    for batch in batch_values
                ],
                0.9,
            )
            initial_rms_value = item.get("initial_weight_rms")
            expected_update_unit = (
                expected_median_rms / float(initial_rms_value)
                if isinstance(initial_rms_value, (int, float))
                and not isinstance(initial_rms_value, bool)
                and float(initial_rms_value) > 0.0
                else None
            )
            for field, expected in (
                ("reference_median_gradient_rms", expected_median_rms),
                ("reference_q90_zero_fraction", expected_q90_zero),
                ("reference_nominal_update_unit", expected_update_unit),
            ):
                observed = item.get(field)
                if expected is None or not isinstance(
                    observed, (int, float)
                ) or isinstance(observed, bool) or not math.isclose(
                    float(observed), expected, rel_tol=1.0e-12, abs_tol=1.0e-15
                ):
                    structural.append(f"{name}:{field}:aggregate_mismatch")
        viable = (
            isinstance(item.get("reference_median_gradient_rms"), (int, float))
            and not isinstance(item.get("reference_median_gradient_rms"), bool)
            and float(item["reference_median_gradient_rms"])
            > REFERENCE_MEDIAN_GRADIENT_RMS_MINIMUM
            and isinstance(item.get("reference_q90_zero_fraction"), (int, float))
            and not isinstance(item.get("reference_q90_zero_fraction"), bool)
            and float(item["reference_q90_zero_fraction"])
            < REFERENCE_Q90_ZERO_FRACTION_MAXIMUM
            and isinstance(item.get("reference_nominal_update_unit"), (int, float))
            and not isinstance(item.get("reference_nominal_update_unit"), bool)
            and math.isfinite(float(item["reference_nominal_update_unit"]))
            and float(item["reference_nominal_update_unit"]) > 0.0
        )
        if item.get("reference_viable") is not viable:
            structural.append(f"{name}:reference_viability_mismatch")
        if not viable:
            gates.append(f"{name}:reference_not_viable")
        norm_delta = item.get("relative_gradient_l2_norm_delta")
        zero_delta = item.get("absolute_zero_fraction_delta")
        if isinstance(norm_delta, (int, float)) and not isinstance(norm_delta, bool):
            if float(norm_delta) > K_NORM_DELTA_MAXIMUM:
                gates.append(f"{name}:relative_gradient_l2_norm_delta:threshold")
        if isinstance(zero_delta, (int, float)) and not isinstance(zero_delta, bool):
            if float(zero_delta) > K_ZERO_FRACTION_DELTA_MAXIMUM:
                gates.append(f"{name}:absolute_zero_fraction_delta:threshold")
        if cosine is None or (
            isinstance(cosine, (int, float))
            and not isinstance(cosine, bool)
            and float(cosine) < K_COSINE_MINIMUM
        ):
            gates.append(f"{name}:gradient_vector_cosine:threshold")
        expected_passed = (
            viable
            and isinstance(norm_delta, (int, float))
            and float(norm_delta) <= K_NORM_DELTA_MAXIMUM
            and isinstance(zero_delta, (int, float))
            and float(zero_delta) <= K_ZERO_FRACTION_DELTA_MAXIMUM
            and isinstance(cosine, (int, float))
            and not isinstance(cosine, bool)
            and float(cosine) >= K_COSINE_MINIMUM
        )
        if item.get("passed") is not expected_passed:
            structural.append(f"{name}:passed_mismatch")
    computed_reference_viable = all(
        isinstance(item, Mapping) and item.get("reference_viable") is True
        for item in diagnostics
    )
    if record.get("reference_viable") is not computed_reference_viable:
        structural.append("reference_viability_mismatch")
    computed_passed = (
        computed_reference_viable
        and all(
            isinstance(item, Mapping) and item.get("passed") is True
            for item in diagnostics
        )
    )
    if record.get("passed") is not computed_passed:
        structural.append("passed_mismatch")
    reasons = structural + gates
    return not reasons, reasons, not structural


def select_k_measurements(
    measurements: Mapping[int, Mapping[str, Any]],
    *,
    k128_sentinel: Mapping[str, Any] | None = None,
    expected_parameters: Sequence[str] = CONV3_CONV_WEIGHTS,
) -> dict[str, Any]:
    """Select K against K=64, requiring K=128 at the K=64 boundary."""

    if tuple(expected_parameters) != CONV3_CONV_WEIGHTS:
        raise _error(
            "expected_parameters",
            f"exactly {CONV3_CONV_WEIGHTS!r}",
            tuple(expected_parameters),
        )
    normalized = _strict_iteration_mapping(measurements)
    if normalized is None:
        return {
            "status": "unresolved_k_incomplete_grid",
            "selected_k": None,
            "k128_sentinel_used": False,
            "candidates": [],
        }
    if set(normalized) != set(K_GRID):
        return {
            "status": "unresolved_k_incomplete_grid",
            "selected_k": None,
            "k128_sentinel_used": False,
            "candidates": [],
        }
    reference = normalized.get(K_REFERENCE)
    if not isinstance(reference, Mapping) or reference.get("reference_viable") is not True:
        return {
            "status": "unresolved_k_reference",
            "selected_k": None,
            "k128_sentinel_used": False,
            "candidates": [],
        }
    candidates = []
    for count in K_GRID:
        record = normalized.get(count)
        if isinstance(record, Mapping) and (
            record.get("candidate_k") != count
            or record.get("reference_k") != K_REFERENCE
        ):
            passed, reasons, complete = (
                False,
                ["wrong_candidate_or_reference_k"],
                False,
            )
        else:
            passed, reasons, complete = _candidate_k_pass(
            normalized.get(count), expected_parameters
            )
        candidates.append(
            {
                "iteration_count": count,
                "passed": passed,
                "complete": complete,
                "reasons": reasons,
            }
        )
    if not all(item["complete"] for item in candidates):
        return {
            "status": "unresolved_k_incomplete_measurement",
            "selected_k": None,
            "k128_sentinel_used": False,
            "candidates": candidates,
        }
    context_fields = (
        "selected_t",
        "cohort_source_indices_sha256",
        "initialization_checkpoint_sha256",
        "initialization_tensor_sha256",
        "free_equilibrium_sha256_by_batch",
    )
    context_values = {
        (
            record.get("selected_t"),
            record.get("cohort_source_indices_sha256"),
            record.get("initialization_checkpoint_sha256"),
            record.get("initialization_tensor_sha256"),
            tuple(record.get("free_equilibrium_sha256_by_batch", ())),
        )
        for record in normalized.values()
    }
    if len(context_values) != 1:
        return {
            "status": "unresolved_k_mixed_context",
            "selected_k": None,
            "k128_sentinel_used": False,
            "candidates": candidates,
        }
    passing = [item["iteration_count"] for item in candidates if item["passed"]]
    if not passing:
        return {
            "status": "unresolved_k_no_pass",
            "selected_k": None,
            "k128_sentinel_used": False,
            "candidates": candidates,
        }
    selected = min(passing)
    if selected != K_REFERENCE:
        return {
            "status": "selected",
            "selected_k": selected,
            "k128_sentinel_used": False,
            "candidates": candidates,
        }
    if k128_sentinel is None:
        return {
            "status": "needs_k128_sentinel",
            "selected_k": None,
            "k128_sentinel_used": False,
            "candidates": candidates,
        }
    if (
        k128_sentinel.get("candidate_k") != K_REFERENCE
        or k128_sentinel.get("reference_k") != K_BOUNDARY_SENTINEL
    ):
        sentinel_passed, reasons, sentinel_complete = (
            False,
            ["wrong_candidate_or_reference_k"],
            False,
        )
    else:
        sentinel_passed, reasons, sentinel_complete = _candidate_k_pass(
            k128_sentinel, expected_parameters
        )
    if not sentinel_complete:
        return {
            "status": "unresolved_k128_incomplete_measurement",
            "selected_k": None,
            "k128_sentinel_used": True,
            "k128_sentinel_passed": False,
            "k128_sentinel_reasons": reasons,
            "candidates": candidates,
        }
    sentinel_context = (
        k128_sentinel.get("selected_t"),
        k128_sentinel.get("cohort_source_indices_sha256"),
        k128_sentinel.get("initialization_checkpoint_sha256"),
        k128_sentinel.get("initialization_tensor_sha256"),
        tuple(k128_sentinel.get("free_equilibrium_sha256_by_batch", ())),
    )
    if sentinel_context != next(iter(context_values)):
        return {
            "status": "unresolved_k128_mixed_context",
            "selected_k": None,
            "k128_sentinel_used": True,
            "k128_sentinel_passed": False,
            "k128_sentinel_reasons": ["mixed_context"],
            "candidates": candidates,
        }
    return {
        "status": "selected" if sentinel_passed else "unresolved_k128_sentinel",
        "selected_k": K_REFERENCE if sentinel_passed else None,
        "k128_sentinel_used": True,
        "k128_sentinel_passed": sentinel_passed,
        "k128_sentinel_reasons": reasons,
        "candidates": candidates,
    }


__all__ = [
    "CONV3_CONV_WEIGHTS",
    "CONV3_PARAMETER_ORDER",
    "GRADIENT_ZERO_EPSILON",
    "K_BATCH_SIZE",
    "K_BOUNDARY_SENTINEL",
    "K_COSINE_MINIMUM",
    "K_EXAMPLES",
    "K_GRID",
    "K_NORM_DELTA_MAXIMUM",
    "K_REFERENCE",
    "K_ZERO_FRACTION_DELTA_MAXIMUM",
    "PerfectDiodeTKStudySpec",
    "PerfectDiodeTKValidationError",
    "REFERENCE_MEDIAN_GRADIENT_RMS_MINIMUM",
    "REFERENCE_Q90_ZERO_FRACTION_MAXIMUM",
    "SCHEME_ORDER",
    "TK_ENTRY_COMPLETION_SCHEMA_VERSION",
    "TK_MANIFEST_SCHEMA_VERSION",
    "TK_PROTOCOL_ID",
    "TK_RUN_SCHEMA_VERSION",
    "TK_SELECTION_SCHEMA_VERSION",
    "TK_STUDY_SCHEMA_VERSION",
    "T_BATCH_SIZE",
    "T_CORE_GRID",
    "T_EXAMPLES",
    "T_EXTENSION_GRID",
    "T_LAYER_CONTRACT",
    "T_REFERENCE",
    "T_RESIDUAL_THRESHOLD",
    "select_k_measurements",
    "select_t_measurements",
    "tk_entry_id",
]
