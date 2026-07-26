"""Immutable contract and dependency-light logic for the perfect-diode LR screen.

The perfect-diode study deliberately has its own schema and identity.  Nothing
in this module changes the historical hard-sigmoid LR v1--v7 contracts.  The
runtime and orchestration layers consume the frozen config through
``PerfectDiodeHparamStudySpec`` and use the pure helpers below for adaptive
probes, restarted center canaries, grid construction, and selection.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .identity import sha256_json


PD_STUDY_SCHEMA_VERSION = "mnist-conv-perfectdiode-hparam-study/v1"
PD_RUN_SCHEMA_VERSION = "mnist-conv-perfectdiode-hparam-run/v1"
PD_STUDY_ID_SCHEMA = "mnist-conv-perfectdiode-hparam-study-id/v1"
PD_PROTOCOL_ID = (
    "conv-perfectdiode-conv12-two-rho-sgd-adam-ordinary-mnist-bs16-v1"
)

PUBLIC_STAGE_SEQUENCE = (
    "audit",
    "assets",
    "fixed_tk_gradient_security",
    "optimizer_probe",
    "rho_canary_core",
    "rho_core_candidates",
    "select_core",
    "rho_canary_extension",
    "rho_extension_candidates",
    "select_final",
    "long_confirm",
    "finalize",
)

MODEL_SEED = 0
TRAIN_SHUFFLE_SEED = 0
TRAIN_SIZE = 55_000
VALIDATION_SIZE = 5_000
TRAIN_BATCH_SIZE = 16
VALIDATION_BATCH_SIZE = 64
STEPS_PER_EPOCH = 3_438
CANDIDATE_EPOCHS = 3
CANDIDATE_TOTAL_STEPS = 10_314
CANARY_STEPS = 640
CONV1_CONFIRM_EPOCHS = 10
CONV1_CONFIRM_STEPS = 34_380
CONV2_CONFIRM_EPOCHS = 30
CONV2_CONFIRM_STEPS = 103_140

PROBE_BATCH_COUNTS = (32, 64, 128)
PROBE_SPLIT_HALF_RELATIVE_TOLERANCE = 0.10
INITIAL_RHO_CENTER = (3e-3, 1e-2)
GRID_FACTOR = 3.0
MAX_CENTER_ATTEMPTS = 6
MINIMUM_FINAL_VALIDATION_ACCURACY = 0.90
PLATEAU_RELATIVE_TOLERANCE = 0.02
MAXIMUM_CELLS_PER_SURFACE = 16


class PerfectDiodeHparamValidationError(ValueError):
    """Raised when data drift from the immutable perfect-diode contract."""


def _error(path: str, expected: str, provided: Any) -> PerfectDiodeHparamValidationError:
    return PerfectDiodeHparamValidationError(
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


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _error(path, "an integer >= 1", value)
    return value


@dataclass(frozen=True)
class FrozenPerfectDiodeRow:
    """One fixed Conv1/Conv2 amplification row."""

    row_id: str
    architecture: str
    scheme: str
    run_name: str
    voltage_amp: float
    current_amp: float
    input_gain: float
    inference_iterations: int
    training_iterations: int
    reference_inference_iterations: int = 64
    reference_training_iterations: int = 64

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "architecture": self.architecture,
            "scheme": self.scheme,
            "run_name": self.run_name,
            "voltage_amp": self.voltage_amp,
            "current_amp": self.current_amp,
            "input_gain": self.input_gain,
            "inference_iterations": self.inference_iterations,
            "training_iterations": self.training_iterations,
            "reference_inference_iterations": self.reference_inference_iterations,
            "reference_training_iterations": self.reference_training_iterations,
        }


_SCHEMES = (
    ("baseline", "mnist_bp_amp_v1_c1", 1.0, 1.0),
    ("ours", "mnist_bp_amp_v4_c1", 4.0, 1.0),
    ("legacy", "mnist_bp_amp_v4_c0p25", 4.0, 0.25),
)

FROZEN_ROW_OBJECTS = tuple(
    FrozenPerfectDiodeRow(
        row_id=f"{architecture}_{scheme}_{run_name.removeprefix('mnist_bp_amp_')}",
        architecture=architecture,
        scheme=scheme,
        run_name=run_name,
        voltage_amp=voltage_amp,
        current_amp=current_amp,
        input_gain=input_gain,
        inference_iterations=iterations,
        training_iterations=iterations,
    )
    for architecture, input_gain, iterations in (
        ("conv1", 40.0, 4),
        ("conv2", 100.0, 6),
    )
    for scheme, run_name, voltage_amp, current_amp in _SCHEMES
)

FROZEN_ROWS = tuple(row.as_dict() for row in FROZEN_ROW_OBJECTS)
SURFACES = tuple(
    (row["architecture"], row["scheme"], optimizer)
    for row in FROZEN_ROWS
    for optimizer in ("sgd", "adam")
)


@dataclass(frozen=True)
class ProbeStabilityDecision:
    """Decision after one optimizer-specific probe length."""

    status: str
    used_batches: int
    next_batches: int | None
    unstable_parameters: tuple[str, ...]


def probe_stability_decision(
    split_half_relative_differences: Mapping[str, float],
    *,
    used_batches: int,
    tolerance: float = PROBE_SPLIT_HALF_RELATIVE_TOLERANCE,
) -> ProbeStabilityDecision:
    """Return ``stable``, ``extend``, or ``unresolved`` for an adaptive probe.

    A parameter is unstable only when its difference is *strictly* greater
    than 10%; exactly 10% passes.  The probe can grow 32 -> 64 -> 128 and no
    farther.
    """

    if used_batches not in PROBE_BATCH_COUNTS:
        raise _error(
            "used_batches",
            f"one of {PROBE_BATCH_COUNTS!r}",
            used_batches,
        )
    threshold = _finite_number(tolerance, "tolerance", nonnegative=True)
    if not isinstance(split_half_relative_differences, Mapping):
        raise _error(
            "split_half_relative_differences",
            "a non-empty parameter-to-difference mapping",
            split_half_relative_differences,
        )
    if not split_half_relative_differences:
        raise _error(
            "split_half_relative_differences",
            "a non-empty parameter-to-difference mapping",
            split_half_relative_differences,
        )
    unstable: list[str] = []
    for name, raw_value in split_half_relative_differences.items():
        if not isinstance(name, str) or not name:
            raise _error(
                "split_half_relative_differences key",
                "a non-empty parameter name",
                name,
            )
        value = _finite_number(
            raw_value,
            f"split_half_relative_differences[{name!r}]",
            nonnegative=True,
        )
        if value > threshold:
            unstable.append(name)
    unstable_parameters = tuple(sorted(unstable))
    if not unstable_parameters:
        return ProbeStabilityDecision("stable", used_batches, None, ())
    index = PROBE_BATCH_COUNTS.index(used_batches)
    if index + 1 < len(PROBE_BATCH_COUNTS):
        return ProbeStabilityDecision(
            "extend",
            used_batches,
            PROBE_BATCH_COUNTS[index + 1],
            unstable_parameters,
        )
    return ProbeStabilityDecision(
        "unresolved",
        used_batches,
        None,
        unstable_parameters,
    )


def safe_center(attempt: int) -> tuple[float, float]:
    """Return the center target for one of the six restarted canary attempts."""

    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise _error("attempt", f"an integer in [0, {MAX_CENTER_ATTEMPTS - 1}]", attempt)
    if not 0 <= attempt < MAX_CENTER_ATTEMPTS:
        raise _error("attempt", f"an integer in [0, {MAX_CENTER_ATTEMPTS - 1}]", attempt)
    scale = GRID_FACTOR**attempt
    return INITIAL_RHO_CENTER[0] / scale, INITIAL_RHO_CENTER[1] / scale


def _axis_values(values: Iterable[float], path: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise _error(path, "a sequence of unique positive finite numbers", values)
    try:
        normalized = tuple(
            _finite_number(value, f"{path}[{index}]", positive=True)
            for index, value in enumerate(values)
        )
    except TypeError as exc:
        raise _error(path, "a sequence of unique positive finite numbers", values) from exc
    if not normalized or len(normalized) != len(set(normalized)):
        raise _error(path, "a sequence of unique positive finite numbers", normalized)
    if normalized != tuple(sorted(normalized)):
        raise _error(path, "a strictly increasing sequence", normalized)
    return normalized


def candidate_grid(
    rho_conv_values: Iterable[float],
    rho_dense_values: Iterable[float],
) -> tuple[tuple[float, float], ...]:
    """Return the deterministic Conv-major Cartesian target grid."""

    conv = _axis_values(rho_conv_values, "rho_conv_values")
    dense = _axis_values(rho_dense_values, "rho_dense_values")
    return tuple((rho_conv, rho_dense) for rho_conv in conv for rho_dense in dense)


def core_grid(
    center_rho_conv: float,
    center_rho_dense: float,
) -> tuple[tuple[float, float], ...]:
    """Return the factor-three 3x3 core around a safety-clean center."""

    center_conv = _finite_number(
        center_rho_conv, "center_rho_conv", positive=True
    )
    center_dense = _finite_number(
        center_rho_dense, "center_rho_dense", positive=True
    )
    return candidate_grid(
        (center_conv / GRID_FACTOR, center_conv, center_conv * GRID_FACTOR),
        (center_dense / GRID_FACTOR, center_dense, center_dense * GRID_FACTOR),
    )


@dataclass(frozen=True)
class BoundaryExpansionPlan:
    """A one-wave factor-three expansion of the implicated outer axes."""

    directions: tuple[tuple[str, str], ...]
    rho_conv_values: tuple[float, ...]
    rho_dense_values: tuple[float, ...]
    new_cells: tuple[tuple[float, float], ...]

    @property
    def required(self) -> bool:
        return bool(self.directions)


def _candidate_pair(
    candidate: Mapping[str, Any],
    *,
    path: str,
) -> tuple[float, float]:
    if not isinstance(candidate, Mapping):
        raise _error(path, "a candidate mapping", candidate)
    return (
        _finite_number(candidate.get("rho_conv"), f"{path}.rho_conv", positive=True),
        _finite_number(candidate.get("rho_dense"), f"{path}.rho_dense", positive=True),
    )


def _boundary_directions(
    plateau: Sequence[Mapping[str, Any]],
    *,
    rho_conv_values: tuple[float, ...],
    rho_dense_values: tuple[float, ...],
) -> tuple[tuple[str, str], ...]:
    if not plateau:
        return ()
    pairs = tuple(
        _candidate_pair(candidate, path=f"plateau[{index}]")
        for index, candidate in enumerate(plateau)
    )
    expected_pairs = set(candidate_grid(rho_conv_values, rho_dense_values))
    if any(pair not in expected_pairs for pair in pairs):
        raise _error("plateau", "candidates on the configured grid", pairs)
    if len(pairs) != len(set(pairs)):
        raise _error("plateau", "unique rho target pairs", pairs)

    edges = (
        ("rho_conv", "lower", lambda pair: pair[0] == rho_conv_values[0]),
        ("rho_conv", "upper", lambda pair: pair[0] == rho_conv_values[-1]),
        ("rho_dense", "lower", lambda pair: pair[1] == rho_dense_values[0]),
        ("rho_dense", "upper", lambda pair: pair[1] == rho_dense_values[-1]),
    )
    covering_single = [
        (axis, direction)
        for axis, direction, predicate in edges
        if all(predicate(pair) for pair in pairs)
    ]
    if covering_single:
        # A plateau at one exact corner implicates both incident axes, matching
        # the established two-rho corner behavior.
        if len(set(pairs)) == 1:
            pair = pairs[0]
            conv_direction = (
                "lower"
                if pair[0] == rho_conv_values[0]
                else "upper"
                if pair[0] == rho_conv_values[-1]
                else None
            )
            dense_direction = (
                "lower"
                if pair[1] == rho_dense_values[0]
                else "upper"
                if pair[1] == rho_dense_values[-1]
                else None
            )
            if conv_direction is not None and dense_direction is not None:
                return (
                    ("rho_conv", conv_direction),
                    ("rho_dense", dense_direction),
                )
        return (covering_single[0],)

    # Preserve the established v6 two-rho rule in all four symmetric
    # directions: a plateau split across one Conv edge and one Dense edge
    # implicates both axes when their union covers every plateau point.
    # Opposite edges of one axis and multiply explainable diagonal patterns
    # remain ambiguous and do not expand automatically.
    conv_edges = edges[:2]
    dense_edges = edges[2:]
    covering_pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for conv_axis, conv_direction, conv_predicate in conv_edges:
        for dense_axis, dense_direction, dense_predicate in dense_edges:
            if all(
                conv_predicate(pair) or dense_predicate(pair) for pair in pairs
            ):
                covering_pairs.append(
                    (
                        (conv_axis, conv_direction),
                        (dense_axis, dense_direction),
                    )
                )
    if len(covering_pairs) == 1:
        return covering_pairs[0]
    return ()


def boundary_expansion_plan(
    plateau: Sequence[Mapping[str, Any]],
    *,
    rho_conv_values: Iterable[float],
    rho_dense_values: Iterable[float],
) -> BoundaryExpansionPlan:
    """Build the only allowed symmetric expansion and list only new cells."""

    if not isinstance(plateau, Sequence) or isinstance(plateau, (str, bytes)):
        raise _error("plateau", "a sequence of candidate mappings", plateau)
    conv = _axis_values(rho_conv_values, "rho_conv_values")
    dense = _axis_values(rho_dense_values, "rho_dense_values")
    directions = _boundary_directions(
        plateau,
        rho_conv_values=conv,
        rho_dense_values=dense,
    )
    expanded_conv = list(conv)
    expanded_dense = list(dense)
    for axis, direction in directions:
        values = expanded_conv if axis == "rho_conv" else expanded_dense
        new_value = (
            values[0] / GRID_FACTOR
            if direction == "lower"
            else values[-1] * GRID_FACTOR
        )
        values.append(new_value)
        values.sort()
    combined = candidate_grid(expanded_conv, expanded_dense)
    original = set(candidate_grid(conv, dense))
    new_cells = tuple(pair for pair in combined if pair not in original)
    if len(combined) > MAXIMUM_CELLS_PER_SURFACE:
        raise RuntimeError(
            "Expected one symmetric expansion to produce at most "
            f"{MAXIMUM_CELLS_PER_SURFACE} cells. Provided value: {len(combined)}."
        )
    return BoundaryExpansionPlan(
        directions=directions,
        rho_conv_values=tuple(expanded_conv),
        rho_dense_values=tuple(expanded_dense),
        new_cells=new_cells,
    )


def select_surface_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    rho_conv_values: Iterable[float],
    rho_dense_values: Iterable[float],
    expansion_available: bool,
    minimum_accuracy: float = MINIMUM_FINAL_VALIDATION_ACCURACY,
    plateau_relative_tolerance: float = PLATEAU_RELATIVE_TOLERANCE,
    required_completed_steps: int = CANDIDATE_TOTAL_STEPS,
) -> dict[str, Any]:
    """Select one row x optimizer surface under the frozen inclusive rule.

    ``candidates`` contains exactly one record for every configured grid cell.
    Canary-rejected cells are represented by ``admissible=false`` records.
    An admissible record must have completed the exact three-epoch step count.
    """

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise _error("candidates", "a non-empty sequence of candidate mappings", candidates)
    if not candidates:
        raise _error("candidates", "a non-empty sequence of candidate mappings", candidates)
    conv = _axis_values(rho_conv_values, "rho_conv_values")
    dense = _axis_values(rho_dense_values, "rho_dense_values")
    grid = candidate_grid(conv, dense)
    expected_pairs = set(grid)
    floor = _finite_number(minimum_accuracy, "minimum_accuracy", nonnegative=True)
    if floor > 1.0:
        raise _error("minimum_accuracy", "a fraction in [0, 1]", minimum_accuracy)
    tolerance = _finite_number(
        plateau_relative_tolerance,
        "plateau_relative_tolerance",
        nonnegative=True,
    )
    steps = _positive_integer(required_completed_steps, "required_completed_steps")
    if not isinstance(expansion_available, bool):
        raise _error("expansion_available", "a boolean", expansion_available)

    by_pair: dict[tuple[float, float], dict[str, Any]] = {}
    identifiers: set[str] = set()
    for index, raw_candidate in enumerate(candidates):
        path = f"candidates[{index}]"
        pair = _candidate_pair(raw_candidate, path=path)
        if pair not in expected_pairs:
            raise _error(path, f"a candidate on the configured grid {grid!r}", pair)
        if pair in by_pair:
            raise _error("candidates", "unique rho target pairs", pair)
        identifier = raw_candidate.get(
            "candidate_id",
            f"rho_conv={pair[0]:.17g}--rho_dense={pair[1]:.17g}",
        )
        if not isinstance(identifier, str) or not identifier.strip():
            raise _error(f"{path}.candidate_id", "a non-empty string", identifier)
        identifier = identifier.strip()
        if identifier in identifiers:
            raise _error("candidates", "unique candidate_id values", identifier)
        identifiers.add(identifier)
        admissible = raw_candidate.get("admissible")
        if not isinstance(admissible, bool):
            raise _error(f"{path}.admissible", "a boolean", admissible)
        normalized: dict[str, Any] = {
            "candidate_id": identifier,
            "rho_conv": pair[0],
            "rho_dense": pair[1],
            "admissible": admissible,
        }
        if admissible:
            completed_steps = raw_candidate.get("completed_steps")
            if completed_steps != steps:
                raise _error(f"{path}.completed_steps", f"exactly {steps}", completed_steps)
            loss = _finite_number(
                raw_candidate.get("final_validation_loss"),
                f"{path}.final_validation_loss",
                nonnegative=True,
            )
            accuracy = _finite_number(
                raw_candidate.get("final_validation_accuracy"),
                f"{path}.final_validation_accuracy",
                nonnegative=True,
            )
            if accuracy > 1.0:
                raise _error(
                    f"{path}.final_validation_accuracy",
                    "a fraction in [0, 1]",
                    accuracy,
                )
            projection_efficiency = _finite_number(
                raw_candidate.get("median_projection_efficiency"),
                f"{path}.median_projection_efficiency",
                nonnegative=True,
            )
            passes = accuracy >= floor
            normalized.update(
                {
                    "completed_steps": completed_steps,
                    "final_validation_loss": loss,
                    "final_validation_accuracy": accuracy,
                    "median_projection_efficiency": projection_efficiency,
                    "passes": passes,
                    "failure_reason": (
                        None
                        if passes
                        else f"final_validation_accuracy_below_{floor:g}"
                    ),
                }
            )
        else:
            reason = raw_candidate.get("inadmissible_reason")
            if not isinstance(reason, str) or not reason.strip():
                raise _error(
                    f"{path}.inadmissible_reason",
                    "a non-empty canary or safety-gate reason",
                    reason,
                )
            normalized.update(
                {
                    "completed_steps": 0,
                    "final_validation_loss": None,
                    "final_validation_accuracy": None,
                    "median_projection_efficiency": None,
                    "passes": False,
                    "failure_reason": reason.strip(),
                }
            )
        by_pair[pair] = normalized
    if set(by_pair) != expected_pairs:
        raise _error(
            "candidates",
            "exactly one result for every configured rho target pair",
            {
                "missing": sorted(expected_pairs - set(by_pair)),
                "extra": sorted(set(by_pair) - expected_pairs),
            },
        )
    evaluated = [by_pair[pair] for pair in grid]
    passing = [candidate for candidate in evaluated if candidate["passes"]]
    if not passing:
        return {
            "status": "unresolved_no_pass",
            "reason": "no_safety_clean_candidate_reached_inclusive_accuracy_gate",
            "selected": None,
            "diagnostic_best": None,
            "minimum_final_validation_loss": None,
            "plateau": [],
            "expansion": BoundaryExpansionPlan((), conv, dense, ()),
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

    def selection_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        rho_conv = float(candidate["rho_conv"])
        rho_dense = float(candidate["rho_dense"])
        return (
            -float(candidate["final_validation_accuracy"]),
            -float(candidate["median_projection_efficiency"]),
            max(rho_conv, rho_dense),
            rho_conv + rho_dense,
            rho_conv,
            rho_dense,
            str(candidate["candidate_id"]),
        )

    diagnostic_best = min(plateau, key=selection_key)
    expansion = boundary_expansion_plan(
        plateau,
        rho_conv_values=conv,
        rho_dense_values=dense,
    )
    if expansion.required:
        return {
            "status": (
                "needs_expansion" if expansion_available else "unresolved_boundary"
            ),
            "reason": "passing_plateau_confined_to_outer_boundary",
            "selected": None,
            "diagnostic_best": diagnostic_best,
            "minimum_final_validation_loss": minimum_loss,
            "plateau": plateau,
            "expansion": expansion,
            "evaluated_candidates": evaluated,
        }
    return {
        "status": "selected",
        "reason": None,
        "selected": diagnostic_best,
        "diagnostic_best": diagnostic_best,
        "minimum_final_validation_loss": minimum_loss,
        "plateau": plateau,
        "expansion": expansion,
        "evaluated_candidates": evaluated,
    }


def _strict_json_loads(text: str, source: Path) -> Any:
    def reject_constant(value: str) -> Any:
        raise _error(str(source), "a finite JSON number", value)

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
        raise PerfectDiodeHparamValidationError(
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


# Set after the readable canonical config is finalized.  Validation compares
# canonical JSON rather than file whitespace.
_FROZEN_CONFIG_SHA256 = (
    "699804e1f7c35f65f50656db4af0b120dd3a3d0f2fb95e3a17b65ebfe7b78535"
)


@dataclass(frozen=True)
class PerfectDiodeHparamStudySpec:
    """Validated immutable study config and its content-derived identity."""

    _data: dict[str, Any]
    study_id: str
    config_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PerfectDiodeHparamStudySpec":
        if not isinstance(value, Mapping):
            raise _error("study", "a mapping", value)
        data = copy.deepcopy(dict(value))
        _reject_non_finite(data)
        config_sha256 = sha256_json(data)
        if config_sha256 != _FROZEN_CONFIG_SHA256:
            raise _error(
                "study",
                "the immutable perfect-diode Conv1/Conv2 v1 config "
                f"(canonical sha256 {_FROZEN_CONFIG_SHA256})",
                config_sha256,
            )
        identity_payload = copy.deepcopy(data)
        identity_payload.pop("name")
        study_id = "lrstudy_" + sha256_json(
            {
                "identity_schema": PD_STUDY_ID_SCHEMA,
                "study": identity_payload,
            }
        )
        return cls(data, study_id, config_sha256)

    @classmethod
    def from_path(cls, path: str | Path) -> "PerfectDiodeHparamStudySpec":
        source = Path(path)
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise PerfectDiodeHparamValidationError(
                f"Expected study config to be readable. Provided value: {source}."
            ) from exc
        return cls.from_dict(_strict_json_loads(text, source))

    @property
    def data(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(row) for row in self._data["rows"])

    @property
    def surfaces(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(SURFACES)


__all__ = [
    "BoundaryExpansionPlan",
    "CANDIDATE_EPOCHS",
    "CANDIDATE_TOTAL_STEPS",
    "CANARY_STEPS",
    "CONV1_CONFIRM_EPOCHS",
    "CONV1_CONFIRM_STEPS",
    "CONV2_CONFIRM_EPOCHS",
    "CONV2_CONFIRM_STEPS",
    "FROZEN_ROWS",
    "FROZEN_ROW_OBJECTS",
    "GRID_FACTOR",
    "INITIAL_RHO_CENTER",
    "MAXIMUM_CELLS_PER_SURFACE",
    "MAX_CENTER_ATTEMPTS",
    "MINIMUM_FINAL_VALIDATION_ACCURACY",
    "PD_PROTOCOL_ID",
    "PD_RUN_SCHEMA_VERSION",
    "PD_STUDY_SCHEMA_VERSION",
    "PLATEAU_RELATIVE_TOLERANCE",
    "PROBE_BATCH_COUNTS",
    "PROBE_SPLIT_HALF_RELATIVE_TOLERANCE",
    "PUBLIC_STAGE_SEQUENCE",
    "PerfectDiodeHparamStudySpec",
    "PerfectDiodeHparamValidationError",
    "ProbeStabilityDecision",
    "STEPS_PER_EPOCH",
    "SURFACES",
    "boundary_expansion_plan",
    "candidate_grid",
    "core_grid",
    "probe_stability_decision",
    "safe_center",
    "select_surface_candidates",
]
