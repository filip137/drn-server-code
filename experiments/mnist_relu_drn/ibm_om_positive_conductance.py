"""Physical positive-conductance calibration primitives for IBM OM studies.

This module deliberately does not assign a physical conductance to AIHWKit's
normalized OM state.  Instead, it loads one immutable, versioned calibration
artifact whose RESET and SET endpoints are expressed in an explicit
conductance unit.  The normalized progress coordinate used here is only

``p = 0 at RESET`` and ``p = 1 at SET``.

The artifact must say whether SET increases or decreases conductance.  Every
conversion consumes the full positive conductance

``G(p) = G_RESET + p * (G_SET - G_RESET)``.

No helper in this module clips a target or a persistent endpoint.  Empty
common windows and out-of-range progress values fail closed.  The module is
independent of training, QAT, program-and-verify, and run orchestration so a
Figure-6 audit can supply the calibration later without changing numerical
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from experiments.artifacts import content_hash, sha256_file
from experiments.mnist_relu_drn.ibm_om_baseline_selection import (
    quad_stack,
    scatter_quads,
)


CALIBRATION_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_positive_conductance_calibration"
)
CALIBRATION_SCHEMA_VERSION = 1
COMMISSIONING_SCHEMA = (
    "ebl.mnist_relu_drn.ibm_om_positive_conductance_commissioning"
)
COMMISSIONING_SCHEMA_VERSION = 1

SET_INCREASES_CONDUCTANCE = "set_increases_conductance"
SET_DECREASES_CONDUCTANCE = "set_decreases_conductance"
CONDUCTANCE_DIRECTIONS = (
    SET_INCREASES_CONDUCTANCE,
    SET_DECREASES_CONDUCTANCE,
)
PROGRESS_COORDINATE = "p=0_RESET,p=1_SET"

PAIRED_ENDPOINTS = "paired_endpoints"
EMPIRICAL_QUANTILES = "empirical_quantiles"
REPRESENTATIONS = (PAIRED_ENDPOINTS, EMPIRICAL_QUANTILES)

DESTINATION_PAIR_BASELINE = "shared_destination_pair"
QUAD_BASELINE = "shared_quad"
BASELINE_POLICIES = (DESTINATION_PAIR_BASELINE, QUAD_BASELINE)

_UNIT_TO_SIEMENS = {
    "S": 1.0,
    "mS": 1e-3,
    "uS": 1e-6,
    "nS": 1e-9,
}
_HEX_DIGITS = frozenset("0123456789abcdef")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in calibration artifact: {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number in calibration artifact: {value}.")


def _expect_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected {label} to be a JSON object.")
    return value


def _expect_keys(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
    label: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    extra = sorted(set(value) - allowed)
    if missing or extra:
        raise ValueError(
            f"Unexpected {label} keys; missing={missing!r}, extra={extra!r}."
        )


def _expect_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected {label} to be a non-empty string.")
    return value.strip()


def _expect_sha256(value: Any, *, label: str) -> str:
    digest = _expect_text(value, label=label)
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise ValueError(f"Expected {label} to be a lowercase SHA-256 digest.")
    return digest


def _finite_vector(value: Any, *, label: str, scale: float) -> torch.Tensor:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Expected {label} to be a non-empty JSON array.")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"Expected {label} to contain only JSON numbers.")
    tensor = torch.tensor(value, dtype=torch.float64) * float(scale)
    if not bool(torch.isfinite(tensor).all()) or bool(torch.any(tensor <= 0.0)):
        raise ValueError(f"Expected every {label} conductance to be strictly positive.")
    return tensor


def _probability_vector(value: Any, *, label: str) -> torch.Tensor:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"Expected {label} to contain at least two probabilities.")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"Expected {label} to contain only JSON numbers.")
    tensor = torch.tensor(value, dtype=torch.float64)
    if (
        not bool(torch.isfinite(tensor).all())
        or float(tensor[0].item()) != 0.0
        or float(tensor[-1].item()) != 1.0
        or bool(torch.any(tensor[1:] <= tensor[:-1]))
    ):
        raise ValueError(
            f"Expected {label} to increase strictly from exactly 0 to exactly 1."
        )
    return tensor


def _direction_is_valid(
    reset_s: torch.Tensor,
    set_s: torch.Tensor,
    *,
    direction: str,
) -> torch.Tensor:
    if direction == SET_INCREASES_CONDUCTANCE:
        return set_s > reset_s
    if direction == SET_DECREASES_CONDUCTANCE:
        return set_s < reset_s
    raise ValueError(f"Expected direction in {CONDUCTANCE_DIRECTIONS!r}.")


def _tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().to(device="cpu")
    digest = sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(
        json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8")
    )
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _inverse_quantile(
    probabilities: torch.Tensor,
    values: torch.Tensor,
    uniform: torch.Tensor,
) -> torch.Tensor:
    indices = torch.searchsorted(probabilities, uniform, right=True)
    indices = torch.clamp(indices, min=1, max=probabilities.numel() - 1)
    low_index = indices - 1
    high_index = indices
    low_probability = probabilities[low_index]
    high_probability = probabilities[high_index]
    fraction = (uniform - low_probability) / (
        high_probability - low_probability
    )
    return values[low_index] + fraction * (
        values[high_index] - values[low_index]
    )


@dataclass(frozen=True)
class PositiveConductanceCalibration:
    """Validated physical RESET/SET calibration loaded from one JSON file."""

    calibration_id: str
    artifact_path: Path
    artifact_sha256: str
    source_sha256: str
    provenance: Mapping[str, Any]
    input_unit: str
    direction: str
    representation: str
    sampling_policy: str
    pairing_policy: str
    reset_values_s: torch.Tensor
    set_values_s: torch.Tensor
    reset_probabilities: torch.Tensor | None = None
    set_probabilities: torch.Tensor | None = None

    def __post_init__(self) -> None:
        _expect_text(self.calibration_id, label="calibration_id")
        _expect_sha256(self.artifact_sha256, label="artifact_sha256")
        _expect_sha256(self.source_sha256, label="source_sha256")
        if self.input_unit not in _UNIT_TO_SIEMENS:
            raise ValueError(f"Unsupported conductance unit: {self.input_unit!r}.")
        if self.direction not in CONDUCTANCE_DIRECTIONS:
            raise ValueError(f"Expected direction in {CONDUCTANCE_DIRECTIONS!r}.")
        if self.representation not in REPRESENTATIONS:
            raise ValueError(f"Expected representation in {REPRESENTATIONS!r}.")
        for name in ("reset_values_s", "set_values_s"):
            value = getattr(self, name)
            if (
                value.ndim != 1
                or value.numel() < 1
                or value.dtype != torch.float64
                or value.device.type != "cpu"
                or not bool(torch.isfinite(value).all())
                or bool(torch.any(value <= 0.0))
            ):
                raise ValueError(f"Expected finite positive CPU float64 {name}.")

    def report(self) -> Mapping[str, Any]:
        """Return the immutable calibration and physical-coordinate contract."""

        report = {
            "schema": CALIBRATION_SCHEMA,
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "calibration_id": self.calibration_id,
            "calibration_path": str(self.artifact_path),
            "calibration_sha256": self.artifact_sha256,
            "source_sha256": self.source_sha256,
            "evidence_class": self.provenance["evidence_class"],
            "input_conductance_unit": self.input_unit,
            "internal_conductance_unit": "S",
            "direction": self.direction,
            "progress_coordinate": PROGRESS_COORDINATE,
            "representation": self.representation,
            "sampling_policy": self.sampling_policy,
            "pairing_policy": self.pairing_policy,
            "reference_subtraction": False,
            "post_handoff_clipping": False,
        }
        if "figure_crop_sha256" in self.provenance:
            report["figure_crop_sha256"] = self.provenance[
                "figure_crop_sha256"
            ]
        return report

    def sample_population(
        self, *, devices: int, assignment_seed: int
    ) -> "PositiveConductancePopulation":
        """Draw a deterministic cell population under the artifact policy."""

        if isinstance(devices, bool) or not isinstance(devices, int) or devices < 1:
            raise ValueError("Expected devices to be a positive integer.")
        if isinstance(assignment_seed, bool) or not isinstance(assignment_seed, int):
            raise ValueError("Expected assignment_seed to be an integer.")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(assignment_seed))
        if self.representation == PAIRED_ENDPOINTS:
            available = int(self.reset_values_s.numel())
            if self.sampling_policy == "without_replacement":
                if devices > available:
                    raise ValueError(
                        "Requested more devices than paired endpoints under "
                        "without-replacement sampling."
                    )
                index = torch.randperm(available, generator=generator)[:devices]
            elif self.sampling_policy == "with_replacement":
                index = torch.randint(available, (devices,), generator=generator)
            else:
                raise RuntimeError("Validated paired sampling policy became invalid.")
            reset_s = self.reset_values_s[index]
            set_s = self.set_values_s[index]
        else:
            reset_u = torch.rand(devices, generator=generator, dtype=torch.float64)
            set_u = (
                reset_u
                if self.pairing_policy == "rank_matched"
                else torch.rand(devices, generator=generator, dtype=torch.float64)
            )
            assert self.reset_probabilities is not None
            assert self.set_probabilities is not None
            reset_s = _inverse_quantile(
                self.reset_probabilities, self.reset_values_s, reset_u
            )
            set_s = _inverse_quantile(
                self.set_probabilities, self.set_values_s, set_u
            )
        valid = _direction_is_valid(reset_s, set_s, direction=self.direction)
        if not bool(valid.all()):
            raise RuntimeError(
                "Calibrated population draw violated the declared RESET-to-SET "
                "direction; the artifact must encode a valid joint distribution."
            )
        population_hash = content_hash(
            {
                "calibration_sha256": self.artifact_sha256,
                "assignment_seed": int(assignment_seed),
                "devices": devices,
                "reset_conductance_s_sha256": _tensor_sha256(reset_s),
                "set_conductance_s_sha256": _tensor_sha256(set_s),
            }
        )
        report = {
            "schema": "ebl.mnist_relu_drn.ibm_om_positive_conductance_population",
            "schema_version": 1,
            "calibration_id": self.calibration_id,
            "calibration_sha256": self.artifact_sha256,
            "source_sha256": self.source_sha256,
            "representation": self.representation,
            "sampling_policy": self.sampling_policy,
            "pairing_policy": self.pairing_policy,
            "joint_model_assumption": (
                "paired_endpoints_retained_by_index"
                if self.representation == PAIRED_ENDPOINTS
                else f"synthetic_joint_draw_from_unpaired_marginals:"
                f"{self.pairing_policy}"
            ),
            "identity_semantics": (
                "artifact_paired_endpoint_record"
                if self.representation == PAIRED_ENDPOINTS
                else "model_sample_not_raw_measured_device_identity"
            ),
            "assignment_seed": int(assignment_seed),
            "devices": devices,
            "direction": self.direction,
            "progress_coordinate": PROGRESS_COORDINATE,
            "conductance_unit": "S",
            "strictly_positive": True,
            "reset_conductance_s_sha256": _tensor_sha256(reset_s),
            "set_conductance_s_sha256": _tensor_sha256(set_s),
            "population_sha256": population_hash,
        }
        return PositiveConductancePopulation(
            calibration_id=self.calibration_id,
            calibration_sha256=self.artifact_sha256,
            source_sha256=self.source_sha256,
            assignment_seed=int(assignment_seed),
            direction=self.direction,
            reset_conductance_s=reset_s.clone(),
            set_conductance_s=set_s.clone(),
            population_sha256=population_hash,
            report=report,
        )


@dataclass(frozen=True)
class PositiveConductancePopulation:
    """One deterministic assignment of positive physical cell endpoints."""

    calibration_id: str
    calibration_sha256: str
    source_sha256: str
    assignment_seed: int
    direction: str
    reset_conductance_s: torch.Tensor
    set_conductance_s: torch.Tensor
    population_sha256: str
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        for digest_name in (
            "calibration_sha256",
            "source_sha256",
            "population_sha256",
        ):
            _expect_sha256(getattr(self, digest_name), label=digest_name)
        reset = self.reset_conductance_s
        set_state = self.set_conductance_s
        if (
            reset.ndim != 1
            or set_state.shape != reset.shape
            or reset.dtype != torch.float64
            or set_state.dtype != torch.float64
            or reset.device.type != "cpu"
            or set_state.device.type != "cpu"
            or reset.numel() < 1
            or not bool(torch.isfinite(reset).all())
            or not bool(torch.isfinite(set_state).all())
            or bool(torch.any(reset <= 0.0))
            or bool(torch.any(set_state <= 0.0))
            or not bool(
                _direction_is_valid(reset, set_state, direction=self.direction).all()
            )
        ):
            raise ValueError("Expected ordered, strictly positive endpoint vectors.")

    @property
    def devices(self) -> int:
        return int(self.reset_conductance_s.numel())


@dataclass(frozen=True)
class PositiveConductanceCommissioning:
    """Per-cell RESET/SET endpoint estimates from explicit physical-G reads."""

    calibration_id: str
    calibration_sha256: str
    source_sha256: str
    population_sha256: str
    commissioning_seed: int
    direction: str
    read_samples: int
    reset_mean_s: torch.Tensor
    set_mean_s: torch.Tensor
    reset_standard_error_s: torch.Tensor
    set_standard_error_s: torch.Tensor
    commissioning_sha256: str
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "calibration_sha256",
            "source_sha256",
            "population_sha256",
            "commissioning_sha256",
        ):
            _expect_sha256(getattr(self, name), label=name)
        shape = self.reset_mean_s.shape
        tensors = (
            self.reset_mean_s,
            self.set_mean_s,
            self.reset_standard_error_s,
            self.set_standard_error_s,
        )
        if (
            self.read_samples < 2
            or len(shape) != 1
            or shape[0] < 1
            or any(
                value.shape != shape
                or value.dtype != torch.float64
                or value.device.type != "cpu"
                or not bool(torch.isfinite(value).all())
                for value in tensors
            )
            or bool(torch.any(self.reset_mean_s <= 0.0))
            or bool(torch.any(self.set_mean_s <= 0.0))
            or bool(torch.any(self.reset_standard_error_s < 0.0))
            or bool(torch.any(self.set_standard_error_s < 0.0))
            or not bool(
                _direction_is_valid(
                    self.reset_mean_s,
                    self.set_mean_s,
                    direction=self.direction,
                ).all()
            )
        ):
            raise ValueError("Invalid positive-conductance commissioning state.")

    @property
    def devices(self) -> int:
        return int(self.reset_mean_s.numel())


@dataclass(frozen=True)
class PositiveConductanceEmbedding:
    """Cell-wise map from normalized RESET-to-SET progress into full G."""

    reset_conductance_s: torch.Tensor
    set_conductance_s: torch.Tensor
    direction: str
    calibration_sha256: str
    source_sha256: str

    @classmethod
    def from_population(
        cls, population: PositiveConductancePopulation
    ) -> "PositiveConductanceEmbedding":
        return cls(
            reset_conductance_s=population.reset_conductance_s,
            set_conductance_s=population.set_conductance_s,
            direction=population.direction,
            calibration_sha256=population.calibration_sha256,
            source_sha256=population.source_sha256,
        )

    @classmethod
    def from_commissioning(
        cls, commissioning: PositiveConductanceCommissioning
    ) -> "PositiveConductanceEmbedding":
        return cls(
            reset_conductance_s=commissioning.reset_mean_s,
            set_conductance_s=commissioning.set_mean_s,
            direction=commissioning.direction,
            calibration_sha256=commissioning.calibration_sha256,
            source_sha256=commissioning.source_sha256,
        )

    def __post_init__(self) -> None:
        _expect_sha256(self.calibration_sha256, label="calibration_sha256")
        _expect_sha256(self.source_sha256, label="source_sha256")
        reset = self.reset_conductance_s.detach().to(
            device="cpu", dtype=torch.float64
        )
        set_state = self.set_conductance_s.detach().to(
            device="cpu", dtype=torch.float64
        )
        if (
            reset.ndim != 1
            or reset.numel() < 1
            or set_state.shape != reset.shape
            or not bool(torch.isfinite(reset).all())
            or not bool(torch.isfinite(set_state).all())
            or bool(torch.any(reset <= 0.0))
            or bool(torch.any(set_state <= 0.0))
            or not bool(
                _direction_is_valid(reset, set_state, direction=self.direction).all()
            )
        ):
            raise ValueError("Expected positive ordered RESET/SET endpoints.")
        object.__setattr__(self, "reset_conductance_s", reset)
        object.__setattr__(self, "set_conductance_s", set_state)

    def to_full_conductance(self, progress: torch.Tensor) -> torch.Tensor:
        """Apply ``G=G_RESET+p(G_SET-G_RESET)`` without clipping ``p``."""

        value = torch.as_tensor(progress)
        if (
            value.numel() < 1
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
            or bool(torch.any(value < 0.0))
            or bool(torch.any(value > 1.0))
        ):
            raise ValueError("Expected finite progress in [0,1]; clipping is forbidden.")
        reset = self.reset_conductance_s.to(
            device=value.device, dtype=torch.float64
        )
        set_state = self.set_conductance_s.to(
            device=value.device, dtype=torch.float64
        )
        progress64 = value.to(dtype=torch.float64)
        try:
            result = reset + progress64 * (set_state - reset)
        except RuntimeError as error:
            raise ValueError(
                "Progress shape is not broadcast-compatible with calibrated cells."
            ) from error
        if not bool(torch.isfinite(result).all()) or bool(torch.any(result <= 0.0)):
            raise RuntimeError("Positive-conductance interpolation became invalid.")
        return result.to(dtype=value.dtype)

    def report(self) -> Mapping[str, Any]:
        return {
            "formula": "G=G_RESET+p*(G_SET-G_RESET)",
            "coordinate": PROGRESS_COORDINATE,
            "direction": self.direction,
            "conductance_unit": "S",
            "reference_subtraction": False,
            "post_handoff_clipping": False,
            "calibration_sha256": self.calibration_sha256,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class CommonConductanceBaseline:
    """Exact common RESET-side baseline for destination pairs or a quad."""

    policy: str
    layout: str
    baseline_position_fraction: float
    direction: str
    baseline_s: torch.Tensor
    group_reset_frontier_s: torch.Tensor
    group_set_frontier_s: torch.Tensor
    group_headroom_s: torch.Tensor
    calibration_sha256: str
    source_sha256: str
    report: Mapping[str, Any]


def load_positive_conductance_calibration(
    path: Path | str,
) -> PositiveConductanceCalibration:
    """Load and strictly validate a version-1 positive-G calibration JSON."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Calibration artifact not found: {artifact_path}.")
    try:
        payload = json.loads(
            artifact_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid calibration JSON: {error}.") from error
    root = _expect_mapping(payload, label="calibration artifact")
    _expect_keys(
        root,
        required=(
            "schema",
            "schema_version",
            "calibration_id",
            "provenance",
            "conductance",
        ),
        label="calibration artifact",
    )
    if root["schema"] != CALIBRATION_SCHEMA:
        raise ValueError(f"Expected calibration schema {CALIBRATION_SCHEMA!r}.")
    if root["schema_version"] != CALIBRATION_SCHEMA_VERSION:
        raise ValueError(
            f"Expected calibration schema_version {CALIBRATION_SCHEMA_VERSION}."
        )
    calibration_id = _expect_text(root["calibration_id"], label="calibration_id")

    provenance = _expect_mapping(root["provenance"], label="provenance")
    _expect_keys(
        provenance,
        required=(
            "citation",
            "year",
            "figure",
            "evidence_class",
            "extraction_method",
            "source_sha256",
        ),
        optional=("doi", "source_url", "notes", "figure_crop_sha256"),
        label="provenance",
    )
    for name in ("citation", "figure", "evidence_class", "extraction_method"):
        _expect_text(provenance[name], label=f"provenance.{name}")
    year = provenance["year"]
    if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2200:
        raise ValueError("Expected provenance.year to be a plausible integer year.")
    source_sha256 = _expect_sha256(
        provenance["source_sha256"], label="provenance.source_sha256"
    )
    if "figure_crop_sha256" in provenance:
        _expect_sha256(
            provenance["figure_crop_sha256"],
            label="provenance.figure_crop_sha256",
        )
    for name in ("doi", "source_url", "notes"):
        if name in provenance:
            _expect_text(provenance[name], label=f"provenance.{name}")

    conductance = _expect_mapping(root["conductance"], label="conductance")
    common_keys = (
        "unit",
        "direction",
        "progress_coordinate",
        "representation",
        "reset",
        "set",
    )
    representation = conductance.get("representation")
    if representation == PAIRED_ENDPOINTS:
        _expect_keys(
            conductance,
            required=(*common_keys, "sampling_policy"),
            label="conductance",
        )
    elif representation == EMPIRICAL_QUANTILES:
        _expect_keys(
            conductance,
            required=(*common_keys, "pairing_policy"),
            label="conductance",
        )
    else:
        raise ValueError(f"Expected representation in {REPRESENTATIONS!r}.")
    unit = conductance["unit"]
    if unit not in _UNIT_TO_SIEMENS:
        raise ValueError(
            f"Expected conductance.unit in {tuple(_UNIT_TO_SIEMENS)!r}."
        )
    direction = conductance["direction"]
    if direction not in CONDUCTANCE_DIRECTIONS:
        raise ValueError(f"Expected direction in {CONDUCTANCE_DIRECTIONS!r}.")
    if conductance["progress_coordinate"] != PROGRESS_COORDINATE:
        raise ValueError(
            f"Expected progress_coordinate to equal {PROGRESS_COORDINATE!r}."
        )
    scale = _UNIT_TO_SIEMENS[unit]

    reset_probabilities = None
    set_probabilities = None
    if representation == PAIRED_ENDPOINTS:
        sampling_policy = conductance["sampling_policy"]
        if sampling_policy not in {"with_replacement", "without_replacement"}:
            raise ValueError("Unexpected paired endpoint sampling_policy.")
        pairing_policy = "paired_by_index"
        reset_values_s = _finite_vector(
            conductance["reset"], label="conductance.reset", scale=scale
        )
        set_values_s = _finite_vector(
            conductance["set"], label="conductance.set", scale=scale
        )
        if reset_values_s.shape != set_values_s.shape:
            raise ValueError("Paired RESET and SET endpoint arrays must match.")
        if not bool(
            _direction_is_valid(
                reset_values_s, set_values_s, direction=direction
            ).all()
        ):
            raise ValueError(
                "Paired endpoints violate the declared conductance direction."
            )
    else:
        sampling_policy = "inverse_empirical_cdf"
        pairing_policy = conductance["pairing_policy"]
        if pairing_policy not in {"independent", "rank_matched"}:
            raise ValueError("Unexpected empirical-quantile pairing_policy.")
        endpoint_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for endpoint in ("reset", "set"):
            spec = _expect_mapping(
                conductance[endpoint], label=f"conductance.{endpoint}"
            )
            _expect_keys(
                spec,
                required=("probabilities", "values"),
                label=f"conductance.{endpoint}",
            )
            probabilities = _probability_vector(
                spec["probabilities"],
                label=f"conductance.{endpoint}.probabilities",
            )
            values_s = _finite_vector(
                spec["values"],
                label=f"conductance.{endpoint}.values",
                scale=scale,
            )
            if probabilities.shape != values_s.shape or bool(
                torch.any(values_s[1:] < values_s[:-1])
            ):
                raise ValueError(
                    f"Expected conductance.{endpoint} quantile values to be "
                    "nondecreasing and match their probabilities."
                )
            endpoint_values.append((probabilities, values_s))
        (reset_probabilities, reset_values_s), (
            set_probabilities,
            set_values_s,
        ) = endpoint_values
        if pairing_policy == "independent":
            if direction == SET_INCREASES_CONDUCTANCE:
                ordered = set_values_s.min() > reset_values_s.max()
            else:
                ordered = set_values_s.max() < reset_values_s.min()
        else:
            union = torch.unique(
                torch.cat((reset_probabilities, set_probabilities)), sorted=True
            )
            reset_at_union = _inverse_quantile(
                reset_probabilities, reset_values_s, union
            )
            set_at_union = _inverse_quantile(
                set_probabilities, set_values_s, union
            )
            ordered = _direction_is_valid(
                reset_at_union, set_at_union, direction=direction
            ).all()
        if not bool(ordered):
            raise ValueError(
                "Empirical marginals cannot guarantee the declared direction "
                "under their pairing_policy; provide paired endpoints or a valid "
                "joint policy."
            )

    return PositiveConductanceCalibration(
        calibration_id=calibration_id,
        artifact_path=artifact_path.resolve(),
        artifact_sha256=sha256_file(artifact_path),
        source_sha256=source_sha256,
        provenance=dict(provenance),
        input_unit=unit,
        direction=direction,
        representation=representation,
        sampling_policy=sampling_policy,
        pairing_policy=pairing_policy,
        reset_values_s=reset_values_s,
        set_values_s=set_values_s,
        reset_probabilities=reset_probabilities,
        set_probabilities=set_probabilities,
    )


def commission_positive_conductance_endpoints(
    population: PositiveConductancePopulation,
    *,
    reset_readings_s: torch.Tensor,
    set_readings_s: torch.Tensor,
    commissioning_seed: int,
) -> PositiveConductanceCommissioning:
    """Estimate per-cell physical endpoints from repeated positive-G reads."""

    if not isinstance(population, PositiveConductancePopulation):
        raise TypeError("Expected one PositiveConductancePopulation.")
    if isinstance(commissioning_seed, bool) or not isinstance(commissioning_seed, int):
        raise ValueError("Expected commissioning_seed to be an integer.")
    reset_reads = torch.as_tensor(reset_readings_s).detach().to(
        device="cpu", dtype=torch.float64
    )
    set_reads = torch.as_tensor(set_readings_s).detach().to(
        device="cpu", dtype=torch.float64
    )
    if (
        reset_reads.ndim != 2
        or set_reads.shape != reset_reads.shape
        or reset_reads.shape[0] < 2
        or reset_reads.shape[1] != population.devices
        or not bool(torch.isfinite(reset_reads).all())
        or not bool(torch.isfinite(set_reads).all())
        or bool(torch.any(reset_reads <= 0.0))
        or bool(torch.any(set_reads <= 0.0))
    ):
        raise ValueError(
            "Expected matched [read_samples, devices] strictly positive "
            "conductance readings in SI units."
        )
    read_samples = int(reset_reads.shape[0])
    reset_mean = reset_reads.mean(dim=0)
    set_mean = set_reads.mean(dim=0)
    if not bool(
        _direction_is_valid(reset_mean, set_mean, direction=population.direction).all()
    ):
        raise ValueError(
            "Commissioned endpoint means violate the declared conductance direction."
        )
    reset_se = reset_reads.std(dim=0, unbiased=True) / math.sqrt(read_samples)
    set_se = set_reads.std(dim=0, unbiased=True) / math.sqrt(read_samples)
    commissioning_hash = content_hash(
        {
            "population_sha256": population.population_sha256,
            "commissioning_seed": int(commissioning_seed),
            "read_samples": read_samples,
            "reset_readings_s_sha256": _tensor_sha256(reset_reads),
            "set_readings_s_sha256": _tensor_sha256(set_reads),
            "reset_mean_s_sha256": _tensor_sha256(reset_mean),
            "set_mean_s_sha256": _tensor_sha256(set_mean),
        }
    )
    report = {
        "schema": COMMISSIONING_SCHEMA,
        "schema_version": COMMISSIONING_SCHEMA_VERSION,
        "algorithm": "per_cell_arithmetic_mean_of_explicit_RESET_and_SET_reads",
        "conductance_unit": "S",
        "direction": population.direction,
        "progress_coordinate": PROGRESS_COORDINATE,
        "read_samples_per_endpoint": read_samples,
        "devices": population.devices,
        "commissioning_seed": int(commissioning_seed),
        "reference_subtraction": False,
        "pooling": "none_per_cell",
        "projection": "none",
        "post_handoff_clipping": False,
        "source_sha256": population.source_sha256,
        "calibration_sha256": population.calibration_sha256,
        "population_sha256": population.population_sha256,
        "reset_readings_s_sha256": _tensor_sha256(reset_reads),
        "set_readings_s_sha256": _tensor_sha256(set_reads),
        "reset_mean_s_sha256": _tensor_sha256(reset_mean),
        "set_mean_s_sha256": _tensor_sha256(set_mean),
        "commissioning_sha256": commissioning_hash,
    }
    return PositiveConductanceCommissioning(
        calibration_id=population.calibration_id,
        calibration_sha256=population.calibration_sha256,
        source_sha256=population.source_sha256,
        population_sha256=population.population_sha256,
        commissioning_seed=int(commissioning_seed),
        direction=population.direction,
        read_samples=read_samples,
        reset_mean_s=reset_mean,
        set_mean_s=set_mean,
        reset_standard_error_s=reset_se,
        set_standard_error_s=set_se,
        commissioning_sha256=commissioning_hash,
        report=report,
    )


def build_common_conductance_baseline(
    reset_conductance_s: torch.Tensor,
    set_conductance_s: torch.Tensor,
    *,
    direction: str,
    policy: str,
    layout: str,
    baseline_position_fraction: float = 0.0,
    calibration_sha256: str,
    source_sha256: str,
) -> CommonConductanceBaseline:
    """Construct one exact shared baseline without projection or clipping.

    For SET-increasing cells the RESET-side common frontier is ``max(RESET)``
    and the SET-side frontier is ``min(SET)``.  For SET-decreasing cells those
    extrema reverse.  ``baseline_position_fraction`` moves from that exact
    RESET-side frontier toward the exact SET-side frontier.
    """

    _expect_sha256(calibration_sha256, label="calibration_sha256")
    _expect_sha256(source_sha256, label="source_sha256")
    if direction not in CONDUCTANCE_DIRECTIONS:
        raise ValueError(f"Expected direction in {CONDUCTANCE_DIRECTIONS!r}.")
    if policy not in BASELINE_POLICIES:
        raise ValueError(f"Expected policy in {BASELINE_POLICIES!r}.")
    alpha = float(baseline_position_fraction)
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("Expected baseline_position_fraction in [0,1].")
    reset = torch.as_tensor(reset_conductance_s).detach().to(
        device="cpu", dtype=torch.float64
    )
    set_state = torch.as_tensor(set_conductance_s).detach().to(
        device="cpu", dtype=torch.float64
    )
    if (
        reset.ndim != 2
        or set_state.shape != reset.shape
        or reset.numel() < 4
        or not bool(torch.isfinite(reset).all())
        or not bool(torch.isfinite(set_state).all())
        or bool(torch.any(reset <= 0.0))
        or bool(torch.any(set_state <= 0.0))
        or not bool(
            _direction_is_valid(reset, set_state, direction=direction).all()
        )
    ):
        raise ValueError(
            "Expected finite rank-2, strictly positive, direction-consistent "
            "RESET/SET conductance tensors."
        )
    reset_quad = quad_stack(reset, layout=layout)
    set_quad = quad_stack(set_state, layout=layout)
    if policy == DESTINATION_PAIR_BASELINE:
        groups = ((0, 2), (1, 3))
    else:
        groups = ((0, 1, 2, 3),)
    reset_frontiers = []
    set_frontiers = []
    for indices in groups:
        reset_group = reset_quad[..., list(indices)]
        set_group = set_quad[..., list(indices)]
        if direction == SET_INCREASES_CONDUCTANCE:
            reset_frontier = reset_group.max(dim=-1).values
            set_frontier = set_group.min(dim=-1).values
        else:
            reset_frontier = reset_group.min(dim=-1).values
            set_frontier = set_group.max(dim=-1).values
        reset_frontiers.append(reset_frontier)
        set_frontiers.append(set_frontier)
    group_reset = torch.stack(reset_frontiers, dim=-1)
    group_set = torch.stack(set_frontiers, dim=-1)
    feasible = _direction_is_valid(group_reset, group_set, direction=direction)
    if not bool(feasible.all()):
        raise ValueError(
            "At least one declared baseline group has no nonzero exact common "
            "RESET-to-SET window; projection and clipping are forbidden."
        )
    group_baseline = group_reset + alpha * (group_set - group_reset)
    if policy == DESTINATION_PAIR_BASELINE:
        baseline_quad = torch.stack(
            (
                group_baseline[..., 0],
                group_baseline[..., 1],
                group_baseline[..., 0],
                group_baseline[..., 1],
            ),
            dim=-1,
        )
    else:
        baseline_quad = group_baseline.expand(*group_baseline.shape[:-1], 4)
    baseline = scatter_quads(
        baseline_quad, shape=tuple(reset.shape), layout=layout
    )
    cell_low = torch.minimum(reset, set_state)
    cell_high = torch.maximum(reset, set_state)
    if (
        bool(torch.any(baseline <= 0.0))
        or bool(torch.any(baseline < cell_low))
        or bool(torch.any(baseline > cell_high))
    ):
        raise RuntimeError("Constructed baseline left exact positive cell support.")
    headroom = torch.abs(group_set - group_baseline)
    report = {
        "schema": "ebl.mnist_relu_drn.ibm_om_positive_conductance_baseline",
        "schema_version": 1,
        "policy": policy,
        "layout": layout,
        "direction": direction,
        "baseline_position_fraction": alpha,
        "conductance_unit": "S",
        "formula": "B=G_RESET_frontier+alpha*(G_SET_frontier-G_RESET_frontier)",
        "reference_subtraction": False,
        "target_projection": "none",
        "post_handoff_clipping": False,
        "source_sha256": source_sha256,
        "calibration_sha256": calibration_sha256,
        "reset_conductance_s_sha256": _tensor_sha256(reset),
        "set_conductance_s_sha256": _tensor_sha256(set_state),
        "baseline_s_sha256": _tensor_sha256(baseline),
        "group_reset_frontier_s_sha256": _tensor_sha256(group_reset),
        "group_set_frontier_s_sha256": _tensor_sha256(group_set),
    }
    return CommonConductanceBaseline(
        policy=policy,
        layout=layout,
        baseline_position_fraction=alpha,
        direction=direction,
        baseline_s=baseline,
        group_reset_frontier_s=group_reset,
        group_set_frontier_s=group_set,
        group_headroom_s=headroom,
        calibration_sha256=calibration_sha256,
        source_sha256=source_sha256,
        report=report,
    )


__all__ = [
    "BASELINE_POLICIES",
    "CALIBRATION_SCHEMA",
    "CALIBRATION_SCHEMA_VERSION",
    "COMMISSIONING_SCHEMA",
    "COMMISSIONING_SCHEMA_VERSION",
    "CONDUCTANCE_DIRECTIONS",
    "DESTINATION_PAIR_BASELINE",
    "EMPIRICAL_QUANTILES",
    "PAIRED_ENDPOINTS",
    "PROGRESS_COORDINATE",
    "QUAD_BASELINE",
    "SET_DECREASES_CONDUCTANCE",
    "SET_INCREASES_CONDUCTANCE",
    "CommonConductanceBaseline",
    "PositiveConductanceCalibration",
    "PositiveConductanceCommissioning",
    "PositiveConductanceEmbedding",
    "PositiveConductancePopulation",
    "build_common_conductance_baseline",
    "commission_positive_conductance_endpoints",
    "load_positive_conductance_calibration",
]
