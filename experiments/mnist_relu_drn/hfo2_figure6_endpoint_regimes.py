"""Figure-6-informed synthetic HfO2 full-RESET/full-SET endpoint regimes.

This module defines an analyst-selected endpoint population in a unitless,
positive state coordinate.  It deliberately does not modify AIHWKit's
``ReRamArrayHfO2PresetDevice`` and must not be presented as a refit of the
unpublished IBM device identities.

The two endpoint operations are

``full_tile_RESET = max(0.05, 0.1 + sigma_RESET * z_RESET)``

and

``full_tile_SET = 2.0 + 0.4295 * z_SET``.

``sigma_RESET`` retains the AIHWKit HfO2 SET-bound standard deviation and
scales it by the RESET-to-SET standard-deviation ratio digitized from IEDM
2022 Figure 6.  The ratio is applied before the requested RESET clamp.  SET
states are never clipped.  Independent and rank-matched endpoint pairing are
kept as separate regimes because Figure 6 publishes marginal CDFs rather
than the underlying per-device pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping

import torch


FULL_TILE_RESET = "full_tile_RESET"
FULL_TILE_SET = "full_tile_SET"

INDEPENDENT_ENDPOINTS = "figure6_ratio_independent"
RANK_MATCHED_ENDPOINTS = "figure6_ratio_rank_matched"
ENDPOINT_REGIMES = (INDEPENDENT_ENDPOINTS, RANK_MATCHED_ENDPOINTS)

# Digitized Baseline-HfO2 Figure-6 marginal statistics, in microSiemens.
FIGURE6_RESET_STD_US = 6.4411
FIGURE6_SET_STD_US = 32.5241
FIGURE6_RESET_TO_SET_STD_RATIO = FIGURE6_RESET_STD_US / FIGURE6_SET_STD_US

# Unitless positive-state model requested for the synthetic control.
RESET_STATE_CENTER = 0.1
SET_STATE_CENTER = 2.0
RESET_STATE_FLOOR = 0.05
HFO2_SET_STATE_STD = 0.4295
HFO2_FIGURE6_RESET_STATE_STD = (
    HFO2_SET_STATE_STD * FIGURE6_RESET_TO_SET_STD_RATIO
)


def _finite_standard_normal(value: torch.Tensor, *, label: str) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim != 1 or tensor.numel() < 1 or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Expected {label} to be a non-empty finite vector.")
    return tensor


def full_tile_reset(
    standard_normal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the synthetic full-RESET endpoint law.

    Returns ``(realized, raw, clipped_mask)``.  Clipping is lower Winsorization:
    every device is retained and every raw value below ``0.05`` is set to
    exactly ``0.05``.
    """

    z = _finite_standard_normal(standard_normal, label="RESET standard normal")
    raw = RESET_STATE_CENTER + HFO2_FIGURE6_RESET_STATE_STD * z
    realized = torch.clamp_min(raw, RESET_STATE_FLOOR)
    return realized, raw, raw < RESET_STATE_FLOOR


def full_tile_set(standard_normal: torch.Tensor) -> torch.Tensor:
    """Apply the synthetic full-SET endpoint law without clipping."""

    z = _finite_standard_normal(standard_normal, label="SET standard normal")
    return SET_STATE_CENTER + HFO2_SET_STATE_STD * z


# Preserve the operation names selected for the experiment while keeping the
# implementation functions in conventional Python snake case.
full_tile_RESET = full_tile_reset
full_tile_SET = full_tile_set


def _derived_seed(base_seed: int, operation: str) -> int:
    digest = sha256(f"{int(base_seed)}\x1f{operation}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1) + 1


def _normal_vector(*, devices: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(devices, generator=generator, dtype=torch.float64)


def _match_ranks(reference: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Permute ``values`` so its ascending ranks match ``reference`` exactly."""

    reference_order = torch.argsort(reference, stable=True)
    sorted_values = torch.sort(values, stable=True).values
    matched = torch.empty_like(values)
    matched[reference_order] = sorted_values
    return matched


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().to(device="cpu")
    digest = sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _summary(value: torch.Tensor) -> Mapping[str, float]:
    tensor = value.detach().to(device="cpu", dtype=torch.float64)
    standard_deviation = (
        float(tensor.std(unbiased=True).item()) if tensor.numel() > 1 else 0.0
    )
    return {
        "minimum": float(tensor.min().item()),
        "median": float(torch.quantile(tensor, 0.5).item()),
        "mean": float(tensor.mean().item()),
        "standard_deviation": standard_deviation,
        "maximum": float(tensor.max().item()),
    }


@dataclass(frozen=True)
class HfO2Figure6EndpointPopulation:
    """One deterministic draw from a declared synthetic endpoint regime."""

    regime: str
    assignment_seed: int
    reset_standard_normal: torch.Tensor
    set_standard_normal: torch.Tensor
    reset_raw_state: torch.Tensor
    reset_state: torch.Tensor
    set_state: torch.Tensor
    reset_clipped: torch.Tensor

    def __post_init__(self) -> None:
        if self.regime not in ENDPOINT_REGIMES:
            raise ValueError(f"Expected regime in {ENDPOINT_REGIMES!r}.")
        if isinstance(self.assignment_seed, bool) or not isinstance(
            self.assignment_seed, int
        ):
            raise ValueError("Expected assignment_seed to be an integer.")
        size = int(self.reset_state.numel())
        if size < 1:
            raise ValueError("Expected at least one endpoint identity.")
        for name in (
            "reset_standard_normal",
            "set_standard_normal",
            "reset_raw_state",
            "reset_state",
            "set_state",
        ):
            value = getattr(self, name)
            if (
                value.ndim != 1
                or value.numel() != size
                or value.dtype != torch.float64
                or value.device.type != "cpu"
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"Expected finite CPU float64 {name} of length {size}."
                )
        if (
            self.reset_clipped.ndim != 1
            or self.reset_clipped.numel() != size
            or self.reset_clipped.dtype != torch.bool
            or self.reset_clipped.device.type != "cpu"
        ):
            raise ValueError("Expected a CPU boolean RESET clipping mask.")
        expected_reset = torch.clamp_min(self.reset_raw_state, RESET_STATE_FLOOR)
        if not torch.equal(self.reset_state, expected_reset):
            raise ValueError("RESET states do not implement the declared lower clamp.")
        if not torch.equal(
            self.reset_clipped, self.reset_raw_state < RESET_STATE_FLOOR
        ):
            raise ValueError("RESET clipping mask does not match the raw states.")
        expected_set = full_tile_set(self.set_standard_normal)
        if not torch.equal(self.set_state, expected_set):
            raise ValueError("SET states do not implement the unclipped endpoint law.")
        if bool(torch.any(self.set_state <= 0.0)):
            raise ValueError(
                "An unclipped SET draw was non-positive; the population is invalid "
                "rather than silently clipped."
            )
        if bool(torch.any(self.set_state <= self.reset_state)):
            raise ValueError(
                "An unclipped SET draw did not exceed its RESET endpoint; the "
                "population is invalid rather than silently repaired."
            )

    @property
    def devices(self) -> int:
        return int(self.reset_state.numel())

    def report(self) -> Mapping[str, Any]:
        """Return distribution, clipping, pairing, and dynamic-range evidence."""

        dynamic_range = self.set_state / self.reset_state
        return {
            "schema": "ebl.mnist_relu_drn.hfo2_figure6_endpoint_population",
            "schema_version": 1,
            "evidence_class": "analyst_defined_figure6_informed_synthetic_control",
            "regime": self.regime,
            "pairing_policy": (
                "independent_gaussian_quantiles"
                if self.regime == INDEPENDENT_ENDPOINTS
                else "rank_matched_gaussian_quantiles"
            ),
            "assignment_seed": self.assignment_seed,
            "devices": self.devices,
            "state_coordinate": "unitless_positive_endpoint_state",
            "full_tile_RESET": {
                "formula": "max(0.05, 0.1 + sigma_RESET*z_RESET)",
                "raw_center": RESET_STATE_CENTER,
                "raw_standard_deviation": HFO2_FIGURE6_RESET_STATE_STD,
                "lower_clip": RESET_STATE_FLOOR,
                "upper_clip": None,
                "clipped_devices": int(self.reset_clipped.sum().item()),
                "clipped_fraction": float(self.reset_clipped.double().mean().item()),
                "raw_summary": _summary(self.reset_raw_state),
                "realized_summary": _summary(self.reset_state),
            },
            "full_tile_SET": {
                "formula": "2.0 + 0.4295*z_SET",
                "center": SET_STATE_CENTER,
                "standard_deviation": HFO2_SET_STATE_STD,
                "lower_clip": None,
                "upper_clip": None,
                "realized_summary": _summary(self.set_state),
            },
            "figure6_reset_to_set_standard_deviation_ratio": (
                FIGURE6_RESET_TO_SET_STD_RATIO
            ),
            "dynamic_range_SET_over_RESET": _summary(dynamic_range),
            "devices_at_least_5x": int((dynamic_range >= 5.0).sum().item()),
            "all_devices_at_least_5x": bool(torch.all(dynamic_range >= 5.0)),
            "reset_state_sha256": _tensor_sha256(self.reset_state),
            "set_state_sha256": _tensor_sha256(self.set_state),
            "stock_aihwkit_preset_modified": False,
        }


def sample_hfo2_figure6_endpoint_population(
    *,
    devices: int,
    assignment_seed: int,
    regime: str = INDEPENDENT_ENDPOINTS,
) -> HfO2Figure6EndpointPopulation:
    """Sample the requested full-RESET/full-SET distributions reproducibly.

    The independent regime uses operation-specific random streams.  The
    rank-matched regime preserves those exact finite marginal samples and
    pairs their ascending ranks.  Neither regime clips, redraws, or otherwise
    repairs SET states.
    """

    if isinstance(devices, bool) or not isinstance(devices, int) or devices < 1:
        raise ValueError("Expected devices to be a positive integer.")
    if isinstance(assignment_seed, bool) or not isinstance(assignment_seed, int):
        raise ValueError("Expected assignment_seed to be an integer.")
    if regime not in ENDPOINT_REGIMES:
        raise ValueError(f"Expected regime in {ENDPOINT_REGIMES!r}.")

    reset_z = _normal_vector(
        devices=devices,
        seed=_derived_seed(assignment_seed, FULL_TILE_RESET),
    )
    set_z = _normal_vector(
        devices=devices,
        seed=_derived_seed(assignment_seed, FULL_TILE_SET),
    )
    if regime == RANK_MATCHED_ENDPOINTS:
        set_z = _match_ranks(reset_z, set_z)

    reset, reset_raw, reset_clipped = full_tile_reset(reset_z)
    set_state = full_tile_set(set_z)
    return HfO2Figure6EndpointPopulation(
        regime=regime,
        assignment_seed=assignment_seed,
        reset_standard_normal=reset_z,
        set_standard_normal=set_z,
        reset_raw_state=reset_raw,
        reset_state=reset,
        set_state=set_state,
        reset_clipped=reset_clipped,
    )


__all__ = [
    "ENDPOINT_REGIMES",
    "FIGURE6_RESET_STD_US",
    "FIGURE6_RESET_TO_SET_STD_RATIO",
    "FIGURE6_SET_STD_US",
    "FULL_TILE_RESET",
    "FULL_TILE_SET",
    "HFO2_FIGURE6_RESET_STATE_STD",
    "HFO2_SET_STATE_STD",
    "INDEPENDENT_ENDPOINTS",
    "RANK_MATCHED_ENDPOINTS",
    "RESET_STATE_CENTER",
    "RESET_STATE_FLOOR",
    "SET_STATE_CENTER",
    "HfO2Figure6EndpointPopulation",
    "full_tile_reset",
    "full_tile_set",
    "full_tile_RESET",
    "full_tile_SET",
    "sample_hfo2_figure6_endpoint_population",
]
