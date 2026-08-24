"""IBM OM ReRAM modifiers for off-chip HWA and pulse-resolved deployment.

The clean DRN tensors remain the optimizer-owned master weights. A compact
modifier samples one independently programmed endpoint for each minibatch;
the pulse-resolved modifier starts the same fixed identities at their sampled
RESET bounds and runs the declared adaptive program-and-verify controller.
Both executions apply the apparent endpoint to the forward pass and retain the
hidden persistent endpoint as the state from which a later device update must
continue, matching the AIHWKit OM device contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterator

import numpy as np
import torch

from experiments.artifacts import sha256_file
from experiments.reram_program_verify.hwa_model import (
    CONDITION_KEY,
    SCHEMA as DEVICE_MODEL_SCHEMA,
    SCHEMA_VERSION as DEVICE_MODEL_SCHEMA_VERSION,
)
from model.resistive.builders import ParameterBinding
from model.variable.parameter import DenseWeight
from training.ibm_reram_endpoint_model import sample_ibm_reram_endpoints
from training.ibm_reram_program_verify import (
    ControllerSettings,
    OM_PRESET,
    PUBLISHED_CORRUPT_PROBABILITY,
    PopulationStepEstimator,
    ProgramVerifyResult,
    derive_seed,
    run_program_verify,
)


_STATE_VERSION = 2
_EXECUTIONS = ("compact_endpoint", "pulse_resolved")
_CORRUPTION_POLICIES = ("counterfactual_repaired", "published")
_TARGET_MAPPINGS = (
    "literal_global",
    "dual_rail_quad_common_window",
    "differential_pair_common_window",
    "cell_aware_exact_bounds_quad",
)
_CELL_AWARE_MODES = ("continuous", "quantized_9_level")
_STREAMS = ("train", "evaluation")
_EVALUATION_STREAM_XOR = 0x4F4D5F50565F4556
_ROOT = Path(__file__).resolve().parents[1]
_ARRAY_POPULATION_SCHEMA = "ebl.ibm_reram.om_array_population"
_ARRAY_POPULATION_SCHEMA_VERSION = 1
_ARRAY_SAMPLING_RECEIPT_SCHEMA = "ebl.ibm_reram.om_array_population_receipt"
_ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION = 1
_REQUIRED_AIHWKIT_VERSION = "1.1.0"
_MAX_EMPTY_COMMON_WINDOW_QUADS = 20
_MAX_EMPTY_COMMON_WINDOW_PAIRS = 20
IBM_RERAM_ENDPOINT_APPLICATION_POLICY = (
    "aihwkit_apparent_forward_persistent_update_state"
)


def _normalize_dual_rail_layouts(
    value: Mapping[str, str] | Sequence[Sequence[str]] | None,
) -> tuple[tuple[str, str], ...] | None:
    if value is None:
        return None
    items = value.items() if isinstance(value, Mapping) else value
    try:
        normalized = tuple(sorted((str(item[0]), str(item[1])) for item in items))
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(
            "Expected dual_rail_layout_by_parameter to be null or a mapping "
            "from stable parameter keys to 'halves' or 'paired'."
        ) from error
    if (
        not normalized
        or len({key for key, _layout in normalized}) != len(normalized)
        or any(not key or layout not in {"halves", "paired"} for key, layout in normalized)
    ):
        raise ValueError(
            "Expected dual_rail_layout_by_parameter to map unique non-empty "
            "parameter keys to 'halves' or 'paired'. Provided value: "
            f"{normalized!r}."
        )
    return normalized


@dataclass(frozen=True)
class IbmReramHwaConfig:
    """Exact OM cap-128 programming intervention for one modifier."""

    execution: str
    assignment_seed: int
    endpoint_seed: int
    corruption_policy: str
    noisy_evaluation: bool
    endpoint_policy: str = "clip_0_1"
    target_out_of_support: str = "error"
    preset: str = OM_PRESET
    controller: str = "adaptive"
    start_protocol: str = "lower_to_target"
    tolerance_step_ratio: float = 0.5
    maximum_program_pulses: int = 128
    target_mapping: str = "literal_global"
    dual_rail_layout_by_parameter: tuple[tuple[str, str], ...] | None = None
    common_window_margin_fraction: float = 0.0
    cell_aware_mode: str | None = None
    cell_aware_signed_levels: int | None = None
    forward_logit_gain: float | None = None

    def __post_init__(self) -> None:
        if self.execution not in _EXECUTIONS:
            raise ValueError(
                f"Expected execution to be one of {_EXECUTIONS!r}. "
                f"Provided value: {self.execution!r}."
            )
        if self.corruption_policy not in _CORRUPTION_POLICIES:
            raise ValueError(
                "Expected corruption_policy to be 'counterfactual_repaired' "
                f"or 'published'. Provided value: {self.corruption_policy!r}."
            )
        for name in ("assignment_seed", "endpoint_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
                raise ValueError(
                    f"Expected {name} to be an integer in [0, 2**63). "
                    f"Provided value: {value!r}."
                )
        exact = {
            "preset": (self.preset, OM_PRESET),
            "controller": (self.controller, "adaptive"),
            "start_protocol": (self.start_protocol, "lower_to_target"),
            "tolerance_step_ratio": (self.tolerance_step_ratio, 0.5),
            "maximum_program_pulses": (self.maximum_program_pulses, 128),
            "endpoint_policy": (self.endpoint_policy, "clip_0_1"),
            "target_out_of_support": (self.target_out_of_support, "error"),
        }
        mismatches = {
            name: {"expected": expected, "provided": provided}
            for name, (provided, expected) in exact.items()
            if provided != expected
        }
        if mismatches:
            raise ValueError(
                "Expected the declared IBM OM adaptive lower-from-RESET cap-128 "
                f"protocol. Provided value: {mismatches!r}."
            )
        if not isinstance(self.noisy_evaluation, bool):
            raise ValueError("Expected noisy_evaluation to be a boolean.")
        if self.target_mapping not in _TARGET_MAPPINGS:
            raise ValueError(
                f"Expected target_mapping to be one of {_TARGET_MAPPINGS!r}. "
                f"Provided value: {self.target_mapping!r}."
            )
        layouts = _normalize_dual_rail_layouts(
            self.dual_rail_layout_by_parameter
        )
        object.__setattr__(self, "dual_rail_layout_by_parameter", layouts)
        margin = self.common_window_margin_fraction
        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(float(margin))
            or not 0.0 <= float(margin) < 0.5
        ):
            raise ValueError(
                "Expected common_window_margin_fraction to be finite in "
                f"[0, 0.5). Provided value: {margin!r}."
            )
        object.__setattr__(self, "common_window_margin_fraction", float(margin))
        if self.forward_logit_gain is not None:
            gain = self.forward_logit_gain
            if (
                isinstance(gain, bool)
                or not isinstance(gain, (int, float))
                or not math.isfinite(float(gain))
                or float(gain) <= 0.0
            ):
                raise ValueError(
                    "Expected forward_logit_gain to be null or a finite "
                    f"positive number. Provided value: {gain!r}."
                )
            object.__setattr__(self, "forward_logit_gain", float(gain))
        if self.target_mapping in {
            "dual_rail_quad_common_window",
            "cell_aware_exact_bounds_quad",
        }:
            if layouts is None:
                raise ValueError(
                    "Expected dual_rail_layout_by_parameter for "
                    f"{self.target_mapping}."
                )
            if self.target_mapping == "cell_aware_exact_bounds_quad":
                if float(margin) != 0.0:
                    raise ValueError(
                        "Expected cell_aware_exact_bounds_quad to use zero "
                        "common_window_margin_fraction."
                    )
                if self.cell_aware_mode not in _CELL_AWARE_MODES:
                    raise ValueError(
                        "Expected cell_aware_mode to be 'continuous' or "
                        "'quantized_9_level' for "
                        "cell_aware_exact_bounds_quad. Provided value: "
                        f"{self.cell_aware_mode!r}."
                    )
                if self.cell_aware_signed_levels != 9:
                    raise ValueError(
                        "Expected cell_aware_signed_levels to equal 9 for "
                        "cell_aware_exact_bounds_quad. Provided value: "
                        f"{self.cell_aware_signed_levels!r}."
                    )
        elif self.target_mapping == "differential_pair_common_window":
            if layouts is not None:
                raise ValueError(
                    "Expected differential_pair_common_window to have null "
                    "dual_rail_layout_by_parameter; pairing is fixed by the "
                    "canonical adjacent plus/minus binding catalog."
                )
        elif layouts is not None or float(margin) != 0.0:
            raise ValueError(
                "Expected literal_global target mapping to have null "
                "dual_rail_layout_by_parameter and zero "
                "common_window_margin_fraction."
            )
        if self.target_mapping != "cell_aware_exact_bounds_quad" and (
            self.cell_aware_mode is not None
            or self.cell_aware_signed_levels is not None
        ):
            raise ValueError(
                "Expected cell-aware fields to be null unless target_mapping "
                "is cell_aware_exact_bounds_quad."
            )


@dataclass(frozen=True)
class IbmReramArrayPopulation:
    """One fixed, flattened physical-cell assignment across named DRN tensors."""

    assignment_seed: int
    corruption_policy: str
    binding_keys: tuple[str, ...]
    binding_shapes: tuple[tuple[int, ...], ...]
    binding_sampling_seeds: tuple[int, ...]
    donor_sampling_seeds: tuple[int, ...]
    nominal_dw_min: float
    dw_min_std: float
    write_noise_std: float
    max_bound: torch.Tensor
    min_bound: torch.Tensor
    dwmin_up: torch.Tensor
    dwmin_down: torch.Tensor
    reference: torch.Tensor
    corrupt: torch.Tensor
    published_corrupt: torch.Tensor
    fingerprint: str
    aihwkit_version: str

    def __post_init__(self) -> None:
        size = int(self.max_bound.numel())
        if size < 1 or sum(math.prod(shape) for shape in self.binding_shapes) != size:
            raise ValueError("Expected population tensors to match binding shapes.")
        if len(self.binding_keys) != len(self.binding_shapes) or len(set(self.binding_keys)) != len(
            self.binding_keys
        ):
            raise ValueError("Expected unique population binding keys and matching shapes.")
        for name in (
            "max_bound",
            "min_bound",
            "dwmin_up",
            "dwmin_down",
            "reference",
        ):
            value = getattr(self, name)
            if value.shape != (size,) or value.dtype != torch.float32 or not bool(
                torch.all(torch.isfinite(value))
            ):
                raise ValueError(f"Expected {name} to be a finite float32 vector.")
        for name in ("corrupt", "published_corrupt"):
            value = getattr(self, name)
            if value.shape != (size,) or value.dtype != torch.bool:
                raise ValueError(f"Expected {name} to be a boolean vector.")
        if bool(torch.any(self.max_bound < self.min_bound)):
            raise ValueError("Expected sampled maximum bounds not below minimum bounds.")
        detected = (
            (torch.abs(self.max_bound - self.min_bound) <= 1e-12)
            & (self.dwmin_up == 0.0)
            & (self.dwmin_down == 0.0)
        )
        if not torch.equal(detected, self.corrupt):
            raise ValueError("Expected final corrupt flags to match collapsed zero-step cells.")
        if self.corruption_policy == "published" and not torch.equal(
            self.corrupt, self.published_corrupt
        ):
            raise ValueError("Expected published policy to retain the sampled corrupt mask.")
        if self.corruption_policy == "counterfactual_repaired" and bool(
            torch.any(self.corrupt)
        ):
            raise ValueError("Expected repaired policy to replace every corrupt cell.")

    @property
    def size(self) -> int:
        return int(self.max_bound.numel())

    @property
    def logical_min(self) -> torch.Tensor:
        return self.min_bound - self.reference

    @property
    def logical_max(self) -> torch.Tensor:
        return self.max_bound - self.reference

    def to(self, device: torch.device | str) -> "IbmReramArrayPopulation":
        target = torch.device(device)
        return IbmReramArrayPopulation(
            assignment_seed=self.assignment_seed,
            corruption_policy=self.corruption_policy,
            binding_keys=self.binding_keys,
            binding_shapes=self.binding_shapes,
            binding_sampling_seeds=self.binding_sampling_seeds,
            donor_sampling_seeds=self.donor_sampling_seeds,
            nominal_dw_min=self.nominal_dw_min,
            dw_min_std=self.dw_min_std,
            write_noise_std=self.write_noise_std,
            max_bound=self.max_bound.to(target),
            min_bound=self.min_bound.to(target),
            dwmin_up=self.dwmin_up.to(target),
            dwmin_down=self.dwmin_down.to(target),
            reference=self.reference.to(target),
            corrupt=self.corrupt.to(target),
            published_corrupt=self.published_corrupt.to(target),
            fingerprint=self.fingerprint,
            aihwkit_version=self.aihwkit_version,
        )

    def tensor_state(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name).detach().cpu().clone()
            for name in (
                "max_bound",
                "min_bound",
                "dwmin_up",
                "dwmin_down",
                "reference",
                "corrupt",
                "published_corrupt",
            )
        }


def _selected_array_population(
    population: IbmReramArrayPopulation,
    selection: torch.Tensor,
) -> IbmReramArrayPopulation:
    """Build an internal physical-identity slice for exact fallback writes."""

    mask = torch.as_tensor(
        selection,
        device=population.max_bound.device,
        dtype=torch.bool,
    )
    if mask.shape != (population.size,) or not bool(torch.any(mask)):
        raise ValueError(
            "Expected a non-empty boolean selection matching the IBM OM "
            "population."
        )
    indices = torch.where(mask)[0].detach().cpu().to(dtype=torch.int64)
    digest = sha256()
    digest.update(population.fingerprint.encode())
    digest.update(indices.numpy().tobytes())
    count = int(indices.numel())
    return IbmReramArrayPopulation(
        assignment_seed=population.assignment_seed,
        corruption_policy=population.corruption_policy,
        binding_keys=("__pulse_resolved_fallback__",),
        binding_shapes=((count, 1),),
        binding_sampling_seeds=(population.assignment_seed,),
        donor_sampling_seeds=(population.assignment_seed,),
        nominal_dw_min=population.nominal_dw_min,
        dw_min_std=population.dw_min_std,
        write_noise_std=population.write_noise_std,
        max_bound=population.max_bound[mask].clone(),
        min_bound=population.min_bound[mask].clone(),
        dwmin_up=population.dwmin_up[mask].clone(),
        dwmin_down=population.dwmin_down[mask].clone(),
        reference=population.reference[mask].clone(),
        corrupt=population.corrupt[mask].clone(),
        published_corrupt=population.published_corrupt[mask].clone(),
        fingerprint=digest.hexdigest(),
        aihwkit_version=population.aihwkit_version,
    )


def _tensor_summary(value: torch.Tensor) -> dict[str, float]:
    flattened = value.detach().reshape(-1)
    if flattened.numel() == 0:
        return {"minimum": 0.0, "mean": 0.0, "maximum": 0.0}
    return {
        "minimum": float(flattened.min().item()),
        "mean": float(flattened.mean().item()),
        "maximum": float(flattened.max().item()),
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor content together with its exact dtype and shape."""

    cpu = value.detach().contiguous().cpu()
    digest = sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _cell_aware_tensor_summary(value: torch.Tensor) -> dict[str, float]:
    """Return device-independent diagnostics for immutable oracle replay."""

    flattened = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return {
        "minimum": float(flattened.min().item()),
        "maximum": float(flattened.max().item()),
        "mean": float(flattened.mean().item()),
    }


def _quad_axes(
    shape: tuple[int, ...],
    layout: str,
    *,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Return canonical source/destination rail axes for a four-cell quad."""

    if len(shape) != 2 or shape[0] % 2 or shape[1] % 2:
        raise ValueError(
            "Expected four-cell dual-rail bindings to be even-by-even rank-2 "
            f"tensors. Provided shape: {shape!r}."
        )
    if layout not in {"halves", "paired"}:
        raise ValueError(
            "Expected a canonical dual-rail layout ('halves' or 'paired'). "
            f"Provided value: {layout!r}."
        )
    input_count = shape[0] // 2
    output_count = shape[1] // 2
    plus_rows = torch.arange(input_count, device=device)
    minus_rows = plus_rows + input_count
    if layout == "halves":
        plus_columns = torch.arange(output_count, device=device)
        minus_columns = plus_columns + output_count
    else:
        plus_columns = torch.arange(output_count, device=device) * 2
        minus_columns = plus_columns + 1
    return (plus_rows, minus_rows), (plus_columns, minus_columns)


def _canonical_differential_pair_layout(
    binding_keys: Sequence[str],
    binding_shapes: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, str, str, tuple[int, int]], ...]:
    """Validate the canonical adjacent G+/G- catalog and return its pairs."""

    keys = tuple(binding_keys)
    shapes = tuple(tuple(shape) for shape in binding_shapes)
    if not keys or len(keys) != len(shapes) or len(keys) % 2:
        raise ValueError(
            "Expected differential-pair common-window bindings to be a "
            "non-empty even-length canonical adjacent plus/minus catalog. "
            f"Provided value: keys={keys!r}, shapes={shapes!r}."
        )
    pairs = []
    for pair_index in range(len(keys) // 2):
        plus_position = 2 * pair_index
        minus_position = plus_position + 1
        plus_key = keys[plus_position]
        minus_key = keys[minus_position]
        expected_plus = f"base.conductance_plus.{pair_index}"
        expected_minus = f"base.conductance_minus.{pair_index}"
        plus_shape = shapes[plus_position]
        minus_shape = shapes[minus_position]
        if plus_key != expected_plus or minus_key != expected_minus:
            raise ValueError(
                "Expected differential-pair common-window bindings in "
                "canonical adjacent plus/minus key and suffix order. "
                f"Provided pair {pair_index}: keys="
                f"({plus_key!r}, {minus_key!r}); expected="
                f"({expected_plus!r}, {expected_minus!r})."
            )
        if (
            len(plus_shape) != 2
            or any(value < 1 for value in plus_shape)
            or minus_shape != plus_shape
        ):
            raise ValueError(
                "Expected each canonical differential plus/minus pair to "
                "have identical non-empty rank-2 shapes. Provided pair "
                f"{pair_index}: plus_shape={plus_shape!r}, "
                f"minus_shape={minus_shape!r}."
            )
        pairs.append(
            (
                pair_index,
                plus_key,
                minus_key,
                (int(plus_shape[0]), int(plus_shape[1])),
            )
        )
    return tuple(pairs)


def _validate_differential_pair_bindings(
    bindings: Sequence[ParameterBinding],
) -> tuple[tuple[int, str, str, tuple[int, int]], ...]:
    """Validate roles and tensors in addition to the stable catalog layout."""

    selected = tuple(bindings)
    pairs = _canonical_differential_pair_layout(
        tuple(binding.key for binding in selected),
        tuple(tuple(binding.state.shape) for binding in selected),
    )
    for pair_index, _plus_key, _minus_key, _shape in pairs:
        plus = selected[2 * pair_index]
        minus = selected[2 * pair_index + 1]
        if (
            plus.role != "conductance_plus"
            or minus.role != "conductance_minus"
            or not isinstance(plus.parameter, DenseWeight)
            or not isinstance(minus.parameter, DenseWeight)
        ):
            raise ValueError(
                "Expected canonical differential-pair bindings to expose "
                "DenseWeight conductance_plus/conductance_minus roles. "
                f"Provided pair {pair_index}: plus_role={plus.role!r}, "
                f"minus_role={minus.role!r}, "
                f"plus_type={type(plus.parameter).__name__!r}, "
                f"minus_type={type(minus.parameter).__name__!r}."
            )
        raw_bounds = (
            plus.parameter.min_cond,
            plus.parameter.max_cond,
            minus.parameter.min_cond,
            minus.parameter.max_cond,
        )
        try:
            plus_min, plus_max, minus_min, minus_max = tuple(
                float(value) for value in raw_bounds
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Expected each canonical differential plus/minus pair to "
                "have finite, increasing, exactly matching conductance "
                f"bounds. Provided pair {pair_index}: bounds={raw_bounds!r}."
            ) from error
        if (
            not all(
                math.isfinite(value)
                for value in (plus_min, plus_max, minus_min, minus_max)
            )
            or not plus_min < plus_max
            or not minus_min < minus_max
            or plus_min != minus_min
            or plus_max != minus_max
        ):
            raise ValueError(
                "Expected each canonical differential plus/minus pair to "
                "have finite, increasing, exactly matching conductance "
                f"bounds. Provided pair {pair_index}: "
                f"plus_bounds=({plus_min!r}, {plus_max!r}), "
                f"minus_bounds=({minus_min!r}, {minus_max!r})."
            )
    return pairs


def _map_cell_aware_exact_bounds_quad(
    global_targets: torch.Tensor,
    population: IbmReramArrayPopulation,
    *,
    layouts: tuple[tuple[str, str], ...],
    mode: str,
    signed_levels: int,
    materialize_evidence: bool = True,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any] | None]:
    """Apply the declared per-cell affine exact-bounds oracle codebook."""

    device = global_targets.device
    dtype = global_targets.dtype
    cell_logical_min = population.logical_min.to(device=device, dtype=dtype)
    cell_logical_max = population.logical_max.to(device=device, dtype=dtype)
    raw_lower = (cell_logical_min + 1.0) / 2.0
    raw_upper = (cell_logical_max + 1.0) / 2.0
    usable_lower = raw_lower.clamp_min(0.0)
    usable_upper = raw_upper.clamp_max(1.0)
    if bool(torch.any(usable_upper < usable_lower)):
        invalid = usable_upper < usable_lower
        raise ValueError(
            "Expected every exact-bound cell to intersect the DRN's global "
            "normalized conductance coordinate. Provided invalid cells: "
            f"{int(invalid.sum().item())}."
        )
    usable_span = usable_upper - usable_lower
    layout_by_key = dict(layouts)
    mapped = torch.empty_like(global_targets)
    shadow_values: list[torch.Tensor] = []
    code_coordinates: list[torch.Tensor] = []
    code_indices: list[torch.Tensor] = []
    baseline_contrasts: list[torch.Tensor] = []
    mapped_contrasts: list[torch.Tensor] = []
    requested_signs: list[torch.Tensor] = []
    realized_signs: list[torch.Tensor] = []
    parameter_reports: dict[str, Any] = {}
    offset = 0
    for key, shape in zip(population.binding_keys, population.binding_shapes):
        count = math.prod(shape)
        source = global_targets[offset : offset + count].reshape(shape)
        target = mapped[offset : offset + count].reshape(shape)
        lower = usable_lower[offset : offset + count].reshape(shape)
        upper = usable_upper[offset : offset + count].reshape(shape)
        span = usable_span[offset : offset + count].reshape(shape)
        row_groups, column_groups = _quad_axes(
            shape,
            layout_by_key[key],
            device=device,
        )
        source_quad = torch.stack(
            tuple(
                source[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        )
        shadow = (
            source_quad[0]
            - source_quad[1]
            - source_quad[2]
            + source_quad[3]
        ) / 2.0
        continuous_code = (4.0 * shadow).clamp(-4.0, 4.0)
        if mode == "quantized_9_level":
            code = torch.sign(continuous_code) * torch.floor(
                torch.abs(continuous_code) + 0.5
            )
            code = code.clamp(-4.0, 4.0)
            integer_code = code.to(dtype=torch.int8)
            indices = (integer_code.to(torch.int16) + 4).to(torch.uint8)
        else:
            code = continuous_code
            integer_code = torch.empty(0, dtype=torch.int8, device=device)
            indices = torch.empty(0, dtype=torch.uint8, device=device)

        signs = (1.0, -1.0, -1.0, 1.0)
        lower_quad = []
        target_quad = []
        # Keep the order (++,+-,-+,--) identical to the logical reconstruction.
        for sign, rows, columns in zip(
            signs,
            (row_groups[0], row_groups[0], row_groups[1], row_groups[1]),
            (
                column_groups[0],
                column_groups[1],
                column_groups[0],
                column_groups[1],
            ),
        ):
            cell_lower = lower[rows[:, None], columns]
            cell_span = span[rows[:, None], columns]
            fraction = (float(sign) * code).clamp_min(0.0) / 4.0
            cell_target = cell_lower + fraction * cell_span
            target[rows[:, None], columns] = cell_target
            if materialize_evidence:
                lower_quad.append(cell_lower)
                target_quad.append(cell_target)
        if not materialize_evidence:
            if mode == "quantized_9_level":
                code_indices.append(indices.reshape(-1))
            offset += count
            continue
        lower_quad_tensor = torch.stack(lower_quad)
        target_quad_tensor = torch.stack(target_quad)
        baseline_contrast = (
            lower_quad_tensor[0]
            - lower_quad_tensor[1]
            - lower_quad_tensor[2]
            + lower_quad_tensor[3]
        ) / 2.0
        mapped_contrast = (
            target_quad_tensor[0]
            - target_quad_tensor[1]
            - target_quad_tensor[2]
            + target_quad_tensor[3]
        ) / 2.0
        requested_sign = torch.sign(code)
        realized_sign = torch.sign(mapped_contrast)
        nonzero = requested_sign != 0
        sign_flip = nonzero & (requested_sign != realized_sign)
        parameter_report: dict[str, Any] = {
            "dual_rail_layout": layout_by_key[key],
            "devices": count,
            "quad_count": int(shadow.numel()),
            "usable_lower": _cell_aware_tensor_summary(lower),
            "usable_upper": _cell_aware_tensor_summary(upper),
            "usable_span": _cell_aware_tensor_summary(span),
            "shadow_logical_weight": _cell_aware_tensor_summary(shadow),
            "code_coordinate": _cell_aware_tensor_summary(code),
            "cell_specific_baseline_contrast": _cell_aware_tensor_summary(
                baseline_contrast
            ),
            "nominal_mapped_logical_contrast": _cell_aware_tensor_summary(
                mapped_contrast
            ),
            "nonzero_quad_count": int(nonzero.sum().item()),
            "logical_sign_flip_count": int(sign_flip.sum().item()),
            "logical_sign_flip_fraction_nonzero": (
                float(sign_flip.sum().item()) / int(nonzero.sum().item())
                if bool(torch.any(nonzero))
                else 0.0
            ),
            "shadow_logical_weight_sha256": _tensor_sha256(shadow),
            "code_coordinate_sha256": _tensor_sha256(code),
            "mapped_target_sha256": _tensor_sha256(target),
        }
        if mode == "quantized_9_level":
            counts = torch.bincount(indices.reshape(-1).to(torch.int64), minlength=9)
            parameter_report["signed_code_sha256"] = _tensor_sha256(
                integer_code
            )
            parameter_report["code_index_sha256"] = _tensor_sha256(indices)
            parameter_report["code_index_counts"] = {
                str(index): int(value)
                for index, value in enumerate(counts.tolist())
            }
        parameter_reports[key] = parameter_report
        shadow_values.append(shadow.reshape(-1))
        code_coordinates.append(code.reshape(-1))
        if mode == "quantized_9_level":
            code_indices.append(indices.reshape(-1))
        baseline_contrasts.append(baseline_contrast.reshape(-1))
        mapped_contrasts.append(mapped_contrast.reshape(-1))
        requested_signs.append(requested_sign.reshape(-1))
        realized_signs.append(realized_sign.reshape(-1))
        offset += count
    if offset != population.size:  # pragma: no cover - population validates it
        raise RuntimeError("Expected cell-aware mapping to cover every cell.")

    if not materialize_evidence:
        support = {
            "below_lower_bound": int((mapped < usable_lower).sum().item()),
            "above_upper_bound": int((mapped > usable_upper).sum().item()),
            "inside_bounds": int(
                ((mapped >= usable_lower) & (mapped <= usable_upper)).sum().item()
            ),
        }
        report: dict[str, Any] = {
            "target_mapping": "cell_aware_exact_bounds_quad",
            "cell_aware_mode": mode,
            "cell_aware_signed_levels": signed_levels,
            "rounding": (
                "round_half_away_from_zero"
                if mode == "quantized_9_level"
                else None
            ),
            "oracle": True,
            "hidden_device_bounds_consumed_by_target_mapper": True,
            "post_mapping_clipping": False,
            "post_programming_rounding": False,
            "population_fingerprint": population.fingerprint,
            "assignment_seed": population.assignment_seed,
            "devices": population.size,
            "quad_count": sum(
                math.prod(shape) // 4 for shape in population.binding_shapes
            ),
            "mapped_target_support": support,
            "mapped_target_outside_0_1": int(
                ((mapped < 0.0) | (mapped > 1.0)).sum().item()
            ),
            "evidence_materialization": "deferred_to_evaluation_stream",
        }
        if mode == "quantized_9_level":
            code_index_all = torch.cat(code_indices)
            counts = torch.bincount(code_index_all.to(torch.int64), minlength=9)
            report["code_index_counts"] = {
                str(index): int(value)
                for index, value in enumerate(counts.tolist())
            }
        return mapped, report, None

    shadow_all = torch.cat(shadow_values)
    code_all = torch.cat(code_coordinates)
    baseline_contrast_all = torch.cat(baseline_contrasts)
    mapped_contrast_all = torch.cat(mapped_contrasts)
    requested_sign_all = torch.cat(requested_signs)
    realized_sign_all = torch.cat(realized_signs)
    nonzero_all = requested_sign_all != 0
    sign_flip_all = nonzero_all & (requested_sign_all != realized_sign_all)
    support = {
        "below_lower_bound": int((mapped < usable_lower).sum().item()),
        "above_upper_bound": int((mapped > usable_upper).sum().item()),
        "inside_bounds": int(
            ((mapped >= usable_lower) & (mapped <= usable_upper)).sum().item()
        ),
    }
    report: dict[str, Any] = {
        "target_mapping": "cell_aware_exact_bounds_quad",
        "cell_aware_mode": mode,
        "cell_aware_signed_levels": signed_levels,
        "rounding": (
            "round_half_away_from_zero"
            if mode == "quantized_9_level"
            else None
        ),
        "oracle": True,
        "hidden_device_bounds_consumed_by_target_mapper": True,
        "bound_coordinate": (
            "intersection_of_exact_cell_logical_bounds_with_global_0_1"
        ),
        "target_rule": "per_cell_lower_plus_sign_selected_fraction_of_cell_span",
        "post_mapping_clipping": False,
        "post_programming_rounding": False,
        "population_fingerprint": population.fingerprint,
        "assignment_seed": population.assignment_seed,
        "corruption_policy": population.corruption_policy,
        "devices": population.size,
        "quad_count": int(shadow_all.numel()),
        "common_window_grouping": None,
        "common_window_group_size": None,
        "common_window_group_count": 0,
        "common_window_empty_group_count": 0,
        "common_window_empty_quad_count": 0,
        "usable_lower": _cell_aware_tensor_summary(usable_lower),
        "usable_upper": _cell_aware_tensor_summary(usable_upper),
        "usable_span": _cell_aware_tensor_summary(usable_span),
        "zero_span_cell_count": int((usable_span == 0.0).sum().item()),
        "shadow_logical_weight": _cell_aware_tensor_summary(shadow_all),
        "code_coordinate": _cell_aware_tensor_summary(code_all),
        "cell_specific_baseline_contrast": _cell_aware_tensor_summary(
            baseline_contrast_all
        ),
        "nominal_mapped_logical_contrast": _cell_aware_tensor_summary(
            mapped_contrast_all
        ),
        "nonzero_quad_count": int(nonzero_all.sum().item()),
        "logical_sign_flip_count": int(sign_flip_all.sum().item()),
        "logical_sign_flip_fraction_nonzero": (
            float(sign_flip_all.sum().item()) / int(nonzero_all.sum().item())
            if bool(torch.any(nonzero_all))
            else 0.0
        ),
        "mapped_target_support": support,
        "global_target_outside_0_1": int(
            ((global_targets < 0.0) | (global_targets > 1.0)).sum().item()
        ),
        "mapped_target_outside_0_1": int(
            ((mapped < 0.0) | (mapped > 1.0)).sum().item()
        ),
        "hashes": {
            "global_source_target": _tensor_sha256(global_targets),
            "cell_logical_min": _tensor_sha256(cell_logical_min),
            "cell_logical_max": _tensor_sha256(cell_logical_max),
            "normalized_exact_lower": _tensor_sha256(raw_lower),
            "normalized_exact_upper": _tensor_sha256(raw_upper),
            "usable_lower": _tensor_sha256(usable_lower),
            "usable_upper": _tensor_sha256(usable_upper),
            "usable_span": _tensor_sha256(usable_span),
            "shadow_logical_weight": _tensor_sha256(shadow_all),
            "code_coordinate": _tensor_sha256(code_all),
            "mapped_target": _tensor_sha256(mapped),
        },
        "parameters": parameter_reports,
    }
    signed_code_all: torch.Tensor | None = None
    code_index_all: torch.Tensor | None = None
    if mode == "quantized_9_level":
        code_index_all = torch.cat(code_indices)
        signed_code_all = (code_index_all.to(torch.int16) - 4).to(torch.int8)
        counts = torch.bincount(code_index_all.to(torch.int64), minlength=9)
        report["signed_code_sha256"] = _tensor_sha256(signed_code_all)
        report["code_index_sha256"] = _tensor_sha256(code_index_all)
        report["code_index_counts"] = {
            str(index): int(value)
            for index, value in enumerate(counts.tolist())
        }
        report["hashes"]["signed_code"] = report["signed_code_sha256"]
        report["hashes"]["code_index"] = report["code_index_sha256"]

    artifact = {
        "schema": "ebl.ibm_reram.om_cell_aware_exact_bounds_codebook",
        "schema_version": 1,
        "target_mapping": "cell_aware_exact_bounds_quad",
        "cell_aware_mode": mode,
        "cell_aware_signed_levels": signed_levels,
        "oracle": True,
        "hidden_device_bounds_consumed_by_target_mapper": True,
        "population_fingerprint": population.fingerprint,
        "assignment_seed": population.assignment_seed,
        "corruption_policy": population.corruption_policy,
        "binding_keys": population.binding_keys,
        "binding_shapes": population.binding_shapes,
        "dual_rail_layout_by_parameter": layouts,
        "cell_logical_min": cell_logical_min.detach().cpu().clone(),
        "cell_logical_max": cell_logical_max.detach().cpu().clone(),
        "normalized_exact_lower": raw_lower.detach().cpu().clone(),
        "normalized_exact_upper": raw_upper.detach().cpu().clone(),
        "usable_lower": usable_lower.detach().cpu().clone(),
        "usable_upper": usable_upper.detach().cpu().clone(),
        "usable_span": usable_span.detach().cpu().clone(),
        "global_source_target": global_targets.detach().cpu().clone(),
        "shadow_logical_weight": shadow_all.detach().cpu().clone(),
        "code_coordinate": code_all.detach().cpu().clone(),
        "signed_code": (
            signed_code_all.detach().cpu().clone()
            if signed_code_all is not None
            else None
        ),
        "code_index": (
            code_index_all.detach().cpu().clone()
            if code_index_all is not None
            else None
        ),
        "requested_target": mapped.detach().cpu().clone(),
        "cell_specific_baseline_contrast": (
            baseline_contrast_all.detach().cpu().clone()
        ),
        "nominal_mapped_logical_contrast": (
            mapped_contrast_all.detach().cpu().clone()
        ),
        "requested_sign": requested_sign_all.detach().cpu().clone(),
        "nominal_mapped_sign": realized_sign_all.detach().cpu().clone(),
        "report": json.loads(json.dumps(report)),
    }
    return mapped, report, artifact


def map_ibm_reram_array_targets(
    global_targets: torch.Tensor,
    population: IbmReramArrayPopulation,
    *,
    target_mapping: str,
    dual_rail_layout_by_parameter: (
        Mapping[str, str] | Sequence[Sequence[str]] | None
    ),
    common_window_margin_fraction: float,
    cell_aware_mode: str | None = None,
    cell_aware_signed_levels: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Map clean global fractions onto one fixed IBM OM array assignment.

    The quad mapping uses the exact four physical identities assigned to the
    Cartesian product of a logical synapse's two source and two destination
    rails.  Corrupt identities are deliberately retained in the intersection;
    the published arm therefore keeps its collapsed/stuck-cell semantics,
    while the repaired arm naturally uses its sampled donor bounds.
    """

    if (
        not isinstance(global_targets, torch.Tensor)
        or not global_targets.is_floating_point()
        or global_targets.ndim != 1
        or global_targets.numel() != population.size
        or not bool(torch.all(torch.isfinite(global_targets)))
    ):
        raise ValueError(
            "Expected finite floating global_targets with shape "
            f"({population.size},). Provided value: "
            f"shape={getattr(global_targets, 'shape', None)!r}."
        )
    if target_mapping not in _TARGET_MAPPINGS:
        raise ValueError(
            f"Expected target_mapping to be one of {_TARGET_MAPPINGS!r}. "
            f"Provided value: {target_mapping!r}."
        )
    layouts = _normalize_dual_rail_layouts(dual_rail_layout_by_parameter)
    margin = common_window_margin_fraction
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(float(margin))
        or not 0.0 <= float(margin) < 0.5
    ):
        raise ValueError(
            "Expected common_window_margin_fraction to be finite in "
            f"[0, 0.5). Provided value: {margin!r}."
        )
    margin = float(margin)
    differential_pairs: tuple[
        tuple[int, str, str, tuple[int, int]], ...
    ] = ()
    if target_mapping in {
        "dual_rail_quad_common_window",
        "cell_aware_exact_bounds_quad",
    }:
        expected_keys = set(population.binding_keys)
        provided_keys = set(dict(layouts or ()))
        if provided_keys != expected_keys:
            raise ValueError(
                "Expected dual_rail_layout_by_parameter keys to equal the "
                "fixed IBM OM population binding keys. Provided value: "
                f"expected={sorted(expected_keys)!r}, "
                f"provided={sorted(provided_keys)!r}."
            )
        outside_global = (global_targets < 0.0) | (global_targets > 1.0)
        if bool(torch.any(outside_global)):
            offending = global_targets[outside_global]
            raise ValueError(
                "Expected quad-mapped clean conductance fractions inside "
                "[0, 1]. Provided extrema: "
                f"minimum={float(offending.min().item())}, "
                f"maximum={float(offending.max().item())}."
            )
        if target_mapping == "cell_aware_exact_bounds_quad":
            if margin != 0.0:
                raise ValueError(
                    "Expected cell_aware_exact_bounds_quad to use zero "
                    "common-window margin."
                )
            if cell_aware_mode not in _CELL_AWARE_MODES:
                raise ValueError(
                    "Expected cell_aware_mode to be 'continuous' or "
                    "'quantized_9_level'. Provided value: "
                    f"{cell_aware_mode!r}."
                )
            if cell_aware_signed_levels != 9:
                raise ValueError(
                    "Expected cell_aware_signed_levels to equal 9. Provided "
                    f"value: {cell_aware_signed_levels!r}."
                )
    elif target_mapping == "differential_pair_common_window":
        if layouts is not None:
            raise ValueError(
                "Expected differential_pair_common_window to have null "
                "layouts; pairing is fixed by the canonical adjacent "
                "plus/minus binding catalog."
            )
        differential_pairs = _canonical_differential_pair_layout(
            population.binding_keys,
            population.binding_shapes,
        )
        outside_global = (global_targets < 0.0) | (global_targets > 1.0)
        if bool(torch.any(outside_global)):
            offending = global_targets[outside_global]
            raise ValueError(
                "Expected differential-pair-mapped clean conductance "
                "fractions inside [0, 1]. Provided extrema: "
                f"minimum={float(offending.min().item())}, "
                f"maximum={float(offending.max().item())}."
            )
    elif layouts is not None or margin != 0.0:
        raise ValueError(
            "Expected literal_global target mapping to have null layouts and "
            "zero common-window margin."
        )
    if target_mapping != "cell_aware_exact_bounds_quad" and (
        cell_aware_mode is not None or cell_aware_signed_levels is not None
    ):
        raise ValueError(
            "Expected cell-aware fields to be null unless target_mapping is "
            "cell_aware_exact_bounds_quad."
        )

    if target_mapping == "cell_aware_exact_bounds_quad":
        assert layouts is not None
        assert cell_aware_mode is not None
        assert cell_aware_signed_levels is not None
        mapped, report, _artifact = _map_cell_aware_exact_bounds_quad(
            global_targets,
            population,
            layouts=layouts,
            mode=cell_aware_mode,
            signed_levels=cell_aware_signed_levels,
        )
        return mapped, report

    device = global_targets.device
    dtype = global_targets.dtype
    lower = ((population.logical_min.to(device=device, dtype=dtype) + 1.0) / 2.0)
    upper = ((population.logical_max.to(device=device, dtype=dtype) + 1.0) / 2.0)
    corrupt = population.corrupt.to(device=device)
    published_corrupt = population.published_corrupt.to(device=device)
    noncorrupt = ~corrupt

    def support_counts(
        targets: torch.Tensor,
        selection: torch.Tensor | None = None,
    ) -> dict[str, int]:
        selected = noncorrupt if selection is None else noncorrupt & selection
        return {
            "below_lower_bound": int((selected & (targets < lower)).sum().item()),
            "above_upper_bound": int((selected & (targets > upper)).sum().item()),
            "inside_bounds": int(
                (selected & (targets >= lower) & (targets <= upper)).sum().item()
            ),
        }

    global_support = support_counts(global_targets)
    report: dict[str, Any] = {
        "target_mapping": target_mapping,
        "common_window_margin_fraction": margin,
        "cell_aware_mode": cell_aware_mode,
        "cell_aware_signed_levels": cell_aware_signed_levels,
        "population_fingerprint": population.fingerprint,
        "corruption_policy": population.corruption_policy,
        "devices": population.size,
        "global_target_support": global_support,
        "global_target_outside_0_1": int(
            ((global_targets < 0.0) | (global_targets > 1.0)).sum().item()
        ),
        "parameters": {},
    }
    if target_mapping == "literal_global":
        mapped = global_targets.detach().clone()
        report.update(
            {
                "common_window_grouping": None,
                "common_window_group_size": None,
                "common_window_group_count": 0,
                "common_window_empty_group_count": 0,
                "corrupt_group_count": 0,
                "published_corrupt_group_count": 0,
                "differential_pair_binding": None,
                "nonempty_inner_common_span": None,
                "quad_count": 0,
                "common_window_empty_quad_count": 0,
                "corrupt_quad_count": 0,
                "published_corrupt_quad_count": 0,
                "raw_common_span": None,
                "inner_common_span": None,
                "mapped_target_below_lower_bound_nonempty_quad": 0,
                "mapped_target_above_upper_bound_nonempty_quad": 0,
                "mapped_target_below_lower_bound_empty_quad": 0,
                "mapped_target_above_upper_bound_empty_quad": 0,
                "mapped_target_below_lower_bound_nonempty_group": 0,
                "mapped_target_above_upper_bound_nonempty_group": 0,
                "mapped_target_below_lower_bound_empty_group": 0,
                "mapped_target_above_upper_bound_empty_group": 0,
                "mapped_target_support": support_counts(mapped),
            }
        )
        return mapped, report

    if target_mapping == "differential_pair_common_window":
        mapped = torch.empty_like(global_targets)
        report["parameter_pairs"] = {}
        provenance_pairs = []
        all_overlap: list[torch.Tensor] = []
        all_corrupt_pair: list[torch.Tensor] = []
        all_published_corrupt_pair: list[torch.Tensor] = []
        all_raw_span: list[torch.Tensor] = []
        all_inner_span: list[torch.Tensor] = []
        all_nonempty_cell: list[torch.Tensor] = []
        offset = 0
        for pair_index, plus_key, minus_key, shape in differential_pairs:
            count = math.prod(shape)
            plus_slice = slice(offset, offset + count)
            minus_slice = slice(offset + count, offset + 2 * count)
            plus_target = global_targets[plus_slice].reshape(shape)
            minus_target = global_targets[minus_slice].reshape(shape)
            plus_lower = lower[plus_slice].reshape(shape)
            minus_lower = lower[minus_slice].reshape(shape)
            plus_upper = upper[plus_slice].reshape(shape)
            minus_upper = upper[minus_slice].reshape(shape)
            plus_corrupt = corrupt[plus_slice].reshape(shape)
            minus_corrupt = corrupt[minus_slice].reshape(shape)
            plus_published_corrupt = published_corrupt[plus_slice].reshape(shape)
            minus_published_corrupt = published_corrupt[minus_slice].reshape(shape)

            # Each logical branch coordinate uses exactly the two identities
            # at that coordinate. Physical support is intersected with the
            # characterized global coordinate before applying the margin.
            common_low = torch.maximum(plus_lower, minus_lower).clamp(0.0, 1.0)
            common_high = torch.minimum(plus_upper, minus_upper).clamp(0.0, 1.0)
            # A point intersection cannot carry a logical fraction.
            overlap = common_high > common_low
            raw_span = (common_high - common_low).clamp_min(0.0)
            inner_span = raw_span * (1.0 - 2.0 * margin)
            inner_low = common_low + margin * raw_span
            baseline = torch.where(
                overlap,
                inner_low,
                (0.5 * (common_low + common_high)).clamp(0.0, 1.0),
            )
            plus_mapped = baseline + plus_target * inner_span
            minus_mapped = baseline + minus_target * inner_span
            mapped[plus_slice] = plus_mapped.reshape(-1)
            mapped[minus_slice] = minus_mapped.reshape(-1)

            group_corrupt = plus_corrupt | minus_corrupt
            group_published_corrupt = (
                plus_published_corrupt | minus_published_corrupt
            )
            nonempty_cell = torch.cat(
                (overlap.reshape(-1), overlap.reshape(-1))
            )
            pair_mapped = torch.cat(
                (plus_mapped.reshape(-1), minus_mapped.reshape(-1))
            )
            pair_lower = torch.cat(
                (plus_lower.reshape(-1), minus_lower.reshape(-1))
            )
            pair_upper = torch.cat(
                (plus_upper.reshape(-1), minus_upper.reshape(-1))
            )
            pair_noncorrupt = torch.cat(
                ((~plus_corrupt).reshape(-1), (~minus_corrupt).reshape(-1))
            )
            pair_below = pair_noncorrupt & (pair_mapped < pair_lower)
            pair_above = pair_noncorrupt & (pair_mapped > pair_upper)
            nonempty_below = int((pair_below & nonempty_cell).sum().item())
            nonempty_above = int((pair_above & nonempty_cell).sum().item())
            empty_below = int((pair_below & ~nonempty_cell).sum().item())
            empty_above = int((pair_above & ~nonempty_cell).sum().item())
            pair_key = f"base.differential_pair.{pair_index}"
            pair_report = {
                "pair_index": pair_index,
                "conductance_plus_key": plus_key,
                "conductance_minus_key": minus_key,
                "shape": list(shape),
                "devices": 2 * count,
                "pair_count": count,
                "common_window_empty_pair_count": int((~overlap).sum().item()),
                "common_window_empty_fraction": float(
                    (~overlap).to(torch.float64).mean().item()
                ),
                "corrupt_pair_count": int(group_corrupt.sum().item()),
                "published_corrupt_pair_count": int(
                    group_published_corrupt.sum().item()
                ),
                "raw_common_span": _tensor_summary(raw_span),
                "inner_common_span": _tensor_summary(inner_span),
                "nonempty_inner_common_span": (
                    _tensor_summary(inner_span[overlap])
                    if bool(torch.any(overlap))
                    else None
                ),
                "nonempty_pair_inner_common_span": (
                    _tensor_summary(inner_span[overlap])
                    if bool(torch.any(overlap))
                    else None
                ),
                "mapped_target_below_lower_bound_nonempty_pair": nonempty_below,
                "mapped_target_above_upper_bound_nonempty_pair": nonempty_above,
                "mapped_target_below_lower_bound_empty_pair": empty_below,
                "mapped_target_above_upper_bound_empty_pair": empty_above,
            }
            report["parameter_pairs"][pair_key] = pair_report
            branch_values = (
                (
                    "conductance_plus",
                    plus_key,
                    minus_key,
                    plus_mapped,
                    plus_lower,
                    plus_upper,
                    plus_corrupt,
                ),
                (
                    "conductance_minus",
                    minus_key,
                    plus_key,
                    minus_mapped,
                    minus_lower,
                    minus_upper,
                    minus_corrupt,
                ),
            )
            for (
                role,
                key,
                peer_key,
                branch_mapped,
                branch_lower,
                branch_upper,
                branch_corrupt,
            ) in branch_values:
                branch_noncorrupt = ~branch_corrupt
                branch_below = branch_noncorrupt & (branch_mapped < branch_lower)
                branch_above = branch_noncorrupt & (branch_mapped > branch_upper)
                report["parameters"][key] = {
                    "differential_role": role,
                    "paired_parameter": peer_key,
                    "pair_index": pair_index,
                    "shape": list(shape),
                    "devices": count,
                    "pair_count": count,
                    "common_window_empty_pair_count": int(
                        (~overlap).sum().item()
                    ),
                    "raw_common_span": _tensor_summary(raw_span),
                    "inner_common_span": _tensor_summary(inner_span),
                    "nonempty_inner_common_span": (
                        _tensor_summary(inner_span[overlap])
                        if bool(torch.any(overlap))
                        else None
                    ),
                    "nonempty_pair_inner_common_span": (
                        _tensor_summary(inner_span[overlap])
                        if bool(torch.any(overlap))
                        else None
                    ),
                    "mapped_target_below_lower_bound_nonempty_pair": int(
                        (branch_below & overlap).sum().item()
                    ),
                    "mapped_target_above_upper_bound_nonempty_pair": int(
                        (branch_above & overlap).sum().item()
                    ),
                    "mapped_target_below_lower_bound_empty_pair": int(
                        (branch_below & ~overlap).sum().item()
                    ),
                    "mapped_target_above_upper_bound_empty_pair": int(
                        (branch_above & ~overlap).sum().item()
                    ),
                }
            provenance_pairs.append(
                {
                    "pair_index": pair_index,
                    "conductance_plus_key": plus_key,
                    "conductance_minus_key": minus_key,
                    "shape": list(shape),
                }
            )
            all_overlap.append(overlap.reshape(-1))
            all_corrupt_pair.append(group_corrupt.reshape(-1))
            all_published_corrupt_pair.append(
                group_published_corrupt.reshape(-1)
            )
            all_raw_span.append(raw_span.reshape(-1))
            all_inner_span.append(inner_span.reshape(-1))
            all_nonempty_cell.extend(
                (overlap.reshape(-1), overlap.reshape(-1))
            )
            offset += 2 * count
        assert offset == population.size

        overlap = torch.cat(all_overlap)
        corrupt_pair = torch.cat(all_corrupt_pair)
        published_corrupt_pair = torch.cat(all_published_corrupt_pair)
        raw_span = torch.cat(all_raw_span)
        inner_span = torch.cat(all_inner_span)
        nonempty_cell = torch.cat(all_nonempty_cell)
        nonempty_support = support_counts(mapped, nonempty_cell)
        empty_support = support_counts(mapped, ~nonempty_cell)
        below_nonempty = nonempty_support["below_lower_bound"]
        above_nonempty = nonempty_support["above_upper_bound"]
        below_empty = empty_support["below_lower_bound"]
        above_empty = empty_support["above_upper_bound"]
        pair_count = int(overlap.numel())
        empty_pair_count = int((~overlap).sum().item())
        corrupt_pair_count = int(corrupt_pair.sum().item())
        published_corrupt_pair_count = int(
            published_corrupt_pair.sum().item()
        )
        report.update(
            {
                "common_window_grouping": "differential_pair",
                "common_window_group_size": 2,
                "common_window_group_count": pair_count,
                "common_window_empty_group_count": empty_pair_count,
                "corrupt_group_count": corrupt_pair_count,
                "published_corrupt_group_count": published_corrupt_pair_count,
                "differential_pair_binding": {
                    "policy": "canonical_adjacent_plus_minus_same_coordinate",
                    "catalog_order": list(population.binding_keys),
                    "pairs": provenance_pairs,
                },
                "pair_count": pair_count,
                "common_window_empty_pair_count": empty_pair_count,
                "common_window_empty_fraction": float(
                    (~overlap).to(torch.float64).mean().item()
                ),
                "corrupt_pair_count": corrupt_pair_count,
                "published_corrupt_pair_count": published_corrupt_pair_count,
                "raw_common_span": _tensor_summary(raw_span),
                "inner_common_span": _tensor_summary(inner_span),
                "nonempty_inner_common_span": (
                    _tensor_summary(inner_span[overlap])
                    if bool(torch.any(overlap))
                    else None
                ),
                "nonempty_pair_inner_common_span": (
                    _tensor_summary(inner_span[overlap])
                    if bool(torch.any(overlap))
                    else None
                ),
                "mapped_target_below_lower_bound_nonempty_group": below_nonempty,
                "mapped_target_above_upper_bound_nonempty_group": above_nonempty,
                "mapped_target_below_lower_bound_empty_group": below_empty,
                "mapped_target_above_upper_bound_empty_group": above_empty,
                "mapped_target_below_lower_bound_nonempty_pair": below_nonempty,
                "mapped_target_above_upper_bound_nonempty_pair": above_nonempty,
                "mapped_target_below_lower_bound_empty_pair": below_empty,
                "mapped_target_above_upper_bound_empty_pair": above_empty,
                "mapped_target_support": support_counts(mapped),
            }
        )
        return mapped, report

    layout_by_key = dict(layouts or ())
    mapped = torch.empty_like(global_targets)
    all_overlap: list[torch.Tensor] = []
    all_corrupt_quad: list[torch.Tensor] = []
    all_published_corrupt_quad: list[torch.Tensor] = []
    all_raw_span: list[torch.Tensor] = []
    all_inner_span: list[torch.Tensor] = []
    all_nonempty_cell: list[torch.Tensor] = []
    offset = 0
    for key, shape in zip(population.binding_keys, population.binding_shapes):
        count = math.prod(shape)
        if len(shape) != 2 or shape[0] % 2 or shape[1] % 2:
            raise ValueError(
                "Expected quad common-window population bindings to be "
                "even-by-even rank-2 tensors. Provided value: "
                f"key={key!r}, shape={shape!r}."
            )
        target = global_targets[offset : offset + count].reshape(shape)
        target_mapped = mapped[offset : offset + count].reshape(shape)
        cell_lower = lower[offset : offset + count].reshape(shape)
        cell_upper = upper[offset : offset + count].reshape(shape)
        cell_corrupt = corrupt[offset : offset + count].reshape(shape)
        cell_published_corrupt = published_corrupt[
            offset : offset + count
        ].reshape(shape)
        input_count = shape[0] // 2
        output_count = shape[1] // 2
        plus_rows = torch.arange(input_count, device=device)
        minus_rows = plus_rows + input_count
        if layout_by_key[key] == "halves":
            plus_columns = torch.arange(output_count, device=device)
            minus_columns = plus_columns + output_count
        else:
            plus_columns = torch.arange(output_count, device=device) * 2
            minus_columns = plus_columns + 1
        row_groups = (plus_rows, minus_rows)
        column_groups = (plus_columns, minus_columns)

        group_lower = torch.stack(
            tuple(
                cell_lower[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        )
        group_upper = torch.stack(
            tuple(
                cell_upper[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        )
        group_corrupt = torch.stack(
            tuple(
                cell_corrupt[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        ).any(dim=0)
        group_published_corrupt = torch.stack(
            tuple(
                cell_published_corrupt[rows[:, None], columns]
                for rows in row_groups
                for columns in column_groups
            )
        ).any(dim=0)

        # Intersect physical support with the characterized global coordinate.
        common_low = group_lower.amax(dim=0).clamp_min(0.0)
        common_high = group_upper.amin(dim=0).clamp_max(1.0)
        # A point intersection cannot carry a logical fraction and is therefore
        # budgeted as an empty IBM quad, even though that one point is reachable.
        overlap = common_high > common_low
        raw_span = (common_high - common_low).clamp_min(0.0)
        inner_span = raw_span * (1.0 - 2.0 * margin)
        inner_low = common_low + margin * raw_span
        baseline = torch.where(
            overlap,
            inner_low,
            (0.5 * (common_low + common_high)).clamp(0.0, 1.0),
        )
        nonempty_cell = torch.empty(shape, dtype=torch.bool, device=device)
        for rows in row_groups:
            for columns in column_groups:
                target_mapped[rows[:, None], columns] = (
                    baseline
                    + target[rows[:, None], columns] * inner_span
                )
                nonempty_cell[rows[:, None], columns] = overlap

        parameter_noncorrupt = ~cell_corrupt.reshape(-1)
        parameter_mapped = target_mapped.reshape(-1)
        parameter_lower = cell_lower.reshape(-1)
        parameter_upper = cell_upper.reshape(-1)
        parameter_nonempty = nonempty_cell.reshape(-1)
        parameter_below = parameter_noncorrupt & (
            parameter_mapped < parameter_lower
        )
        parameter_above = parameter_noncorrupt & (
            parameter_mapped > parameter_upper
        )

        parameter_report = {
            "dual_rail_layout": layout_by_key[key],
            "devices": count,
            "quad_count": int(overlap.numel()),
            "common_window_empty_quad_count": int((~overlap).sum().item()),
            "common_window_empty_fraction": float(
                (~overlap).to(torch.float64).mean().item()
            ),
            "corrupt_quad_count": int(group_corrupt.sum().item()),
            "published_corrupt_quad_count": int(
                group_published_corrupt.sum().item()
            ),
            "raw_common_span": _tensor_summary(raw_span),
            "inner_common_span": _tensor_summary(inner_span),
            "mapped_target_below_lower_bound_nonempty_quad": int(
                (parameter_below & parameter_nonempty).sum().item()
            ),
            "mapped_target_above_upper_bound_nonempty_quad": int(
                (parameter_above & parameter_nonempty).sum().item()
            ),
            "mapped_target_below_lower_bound_empty_quad": int(
                (parameter_below & ~parameter_nonempty).sum().item()
            ),
            "mapped_target_above_upper_bound_empty_quad": int(
                (parameter_above & ~parameter_nonempty).sum().item()
            ),
        }
        report["parameters"][key] = parameter_report
        all_overlap.append(overlap.reshape(-1))
        all_corrupt_quad.append(group_corrupt.reshape(-1))
        all_published_corrupt_quad.append(group_published_corrupt.reshape(-1))
        all_raw_span.append(raw_span.reshape(-1))
        all_inner_span.append(inner_span.reshape(-1))
        all_nonempty_cell.append(nonempty_cell.reshape(-1))
        offset += count
    assert offset == population.size

    overlap = torch.cat(all_overlap)
    corrupt_quad = torch.cat(all_corrupt_quad)
    published_corrupt_quad = torch.cat(all_published_corrupt_quad)
    raw_span = torch.cat(all_raw_span)
    inner_span = torch.cat(all_inner_span)
    nonempty_cell = torch.cat(all_nonempty_cell)
    nonempty_support = support_counts(mapped, nonempty_cell)
    empty_support = support_counts(mapped, ~nonempty_cell)
    report.update(
        {
            "common_window_grouping": "dual_rail_quad",
            "common_window_group_size": 4,
            "common_window_group_count": int(overlap.numel()),
            "common_window_empty_group_count": int((~overlap).sum().item()),
            "corrupt_group_count": int(corrupt_quad.sum().item()),
            "published_corrupt_group_count": int(
                published_corrupt_quad.sum().item()
            ),
            "differential_pair_binding": None,
            "quad_count": int(overlap.numel()),
            "common_window_empty_quad_count": int((~overlap).sum().item()),
            "common_window_empty_fraction": float(
                (~overlap).to(torch.float64).mean().item()
            ),
            "corrupt_quad_count": int(corrupt_quad.sum().item()),
            "published_corrupt_quad_count": int(
                published_corrupt_quad.sum().item()
            ),
            "raw_common_span": _tensor_summary(raw_span),
            "inner_common_span": _tensor_summary(inner_span),
            "nonempty_inner_common_span": (
                _tensor_summary(inner_span[overlap])
                if bool(torch.any(overlap))
                else None
            ),
            "nonempty_quad_inner_common_span": (
                _tensor_summary(inner_span[overlap])
                if bool(torch.any(overlap))
                else None
            ),
            "mapped_target_below_lower_bound_nonempty_quad": nonempty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_nonempty_quad": nonempty_support[
                "above_upper_bound"
            ],
            "mapped_target_below_lower_bound_empty_quad": empty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_empty_quad": empty_support[
                "above_upper_bound"
            ],
            "mapped_target_below_lower_bound_nonempty_group": nonempty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_nonempty_group": nonempty_support[
                "above_upper_bound"
            ],
            "mapped_target_below_lower_bound_empty_group": empty_support[
                "below_lower_bound"
            ],
            "mapped_target_above_upper_bound_empty_group": empty_support[
                "above_upper_bound"
            ],
            "mapped_target_support": support_counts(mapped),
        }
    )
    return mapped, report


def validate_ibm_reram_target_mapping_preflight(
    report: Mapping[str, Any],
    *,
    maximum_empty_quads: int = _MAX_EMPTY_COMMON_WINDOW_QUADS,
    maximum_empty_pairs: int = _MAX_EMPTY_COMMON_WINDOW_PAIRS,
) -> None:
    """Fail closed on unsupported nonempty common-window groups."""

    target_mapping = report.get("target_mapping")
    if target_mapping == "cell_aware_exact_bounds_quad":
        support = report.get("mapped_target_support")
        devices = report.get("devices")
        if (
            report.get("oracle") is not True
            or report.get("hidden_device_bounds_consumed_by_target_mapper")
            is not True
            or report.get("cell_aware_mode") not in _CELL_AWARE_MODES
            or report.get("cell_aware_signed_levels") != 9
            or report.get("post_mapping_clipping") is not False
            or report.get("post_programming_rounding") is not False
            or not isinstance(devices, int)
            or devices < 1
            or not isinstance(support, Mapping)
            or support.get("below_lower_bound") != 0
            or support.get("above_upper_bound") != 0
            or support.get("inside_bounds") != devices
            or report.get("mapped_target_outside_0_1") != 0
        ):
            raise ValueError(
                "Expected a strict in-support cell-aware exact-bounds oracle "
                "target-mapping preflight report."
            )
        if report.get("cell_aware_mode") == "quantized_9_level":
            counts = report.get("code_index_counts")
            quad_count = report.get("quad_count")
            if (
                not isinstance(counts, Mapping)
                or set(counts) != {str(index) for index in range(9)}
                or not all(isinstance(value, int) and value >= 0 for value in counts.values())
                or sum(counts.values()) != quad_count
            ):
                raise ValueError(
                    "Expected exact nine-level code-index coverage in the "
                    "cell-aware preflight report."
                )
        return
    if target_mapping not in {
        "dual_rail_quad_common_window",
        "differential_pair_common_window",
    }:
        return
    if target_mapping == "dual_rail_quad_common_window":
        if (
            isinstance(maximum_empty_quads, bool)
            or not isinstance(maximum_empty_quads, int)
            or maximum_empty_quads < 0
        ):
            raise ValueError(
                "Expected maximum_empty_quads to be a non-negative integer."
            )
        below = report.get("mapped_target_below_lower_bound_nonempty_quad")
        above = report.get("mapped_target_above_upper_bound_nonempty_quad")
        empty = report.get("common_window_empty_quad_count")
        if not all(
            isinstance(value, int) and value >= 0
            for value in (below, above, empty)
        ):
            raise ValueError(
                "Expected a strict IBM OM quad target-mapping preflight report."
            )
        if below or above:
            raise ValueError(
                "Expected zero mapped targets outside physical support on "
                "nonempty quad windows. Provided value: "
                f"below={below}, above={above}."
            )
        if empty > maximum_empty_quads:
            raise ValueError(
                "Expected no more than "
                f"{maximum_empty_quads} empty quad common windows. Provided "
                f"value: {empty}."
            )
        return

    if (
        isinstance(maximum_empty_pairs, bool)
        or not isinstance(maximum_empty_pairs, int)
        or maximum_empty_pairs < 0
    ):
        raise ValueError(
            "Expected maximum_empty_pairs to be a non-negative integer."
        )
    below = report.get("mapped_target_below_lower_bound_nonempty_pair")
    above = report.get("mapped_target_above_upper_bound_nonempty_pair")
    empty = report.get("common_window_empty_pair_count")
    pair_count = report.get("pair_count")
    if not all(
        isinstance(value, int) and value >= 0
        for value in (below, above, empty, pair_count)
    ):
        raise ValueError(
            "Expected a strict IBM OM differential-pair target-mapping "
            "preflight report."
        )
    expected_generic = {
        "common_window_grouping": "differential_pair",
        "common_window_group_size": 2,
        "common_window_group_count": pair_count,
        "common_window_empty_group_count": empty,
        "mapped_target_below_lower_bound_nonempty_group": below,
        "mapped_target_above_upper_bound_nonempty_group": above,
        "mapped_target_below_lower_bound_empty_group": report.get(
            "mapped_target_below_lower_bound_empty_pair"
        ),
        "mapped_target_above_upper_bound_empty_group": report.get(
            "mapped_target_above_upper_bound_empty_pair"
        ),
    }
    if any(report.get(key) != value for key, value in expected_generic.items()):
        raise ValueError(
            "Expected differential-pair-specific and generic common-window "
            "preflight fields to agree exactly."
        )
    if empty > pair_count:
        raise ValueError(
            "Expected common_window_empty_pair_count not to exceed pair_count."
        )
    if below or above:
        raise ValueError(
            "Expected zero mapped targets outside physical support on "
            "nonempty differential-pair windows. Provided value: "
            f"below={below}, above={above}."
        )
    if empty > maximum_empty_pairs:
        raise ValueError(
            "Expected no more than "
            f"{maximum_empty_pairs} empty differential-pair common windows. "
            f"Provided value: {empty}."
        )


def _artifact(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Expected --device-model to name a readable IBM OM HWA JSON "
            f"artifact. Provided value: {str(resolved)!r}."
        ) from error
    if not isinstance(value, dict) or (
        value.get("schema") != DEVICE_MODEL_SCHEMA
        or value.get("schema_version") != DEVICE_MODEL_SCHEMA_VERSION
    ):
        raise ValueError(
            "Expected --device-model to use the IBM OM HWA schema version 1."
        )
    programming = value.get("programming")
    expected = {
        "controller": "adaptive",
        "start_protocol": "lower_to_target",
        "condition_key": CONDITION_KEY,
        "tolerance_step_ratio": 0.5,
        "maximum_program_pulses": 128,
    }
    if not isinstance(programming, Mapping) or any(
        programming.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError(
            "Expected --device-model to contain the declared OM cap-128 "
            "adaptive lower-from-RESET protocol."
        )
    endpoint_models = value.get("endpoint_models")
    estimators = value.get("step_estimators")
    if (
        not isinstance(endpoint_models, Mapping)
        or set(endpoint_models) != {"continuous", "published_corruption"}
        or not isinstance(estimators, Mapping)
        or set(estimators) != {"continuous", "published_corruption"}
    ):
        raise ValueError(
            "Expected --device-model to contain matched continuous and "
            "published-corruption endpoint models and step estimators."
        )
    for branch, published_corruption in (
        ("continuous", False),
        ("published_corruption", True),
    ):
        model = endpoint_models[branch]
        if (
            not isinstance(model, Mapping)
            or model.get("schema")
            != "ebl.ibm_reram.bounded_piecewise_uniform_endpoint_model"
            or model.get("schema_version") != 2
        ):
            raise ValueError(
                f"Expected {branch} to contain endpoint-model schema version 2."
            )
        condition = model.get("conditions", {}).get(CONDITION_KEY)
        if not isinstance(condition, Mapping) or condition.get("adequate") is not True:
            raise ValueError(
                f"Expected embedded {branch} endpoint condition to pass its "
                "held-out adequacy gate."
            )
        metadata = model.get("metadata")
        expected_metadata = {
            "preset": OM_PRESET,
            "execution_profile": "hwa_production_cap128",
            "enable_published_corruption": published_corruption,
        }
        if not isinstance(metadata, Mapping) or any(
            metadata.get(key) != expected_value
            for key, expected_value in expected_metadata.items()
        ):
            raise ValueError(
                f"Expected embedded {branch} endpoint metadata to match the "
                "declared OM cap-128 corruption arm."
            )
        PopulationStepEstimator.from_mapping(estimators[branch])
    return value, sha256_file(resolved)


def build_ibm_reram_cell_aware_exact_bounds_codebook(
    global_targets: torch.Tensor,
    population: IbmReramArrayPopulation,
    *,
    dual_rail_layout_by_parameter: (
        Mapping[str, str] | Sequence[Sequence[str]]
    ),
    cell_aware_mode: str,
    cell_aware_signed_levels: int = 9,
) -> dict[str, Any]:
    """Build the immutable tensor artifact for one exact-bounds mapping."""

    mapped, report = map_ibm_reram_array_targets(
        global_targets,
        population,
        target_mapping="cell_aware_exact_bounds_quad",
        dual_rail_layout_by_parameter=dual_rail_layout_by_parameter,
        common_window_margin_fraction=0.0,
        cell_aware_mode=cell_aware_mode,
        cell_aware_signed_levels=cell_aware_signed_levels,
    )
    validate_ibm_reram_target_mapping_preflight(report)
    layouts = _normalize_dual_rail_layouts(dual_rail_layout_by_parameter)
    assert layouts is not None
    rebuilt, rebuilt_report, artifact = _map_cell_aware_exact_bounds_quad(
        global_targets,
        population,
        layouts=layouts,
        mode=cell_aware_mode,
        signed_levels=cell_aware_signed_levels,
    )
    if not torch.equal(mapped, rebuilt) or report != rebuilt_report:
        raise RuntimeError(
            "Expected immutable cell-aware codebook reconstruction to match "
            "the authoritative target mapper exactly."
        )
    if artifact is None:  # pragma: no cover - evidence is requested above
        raise RuntimeError("Expected an immutable cell-aware codebook artifact.")
    return artifact


def _native_seed(seed: int) -> int:
    return int(seed % (2**31 - 2) + 1)


def _sample_tile_hidden(
    shape: tuple[int, ...],
    *,
    construction_seed: int,
    published_corruption: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, float | str]]:
    if len(shape) != 2:
        raise ValueError(
            "Expected IBM OM HWA only for two-dimensional DenseWeight tensors. "
            f"Provided value: {shape!r}."
        )
    try:
        import aihwkit
        from aihwkit.simulator.configs import SingleRPUConfig
        from aihwkit.simulator.presets.devices import ReRamArrayOMPresetDevice
        from aihwkit.simulator.tiles import AnalogTile
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Expected AIHWKit 1.1.0 to sample the published IBM OM cell "
            f"assignment. Provided value: {error}."
        ) from error
    version = str(getattr(aihwkit, "__version__", "unknown"))
    if version != "1.1.0":
        raise RuntimeError(
            "Expected AIHWKit version '1.1.0' for IBM OM identity sampling. "
            f"Provided value: {version!r}."
        )
    device = ReRamArrayOMPresetDevice()
    device.corrupt_devices_prob = (
        PUBLISHED_CORRUPT_PROBABILITY[OM_PRESET] if published_corruption else 0.0
    )
    device.construction_seed = _native_seed(construction_seed)
    tile = AnalogTile(
        shape[1],
        shape[0],
        SingleRPUConfig(device=device),
        bias=False,
    )
    raw = tile.get_hidden_parameters()
    names = ("max_bound", "min_bound", "dwmin_up", "dwmin_down", "reference")
    missing = tuple(name for name in names if name not in raw)
    if missing:
        raise RuntimeError(
            "Expected AIHWKit OM hidden device parameters. "
            f"Provided value: missing={missing!r}."
        )
    hidden = {}
    for name in names:
        tensor = raw[name].detach().to(device="cpu", dtype=torch.float32)
        if tensor.shape != (shape[1], shape[0]):
            raise RuntimeError(
                "Expected AIHWKit hidden arrays in (post, pre) layout. "
                f"Provided value: name={name!r}, shape={tuple(tensor.shape)!r}."
            )
        hidden[name] = tensor.transpose(0, 1).contiguous().reshape(-1)
    metadata: dict[str, float | str] = {
        "aihwkit_version": version,
        "nominal_dw_min": float(device.dw_min),
        "dw_min_std": float(device.dw_min_std),
        "write_noise_std": float(device.write_noise_std),
    }
    return hidden, metadata


def _population_fingerprint(
    *,
    assignment_seed: int,
    corruption_policy: str,
    keys: Sequence[str],
    shapes: Sequence[tuple[int, ...]],
    binding_sampling_seeds: Sequence[int],
    donor_sampling_seeds: Sequence[int],
    scalar_parameters: Mapping[str, float | str],
    tensors: Mapping[str, torch.Tensor],
) -> str:
    digest = sha256()
    digest.update(str(assignment_seed).encode())
    digest.update(corruption_policy.encode())
    for key, shape, binding_seed, donor_seed in zip(
        keys,
        shapes,
        binding_sampling_seeds,
        donor_sampling_seeds,
    ):
        digest.update(key.encode())
        digest.update(repr(tuple(shape)).encode())
        digest.update(str(int(binding_seed)).encode())
        digest.update(str(int(donor_seed)).encode())
    for name in sorted(scalar_parameters):
        value = scalar_parameters[name]
        digest.update(name.encode())
        digest.update(
            (float(value).hex() if isinstance(value, float) else str(value)).encode()
        )
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _sample_om_array_population_layout(
    binding_keys: Sequence[str],
    binding_shapes: Sequence[tuple[int, ...]],
    *,
    assignment_seed: int,
    corruption_policy: str,
) -> IbmReramArrayPopulation:
    """Sample a literal published assignment for one canonical binding layout."""

    if corruption_policy not in _CORRUPTION_POLICIES:
        raise ValueError(
            f"Expected corruption_policy in {_CORRUPTION_POLICIES!r}."
        )
    keys = tuple(binding_keys)
    shapes = tuple(tuple(int(value) for value in shape) for shape in binding_shapes)
    if (
        not keys
        or len(keys) != len(shapes)
        or len(set(keys)) != len(keys)
        or any(not key for key in keys)
        or any(len(shape) != 2 or any(value < 1 for value in shape) for shape in shapes)
    ):
        raise ValueError(
            "Expected unique named two-dimensional IBM OM binding keys and shapes."
        )
    base_seeds = tuple(
        _native_seed(derive_seed(assignment_seed, OM_PRESET, key, "published"))
        for key in keys
    )
    donor_seeds = tuple(
        _native_seed(derive_seed(assignment_seed, OM_PRESET, key, "repair_donor"))
        for key in keys
    )
    names = ("max_bound", "min_bound", "dwmin_up", "dwmin_down", "reference")
    pieces: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    corrupt_pieces = []
    published_pieces = []
    reference_metadata: dict[str, float | str] | None = None
    for key, shape, base_seed, donor_seed in zip(keys, shapes, base_seeds, donor_seeds):
        base, metadata = _sample_tile_hidden(
            shape,
            construction_seed=base_seed,
            published_corruption=True,
        )
        if reference_metadata is None:
            reference_metadata = metadata
        elif metadata != reference_metadata:
            raise RuntimeError("Expected identical OM preset scalar parameters per binding.")
        published_corrupt = (
            (torch.abs(base["max_bound"] - base["min_bound"]) <= 1e-12)
            & (base["dwmin_up"] == 0.0)
            & (base["dwmin_down"] == 0.0)
        )
        final = {name: value.clone() for name, value in base.items()}
        if corruption_policy == "counterfactual_repaired" and bool(
            torch.any(published_corrupt)
        ):
            donor, donor_metadata = _sample_tile_hidden(
                shape,
                construction_seed=donor_seed,
                published_corruption=False,
            )
            if donor_metadata != metadata:
                raise RuntimeError("Expected repair donors from the same OM preset.")
            for name in names:
                final[name][published_corrupt] = donor[name][published_corrupt]
        final_corrupt = (
            (torch.abs(final["max_bound"] - final["min_bound"]) <= 1e-12)
            & (final["dwmin_up"] == 0.0)
            & (final["dwmin_down"] == 0.0)
        )
        for name in names:
            pieces[name].append(final[name])
        corrupt_pieces.append(final_corrupt)
        published_pieces.append(published_corrupt)
    assert reference_metadata is not None
    joined = {name: torch.cat(values) for name, values in pieces.items()}
    joined["corrupt"] = torch.cat(corrupt_pieces)
    joined["published_corrupt"] = torch.cat(published_pieces)
    fingerprint = _population_fingerprint(
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        keys=keys,
        shapes=shapes,
        binding_sampling_seeds=base_seeds,
        donor_sampling_seeds=donor_seeds,
        scalar_parameters=reference_metadata,
        tensors=joined,
    )
    return IbmReramArrayPopulation(
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=base_seeds,
        donor_sampling_seeds=donor_seeds,
        nominal_dw_min=float(reference_metadata["nominal_dw_min"]),
        dw_min_std=float(reference_metadata["dw_min_std"]),
        write_noise_std=float(reference_metadata["write_noise_std"]),
        max_bound=joined["max_bound"],
        min_bound=joined["min_bound"],
        dwmin_up=joined["dwmin_up"],
        dwmin_down=joined["dwmin_down"],
        reference=joined["reference"],
        corrupt=joined["corrupt"],
        published_corrupt=joined["published_corrupt"],
        fingerprint=fingerprint,
        aihwkit_version=str(reference_metadata["aihwkit_version"]),
    )


def _binding_layout(
    bindings: Sequence[ParameterBinding],
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    selected = tuple(bindings)
    if not selected or any(
        not isinstance(binding.parameter, DenseWeight) for binding in selected
    ):
        raise ValueError(
            "Expected at least one named DenseWeight binding for IBM OM HWA."
        )
    keys = tuple(binding.key for binding in selected)
    shapes = tuple(tuple(binding.state.shape) for binding in selected)
    if len(set(keys)) != len(keys):
        raise ValueError("Expected unique IBM OM DenseWeight binding keys.")
    return keys, shapes


def sample_om_array_population(
    bindings: Sequence[ParameterBinding],
    *,
    assignment_seed: int,
    corruption_policy: str,
) -> IbmReramArrayPopulation:
    """Sample a literal published assignment, optionally repairing defects only."""

    keys, shapes = _binding_layout(bindings)
    return _sample_om_array_population_layout(
        keys,
        shapes,
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
    )


_ARRAY_POPULATION_FIELDS = {
    "schema",
    "schema_version",
    "assignment_seed",
    "corruption_policy",
    "binding_keys_json",
    "binding_shapes_json",
    "binding_sampling_seeds",
    "donor_sampling_seeds",
    "nominal_dw_min",
    "dw_min_std",
    "write_noise_std",
    "max_bound",
    "min_bound",
    "dwmin_up",
    "dwmin_down",
    "reference",
    "corrupt",
    "published_corrupt",
    "fingerprint",
    "aihwkit_version",
}


def save_om_array_population(
    path: Path,
    population: IbmReramArrayPopulation,
) -> None:
    """Write a cross-PyTorch-version, pickle-free OM array population."""

    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(_ARRAY_POPULATION_SCHEMA),
            schema_version=np.asarray(
                _ARRAY_POPULATION_SCHEMA_VERSION, dtype=np.int64
            ),
            assignment_seed=np.asarray(population.assignment_seed, dtype=np.int64),
            corruption_policy=np.asarray(population.corruption_policy),
            binding_keys_json=np.asarray(
                json.dumps(
                    list(population.binding_keys),
                    allow_nan=False,
                    separators=(",", ":"),
                )
            ),
            binding_shapes_json=np.asarray(
                json.dumps(
                    [list(shape) for shape in population.binding_shapes],
                    allow_nan=False,
                    separators=(",", ":"),
                )
            ),
            binding_sampling_seeds=np.asarray(
                population.binding_sampling_seeds, dtype=np.int64
            ),
            donor_sampling_seeds=np.asarray(
                population.donor_sampling_seeds, dtype=np.int64
            ),
            nominal_dw_min=np.asarray(population.nominal_dw_min, dtype=np.float64),
            dw_min_std=np.asarray(population.dw_min_std, dtype=np.float64),
            write_noise_std=np.asarray(
                population.write_noise_std, dtype=np.float64
            ),
            max_bound=population.max_bound.detach().cpu().numpy(),
            min_bound=population.min_bound.detach().cpu().numpy(),
            dwmin_up=population.dwmin_up.detach().cpu().numpy(),
            dwmin_down=population.dwmin_down.detach().cpu().numpy(),
            reference=population.reference.detach().cpu().numpy(),
            corrupt=population.corrupt.detach().cpu().numpy(),
            published_corrupt=population.published_corrupt.detach().cpu().numpy(),
            fingerprint=np.asarray(population.fingerprint),
            aihwkit_version=np.asarray(population.aihwkit_version),
        )


def _npz_scalar(payload: Mapping[str, np.ndarray], name: str) -> Any:
    value = payload[name]
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"Expected scalar non-object OM population field {name!r}.")
    return value.item()


def load_om_array_population(path: Path) -> IbmReramArrayPopulation:
    """Load and independently revalidate an OM array population artifact."""

    source = path.expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as raw:
            if set(raw.files) != _ARRAY_POPULATION_FIELDS:
                raise ValueError(
                    "Expected exact OM array population fields. "
                    f"Provided value: {sorted(raw.files)!r}."
                )
            payload = {name: raw[name].copy() for name in raw.files}
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Expected a readable pickle-free OM array population at {str(source)!r}."
        ) from error
    if (
        str(_npz_scalar(payload, "schema")) != _ARRAY_POPULATION_SCHEMA
        or int(_npz_scalar(payload, "schema_version"))
        != _ARRAY_POPULATION_SCHEMA_VERSION
    ):
        raise ValueError("Expected OM array population schema version 1.")
    if payload["schema_version"].dtype != np.dtype(np.int64):
        raise ValueError("Expected int64 OM array population schema version.")
    if payload["assignment_seed"].dtype != np.dtype(np.int64):
        raise ValueError("Expected int64 OM array assignment seed.")
    try:
        keys_value = json.loads(str(_npz_scalar(payload, "binding_keys_json")))
        shapes_value = json.loads(str(_npz_scalar(payload, "binding_shapes_json")))
    except json.JSONDecodeError as error:
        raise ValueError("Expected canonical JSON OM array binding layout.") from error
    if (
        not isinstance(keys_value, list)
        or not all(isinstance(key, str) and key for key in keys_value)
        or not isinstance(shapes_value, list)
        or not all(
            isinstance(shape, list)
            and len(shape) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in shape)
            for shape in shapes_value
        )
    ):
        raise ValueError("Expected valid named two-dimensional OM binding layout.")
    keys = tuple(keys_value)
    shapes = tuple(tuple(shape) for shape in shapes_value)
    binding_seeds_array = payload["binding_sampling_seeds"]
    donor_seeds_array = payload["donor_sampling_seeds"]
    if (
        binding_seeds_array.dtype != np.dtype(np.int64)
        or donor_seeds_array.dtype != np.dtype(np.int64)
        or binding_seeds_array.shape != (len(keys),)
        or donor_seeds_array.shape != (len(keys),)
    ):
        raise ValueError("Expected one int64 base and donor seed per OM binding.")
    tensor_names = (
        "max_bound",
        "min_bound",
        "dwmin_up",
        "dwmin_down",
        "reference",
    )
    size = sum(math.prod(shape) for shape in shapes)
    tensors: dict[str, torch.Tensor] = {}
    for name in tensor_names:
        value = payload[name]
        if (
            value.dtype != np.dtype(np.float32)
            or value.shape != (size,)
            or not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"Expected finite float32 OM population vector {name!r}.")
        tensors[name] = torch.from_numpy(value.copy())
    for name in ("corrupt", "published_corrupt"):
        value = payload[name]
        if value.dtype != np.dtype(np.bool_) or value.shape != (size,):
            raise ValueError(f"Expected boolean OM population vector {name!r}.")
        tensors[name] = torch.from_numpy(value.copy())
    scalar_names = ("nominal_dw_min", "dw_min_std", "write_noise_std")
    scalars = {}
    for name in scalar_names:
        if payload[name].dtype != np.dtype(np.float64):
            raise ValueError(f"Expected float64 OM population scalar {name!r}.")
        value = float(_npz_scalar(payload, name))
        if not math.isfinite(value):
            raise ValueError(f"Expected finite OM population scalar {name!r}.")
        scalars[name] = value
    aihwkit_version = str(_npz_scalar(payload, "aihwkit_version"))
    if aihwkit_version != _REQUIRED_AIHWKIT_VERSION:
        raise ValueError(
            "Expected AIHWKit version '1.1.0' in the OM population artifact."
        )
    assignment_seed = int(_npz_scalar(payload, "assignment_seed"))
    corruption_policy = str(_npz_scalar(payload, "corruption_policy"))
    base_seeds = tuple(int(value) for value in binding_seeds_array.tolist())
    donor_seeds = tuple(int(value) for value in donor_seeds_array.tolist())
    expected_base_seeds = tuple(
        _native_seed(derive_seed(assignment_seed, OM_PRESET, key, "published"))
        for key in keys
    )
    expected_donor_seeds = tuple(
        _native_seed(derive_seed(assignment_seed, OM_PRESET, key, "repair_donor"))
        for key in keys
    )
    if base_seeds != expected_base_seeds or donor_seeds != expected_donor_seeds:
        raise ValueError("Expected binding seeds derived from the OM assignment seed.")
    fingerprint = _population_fingerprint(
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        keys=keys,
        shapes=shapes,
        binding_sampling_seeds=base_seeds,
        donor_sampling_seeds=donor_seeds,
        scalar_parameters={"aihwkit_version": aihwkit_version, **scalars},
        tensors=tensors,
    )
    if str(_npz_scalar(payload, "fingerprint")) != fingerprint:
        raise ValueError("Expected recomputed OM population fingerprint to match.")
    return IbmReramArrayPopulation(
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        binding_keys=keys,
        binding_shapes=shapes,
        binding_sampling_seeds=base_seeds,
        donor_sampling_seeds=donor_seeds,
        nominal_dw_min=scalars["nominal_dw_min"],
        dw_min_std=scalars["dw_min_std"],
        write_noise_std=scalars["write_noise_std"],
        max_bound=tensors["max_bound"],
        min_bound=tensors["min_bound"],
        dwmin_up=tensors["dwmin_up"],
        dwmin_down=tensors["dwmin_down"],
        reference=tensors["reference"],
        corrupt=tensors["corrupt"],
        published_corrupt=tensors["published_corrupt"],
        fingerprint=fingerprint,
        aihwkit_version=aihwkit_version,
    )


def sample_om_array_population_external(
    bindings: Sequence[ParameterBinding],
    *,
    assignment_seed: int,
    corruption_policy: str,
    aihwkit_python: Path,
    population_path: Path,
    receipt_path: Path,
) -> tuple[IbmReramArrayPopulation, dict[str, Any]]:
    """Sample OM cells in pinned AIHWKit while training stays in CUDA PyTorch."""

    keys, shapes = _binding_layout(bindings)
    sampler = aihwkit_python.expanduser().resolve()
    if not sampler.is_file() or not os.access(sampler, os.X_OK):
        raise RuntimeError(
            "Expected EBL_AIHWKIT_PYTHON to identify an executable pinned "
            f"AIHWKit interpreter. Provided value: {str(sampler)!r}."
        )
    destination = population_path.expanduser().resolve()
    receipt_destination = receipt_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_destination.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "preset": OM_PRESET,
        "assignment_seed": assignment_seed,
        "corruption_policy": corruption_policy,
        "binding_keys": list(keys),
        "binding_shapes": [list(shape) for shape in shapes],
        "required_aihwkit_version": _REQUIRED_AIHWKIT_VERSION,
    }
    command = [
        str(sampler),
        "-m",
        "experiments.reram_program_verify.hwa_population_sampler",
        "--request-json",
        json.dumps(request, allow_nan=False, sort_keys=True, separators=(",", ":")),
        "--output",
        str(destination),
        "--receipt",
        str(receipt_destination),
    ]
    environment = os.environ.copy()
    environment.pop("EBL_AIHWKIT_PYTHON", None)
    completed = subprocess.run(
        command,
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(
            "Expected the pinned external OM HWA population sampler to "
            f"complete successfully. Exit={completed.returncode}; tail={detail!r}."
        )
    try:
        receipt = json.loads(receipt_destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Expected a readable OM array sampling receipt.") from error
    expected_receipt_keys = {
        "schema",
        "schema_version",
        "backend",
        "python_executable",
        "python_version",
        "torch_version",
        "aihwkit_version",
        "request",
        "num_cells",
        "population_fingerprint",
        "population_sha256",
        "sampler_source_sha256",
        "population_implementation_sha256",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_receipt_keys
        or receipt.get("schema") != _ARRAY_SAMPLING_RECEIPT_SCHEMA
        or receipt.get("schema_version") != _ARRAY_SAMPLING_RECEIPT_SCHEMA_VERSION
        or receipt.get("backend") != "external_pinned_aihwkit_python"
        or receipt.get("aihwkit_version") != _REQUIRED_AIHWKIT_VERSION
        or receipt.get("request") != request
        or receipt.get("population_sha256") != sha256_file(destination)
        or receipt.get("sampler_source_sha256")
        != sha256_file(_ROOT / "experiments/reram_program_verify/hwa_population_sampler.py")
        or receipt.get("population_implementation_sha256")
        != sha256_file(Path(__file__).resolve())
    ):
        raise RuntimeError("Expected the OM array sampling receipt to match its request and sources.")
    population = load_om_array_population(destination)
    if (
        population.assignment_seed != assignment_seed
        or population.corruption_policy != corruption_policy
        or population.binding_keys != keys
        or population.binding_shapes != shapes
        or population.fingerprint != receipt.get("population_fingerprint")
        or population.size != receipt.get("num_cells")
    ):
        raise RuntimeError("Expected the sampled OM population to match its declared layout.")
    return population, receipt


class _ArrayPlant:
    """Network-scale pulse plant using one exact checkpointable RNG stream."""

    def __init__(
        self,
        population: IbmReramArrayPopulation,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        self.population = population.to(device)
        self.device = device
        self.generator = generator
        self.persistent = self.population.logical_min.clone()
        write_scale = self.population.write_noise_std * self.population.nominal_dw_min
        self.apparent = self.persistent + write_scale * self._normal_all()

    @property
    def size(self) -> int:
        return self.population.size

    def _normal_all(self) -> torch.Tensor:
        return torch.randn(
            (self.size,),
            dtype=torch.float32,
            device=self.device,
            generator=self.generator,
        )

    def pulse(self, directions: torch.Tensor) -> None:
        direction = torch.as_tensor(directions, dtype=torch.int8, device=self.device)
        if direction.shape != (self.size,) or bool(
            torch.any((direction < -1) | (direction > 1))
        ):
            raise ValueError("Expected one {-1,0,1} pulse direction per device.")
        active = direction != 0
        if not bool(torch.any(active)):
            return
        population = self.population
        physical = self.persistent + population.reference
        cycle = self._normal_all()
        candidate = physical.clone()
        up = direction > 0
        down = direction < 0
        if bool(torch.any(up)):
            normalized = torch.where(
                population.max_bound > 0.0,
                physical / population.max_bound,
                torch.zeros_like(physical),
            )
            response = population.dwmin_up * (
                1.0 - normalized + population.dw_min_std * cycle
            )
            candidate[up] = physical[up] + response[up]
        if bool(torch.any(down)):
            normalized = torch.where(
                population.min_bound < 0.0,
                physical / population.min_bound,
                torch.zeros_like(physical),
            )
            response = population.dwmin_down * (
                1.0 - normalized + population.dw_min_std * cycle
            )
            candidate[down] = physical[down] - response[down]
        candidate = torch.maximum(candidate, population.min_bound)
        candidate = torch.minimum(candidate, population.max_bound)
        self.persistent[active] = (candidate - population.reference)[active]
        write_scale = population.write_noise_std * population.nominal_dw_min
        apparent = self.persistent + write_scale * self._normal_all()
        self.apparent[active] = apparent[active]

    def controller_port(self) -> "_ArrayControllerPort":
        return _ArrayControllerPort(self)


class _ArrayControllerPort:
    def __init__(self, plant: _ArrayPlant) -> None:
        self._plant = plant

    @property
    def size(self) -> int:
        return self._plant.size

    def verify(self) -> torch.Tensor:
        return (self._plant.apparent.clone() + 1.0) / 2.0

    def apply_identical_pulses(
        self,
        directions: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        direction = torch.as_tensor(
            directions, dtype=torch.int8, device=self._plant.device
        )
        count = torch.as_tensor(counts, dtype=torch.int64, device=self._plant.device)
        if direction.shape != (self.size,) or count.shape != (self.size,) or bool(
            torch.any(count < 0)
        ):
            raise ValueError("Expected valid directions and pulse counts per device.")
        for pulse_index in range(int(count.max().item()) if count.numel() else 0):
            self._plant.pulse(
                torch.where(
                    count > pulse_index,
                    direction,
                    torch.zeros_like(direction),
                )
            )


def _result_mapping(result: ProgramVerifyResult) -> dict[str, Any]:
    count = int(result.accepted.numel())
    return {
        "devices": count,
        "accepted": int(result.accepted.sum().item()),
        "success_fraction": float(result.accepted.float().mean().item()),
        "budget_exhausted": int(result.budget_exhausted.sum().item()),
        "nonfinite": int(result.nonfinite.sum().item()),
        "pulse_count": {
            "mean": float(result.total_pulses.float().mean().item()),
            "median": float(result.total_pulses.float().median().item()),
            "maximum": int(result.total_pulses.max().item()),
            "set_mean": float(result.set_count.float().mean().item()),
            "reset_mean": float(result.reset_count.float().mean().item()),
        },
        "verify_count_mean": float(result.verify_count.float().mean().item()),
        "reversal_count_mean": float(result.reversals.float().mean().item()),
    }


class IbmReramHwaParameterModifier:
    """Temporary IBM OM endpoints backed by one fixed physical assignment."""

    def __init__(
        self,
        bindings: Sequence[ParameterBinding],
        config: IbmReramHwaConfig,
        *,
        device_model_path: Path,
        conductance_min: float,
        conductance_max: float,
        population_path: Path | None = None,
        population_receipt_path: Path | None = None,
    ) -> None:
        self._bindings = tuple(bindings)
        if not self._bindings:
            raise ValueError("Expected IBM OM HWA to receive named dense bindings.")
        if not math.isfinite(conductance_min) or not math.isfinite(
            conductance_max
        ) or not conductance_min < conductance_max:
            raise ValueError("Expected finite increasing DRN conductance bounds.")
        self._config = config
        self._conductance_min = float(conductance_min)
        self._conductance_max = float(conductance_max)
        differential_binding_pairs = ()
        if config.target_mapping == "differential_pair_common_window":
            differential_binding_pairs = _validate_differential_pair_bindings(
                self._bindings
            )
        self._artifact, self._artifact_sha256 = _artifact(device_model_path)
        if (population_path is None) != (population_receipt_path is None):
            raise ValueError(
                "Expected IBM OM population and receipt paths together or neither."
            )
        self._population_artifact_paths: tuple[Path, Path] | None = None
        self._population_sampling_receipt: dict[str, Any] | None = None
        if population_path is None:
            self._population = sample_om_array_population(
                self._bindings,
                assignment_seed=config.assignment_seed,
                corruption_policy=config.corruption_policy,
            )
        else:
            raw_sampler = os.environ.get("EBL_AIHWKIT_PYTHON")
            if not raw_sampler:
                raise RuntimeError(
                    "Expected IBM OM DRN execution to set EBL_AIHWKIT_PYTHON "
                    "to the pinned AIHWKit 1.1.0 interpreter."
                )
            assert population_receipt_path is not None
            self._population, self._population_sampling_receipt = (
                sample_om_array_population_external(
                    self._bindings,
                    assignment_seed=config.assignment_seed,
                    corruption_policy=config.corruption_policy,
                    aihwkit_python=Path(raw_sampler),
                    population_path=population_path,
                    receipt_path=population_receipt_path,
                )
            )
            self._population_artifact_paths = (
                population_path.expanduser().resolve(),
                population_receipt_path.expanduser().resolve(),
            )
        if config.target_mapping == "differential_pair_common_window":
            population_pairs = _canonical_differential_pair_layout(
                self._population.binding_keys,
                self._population.binding_shapes,
            )
            if population_pairs != differential_binding_pairs:
                raise ValueError(
                    "Expected the fixed IBM OM population to preserve the "
                    "canonical differential-pair binding keys and shapes. "
                    f"Provided population={population_pairs!r}, "
                    f"bindings={differential_binding_pairs!r}."
                )
        self._population_cache: dict[str, IbmReramArrayPopulation] = {
            "cpu": self._population
        }
        model_step = float(self._artifact["programming"]["nominal_dw_min"])
        if not math.isclose(
            model_step,
            self._population.nominal_dw_min,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Expected device-model nominal OM step to match AIHWKit identity "
                f"sampling. Provided value: model={model_step}, "
                f"sampled={self._population.nominal_dw_min}."
            )
        if config.target_mapping in {
            "dual_rail_quad_common_window",
            "cell_aware_exact_bounds_quad",
        }:
            expected_layout_keys = set(self._population.binding_keys)
            provided_layout_keys = set(
                dict(config.dual_rail_layout_by_parameter or ())
            )
            if provided_layout_keys != expected_layout_keys:
                raise ValueError(
                    "Expected dual_rail_layout_by_parameter keys to equal the "
                    "fixed IBM OM population binding keys. Provided value: "
                    f"expected={sorted(expected_layout_keys)!r}, "
                    f"provided={sorted(provided_layout_keys)!r}."
                )
            invalid_shapes = {
                key: shape
                for key, shape in zip(
                    self._population.binding_keys,
                    self._population.binding_shapes,
                )
                if len(shape) != 2 or shape[0] % 2 or shape[1] % 2
            }
            if invalid_shapes:
                raise ValueError(
                    "Expected four-cell dual-rail bindings to be even-by-even "
                    f"rank-2 tensors. Provided value: {invalid_shapes!r}."
                )
        self._generators: dict[str, dict[str, torch.Generator]] = {
            stream: {} for stream in _STREAMS
        }
        self._pending_generator_states: dict[str, dict[str, torch.Tensor]] = {
            stream: {} for stream in _STREAMS
        }
        self._active_stream: str | None = None
        self._last_report: dict[str, Any] | None = None
        self._last_deployment: dict[str, Any] | None = None

    def _population_on(self, device: torch.device) -> IbmReramArrayPopulation:
        key = str(device)
        population = self._population_cache.get(key)
        if population is None:
            population = self._population.to(device)
            self._population_cache[key] = population
        return population

    @property
    def config(self) -> IbmReramHwaConfig:
        return self._config

    @property
    def population_fingerprint(self) -> str:
        return self._population.fingerprint

    @property
    def device_model_sha256(self) -> str:
        return self._artifact_sha256

    @property
    def population_artifact_paths(self) -> tuple[Path, Path] | None:
        return self._population_artifact_paths

    @property
    def population_sampling_receipt(self) -> dict[str, Any] | None:
        if self._population_sampling_receipt is None:
            return None
        return json.loads(json.dumps(self._population_sampling_receipt))

    @property
    def programming_report(self) -> dict[str, Any] | None:
        return None if self._last_report is None else dict(self._last_report)

    @property
    def last_deployment_bundle(self) -> dict[str, Any] | None:
        return self._last_deployment

    def _generator(self, stream: str, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self._generators[stream].get(key)
        if generator is not None:
            return generator
        generator = torch.Generator(device=device)
        seed = self._config.endpoint_seed
        if stream == "evaluation":
            seed ^= _EVALUATION_STREAM_XOR
        generator.manual_seed(seed)
        pending = self._pending_generator_states[stream].pop(key, None)
        if pending is not None:
            generator.set_state(pending)
        self._generators[stream][key] = generator
        return generator

    def _global_targets(
        self,
    ) -> tuple[torch.Tensor, list[tuple[int, tuple[int, ...]]]]:
        flat = []
        layout = []
        offset = 0
        span = self._conductance_max - self._conductance_min
        for binding in self._bindings:
            state = binding.state
            target = (state.detach() - self._conductance_min) / span
            if not bool(torch.all(torch.isfinite(target))):
                raise ValueError(
                    f"Expected finite clean target for {binding.key!r}."
                )
            flat.append(target.reshape(-1))
            layout.append((offset, tuple(state.shape)))
            offset += target.numel()
        return torch.cat(flat), layout

    def preflight_target_mapping(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return authoritative mapped targets and a read-only mapping report."""

        global_targets, _layout = self._global_targets()
        population = self._population_on(global_targets.device)
        mapped, report = map_ibm_reram_array_targets(
            global_targets,
            population,
            target_mapping=self._config.target_mapping,
            dual_rail_layout_by_parameter=(
                self._config.dual_rail_layout_by_parameter
            ),
            common_window_margin_fraction=(
                self._config.common_window_margin_fraction
            ),
            cell_aware_mode=self._config.cell_aware_mode,
            cell_aware_signed_levels=(
                self._config.cell_aware_signed_levels
            ),
        )
        validate_ibm_reram_target_mapping_preflight(report)
        return mapped.detach().clone(), report

    def _targets(
        self,
        *,
        materialize_evidence: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[tuple[int, tuple[int, ...]]],
        dict[str, Any],
        dict[str, Any] | None,
    ]:
        global_targets, layout = self._global_targets()
        population = self._population_on(global_targets.device)
        oracle_codebook = None
        if self._config.target_mapping == "cell_aware_exact_bounds_quad":
            layouts = self._config.dual_rail_layout_by_parameter
            mode = self._config.cell_aware_mode
            levels = self._config.cell_aware_signed_levels
            assert layouts is not None and mode is not None and levels is not None
            targets, report, oracle_codebook = (
                _map_cell_aware_exact_bounds_quad(
                    global_targets,
                    population,
                    layouts=layouts,
                    mode=mode,
                    signed_levels=levels,
                    materialize_evidence=materialize_evidence,
                )
            )
        else:
            targets, report = map_ibm_reram_array_targets(
                global_targets,
                population,
                target_mapping=self._config.target_mapping,
                dual_rail_layout_by_parameter=(
                    self._config.dual_rail_layout_by_parameter
                ),
                common_window_margin_fraction=(
                    self._config.common_window_margin_fraction
                ),
                cell_aware_mode=self._config.cell_aware_mode,
                cell_aware_signed_levels=(
                    self._config.cell_aware_signed_levels
                ),
            )
        validate_ibm_reram_target_mapping_preflight(report)
        return global_targets, targets, layout, report, oracle_codebook

    def _write_endpoints(
        self,
        endpoint: torch.Tensor,
        layout: Sequence[tuple[int, tuple[int, ...]]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        snapshots = []
        span = self._conductance_max - self._conductance_min
        for binding, (offset, shape) in zip(self._bindings, layout):
            state = binding.state
            count = math.prod(shape)
            clean = state.detach().clone()
            snapshots.append((state, clean))
            programmed = endpoint[offset : offset + count].reshape(shape)
            state.copy_(self._conductance_min + span * programmed.to(state))
            binding.parameter.clamp_()
        return snapshots

    def _run_exact_programming(
        self,
        targets: torch.Tensor,
        population: IbmReramArrayPopulation,
        *,
        generator: torch.Generator,
    ) -> tuple[ProgramVerifyResult, _ArrayPlant]:
        """Run the declared controller on exactly the supplied identities."""

        plant = _ArrayPlant(population, generator=generator, device=targets.device)
        branch = (
            "continuous"
            if self._config.corruption_policy == "counterfactual_repaired"
            else "published_corruption"
        )
        estimator = PopulationStepEstimator.from_mapping(
            self._artifact["step_estimators"][branch]
        )
        adaptive = self._artifact["programming"]["adaptive"]
        settings = ControllerSettings(
            kind="adaptive",
            eta=float(adaptive["eta"]),
            maximum_batch=int(adaptive["maximum_batch"]),
            epsilon=float(adaptive["epsilon"]),
            force_one_within_steps=float(adaptive["force_one_within_steps"]),
        )
        tolerance = (
            self._config.tolerance_step_ratio
            * population.nominal_dw_min
            / 2.0
        )
        result = run_program_verify(
            plant.controller_port(),
            targets=targets,
            tolerance=tolerance,
            maximum_pulses=self._config.maximum_program_pulses,
            settings=settings,
            estimator=estimator,
        )
        return result, plant

    def _compact(
        self,
        targets: torch.Tensor,
        *,
        generator: torch.Generator,
        mapping_report: Mapping[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        population = self._population_on(targets.device)
        branch = (
            "continuous"
            if self._config.corruption_policy == "counterfactual_repaired"
            else "published_corruption"
        )
        model = self._artifact["endpoint_models"][branch]
        lower = (population.logical_min + 1.0) / 2.0
        upper = (population.logical_max + 1.0) / 2.0
        noncorrupt = ~population.corrupt
        below = noncorrupt & (targets < lower)
        above = noncorrupt & (targets > upper)
        inside = noncorrupt & ~(below | above)
        outside_fixed_support = below | above
        if (
            self._config.target_mapping == "cell_aware_exact_bounds_quad"
            and bool(torch.any(outside_fixed_support))
        ):
            raise RuntimeError(
                "Expected the exact-bounds oracle mapper to place every "
                "non-corrupt target inside its own cell support without "
                "clipping or fallback."
            )

        # Empty common-window intersections are an explicitly budgeted mapper
        # failure. They can expose a target/reachability class absent from the
        # compact calibration fit. Preserve that null as unsupported and run
        # the physical controller only on those cells; literal/global mapping
        # remains fail-closed in the compact sampler.
        pulse_fallback_mask = torch.zeros_like(population.corrupt)
        mapping_group_suffix = None
        if self._config.target_mapping == "dual_rail_quad_common_window":
            mapping_group_suffix = "quad"
        elif self._config.target_mapping == "differential_pair_common_window":
            mapping_group_suffix = "pair"
        if mapping_group_suffix is not None:
            expected_below = mapping_report.get(
                f"mapped_target_below_lower_bound_empty_{mapping_group_suffix}"
            )
            expected_above = mapping_report.get(
                f"mapped_target_above_upper_bound_empty_{mapping_group_suffix}"
            )
            nonempty_below = mapping_report.get(
                "mapped_target_below_lower_bound_nonempty_"
                f"{mapping_group_suffix}"
            )
            nonempty_above = mapping_report.get(
                "mapped_target_above_upper_bound_nonempty_"
                f"{mapping_group_suffix}"
            )
            if (
                expected_below != int(below.sum().item())
                or expected_above != int(above.sum().item())
                or nonempty_below != 0
                or nonempty_above != 0
            ):
                raise RuntimeError(
                    "Expected compact exact-fallback cells to equal the "
                    "preflighted out-of-bound cells from empty "
                    f"{mapping_group_suffix} windows."
                )
            pulse_fallback_mask = outside_fixed_support
        compact_mask = ~pulse_fallback_mask

        stuck = (population.logical_min + 1.0) / 2.0
        write_sigma = (
            population.write_noise_std * population.nominal_dw_min / 2.0
        )
        raw_endpoint = torch.full_like(targets, float("nan"))
        endpoint = torch.full_like(targets, float("nan"))
        persistent_endpoint = torch.full_like(targets, float("nan"))
        accepted = torch.zeros_like(population.corrupt)
        endpoint_was_clipped = torch.zeros_like(population.corrupt)

        if bool(torch.any(compact_mask)):
            corrupt_apparent = stuck[compact_mask] + write_sigma * torch.randn(
                (int(compact_mask.sum().item()),),
                dtype=stuck.dtype,
                device=stuck.device,
                generator=generator,
            )
            sample = sample_ibm_reram_endpoints(
                targets[compact_mask],
                model,
                condition_key=CONDITION_KEY,
                generator=generator,
                corrupt_mask=population.corrupt[compact_mask],
                lower_bound=lower[compact_mask],
                upper_bound=upper[compact_mask],
                corrupt_apparent_endpoint=corrupt_apparent,
                corrupt_persistent_endpoint=stuck[compact_mask],
                target_out_of_support=self._config.target_out_of_support,
                endpoint_policy=self._config.endpoint_policy,
            )
            raw_endpoint[compact_mask] = sample.raw_endpoint
            endpoint[compact_mask] = sample.endpoint
            persistent_endpoint[compact_mask] = sample.persistent_endpoint
            accepted[compact_mask] = sample.accepted
            endpoint_was_clipped[compact_mask] = sample.endpoint_was_clipped

        fallback_indices = torch.where(pulse_fallback_mask)[0]
        fallback_result: ProgramVerifyResult | None = None
        fallback_plant: _ArrayPlant | None = None
        if fallback_indices.numel():
            fallback_population = _selected_array_population(
                population,
                pulse_fallback_mask,
            )
            fallback_result, fallback_plant = self._run_exact_programming(
                targets[pulse_fallback_mask],
                fallback_population,
                generator=generator,
            )
            fallback_raw = fallback_result.apparent_endpoint
            fallback_endpoint = fallback_raw.clamp(0.0, 1.0)
            fallback_persistent = (fallback_plant.persistent + 1.0) / 2.0
            raw_endpoint[pulse_fallback_mask] = fallback_raw
            endpoint[pulse_fallback_mask] = fallback_endpoint
            persistent_endpoint[pulse_fallback_mask] = fallback_persistent
            accepted[pulse_fallback_mask] = fallback_result.accepted
            endpoint_was_clipped[pulse_fallback_mask] = (
                fallback_endpoint != fallback_raw
            )

        if not bool(
            torch.all(torch.isfinite(raw_endpoint))
            and torch.all(torch.isfinite(endpoint))
            and torch.all(torch.isfinite(persistent_endpoint))
        ):
            raise RuntimeError(
                "Expected compact and pulse-resolved fallback endpoints to "
                "partition every physical identity."
            )

        tolerance = (
            self._config.tolerance_step_ratio
            * population.nominal_dw_min
            / 2.0
        )
        window_reachable = noncorrupt & (
            (targets + tolerance >= lower) & (targets - tolerance <= upper)
        )
        fallback_index_bytes = (
            fallback_indices.detach().cpu().to(dtype=torch.int64).numpy().tobytes()
        )
        fallback_index_sha256 = sha256(fallback_index_bytes).hexdigest()
        if self._config.target_mapping == "differential_pair_common_window":
            fallback_policy = (
                "pulse_resolved_noncorrupt_out_of_bound_empty_pair_only"
            )
        elif self._config.target_mapping == "cell_aware_exact_bounds_quad":
            fallback_policy = "none_cell_aware_exact_bounds_all_targets_in_support"
        else:
            fallback_policy = (
                "pulse_resolved_noncorrupt_out_of_bound_empty_quad_only"
            )
        fallback_report: dict[str, Any] = {
            "policy": fallback_policy,
            "devices": int(fallback_indices.numel()),
            "selection_indices": fallback_indices.detach().cpu().tolist(),
            "selection_indices_sha256": fallback_index_sha256,
            "target_below_lower_bound": int(
                (below & pulse_fallback_mask).sum().item()
            ),
            "target_above_upper_bound": int(
                (above & pulse_fallback_mask).sum().item()
            ),
            "acceptance_window_unreachable": int(
                (pulse_fallback_mask & ~window_reachable).sum().item()
            ),
            "start_state": "sampled_fully_reset_bound",
            "controller": self._config.controller,
            "maximum_program_pulses": self._config.maximum_program_pulses,
            "random_stream_order": "after_compact_endpoint_sampling",
        }
        if fallback_result is not None:
            fallback_report.update(_result_mapping(fallback_result))
            assert fallback_plant is not None
            fallback_physical = (
                fallback_plant.persistent + fallback_plant.population.reference
            )
            fallback_saturated = (
                torch.abs(
                    fallback_physical - fallback_plant.population.min_bound
                )
                <= 2e-6
            ) | (
                torch.abs(
                    fallback_physical - fallback_plant.population.max_bound
                )
                <= 2e-6
            )
            fallback_report["saturated"] = int(
                fallback_saturated.sum().item()
            )
            fallback_report["endpoint_clipped"] = int(
                endpoint_was_clipped[pulse_fallback_mask].sum().item()
            )
        else:
            fallback_report.update(
                {
                    "accepted": 0,
                    "success_fraction": None,
                    "budget_exhausted": 0,
                    "nonfinite": 0,
                    "pulse_count": {
                        "mean": None,
                        "median": None,
                        "maximum": 0,
                        "set_mean": None,
                        "reset_mean": None,
                    },
                    "verify_count_mean": None,
                    "reversal_count_mean": None,
                    "saturated": 0,
                    "endpoint_clipped": 0,
                }
            )

        failed_noncorrupt = noncorrupt & ~accepted
        if self._config.target_mapping == "differential_pair_common_window":
            fallback_execution_detail = (
                "compact_endpoint_with_exact_empty_pair_fallback"
            )
        elif self._config.target_mapping == "cell_aware_exact_bounds_quad":
            fallback_execution_detail = "compact_endpoint_exact_bounds_in_support"
        else:
            fallback_execution_detail = (
                "compact_endpoint_with_exact_empty_quad_fallback"
            )
        report = {
            "execution": "compact_endpoint",
            "execution_detail": (
                fallback_execution_detail
                if (
                    fallback_indices.numel()
                    or self._config.target_mapping
                    == "cell_aware_exact_bounds_quad"
                )
                else "compact_endpoint_only"
            ),
            "devices": int(targets.numel()),
            "compact_endpoint_devices": int(compact_mask.sum().item()),
            "pulse_resolved_fallback_devices": int(fallback_indices.numel()),
            "accepted": int(accepted.sum().item()),
            "success_fraction": float(accepted.float().mean().item()),
            "failed_noncorrupt": int(failed_noncorrupt.sum().item()),
            "corrupt": int(population.corrupt.sum().item()),
            "published_corrupt": int(
                population.published_corrupt.sum().item()
            ),
            "accepted_corrupt": int(
                (accepted & population.corrupt).sum().item()
            ),
            "target_below_lower_bound": int(below.sum().item()),
            "target_inside_bounds": int(inside.sum().item()),
            "target_above_upper_bound": int(above.sum().item()),
            "acceptance_window_unreachable": int(
                (noncorrupt & ~window_reachable).sum().item()
            ),
            "endpoint_clipped": int(endpoint_was_clipped.sum().item()),
            "pulse_resolved_fallback": fallback_report,
        }
        empty_long = torch.empty(0, dtype=torch.int64)
        empty_bool = torch.empty(0, dtype=torch.bool)
        fallback_deployment = {
            "policy": fallback_report["policy"],
            "selection_indices": fallback_indices.detach().cpu().clone(),
            "selection_indices_sha256": fallback_index_sha256,
            "requested_target": (
                targets[pulse_fallback_mask].detach().cpu().clone()
            ),
            "raw_apparent_endpoint": (
                raw_endpoint[pulse_fallback_mask].detach().cpu().clone()
            ),
            "apparent_endpoint": (
                endpoint[pulse_fallback_mask].detach().cpu().clone()
            ),
            "persistent_endpoint": (
                persistent_endpoint[pulse_fallback_mask].detach().cpu().clone()
            ),
            "accepted": accepted[pulse_fallback_mask].detach().cpu().clone(),
            "budget_exhausted": (
                fallback_result.budget_exhausted.detach().cpu().clone()
                if fallback_result is not None
                else empty_bool
            ),
            "set_count": (
                fallback_result.set_count.detach().cpu().clone()
                if fallback_result is not None
                else empty_long
            ),
            "reset_count": (
                fallback_result.reset_count.detach().cpu().clone()
                if fallback_result is not None
                else empty_long
            ),
            "total_pulses": (
                fallback_result.total_pulses.detach().cpu().clone()
                if fallback_result is not None
                else empty_long
            ),
            "verify_count": (
                fallback_result.verify_count.detach().cpu().clone()
                if fallback_result is not None
                else empty_long
            ),
            "reversals": (
                fallback_result.reversals.detach().cpu().clone()
                if fallback_result is not None
                else empty_long
            ),
            "report": fallback_report,
        }
        deployment = {
            "schema": "ebl.ibm_reram.om_compact_endpoint_deployment",
            "schema_version": 1,
            "device_model_sha256": self._artifact_sha256,
            "population_fingerprint": population.fingerprint,
            "config": asdict(self._config),
            "binding_keys": population.binding_keys,
            "binding_shapes": population.binding_shapes,
            "endpoint_generation_policy": (
                "compact_covered_exact_out_of_bound_empty_pair_fallback"
                if self._config.target_mapping
                == "differential_pair_common_window"
                else (
                    "compact_exact_bounds_in_support_only"
                    if self._config.target_mapping
                    == "cell_aware_exact_bounds_quad"
                    else "compact_covered_exact_out_of_bound_empty_quad_fallback"
                )
            ),
            "requested_target": targets.detach().cpu().clone(),
            "apparent_endpoint": endpoint.detach().cpu().clone(),
            "raw_apparent_endpoint": raw_endpoint.detach().cpu().clone(),
            "persistent_endpoint": persistent_endpoint.detach().cpu().clone(),
            "accepted": accepted.detach().cpu().clone(),
            "compact_endpoint_mask": compact_mask.detach().cpu().clone(),
            "pulse_resolved_fallback_mask": (
                pulse_fallback_mask.detach().cpu().clone()
            ),
            "pulse_resolved_fallback": fallback_deployment,
            "generator_state_after_programming": generator.get_state()
            .detach()
            .cpu()
            .clone(),
        }
        return endpoint.to(dtype=targets.dtype), report, deployment

    def _pulse_resolved(
        self,
        targets: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        population = self._population_on(targets.device)
        result, plant = self._run_exact_programming(
            targets,
            population,
            generator=generator,
        )
        tolerance = (
            self._config.tolerance_step_ratio
            * population.nominal_dw_min
            / 2.0
        )
        raw_endpoint = result.apparent_endpoint
        endpoint = raw_endpoint.clamp(0.0, 1.0)
        lower = (population.logical_min + 1.0) / 2.0
        upper = (population.logical_max + 1.0) / 2.0
        noncorrupt = ~population.corrupt
        below = noncorrupt & (targets < lower)
        above = noncorrupt & (targets > upper)
        inside = noncorrupt & ~(below | above)
        window_reachable = noncorrupt & (
            (targets + tolerance >= lower) & (targets - tolerance <= upper)
        )
        persistent_endpoint = (plant.persistent + 1.0) / 2.0
        accepted_noncorrupt = result.accepted & noncorrupt
        apparent_residual = raw_endpoint - targets
        persistent_residual = persistent_endpoint - targets
        physical = plant.persistent + population.reference
        saturated = (
            (torch.abs(physical - population.min_bound) <= 2e-6)
            | (torch.abs(physical - population.max_bound) <= 2e-6)
        )

        def residual_summary(
            residual: torch.Tensor,
            selection: torch.Tensor,
        ) -> dict[str, float | int | None]:
            selected = residual[selection]
            if selected.numel() == 0:
                return {"count": 0, "bias": None, "mae": None, "rmse": None}
            return {
                "count": int(selected.numel()),
                "bias": float(selected.mean().item()),
                "mae": float(selected.abs().mean().item()),
                "rmse": float(selected.square().mean().sqrt().item()),
            }

        report = {
            "execution": "pulse_resolved",
            **_result_mapping(result),
            "corrupt": int(population.corrupt.sum().item()),
            "published_corrupt": int(population.published_corrupt.sum().item()),
            "accepted_corrupt": int(
                (result.accepted & population.corrupt).sum().item()
            ),
            "budget_exhausted_noncorrupt": int(
                (result.budget_exhausted & noncorrupt).sum().item()
            ),
            "budget_exhausted_corrupt": int(
                (result.budget_exhausted & population.corrupt).sum().item()
            ),
            "target_below_lower_bound": int(below.sum().item()),
            "target_inside_bounds": int(inside.sum().item()),
            "target_above_upper_bound": int(above.sum().item()),
            "acceptance_window_unreachable": int(
                (noncorrupt & ~window_reachable).sum().item()
            ),
            "saturated": int(saturated.sum().item()),
            "endpoint_clipped": int((endpoint != raw_endpoint).sum().item()),
            "accepted_noncorrupt_apparent_residual": residual_summary(
                apparent_residual,
                accepted_noncorrupt,
            ),
            "accepted_noncorrupt_persistent_residual": residual_summary(
                persistent_residual,
                accepted_noncorrupt,
            ),
        }
        deployment = {
            "schema": "ebl.ibm_reram.om_pulse_resolved_deployment",
            "schema_version": 1,
            "device_model_sha256": self._artifact_sha256,
            "population_fingerprint": population.fingerprint,
            "population_sampling_receipt": self.population_sampling_receipt,
            "config": asdict(self._config),
            "binding_keys": population.binding_keys,
            "binding_shapes": population.binding_shapes,
            "binding_sampling_seeds": population.binding_sampling_seeds,
            "donor_sampling_seeds": population.donor_sampling_seeds,
            "population_scalars": {
                "preset": OM_PRESET,
                "aihwkit_version": population.aihwkit_version,
                "nominal_dw_min": population.nominal_dw_min,
                "dw_min_std": population.dw_min_std,
                "write_noise_std": population.write_noise_std,
            },
            "population": population.tensor_state(),
            "requested_target": targets.detach().cpu().clone(),
            "persistent_endpoint": persistent_endpoint.detach().cpu().clone(),
            "raw_apparent_endpoint": raw_endpoint.detach().cpu().clone(),
            "apparent_endpoint": endpoint.detach().cpu().clone(),
            "accepted": result.accepted.detach().cpu().clone(),
            "budget_exhausted": result.budget_exhausted.detach().cpu().clone(),
            "corrupt": population.corrupt.detach().cpu().clone(),
            "target_below_lower_bound": below.detach().cpu().clone(),
            "target_inside_bounds": inside.detach().cpu().clone(),
            "target_above_upper_bound": above.detach().cpu().clone(),
            "acceptance_window_reachable": window_reachable.detach().cpu().clone(),
            "saturated": saturated.detach().cpu().clone(),
            "set_count": result.set_count.detach().cpu().clone(),
            "reset_count": result.reset_count.detach().cpu().clone(),
            "total_pulses": result.total_pulses.detach().cpu().clone(),
            "verify_count": result.verify_count.detach().cpu().clone(),
            "reversals": result.reversals.detach().cpu().clone(),
            "generator_state_after_programming": generator.get_state()
            .detach()
            .cpu()
            .clone(),
            "report": report,
        }
        return endpoint.to(dtype=targets.dtype), report, deployment

    @contextmanager
    def _context(self, stream: str, *, enabled: bool) -> Iterator["IbmReramHwaParameterModifier"]:
        if self._active_stream is not None:
            raise RuntimeError(
                "Expected IBM OM modifier contexts not to overlap. "
                f"Provided value: active={self._active_stream!r}, requested={stream!r}."
            )
        self._active_stream = stream
        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            if enabled:
                with torch.no_grad():
                    (
                        global_targets,
                        targets,
                        layout,
                        mapping_report,
                        oracle_codebook,
                    ) = self._targets(
                        materialize_evidence=stream != "train"
                    )
                    generator = self._generator(stream, targets.device)
                    if self._config.execution == "compact_endpoint":
                        endpoint, report, deployment = self._compact(
                            targets,
                            generator=generator,
                            mapping_report=mapping_report,
                        )
                    else:
                        endpoint, report, deployment = self._pulse_resolved(
                            targets, generator=generator
                        )
                    snapshots = self._write_endpoints(endpoint, layout)
                    deployment["global_requested_target"] = (
                        global_targets.detach().cpu().clone()
                    )
                    deployment["target_mapping_report"] = mapping_report
                    deployment["oracle_codebook"] = oracle_codebook
                    self._last_report = {
                        **report,
                        "endpoint_application_policy": (
                            IBM_RERAM_ENDPOINT_APPLICATION_POLICY
                        ),
                        "target_mapping": self._config.target_mapping,
                        "target_mapping_report": mapping_report,
                        "assignment_seed": self._config.assignment_seed,
                        "endpoint_seed": self._config.endpoint_seed,
                        "corruption_policy": self._config.corruption_policy,
                        "population_fingerprint": self._population.fingerprint,
                        "device_model_sha256": self._artifact_sha256,
                        "population_sampling_backend": (
                            self._population_sampling_receipt.get("backend")
                            if self._population_sampling_receipt is not None
                            else "in_process_aihwkit"
                        ),
                        "population_receipt_sha256": (
                            sha256_file(self._population_artifact_paths[1])
                            if self._population_artifact_paths is not None
                            else None
                        ),
                    }
                    deployment["population_sampling_receipt"] = (
                        self.population_sampling_receipt
                    )
                    deployment["endpoint_application_policy"] = (
                        IBM_RERAM_ENDPOINT_APPLICATION_POLICY
                    )
                    deployment["report"] = dict(self._last_report)
                    self._last_deployment = deployment
            yield self
        finally:
            with torch.no_grad():
                for state, clean in snapshots:
                    state.copy_(clean)
            self._active_stream = None

    def training_context(self):
        return self._context("train", enabled=True)

    def evaluation_context(self):
        return self._context(
            "evaluation", enabled=self._config.noisy_evaluation
        )

    def state_dict(self) -> dict[str, Any]:
        if self._active_stream is not None:
            raise RuntimeError("Expected modifier state outside an active context.")
        streams = {}
        for stream in _STREAMS:
            states = {
                key: value.detach().cpu().clone()
                for key, value in self._pending_generator_states[stream].items()
            }
            states.update(
                {
                    key: generator.get_state().detach().cpu().clone()
                    for key, generator in self._generators[stream].items()
                }
            )
            streams[stream] = states
        return {
            "version": _STATE_VERSION,
            "endpoint_application_policy": (
                IBM_RERAM_ENDPOINT_APPLICATION_POLICY
            ),
            "config": asdict(self._config),
            "conductance_bounds": [self._conductance_min, self._conductance_max],
            "device_model_sha256": self._artifact_sha256,
            "population": {
                "fingerprint": self._population.fingerprint,
                "binding_keys": self._population.binding_keys,
                "binding_shapes": self._population.binding_shapes,
                "binding_sampling_seeds": self._population.binding_sampling_seeds,
                "donor_sampling_seeds": self._population.donor_sampling_seeds,
                "tensors": self._population.tensor_state(),
            },
            "generator_states": streams,
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        if self._active_stream is not None:
            raise RuntimeError("Expected modifier restore outside an active context.")
        expected = {
            "version",
            "endpoint_application_policy",
            "config",
            "conductance_bounds",
            "device_model_sha256",
            "population",
            "generator_states",
        }
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected:
            raise ValueError("Expected an exact IBM OM HWA modifier state mapping.")
        if (
            state_dict["endpoint_application_policy"]
            != IBM_RERAM_ENDPOINT_APPLICATION_POLICY
        ):
            raise ValueError(
                "Expected the saved IBM OM endpoint application policy to "
                "match the apparent-forward, persistent-update-state contract."
            )
        saved_config = state_dict["config"]
        if isinstance(saved_config, Mapping):
            saved_config = dict(saved_config)
            saved_config.setdefault("target_mapping", "literal_global")
            saved_config.setdefault("dual_rail_layout_by_parameter", None)
            saved_config.setdefault("common_window_margin_fraction", 0.0)
            saved_config.setdefault("cell_aware_mode", None)
            saved_config.setdefault("cell_aware_signed_levels", None)
            saved_config.setdefault("forward_logit_gain", None)
        if state_dict["version"] != _STATE_VERSION or saved_config != asdict(
            self._config
        ):
            raise ValueError("Expected saved IBM OM HWA version and config to match.")
        if list(state_dict["conductance_bounds"]) != [
            self._conductance_min,
            self._conductance_max,
        ] or state_dict["device_model_sha256"] != self._artifact_sha256:
            raise ValueError(
                "Expected saved conductance bounds and device-model digest to match."
            )
        population = state_dict["population"]
        if not isinstance(population, Mapping) or population.get(
            "fingerprint"
        ) != self._population.fingerprint:
            raise ValueError("Expected saved IBM OM physical assignment to match.")
        if tuple(population.get("binding_keys", ())) != self._population.binding_keys or tuple(
            tuple(shape) for shape in population.get("binding_shapes", ())
        ) != self._population.binding_shapes:
            raise ValueError("Expected saved IBM OM binding layout to match.")
        tensors = population.get("tensors")
        if not isinstance(tensors, Mapping) or any(
            name not in tensors
            or not isinstance(tensors[name], torch.Tensor)
            or not torch.equal(tensors[name].cpu(), current)
            for name, current in self._population.tensor_state().items()
        ):
            raise ValueError("Expected saved IBM OM population tensors to match exactly.")
        raw_streams = state_dict["generator_states"]
        if not isinstance(raw_streams, Mapping) or set(raw_streams) != set(_STREAMS):
            raise ValueError("Expected train and evaluation IBM OM RNG streams.")
        pending: dict[str, dict[str, torch.Tensor]] = {stream: {} for stream in _STREAMS}
        for stream in _STREAMS:
            if not isinstance(raw_streams[stream], Mapping):
                raise ValueError("Expected IBM OM RNG states to be keyed by device.")
            for key, value in raw_streams[stream].items():
                if (
                    not isinstance(key, str)
                    or not isinstance(value, torch.Tensor)
                    or value.dtype != torch.uint8
                    or value.device.type != "cpu"
                ):
                    raise ValueError("Expected CPU uint8 IBM OM generator states.")
                probe = torch.Generator(device=torch.device(key))
                probe.set_state(value)
                pending[stream][key] = value.detach().clone()
        self._generators = {stream: {} for stream in _STREAMS}
        self._pending_generator_states = pending


def build_ibm_reram_hwa_modifier(
    bindings: Sequence[ParameterBinding],
    config: IbmReramHwaConfig,
    *,
    device_model_path: Path,
    conductance_min: float,
    conductance_max: float,
    population_path: Path | None = None,
    population_receipt_path: Path | None = None,
) -> IbmReramHwaParameterModifier:
    return IbmReramHwaParameterModifier(
        bindings,
        config,
        device_model_path=device_model_path,
        conductance_min=conductance_min,
        conductance_max=conductance_max,
        population_path=population_path,
        population_receipt_path=population_receipt_path,
    )


__all__ = [
    "IBM_RERAM_ENDPOINT_APPLICATION_POLICY",
    "IbmReramArrayPopulation",
    "IbmReramHwaConfig",
    "IbmReramHwaParameterModifier",
    "build_ibm_reram_cell_aware_exact_bounds_codebook",
    "build_ibm_reram_hwa_modifier",
    "load_om_array_population",
    "map_ibm_reram_array_targets",
    "sample_om_array_population",
    "sample_om_array_population_external",
    "save_om_array_population",
    "validate_ibm_reram_target_mapping_preflight",
]
