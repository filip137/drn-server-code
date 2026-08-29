"""Persistent-endpoint ensemble primitives for Winsorized IBM OM QAT.

This module keeps the two relevant operations deliberately separate:

* :func:`build_persistent_endpoint_codebook_ensemble` runs the exact
  raw-active pulse plant from sampled RESET for every supported integer code;
  and
* :class:`PersistentEndpointCodebookEnsemble` gathers those persistent
  endpoints for the integer codes requested by the current logical master.

The table is a finite empirical ensemble, not a differentiable device model.
An identity STE is implemented by evaluating the circuit gradient at the
gathered endpoint and then applying the existing logical-master lift.  In
particular, the circuit sees complete conductance ``G=2*x``.  Apparent verify
values are never used for inference and neither lookup nor handoff clips.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from experiments.mnist_relu_drn.ibm_om_winsorized_qat import (
    WinsorizedQatLayerTemplate,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation
from training.ibm_reram_program_verify import (
    ControllerSettings,
    IbmReramRawActivePlant,
    make_buffered_normal_draws,
    run_program_verify,
)
from training.ibm_reram_raw_active_program_verify import (
    RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT,
    array_population_as_pulse_population,
    matched_trajectory_seeds,
    required_cuda_random_draws,
)


ENSEMBLE_SIZE = 4
LOSS_KINDS = ("deterministic", "mean2", "tail4")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_positive_float(value: float, *, name: str) -> float:
    selected = float(value)
    if not math.isfinite(selected) or selected <= 0.0:
        raise ValueError(f"Expected {name} to be positive and finite.")
    return selected


def _valid_table_mask(maximum_index: torch.Tensor, levels: int) -> torch.Tensor:
    level = torch.arange(levels, device=maximum_index.device).reshape(1, 1, levels)
    return level <= maximum_index.reshape(1, -1, 1)


@dataclass(frozen=True)
class PersistentEndpointLevelDiagnostics:
    """Compact controller diagnostics for one sample/target-level build."""

    sample_index: int
    endpoint_seed: int
    level: int
    eligible_cells: int
    apparent_accepted: int
    persistent_inside_tolerance: int
    budget_exhausted: int
    mean_total_pulses: float

    def __post_init__(self) -> None:
        integers = (
            self.sample_index,
            self.endpoint_seed,
            self.level,
            self.eligible_cells,
            self.apparent_accepted,
            self.persistent_inside_tolerance,
            self.budget_exhausted,
        )
        if any(not _is_int(value) for value in integers):
            raise TypeError("Expected integer persistent-endpoint diagnostics.")
        if (
            self.sample_index < 0
            or self.level < 0
            or self.eligible_cells < 1
            or self.apparent_accepted < 0
            or self.persistent_inside_tolerance < 0
            or self.budget_exhausted < 0
            or self.apparent_accepted > self.eligible_cells
            or self.persistent_inside_tolerance > self.eligible_cells
            or self.budget_exhausted > self.eligible_cells
            or not math.isfinite(float(self.mean_total_pulses))
            or self.mean_total_pulses < 0.0
        ):
            raise ValueError("Invalid persistent-endpoint level diagnostics.")

    def report(self) -> dict[str, int | float]:
        return {
            "sample_index": self.sample_index,
            "endpoint_seed": self.endpoint_seed,
            "level": self.level,
            "eligible_cells": self.eligible_cells,
            "apparent_accepted": self.apparent_accepted,
            "persistent_inside_tolerance": self.persistent_inside_tolerance,
            "budget_exhausted": self.budget_exhausted,
            "mean_total_pulses": float(self.mean_total_pulses),
        }


@dataclass(frozen=True)
class PersistentEndpointCodebookEnsemble:
    """Four exact persistent P&V response tables for one frozen assignment.

    ``persistent_raw_x`` is sample-major with shape ``(4, cells, levels)``.
    Cell/level pairs beyond that cell's commissioned support are represented
    by NaN, so an accidental unsupported lookup fails closed instead of
    silently inventing or clipping a conductance.
    """

    assignment_seed: int
    population_fingerprint: str
    binding_shapes: tuple[tuple[int, ...], ...]
    endpoint_seeds: tuple[int, int, int, int]
    spacing_raw_x: float
    tolerance_raw_x: float
    maximum_program_pulses: int
    baseline_raw_x: torch.Tensor
    maximum_index: torch.Tensor
    persistent_raw_x: torch.Tensor
    level_diagnostics: tuple[PersistentEndpointLevelDiagnostics, ...] = ()

    def __post_init__(self) -> None:
        if not _is_int(self.assignment_seed):
            raise TypeError("Expected assignment_seed to be an integer.")
        if not isinstance(self.population_fingerprint, str) or not self.population_fingerprint:
            raise ValueError("Expected a nonempty population fingerprint.")
        shapes = tuple(tuple(value for value in shape) for shape in self.binding_shapes)
        if (
            not shapes
            or any(not shape for shape in shapes)
            or any(not _is_int(value) or value < 1 for shape in shapes for value in shape)
        ):
            raise ValueError("Expected nonempty positive integer binding shapes.")
        seeds = tuple(self.endpoint_seeds)
        if (
            len(seeds) != ENSEMBLE_SIZE
            or len(set(seeds)) != ENSEMBLE_SIZE
            or any(not _is_int(seed) for seed in seeds)
        ):
            raise ValueError("Expected exactly four unique integer endpoint seeds.")
        _validate_positive_float(self.spacing_raw_x, name="spacing_raw_x")
        _validate_positive_float(self.tolerance_raw_x, name="tolerance_raw_x")
        if not _is_int(self.maximum_program_pulses) or self.maximum_program_pulses < 1:
            raise ValueError("Expected a positive integer maximum_program_pulses.")

        baseline = self.baseline_raw_x
        maximum = self.maximum_index
        table = self.persistent_raw_x
        if (
            not isinstance(baseline, torch.Tensor)
            or not baseline.is_floating_point()
            or baseline.ndim != 1
            or baseline.requires_grad
            or not isinstance(maximum, torch.Tensor)
            or maximum.dtype != torch.int64
            or maximum.ndim != 1
            or maximum.requires_grad
            or not isinstance(table, torch.Tensor)
            or table.dtype != torch.float32
            or table.ndim != 3
            or table.requires_grad
        ):
            raise TypeError(
                "Expected detached floating baseline, int64 capacity, and "
                "float32 endpoint-table tensors."
            )
        cells = sum(math.prod(shape) for shape in shapes)
        if baseline.shape != (cells,) or maximum.shape != (cells,):
            raise ValueError("Expected binding shapes to cover every table cell.")
        if baseline.device != maximum.device or baseline.device != table.device:
            raise ValueError("Expected all endpoint-table tensors on one device.")
        if (
            not bool(torch.all(torch.isfinite(baseline)))
            or bool(torch.any(baseline < 0.0))
            or bool(torch.any(baseline > 1.0))
            or bool(torch.any(maximum < 0))
        ):
            raise ValueError("Expected finite public raw-x baselines and nonnegative capacities.")
        levels = int(maximum.max().item()) + 1
        if table.shape != (ENSEMBLE_SIZE, cells, levels):
            raise ValueError(
                "Expected endpoint table shape (4, cells, global_maximum_index+1)."
            )
        valid = _valid_table_mask(maximum, levels).expand_as(table)
        if (
            not bool(torch.all(torch.isfinite(table[valid])))
            or bool(torch.any(table[valid] < 0.0))
            or bool(torch.any(table[valid] > 1.0))
            or not bool(torch.all(torch.isnan(table[~valid])))
        ):
            raise ValueError(
                "Expected finite in-range persistent endpoints exactly on supported "
                "cell/level pairs and NaN everywhere else."
            )
        diagnostics = tuple(self.level_diagnostics)
        if diagnostics:
            expected_pairs = tuple(
                (sample, level)
                for sample in range(ENSEMBLE_SIZE)
                for level in range(levels)
            )
            observed_pairs = tuple(
                (item.sample_index, item.level) for item in diagnostics
            )
            if len(diagnostics) != ENSEMBLE_SIZE * levels or observed_pairs != expected_pairs:
                raise ValueError("Expected one ordered diagnostic per sample and level.")
            for item in diagnostics:
                expected_eligible = int((maximum >= item.level).sum().item())
                if (
                    item.endpoint_seed != seeds[item.sample_index]
                    or item.eligible_cells != expected_eligible
                ):
                    raise ValueError("Endpoint build diagnostics do not match the table.")

    @property
    def num_cells(self) -> int:
        return int(self.baseline_raw_x.numel())

    @property
    def num_levels(self) -> int:
        return int(self.persistent_raw_x.shape[2])

    @property
    def device(self) -> torch.device:
        return self.persistent_raw_x.device

    @property
    def build_report(self) -> dict[str, object]:
        """Return JSON-ready compact provenance for table construction."""

        return {
            "assignment_seed": self.assignment_seed,
            "population_fingerprint": self.population_fingerprint,
            "endpoint_seeds": list(self.endpoint_seeds),
            "controller": "one_pulse",
            "spacing_raw_x": self.spacing_raw_x,
            "tolerance_raw_x": self.tolerance_raw_x,
            "maximum_program_pulses": self.maximum_program_pulses,
            "cells": self.num_cells,
            "levels": self.num_levels,
            "diagnostics_complete": bool(self.level_diagnostics),
            "per_endpoint_level": [
                item.report() for item in self.level_diagnostics
            ],
        }

    def to(self, device: torch.device | str) -> "PersistentEndpointCodebookEnsemble":
        """Move the immutable empirical table without changing its values."""

        selected = torch.device(device)
        return PersistentEndpointCodebookEnsemble(
            assignment_seed=self.assignment_seed,
            population_fingerprint=self.population_fingerprint,
            binding_shapes=self.binding_shapes,
            endpoint_seeds=self.endpoint_seeds,
            spacing_raw_x=self.spacing_raw_x,
            tolerance_raw_x=self.tolerance_raw_x,
            maximum_program_pulses=self.maximum_program_pulses,
            baseline_raw_x=self.baseline_raw_x.to(selected),
            maximum_index=self.maximum_index.to(selected),
            persistent_raw_x=self.persistent_raw_x.to(selected),
            level_diagnostics=self.level_diagnostics,
        )

    def _sample_indices(self, sample_indices: Sequence[int] | None) -> tuple[int, ...]:
        selected = (
            tuple(range(ENSEMBLE_SIZE))
            if sample_indices is None
            else tuple(sample_indices)
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(not _is_int(value) or value < 0 or value >= ENSEMBLE_SIZE for value in selected)
        ):
            raise ValueError("Expected unique endpoint-sample indices in [0,4).")
        return selected

    def lookup_raw_x(
        self,
        requested_index: torch.Tensor,
        *,
        sample_indices: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Gather persistent raw-x endpoints for the current physical codes."""

        requested = requested_index
        if (
            not isinstance(requested, torch.Tensor)
            or requested.dtype != torch.int64
            or requested.shape != (self.num_cells,)
            or requested.device != self.device
            or requested.requires_grad
        ):
            raise TypeError(
                "Expected a detached int64 requested-index vector on the table device."
            )
        if bool(torch.any(requested < 0)) or bool(
            torch.any(requested > self.maximum_index)
        ):
            raise ValueError("Requested endpoint code exceeds commissioned support.")
        selected = self._sample_indices(sample_indices)
        sample = torch.tensor(selected, dtype=torch.int64, device=self.device)
        bank = self.persistent_raw_x.index_select(0, sample)
        gather_index = requested.reshape(1, -1, 1).expand(len(selected), -1, 1)
        result = torch.gather(bank, dim=2, index=gather_index).squeeze(2)
        if not bool(torch.all(torch.isfinite(result))):
            raise RuntimeError("A supported endpoint lookup produced a non-finite value.")
        return result

    def lookup_full_conductance(
        self,
        requested_index: torch.Tensor,
        *,
        sample_indices: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Return literal complete ``G=2*x`` with validation and no clipping."""

        raw_x = self.lookup_raw_x(
            requested_index, sample_indices=sample_indices
        )
        full_g = 2.0 * raw_x
        if (
            not bool(torch.all(torch.isfinite(full_g)))
            or bool(torch.any(full_g < 0.0))
            or bool(torch.any(full_g > 2.0))
        ):
            raise RuntimeError("Persistent raw x did not map to full G in [0,2].")
        return full_g

    def full_conductance_views(
        self,
        requested_index_by_layer: Sequence[torch.Tensor],
        *,
        sample_indices: Sequence[int] | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Gather and split sample-major full-G views in frozen binding order."""

        requested_layers = tuple(requested_index_by_layer)
        if len(requested_layers) != len(self.binding_shapes):
            raise ValueError("Expected one requested-index tensor per physical binding.")
        for value, shape in zip(requested_layers, self.binding_shapes):
            if value.shape != shape:
                raise ValueError("Requested-index layer shape does not match its binding.")
        flat = torch.cat(tuple(value.reshape(-1) for value in requested_layers))
        full = self.lookup_full_conductance(flat, sample_indices=sample_indices)
        views = []
        for sample in full:
            layers = []
            offset = 0
            for shape in self.binding_shapes:
                count = math.prod(shape)
                layers.append(sample[offset : offset + count].reshape(shape))
                offset += count
            if offset != self.num_cells:  # pragma: no cover - dataclass validates this
                raise RuntimeError("Binding shapes failed to cover the endpoint table.")
            views.append(tuple(layers))
        return tuple(views)


def _flatten_template_tensors(
    templates: Sequence[WinsorizedQatLayerTemplate],
    population: IbmReramArrayPopulation,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    selected = tuple(templates)
    if len(selected) != len(population.binding_shapes) or not selected:
        raise ValueError("Expected one Winsorized QAT template per array binding.")
    spacings = tuple(float(value.level_spacing_raw_x) for value in selected)
    if any(
        not math.isclose(value, spacings[0], rel_tol=0.0, abs_tol=1e-15)
        for value in spacings
    ):
        raise ValueError("Expected one common raw-x spacing across all layers.")
    for template, shape in zip(selected, population.binding_shapes):
        if tuple(template.baseline_raw_x.shape) != tuple(shape):
            raise ValueError("QAT template shape does not match its population binding.")
    baseline = torch.cat(
        tuple(value.baseline_raw_x.detach().cpu().to(torch.float32).reshape(-1) for value in selected)
    )
    baseline_full_g = torch.cat(
        tuple(value.baseline_full_g.detach().cpu().to(torch.float32).reshape(-1) for value in selected)
    )
    upper = torch.cat(
        tuple(value.cell_upper_raw_x.detach().cpu().to(torch.float32).reshape(-1) for value in selected)
    )
    if not torch.equal(baseline, baseline_full_g / 2.0):
        raise ValueError("Expected exact Winsorized baseline parity G=2*x.")
    return baseline, baseline_full_g, upper, spacings[0]


def _maximum_supported_index(
    upper_raw_x: torch.Tensor,
    baseline_raw_x: torch.Tensor,
    *,
    spacing_raw_x: float,
) -> torch.Tensor:
    return torch.floor(
        (
            upper_raw_x.to(torch.float64)
            - baseline_raw_x.to(torch.float64)
            + RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT
        )
        / float(spacing_raw_x)
        + 1e-12
    ).to(torch.int64)


def build_persistent_endpoint_codebook_ensemble(
    population: IbmReramArrayPopulation,
    templates: Sequence[WinsorizedQatLayerTemplate],
    *,
    endpoint_seeds: Sequence[int],
    tolerance_raw_x: float,
    maximum_program_pulses: int,
    device: torch.device | str,
) -> PersistentEndpointCodebookEnsemble:
    """Build four exact target-conditioned persistent endpoint codebooks.

    A codebook member uses one endpoint seed for every integer target.  On
    CUDA, its per-cell normal-draw matrix is generated exactly once and shared
    by fresh lower-initialized plants for all levels.  This preserves the
    repository's matched-stream contract while avoiding repeated RNG staging.
    Unsupported cell/level pairs are excluded from the controller and remain
    NaN in the returned table.
    """

    if not isinstance(population, IbmReramArrayPopulation):
        raise TypeError("Expected one frozen IbmReramArrayPopulation.")
    seeds = tuple(endpoint_seeds)
    if (
        len(seeds) != ENSEMBLE_SIZE
        or len(set(seeds)) != ENSEMBLE_SIZE
        or any(not _is_int(seed) for seed in seeds)
    ):
        raise ValueError("Expected exactly four unique integer endpoint seeds.")
    tolerance = _validate_positive_float(tolerance_raw_x, name="tolerance_raw_x")
    if not _is_int(maximum_program_pulses) or maximum_program_pulses < 1:
        raise ValueError("Expected a positive integer maximum_program_pulses.")
    execution_device = torch.device(device)
    if execution_device.type not in {"cpu", "cuda"}:
        raise ValueError("Expected endpoint-table execution on CPU or CUDA.")
    if execution_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Expected CUDA to be available for endpoint-table construction.")
    if bool(torch.any(population.corrupt)):
        raise ValueError("Persistent endpoint QAT requires a fully repaired population.")

    baseline, baseline_full_g, template_upper, spacing = _flatten_template_tensors(
        templates, population
    )
    _validate_positive_float(spacing, name="spacing_raw_x")
    pulse_population_cpu = array_population_as_pulse_population(population)
    raw_lower = ((pulse_population_cpu.min_bound + 1.0) / 2.0).to(torch.float32)
    raw_upper = ((pulse_population_cpu.max_bound + 1.0) / 2.0).to(torch.float32)
    support_tolerance = RAW_ACTIVE_SUPPORT_TOLERANCE_UNIT
    if (
        baseline.shape != raw_lower.shape
        or not bool(torch.all(torch.isfinite(baseline)))
        or not bool(torch.all(torch.isfinite(template_upper)))
        or bool(torch.any(baseline < raw_lower - support_tolerance))
        or bool(torch.any(baseline > raw_upper + support_tolerance))
        or float((template_upper.to(torch.float64) - raw_upper.to(torch.float64)).abs().max().item())
        > support_tolerance
    ):
        raise ValueError("Winsorized QAT templates do not match population raw-x support.")
    maximum = _maximum_supported_index(
        template_upper, baseline, spacing_raw_x=spacing
    )
    if bool(torch.any(maximum < 0)):
        raise ValueError("Winsorized baseline lies above one cell's support.")
    levels = int(maximum.max().item()) + 1
    table = torch.full(
        (ENSEMBLE_SIZE, population.size, levels),
        float("nan"),
        dtype=torch.float32,
        device="cpu",
    )
    diagnostics: list[PersistentEndpointLevelDiagnostics] = []

    pulse_population = pulse_population_cpu.to(execution_device)
    baseline_full_g_device = baseline_full_g.to(execution_device)
    baseline_device = baseline.to(execution_device)
    maximum_device = maximum.to(execution_device)
    raw_lower_device = raw_lower.to(execution_device)
    raw_upper_device = raw_upper.to(execution_device)
    maximum_draws = (
        required_cuda_random_draws(
            population, maximum_program_pulses=maximum_program_pulses
        )
        if execution_device.type == "cuda"
        else None
    )

    for sample_index, endpoint_seed in enumerate(seeds):
        trajectory_seeds = matched_trajectory_seeds(
            population, endpoint_seed=endpoint_seed
        )
        normal_draws = (
            make_buffered_normal_draws(
                trajectory_seeds,
                maximum_random_draws=maximum_draws,
                device=execution_device,
            )
            if maximum_draws is not None
            else None
        )
        for level in range(levels):
            eligible = maximum_device >= level
            if not bool(torch.any(eligible)):  # pragma: no cover - levels derives from maximum
                continue
            # Match quantize_winsorized_logical_master bit-for-bit: form the
            # float32 full-G offset first, then divide complete G by two.
            offset_g = torch.tensor(
                2.0 * float(level) * spacing,
                dtype=torch.float64,
                device=execution_device,
            ).to(torch.float32)
            level_target = (baseline_full_g_device + offset_g) / 2.0
            target = torch.where(eligible, level_target, baseline_device)
            if (
                not bool(torch.all(torch.isfinite(target)))
                or bool(torch.any(target < raw_lower_device - support_tolerance))
                or bool(torch.any(target > raw_upper_device + support_tolerance))
            ):
                raise RuntimeError("A generated endpoint-code target left exact support.")

            plant = IbmReramRawActivePlant(
                pulse_population,
                seeds=trajectory_seeds,
                device=execution_device,
                maximum_random_draws=maximum_draws,
                normal_draws=normal_draws,
            )
            plant.initialize_at_sampled_lower()
            result = run_program_verify(
                plant.controller_port(),
                targets=target,
                tolerance=tolerance,
                maximum_pulses=maximum_program_pulses,
                settings=ControllerSettings(kind="one_pulse"),
                eligible=eligible,
            )
            persistent = (plant.persistent + 1.0) / 2.0
            selected = persistent[eligible]
            if (
                bool(torch.any(result.nonfinite[eligible]))
                or not bool(torch.all(torch.isfinite(selected)))
                or bool(torch.any(selected < 0.0))
                or bool(torch.any(selected > 1.0))
            ):
                raise RuntimeError("Exact persistent endpoint table became invalid.")
            eligible_cells = int(eligible.sum().item())
            persistent_inside = eligible & (torch.abs(persistent - target) <= tolerance)
            diagnostics.append(
                PersistentEndpointLevelDiagnostics(
                    sample_index=sample_index,
                    endpoint_seed=endpoint_seed,
                    level=level,
                    eligible_cells=eligible_cells,
                    apparent_accepted=int(result.accepted[eligible].sum().item()),
                    persistent_inside_tolerance=int(
                        persistent_inside[eligible].sum().item()
                    ),
                    budget_exhausted=int(
                        result.budget_exhausted[eligible].sum().item()
                    ),
                    mean_total_pulses=float(
                        result.total_pulses[eligible]
                        .to(torch.float64)
                        .mean()
                        .item()
                    ),
                )
            )
            row = table[sample_index, :, level]
            row[eligible.detach().cpu()] = selected.detach().cpu().to(torch.float32)
            del plant, result, persistent, selected
        del normal_draws

    return PersistentEndpointCodebookEnsemble(
        assignment_seed=int(population.assignment_seed),
        population_fingerprint=population.fingerprint,
        binding_shapes=tuple(tuple(shape) for shape in population.binding_shapes),
        endpoint_seeds=seeds,  # type: ignore[arg-type]
        spacing_raw_x=spacing,
        tolerance_raw_x=tolerance,
        maximum_program_pulses=maximum_program_pulses,
        baseline_raw_x=baseline,
        maximum_index=maximum,
        persistent_raw_x=table,
        level_diagnostics=tuple(diagnostics),
    )


@dataclass(frozen=True)
class EndpointGradientMixtureWeights:
    """Scalar weights for one declared endpoint-ensemble objective."""

    kind: str
    ideal_weight: float
    persistent_weights: tuple[float, ...]
    tail_sample_index: int | None


def _finite_losses(values: Sequence[torch.Tensor | float]) -> tuple[float, ...]:
    result = []
    for value in values:
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Expected each endpoint loss to be scalar.")
            selected = float(value.detach().item())
        else:
            selected = float(value)
        if not math.isfinite(selected):
            raise ValueError("Expected finite endpoint losses.")
        result.append(selected)
    return tuple(result)


def gradient_mixture_weights(
    kind: str,
    persistent_losses: Sequence[torch.Tensor | float],
    *,
    ideal_loss: torch.Tensor | float | None = None,
) -> EndpointGradientMixtureWeights:
    """Return the fixed deterministic, mean-two, or tail-four objective weights."""

    if kind not in LOSS_KINDS:
        raise ValueError(f"Expected gradient-mixture kind in {LOSS_KINDS!r}.")
    losses = _finite_losses(persistent_losses)
    if ideal_loss is not None:
        _finite_losses((ideal_loss,))
    if kind == "deterministic":
        if losses or ideal_loss is None:
            raise ValueError("Deterministic mixing requires only one ideal loss.")
        return EndpointGradientMixtureWeights(kind, 1.0, (), None)
    if kind == "mean2":
        if len(losses) != 2 or ideal_loss is None:
            raise ValueError(
                "Mean-two mixing requires one ideal and exactly two persistent losses."
            )
        return EndpointGradientMixtureWeights(
            kind, 0.25, (0.375, 0.375), None
        )
    if len(losses) != ENSEMBLE_SIZE or ideal_loss is None:
        raise ValueError("Tail-four mixing requires one ideal and four persistent losses.")
    tail = max(range(ENSEMBLE_SIZE), key=losses.__getitem__)
    persistent = [0.5 / ENSEMBLE_SIZE] * ENSEMBLE_SIZE
    persistent[tail] += 0.25
    return EndpointGradientMixtureWeights(
        kind,
        0.25,
        tuple(persistent),
        tail,
    )


def mix_logical_gradients(
    kind: str,
    persistent_losses: Sequence[torch.Tensor | float],
    persistent_gradients: Sequence[Sequence[torch.Tensor]],
    *,
    ideal_loss: torch.Tensor | float | None = None,
    ideal_gradients: Sequence[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Mix already lifted logical gradients using the declared objective.

    Tail selection is a stop-gradient argmax over the four persistent losses.
    Every input tensor is left untouched.
    """

    weights = gradient_mixture_weights(
        kind, persistent_losses, ideal_loss=ideal_loss
    )
    samples = tuple(tuple(layer for layer in sample) for sample in persistent_gradients)
    if len(samples) != len(weights.persistent_weights):
        raise ValueError("Persistent gradient/sample count does not match the objective.")
    ideal = None if ideal_gradients is None else tuple(ideal_gradients)
    if (weights.ideal_weight > 0.0) != (ideal is not None):
        raise ValueError("Ideal gradients must be present exactly when ideal loss is weighted.")
    reference = ideal if ideal is not None else (samples[0] if samples else None)
    if reference is None:
        raise ValueError("Expected at least one gradient source.")
    layers = len(reference)
    if layers < 1 or any(len(sample) != layers for sample in samples):
        raise ValueError("Expected matched nonempty logical-gradient layer tuples.")
    if ideal is not None and len(ideal) != layers:
        raise ValueError("Ideal logical-gradient layer count mismatch.")

    result = []
    for layer_index in range(layers):
        sources = [sample[layer_index] for sample in samples]
        if ideal is not None:
            sources.append(ideal[layer_index])
        first = sources[0]
        if (
            not first.is_floating_point()
            or any(
                value.shape != first.shape
                or value.dtype != first.dtype
                or value.device != first.device
                or not value.is_floating_point()
                or not bool(torch.all(torch.isfinite(value)))
                for value in sources
            )
        ):
            raise ValueError("Expected finite matched floating logical gradients.")
        mixed = torch.zeros_like(first)
        if ideal is not None:
            mixed.add_(ideal[layer_index], alpha=weights.ideal_weight)
        for weight, sample in zip(weights.persistent_weights, samples):
            mixed.add_(sample[layer_index], alpha=weight)
        if not bool(torch.all(torch.isfinite(mixed))):
            raise FloatingPointError("Mixed logical gradient became non-finite.")
        result.append(mixed)
    return tuple(result)


__all__ = [
    "ENSEMBLE_SIZE",
    "LOSS_KINDS",
    "EndpointGradientMixtureWeights",
    "PersistentEndpointCodebookEnsemble",
    "PersistentEndpointLevelDiagnostics",
    "build_persistent_endpoint_codebook_ensemble",
    "gradient_mixture_weights",
    "mix_logical_gradients",
]
