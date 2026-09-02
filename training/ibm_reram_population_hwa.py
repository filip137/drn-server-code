"""Array-agnostic IBM-OM programming-error HWA for positive conductances.

The optimizer owns clean physical DRN conductances ``G`` in the global range
``[0, 2]``.  For each training minibatch this module converts every physical
cell target to ``x=G/2``, draws a fresh accepted, healthy, target-conditioned
apparent endpoint, ramps the programming error, and exposes

``G_hwa = 2 * (x + alpha * (x_apparent - x))``.

The same temporary tensor spans the complete forward/backward computation and
the clean master is restored before the optimizer step.  No fixed array
identity, per-cell bounds, reference state, corruption mask, or persistent
pulse state is accepted by this API.  Those belong to independent A--D
deployment arms, not to the population-HWA kernel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch

from experiments.artifacts import sha256_file
from model.resistive.builders import ParameterBinding
from training.ibm_reram_endpoint_model import (
    IbmReramAcceptedEndpointModel,
    build_ibm_reram_accepted_endpoint_model,
)
from training.ibm_reram_program_verify import derive_seed
from training.ibm_reram_raw_active_program_verify import RAW_ACTIVE_COORDINATE


POSITIVE_G_MINIMUM = 0.0
POSITIVE_G_MAXIMUM = 2.0
POSITIVE_G_TO_RAW_ACTIVE_X = "x=G/2"
POPULATION_HWA_POLICY = (
    "fresh_accepted_healthy_raw_active_programming_error_per_physical_cell"
)
_STATE_VERSION = 1
_SEED_STREAM = "ibm_om_positive_g_population_hwa_v1"


@dataclass(frozen=True)
class IbmReramPositiveGPopulationHwaConfig:
    """Configuration for the array-agnostic positive-``G`` HWA sampler."""

    seed: int
    ramp_epochs: int = 10
    initial_strength: float = 0.0
    final_strength: float = 1.0
    maximum_rejection_rounds: int = 64

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("Expected population-HWA seed in [0, 2**63).")
        if (
            isinstance(self.ramp_epochs, bool)
            or not isinstance(self.ramp_epochs, int)
            or self.ramp_epochs < 1
        ):
            raise ValueError("Expected population-HWA ramp_epochs >= 1.")
        if (
            isinstance(self.maximum_rejection_rounds, bool)
            or not isinstance(self.maximum_rejection_rounds, int)
            or self.maximum_rejection_rounds < 1
        ):
            raise ValueError(
                "Expected population-HWA maximum_rejection_rounds >= 1."
            )
        for name in ("initial_strength", "final_strength"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"Expected {name} to be finite in [0, 1].")
        if float(self.final_strength) < float(self.initial_strength):
            raise ValueError(
                "Expected final_strength to be greater than or equal to "
                "initial_strength."
            )


@dataclass(frozen=True)
class PositiveGProgrammingErrorDraw:
    """One complete physical-cell HWA realization."""

    target_g: torch.Tensor
    target_x: torch.Tensor
    residual_x: torch.Tensor
    apparent_x: torch.Tensor
    applied_x: torch.Tensor
    applied_g: torch.Tensor
    epoch: int
    strength: float
    rejection_rounds: int
    rejected_scalar_draws: int
    total_scalar_draws: int


def load_raw_active_accepted_endpoint_model(
    path: Path,
    *,
    condition_key: str,
    expected_sha256: str | None = None,
) -> tuple[IbmReramAcceptedEndpointModel, str, Path]:
    """Load a healthy raw-active endpoint artifact with coordinate checks."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            "Expected the raw-active endpoint artifact to exist. "
            f"Provided value: {str(resolved)!r}."
        )
    digest = sha256_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            "Expected the raw-active endpoint artifact SHA-256 to match."
        )
    try:
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "Expected a readable raw-active endpoint JSON artifact."
        ) from error
    if not isinstance(artifact, Mapping):
        raise ValueError("Expected the raw-active endpoint artifact to be an object.")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get(
        "enable_published_corruption"
    ) is not False:
        raise ValueError(
            "Expected the population-HWA artifact to characterize a healthy "
            "counterfactual-repaired population."
        )
    model = build_ibm_reram_accepted_endpoint_model(
        artifact,
        condition_key=condition_key,
        expected_coordinate=RAW_ACTIVE_COORDINATE,
    )
    return model, digest, resolved


class IbmReramPositiveGProgrammingErrorSampler:
    """Replayable population sampler with an explicit ten-epoch-style ramp."""

    def __init__(
        self,
        model: IbmReramAcceptedEndpointModel,
        config: IbmReramPositiveGPopulationHwaConfig,
        *,
        artifact_sha256: str,
    ) -> None:
        if not isinstance(model, IbmReramAcceptedEndpointModel):
            raise TypeError("Expected a compiled accepted-endpoint model.")
        if model.coordinate != RAW_ACTIVE_COORDINATE:
            raise ValueError(
                "Positive-G HWA requires the raw-active endpoint coordinate."
            )
        if not isinstance(config, IbmReramPositiveGPopulationHwaConfig):
            raise TypeError("Expected a positive-G population-HWA config.")
        if (
            not isinstance(artifact_sha256, str)
            or len(artifact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in artifact_sha256)
        ):
            raise ValueError("Expected a lowercase endpoint-artifact SHA-256.")
        self._model = model
        self._config = config
        self._artifact_sha256 = artifact_sha256
        self._resolved_seed = derive_seed(
            config.seed,
            model.fingerprint,
            _SEED_STREAM,
        )
        self._models: dict[str, IbmReramAcceptedEndpointModel] = {}
        self._generators: dict[str, torch.Generator] = {}
        self._pending_generator_states: dict[str, torch.Tensor] = {}
        self._epoch: int | None = None
        self._draw_ordinal = 0
        self._rejected_scalar_draws = 0
        self._total_scalar_draws = 0

    @property
    def config(self) -> IbmReramPositiveGPopulationHwaConfig:
        return self._config

    @property
    def model_fingerprint(self) -> str:
        return self._model.fingerprint

    @property
    def artifact_sha256(self) -> str:
        return self._artifact_sha256

    @property
    def resolved_seed(self) -> int:
        return self._resolved_seed

    @property
    def epoch(self) -> int | None:
        return self._epoch

    @property
    def strength(self) -> float:
        if self._epoch is None:
            raise RuntimeError(
                "Set the population-HWA epoch before drawing programming error."
            )
        fraction = min(float(self._epoch) / float(self._config.ramp_epochs), 1.0)
        return float(self._config.initial_strength) + fraction * (
            float(self._config.final_strength)
            - float(self._config.initial_strength)
        )

    def set_epoch(self, epoch: int) -> None:
        """Select a nonnegative one-based training epoch (zero is clean)."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Expected a nonnegative population-HWA epoch.")
        self._epoch = int(epoch)

    def _key(self, tensor: torch.Tensor) -> str:
        return f"{tensor.device}|{tensor.dtype}"

    def _model_for(self, tensor: torch.Tensor) -> IbmReramAcceptedEndpointModel:
        key = self._key(tensor)
        model = self._models.get(key)
        if model is None:
            model = self._model.to(tensor.device, dtype=tensor.dtype)
            self._models[key] = model
        return model

    def _generator_for(self, tensor: torch.Tensor) -> torch.Generator:
        key = str(tensor.device)
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(self._resolved_seed)
            pending = self._pending_generator_states.pop(key, None)
            if pending is not None:
                generator.set_state(pending)
            self._generators[key] = generator
        return generator

    def sample_full_conductance(
        self,
        target_g: torch.Tensor,
    ) -> PositiveGProgrammingErrorDraw:
        """Draw valid apparent raw-``x`` endpoints without clipping.

        Apparent draws outside the public physical range are rejected and
        redrawn from the same target-conditioned kernel.  The count is part
        of the returned result and sampler continuation state, so this
        hardware-adaptation policy is visible rather than a hidden clamp.
        """

        if self._epoch is None:
            raise RuntimeError(
                "Set the population-HWA epoch before drawing programming error."
            )
        if (
            not isinstance(target_g, torch.Tensor)
            or target_g.numel() < 1
            or target_g.dtype not in {torch.float32, torch.float64}
            or not bool(torch.all(torch.isfinite(target_g)))
            or bool(torch.any(target_g < POSITIVE_G_MINIMUM))
            or bool(torch.any(target_g > POSITIVE_G_MAXIMUM))
        ):
            raise ValueError("Expected finite positive DRN conductances in [0, 2].")
        target_x = target_g.detach() / POSITIVE_G_MAXIMUM
        model = self._model_for(target_x)
        generator = self._generator_for(target_x)
        flat_target = target_x.reshape(-1)
        flat_residual = torch.empty_like(flat_target)
        flat_apparent = torch.empty_like(flat_target)
        pending = torch.arange(
            flat_target.numel(), device=flat_target.device, dtype=torch.long
        )
        rejected = 0
        rounds = 0
        total = 0
        while pending.numel() and rounds < self._config.maximum_rejection_rounds:
            rounds += 1
            sample = model.sample(flat_target[pending], generator=generator)
            total += int(pending.numel())
            valid = (
                torch.isfinite(sample.apparent_x)
                & (sample.apparent_x >= 0.0)
                & (sample.apparent_x <= 1.0)
            )
            accepted_indices = pending[valid]
            flat_residual[accepted_indices] = sample.residual_x[valid]
            flat_apparent[accepted_indices] = sample.apparent_x[valid]
            invalid_indices = pending[~valid]
            rejected += int(invalid_indices.numel())
            pending = invalid_indices
        if pending.numel():
            raise RuntimeError(
                "Could not draw an in-range raw-active apparent endpoint after "
                f"{self._config.maximum_rejection_rounds} rejection rounds; "
                f"remaining={int(pending.numel())}."
            )
        residual = flat_residual.reshape(target_x.shape)
        apparent = flat_apparent.reshape(target_x.shape)
        strength = self.strength
        applied_x = target_x + strength * residual
        applied_g = POSITIVE_G_MAXIMUM * applied_x
        if (
            not bool(torch.all(torch.isfinite(applied_g)))
            or bool(torch.any(applied_g < POSITIVE_G_MINIMUM))
            or bool(torch.any(applied_g > POSITIVE_G_MAXIMUM))
        ):
            raise RuntimeError(
                "Population-HWA interpolation left the positive G=[0,2] range."
            )
        self._draw_ordinal += 1
        self._rejected_scalar_draws += rejected
        self._total_scalar_draws += total
        return PositiveGProgrammingErrorDraw(
            target_g=target_g.detach().clone(),
            target_x=target_x,
            residual_x=residual,
            apparent_x=apparent,
            applied_x=applied_x,
            applied_g=applied_g,
            epoch=int(self._epoch),
            strength=strength,
            rejection_rounds=rounds,
            rejected_scalar_draws=rejected,
            total_scalar_draws=total,
        )

    def report(self) -> dict[str, Any]:
        return {
            "policy": POPULATION_HWA_POLICY,
            "coordinate": RAW_ACTIVE_COORDINATE,
            "conductance_coordinate": POSITIVE_G_TO_RAW_ACTIVE_X,
            "conductance_bounds": [POSITIVE_G_MINIMUM, POSITIVE_G_MAXIMUM],
            "accepted_noncorrupt_residuals_only": True,
            "fixed_array_identity_during_training": False,
            "fixed_per_cell_bounds_during_training": False,
            "fixed_reference_state_during_training": False,
            "corrupt_device_mask_during_training": False,
            "persistent_endpoint_state_during_training": False,
            "physical_programming_during_training": False,
            "same_sample_for_forward_and_backward": True,
            "endpoint_range_policy": "rejection_resample_no_clipping",
            "model_fingerprint": self._model.fingerprint,
            "artifact_sha256": self._artifact_sha256,
            "configured_seed": self._config.seed,
            "resolved_seed": self._resolved_seed,
            "epoch": self._epoch,
            "strength": None if self._epoch is None else self.strength,
            "draw_ordinal": self._draw_ordinal,
            "total_scalar_draws": self._total_scalar_draws,
            "rejected_scalar_draws": self._rejected_scalar_draws,
        }

    def state_dict(self) -> dict[str, Any]:
        states = {
            key: value.detach().cpu().clone()
            for key, value in self._pending_generator_states.items()
        }
        states.update(
            {
                key: generator.get_state().detach().cpu().clone()
                for key, generator in self._generators.items()
            }
        )
        return {
            "version": _STATE_VERSION,
            "policy": POPULATION_HWA_POLICY,
            "config": asdict(self._config),
            "coordinate": RAW_ACTIVE_COORDINATE,
            "artifact_sha256": self._artifact_sha256,
            "model_fingerprint": self._model.fingerprint,
            "resolved_seed": self._resolved_seed,
            "epoch": self._epoch,
            "draw_ordinal": self._draw_ordinal,
            "total_scalar_draws": self._total_scalar_draws,
            "rejected_scalar_draws": self._rejected_scalar_draws,
            "generator_states": states,
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        expected = {
            "version",
            "policy",
            "config",
            "coordinate",
            "artifact_sha256",
            "model_fingerprint",
            "resolved_seed",
            "epoch",
            "draw_ordinal",
            "total_scalar_draws",
            "rejected_scalar_draws",
            "generator_states",
        }
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected:
            raise ValueError("Expected an exact positive-G population-HWA state.")
        invariant = {
            "version": _STATE_VERSION,
            "policy": POPULATION_HWA_POLICY,
            "config": asdict(self._config),
            "coordinate": RAW_ACTIVE_COORDINATE,
            "artifact_sha256": self._artifact_sha256,
            "model_fingerprint": self._model.fingerprint,
            "resolved_seed": self._resolved_seed,
        }
        mismatch = {
            name: {"expected": value, "provided": state_dict.get(name)}
            for name, value in invariant.items()
            if state_dict.get(name) != value
        }
        if mismatch:
            raise ValueError(
                "Positive-G population-HWA continuation does not match: "
                f"{mismatch!r}."
            )
        epoch = state_dict["epoch"]
        if epoch is not None:
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
                raise ValueError("Invalid saved population-HWA epoch.")
        counters = {}
        for name in (
            "draw_ordinal",
            "total_scalar_draws",
            "rejected_scalar_draws",
        ):
            value = state_dict[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid saved population-HWA {name}.")
            counters[name] = value
        raw_states = state_dict["generator_states"]
        if not isinstance(raw_states, Mapping):
            raise ValueError("Expected population-HWA generator states by device.")
        pending: dict[str, torch.Tensor] = {}
        for key, value in raw_states.items():
            if (
                not isinstance(key, str)
                or not isinstance(value, torch.Tensor)
                or value.dtype != torch.uint8
                or value.device.type != "cpu"
            ):
                raise ValueError("Expected CPU uint8 population-HWA RNG states.")
            probe = torch.Generator(device=torch.device(key))
            probe.set_state(value)
            pending[key] = value.detach().clone()
        self._epoch = epoch
        self._draw_ordinal = counters["draw_ordinal"]
        self._total_scalar_draws = counters["total_scalar_draws"]
        self._rejected_scalar_draws = counters["rejected_scalar_draws"]
        self._generators = {}
        self._pending_generator_states = pending


class IbmReramPositiveGPopulationHwaModifier:
    """Temporarily replace clean positive DRN tensors for one minibatch."""

    def __init__(
        self,
        bindings: Sequence[ParameterBinding],
        sampler: IbmReramPositiveGProgrammingErrorSampler,
    ) -> None:
        self._bindings = tuple(bindings)
        if not self._bindings:
            raise ValueError("Expected positive-G population HWA to receive bindings.")
        if not isinstance(sampler, IbmReramPositiveGProgrammingErrorSampler):
            raise TypeError("Expected a positive-G programming-error sampler.")
        for binding in self._bindings:
            parameter = binding.parameter
            minimum = getattr(parameter, "min_cond", None)
            maximum = getattr(parameter, "max_cond", None)
            if (
                minimum is None
                or maximum is None
                or not math.isclose(float(minimum), 0.0)
                or not math.isclose(float(maximum), 2.0)
            ):
                raise ValueError(
                    "Positive-G population HWA requires every DRN parameter "
                    f"to use conductance bounds [0,2]; {binding.key!r} has "
                    f"[{parameter.min_cond!r},{parameter.max_cond!r}]."
                )
        self._sampler = sampler
        self._active = False
        self._last_draw: PositiveGProgrammingErrorDraw | None = None

    @property
    def sampler(self) -> IbmReramPositiveGProgrammingErrorSampler:
        return self._sampler

    @property
    def programming_report(self) -> dict[str, Any]:
        return self._sampler.report()

    def set_epoch(self, epoch: int) -> None:
        self._sampler.set_epoch(epoch)

    @contextmanager
    def training_context(self) -> Iterator["IbmReramPositiveGPopulationHwaModifier"]:
        if self._active:
            raise RuntimeError("Positive-G population-HWA contexts cannot overlap.")
        self._active = True
        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            with torch.no_grad():
                clean_flat = []
                layout = []
                offset = 0
                for binding in self._bindings:
                    state = binding.state
                    clean = state.detach().clone()
                    snapshots.append((state, clean))
                    clean_flat.append(clean.reshape(-1))
                    layout.append((offset, clean.numel(), tuple(clean.shape)))
                    offset += clean.numel()
                draw = self._sampler.sample_full_conductance(
                    torch.cat(clean_flat)
                )
                for binding, (start, count, shape) in zip(
                    self._bindings, layout, strict=True
                ):
                    value = draw.applied_g[start : start + count].reshape(shape)
                    binding.state.copy_(value.to(binding.state))
                self._last_draw = draw
            yield self
        finally:
            with torch.no_grad():
                for state, clean in snapshots:
                    state.copy_(clean)
            self._active = False

    def evaluation_context(self):
        # HWA selection is performed on the clean master or by a separate,
        # explicitly programmed A--D deployment.  There is no noisy-eval mode.
        return nullcontext(self)

    def state_dict(self) -> dict[str, Any]:
        if self._active:
            raise RuntimeError("Expected modifier state outside an active context.")
        return {
            "version": _STATE_VERSION,
            "binding_keys": [binding.key for binding in self._bindings],
            "binding_shapes": [list(binding.state.shape) for binding in self._bindings],
            "sampler": self._sampler.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        if self._active:
            raise RuntimeError("Expected modifier restore outside an active context.")
        expected = {"version", "binding_keys", "binding_shapes", "sampler"}
        if not isinstance(state_dict, Mapping) or set(state_dict) != expected:
            raise ValueError("Expected an exact positive-G modifier state.")
        if (
            state_dict["version"] != _STATE_VERSION
            or list(state_dict["binding_keys"])
            != [binding.key for binding in self._bindings]
            or list(state_dict["binding_shapes"])
            != [list(binding.state.shape) for binding in self._bindings]
        ):
            raise ValueError("Positive-G modifier binding layout does not match.")
        self._sampler.load_state_dict(state_dict["sampler"])


def build_ibm_reram_positive_g_population_hwa_modifier(
    bindings: Sequence[ParameterBinding],
    config: IbmReramPositiveGPopulationHwaConfig,
    *,
    endpoint_model_path: Path,
    condition_key: str,
    expected_sha256: str | None = None,
) -> IbmReramPositiveGPopulationHwaModifier:
    model, digest, _resolved = load_raw_active_accepted_endpoint_model(
        endpoint_model_path,
        condition_key=condition_key,
        expected_sha256=expected_sha256,
    )
    sampler = IbmReramPositiveGProgrammingErrorSampler(
        model,
        config,
        artifact_sha256=digest,
    )
    return IbmReramPositiveGPopulationHwaModifier(bindings, sampler)


__all__ = [
    "IbmReramPositiveGPopulationHwaConfig",
    "IbmReramPositiveGPopulationHwaModifier",
    "IbmReramPositiveGProgrammingErrorSampler",
    "POPULATION_HWA_POLICY",
    "POSITIVE_G_MAXIMUM",
    "POSITIVE_G_MINIMUM",
    "POSITIVE_G_TO_RAW_ACTIVE_X",
    "PositiveGProgrammingErrorDraw",
    "build_ibm_reram_positive_g_population_hwa_modifier",
    "load_raw_active_accepted_endpoint_model",
]
