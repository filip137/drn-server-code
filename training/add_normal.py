"""Temporary additive-Gaussian perturbations for trainable DRN weights.

The modifier keeps each parameter tensor as a clean master weight.  Entering a
training context samples one realization for a complete minibatch; entering an
enabled evaluation context samples one realization for a complete loader.
Every perturbation and restoration is performed in place so optimizers retain
the identity of the parameter tensors.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Any, Iterator

import torch

from model.variable.parameter import ConvWeight, DenseWeight


_STATE_DICT_VERSION = 1
_MAX_TORCH_SEED = 2**64 - 1
_EVALUATION_STREAM_XOR = 0x9E3779B97F4A7C15
_STREAMS = ("train", "evaluation")


@dataclass(frozen=True)
class AddNormalConfig:
    """Configuration for temporary additive Gaussian weight noise.

    ``std_dev`` is relative to the current maximum absolute clean value of
    each eligible tensor.  ``seed`` overrides the run seed when provided.
    ``noisy_evaluation`` controls whether :meth:`evaluation_context` applies a
    fixed realization or leaves the clean weights untouched.
    """

    std_dev: float
    seed: int | None = None
    noisy_evaluation: bool = False

    def __post_init__(self) -> None:
        std_dev = _validated_non_negative_real(
            "add_normal.std_dev",
            self.std_dev,
        )
        seed = _validated_optional_seed("add_normal.seed", self.seed)
        if not isinstance(self.noisy_evaluation, bool):
            raise ValueError(
                "Expected add_normal.noisy_evaluation to be a boolean. "
                f"Provided value: {self.noisy_evaluation!r}."
            )

        # Store JSON-compatible canonical scalar types.  Besides producing
        # stable provenance, this makes state/config comparisons unambiguous.
        object.__setattr__(self, "std_dev", std_dev)
        object.__setattr__(self, "seed", seed)


class AddNormalParameterModifier:
    """Implement additive-normal noise through ``ParameterModifier`` contexts."""

    def __init__(
        self,
        parameters: Iterable[Any],
        config: AddNormalConfig,
        *,
        run_seed: int | None = None,
    ) -> None:
        if not isinstance(config, AddNormalConfig):
            raise ValueError(
                "Expected config to be an AddNormalConfig. "
                f"Provided value: {config!r}."
            )
        normalized_run_seed = _validated_optional_seed("run_seed", run_seed)

        eligible = []
        seen_parameter_ids: set[int] = set()
        for parameter in parameters:
            parameter_id = id(parameter)
            if (
                isinstance(parameter, (DenseWeight, ConvWeight))
                and parameter_id not in seen_parameter_ids
            ):
                eligible.append(parameter)
                seen_parameter_ids.add(parameter_id)

        if config.seed is not None:
            resolved_seed = config.seed
        elif normalized_run_seed is not None:
            resolved_seed = normalized_run_seed
        else:
            # Reading the initial seed does not consume the global RNG stream.
            resolved_seed = int(torch.initial_seed())

        self._parameters = tuple(eligible)
        self._config = config
        self._resolved_seed = resolved_seed
        self._generators: dict[
            str, dict[str, torch.Generator]
        ] = {stream: {} for stream in _STREAMS}
        self._pending_generator_states: dict[
            str, dict[str, torch.Tensor]
        ] = {stream: {} for stream in _STREAMS}
        self._active_stream: str | None = None

    @property
    def config(self) -> AddNormalConfig:
        """Return the immutable, normalized configuration."""

        return self._config

    @property
    def resolved_seed(self) -> int:
        """Return the seed selected by modifier/run/global precedence."""

        return self._resolved_seed

    def training_context(self):
        """Sample one realization held for a complete training minibatch."""

        return self._noise_context(
            "train",
            enabled=self._config.std_dev > 0.0,
        )

    def evaluation_context(self):
        """Hold one evaluation realization for a loader, when configured."""

        return self._noise_context(
            "evaluation",
            enabled=(
                self._config.noisy_evaluation
                and self._config.std_dev > 0.0
            ),
        )

    def _stream_seed(self, stream: str) -> int:
        if stream == "train":
            return self._resolved_seed
        return self._resolved_seed ^ _EVALUATION_STREAM_XOR

    def _generator(
        self,
        stream: str,
        device: torch.device,
    ) -> torch.Generator:
        key = _device_key(device)
        generator = self._generators[stream].get(key)
        if generator is not None:
            return generator

        generator = torch.Generator(device=torch.device(key))
        generator.manual_seed(self._stream_seed(stream))
        pending = self._pending_generator_states[stream].pop(key, None)
        if pending is not None:
            generator.set_state(pending)
        self._generators[stream][key] = generator
        return generator

    @contextmanager
    def _noise_context(
        self,
        stream: str,
        *,
        enabled: bool,
    ) -> Iterator["AddNormalParameterModifier"]:
        if self._active_stream is not None:
            raise RuntimeError(
                "Expected add-normal modifier contexts not to overlap. "
                f"Provided value: active={self._active_stream!r}, "
                f"requested={stream!r}."
            )

        self._active_stream = stream
        snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
        try:
            if enabled:
                with torch.no_grad():
                    for parameter in self._parameters:
                        state = parameter.state
                        clean = state.detach().clone()
                        snapshots.append((state, clean))
                        generator = self._generator(stream, state.device)
                        noise = torch.randn(
                            state.shape,
                            dtype=state.dtype,
                            device=state.device,
                            generator=generator,
                        )
                        scale = (
                            self._config.std_dev
                            * clean.detach().abs().max()
                        )
                        state.add_(noise * scale)
                        parameter.clamp_()
            yield self
        finally:
            restore_error: BaseException | None = None
            with torch.no_grad():
                for state, clean in snapshots:
                    try:
                        state.copy_(clean)
                    except BaseException as error:  # pragma: no cover
                        if restore_error is None:
                            restore_error = error
            self._active_stream = None
            if restore_error is not None:  # pragma: no cover
                raise RuntimeError(
                    "Failed to restore a clean add-normal parameter tensor."
                ) from restore_error

    def state_dict(self) -> dict[str, Any]:
        """Return exact continuation state for both per-device RNG streams."""

        if self._active_stream is not None:
            raise RuntimeError(
                "Expected state_dict outside an active add-normal context. "
                f"Provided value: active={self._active_stream!r}."
            )

        generator_states: dict[str, dict[str, torch.Tensor]] = {}
        for stream in _STREAMS:
            states = {
                key: state.detach().cpu().clone()
                for key, state in self._pending_generator_states[stream].items()
            }
            states.update(
                {
                    key: generator.get_state().detach().cpu().clone()
                    for key, generator in self._generators[stream].items()
                }
            )
            generator_states[stream] = dict(sorted(states.items()))

        return {
            "version": _STATE_DICT_VERSION,
            "config": asdict(self._config),
            "resolved_seed": self._resolved_seed,
            "generator_states": generator_states,
        }

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Validate then atomically replace both RNG continuations."""

        if self._active_stream is not None:
            raise RuntimeError(
                "Expected load_state_dict outside an active add-normal "
                "context. "
                f"Provided value: active={self._active_stream!r}."
            )

        resolved_seed, generator_states = self._validated_state_dict(
            state_dict
        )

        # Everything that can fail has been staged above.  Commit only after
        # the complete checkpoint has passed validation.
        self._resolved_seed = resolved_seed
        self._generators = {stream: {} for stream in _STREAMS}
        self._pending_generator_states = generator_states

    def _validated_state_dict(
        self,
        state_dict: Mapping,
    ) -> tuple[int, dict[str, dict[str, torch.Tensor]]]:
        if not isinstance(state_dict, Mapping):
            raise ValueError(
                "Expected add-normal state_dict to be an object. "
                f"Provided value: {state_dict!r}."
            )

        expected_keys = {
            "version",
            "config",
            "resolved_seed",
            "generator_states",
        }
        provided_keys = set(state_dict)
        if provided_keys != expected_keys:
            raise ValueError(
                "Expected add-normal state_dict keys to be exactly: config, "
                "generator_states, resolved_seed, version. "
                f"Provided value: {sorted(provided_keys, key=repr)!r}."
            )

        version = state_dict["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, Integral)
            or int(version) != _STATE_DICT_VERSION
        ):
            raise ValueError(
                "Expected add-normal state_dict.version to be "
                f"{_STATE_DICT_VERSION}. Provided value: {version!r}."
            )

        saved_config = _validated_saved_config(state_dict["config"])
        current_config = asdict(self._config)
        if saved_config != current_config:
            raise ValueError(
                "Expected add-normal state_dict.config to match the current "
                "modifier configuration. "
                f"Provided value: saved={saved_config!r}, "
                f"current={current_config!r}."
            )

        resolved_seed = _validated_required_seed(
            "add-normal state_dict.resolved_seed",
            state_dict["resolved_seed"],
        )
        if (
            self._config.seed is not None
            and resolved_seed != self._config.seed
        ):
            raise ValueError(
                "Expected add-normal state_dict.resolved_seed to match the "
                "explicit modifier seed. "
                f"Provided value: saved={resolved_seed!r}, "
                f"configured={self._config.seed!r}."
            )

        raw_streams = state_dict["generator_states"]
        if (
            not isinstance(raw_streams, Mapping)
            or set(raw_streams) != set(_STREAMS)
        ):
            raise ValueError(
                "Expected add-normal state_dict.generator_states to contain "
                "exactly train and evaluation mappings. "
                f"Provided value: {raw_streams!r}."
            )

        staged: dict[str, dict[str, torch.Tensor]] = {
            stream: {} for stream in _STREAMS
        }
        for stream in _STREAMS:
            raw_device_states = raw_streams[stream]
            if not isinstance(raw_device_states, Mapping):
                raise ValueError(
                    "Expected each add-normal generator stream to be an "
                    "object keyed by canonical device string. "
                    f"Provided value: stream={stream!r}, "
                    f"states={raw_device_states!r}."
                )
            for key, raw_state in raw_device_states.items():
                device = _validated_device_key(key)
                state = _validated_generator_state(
                    raw_state,
                    stream=stream,
                    device_key=key,
                    device=device,
                )
                staged[stream][key] = state

        return resolved_seed, staged


def build_add_normal_modifier(
    parameters: Iterable[Any],
    config: AddNormalConfig,
    *,
    run_seed: int | None = None,
) -> AddNormalParameterModifier | None:
    """Build the modifier, returning ``None`` for exactly zero noise."""

    if not isinstance(config, AddNormalConfig):
        raise ValueError(
            "Expected config to be an AddNormalConfig. "
            f"Provided value: {config!r}."
        )
    normalized_run_seed = _validated_optional_seed("run_seed", run_seed)
    if config.std_dev == 0.0:
        return None
    return AddNormalParameterModifier(
        parameters,
        config,
        run_seed=normalized_run_seed,
    )


def _validated_non_negative_real(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(
            f"Expected {name} to be a finite number >= 0.0. "
            f"Provided value: {value!r}."
        )
    return float(value)


def _validated_optional_seed(name: str, value: Any) -> int | None:
    if value is None:
        return None
    return _validated_required_seed(name, value)


def _validated_required_seed(name: str, value: Any) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= int(value) <= _MAX_TORCH_SEED
    ):
        raise ValueError(
            f"Expected {name} to be an integer in "
            f"[0, {_MAX_TORCH_SEED}]. Provided value: {value!r}."
        )
    return int(value)


def _validated_saved_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "Expected add-normal state_dict.config to be a normalized object "
            "containing noisy_evaluation, seed, and std_dev. "
            f"Provided value: {value!r}."
        )
    expected_keys = {"std_dev", "seed", "noisy_evaluation"}
    if set(value) != expected_keys:
        raise ValueError(
            "Expected add-normal state_dict.config keys to be exactly: "
            "noisy_evaluation, seed, std_dev. "
            f"Provided value: {sorted(value, key=repr)!r}."
        )
    if not isinstance(value["std_dev"], float):
        raise ValueError(
            "Expected add-normal state_dict.config.std_dev to be a normalized "
            "floating-point number. "
            f"Provided value: {value['std_dev']!r}."
        )
    seed = value["seed"]
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int)
    ):
        raise ValueError(
            "Expected add-normal state_dict.config.seed to be null or a "
            "normalized integer. "
            f"Provided value: {seed!r}."
        )
    if not isinstance(value["noisy_evaluation"], bool):
        raise ValueError(
            "Expected add-normal state_dict.config.noisy_evaluation to be a "
            "boolean. "
            f"Provided value: {value['noisy_evaluation']!r}."
        )
    try:
        normalized = AddNormalConfig(
            std_dev=value["std_dev"],
            seed=seed,
            noisy_evaluation=value["noisy_evaluation"],
        )
    except ValueError as error:
        raise ValueError(
            "Expected add-normal state_dict.config to be valid normalized "
            f"configuration. Provided value: {value!r}."
        ) from error
    return asdict(normalized)


def _device_key(device: torch.device) -> str:
    concrete = torch.device(device)
    if concrete.type == "cuda" and concrete.index is None:
        return f"cuda:{torch.cuda.current_device()}"
    return str(concrete)


def _validated_device_key(value: Any) -> torch.device:
    if not isinstance(value, str):
        raise ValueError(
            "Expected each add-normal generator state to be keyed by a "
            f"canonical device string. Provided value: {value!r}."
        )
    try:
        device = torch.device(value)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "Expected each add-normal generator state to be keyed by a "
            f"canonical device string. Provided value: {value!r}."
        ) from error
    if (
        str(device) != value
        or (device.type == "cuda" and device.index is None)
    ):
        raise ValueError(
            "Expected each add-normal generator state to be keyed by a "
            f"canonical concrete device string. Provided value: {value!r}."
        )
    return device


def _validated_generator_state(
    value: Any,
    *,
    stream: str,
    device_key: str,
    device: torch.device,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.uint8
        or value.ndim != 1
        or value.device.type != "cpu"
    ):
        raise ValueError(
            "Expected each add-normal generator state to be a "
            "one-dimensional CPU uint8 tensor. "
            f"Provided value: stream={stream!r}, device={device_key!r}, "
            f"state={value!r}."
        )
    staged = value.detach().clone()
    try:
        probe = torch.Generator(device=device)
        probe.set_state(staged)
    except Exception as error:
        raise ValueError(
            "Expected each add-normal generator state to be compatible with "
            "its declared device. "
            f"Provided value: stream={stream!r}, device={device_key!r}, "
            f"error={str(error)!r}."
        ) from error
    return staged


__all__ = [
    "AddNormalConfig",
    "AddNormalParameterModifier",
    "build_add_normal_modifier",
]
