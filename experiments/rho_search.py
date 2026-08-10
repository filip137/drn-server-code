#!/usr/bin/env python3
"""Probe optimizer update units and run a layerwise Conv two-rho grid."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from model.function.interaction import load_function_checkpoint_artifact

try:
    from experiments.reporting import (
        complete_run,
        fail_run,
        runtime_context,
        start_run,
    )
except ModuleNotFoundError:  # Support direct execution from the repository root.
    from reporting import complete_run, fail_run, runtime_context, start_run


PARAMETER_RE = re.compile(r"^(ConvWeight|DenseWeight|Bias)_(\d+)$")
DEFAULT_PROBE_COUNTS = (32, 64, 128)
DEFAULT_CANARY_STEPS = 640


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_zero_bias_checkpoint(
    checkpoint: str | Path,
    expected_bias_names: Sequence[str],
) -> dict[str, Any]:
    """Fail closed unless every declared Bias tensor is present and exact zero."""

    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Expected zero-bias checkpoint to exist: {path}.")
    states, schema, source_format = load_function_checkpoint_artifact(
        path, map_location="cpu"
    )
    if schema is None:
        raise RuntimeError(
            "The zero-bias contract requires a versioned checkpoint with named "
            f"parameter schema entries: {path}."
        )

    expected = [str(name).strip() for name in expected_bias_names]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError(
            "Expected a non-empty, unique sequence of zero-bias parameter names. "
            f"Provided value: {expected!r}."
        )
    bias_entries: list[tuple[str, str, torch.Tensor]] = []
    for entry, state in zip(schema, states):
        name = str(entry["name"]).strip()
        parameter_type = str(entry["type"])
        name_is_bias = name.startswith("Bias_")
        type_is_bias = parameter_type.rsplit(".", 1)[-1] == "Bias"
        if name_is_bias != type_is_bias:
            raise RuntimeError(
                "Checkpoint Bias name/type classification mismatch for "
                f"{name!r} ({parameter_type!r}) in {path}."
            )
        if name_is_bias:
            bias_entries.append((name, parameter_type, state))

    observed = [name for name, _parameter_type, _state in bias_entries]
    if len(set(observed)) != len(observed) or set(observed) != set(expected):
        raise RuntimeError(
            "Checkpoint Bias parameter set does not match the zero-bias contract: "
            f"expected={sorted(expected)!r}, observed={sorted(observed)!r}, "
            f"checkpoint={path}."
        )

    per_parameter = []
    total_elements = 0
    total_nonzero = 0
    by_name = {name: (parameter_type, state) for name, parameter_type, state in bias_entries}
    for name in expected:
        parameter_type, state = by_name[name]
        nonzero = int(torch.count_nonzero(state).item())
        elements = int(state.numel())
        total_elements += elements
        total_nonzero += nonzero
        per_parameter.append(
            {
                "name": name,
                "type": parameter_type,
                "shape": [int(value) for value in state.shape],
                "dtype": str(state.dtype),
                "element_count": elements,
                "nonzero_element_count": nonzero,
                "exact_zero": nonzero == 0,
            }
        )
    if total_nonzero != 0:
        raise RuntimeError(
            "Zero-bias checkpoint verification failed: "
            f"{total_nonzero} nonzero Bias elements in {path}."
        )
    return {
        "schema_version": "zero-bias-checkpoint-verification/v1",
        "checkpoint": str(path),
        "checkpoint_sha256": _file_sha256(path),
        "checkpoint_source_format": source_format,
        "expected_bias_names": expected,
        "bias_tensor_count": len(per_parameter),
        "bias_element_count": total_elements,
        "nonzero_bias_element_count": total_nonzero,
        "all_exact_zero": True,
        "parameters": per_parameter,
    }


def verify_zero_bias_run_checkpoints(
    run_dir: str | Path,
    expected_bias_names: Sequence[str],
    *,
    require_terminal: bool = True,
) -> dict[str, Any]:
    """Verify every model checkpoint emitted by a trainer run directory."""

    root = Path(run_dir).expanduser().resolve()
    terminal_paths = [root / "final_model.pt", root / "best_model.pt"]
    if require_terminal:
        missing = [str(path) for path in terminal_paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                "Zero-bias checkpoint verification requires both terminal model "
                f"checkpoints. Missing: {missing!r}."
            )
    model_paths = {path for path in terminal_paths if path.is_file()}
    checkpoint_dir = root / "checkpoints"
    if checkpoint_dir.is_dir():
        model_paths.update(checkpoint_dir.glob("*_model.pt"))
    if not model_paths:
        raise RuntimeError(
            f"Expected at least one saved model checkpoint beneath {root}."
        )
    records = [
        verify_zero_bias_checkpoint(path, expected_bias_names)
        for path in sorted(model_paths, key=lambda value: str(value))
    ]
    return {
        "schema_version": "zero-bias-run-checkpoint-verification/v1",
        "run_dir": str(root),
        "checkpoint_count": len(records),
        "all_exact_zero": True,
        "checkpoints": records,
    }


def _positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Expected {label} to be positive and finite. Provided value: {value!r}.") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"Expected {label} to be positive and finite. Provided value: {value!r}.")
    return number


def rms(tensor: torch.Tensor) -> float:
    value = tensor.detach().to(dtype=torch.float64)
    return float(torch.sqrt(torch.mean(value.square())).item())


def parameter_sha256(states: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in states.items():
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def linear_quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError(f"Expected a non-empty sample. Provided value: {values!r}.")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"Expected q to be in [0, 1]. Provided value: {q!r}.")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parameter_topology(names: Sequence[str]) -> dict[str, Any]:
    normalized = [str(name).strip() for name in names]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Expected unique parameter names. Provided value: {normalized!r}.")

    kinds: dict[str, str] = {}
    conv_by_index: dict[int, str] = {}
    biases: dict[str, int] = {}
    for name in normalized:
        match = PARAMETER_RE.fullmatch(name)
        if match is None:
            raise ValueError(
                "Expected only ConvWeight_i, DenseWeight_i, and Bias_i parameters. "
                f"Provided value: {name!r}."
            )
        kind, raw_index = match.groups()
        kinds[name] = kind
        index = int(raw_index)
        if kind == "ConvWeight":
            conv_by_index[index] = name
        elif kind == "Bias":
            biases[name] = index

    mapping = {}
    for bias, index in biases.items():
        if index not in conv_by_index:
            raise ValueError(
                f"Expected {bias!r} to have an attached ConvWeight_{index}. "
                f"Provided parameter names: {normalized!r}."
            )
        mapping[bias] = conv_by_index[index]
    if not conv_by_index or not any(kind == "DenseWeight" for kind in kinds.values()):
        raise ValueError(
            "Expected at least one ConvWeight and one DenseWeight. "
            f"Provided parameter names: {normalized!r}."
        )
    return {
        "parameter_names": normalized,
        "kinds": kinds,
        "bias_to_weight": mapping,
        "weight_names": [
            name for name in normalized if kinds[name] in {"ConvWeight", "DenseWeight"}
        ],
    }


def nominal_proposal(gradient: torch.Tensor, optimizer_name: str, eps: float = 1e-8) -> torch.Tensor:
    if optimizer_name == "SGD":
        return -gradient
    if optimizer_name == "Adam":
        return -gradient / (gradient.abs() + eps)
    raise ValueError(
        "Expected optimizer_name to be one of ('SGD', 'Adam'). "
        f"Provided value: {optimizer_name!r}."
    )


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-30)


class OptimizerProbe:
    """Collect nominal-LR-one proposal units from the trainer's gradients."""

    def __init__(
        self,
        optimizer_name: str,
        batch_counts: Sequence[int] = DEFAULT_PROBE_COUNTS,
        stability_tolerance: float = 0.10,
        adam_eps: float = 1e-8,
        stability_scope: str = "all",
    ):
        counts = tuple(int(value) for value in batch_counts)
        if not counts or any(value < 2 or value % 2 for value in counts):
            raise ValueError(
                "Expected probe batch counts to be positive even integers >= 2. "
                f"Provided value: {counts!r}."
            )
        if tuple(sorted(set(counts))) != counts:
            raise ValueError(
                "Expected probe batch counts to be strictly increasing. "
                f"Provided value: {counts!r}."
            )
        self.optimizer_name = optimizer_name
        self.batch_counts = counts
        self.stability_tolerance = float(stability_tolerance)
        if not math.isfinite(self.stability_tolerance) or self.stability_tolerance < 0.0:
            raise ValueError(
                "Expected stability_tolerance to be non-negative and finite. "
                f"Provided value: {stability_tolerance!r}."
            )
        self.adam_eps = _positive(adam_eps, "Adam epsilon")
        if stability_scope not in {"all", "weights_only"}:
            raise ValueError(
                "Expected stability_scope to be one of ('all', 'weights_only'). "
                f"Provided value: {stability_scope!r}."
            )
        self.stability_scope = stability_scope
        self.topology: dict[str, Any] | None = None
        self.parameter_refs: dict[str, Any] = {}
        self.initial_states: dict[str, torch.Tensor] = {}
        self.initial_weight_rms: dict[str, float] = {}
        self.samples: dict[str, list[float]] = {}
        self.split_statistics: dict[str, dict[str, float]] = {}
        self.used_batches = 0
        self.stable = False

    @staticmethod
    def _statistic(name: str, values: Sequence[float]) -> float:
        return linear_quantile(values, 0.9 if name.startswith("Bias_") else 0.5)

    def __call__(self, batch: Mapping[str, Any]) -> bool:
        parameters = tuple(batch["parameters"])
        gradients = tuple(batch["gradients"])
        names = [str(getattr(param, "name", "")).strip() for param in parameters]
        if len(parameters) != len(gradients):
            raise RuntimeError(
                f"Expected one gradient per parameter. Provided values: "
                f"parameters={len(parameters)}, gradients={len(gradients)}."
            )

        if self.topology is None:
            self.topology = parameter_topology(names)
            self.parameter_refs = dict(zip(names, parameters))
            self.initial_states = {
                name: param.state.detach().clone()
                for name, param in zip(names, parameters)
            }
            self.initial_weight_rms = {
                name: rms(self.initial_states[name])
                for name in self.topology["weight_names"]
            }
            invalid = {
                name: value
                for name, value in self.initial_weight_rms.items()
                if not math.isfinite(value) or value <= 0.0
            }
            if invalid:
                raise RuntimeError(
                    "Expected every initial weight RMS to be positive and finite. "
                    f"Provided value: {invalid!r}."
                )
            self.samples = {name: [] for name in names}
        elif names != self.topology["parameter_names"]:
            raise RuntimeError(
                "Expected stable parameter names during the probe. "
                f"Provided value: {names!r}."
            )

        for name, gradient in zip(names, gradients):
            attached = self.topology["bias_to_weight"].get(name, name)
            denominator = self.initial_weight_rms[attached]
            delta = nominal_proposal(
                gradient.detach(),
                self.optimizer_name,
                eps=self.adam_eps,
            )
            unit = rms(delta) / denominator
            if not math.isfinite(unit) or unit < 0.0:
                raise RuntimeError(
                    f"Expected a finite non-negative proposal unit for {name!r}. "
                    f"Provided value: {unit!r}."
                )
            self.samples[name].append(unit)

        self.used_batches += 1
        if self.used_batches not in self.batch_counts:
            return False

        half = self.used_batches // 2
        self.split_statistics = {}
        unstable = []
        for name, values in self.samples.items():
            first = self._statistic(name, values[:half])
            second = self._statistic(name, values[half:self.used_batches])
            difference = _relative_difference(first, second)
            self.split_statistics[name] = {
                "first_half": first,
                "second_half": second,
                "relative_difference": difference,
            }
            required_for_stability = (
                self.stability_scope == "all" or not name.startswith("Bias_")
            )
            if required_for_stability and difference > self.stability_tolerance:
                unstable.append(name)
        self.stable = not unstable
        return self.stable or self.used_batches == self.batch_counts[-1]

    def result(self) -> dict[str, Any]:
        if self.topology is None or self.used_batches not in self.batch_counts:
            raise RuntimeError(
                "Expected the probe to reach a configured batch count. "
                f"Provided value: {self.used_batches!r}."
            )
        mutated = [
            name
            for name, initial in self.initial_states.items()
            if not torch.equal(initial, self.parameter_refs[name].state.detach())
        ]
        if mutated:
            raise RuntimeError(
                "Expected probing not to mutate parameters. "
                f"Provided value: {mutated!r}."
            )
        weight_units = {
            name: self._statistic(name, self.samples[name][:self.used_batches])
            for name in self.topology["weight_names"]
        }
        bias_units = {
            name: self._statistic(name, self.samples[name][:self.used_batches])
            for name in self.topology["bias_to_weight"]
        }
        invalid = {
            name: value
            for name, value in weight_units.items()
            if not math.isfinite(value) or value <= 0.0
        }
        unstable = [
            name
            for name, values in self.split_statistics.items()
            if (
                self.stability_scope == "all" or not name.startswith("Bias_")
            )
            and values["relative_difference"] > self.stability_tolerance
        ]
        complete = self.stable and not invalid
        return {
            "schema_version": "conv-rho-probe/v1",
            "status": "complete" if complete else "unresolved_probe",
            "probe_stable": self.stable,
            "proposal_units_valid": not invalid,
            "invalid_weight_proposal_units": invalid,
            "optimizer_name": self.optimizer_name,
            "parameter_names": self.topology["parameter_names"],
            "weight_names": self.topology["weight_names"],
            "bias_to_weight": self.topology["bias_to_weight"],
            "used_batches": self.used_batches,
            "batch_counts": list(self.batch_counts),
            "stability_tolerance": self.stability_tolerance,
            "stability_scope": self.stability_scope,
            "unstable_parameters": unstable,
            "split_half_statistics": self.split_statistics,
            "initial_parameter_sha256": parameter_sha256(self.initial_states),
            "initial_weight_rms_by_parameter": self.initial_weight_rms,
            "normalization_unit_by_weight": weight_units,
            "bias_q90_unit_by_parameter": bias_units,
            "proposal_unit_samples_by_parameter": {
                name: values[:self.used_batches] for name, values in self.samples.items()
            },
            "official_test_read": False,
        }


class SafetyRejection(RuntimeError):
    """A restarted canary or candidate crossed a terminal safety gate."""

    def __init__(self, failure: Mapping[str, Any]):
        self.failure = dict(failure)
        super().__init__(
            "Training safety rejection: "
            + json.dumps(self.failure, sort_keys=True, allow_nan=False)
        )


class TrainingSafetyMonitor:
    """Observe optimizer transitions without treating clipping as rejection.

    Bound occupancy and projection efficiency are retained for diagnosis and
    selection tie-breaking only.  The terminal gates are sustained loss-EMA
    growth and sustained bounded-weight gradient-RMS growth; the trainer
    independently fails on any non-finite loss, gradient, parameter, update,
    or optimizer state.
    """

    def __init__(
        self,
        *,
        warmup_steps: int = 32,
        ema_decay: float = 0.98,
        loss_factor: float = 4.0,
        gradient_factor: float = 100.0,
        persistence: int = 8,
        zero_proposal_epsilon: float = 1e-30,
        bound_occupancy_increase_maximum: float | None = None,
        projection_efficiency_minimum: float | None = None,
        boundary_persistence: int = 16,
    ):
        self.warmup_steps = int(warmup_steps)
        self.ema_decay = float(ema_decay)
        self.loss_factor = float(loss_factor)
        self.gradient_factor = float(gradient_factor)
        self.persistence = int(persistence)
        self.zero_proposal_epsilon = float(zero_proposal_epsilon)
        self.bound_occupancy_increase_maximum = (
            None
            if bound_occupancy_increase_maximum is None
            else float(bound_occupancy_increase_maximum)
        )
        self.projection_efficiency_minimum = (
            None
            if projection_efficiency_minimum is None
            else float(projection_efficiency_minimum)
        )
        self.boundary_persistence = int(boundary_persistence)
        if self.warmup_steps <= 0 or self.persistence <= 0:
            raise ValueError("Safety warmup and persistence must be positive.")
        for value, label in (
            (self.ema_decay, "ema_decay"),
            (self.loss_factor, "loss_factor"),
            (self.gradient_factor, "gradient_factor"),
            (self.zero_proposal_epsilon, "zero_proposal_epsilon"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Expected finite non-negative {label}, got {value!r}.")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(
                f"Expected ema_decay in [0, 1), got {self.ema_decay!r}."
            )
        if self.boundary_persistence <= 0:
            raise ValueError("Boundary-gate persistence must be positive.")
        if (
            self.bound_occupancy_increase_maximum is not None
            and not 0.0 <= self.bound_occupancy_increase_maximum <= 1.0
        ):
            raise ValueError(
                "Expected bound_occupancy_increase_maximum in [0, 1] or None, "
                f"got {self.bound_occupancy_increase_maximum!r}."
            )
        if (
            self.projection_efficiency_minimum is not None
            and not 0.0 <= self.projection_efficiency_minimum <= 1.0
        ):
            raise ValueError(
                "Expected projection_efficiency_minimum in [0, 1] or None, "
                f"got {self.projection_efficiency_minimum!r}."
            )

        self.step = 0
        self.loss_ema: float | None = None
        self.prior_loss_ema_minimum = math.inf
        self.loss_streak = 0
        self.maximum_loss_streak = 0
        self.gradient_samples: dict[str, list[float]] = {}
        self.gradient_reference: dict[str, float] = {}
        self.gradient_streak: dict[str, int] = {}
        self.maximum_gradient_streak: dict[str, int] = {}
        self.maximum_gradient_rms: dict[str, float] = {}
        self.initial_occupancy: dict[str, float] = {}
        self.maximum_occupancy: dict[str, float] = {}
        self.final_occupancy: dict[str, float] = {}
        self.projection_efficiencies: dict[str, list[float]] = {}
        self.proposed_update_rms: dict[str, list[float]] = {}
        self.applied_update_rms: dict[str, list[float]] = {}
        self.occupancy_increase_streak: dict[str, int] = {}
        self.maximum_occupancy_increase_streak: dict[str, int] = {}
        self.maximum_occupancy_increase: dict[str, float] = {}
        self.projection_efficiency_streak: dict[str, int] = {}
        self.maximum_projection_efficiency_streak: dict[str, int] = {}
        self.failure: dict[str, Any] | None = None

    @staticmethod
    def _occupancy(state: torch.Tensor, parameter: Any) -> float:
        lower = float(parameter.min_cond)
        upper = float(parameter.max_cond)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(
                f"Expected finite ordered bounds for {parameter.name!r}; "
                f"provided {(parameter.min_cond, parameter.max_cond)!r}."
            )
        values = state.detach()
        return float(((values <= lower) | (values >= upper)).to(torch.float64).mean().item())

    def _failure_record(
        self,
        kind: str,
        parameter: str | None,
        *,
        persistence: int | None = None,
    ) -> dict[str, Any]:
        persistence = self.persistence if persistence is None else int(persistence)
        return {
            "kind": kind,
            "parameter": parameter,
            "onset_step": self.step - persistence + 1,
            "confirmed_step": self.step,
        }

    def __call__(self, batch: Mapping[str, Any]) -> bool:
        if self.failure is not None:
            raise RuntimeError("Safety monitor called after a confirmed rejection.")
        self.step += 1
        loss = float(batch["loss"])
        if not math.isfinite(loss):
            raise SafetyRejection(
                {
                    "kind": "nonfinite_loss",
                    "parameter": None,
                    "onset_step": self.step,
                    "confirmed_step": self.step,
                }
            )
        self.loss_ema = (
            loss
            if self.loss_ema is None
            else self.ema_decay * self.loss_ema + (1.0 - self.ema_decay) * loss
        )

        parameters = {
            str(getattr(parameter, "name", "")).strip(): parameter
            for parameter in batch["parameters"]
        }
        gradients = {
            str(getattr(parameter, "name", "")).strip(): gradient
            for parameter, gradient in zip(batch["parameters"], batch["gradients"])
        }
        pre = batch["pre_optimizer_states"]
        proposed = batch["post_optimizer_states"]
        applied = batch["post_projection_states"]
        names = sorted(pre)
        if not names or set(names) != set(proposed) or set(names) != set(applied):
            raise RuntimeError(
                "Expected matched bounded-weight transition states. "
                f"Provided names: pre={sorted(pre)!r}, "
                f"proposed={sorted(proposed)!r}, applied={sorted(applied)!r}."
            )

        failures: list[dict[str, Any]] = []
        for name in names:
            if name not in parameters or name not in gradients:
                raise RuntimeError(f"Missing parameter or gradient for {name!r}.")
            gradient_rms = rms(gradients[name])
            proposal_rms = rms(proposed[name] - pre[name])
            achieved_rms = rms(applied[name] - pre[name])
            occupancy = self._occupancy(applied[name], parameters[name])
            for value, label in (
                (gradient_rms, "gradient RMS"),
                (proposal_rms, "proposed update RMS"),
                (achieved_rms, "applied update RMS"),
                (occupancy, "bound occupancy"),
            ):
                if not math.isfinite(value) or value < 0.0:
                    raise SafetyRejection(
                        {
                            "kind": "nonfinite_diagnostic",
                            "parameter": name,
                            "diagnostic": label,
                            "onset_step": self.step,
                            "confirmed_step": self.step,
                        }
                    )

            if name not in self.gradient_samples:
                initial = self._occupancy(pre[name], parameters[name])
                self.gradient_samples[name] = []
                self.gradient_streak[name] = 0
                self.maximum_gradient_streak[name] = 0
                self.maximum_gradient_rms[name] = gradient_rms
                self.initial_occupancy[name] = initial
                self.maximum_occupancy[name] = initial
                self.projection_efficiencies[name] = []
                self.proposed_update_rms[name] = []
                self.applied_update_rms[name] = []
                self.occupancy_increase_streak[name] = 0
                self.maximum_occupancy_increase_streak[name] = 0
                self.maximum_occupancy_increase[name] = 0.0
                self.projection_efficiency_streak[name] = 0
                self.maximum_projection_efficiency_streak[name] = 0

            self.maximum_gradient_rms[name] = max(
                self.maximum_gradient_rms[name], gradient_rms
            )
            self.maximum_occupancy[name] = max(
                self.maximum_occupancy[name], occupancy
            )
            self.final_occupancy[name] = occupancy
            self.proposed_update_rms[name].append(proposal_rms)
            self.applied_update_rms[name].append(achieved_rms)
            occupancy_increase = occupancy - self.initial_occupancy[name]
            self.maximum_occupancy_increase[name] = max(
                self.maximum_occupancy_increase[name], occupancy_increase
            )
            if self.bound_occupancy_increase_maximum is not None:
                if occupancy_increase > self.bound_occupancy_increase_maximum:
                    self.occupancy_increase_streak[name] += 1
                    self.maximum_occupancy_increase_streak[name] = max(
                        self.maximum_occupancy_increase_streak[name],
                        self.occupancy_increase_streak[name],
                    )
                    if (
                        self.occupancy_increase_streak[name]
                        == self.boundary_persistence
                    ):
                        failures.append(
                            self._failure_record(
                                "bound_occupancy_increase",
                                name,
                                persistence=self.boundary_persistence,
                            )
                        )
                else:
                    self.occupancy_increase_streak[name] = 0
            if proposal_rms > self.zero_proposal_epsilon:
                efficiency = achieved_rms / proposal_rms
                if math.isfinite(efficiency) and efficiency >= 0.0:
                    self.projection_efficiencies[name].append(efficiency)
                    if self.projection_efficiency_minimum is not None:
                        if efficiency < self.projection_efficiency_minimum:
                            self.projection_efficiency_streak[name] += 1
                            self.maximum_projection_efficiency_streak[name] = max(
                                self.maximum_projection_efficiency_streak[name],
                                self.projection_efficiency_streak[name],
                            )
                            if (
                                self.projection_efficiency_streak[name]
                                == self.boundary_persistence
                            ):
                                failures.append(
                                    self._failure_record(
                                        "projection_efficiency_below_minimum",
                                        name,
                                        persistence=self.boundary_persistence,
                                    )
                                )
                        else:
                            self.projection_efficiency_streak[name] = 0
            elif self.projection_efficiency_minimum is not None:
                self.projection_efficiency_streak[name] = 0

            if self.step <= self.warmup_steps:
                self.gradient_samples[name].append(gradient_rms)
                if self.step == self.warmup_steps:
                    self.gradient_reference[name] = linear_quantile(
                        self.gradient_samples[name], 0.5
                    )
            else:
                reference = self.gradient_reference[name]
                if gradient_rms > self.gradient_factor * reference:
                    self.gradient_streak[name] += 1
                    self.maximum_gradient_streak[name] = max(
                        self.maximum_gradient_streak[name],
                        self.gradient_streak[name],
                    )
                    if self.gradient_streak[name] == self.persistence:
                        failures.append(
                            self._failure_record("gradient_rms_explosion", name)
                        )
                else:
                    self.gradient_streak[name] = 0

        if self.step <= self.warmup_steps:
            self.prior_loss_ema_minimum = min(
                self.prior_loss_ema_minimum, self.loss_ema
            )
        else:
            if self.loss_ema > self.loss_factor * self.prior_loss_ema_minimum:
                self.loss_streak += 1
                self.maximum_loss_streak = max(
                    self.maximum_loss_streak, self.loss_streak
                )
                if self.loss_streak == self.persistence:
                    failures.append(self._failure_record("loss_ema_explosion", None))
            else:
                self.loss_streak = 0
            self.prior_loss_ema_minimum = min(
                self.prior_loss_ema_minimum, self.loss_ema
            )

        if failures:
            self.failure = min(
                failures,
                key=lambda item: (
                    item["onset_step"],
                    item["kind"],
                    item["parameter"] or "",
                ),
            )
            raise SafetyRejection(self.failure)
        return False

    def summary(self) -> dict[str, Any]:
        projection_by_parameter = {
            name: (
                linear_quantile(values, 0.5) if values else None
            )
            for name, values in self.projection_efficiencies.items()
        }
        all_projection = [
            value
            for values in self.projection_efficiencies.values()
            for value in values
        ]
        return {
            "schema_version": "conv-rho-training-safety/v1",
            "processed_steps": self.step,
            "terminal_gates": {
                "nonfinite_values": True,
                "warmup_steps": self.warmup_steps,
                "loss_ema_decay": self.ema_decay,
                "loss_ema_factor": self.loss_factor,
                "gradient_rms_factor": self.gradient_factor,
                "persistence_steps": self.persistence,
                "bound_occupancy_increase_maximum": (
                    self.bound_occupancy_increase_maximum
                ),
                "projection_efficiency_minimum": (
                    self.projection_efficiency_minimum
                ),
                "zero_proposal_epsilon": self.zero_proposal_epsilon,
                "boundary_persistence_steps": self.boundary_persistence,
            },
            "report_only_diagnostics": {
                "bound_occupancy": (
                    self.bound_occupancy_increase_maximum is None
                ),
                "projection_efficiency": (
                    self.projection_efficiency_minimum is None
                ),
                "used_for_rejection": bool(
                    self.bound_occupancy_increase_maximum is not None
                    or self.projection_efficiency_minimum is not None
                ),
            },
            "loss_ema": self.loss_ema,
            "prior_loss_ema_minimum": (
                self.prior_loss_ema_minimum
                if math.isfinite(self.prior_loss_ema_minimum)
                else None
            ),
            "maximum_loss_violation_streak": self.maximum_loss_streak,
            "first_warmup_gradient_rms_median_by_parameter": (
                self.gradient_reference
            ),
            "maximum_gradient_rms_by_parameter": self.maximum_gradient_rms,
            "maximum_gradient_violation_streak_by_parameter": (
                self.maximum_gradient_streak
            ),
            "initial_bound_occupancy_by_parameter": self.initial_occupancy,
            "maximum_bound_occupancy_by_parameter": self.maximum_occupancy,
            "final_bound_occupancy_by_parameter": self.final_occupancy,
            "maximum_bound_occupancy_increase_by_parameter": (
                self.maximum_occupancy_increase
            ),
            "maximum_bound_occupancy_increase_streak_by_parameter": (
                self.maximum_occupancy_increase_streak
            ),
            "maximum_projection_efficiency_violation_streak_by_parameter": (
                self.maximum_projection_efficiency_streak
            ),
            "median_projection_efficiency_by_parameter": projection_by_parameter,
            "median_projection_efficiency": (
                linear_quantile(all_projection, 0.5)
                if all_projection
                else None
            ),
            "median_proposed_update_rms_by_parameter": {
                name: linear_quantile(values, 0.5)
                for name, values in self.proposed_update_rms.items()
                if values
            },
            "median_applied_update_rms_by_parameter": {
                name: linear_quantile(values, 0.5)
                for name, values in self.applied_update_rms.items()
                if values
            },
            "safety_failure": self.failure,
        }


def derive_learning_rates(
    probe: Mapping[str, Any],
    rho_conv: float,
    rho_dense: float,
    bias_policy: str = "q90_cap",
) -> dict[str, float]:
    if probe.get("probe_stable") is not True:
        raise ValueError(
            "Expected a stable optimizer probe before deriving learning rates. "
            f"Provided value: {probe.get('status')!r}."
        )
    rho_conv = _positive(rho_conv, "rho_conv")
    rho_dense = _positive(rho_dense, "rho_dense")
    if bias_policy not in {"q90_cap", "tied", "zero"}:
        raise ValueError(
            "Expected bias_policy to be one of ('q90_cap', 'tied', 'zero'). "
            f"Provided value: {bias_policy!r}."
        )

    names = [str(name) for name in probe["parameter_names"]]
    topology = parameter_topology(names)
    units = probe["normalization_unit_by_weight"]
    rates: dict[str, float] = {}
    for name in topology["weight_names"]:
        unit = _positive(units.get(name), f"proposal unit for {name!r}")
        target = rho_conv if name.startswith("ConvWeight_") else rho_dense
        rates[name] = target / unit

    bias_units = (
        {} if bias_policy == "zero" else probe["bias_q90_unit_by_parameter"]
    )
    for bias, attached in topology["bias_to_weight"].items():
        if bias_policy == "zero":
            rates[bias] = 0.0
            continue
        attached_rate = rates[attached]
        unit = float(bias_units[bias])
        if not math.isfinite(unit) or unit < 0.0:
            raise ValueError(
                f"Expected bias unit for {bias!r} to be non-negative and finite. "
                f"Provided value: {unit!r}."
            )
        rates[bias] = (
            attached_rate
            if bias_policy == "tied" or unit == 0.0
            else min(attached_rate, rho_conv / unit)
        )
    return {name: float(rates[name]) for name in names}


def _optimizer_config(name: str, rates: Sequence[float]) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": name,
        "learning_rate": list(rates),
        "lr_decay": 1.0,
        "momentum": 0.0,
        "weight_decay": 0.0,
    }
    if name == "Adam":
        config.update(betas=[0.9, 0.999], eps=1e-8)
    return config


def prepare_training_config(
    base: Mapping[str, Any],
    optimizer_name: str,
    rates: Sequence[float],
    *,
    epochs: int,
    max_batches: int | None,
    max_validation_batches: int | None,
    split_seed: int,
    shuffle_seed: int,
    validation_batch_size: int,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    dataset_key = str(config.get("lab", {}).get("dataset_key", "mnist"))
    if dataset_key != "mnist":
        raise ValueError(
            "Expected the two-rho search source config to use dataset_key 'mnist'. "
            f"Provided value: {dataset_key!r}."
        )
    dataset = config["datasets"][dataset_key]
    params = dataset["params"]
    dataset["factory"] = "labs.datasets.MnistTrainValidationDataset"
    params.setdefault("name", dataset_key)
    params["train"] = True
    params["split_seed"] = int(split_seed)
    params["shuffle_seed"] = int(shuffle_seed)
    params["validation_batch_size"] = int(validation_batch_size)

    config.setdefault("lab", {})["epochs"] = int(epochs)
    config["training_algorithm"] = "BP"
    config["lr"] = list(rates)
    config["lr_decay"] = 1.0
    config["optimizer"] = _optimizer_config(optimizer_name, rates)
    config["max_batches"] = max_batches
    config["max_test_batches"] = max_validation_batches
    return config


def _configured_rate_count(config: Mapping[str, Any]) -> int:
    optimizer = config.get("optimizer", {})
    values = optimizer.get("learning_rate") if isinstance(optimizer, Mapping) else None
    if not isinstance(values, (list, tuple)):
        values = config.get("lr")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(
            "Expected the source config to contain an explicit optimizer learning-rate vector. "
            f"Provided value: {values!r}."
        )
    return len(values)


def _run_trainer(
    config_path: Path,
    output_dir: Path,
    *,
    device: str | None,
    gradient_callback=None,
    optimizer_step_callback=None,
    apply_optimizer_steps: bool = True,
    reporting_run_dir: Path | None = None,
    checkpoint_every_epoch: bool = False,
) -> dict[str, Any]:
    from labs.mnist_train import train_mnist_conv

    config = json.loads(config_path.read_text(encoding="utf-8"))
    return train_mnist_conv(
        config_path=config_path,
        epochs=int(config["lab"]["epochs"]),
        lr=config["lr"],
        beta=config.get("beta"),
        log_interval=int(config.get("log_interval", 100)),
        max_batches=config.get("max_batches"),
        max_test_batches=config.get("max_test_batches"),
        device=device,
        dataset_key=str(config["lab"].get("dataset_key", "mnist")),
        output_dir=output_dir,
        model_key=str(config["lab"]["model_key"]),
        training_algorithm="BP",
        seed=config.get("seed"),
        lr_decay=1.0,
        gradient_callback=gradient_callback,
        optimizer_step_callback=optimizer_step_callback,
        apply_optimizer_steps=apply_optimizer_steps,
        reporting_run_dir=reporting_run_dir,
        checkpoint_every_epoch=checkpoint_every_epoch,
    )


def _float_token(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p").replace("+", "")


def _summary_row(index: int, rho_conv: float, rho_dense: float, cell_dir: Path) -> dict[str, Any]:
    cell = json.loads((cell_dir / "cell.json").read_text(encoding="utf-8"))
    metrics_path = cell_dir / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if cell["status"] in {"complete", "candidate_rejected_post_training_tk"}
        and metrics_path.exists()
        else {}
    )
    diagnostics_path = cell_dir / "safety_diagnostics.json"
    diagnostics = (
        json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics_path.exists()
        else {}
    )
    return {
        "index": index,
        "optimizer": cell["optimizer"],
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "status": cell["status"],
        "final_validation_loss": metrics.get("final_test_loss"),
        "final_validation_accuracy": metrics.get("final_test_accuracy"),
        "best_validation_accuracy": metrics.get("best_test_accuracy"),
        "selection_eligible": cell.get("selection_eligible"),
        "median_projection_efficiency": diagnostics.get(
            "median_projection_efficiency"
        ),
        "learning_rates": json.dumps(cell["learning_rate_vector"]),
        "path": str(cell_dir),
    }


def _terminal_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "best_epoch": metrics.get("best_epoch"),
        "train": {
            "final_loss": metrics.get("final_train_loss"),
            "final_accuracy": metrics.get("final_train_accuracy"),
            "best_accuracy": metrics.get("best_train_accuracy"),
        },
        "validation": {
            "final_loss": metrics.get("final_test_loss"),
            "final_accuracy": metrics.get("final_test_accuracy"),
            "best_accuracy": metrics.get("best_test_accuracy"),
        },
    }


def _write_summary(output_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_json(output_root / "summary.json", list(rows))
    fieldnames = list(rows[0]) if rows else [
        "index", "optimizer", "rho_conv", "rho_dense", "status",
        "final_validation_loss", "final_validation_accuracy",
        "best_validation_accuracy", "selection_eligible",
        "median_projection_efficiency", "learning_rates", "path",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _indexed_cells(
    rhos_conv: Sequence[float],
    rhos_dense: Sequence[float],
    index: int | None,
) -> tuple[list[tuple[float, float]], list[tuple[int, float, float]]]:
    pairs = list(itertools.product(rhos_conv, rhos_dense))
    if index is None:
        return pairs, [
            (cell_index, rho_conv, rho_dense)
            for cell_index, (rho_conv, rho_dense) in enumerate(pairs)
        ]
    if index < 0 or index >= len(pairs):
        raise ValueError(
            f"Expected --index in [0, {len(pairs) - 1}]. Provided value: {index!r}."
        )
    rho_conv, rho_dense = pairs[index]
    return pairs, [(index, rho_conv, rho_dense)]


def _git_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    fingerprint = hashlib.sha256()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=root,
            check=False,
            capture_output=True,
        )
    except OSError:
        commit_value = None
        dirty_value = None
    else:
        fingerprint.update(status.stdout.encode("utf-8"))
        fingerprint.update(diff.stdout)
        commit_value = commit.stdout.strip() if commit.returncode == 0 else None
        dirty_value = bool(status.stdout.strip()) if status.returncode == 0 else None
    fingerprint.update(Path(__file__).read_bytes())
    if commit_value is None:
        archive_commit = os.environ.get("EXPERIMENT_SOURCE_COMMIT", "").strip()
        archive_sha256 = os.environ.get(
            "EXPERIMENT_SOURCE_ARCHIVE_SHA256", ""
        ).strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}", archive_commit) is None:
            raise RuntimeError(
                "Git source identity is unavailable. A frozen source archive "
                "must provide EXPERIMENT_SOURCE_COMMIT as exactly 40 hexadecimal "
                "characters."
            )
        if re.fullmatch(r"[0-9a-fA-F]{64}", archive_sha256) is None:
            raise RuntimeError(
                "Git source identity is unavailable. A frozen source archive "
                "must provide EXPERIMENT_SOURCE_ARCHIVE_SHA256 as exactly 64 "
                "hexadecimal characters."
            )
        archive_commit = archive_commit.lower()
        archive_sha256 = archive_sha256.lower()
        return {
            "commit": archive_commit,
            "dirty": False,
            "working_tree_sha256": archive_sha256,
            "source_kind": "frozen_archive",
            "source_archive_sha256": archive_sha256,
        }
    return {
        "commit": commit_value,
        "dirty": dirty_value,
        "working_tree_sha256": fingerprint.hexdigest(),
    }


def _cell_signature(
    resolved: Mapping[str, Any],
    probe: Mapping[str, Any],
    rho_conv: float,
    rho_dense: float,
) -> dict[str, Any]:
    probe_digest = hashlib.sha256(
        json.dumps(probe, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    signature = {
        "source_config_sha256": resolved["source_config_sha256"],
        "code": resolved["code"],
        "optimizer": resolved["optimizer"],
        "bias_policy": resolved["bias_policy"],
        "probe_sha256": probe_digest,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "epochs": resolved["epochs"],
        "max_batches": resolved["max_batches"],
        "max_validation_batches": resolved["max_validation_batches"],
        "split_seed": resolved["split_seed"],
        "shuffle_seed": resolved["shuffle_seed"],
        "canary_steps": resolved.get("canary_steps"),
        "expected_candidate_steps": resolved.get("expected_candidate_steps"),
        "minimum_validation_accuracy": resolved.get(
            "minimum_validation_accuracy"
        ),
        "safety": resolved.get("safety"),
    }
    if "checkpoint_every_epoch" in resolved:
        signature["checkpoint_every_epoch"] = resolved["checkpoint_every_epoch"]
    if "post_candidate_gate_name" in resolved:
        signature["post_candidate_gate_name"] = resolved[
            "post_candidate_gate_name"
        ]
    return signature


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_path = Path(args.config).expanduser().resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    optimizer_name = args.optimizer or str(base.get("optimizer", {}).get("name", ""))
    if optimizer_name not in {"SGD", "Adam"}:
        raise ValueError(
            "Expected --optimizer or config optimizer name to be one of ('SGD', 'Adam'). "
            f"Provided value: {optimizer_name!r}."
        )
    epochs = int(args.epochs if args.epochs is not None else base["lab"]["epochs"])
    if epochs <= 0:
        raise ValueError(f"Expected epochs to be positive. Provided value: {epochs!r}.")
    rhos_conv = [_positive(value, "rho_conv") for value in args.rho_conv]
    rhos_dense = [_positive(value, "rho_dense") for value in args.rho_dense]
    counts = tuple(args.probe_batches)
    canary_steps = int(getattr(args, "canary_steps", DEFAULT_CANARY_STEPS))
    if canary_steps <= 0:
        raise ValueError(
            f"Expected --canary-steps to be positive. Provided value: {canary_steps!r}."
        )
    expected_candidate_steps = getattr(args, "expected_candidate_steps", None)
    if expected_candidate_steps is not None:
        expected_candidate_steps = int(expected_candidate_steps)
        if expected_candidate_steps <= 0:
            raise ValueError(
                "Expected --expected-candidate-steps to be positive when supplied. "
                f"Provided value: {expected_candidate_steps!r}."
            )
    minimum_validation_accuracy = float(
        getattr(args, "minimum_validation_accuracy", 0.90)
    )
    if not 0.0 <= minimum_validation_accuracy <= 1.0:
        raise ValueError(
            "Expected --minimum-validation-accuracy in [0, 1]. "
            f"Provided value: {minimum_validation_accuracy!r}."
        )
    checkpoint_every_epoch = bool(
        getattr(args, "checkpoint_every_epoch", False)
    )
    post_candidate_gate_name = getattr(args, "post_candidate_gate_name", None)
    post_candidate_callback = getattr(args, "post_candidate_callback", None)
    if post_candidate_gate_name is not None and not isinstance(
        post_candidate_gate_name, str
    ):
        raise TypeError("post_candidate_gate_name must be a string when supplied.")
    if post_candidate_callback is not None and not callable(post_candidate_callback):
        raise TypeError("post_candidate_callback must be callable when supplied.")
    safety = {
        "warmup_steps": int(getattr(args, "safety_warmup_steps", 32)),
        "ema_decay": float(getattr(args, "safety_ema_decay", 0.98)),
        "loss_factor": float(getattr(args, "safety_loss_factor", 4.0)),
        "gradient_factor": float(
            getattr(args, "safety_gradient_factor", 100.0)
        ),
        "persistence": int(getattr(args, "safety_persistence", 8)),
        "bound_occupancy_increase_maximum": getattr(
            args, "safety_bound_occupancy_increase_maximum", None
        ),
        "projection_efficiency_minimum": getattr(
            args, "safety_projection_efficiency_minimum", None
        ),
        "zero_proposal_epsilon": float(
            getattr(args, "safety_zero_proposal_epsilon", 1e-30)
        ),
        "boundary_persistence": int(
            getattr(args, "safety_boundary_persistence", 16)
        ),
    }
    safety["bound_occupancy"] = (
        "report_only"
        if safety["bound_occupancy_increase_maximum"] is None
        else "reject_persistent_increase"
    )
    safety["projection_efficiency"] = (
        "report_only"
        if safety["projection_efficiency_minimum"] is None
        else "reject_persistent_low_efficiency"
    )
    # Validate the supplied safety values before writing a resolved contract.
    TrainingSafetyMonitor(
        warmup_steps=safety["warmup_steps"],
        ema_decay=safety["ema_decay"],
        loss_factor=safety["loss_factor"],
        gradient_factor=safety["gradient_factor"],
        persistence=safety["persistence"],
        bound_occupancy_increase_maximum=safety[
            "bound_occupancy_increase_maximum"
        ],
        projection_efficiency_minimum=safety[
            "projection_efficiency_minimum"
        ],
        zero_proposal_epsilon=safety["zero_proposal_epsilon"],
        boundary_persistence=safety["boundary_persistence"],
    )
    rate_count = _configured_rate_count(base)
    output_root = Path(args.output_root).expanduser().resolve()
    study_id = getattr(args, "study_id", None) or output_root.name
    target = getattr(args, "target", None)
    all_pairs, selected_cells = _indexed_cells(rhos_conv, rhos_dense, args.index)
    if args.collect_only and args.index is not None:
        raise ValueError("--collect-only and --index are mutually exclusive.")
    resolved = {
        "schema_version": "conv-rho-search/v1",
        "source_config": str(base_path),
        "source_config_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "code": _git_state(),
        "optimizer": optimizer_name,
        "rho_conv": rhos_conv,
        "rho_dense": rhos_dense,
        "bias_policy": args.bias_policy,
        "probe_stability_scope": (
            "weights_only" if args.bias_policy == "zero" else "all"
        ),
        "probe_batches": list(counts),
        "stability_tolerance": args.stability_tolerance,
        "epochs": epochs,
        "max_batches": args.max_batches,
        "max_validation_batches": args.max_validation_batches,
        "split_seed": args.split_seed,
        "shuffle_seed": args.shuffle_seed,
        "validation_batch_size": args.validation_batch_size,
        "device": args.device,
        "canary_steps": canary_steps,
        "expected_candidate_steps": expected_candidate_steps,
        "minimum_validation_accuracy": minimum_validation_accuracy,
        "safety": safety,
        "target": target,
    }
    if checkpoint_every_epoch:
        resolved["checkpoint_every_epoch"] = True
    if post_candidate_gate_name is not None:
        resolved["post_candidate_gate_name"] = post_candidate_gate_name
    if args.dry_run:
        planned = dict(resolved)
        planned["cells"] = len(all_pairs)
        planned["selected_indices"] = [index for index, _, _ in selected_cells]
        planned["mode"] = "collect" if args.collect_only else "run"
        return planned

    output_root.mkdir(parents=True, exist_ok=True)
    resolved_path = output_root / "resolved.json"
    if resolved_path.exists():
        existing_resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        if existing_resolved != resolved:
            raise RuntimeError(
                "Existing resolved.json does not match this rho-search command. "
                f"Path: {resolved_path}."
            )
    elif args.index is not None or args.collect_only:
        raise RuntimeError(
            "Indexed and collection modes require a completed probe invocation to "
            f"create {resolved_path} first."
        )
    else:
        _write_json(resolved_path, resolved)
    probe_path = output_root / "probe.json"
    probe = None
    if probe_path.exists() and not args.force:
        candidate = json.loads(probe_path.read_text(encoding="utf-8"))
        if candidate.get("search_signature") == resolved:
            probe = candidate
    if probe is None:
        probe_dir = output_root / "probe"
        probe_config = prepare_training_config(
            base,
            optimizer_name,
            [0.0] * rate_count,
            epochs=1,
            max_batches=max(counts),
            max_validation_batches=1,
            split_seed=args.split_seed,
            shuffle_seed=args.shuffle_seed,
            validation_batch_size=args.validation_batch_size,
        )
        probe_config_path = probe_dir / "source_config.json"
        _write_json(probe_config_path, probe_config)
        callback = OptimizerProbe(
            optimizer_name,
            counts,
            args.stability_tolerance,
            adam_eps=float(probe_config["optimizer"].get("eps", 1e-8)),
            stability_scope=resolved["probe_stability_scope"],
        )
        _run_trainer(
            probe_config_path,
            probe_dir,
            device=args.device,
            gradient_callback=callback,
            apply_optimizer_steps=False,
        )
        probe = callback.result()
        probe_metrics = json.loads(
            (probe_dir / "metrics.json").read_text(encoding="utf-8")
        )
        probe["dataset_provenance"] = probe_metrics["dataset_provenance"]
        probe["search_signature"] = resolved
        _write_json(probe_path, probe)

    zero_bias_names = [
        str(name).strip()
        for name in probe.get("parameter_names", ())
        if str(name).strip().startswith("Bias_")
    ]
    if args.bias_policy == "zero":
        probe_zero_bias_verification = verify_zero_bias_run_checkpoints(
            output_root / "probe",
            zero_bias_names,
        )
        probe["zero_bias_checkpoint_verification"] = (
            probe_zero_bias_verification
        )
        _write_json(probe_path, probe)

    if probe.get("status") != "complete":
        return {
            "probe": str(probe_path),
            "status": "unresolved_probe",
            "probe_stable": probe.get("probe_stable") is True,
            "proposal_units_valid": probe.get("proposal_units_valid") is True,
            "unstable_parameters": probe.get("unstable_parameters", []),
            "invalid_weight_proposal_units": probe.get(
                "invalid_weight_proposal_units", {}
            ),
        }
    if args.probe_only:
        return {"probe": str(probe_path), "status": "complete"}

    if args.collect_only:
        rows = []
        missing = []
        for index, (rho_conv, rho_dense) in enumerate(all_pairs):
            name = (
                f"{index:03d}_rc_{_float_token(rho_conv)}_"
                f"rd_{_float_token(rho_dense)}"
            )
            cell_dir = output_root / "cells" / name
            cell_path = cell_dir / "cell.json"
            if not cell_path.exists():
                missing.append(str(cell_path))
                continue
            cell = json.loads(cell_path.read_text(encoding="utf-8"))
            expected_signature = _cell_signature(
                resolved,
                probe,
                rho_conv,
                rho_dense,
            )
            if cell.get("signature") != expected_signature:
                raise RuntimeError(
                    f"Cell signature mismatch while collecting {cell_path}."
                )
            if cell.get("status") not in {
                "complete",
                "canary_rejected_nonfinite",
                "canary_rejected_safety",
                "candidate_rejected_nonfinite",
                "candidate_rejected_safety",
                "candidate_rejected_post_training_tk",
            }:
                missing.append(f"{cell_path} (status={cell.get('status')!r})")
                continue
            if cell["status"] in {
                "complete",
                "candidate_rejected_post_training_tk",
            } and not (cell_dir / "metrics.json").exists():
                missing.append(str(cell_dir / "metrics.json"))
                continue
            rows.append(_summary_row(index, rho_conv, rho_dense, cell_dir))
        if missing:
            raise RuntimeError(
                "Cannot collect an incomplete rho grid. Missing or incomplete entries: "
                + ", ".join(missing)
            )
        _write_summary(output_root, rows)
        return {
            "status": "complete",
            "probe": str(probe_path),
            "summary": str(output_root / "summary.json"),
            "cells": len(rows),
        }

    rows = []
    for index, rho_conv, rho_dense in selected_cells:
        name = (
            f"{index:03d}_rc_{_float_token(rho_conv)}_"
            f"rd_{_float_token(rho_dense)}"
        )
        cell_dir = output_root / "cells" / name
        cell_path = cell_dir / "cell.json"
        signature = _cell_signature(resolved, probe, rho_conv, rho_dense)
        if cell_path.exists() and not args.force:
            existing = json.loads(cell_path.read_text(encoding="utf-8"))
            reusable = (
                existing.get("signature") == signature
                and (
                    (
                        existing.get("status") == "complete"
                        and (cell_dir / "metrics.json").exists()
                    )
                    or existing.get("status")
                    in {
                        "canary_rejected_nonfinite",
                        "canary_rejected_safety",
                        "candidate_rejected_nonfinite",
                        "candidate_rejected_safety",
                        "candidate_rejected_post_training_tk",
                    }
                    or (
                        getattr(args, "canary_only", False)
                        and existing.get("status") == "canary_clean"
                    )
                )
            )
            if (
                reusable
                and post_candidate_gate_name is not None
                and existing.get("status") in {
                    "complete",
                    "candidate_rejected_post_training_tk",
                }
            ):
                result_path = cell_dir / "result.json"
                reusable = result_path.is_file()
                if reusable:
                    gate_path = cell_dir / "epochwise_k64_viability.json"
                    gate_receipt = existing.get("post_candidate_gate")
                    result_record = json.loads(
                        result_path.read_text(encoding="utf-8")
                    )
                    completion_gate = result_record.get("completion", {}).get(
                        "post_candidate_gate"
                    )
                    gate_sha256 = (
                        _file_sha256(gate_path) if gate_path.is_file() else None
                    )
                    artifact_bound = any(
                        str(artifact.get("path", "")).endswith(
                            "/epochwise_k64_viability.json"
                        )
                        and artifact.get("sha256") == gate_sha256
                        for artifact in result_record.get("artifacts", ())
                    )
                    reusable = bool(
                        isinstance(gate_receipt, Mapping)
                        and gate_receipt.get("name") == post_candidate_gate_name
                        and gate_receipt.get("artifact_sha256") == gate_sha256
                        and completion_gate == gate_receipt
                        and artifact_bound
                    )
            if reusable:
                if args.bias_policy == "zero" and existing.get("status") in {
                    "complete",
                    "candidate_rejected_post_training_tk",
                    "canary_clean",
                }:
                    verification_root = (
                        cell_dir
                        if existing["status"]
                        in {"complete", "candidate_rejected_post_training_tk"}
                        else cell_dir / "canary"
                    )
                    zero_bias_verification = verify_zero_bias_run_checkpoints(
                        verification_root,
                        zero_bias_names,
                    )
                    existing["zero_bias_checkpoint_verification"] = (
                        zero_bias_verification
                    )
                    diagnostics_path = (
                        cell_dir / "safety_diagnostics.json"
                        if existing["status"]
                        in {"complete", "candidate_rejected_post_training_tk"}
                        else cell_dir / "canary" / "safety_diagnostics.json"
                    )
                    diagnostics = (
                        json.loads(diagnostics_path.read_text(encoding="utf-8"))
                        if diagnostics_path.is_file()
                        else {}
                    )
                    diagnostics["zero_bias_checkpoint_verification"] = (
                        zero_bias_verification
                    )
                    _write_json(diagnostics_path, diagnostics)
                    _write_json(cell_path, existing)
                rows.append(_summary_row(index, rho_conv, rho_dense, cell_dir))
                continue

        rates_by_name = derive_learning_rates(
            probe,
            rho_conv,
            rho_dense,
            args.bias_policy,
        )
        rate_vector = [rates_by_name[name] for name in probe["parameter_names"]]
        cell = {
            "index": index,
            "optimizer": optimizer_name,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "bias_policy": args.bias_policy,
            "learning_rates_by_parameter": rates_by_name,
            "learning_rate_vector": rate_vector,
            "signature": signature,
            "status": "running",
        }
        _write_json(cell_path, cell)
        candidate_config = prepare_training_config(
            base,
            optimizer_name,
            rate_vector,
            epochs=epochs,
            max_batches=args.max_batches,
            max_validation_batches=args.max_validation_batches,
            split_seed=args.split_seed,
            shuffle_seed=args.shuffle_seed,
            validation_batch_size=args.validation_batch_size,
        )
        config_path = cell_dir / "source_config.json"
        _write_json(config_path, candidate_config)
        from labs.mnist_train import NonFiniteTrainingError

        reporting_manifest = {
            "study_id": study_id,
            "run_id": name,
            "arm_id": f"{optimizer_name.lower()}-rho-{rho_conv:g}-{rho_dense:g}",
            "evidence_class": "ordinary_mnist_selection",
            "smoke": bool(getattr(args, "smoke", False)),
            "configuration": {
                "path": str(config_path),
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "resolved": candidate_config,
                "optimizer": optimizer_name,
                "learning_rates": rate_vector,
                "rho_conv": rho_conv,
                "rho_dense": rho_dense,
                "epochs": epochs,
                "canary_steps": canary_steps,
                "safety": safety,
            },
            "dataset": {
                "key": "mnist",
                "variant": "ordinary",
                "evaluation_split": "validation",
                "split_seed": args.split_seed,
                "official_test_read": False,
            },
            "command": getattr(args, "reporting_command", None)
            or {
                "module": "experiments.rho_search",
                "cell_index": index,
            },
            "git": resolved["code"],
            "runtime": {
                **runtime_context(target=target),
                "device": args.device,
            },
            "inputs": [
                {
                    "role": "source_config",
                    "path": str(base_path),
                    "sha256": resolved["source_config_sha256"],
                },
                {
                    "role": "optimizer_probe",
                    "path": str(probe_path),
                    "sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
                },
            ],
        }
        canary_path = cell_dir / "canary.json"
        canary = None
        if canary_path.exists() and not args.force:
            candidate_canary = json.loads(canary_path.read_text(encoding="utf-8"))
            if (
                candidate_canary.get("signature") == signature
                and candidate_canary.get("status")
                in {"clean", "rejected_nonfinite", "rejected_safety"}
            ):
                canary = candidate_canary
        if canary is None:
            canary_dir = cell_dir / "canary"
            canary_config = prepare_training_config(
                base,
                optimizer_name,
                rate_vector,
                epochs=1,
                max_batches=canary_steps,
                max_validation_batches=1,
                split_seed=args.split_seed,
                shuffle_seed=args.shuffle_seed,
                validation_batch_size=args.validation_batch_size,
            )
            canary_config_path = canary_dir / "source_config.json"
            _write_json(canary_config_path, canary_config)
            canary_monitor = TrainingSafetyMonitor(
                warmup_steps=safety["warmup_steps"],
                ema_decay=safety["ema_decay"],
                loss_factor=safety["loss_factor"],
                gradient_factor=safety["gradient_factor"],
                persistence=safety["persistence"],
                bound_occupancy_increase_maximum=safety[
                    "bound_occupancy_increase_maximum"
                ],
                projection_efficiency_minimum=safety[
                    "projection_efficiency_minimum"
                ],
                zero_proposal_epsilon=safety["zero_proposal_epsilon"],
                boundary_persistence=safety["boundary_persistence"],
            )
            canary = {
                "schema_version": "conv-rho-canary/v1",
                "signature": signature,
                "requested_steps": canary_steps,
                "status": "running",
            }
            canary_zero_bias_verification = None
            _write_json(canary_path, canary)
            try:
                _run_trainer(
                    canary_config_path,
                    canary_dir,
                    device=args.device,
                    optimizer_step_callback=canary_monitor,
                )
                if args.bias_policy == "zero":
                    canary_zero_bias_verification = (
                        verify_zero_bias_run_checkpoints(
                            canary_dir,
                            zero_bias_names,
                        )
                    )
                    canary["zero_bias_checkpoint_verification"] = (
                        canary_zero_bias_verification
                    )
            except SafetyRejection as error:
                canary["status"] = "rejected_safety"
                canary["failure"] = error.failure
            except NonFiniteTrainingError as error:
                canary["status"] = "rejected_nonfinite"
                canary["failure"] = {
                    "kind": "nonfinite_training_value",
                    "message": str(error),
                    "confirmed_step": canary_monitor.step + 1,
                }
            except BaseException:
                canary["status"] = "failed"
                _write_json(canary_path, canary)
                raise
            else:
                if canary_monitor.step != canary_steps:
                    raise RuntimeError(
                        f"Expected canary to complete {canary_steps} optimizer steps. "
                        f"Provided value: {canary_monitor.step}."
                    )
                canary["status"] = "clean"
            finally:
                diagnostics = canary_monitor.summary()
                if canary.get("failure") is not None:
                    diagnostics["safety_failure"] = canary["failure"]
                if canary_zero_bias_verification is not None:
                    diagnostics["zero_bias_checkpoint_verification"] = (
                        canary_zero_bias_verification
                    )
                _write_json(canary_dir / "safety_diagnostics.json", diagnostics)
            canary["completed_steps"] = canary_monitor.step
            _write_json(canary_path, canary)

        if canary["status"] == "clean" and args.bias_policy == "zero":
            canary_zero_bias_verification = verify_zero_bias_run_checkpoints(
                cell_dir / "canary",
                zero_bias_names,
            )
            canary["zero_bias_checkpoint_verification"] = (
                canary_zero_bias_verification
            )
            canary_diagnostics_path = (
                cell_dir / "canary" / "safety_diagnostics.json"
            )
            canary_diagnostics = (
                json.loads(
                    canary_diagnostics_path.read_text(encoding="utf-8")
                )
                if canary_diagnostics_path.is_file()
                else {}
            )
            canary_diagnostics["zero_bias_checkpoint_verification"] = (
                canary_zero_bias_verification
            )
            _write_json(canary_diagnostics_path, canary_diagnostics)
            _write_json(canary_path, canary)

        if canary["status"] != "clean":
            cell["status"] = f"canary_{canary['status']}"
            cell["canary"] = canary
            _write_json(cell_path, cell)
            if not getattr(args, "canary_only", False):
                start_run(cell_dir, reporting_manifest)
                fail_run(
                    cell_dir,
                    error=SafetyRejection(
                        canary.get(
                            "failure",
                            {"kind": canary["status"], "parameter": None},
                        )
                    ),
                )
            rows.append(_summary_row(index, rho_conv, rho_dense, cell_dir))
            continue

        if getattr(args, "canary_only", False):
            cell["status"] = "canary_clean"
            cell["canary"] = canary
            _write_json(cell_path, cell)
            rows.append(_summary_row(index, rho_conv, rho_dense, cell_dir))
            continue

        start_run(cell_dir, reporting_manifest)
        candidate_monitor = TrainingSafetyMonitor(
            warmup_steps=safety["warmup_steps"],
            ema_decay=safety["ema_decay"],
            loss_factor=safety["loss_factor"],
            gradient_factor=safety["gradient_factor"],
            persistence=safety["persistence"],
            bound_occupancy_increase_maximum=safety[
                "bound_occupancy_increase_maximum"
            ],
            projection_efficiency_minimum=safety[
                "projection_efficiency_minimum"
            ],
            zero_proposal_epsilon=safety["zero_proposal_epsilon"],
            boundary_persistence=safety["boundary_persistence"],
        )
        candidate_zero_bias_verification = None
        candidate_completion = None
        candidate_terminal_metrics = None
        try:
            _run_trainer(
                config_path,
                cell_dir,
                device=args.device,
                optimizer_step_callback=candidate_monitor,
                reporting_run_dir=cell_dir,
                checkpoint_every_epoch=checkpoint_every_epoch,
            )
            if args.bias_policy == "zero":
                candidate_zero_bias_verification = (
                    verify_zero_bias_run_checkpoints(
                        cell_dir,
                        zero_bias_names,
                    )
                )
                cell["zero_bias_checkpoint_verification"] = (
                    candidate_zero_bias_verification
                )
        except SafetyRejection as error:
            cell["status"] = "candidate_rejected_safety"
            cell["error"] = str(error)
            cell["safety_failure"] = error.failure
            _write_json(cell_path, cell)
            fail_run(cell_dir, error=error)
        except NonFiniteTrainingError as error:
            cell["status"] = "candidate_rejected_nonfinite"
            cell["error"] = str(error)
            cell["safety_failure"] = {
                "kind": "nonfinite_training_value",
                "message": str(error),
                "confirmed_step": candidate_monitor.step + 1,
            }
            _write_json(cell_path, cell)
            fail_run(cell_dir, error=error)
        except BaseException as error:
            cell["status"] = "failed"
            cell["error"] = str(error)
            _write_json(cell_path, cell)
            fail_run(cell_dir, error=error)
            raise
        else:
            if (
                expected_candidate_steps is not None
                and candidate_monitor.step != expected_candidate_steps
            ):
                error = RuntimeError(
                    "Candidate optimizer-step count does not match the resolved "
                    f"contract: expected={expected_candidate_steps}, "
                    f"observed={candidate_monitor.step}."
                )
                cell["status"] = "failed"
                cell["error"] = str(error)
                _write_json(cell_path, cell)
                fail_run(cell_dir, error=error)
                raise error
            cell["status"] = (
                "candidate_trained_pending_post_gate"
                if post_candidate_gate_name is not None
                else "complete"
            )
            metrics = json.loads(
                (cell_dir / "metrics.json").read_text(encoding="utf-8")
            )
            final_accuracy = float(metrics["final_test_accuracy"])
            accuracy_selection_eligible = (
                final_accuracy >= minimum_validation_accuracy
            )
            cell["selection_eligible"] = accuracy_selection_eligible
            if post_candidate_gate_name is not None:
                cell["training_accuracy_selection_eligible"] = (
                    accuracy_selection_eligible
                )
            cell["completed_steps"] = candidate_monitor.step
            _write_json(cell_path, cell)
            completion = {
                "criteria_met": cell["selection_eligible"],
                "rho_cell_complete": True,
                "safety_admissible": True,
                "minimum_validation_accuracy": minimum_validation_accuracy,
                "official_test_read": False,
            }
            if args.bias_policy == "zero":
                completion["zero_bias_checkpoint_verification"] = (
                    candidate_zero_bias_verification
                )
            candidate_completion = completion
            candidate_terminal_metrics = _terminal_metrics(metrics)
        finally:
            diagnostics = candidate_monitor.summary()
            if cell.get("safety_failure") is not None:
                diagnostics["safety_failure"] = cell["safety_failure"]
            if candidate_zero_bias_verification is not None:
                diagnostics["zero_bias_checkpoint_verification"] = (
                    candidate_zero_bias_verification
                )
            _write_json(cell_dir / "safety_diagnostics.json", diagnostics)
        if candidate_completion is not None:
            if post_candidate_gate_name is not None:
                if post_candidate_callback is None:
                    error = RuntimeError(
                        "A post-candidate scientific gate was declared but no "
                        "callback was supplied for this new candidate."
                    )
                    cell["status"] = "failed"
                    cell["error"] = str(error)
                    _write_json(cell_path, cell)
                    fail_run(cell_dir, error=error)
                    raise error
                try:
                    gate = post_candidate_callback(cell_dir, cell)
                except BaseException as error:
                    cell["status"] = "failed"
                    cell["selection_eligible"] = False
                    cell["error"] = str(error)
                    _write_json(cell_path, cell)
                    fail_run(cell_dir, error=error)
                    raise
                if not isinstance(gate, Mapping) or not isinstance(
                    gate.get("passed"), bool
                ):
                    error = RuntimeError(
                        "The post-candidate scientific gate must return a mapping "
                        "with a boolean 'passed' field."
                    )
                    cell["status"] = "failed"
                    cell["error"] = str(error)
                    _write_json(cell_path, cell)
                    fail_run(cell_dir, error=error)
                    raise error
                gate_artifact = Path(str(gate.get("summary_path", "")))
                gate_artifact_sha256 = gate.get("summary_sha256")
                if (
                    not gate_artifact.is_file()
                    or gate_artifact_sha256 != _file_sha256(gate_artifact)
                    or gate_artifact.parent.resolve() != cell_dir.resolve()
                ):
                    error = RuntimeError(
                        "The post-candidate scientific gate must provide a "
                        "hash-verified flat artifact at the candidate root."
                    )
                    cell["status"] = "failed"
                    cell["error"] = str(error)
                    _write_json(cell_path, cell)
                    fail_run(cell_dir, error=error)
                    raise error
                gate_passed = gate["passed"] is True
                gate_summary = {
                    "name": post_candidate_gate_name,
                    "status": gate.get("status"),
                    "passed": gate_passed,
                    "artifact": str(gate_artifact.resolve()),
                    "artifact_sha256": gate_artifact_sha256,
                }
                cell["post_candidate_gate"] = gate_summary
                cell["selection_eligible"] = bool(
                    cell["selection_eligible"] and gate_passed
                )
                if not gate_passed:
                    cell["status"] = "candidate_rejected_post_training_tk"
                else:
                    cell["status"] = "complete"
                _write_json(cell_path, cell)
                candidate_completion["post_candidate_gate"] = gate_summary
                candidate_completion["post_training_tk_admissible"] = gate_passed
                candidate_completion["criteria_met"] = cell["selection_eligible"]
            complete_run(
                cell_dir,
                terminal_metrics=candidate_terminal_metrics,
                completion=candidate_completion,
            )
        rows.append(_summary_row(index, rho_conv, rho_dense, cell_dir))

    if args.index is None:
        _write_summary(output_root, rows)
        summary_path = output_root / "summary.json"
    else:
        summary_path = output_root / "shards" / f"{args.index:03d}.json"
        _write_json(summary_path, rows[0])
    return {
        "status": "complete",
        "probe": str(probe_path),
        "summary": str(summary_path),
        "cells": len(rows),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Normal Conv source_config.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--rho-conv", type=float, nargs="+", required=True)
    parser.add_argument("--rho-dense", type=float, nargs="+", required=True)
    parser.add_argument("--optimizer", choices=("SGD", "Adam"))
    parser.add_argument(
        "--bias-policy",
        choices=("q90_cap", "tied", "zero"),
        default="q90_cap",
    )
    parser.add_argument("--probe-batches", type=int, nargs="+", default=list(DEFAULT_PROBE_COUNTS))
    parser.add_argument("--stability-tolerance", type=float, default=0.10)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--checkpoint-every-epoch",
        action="store_true",
        help="Save candidate model/optimizer checkpoints at epoch 0 and every epoch.",
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--validation-batch-size", type=int, default=128)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--study-id")
    parser.add_argument("--target")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--canary-only",
        action="store_true",
        help="Run or resume the selected cell's restarted canary without promotion.",
    )
    parser.add_argument("--canary-steps", type=int, default=DEFAULT_CANARY_STEPS)
    parser.add_argument("--expected-candidate-steps", type=int)
    parser.add_argument("--minimum-validation-accuracy", type=float, default=0.90)
    parser.add_argument("--safety-warmup-steps", type=int, default=32)
    parser.add_argument("--safety-ema-decay", type=float, default=0.98)
    parser.add_argument("--safety-loss-factor", type=float, default=4.0)
    parser.add_argument("--safety-gradient-factor", type=float, default=100.0)
    parser.add_argument("--safety-persistence", type=int, default=8)
    parser.add_argument("--safety-bound-occupancy-increase-maximum", type=float)
    parser.add_argument("--safety-projection-efficiency-minimum", type=float)
    parser.add_argument("--safety-zero-proposal-epsilon", type=float, default=1e-30)
    parser.add_argument("--safety-boundary-persistence", type=int, default=16)
    parser.add_argument("--index", type=int, help="Run one Cartesian-grid cell by index")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Validate all indexed cells and write the combined summary",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.reporting_command = [
        sys.executable,
        "-m",
        "experiments.rho_search",
        *(list(argv) if argv is not None else sys.argv[1:]),
    ]
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
