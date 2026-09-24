"""Persistent additive Gaussian programming noise for conductance weights.

The model represents one programming event. For every eligible conductance
weight, it draws one standard-normal tensor on CPU in float64 and writes

    G_written = clip(G_target + eta * (G_max - G_min) * Z, G_min, G_max).

The draw for a parameter is keyed independently of parameter traversal order,
runtime device, tensor dtype, target values, bounds, and eta. Consequently,
matched models with the same parameter geometry receive common random numbers.
The written tensor remains fixed until this function is called again.
"""

from __future__ import annotations

import hashlib
import json
import math
import operator
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from model.variable.parameter import Bias, ConvWeight, DenseWeight


MODEL_NAME = "persistent_additive_gaussian_full_scale"
MODEL_VERSION = f"{MODEL_NAME}/v1"
RECEIPT_SCHEMA_VERSION = "weight-write-noise-receipt/v1"

_CANONICAL_WEIGHT_NAME = re.compile(r"^(ConvWeight|DenseWeight)_[0-9]+$")
_MAX_PROGRAMMING_SEED = (1 << 63) - 1


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _normalize_eta(eta: Any) -> float:
    if isinstance(eta, (bool, np.bool_)):
        raise ValueError(
            "Expected weight-write-noise eta to be a finite number >= 0; "
            f"provided value: {eta!r}."
        )
    try:
        normalized = float(eta)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "Expected weight-write-noise eta to be a finite number >= 0; "
            f"provided value: {eta!r}."
        ) from None
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(
            "Expected weight-write-noise eta to be a finite number >= 0; "
            f"provided value: {eta!r}."
        )
    return normalized


def _normalize_programming_seed(programming_seed: Any) -> int:
    if isinstance(programming_seed, (bool, np.bool_)):
        raise ValueError(
            "Expected programming_seed to be an integer in "
            f"[0, {_MAX_PROGRAMMING_SEED}]; provided value: "
            f"{programming_seed!r}."
        )
    try:
        normalized = operator.index(programming_seed)
    except TypeError:
        raise ValueError(
            "Expected programming_seed to be an integer in "
            f"[0, {_MAX_PROGRAMMING_SEED}]; provided value: "
            f"{programming_seed!r}."
        ) from None
    if normalized < 0 or normalized > _MAX_PROGRAMMING_SEED:
        raise ValueError(
            "Expected programming_seed to be an integer in "
            f"[0, {_MAX_PROGRAMMING_SEED}]; provided value: "
            f"{programming_seed!r}."
        )
    return int(normalized)


def _canonical_weight_name(parameter: Any) -> str | None:
    """Return a canonical eligible name, or None for a non-weight."""

    raw_name = getattr(parameter, "name", None)
    stripped_name = raw_name.strip() if isinstance(raw_name, str) else None
    name_match = (
        _CANONICAL_WEIGHT_NAME.fullmatch(stripped_name)
        if stripped_name is not None
        else None
    )

    # Biases are never conductance weights, even if a caller has renamed one.
    if type(parameter) is Bias or type(parameter).__name__ == "Bias":
        return None

    exact_weight_type = type(parameter) in (ConvWeight, DenseWeight)
    if not exact_weight_type and name_match is None:
        return None
    if name_match is None:
        raise ValueError(
            "Eligible ConvWeight/DenseWeight parameters require a canonical "
            "name matching 'ConvWeight_<index>' or 'DenseWeight_<index>'; "
            f"provided name: {raw_name!r}."
        )

    if exact_weight_type:
        expected_prefix = f"{type(parameter).__name__}_"
        if not stripped_name.startswith(expected_prefix):
            raise ValueError(
                "Conductance-weight class and canonical parameter name disagree; "
                f"class={type(parameter).__name__!r}, name={raw_name!r}."
            )
    return stripped_name


def _explicit_bounds(parameter: Any, canonical_name: str) -> tuple[float, float]:
    provided = {
        "min_cond": getattr(parameter, "min_cond", None),
        "max_cond": getattr(parameter, "max_cond", None),
    }
    try:
        lower = float(provided["min_cond"])
        upper = float(provided["max_cond"])
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"Weight {canonical_name!r} requires explicit finite bounds with "
            f"min_cond < max_cond; provided bounds: {provided!r}."
        ) from None
    span = upper - lower
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or not math.isfinite(span)
        or lower >= upper
    ):
        raise ValueError(
            f"Weight {canonical_name!r} requires explicit finite bounds with "
            f"min_cond < max_cond; provided bounds: {provided!r}."
        )
    return lower, upper


def _parameter_draw(
    *,
    programming_seed: int,
    canonical_name: str,
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, str, int]:
    key = _canonical_json_bytes(
        {
            "model_version": MODEL_VERSION,
            "programming_seed": programming_seed,
            "canonical_parameter_name": canonical_name,
            "shape": list(shape),
        }
    )
    key_digest = hashlib.sha256(key)
    derived_seed = (
        int.from_bytes(key_digest.digest()[:8], byteorder="big")
        & _MAX_PROGRAMMING_SEED
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed)
    draw = torch.randn(
        shape,
        dtype=torch.float64,
        device="cpu",
        generator=generator,
    )
    return draw, key_digest.hexdigest(), derived_seed


def _tensor_sha256(tensor: torch.Tensor, *, logical_dtype: str) -> str:
    canonical = tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
    header = _canonical_json_bytes(
        {
            "encoding": "little-endian-float64-c-order",
            "logical_dtype": logical_dtype,
            "shape": list(canonical.shape),
        }
    )
    array = canonical.numpy().astype("<f8", copy=False)
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _aggregate_sha256(receipts: list[dict[str, Any]], field: str) -> str:
    entries = [
        {
            "canonical_name": receipt["canonical_name"],
            "shape": receipt["shape"],
            # Keep this digest comparable across target/written content. The
            # caller's field selects the per-parameter digest; its label is
            # not itself part of the hashed tensor identity.
            "sha256": receipt[field],
        }
        for receipt in receipts
    ]
    return hashlib.sha256(_canonical_json_bytes(entries)).hexdigest()


def _fraction(count: int, total: int) -> float:
    return float(count / total)


def apply_weight_write_noise(
    parameters: Iterable[Any],
    base_states: Iterable[torch.Tensor],
    *,
    eta: float,
    programming_seed: int,
) -> dict[str, Any]:
    """Program eligible conductance weights from clean base states.

    parameters and base_states are parallel iterables. Only exact ConvWeight
    and DenseWeight objects or duck-typed parameters with canonical
    ConvWeight_<index> and DenseWeight_<index> names are eligible. Biases and
    all other parameters are ignored and left unchanged.

    All validation and numerical work finishes before runtime states are
    written, so ordinary validation failures cannot leave a partially written
    model. At eta == 0, eligible runtime states receive exact copies of their
    base tensors rather than float64 round trips.

    Receipt error fields without a suffix use conductance units. Fields ending
    in _fraction divide each error by that parameter's conductance span before
    pooling. Aggregate means are element-weighted; aggregate requested_sigma
    and RMS fields are pooled root-mean-square quantities.
    """

    normalized_eta = _normalize_eta(eta)
    normalized_seed = _normalize_programming_seed(programming_seed)
    parameter_list = list(parameters)
    base_state_list = list(base_states)
    if len(parameter_list) != len(base_state_list):
        raise ValueError(
            "parameters and base_states must have the same length; "
            f"received {len(parameter_list)} and {len(base_state_list)}."
        )

    planned_writes: list[tuple[torch.Tensor, torch.Tensor]] = []
    parameter_receipts: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    aggregate_counts = {
        "element_count": 0,
        "pre_existing_lower_rail_count": 0,
        "pre_existing_upper_rail_count": 0,
        "pre_existing_rail_count": 0,
        "pre_existing_interior_count": 0,
        "final_lower_rail_count": 0,
        "final_upper_rail_count": 0,
        "final_rail_count": 0,
        "lower_clipping_count": 0,
        "upper_clipping_count": 0,
        "clipping_count": 0,
        "new_lower_clipping_count": 0,
        "new_upper_clipping_count": 0,
        "new_clipping_count": 0,
    }
    aggregate_error_sum = 0.0
    aggregate_absolute_error_sum = 0.0
    aggregate_error_square_sum = 0.0
    aggregate_error_fraction_sum = 0.0
    aggregate_absolute_error_fraction_sum = 0.0
    aggregate_error_fraction_square_sum = 0.0
    aggregate_requested_variance_sum = 0.0

    for parameter, base_state in zip(parameter_list, base_state_list):
        canonical_name = _canonical_weight_name(parameter)
        if canonical_name is None:
            continue
        if canonical_name in seen_names:
            raise ValueError(
                "Canonical conductance-weight names must be unique; duplicate "
                f"name: {canonical_name!r}."
            )
        seen_names.add(canonical_name)

        runtime_state = getattr(parameter, "state", None)
        if not isinstance(runtime_state, torch.Tensor):
            raise TypeError(
                f"Weight {canonical_name!r} must expose a Tensor state; "
                f"provided type: {type(runtime_state).__name__}."
            )
        if not isinstance(base_state, torch.Tensor):
            raise TypeError(
                f"Base state for weight {canonical_name!r} must be a Tensor; "
                f"provided type: {type(base_state).__name__}."
            )
        if tuple(runtime_state.shape) != tuple(base_state.shape):
            raise ValueError(
                f"Runtime/base shape mismatch for weight {canonical_name!r}: "
                f"runtime={tuple(runtime_state.shape)!r}, "
                f"base={tuple(base_state.shape)!r}."
            )
        if runtime_state.dtype != base_state.dtype:
            raise ValueError(
                f"Runtime/base dtype mismatch for weight {canonical_name!r}: "
                f"runtime={runtime_state.dtype}, base={base_state.dtype}."
            )
        if not base_state.is_floating_point():
            raise TypeError(
                f"Weight {canonical_name!r} requires a floating-point base "
                f"state; provided dtype: {base_state.dtype}."
            )
        if base_state.numel() == 0:
            raise ValueError(f"Weight {canonical_name!r} has an empty base state.")

        lower, upper = _explicit_bounds(parameter, canonical_name)
        span = upper - lower
        requested_sigma = normalized_eta * span
        if not math.isfinite(requested_sigma):
            raise ValueError(
                f"Requested sigma is not finite for weight {canonical_name!r}: "
                f"eta={normalized_eta!r}, span={span!r}."
            )

        base_cpu = base_state.detach().to(device="cpu").contiguous()
        if not bool(torch.isfinite(base_cpu).all()):
            raise ValueError(f"Base state for weight {canonical_name!r} must be finite.")
        effective_lower = torch.tensor(lower, dtype=base_cpu.dtype, device="cpu")
        effective_upper = torch.tensor(upper, dtype=base_cpu.dtype, device="cpu")
        if not bool(torch.isfinite(effective_lower)) or not bool(
            torch.isfinite(effective_upper)
        ):
            raise ValueError(
                f"Bounds for weight {canonical_name!r} are not representable in "
                f"dtype {base_cpu.dtype}: min_cond={lower!r}, max_cond={upper!r}."
            )
        if not bool(effective_lower < effective_upper):
            raise ValueError(
                f"Bounds for weight {canonical_name!r} collapse in dtype "
                f"{base_cpu.dtype}: min_cond={lower!r}, max_cond={upper!r}."
            )
        if not bool(
            ((base_cpu >= effective_lower) & (base_cpu <= effective_upper)).all()
        ):
            raise ValueError(
                f"Base state for weight {canonical_name!r} lies outside its "
                f"explicit interval [{lower!r}, {upper!r}]."
            )

        shape = tuple(int(dim) for dim in base_cpu.shape)
        draw, draw_key_sha256, derived_seed = _parameter_draw(
            programming_seed=normalized_seed,
            canonical_name=canonical_name,
            shape=shape,
        )
        pre_lower = base_cpu == effective_lower
        pre_upper = base_cpu == effective_upper
        pre_any = pre_lower | pre_upper
        pre_interior = ~pre_any

        if normalized_eta == 0.0:
            written_cpu = base_cpu.clone()
            clipped_lower = torch.zeros(shape, dtype=torch.bool, device="cpu")
            clipped_upper = torch.zeros(shape, dtype=torch.bool, device="cpu")
        else:
            target_float64 = base_cpu.to(dtype=torch.float64)
            # Stored rails can differ by one representation rounding from their
            # Python floats. Use the declared physical rail in the formula.
            target_float64 = torch.where(
                pre_lower,
                torch.as_tensor(lower, dtype=torch.float64),
                target_float64,
            )
            target_float64 = torch.where(
                pre_upper,
                torch.as_tensor(upper, dtype=torch.float64),
                target_float64,
            )
            proposed = target_float64 + requested_sigma * draw
            if not bool(torch.isfinite(proposed).all()):
                raise ValueError(
                    f"Programming proposal for weight {canonical_name!r} is "
                    "non-finite."
                )
            clipped_lower = proposed < lower
            clipped_upper = proposed > upper
            written_cpu = proposed.clamp(min=lower, max=upper).to(
                dtype=base_cpu.dtype
            )

        clipped_any = clipped_lower | clipped_upper
        # A target that starts on either rail is not an interior target, even
        # if an unusually large proposal crosses the complete interval and is
        # clipped at the opposite rail.
        new_lower_clipping = clipped_lower & pre_interior
        new_upper_clipping = clipped_upper & pre_interior
        new_clipping = new_lower_clipping | new_upper_clipping
        final_lower = written_cpu == effective_lower
        final_upper = written_cpu == effective_upper
        final_any = final_lower | final_upper
        error = written_cpu.to(dtype=torch.float64) - base_cpu.to(dtype=torch.float64)

        element_count = int(base_cpu.numel())
        counts = {
            "element_count": element_count,
            "pre_existing_lower_rail_count": int(pre_lower.sum().item()),
            "pre_existing_upper_rail_count": int(pre_upper.sum().item()),
            "pre_existing_rail_count": int(pre_any.sum().item()),
            "pre_existing_interior_count": int(pre_interior.sum().item()),
            "final_lower_rail_count": int(final_lower.sum().item()),
            "final_upper_rail_count": int(final_upper.sum().item()),
            "final_rail_count": int(final_any.sum().item()),
            "lower_clipping_count": int(clipped_lower.sum().item()),
            "upper_clipping_count": int(clipped_upper.sum().item()),
            "clipping_count": int(clipped_any.sum().item()),
            "new_lower_clipping_count": int(new_lower_clipping.sum().item()),
            "new_upper_clipping_count": int(new_upper_clipping.sum().item()),
            "new_clipping_count": int(new_clipping.sum().item()),
        }
        error_sum = float(error.sum().item())
        absolute_error_sum = float(error.abs().sum().item())
        error_square_sum = float(torch.sum(error * error).item())
        error_fraction_sum = error_sum / span
        absolute_error_fraction_sum = absolute_error_sum / span
        error_fraction_square_sum = error_square_sum / (span * span)

        receipt = {
            "canonical_name": canonical_name,
            "class_name": type(parameter).__name__,
            "shape": list(shape),
            "dtype": str(base_cpu.dtype),
            "runtime_device": str(runtime_state.device),
            "lower_bound": lower,
            "upper_bound": upper,
            "conductance_span": span,
            "requested_sigma": requested_sigma,
            **counts,
            "pre_existing_lower_rail_fraction": _fraction(
                counts["pre_existing_lower_rail_count"], element_count
            ),
            "pre_existing_upper_rail_fraction": _fraction(
                counts["pre_existing_upper_rail_count"], element_count
            ),
            "pre_existing_rail_fraction": _fraction(
                counts["pre_existing_rail_count"], element_count
            ),
            "pre_existing_interior_fraction": _fraction(
                counts["pre_existing_interior_count"], element_count
            ),
            "final_lower_rail_fraction": _fraction(
                counts["final_lower_rail_count"], element_count
            ),
            "final_upper_rail_fraction": _fraction(
                counts["final_upper_rail_count"], element_count
            ),
            "final_rail_fraction": _fraction(
                counts["final_rail_count"], element_count
            ),
            "lower_clipping_fraction": _fraction(
                counts["lower_clipping_count"], element_count
            ),
            "upper_clipping_fraction": _fraction(
                counts["upper_clipping_count"], element_count
            ),
            "clipping_fraction": _fraction(
                counts["clipping_count"], element_count
            ),
            "new_lower_clipping_fraction": _fraction(
                counts["new_lower_clipping_count"], element_count
            ),
            "new_upper_clipping_fraction": _fraction(
                counts["new_upper_clipping_count"], element_count
            ),
            "new_clipping_fraction": _fraction(
                counts["new_clipping_count"], element_count
            ),
            "new_lower_clipping_fraction_of_interior": (
                _fraction(
                    counts["new_lower_clipping_count"],
                    counts["pre_existing_interior_count"],
                )
                if counts["pre_existing_interior_count"]
                else 0.0
            ),
            "new_upper_clipping_fraction_of_interior": (
                _fraction(
                    counts["new_upper_clipping_count"],
                    counts["pre_existing_interior_count"],
                )
                if counts["pre_existing_interior_count"]
                else 0.0
            ),
            "new_clipping_fraction_of_interior": (
                _fraction(
                    counts["new_clipping_count"],
                    counts["pre_existing_interior_count"],
                )
                if counts["pre_existing_interior_count"]
                else 0.0
            ),
            "signed_mean_error": error_sum / element_count,
            "mean_absolute_error": absolute_error_sum / element_count,
            "rms_error": math.sqrt(error_square_sum / element_count),
            "signed_mean_error_fraction": error_fraction_sum / element_count,
            "mean_absolute_error_fraction": (
                absolute_error_fraction_sum / element_count
            ),
            "rms_error_fraction": math.sqrt(
                error_fraction_square_sum / element_count
            ),
            "draw_key_sha256": draw_key_sha256,
            "derived_draw_seed": derived_seed,
            "target_sha256": _tensor_sha256(
                base_cpu,
                logical_dtype=str(base_cpu.dtype),
            ),
            "draw_sha256": _tensor_sha256(
                draw,
                logical_dtype="torch.float64",
            ),
            "written_sha256": _tensor_sha256(
                written_cpu,
                logical_dtype=str(base_cpu.dtype),
            ),
        }
        parameter_receipts.append(receipt)
        planned_writes.append((runtime_state, written_cpu))

        for key in aggregate_counts:
            aggregate_counts[key] += counts[key]
        aggregate_error_sum += error_sum
        aggregate_absolute_error_sum += absolute_error_sum
        aggregate_error_square_sum += error_square_sum
        aggregate_error_fraction_sum += error_fraction_sum
        aggregate_absolute_error_fraction_sum += absolute_error_fraction_sum
        aggregate_error_fraction_square_sum += error_fraction_square_sum
        aggregate_requested_variance_sum += (
            element_count * requested_sigma * requested_sigma
        )

    if not parameter_receipts:
        raise ValueError(
            "No eligible conductance weights were found. Expected exact "
            "ConvWeight/DenseWeight parameters or canonical names matching "
            "'ConvWeight_<index>'/'DenseWeight_<index>'."
        )

    # Sort receipt content independently of traversal order. Writes can remain
    # in input order because each target has already received its keyed draw.
    parameter_receipts.sort(key=lambda receipt: receipt["canonical_name"])
    total = aggregate_counts["element_count"]
    requested_sigmas = [
        receipt["requested_sigma"] for receipt in parameter_receipts
    ]
    aggregate_receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "persistence": "program_once",
        "boundary_handling": "clip",
        "rng_device": "cpu",
        "rng_dtype": "torch.float64",
        "torch_version": torch.__version__,
        "programming_seed": normalized_seed,
        "eta": normalized_eta,
        "input_parameter_count": len(parameter_list),
        "selected_parameter_count": len(parameter_receipts),
        **aggregate_counts,
        # If parameters have different spans, this is their element-weighted
        # pooled requested RMS, rather than an arbitrary layer average.
        "requested_sigma": math.sqrt(
            aggregate_requested_variance_sum / total
        ),
        "requested_sigma_min": min(requested_sigmas),
        "requested_sigma_max": max(requested_sigmas),
        "pre_existing_lower_rail_fraction": _fraction(
            aggregate_counts["pre_existing_lower_rail_count"], total
        ),
        "pre_existing_upper_rail_fraction": _fraction(
            aggregate_counts["pre_existing_upper_rail_count"], total
        ),
        "pre_existing_rail_fraction": _fraction(
            aggregate_counts["pre_existing_rail_count"], total
        ),
        "pre_existing_interior_fraction": _fraction(
            aggregate_counts["pre_existing_interior_count"], total
        ),
        "final_lower_rail_fraction": _fraction(
            aggregate_counts["final_lower_rail_count"], total
        ),
        "final_upper_rail_fraction": _fraction(
            aggregate_counts["final_upper_rail_count"], total
        ),
        "final_rail_fraction": _fraction(
            aggregate_counts["final_rail_count"], total
        ),
        "lower_clipping_fraction": _fraction(
            aggregate_counts["lower_clipping_count"], total
        ),
        "upper_clipping_fraction": _fraction(
            aggregate_counts["upper_clipping_count"], total
        ),
        "clipping_fraction": _fraction(
            aggregate_counts["clipping_count"], total
        ),
        "new_lower_clipping_fraction": _fraction(
            aggregate_counts["new_lower_clipping_count"], total
        ),
        "new_upper_clipping_fraction": _fraction(
            aggregate_counts["new_upper_clipping_count"], total
        ),
        "new_clipping_fraction": _fraction(
            aggregate_counts["new_clipping_count"], total
        ),
        "new_lower_clipping_fraction_of_interior": (
            _fraction(
                aggregate_counts["new_lower_clipping_count"],
                aggregate_counts["pre_existing_interior_count"],
            )
            if aggregate_counts["pre_existing_interior_count"]
            else 0.0
        ),
        "new_upper_clipping_fraction_of_interior": (
            _fraction(
                aggregate_counts["new_upper_clipping_count"],
                aggregate_counts["pre_existing_interior_count"],
            )
            if aggregate_counts["pre_existing_interior_count"]
            else 0.0
        ),
        "new_clipping_fraction_of_interior": (
            _fraction(
                aggregate_counts["new_clipping_count"],
                aggregate_counts["pre_existing_interior_count"],
            )
            if aggregate_counts["pre_existing_interior_count"]
            else 0.0
        ),
        "signed_mean_error": aggregate_error_sum / total,
        "mean_absolute_error": aggregate_absolute_error_sum / total,
        "rms_error": math.sqrt(aggregate_error_square_sum / total),
        "signed_mean_error_fraction": aggregate_error_fraction_sum / total,
        "mean_absolute_error_fraction": (
            aggregate_absolute_error_fraction_sum / total
        ),
        "rms_error_fraction": math.sqrt(
            aggregate_error_fraction_square_sum / total
        ),
        "target_sha256": _aggregate_sha256(
            parameter_receipts,
            "target_sha256",
        ),
        "draw_sha256": _aggregate_sha256(
            parameter_receipts,
            "draw_sha256",
        ),
        "written_sha256": _aggregate_sha256(
            parameter_receipts,
            "written_sha256",
        ),
        "parameters": parameter_receipts,
    }

    with torch.no_grad():
        for runtime_state, written_cpu in planned_writes:
            runtime_state.copy_(
                written_cpu.to(
                    device=runtime_state.device,
                    dtype=runtime_state.dtype,
                )
            )

    return aggregate_receipt


__all__ = [
    "MODEL_NAME",
    "MODEL_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "apply_weight_write_noise",
]
