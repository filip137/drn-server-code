"""Sample raw gradients and realized optimizer updates during Conv training."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


GRADIENT_TRACE_SCHEMA = "mnist-conv-gradient-update-trace/v1"
GRADIENT_TRACE_METADATA_SCHEMA = "mnist-conv-gradient-update-trace-metadata/v1"


def sampled_batch_positions(total_batches: int, sample_count: int) -> tuple[int, ...]:
    """Return evenly spaced one-based batch positions, including both endpoints."""

    total = int(total_batches)
    count = int(sample_count)
    if total <= 0:
        raise ValueError(
            f"Expected total_batches to be positive. Provided value: {total_batches!r}."
        )
    if count <= 0:
        raise ValueError(
            f"Expected sample_count to be positive. Provided value: {sample_count!r}."
        )
    if count >= total:
        return tuple(range(1, total + 1))
    if count == 1:
        return (1,)
    values = {
        1 + round(index * (total - 1) / (count - 1))
        for index in range(count)
    }
    return tuple(sorted(values))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_index_sequence_hash(indices: Sequence[int]) -> str:
    from labs.datasets import stable_index_sequence_hash

    return stable_index_sequence_hash(tuple(int(value) for value in indices))


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected finite {label}. Provided value: {value!r}.")
    return result


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    value = numerator / denominator
    return value if math.isfinite(value) else None


def _tensor_metrics(value: Any) -> dict[str, float | int]:
    import torch

    vector = value.detach().reshape(-1).to(dtype=torch.float64)
    if not bool(torch.isfinite(vector).all().item()):
        count = int((~torch.isfinite(vector)).sum().item())
        raise ValueError(f"Expected a finite tensor. Found {count} non-finite values.")
    element_count = int(vector.numel())
    l2 = float(torch.linalg.vector_norm(vector).item())
    rms = float(torch.sqrt(torch.mean(vector * vector)).item())
    return {
        "element_count": element_count,
        "l2": l2,
        "rms": rms,
        "abs_mean": float(vector.abs().mean().item()),
        "abs_max": float(vector.abs().max().item()),
        "abs_q50": float(torch.quantile(vector.abs(), 0.50).item()),
        "abs_q90": float(torch.quantile(vector.abs(), 0.90).item()),
        "abs_q99": float(torch.quantile(vector.abs(), 0.99).item()),
        "zero_fraction": float(
            (vector.abs() <= 1.0e-12).to(torch.float64).mean().item()
        ),
    }


def _tensor_sha256(value: Any) -> str:
    vector = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(vector.dtype).encode("utf-8"))
    digest.update(str(tuple(vector.shape)).encode("utf-8"))
    digest.update(vector.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _descent_cosine(gradient: Any, update: Any) -> float | None:
    import torch

    grad = gradient.detach().reshape(-1).to(dtype=torch.float64)
    delta = update.detach().reshape(-1).to(dtype=torch.float64)
    denominator = float(
        torch.linalg.vector_norm(grad).item()
        * torch.linalg.vector_norm(delta).item()
    )
    if denominator <= 0.0:
        return None
    value = float(torch.dot(-grad, delta).item() / denominator)
    return _finite(value, "descent cosine")


def _vector_cosine(left: Any, right: Any) -> float | None:
    import torch

    left_vector = left.detach().reshape(-1).to(dtype=torch.float64)
    right_vector = right.detach().reshape(-1).to(dtype=torch.float64)
    denominator = float(
        torch.linalg.vector_norm(left_vector).item()
        * torch.linalg.vector_norm(right_vector).item()
    )
    if denominator <= 0.0:
        return None
    return _finite(
        float(torch.dot(left_vector, right_vector).item() / denominator),
        "vector cosine",
    )


def _sign_disagreement_fraction(left: Any, right: Any) -> float:
    import torch

    left_vector = left.detach().reshape(-1)
    right_vector = right.detach().reshape(-1)
    return float(
        (torch.sign(left_vector) != torch.sign(right_vector))
        .to(torch.float64)
        .mean()
        .item()
    )


def _effective_bounds(parameter: Any) -> tuple[float | None, float | None]:
    lower = getattr(parameter, "min_cond", None)
    upper = getattr(parameter, "max_cond", None)
    if lower is None and bool(getattr(parameter, "_non_negative", False)):
        lower = 0.0
    lower_value = float(lower) if lower is not None else None
    upper_value = float(upper) if upper is not None else None
    if lower_value is not None and not math.isfinite(lower_value):
        lower_value = None
    if upper_value is not None and not math.isfinite(upper_value):
        upper_value = None
    return lower_value, upper_value


def _bound_fraction(value: Any, bound: float | None) -> float | None:
    import torch

    if bound is None:
        return None
    vector = value.detach().reshape(-1)
    return float(
        torch.isclose(
            vector,
            torch.as_tensor(bound, dtype=vector.dtype, device=vector.device),
            rtol=0.0,
            atol=1.0e-12,
        )
        .to(torch.float64)
        .mean()
        .item()
    )


def _optimizer_state_metrics(
    state: Mapping[str, Any],
    *,
    gradient: Any,
    optimizer_group: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {
        "optimizer_step": None,
        "adam_exp_avg_rms": None,
        "adam_exp_avg_sq_mean": None,
        "adam_rms_denominator": None,
        "adam_max_exp_avg_sq_mean": None,
        "adam_bias_corrected_exp_avg_rms": None,
        "adam_bias_corrected_exp_avg_sq_mean": None,
        "gradient_momentum_cosine": None,
        "gradient_momentum_sign_disagreement_fraction": None,
    }
    step = state.get("step")
    if step is not None:
        step_value = float(step.detach().cpu().item()) if torch.is_tensor(step) else float(step)
        result["optimizer_step"] = _finite(step_value, "optimizer step")
    exp_avg = state.get("exp_avg")
    if torch.is_tensor(exp_avg):
        result["adam_exp_avg_rms"] = _tensor_metrics(exp_avg)["rms"]
        result["gradient_momentum_cosine"] = _vector_cosine(gradient, exp_avg)
        result["gradient_momentum_sign_disagreement_fraction"] = (
            _sign_disagreement_fraction(gradient, exp_avg)
        )
    exp_avg_sq = state.get("exp_avg_sq")
    if torch.is_tensor(exp_avg_sq):
        vector = exp_avg_sq.detach().reshape(-1).to(dtype=torch.float64)
        mean = float(vector.mean().item())
        result["adam_exp_avg_sq_mean"] = _finite(mean, "Adam exp_avg_sq mean")
        result["adam_rms_denominator"] = _finite(
            math.sqrt(max(mean, 0.0)), "Adam RMS denominator"
        )
    if (
        result["optimizer_step"] is not None
        and torch.is_tensor(exp_avg)
        and torch.is_tensor(exp_avg_sq)
    ):
        beta1, beta2 = tuple(float(value) for value in optimizer_group["betas"])
        step_number = int(result["optimizer_step"])
        if step_number <= 0:
            raise ValueError(f"Expected a positive Adam step. Found {step_number}.")
        corrected_exp_avg = exp_avg / (1.0 - beta1**step_number)
        corrected_exp_avg_sq = exp_avg_sq / (1.0 - beta2**step_number)
        result["adam_bias_corrected_exp_avg_rms"] = _tensor_metrics(
            corrected_exp_avg
        )["rms"]
        result["adam_bias_corrected_exp_avg_sq_mean"] = _finite(
            float(
                corrected_exp_avg_sq.detach()
                .reshape(-1)
                .to(dtype=torch.float64)
                .mean()
                .item()
            ),
            "Adam bias-corrected exp_avg_sq mean",
        )
    max_exp_avg_sq = state.get("max_exp_avg_sq")
    if torch.is_tensor(max_exp_avg_sq):
        result["adam_max_exp_avg_sq_mean"] = _finite(
            float(max_exp_avg_sq.detach().to(dtype=torch.float64).mean().item()),
            "Adam max_exp_avg_sq mean",
        )
    return result


class GradientTraceRecorder:
    """Record selected post-step transitions without changing training state."""

    track_all_parameters = True

    def __init__(
        self,
        *,
        output_path: str | Path,
        metadata_path: str | Path,
        parameters: Sequence[Any],
        optimizer: Any,
        batch_indices_by_epoch: Sequence[Sequence[Sequence[int]]],
        effective_batches_per_epoch: int,
        samples_per_epoch: int,
        dataset_provenance: Mapping[str, Any],
        resume: bool = False,
    ) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists() and not resume:
            raise FileExistsError(
                f"Expected a new gradient trace path. Provided value: {self.output_path}."
            )
        self.parameters = tuple(parameters)
        self.optimizer = optimizer
        self.names = tuple(
            str(getattr(parameter, "name", "")).strip()
            for parameter in self.parameters
        )
        if not self.names or any(not name for name in self.names):
            raise ValueError(f"Expected every traced parameter to have a name: {self.names!r}.")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"Expected unique traced parameter names: {self.names!r}.")
        self.initial_parameter_metrics = {
            name: {
                **_tensor_metrics(parameter.state),
                "sha256": _tensor_sha256(parameter.state),
            }
            for name, parameter in zip(self.names, self.parameters)
        }

        self.batch_indices_by_epoch = tuple(
            tuple(tuple(int(index) for index in batch) for batch in epoch)
            for epoch in batch_indices_by_epoch
        )
        if not self.batch_indices_by_epoch:
            raise ValueError("Expected at least one epoch of batch source indices.")
        self.effective_batches_per_epoch = int(effective_batches_per_epoch)
        self.samples_per_epoch = int(samples_per_epoch)
        self.positions = sampled_batch_positions(
            self.effective_batches_per_epoch, self.samples_per_epoch
        )
        if any(
            len(epoch) < self.effective_batches_per_epoch
            for epoch in self.batch_indices_by_epoch
        ):
            raise ValueError(
                "Expected each epoch's source-index stream to cover every effective batch."
            )
        self.recorded_steps = 0
        self.recorded_rows = 0
        self.metadata = {
            "schema_version": GRADIENT_TRACE_METADATA_SCHEMA,
            "status": "running",
            "trace_path": str(self.output_path),
            "parameter_names": list(self.names),
            "initial_parameters": self.initial_parameter_metrics,
            "epoch_count": len(self.batch_indices_by_epoch),
            "effective_batches_per_epoch": self.effective_batches_per_epoch,
            "samples_per_epoch": self.samples_per_epoch,
            "sampled_batch_positions": list(self.positions),
            "dataset_provenance": dict(dataset_provenance),
            "sampled_source_batches": [
                {
                    "epoch": epoch_index + 1,
                    "batch": batch_position,
                    "source_indices": list(epoch_batches[batch_position - 1]),
                    "source_indices_sha256": _stable_index_sequence_hash(
                        epoch_batches[batch_position - 1]
                    ),
                }
                for epoch_index, epoch_batches in enumerate(self.batch_indices_by_epoch)
                for batch_position in self.positions
            ],
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        temporary = self.metadata_path.with_name(
            f".{self.metadata_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(
                json.dumps(self.metadata, indent=2, sort_keys=True, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.metadata_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def should_record(self, *, epoch: int, batch: int, total_batches: int) -> bool:
        if int(total_batches) != self.effective_batches_per_epoch:
            raise ValueError(
                "Expected a stable effective batch count across epochs. "
                f"Provided value: expected={self.effective_batches_per_epoch}, "
                f"actual={total_batches}."
            )
        if not 1 <= int(epoch) <= len(self.batch_indices_by_epoch):
            raise ValueError(f"Unexpected trace epoch: {epoch!r}.")
        return int(batch) in self.positions

    def _optimizer_group(self, parameter: Any) -> Mapping[str, Any]:
        state = parameter.state
        matches = [
            group
            for group in self.optimizer.param_groups
            if any(candidate is state for candidate in group["params"])
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one optimizer group for {parameter.name!r}; "
                f"found {len(matches)}."
            )
        return matches[0]

    def _fresh_shadow_proposal(
        self,
        *,
        gradient: Any,
        optimizer_group: Mapping[str, Any],
    ) -> tuple[Any, str]:
        import torch

        learning_rate = _finite(optimizer_group["lr"], "optimizer learning rate")
        optimizer_name = self._optimizer_name()
        if optimizer_name == "SGD":
            if any(
                float(optimizer_group.get(key, 0.0)) != 0.0
                for key in ("momentum", "weight_decay")
            ):
                raise ValueError(
                    "The diagnostic fresh-SGD proposal requires momentum=weight_decay=0."
                )
            return -learning_rate * gradient.detach(), "fresh_sgd"
        if optimizer_name == "Adam":
            if float(optimizer_group.get("weight_decay", 0.0)) != 0.0:
                raise ValueError(
                    "The diagnostic fresh-Adam proposal requires weight_decay=0."
                )
            epsilon = float(optimizer_group["eps"])
            denominator = gradient.detach().abs() + torch.as_tensor(
                epsilon,
                dtype=gradient.dtype,
                device=gradient.device,
            )
            return -learning_rate * gradient.detach() / denominator, "fresh_adam"
        raise ValueError(
            f"Unsupported optimizer for the gradient trace: {optimizer_name!r}."
        )

    def _optimizer_name(self) -> str:
        import torch

        if isinstance(self.optimizer, torch.optim.Adam):
            return "Adam"
        if isinstance(self.optimizer, torch.optim.SGD):
            return "SGD"
        return type(self.optimizer).__name__

    def __call__(self, payload: Mapping[str, Any]) -> bool:
        import torch

        epoch = int(payload["epoch"])
        batch = int(payload["batch"])
        total_batches = int(payload["total_batches"])
        if not self.should_record(epoch=epoch, batch=batch, total_batches=total_batches):
            raise RuntimeError(
                f"Gradient trace callback was invoked for unsampled step {(epoch, batch)!r}."
            )
        if payload.get("optimizer") is not self.optimizer:
            raise RuntimeError("Expected the trace callback to receive its bound optimizer.")
        parameters = tuple(payload["parameters"])
        gradients = tuple(payload["gradients"])
        if (
            len(parameters) != len(self.parameters)
            or any(left is not right for left, right in zip(parameters, self.parameters))
            or len(gradients) != len(parameters)
        ):
            raise RuntimeError("Expected traced parameter and gradient order to remain fixed.")

        source_indices = self.batch_indices_by_epoch[epoch - 1][batch - 1]
        pre = payload["pre_optimizer_states"]
        post_optimizer = payload["post_optimizer_states"]
        post_projection = payload["post_projection_states"]
        rows = []
        for parameter, gradient in zip(parameters, gradients):
            name = str(parameter.name).strip()
            if name not in pre or name not in post_optimizer or name not in post_projection:
                raise RuntimeError(f"Missing traced transition states for {name!r}.")
            before = pre[name]
            optimizer_value = post_optimizer[name]
            projected = post_projection[name]
            raw_update = optimizer_value - before
            applied_update = projected - before
            projection_update = projected - optimizer_value
            gradient_metrics = _tensor_metrics(gradient)
            parameter_metrics = _tensor_metrics(before)
            raw_update_metrics = _tensor_metrics(raw_update)
            applied_update_metrics = _tensor_metrics(applied_update)
            projection_metrics = _tensor_metrics(projection_update)
            optimizer_group = self._optimizer_group(parameter)
            learning_rate = _finite(
                optimizer_group["lr"], f"learning rate for {parameter.name!r}"
            )
            nominal_sgd_update_rms = learning_rate * float(gradient_metrics["rms"])
            fresh_proposal, fresh_proposal_kind = self._fresh_shadow_proposal(
                gradient=gradient,
                optimizer_group=optimizer_group,
            )
            fresh_proposal_metrics = _tensor_metrics(fresh_proposal)
            lower, upper = _effective_bounds(parameter)
            changed_by_projection = float(
                (projected != optimizer_value).to(torch.float64).mean().item()
            )
            optimizer_state = self.optimizer.state.get(parameter.state, {})
            row = {
                "schema_version": GRADIENT_TRACE_SCHEMA,
                "epoch": epoch,
                "batch": batch,
                "total_batches": total_batches,
                "global_step": (epoch - 1) * total_batches + batch,
                "source_indices": list(source_indices),
                "source_indices_sha256": _stable_index_sequence_hash(source_indices),
                "loss": _finite(payload["loss"], "trace loss"),
                "optimizer": self._optimizer_name(),
                "parameter_name": name,
                "parameter_type": type(parameter).__name__,
                "parameter_shape": list(parameter.state.shape),
                "learning_rate": learning_rate,
                "gradient_l2": gradient_metrics["l2"],
                "gradient_rms": gradient_metrics["rms"],
                "gradient_abs_mean": gradient_metrics["abs_mean"],
                "gradient_abs_max": gradient_metrics["abs_max"],
                "gradient_abs_q50": gradient_metrics["abs_q50"],
                "gradient_abs_q90": gradient_metrics["abs_q90"],
                "gradient_abs_q99": gradient_metrics["abs_q99"],
                "gradient_zero_fraction": gradient_metrics["zero_fraction"],
                "parameter_l2_before": parameter_metrics["l2"],
                "parameter_rms_before": parameter_metrics["rms"],
                "parameter_initial_rms": self.initial_parameter_metrics[name]["rms"],
                "parameter_initial_sha256": self.initial_parameter_metrics[name][
                    "sha256"
                ],
                "raw_optimizer_update_l2": raw_update_metrics["l2"],
                "raw_optimizer_update_rms": raw_update_metrics["rms"],
                "applied_update_l2": applied_update_metrics["l2"],
                "applied_update_rms": applied_update_metrics["rms"],
                "projection_update_l2": projection_metrics["l2"],
                "projection_update_rms": projection_metrics["rms"],
                "nominal_sgd_update_rms": nominal_sgd_update_rms,
                "fresh_shadow_proposal_kind": fresh_proposal_kind,
                "fresh_shadow_proposal_rms": fresh_proposal_metrics["rms"],
                "raw_optimizer_over_fresh_shadow_rms": _ratio(
                    float(raw_update_metrics["rms"]),
                    float(fresh_proposal_metrics["rms"]),
                ),
                "applied_over_fresh_shadow_rms": _ratio(
                    float(applied_update_metrics["rms"]),
                    float(fresh_proposal_metrics["rms"]),
                ),
                "raw_optimizer_fresh_shadow_cosine": _vector_cosine(
                    raw_update, fresh_proposal
                ),
                "applied_fresh_shadow_cosine": _vector_cosine(
                    applied_update, fresh_proposal
                ),
                "applied_update_over_parameter_rms": _ratio(
                    float(applied_update_metrics["rms"]),
                    float(parameter_metrics["rms"]),
                ),
                "applied_update_over_initial_parameter_rms": _ratio(
                    float(applied_update_metrics["rms"]),
                    float(self.initial_parameter_metrics[name]["rms"]),
                ),
                "applied_over_nominal_sgd_update_rms": _ratio(
                    float(applied_update_metrics["rms"]), nominal_sgd_update_rms
                ),
                "projection_efficiency_l2": _ratio(
                    float(applied_update_metrics["l2"]),
                    float(raw_update_metrics["l2"]),
                ),
                "projection_changed_fraction": changed_by_projection,
                "descent_cosine_raw_optimizer_update": _descent_cosine(
                    gradient, raw_update
                ),
                "descent_cosine_applied_update": _descent_cosine(
                    gradient, applied_update
                ),
                "lower_bound": lower,
                "upper_bound": upper,
                "lower_bound_fraction_before": _bound_fraction(before, lower),
                "upper_bound_fraction_before": _bound_fraction(before, upper),
                "lower_bound_fraction_after": _bound_fraction(projected, lower),
                "upper_bound_fraction_after": _bound_fraction(projected, upper),
                **_optimizer_state_metrics(
                    optimizer_state,
                    gradient=gradient,
                    optimizer_group=optimizer_group,
                ),
            }
            rows.append(row)

        with self.output_path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.recorded_steps += 1
        self.recorded_rows += len(rows)
        return False

    def finalize(self) -> dict[str, Any]:
        expected_steps = len(self.batch_indices_by_epoch) * len(self.positions)
        expected_rows = expected_steps * len(self.parameters)
        if self.recorded_steps != expected_steps or self.recorded_rows != expected_rows:
            raise RuntimeError(
                "Expected every planned gradient-trace row to be recorded. "
                f"Provided value: steps={self.recorded_steps}/{expected_steps}, "
                f"rows={self.recorded_rows}/{expected_rows}."
            )
        self.metadata.update(
            {
                "status": "complete",
                "recorded_steps": self.recorded_steps,
                "recorded_rows": self.recorded_rows,
                "trace_size_bytes": self.output_path.stat().st_size,
                "trace_sha256": _sha256_file(self.output_path),
            }
        )
        self._write_metadata()
        return dict(self.metadata)


__all__ = [
    "GRADIENT_TRACE_METADATA_SCHEMA",
    "GRADIENT_TRACE_SCHEMA",
    "GradientTraceRecorder",
    "sampled_batch_positions",
]
