"""One-batch functional smoke for perfect-diode Conv training.

This module deliberately tests capabilities rather than reproducing an exact
host environment.  Scientific inputs (study, row, optimizer, initialization,
and learning-rate vector) remain fixed, while Python/PyTorch/CUDA versions are
recorded as provenance.
"""

from __future__ import annotations

import math
import platform
from pathlib import Path
import sys
import time
from typing import Any, Callable, Mapping

from . import perfectdiode_hparam_runtime as _runtime
from .identity import sha256_file
from .io import atomic_create_json
from .lr_engine import (
    canonical_parameter_name,
    parameter_state_diagnostics,
    parameter_tensor_digest,
    training_step,
)
from .lr_step import require_finite_optimizer_state
from .perfectdiode_hparam_spec import PerfectDiodeHparamStudySpec


FUNCTIONAL_PREFLIGHT_SCHEMA = "experiment-functional-preflight/v1"
FUNCTIONAL_SMOKE_KIND = "perfectdiode-conv-one-batch/v1"


class FunctionalSmokeError(RuntimeError):
    """The target could not complete the bounded functional smoke."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Expected {label} to be non-empty text. Provided value: {value!r}."
        )
    return value.strip()


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(
            f"Expected {label} to be a finite number. Provided value: {value!r}."
        )
    return float(value)


def _rates(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "Expected raw_learning_rates_by_parameter to be a non-empty "
            f"mapping. Provided value: {value!r}."
        )
    result: dict[str, float] = {}
    for raw_name, raw_rate in value.items():
        name = _text(raw_name, "a learning-rate parameter name")
        rate = _finite(raw_rate, f"learning rate for {name!r}")
        if rate <= 0.0:
            raise ValueError(
                f"Expected learning rate for {name!r} to be positive. "
                f"Provided value: {raw_rate!r}."
            )
        result[name] = rate
    return result


def _transition_records(transition: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in transition.parameters:
        record = {
            "name": _text(item.name, "transition parameter name"),
            "bounded_gate": bool(item.bounded_gate),
            "gradient_rms": _finite(
                item.gradient_rms, f"gradient RMS for {item.name!r}"
            ),
            "proposed_update_rms": _finite(
                item.proposed_update_rms,
                f"proposed update RMS for {item.name!r}",
            ),
            "normalized_update": _finite(
                item.normalized_update,
                f"normalized update for {item.name!r}",
            ),
            "projection_efficiency": _finite(
                item.projection_efficiency,
                f"projection efficiency for {item.name!r}",
            ),
        }
        records.append(record)
    if not records:
        raise FunctionalSmokeError(
            "Expected the one-batch optimizer step to report at least one "
            "parameter transition. Provided value: an empty transition."
        )
    return records


def capture_environment(device: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise FunctionalSmokeError(
            "Expected the target to import PyTorch. Provided value: import "
            f"failed with {exc!r}."
        ) from exc

    cuda_requested = str(device).startswith("cuda")
    cuda_available = bool(torch.cuda.is_available())
    if cuda_requested and not cuda_available:
        raise FunctionalSmokeError(
            f"Expected CUDA to be available for device {device!r}. "
            f"Provided value: torch.cuda.is_available()={cuda_available!r}."
        )
    gpu_name: str | None = None
    cuda_runtime: str | None = None
    cuda_devices: list[dict[str, Any]] = []
    if cuda_requested:
        try:
            device_index = torch.device(device).index
            index = 0 if device_index is None else int(device_index)
            gpu_name = str(torch.cuda.get_device_name(index))
            cuda_runtime = (
                None if torch.version.cuda is None else str(torch.version.cuda)
            )
            for device_number in range(int(torch.cuda.device_count())):
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_number)
                cuda_devices.append(
                    {
                        "index": device_number,
                        "name": str(torch.cuda.get_device_name(device_number)),
                        "free_memory_bytes": int(free_bytes),
                        "total_memory_bytes": int(total_bytes),
                    }
                )
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise FunctionalSmokeError(
                f"Expected CUDA device {device!r} to be usable. "
                f"Provided value: {exc!r}."
            ) from exc
    return {
        "python_version": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "pytorch_version": str(torch.__version__),
        "cuda_requested": cuda_requested,
        "cuda_available": cuda_available,
        "cuda_runtime": cuda_runtime,
        "cuda_device_count": len(cuda_devices),
        "cuda_devices": cuda_devices,
        "gpu_name": gpu_name,
        "device": str(device),
    }


def execute_one_batch(
    runtime: Any,
    bundle: Any,
    *,
    learning_rates_by_parameter: Mapping[str, float],
    checkpoint_path: str | Path,
    step_function: Callable[..., Any] = training_step,
    save_checkpoint: Callable[[Any, Path], Path] = _runtime._save_checkpoint_atomic,
) -> dict[str, Any]:
    """Execute exactly one real training batch and write one checkpoint."""

    rates = _rates(learning_rates_by_parameter)
    parameter_names = {
        canonical_parameter_name(parameter) for parameter in runtime.parameters
    }
    if set(rates) != parameter_names:
        raise ValueError(
            "Expected raw_learning_rates_by_parameter to cover every runtime "
            f"parameter exactly. Provided value: rates={sorted(rates)!r}, "
            f"parameters={sorted(parameter_names)!r}."
        )

    bundle.reset_train_shuffle()
    expected_epochs = bundle.train_batch_indices(num_epochs=1)
    if not expected_epochs or not expected_epochs[0]:
        raise FunctionalSmokeError(
            "Expected the deterministic training stream to contain one batch. "
            f"Provided value: {expected_epochs!r}."
        )
    expected_indices = tuple(int(item) for item in expected_epochs[0][0])
    iterator = iter(bundle.train_loader)
    try:
        batch = next(iterator)
    except StopIteration as exc:
        raise FunctionalSmokeError(
            "Expected the training loader to yield one batch. "
            "Provided value: an empty loader."
        ) from exc

    before_digest = parameter_tensor_digest(runtime.parameters)
    started = time.monotonic()
    result = step_function(
        runtime,
        batch,
        learning_rate=rates,
        restore=False,
        zero_proposal_epsilon=1.0e-12,
    )
    elapsed = max(time.monotonic() - started, 0.0)
    require_finite_optimizer_state(
        runtime.optimizer,
        context="functional smoke optimizer state after step 1",
    )
    if tuple(int(item) for item in result.source_indices) != expected_indices:
        raise FunctionalSmokeError(
            "Expected the one-batch smoke to use the first deterministic "
            f"training minibatch. Provided value: {result.source_indices!r}; "
            f"expected value: {expected_indices!r}."
        )
    loss = _finite(result.loss, "one-batch training loss")
    accuracy = _finite(result.accuracy, "one-batch training accuracy")
    if not 0.0 <= accuracy <= 1.0:
        raise FunctionalSmokeError(
            "Expected one-batch training accuracy in [0, 1]. "
            f"Provided value: {accuracy!r}."
        )
    transitions = _transition_records(result.transition)
    after = parameter_state_diagnostics(runtime.parameters)
    after_digest = parameter_tensor_digest(runtime.parameters)
    destination = Path(checkpoint_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "Expected a fresh disposable checkpoint path. "
            f"Provided value: {destination}."
        )
    save_checkpoint(runtime, destination)
    if not destination.is_file() or destination.is_symlink():
        raise FunctionalSmokeError(
            "Expected the one-batch smoke to write a regular checkpoint. "
            f"Provided value: {destination}."
        )
    return {
        "kind": FUNCTIONAL_SMOKE_KIND,
        "status": "passed",
        "completed_steps": 1,
        "sample_count": int(result.sample_count),
        "source_indices": list(expected_indices),
        "loss": loss,
        "accuracy": accuracy,
        "elapsed_seconds": elapsed,
        "parameter_tensor_sha256_before": before_digest,
        "parameter_tensor_sha256_after": after_digest,
        "parameter_states_after": after,
        "parameter_transitions": transitions,
        "checkpoint": {
            "path": destination.name,
            "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size,
            "disposable": True,
        },
    }


def validate_functional_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the compact receipt consumed by the generic executor."""

    if not isinstance(value, Mapping):
        raise TypeError(
            "Expected a functional preflight receipt mapping. "
            f"Provided value: {type(value).__name__}."
        )
    if value.get("schema_version") != FUNCTIONAL_PREFLIGHT_SCHEMA:
        raise ValueError(
            f"Expected schema_version {FUNCTIONAL_PREFLIGHT_SCHEMA!r}. "
            f"Provided value: {value.get('schema_version')!r}."
        )
    if value.get("status") != "passed" or value.get("passed") is not True:
        raise ValueError(
            "Expected functional preflight status='passed' and passed=true. "
            f"Provided value: status={value.get('status')!r}, "
            f"passed={value.get('passed')!r}."
        )
    if value.get("official_test_read") is not False:
        raise ValueError(
            "Expected official_test_read=false. "
            f"Provided value: {value.get('official_test_read')!r}."
        )
    smoke = value.get("smoke")
    if not isinstance(smoke, Mapping):
        raise ValueError(
            "Expected smoke to be a mapping. "
            f"Provided value: {smoke!r}."
        )
    if (
        smoke.get("kind") != FUNCTIONAL_SMOKE_KIND
        or smoke.get("status") != "passed"
        or smoke.get("completed_steps") != 1
    ):
        raise ValueError(
            "Expected a passed perfect-diode one-batch smoke with exactly one "
            f"completed step. Provided value: {smoke!r}."
        )
    _finite(smoke.get("loss"), "smoke loss")
    _finite(smoke.get("accuracy"), "smoke accuracy")
    checkpoint = smoke.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("disposable") is not True
        or not isinstance(checkpoint.get("sha256"), str)
        or len(checkpoint["sha256"]) != 64
    ):
        raise ValueError(
            "Expected smoke.checkpoint to describe a disposable SHA-256-bound "
            f"checkpoint. Provided value: {checkpoint!r}."
        )
    return dict(value)


def run_perfectdiode_one_batch_smoke(
    study: Mapping[str, Any] | PerfectDiodeHparamStudySpec,
    shard_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    target: str,
    data_root: str | Path,
    device: str,
    output_dir: str | Path,
    download: bool = False,
) -> dict[str, Any]:
    """Load fixed scientific inputs and publish one immutable smoke receipt."""

    attempt = _text(attempt_id, "attempt_id")
    target_name = _text(target, "target")
    entry_id = _text(payload.get("entry_id"), "payload.entry_id")
    root = Path(shard_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Expected functional-smoke output_dir outside the immutable input "
            f"bundle. Provided value: {output}."
        )
    receipt_path = output / "functional_preflight.json"
    checkpoint_path = output / "disposable_checkpoint.pt"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            "Expected a fresh attempt-specific functional-smoke output "
            f"directory. Provided value: {output}."
        )
    for path in (receipt_path, checkpoint_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                "Expected a fresh attempt-specific functional-smoke output. "
                f"Provided value: {path}."
            )

    environment = capture_environment(device)
    data, spec = _runtime._study_and_spec(study)
    prepared = _runtime._prepare_training_entry(
        data,
        spec,
        root,
        payload,
        data_root=data_root,
        download=download,
    )
    row = prepared["row"]
    assets = prepared["assets"]
    resolved_payload = prepared["payload"]
    rates = prepared["rates"]
    optimizer_name = prepared["optimizer_name"]
    runtime = _runtime._build_fresh_runtime(
        data,
        row,
        assets,
        optimizer_name=optimizer_name,
        rates=rates,
        device=device,
    )
    try:
        benchmark_started, benchmark_cuda = _runtime._runtime_benchmark_start(
            device
        )
        smoke = execute_one_batch(
            runtime,
            assets.bundle,
            learning_rates_by_parameter=rates,
            checkpoint_path=checkpoint_path,
        )
        smoke["benchmark"] = _runtime._runtime_benchmark_finish(
            benchmark_started,
            benchmark_cuda,
            int(smoke["completed_steps"]),
        )
    finally:
        del runtime

    receipt = {
        "schema_version": FUNCTIONAL_PREFLIGHT_SCHEMA,
        "status": "passed",
        "passed": True,
        "attempt_id": attempt,
        "target": target_name,
        "official_test_read": False,
        "capabilities": {
            "required_imports": ["torch", "labs.datasets"],
            "dataset_readable": True,
            "output_writable": True,
            "real_training_step": True,
            "finite_numerics": True,
            "checkpoint_written": True,
            "result_schema_valid": True,
        },
        "environment": environment,
        "scientific_binding": {
            "study_id": spec.study_id,
            "config_sha256": spec.config_sha256,
            "entry_id": entry_id,
            "row_id": row["row_id"],
            "architecture": row["architecture"],
            "scheme": row["scheme"],
            "optimizer": optimizer_name,
            "cell_id": resolved_payload["cell_id"],
            "raw_learning_rates_by_parameter": rates,
            "initialization_checkpoint_sha256": assets.initialization[
                "checkpoint_sha256"
            ],
            "initialization_tensor_sha256": assets.initialization[
                "parameter_tensor_sha256"
            ],
            "train_indices_sha256": assets.bundle.train_indices_hash,
        },
        "smoke": smoke,
    }
    validate_functional_preflight(receipt)
    output.mkdir(parents=True, exist_ok=True)
    atomic_create_json(receipt_path, receipt, canonical=True)
    return receipt


__all__ = [
    "FUNCTIONAL_PREFLIGHT_SCHEMA",
    "FUNCTIONAL_SMOKE_KIND",
    "FunctionalSmokeError",
    "capture_environment",
    "execute_one_batch",
    "run_perfectdiode_one_batch_smoke",
    "validate_functional_preflight",
]
