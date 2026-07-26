"""Numerical runtime for the Conv2 SGD/Adam boundary-extension diagnostic.

This module is a new execution path.  It deliberately does not alter the
v1--v6 stage implementation.  Its responsibilities are:

* stage and verify one shared seed-0 Conv2 initialization;
* freeze the ordinary-MNIST split and the first 32 shuffled minibatches;
* measure optimizer-specific unit-LR shadow proposals with exact restoration;
* derive explicit learning rates for every scientific parameter; and
* execute a five-epoch candidate while replaying the same first eight batches
  at initialization and after every epoch.

Only the MNIST training split is instantiated.  The official test split is
never read.
"""

from __future__ import annotations

import copy
import hashlib
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .identity import canonical_json_bytes, sha256_file, sha256_json
from .io import atomic_write_csv, atomic_write_json, read_json
from .lr_engine import (
    LRStudyNumericalError,
    build_loader_bundle,
    build_model_runtime,
    canonical_parameter_name,
    evaluate_validation,
    parameter_state_diagnostics,
    parameter_tensor_digest,
    training_step,
    validate_explicit_optimizer_contract,
)
from .lr_step import OptimizerStateNumericalError, require_finite_optimizer_state


PROBE_SCHEMA_VERSION = "mnist-conv-lr-optimizer-probe/v1"
PROBE_BATCH_SCHEMA_VERSION = "mnist-conv-lr-optimizer-probe-batches/v1"
INITIALIZATION_SCHEMA_VERSION = "mnist-conv-lr-optimizer-initialization/v1"
REPLAY_SCHEMA_VERSION = "mnist-conv-lr-optimizer-epoch-replay/v1"
RESULT_SCHEMA_VERSION = "mnist-conv-lr-optimizer-boundary-result/v1"

NORMALIZATION_BATCHES = 32
REPLAY_BATCHES = 8
EPOCHS = 5
STEPS_PER_EPOCH = 3_438
TOTAL_STEPS = 17_190
TRAINING_EXAMPLES = 55_000
VALIDATION_EXAMPLES = 5_000
BIAS_RHO_CAP = 1.0e-3

FROZEN_VALIDATION_SPLIT_SHA256 = (
    "4801b7805d54dbe2197b98320cfe8a34d5d853c4ab568bcf1d7f69e136c94bb4"
)
FROZEN_REPLAY_BATCH_ORDER_SHA256 = (
    "1d67f374dea1d58ed7d779ed5eb14471fce3b6c64e614c47f47fda5526bc5673"
)
FROZEN_NORMALIZATION_BATCH_ORDER_SHA256 = (
    "1c65076487a87e0badfe19b80edc8263f50c4b765dc317299dec4a8f1fad460b"
)
FROZEN_TRAINING_BATCH_ORDER_SHA256 = (
    "e774de39fe6f9529dafded0b14356727b3e8510bd42f3a92f7416dcab0592d14"
)
FROZEN_INITIALIZATION_CHECKPOINT_SHA256 = (
    "dc16ef2bebd2c9e6afe307b41555683c3eec29b9481ab64ea3f412382c837c7e"
)
FROZEN_INITIALIZATION_TENSOR_SHA256 = (
    "e9aa471dc5558e525758b345de12c53f0a086c5a1b022024c2fc349ec138e2e3"
)

WEIGHT_NAMES = ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
BIAS_NAMES = ("Bias_0", "Bias_1")
PARAMETER_NAMES = (
    "ConvWeight_0",
    "Bias_0",
    "ConvWeight_1",
    "Bias_1",
    "DenseWeight_0",
)
BIAS_TO_WEIGHT = {
    "Bias_0": "ConvWeight_0",
    "Bias_1": "ConvWeight_1",
}


class BoundaryRuntimeError(RuntimeError):
    """Raised when an immutable scientific runtime contract is violated."""


@dataclass
class BoundaryAssets:
    """In-memory handles plus immutable metadata for one process."""

    study_dir: Path
    checkpoint_path: Path
    initialization: dict[str, Any]
    bundle: Any
    probe_batches: tuple[tuple[Any, Any, Any], ...]
    minibatches: dict[str, Any]
    minibatches_path: Path


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(
            f"Expected {label} to be a {qualifier} number. "
            f"Provided value: {value!r}."
        )
    return float(value)


def _integer(value: Any, label: str, *, expected: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Expected {label} to be an integer. Provided value: {value!r}."
        )
    if expected is not None and value != expected:
        raise ValueError(
            f"Expected {label} to be exactly {expected}. "
            f"Provided value: {value!r}."
        )
    return int(value)


def _sha256(value: Any, label: str, *, nullable: bool = False) -> str | None:
    import re

    if nullable and value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        expected = "null or a lowercase SHA-256 digest" if nullable else (
            "a lowercase SHA-256 digest"
        )
        raise ValueError(
            f"Expected {label} to be {expected}. Provided value: {value!r}."
        )
    return value


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    """Return the deterministic type-7/linear quantile used by torch/numpy."""

    ordered = sorted(_finite(value, "quantile sample") for value in values)
    if not ordered:
        raise ValueError(
            "Expected quantile samples to be non-empty. Provided value: []."
        )
    q = _finite(probability, "quantile probability")
    if not 0.0 <= q <= 1.0:
        raise ValueError(
            "Expected quantile probability to be in [0, 1]. "
            f"Provided value: {probability!r}."
        )
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _rms(tensor: Any) -> float:
    import torch

    values = tensor.detach().to(dtype=torch.float64)
    if values.numel() == 0:
        raise ValueError(
            f"Expected a non-empty tensor. Provided shape: {tuple(values.shape)!r}."
        )
    result = float(torch.sqrt(torch.mean(values * values)).item())
    return _finite(result, "tensor RMS")


def _parameter_by_name(parameters: Iterable[Any]) -> dict[str, Any]:
    result = {
        canonical_parameter_name(parameter): parameter for parameter in parameters
    }
    if set(result) != set(PARAMETER_NAMES):
        raise BoundaryRuntimeError(
            "Expected exactly the five Conv2 scientific parameters. "
            f"Provided value: {sorted(result)!r}."
        )
    return result


def _row_by_id(study: Mapping[str, Any], row_id: str) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in study.get("rows", ())
        if row.get("row_id") == row_id
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one study row with row_id {row_id!r}. "
            f"Provided value: {len(rows)}."
        )
    row = rows[0]
    if row.get("architecture") != "conv2":
        raise ValueError(
            "Expected the optimizer-boundary row architecture to be 'conv2'. "
            f"Provided value: {row.get('architecture')!r}."
        )
    return row


def _optimizer_contract(
    study: Mapping[str, Any], optimizer_name: str
) -> dict[str, Any]:
    key = str(optimizer_name).lower()
    if key not in {"sgd", "adam"}:
        raise ValueError(
            "Expected optimizer_name to be 'sgd' or 'adam'. "
            f"Provided value: {optimizer_name!r}."
        )
    arms = study.get("optimizer_arms")
    if not isinstance(arms, Mapping) or key not in arms:
        raise ValueError(
            "Expected study.optimizer_arms to contain both 'sgd' and 'adam'. "
            f"Provided value: {arms!r}."
        )
    return validate_explicit_optimizer_contract(arms[key])


def _optimizer_name_lower(contract: Mapping[str, Any]) -> str:
    name = validate_explicit_optimizer_contract(contract)["name"]
    return name.lower()


def _hash_update_typed(digest: Any, value: Any) -> None:
    """Hash a nested optimizer state without relying on pickle serialization."""

    import torch

    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T")
        _hash_update_typed(digest, str(tensor.dtype))
        _hash_update_typed(digest, [int(item) for item in tensor.shape])
        raw = tensor.numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big", signed=False))
        digest.update(raw)
    elif isinstance(value, Mapping):
        digest.update(b"M")
        digest.update(len(value).to_bytes(8, "big", signed=False))
        # Optimizer state dictionaries use integer and string keys.  Sort by a
        # typed representation so cross-type comparisons never occur.
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _hash_update_typed(digest, key)
            _hash_update_typed(digest, value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"L" if isinstance(value, list) else b"Q")
        digest.update(len(value).to_bytes(8, "big", signed=False))
        for item in value:
            _hash_update_typed(digest, item)
    elif value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        digest.update(b"I")
        encoded = str(value).encode("ascii")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise OptimizerStateNumericalError(
                "Expected optimizer state digest input to be finite. "
                f"Provided value: {value!r}."
            )
        digest.update(b"F")
        digest.update(float(value).hex().encode("ascii"))
    elif isinstance(value, str):
        digest.update(b"S")
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    else:
        raise TypeError(
            "Expected optimizer state to contain tensors, mappings, sequences, "
            "numbers, booleans, strings, or null. "
            f"Provided value: {type(value).__name__}."
        )


def optimizer_state_digest(optimizer: Any) -> str:
    """Hash parameters groups and the complete mutable optimizer state."""

    require_finite_optimizer_state(optimizer, context="optimizer digest state")
    digest = hashlib.sha256()
    digest.update(b"mnist-conv-complete-optimizer-state/v1\0")
    _hash_update_typed(digest, optimizer.state_dict())
    return digest.hexdigest()


def _publish_immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    """Write canonical JSON once; existing bytes must be identical."""

    encoded = canonical_json_bytes(dict(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise BoundaryRuntimeError(
                f"Expected immutable artifact to be a regular file: {path}."
            )
        if path.read_bytes() != encoded:
            raise BoundaryRuntimeError(
                f"Expected immutable artifact {path} to retain identical bytes. "
                "Provided value: conflicting content."
            )
        return path
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise BoundaryRuntimeError(
                f"Expected concurrently published artifact {path} to have "
                "identical bytes. Provided value: conflicting content."
            )
        return path
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return path


def _save_checkpoint_atomic(runtime: Any, target: Path) -> Path:
    """Save a model-only checkpoint and atomically replace the destination."""

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.unlink()
        runtime.save(temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _copy_checkpoint_once(source: Path, target: Path) -> Path:
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(
            "Expected initialization_source to be a regular checkpoint file. "
            f"Provided value: {source}."
        )
    source_hash = sha256_file(source)
    if source_hash != FROZEN_INITIALIZATION_CHECKPOINT_SHA256:
        raise BoundaryRuntimeError(
            "Expected initialization_source to be the exact frozen Conv2 "
            "checkpoint. "
            f"Provided value: observed_sha256={source_hash!r}, "
            f"expected_sha256={FROZEN_INITIALIZATION_CHECKPOINT_SHA256!r}."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise BoundaryRuntimeError(
                f"Expected initialization target to be a regular file: {target}."
            )
        if sha256_file(target) != sha256_file(source):
            raise BoundaryRuntimeError(
                "Expected an existing initialization checkpoint to match the "
                f"declared source exactly. Provided target: {target}."
            )
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def _clone_batch(batch: Any) -> tuple[Any, Any, Any]:
    import torch

    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise ValueError(
            "Expected an indexed MNIST batch as (images, labels, indices). "
            f"Provided value: {type(batch).__name__}."
        )
    values = []
    for item in batch:
        if not torch.is_tensor(item):
            raise TypeError(
                "Expected every indexed MNIST batch component to be a tensor. "
                f"Provided value: {type(item).__name__}."
            )
        values.append(item.detach().cpu().clone())
    return values[0], values[1], values[2]


def _source_indices(batch: Sequence[Any]) -> tuple[int, ...]:
    return tuple(int(value) for value in batch[2].reshape(-1).tolist())


def _batch_order_hash(batches: Sequence[Sequence[int]]) -> str:
    from labs.datasets import stable_batch_order_hash

    return stable_batch_order_hash(batches)


def _freeze_probe_batches(bundle: Any) -> tuple[tuple[Any, Any, Any], ...]:
    bundle.reset_train_shuffle()
    batches: list[tuple[Any, Any, Any]] = []
    for index, batch in enumerate(bundle.train_loader):
        if index >= NORMALIZATION_BATCHES:
            break
        batches.append(_clone_batch(batch))
    if len(batches) != NORMALIZATION_BATCHES:
        raise BoundaryRuntimeError(
            f"Expected exactly {NORMALIZATION_BATCHES} probe minibatches. "
            f"Provided value: {len(batches)}."
        )
    expected = bundle.train_batch_indices(num_epochs=1)[0][:NORMALIZATION_BATCHES]
    observed = tuple(_source_indices(batch) for batch in batches)
    if observed != tuple(expected):
        raise BoundaryRuntimeError(
            "Expected the materialized probe batches to match the deterministic "
            "train shuffle exactly."
        )
    return tuple(batches)


def _probe_minibatch_payload(bundle: Any, batches: Sequence[Any]) -> dict[str, Any]:
    normalization = [_source_indices(batch) for batch in batches]
    replay = normalization[:REPLAY_BATCHES]
    return {
        "schema_version": PROBE_BATCH_SCHEMA_VERSION,
        "official_test_read": False,
        "batch_size": int(bundle.batch_size),
        "normalization_batch_count": NORMALIZATION_BATCHES,
        "replay_batch_count": REPLAY_BATCHES,
        "train_indices_sha256": bundle.train_indices_hash,
        "validation_indices_sha256": bundle.validation_indices_hash,
        "normalization_batches": [list(batch) for batch in normalization],
        "replay_batches": [list(batch) for batch in replay],
        "normalization_batch_order_sha256": _batch_order_hash(normalization),
        "replay_batch_order_sha256": _batch_order_hash(replay),
    }


def _resolve_initialization_source(
    initialization_source: str | Path | None,
) -> Path | None:
    if initialization_source is not None:
        return Path(initialization_source).expanduser().resolve()
    environment_value = os.environ.get("MNIST_CONV_BOUNDARY_INITIALIZATION")
    if environment_value:
        return Path(environment_value).expanduser().resolve()
    return None


def verify_shared_initialization(
    study: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    device: str,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify natural and checkpoint-loaded parameter identity for all schemes."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(
            "Expected shared initialization checkpoint to be a regular file. "
            f"Provided value: {checkpoint}."
        )
    checkpoint_sha256 = sha256_file(checkpoint)
    expected = (
        FROZEN_INITIALIZATION_CHECKPOINT_SHA256
        if expected_checkpoint_sha256 is None
        else _sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    )
    if expected != FROZEN_INITIALIZATION_CHECKPOINT_SHA256:
        raise BoundaryRuntimeError(
            "Expected the entry checkpoint binding to equal the frozen Conv2 "
            f"checkpoint SHA-256. Provided value: {expected!r}."
        )
    if checkpoint_sha256 != expected:
        raise BoundaryRuntimeError(
            "Expected the shared initialization checkpoint SHA-256 to match "
            f"the frozen parent. Provided value: observed={checkpoint_sha256!r}, "
            f"expected={expected!r}."
        )

    natural: dict[str, str] = {}
    loaded: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}
    sgd_contract = _optimizer_contract(study, "sgd")
    for row in study["rows"]:
        row_id = str(row["row_id"])
        natural_runtime = build_model_runtime(
            study,
            row,
            device=device,
            learning_rate={name: 1.0 for name in PARAMETER_NAMES},
            optimizer_contract=sgd_contract,
        )
        natural[row_id] = parameter_tensor_digest(natural_runtime.parameters)
        diagnostics[row_id] = parameter_state_diagnostics(
            natural_runtime.parameters
        )
        loaded_runtime = build_model_runtime(
            study,
            row,
            device=device,
            initialization_checkpoint=checkpoint,
            learning_rate={name: 1.0 for name in PARAMETER_NAMES},
            optimizer_contract=sgd_contract,
        )
        loaded[row_id] = parameter_tensor_digest(loaded_runtime.parameters)
    if len(set(natural.values())) != 1:
        raise BoundaryRuntimeError(
            "Expected all three amplification rows to have identical natural "
            f"seed-0 parameter tensors. Provided value: {natural!r}."
        )
    tensor_sha256 = next(iter(natural.values()))
    if tensor_sha256 != FROZEN_INITIALIZATION_TENSOR_SHA256:
        raise BoundaryRuntimeError(
            "Expected the natural seed-0 Conv2 parameter tensor hash to match "
            f"the frozen checkpoint lineage. Provided value: {tensor_sha256!r}."
        )
    if set(loaded.values()) != {tensor_sha256}:
        raise BoundaryRuntimeError(
            "Expected the shared checkpoint to load the same seed-0 parameter "
            f"tensors for every row. Provided value: {loaded!r}."
        )
    return {
        "schema_version": INITIALIZATION_SCHEMA_VERSION,
        "official_test_read": False,
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256,
        "parameter_tensor_sha256": tensor_sha256,
        "natural_parameter_tensor_sha256_by_row": natural,
        "loaded_parameter_tensor_sha256_by_row": loaded,
        "initial_parameter_diagnostics_by_row": diagnostics,
    }


def load_verified_shared_initialization(
    study: Mapping[str, Any],
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the one-time full verification record and recheck bound bytes."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    metadata_file = Path(metadata_path).expanduser().resolve()
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise FileNotFoundError(
            "Expected the staged Conv2 initialization checkpoint to exist. "
            f"Provided value: {checkpoint}."
        )
    if metadata_file.is_symlink() or not metadata_file.is_file():
        raise FileNotFoundError(
            "Expected the shared-initialization verification metadata to exist. "
            f"Provided value: {metadata_file}."
        )
    metadata = read_json(metadata_file)
    if metadata.get("schema_version") != INITIALIZATION_SCHEMA_VERSION:
        raise BoundaryRuntimeError(
            f"Expected initialization metadata schema {INITIALIZATION_SCHEMA_VERSION!r}. "
            f"Provided value: {metadata.get('schema_version')!r}."
        )
    if metadata.get("official_test_read") is not False:
        raise BoundaryRuntimeError(
            "Expected initialization metadata official_test_read to be false."
        )
    expected = (
        FROZEN_INITIALIZATION_CHECKPOINT_SHA256
        if expected_checkpoint_sha256 is None
        else _sha256(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    )
    observed_checkpoint = sha256_file(checkpoint)
    if (
        expected != FROZEN_INITIALIZATION_CHECKPOINT_SHA256
        or observed_checkpoint != expected
        or metadata.get("checkpoint_sha256") != expected
    ):
        raise BoundaryRuntimeError(
            "Expected checkpoint bytes, entry binding, and initialization "
            "metadata to agree with the frozen Conv2 SHA-256. "
            f"Provided value: entry={expected!r}, observed={observed_checkpoint!r}, "
            f"metadata={metadata.get('checkpoint_sha256')!r}."
        )
    if metadata.get("parameter_tensor_sha256") != FROZEN_INITIALIZATION_TENSOR_SHA256:
        raise BoundaryRuntimeError(
            "Expected initialization metadata to bind the frozen parameter "
            f"tensor hash. Provided value: {metadata.get('parameter_tensor_sha256')!r}."
        )
    expected_rows = {str(row["row_id"]) for row in study["rows"]}
    for field in (
        "natural_parameter_tensor_sha256_by_row",
        "loaded_parameter_tensor_sha256_by_row",
    ):
        values = metadata.get(field)
        if not isinstance(values, Mapping) or set(values) != expected_rows:
            raise BoundaryRuntimeError(
                f"Expected initialization metadata {field!r} to cover all rows. "
                f"Provided value: {values!r}."
            )
        if set(values.values()) != {FROZEN_INITIALIZATION_TENSOR_SHA256}:
            raise BoundaryRuntimeError(
                f"Expected every {field!r} value to equal the frozen tensor hash. "
                f"Provided value: {values!r}."
            )
    return metadata


def prepare_boundary_assets(
    study: Mapping[str, Any],
    study_dir: str | Path,
    *,
    data_root: str | Path,
    download: bool,
    device: str,
    initialization_source: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> BoundaryAssets:
    """Build/verify the shared initialization, loaders, and frozen minibatches."""

    root = Path(study_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if study.get("dataset", {}).get("official_test", {}).get("read_allowed") is not False:
        raise BoundaryRuntimeError(
            "Expected study.dataset.official_test.read_allowed to be false. "
            f"Provided value: {study.get('dataset', {}).get('official_test')!r}."
        )
    checkpoint = root / "initialization" / "conv2.pt"
    source = _resolve_initialization_source(initialization_source)
    if not checkpoint.exists():
        if source is None:
            raise FileNotFoundError(
                "Expected an explicit initialization_source (or "
                "MNIST_CONV_BOUNDARY_INITIALIZATION) when staging a new study. "
                "Provided value: None."
            )
        _copy_checkpoint_once(source, checkpoint)
    initialization_path = root / "initialization" / "conv2.json"
    if source is not None:
        _copy_checkpoint_once(source, checkpoint)
    if initialization_path.is_file():
        initialization = load_verified_shared_initialization(
            study,
            checkpoint,
            initialization_path,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
    else:
        initialization = verify_shared_initialization(
            study,
            checkpoint,
            device=device,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
        _publish_immutable_json(initialization_path, initialization)

    bundle = build_loader_bundle(
        study,
        data_root=data_root,
        download=download,
        return_source_indices=True,
    )
    if (
        int(bundle.batch_size) != 16
        or len(bundle.train_indices) != TRAINING_EXAMPLES
        or len(bundle.validation_indices) != VALIDATION_EXAMPLES
    ):
        raise BoundaryRuntimeError(
            "Expected the frozen 55,000/5,000 ordinary-MNIST split with "
            f"batch size 16. Provided value: train={len(bundle.train_indices)}, "
            f"validation={len(bundle.validation_indices)}, "
            f"batch_size={bundle.batch_size}."
        )
    if bundle.validation_indices_hash != FROZEN_VALIDATION_SPLIT_SHA256:
        raise BoundaryRuntimeError(
            "Expected the ordinary-MNIST validation split hash to match the "
            f"frozen protocol. Provided value: {bundle.validation_indices_hash!r}."
        )
    batches = _freeze_probe_batches(bundle)
    minibatches = _probe_minibatch_payload(bundle, batches)
    if (
        minibatches["normalization_batch_order_sha256"]
        != FROZEN_NORMALIZATION_BATCH_ORDER_SHA256
        or minibatches["replay_batch_order_sha256"]
        != FROZEN_REPLAY_BATCH_ORDER_SHA256
    ):
        raise BoundaryRuntimeError(
            "Expected the first 32 normalization and first eight replay "
            "minibatch hashes to match the frozen ordinary-MNIST stream. "
            f"Provided value: normalization="
            f"{minibatches['normalization_batch_order_sha256']!r}, replay="
            f"{minibatches['replay_batch_order_sha256']!r}."
        )
    minibatches_path = root / "probes" / "minibatches.json"
    _publish_immutable_json(minibatches_path, minibatches)
    return BoundaryAssets(
        study_dir=root,
        checkpoint_path=checkpoint,
        initialization=initialization,
        bundle=bundle,
        probe_batches=batches,
        minibatches=minibatches,
        minibatches_path=minibatches_path,
    )


def _transition_proposal_tensor(item: Any) -> Any:
    if hasattr(item, "post_optimizer_pre_projection"):
        return item.post_optimizer_pre_projection - item.pre_update
    return item.post_sgd_pre_projection - item.pre_update


def measure_optimizer_probe(
    study: Mapping[str, Any],
    assets: BoundaryAssets,
    *,
    row_id: str,
    optimizer_name: str,
    device: str,
    publish: bool = True,
) -> dict[str, Any]:
    """Measure 32 independent nominal-LR shadow proposals for one arm."""

    row = _row_by_id(study, row_id)
    optimizer_contract = _optimizer_contract(study, optimizer_name)
    optimizer_key = _optimizer_name_lower(optimizer_contract)
    rates = {name: 1.0 for name in PARAMETER_NAMES}
    runtime = build_model_runtime(
        study,
        row,
        device=device,
        initialization_checkpoint=assets.checkpoint_path,
        learning_rate=rates,
        optimizer_contract=optimizer_contract,
    )
    parameters = _parameter_by_name(runtime.parameters)
    initial_rms_by_weight = {
        name: _rms(parameters[name].state) for name in WEIGHT_NAMES
    }
    tensor_before_all = parameter_tensor_digest(runtime.parameters)
    optimizer_before_all = optimizer_state_digest(runtime.optimizer)
    proposal_units: dict[str, list[float]] = {
        name: [] for name in PARAMETER_NAMES
    }
    records: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(assets.probe_batches):
        parameter_before = parameter_tensor_digest(runtime.parameters)
        optimizer_before = optimizer_state_digest(runtime.optimizer)
        result = training_step(
            runtime,
            batch,
            learning_rate=rates,
            restore=True,
        )
        parameter_after = parameter_tensor_digest(runtime.parameters)
        optimizer_after = optimizer_state_digest(runtime.optimizer)
        if parameter_after != parameter_before:
            raise BoundaryRuntimeError(
                "Expected every probe shadow proposal to restore all parameter "
                f"bytes exactly. Provided batch index: {batch_index}."
            )
        if optimizer_after != optimizer_before:
            raise BoundaryRuntimeError(
                "Expected every probe shadow proposal to restore the complete "
                f"{optimizer_contract['name']} state exactly. "
                f"Provided batch index: {batch_index}."
            )
        transition_by_name = result.transition.by_name
        if set(transition_by_name) != set(PARAMETER_NAMES):
            raise BoundaryRuntimeError(
                "Expected each probe transition to cover all Conv2 parameters. "
                f"Provided value: {sorted(transition_by_name)!r}."
            )
        parameter_records: dict[str, Any] = {}
        for name in PARAMETER_NAMES:
            item = transition_by_name[name]
            attached_weight = BIAS_TO_WEIGHT.get(name, name)
            denominator = initial_rms_by_weight[attached_weight]
            unit = float(item.proposed_update_rms) / denominator
            unit = _finite(unit, f"nominal proposal unit for {name!r}", positive=True)
            proposal_units[name].append(unit)
            parameter_records[name] = {
                "gradient_rms": float(item.gradient_rms),
                "proposed_update_rms": float(item.proposed_update_rms),
                "proposal_over_initial_attached_weight_rms": unit,
                "projection_efficiency": float(item.projection_efficiency),
                "proposed_bound_crossing_fraction": (
                    item.proposed_bound_crossing_fraction
                ),
            }
        records.append(
            {
                "batch_index": batch_index,
                "source_indices": list(result.source_indices),
                "loss": result.loss,
                "accuracy": result.accuracy,
                "parameter_tensor_sha256_before": parameter_before,
                "parameter_tensor_sha256_after": parameter_after,
                "optimizer_state_sha256_before": optimizer_before,
                "optimizer_state_sha256_after": optimizer_after,
                "parameters": parameter_records,
            }
        )

    if parameter_tensor_digest(runtime.parameters) != tensor_before_all:
        raise BoundaryRuntimeError(
            "Expected all probe shadow proposals together to leave parameter "
            "tensors unchanged."
        )
    if optimizer_state_digest(runtime.optimizer) != optimizer_before_all:
        raise BoundaryRuntimeError(
            "Expected all probe shadow proposals together to leave the complete "
            "optimizer state unchanged."
        )
    weight_units = {
        name: _linear_quantile(proposal_units[name], 0.5)
        for name in WEIGHT_NAMES
    }
    bias_units = {
        name: _linear_quantile(proposal_units[name], 0.9)
        for name in BIAS_NAMES
    }
    payload = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "official_test_read": False,
        "row_id": row_id,
        "scheme": row["scheme"],
        "optimizer_name": optimizer_contract["name"],
        "optimizer_parameters": optimizer_contract,
        "nominal_learning_rate_by_parameter": rates,
        "normalization_batch_count": NORMALIZATION_BATCHES,
        "normalization_batch_order_sha256": assets.minibatches[
            "normalization_batch_order_sha256"
        ],
        "replay_batch_order_sha256": assets.minibatches[
            "replay_batch_order_sha256"
        ],
        "probe_minibatches_file_sha256": sha256_file(assets.minibatches_path),
        "initialization_checkpoint_sha256": assets.initialization[
            "checkpoint_sha256"
        ],
        "initialization_tensor_sha256": assets.initialization[
            "parameter_tensor_sha256"
        ],
        "initial_weight_rms_by_parameter": initial_rms_by_weight,
        "weight_unit_statistic": "median",
        "shadow_weight_unit_by_parameter": weight_units,
        "normalization_unit_by_weight": weight_units,
        "weight_normalization_role": (
            "authoritative_fresh_optimizer_normalization"
            if optimizer_key == "adam"
            else "verification_only_historical_sgd_units_remain_authoritative"
        ),
        "bias_unit_statistic": "q90",
        "bias_q90_unit_by_parameter": bias_units,
        "bias_weight_mapping": dict(BIAS_TO_WEIGHT),
        "proposal_unit_samples_by_parameter": proposal_units,
        "records": records,
        "restoration": {
            "parameter_tensor_sha256": tensor_before_all,
            "optimizer_state_sha256": optimizer_before_all,
            "verified_after_every_shadow_proposal": True,
        },
    }
    payload["probe_content_sha256"] = sha256_json(payload)
    if publish:
        path = (
            assets.study_dir
            / "probes"
            / f"{row['scheme']}--{optimizer_key}"
            / "probe.json"
        )
        _publish_immutable_json(path, payload)
        payload = dict(payload)
        payload["artifact_path"] = str(path)
        payload["artifact_sha256"] = sha256_file(path)
    return payload


def load_optimizer_probe(
    study_dir: str | Path,
    *,
    scheme: str,
    optimizer_name: str,
) -> dict[str, Any]:
    path = (
        Path(study_dir).expanduser().resolve()
        / "probes"
        / f"{scheme}--{optimizer_name.lower()}"
        / "probe.json"
    )
    value = read_json(path)
    if value.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise BoundaryRuntimeError(
            f"Expected {path} to use {PROBE_SCHEMA_VERSION!r}. "
            f"Provided value: {value.get('schema_version')!r}."
        )
    scientific = dict(value)
    observed_content_hash = scientific.pop("probe_content_sha256", None)
    if observed_content_hash != sha256_json(scientific):
        raise BoundaryRuntimeError(
            f"Expected optimizer probe content hash to verify exactly: {path}."
        )
    value["artifact_path"] = str(path)
    value["artifact_sha256"] = sha256_file(path)
    return value


def compose_effective_optimizer_probe(
    measured_probe: Mapping[str, Any],
    *,
    source_weight_units_by_parameter: Mapping[str, float] | None = None,
    source_weight_probe_sha256: str | None = None,
) -> dict[str, Any]:
    """Compose authoritative weight and bias-normalization provenance.

    Adam takes both statistics from its fresh 32-shadow artifact.  SGD takes
    only the bias Q90 from the fresh shadow artifact: its weight units and
    normalization probe hash remain those frozen in the hash-verified v5/v6
    source run specs.
    """

    probe = copy.deepcopy(dict(measured_probe))
    optimizer_name = str(probe.get("optimizer_name", "")).lower()
    if optimizer_name not in {"sgd", "adam"}:
        raise ValueError(
            "Expected measured_probe.optimizer_name to be SGD or Adam. "
            f"Provided value: {probe.get('optimizer_name')!r}."
        )
    artifact_hash = _sha256(
        probe.get("artifact_sha256"), "measured_probe.artifact_sha256"
    )
    observed = probe.get("shadow_weight_unit_by_parameter") or probe.get(
        "normalization_unit_by_weight"
    )
    if not isinstance(observed, Mapping) or set(observed) != set(WEIGHT_NAMES):
        raise ValueError(
            "Expected measured probe shadow weight units for all Conv2 weights. "
            f"Provided value: {observed!r}."
        )
    observed_units = {
        name: _finite(
            observed[name], f"measured shadow weight unit {name!r}", positive=True
        )
        for name in WEIGHT_NAMES
    }
    if optimizer_name == "adam":
        if (
            source_weight_units_by_parameter is not None
            or source_weight_probe_sha256 is not None
        ):
            raise ValueError(
                "Expected Adam normalization not to receive historical SGD "
                "weight units or probe hashes."
            )
        effective_units = observed_units
        weight_probe_hash = artifact_hash
        source_mode = "fresh_adam_32_shadow_proposals"
    else:
        if (
            not isinstance(source_weight_units_by_parameter, Mapping)
            or set(source_weight_units_by_parameter) != set(WEIGHT_NAMES)
        ):
            raise ValueError(
                "Expected historical SGD source weight units for exactly "
                f"{list(WEIGHT_NAMES)!r}. "
                f"Provided value: {source_weight_units_by_parameter!r}."
            )
        weight_probe_hash = _sha256(
            source_weight_probe_sha256, "source_weight_probe_sha256"
        )
        effective_units = {
            name: _finite(
                source_weight_units_by_parameter[name],
                f"historical SGD weight unit {name!r}",
                positive=True,
            )
            for name in WEIGHT_NAMES
        }
        source_mode = "hash_verified_v5_v6_source_run_spec"

    probe["shadow_weight_unit_by_parameter"] = observed_units
    probe["normalization_unit_by_weight"] = effective_units
    probe["weight_normalization_probe_sha256"] = weight_probe_hash
    probe["bias_q90_probe_sha256"] = artifact_hash
    probe["weight_normalization_source"] = source_mode
    probe["shadow_vs_authoritative_weight_unit_diagnostics"] = {
        name: {
            "absolute_difference": observed_units[name] - effective_units[name],
            "ratio": observed_units[name] / effective_units[name],
        }
        for name in WEIGHT_NAMES
    }
    probe["effective_probe_composed"] = True
    return probe


def derive_raw_learning_rates(
    probe: Mapping[str, Any],
    *,
    rho_conv: float,
    rho_dense: float,
    bias_policy: str,
    bias_rho_cap: float = BIAS_RHO_CAP,
) -> dict[str, float]:
    """Derive the explicit five-parameter LR vector from a frozen probe."""

    conv_target = _finite(rho_conv, "rho_conv", positive=True)
    dense_target = _finite(rho_dense, "rho_dense", positive=True)
    if bias_policy not in {"attached", "capped"}:
        raise ValueError(
            "Expected bias_policy to be 'attached' or 'capped'. "
            f"Provided value: {bias_policy!r}."
        )
    if (
        str(probe.get("optimizer_name", "")).lower() == "sgd"
        and probe.get("effective_probe_composed") is not True
    ):
        raise ValueError(
            "Expected SGD learning-rate derivation to use a composed effective "
            "probe with historical source weight units and fresh bias Q90."
        )
    units = probe.get("normalization_unit_by_weight")
    if not isinstance(units, Mapping) or set(units) != set(WEIGHT_NAMES):
        raise ValueError(
            "Expected probe.normalization_unit_by_weight to contain exactly "
            f"{list(WEIGHT_NAMES)!r}. Provided value: {units!r}."
        )
    rates = {
        name: (
            conv_target if name.startswith("ConvWeight_") else dense_target
        )
        / _finite(units[name], f"normalization unit {name!r}", positive=True)
        for name in WEIGHT_NAMES
    }
    bias_units = probe.get("bias_q90_unit_by_parameter")
    if not isinstance(bias_units, Mapping) or set(bias_units) != set(BIAS_NAMES):
        raise ValueError(
            "Expected probe.bias_q90_unit_by_parameter to contain exactly "
            f"{list(BIAS_NAMES)!r}. Provided value: {bias_units!r}."
        )
    cap = _finite(bias_rho_cap, "bias_rho_cap", positive=True)
    for bias, weight in BIAS_TO_WEIGHT.items():
        attached = rates[weight]
        rates[bias] = (
            attached
            if bias_policy == "attached"
            else min(
                attached,
                cap
                / _finite(
                    bias_units[bias],
                    f"bias Q90 normalization unit {bias!r}",
                    positive=True,
                ),
            )
        )
        if rates[bias] > attached:
            raise BoundaryRuntimeError(
                f"Expected capped LR for {bias!r} not to exceed {weight!r}."
            )
    return {name: float(rates[name]) for name in PARAMETER_NAMES}


def _effective_probe_for_entry(
    measured_probe: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a measured artifact to the entry's authoritative normalization."""

    optimizer_name = str(entry["optimizer"]).lower()
    if optimizer_name == "adam":
        effective = compose_effective_optimizer_probe(measured_probe)
    elif optimizer_name == "sgd":
        units = entry.get("normalization_unit_by_weight") or entry.get(
            "weight_units_by_parameter"
        )
        effective = compose_effective_optimizer_probe(
            measured_probe,
            source_weight_units_by_parameter=units,
            source_weight_probe_sha256=entry.get("probe_sha256"),
        )
    else:
        raise ValueError(
            "Expected entry.optimizer to be 'sgd' or 'adam'. "
            f"Provided value: {entry['optimizer']!r}."
        )
    expected_weight_hash = _sha256(
        entry.get("probe_sha256"), "entry.probe_sha256"
    )
    if effective["weight_normalization_probe_sha256"] != expected_weight_hash:
        raise BoundaryRuntimeError(
            "Expected entry.probe_sha256 to bind the authoritative weight "
            "normalization probe. "
            f"Provided value: entry={expected_weight_hash!r}, effective="
            f"{effective['weight_normalization_probe_sha256']!r}."
        )
    expected_bias_hash = entry.get("bias_q90_probe_sha256")
    if expected_bias_hash is not None and effective[
        "bias_q90_probe_sha256"
    ] != _sha256(expected_bias_hash, "entry.bias_q90_probe_sha256"):
        raise BoundaryRuntimeError(
            "Expected entry.bias_q90_probe_sha256 to bind the fresh shadow "
            "artifact used for bias Q90."
        )
    return effective


def _hidden_saturation(runtime: Any, batch: Sequence[Any], *, v_off: float) -> list[float]:
    import torch

    images = batch[0].to(runtime.device)
    runtime.network.set_input(images, reset=True)
    with torch.no_grad():
        runtime.minimizer_inference.compute_equilibrium()
    values: list[float] = []
    for layer in runtime.energy_fn.layers()[1:-1]:
        state = layer.state.detach()
        if not bool(torch.isfinite(state).all().item()):
            count = int((~torch.isfinite(state)).sum().item())
            raise LRStudyNumericalError(
                "Expected hidden-layer replay state to contain only finite "
                f"values. Provided value: {count} non-finite element(s)."
            )
        outside = (state < -float(v_off)) | (state > float(v_off))
        values.append(float(outside.to(dtype=torch.float64).mean().item()))
    if len(values) != 2:
        raise BoundaryRuntimeError(
            "Expected exactly two hidden-layer saturation values for Conv2. "
            f"Provided value: {values!r}."
        )
    return values


def _displacement(
    runtime: Any,
    initial_states: Mapping[str, Any],
    initial_weight_rms: Mapping[str, float],
) -> dict[str, float]:
    parameters = _parameter_by_name(runtime.parameters)
    result: dict[str, float] = {}
    for name in PARAMETER_NAMES:
        denominator_name = BIAS_TO_WEIGHT.get(name, name)
        if denominator_name not in initial_weight_rms:
            raise BoundaryRuntimeError(
                f"Expected initial weight RMS for {denominator_name!r}."
            )
        delta = parameters[name].state.detach() - initial_states[name].to(
            parameters[name].state.device
        )
        result[name] = _rms(delta) / float(initial_weight_rms[denominator_name])
    return result


def replay_shadow_updates(
    runtime: Any,
    replay_batches: Sequence[Any],
    *,
    learning_rates_by_parameter: Mapping[str, float],
    initial_states: Mapping[str, Any],
    initial_weight_rms: Mapping[str, float],
    checkpoint_label: str,
    v_off: float,
) -> dict[str, Any]:
    """Replay eight batches and prove exact parameter/optimizer restoration."""

    if len(replay_batches) != REPLAY_BATCHES:
        raise ValueError(
            f"Expected exactly {REPLAY_BATCHES} replay minibatches. "
            f"Provided value: {len(replay_batches)}."
        )
    weight_rhos: dict[str, list[float]] = {name: [] for name in WEIGHT_NAMES}
    bias_rhos: dict[str, list[float]] = {name: [] for name in BIAS_NAMES}
    bias_ratios: dict[str, list[float]] = {name: [] for name in BIAS_NAMES}
    projection: dict[str, list[float]] = {name: [] for name in WEIGHT_NAMES}
    saturation: list[list[float]] = []
    batch_records: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(replay_batches):
        saturation_values = _hidden_saturation(runtime, batch, v_off=v_off)
        parameter_before = parameter_tensor_digest(runtime.parameters)
        optimizer_before = optimizer_state_digest(runtime.optimizer)
        result = training_step(
            runtime,
            batch,
            learning_rate=learning_rates_by_parameter,
            restore=True,
        )
        parameter_after = parameter_tensor_digest(runtime.parameters)
        optimizer_after = optimizer_state_digest(runtime.optimizer)
        if parameter_after != parameter_before or optimizer_after != optimizer_before:
            raise BoundaryRuntimeError(
                "Expected epoch replay to restore parameters and complete "
                f"optimizer state exactly at {checkpoint_label!r}, "
                f"batch {batch_index}."
            )
        by_name = result.transition.by_name
        for weight in WEIGHT_NAMES:
            rho = float(by_name[weight].proposed_update_rms) / float(
                initial_weight_rms[weight]
            )
            weight_rhos[weight].append(rho)
            projection[weight].append(float(by_name[weight].projection_efficiency))
        for bias, weight in BIAS_TO_WEIGHT.items():
            bias_rho = float(by_name[bias].proposed_update_rms) / float(
                initial_weight_rms[weight]
            )
            bias_rhos[bias].append(bias_rho)
            weight_rho = weight_rhos[weight][-1]
            bias_ratios[bias].append(
                bias_rho / weight_rho if weight_rho > 0.0 else 0.0
            )
        saturation.append(saturation_values)
        batch_records.append(
            {
                "batch_index": batch_index,
                "source_indices": list(result.source_indices),
                "loss": result.loss,
                "accuracy": result.accuracy,
                "weight_rho": {
                    name: weight_rhos[name][-1] for name in WEIGHT_NAMES
                },
                "bias_rho": {
                    name: bias_rhos[name][-1] for name in BIAS_NAMES
                },
                "bias_weight_step_ratio": {
                    name: bias_ratios[name][-1] for name in BIAS_NAMES
                },
                "projection_efficiency": {
                    name: projection[name][-1] for name in WEIGHT_NAMES
                },
                "hidden_saturation_by_layer": saturation_values,
                "parameter_tensor_sha256_before": parameter_before,
                "parameter_tensor_sha256_after": parameter_after,
                "optimizer_state_sha256_before": optimizer_before,
                "optimizer_state_sha256_after": optimizer_after,
            }
        )
    achieved_weight_rho_median = {
        name: _linear_quantile(values, 0.5)
        for name, values in weight_rhos.items()
    }
    bias_weight_step_ratio_median = {
        name: _linear_quantile(values, 0.5)
        for name, values in bias_ratios.items()
    }
    return {
        "checkpoint": checkpoint_label,
        "replay_batch_count": REPLAY_BATCHES,
        "achieved_weight_rho_median_by_parameter": achieved_weight_rho_median,
        # Compatibility aliases for the boundary plotter.  The statistic is
        # still explicitly recorded by the canonical ``*_median`` keys.
        "achieved_weight_rho_by_parameter": achieved_weight_rho_median,
        "achieved_weight_rho_q90_by_parameter": {
            name: _linear_quantile(values, 0.9)
            for name, values in weight_rhos.items()
        },
        "bias_rho_median_by_parameter": {
            name: _linear_quantile(values, 0.5)
            for name, values in bias_rhos.items()
        },
        "bias_weight_step_ratio_median_by_parameter": (
            bias_weight_step_ratio_median
        ),
        "bias_weight_step_ratio_by_bias": bias_weight_step_ratio_median,
        "projection_efficiency_median_by_parameter": {
            name: _linear_quantile(values, 0.5)
            for name, values in projection.items()
        },
        "hidden_saturation_mean_by_layer": [
            sum(values[layer] for values in saturation) / len(saturation)
            for layer in range(2)
        ],
        "displacement_over_initial_attached_weight_rms_by_parameter": _displacement(
            runtime, initial_states, initial_weight_rms
        ),
        "restoration_verified_after_every_shadow_step": True,
        "batches": batch_records,
    }


_CALIBRATION_SECOND_LAYER = {
    "conv2_baseline_v1_c1": 0.0,
    "conv2_ours_v4_c1": 0.04516103316326531,
    "conv2_legacy_v4_c0p25": 0.5926881128427933,
}


def _run_dataset(bundle: Any) -> dict[str, Any]:
    return {
        "name": "mnist",
        "input_shape": [2, 28, 28],
        "batch_size": 16,
        "max_batches": None,
        "normalization": {"mean": 0.1307, "std": 0.3081, "scale": 0.3},
        "affine": {
            "enabled": False,
            "preset": "ordinary_identity",
            "degrees": 0.0,
            "translate": [0.0, 0.0],
            "scale": [1.0, 1.0],
            "shear": 0.0,
            "seed": 1729,
            "interpolation": "bilinear",
            "fill": 0.0,
        },
        "train_shuffle_seed": 0,
        "validation": {
            "source": "mnist_train",
            "size": VALIDATION_EXAMPLES,
            "stratified": True,
            "samples_per_class": 500,
            "split_seed": 0,
            "batch_size": 128,
            "indices_sha256": bundle.validation_indices_hash,
            "official_test_enabled": False,
        },
    }


def _run_architecture() -> dict[str, Any]:
    return {
        "profile": "conv2",
        "channels": [64, 128],
        "kernel_sizes": [3, 3],
        "strides": [2, 2],
        "paddings": [1, 1],
        "pooling": {"mode": "none"},
        "output_dim": 20,
    }


def _run_model(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "resistive_conv",
        "non_linearity": "hard_sigmoid",
        "quadratic_diode_param": {},
        "exponential_diode_param": {},
        "hard_sigmoid_param": {"g_on": 100.0, "g_off": 0.0, "v_off": 4.0},
        "voltage_amp": float(row["voltage_amp"]),
        "current_amp": float(row["current_amp"]),
        "input_gain": float(row["input_gain"]),
        "weight_min": 0.0,
        "weight_max": 100.0,
        "weight_init_mode": "kaiming_uniform",
        "weight_gains": [1.0, 1.0, 1.0],
        "trainable_parameters": {
            "weights": True,
            "biases": True,
            "amplification": False,
            "hard_sigmoid_v_off": False,
        },
        "amplification_min": 1.0e-6,
        "amplification_max": None,
    }


def _run_solver(study: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    minimizer = copy.deepcopy(study["solver"]["minimizer"])
    return {
        "energy_mode": study["solver"]["energy_mode"],
        "inference_iterations": int(row["inference_iterations"]),
        "training_iterations": int(row["training_iterations"]),
        "minimizer": minimizer,
    }


def _run_training(
    optimizer_contract: Mapping[str, Any],
    learning_rates: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "algorithm": "BP",
        "optimizer": dict(optimizer_contract),
        "batch_state_policy": "reset_each_batch",
        "beta": 0.1,
        "epochs": EPOCHS,
        "schedule": {
            "name": "constant",
            "interval": "optimizer_step",
            "total_steps": TOTAL_STEPS,
            "scheduler_enabled": False,
        },
        "learning_rates_by_parameter": {
            name: float(learning_rates[name]) for name in PARAMETER_NAMES
        },
        "checkpoint_rule": "min_validation_loss",
        "pruning": {
            "enabled": False,
            "after_epoch": None,
            "min_best_validation_accuracy": None,
        },
        "diagnostics": {
            "enabled": True,
            "bounded_parameter_classes": ["ConvWeight", "DenseWeight"],
            "bias_global_gate": False,
            "range_epsilon": 1.0e-8,
            "projection_epsilon": 1.0e-12,
            "early_fraction": 1.0,
            "projection_efficiency_threshold": 0.5,
            "projection_persistence": 16,
            "occupancy_delta_threshold": 0.2,
            "occupancy_persistence": 16,
        },
    }


def _run_calibration(row: Mapping[str, Any]) -> dict[str, Any]:
    row_id = str(row["row_id"])
    return {
        "calibration_id": f"conv-hardsigmoid-sat30-20260718-{row_id}",
        "kind": "hard_sigmoid_saturation",
        "scope": "first_hidden_layer",
        "target_initial_saturation": 0.3,
        "measured_initial_saturation": 0.29999993771922834,
        "layer_measurements": [
            {"layer_index": 1, "measured_saturation": 0.29999993771922834},
            {
                "layer_index": 2,
                "measured_saturation": _CALIBRATION_SECOND_LAYER[row_id],
            },
        ],
        "sample_count": 256,
        "batch_size": 64,
        "model_seed": 0,
        "affine_seed": 1729,
        "settling_iterations": 64,
        "adaptive_equilibrium": False,
        "g_on": 100.0,
        "g_off": 0.0,
        "v_off": 4.0,
    }


def _study_id(study: Mapping[str, Any]) -> str:
    from .lr_optimizer_boundary_spec import OptimizerBoundaryStudySpec

    return OptimizerBoundaryStudySpec.from_dict(study).study_id


def build_boundary_run_payload(
    study: Mapping[str, Any],
    assets: BoundaryAssets,
    entry: Mapping[str, Any],
    probe: Mapping[str, Any],
    *,
    training_batch_order_sha256: str,
) -> dict[str, Any]:
    """Build and validate the standalone v7 RunSpec payload."""

    from .specs import OPTIMIZER_BOUNDARY_PROTOCOL_ID, RunSpec

    row = _row_by_id(study, str(entry["row_id"]))
    optimizer_key = str(entry["optimizer"]).lower()
    optimizer_contract = _optimizer_contract(study, optimizer_key)
    rates = {
        name: _finite(
            entry["raw_learning_rates_by_parameter"][name],
            f"entry raw LR {name!r}",
            positive=True,
        )
        for name in PARAMETER_NAMES
    }
    stage = str(entry["stage"])
    execution_mode = (
        "reuse" if str(entry.get("mode", "")).startswith("reuse") else "new"
    )
    parent_completion = entry.get("parent_entry_completion_sha256")
    if stage == "main_grid" and execution_mode == "new":
        parent_completion = None
    parent_decision = entry.get("parent_decision_sha256")
    parent_study = entry.get("parent_study_config_sha256") or entry.get(
        "parent_sha256"
    )
    probe_sha = entry.get("probe_sha256") or probe.get("artifact_sha256")
    code_sha = entry.get("code_fingerprint_sha256") or entry.get(
        "code_fingerprint"
    )
    normalization_batch_sha = assets.minibatches[
        "normalization_batch_order_sha256"
    ]
    replay_batch_sha = assets.minibatches["replay_batch_order_sha256"]
    checkpoint_relative = "initialization/conv2.pt"
    bias_policy = str(entry["bias_policy"])
    provenance = {
        "study_id": _study_id(study),
        "row_id": row["row_id"],
        "candidate_stage": stage,
        "execution_mode": execution_mode,
        "optimizer_name": optimizer_contract["name"],
        "optimizer_parameters": optimizer_contract,
        "rho_conv": float(entry["rho_conv"]),
        "rho_dense": float(entry["rho_dense"]),
        "normalization_unit_by_weight": {
            name: float(probe["normalization_unit_by_weight"][name])
            for name in WEIGHT_NAMES
        },
        "bias_q90_unit_by_parameter": {
            name: float(probe["bias_q90_unit_by_parameter"][name])
            for name in BIAS_NAMES
        },
        "bias_weight_mapping": dict(BIAS_TO_WEIGHT),
        "bias_lr_policy": bias_policy,
        "bias_rho_cap": BIAS_RHO_CAP if bias_policy == "capped" else None,
        "raw_learning_rates_by_parameter": rates,
        "probe_sha256": _sha256(probe_sha, "entry.probe_sha256"),
        "bias_q90_probe_sha256": _sha256(
            entry.get("bias_q90_probe_sha256")
            or probe.get("bias_q90_probe_sha256")
            or probe.get("artifact_sha256"),
            "entry.bias_q90_probe_sha256",
        ),
        "parent_study_config_sha256": _sha256(
            parent_study, "entry.parent_study_config_sha256"
        ),
        "parent_entry_completion_sha256": _sha256(
            parent_completion,
            "entry.parent_entry_completion_sha256",
            nullable=True,
        ),
        "parent_decision_sha256": _sha256(
            parent_decision,
            "entry.parent_decision_sha256",
            nullable=True,
        ),
        "split_sha256": assets.bundle.validation_indices_hash,
        "normalization_minibatches_sha256": normalization_batch_sha,
        "replay_minibatches_sha256": replay_batch_sha,
        "training_batch_order_sha256": _sha256(
            training_batch_order_sha256, "training_batch_order_sha256"
        ),
        "initialization_checkpoint_sha256": assets.initialization[
            "checkpoint_sha256"
        ],
        "initialization_tensor_sha256": assets.initialization[
            "parameter_tensor_sha256"
        ],
        "code_fingerprint_sha256": _sha256(
            code_sha, "entry.code_fingerprint_sha256"
        ),
        "official_test_read": False,
    }
    payload = {
        "schema_version": "mnist-conv-run/v7",
        "label": f"{entry['entry_id']}--seed0-optimizer-boundary-diagnostic",
        "seed": 0,
        "replicate_id": None,
        "protocol_id": OPTIMIZER_BOUNDARY_PROTOCOL_ID,
        "category": "diagnostic",
        "run": {
            "dataset": _run_dataset(assets.bundle),
            "architecture": _run_architecture(),
            "model": _run_model(row),
            "solver": _run_solver(study, row),
            "training": _run_training(optimizer_contract, rates),
            "initialization": {
                "checkpoint": {
                    "path": checkpoint_relative,
                    "sha256": assets.initialization["checkpoint_sha256"],
                    "format": "drn.function.parameters/v1",
                    "source_run_id": None,
                    "role": "initialization",
                }
            },
            "calibration": _run_calibration(row),
            "lr_provenance": provenance,
        },
    }
    return RunSpec.from_dict(payload).to_dict()


def _entry_contract(entry: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "entry_id",
        "row_id",
        "scheme",
        "optimizer",
        "stage",
        "bias_policy",
        "rho_conv",
        "rho_dense",
        "expected_steps",
        "mode",
    }
    missing = sorted(required - set(entry))
    if missing:
        raise ValueError(
            f"Expected boundary entry to contain {sorted(required)!r}. "
            f"Provided value: missing={missing!r}."
        )
    if str(entry["optimizer"]).lower() not in {"sgd", "adam"}:
        raise ValueError(
            "Expected entry.optimizer to be 'sgd' or 'adam'. "
            f"Provided value: {entry['optimizer']!r}."
        )
    _integer(entry["expected_steps"], "entry.expected_steps", expected=TOTAL_STEPS)
    if entry.get("official_test_read", False) is not False:
        raise ValueError(
            "Expected entry.official_test_read to be false. "
            f"Provided value: {entry.get('official_test_read')!r}."
        )
    return dict(entry)


def _candidate_minibatch_payload(
    *,
    entry_id: str,
    batches: Sequence[Sequence[int]],
    bundle: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "mnist-conv-lr-optimizer-training-minibatches/v1",
        "entry_id": entry_id,
        "official_test_read": False,
        "train_indices_sha256": bundle.train_indices_hash,
        "batch_count": len(batches),
        "batch_order_sha256": _batch_order_hash(batches),
        "batches": [list(batch) for batch in batches],
    }


def _allocate_attempt_output_dir(
    base_entry_dir: Path,
    entry: Mapping[str, Any],
) -> tuple[Path, str | None]:
    """Choose a new directory for an explicit restart without touching stale state."""

    resume_action = entry.get("resume_action")
    restart_actions = {
        "restart_from_initialization",
        "restart_fresh",
        "restart_incomplete_adam",
    }
    if resume_action not in restart_actions:
        base_entry_dir.mkdir(parents=True, exist_ok=True)
        return base_entry_dir, None
    if str(entry["optimizer"]).lower() != "adam":
        raise BoundaryRuntimeError(
            "Expected restart-from-initialization attempt directories only for "
            f"incomplete Adam entries. Provided optimizer: {entry['optimizer']!r}."
        )
    attempts_root = base_entry_dir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    explicit = entry.get("attempt_id")
    if explicit is not None:
        import re

        if (
            not isinstance(explicit, str)
            or re.fullmatch(r"attempt-[0-9]{6}", explicit) is None
        ):
            raise ValueError(
                "Expected entry.attempt_id to match 'attempt-NNNNNN'. "
                f"Provided value: {explicit!r}."
            )
        target = attempts_root / explicit
        try:
            target.mkdir()
        except FileExistsError as exc:
            raise BoundaryRuntimeError(
                "Expected an explicit restart attempt directory not to exist. "
                f"Provided value: {target}."
            ) from exc
        return target, target.relative_to(base_entry_dir).as_posix()
    for index in range(1, 1_000_000):
        target = attempts_root / f"attempt-{index:06d}"
        try:
            target.mkdir()
        except FileExistsError:
            continue
        return target, target.relative_to(base_entry_dir).as_posix()
    raise BoundaryRuntimeError(
        f"Expected an available Adam attempt directory under {attempts_root}."
    )


def execute_optimizer_boundary_entry(
    study: Mapping[str, Any],
    study_dir: str | Path,
    entry: Mapping[str, Any],
    *,
    data_root: str | Path,
    download: bool,
    device: str,
) -> dict[str, Any]:
    """Execute one new five-epoch entry and publish its scientific outputs.

    The surrounding runner owns ``result.json`` and writes ``complete.json``
    last.  This function never resumes Adam from model-only checkpoints: every
    invocation reconstructs parameters and optimizer state from the shared
    initialization.
    """

    entry = _entry_contract(entry)
    if str(entry["mode"]).startswith("reuse"):
        raise BoundaryRuntimeError(
            "Expected reused SGD cells to be handled by the hash-audit path, not "
            f"the numerical executor. Provided entry: {entry['entry_id']!r}."
        )
    expected_checkpoint = entry.get("checkpoint_sha256")
    initialization_source = entry.get("initialization_source")
    assets = prepare_boundary_assets(
        study,
        study_dir,
        data_root=data_root,
        download=download,
        device=device,
        initialization_source=initialization_source,
        expected_checkpoint_sha256=expected_checkpoint,
    )
    row = _row_by_id(study, str(entry["row_id"]))
    optimizer_key = str(entry["optimizer"]).lower()
    optimizer_contract = _optimizer_contract(study, optimizer_key)
    try:
        probe = load_optimizer_probe(
            assets.study_dir,
            scheme=str(entry["scheme"]),
            optimizer_name=optimizer_key,
        )
    except FileNotFoundError:
        probe = measure_optimizer_probe(
            study,
            assets,
            row_id=str(entry["row_id"]),
            optimizer_name=optimizer_key,
            device=device,
            publish=True,
        )
    probe = _effective_probe_for_entry(probe, entry)
    derived_rates = derive_raw_learning_rates(
        probe,
        rho_conv=float(entry["rho_conv"]),
        rho_dense=float(entry["rho_dense"]),
        bias_policy=str(entry["bias_policy"]),
    )
    provided_rates = entry.get("raw_learning_rates_by_parameter")
    if provided_rates is not None:
        normalized_provided = {
            name: _finite(
                provided_rates[name],
                f"entry.raw_learning_rates_by_parameter.{name}",
                positive=True,
            )
            for name in PARAMETER_NAMES
        }
        for name in PARAMETER_NAMES:
            if not math.isclose(
                normalized_provided[name],
                derived_rates[name],
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ):
                raise BoundaryRuntimeError(
                    "Expected entry raw learning rates to equal the "
                    f"optimizer-specific probe derivation for {name!r}. "
                    f"Provided value: entry={normalized_provided[name]!r}, "
                    f"derived={derived_rates[name]!r}."
                )
        rates = normalized_provided
    else:
        rates = derived_rates
        entry["raw_learning_rates_by_parameter"] = rates

    expected_batches_by_epoch = assets.bundle.train_batch_indices(
        num_epochs=EPOCHS
    )
    expected_training_batches = [
        batch for epoch in expected_batches_by_epoch for batch in epoch
    ]
    expected_training_hash = _batch_order_hash(expected_training_batches)
    if expected_training_hash != FROZEN_TRAINING_BATCH_ORDER_SHA256:
        raise BoundaryRuntimeError(
            "Expected the full five-epoch training order hash to match the "
            f"frozen protocol. Provided value: {expected_training_hash!r}."
        )
    run_payload = build_boundary_run_payload(
        study,
        assets,
        entry,
        probe,
        training_batch_order_sha256=expected_training_hash,
    )
    base_output_dir = (
        assets.study_dir
        / "stages"
        / str(entry["stage"])
        / "entries"
        / str(entry["entry_id"])
    )
    output_dir, attempt_relative_path = _allocate_attempt_output_dir(
        base_output_dir, entry
    )
    atomic_write_json(
        output_dir / "run_spec.v7.json",
        run_payload,
        canonical=True,
    )

    runtime = build_model_runtime(
        study,
        row,
        device=device,
        initialization_checkpoint=assets.checkpoint_path,
        learning_rate=rates,
        optimizer_contract=optimizer_contract,
    )
    parameters = _parameter_by_name(runtime.parameters)
    initial_states = {
        name: parameter.state.detach().cpu().clone()
        for name, parameter in parameters.items()
    }
    initial_weight_rms = {
        name: _rms(parameters[name].state) for name in WEIGHT_NAMES
    }
    if parameter_tensor_digest(runtime.parameters) != assets.initialization[
        "parameter_tensor_sha256"
    ]:
        raise BoundaryRuntimeError(
            "Expected candidate runtime to start from the verified shared "
            "initialization tensor hash."
        )
    require_finite_optimizer_state(
        runtime.optimizer,
        context=f"{optimizer_contract['name']} candidate initial state",
    )

    replay_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    observed_training_batches: list[tuple[int, ...]] = []
    projection_values: list[float] = []
    projection_by_weight: dict[str, list[float]] = {
        name: [] for name in WEIGHT_NAMES
    }
    completed_steps = 0
    safety_failure: dict[str, Any] | None = None
    best_epoch: int | None = None
    best_validation_loss = math.inf
    best_checkpoint = output_dir / "best_validation.pt"
    final_checkpoint = output_dir / "final.pt"

    try:
        initial_validation = evaluate_validation(
            runtime, assets.bundle.validation_loader
        )
        initial_validation.pop("source_indices")
        initial_validation.update({"epoch": 0, "step": 0})
        validation_records.append(initial_validation)
        initial_replay = replay_shadow_updates(
            runtime,
            assets.probe_batches[:REPLAY_BATCHES],
            learning_rates_by_parameter=rates,
            initial_states=initial_states,
            initial_weight_rms=initial_weight_rms,
            checkpoint_label="initialization",
            v_off=4.0,
        )
        initial_replay["epoch"] = 0
        initial_replay["step"] = 0
        initial_replay["validation_loss"] = initial_validation["loss"]
        initial_replay["validation_accuracy"] = initial_validation["accuracy"]
        replay_records.append(initial_replay)
    except (LRStudyNumericalError, OptimizerStateNumericalError, FloatingPointError) as exc:
        safety_failure = {
            "kind": "non_finite_initial_replay",
            "attempted_step": 0,
            "detail": str(exc),
        }

    assets.bundle.reset_train_shuffle()
    if safety_failure is None:
        for epoch in range(1, EPOCHS + 1):
            epoch_seen = 0
            epoch_loss = 0.0
            epoch_correct = 0.0
            batch_count = 0
            for batch_count, batch in enumerate(
                assets.bundle.train_loader, start=1
            ):
                attempted_step = completed_steps + 1
                expected_indices = tuple(
                    expected_batches_by_epoch[epoch - 1][batch_count - 1]
                )
                try:
                    result = training_step(
                        runtime,
                        batch,
                        learning_rate=rates,
                        restore=False,
                    )
                    require_finite_optimizer_state(
                        runtime.optimizer,
                        context=(
                            f"{optimizer_contract['name']} real training state "
                            f"after step {attempted_step}"
                        ),
                    )
                except (
                    LRStudyNumericalError,
                    OptimizerStateNumericalError,
                    FloatingPointError,
                ) as exc:
                    safety_failure = {
                        "kind": "non_finite_optimizer_transition",
                        "attempted_step": attempted_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_count,
                        "detail": str(exc),
                    }
                    step_rows.append(
                        {
                            "step": attempted_step,
                            "epoch": epoch,
                            "batch_in_epoch": batch_count,
                            "status": "safety_failure",
                            "loss": None,
                            "accuracy": None,
                            "sample_count": None,
                            "median_weight_projection_efficiency": None,
                        }
                    )
                    break
                if result.source_indices != expected_indices:
                    raise BoundaryRuntimeError(
                        "Expected candidate training minibatch order to match "
                        f"the frozen shuffle at epoch {epoch}, batch {batch_count}."
                    )
                completed_steps += 1
                observed_training_batches.append(result.source_indices)
                epoch_seen += result.sample_count
                epoch_loss += result.loss * result.sample_count
                epoch_correct += result.accuracy * result.sample_count
                by_name = result.transition.by_name
                step_projection = [
                    float(by_name[name].projection_efficiency)
                    for name in WEIGHT_NAMES
                ]
                for name, value in zip(WEIGHT_NAMES, step_projection):
                    projection_values.append(value)
                    projection_by_weight[name].append(value)
                step_rows.append(
                    {
                        "step": completed_steps,
                        "epoch": epoch,
                        "batch_in_epoch": batch_count,
                        "status": "complete",
                        "loss": result.loss,
                        "accuracy": result.accuracy,
                        "sample_count": result.sample_count,
                        "median_weight_projection_efficiency": _linear_quantile(
                            step_projection, 0.5
                        ),
                    }
                )
            if safety_failure is not None:
                break
            if batch_count != STEPS_PER_EPOCH or epoch_seen != TRAINING_EXAMPLES:
                raise BoundaryRuntimeError(
                    "Expected each epoch to contain exactly 3,438 batches and "
                    f"55,000 examples. Provided epoch {epoch}: "
                    f"batches={batch_count}, examples={epoch_seen}."
                )
            try:
                validation = evaluate_validation(
                    runtime, assets.bundle.validation_loader
                )
                validation.pop("source_indices")
                validation.update(
                    {
                        "epoch": epoch,
                        "step": completed_steps,
                        "train_loss": epoch_loss / epoch_seen,
                        "train_accuracy": epoch_correct / epoch_seen,
                    }
                )
                replay = replay_shadow_updates(
                    runtime,
                    assets.probe_batches[:REPLAY_BATCHES],
                    learning_rates_by_parameter=rates,
                    initial_states=initial_states,
                    initial_weight_rms=initial_weight_rms,
                    checkpoint_label=f"epoch_{epoch}",
                    v_off=4.0,
                )
            except (
                LRStudyNumericalError,
                OptimizerStateNumericalError,
                FloatingPointError,
            ) as exc:
                safety_failure = {
                    "kind": "non_finite_epoch_validation_or_replay",
                    "attempted_step": completed_steps,
                    "epoch": epoch,
                    "detail": str(exc),
                }
                break
            validation_records.append(validation)
            replay["epoch"] = epoch
            replay["step"] = completed_steps
            replay["validation_loss"] = validation["loss"]
            replay["validation_accuracy"] = validation["accuracy"]
            replay_records.append(replay)
            if float(validation["loss"]) < best_validation_loss:
                best_validation_loss = float(validation["loss"])
                best_epoch = epoch
                _save_checkpoint_atomic(runtime, best_checkpoint)

    _save_checkpoint_atomic(runtime, final_checkpoint)
    if best_epoch is None:
        _save_checkpoint_atomic(runtime, best_checkpoint)
    completed = safety_failure is None and completed_steps == TOTAL_STEPS
    if safety_failure is None and not completed:
        raise BoundaryRuntimeError(
            f"Expected a successful candidate to complete {TOTAL_STEPS} steps. "
            f"Provided value: {completed_steps}."
        )
    if completed and len(validation_records) != EPOCHS + 1:
        raise BoundaryRuntimeError(
            "Expected initialization plus five full validation records. "
            f"Provided value: {len(validation_records)}."
        )
    minibatch_payload = _candidate_minibatch_payload(
        entry_id=str(entry["entry_id"]),
        batches=observed_training_batches,
        bundle=assets.bundle,
    )
    if completed and minibatch_payload["batch_order_sha256"] != expected_training_hash:
        raise BoundaryRuntimeError(
            "Expected completed candidate minibatches to have the frozen "
            "five-epoch order hash."
        )
    replay_payload = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "entry_id": entry["entry_id"],
        "attempt_relative_path": attempt_relative_path,
        "official_test_read": False,
        "replay_batch_order_sha256": assets.minibatches[
            "replay_batch_order_sha256"
        ],
        "records": replay_records,
    }
    final_validation = validation_records[-1] if completed else None
    parent_sha = entry.get("parent_sha256") or entry.get(
        "parent_study_config_sha256"
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "entry_id": entry["entry_id"],
        "attempt_relative_path": attempt_relative_path,
        "row_id": entry["row_id"],
        "scheme": entry["scheme"],
        "optimizer": optimizer_key,
        "stage": entry["stage"],
        "rho_conv": float(entry["rho_conv"]),
        "rho_dense": float(entry["rho_dense"]),
        "bias_policy": entry["bias_policy"],
        "status": "complete" if completed else "safety_failure",
        "training_completed": completed,
        "completed_steps": completed_steps,
        "expected_steps": TOTAL_STEPS,
        "admissible": completed,
        "safety_failure": safety_failure,
        "official_test_read": False,
        "optimizer_parameters": optimizer_contract,
        "raw_learning_rates_by_parameter": rates,
        "normalization_unit_by_weight": probe[
            "normalization_unit_by_weight"
        ],
        "bias_q90_unit_by_parameter": probe[
            "bias_q90_unit_by_parameter"
        ],
        "probe_sha256": probe["weight_normalization_probe_sha256"],
        "bias_q90_probe_sha256": probe["bias_q90_probe_sha256"],
        "parent_sha256": _sha256(parent_sha, "entry.parent_sha256"),
        "parent_study_config_sha256": _sha256(
            entry.get("parent_study_config_sha256"),
            "entry.parent_study_config_sha256",
        ),
        "parent_entry_completion_sha256": _sha256(
            entry.get("parent_entry_completion_sha256"),
            "entry.parent_entry_completion_sha256",
            nullable=True,
        ),
        "parent_decision_sha256": _sha256(
            entry.get("parent_decision_sha256"),
            "entry.parent_decision_sha256",
            nullable=True,
        ),
        "minibatch_sha256": _sha256(
            entry.get("minibatch_sha256"),
            "entry.minibatch_sha256",
        ),
        "checkpoint_sha256": assets.initialization["checkpoint_sha256"],
        "code_fingerprint": _sha256(
            entry.get("code_fingerprint_sha256")
            or entry.get("code_fingerprint"),
            "entry.code_fingerprint",
        ),
        "normalization_minibatches_sha256": assets.minibatches[
            "normalization_batch_order_sha256"
        ],
        "replay_minibatches_sha256": assets.minibatches[
            "replay_batch_order_sha256"
        ],
        "training_batch_order_sha256": minibatch_payload[
            "batch_order_sha256"
        ],
        "initialization_tensor_sha256": assets.initialization[
            "parameter_tensor_sha256"
        ],
        "final_validation_loss": (
            None if final_validation is None else final_validation["loss"]
        ),
        "final_validation_accuracy": (
            None if final_validation is None else final_validation["accuracy"]
        ),
        "median_projection_efficiency": (
            None
            if not projection_values
            else _linear_quantile(projection_values, 0.5)
        ),
        "median_projection_efficiency_by_parameter": {
            name: _linear_quantile(values, 0.5)
            for name, values in projection_by_weight.items()
            if values
        },
        "best_validation_epoch": best_epoch,
        "validation_metrics": validation_records,
        "epoch_replay_records": replay_records,
        "best_checkpoint_sha256": sha256_file(best_checkpoint),
        "final_checkpoint_sha256": sha256_file(final_checkpoint),
        "final_parameter_tensor_sha256": parameter_tensor_digest(
            runtime.parameters
        ),
        "run_spec_v7_sha256": sha256_file(output_dir / "run_spec.v7.json"),
    }
    atomic_write_csv(
        output_dir / "step_log.csv",
        [
            "step",
            "epoch",
            "batch_in_epoch",
            "status",
            "loss",
            "accuracy",
            "sample_count",
            "median_weight_projection_efficiency",
        ],
        step_rows,
    )
    atomic_write_json(
        output_dir / "minibatches.json", minibatch_payload, canonical=True
    )
    atomic_write_json(
        output_dir / "validation.json", validation_records, canonical=True
    )
    atomic_write_json(
        output_dir / "epoch_replay.json", replay_payload, canonical=True
    )
    atomic_write_json(output_dir / "summary.json", result, canonical=True)
    return result


def execute_boundary_shadow_benchmark(
    study: Mapping[str, Any],
    study_dir: str | Path,
    entry: Mapping[str, Any],
    *,
    data_root: str | Path,
    device: str,
    warmup_steps: int = 32,
    measured_steps: int = 256,
    ready_path: str | Path | None = None,
    start_path: str | Path | None = None,
    start_timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    """Time one disposable persistent-optimizer child for packing benchmarks.

    "Shadow" here means that the whole runtime is discarded after measurement;
    these are real persistent optimizer transitions, not restored replay steps.
    The outer runner launches multiple children and samples device-wide memory.
    """

    import time
    import torch

    entry = _entry_contract(entry)
    warmup = _integer(warmup_steps, "warmup_steps")
    measured = _integer(measured_steps, "measured_steps")
    if warmup < 0 or measured <= 0:
        raise ValueError(
            "Expected warmup_steps >= 0 and measured_steps > 0. "
            f"Provided value: warmup_steps={warmup}, measured_steps={measured}."
        )
    if (ready_path is None) != (start_path is None):
        raise ValueError(
            "Expected ready_path and start_path to be provided together or both "
            f"omitted. Provided value: ready_path={ready_path!r}, "
            f"start_path={start_path!r}."
        )
    timeout = _finite(
        start_timeout_seconds, "start_timeout_seconds", positive=True
    )
    assets = prepare_boundary_assets(
        study,
        study_dir,
        data_root=data_root,
        download=False,
        device=device,
        expected_checkpoint_sha256=entry.get("checkpoint_sha256"),
    )
    probe = load_optimizer_probe(
        assets.study_dir,
        scheme=str(entry["scheme"]),
        optimizer_name=str(entry["optimizer"]),
    )
    probe = _effective_probe_for_entry(probe, entry)
    rates = derive_raw_learning_rates(
        probe,
        rho_conv=float(entry["rho_conv"]),
        rho_dense=float(entry["rho_dense"]),
        bias_policy=str(entry["bias_policy"]),
    )
    provided_rates = entry.get("raw_learning_rates_by_parameter")
    if provided_rates is not None:
        for name in PARAMETER_NAMES:
            provided = _finite(
                provided_rates[name],
                f"entry.raw_learning_rates_by_parameter.{name}",
                positive=True,
            )
            if not math.isclose(
                provided, rates[name], rel_tol=1.0e-12, abs_tol=1.0e-15
            ):
                raise BoundaryRuntimeError(
                    f"Expected benchmark LR for {name!r} to match its probe."
                )
        rates = {name: float(provided_rates[name]) for name in PARAMETER_NAMES}

    requested = torch.device(device)
    if requested.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(requested)
    row = _row_by_id(study, str(entry["row_id"]))
    optimizer_contract = _optimizer_contract(study, str(entry["optimizer"]))
    runtime = build_model_runtime(
        study,
        row,
        device=device,
        initialization_checkpoint=assets.checkpoint_path,
        learning_rate=rates,
        optimizer_contract=optimizer_contract,
    )
    assets.bundle.reset_train_shuffle()
    expected = assets.bundle.train_batch_indices(num_epochs=1)[0]
    loader_iterator = iter(assets.bundle.train_loader)
    completed = 0
    for batch_index in range(warmup):
        try:
            batch = next(loader_iterator)
        except StopIteration as exc:
            raise BoundaryRuntimeError(
                "Expected enough training batches for benchmark warmup. "
                f"Provided value: stopped after {batch_index}."
            ) from exc
        result = training_step(
            runtime,
            batch,
            learning_rate=rates,
            restore=False,
        )
        if result.source_indices != tuple(expected[batch_index]):
            raise BoundaryRuntimeError(
                "Expected benchmark minibatches to match the frozen train "
                f"stream at batch {batch_index}."
            )
        require_finite_optimizer_state(
            runtime.optimizer,
            context=f"{optimizer_contract['name']} benchmark state",
        )
        completed += 1

    if ready_path is not None and start_path is not None:
        ready = Path(ready_path).expanduser().resolve()
        start = Path(start_path).expanduser().resolve()
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.touch(exist_ok=False)
        wait_started = time.monotonic()
        while not start.is_file():
            if time.monotonic() - wait_started > timeout:
                raise TimeoutError(
                    "Expected the benchmark start barrier to appear within "
                    f"{timeout:g} seconds. Provided path: {start}."
                )
            time.sleep(0.05)

    if requested.type == "cuda":
        torch.cuda.synchronize(requested)
    started = time.perf_counter()
    measured_completed = 0
    for offset in range(measured):
        batch_index = warmup + offset
        try:
            batch = next(loader_iterator)
        except StopIteration as exc:
            raise BoundaryRuntimeError(
                "Expected enough training batches for benchmark measurement. "
                f"Provided value: stopped after {offset} measured steps."
            ) from exc
        result = training_step(
            runtime,
            batch,
            learning_rate=rates,
            restore=False,
        )
        if result.source_indices != tuple(expected[batch_index]):
            raise BoundaryRuntimeError(
                "Expected benchmark minibatches to match the frozen train "
                f"stream at batch {batch_index}."
            )
        require_finite_optimizer_state(
            runtime.optimizer,
            context=f"{optimizer_contract['name']} benchmark state",
        )
        completed += 1
        measured_completed += 1
    if requested.type == "cuda":
        torch.cuda.synchronize(requested)
    elapsed = time.perf_counter() - started
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise BoundaryRuntimeError(
            f"Expected positive finite benchmark elapsed time. Provided value: {elapsed!r}."
        )
    mib = float(1024**2)
    return {
        "schema_version": "mnist-conv-lr-optimizer-boundary-benchmark-child/v1",
        "entry_id": entry["entry_id"],
        "scheme": entry["scheme"],
        "optimizer": str(entry["optimizer"]).lower(),
        "warmup_steps": warmup,
        "measured_steps": measured,
        "barrier_synchronized": ready_path is not None,
        "successful_steps": completed,
        "measured_successful_steps": measured_completed,
        "elapsed_seconds": elapsed,
        "steps_per_second": measured_completed / elapsed,
        "peak_allocated_mib": (
            float(torch.cuda.max_memory_allocated(requested)) / mib
            if requested.type == "cuda"
            else 0.0
        ),
        "peak_reserved_mib": (
            float(torch.cuda.max_memory_reserved(requested)) / mib
            if requested.type == "cuda"
            else 0.0
        ),
        "optimizer_parameters": optimizer_contract,
        "raw_learning_rates_by_parameter": rates,
        "probe_sha256": probe["weight_normalization_probe_sha256"],
        "bias_q90_probe_sha256": probe["bias_q90_probe_sha256"],
        "checkpoint_sha256": assets.initialization["checkpoint_sha256"],
        "official_test_read": False,
    }


__all__ = [
    "BIAS_NAMES",
    "BIAS_RHO_CAP",
    "BIAS_TO_WEIGHT",
    "BoundaryAssets",
    "BoundaryRuntimeError",
    "NORMALIZATION_BATCHES",
    "PARAMETER_NAMES",
    "REPLAY_BATCHES",
    "TOTAL_STEPS",
    "WEIGHT_NAMES",
    "build_boundary_run_payload",
    "compose_effective_optimizer_probe",
    "derive_raw_learning_rates",
    "execute_boundary_shadow_benchmark",
    "execute_optimizer_boundary_entry",
    "load_verified_shared_initialization",
    "load_optimizer_probe",
    "measure_optimizer_probe",
    "optimizer_state_digest",
    "prepare_boundary_assets",
    "replay_shadow_updates",
    "verify_shared_initialization",
]
