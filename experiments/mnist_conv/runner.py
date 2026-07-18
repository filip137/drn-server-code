"""Crash-safe attempts and manifest-last publication for one Conv run."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import re
import socket
import sys
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backend import ExecutionContext, TrainingBackend, build_engine_config, train_with_mnist_backend
from .identity import canonical_json_bytes, normalize_code_provenance, run_fingerprint, sha256_file
from .io import atomic_write_csv, atomic_write_json, read_json, relative_posix
from .layout import ResultLayout
from .protocol import ensure_run_executable
from .specs import RunSpec


STATUS_SCHEMA_VERSION = "mnist-conv-status/v1"
BUNDLE_MANIFEST_SCHEMA_VERSION = "mnist-conv-result-manifest/v1"
HISTORY_COLUMNS = ["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy", "learning_rate"]
PUBLIC_FILES = [
    "config.resolved.json", "status.json", "metrics.json", "history.csv",
    "checkpoints/best.pt", "checkpoints/final.pt", "weights/best.npz",
    "weights/final.npz", "logs/train.log",
]
CANONICAL_METRIC_PATHS = {
    "config_path": "config.resolved.json",
    "best_checkpoint_path": "checkpoints/best.pt",
    "checkpoint_path": "checkpoints/final.pt",
    "weights_best_path": "weights/best.npz",
    "weights_final_path": "weights/final.npz",
    "history_path": "history.csv",
}


class RunBusyError(RuntimeError): pass
class RetryExhaustedError(RuntimeError): pass
class InvalidBundleError(RuntimeError): pass


@dataclass(frozen=True)
class ExecutionPolicy:
    max_attempts: int = 3
    retry_failed: bool = False
    retry_pruned: bool = False
    reclaim_stale_after_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_attempts < 1: raise ValueError(f"Expected max_attempts >= 1. Provided value: {self.max_attempts!r}.")
        if self.reclaim_stale_after_seconds is not None and self.reclaim_stale_after_seconds <= 0:
            raise ValueError(f"Expected reclaim_stale_after_seconds > 0. Provided value: {self.reclaim_stale_after_seconds!r}.")


@dataclass(frozen=True)
class RunExecution:
    run_id: str
    status: str
    run_dir: Path | None
    attempt_id: str | None


def _now() -> str: return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
def _attempt_id() -> str: return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}-{os.getpid()}-{uuid.uuid4().hex[:10]}"


def _status(run_id: str, attempt_id: str, state: str, message: str | None = None) -> dict[str, Any]:
    value = {"schema_version": STATUS_SCHEMA_VERSION, "run_id": run_id, "attempt_id": attempt_id, "state": state, "updated_at": _now(), "host": socket.gethostname(), "pid": os.getpid()}
    if message is not None: value["message"] = message
    return value


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _recover_stale_claim(path: Path, attempt_root: Path, threshold: float | None) -> None:
    if not path.exists(): return
    try:
        claim = read_json(path)
        age = (datetime.now(timezone.utc) - _parse_time(claim["updated_at"])).total_seconds()
    except Exception as exc:
        try:
            age = (
                datetime.now(timezone.utc)
                - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            ).total_seconds()
        except OSError as stat_exc:
            raise RunBusyError(
                f"Expected active claim to be readable or stat-able. Provided path: {path}; error: {stat_exc}."
            ) from stat_exc
        if threshold is None or age <= threshold:
            raise RunBusyError(
                "Expected active claim to contain valid status JSON, or an explicitly recoverable stale claim. "
                f"Provided path: {path}; age={age:.1f}s; error: {exc}."
            ) from exc
        recovered = attempt_root / "recovered_claims" / f"invalid-{uuid.uuid4().hex}.json"
        recovered.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, recovered)
        return
    if threshold is None or age <= threshold:
        raise RunBusyError(f"Expected no active claim. Provided claim age {age:.1f}s at {path}.")
    recovered = attempt_root / "recovered_claims" / f"{uuid.uuid4().hex}.json"
    recovered.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, recovered)
    old_attempt = claim.get("attempt_id")
    if old_attempt:
        old_status = attempt_root / str(old_attempt) / "status.json"
        if old_status.parent.exists(): atomic_write_json(old_status, _status(claim.get("run_id", "unknown"), str(old_attempt), "stale", f"claim recovered after {age:.1f}s"))


def _create_claim(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("x", encoding="utf-8") as handle:
            created = True
            json.dump(status, handle, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    except FileExistsError as exc: raise RunBusyError(f"Expected no active claim. Provided value: {path}.") from exc
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def _remove_owned_claim(path: Path, attempt_id: str) -> None:
    """Remove this attempt's claim without deleting a replacement claimant."""

    try:
        claim = read_json(path)
    except FileNotFoundError:
        return
    except Exception:
        # An active process only writes complete claim JSON atomically.  An
        # unreadable value can therefore no longer be proven to belong to us.
        return
    if claim.get("attempt_id") != attempt_id:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


class _Heartbeat:
    def __init__(self, claim: Path, run_id: str, attempt_id: str, interval: float = 30.0):
        self.claim, self.run_id, self.attempt_id, self.interval = claim, run_id, attempt_id, interval
        self.stop_event = threading.Event(); self.thread = threading.Thread(target=self._run, daemon=True)
    def start(self) -> None: self.thread.start()
    def stop(self) -> None: self.stop_event.set(); self.thread.join(timeout=max(1.0, self.interval + 1.0))
    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            if not self.claim.exists(): return
            try:
                current = read_json(self.claim)
            except Exception:
                return
            if current.get("attempt_id") != self.attempt_id:
                return
            atomic_write_json(self.claim, _status(self.run_id, self.attempt_id, "running"))


def _latest_attempt_state(root: Path) -> tuple[str, str] | None:
    values = []
    for path in root.glob("*/status.json"):
        try: values.append((path.parent.name, str(read_json(path).get("state"))))
        except Exception: continue
    return sorted(values)[-1] if values else None


def _relativize(value: Any, root: Path) -> Any:
    if isinstance(value, dict): return {key: _relativize(item, root) for key, item in value.items()}
    if isinstance(value, list): return [_relativize(item, root) for item in value]
    if isinstance(value, str) and os.path.isabs(value):
        try: return Path(value).resolve().relative_to(root.resolve()).as_posix() or "."
        except ValueError as exc:
            raise InvalidBundleError(
                "Expected every absolute path in backend public JSON to point inside the staging bundle. "
                f"Provided value: {value!r}."
            ) from exc
    return value


def _is_absolute_serialized_path(value: str) -> bool:
    """Recognize both native and common cross-platform absolute paths."""

    return (
        os.path.isabs(value)
        or re.match(r"^[A-Za-z]:[\\/]", value) is not None
        or value.startswith("\\\\")
    )


def _reject_absolute_serialized_paths(value: Any, *, context: str, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_absolute_serialized_paths(
                item,
                context=context,
                location=f"{location}.{key}",
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_serialized_paths(
                item,
                context=context,
                location=f"{location}[{index}]",
            )
        return
    if isinstance(value, str) and _is_absolute_serialized_path(value):
        raise InvalidBundleError(
            f"Expected {context} to contain only relocatable paths. "
            f"Provided absolute path at {location}: {value!r}."
        )


def _validate_metrics_paths(bundle: Path, metrics: dict[str, Any]) -> None:
    """Require canonical metrics to reference only published artifacts."""

    _reject_absolute_serialized_paths(metrics, context="metrics.json")
    if "history_paths" in metrics:
        raise InvalidBundleError(
            "Expected canonical metrics.json to use history_path instead of deleted backend history arrays. "
            f"Provided value: {metrics['history_paths']!r}."
        )
    if metrics.get("run_dir") != ".":
        raise InvalidBundleError(
            "Expected canonical metrics run_dir to identify the relocatable bundle as '.'. "
            f"Provided value: {metrics.get('run_dir')!r}."
        )
    for key, expected in CANONICAL_METRIC_PATHS.items():
        observed = metrics.get(key)
        if observed != expected:
            raise InvalidBundleError(
                f"Expected canonical metrics path {key!r} to be {expected!r}. "
                f"Provided value: {observed!r}."
            )
        artifact = bundle / expected
        if not artifact.is_file():
            raise InvalidBundleError(
                f"Expected canonical metrics path {key!r} to reference a published file. "
                f"Provided value: {expected!r}."
            )


def _move_if_needed(source: Path, target: Path) -> None:
    if target.exists(): return
    if source.exists(): target.parent.mkdir(parents=True, exist_ok=True); os.replace(source, target)


def _normalize_backend_output(bundle: Path, spec: RunSpec, backend_result: Any) -> None:
    bundle.joinpath("checkpoints").mkdir(exist_ok=True); bundle.joinpath("weights").mkdir(exist_ok=True)
    bundle.joinpath("logs").mkdir(exist_ok=True); bundle.joinpath("logs/tensorboard").mkdir(exist_ok=True)
    _move_if_needed(bundle / "best_model.pt", bundle / "checkpoints" / "best.pt")
    _move_if_needed(bundle / "final_model.pt", bundle / "checkpoints" / "final.pt")
    _move_if_needed(bundle / "weights_best.npz", bundle / "weights" / "best.npz")
    _move_if_needed(bundle / "weights_final.npz", bundle / "weights" / "final.npz")
    for path in list(bundle.glob("events.out.tfevents.*")): os.replace(path, bundle / "logs/tensorboard" / path.name)
    legacy_tensorboard = bundle / "tensorboard"
    if legacy_tensorboard.is_dir():
        for path in legacy_tensorboard.iterdir():
            os.replace(path, bundle / "logs/tensorboard" / path.name)
        legacy_tensorboard.rmdir()
    # Backend JSON snapshots contain host-specific absolute paths.  The public
    # config and metrics already retain the scientific and result provenance,
    # so omit these operational duplicates from the relocatable bundle.
    for source in ("config.json", "config.used.json", "run_metadata.json", "engine_config.json"):
        (bundle / source).unlink(missing_ok=True)
    if not (bundle / "history.csv").exists():
        try:
            import numpy as np
            arrays = {name: np.load(bundle / filename) for name, filename in {
                "train_loss": "loss_train.npy", "test_loss": "loss_test.npy",
                "train_accuracy": "accuracy_train.npy", "test_accuracy": "accuracy_test.npy",
            }.items()}
        except Exception as exc: raise InvalidBundleError(f"Expected backend histories or history.csv. Provided error: {exc}.") from exc
        lengths = {len(value) for value in arrays.values()}
        if len(lengths) != 1: raise InvalidBundleError(f"Expected equal history lengths. Provided lengths: {sorted(lengths)!r}.")
        learning_rates = backend_result.get("learning_rate") if isinstance(backend_result, dict) else None
        history_length = next(iter(lengths))
        if not isinstance(learning_rates, list) or len(learning_rates) != history_length:
            raise InvalidBundleError(
                "Expected backend result learning_rate history with one explicit vector per epoch. "
                f"Provided value: {learning_rates!r}."
            )
        rows = []
        for index in range(history_length):
            learning_rate = learning_rates[index]
            if not isinstance(learning_rate, (list, tuple)):
                raise InvalidBundleError(
                    f"Expected learning-rate vector at epoch {index + 1}. Provided value: {learning_rate!r}."
                )
            rows.append({"epoch": index + 1, **{key: float(value[index]) for key, value in arrays.items()}, "learning_rate": json.dumps([float(item) for item in learning_rate], separators=(",", ":"))})
        atomic_write_csv(bundle / "history.csv", HISTORY_COLUMNS, rows)
        for filename in ("loss_train.npy", "loss_test.npy", "accuracy_train.npy", "accuracy_test.npy"):
            (bundle / filename).unlink(missing_ok=True)
    metrics_path = bundle / "metrics.json"
    if metrics_path.exists():
        metrics = _relativize(read_json(metrics_path), bundle)
        if not isinstance(metrics, dict):
            raise InvalidBundleError(
                f"Expected backend metrics.json to contain an object. Provided value: {metrics!r}."
            )
        # The numerical backend reports its transient config snapshot and four
        # NPY histories.  Those files are deliberately retired during bundle
        # normalization, so publish their canonical replacements instead.
        metrics.pop("history_paths", None)
        metrics["run_dir"] = "."
        metrics.update(CANONICAL_METRIC_PATHS)
        _validate_metrics_paths(bundle, metrics)
        atomic_write_json(metrics_path, metrics)
    _canonicalize_npz_schema(bundle / "checkpoints/best.pt", bundle / "weights/best.npz")
    _canonicalize_npz_schema(bundle / "checkpoints/final.pt", bundle / "weights/final.npz")


def _checkpoint_tensors(path: Path) -> tuple[list[Any], list[dict[str, Any]] | None]:
    if not path.is_file() or path.stat().st_size == 0: raise InvalidBundleError(f"Expected non-empty checkpoint. Provided value: {path}.")
    try:
        from model.function.interaction import load_function_checkpoint_artifact
        tensors, schema, _ = load_function_checkpoint_artifact(path, map_location="cpu")
    except Exception as exc: raise InvalidBundleError(f"Expected loadable torch checkpoint. Provided path: {path}; error: {exc}.") from exc
    if not tensors:
        raise InvalidBundleError(f"Expected a non-empty checkpoint tensor list. Provided value: {path}.")
    return tensors, schema


def _sanitized_names(schema: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    used: set[str] = set()
    for index, entry in enumerate(schema):
        base = "".join(character if character.isalnum() or character == "_" else "_" for character in str(entry["name"]).strip()).strip("_") or f"param_{index}"
        name, suffix = base, 1
        while name in used:
            name = f"{base}_{suffix}"; suffix += 1
        used.add(name); names.append(name)
    return names


def _canonicalize_npz_schema(checkpoint_path: Path, npz_path: Path) -> None:
    tensors, schema = _checkpoint_tensors(checkpoint_path)
    if schema is None:
        return
    try:
        import numpy as np
        with np.load(npz_path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        names = [str(item) for item in arrays["param_names"].tolist()]
        types = [str(item) for item in arrays["param_types"].tolist()]
        shapes = json.loads(str(arrays["param_shapes_json"].item()))
    except Exception as exc:
        raise InvalidBundleError(f"Expected complete NPZ parameter schema. Provided path: {npz_path}; error: {exc}.") from exc
    expected_names = _sanitized_names(schema)
    expected_short_types = [entry["type"].rsplit(".", 1)[-1] for entry in schema]
    if names != expected_names or types not in (expected_short_types, [entry["type"] for entry in schema]) or shapes != [entry["shape"] for entry in schema]:
        raise InvalidBundleError(
            "Expected checkpoint and NPZ name/type/shape schemas to agree. "
            f"Provided names={names!r}, types={types!r}, shapes={shapes!r}."
        )
    for name, tensor in zip(names, tensors):
        array = arrays.get(name)
        if array is None or tuple(array.shape) != tuple(tensor.shape) or str(array.dtype) != str(tensor.dtype).removeprefix("torch."):
            raise InvalidBundleError(f"Expected NPZ array {name!r} to match checkpoint shape/dtype. Provided value: {npz_path}.")
        expected = tensor.detach().cpu().numpy()
        if not np.array_equal(array, expected):
            raise InvalidBundleError(
                f"Expected NPZ array {name!r} values to exactly match the checkpoint tensor. "
                f"Provided value: {npz_path}."
            )
    arrays["param_types"] = np.asarray([entry["type"] for entry in schema])
    temporary = npz_path.with_name(f".{npz_path.name}.{uuid.uuid4().hex}.npz")
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, npz_path)
    finally:
        temporary.unlink(missing_ok=True)


def _npz_contents(path: Path) -> tuple[list[dict[str, Any]], list[Any]]:
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as data:
            required = {"param_names", "param_types", "param_shapes_json"}
            if not required <= set(data.files): raise InvalidBundleError(f"Expected NPZ schema fields {sorted(required)!r}. Provided value: {path}.")
            names = [str(item) for item in data["param_names"].tolist()]
            types = [str(item) for item in data["param_types"].tolist()]
            shapes = json.loads(str(data["param_shapes_json"].item()))
            if not names or len(names) != len(set(names)) or len(types) != len(names) or len(shapes) != len(names) or any(name not in data for name in names): raise InvalidBundleError(f"Expected a complete ordered NPZ schema. Provided value: {path}.")
            expected_fields = {
                "param_names", "param_types", "param_shapes_json", "metadata_json", *names,
            }
            if set(data.files) != expected_fields:
                raise InvalidBundleError(
                    "Expected NPZ contents to contain exactly the declared parameters and canonical metadata fields. "
                    f"Provided missing={sorted(expected_fields - set(data.files))!r}, "
                    f"extra={sorted(set(data.files) - expected_fields)!r} at {path}."
                )
            metadata = json.loads(str(data["metadata_json"].item()))
            if not isinstance(metadata, dict):
                raise InvalidBundleError(
                    f"Expected NPZ metadata_json to encode an object. Provided value: {metadata!r}."
                )
            _reject_absolute_serialized_paths(metadata, context=f"NPZ metadata_json at {path}")
            canonical_json_bytes(metadata)
            arrays = [data[name].copy() for name in names]
            if any(not np.isfinite(array).all() for array in arrays): raise InvalidBundleError(f"Expected finite NPZ arrays. Provided value: {path}.")
            result = []
            for name, type_name, shape, array in zip(names, types, shapes, arrays):
                if list(array.shape) != list(shape): raise InvalidBundleError(f"Expected NPZ declared and actual shapes to agree for {name!r}. Provided value: {path}.")
                result.append({"name": name, "type": type_name, "shape": list(shape), "dtype": str(array.dtype)})
            return result, arrays
    except InvalidBundleError: raise
    except Exception as exc: raise InvalidBundleError(f"Expected readable NPZ weights. Provided path: {path}; error: {exc}.") from exc


def _validate_checkpoint_npz_pair(checkpoint_path: Path, npz_path: Path) -> list[dict[str, Any]]:
    """Validate one checkpoint/NPZ pair, including exact parameter values."""

    tensors, checkpoint_schema = _checkpoint_tensors(checkpoint_path)
    npz_schema, arrays = _npz_contents(npz_path)
    if len(arrays) != len(tensors):
        raise InvalidBundleError(
            "Expected checkpoint and NPZ parameter counts to agree. "
            f"Provided checkpoint={len(tensors)}, NPZ={len(arrays)}."
        )

    if checkpoint_schema is None:
        expected_schema = [
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}
            for tensor in tensors
        ]
        actual_schema = [
            {"shape": entry["shape"], "dtype": entry["dtype"]}
            for entry in npz_schema
        ]
    else:
        expected_schema = [
            {
                "name": name,
                "type": entry["type"],
                "shape": entry["shape"],
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
            for name, entry, tensor in zip(
                _sanitized_names(checkpoint_schema), checkpoint_schema, tensors
            )
        ]
        actual_schema = npz_schema
    if actual_schema != expected_schema:
        raise InvalidBundleError(
            "Expected checkpoint and NPZ name, type, shape, dtype, and count schemas to agree. "
            f"Provided NPZ={actual_schema!r}, checkpoint={expected_schema!r}."
        )

    try:
        import numpy as np
        for index, (array, tensor) in enumerate(zip(arrays, tensors)):
            expected = tensor.detach().cpu().numpy()
            if not np.array_equal(array, expected):
                name = npz_schema[index].get("name", f"parameter {index}")
                raise InvalidBundleError(
                    f"Expected NPZ array {name!r} values to exactly match the checkpoint tensor. "
                    f"Provided value: {npz_path}."
                )
    except InvalidBundleError:
        raise
    except Exception as exc:
        raise InvalidBundleError(
            f"Expected checkpoint tensors to be exactly comparable with NPZ arrays. "
            f"Provided checkpoint={checkpoint_path}, NPZ={npz_path}; error: {exc}."
        ) from exc
    return npz_schema


def _validate_history(
    path: Path,
    epochs: int,
    initial_learning_rate: list[float],
    lr_decay: float,
) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    except OSError as exc: raise InvalidBundleError(f"Expected readable history.csv. Provided error: {exc}.") from exc
    if not rows or set(rows[0]) != set(HISTORY_COLUMNS): raise InvalidBundleError(f"Expected history columns {HISTORY_COLUMNS!r}. Provided value: {list(rows[0]) if rows else None!r}.")
    if len(rows) != epochs: raise InvalidBundleError(f"Expected {epochs} history rows. Provided value: {len(rows)}.")
    parsed = []
    for index, row in enumerate(rows, 1):
        if int(row["epoch"]) != index: raise InvalidBundleError(f"Expected contiguous epoch {index}. Provided value: {row['epoch']!r}.")
        values = {key: float(row[key]) for key in ("train_loss", "train_accuracy", "test_loss", "test_accuracy")}
        if not all(math.isfinite(item) for item in values.values()): raise InvalidBundleError(f"Expected finite history at epoch {index}. Provided value: {values!r}.")
        if values["train_loss"] < 0 or values["test_loss"] < 0:
            raise InvalidBundleError(f"Expected non-negative losses at epoch {index}. Provided value: {values!r}.")
        if not 0 <= values["train_accuracy"] <= 1 or not 0 <= values["test_accuracy"] <= 1:
            raise InvalidBundleError(f"Expected accuracies in [0, 1] at epoch {index}. Provided value: {values!r}.")
        lr = json.loads(row["learning_rate"])
        if (
            not isinstance(lr, list)
            or len(lr) != len(initial_learning_rate)
            or any(isinstance(item, bool) or not math.isfinite(float(item)) for item in lr)
        ):
            raise InvalidBundleError(f"Expected a complete finite learning-rate vector. Provided value: {lr!r}.")
        expected_lr = [value * lr_decay ** (index - 1) for value in initial_learning_rate]
        if any(
            not math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=0.0)
            for observed, expected in zip(lr, expected_lr)
        ):
            raise InvalidBundleError(
                f"Expected epoch {index} learning rates {expected_lr!r}. Provided value: {lr!r}."
            )
        parsed.append({"epoch": index, **values, "learning_rate": lr})
    return parsed


def _validate_scientific_artifacts(bundle: Path, spec: RunSpec) -> dict[str, Any]:
    for relative in PUBLIC_FILES:
        if not (bundle / relative).is_file(): raise InvalidBundleError(f"Expected public artifact {relative!r}. Provided bundle: {bundle}.")
    if not (bundle / "logs/tensorboard").is_dir(): raise InvalidBundleError(f"Expected logs/tensorboard directory. Provided bundle: {bundle}.")
    training = spec.data["run"]["training"]
    rows = _validate_history(
        bundle / "history.csv",
        training["epochs"],
        training["learning_rate"],
        training["lr_decay"],
    )
    metrics = read_json(bundle / "metrics.json")
    required_metrics = {"best_epoch", "best_test_accuracy", "final_test_accuracy", "final_train_loss", "final_test_loss"}
    if not isinstance(metrics, dict) or not required_metrics <= set(metrics): raise InvalidBundleError(f"Expected metrics fields {sorted(required_metrics)!r}. Provided value: {metrics!r}.")
    _validate_metrics_paths(bundle, metrics)
    for key in required_metrics - {"best_epoch"}:
        if isinstance(metrics[key], bool) or not isinstance(metrics[key], (int, float)) or not math.isfinite(float(metrics[key])):
            raise InvalidBundleError(f"Expected finite numeric metric {key}. Provided value: {metrics[key]!r}.")
    if type(metrics["best_epoch"]) is not int:
        raise InvalidBundleError(f"Expected best_epoch to be an integer. Provided value: {metrics['best_epoch']!r}.")
    best_epoch = metrics["best_epoch"]; expected_best = max(rows, key=lambda row: row["test_accuracy"])
    if best_epoch != expected_best["epoch"] or not math.isclose(float(metrics["best_test_accuracy"]), expected_best["test_accuracy"], rel_tol=1e-9, abs_tol=1e-12):
        raise InvalidBundleError(f"Expected best metric/epoch to match history. Provided best_epoch={best_epoch!r}, expected={expected_best!r}.")
    final_row = rows[-1]
    for metric_key, history_key in (
        ("final_test_accuracy", "test_accuracy"),
        ("final_train_loss", "train_loss"),
        ("final_test_loss", "test_loss"),
    ):
        if not math.isclose(float(metrics[metric_key]), final_row[history_key], rel_tol=1e-9, abs_tol=1e-12):
            raise InvalidBundleError(
                f"Expected {metric_key} to match final history. Provided metric={metrics[metric_key]!r}, history={final_row[history_key]!r}."
            )
    _, best_checkpoint_schema = _checkpoint_tensors(bundle / "checkpoints/best.pt")
    _, final_checkpoint_schema = _checkpoint_tensors(bundle / "checkpoints/final.pt")
    if final_checkpoint_schema != best_checkpoint_schema:
        raise InvalidBundleError("Expected best and final checkpoint schemas to agree exactly.")
    best_npz = _validate_checkpoint_npz_pair(
        bundle / "checkpoints/best.pt", bundle / "weights/best.npz"
    )
    final_npz = _validate_checkpoint_npz_pair(
        bundle / "checkpoints/final.pt", bundle / "weights/final.npz"
    )
    if best_npz != final_npz:
        raise InvalidBundleError(
            "Expected best and final NPZ parameter schemas to agree exactly. "
            f"Provided best={best_npz!r}, final={final_npz!r}."
        )
    return metrics


def _artifact_records(bundle: Path) -> list[dict[str, Any]]:
    if bundle.is_symlink():
        raise InvalidBundleError(f"Expected a real result bundle directory, not a symlink. Provided value: {bundle}.")
    records = []
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink(): raise InvalidBundleError(f"Expected regular artifacts, not symlinks. Provided value: {path}.")
        if path == bundle / "manifest.json" or not path.is_file(): continue
        records.append({"path": path.relative_to(bundle).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return records


def _write_bundle_manifest(bundle: Path, spec: RunSpec, run_id: str, attempt_id: str, code_provenance: dict[str, Any], metrics: dict[str, Any]) -> None:
    value = {"schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION, "run_id": run_id, "attempt_id": attempt_id, "state": "complete", "completed_at": _now(), "code_provenance": code_provenance, "config_sha256": sha256_file(bundle / "config.resolved.json"), "best_epoch": metrics["best_epoch"], "artifacts": _artifact_records(bundle)}
    atomic_write_json(bundle / "manifest.json", value, canonical=True)


def validate_bundle(run_dir: str | Path, expected_run_id: str | None = None) -> dict[str, Any]:
    bundle = Path(run_dir); manifest_path = bundle / "manifest.json"
    if bundle.is_symlink() or manifest_path.is_symlink():
        raise InvalidBundleError(f"Expected a real bundle and manifest, not symlinks. Provided bundle: {bundle}.")
    if not manifest_path.is_file(): raise InvalidBundleError(f"Expected manifest.json as completion marker. Provided bundle: {bundle}.")
    manifest = read_json(manifest_path)
    manifest_keys = {
        "schema_version", "run_id", "attempt_id", "state", "completed_at",
        "code_provenance", "config_sha256", "best_epoch", "artifacts",
    }
    if not isinstance(manifest, dict) or set(manifest) != manifest_keys:
        raise InvalidBundleError(
            f"Expected result manifest keys {sorted(manifest_keys)!r}. "
            f"Provided value: {sorted(manifest) if isinstance(manifest, dict) else manifest!r}."
        )
    if manifest["schema_version"] != BUNDLE_MANIFEST_SCHEMA_VERSION or manifest["state"] != "complete": raise InvalidBundleError(f"Expected complete v1 result manifest. Provided value: {manifest!r}.")
    if not isinstance(manifest["run_id"], str) or re.fullmatch(r"run_[0-9a-f]{64}", manifest["run_id"]) is None:
        raise InvalidBundleError(f"Expected a canonical run id in the manifest. Provided value: {manifest['run_id']!r}.")
    if not isinstance(manifest["attempt_id"], str) or not manifest["attempt_id"]:
        raise InvalidBundleError(f"Expected a non-empty attempt id in the manifest. Provided value: {manifest['attempt_id']!r}.")
    try:
        _parse_time(manifest["completed_at"])
    except Exception as exc:
        raise InvalidBundleError(f"Expected an ISO-8601 completion time. Provided value: {manifest['completed_at']!r}.") from exc
    if expected_run_id is not None and manifest.get("run_id") != expected_run_id: raise InvalidBundleError(f"Expected run id {expected_run_id!r}. Provided value: {manifest.get('run_id')!r}.")
    records = manifest["artifacts"]
    if not isinstance(records, list) or not records:
        raise InvalidBundleError(f"Expected a non-empty artifact record list. Provided value: {records!r}.")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
            raise InvalidBundleError(
                "Expected each artifact record to contain exactly path, sha256, and bytes. "
                f"Provided value: {record!r}."
            )
        if (
            not isinstance(record["path"], str)
            or "\\" in record["path"]
        ):
            raise InvalidBundleError(f"Expected a portable relative artifact path. Provided value: {record['path']!r}.")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise InvalidBundleError(f"Expected a portable relative artifact path. Provided value: {record['path']!r}.")
        if not isinstance(record["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
            raise InvalidBundleError(f"Expected a lowercase artifact SHA-256. Provided value: {record['sha256']!r}.")
        if type(record["bytes"]) is not int or record["bytes"] < 0:
            raise InvalidBundleError(f"Expected a non-negative artifact byte count. Provided value: {record['bytes']!r}.")
    actual_records = _artifact_records(bundle)
    if records != actual_records:
        raise InvalidBundleError(
            "Expected manifest artifact records to be complete, unique, sorted, and to exactly match every bundle file. "
            f"Provided records={records!r}; actual={actual_records!r}."
        )
    status = read_json(bundle / "status.json")
    status_keys = {
        "schema_version", "run_id", "attempt_id", "state", "updated_at", "host", "pid",
    }
    if not isinstance(status, dict) or set(status) != status_keys:
        raise InvalidBundleError(
            f"Expected completed bundle status keys {sorted(status_keys)!r}. Provided value: {status!r}."
        )
    if (
        status["schema_version"] != STATUS_SCHEMA_VERSION
        or status["run_id"] != manifest["run_id"]
        or status["attempt_id"] != manifest["attempt_id"]
        or status["state"] != "complete"
        or not isinstance(status["host"], str)
        or not status["host"]
        or type(status["pid"]) is not int
        or status["pid"] < 1
    ):
        raise InvalidBundleError(
            "Expected bundle status to identify the same completed run and attempt. "
            f"Provided value: {status!r}."
        )
    try:
        _parse_time(status["updated_at"])
    except Exception as exc:
        raise InvalidBundleError(
            f"Expected bundle status updated_at to be ISO-8601. Provided value: {status['updated_at']!r}."
        ) from exc
    spec = RunSpec.from_dict(read_json(bundle / "config.resolved.json"))
    observed_config_sha = sha256_file(bundle / "config.resolved.json")
    if manifest["config_sha256"] != observed_config_sha:
        raise InvalidBundleError(
            f"Expected manifest config SHA-256 {observed_config_sha!r}. "
            f"Provided value: {manifest['config_sha256']!r}."
        )
    provenance = normalize_code_provenance(manifest["code_provenance"])
    if run_fingerprint(spec, provenance) != manifest["run_id"]: raise InvalidBundleError(f"Expected config/code identity to match manifest. Provided bundle: {bundle}.")
    metrics = _validate_scientific_artifacts(bundle, spec)
    if type(manifest["best_epoch"]) is not int or manifest["best_epoch"] != int(metrics["best_epoch"]):
        raise InvalidBundleError(
            f"Expected manifest best_epoch to match metrics. Provided manifest={manifest['best_epoch']!r}, "
            f"metrics={metrics['best_epoch']!r}."
        )
    return manifest


def _validate_initialization_checkpoint(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    layout: ResultLayout,
    target_run_id: str,
) -> None:
    """Validate an initialization artifact and any declared source-bundle link."""

    if checkpoint_path.is_symlink():
        raise InvalidBundleError(
            f"Expected initialization checkpoint to be a regular file, not a symlink. Provided value: {checkpoint_path}."
        )
    if not checkpoint_path.is_file():
        raise InvalidBundleError(f"Expected initialization checkpoint file to exist. Provided value: {checkpoint_path}.")
    observed_sha = sha256_file(checkpoint_path)
    if observed_sha != checkpoint["sha256"]:
        raise InvalidBundleError(
            f"Expected initialization checkpoint SHA-256 {checkpoint['sha256']!r}. "
            f"Provided value: {observed_sha!r} at {checkpoint_path}."
        )

    try:
        from model.function.interaction import (
            FUNCTION_CHECKPOINT_FORMAT,
            FUNCTION_CHECKPOINT_VERSION,
            load_function_checkpoint_artifact,
        )
        _, _, source_format = load_function_checkpoint_artifact(checkpoint_path, map_location="cpu")
    except Exception as exc:
        raise InvalidBundleError(
            f"Expected a structurally valid initialization checkpoint. Provided path: {checkpoint_path}; error: {exc}."
        ) from exc
    decoded_format = (
        f"{FUNCTION_CHECKPOINT_FORMAT}/v{FUNCTION_CHECKPOINT_VERSION}"
        if source_format == "versioned"
        else "legacy_tensor_list"
    )
    if checkpoint["format"] != decoded_format:
        raise InvalidBundleError(
            f"Expected initialization checkpoint format declaration {decoded_format!r}. "
            f"Provided value: {checkpoint['format']!r}."
        )

    role = checkpoint["role"]
    if role not in {"best", "final", "initialization"}:
        raise InvalidBundleError(
            "Expected initialization checkpoint role to be 'best', 'final', or 'initialization'. "
            f"Provided value: {role!r}."
        )
    source_run_id = checkpoint["source_run_id"]
    if source_run_id is None:
        try:
            checkpoint_path.relative_to(layout.runs_root)
        except ValueError:
            return
        raise InvalidBundleError(
            "Expected a checkpoint inside results/runs to declare its source_run_id. "
            f"Provided path: {checkpoint_path}."
        )
    if re.fullmatch(r"run_[0-9a-f]{64}", source_run_id) is None:
        raise InvalidBundleError(
            f"Expected source_run_id in canonical SHA-256 form. Provided value: {source_run_id!r}."
        )
    if source_run_id == target_run_id:
        raise InvalidBundleError(
            f"Expected initialization source_run_id to differ from the target run id. Provided value: {source_run_id!r}."
        )
    if role not in {"best", "final"}:
        raise InvalidBundleError(
            "Expected a source-run checkpoint role to be 'best' or 'final'. "
            f"Provided value: {role!r}."
        )
    try:
        source_dir = layout.find_run_dir(source_run_id)
    except (ValueError, RuntimeError) as exc:
        raise InvalidBundleError(
            f"Expected one canonical source run for initialization. Provided value: {source_run_id!r}; error: {exc}."
        ) from exc
    if source_dir is None:
        raise InvalidBundleError(
            f"Expected declared initialization source run to exist. Provided value: {source_run_id!r}."
        )
    validate_bundle(source_dir, source_run_id)
    expected_path = source_dir / "checkpoints" / f"{role}.pt"
    if checkpoint_path != expected_path:
        raise InvalidBundleError(
            f"Expected source-run role {role!r} at {expected_path}. Provided value: {checkpoint_path}."
        )


def _validate_pruned_best_artifacts(bundle: Path) -> None:
    """Require a self-consistent best checkpoint before recording pruning."""

    (bundle / "checkpoints").mkdir(exist_ok=True)
    (bundle / "weights").mkdir(exist_ok=True)
    checkpoint_path = bundle / "checkpoints/best.pt"
    weights_path = bundle / "weights/best.npz"
    _move_if_needed(bundle / "best_model.pt", checkpoint_path)
    _move_if_needed(bundle / "weights_best.npz", weights_path)
    if not checkpoint_path.is_file() or not weights_path.is_file():
        raise InvalidBundleError(
            "Expected a pruned backend to save both the current best checkpoint and matching NPZ weights "
            f"before returning. Provided checkpoint={checkpoint_path.is_file()}, weights={weights_path.is_file()}."
        )
    _canonicalize_npz_schema(checkpoint_path, weights_path)
    _validate_checkpoint_npz_pair(checkpoint_path, weights_path)


def execute_run(spec: RunSpec, context: ExecutionContext, layout: ResultLayout, code_provenance: dict[str, Any], *, backend: TrainingBackend | None = None, policy: ExecutionPolicy | None = None) -> RunExecution:
    ensure_run_executable(spec)
    policy = policy or ExecutionPolicy(); code_provenance = normalize_code_provenance(code_provenance)
    run_id, label = run_fingerprint(spec, code_provenance), spec.data["label"]
    published = layout.find_run_dir(run_id)
    if published is not None: validate_bundle(published, run_id); return RunExecution(run_id, "already_complete", published, None)
    final_dir = layout.run_dir(label, run_id)
    checkpoint = spec.data["run"]["initialization"]["checkpoint"]
    operational_context = context
    if checkpoint is not None:
        checkpoint_path = layout.root / checkpoint["path"]
        _validate_initialization_checkpoint(checkpoint_path, checkpoint, layout, run_id)
        operational_context = context.with_initialization_checkpoint(checkpoint_path)
    attempt_root = layout.attempt_root(run_id); attempt_root.mkdir(parents=True, exist_ok=True)
    _recover_stale_claim(layout.claim_path(run_id), attempt_root, policy.reclaim_stale_after_seconds)
    attempts = [path for path in attempt_root.iterdir() if path.is_dir() and path.name != "recovered_claims"]
    latest = _latest_attempt_state(attempt_root)
    if latest and latest[1] == "failed" and not policy.retry_failed: raise RetryExhaustedError(f"Expected retry_failed=true to retry {run_id}. Provided latest attempt: {latest!r}.")
    if latest and latest[1] == "pruned" and not policy.retry_pruned:
        return RunExecution(run_id, "pruned", None, latest[0])
    if len(attempts) >= policy.max_attempts: raise RetryExhaustedError(f"Expected fewer than {policy.max_attempts} attempts. Provided {len(attempts)} for {run_id}.")
    attempt_id = _attempt_id(); attempt_dir = layout.attempt_dir(run_id, attempt_id); bundle = attempt_dir / ".staging"
    running = _status(run_id, attempt_id, "running"); _create_claim(layout.claim_path(run_id), running)
    heartbeat: _Heartbeat | None = None
    heartbeat_started = False
    try:
        attempt_dir.mkdir(parents=True)
        bundle.mkdir()
        atomic_write_json(attempt_dir / "status.json", running)
        heartbeat = _Heartbeat(layout.claim_path(run_id), run_id, attempt_id)
        heartbeat.start()
        heartbeat_started = True
        atomic_write_json(bundle / "config.resolved.json", spec.to_dict(), canonical=True)
        engine_path = bundle / "engine_config.json"
        atomic_write_json(engine_path, build_engine_config(spec, operational_context))
        (bundle / "logs").mkdir()
        log_path = bundle / "logs/train.log"
        selected_backend = backend or train_with_mnist_backend
        with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = selected_backend(spec=spec, context=operational_context, engine_config_path=engine_path, output_dir=bundle)
        if isinstance(result, dict) and result.get("status") == "pruned":
            _validate_pruned_best_artifacts(bundle)
            atomic_write_json(attempt_dir / "status.json", _status(run_id, attempt_id, "pruned", json.dumps(result, sort_keys=True)))
            return RunExecution(run_id, "pruned", None, attempt_id)
        _normalize_backend_output(bundle, spec, result)
        atomic_write_json(bundle / "status.json", _status(run_id, attempt_id, "complete"))
        metrics = _validate_scientific_artifacts(bundle, spec)
        _write_bundle_manifest(bundle, spec, run_id, attempt_id, code_provenance, metrics)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists(): raise InvalidBundleError(f"Expected exact-once publication. Provided existing path: {final_dir}.")
        os.replace(bundle, final_dir)
        atomic_write_json(attempt_dir / "status.json", _status(run_id, attempt_id, "complete", f"published:{relative_posix(final_dir, layout.root)}"))
        return RunExecution(run_id, "complete", final_dir, attempt_id)
    except BaseException as exc:
        try:
            attempt_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(attempt_dir / "status.json", _status(run_id, attempt_id, "failed", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
        raise
    finally:
        if heartbeat is not None and heartbeat_started:
            heartbeat.stop()
        _remove_owned_claim(layout.claim_path(run_id), attempt_id)
