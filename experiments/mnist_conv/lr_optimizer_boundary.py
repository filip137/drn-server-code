"""Immutable policy helpers for the Conv2 SGD/Adam boundary diagnostic.

The numerical optimizer implementation lives in :mod:`lr_engine` and
:mod:`lr_stages`.  This module owns the experiment topology and the pieces
that must remain usable without constructing MNIST or a model: grid
enumeration, fail-closed source reuse, selection, conditional sentinels,
bias-capped confirmations, completion validation, and deterministic packing.

This is a new lineage.  Nothing in this module changes the interpretation or
identity of the v1--v6 learning-rate studies.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import canonical_json_bytes, sha256_file, sha256_json
from .io import atomic_write_bytes, read_json


STUDY_SCHEMA_VERSION = "mnist-conv-lr-optimizer-boundary/v1"
RUN_SCHEMA_VERSION = "mnist-conv-run/v7"
MANIFEST_SCHEMA_VERSION = "mnist-conv-lr-optimizer-boundary-manifest/v1"
RESULT_SCHEMA_VERSION = "mnist-conv-lr-optimizer-boundary-result/v1"
SELECTION_SCHEMA_VERSION = "mnist-conv-lr-optimizer-boundary-selection/v1"
COMPLETION_SCHEMA_VERSION = "mnist-conv-lr-optimizer-boundary-completion/v1"
PACK_SCHEMA_VERSION = "mnist-conv-lr-optimizer-boundary-pack/v1"
STAGE_PLAN_SCHEMA_VERSION = "mnist-conv-lr-optimizer-boundary-stage-plan/v1"
PROBE_BUNDLE_SCHEMA_VERSION = "mnist-conv-lr-optimizer-boundary-probes/v1"
PROTOCOL_ID = "conv-hardsigmoid-lr-conv2-sgd-adam-boundary-constant-bs16-v1"

SCHEMES = ("baseline", "ours", "legacy")
OPTIMIZERS = ("sgd", "adam")
STAGES = ("main_grid", "upper_sentinel", "bias_capped_confirmation")
ROW_IDS = {
    "baseline": "conv2_baseline_v1_c1",
    "ours": "conv2_ours_v4_c1",
    "legacy": "conv2_legacy_v4_c0p25",
}
REUSE_SOURCE_DIRECTORIES = {
    ("baseline", 0.01): "conv2_baseline_v1_c1--grid-c03-d01",
    ("baseline", 0.03): "conv2_baseline_v1_c1--grid-c03-d02",
    ("ours", 0.003): "conv2_ours_v4_c1--scheme-rho--c03-d00",
    ("ours", 0.01): "conv2_ours_v4_c1--scheme-rho--c03-d01",
    ("legacy", 0.003): "conv2_legacy_v4_c0p25--scheme-rho--c03-d00",
    ("legacy", 0.01): "conv2_legacy_v4_c0p25--scheme-rho--c03-d01",
}
RHO_CONV_MAIN = (0.01, 0.015, 0.03)
RHO_CONV_SENTINEL = 0.1
RHO_DENSE_BY_SCHEME = {
    "baseline": (0.01, 0.03),
    "ours": (0.003, 0.01),
    "legacy": (0.003, 0.01),
}
PLATEAU_RELATIVE_TOLERANCE = 0.02
BIAS_RHO_CAP = 1.0e-3
EPOCHS = 5
STEPS_PER_EPOCH = 3_438
TOTAL_STEPS = 17_190
PROBE_BATCHES = 32
REPLAY_BATCHES = 8
OFFICIAL_TEST_READ = False
CONCURRENCY_LEVELS = (1, 2, 4, 8, 12, 16)
GPU_CAPACITY_MIB = 32_768.0
MEMORY_HEADROOM_FRACTION = 0.10
THROUGHPUT_FRACTION = 0.90
FROZEN_SPLIT_SHA256 = (
    "4801b7805d54dbe2197b98320cfe8a34d5d853c4ab568bcf1d7f69e136c94bb4"
)
FROZEN_TRAIN_BATCH_ORDER_SHA256 = (
    "e774de39fe6f9529dafded0b14356727b3e8510bd42f3a92f7416dcab0592d14"
)
FROZEN_INITIALIZATION_CHECKPOINT_SHA256 = (
    "dc16ef2bebd2c9e6afe307b41555683c3eec29b9481ab64ea3f412382c837c7e"
)
FROZEN_INITIALIZATION_TENSOR_SHA256 = (
    "e9aa471dc5558e525758b345de12c53f0a086c5a1b022024c2fc349ec138e2e3"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_PARAMETER_RE = re.compile(r"(?:ConvWeight|DenseWeight|Bias)_[0-9]+")


def _error(expected: str, provided: Any, path: str = "value") -> ValueError:
    return ValueError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _digest(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise _error("a lowercase SHA-256 digest", value, path)
    return value


def _finite(value: Any, path: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise _error(f"a {qualifier} number", value, path)
    return float(value)


def _rho(value: Any, path: str) -> float:
    return _finite(value, path, positive=True)


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1.0e-15)


def _canonical_id_float(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def arm_id(scheme: str, optimizer: str) -> str:
    if scheme not in SCHEMES:
        raise _error(f"one of {SCHEMES!r}", scheme, "scheme")
    if optimizer not in OPTIMIZERS:
        raise _error(f"one of {OPTIMIZERS!r}", optimizer, "optimizer")
    return f"{scheme}--{optimizer}"


def entry_id(
    *,
    scheme: str,
    optimizer: str,
    stage: str,
    rho_conv: float,
    rho_dense: float,
) -> str:
    arm_id(scheme, optimizer)
    if stage not in STAGES:
        raise _error(f"one of {STAGES!r}", stage, "stage")
    return (
        f"{stage}--{scheme}--{optimizer}"
        f"--c{_canonical_id_float(_rho(rho_conv, 'rho_conv'))}"
        f"--d{_canonical_id_float(_rho(rho_dense, 'rho_dense'))}"
    )


def main_grid_entries() -> list[dict[str, Any]]:
    """Return the exact 36-cell grid in deterministic pack order."""

    entries: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for optimizer in OPTIMIZERS:
            for rho_conv in RHO_CONV_MAIN:
                for rho_dense in RHO_DENSE_BY_SCHEME[scheme]:
                    reuse = optimizer == "sgd" and _same(rho_conv, 0.01)
                    entries.append(
                        {
                            "entry_index": len(entries),
                            "entry_id": entry_id(
                                scheme=scheme,
                                optimizer=optimizer,
                                stage="main_grid",
                                rho_conv=rho_conv,
                                rho_dense=rho_dense,
                            ),
                            "row_id": ROW_IDS[scheme],
                            "architecture": "conv2",
                            "scheme": scheme,
                            "optimizer": optimizer,
                            "stage": "main_grid",
                            "rho_conv": rho_conv,
                            "rho_dense": rho_dense,
                            "bias_policy": "attached",
                            "mode": (
                                "reuse_hash_verified_sgd"
                                if reuse
                                else "new_five_epoch_training"
                            ),
                            "epochs": EPOCHS,
                            "expected_steps": TOTAL_STEPS,
                            "official_test_read": False,
                        }
                    )
    counts = {
        mode: sum(entry["mode"] == mode for entry in entries)
        for mode in ("reuse_hash_verified_sgd", "new_five_epoch_training")
    }
    if len(entries) != 36 or counts != {
        "reuse_hash_verified_sgd": 6,
        "new_five_epoch_training": 30,
    }:
        raise RuntimeError(
            "Expected the frozen main grid to contain 36 cells: six reused SGD "
            f"and 30 new. Provided value: entries={len(entries)}, counts={counts!r}."
        )
    new_sgd = sum(
        entry["optimizer"] == "sgd"
        and entry["mode"] == "new_five_epoch_training"
        for entry in entries
    )
    new_adam = sum(
        entry["optimizer"] == "adam"
        and entry["mode"] == "new_five_epoch_training"
        for entry in entries
    )
    if (new_sgd, new_adam) != (12, 18):
        raise RuntimeError(
            "Expected exactly 12 new SGD and 18 new Adam main runs. "
            f"Provided value: {(new_sgd, new_adam)!r}."
        )
    return entries


def reuse_entry_ids() -> tuple[str, ...]:
    return tuple(
        entry["entry_id"]
        for entry in main_grid_entries()
        if entry["mode"] == "reuse_hash_verified_sgd"
    )


def _portable_relative(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _error("a non-empty portable relative path", value, path)
    relative = Path(value)
    if relative.is_absolute() or relative == Path(".") or ".." in relative.parts:
        raise _error("a contained portable relative path", value, path)
    if relative.as_posix() != value:
        raise _error("a normalized portable relative path", value, path)
    return value


def artifact_record(root: str | Path, relative_path: str) -> dict[str, Any]:
    source_root = Path(root).expanduser().resolve()
    relative = _portable_relative(relative_path, "relative_path")
    target = source_root / relative
    if target.is_symlink() or not target.is_file():
        raise _error("an existing regular non-symlink file", str(target), "artifact")
    return {
        "path": relative,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def reuse_cells_from_root(reuse_root: str | Path) -> list[dict[str, Any]]:
    """Freeze the exact six staged source directories into study-spec records.

    This audit understands the older v5/v6 candidate shapes, but it does not
    trust copied completion metadata alone: every output and completion marker
    is rehashed from the staged source tree.
    """

    root = Path(reuse_root).expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise _error(
            "an existing non-symlink reuse directory", str(root), "reuse_root"
        )
    observed_directories = {
        path.name
        for path in root.iterdir()
        if path.is_dir() and not path.is_symlink()
    }
    expected_directories = set(REUSE_SOURCE_DIRECTORIES.values())
    if observed_directories != expected_directories:
        raise _error(
            f"exactly the six source directories {sorted(expected_directories)!r}",
            sorted(observed_directories),
            "reuse_root",
        )

    expected_protocol_by_scheme = {
        "baseline": (
            "mnist-conv-run/v5",
            "conv-hardsigmoid-lr-conv2-two-rho-median-constant-sgd-bs16-v6",
        ),
        "ours": (
            "mnist-conv-run/v6",
            "conv-hardsigmoid-lr-conv2-amplified-scheme-two-rho-median-constant-sgd-bs16",
        ),
        "legacy": (
            "mnist-conv-run/v6",
            "conv-hardsigmoid-lr-conv2-amplified-scheme-two-rho-median-constant-sgd-bs16",
        ),
    }
    cells: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for rho_dense in RHO_DENSE_BY_SCHEME[scheme]:
            source_entry_id = REUSE_SOURCE_DIRECTORIES[(scheme, rho_dense)]
            source = root / source_entry_id
            completion_path = source / "complete.json"
            completion = read_json(completion_path)
            if completion.get("state") != "complete":
                raise RuntimeError(
                    "Expected a completed source cell; fallback reruns are "
                    f"prohibited. Provided value: {completion_path}."
                )
            source_collection_id = completion.get("study_id") or completion.get(
                "sweep_id"
            )
            if (
                not isinstance(source_collection_id, str)
                or re.fullmatch(r"(?:lrstudy_|lrsweep_)[0-9a-f]{64}", source_collection_id)
                is None
            ):
                raise _error(
                    "a content-addressed source study or sweep id",
                    source_collection_id,
                    f"{source_entry_id}.complete.json",
                )
            summary_path = source / "summary.json"
            summary = read_json(summary_path)
            expected_row_id = ROW_IDS[scheme]
            row = summary.get("row")
            if (
                summary.get("status") != "complete"
                or summary.get("training_completed") is not True
                or summary.get("admissible") is not True
                or summary.get("completed_steps") != TOTAL_STEPS
                or not isinstance(row, Mapping)
                or row.get("row_id") != expected_row_id
                or row.get("scheme") != scheme
                or not _same(float(summary.get("rho_conv")), 0.01)
                or not _same(float(summary.get("rho_dense")), rho_dense)
            ):
                raise RuntimeError(
                    "Expected a complete 17,190-step source summary at the "
                    "declared row and rho coordinates; aborting without rerun. "
                    f"Provided value: {summary_path}."
                )
            run_specs = sorted(source.glob("run_spec.v*.json"))
            if len(run_specs) != 1:
                raise _error(
                    "exactly one run_spec.v5.json or run_spec.v6.json",
                    [path.name for path in run_specs],
                    source_entry_id,
                )
            run_spec_path = run_specs[0]
            run_spec = read_json(run_spec_path)
            expected_schema, expected_protocol = expected_protocol_by_scheme[scheme]
            validation = (
                run_spec.get("run", {}).get("dataset", {}).get("validation", {})
            )
            optimizer = (
                run_spec.get("run", {}).get("training", {}).get("optimizer", {})
            )
            lr_provenance = run_spec.get("run", {}).get("lr_provenance")
            run_rates = (
                run_spec.get("run", {})
                .get("training", {})
                .get("learning_rates_by_parameter")
            )
            summary_rates = summary.get("learning_rates_by_parameter")
            if (
                run_spec.get("schema_version") != expected_schema
                or run_spec.get("protocol_id") != expected_protocol
                or validation.get("official_test_enabled") is not False
                or optimizer
                != {"momentum": 0.0, "name": "SGD", "weight_decay": 0.0}
                or not isinstance(lr_provenance, Mapping)
                or lr_provenance.get("row_id") != expected_row_id
                or not _same(float(lr_provenance.get("rho_conv")), 0.01)
                or not _same(float(lr_provenance.get("rho_dense")), rho_dense)
                or lr_provenance.get("split_sha256")
                != FROZEN_SPLIT_SHA256
                or lr_provenance.get("batch_order_sha256")
                != FROZEN_TRAIN_BATCH_ORDER_SHA256
                or lr_provenance.get("initialization_checkpoint_sha256")
                != FROZEN_INITIALIZATION_CHECKPOINT_SHA256
                or lr_provenance.get("initialization_tensor_sha256")
                != FROZEN_INITIALIZATION_TENSOR_SHA256
                or not isinstance(run_rates, Mapping)
                or not isinstance(summary_rates, Mapping)
            ):
                raise RuntimeError(
                    "Expected the frozen ordinary-MNIST constant plain-SGD source "
                    f"protocol with official test disabled. Provided value: {run_spec_path}."
                )
            normalized_run_rates = _validate_parameter_rates(
                run_rates, f"{source_entry_id}.run_spec.learning_rates"
            )
            normalized_summary_rates = _validate_parameter_rates(
                summary_rates, f"{source_entry_id}.summary.learning_rates"
            )
            if normalized_run_rates != normalized_summary_rates:
                raise RuntimeError(
                    "Expected source run-spec and summary raw per-parameter LRs "
                    f"to match exactly. Provided value: {source_entry_id}."
                )
            median_units = lr_provenance.get("median_unit_by_weight")
            if (
                not isinstance(median_units, Mapping)
                or set(median_units)
                != {"ConvWeight_0", "ConvWeight_1", "DenseWeight_0"}
            ):
                raise RuntimeError(
                    "Expected source LR provenance to bind all three weight "
                    f"normalization units. Provided value: {source_entry_id}."
                )
            for name, target in (
                ("ConvWeight_0", 0.01),
                ("ConvWeight_1", 0.01),
                ("DenseWeight_0", rho_dense),
            ):
                achieved_target = normalized_run_rates[name] * _finite(
                    median_units[name],
                    f"{source_entry_id}.median_unit_by_weight.{name}",
                    positive=True,
                )
                if not math.isclose(
                    achieved_target, target, rel_tol=1.0e-12, abs_tol=1.0e-15
                ):
                    raise RuntimeError(
                        "Expected source raw weight LR multiplied by its frozen "
                        f"probe unit to equal the declared rho target. Provided "
                        f"value: entry={source_entry_id}, parameter={name}, "
                        f"target={target}, observed={achieved_target}."
                    )
            if (
                not _same(
                    normalized_run_rates["Bias_0"],
                    normalized_run_rates["ConvWeight_0"],
                )
                or not _same(
                    normalized_run_rates["Bias_1"],
                    normalized_run_rates["ConvWeight_1"],
                )
            ):
                raise RuntimeError(
                    "Expected every reused source bias LR to equal its attached "
                    f"ConvWeight LR. Provided value: {source_entry_id}."
                )
            completion_outputs = completion.get("outputs")
            if not isinstance(completion_outputs, list) or len(completion_outputs) != 8:
                raise _error(
                    "exactly eight completion-bound outputs",
                    completion_outputs,
                    f"{source_entry_id}.complete.outputs",
                )
            outputs: list[dict[str, Any]] = []
            for output in completion_outputs:
                name = Path(str(output.get("path"))).name
                target = source / name
                observed = artifact_record(source, name)
                if (
                    observed["sha256"] != output.get("sha256")
                    or observed["bytes"] != output.get("bytes")
                ):
                    raise RuntimeError(
                        "Expected each staged source output to match its original "
                        "completion hash exactly; aborting without rerun. "
                        f"Provided value: {target}."
                    )
                outputs.append(
                    {
                        "name": name,
                        "sha256": observed["sha256"],
                        "bytes": observed["bytes"],
                    }
                )
            outputs.sort(key=lambda record: record["name"])
            output_by_name = {record["name"]: record for record in outputs}
            cells.append(
                {
                    "row_id": expected_row_id,
                    "rho_conv": 0.01,
                    "rho_dense": rho_dense,
                    "source_entry_id": source_entry_id,
                    "source_collection_id": source_collection_id,
                    "expected_completed_steps": TOTAL_STEPS,
                    "run_spec_sha256": output_by_name[run_spec_path.name][
                        "sha256"
                    ],
                    "candidate_summary_sha256": output_by_name["summary.json"][
                        "sha256"
                    ],
                    "completion_sha256": sha256_file(completion_path),
                    "completion_bytes": completion_path.stat().st_size,
                    "outputs": outputs,
                }
            )
    return sorted(
        cells,
        key=lambda cell: (
            cell["row_id"],
            cell["rho_conv"],
            cell["rho_dense"],
        ),
    )


def verify_reuse_cells_against_root(
    cells: Sequence[Mapping[str, Any]], reuse_root: str | Path
) -> list[dict[str, Any]]:
    """Re-audit a frozen spec against current bytes and require exact equality."""

    observed = reuse_cells_from_root(reuse_root)
    provided = [copy.deepcopy(dict(cell)) for cell in cells]
    provided.sort(
        key=lambda cell: (
            cell["row_id"],
            float(cell["rho_conv"]),
            float(cell["rho_dense"]),
        )
    )
    if provided != observed:
        raise RuntimeError(
            "Expected all six reused SGD cells to retain their frozen hashes and "
            "metadata; aborting rather than silently rerunning them."
        )
    return observed


def verify_artifact_binding(
    binding: Mapping[str, Any], *, reuse_root: str | Path
) -> dict[str, Any]:
    """Recompute every source hash; never regenerate or fall back on failure."""

    if not isinstance(binding, Mapping):
        raise _error("an artifact binding object", binding, "binding")
    entry = binding.get("entry")
    if not isinstance(entry, Mapping):
        raise _error("a source-cell object", entry, "binding.entry")
    expected_cell = {
        "scheme": entry.get("scheme"),
        "optimizer": entry.get("optimizer"),
        "rho_conv": entry.get("rho_conv"),
        "rho_dense": entry.get("rho_dense"),
    }
    if (
        expected_cell["scheme"] not in SCHEMES
        or expected_cell["optimizer"] != "sgd"
        or not _same(_rho(expected_cell["rho_conv"], "entry.rho_conv"), 0.01)
        or not any(
            _same(_rho(expected_cell["rho_dense"], "entry.rho_dense"), value)
            for value in RHO_DENSE_BY_SCHEME[expected_cell["scheme"]]
        )
    ):
        raise _error(
            "one of the six SGD rho_conv=0.01 reuse cells",
            expected_cell,
            "binding.entry",
        )
    expected_id = entry_id(
        scheme=str(expected_cell["scheme"]),
        optimizer="sgd",
        stage="main_grid",
        rho_conv=float(expected_cell["rho_conv"]),
        rho_dense=float(expected_cell["rho_dense"]),
    )
    if entry.get("entry_id") != expected_id:
        raise _error(expected_id, entry.get("entry_id"), "binding.entry.entry_id")

    if "source_root" in binding:
        raise _error(
            "a path-free reuse binding without source_root",
            binding.get("source_root"),
            "source_root",
        )
    source_root = Path(reuse_root).expanduser().resolve()
    if source_root.is_symlink() or not source_root.is_dir():
        raise _error(
            "an existing regular reuse root", str(source_root), "reuse_root"
        )

    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) < 2:
        raise _error(
            "at least a completion and summary artifact", artifacts, "artifacts"
        )
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(artifacts):
        path = f"artifacts[{index}]"
        if not isinstance(value, Mapping) or set(value) != {
            "path",
            "sha256",
            "bytes",
        }:
            raise _error(
                "an object with path, sha256, and bytes", value, path
            )
        relative = _portable_relative(value["path"], f"{path}.path")
        expected_hash = _digest(value["sha256"], f"{path}.sha256")
        expected_bytes = value["bytes"]
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise _error(
                "a non-negative integer", expected_bytes, f"{path}.bytes"
            )
        target = source_root / relative
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(
                "Expected every frozen reuse artifact to exist as a regular file. "
                f"Provided value: {target}."
            )
        observed_hash = sha256_file(target)
        observed_bytes = target.stat().st_size
        if (observed_hash, observed_bytes) != (expected_hash, expected_bytes):
            raise RuntimeError(
                "Expected the frozen reuse artifact hash and size to match exactly; "
                "rerunning the cell is prohibited. "
                f"Provided value: path={target}, expected_sha256={expected_hash!r}, "
                f"observed_sha256={observed_hash!r}, expected_bytes={expected_bytes}, "
                f"observed_bytes={observed_bytes}."
            )
        normalized.append(
            {"path": relative, "sha256": expected_hash, "bytes": expected_bytes}
        )
    paths = [record["path"] for record in normalized]
    if paths != sorted(set(paths)):
        raise _error("sorted artifacts with unique paths", paths, "artifacts")
    if not any(Path(path).name == "summary.json" for path in paths):
        raise _error("artifacts containing summary.json", paths, "artifacts")
    if not any(Path(path).name == "complete.json" for path in paths):
        raise _error("artifacts containing complete.json", paths, "artifacts")
    source_entry_id = binding.get("source_entry_id")
    expected_source_entry_id = REUSE_SOURCE_DIRECTORIES[
        (str(expected_cell["scheme"]), float(expected_cell["rho_dense"]))
    ]
    if (
        source_entry_id != expected_source_entry_id
        or {Path(path).parts[0] for path in paths} != {source_entry_id}
    ):
        raise _error(
            f"source entry {expected_source_entry_id!r} with every artifact "
            "under that portable prefix",
            {"source_entry_id": source_entry_id, "paths": paths},
            "source_entry_id",
        )
    return {
        "entry": {
            "entry_id": expected_id,
            "scheme": expected_cell["scheme"],
            "optimizer": "sgd",
            "rho_conv": float(expected_cell["rho_conv"]),
            "rho_dense": float(expected_cell["rho_dense"]),
        },
        "source_id": binding.get("source_id"),
        "source_entry_id": source_entry_id,
        "artifacts": normalized,
        "binding_sha256": sha256_json(
            {
                "entry_id": expected_id,
                "source_id": binding.get("source_id"),
                "source_entry_id": source_entry_id,
                "artifacts": normalized,
            }
        ),
    }


def verify_reuse_bindings(
    bindings: Sequence[Mapping[str, Any]], *, reuse_root: str | Path
) -> dict[str, dict[str, Any]]:
    """Validate the exact six-cell reuse set and return it by entry id."""

    normalized = [
        verify_artifact_binding(binding, reuse_root=reuse_root)
        for binding in bindings
    ]
    by_id = {binding["entry"]["entry_id"]: binding for binding in normalized}
    expected = set(reuse_entry_ids())
    if len(by_id) != len(normalized) or set(by_id) != expected:
        raise _error(
            f"exactly the six reuse entry ids {sorted(expected)!r}",
            sorted(by_id),
            "bindings",
        )
    return by_id


def _validate_parameter_rates(value: Any, path: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise _error("a non-empty per-parameter LR object", value, path)
    rates: dict[str, float] = {}
    for name, rate in value.items():
        if not isinstance(name, str) or _PARAMETER_RE.fullmatch(name) is None:
            raise _error("a scientific parameter name", name, f"{path}.key")
        rates[name] = _finite(rate, f"{path}.{name}", positive=True)
    required = {
        "ConvWeight_0",
        "Bias_0",
        "ConvWeight_1",
        "Bias_1",
        "DenseWeight_0",
    }
    if set(rates) != required:
        raise _error(
            f"exactly Conv2 parameters {sorted(required)!r}", sorted(rates), path
        )
    return dict(sorted(rates.items()))


def _optimizer_contract(value: Any, optimizer: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error("an optimizer contract object", value, "optimizer_parameters")
    expected = (
        {
            "name": "SGD",
            "momentum": 0.0,
            "weight_decay": 0.0,
        }
        if optimizer == "sgd"
        else {
            "name": "Adam",
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
        }
    )
    if dict(value) != expected:
        raise _error(f"the exact frozen {optimizer} contract", value)
    return copy.deepcopy(expected)


def build_main_manifest(
    *,
    study_id: str,
    study_config_sha256: str,
    code_fingerprint: str,
    parent_sha256: str,
    checkpoint_sha256: str,
    minibatch_sha256: str,
    probe_sha256_by_arm: Mapping[str, str],
    bias_q90_probe_sha256_by_arm: Mapping[str, str],
    normalization_unit_by_arm: Mapping[str, Mapping[str, float]],
    bias_q90_unit_by_arm: Mapping[str, Mapping[str, float]],
    raw_learning_rates_by_entry: Mapping[str, Mapping[str, float]],
    optimizer_parameters_by_name: Mapping[str, Mapping[str, Any]],
    reuse_bindings: Sequence[Mapping[str, Any]],
    reuse_root: str | Path,
) -> dict[str, Any]:
    """Bind all main cells after SGD/Adam probes have frozen raw LRs."""

    if not isinstance(study_id, str) or not study_id.startswith("lrstudy_"):
        raise _error("a content-addressed lrstudy id", study_id, "study_id")
    digests = {
        "study_config_sha256": _digest(
            study_config_sha256, "study_config_sha256"
        ),
        "code_fingerprint": _digest(code_fingerprint, "code_fingerprint"),
        "parent_sha256": _digest(parent_sha256, "parent_sha256"),
        "checkpoint_sha256": _digest(checkpoint_sha256, "checkpoint_sha256"),
        "minibatch_sha256": _digest(minibatch_sha256, "minibatch_sha256"),
    }
    verified_reuse = verify_reuse_bindings(
        reuse_bindings, reuse_root=reuse_root
    )
    expected_arm_ids = {
        arm_id(scheme, optimizer)
        for scheme in SCHEMES
        for optimizer in OPTIMIZERS
    }
    if set(probe_sha256_by_arm) != expected_arm_ids:
        raise _error(
            f"probe hashes for {sorted(expected_arm_ids)!r}",
            sorted(probe_sha256_by_arm),
            "probe_sha256_by_arm",
        )
    probe_hashes = {
        key: _digest(value, f"probe_sha256_by_arm.{key}")
        for key, value in probe_sha256_by_arm.items()
    }
    if set(bias_q90_probe_sha256_by_arm) != expected_arm_ids:
        raise _error(
            f"bias-Q90 probe hashes for {sorted(expected_arm_ids)!r}",
            sorted(bias_q90_probe_sha256_by_arm),
            "bias_q90_probe_sha256_by_arm",
        )
    bias_probe_hashes = {
        key: _digest(value, f"bias_q90_probe_sha256_by_arm.{key}")
        for key, value in bias_q90_probe_sha256_by_arm.items()
    }
    if set(normalization_unit_by_arm) != expected_arm_ids:
        raise _error(
            f"weight normalization units for {sorted(expected_arm_ids)!r}",
            sorted(normalization_unit_by_arm),
            "normalization_unit_by_arm",
        )
    normalization_units = {
        key: _probe_weight_units(
            {"normalization_unit_by_weight": value},
            f"normalization_unit_by_arm.{key}",
        )
        for key, value in normalization_unit_by_arm.items()
    }
    if set(bias_q90_unit_by_arm) != expected_arm_ids:
        raise _error(
            f"bias-Q90 units for {sorted(expected_arm_ids)!r}",
            sorted(bias_q90_unit_by_arm),
            "bias_q90_unit_by_arm",
        )
    bias_q90_units = {
        key: _probe_bias_units(
            {"bias_q90_unit_by_parameter": value},
            f"bias_q90_unit_by_arm.{key}",
        )
        for key, value in bias_q90_unit_by_arm.items()
    }
    optimizer_contracts = {
        optimizer: _optimizer_contract(
            optimizer_parameters_by_name.get(optimizer), optimizer
        )
        for optimizer in OPTIMIZERS
    }
    entries = main_grid_entries()
    if set(raw_learning_rates_by_entry) != {
        entry["entry_id"] for entry in entries
    }:
        raise _error(
            "raw per-parameter LRs for every one of the 36 main entries",
            sorted(raw_learning_rates_by_entry),
            "raw_learning_rates_by_entry",
        )
    bound_entries = []
    for entry in entries:
        rates = _validate_parameter_rates(
            raw_learning_rates_by_entry[entry["entry_id"]],
            f"raw_learning_rates_by_entry.{entry['entry_id']}",
        )
        if entry["bias_policy"] == "attached":
            for bias, weight in (
                ("Bias_0", "ConvWeight_0"),
                ("Bias_1", "ConvWeight_1"),
            ):
                if not _same(rates[bias], rates[weight]):
                    raise _error(
                        f"{bias} LR to equal attached {weight} LR",
                        rates,
                        entry["entry_id"],
                    )
        key = arm_id(entry["scheme"], entry["optimizer"])
        bound = {
            **entry,
            "protocol_id": PROTOCOL_ID,
            "run_schema_version": RUN_SCHEMA_VERSION,
            "raw_learning_rates_by_parameter": rates,
            "optimizer_parameters": optimizer_contracts[entry["optimizer"]],
            "probe_sha256": probe_hashes[key],
            "bias_q90_probe_sha256": bias_probe_hashes[key],
            "normalization_unit_by_weight": normalization_units[key],
            "bias_q90_unit_by_parameter": bias_q90_units[key],
            "parent_sha256": digests["parent_sha256"],
            "parent_study_config_sha256": digests["study_config_sha256"],
            "parent_entry_completion_sha256": None,
            "checkpoint_sha256": digests["checkpoint_sha256"],
            "minibatch_sha256": digests["minibatch_sha256"],
            "code_fingerprint": digests["code_fingerprint"],
        }
        if entry["mode"] == "reuse_hash_verified_sgd":
            reuse_binding = verified_reuse[entry["entry_id"]]
            completion_record = next(
                record
                for record in reuse_binding["artifacts"]
                if Path(record["path"]).name == "complete.json"
            )
            bound["parent_entry_completion_sha256"] = completion_record[
                "sha256"
            ]
            summary_record = next(
                record
                for record in reuse_binding["artifacts"]
                if Path(record["path"]).name == "summary.json"
            )
            source_summary = read_json(
                Path(reuse_root).expanduser().resolve()
                / summary_record["path"]
            )
            source_rates = _validate_parameter_rates(
                source_summary.get("learning_rates_by_parameter"),
                f"{entry['entry_id']}.source_summary.learning_rates",
            )
            for name, source_rate in source_rates.items():
                if not math.isclose(
                    rates[name], source_rate, rel_tol=1.0e-12, abs_tol=1.0e-15
                ):
                    raise RuntimeError(
                        "Expected the newly bound SGD probe LR to reproduce the "
                        "hash-verified reused source trajectory exactly. "
                        f"Provided value: entry={entry['entry_id']}, "
                        f"parameter={name}, derived={rates[name]!r}, "
                        f"source={source_rate!r}."
                    )
            bound["reuse_binding"] = reuse_binding
        else:
            bound["reuse_binding"] = None
        bound_entries.append(bound)
    scientific = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "study_schema_version": STUDY_SCHEMA_VERSION,
        "run_schema_version": RUN_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "study_id": study_id,
        **digests,
        "study_role": "ordinary_mnist_local_high_rho_optimizer_diagnostic",
        "optimizer_scope": "local_high_rho_surface_only",
        "medium_affine_handoff_replaced": False,
        "final_paper_optimizer_changed": False,
        "official_test_read": False,
        "seed": 0,
        "batch_size": 16,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_steps": TOTAL_STEPS,
        "probe_batches": PROBE_BATCHES,
        "replay_batches": REPLAY_BATCHES,
        "rho_conv_main": list(RHO_CONV_MAIN),
        "rho_conv_sentinel": RHO_CONV_SENTINEL,
        "rho_dense_by_scheme": {
            key: list(value) for key, value in RHO_DENSE_BY_SCHEME.items()
        },
        "plateau_relative_tolerance": PLATEAU_RELATIVE_TOLERANCE,
        "bias_rho_cap": BIAS_RHO_CAP,
        "optimizer_arms": optimizer_contracts,
        "entries": bound_entries,
    }
    return {
        **scientific,
        "manifest_id": "lroptboundary_" + sha256_json(scientific),
    }


def publish_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path)
    encoded = canonical_json_bytes(dict(value)) + b"\n"
    if target.exists():
        if target.read_bytes() != encoded:
            raise RuntimeError(
                f"Expected immutable artifact {target} to retain identical bytes. "
                "Provided value: conflicting content."
            )
        return target
    return atomic_write_bytes(target, encoded)


def load_main_manifest(
    path: str | Path, *, reuse_root: str | Path | None = None
) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise _error(
            f"schema {MANIFEST_SCHEMA_VERSION!r}",
            manifest.get("schema_version"),
            "manifest.schema_version",
        )
    scientific = dict(manifest)
    observed_id = scientific.pop("manifest_id", None)
    expected_id = "lroptboundary_" + sha256_json(scientific)
    if observed_id != expected_id:
        raise _error(expected_id, observed_id, "manifest.manifest_id")
    reuse = [
        entry["reuse_binding"]
        for entry in manifest["entries"]
        if entry["mode"] == "reuse_hash_verified_sgd"
    ]
    if reuse_root is not None:
        verify_reuse_bindings(reuse, reuse_root=reuse_root)
    return manifest


def _probe_weight_units(probe: Mapping[str, Any], path: str) -> dict[str, float]:
    raw = (
        probe.get("weight_units_by_parameter")
        or probe.get("median_units_by_weight")
        or probe.get("optimizer_units_by_weight")
        or probe.get("normalization_unit_by_weight")
    )
    if not isinstance(raw, Mapping):
        raise _error("per-weight optimizer proposal units", raw, path)
    expected = {"ConvWeight_0", "ConvWeight_1", "DenseWeight_0"}
    if set(raw) != expected:
        raise _error(
            f"exactly weights {sorted(expected)!r}", sorted(raw), path
        )
    return {
        name: _finite(value, f"{path}.{name}", positive=True)
        for name, value in raw.items()
    }


def _probe_bias_units(probe: Mapping[str, Any], path: str) -> dict[str, float]:
    raw = (
        probe.get("q90_bias_units_by_parameter")
        or probe.get("q90_units_by_bias")
        or probe.get("bias_q90_units")
        or probe.get("bias_q90_unit_by_parameter")
    )
    if not isinstance(raw, Mapping) or set(raw) != {"Bias_0", "Bias_1"}:
        raise _error(
            "Q90 optimizer proposal units for Bias_0 and Bias_1", raw, path
        )
    return {
        name: _finite(value, f"{path}.{name}", positive=True)
        for name, value in raw.items()
    }


def normalize_probe_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the six optimizer-specific probes consumed by all stages."""

    if not isinstance(value, Mapping):
        raise _error("a probe bundle object", value, "probe_bundle")
    if (
        value.get("schema_version") != PROBE_BUNDLE_SCHEMA_VERSION
        or value.get("official_test_read") is not False
        or not isinstance(value.get("study_id"), str)
        or re.fullmatch(r"lrstudy_[0-9a-f]{64}", value["study_id"]) is None
    ):
        raise _error(
            f"schema {PROBE_BUNDLE_SCHEMA_VERSION!r}, a content-addressed "
            "study id, and official_test_read=false",
            {
                "schema_version": value.get("schema_version"),
                "study_id": value.get("study_id"),
                "official_test_read": value.get("official_test_read"),
            },
            "probe_bundle",
        )
    raw_probes = value.get("probes_by_arm")
    if not isinstance(raw_probes, Mapping):
        raise _error(
            "an object keyed by all six arm ids",
            raw_probes,
            "probe_bundle.probes_by_arm",
        )
    expected = {
        arm_id(scheme, optimizer)
        for scheme in SCHEMES
        for optimizer in OPTIMIZERS
    }
    if set(raw_probes) != expected:
        raise _error(
            f"exactly probes {sorted(expected)!r}",
            sorted(raw_probes),
            "probe_bundle.probes_by_arm",
        )
    probes: dict[str, Any] = {}
    for key in sorted(expected):
        probe = raw_probes[key]
        if not isinstance(probe, Mapping):
            raise _error("a probe object", probe, f"probes_by_arm.{key}")
        scheme, optimizer = key.split("--", maxsplit=1)
        observed_optimizer = probe.get("optimizer") or str(
            probe.get("optimizer_name", "")
        ).lower()
        if (
            probe.get("scheme") != scheme
            or observed_optimizer != optimizer
            or probe.get("official_test_read") is not False
        ):
            raise _error(
                "matching scheme/optimizer with official_test_read=false",
                probe,
                f"probes_by_arm.{key}",
            )
        probe_sha = probe.get("probe_sha256") or probe.get("artifact_sha256")
        weight_probe_sha = (
            probe.get("weight_normalization_probe_sha256") or probe_sha
        )
        if probe_sha != weight_probe_sha:
            raise _error(
                "probe_sha256 to identify the authoritative weight "
                "normalization probe",
                {
                    "probe_sha256": probe_sha,
                    "weight_normalization_probe_sha256": weight_probe_sha,
                },
                f"probes_by_arm.{key}",
            )
        bias_probe_sha = probe.get("bias_q90_probe_sha256")
        probes[key] = {
            **dict(probe),
            "probe_sha256": _digest(
                probe_sha, f"probes_by_arm.{key}.probe_sha256"
            ),
            "weight_normalization_probe_sha256": _digest(
                weight_probe_sha,
                f"probes_by_arm.{key}.weight_normalization_probe_sha256",
            ),
            "bias_q90_probe_sha256": _digest(
                bias_probe_sha,
                f"probes_by_arm.{key}.bias_q90_probe_sha256",
            ),
            "weight_units_by_parameter": _probe_weight_units(
                probe, f"probes_by_arm.{key}.weight_units"
            ),
            "q90_bias_units_by_parameter": _probe_bias_units(
                probe, f"probes_by_arm.{key}.bias_units"
            ),
            "optimizer_parameters": _optimizer_contract(
                probe.get("optimizer_parameters"), optimizer
            ),
        }
    required_hashes = {
        field: _digest(value.get(field), f"probe_bundle.{field}")
        for field in (
            "code_fingerprint",
            "parent_sha256",
            "checkpoint_sha256",
            "minibatch_sha256",
        )
    }
    return {
        "schema_version": PROBE_BUNDLE_SCHEMA_VERSION,
        "study_id": value.get("study_id"),
        **required_hashes,
        "probes_by_arm": probes,
        "official_test_read": False,
    }


def raw_learning_rates_from_probe(
    probe: Mapping[str, Any],
    *,
    rho_conv: float,
    rho_dense: float,
) -> dict[str, float]:
    """Derive optimizer-specific attached-bias rates from one frozen probe."""

    units = _probe_weight_units(probe, "probe.weight_units")
    rates = {
        "ConvWeight_0": _rho(rho_conv, "rho_conv") / units["ConvWeight_0"],
        "ConvWeight_1": _rho(rho_conv, "rho_conv") / units["ConvWeight_1"],
        "DenseWeight_0": _rho(rho_dense, "rho_dense")
        / units["DenseWeight_0"],
    }
    rates["Bias_0"] = rates["ConvWeight_0"]
    rates["Bias_1"] = rates["ConvWeight_1"]
    return dict(sorted(rates.items()))


def raw_main_learning_rates(
    probe_bundle: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    normalized = normalize_probe_bundle(probe_bundle)
    result = {}
    for entry in main_grid_entries():
        probe = normalized["probes_by_arm"][
            arm_id(entry["scheme"], entry["optimizer"])
        ]
        result[entry["entry_id"]] = raw_learning_rates_from_probe(
            probe,
            rho_conv=entry["rho_conv"],
            rho_dense=entry["rho_dense"],
        )
    return result


def bind_stage_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    main_manifest: Mapping[str, Any],
    probe_bundle: Mapping[str, Any],
    stage: str,
    stage_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a conditional stage to probes, parent hashes, and raw LRs."""

    if stage not in {"upper_sentinel", "bias_capped_confirmation"}:
        raise _error(
            "'upper_sentinel' or 'bias_capped_confirmation'", stage, "stage"
        )
    probes = normalize_probe_bundle(probe_bundle)
    if probes["study_id"] != main_manifest["study_id"]:
        raise _error(
            "probe bundle study_id to match the main manifest",
            probes["study_id"],
            "probe_bundle.study_id",
        )
    bound: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(entries):
        entry = dict(raw_entry)
        if entry.get("stage") != stage:
            raise _error(stage, entry.get("stage"), f"entries[{index}].stage")
        key = arm_id(entry["scheme"], entry["optimizer"])
        probe = probes["probes_by_arm"][key]
        if stage == "upper_sentinel":
            rates = raw_learning_rates_from_probe(
                probe,
                rho_conv=entry["rho_conv"],
                rho_dense=entry["rho_dense"],
            )
        else:
            rates = _validate_parameter_rates(
                entry.get("raw_learning_rates_by_parameter"),
                f"entries[{index}].raw_learning_rates_by_parameter",
            )
        bound.append(
            {
                **entry,
                "entry_index": index,
                "protocol_id": PROTOCOL_ID,
                "run_schema_version": RUN_SCHEMA_VERSION,
                "raw_learning_rates_by_parameter": rates,
                "optimizer_parameters": probe["optimizer_parameters"],
                "probe_sha256": probe["probe_sha256"],
                "bias_q90_probe_sha256": probe[
                    "bias_q90_probe_sha256"
                ],
                "normalization_unit_by_weight": probe[
                    "weight_units_by_parameter"
                ],
                "bias_q90_unit_by_parameter": probe[
                    "q90_bias_units_by_parameter"
                ],
                "parent_sha256": _digest(
                    entry.get("parent_sha256")
                    or main_manifest["parent_sha256"],
                    f"entries[{index}].parent_sha256",
                ),
                "parent_study_config_sha256": main_manifest[
                    "study_config_sha256"
                ],
                "parent_entry_completion_sha256": (
                    entry.get("parent_entry_completion_sha256")
                ),
                "parent_decision_sha256": entry.get(
                    "parent_decision_sha256"
                ),
                "checkpoint_sha256": probes["checkpoint_sha256"],
                "minibatch_sha256": probes["minibatch_sha256"],
                "code_fingerprint": probes["code_fingerprint"],
                "official_test_read": False,
            }
        )
    scientific = {
        "schema_version": STAGE_PLAN_SCHEMA_VERSION,
        "study_id": main_manifest["study_id"],
        "main_manifest_id": main_manifest["manifest_id"],
        "main_manifest_sha256": sha256_json(main_manifest),
        "stage": stage,
        "entry_count": len(bound),
        "entries": bound,
        "stage_provenance": (
            dict(sorted(stage_provenance.items()))
            if stage_provenance is not None
            else {}
        ),
        "official_test_read": False,
    }
    return {
        **scientific,
        "plan_id": "lrstageplan_" + sha256_json(scientific),
    }


def load_execution_plan(
    path: str | Path, *, reuse_root: str | Path | None = None
) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") == MANIFEST_SCHEMA_VERSION:
        return load_main_manifest(path, reuse_root=reuse_root)
    if value.get("schema_version") != STAGE_PLAN_SCHEMA_VERSION:
        raise _error(
            f"{MANIFEST_SCHEMA_VERSION!r} or {STAGE_PLAN_SCHEMA_VERSION!r}",
            value.get("schema_version"),
            "plan.schema_version",
        )
    scientific = dict(value)
    observed_id = scientific.pop("plan_id", None)
    expected_id = "lrstageplan_" + sha256_json(scientific)
    if observed_id != expected_id:
        raise _error(expected_id, observed_id, "plan.plan_id")
    if not isinstance(value.get("entries"), list):
        raise _error("an entry list", value.get("entries"), "plan.entries")
    return value


def validate_result(
    value: Mapping[str, Any],
    *,
    require_provenance: bool = True,
) -> dict[str, Any]:
    """Validate one new-lineage result used for selection or completion."""

    if not isinstance(value, Mapping):
        raise _error("a result object", value, "result")
    scheme = value.get("scheme")
    optimizer = value.get("optimizer")
    stage = value.get("stage")
    arm_id(str(scheme), str(optimizer))
    if stage not in STAGES:
        raise _error(f"one of {STAGES!r}", stage, "result.stage")
    rho_conv = _rho(value.get("rho_conv"), "result.rho_conv")
    rho_dense = _rho(value.get("rho_dense"), "result.rho_dense")
    expected_entry_id = entry_id(
        scheme=str(scheme),
        optimizer=str(optimizer),
        stage=str(stage),
        rho_conv=rho_conv,
        rho_dense=rho_dense,
    )
    if value.get("entry_id") != expected_entry_id:
        raise _error(expected_entry_id, value.get("entry_id"), "result.entry_id")
    if value.get("official_test_read") is not False:
        raise _error("false", value.get("official_test_read"), "official_test_read")
    status = value.get("status")
    completed_steps = value.get("completed_steps")
    if type(completed_steps) is not int or not 0 <= completed_steps <= TOTAL_STEPS:
        raise _error(
            f"an integer in [0, {TOTAL_STEPS}]", completed_steps, "completed_steps"
        )
    if status == "complete":
        if completed_steps != TOTAL_STEPS:
            raise _error(
                f"exactly {TOTAL_STEPS} for a completed candidate",
                completed_steps,
                "completed_steps",
            )
    elif status == "safety_failure":
        safety_failure = value.get("safety_failure")
        post_training_replay_failure = (
            isinstance(safety_failure, Mapping)
            and safety_failure.get("kind")
            == "non_finite_epoch_validation_or_replay"
            and safety_failure.get("attempted_step") == TOTAL_STEPS
            and safety_failure.get("epoch") == EPOCHS
            and completed_steps == TOTAL_STEPS
        )
        if (
            not isinstance(safety_failure, Mapping)
            or (
                completed_steps >= TOTAL_STEPS
                and not post_training_replay_failure
            )
        ):
            raise _error(
                "an early recorded safety failure, or the explicit non-finite "
                "epoch-5 validation/replay gate after exactly 17,190 optimizer "
                "steps",
                {
                    "completed_steps": completed_steps,
                    "safety_failure": safety_failure,
                },
                "result",
            )
    else:
        raise _error("'complete' or 'safety_failure'", status, "result.status")
    admissible = bool(value.get("admissible", status == "complete"))
    if admissible and status != "complete":
        raise _error(
            "false when status is safety_failure", admissible, "result.admissible"
        )
    normalized = dict(value)
    normalized.update(
        {
            "scheme": scheme,
            "optimizer": optimizer,
            "stage": stage,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "completed_steps": completed_steps,
            "admissible": admissible,
        }
    )
    if status == "complete":
        for field in (
            "final_validation_loss",
            "final_validation_accuracy",
            "median_projection_efficiency",
        ):
            normalized[field] = _finite(value.get(field), f"result.{field}")
    if require_provenance:
        for field in (
            "probe_sha256",
            "bias_q90_probe_sha256",
            "parent_sha256",
            "minibatch_sha256",
            "checkpoint_sha256",
            "code_fingerprint",
        ):
            normalized[field] = _digest(value.get(field), f"result.{field}")
        normalized["raw_learning_rates_by_parameter"] = _validate_parameter_rates(
            value.get("raw_learning_rates_by_parameter"),
            "result.raw_learning_rates_by_parameter",
        )
        normalized["optimizer_parameters"] = _optimizer_contract(
            value.get("optimizer_parameters"), str(optimizer)
        )
        normalized["normalization_unit_by_weight"] = _probe_weight_units(
            value, "result.normalization_unit_by_weight"
        )
        normalized["bias_q90_unit_by_parameter"] = _probe_bias_units(
            value, "result.bias_q90_unit_by_parameter"
        )
        normalized["parent_study_config_sha256"] = _digest(
            value.get("parent_study_config_sha256"),
            "result.parent_study_config_sha256",
        )
        for field in (
            "parent_entry_completion_sha256",
            "parent_result_sha256",
            "parent_decision_sha256",
        ):
            item = value.get(field)
            normalized[field] = (
                None if item is None else _digest(item, f"result.{field}")
            )
        parent_entry_id = value.get("parent_entry_id")
        if parent_entry_id is not None and (
            not isinstance(parent_entry_id, str) or not parent_entry_id
        ):
            raise _error(
                "a non-empty string or null",
                parent_entry_id,
                "result.parent_entry_id",
            )
        normalized["parent_entry_id"] = parent_entry_id
    return normalized


def _eligible_records(
    records: Sequence[Mapping[str, Any]], *, require_provenance: bool
) -> list[dict[str, Any]]:
    normalized = [
        validate_result(record, require_provenance=require_provenance)
        for record in records
    ]
    arms = {(record["scheme"], record["optimizer"]) for record in normalized}
    if len(arms) != 1:
        raise _error("records for exactly one scheme x optimizer arm", arms)
    return [
        record
        for record in normalized
        if record["status"] == "complete" and record["admissible"]
    ]


def select_arm(
    records: Sequence[Mapping[str, Any]],
    *,
    require_provenance: bool = True,
) -> dict[str, Any]:
    """Apply the frozen loss plateau and deterministic tie breakers."""

    normalized_records = [
        validate_result(record, require_provenance=require_provenance)
        for record in records
    ]
    selection_records = [
        record
        for record in normalized_records
        if record["stage"] in {"main_grid", "upper_sentinel"}
        and record.get("bias_policy", "attached") == "attached"
    ]
    eligible = _eligible_records(
        selection_records, require_provenance=require_provenance
    )
    if not eligible:
        first = records[0] if records else {}
        return {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "scheme": first.get("scheme"),
            "optimizer": first.get("optimizer"),
            "status": "unresolved_no_admissible_candidate",
            "minimum_final_validation_loss": None,
            "plateau": [],
            "selected": None,
            "trigger_upper_sentinel": False,
            "boundary_statuses": [],
            "official_test_read": False,
        }
    minimum_loss = min(record["final_validation_loss"] for record in eligible)
    plateau_limit = minimum_loss * (1.0 + PLATEAU_RELATIVE_TOLERANCE)
    plateau = [
        record
        for record in eligible
        if record["final_validation_loss"] <= plateau_limit
    ]
    ranked = sorted(
        plateau,
        key=lambda record: (
            -record["final_validation_accuracy"],
            -record["median_projection_efficiency"],
            max(record["rho_conv"], record["rho_dense"]),
            record["rho_conv"] + record["rho_dense"],
            record["final_validation_loss"],
            record["entry_id"],
        ),
    )
    selected = ranked[0]
    reaches_main_high = any(
        _same(record["rho_conv"], max(RHO_CONV_MAIN)) for record in plateau
    )
    sentinel_evaluated = any(
        record["stage"] == "upper_sentinel" for record in normalized_records
    )
    reaches_sentinel = any(
        _same(record["rho_conv"], RHO_CONV_SENTINEL) for record in plateau
    )
    reaches_adam_low = selected["optimizer"] == "adam" and any(
        _same(record["rho_conv"], min(RHO_CONV_MAIN)) for record in plateau
    )
    statuses: list[str] = []
    if reaches_sentinel:
        statuses.append("unbracketed_high")
    if reaches_adam_low:
        statuses.append("unbracketed_low")
    if not statuses and (not reaches_main_high or sentinel_evaluated):
        statuses.append("bracketed")
    result = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "scheme": selected["scheme"],
        "optimizer": selected["optimizer"],
        "status": (
            "pending_upper_sentinel"
            if reaches_main_high and not sentinel_evaluated
            else "selected"
        ),
        "minimum_final_validation_loss": minimum_loss,
        "plateau_limit": plateau_limit,
        "plateau": [
            {
                "entry_id": record["entry_id"],
                "rho_conv": record["rho_conv"],
                "rho_dense": record["rho_dense"],
                "final_validation_loss": record["final_validation_loss"],
                "final_validation_accuracy": record[
                    "final_validation_accuracy"
                ],
                "median_projection_efficiency": record[
                    "median_projection_efficiency"
                ],
            }
            for record in sorted(plateau, key=lambda item: item["entry_id"])
        ],
        "selected": {
            **{
                key: selected[key]
                for key in (
                    "entry_id",
                    "rho_conv",
                    "rho_dense",
                    "final_validation_loss",
                    "final_validation_accuracy",
                    "median_projection_efficiency",
                )
            },
            **{
                key: selected[key]
                for key in ("result_sha256", "completion_sha256")
                if key in selected
            },
        },
        "trigger_upper_sentinel": reaches_main_high and not sentinel_evaluated,
        "boundary_statuses": statuses,
        "official_test_read": False,
    }
    return result


def select_all_arms(
    records: Sequence[Mapping[str, Any]],
    *,
    require_provenance: bool = True,
) -> dict[str, dict[str, Any]]:
    by_arm: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (str(record.get("scheme")), str(record.get("optimizer")))
        arm_id(*key)
        by_arm.setdefault(key, []).append(record)
    expected = {(scheme, optimizer) for scheme in SCHEMES for optimizer in OPTIMIZERS}
    if set(by_arm) != expected:
        raise _error(
            f"records for all six arms {sorted(expected)!r}",
            sorted(by_arm),
            "records",
        )
    return {
        arm_id(scheme, optimizer): select_arm(
            by_arm[(scheme, optimizer)],
            require_provenance=require_provenance,
        )
        for scheme in SCHEMES
        for optimizer in OPTIMIZERS
    }


def upper_sentinel_entries(
    selections_by_arm: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Generate two predeclared rho_conv=0.1 cells for each triggered arm."""

    entries: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for optimizer in OPTIMIZERS:
            selection = selections_by_arm.get(arm_id(scheme, optimizer))
            if not isinstance(selection, Mapping):
                raise _error(
                    "one selection per arm",
                    selection,
                    f"selections_by_arm.{arm_id(scheme, optimizer)}",
                )
            if not selection.get("trigger_upper_sentinel"):
                continue
            for rho_dense in RHO_DENSE_BY_SCHEME[scheme]:
                entries.append(
                    {
                        "entry_index": len(entries),
                        "entry_id": entry_id(
                            scheme=scheme,
                            optimizer=optimizer,
                            stage="upper_sentinel",
                            rho_conv=RHO_CONV_SENTINEL,
                            rho_dense=rho_dense,
                        ),
                        "row_id": ROW_IDS[scheme],
                        "architecture": "conv2",
                        "scheme": scheme,
                        "optimizer": optimizer,
                        "stage": "upper_sentinel",
                        "rho_conv": RHO_CONV_SENTINEL,
                        "rho_dense": rho_dense,
                        "bias_policy": "attached",
                        "mode": "new_five_epoch_training",
                        "epochs": EPOCHS,
                        "expected_steps": TOTAL_STEPS,
                        "official_test_read": False,
                    }
                )
    return entries


def capped_bias_learning_rates(
    attached_learning_rates: Mapping[str, float],
    *,
    q90_nominal_bias_units: Mapping[str, float],
    bias_to_weight: Mapping[str, str],
    bias_rho_cap: float = BIAS_RHO_CAP,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Cap initial bias rho without changing any weight LR or increasing a bias."""

    rates = _validate_parameter_rates(
        attached_learning_rates, "attached_learning_rates"
    )
    cap = _rho(bias_rho_cap, "bias_rho_cap")
    expected_mapping = {
        "Bias_0": "ConvWeight_0",
        "Bias_1": "ConvWeight_1",
    }
    if dict(bias_to_weight) != expected_mapping:
        raise _error(
            f"the Conv2 bias mapping {expected_mapping!r}",
            bias_to_weight,
            "bias_to_weight",
        )
    if set(q90_nominal_bias_units) != set(expected_mapping):
        raise _error(
            f"Q90 units for {sorted(expected_mapping)!r}",
            sorted(q90_nominal_bias_units),
            "q90_nominal_bias_units",
        )
    capped = dict(rates)
    details: dict[str, Any] = {}
    for bias, weight in expected_mapping.items():
        attached = rates[bias]
        if not _same(attached, rates[weight]):
            raise _error(
                f"{bias} attached LR to equal {weight}", rates, bias
            )
        unit = _finite(
            q90_nominal_bias_units[bias],
            f"q90_nominal_bias_units.{bias}",
            positive=True,
        )
        absolute_cap_lr = cap / unit
        capped_lr = min(attached, absolute_cap_lr)
        if capped_lr > attached:
            raise RuntimeError(
                "Expected a capped bias LR never to exceed its attached LR."
            )
        capped[bias] = capped_lr
        details[bias] = {
            "attached_weight": weight,
            "attached_learning_rate": attached,
            "q90_nominal_bias_unit": unit,
            "absolute_cap_learning_rate": absolute_cap_lr,
            "capped_learning_rate": capped_lr,
            "cap_increased_learning_rate": False,
        }
    return dict(sorted(capped.items())), {
        "bias_rho_cap": cap,
        "units_definition": (
            "optimizer-specific Q90 nominal-LR bias proposal divided by "
            "attached weight initial RMS"
        ),
        "biases": details,
    }


def bias_capped_confirmation_entries(
    selections_by_arm: Mapping[str, Mapping[str, Any]],
    result_by_entry_id: Mapping[str, Mapping[str, Any]],
    q90_bias_units_by_arm: Mapping[str, Mapping[str, float]],
) -> list[dict[str, Any]]:
    """Create exactly one capped confirmation for each selected arm."""

    confirmations: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for optimizer in OPTIMIZERS:
            key = arm_id(scheme, optimizer)
            selection = selections_by_arm.get(key)
            if (
                not isinstance(selection, Mapping)
                or selection.get("status") != "selected"
                or selection.get("trigger_upper_sentinel") is not False
                or selection.get("selected") is None
            ):
                raise _error(
                    "a fully resolved post-sentinel selection with "
                    "status='selected' and trigger_upper_sentinel=false",
                    selection,
                    key,
                )
            parent_id = selection["selected"]["entry_id"]
            parent = result_by_entry_id.get(parent_id)
            if not isinstance(parent, Mapping):
                raise _error(
                    "the selected parent result", parent, f"result_by_entry_id.{parent_id}"
                )
            selected_record = selection["selected"]
            if (
                parent.get("scheme") != scheme
                or parent.get("optimizer") != optimizer
                or parent.get("stage")
                not in {"main_grid", "upper_sentinel"}
                or parent.get("bias_policy", "attached") != "attached"
                or parent.get("result_sha256")
                != selected_record.get("result_sha256")
                or parent.get("completion_sha256")
                != selected_record.get("completion_sha256")
            ):
                raise RuntimeError(
                    "Expected each resolved selection to bind the exact "
                    "attached parent result and completion hashes before "
                    f"creating a capped confirmation. Provided arm: {key!r}."
                )
            rates, cap = capped_bias_learning_rates(
                parent["raw_learning_rates_by_parameter"],
                q90_nominal_bias_units=q90_bias_units_by_arm.get(key, {}),
                bias_to_weight={
                    "Bias_0": "ConvWeight_0",
                    "Bias_1": "ConvWeight_1",
                },
            )
            rho_conv = float(selection["selected"]["rho_conv"])
            rho_dense = float(selection["selected"]["rho_dense"])
            confirmations.append(
                {
                    "entry_index": len(confirmations),
                    "entry_id": entry_id(
                        scheme=scheme,
                        optimizer=optimizer,
                        stage="bias_capped_confirmation",
                        rho_conv=rho_conv,
                        rho_dense=rho_dense,
                    ),
                    "row_id": ROW_IDS[scheme],
                    "architecture": "conv2",
                    "scheme": scheme,
                    "optimizer": optimizer,
                    "stage": "bias_capped_confirmation",
                    "rho_conv": rho_conv,
                    "rho_dense": rho_dense,
                    "bias_policy": "capped",
                    "mode": "new_five_epoch_training",
                    "epochs": EPOCHS,
                    "expected_steps": TOTAL_STEPS,
                    "parent_entry_id": parent_id,
                    "parent_result_sha256": _digest(
                        parent["result_sha256"], f"{parent_id}.result_sha256"
                    ),
                    "parent_entry_completion_sha256": _digest(
                        parent["completion_sha256"],
                        f"{parent_id}.completion_sha256",
                    ),
                    "parent_sha256": _digest(
                        parent["completion_sha256"],
                        f"{parent_id}.completion_sha256",
                    ),
                    "raw_learning_rates_by_parameter": rates,
                    "bias_cap": cap,
                    "official_test_read": False,
                }
            )
    if len(confirmations) != 6:
        raise RuntimeError(
            "Expected exactly six bias-capped confirmations. "
            f"Provided value: {len(confirmations)}."
        )
    return confirmations


def execution_entry_provenance(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical execution identity checked before resume and aggregation."""

    optimizer = str(entry.get("optimizer"))
    if optimizer not in OPTIMIZERS:
        raise _error(f"one of {OPTIMIZERS!r}", optimizer, "entry.optimizer")

    def optional_digest(field: str) -> str | None:
        value = entry.get(field)
        return None if value is None else _digest(value, f"entry.{field}")

    parent_entry_id = entry.get("parent_entry_id")
    if parent_entry_id is not None and (
        not isinstance(parent_entry_id, str) or not parent_entry_id
    ):
        raise _error(
            "a non-empty parent entry id or null",
            parent_entry_id,
            "entry.parent_entry_id",
        )
    return {
        "optimizer_parameters": _optimizer_contract(
            entry.get("optimizer_parameters"), optimizer
        ),
        "probe_sha256": _digest(
            entry.get("probe_sha256"), "entry.probe_sha256"
        ),
        "bias_q90_probe_sha256": _digest(
            entry.get("bias_q90_probe_sha256"),
            "entry.bias_q90_probe_sha256",
        ),
        "normalization_unit_by_weight": _probe_weight_units(
            entry, "entry.normalization_unit_by_weight"
        ),
        "bias_q90_unit_by_parameter": _probe_bias_units(
            entry, "entry.bias_q90_unit_by_parameter"
        ),
        "parent_sha256": _digest(
            entry.get("parent_sha256"), "entry.parent_sha256"
        ),
        "parent_study_config_sha256": _digest(
            entry.get("parent_study_config_sha256"),
            "entry.parent_study_config_sha256",
        ),
        "parent_entry_completion_sha256": optional_digest(
            "parent_entry_completion_sha256"
        ),
        "parent_entry_id": parent_entry_id,
        "parent_result_sha256": optional_digest("parent_result_sha256"),
        "parent_decision_sha256": optional_digest("parent_decision_sha256"),
        "raw_learning_rates_by_parameter": _validate_parameter_rates(
            entry.get("raw_learning_rates_by_parameter"),
            "entry.raw_learning_rates_by_parameter",
        ),
        "minibatch_sha256": _digest(
            entry.get("minibatch_sha256"), "entry.minibatch_sha256"
        ),
        "checkpoint_sha256": _digest(
            entry.get("checkpoint_sha256"), "entry.checkpoint_sha256"
        ),
        "code_fingerprint": _digest(
            entry.get("code_fingerprint"), "entry.code_fingerprint"
        ),
    }


def assert_result_matches_entry(
    result: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed when a result belongs to stale execution provenance."""

    normalized = validate_result(result)
    expected = execution_entry_provenance(entry)
    observed = {
        key: normalized.get(key)
        for key in expected
    }
    if observed != expected:
        raise RuntimeError(
            "Expected result provenance to match the current plan entry "
            "exactly. Provided value: "
            f"expected={expected!r}, observed={observed!r}."
        )
    return normalized


def completion_action(
    entry_dir: str | Path,
    *,
    optimizer: str,
    expected_entry_id: str,
    expected_entry: Mapping[str, Any] | None = None,
) -> str:
    """Return a safe resume action without deleting an incomplete attempt."""

    if optimizer not in OPTIMIZERS:
        raise _error(f"one of {OPTIMIZERS!r}", optimizer, "optimizer")
    root = Path(entry_dir)
    completion_path = root / "complete.json"
    if completion_path.is_file():
        completion = read_json(completion_path)
        if completion.get("schema_version") != COMPLETION_SCHEMA_VERSION:
            raise _error(
                f"schema {COMPLETION_SCHEMA_VERSION!r}",
                completion.get("schema_version"),
                "completion.schema_version",
            )
        if (
            completion.get("state") not in {"complete", "safety_failure"}
            or completion.get("entry_id") != expected_entry_id
            or completion.get("official_test_read") is not False
        ):
            raise _error(
                "a matching terminal completion record", completion, "completion"
            )
        if expected_entry is None:
            raise _error(
                "the current plan entry when validating a completion",
                expected_entry,
                "expected_entry",
            )
        expected_provenance = execution_entry_provenance(expected_entry)
        expected_provenance_sha256 = sha256_json(expected_provenance)
        if (
            completion.get("entry_provenance") != expected_provenance
            or completion.get("entry_provenance_sha256")
            != expected_provenance_sha256
        ):
            raise RuntimeError(
                "Expected immutable completion provenance to match the "
                "current plan entry exactly; refusing stale coordinate reuse."
            )
        outputs = completion.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise _error("a non-empty output list", outputs, "completion.outputs")
        for index, record in enumerate(outputs):
            if not isinstance(record, Mapping) or set(record) != {
                "path",
                "sha256",
                "bytes",
            }:
                raise _error(
                    "a path/hash/size record",
                    record,
                    f"completion.outputs[{index}]",
                )
            relative = _portable_relative(
                record["path"], f"completion.outputs[{index}].path"
            )
            target = root / relative
            observed = (
                (sha256_file(target), target.stat().st_size)
                if target.is_file() and not target.is_symlink()
                else (None, None)
            )
            expected = (
                _digest(
                    record["sha256"], f"completion.outputs[{index}].sha256"
                ),
                record["bytes"],
            )
            if observed != expected:
                raise RuntimeError(
                    "Expected every immutable completed output to match its hash. "
                    f"Provided value: path={target}, expected={expected!r}, "
                    f"observed={observed!r}."
                )
        return "skip_immutable_complete"
    if root.exists() and any(root.iterdir()):
        return (
            "restart_from_initialization"
            if optimizer == "adam"
            else "restart_or_resume_sgd_under_executor_contract"
        )
    return "run_from_initialization"


def terminal_completion(
    *,
    entry_id_value: str,
    result: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    entry: Mapping[str, Any],
    result_relative_path: str = "result.json",
) -> dict[str, Any]:
    normalized = assert_result_matches_entry(result, entry)
    state = (
        "complete" if normalized["status"] == "complete" else "safety_failure"
    )
    records = sorted((dict(record) for record in outputs), key=lambda item: item["path"])
    relative_result = _portable_relative(
        result_relative_path, "result_relative_path"
    )
    if relative_result not in {record["path"] for record in records}:
        raise _error(
            "result_relative_path to identify one immutable output",
            relative_result,
            "result_relative_path",
        )
    entry_provenance = execution_entry_provenance(entry)
    return {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "state": state,
        "entry_id": entry_id_value,
        "entry_provenance": entry_provenance,
        "entry_provenance_sha256": sha256_json(entry_provenance),
        "result_relative_path": relative_result,
        "completed_steps": normalized["completed_steps"],
        "safety_failure": normalized.get("safety_failure"),
        "optimizer_parameters": normalized["optimizer_parameters"],
        "probe_sha256": normalized["probe_sha256"],
        "bias_q90_probe_sha256": normalized["bias_q90_probe_sha256"],
        "normalization_unit_by_weight": normalized[
            "normalization_unit_by_weight"
        ],
        "bias_q90_unit_by_parameter": normalized[
            "bias_q90_unit_by_parameter"
        ],
        "parent_sha256": normalized["parent_sha256"],
        "parent_study_config_sha256": normalized[
            "parent_study_config_sha256"
        ],
        "parent_entry_completion_sha256": normalized[
            "parent_entry_completion_sha256"
        ],
        "parent_entry_id": normalized["parent_entry_id"],
        "parent_result_sha256": normalized["parent_result_sha256"],
        "parent_decision_sha256": normalized["parent_decision_sha256"],
        "raw_learning_rates_by_parameter": normalized[
            "raw_learning_rates_by_parameter"
        ],
        "minibatch_sha256": normalized["minibatch_sha256"],
        "checkpoint_sha256": normalized["checkpoint_sha256"],
        "code_fingerprint": normalized["code_fingerprint"],
        "outputs": records,
        "official_test_read": False,
    }


def pack_entries(
    entries: Sequence[Mapping[str, Any]],
    *,
    scheme: str,
    concurrency: int,
    stage: str,
    benchmark_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create deterministic one-GPU waves for one scheme and stage."""

    if scheme not in SCHEMES:
        raise _error(f"one of {SCHEMES!r}", scheme, "scheme")
    if stage not in STAGES:
        raise _error(f"one of {STAGES!r}", stage, "stage")
    if type(concurrency) is not int or concurrency not in CONCURRENCY_LEVELS:
        raise _error(
            f"one of {CONCURRENCY_LEVELS!r}", concurrency, "concurrency"
        )
    selected = [
        dict(entry)
        for entry in entries
        if entry.get("scheme") == scheme
        and entry.get("stage") == stage
        and entry.get("mode") == "new_five_epoch_training"
    ]
    if stage == "main_grid" and len(selected) != 10:
        raise _error(
            "exactly ten new main entries per scheme",
            len(selected),
            "entries",
        )
    waves = [
        {
            "wave_index": wave_index,
            "entry_ids": [
                entry["entry_id"]
                for entry in selected[start : start + concurrency]
            ],
        }
        for wave_index, start in enumerate(range(0, len(selected), concurrency))
    ]
    scientific = {
        "schema_version": PACK_SCHEMA_VERSION,
        "scheme": scheme,
        "stage": stage,
        "concurrency": concurrency,
        "entry_count": len(selected),
        "waves": waves,
        "resource_contract": {
            "account": "fmu@v100",
            "partition": "gpu_p13",
            "qos": "qos_gpu-t3",
            "constraint": "v100-32g",
            "gpus": 1,
            "cpus": 16,
            "walltime_hours": 20,
        },
        "official_test_read": False,
    }
    if benchmark_binding is not None:
        required = {
            "benchmark_id",
            "benchmark_sha256",
            "manifest_id",
            "manifest_sha256",
            "study_id",
            "study_config_sha256",
            "selected_concurrency",
        }
        if set(benchmark_binding) != required:
            raise _error(
                f"benchmark binding fields {sorted(required)!r}",
                sorted(benchmark_binding),
                "benchmark_binding",
            )
        if benchmark_binding["selected_concurrency"] != concurrency:
            raise _error(
                "benchmark-selected concurrency to equal pack concurrency",
                {
                    "selected": benchmark_binding["selected_concurrency"],
                    "pack": concurrency,
                },
                "benchmark_binding",
            )
        scientific["benchmark_binding"] = {
            key: benchmark_binding[key] for key in sorted(required)
        }
    return {**scientific, "pack_id": "lrpack_" + sha256_json(scientific)}


def select_concurrency(
    levels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pick the largest safe width retaining at least 90% of best throughput."""

    if [level.get("concurrency") for level in levels] != list(CONCURRENCY_LEVELS):
        raise _error(
            f"levels ordered as {list(CONCURRENCY_LEVELS)!r}",
            [level.get("concurrency") for level in levels],
            "levels",
        )
    normalized = []
    for level in levels:
        concurrency = int(level["concurrency"])
        failures = level.get("failed_children")
        if type(failures) is not int or failures < 0:
            raise _error(
                "a non-negative integer",
                failures,
                f"level[{concurrency}].failed_children",
            )
        memory = _finite(
            level.get("peak_gpu_memory_mib"),
            f"level[{concurrency}].peak_gpu_memory_mib",
        )
        throughput = _finite(
            level.get("aggregate_steps_per_second"),
            f"level[{concurrency}].aggregate_steps_per_second",
        )
        normalized.append(
            {
                "concurrency": concurrency,
                "failed_children": failures,
                "peak_gpu_memory_mib": memory,
                "aggregate_steps_per_second": throughput,
            }
        )
    safe = [
        level
        for level in normalized
        if level["failed_children"] == 0
        and level["peak_gpu_memory_mib"]
        <= GPU_CAPACITY_MIB * (1.0 - MEMORY_HEADROOM_FRACTION)
        and level["aggregate_steps_per_second"] > 0.0
    ]
    if not safe:
        raise RuntimeError(
            "Expected at least one zero-failure concurrency level to fit a "
            "V100-32GB with 10% memory headroom."
        )
    best_throughput = max(
        level["aggregate_steps_per_second"] for level in safe
    )
    eligible = [
        level
        for level in safe
        if level["aggregate_steps_per_second"]
        >= THROUGHPUT_FRACTION * best_throughput
    ]
    selected = max(eligible, key=lambda level: level["concurrency"])
    return {
        "selected_concurrency": selected["concurrency"],
        "selected_level": selected,
        "best_safe_throughput": best_throughput,
        "memory_headroom_fraction": MEMORY_HEADROOM_FRACTION,
        "throughput_fraction": THROUGHPUT_FRACTION,
        "levels": normalized,
    }


__all__ = [
    "BIAS_RHO_CAP",
    "COMPLETION_SCHEMA_VERSION",
    "CONCURRENCY_LEVELS",
    "EPOCHS",
    "MANIFEST_SCHEMA_VERSION",
    "OPTIMIZERS",
    "PACK_SCHEMA_VERSION",
    "PLATEAU_RELATIVE_TOLERANCE",
    "PROTOCOL_ID",
    "REPLAY_BATCHES",
    "RHO_CONV_MAIN",
    "RHO_CONV_SENTINEL",
    "RHO_DENSE_BY_SCHEME",
    "ROW_IDS",
    "RUN_SCHEMA_VERSION",
    "SCHEMES",
    "SELECTION_SCHEMA_VERSION",
    "STAGES",
    "STEPS_PER_EPOCH",
    "STUDY_SCHEMA_VERSION",
    "TOTAL_STEPS",
    "arm_id",
    "artifact_record",
    "bias_capped_confirmation_entries",
    "build_main_manifest",
    "capped_bias_learning_rates",
    "completion_action",
    "entry_id",
    "load_main_manifest",
    "main_grid_entries",
    "pack_entries",
    "publish_once",
    "reuse_entry_ids",
    "select_all_arms",
    "select_arm",
    "select_concurrency",
    "terminal_completion",
    "upper_sentinel_entries",
    "validate_result",
    "verify_artifact_binding",
    "verify_reuse_bindings",
]
