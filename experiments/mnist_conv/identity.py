"""Canonical JSON and content fingerprints for Conv runs and sweeps."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from .specs import ExpandedRun, RunSpec, SweepSpec


CODE_FINGERPRINT_SCHEMA = "mnist-conv-code/v1"
CODE_IDENTITY_SCHEMA = "mnist-conv-code-identity/v1"
RUN_ID_SCHEMA = "mnist-conv-run-id/v1"
SWEEP_ID_SCHEMA = "mnist-conv-sweep-id/v1"


def _identity_error(expected: str, provided: Any, path: str) -> ValueError:
    return ValueError(f"Expected {path} to be {expected}. Provided value: {provided!r}.")


def _is_json_pointer(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("/") and value != "/"


def _validate_sweep_identity_parts(
    entries: Any,
    varying_fields: Any,
    collection: Any,
) -> None:
    """Validate the complete comparison contract before it is hashed.

    This deliberately excludes the sweep's display name and all filesystem
    paths.  A copied manifest and a manifest generated under another display
    name therefore retain the same scientific identity.
    """

    if not isinstance(varying_fields, list):
        raise _identity_error("a list of unique JSON pointers", varying_fields, "varying_fields")
    if (
        any(not _is_json_pointer(pointer) for pointer in varying_fields)
        or len(varying_fields) != len(set(varying_fields))
        or varying_fields != sorted(varying_fields)
    ):
        raise _identity_error(
            "a sorted list of unique non-root JSON pointers",
            varying_fields,
            "varying_fields",
        )

    collection_keys = {"expected_seeds", "required_cases", "group_by"}
    if not isinstance(collection, dict) or set(collection) != collection_keys:
        raise _identity_error(
            f"an object with exactly keys {sorted(collection_keys)!r}",
            collection,
            "collection",
        )
    expected_seeds = collection["expected_seeds"]
    if (
        not isinstance(expected_seeds, list)
        or not expected_seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in expected_seeds)
        or len(expected_seeds) != len(set(expected_seeds))
    ):
        raise _identity_error(
            "a non-empty list of unique non-negative integers",
            expected_seeds,
            "collection.expected_seeds",
        )
    required_cases = collection["required_cases"]
    if (
        not isinstance(required_cases, list)
        or not required_cases
        or any(not isinstance(case_id, str) or not case_id.strip() for case_id in required_cases)
        or len(required_cases) != len(set(required_cases))
    ):
        raise _identity_error(
            "a non-empty list of unique non-empty case ids",
            required_cases,
            "collection.required_cases",
        )
    group_by = collection["group_by"]
    if (
        not isinstance(group_by, list)
        or any(not _is_json_pointer(pointer) for pointer in group_by)
        or len(group_by) != len(set(group_by))
        or set(group_by) & {"/seed", "/replicate_id"}
    ):
        raise _identity_error(
            "a list of unique non-root JSON pointers excluding /seed and /replicate_id",
            group_by,
            "collection.group_by",
        )

    if not isinstance(entries, list) or not entries:
        raise _identity_error("a non-empty list", entries, "entries")
    entry_keys = {"logical_key", "case_id", "axes", "run_id"}
    seen_logical_keys: set[str] = set()
    seen_run_ids: set[str] = set()
    case_order: list[str] = []
    closed_cases: set[str] = set()
    previous_case: str | None = None
    expected_axis_paths: list[str] | None = None
    for index, entry in enumerate(entries):
        path = f"entries[{index}]"
        if not isinstance(entry, dict) or set(entry) != entry_keys:
            raise _identity_error(
                f"an object with exactly keys {sorted(entry_keys)!r}",
                entry,
                path,
            )
        case_id = entry["case_id"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise _identity_error("a non-empty string", case_id, f"{path}.case_id")
        if case_id not in required_cases:
            raise _identity_error(
                "one of collection.required_cases",
                case_id,
                f"{path}.case_id",
            )
        if case_id != previous_case:
            if case_id in closed_cases:
                raise _identity_error(
                    "entries grouped contiguously in required-case order",
                    case_id,
                    f"{path}.case_id",
                )
            if previous_case is not None:
                closed_cases.add(previous_case)
            case_order.append(case_id)
            previous_case = case_id

        axes = entry["axes"]
        if not isinstance(axes, dict):
            raise _identity_error("an object", axes, f"{path}.axes")
        axis_paths = sorted(axes)
        if any(not _is_json_pointer(pointer) for pointer in axis_paths):
            raise _identity_error("an object keyed by non-root JSON pointers", axes, f"{path}.axes")
        if any(pointer not in varying_fields for pointer in axis_paths):
            raise _identity_error(
                "axes whose paths are declared by varying_fields",
                axis_paths,
                f"{path}.axes",
            )
        if expected_axis_paths is None:
            expected_axis_paths = axis_paths
        elif axis_paths != expected_axis_paths:
            raise _identity_error(
                f"the common axis paths {expected_axis_paths!r}",
                axis_paths,
                f"{path}.axes",
            )

        expected_logical_key = canonical_json_bytes(
            {"case_id": case_id, "axes": axes}
        ).decode("utf-8")
        logical_key = entry["logical_key"]
        if logical_key != expected_logical_key:
            raise _identity_error(expected_logical_key, logical_key, f"{path}.logical_key")
        if logical_key in seen_logical_keys:
            raise _identity_error("a unique logical key", logical_key, f"{path}.logical_key")
        seen_logical_keys.add(logical_key)

        run_id = entry["run_id"]
        if (
            not isinstance(run_id, str)
            or not run_id.startswith("run_")
            or len(run_id) != 68
            or any(character not in "0123456789abcdef" for character in run_id[4:])
        ):
            raise _identity_error("a canonical run SHA-256 id", run_id, f"{path}.run_id")
        if run_id in seen_run_ids:
            raise _identity_error("a unique run id", run_id, f"{path}.run_id")
        seen_run_ids.add(run_id)

    if case_order != required_cases:
        raise _identity_error(
            f"entries grouped in required-case order {required_cases!r}",
            case_order,
            "entries",
        )


def _reject_non_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(
            f"Expected canonical JSON to contain only finite numbers. Provided value at {path}: {value!r}."
        )
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes suitable for hashing."""

    _reject_non_finite(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_paths(repo_root: Path) -> Iterable[Path]:
    explicit = [
        repo_root / "labs" / "mnist_train.py",
        repo_root / "labs" / "custom_classes.py",
        repo_root / "labs" / "custom_minimizer.py",
        repo_root / "labs" / "datasets.py",
        repo_root
        / "experiments"
        / "run_mnist_conv_lr_conv1_scheme_rho_sweep.py",
        repo_root
        / "experiments"
        / "run_mnist_conv_lr_conv1_scheme_rho_sweep_jeanzay.slurm",
        repo_root
        / "experiments"
        / "run_mnist_conv_lr_conv2_scheme_rho_sweep.py",
        repo_root
        / "experiments"
        / "run_mnist_conv_lr_conv2_scheme_rho_sweep_jeanzay.slurm",
        repo_root
        / "experiments"
        / "run_mnist_conv_lr_optimizer_boundary.py",
        repo_root
        / "experiments"
        / "run_mnist_conv_lr_optimizer_boundary_jeanzay.slurm",
        repo_root
        / "experiments"
        / "run_mnist_conv_lr_optimizer_boundary_benchmark_jeanzay.slurm",
        repo_root
        / "experiments"
        / "submit_mnist_conv_lr_optimizer_boundary_jeanzay.py",
        repo_root
        / "experiments"
        / "run_mnist_conv_lr_stage_slurm.sh",
        repo_root
        / "experiments"
        / "supervise_mnist_conv_lr_v7_jeanzay.py",
    ]
    for path in explicit:
        if path.is_file():
            yield path
    for directory in (
        repo_root / "experiments" / "mnist_conv",
        repo_root / "model",
        repo_root / "training",
    ):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" not in path.parts:
                yield path


def code_fingerprint(repo_root: str | Path | None = None) -> str:
    """Hash the active Conv engine and canonical runner sources.

    The digest is based on relative file names and contents, not timestamps or
    absolute checkout paths.  It therefore works for the same source copied to
    local machines and Slurm scratch checkouts, including an uncommitted new
    runner during development.
    """

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    digest = hashlib.sha256()
    digest.update(CODE_FINGERPRINT_SCHEMA.encode("utf-8"))
    paths = sorted(set(_code_paths(root)), key=lambda item: item.relative_to(root).as_posix())
    if not paths:
        raise ValueError(f"Expected code files below repository root. Provided value: {root}.")
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = path.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def code_provenance(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Expose the effective content identity and its Git provenance."""

    root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    effective = code_fingerprint(root)
    try:
        revision_result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        revision = revision_result.stdout.strip()
        tracked_diff = subprocess.run(
            [
                "git", "-C", str(root), "diff", "--binary", "HEAD", "--",
                "model", "training", "labs/mnist_train.py", "labs/custom_classes.py",
                "labs/custom_minimizer.py", "labs/datasets.py", "experiments/mnist_conv",
            ],
            check=True,
            capture_output=True,
        ).stdout
        untracked_result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
            check=True,
            capture_output=True,
            text=True,
        )
        untracked = []
        for raw in untracked_result.stdout.splitlines():
            path = root / raw
            if path.is_file() and (
                raw.startswith("experiments/mnist_conv/")
                or raw.startswith("model/")
                or raw.startswith("training/")
                or raw in {
                    "labs/mnist_train.py", "labs/custom_classes.py",
                    "labs/custom_minimizer.py", "labs/datasets.py",
                }
            ):
                untracked.append(path)
        dirty = hashlib.sha256()
        dirty.update(tracked_diff)
        for path in sorted(untracked):
            relative = path.relative_to(root).as_posix().encode()
            dirty.update(relative + b"\0" + path.read_bytes())
        dirty_source_digest = dirty.hexdigest() if tracked_diff or untracked else None
    except (OSError, subprocess.CalledProcessError):
        revision = None
        dirty_source_digest = effective
    return {
        "git_revision": revision,
        "dirty_source_digest": dirty_source_digest,
        "effective_code_fingerprint": effective,
    }


def normalize_code_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete source identity used by run and sweep hashes."""

    if not isinstance(value, Mapping):
        raise ValueError(f"Expected code provenance to be an object. Provided value: {value!r}.")
    required = {"git_revision", "dirty_source_digest", "effective_code_fingerprint"}
    if set(value) != required:
        raise ValueError(
            f"Expected code provenance keys {sorted(required)!r}. Provided value: {sorted(value)!r}."
        )
    revision = value["git_revision"]
    dirty = value["dirty_source_digest"]
    effective = value["effective_code_fingerprint"]
    if revision is not None and (
        not isinstance(revision, str)
        or len(revision) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError(f"Expected git_revision to be null or a lowercase Git object id. Provided value: {revision!r}.")
    for name, digest in (("dirty_source_digest", dirty), ("effective_code_fingerprint", effective)):
        if digest is None and name == "dirty_source_digest":
            continue
        if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Expected {name} to be {'null or ' if name == 'dirty_source_digest' else ''}a lowercase SHA-256 digest. Provided value: {digest!r}.")
    return {
        "git_revision": revision,
        "dirty_source_digest": dirty,
        "effective_code_fingerprint": effective,
    }


def code_identity(source_provenance: Mapping[str, Any]) -> str:
    normalized = normalize_code_provenance(source_provenance)
    return "code_" + sha256_json(
        {"schema_version": CODE_IDENTITY_SCHEMA, "provenance": normalized}
    )


def run_fingerprint(spec: RunSpec, source_provenance: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": RUN_ID_SCHEMA,
        "code_identity": code_identity(source_provenance),
        "run_spec": spec.identity_payload(),
    }
    return "run_" + sha256_json(payload)


def sweep_fingerprint(
    spec: SweepSpec,
    expanded: list[ExpandedRun],
    run_ids: list[str],
    source_provenance: Mapping[str, Any],
) -> str:
    if len(expanded) != len(run_ids):
        raise ValueError(
            "Expected one run id per expanded run. "
            f"Provided expanded={len(expanded)}, run_ids={len(run_ids)}."
        )
    entries = [
        {
            "logical_key": item.logical_key,
            "case_id": item.case_id,
            "axes": item.axes,
            "run_id": run_id,
        }
        for item, run_id in zip(expanded, run_ids)
    ]
    return sweep_entries_fingerprint(
        entries,
        varying_fields=spec.data["varying_fields"],
        collection=spec.data["collection"],
        source_provenance=source_provenance,
    )


def sweep_entries_fingerprint(
    entries: list[dict[str, Any]],
    *,
    varying_fields: list[str],
    collection: dict[str, Any],
    source_provenance: Mapping[str, Any],
) -> str:
    """Hash the immutable logical job plan represented in a sweep manifest."""

    provenance = normalize_code_provenance(source_provenance)
    _validate_sweep_identity_parts(entries, varying_fields, collection)
    payload = {
        "schema_version": SWEEP_ID_SCHEMA,
        "code_identity": code_identity(provenance),
        "varying_fields": varying_fields,
        "collection": collection,
        "entries": entries,
    }
    return "sweep_" + sha256_json(payload)
