"""Numerical execution adapter for the Conv3 perfect-diode LR-v2 study.

The immutable v2 specification owns scientific identity and routing.  This
module owns the numerical surface lifecycle while deliberately reusing the
tested Conv1/Conv2 v1 kernels for:

* optimizer-specific 32/64/128 restored probes;
* restarted 640-step canaries;
* restarted, exact 10,314-step three-epoch candidates;
* the inclusive selection rule and the single boundary expansion.

One surface is always executed on exactly one manifest-bound host.  Every
stage writes a hash-verified completion marker last.  Conditional work emits
an explicit zero-work result, and finalization fails closed until a
post-training T/K audit has passed for all three selected epoch checkpoints.
The official MNIST test split is never instantiated.
"""

from __future__ import annotations

import copy
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .identity import code_provenance, sha256_file, sha256_json
from .io import atomic_write_json, read_json
from .lr_engine import (
    build_loader_bundle,
    build_model_runtime,
    evaluate_validation,
    parameter_tensor_digest,
)
from .perfectdiode_conv3_hparam_v2_spec import (
    ALLOWED_HOSTS,
    CANARY_STEPS,
    CANDIDATE_TOTAL_STEPS,
    EXECUTION_AUTHORITY_KEYS,
    MAXIMUM_CELLS_PER_SURFACE,
    PD_CONV3_SURFACE_MANIFEST_SCHEMA_VERSION,
    PUBLIC_STAGE_SEQUENCE,
    SHARED_ASSET_HASH_KEYS,
    PerfectDiodeConv3HparamStudySpec,
    representative_preflight_plan,
    surface_rho_policy,
    validate_surface_manifest,
)
from .perfectdiode_hparam_runtime import (
    PerfectDiodeRuntimeError,
    _materialize_train_batches,
    _optimizer_contract,
    _save_checkpoint_atomic,
    _write_step_log,
    derive_raw_learning_rates,
    measure_adaptive_optimizer_probe,
    require_official_test_excluded,
    rho_cell_id,
    run_canary,
    run_epoch_training,
)
from .perfectdiode_hparam_spec import (
    candidate_grid,
    core_grid,
    select_surface_candidates,
)
from .perfectdiode_tk_runtime import (
    LoadedTKAssets,
    PerfectDiodeTKRuntimeError,
    TK_ASSET_COMPLETION_SCHEMA_VERSION,
    TK_ASSET_SCHEMA_VERSION,
    TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION,
    _collect_gradients_for_k,
    compare_k_gradients,
    load_shared_assets as load_tk_shared_assets,
    parameters_in_contract_order,
    run_operating_point_audit,
    validate_shared_assets as validate_tk_shared_assets,
)
from .perfectdiode_tk_spec import (
    CONV3_CONV_WEIGHTS,
    K_BOUNDARY_SENTINEL,
    K_BATCH_SIZE,
    K_EXAMPLES,
    K_REFERENCE,
    T_REFERENCE,
    PerfectDiodeTKStudySpec,
    _candidate_k_pass,
    _candidate_t_pass,
)


EXECUTION_STAGE_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-execution-stage/v2"
)
EXECUTION_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-execution-completion/v2"
)
CELL_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-cell-completion/v2"
)
POST_TK_AUDIT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-post-training-tk-audit/v2"
)
EPOCH_K_VIABILITY_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-epoch-k-reference-viability/v2"
)
SURFACE_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-surface-completion/v2"
)
PREFLIGHT_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-preflight-completion/v2"
)
FIXED_TK_GATE_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-fixed-tk-gate-completion/v1"
)
FIXED_TK_GATE_RESULT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-fixed-tk-gate-result/v1"
)
ENVIRONMENT_RECEIPT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-environment-receipt/v2"
)
ENVIRONMENT_CONTRACT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-hparam-environment-contract/v2"
)
WORKER_LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "run_mnist_conv_perfectdiode_hparam_v2_worker.py"
)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TK_CONFIG = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_conv3_tk_ordinary_mnist_v1.json"
)


class PerfectDiodeConv3ExecutionError(RuntimeError):
    """Raised when an LR-v2 execution or artifact fails closed."""


class NumericalBackend(Protocol):
    """Injectable numerical boundary used by focused contract tests."""

    def load_assets(
        self,
        context: "ExecutionContext",
        *,
        asset_dir: Path,
        data_root: Path,
        download: bool,
        device: str,
    ) -> Any: ...

    def optimizer_probe(
        self,
        context: "ExecutionContext",
        assets: Any,
        *,
        output_dir: Path,
        device: str,
    ) -> Mapping[str, Any]: ...

    def canary(
        self,
        context: "ExecutionContext",
        assets: Any,
        probe: Mapping[str, Any],
        *,
        rho_conv: float,
        rho_dense: float,
        output_dir: Path,
        device: str,
    ) -> Mapping[str, Any]: ...

    def candidate(
        self,
        context: "ExecutionContext",
        assets: Any,
        probe: Mapping[str, Any],
        *,
        rho_conv: float,
        rho_dense: float,
        output_dir: Path,
        device: str,
    ) -> Mapping[str, Any]: ...


def _error(path: str, expected: str, provided: Any) -> ValueError:
    return ValueError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "a mapping", value)
    return copy.deepcopy(dict(value))


def _finite_positive(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise _error(path, "a positive finite number", value)
    return float(value)


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(path, "a lowercase SHA-256 digest", value)
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(path, "a non-empty string", value)
    return value


def _nonnegative_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _error(path, "a non-negative integer", value)
    return int(value)


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, path)


def validate_environment_receipt(
    raw: Mapping[str, Any],
    *,
    expected_host: str | None = None,
) -> dict[str, Any]:
    """Validate one exact tmux/Python/Torch/CUDA host receipt."""

    value = _mapping(raw, "environment receipt")
    expected_keys = {
        "schema_version",
        "execution_backend",
        "execution_host",
        "tmux",
        "python",
        "torch",
        "cuda",
        "official_test_read",
    }
    if set(value) != expected_keys:
        raise _error(
            "environment receipt",
            f"an object with exactly keys {sorted(expected_keys)!r}",
            {
                "missing": sorted(expected_keys - set(value)),
                "extra": sorted(set(value) - expected_keys),
            },
        )
    if value.get("schema_version") != ENVIRONMENT_RECEIPT_SCHEMA_VERSION:
        raise _error(
            "environment receipt.schema_version",
            f"exactly {ENVIRONMENT_RECEIPT_SCHEMA_VERSION!r}",
            value.get("schema_version"),
        )
    if value.get("execution_backend") != "tmux":
        raise _error(
            "environment receipt.execution_backend",
            "exactly 'tmux'",
            value.get("execution_backend"),
        )
    host = value.get("execution_host")
    if host not in ALLOWED_HOSTS:
        raise _error(
            "environment receipt.execution_host",
            f"one of {ALLOWED_HOSTS!r}",
            host,
        )
    if expected_host is not None and host != expected_host:
        raise _error(
            "environment receipt.execution_host",
            f"exactly {expected_host!r}",
            host,
        )
    if value.get("official_test_read") is not False:
        raise _error(
            "environment receipt.official_test_read",
            "exactly false",
            value.get("official_test_read"),
        )

    tmux = _mapping(value.get("tmux"), "environment receipt.tmux")
    if set(tmux) != {
        "attestation_mode",
        "session_name",
        "pane_token",
    }:
        raise _error(
            "environment receipt.tmux",
            "exactly attestation_mode, session_name, and pane_token",
            tmux,
        )
    attestation_mode = tmux.get("attestation_mode")
    if attestation_mode not in {
        "direct_tmux_session",
        "delegated_remote_tmux_session",
    }:
        raise _error(
            "environment receipt.tmux.attestation_mode",
            "'direct_tmux_session' or 'delegated_remote_tmux_session'",
            attestation_mode,
        )
    if host == "main" and attestation_mode != "direct_tmux_session":
        raise _error(
            "environment receipt.tmux.attestation_mode",
            "exactly 'direct_tmux_session' for main",
            attestation_mode,
        )
    if (
        host == "akibscomputer"
        and attestation_mode != "delegated_remote_tmux_session"
    ):
        raise _error(
            "environment receipt.tmux.attestation_mode",
            "exactly 'delegated_remote_tmux_session' for akibscomputer",
            attestation_mode,
        )
    session_name = _nonempty_string(
        tmux.get("session_name"), "environment receipt.tmux.session_name"
    )
    if session_name != host:
        raise _error(
            "environment receipt.tmux.session_name",
            f"exactly the execution host {host!r}",
            session_name,
        )
    pane_token = _nonempty_string(
        tmux.get("pane_token"), "environment receipt.tmux.pane_token"
    )
    if not pane_token.startswith("%"):
        raise _error(
            "environment receipt.tmux.pane_token",
            "a tmux pane id beginning with '%'",
            pane_token,
        )

    python = _mapping(value.get("python"), "environment receipt.python")
    if set(python) != {"implementation", "version", "executable"}:
        raise _error(
            "environment receipt.python",
            "exactly implementation, version, and executable",
            python,
        )
    _nonempty_string(
        python.get("implementation"),
        "environment receipt.python.implementation",
    )
    _nonempty_string(
        python.get("version"), "environment receipt.python.version"
    )
    executable = Path(
        _nonempty_string(
            python.get("executable"),
            "environment receipt.python.executable",
        )
    )
    if not executable.is_absolute():
        raise _error(
            "environment receipt.python.executable",
            "an absolute path",
            str(executable),
        )

    torch = _mapping(value.get("torch"), "environment receipt.torch")
    if set(torch) != {
        "version",
        "cuda_build_version",
        "cudnn_version",
    }:
        raise _error(
            "environment receipt.torch",
            "exactly version, cuda_build_version, and cudnn_version",
            torch,
        )
    _nonempty_string(
        torch.get("version"), "environment receipt.torch.version"
    )
    _optional_string(
        torch.get("cuda_build_version"),
        "environment receipt.torch.cuda_build_version",
    )
    cudnn_version = torch.get("cudnn_version")
    if cudnn_version is not None:
        _nonnegative_integer(
            cudnn_version, "environment receipt.torch.cudnn_version"
        )

    cuda = _mapping(value.get("cuda"), "environment receipt.cuda")
    if set(cuda) != {
        "available",
        "visible_devices",
        "device_count",
        "devices",
    }:
        raise _error(
            "environment receipt.cuda",
            "exactly available, visible_devices, device_count, and devices",
            cuda,
        )
    available = cuda.get("available")
    if not isinstance(available, bool):
        raise _error(
            "environment receipt.cuda.available", "a boolean", available
        )
    visible_devices = cuda.get("visible_devices")
    if visible_devices is not None and not isinstance(
        visible_devices, str
    ):
        raise _error(
            "environment receipt.cuda.visible_devices",
            "a string or null",
            visible_devices,
        )
    device_count = _nonnegative_integer(
        cuda.get("device_count"),
        "environment receipt.cuda.device_count",
    )
    devices = cuda.get("devices")
    if not isinstance(devices, list) or len(devices) != device_count:
        raise _error(
            "environment receipt.cuda.devices",
            f"a list of length {device_count}",
            devices,
        )
    if available != (device_count > 0):
        raise _error(
            "environment receipt.cuda",
            "available=true if and only if device_count is positive",
            {"available": available, "device_count": device_count},
        )
    for index, raw_device in enumerate(devices):
        device = _mapping(
            raw_device, f"environment receipt.cuda.devices[{index}]"
        )
        if set(device) != {
            "index",
            "name",
            "compute_capability",
            "total_memory_bytes",
        }:
            raise _error(
                f"environment receipt.cuda.devices[{index}]",
                "exactly index, name, compute_capability, and "
                "total_memory_bytes",
                device,
            )
        if device.get("index") != index:
            raise _error(
                f"environment receipt.cuda.devices[{index}].index",
                f"exactly {index}",
                device.get("index"),
            )
        _nonempty_string(
            device.get("name"),
            f"environment receipt.cuda.devices[{index}].name",
        )
        capability = device.get("compute_capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                for item in capability
            )
        ):
            raise _error(
                f"environment receipt.cuda.devices[{index}]."
                "compute_capability",
                "two non-negative integers",
                capability,
            )
        memory = _nonnegative_integer(
            device.get("total_memory_bytes"),
            f"environment receipt.cuda.devices[{index}]."
            "total_memory_bytes",
        )
        if memory == 0:
            raise _error(
                f"environment receipt.cuda.devices[{index}]."
                "total_memory_bytes",
                "a positive integer",
                memory,
            )
    return value


def observe_execution_environment(host: str) -> dict[str, Any]:
    """Observe the current tmux lane and exact Python/Torch/CUDA semantics."""

    if host not in ALLOWED_HOSTS:
        raise _error("host", f"one of {ALLOWED_HOSTS!r}", host)
    tmux_value = os.environ.get("TMUX")
    pane_id = os.environ.get("TMUX_PANE")
    if tmux_value and pane_id:
        try:
            completed = subprocess.run(
                [
                    "tmux",
                    "display-message",
                    "-p",
                    "-t",
                    pane_id,
                    "#{session_name}",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise PerfectDiodeConv3ExecutionError(
                "Expected the current tmux session name to be observable."
            ) from exc
        session_name = completed.stdout.strip()
        pane_token = pane_id
        attestation_mode = "direct_tmux_session"
    elif host == "akibscomputer":
        session_name = os.environ.get("PD_LR_TMUX_SESSION", "")
        pane_token = os.environ.get("PD_LR_TMUX_PANE_TOKEN", "")
        attestation_mode = "delegated_remote_tmux_session"
        if session_name != host or not pane_token.startswith("%"):
            raise PerfectDiodeConv3ExecutionError(
                "Expected remote Akib execution to carry the versioned "
                "runner's delegated lane attestation: "
                "PD_LR_TMUX_SESSION=akibscomputer and a '%' prefixed "
                "PD_LR_TMUX_PANE_TOKEN."
            )
    else:
        raise PerfectDiodeConv3ExecutionError(
            "Expected main environment capture and execution to run inside "
            "the actual tmux main session."
        )
    if session_name != host:
        raise PerfectDiodeConv3ExecutionError(
            "Expected the current tmux session to match the requested host. "
            f"Expected {host!r}; observed {session_name!r}."
        )

    try:
        import torch
    except Exception as exc:
        raise PerfectDiodeConv3ExecutionError(
            "Expected PyTorch to import while capturing the execution environment."
        ) from exc
    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if available else 0
    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        capability = torch.cuda.get_device_capability(index)
        devices.append(
            {
                "index": index,
                "name": str(properties.name),
                "compute_capability": [
                    int(capability[0]),
                    int(capability[1]),
                ],
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    cudnn_version = torch.backends.cudnn.version()
    receipt = {
        "schema_version": ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
        "execution_backend": "tmux",
        "execution_host": host,
        "tmux": {
            "attestation_mode": attestation_mode,
            "session_name": session_name,
            "pane_token": pane_token,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": sys.version,
            "executable": str(Path(sys.executable).resolve()),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_build_version": (
                None if torch.version.cuda is None else str(torch.version.cuda)
            ),
            "cudnn_version": (
                None if cudnn_version is None else int(cudnn_version)
            ),
        },
        "cuda": {
            "available": available,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_count": device_count,
            "devices": devices,
        },
        "official_test_read": False,
    }
    return validate_environment_receipt(receipt, expected_host=host)


def build_environment_contract(
    main_receipt: Mapping[str, Any],
    akibscomputer_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the canonical two-lane environment contract."""

    receipts = {
        "main": validate_environment_receipt(
            main_receipt, expected_host="main"
        ),
        "akibscomputer": validate_environment_receipt(
            akibscomputer_receipt, expected_host="akibscomputer"
        ),
    }
    return {
        "schema_version": ENVIRONMENT_CONTRACT_SCHEMA_VERSION,
        "execution_backend": "tmux",
        "hosts": list(ALLOWED_HOSTS),
        "exact_live_receipt_match_required": True,
        "host_runtime_policy": {
            "mode": "host_specific_exact_receipts",
            "cross_host_software_equality_required": False,
            "same_surface_stays_on_bound_host": True,
        },
        "host_receipts": copy.deepcopy(receipts),
        "host_receipt_content_sha256s": {
            host: sha256_json(receipts[host]) for host in ALLOWED_HOSTS
        },
        "official_test_read": False,
    }


def validate_environment_contract(
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and canonically reconstruct a two-lane contract."""

    value = _mapping(raw, "environment contract")
    expected_keys = {
        "schema_version",
        "execution_backend",
        "hosts",
        "exact_live_receipt_match_required",
        "host_runtime_policy",
        "host_receipts",
        "host_receipt_content_sha256s",
        "official_test_read",
    }
    if set(value) != expected_keys:
        raise _error(
            "environment contract",
            f"an object with exactly keys {sorted(expected_keys)!r}",
            {
                "missing": sorted(expected_keys - set(value)),
                "extra": sorted(set(value) - expected_keys),
            },
        )
    if value.get("schema_version") != ENVIRONMENT_CONTRACT_SCHEMA_VERSION:
        raise _error(
            "environment contract.schema_version",
            f"exactly {ENVIRONMENT_CONTRACT_SCHEMA_VERSION!r}",
            value.get("schema_version"),
        )
    receipts = _mapping(
        value.get("host_receipts"), "environment contract.host_receipts"
    )
    if set(receipts) != set(ALLOWED_HOSTS):
        raise _error(
            "environment contract.host_receipts",
            f"exactly {ALLOWED_HOSTS!r}",
            sorted(receipts),
        )
    expected = build_environment_contract(
        receipts["main"], receipts["akibscomputer"]
    )
    if value != expected:
        raise _error(
            "environment contract",
            "the canonical contract reconstructed from its two receipts",
            value,
        )
    return expected


def verify_current_execution_environment(
    *,
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    host: str,
) -> dict[str, Any]:
    """Verify contract, bound host receipt, and a fresh live observation."""

    validated_contract = validate_environment_contract(contract)
    validated_receipt = validate_environment_receipt(
        receipt, expected_host=host
    )
    expected_receipt = validated_contract["host_receipts"][host]
    if validated_receipt != expected_receipt:
        raise PerfectDiodeConv3ExecutionError(
            "Expected the execution receipt semantics to equal the "
            "contract-bound host receipt."
        )
    observed = observe_execution_environment(host)
    if observed != validated_receipt:
        raise PerfectDiodeConv3ExecutionError(
            "Expected the freshly observed tmux/Python/Torch/CUDA environment "
            "to equal the manifest-bound receipt. "
            f"Expected digest {sha256_json(validated_receipt)!r}; "
            f"observed digest {sha256_json(observed)!r}."
        )
    return observed


def derive_shared_asset_hashes(
    *,
    study_path: str | Path,
    data_root: str | Path,
    download: bool = False,
) -> dict[str, str]:
    """Derive all nine manifest hashes from copied T/K assets and loaders."""

    resolved_study = Path(study_path).expanduser().resolve()
    if (
        not resolved_study.is_file()
        or resolved_study.name != "study.resolved.json"
    ):
        raise _error(
            "study_path",
            "an existing file named study.resolved.json",
            resolved_study,
        )
    spec = PerfectDiodeConv3HparamStudySpec.from_path(resolved_study)
    upstream = _mapping(
        spec.data.get("upstream_tk"), "study.upstream_tk"
    )
    asset_dir = _resolve_relative(
        resolved_study.parent,
        upstream.get("shared_assets_artifact"),
        "study.upstream_tk.shared_assets_artifact",
    )
    tk_spec = PerfectDiodeTKStudySpec.from_path(DEFAULT_TK_CONFIG)
    if (
        tk_spec.study_id != upstream.get("study_id")
        or tk_spec.config_sha256 != upstream.get("config_sha256")
    ):
        raise PerfectDiodeConv3ExecutionError(
            "Expected the shared-asset derivation to use the exact frozen "
            "upstream T/K study."
        )
    metadata = validate_tk_shared_assets(tk_spec, asset_dir)
    loaded = load_tk_shared_assets(
        tk_spec,
        asset_dir=asset_dir,
        data_root=Path(data_root).expanduser().resolve(),
        download=download,
    )
    bundle = build_loader_bundle(
        spec.data,
        data_root=Path(data_root).expanduser().resolve(),
        download=download,
        return_source_indices=True,
    )
    if (
        tuple(bundle.train_indices)
        != tuple(metadata["split"]["train_indices"])
        or tuple(bundle.validation_indices)
        != tuple(metadata["split"]["validation_indices"])
        or bundle.train_indices_hash != loaded.train_indices_sha256
        or bundle.validation_indices_hash
        != loaded.validation_indices_sha256
    ):
        raise PerfectDiodeConv3ExecutionError(
            "Expected the actual LR loader to reproduce the copied T/K "
            "train/validation split exactly."
        )
    epoch_hashes = tuple(
        str(value)
        for value in bundle.train_batch_order_hashes(num_epochs=3)
    )
    if len(epoch_hashes) != 3:
        raise _error(
            "actual loader three-epoch order",
            "exactly three SHA-256 values",
            epoch_hashes,
        )
    result = {
        "train_indices_sha256": loaded.train_indices_sha256,
        "validation_indices_sha256": loaded.validation_indices_sha256,
        "initialization_checkpoint_sha256": loaded.checkpoint_sha256,
        "initialization_tensor_sha256": loaded.parameter_tensor_sha256,
        "t_cohort_indices_sha256": loaded.t_indices_sha256,
        "k_cohort_indices_sha256": loaded.k_indices_sha256,
        "train_batch_order_epoch_1_sha256": epoch_hashes[0],
        "train_batch_order_epoch_2_sha256": epoch_hashes[1],
        "train_batch_order_epoch_3_sha256": epoch_hashes[2],
    }
    if set(result) != set(SHARED_ASSET_HASH_KEYS):
        raise AssertionError("shared-asset hash derivation key drift")
    return {
        key: _sha256(result[key], f"shared_asset_hashes.{key}")
        for key in SHARED_ASSET_HASH_KEYS
    }


def _relative_path(value: Any, path: str) -> Path:
    if not isinstance(value, str) or not value:
        raise _error(path, "a non-empty relative path", value)
    result = Path(value)
    if result.is_absolute() or ".." in result.parts:
        raise _error(path, "a traversal-free relative path", value)
    return result


def _resolve_relative(root: Path, value: Any, path: str) -> Path:
    relative = _relative_path(value, path)
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise _error(path, "a path within the manifest root", value) from exc
    return resolved


def _artifact(path: Path, *, base: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(base).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _all_artifacts(root: Path, *, exclude: Sequence[Path] = ()) -> list[dict[str, Any]]:
    excluded = {path.resolve() for path in exclude}
    return [
        _artifact(path, base=root)
        for path in sorted(
            (
                item
                for item in root.rglob("*")
                if item.is_file() and item.resolve() not in excluded
            ),
            key=lambda item: item.relative_to(root).as_posix(),
        )
    ]


def _validate_artifacts(
    root: Path, records: Any, *, path: str
) -> None:
    if not isinstance(records, list):
        raise _error(path, "a list of artifact records", records)
    observed_paths: set[str] = set()
    for index, raw in enumerate(records):
        record = _mapping(raw, f"{path}[{index}]")
        relative = _relative_path(
            record.get("path"), f"{path}[{index}].path"
        )
        name = relative.as_posix()
        if name in observed_paths:
            raise _error(path, "unique artifact paths", name)
        observed_paths.add(name)
        artifact_path = _resolve_relative(
            root, name, f"{path}[{index}].path"
        )
        if (
            not artifact_path.is_file()
            or sha256_file(artifact_path)
            != _sha256(record.get("sha256"), f"{path}[{index}].sha256")
            or artifact_path.stat().st_size != record.get("bytes")
        ):
            raise PerfectDiodeConv3ExecutionError(
                "Expected hash- and size-verified artifact. "
                f"Provided value: {artifact_path}."
            )


def _require_official_test_false_recursive(
    value: Any, *, path: str
) -> None:
    if isinstance(value, Mapping):
        if (
            "official_test_read" in value
            and value.get("official_test_read") is not False
        ):
            raise _error(
                f"{path}.official_test_read",
                "exactly false",
                value.get("official_test_read"),
            )
        for key, item in value.items():
            _require_official_test_false_recursive(
                item, path=f"{path}.{key}"
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_official_test_false_recursive(
                item, path=f"{path}[{index}]"
            )


@dataclass(frozen=True)
class ExecutionContext:
    """One manifest-bound Conv3 scheme x optimizer surface."""

    spec: PerfectDiodeConv3HparamStudySpec
    manifest: dict[str, Any]
    manifest_path: Path
    surface: dict[str, Any]
    row: dict[str, Any]
    surface_root: Path
    host: str
    observed_authority: dict[str, str]

    @property
    def surface_id(self) -> str:
        return str(self.surface["surface_id"])

    @property
    def manifest_id(self) -> str:
        return str(self.manifest["manifest_id"])


@dataclass(frozen=True)
class LoadedV2Assets:
    """Validated imported T/K assets plus the v2 ordinary-MNIST loader."""

    asset_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    parameter_tensor_sha256: str
    bundle: Any
    asset_metadata: dict[str, Any]
    epoch_order_sha256s: tuple[str, str, str]
    tk_spec: PerfectDiodeTKStudySpec
    tk_assets: LoadedTKAssets


def _contract_parameter_tensor_digest(parameters: Sequence[Any]) -> str:
    """Hash Conv3 parameters in the frozen T/K scientific name order."""

    return parameter_tensor_digest(parameters_in_contract_order(parameters))


def _row_for_surface(
    spec: PerfectDiodeConv3HparamStudySpec, surface: Mapping[str, Any]
) -> dict[str, Any]:
    rows = [
        row
        for row in spec.rows
        if row.get("row_id") == surface.get("row_id")
    ]
    if len(rows) != 1:
        raise _error(
            "surface.row_id",
            "exactly one row in the resolved v2 study",
            surface.get("row_id"),
        )
    return copy.deepcopy(rows[0])


def _validate_surface(
    spec: PerfectDiodeConv3HparamStudySpec,
    manifest: Mapping[str, Any],
    surface: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(surface, "surface")
    row = _row_for_surface(spec, value)
    surface_id = f"{row['row_id']}--{value.get('optimizer')}"
    expected = {
        "surface_id": surface_id,
        "architecture": "conv3",
        "scheme": row["scheme"],
        "host": row["upstream_tk"]["execution_host"],
        "inference_iterations": row["inference_iterations"],
        "training_iterations": row["training_iterations"],
        "input_gain": 360.0,
        "lr_eligible": bool(row["lr_eligible"]),
        "zero_work": not bool(row["lr_eligible"]),
        "zero_work_reason": row["zero_work_reason"],
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise _error(
                f"surface.{key}", f"exactly {expected_value!r}", value.get(key)
            )
    if value.get("optimizer") not in {"sgd", "adam"}:
        raise _error(
            "surface.optimizer", "'sgd' or 'adam'", value.get("optimizer")
        )
    if (
        value.get("canary_steps_per_cell") != CANARY_STEPS
        or value.get("candidate_total_steps") != CANDIDATE_TOTAL_STEPS
        or value.get("maximum_cells") != MAXIMUM_CELLS_PER_SURFACE
        or value.get("maximum_expansion_waves") != 1
        or value.get("long_confirmation") is not False
    ):
        raise _error(
            "surface numerical limits",
            "640-step canaries, 10,314-step candidates, <=16 cells, one "
            "expansion, and no long confirmation",
            value,
        )
    if value.get("shared_asset_hashes") != manifest.get(
        "shared_asset_hashes"
    ):
        raise _error(
            "surface.shared_asset_hashes",
            "the manifest-level shared asset hashes",
            value.get("shared_asset_hashes"),
        )
    if value.get("execution_source") != manifest.get("execution_source"):
        raise _error(
            "surface.execution_source",
            "the manifest-level execution source",
            value.get("execution_source"),
        )
    source = _mapping(value["execution_source"], "surface.execution_source")
    host_environments = _mapping(
        source.get("host_environment_sha256s"),
        "surface.execution_source.host_environment_sha256s",
    )
    if value.get("execution_environment_sha256") != host_environments.get(
        value["host"]
    ):
        raise _error(
            "surface.execution_environment_sha256",
            "the environment SHA-256 assigned to the surface host",
            value.get("execution_environment_sha256"),
        )
    expected_output = f"surfaces/{surface_id}"
    for key, expected_path in (
        ("output_path", expected_output),
        ("completion_path", f"{expected_output}/completion.json"),
        (
            "finalization_path",
            f"{expected_output}/stages/finalize_lr/result.json",
        ),
    ):
        if value.get(key) != expected_path:
            raise _error(
                f"surface.{key}", f"exactly {expected_path!r}", value.get(key)
            )
    expected_policy = (
        surface_rho_policy(spec, row["scheme"])
        if value["lr_eligible"]
        else None
    )
    if value.get("rho_policy") != expected_policy:
        raise _error(
            "surface.rho_policy",
            f"exactly the resolved template-derived policy {expected_policy!r}",
            value.get("rho_policy"),
        )
    return value


def load_execution_context(
    *,
    study_path: str | Path,
    manifest_path: str | Path,
    surface_id: str,
    host: str,
    source_archive_path: str | Path,
    environment_contract_path: str | Path,
    execution_environment_path: str | Path,
) -> ExecutionContext:
    """Load and verify one surface plus immutable execution authority."""

    if host not in ALLOWED_HOSTS:
        raise _error("host", f"one of {ALLOWED_HOSTS!r}", host)
    provided_study = Path(study_path).expanduser().resolve()
    if not provided_study.is_file():
        raise _error(
            "study_path",
            "an existing resolved-study JSON file",
            provided_study,
        )
    spec = PerfectDiodeConv3HparamStudySpec.from_path(provided_study)
    source_manifest = Path(manifest_path).expanduser().resolve()
    manifest = validate_surface_manifest(
        read_json(source_manifest),
        spec=spec,
    )
    if (
        manifest.get("schema_version")
        != PD_CONV3_SURFACE_MANIFEST_SCHEMA_VERSION
        or manifest.get("public_stage_sequence") != list(PUBLIC_STAGE_SEQUENCE)
        or manifest.get("relative_paths_resolve_from")
        != "surface_manifest_parent"
    ):
        raise _error(
            "surface manifest",
            "the exact Conv3 LR-v2 public-stage and relative-path contract",
            {
                "schema_version": manifest.get("schema_version"),
                "public_stage_sequence": manifest.get(
                    "public_stage_sequence"
                ),
                "relative_paths_resolve_from": manifest.get(
                    "relative_paths_resolve_from"
                ),
            },
        )
    root = source_manifest.parent
    source = _mapping(manifest.get("execution_source"), "execution_source")
    if set(source) != set(EXECUTION_AUTHORITY_KEYS):
        raise _error(
            "execution_source",
            f"exactly {EXECUTION_AUTHORITY_KEYS!r}",
            {
                "missing": sorted(set(EXECUTION_AUTHORITY_KEYS) - set(source)),
                "extra": sorted(set(source) - set(EXECUTION_AUTHORITY_KEYS)),
            },
        )
    if (
        not isinstance(source.get("source_commit"), str)
        or len(source["source_commit"]) not in {40, 64}
        or any(
            character not in "0123456789abcdef"
            for character in source["source_commit"]
        )
    ):
        raise _error(
            "execution_source.source_commit",
            "a lowercase Git object id",
            source.get("source_commit"),
        )
    for key in (
        "source_archive_sha256",
        "effective_code_fingerprint",
        "worker_launcher_sha256",
        "environment_contract_sha256",
        "resolved_study_sha256",
    ):
        _sha256(source.get(key), f"execution_source.{key}")
    host_environments = _mapping(
        source.get("host_environment_sha256s"),
        "execution_source.host_environment_sha256s",
    )
    if set(host_environments) != set(ALLOWED_HOSTS):
        raise _error(
            "execution_source.host_environment_sha256s",
            f"exactly {ALLOWED_HOSTS!r}",
            sorted(host_environments),
        )
    for lane in ALLOWED_HOSTS:
        _sha256(
            host_environments[lane],
            f"execution_source.host_environment_sha256s.{lane}",
        )
    shared = _mapping(
        manifest.get("shared_asset_hashes"), "shared_asset_hashes"
    )
    if set(shared) != set(SHARED_ASSET_HASH_KEYS):
        raise _error(
            "shared_asset_hashes",
            f"exactly {SHARED_ASSET_HASH_KEYS!r}",
            sorted(shared),
        )
    for key in SHARED_ASSET_HASH_KEYS:
        _sha256(shared[key], f"shared_asset_hashes.{key}")
    resolved_study = _resolve_relative(
        root,
        source.get("resolved_study_path"),
        "execution_source.resolved_study_path",
    )
    if provided_study != resolved_study:
        raise _error(
            "study_path",
            f"the manifest-bound resolved study {resolved_study}",
            provided_study,
        )
    if sha256_file(provided_study) != source.get("resolved_study_sha256"):
        raise PerfectDiodeConv3ExecutionError(
            "Expected the resolved study file SHA-256 to match execution authority."
        )
    if (
        spec.study_id != manifest.get("study_id")
        or spec.config_sha256 != manifest.get("config_sha256")
    ):
        raise PerfectDiodeConv3ExecutionError(
            "Expected the surface manifest to match the exact resolved study."
        )
    normalized_surfaces = [
        _validate_surface(spec, manifest, value)
        for value in manifest["surfaces"]
    ]
    identifiers = [value["surface_id"] for value in normalized_surfaces]
    indexes = [value.get("surface_index") for value in normalized_surfaces]
    if len(set(identifiers)) != 6 or indexes != list(range(6)):
        raise _error(
            "surface manifest.surfaces",
            "six unique surfaces indexed 0 through 5",
            {"surface_ids": identifiers, "surface_indexes": indexes},
        )
    surfaces = [
        value
        for value in normalized_surfaces
        if value.get("surface_id") == surface_id
    ]
    if len(surfaces) != 1:
        raise _error(
            "surface_id",
            "exactly one manifest surface",
            surface_id,
        )
    surface = _validate_surface(spec, manifest, surfaces[0])
    if surface["host"] != host:
        raise _error(
            "host",
            f"the manifest-bound host {surface['host']!r}",
            host,
        )
    if not WORKER_LAUNCHER.is_file():
        raise PerfectDiodeConv3ExecutionError(
            f"Expected the v2 worker launcher to exist: {WORKER_LAUNCHER}."
        )
    if sha256_file(WORKER_LAUNCHER) != source.get("worker_launcher_sha256"):
        raise PerfectDiodeConv3ExecutionError(
            "Expected the running worker launcher SHA-256 to match execution authority."
        )
    provenance = code_provenance(REPO_ROOT)
    if provenance.get("dirty_source_digest") is not None:
        raise PerfectDiodeConv3ExecutionError(
            "Expected a clean source checkout with dirty_source_digest=null. "
            f"Provided value: {provenance.get('dirty_source_digest')!r}."
        )
    archive = Path(source_archive_path).expanduser().resolve()
    environment_contract = Path(environment_contract_path).expanduser().resolve()
    execution_environment = (
        Path(execution_environment_path).expanduser().resolve()
    )
    for label, path in (
        ("source archive", archive),
        ("environment contract receipt", environment_contract),
        ("execution environment receipt", execution_environment),
    ):
        if not path.is_file():
            raise _error(label, "an existing regular file", path)
    environment_contract_value = _mapping(
        read_json(environment_contract), "environment contract"
    )
    environment_receipt = _mapping(
        read_json(execution_environment), "execution environment receipt"
    )
    observed = {
        "source_commit": provenance.get("git_revision"),
        "source_archive_sha256": sha256_file(archive),
        "effective_code_fingerprint": provenance.get(
            "effective_code_fingerprint"
        ),
        "environment_contract_sha256": sha256_file(environment_contract),
        "execution_environment_sha256": sha256_file(execution_environment),
    }
    expected_observed = {
        "source_commit": source["source_commit"],
        "source_archive_sha256": source["source_archive_sha256"],
        "effective_code_fingerprint": source["effective_code_fingerprint"],
        "environment_contract_sha256": source[
            "environment_contract_sha256"
        ],
        "execution_environment_sha256": surface[
            "execution_environment_sha256"
        ],
    }
    if observed != expected_observed:
        raise _error(
            "observed execution authority",
            f"exactly {expected_observed!r}",
            observed,
        )
    verify_current_execution_environment(
        contract=environment_contract_value,
        receipt=environment_receipt,
        host=host,
    )
    surface_root = _resolve_relative(
        root, surface["output_path"], "surface.output_path"
    )
    return ExecutionContext(
        spec=spec,
        manifest=manifest,
        manifest_path=source_manifest,
        surface=surface,
        row=_row_for_surface(spec, surface),
        surface_root=surface_root,
        host=host,
        observed_authority={key: str(value) for key, value in observed.items()},
    )


def _base_result(
    context: ExecutionContext,
    stage: str,
    *,
    status: str,
    zero_work: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_STAGE_SCHEMA_VERSION,
        "study_id": context.spec.study_id,
        "config_sha256": context.spec.config_sha256,
        "manifest_id": context.manifest_id,
        "stage": stage,
        "surface_id": context.surface_id,
        "row_id": context.row["row_id"],
        "architecture": "conv3",
        "scheme": context.surface["scheme"],
        "optimizer": context.surface["optimizer"],
        "host": context.host,
        "execution_environment_sha256": context.surface[
            "execution_environment_sha256"
        ],
        "status": status,
        "zero_work": bool(zero_work),
        "reason": reason,
        "official_test_read": False,
    }


def _zero_work(
    context: ExecutionContext,
    stage: str,
    reason: str,
    *,
    status: str = "zero_work",
    dependency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = _base_result(
        context,
        stage,
        status=status,
        zero_work=True,
        reason=reason,
    )
    if dependency is not None:
        result["dependency"] = copy.deepcopy(dict(dependency))
    return result


def _stage_dir(context: ExecutionContext, stage: str) -> Path:
    return context.surface_root / "stages" / stage


def _validate_completion(
    context: ExecutionContext,
    stage: str,
    stage_dir: Path,
) -> dict[str, Any]:
    completion_path = stage_dir / "completion.json"
    result_path = stage_dir / "result.json"
    if not completion_path.is_file() or not result_path.is_file():
        raise PerfectDiodeConv3ExecutionError(
            f"Expected a complete immutable stage: {stage_dir}."
        )
    completion = _mapping(
        read_json(completion_path), f"{stage} completion"
    )
    expected = {
        "schema_version": EXECUTION_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": context.spec.study_id,
        "config_sha256": context.spec.config_sha256,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "stage": stage,
        "official_test_read": False,
    }
    for key, expected_value in expected.items():
        if completion.get(key) != expected_value:
            raise _error(
                f"{stage} completion.{key}",
                f"exactly {expected_value!r}",
                completion.get(key),
            )
    _validate_artifacts(
        stage_dir, completion.get("outputs"), path=f"{stage} completion.outputs"
    )
    result = _mapping(read_json(result_path), f"{stage} result")
    for key in (
        "schema_version",
        "study_id",
        "config_sha256",
        "manifest_id",
        "surface_id",
        "stage",
        "official_test_read",
    ):
        expected_value = (
            EXECUTION_STAGE_SCHEMA_VERSION
            if key == "schema_version"
            else False
            if key == "official_test_read"
            else getattr(context.spec, key)
            if key in {"study_id", "config_sha256"}
            else context.manifest_id
            if key == "manifest_id"
            else context.surface_id
            if key == "surface_id"
            else stage
        )
        if result.get(key) != expected_value:
            raise _error(
                f"{stage} result.{key}",
                f"exactly {expected_value!r}",
                result.get(key),
            )
    return result


def _load_stage_result(
    context: ExecutionContext, stage: str
) -> dict[str, Any]:
    return _validate_completion(context, stage, _stage_dir(context, stage))


def _publish_stage(
    context: ExecutionContext,
    stage: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    destination = _stage_dir(context, stage)
    completion_path = destination / "completion.json"
    result_path = destination / "result.json"
    if completion_path.exists():
        return _validate_completion(context, stage, destination)
    if result_path.exists():
        raise PerfectDiodeConv3ExecutionError(
            "Expected an interrupted stage with result.json but no completion "
            f"to be inspected before retry. Provided value: {destination}."
        )
    normalized = _mapping(result, f"{stage} result")
    require_official_test_excluded(normalized, require_result_marker=True)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_path, normalized, canonical=True)
    completion = {
        "schema_version": EXECUTION_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": context.spec.study_id,
        "config_sha256": context.spec.config_sha256,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "stage": stage,
        "official_test_read": False,
        "outputs": _all_artifacts(
            destination, exclude=(completion_path,)
        ),
    }
    atomic_write_json(completion_path, completion, canonical=True)
    return _validate_completion(context, stage, destination)


def _cell_dir(
    context: ExecutionContext, stage: str, cell_id: str
) -> Path:
    return _stage_dir(context, stage) / "cells" / cell_id


def _validate_cell_completion(
    context: ExecutionContext,
    stage: str,
    cell_id: str,
    destination: Path,
) -> dict[str, Any]:
    completion_path = destination / "completion.json"
    result_path = destination / "result.json"
    if not completion_path.is_file() or not result_path.is_file():
        raise PerfectDiodeConv3ExecutionError(
            f"Expected a complete immutable cell: {destination}."
        )
    completion = _mapping(read_json(completion_path), "cell completion")
    expected = {
        "schema_version": CELL_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "stage": stage,
        "cell_id": cell_id,
        "official_test_read": False,
    }
    for key, expected_value in expected.items():
        if completion.get(key) != expected_value:
            raise _error(
                f"cell completion.{key}",
                f"exactly {expected_value!r}",
                completion.get(key),
            )
    _validate_artifacts(
        destination,
        completion.get("outputs"),
        path="cell completion.outputs",
    )
    result = _mapping(read_json(result_path), "cell result")
    if (
        result.get("cell_id") != cell_id
        or result.get("surface_id") != context.surface_id
        or result.get("official_test_read") is not False
    ):
        raise PerfectDiodeConv3ExecutionError(
            f"Expected a matching official-test-free cell result: {result_path}."
        )
    return result


def _publish_cell(
    context: ExecutionContext,
    stage: str,
    cell_id: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    destination = _cell_dir(context, stage, cell_id)
    completion_path = destination / "completion.json"
    result_path = destination / "result.json"
    if completion_path.exists():
        return _validate_cell_completion(
            context, stage, cell_id, destination
        )
    if result_path.exists():
        raise PerfectDiodeConv3ExecutionError(
            "Expected an interrupted cell with result.json but no completion "
            f"to be inspected before retry. Provided value: {destination}."
        )
    destination.mkdir(parents=True, exist_ok=True)
    normalized = _mapping(result, "cell result")
    require_official_test_excluded(normalized, require_result_marker=True)
    atomic_write_json(result_path, normalized, canonical=True)
    completion = {
        "schema_version": CELL_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "stage": stage,
        "cell_id": cell_id,
        "official_test_read": False,
        "outputs": _all_artifacts(
            destination, exclude=(completion_path,)
        ),
    }
    atomic_write_json(completion_path, completion, canonical=True)
    return _validate_cell_completion(context, stage, cell_id, destination)


def _read_imported_assets(asset_dir: Path) -> tuple[dict[str, Any], Path]:
    assets_path = asset_dir / "assets.json"
    completion_path = asset_dir / "completion.json"
    if not assets_path.is_file() or not completion_path.is_file():
        raise PerfectDiodeConv3ExecutionError(
            f"Expected a complete imported T/K asset bundle: {asset_dir}."
        )
    assets = _mapping(read_json(assets_path), "T/K assets")
    completion = _mapping(read_json(completion_path), "T/K asset completion")
    if (
        assets.get("schema_version") != TK_ASSET_SCHEMA_VERSION
        or assets.get("architecture") != "conv3"
        or assets.get("official_test_read") is not False
        or completion.get("schema_version")
        != TK_ASSET_COMPLETION_SCHEMA_VERSION
        or completion.get("state") != "complete"
        or completion.get("study_id") != assets.get("study_id")
        or completion.get("config_sha256") != assets.get("config_sha256")
        or completion.get("official_test_read") is not False
    ):
        raise PerfectDiodeConv3ExecutionError(
            "Expected a complete Conv3 T/K asset bundle excluding the official test set."
        )
    _validate_artifacts(
        asset_dir,
        completion.get("outputs"),
        path="T/K asset completion.outputs",
    )
    checkpoint = _resolve_relative(
        asset_dir,
        assets.get("initialization", {}).get("path"),
        "T/K assets.initialization.path",
    )
    return assets, checkpoint


class V1NumericalBackend:
    """Thin Conv3 adapter around the tested perfect-diode v1 kernels."""

    def load_assets(
        self,
        context: ExecutionContext,
        *,
        asset_dir: Path,
        data_root: Path,
        download: bool,
        device: str,
    ) -> LoadedV2Assets:
        root = asset_dir.expanduser().resolve()
        metadata, checkpoint = _read_imported_assets(root)
        upstream = _mapping(
            context.spec.data.get("upstream_tk"), "study.upstream_tk"
        )
        if (
            metadata.get("study_id") != upstream.get("study_id")
            or metadata.get("config_sha256") != upstream.get("config_sha256")
        ):
            raise PerfectDiodeConv3ExecutionError(
                "Expected imported assets from the exact upstream T/K study."
            )
        expected = context.surface["shared_asset_hashes"]
        initialization = _mapping(
            metadata.get("initialization"), "T/K assets.initialization"
        )
        split = _mapping(metadata.get("split"), "T/K assets.split")
        cohorts = _mapping(metadata.get("cohorts"), "T/K assets.cohorts")
        observed_static = {
            "train_indices_sha256": split.get("train_indices_sha256"),
            "validation_indices_sha256": split.get(
                "validation_indices_sha256"
            ),
            "initialization_checkpoint_sha256": initialization.get("sha256"),
            "initialization_tensor_sha256": initialization.get(
                "parameter_tensor_sha256"
            ),
            "t_cohort_indices_sha256": cohorts.get("t", {}).get(
                "source_indices_sha256"
            ),
            "k_cohort_indices_sha256": cohorts.get("k", {}).get(
                "source_indices_sha256"
            ),
        }
        for key, value in observed_static.items():
            if value != expected.get(key):
                raise _error(
                    f"imported assets.{key}",
                    f"the manifest-bound digest {expected.get(key)!r}",
                    value,
                )
        if sha256_file(checkpoint) != expected[
            "initialization_checkpoint_sha256"
        ]:
            raise PerfectDiodeConv3ExecutionError(
                "Expected the imported initialization checkpoint SHA-256 to verify."
            )
        bundle = build_loader_bundle(
            context.spec.data,
            data_root=data_root,
            download=download,
            return_source_indices=True,
        )
        if (
            tuple(bundle.train_indices) != tuple(split.get("train_indices", ()))
            or tuple(bundle.validation_indices)
            != tuple(split.get("validation_indices", ()))
            or bundle.train_indices_hash
            != expected["train_indices_sha256"]
            or bundle.validation_indices_hash
            != expected["validation_indices_sha256"]
        ):
            raise PerfectDiodeConv3ExecutionError(
                "Expected regenerated ordinary-MNIST split indices to match imported assets."
            )
        epoch_hashes = tuple(
            str(value)
            for value in bundle.train_batch_order_hashes(num_epochs=3)
        )
        expected_epochs = tuple(
            expected[f"train_batch_order_epoch_{epoch}_sha256"]
            for epoch in range(1, 4)
        )
        if epoch_hashes != expected_epochs:
            raise _error(
                "regenerated three-epoch batch order",
                f"the manifest-bound order {expected_epochs!r}",
                epoch_hashes,
            )
        tk_spec = PerfectDiodeTKStudySpec.from_path(DEFAULT_TK_CONFIG)
        if (
            tk_spec.study_id != upstream.get("study_id")
            or tk_spec.config_sha256 != upstream.get("config_sha256")
        ):
            raise PerfectDiodeConv3ExecutionError(
                "Expected the frozen K-cohort replay to use the exact "
                "manifest-bound upstream T/K study."
            )
        tk_assets = load_tk_shared_assets(
            tk_spec,
            asset_dir=root,
            data_root=data_root,
            download=download,
        )
        loaded = LoadedV2Assets(
            asset_dir=root,
            checkpoint_path=checkpoint,
            checkpoint_sha256=expected[
                "initialization_checkpoint_sha256"
            ],
            parameter_tensor_sha256=expected[
                "initialization_tensor_sha256"
            ],
            bundle=bundle,
            asset_metadata=metadata,
            epoch_order_sha256s=(
                epoch_hashes[0],
                epoch_hashes[1],
                epoch_hashes[2],
            ),
            tk_spec=tk_spec,
            tk_assets=tk_assets,
        )
        runtime = self._fresh_runtime(
            context,
            loaded,
            learning_rate=1.0,
            device=device,
        )
        del runtime
        return loaded

    def _fresh_runtime(
        self,
        context: ExecutionContext,
        assets: LoadedV2Assets,
        *,
        learning_rate: Any,
        device: str,
    ) -> Any:
        runtime = build_model_runtime(
            context.spec.data,
            context.row,
            device=device,
            initialization_checkpoint=assets.checkpoint_path,
            learning_rate=learning_rate,
            optimizer_contract=_optimizer_contract(
                context.spec.data, context.surface["optimizer"]
            ),
        )
        digest = _contract_parameter_tensor_digest(runtime.parameters)
        if digest != assets.parameter_tensor_sha256:
            raise PerfectDiodeConv3ExecutionError(
                "Expected every numerical entry to restart from the shared "
                "seed-0 Conv3 initialization with parameter tensor SHA-256 "
                f"{assets.parameter_tensor_sha256!r}. Provided value: "
                f"{digest!r}."
            )
        return runtime

    def optimizer_probe(
        self,
        context: ExecutionContext,
        assets: LoadedV2Assets,
        *,
        output_dir: Path,
        device: str,
    ) -> Mapping[str, Any]:
        del output_dir
        runtime = self._fresh_runtime(
            context, assets, learning_rate=1.0, device=device
        )
        batches = _materialize_train_batches(assets.bundle, count=128)
        result = measure_adaptive_optimizer_probe(runtime, batches)
        del runtime
        result.update(
            {
                "study_id": context.spec.study_id,
                "config_sha256": context.spec.config_sha256,
                "manifest_id": context.manifest_id,
                "surface_id": context.surface_id,
                "row_id": context.row["row_id"],
                "architecture": "conv3",
                "scheme": context.surface["scheme"],
                "optimizer": context.surface["optimizer"],
                "initialization_checkpoint_sha256": assets.checkpoint_sha256,
                "initialization_tensor_sha256": (
                    assets.parameter_tensor_sha256
                ),
            }
        )
        return result

    def canary(
        self,
        context: ExecutionContext,
        assets: LoadedV2Assets,
        probe: Mapping[str, Any],
        *,
        rho_conv: float,
        rho_dense: float,
        output_dir: Path,
        device: str,
    ) -> Mapping[str, Any]:
        rates = derive_raw_learning_rates(
            probe, rho_conv=rho_conv, rho_dense=rho_dense
        )
        runtime = self._fresh_runtime(
            context, assets, learning_rate=rates, device=device
        )
        assets.bundle.reset_train_shuffle()
        expected_batches = assets.bundle.train_batch_indices(num_epochs=1)[0][
            :CANARY_STEPS
        ]
        raw = run_canary(
            runtime,
            assets.bundle.train_loader,
            learning_rates_by_parameter=rates,
            expected_source_indices=expected_batches,
        )
        del runtime
        rows = raw.pop("step_rows")
        _write_step_log(
            output_dir / "steps.csv",
            (
                {
                    **row,
                    "cell_id": rho_cell_id(
                        study_id=context.spec.study_id,
                        surface_id=context.surface_id,
                        rho_conv=rho_conv,
                        rho_dense=rho_dense,
                    ),
                    "phase": "canary",
                }
                for row in rows
            ),
        )
        raw.update(
            {
                "architecture": "conv3",
                "raw_learning_rates_by_parameter": rates,
            }
        )
        return raw

    def candidate(
        self,
        context: ExecutionContext,
        assets: LoadedV2Assets,
        probe: Mapping[str, Any],
        *,
        rho_conv: float,
        rho_dense: float,
        output_dir: Path,
        device: str,
    ) -> Mapping[str, Any]:
        rates = derive_raw_learning_rates(
            probe, rho_conv=rho_conv, rho_dense=rho_dense
        )
        runtime = self._fresh_runtime(
            context, assets, learning_rate=rates, device=device
        )
        epoch_checkpoints: list[dict[str, Any]] = []
        epoch_k_viability: list[dict[str, Any]] = []
        tk_rows = [
            row
            for row in assets.tk_spec.rows
            if row.get("scheme") == context.surface["scheme"]
        ]
        if len(tk_rows) != 1:
            raise _error(
                "epochwise K replay row",
                f"one T/K row for scheme {context.surface['scheme']!r}",
                tk_rows,
            )
        tk_row = tk_rows[0]

        def validation_with_checkpoint(
            current_runtime: Any, validation_loader: Any
        ) -> Mapping[str, Any]:
            validation = evaluate_validation(
                current_runtime, validation_loader
            )
            epoch = len(epoch_checkpoints) + 1
            path = output_dir / f"epoch_{epoch:03d}.pt"
            _save_checkpoint_atomic(current_runtime, path)
            epoch_checkpoints.append(
                {
                    "epoch": epoch,
                    "path": path.name,
                    "sha256": sha256_file(path),
                }
            )
            checkpoint_sha = epoch_checkpoints[-1]["sha256"]
            tensor_sha = _contract_parameter_tensor_digest(
                current_runtime.parameters
            )
            trained_assets = replace(
                assets.tk_assets,
                checkpoint=path,
                checkpoint_sha256=checkpoint_sha,
                parameter_tensor_sha256=tensor_sha,
            )
            try:
                reference = _collect_gradients_for_k(
                    assets.tk_spec,
                    tk_row,
                    selected_t=int(context.surface["inference_iterations"]),
                    training_iterations=K_REFERENCE,
                    assets=trained_assets,
                    device=device,
                )
                measurement = compare_k_gradients(
                    reference,
                    reference,
                    candidate_k=K_REFERENCE,
                    reference_k=K_REFERENCE,
                )
                measurement.update(
                    {
                        "cohort_examples": K_EXAMPLES,
                        "batch_size": K_BATCH_SIZE,
                        "official_test_read": False,
                        "cohort_source_indices_sha256": (
                            trained_assets.k_indices_sha256
                        ),
                        "initialization_checkpoint_sha256": checkpoint_sha,
                        "initialization_tensor_sha256": tensor_sha,
                        "selected_t": int(
                            context.surface["inference_iterations"]
                        ),
                        "fixed_step_minimization": True,
                        "batch_state_policy": "reset_each_batch",
                    }
                )
                evidence: dict[str, Any] = {
                    "schema_version": EPOCH_K_VIABILITY_SCHEMA_VERSION,
                    "epoch": epoch,
                    "checkpoint_path": path.name,
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_tensor_sha256": tensor_sha,
                    "high_k_reference": K_REFERENCE,
                    "official_test_read": False,
                    "measurement": measurement,
                }
                passed, reasons = _validate_epoch_k_viability_evidence(
                    evidence,
                    checkpoint=epoch_checkpoints[-1],
                    expected_epoch=epoch,
                    expected_selected_t=int(
                        context.surface["inference_iterations"]
                    ),
                    expected_k_cohort_sha256=str(
                        context.surface["shared_asset_hashes"][
                            "k_cohort_indices_sha256"
                        ]
                    ),
                )
                evidence.update(
                    {
                        "status": "passed" if passed else "failed",
                        "passed": passed,
                        "validation_reasons": reasons,
                    }
                )
            except (PerfectDiodeTKRuntimeError, FloatingPointError) as exc:
                evidence = {
                    "schema_version": EPOCH_K_VIABILITY_SCHEMA_VERSION,
                    "epoch": epoch,
                    "checkpoint_path": path.name,
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_tensor_sha256": tensor_sha,
                    "high_k_reference": K_REFERENCE,
                    "official_test_read": False,
                    "measurement_error": f"{type(exc).__name__}:{exc}",
                    "measurement": None,
                }
                passed, reasons = _validate_epoch_k_viability_evidence(
                    evidence,
                    checkpoint=epoch_checkpoints[-1],
                    expected_epoch=epoch,
                    expected_selected_t=int(
                        context.surface["inference_iterations"]
                    ),
                    expected_k_cohort_sha256=str(
                        context.surface["shared_asset_hashes"][
                            "k_cohort_indices_sha256"
                        ]
                    ),
                )
                evidence.update(
                    {
                        "status": "failed",
                        "passed": passed,
                        "validation_reasons": reasons,
                    }
                )
            epoch_k_viability.append(evidence)
            return validation

        # The v1 candidate kernel is architecture-agnostic after its immutable
        # contract lookup.  Conv2 is used solely as the existing 3 x 3,438
        # contract token; the runtime itself is the manifest-bound Conv3 model.
        raw = run_epoch_training(
            runtime,
            assets.bundle,
            architecture="conv2",
            mode="candidate",
            learning_rates_by_parameter=rates,
            output_dir=output_dir,
            validation_function=validation_with_checkpoint,
        )
        del runtime
        rows = raw.pop("step_rows")
        cell_id = rho_cell_id(
            study_id=context.spec.study_id,
            surface_id=context.surface_id,
            rho_conv=rho_conv,
            rho_dense=rho_dense,
        )
        _write_step_log(
            output_dir / "steps.csv",
            (
                {**row, "cell_id": cell_id, "phase": "candidate"}
                for row in rows
            ),
        )
        if raw.get("training_completed") is True:
            if (
                raw.get("completed_steps") != CANDIDATE_TOTAL_STEPS
                or len(epoch_checkpoints) != 3
                or len(epoch_k_viability) != 3
            ):
                raise PerfectDiodeConv3ExecutionError(
                    "Expected a completed Conv3 candidate to contain exactly "
                    "10,314 steps and three epoch checkpoints."
                )
        raw.update(
            {
                "architecture": "conv3",
                "epoch_checkpoints": epoch_checkpoints,
                "epoch_k_reference_viability": epoch_k_viability,
                "epoch_k_reference_viability_passed": (
                    len(epoch_k_viability) == 3
                    and all(
                        evidence.get("passed") is True
                        for evidence in epoch_k_viability
                    )
                ),
                "v1_candidate_contract_adapter": {
                    "contract_lookup_architecture_token": "conv2",
                    "runtime_architecture": "conv3",
                    "epochs": 3,
                    "steps_per_epoch": 3_438,
                    "total_steps": CANDIDATE_TOTAL_STEPS,
                },
            }
        )
        return raw


def _assets_summary(assets: Any) -> dict[str, Any]:
    if isinstance(assets, LoadedV2Assets):
        return {
            "asset_dir": str(assets.asset_dir),
            "initialization_checkpoint_sha256": assets.checkpoint_sha256,
            "initialization_tensor_sha256": assets.parameter_tensor_sha256,
            "train_indices_sha256": assets.bundle.train_indices_hash,
            "validation_indices_sha256": (
                assets.bundle.validation_indices_hash
            ),
            "train_batch_order_epoch_sha256s": list(
                assets.epoch_order_sha256s
            ),
        }
    if isinstance(assets, Mapping):
        return copy.deepcopy(dict(assets))
    return {"backend_asset_type": type(assets).__name__}


def _expected_shared_asset_dir(context: ExecutionContext) -> Path:
    upstream = _mapping(
        context.spec.data.get("upstream_tk"), "study.upstream_tk"
    )
    return _resolve_relative(
        context.manifest_path.parent,
        upstream.get("shared_assets_artifact"),
        "study.upstream_tk.shared_assets_artifact",
    )


def _load_assets_for_stage(
    context: ExecutionContext,
    backend: NumericalBackend,
    *,
    asset_dir: Path | None,
    data_root: Path | None,
    download: bool,
    device: str,
) -> Any:
    if asset_dir is None or data_root is None:
        raise ValueError(
            "Expected asset_dir and data_root for a numerical v2 stage. "
            f"Provided value: asset_dir={asset_dir!r}, data_root={data_root!r}."
        )
    expected_asset_dir = _expected_shared_asset_dir(context)
    if asset_dir.resolve() != expected_asset_dir:
        raise _error(
            "asset_dir",
            f"the copied, resolved-study-bound asset directory {expected_asset_dir}",
            asset_dir,
        )
    return backend.load_assets(
        context,
        asset_dir=asset_dir,
        data_root=data_root,
        download=download,
        device=device,
    )


def _probe_result(context: ExecutionContext) -> dict[str, Any]:
    stage = _load_stage_result(context, "optimizer_probe")
    probe = stage.get("probe")
    if not isinstance(probe, Mapping):
        raise _error(
            "optimizer_probe result.probe", "a probe mapping", probe
        )
    return copy.deepcopy(dict(probe))


def _pair(value: Mapping[str, Any], path: str) -> tuple[float, float]:
    return (
        _finite_positive(value.get("rho_conv"), f"{path}.rho_conv"),
        _finite_positive(value.get("rho_dense"), f"{path}.rho_dense"),
    )


def _cell_identity(
    context: ExecutionContext, rho_conv: float, rho_dense: float
) -> str:
    return rho_cell_id(
        study_id=context.spec.study_id,
        surface_id=context.surface_id,
        rho_conv=rho_conv,
        rho_dense=rho_dense,
    )


def _decorate_cell_result(
    context: ExecutionContext,
    stage: str,
    *,
    rho_conv: float,
    rho_dense: float,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    cell_id = _cell_identity(context, rho_conv, rho_dense)
    result = copy.deepcopy(dict(raw))
    result.update(
        {
            "study_id": context.spec.study_id,
            "config_sha256": context.spec.config_sha256,
            "manifest_id": context.manifest_id,
            "stage": stage,
            "surface_id": context.surface_id,
            "cell_id": cell_id,
            "row_id": context.row["row_id"],
            "architecture": "conv3",
            "scheme": context.surface["scheme"],
            "optimizer": context.surface["optimizer"],
            "host": context.host,
            "rho_conv": float(rho_conv),
            "rho_dense": float(rho_dense),
            "official_test_read": False,
        }
    )
    return result


def _run_canary_cell(
    context: ExecutionContext,
    backend: NumericalBackend,
    assets: Any,
    probe: Mapping[str, Any],
    *,
    stage: str,
    rho_conv: float,
    rho_dense: float,
    device: str,
) -> dict[str, Any]:
    cell_id = _cell_identity(context, rho_conv, rho_dense)
    destination = _cell_dir(context, stage, cell_id)
    if (destination / "completion.json").exists():
        return _validate_cell_completion(
            context, stage, cell_id, destination
        )
    raw = backend.canary(
        context,
        assets,
        probe,
        rho_conv=rho_conv,
        rho_dense=rho_dense,
        output_dir=destination,
        device=device,
    )
    return _publish_cell(
        context,
        stage,
        cell_id,
        _decorate_cell_result(
            context,
            stage,
            rho_conv=rho_conv,
            rho_dense=rho_dense,
            raw=raw,
        ),
    )


def _zero_work_cell(
    context: ExecutionContext,
    stage: str,
    *,
    rho_conv: float,
    rho_dense: float,
    reason: str,
) -> dict[str, Any]:
    cell_id = _cell_identity(context, rho_conv, rho_dense)
    return _publish_cell(
        context,
        stage,
        cell_id,
        _decorate_cell_result(
            context,
            stage,
            rho_conv=rho_conv,
            rho_dense=rho_dense,
            raw={
                "status": "zero_work",
                "zero_work": True,
                "reason": reason,
                "training_completed": False,
                "safety_admissible": False,
                "completed_steps": 0,
            },
        ),
    )


def _run_candidate_cell(
    context: ExecutionContext,
    backend: NumericalBackend,
    assets: Any,
    probe: Mapping[str, Any],
    *,
    stage: str,
    rho_conv: float,
    rho_dense: float,
    device: str,
) -> dict[str, Any]:
    cell_id = _cell_identity(context, rho_conv, rho_dense)
    destination = _cell_dir(context, stage, cell_id)
    if (destination / "completion.json").exists():
        return _validate_cell_completion(
            context, stage, cell_id, destination
        )
    raw = backend.candidate(
        context,
        assets,
        probe,
        rho_conv=rho_conv,
        rho_dense=rho_dense,
        output_dir=destination,
        device=device,
    )
    return _publish_cell(
        context,
        stage,
        cell_id,
        _decorate_cell_result(
            context,
            stage,
            rho_conv=rho_conv,
            rho_dense=rho_dense,
            raw=raw,
        ),
    )


def _fixed_core(policy: Mapping[str, Any]) -> tuple[tuple[float, float], ...]:
    cells = policy.get("core_cells")
    if not isinstance(cells, list) or len(cells) != 9:
        raise _error(
            "rho_policy.core_cells",
            "the exact fixed nine-cell grid",
            cells,
        )
    pairs = tuple(_pair(cell, f"rho_policy.core_cells[{index}]") for index, cell in enumerate(cells))
    conv = tuple(sorted({pair[0] for pair in pairs}))
    dense = tuple(sorted({pair[1] for pair in pairs}))
    if pairs != candidate_grid(conv, dense) or len(conv) != 3 or len(dense) != 3:
        raise _error(
            "rho_policy.core_cells",
            "a Conv-major 3x3 Cartesian grid",
            pairs,
        )
    return pairs


def _axes_from_cells(
    cells: Sequence[Mapping[str, Any]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    pairs = tuple(_pair(cell, f"cells[{index}]") for index, cell in enumerate(cells))
    conv = tuple(sorted({pair[0] for pair in pairs}))
    dense = tuple(sorted({pair[1] for pair in pairs}))
    if set(pairs) != set(candidate_grid(conv, dense)):
        raise _error("cells", "one complete Cartesian grid", pairs)
    return conv, dense


def _selection_json(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    expansion = result.get("expansion")
    if expansion is not None and not isinstance(expansion, Mapping):
        result["expansion"] = {
            "required": bool(expansion.required),
            "directions": [
                {"axis": axis, "direction": direction}
                for axis, direction in expansion.directions
            ],
            "rho_conv_values": list(expansion.rho_conv_values),
            "rho_dense_values": list(expansion.rho_dense_values),
            "new_cells": [
                {"rho_conv": pair[0], "rho_dense": pair[1]}
                for pair in expansion.new_cells
            ],
        }
    return result


def _candidate_selector_records(
    candidate_stage: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records = candidate_stage.get("candidate_records")
    if not isinstance(records, list):
        raise _error(
            "candidate stage.candidate_records",
            "a complete list",
            records,
        )
    return copy.deepcopy(records)


def _validate_epoch_k_viability_evidence(
    evidence: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    expected_epoch: int,
    expected_selected_t: int | None = None,
    expected_k_cohort_sha256: str | None = None,
) -> tuple[bool, list[str]]:
    """Recompute the complete high-K viability decision from raw evidence."""

    reasons: list[str] = []
    if evidence.get("schema_version") != EPOCH_K_VIABILITY_SCHEMA_VERSION:
        reasons.append("wrong_schema_version")
    if evidence.get("epoch") != expected_epoch:
        reasons.append("wrong_epoch")
    if evidence.get("checkpoint_path") != checkpoint.get("path"):
        reasons.append("wrong_checkpoint_path")
    if evidence.get("checkpoint_sha256") != checkpoint.get("sha256"):
        reasons.append("wrong_checkpoint_sha256")
    if evidence.get("high_k_reference") != K_REFERENCE:
        reasons.append("wrong_high_k_reference")
    if evidence.get("official_test_read") is not False:
        reasons.append("wrong_official_test_read")
    measurement = evidence.get("measurement")
    if not isinstance(measurement, Mapping):
        return False, reasons + ["missing_measurement"]
    if (
        measurement.get("candidate_k") != K_REFERENCE
        or measurement.get("reference_k") != K_REFERENCE
    ):
        reasons.append("wrong_self_reference_identity")
    if measurement.get("initialization_checkpoint_sha256") != checkpoint.get(
        "sha256"
    ):
        reasons.append("measurement_checkpoint_mismatch")
    if measurement.get("initialization_tensor_sha256") != evidence.get(
        "checkpoint_tensor_sha256"
    ):
        reasons.append("measurement_tensor_mismatch")
    if (
        expected_selected_t is not None
        and measurement.get("selected_t") != expected_selected_t
    ):
        reasons.append("wrong_selected_t")
    if (
        expected_k_cohort_sha256 is not None
        and measurement.get("cohort_source_indices_sha256")
        != expected_k_cohort_sha256
    ):
        reasons.append("wrong_k_cohort_sha256")
    measurement_passed, measurement_reasons, measurement_complete = (
        _candidate_k_pass(measurement, CONV3_CONV_WEIGHTS)
    )
    reasons.extend(f"measurement:{reason}" for reason in measurement_reasons)
    if not measurement_complete:
        reasons.append("measurement_incomplete")
    if measurement.get("reference_viable") is not measurement_passed:
        # A K=64 self-comparison can fail only through reference viability.
        reasons.append("self_reference_viability_mismatch")
    return not reasons and measurement_passed, reasons


def _candidate_epoch_k_viability(
    candidate: Mapping[str, Any],
    *,
    expected_selected_t: int,
    expected_k_cohort_sha256: str,
) -> tuple[bool, list[str]]:
    checkpoints = candidate.get("epoch_checkpoints")
    evidence = candidate.get("epoch_k_reference_viability")
    if (
        not isinstance(checkpoints, list)
        or len(checkpoints) != 3
        or not isinstance(evidence, list)
        or len(evidence) != 3
    ):
        return False, ["expected_three_checkpoint_viability_records"]
    reasons: list[str] = []
    all_passed = True
    for epoch, (checkpoint, record) in enumerate(
        zip(checkpoints, evidence), start=1
    ):
        if not isinstance(checkpoint, Mapping) or not isinstance(
            record, Mapping
        ):
            all_passed = False
            reasons.append(f"epoch_{epoch}:invalid_record")
            continue
        passed, epoch_reasons = _validate_epoch_k_viability_evidence(
            record,
            checkpoint=checkpoint,
            expected_epoch=epoch,
            expected_selected_t=expected_selected_t,
            expected_k_cohort_sha256=expected_k_cohort_sha256,
        )
        expected_status = "passed" if passed else "failed"
        if record.get("passed") is not passed:
            epoch_reasons.append("recorded_passed_mismatch")
        if record.get("status") != expected_status:
            epoch_reasons.append("recorded_status_mismatch")
        if record.get("validation_reasons") != epoch_reasons:
            epoch_reasons.append("recorded_reasons_mismatch")
        if epoch_reasons:
            passed = False
        all_passed = all_passed and passed
        reasons.extend(
            f"epoch_{epoch}:{reason}" for reason in epoch_reasons
        )
    if candidate.get("epoch_k_reference_viability_passed") is not all_passed:
        reasons.append("candidate_viability_summary_mismatch")
        all_passed = False
    return all_passed and not reasons, reasons


def _candidate_record(
    context: ExecutionContext,
    canary: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    rho_conv, rho_dense = _pair(candidate, "candidate")
    identifier = str(candidate["cell_id"])
    if canary.get("safety_clean") is not True:
        failure = canary.get("safety_failure") or canary.get("reason")
        return {
            "candidate_id": identifier,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "admissible": False,
            "inadmissible_reason": f"canary_rejected:{failure}",
            "candidate_result_path": None,
        }
    if (
        candidate.get("training_completed") is not True
        or candidate.get("safety_admissible") is not True
    ):
        failure = candidate.get("safety_failure") or candidate.get("reason")
        return {
            "candidate_id": identifier,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "admissible": False,
            "inadmissible_reason": f"candidate_safety_failure:{failure}",
            "candidate_result_path": None,
        }
    viability_passed, viability_reasons = _candidate_epoch_k_viability(
        candidate,
        expected_selected_t=int(context.surface["inference_iterations"]),
        expected_k_cohort_sha256=str(
            context.surface["shared_asset_hashes"][
                "k_cohort_indices_sha256"
            ]
        ),
    )
    if not viability_passed:
        return {
            "candidate_id": identifier,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "admissible": False,
            "inadmissible_reason": (
                "epoch_k_reference_viability_failed:"
                + ",".join(viability_reasons)
            ),
            "candidate_result_path": None,
        }
    projection = candidate.get("median_projection_efficiency")
    if (
        isinstance(projection, bool)
        or not isinstance(projection, (int, float))
        or not math.isfinite(float(projection))
        or float(projection) < 0.0
    ):
        return {
            "candidate_id": identifier,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "admissible": False,
            "inadmissible_reason": "candidate_projection_efficiency_missing_or_nonfinite",
            "candidate_result_path": None,
        }
    return {
        "candidate_id": identifier,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "admissible": True,
        "completed_steps": candidate.get("completed_steps"),
        "final_validation_loss": candidate.get("final_validation_loss"),
        "final_validation_accuracy": candidate.get(
            "final_validation_accuracy"
        ),
        "median_projection_efficiency": float(projection),
        "candidate_result_path": (
            f"stages/{candidate['stage']}/cells/{identifier}/result.json"
        ),
    }


def _execute_audit(context: ExecutionContext) -> dict[str, Any]:
    return {
        **_base_result(context, "audit", status="complete"),
        "manifest_schema_version": context.manifest["schema_version"],
        "public_stage_sequence": list(PUBLIC_STAGE_SEQUENCE),
        "routing_verified": True,
        "execution_authority_verified": True,
        "official_test_enabled": False,
        "long_confirmation": False,
    }


def _execute_assets(
    context: ExecutionContext,
    backend: NumericalBackend,
    *,
    asset_dir: Path | None,
    data_root: Path | None,
    download: bool,
    device: str,
) -> dict[str, Any]:
    _load_stage_result(context, "audit")
    if not context.surface["lr_eligible"]:
        return _zero_work(
            context,
            "import_tk",
            str(context.surface["zero_work_reason"]),
        )
    assets = _load_assets_for_stage(
        context,
        backend,
        asset_dir=asset_dir,
        data_root=data_root,
        download=download,
        device=device,
    )
    return {
        **_base_result(context, "import_tk", status="complete"),
        "assets": _assets_summary(assets),
        "shared_asset_hashes": copy.deepcopy(
            context.surface["shared_asset_hashes"]
        ),
    }


def _execute_probe(
    context: ExecutionContext,
    backend: NumericalBackend,
    *,
    asset_dir: Path | None,
    data_root: Path | None,
    download: bool,
    device: str,
) -> dict[str, Any]:
    assets_stage = _load_stage_result(context, "import_tk")
    if assets_stage.get("zero_work") is True:
        return _zero_work(
            context,
            "optimizer_probe",
            str(assets_stage["reason"]),
            dependency=assets_stage,
        )
    assets = _load_assets_for_stage(
        context,
        backend,
        asset_dir=asset_dir,
        data_root=data_root,
        download=download,
        device=device,
    )
    raw = _mapping(
        backend.optimizer_probe(
            context,
            assets,
            output_dir=_stage_dir(context, "optimizer_probe"),
            device=device,
        ),
        "optimizer probe",
    )
    require_official_test_excluded(raw, require_result_marker=True)
    status = "complete" if raw.get("probe_stable") is True else "unresolved_probe"
    return {
        **_base_result(context, "optimizer_probe", status=status),
        "probe": raw,
    }


def _execute_core_canaries(
    context: ExecutionContext,
    backend: NumericalBackend,
    *,
    asset_dir: Path | None,
    data_root: Path | None,
    download: bool,
    device: str,
) -> dict[str, Any]:
    probe_stage = _load_stage_result(context, "optimizer_probe")
    if (
        probe_stage.get("zero_work") is True
        or probe_stage.get("status") != "complete"
    ):
        return _zero_work(
            context,
            "rho_canary_core",
            str(probe_stage.get("reason") or "unresolved_probe"),
            status="unresolved_probe",
            dependency=probe_stage,
        )
    probe = _probe_result(context)
    assets = _load_assets_for_stage(
        context,
        backend,
        asset_dir=asset_dir,
        data_root=data_root,
        download=download,
        device=device,
    )
    policy = _mapping(context.surface["rho_policy"], "rho_policy")
    mode = policy.get("core_mode")
    cells: list[dict[str, Any]] = []
    center_attempts: list[dict[str, Any]] = []
    if mode == "fixed_high_3x3":
        pairs = _fixed_core(policy)
    elif mode == "adaptive_safe_center_factor_three_3x3":
        center_conv = _finite_positive(
            policy.get("initial_rho_conv"), "rho_policy.initial_rho_conv"
        )
        center_dense = _finite_positive(
            policy.get("initial_rho_dense"), "rho_policy.initial_rho_dense"
        )
        divisor = _finite_positive(
            policy.get("failure_scale_divisor"),
            "rho_policy.failure_scale_divisor",
        )
        attempts = policy.get("maximum_center_attempts")
        if (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or attempts < 1
        ):
            raise _error(
                "rho_policy.maximum_center_attempts",
                "an integer >= 1",
                attempts,
            )
        safe_center: tuple[float, float] | None = None
        attempt_by_pair: dict[tuple[float, float], dict[str, Any]] = {}
        for attempt in range(attempts):
            pair = (
                center_conv / (divisor**attempt),
                center_dense / (divisor**attempt),
            )
            result = _run_canary_cell(
                context,
                backend,
                assets,
                probe,
                stage="rho_canary_core",
                rho_conv=pair[0],
                rho_dense=pair[1],
                device=device,
            )
            attempt_record = {"attempt": attempt, **result}
            center_attempts.append(attempt_record)
            attempt_by_pair[pair] = result
            if result.get("safety_clean") is True:
                safe_center = pair
                break
        if safe_center is None:
            return {
                **_base_result(
                    context,
                    "rho_canary_core",
                    status="unresolved_no_safe_center",
                    zero_work=False,
                    reason="legacy_center_canary_failed_all_attempts",
                ),
                "center_attempts": center_attempts,
                "cells": [],
            }
        pairs = core_grid(*safe_center)
        for pair in pairs:
            if pair in attempt_by_pair:
                result = attempt_by_pair[pair]
            else:
                result = _run_canary_cell(
                    context,
                    backend,
                    assets,
                    probe,
                    stage="rho_canary_core",
                    rho_conv=pair[0],
                    rho_dense=pair[1],
                    device=device,
                )
            cells.append(result)
    else:
        raise _error(
            "rho_policy.core_mode",
            "'fixed_high_3x3' or 'adaptive_safe_center_factor_three_3x3'",
            mode,
        )
    if mode == "fixed_high_3x3":
        cells = [
            _run_canary_cell(
                context,
                backend,
                assets,
                probe,
                stage="rho_canary_core",
                rho_conv=pair[0],
                rho_dense=pair[1],
                device=device,
            )
            for pair in pairs
        ]
    conv, dense = _axes_from_cells(cells)
    return {
        **_base_result(context, "rho_canary_core", status="complete"),
        "core_mode": mode,
        "center_attempts": center_attempts,
        "rho_conv_values": list(conv),
        "rho_dense_values": list(dense),
        "cells": cells,
    }


def _execute_candidate_stage(
    context: ExecutionContext,
    backend: NumericalBackend,
    *,
    stage: str,
    canary_stage_name: str,
    asset_dir: Path | None,
    data_root: Path | None,
    download: bool,
    device: str,
) -> dict[str, Any]:
    canaries = _load_stage_result(context, canary_stage_name)
    if canaries.get("status") != "complete":
        return _zero_work(
            context,
            stage,
            str(canaries.get("reason") or canaries.get("status")),
            status=str(canaries.get("status") or "zero_work"),
            dependency=canaries,
        )
    probe = _probe_result(context)
    assets = _load_assets_for_stage(
        context,
        backend,
        asset_dir=asset_dir,
        data_root=data_root,
        download=download,
        device=device,
    )
    results: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for index, canary in enumerate(canaries.get("cells", [])):
        rho_conv, rho_dense = _pair(canary, f"{canary_stage_name}.cells[{index}]")
        if canary.get("safety_clean") is True:
            candidate = _run_candidate_cell(
                context,
                backend,
                assets,
                probe,
                stage=stage,
                rho_conv=rho_conv,
                rho_dense=rho_dense,
                device=device,
            )
        else:
            candidate = _zero_work_cell(
                context,
                stage,
                rho_conv=rho_conv,
                rho_dense=rho_dense,
                reason="canary_rejected",
            )
        results.append(candidate)
        records.append(_candidate_record(context, canary, candidate))
    expected_count = len(canaries.get("cells", []))
    if expected_count < 1 or len(records) != expected_count:
        raise PerfectDiodeConv3ExecutionError(
            f"Expected one candidate record per {canary_stage_name} cell."
        )
    return {
        **_base_result(context, stage, status="complete"),
        "canary_stage": canary_stage_name,
        "candidate_records": records,
        "cells": results,
    }


def _execute_select_core(context: ExecutionContext) -> dict[str, Any]:
    candidates = _load_stage_result(context, "rho_core_candidates")
    canaries = _load_stage_result(context, "rho_canary_core")
    if candidates.get("status") != "complete":
        return _zero_work(
            context,
            "select_core",
            str(candidates.get("reason") or candidates.get("status")),
            status=str(candidates.get("status") or "zero_work"),
            dependency=candidates,
        )
    conv = canaries.get("rho_conv_values")
    dense = canaries.get("rho_dense_values")
    selection = select_surface_candidates(
        _candidate_selector_records(candidates),
        rho_conv_values=conv,
        rho_dense_values=dense,
        expansion_available=True,
        minimum_accuracy=float(
            context.spec.data["candidate_training"][
                "minimum_final_validation_accuracy"
            ]
        ),
        plateau_relative_tolerance=float(
            context.spec.data["selection"]["plateau_relative_to_minimum"]
        ),
        required_completed_steps=CANDIDATE_TOTAL_STEPS,
    )
    normalized = _selection_json(selection)
    return {
        **_base_result(
            context, "select_core", status=str(normalized["status"])
        ),
        "selection": normalized,
    }


def _execute_extension_canaries(
    context: ExecutionContext,
    backend: NumericalBackend,
    *,
    asset_dir: Path | None,
    data_root: Path | None,
    download: bool,
    device: str,
) -> dict[str, Any]:
    core = _load_stage_result(context, "select_core")
    selection = core.get("selection")
    if (
        core.get("status") != "needs_expansion"
        or not isinstance(selection, Mapping)
    ):
        return _zero_work(
            context,
            "rho_canary_expansion",
            "core_selection_did_not_request_expansion",
            dependency=core,
        )
    expansion = _mapping(selection.get("expansion"), "core expansion")
    new_cells = expansion.get("new_cells")
    if not isinstance(new_cells, list) or not new_cells:
        raise _error(
            "core expansion.new_cells",
            "a non-empty single-wave cell list",
            new_cells,
        )
    if len(new_cells) > 7:
        raise _error(
            "core expansion.new_cells", "at most seven cells", new_cells
        )
    probe = _probe_result(context)
    assets = _load_assets_for_stage(
        context,
        backend,
        asset_dir=asset_dir,
        data_root=data_root,
        download=download,
        device=device,
    )
    cells = []
    for index, cell in enumerate(new_cells):
        rho_conv, rho_dense = _pair(cell, f"new_cells[{index}]")
        cells.append(
            _run_canary_cell(
                context,
                backend,
                assets,
                probe,
                stage="rho_canary_expansion",
                rho_conv=rho_conv,
                rho_dense=rho_dense,
                device=device,
            )
        )
    return {
        **_base_result(
            context, "rho_canary_expansion", status="complete"
        ),
        "expansion": expansion,
        "cells": cells,
    }


def _execute_select_expanded(context: ExecutionContext) -> dict[str, Any]:
    core = _load_stage_result(context, "select_core")
    if core.get("status") != "needs_expansion":
        selection = core.get("selection")
        if not isinstance(selection, Mapping):
            return _zero_work(
                context,
                "select_expanded",
                str(core.get("reason") or core.get("status")),
                status=str(core.get("status") or "zero_work"),
                dependency=core,
            )
        return {
            **_base_result(
                context, "select_expanded", status=str(selection["status"])
            ),
            "expansion_used": False,
            "selection": copy.deepcopy(dict(selection)),
        }
    extension = _load_stage_result(context, "rho_expansion_candidates")
    if extension.get("status") != "complete":
        return _zero_work(
            context,
            "select_expanded",
            str(extension.get("reason") or extension.get("status")),
            status="unresolved_boundary",
            dependency=extension,
        )
    core_candidates = _load_stage_result(context, "rho_core_candidates")
    core_selection = _mapping(core["selection"], "core selection")
    expansion = _mapping(
        core_selection.get("expansion"), "core selection.expansion"
    )
    combined = (
        _candidate_selector_records(core_candidates)
        + _candidate_selector_records(extension)
    )
    selection = select_surface_candidates(
        combined,
        rho_conv_values=expansion["rho_conv_values"],
        rho_dense_values=expansion["rho_dense_values"],
        expansion_available=False,
        minimum_accuracy=float(
            context.spec.data["candidate_training"][
                "minimum_final_validation_accuracy"
            ]
        ),
        plateau_relative_tolerance=float(
            context.spec.data["selection"]["plateau_relative_to_minimum"]
        ),
        required_completed_steps=CANDIDATE_TOTAL_STEPS,
    )
    normalized = _selection_json(selection)
    return {
        **_base_result(
            context, "select_expanded", status=str(normalized["status"])
        ),
        "expansion_used": True,
        "core_selection": core_selection,
        "selection": normalized,
    }


def _selected_candidate(
    context: ExecutionContext, final_selection: Mapping[str, Any]
) -> tuple[dict[str, Any], Path]:
    selection = _mapping(
        final_selection.get("selection"), "select_expanded.selection"
    )
    selected = _mapping(selection.get("selected"), "selection.selected")
    cell_id = selected.get("candidate_id")
    if not isinstance(cell_id, str) or not cell_id:
        raise _error(
            "selection.selected.candidate_id", "a non-empty id", cell_id
        )
    stage_names = (
        ("rho_expansion_candidates", "rho_core_candidates")
        if final_selection.get("expansion_used") is True
        else ("rho_core_candidates",)
    )
    for stage in stage_names:
        path = _cell_dir(context, stage, cell_id) / "result.json"
        if path.is_file():
            result = _validate_cell_completion(
                context, stage, cell_id, path.parent
            )
            return result, path.parent
    raise PerfectDiodeConv3ExecutionError(
        f"Expected the selected candidate cell result for {cell_id}."
    )


def _bind_k_measurement_context(
    measurement: Mapping[str, Any],
    *,
    assets: LoadedTKAssets,
    checkpoint_sha256: str,
    checkpoint_tensor_sha256: str,
    selected_t: int,
) -> dict[str, Any]:
    value = _mapping(measurement, "K measurement")
    value.update(
        {
            "cohort_examples": K_EXAMPLES,
            "batch_size": K_BATCH_SIZE,
            "official_test_read": False,
            "cohort_source_indices_sha256": assets.k_indices_sha256,
            "initialization_checkpoint_sha256": checkpoint_sha256,
            "initialization_tensor_sha256": checkpoint_tensor_sha256,
            "selected_t": selected_t,
            "fixed_step_minimization": True,
            "batch_state_policy": "reset_each_batch",
        }
    )
    return value


def run_builtin_post_training_tk_audit(
    *,
    context: ExecutionContext,
    selected_candidate: Mapping[str, Any],
    candidate_dir: Path,
    epoch_checkpoints: Sequence[Mapping[str, Any]],
    asset_dir: str | Path,
    data_root: str | Path,
    device: str,
    download: bool = False,
    tk_config_path: str | Path = DEFAULT_TK_CONFIG,
) -> dict[str, Any]:
    """Replay all three selected epoch checkpoints with the T/K runtime."""

    resolved_asset_dir = Path(asset_dir).expanduser().resolve()
    expected_asset_dir = _expected_shared_asset_dir(context)
    if resolved_asset_dir != expected_asset_dir:
        raise _error(
            "asset_dir",
            f"the copied, resolved-study-bound asset directory {expected_asset_dir}",
            resolved_asset_dir,
        )
    tk_spec = PerfectDiodeTKStudySpec.from_path(tk_config_path)
    upstream = _mapping(
        context.spec.data.get("upstream_tk"), "study.upstream_tk"
    )
    if (
        tk_spec.study_id != upstream.get("study_id")
        or tk_spec.config_sha256 != upstream.get("config_sha256")
    ):
        raise PerfectDiodeConv3ExecutionError(
            "Expected the built-in post-training audit to use the exact "
            "manifest-bound upstream T/K study."
        )
    rows = [
        row
        for row in tk_spec.rows
        if row.get("scheme") == context.surface["scheme"]
    ]
    if len(rows) != 1:
        raise _error(
            "T/K audit row",
            f"one row for scheme {context.surface['scheme']!r}",
            rows,
        )
    tk_row = rows[0]
    base_assets = load_tk_shared_assets(
        tk_spec,
        asset_dir=resolved_asset_dir,
        data_root=data_root,
        download=download,
    )
    shared = context.surface["shared_asset_hashes"]
    observed_assets = {
        "initialization_checkpoint_sha256": base_assets.checkpoint_sha256,
        "initialization_tensor_sha256": base_assets.parameter_tensor_sha256,
        "train_indices_sha256": base_assets.train_indices_sha256,
        "validation_indices_sha256": base_assets.validation_indices_sha256,
        "t_cohort_indices_sha256": base_assets.t_indices_sha256,
        "k_cohort_indices_sha256": base_assets.k_indices_sha256,
    }
    for key, observed in observed_assets.items():
        if observed != shared[key]:
            raise _error(
                f"post-training T/K assets.{key}",
                f"the manifest-bound digest {shared[key]!r}",
                observed,
            )
    if len(epoch_checkpoints) != 3:
        raise _error(
            "epoch_checkpoints", "exactly three records", epoch_checkpoints
        )
    epoch_audits: list[dict[str, Any]] = []
    selected_t = int(context.surface["inference_iterations"])
    selected_k = int(context.surface["training_iterations"])
    for expected_epoch, raw_checkpoint in enumerate(
        epoch_checkpoints, start=1
    ):
        checkpoint_record = _mapping(
            raw_checkpoint, f"epoch_checkpoints[{expected_epoch - 1}]"
        )
        if checkpoint_record.get("epoch") != expected_epoch:
            raise _error(
                f"epoch_checkpoints[{expected_epoch - 1}].epoch",
                f"exactly {expected_epoch}",
                checkpoint_record.get("epoch"),
            )
        checkpoint = _resolve_relative(
            candidate_dir,
            checkpoint_record.get("path"),
            f"epoch_checkpoints[{expected_epoch - 1}].path",
        )
        checkpoint_sha = sha256_file(checkpoint)
        if checkpoint_sha != checkpoint_record.get("sha256"):
            raise PerfectDiodeConv3ExecutionError(
                f"Expected epoch {expected_epoch} checkpoint SHA-256 to verify."
            )
        runtime = build_model_runtime(
            tk_spec.data,
            tk_row,
            device=device,
            initialization_checkpoint=checkpoint,
            learning_rate=1.0,
        )
        tensor_sha = _contract_parameter_tensor_digest(runtime.parameters)
        del runtime
        trained_assets: LoadedTKAssets = replace(
            base_assets,
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            parameter_tensor_sha256=tensor_sha,
        )
        selected_audit = run_operating_point_audit(
            tk_spec,
            tk_row,
            selected_t=selected_t,
            selected_k=selected_k,
            assets=trained_assets,
            device=device,
            entry_id=(
                f"{context.surface_id}--epoch-{expected_epoch:03d}"
            ),
            execution_host=context.host,
            t_extension_used=bool(
                context.row["upstream_tk"]["t_extension_used"]
            ),
            execution_environment_sha256=context.surface[
                "execution_environment_sha256"
            ],
        )
        reference_t = copy.deepcopy(
            selected_audit["t64_sentinel_measurement"]
        )
        selected_t_passed, _selected_t_reasons, selected_t_complete = (
            _candidate_t_pass(
                selected_audit["t_measurement"],
                expected_iteration=selected_t,
            )
        )
        reference_t_passed, _t64_reasons, t64_complete = _candidate_t_pass(
            reference_t, expected_iteration=64
        )
        selected_t_passed = selected_t_passed and selected_t_complete
        reference_t_passed = reference_t_passed and t64_complete
        audit_k_measurement = _mapping(
            selected_audit["k_measurement"],
            f"epoch {expected_epoch} K measurement",
        )
        if selected_k == K_REFERENCE:
            k64_reference = _collect_gradients_for_k(
                tk_spec,
                tk_row,
                selected_t=selected_t,
                training_iterations=K_REFERENCE,
                assets=trained_assets,
                device=device,
            )
            selected_k_measurement = _bind_k_measurement_context(
                compare_k_gradients(
                    k64_reference,
                    k64_reference,
                    candidate_k=K_REFERENCE,
                    reference_k=K_REFERENCE,
                ),
                assets=trained_assets,
                checkpoint_sha256=checkpoint_sha,
                checkpoint_tensor_sha256=tensor_sha,
                selected_t=selected_t,
            )
            k128_sentinel_measurement: dict[str, Any] | None = (
                audit_k_measurement
            )
        else:
            selected_k_measurement = audit_k_measurement
            k128_sentinel_measurement = None
        selected_k_passed, _k_reasons, k_complete = _candidate_k_pass(
            selected_k_measurement, CONV3_CONV_WEIGHTS
        )
        selected_k_passed = selected_k_passed and k_complete
        reference_viable = (
            selected_k_measurement.get("reference_viable") is True
        )
        if k128_sentinel_measurement is not None:
            sentinel_passed, _sentinel_reasons, sentinel_complete = (
                _candidate_k_pass(
                    k128_sentinel_measurement, CONV3_CONV_WEIGHTS
                )
            )
            sentinel_passed = sentinel_passed and sentinel_complete
        else:
            sentinel_passed = None
        extension_passed = (
            selected_audit.get("t256_extension_sentinel_passed") is True
            if bool(context.row["upstream_tk"]["t_extension_used"])
            else True
        )
        passed = (
            selected_t_passed
            and reference_t_passed
            and extension_passed
            and selected_k_passed
            and reference_viable
            and (
                sentinel_passed is True
                if selected_k == K_REFERENCE
                else True
            )
        )
        epoch_audits.append(
            {
                "epoch": expected_epoch,
                "checkpoint_path": checkpoint_record["path"],
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_tensor_sha256": tensor_sha,
                "selected_t_residual_gate_passed": selected_t_passed,
                "reference_t64_residual_gate_passed": reference_t_passed,
                "selected_k_gradient_gate_passed": selected_k_passed,
                "reference_gradient_viable": reference_viable,
                "k128_sentinel_gate_passed": sentinel_passed,
                "t256_extension_residual_gate_passed": (
                    extension_passed
                    if bool(
                        context.row["upstream_tk"]["t_extension_used"]
                    )
                    else None
                ),
                "passed": passed,
                "official_test_read": False,
                "selected_operating_point_audit": selected_audit,
                "reference_t64_measurement": reference_t,
                "selected_k_vs_k64_measurement": selected_k_measurement,
                "k128_sentinel_measurement": (
                    k128_sentinel_measurement
                ),
            }
        )
    all_passed = all(record["passed"] for record in epoch_audits)
    return {
        "schema_version": POST_TK_AUDIT_SCHEMA_VERSION,
        "surface_id": context.surface_id,
        "row_id": context.row["row_id"],
        "scheme": context.surface["scheme"],
        "selected_cell_id": selected_candidate["cell_id"],
        "selected_t": selected_t,
        "selected_k": selected_k,
        "status": "passed" if all_passed else "failed",
        "passed": all_passed,
        "official_test_read": False,
        "epoch_audits": epoch_audits,
    }


def _validate_post_epoch_measurements(
    context: ExecutionContext,
    record: Mapping[str, Any],
    *,
    checkpoint: Mapping[str, Any],
    expected_epoch: int,
) -> tuple[bool, list[str]]:
    """Derive a post-training epoch decision from complete T/K measurements."""

    reasons: list[str] = []
    audit = record.get("selected_operating_point_audit")
    if not isinstance(audit, Mapping):
        return False, ["missing_operating_point_audit"]
    expected_audit = {
        "schema_version": TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION,
        "study_id": context.spec.data["upstream_tk"]["study_id"],
        "config_sha256": context.spec.data["upstream_tk"][
            "config_sha256"
        ],
        "entry_id": (
            f"{context.surface_id}--epoch-{expected_epoch:03d}"
        ),
        "row_id": context.row["row_id"],
        "architecture": "conv3",
        "scheme": context.surface["scheme"],
        "execution_backend": "tmux",
        "execution_host": context.host,
        "execution_environment_sha256": context.surface[
            "execution_environment_sha256"
        ],
        "official_test_read": False,
        "selected_t": context.surface["inference_iterations"],
        "selected_k": context.surface["training_iterations"],
        "k_audit_reference": (
            K_BOUNDARY_SENTINEL
            if int(context.surface["training_iterations"]) == K_REFERENCE
            else K_REFERENCE
        ),
        "fresh_replay_after_selection": True,
        "t_extension_used": bool(
            context.row["upstream_tk"]["t_extension_used"]
        ),
    }
    for key, expected_value in expected_audit.items():
        if audit.get(key) != expected_value:
            reasons.append(f"audit_{key}_mismatch")
    selected_t = audit.get("t_measurement")
    selected_t_passed, selected_t_reasons, selected_t_complete = (
        _candidate_t_pass(
            selected_t,
            expected_iteration=int(context.surface["inference_iterations"]),
        )
    )
    reasons.extend(
        f"selected_t:{reason}" for reason in selected_t_reasons
    )
    if not selected_t_complete:
        reasons.append("selected_t_incomplete")
    t64 = audit.get("t64_sentinel_measurement")
    t64_passed, t64_reasons, t64_complete = _candidate_t_pass(
        t64, expected_iteration=64
    )
    reasons.extend(f"t64:{reason}" for reason in t64_reasons)
    if not t64_complete:
        reasons.append("t64_incomplete")
    if record.get("reference_t64_measurement") != t64:
        reasons.append("reference_t64_copy_mismatch")
    t_extension_used = bool(
        context.row["upstream_tk"]["t_extension_used"]
    )
    extension_passed = True
    t256: Any = None
    if t_extension_used:
        t256 = audit.get("t256_extension_sentinel_measurement")
        extension_passed, t256_reasons, t256_complete = _candidate_t_pass(
            t256, expected_iteration=256
        )
        reasons.extend(f"t256:{reason}" for reason in t256_reasons)
        if not t256_complete:
            reasons.append("t256_incomplete")
    elif audit.get("t256_extension_sentinel_measurement") is not None:
        reasons.append("unexpected_t256_measurement")
    selected_k_value = int(context.surface["training_iterations"])
    audit_k_measurement = audit.get("k_measurement")
    audit_k_passed, audit_k_reasons, audit_k_complete = _candidate_k_pass(
        audit_k_measurement, CONV3_CONV_WEIGHTS
    )
    reasons.extend(f"audit_k:{reason}" for reason in audit_k_reasons)
    if not audit_k_complete:
        reasons.append("audit_k_incomplete")
    expected_reference_k = (
        K_BOUNDARY_SENTINEL
        if selected_k_value == K_REFERENCE
        else K_REFERENCE
    )
    if not isinstance(audit_k_measurement, Mapping) or (
        audit_k_measurement.get("candidate_k") != selected_k_value
        or audit_k_measurement.get("reference_k") != expected_reference_k
    ):
        reasons.append("audit_k_identity_mismatch")
    selected_k_measurement = record.get(
        "selected_k_vs_k64_measurement"
    )
    selected_k_passed, selected_k_reasons, selected_k_complete = (
        _candidate_k_pass(selected_k_measurement, CONV3_CONV_WEIGHTS)
    )
    reasons.extend(
        f"selected_k_vs_k64:{reason}" for reason in selected_k_reasons
    )
    if not selected_k_complete:
        reasons.append("selected_k_vs_k64_incomplete")
    if not isinstance(selected_k_measurement, Mapping) or (
        selected_k_measurement.get("candidate_k") != selected_k_value
        or selected_k_measurement.get("reference_k") != K_REFERENCE
    ):
        reasons.append("selected_k_vs_k64_identity_mismatch")
    k128_sentinel_measurement = record.get(
        "k128_sentinel_measurement"
    )
    if selected_k_value == K_REFERENCE:
        k128_sentinel_passed, sentinel_reasons, sentinel_complete = (
            _candidate_k_pass(
                k128_sentinel_measurement, CONV3_CONV_WEIGHTS
            )
        )
        reasons.extend(
            f"k128_sentinel:{reason}" for reason in sentinel_reasons
        )
        if not sentinel_complete:
            reasons.append("k128_sentinel_incomplete")
        if not isinstance(k128_sentinel_measurement, Mapping) or (
            k128_sentinel_measurement.get("candidate_k") != K_REFERENCE
            or k128_sentinel_measurement.get("reference_k")
            != K_BOUNDARY_SENTINEL
        ):
            reasons.append("k128_sentinel_identity_mismatch")
        if k128_sentinel_measurement != audit_k_measurement:
            reasons.append("k128_sentinel_audit_copy_mismatch")
    else:
        k128_sentinel_passed = None
        if k128_sentinel_measurement is not None:
            reasons.append("unexpected_k128_sentinel")
        if selected_k_measurement != audit_k_measurement:
            reasons.append("selected_k_audit_copy_mismatch")
    for name, measurement, cohort_key in (
        ("selected_t", selected_t, "t_cohort_indices_sha256"),
        ("t64", t64, "t_cohort_indices_sha256"),
        ("t256", t256, "t_cohort_indices_sha256"),
        (
            "audit_k",
            audit_k_measurement,
            "k_cohort_indices_sha256",
        ),
        (
            "selected_k_vs_k64",
            selected_k_measurement,
            "k_cohort_indices_sha256",
        ),
        (
            "k128_sentinel",
            k128_sentinel_measurement,
            "k_cohort_indices_sha256",
        ),
    ):
        if not isinstance(measurement, Mapping):
            continue
        if measurement.get("initialization_checkpoint_sha256") != checkpoint.get(
            "sha256"
        ):
            reasons.append(f"{name}_checkpoint_mismatch")
        if measurement.get("initialization_tensor_sha256") != record.get(
            "checkpoint_tensor_sha256"
        ):
            reasons.append(f"{name}_tensor_mismatch")
        expected_cohort_sha256 = context.surface["shared_asset_hashes"][
            cohort_key
        ]
        if (
            measurement.get("cohort_source_indices_sha256")
            != expected_cohort_sha256
        ):
            reasons.append(f"{name}_cohort_mismatch")
    reference_viable = (
        isinstance(selected_k_measurement, Mapping)
        and selected_k_measurement.get("reference_viable") is True
    )
    expected_runtime_audit_passed = (
        selected_t_passed
        and (extension_passed if t_extension_used else t64_passed)
        and audit_k_passed
    )
    if audit.get("passed") is not expected_runtime_audit_passed:
        reasons.append("operating_point_audit_passed_mismatch")
    expected_runtime_status = (
        "passed" if expected_runtime_audit_passed else "failed"
    )
    if audit.get("status") != expected_runtime_status:
        reasons.append("operating_point_audit_status_mismatch")
    if audit.get("t64_sentinel_passed") is not t64_passed:
        reasons.append("t64_sentinel_passed_mismatch")
    expected_t256_passed = extension_passed if t_extension_used else None
    if audit.get("t256_extension_sentinel_passed") is not expected_t256_passed:
        reasons.append("t256_extension_sentinel_passed_mismatch")
    expected_gates = {
        "selected_t_residual_gate_passed": selected_t_passed,
        "reference_t64_residual_gate_passed": t64_passed,
        "selected_k_gradient_gate_passed": selected_k_passed,
        "reference_gradient_viable": reference_viable,
        "k128_sentinel_gate_passed": k128_sentinel_passed,
        "t256_extension_residual_gate_passed": (
            extension_passed if t_extension_used else None
        ),
    }
    for key, expected_value in expected_gates.items():
        if record.get(key) is not expected_value:
            reasons.append(f"{key}_mismatch")
    passed = (
        selected_t_passed
        and t64_passed
        and (extension_passed if t_extension_used else True)
        and selected_k_passed
        and reference_viable
        and (
            k128_sentinel_passed is True
            if selected_k_value == K_REFERENCE
            else True
        )
        and not reasons
    )
    if record.get("passed") is not passed:
        reasons.append("epoch_passed_mismatch")
        passed = False
    if record.get("epoch") != expected_epoch:
        reasons.append("wrong_epoch")
        passed = False
    return bool(passed and not reasons), reasons


def _validate_post_tk_audit(
    context: ExecutionContext,
    audit: Mapping[str, Any],
    *,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    value = _mapping(audit, "post-training T/K audit")
    require_official_test_excluded(value, require_result_marker=True)
    _require_official_test_false_recursive(
        value, path="post-training T/K audit"
    )
    expected_top = {
        "schema_version": POST_TK_AUDIT_SCHEMA_VERSION,
        "surface_id": context.surface_id,
        "row_id": context.row["row_id"],
        "scheme": context.surface["scheme"],
        "selected_cell_id": selected["cell_id"],
        "selected_t": context.surface["inference_iterations"],
        "selected_k": context.surface["training_iterations"],
    }
    for key, expected_value in expected_top.items():
        if value.get(key) != expected_value:
            raise _error(
                f"post-training T/K audit.{key}",
                f"exactly {expected_value!r}",
                value.get(key),
            )
    expected_checkpoints = selected.get("epoch_checkpoints")
    if not isinstance(expected_checkpoints, list) or len(expected_checkpoints) != 3:
        raise _error(
            "selected candidate.epoch_checkpoints",
            "three epoch checkpoints",
            expected_checkpoints,
        )
    audits = value.get("epoch_audits")
    if not isinstance(audits, list) or len(audits) != 3:
        raise _error(
            "post-training T/K audit.epoch_audits",
            "three epoch audit records",
            audits,
        )
    all_passed = True
    normalized_audits: list[dict[str, Any]] = []
    for index, (raw, checkpoint) in enumerate(
        zip(audits, expected_checkpoints), start=1
    ):
        record = _mapping(raw, f"epoch_audits[{index - 1}]")
        if (
            record.get("epoch") != index
            or record.get("checkpoint_path") != checkpoint.get("path")
            or record.get("checkpoint_sha256") != checkpoint.get("sha256")
            or record.get("official_test_read") is not False
        ):
            raise _error(
                f"epoch_audits[{index - 1}]",
                "the matching official-test-free epoch checkpoint identity",
                record,
            )
        gate_passed, reasons = _validate_post_epoch_measurements(
            context,
            record,
            checkpoint=checkpoint,
            expected_epoch=index,
        )
        if reasons:
            raise _error(
                f"epoch_audits[{index - 1}]",
                "complete internally consistent T/K measurements",
                reasons,
            )
        all_passed = all_passed and gate_passed
        normalized_audits.append(record)
    if value.get("passed") is not all_passed:
        raise _error(
            "post-training T/K audit.passed",
            f"exactly {all_passed!r} from all epoch audits",
            value.get("passed"),
        )
    expected_status = "passed" if all_passed else "failed"
    if value.get("status") != expected_status:
        raise _error(
            "post-training T/K audit.status",
            f"exactly {expected_status!r}",
            value.get("status"),
        )
    value["epoch_audits"] = normalized_audits
    return value


def _execute_post_tk_audit(
    context: ExecutionContext,
    *,
    asset_dir: Path | None,
    data_root: Path | None,
    download: bool,
    device: str,
    tk_config_path: Path,
) -> dict[str, Any]:
    final = _load_stage_result(context, "select_expanded")
    if final.get("status") != "selected":
        return _zero_work(
            context,
            "post_training_tk",
            "final_selection_not_selected",
            status=str(final.get("status") or "zero_work"),
            dependency=final,
    )
    selected, candidate_dir = _selected_candidate(context, final)
    if asset_dir is None or data_root is None:
        return _zero_work(
            context,
            "post_training_tk",
            "post_training_tk_executor_not_configured",
            status="unresolved_post_training_tk",
            dependency=final,
        )
    epoch_checkpoints = selected.get("epoch_checkpoints")
    if not isinstance(epoch_checkpoints, list) or len(epoch_checkpoints) != 3:
        raise _error(
            "selected candidate.epoch_checkpoints",
            "exactly three epoch checkpoint records",
            epoch_checkpoints,
        )
    raw = run_builtin_post_training_tk_audit(
        context=context,
        selected_candidate=copy.deepcopy(selected),
        candidate_dir=candidate_dir,
        epoch_checkpoints=copy.deepcopy(epoch_checkpoints),
        asset_dir=asset_dir,
        data_root=data_root,
        device=device,
        download=download,
        tk_config_path=tk_config_path,
    )
    audit_input = {
        "mode": "built_in_tk_runtime",
        "tk_config_path": str(tk_config_path),
        "tk_config_sha256": sha256_file(tk_config_path),
    }
    audit = _validate_post_tk_audit(
        context, _mapping(raw, "post-training T/K audit"), selected=selected
    )
    status = (
        "complete"
        if audit["passed"] is True
        else "unresolved_post_training_tk"
    )
    return {
        **_base_result(
            context,
            "post_training_tk",
            status=status,
            zero_work=False,
            reason=(
                None
                if audit["passed"] is True
                else "one_or_more_epoch_tk_gates_failed"
            ),
        ),
        "selected_cell_id": selected["cell_id"],
        "audit_input": audit_input,
        "audit": audit,
    }


def _publish_surface_completion(
    context: ExecutionContext, final_result: Mapping[str, Any]
) -> None:
    target = context.surface_root / "completion.json"
    if target.exists():
        existing = _mapping(read_json(target), "surface completion")
        expected = {
            "schema_version": SURFACE_COMPLETION_SCHEMA_VERSION,
            "state": "complete",
            "study_id": context.spec.study_id,
            "config_sha256": context.spec.config_sha256,
            "manifest_id": context.manifest_id,
            "surface_id": context.surface_id,
            "host": context.host,
            "official_test_read": False,
        }
        for key, expected_value in expected.items():
            if existing.get(key) != expected_value:
                raise _error(
                    f"surface completion.{key}",
                    f"exactly {expected_value!r}",
                    existing.get(key),
                )
        _validate_artifacts(
            context.surface_root,
            existing.get("outputs"),
            path="surface completion.outputs",
        )
        return
    completion = {
        "schema_version": SURFACE_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": context.spec.study_id,
        "config_sha256": context.spec.config_sha256,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "host": context.host,
        "status": final_result["status"],
        "zero_work": final_result["zero_work"],
        "official_test_read": False,
        "outputs": [
            _artifact(
                _stage_dir(context, "finalize_lr") / "completion.json",
                base=context.surface_root,
            ),
            _artifact(
                _stage_dir(context, "finalize_lr") / "result.json",
                base=context.surface_root,
            ),
        ],
    }
    atomic_write_json(target, completion, canonical=True)


def _execute_finalize(context: ExecutionContext) -> dict[str, Any]:
    audit_stage = _load_stage_result(context, "post_training_tk")
    final_selection = _load_stage_result(context, "select_expanded")
    if (
        audit_stage.get("status") != "complete"
        or audit_stage.get("audit", {}).get("passed") is not True
    ):
        return _zero_work(
            context,
            "finalize_lr",
            str(
                audit_stage.get("reason")
                or audit_stage.get("status")
                or "unresolved_post_training_tk"
            ),
            status=(
                "unresolved_post_training_tk"
                if final_selection.get("status") == "selected"
                else str(final_selection.get("status") or "zero_work")
            ),
            dependency=audit_stage,
        )
    selected, _candidate_dir = _selected_candidate(context, final_selection)
    selection = _mapping(
        final_selection.get("selection"), "select_expanded.selection"
    )
    selected_contract = _mapping(
        selection.get("selected"), "select_expanded.selection.selected"
    )
    return {
        **_base_result(
            context,
            "finalize_lr",
            status=context.spec.data["selection"]["selected_status"],
        ),
        "selected_t": context.surface["inference_iterations"],
        "selected_k": context.surface["training_iterations"],
        "selected_cell_id": selected["cell_id"],
        "selected_rho": {
            "rho_conv": selected_contract["rho_conv"],
            "rho_dense": selected_contract["rho_dense"],
        },
        "raw_learning_rates_by_parameter": copy.deepcopy(
            selected["raw_learning_rates_by_parameter"]
        ),
        "final_validation_loss": selected["final_validation_loss"],
        "final_validation_accuracy": selected[
            "final_validation_accuracy"
        ],
        "median_projection_efficiency": selected[
            "median_projection_efficiency"
        ],
        "epoch_checkpoints": copy.deepcopy(
            selected["epoch_checkpoints"]
        ),
        "post_training_tk": copy.deepcopy(audit_stage["audit"]),
        "expansion_used": final_selection["expansion_used"],
        "long_confirmation": False,
    }


def _preflight_dir(context: ExecutionContext) -> Path:
    return (
        context.manifest_path.parent
        / "preflight"
        / "representative_canary"
        / context.host
    )


def _fixed_tk_gate_dir(context: ExecutionContext) -> Path:
    return (
        context.manifest_path.parent
        / "preflight"
        / "fixed_tk_gate"
        / context.host
        / context.row["row_id"]
    )


def _fixed_tk_audit_passed(
    audit: Mapping[str, Any],
    *,
    selected_t: int,
    selected_k: int,
) -> bool:
    t_passed, _t_reasons, t_complete = _candidate_t_pass(
        _mapping(audit.get("t_measurement"), "fixed T/K gate T measurement"),
        expected_iteration=selected_t,
    )
    t64_passed, _t64_reasons, t64_complete = _candidate_t_pass(
        _mapping(
            audit.get("t64_sentinel_measurement"),
            "fixed T/K gate T=64 measurement",
        ),
        expected_iteration=T_REFERENCE,
    )
    k_measurement = _mapping(
        audit.get("k_measurement"), "fixed T/K gate K measurement"
    )
    k_passed, _k_reasons, k_complete = _candidate_k_pass(
        k_measurement, CONV3_CONV_WEIGHTS
    )
    return (
        audit.get("selected_t") == selected_t
        and audit.get("selected_k") == selected_k
        and audit.get("k_audit_reference") == K_REFERENCE
        and audit.get("fresh_replay_after_selection") is True
        and audit.get("t64_sentinel_passed") is True
        and t_passed
        and t_complete
        and t64_passed
        and t64_complete
        and k_passed
        and k_complete
        and k_measurement.get("candidate_k") == selected_k
        and k_measurement.get("reference_k") == K_REFERENCE
        and k_measurement.get("selected_t") == selected_t
        and k_measurement.get("reference_viable") is True
        and audit.get("passed") is True
        and audit.get("status") == "passed"
    )


def _validate_fixed_tk_gate_completion(
    context: ExecutionContext,
    destination: Path,
) -> dict[str, Any]:
    completion_path = destination / "completion.json"
    result_path = destination / "result.json"
    if not completion_path.is_file() or not result_path.is_file():
        raise PerfectDiodeConv3ExecutionError(
            f"Expected a complete fixed T/K preflight gate: {destination}."
        )
    completion = _mapping(
        read_json(completion_path), "fixed T/K gate completion"
    )
    expected_completion = {
        "schema_version": FIXED_TK_GATE_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": context.spec.study_id,
        "config_sha256": context.spec.config_sha256,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "row_id": context.row["row_id"],
        "stage": "fixed_tk_gate",
        "gate_id": f"{context.row['row_id']}--t8-k8",
        "representative_surface_id": context.surface_id,
        "selected_t": context.surface["inference_iterations"],
        "selected_k": context.surface["training_iterations"],
        "official_test_read": False,
    }
    for key, expected_value in expected_completion.items():
        if completion.get(key) != expected_value:
            raise _error(
                f"fixed T/K gate completion.{key}",
                f"exactly {expected_value!r}",
                completion.get(key),
            )
    _validate_artifacts(
        destination,
        completion.get("outputs"),
        path="fixed T/K gate completion.outputs",
    )
    expected_outputs = [_artifact(result_path, base=destination)]
    if completion.get("outputs") != expected_outputs:
        raise _error(
            "fixed T/K gate completion.outputs",
            f"exactly {expected_outputs!r}",
            completion.get("outputs"),
        )
    result = _mapping(read_json(result_path), "fixed T/K gate result")
    audit = _mapping(
        result.get("operating_point_audit"),
        "fixed T/K gate result.operating_point_audit",
    )
    expected_result = {
        "schema_version": FIXED_TK_GATE_RESULT_SCHEMA_VERSION,
        "study_id": context.spec.study_id,
        "config_sha256": context.spec.config_sha256,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "row_id": context.row["row_id"],
        "scheme": context.surface["scheme"],
        "optimizer": "adam",
        "host": context.host,
        "stage": "fixed_tk_gate",
        "gate_id": f"{context.row['row_id']}--t8-k8",
        "representative_surface_id": context.surface_id,
        "surface_manifest_sha256": sha256_file(context.manifest_path),
        "selected_t": context.surface["inference_iterations"],
        "selected_k": context.surface["training_iterations"],
        "diagnostic_selected_t": context.row["upstream_tk"]["selected_t"],
        "diagnostic_selected_k": context.row["upstream_tk"]["selected_k"],
        "official_test_read": False,
    }
    for key, expected_value in expected_result.items():
        if result.get(key) != expected_value:
            raise _error(
                f"fixed T/K gate result.{key}",
                f"exactly {expected_value!r}",
                result.get(key),
            )
    if (
        result.get("restart_from_shared_initialization") is not True
        or result.get("shared_asset_hashes")
        != context.surface["shared_asset_hashes"]
        or result.get("execution_source")
        != context.surface["execution_source"]
    ):
        raise PerfectDiodeConv3ExecutionError(
            "Expected the fixed T/K gate result to preserve the exact "
            "manifest-bound initialization, shared assets, and execution source."
        )
    expected_audit = {
        "schema_version": TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION,
        "study_id": context.spec.data["upstream_tk"]["study_id"],
        "config_sha256": context.spec.data["upstream_tk"]["config_sha256"],
        "entry_id": f"{context.row['row_id']}--lr-fixed-tk-gate",
        "row_id": context.row["row_id"],
        "scheme": context.surface["scheme"],
        "execution_backend": "tmux",
        "execution_host": context.host,
        "execution_environment_sha256": context.surface[
            "execution_environment_sha256"
        ],
        "selected_t": context.surface["inference_iterations"],
        "selected_k": context.surface["training_iterations"],
        "k_audit_reference": K_REFERENCE,
        "fresh_replay_after_selection": True,
        "t_extension_used": False,
        "official_test_read": False,
    }
    for key, expected_value in expected_audit.items():
        if audit.get(key) != expected_value:
            raise _error(
                f"fixed T/K gate audit.{key}",
                f"exactly {expected_value!r}",
                audit.get(key),
            )
    shared = context.surface["shared_asset_hashes"]
    t_measurement = _mapping(
        audit.get("t_measurement"), "fixed T/K gate T measurement"
    )
    t64_measurement = _mapping(
        audit.get("t64_sentinel_measurement"),
        "fixed T/K gate T=64 measurement",
    )
    k_measurement = _mapping(
        audit.get("k_measurement"), "fixed T/K gate K measurement"
    )
    for label, measurement, cohort_key in (
        ("T=8", t_measurement, "t_cohort_indices_sha256"),
        ("T=64", t64_measurement, "t_cohort_indices_sha256"),
        ("K=8-vs-64", k_measurement, "k_cohort_indices_sha256"),
    ):
        expected_measurement = {
            "cohort_source_indices_sha256": shared[cohort_key],
            "initialization_checkpoint_sha256": shared[
                "initialization_checkpoint_sha256"
            ],
            "initialization_tensor_sha256": shared[
                "initialization_tensor_sha256"
            ],
            "official_test_read": False,
        }
        for key, expected_value in expected_measurement.items():
            if measurement.get(key) != expected_value:
                raise _error(
                    f"fixed T/K gate {label} measurement.{key}",
                    f"exactly {expected_value!r}",
                    measurement.get(key),
                )
    if (
        audit.get("t256_extension_sentinel_measurement") is not None
        or audit.get("t256_extension_sentinel_passed") is not None
    ):
        raise _error(
            "fixed T/K gate T=256 extension",
            "no extension measurement for the fixed core-grid T=8 point",
            {
                "measurement": audit.get(
                    "t256_extension_sentinel_measurement"
                ),
                "passed": audit.get("t256_extension_sentinel_passed"),
            },
        )
    _require_official_test_false_recursive(
        result, path="fixed T/K gate result"
    )
    passed = _fixed_tk_audit_passed(
        audit,
        selected_t=int(context.surface["inference_iterations"]),
        selected_k=int(context.surface["training_iterations"]),
    )
    expected_status = "passed" if passed else "failed"
    if (
        result.get("passed") is not passed
        or completion.get("passed") is not passed
        or result.get("status") != expected_status
        or completion.get("status") != expected_status
    ):
        raise _error(
            "fixed T/K gate pass state",
            f"passed={passed!r} and status={expected_status!r} from the "
            "residual and gradient audit",
            {
                "result_passed": result.get("passed"),
                "completion_passed": completion.get("passed"),
                "result_status": result.get("status"),
                "completion_status": completion.get("status"),
            },
        )
    return result


def execute_fixed_operating_point_preflight_gate(
    context: ExecutionContext,
    *,
    backend: NumericalBackend | None = None,
    asset_dir: str | Path | None = None,
    data_root: str | Path | None = None,
    download: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    """Audit the manifest-bound user-fixed T/K point at initialization."""

    operating_point = context.spec.data.get("operating_point")
    if not isinstance(operating_point, Mapping):
        raise PerfectDiodeConv3ExecutionError(
            "Expected a frozen LR operating-point amendment before running "
            "the fixed T/K preflight gate. Provided value: None."
        )
    expected_operating_point = {
        "mode": "user_fixed_after_residual_gradient_review",
        "inference_iterations": 8,
        "training_iterations": 8,
        "reference_inference_iterations": 64,
        "reference_training_iterations": 64,
        "shared_across_schemes": True,
        "retain_upstream_diagnostic_selection": True,
        "require_fresh_manifest_bound_preflight_gate": True,
    }
    for key, expected_value in expected_operating_point.items():
        if operating_point.get(key) != expected_value:
            raise _error(
                f"study.operating_point.{key}",
                f"exactly {expected_value!r}",
                operating_point.get(key),
            )
    if (
        context.surface["optimizer"] != "adam"
        or context.surface["inference_iterations"] != 8
        or context.surface["training_iterations"] != 8
    ):
        raise _error(
            "fixed T/K gate surface",
            "the canonical Adam surface with manifest-bound T=8 and K=8",
            {
                "surface_id": context.surface_id,
                "optimizer": context.surface["optimizer"],
                "T": context.surface["inference_iterations"],
                "K": context.surface["training_iterations"],
            },
        )
    destination = _fixed_tk_gate_dir(context)
    completion_path = destination / "completion.json"
    result_path = destination / "result.json"
    if completion_path.exists():
        return _validate_fixed_tk_gate_completion(context, destination)
    if result_path.exists():
        raise PerfectDiodeConv3ExecutionError(
            "Expected an interrupted fixed T/K gate with result.json but no "
            f"completion to be inspected before retry: {destination}."
        )
    numerical = backend or V1NumericalBackend()
    assets = _load_assets_for_stage(
        context,
        numerical,
        asset_dir=(
            None
            if asset_dir is None
            else Path(asset_dir).expanduser().resolve()
        ),
        data_root=(
            None
            if data_root is None
            else Path(data_root).expanduser().resolve()
        ),
        download=download,
        device=device,
    )
    if not isinstance(assets, LoadedV2Assets):
        raise _error(
            "fixed T/K gate assets",
            "the validated LoadedV2Assets production bundle",
            type(assets).__name__,
        )
    tk_rows = [
        row
        for row in assets.tk_spec.rows
        if row.get("scheme") == context.surface["scheme"]
    ]
    if len(tk_rows) != 1:
        raise _error(
            "fixed T/K gate T/K row",
            f"one row for scheme {context.surface['scheme']!r}",
            tk_rows,
        )
    audit = run_operating_point_audit(
        assets.tk_spec,
        tk_rows[0],
        selected_t=8,
        selected_k=8,
        assets=assets.tk_assets,
        device=device,
        entry_id=f"{context.row['row_id']}--lr-fixed-tk-gate",
        execution_host=context.host,
        t_extension_used=False,
        execution_environment_sha256=context.surface[
            "execution_environment_sha256"
        ],
    )
    passed = _fixed_tk_audit_passed(
        audit,
        selected_t=8,
        selected_k=8,
    )
    result = {
        **_base_result(
            context,
            "fixed_tk_gate",
            status="passed" if passed else "failed",
            reason=None if passed else "fixed_tk_residual_or_gradient_gate_failed",
        ),
        "schema_version": FIXED_TK_GATE_RESULT_SCHEMA_VERSION,
        "surface_manifest_sha256": sha256_file(context.manifest_path),
        "gate_id": f"{context.row['row_id']}--t8-k8",
        "representative_surface_id": context.surface_id,
        "selected_t": 8,
        "selected_k": 8,
        "diagnostic_selected_t": context.row["upstream_tk"]["selected_t"],
        "diagnostic_selected_k": context.row["upstream_tk"]["selected_k"],
        "operating_point_mode": operating_point["mode"],
        "restart_from_shared_initialization": True,
        "shared_asset_hashes": copy.deepcopy(
            context.surface["shared_asset_hashes"]
        ),
        "execution_source": copy.deepcopy(
            context.surface["execution_source"]
        ),
        "passed": passed,
        "operating_point_audit": audit,
    }
    require_official_test_excluded(result, require_result_marker=True)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(result_path, result, canonical=True)
    completion = {
        "schema_version": FIXED_TK_GATE_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": context.spec.study_id,
        "config_sha256": context.spec.config_sha256,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "row_id": context.row["row_id"],
        "stage": "fixed_tk_gate",
        "gate_id": f"{context.row['row_id']}--t8-k8",
        "representative_surface_id": context.surface_id,
        "selected_t": 8,
        "selected_k": 8,
        "status": result["status"],
        "passed": passed,
        "official_test_read": False,
        "outputs": _all_artifacts(
            destination, exclude=(completion_path,)
        ),
    }
    atomic_write_json(completion_path, completion, canonical=True)
    return _validate_fixed_tk_gate_completion(context, destination)


def _validate_preflight_completion(
    context: ExecutionContext,
    destination: Path,
) -> dict[str, Any]:
    completion_path = destination / "completion.json"
    result_path = destination / "result.json"
    if not completion_path.is_file() or not result_path.is_file():
        raise PerfectDiodeConv3ExecutionError(
            f"Expected a complete representative preflight: {destination}."
        )
    completion = _mapping(
        read_json(completion_path), "representative preflight completion"
    )
    expected = {
        "schema_version": PREFLIGHT_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": context.spec.study_id,
        "config_sha256": context.spec.config_sha256,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "stage": "preflight_canary",
        "official_test_read": False,
    }
    for key, expected_value in expected.items():
        if completion.get(key) != expected_value:
            raise _error(
                f"representative preflight completion.{key}",
                f"exactly {expected_value!r}",
                completion.get(key),
            )
    _validate_artifacts(
        destination,
        completion.get("outputs"),
        path="representative preflight completion.outputs",
    )
    result = _mapping(
        read_json(result_path), "representative preflight result"
    )
    if (
        result.get("manifest_id") != context.manifest_id
        or result.get("surface_id") != context.surface_id
        or result.get("stage") != "preflight_canary"
        or result.get("official_test_read") is not False
    ):
        raise PerfectDiodeConv3ExecutionError(
            "Expected the representative preflight result to match the "
            "manifest-bound surface and exclude the official test split."
        )
    return result


def execute_representative_preflight_canary(
    context: ExecutionContext,
    *,
    backend: NumericalBackend | None = None,
    asset_dir: str | Path | None = None,
    data_root: str | Path | None = None,
    download: bool = False,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run one isolated manifest-selected Adam canary through production code."""

    plan = representative_preflight_plan(
        context.manifest,
        spec=context.spec,
        host=context.host,
    )
    expected_plan = {
        "status": "required",
        "stage": "preflight_canary",
        "same_public_runner": True,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "row_id": context.row["row_id"],
        "optimizer": context.surface["optimizer"],
        "host": context.host,
        "steps": CANARY_STEPS,
        "restart_from_shared_initialization": True,
        "execution_environment_sha256": context.surface[
            "execution_environment_sha256"
        ],
        "official_test_read": False,
    }
    for key, expected_value in expected_plan.items():
        if plan.get(key) != expected_value:
            raise _error(
                f"representative preflight plan.{key}",
                f"exactly {expected_value!r}",
                plan.get(key),
            )
    if (
        plan.get("shared_asset_hashes")
        != context.surface["shared_asset_hashes"]
        or plan.get("execution_source")
        != context.surface["execution_source"]
    ):
        raise PerfectDiodeConv3ExecutionError(
            "Expected the representative preflight plan to preserve the "
            "surface's exact asset and execution authority."
        )
    rho_conv = _finite_positive(
        plan.get("rho_conv"), "representative preflight plan.rho_conv"
    )
    rho_dense = _finite_positive(
        plan.get("rho_dense"), "representative preflight plan.rho_dense"
    )
    destination = _preflight_dir(context)
    completion_path = destination / "completion.json"
    result_path = destination / "result.json"
    if completion_path.exists():
        return _validate_preflight_completion(context, destination)
    if result_path.exists():
        raise PerfectDiodeConv3ExecutionError(
            "Expected an interrupted representative preflight with result.json "
            f"but no completion to be inspected before retry: {destination}."
        )
    numerical = backend or V1NumericalBackend()
    assets = _load_assets_for_stage(
        context,
        numerical,
        asset_dir=(
            None
            if asset_dir is None
            else Path(asset_dir).expanduser().resolve()
        ),
        data_root=(
            None
            if data_root is None
            else Path(data_root).expanduser().resolve()
        ),
        download=download,
        device=device,
    )
    destination.mkdir(parents=True, exist_ok=True)
    probe = _mapping(
        numerical.optimizer_probe(
            context,
            assets,
            output_dir=destination / "optimizer_probe",
            device=device,
        ),
        "representative preflight optimizer probe",
    )
    require_official_test_excluded(probe, require_result_marker=True)
    canary: dict[str, Any] | None = None
    if probe.get("probe_stable") is True:
        canary = _mapping(
            numerical.canary(
                context,
                assets,
                probe,
                rho_conv=rho_conv,
                rho_dense=rho_dense,
                output_dir=destination / "cell",
                device=device,
            ),
            "representative preflight canary",
        )
        require_official_test_excluded(
            canary, require_result_marker=True
        )
    passed = (
        probe.get("probe_stable") is True
        and isinstance(canary, Mapping)
        and canary.get("safety_clean") is True
        and canary.get("completed_steps") == CANARY_STEPS
    )
    result = {
        **_base_result(
            context,
            "preflight_canary",
            status="complete" if passed else "failed_preflight_canary",
            reason=None if passed else "representative_canary_did_not_pass",
        ),
        "passed": passed,
        "steps": CANARY_STEPS,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "restart_from_shared_initialization": True,
        "isolated_from_public_stage_outputs": True,
        "output_namespace": (
            f"preflight/representative_canary/{context.host}"
        ),
        "shared_asset_hashes": copy.deepcopy(
            context.surface["shared_asset_hashes"]
        ),
        "execution_source": copy.deepcopy(
            context.surface["execution_source"]
        ),
        "probe": probe,
        "canary": canary,
    }
    require_official_test_excluded(result, require_result_marker=True)
    atomic_write_json(result_path, result, canonical=True)
    completion = {
        "schema_version": PREFLIGHT_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "study_id": context.spec.study_id,
        "config_sha256": context.spec.config_sha256,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "stage": "preflight_canary",
        "status": result["status"],
        "passed": passed,
        "official_test_read": False,
        "outputs": _all_artifacts(
            destination, exclude=(completion_path,)
        ),
    }
    atomic_write_json(completion_path, completion, canonical=True)
    return _validate_preflight_completion(context, destination)


def execute_surface_stage(
    context: ExecutionContext,
    stage: str,
    *,
    backend: NumericalBackend | None = None,
    asset_dir: str | Path | None = None,
    data_root: str | Path | None = None,
    download: bool = False,
    device: str = "cuda",
    tk_config_path: str | Path = DEFAULT_TK_CONFIG,
) -> dict[str, Any]:
    """Execute or resume one public surface stage without launching a job."""

    if stage not in PUBLIC_STAGE_SEQUENCE:
        raise _error(
            "stage", f"one of {PUBLIC_STAGE_SEQUENCE!r}", stage
        )
    destination = _stage_dir(context, stage)
    if (destination / "completion.json").exists():
        return _validate_completion(context, stage, destination)
    numerical = backend or V1NumericalBackend()
    asset_path = (
        None if asset_dir is None else Path(asset_dir).expanduser().resolve()
    )
    data_path = (
        None if data_root is None else Path(data_root).expanduser().resolve()
    )
    resolved_tk_config = Path(tk_config_path).expanduser().resolve()
    handlers: dict[str, Callable[[], dict[str, Any]]] = {
        "audit": lambda: _execute_audit(context),
        "import_tk": lambda: _execute_assets(
            context,
            numerical,
            asset_dir=asset_path,
            data_root=data_path,
            download=download,
            device=device,
        ),
        "optimizer_probe": lambda: _execute_probe(
            context,
            numerical,
            asset_dir=asset_path,
            data_root=data_path,
            download=download,
            device=device,
        ),
        "rho_canary_core": lambda: _execute_core_canaries(
            context,
            numerical,
            asset_dir=asset_path,
            data_root=data_path,
            download=download,
            device=device,
        ),
        "rho_core_candidates": lambda: _execute_candidate_stage(
            context,
            numerical,
            stage="rho_core_candidates",
            canary_stage_name="rho_canary_core",
            asset_dir=asset_path,
            data_root=data_path,
            download=download,
            device=device,
        ),
        "select_core": lambda: _execute_select_core(context),
        "rho_canary_expansion": lambda: _execute_extension_canaries(
            context,
            numerical,
            asset_dir=asset_path,
            data_root=data_path,
            download=download,
            device=device,
        ),
        "rho_expansion_candidates": lambda: _execute_candidate_stage(
            context,
            numerical,
            stage="rho_expansion_candidates",
            canary_stage_name="rho_canary_expansion",
            asset_dir=asset_path,
            data_root=data_path,
            download=download,
            device=device,
        ),
        "select_expanded": lambda: _execute_select_expanded(context),
        "post_training_tk": lambda: _execute_post_tk_audit(
            context,
            asset_dir=asset_path,
            data_root=data_path,
            download=download,
            device=device,
            tk_config_path=resolved_tk_config,
        ),
        "finalize_lr": lambda: _execute_finalize(context),
    }
    result = _publish_stage(context, stage, handlers[stage]())
    if stage == "finalize_lr":
        _publish_surface_completion(context, result)
    return result


__all__ = [
    "CELL_COMPLETION_SCHEMA_VERSION",
    "DEFAULT_TK_CONFIG",
    "EPOCH_K_VIABILITY_SCHEMA_VERSION",
    "ENVIRONMENT_CONTRACT_SCHEMA_VERSION",
    "ENVIRONMENT_RECEIPT_SCHEMA_VERSION",
    "EXECUTION_COMPLETION_SCHEMA_VERSION",
    "EXECUTION_STAGE_SCHEMA_VERSION",
    "ExecutionContext",
    "FIXED_TK_GATE_COMPLETION_SCHEMA_VERSION",
    "FIXED_TK_GATE_RESULT_SCHEMA_VERSION",
    "LoadedV2Assets",
    "NumericalBackend",
    "PREFLIGHT_COMPLETION_SCHEMA_VERSION",
    "POST_TK_AUDIT_SCHEMA_VERSION",
    "PerfectDiodeConv3ExecutionError",
    "SURFACE_COMPLETION_SCHEMA_VERSION",
    "V1NumericalBackend",
    "build_environment_contract",
    "derive_shared_asset_hashes",
    "execute_fixed_operating_point_preflight_gate",
    "execute_representative_preflight_canary",
    "execute_surface_stage",
    "load_execution_context",
    "observe_execution_environment",
    "run_builtin_post_training_tk_audit",
    "validate_environment_contract",
    "validate_environment_receipt",
    "verify_current_execution_environment",
]
