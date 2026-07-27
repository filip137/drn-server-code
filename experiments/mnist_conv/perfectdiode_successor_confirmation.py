"""Immutable successor runtime for the Conv1/Conv2 perfect-diode confirmations.

The parent LR screen remains immutable.  This module consumes its verified
seed-0 initialization, split, optimizer-probe, fixed-T/K security, and
three-epoch candidate records through a separate twelve-entry input bundle.
Production entries always restart from the seed-0 initialization and use the
exact raw learning-rate vector re-derived from the staged optimizer probe.
The official MNIST test split is prohibited throughout.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .identity import code_fingerprint, sha256_file, sha256_json
from .io import atomic_write_json, read_json
from .lr_artifacts import (
    load_stage_manifest,
    validate_entry_completion,
)
from . import perfectdiode_hparam_runtime as _hparam_runtime
from .perfectdiode_hparam_runtime import (
    derive_raw_learning_rates,
    require_official_test_excluded,
)
from .perfectdiode_hparam_spec import (
    CANDIDATE_EPOCHS,
    CANDIDATE_TOTAL_STEPS,
    PerfectDiodeHparamStudySpec,
)


CONFIG_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-best-observed-confirmation/v1"
)
BUNDLE_MANIFEST_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-successor-input-bundle/v1"
)
ENTRY_COMPLETION_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-successor-entry-completion/v1"
)
STATUS_SCHEMA_VERSION = "mnist-conv-perfectdiode-successor-status/v1"
SMOKE_RECEIPT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-successor-smoke-receipt/v1"
)
SMOKE_PACK_RECEIPT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-successor-smoke-pack-receipt/v1"
)
CANARY_CHILD_RECEIPT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-successor-canary-child-receipt/v1"
)
PRODUCTION_PACK_RECEIPT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-successor-production-pack-receipt/v1"
)
PREFLIGHT_SCHEMA_VERSION = (
    "mnist-conv-perfectdiode-successor-preflight/v1"
)
CUSTOM_PREFLIGHT_RECEIPT_SCHEMA_VERSION = (
    "perfectdiode-successor-preflight-receipt/v1"
)
LAUNCH_AUTHORIZATION_SCHEMA_VERSION = (
    "perfectdiode-successor-launch-authorization/v1"
)
GATE_INPUT_SCHEMA_VERSION = "perfectdiode-successor-gate-input/v1"
CANARY_GATE_SCHEMA_VERSION = "perfectdiode-successor-canary-gate/v1"
SCHEDULED_PREFLIGHT_RECEIPT_SCHEMA_VERSION = (
    "scheduled-run-preflight-receipt/v1"
)

PARENT_STUDY_ID = (
    "lrstudy_5afe8bc7c9180cd1c677d1d87a497e1d6a89c1599cd015d7a6c127ebb18f2e46"
)
PARENT_CONFIG_FILE_SHA256 = (
    "647f92e01c8619951367d71cadd6320c6924c0223a2f5ed344f7ff1739bac5c1"
)
PARENT_CONFIG_SHA256 = (
    "699804e1f7c35f65f50656db4af0b120dd3a3d0f2fb95e3a17b65ebfe7b78535"
)
HIGH_RHO_CONTINUATION_ID = (
    "pdcontinuation_6d967e1bddba4a70cb6bcd81e7f739488389271553fa7eb8f099de688fa4922b"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_LAUNCHER = (
    REPO_ROOT
    / "experiments"
    / "run_mnist_conv_perfectdiode_successor_confirmation.py"
)
RUNTIME_MODULE = Path(__file__).resolve()
JEAN_ZAY_WRAPPER = (
    REPO_ROOT
    / "experiments"
    / "run_mnist_conv_perfectdiode_successor_confirmation_jeanzay.slurm"
)
SCHEDULED_PREFLIGHT_SCRIPT = (
    REPO_ROOT
    / "skills"
    / "scheduled-run-preflight"
    / "scripts"
    / "preflight_scheduled_runner.py"
)
JEAN_ZAY_SUBMITTER = (
    REPO_ROOT
    / "experiments"
    / "submit_mnist_conv_perfectdiode_successor_confirmation_jeanzay.py"
)
JEAN_ZAY_SUPERVISOR = (
    REPO_ROOT
    / "experiments"
    / "supervise_mnist_conv_perfectdiode_successor_confirmation_jeanzay.py"
)
SEMANTIC_CANARY_VERIFIER = (
    REPO_ROOT
    / "experiments"
    / "verify_mnist_conv_perfectdiode_successor_canary.py"
)
EXPERIMENT_PLAN_VALIDATOR = (
    REPO_ROOT
    / "skills"
    / "run-experiment-pipeline"
    / "scripts"
    / "validate_experiment_plan.py"
)
JEAN_ZAY_PYTHON = (
    "/lustre/fshomisc/sup/hpe/pub/miniforge/24.9.0/envs/"
    "pytorch-gpu-2.5.0+py3.12.7/bin/python"
)

ARCHITECTURE_ORDER = ("conv1", "conv2")
SCHEME_ORDER = ("baseline", "ours", "legacy")
OPTIMIZER_ORDER = ("sgd", "adam")
FIXED_ENTRY_PACKS = tuple((index, index + 1) for index in range(0, 12, 2))
RUNS_PER_GPU = 2
PRODUCTION_PACK_COUNT = len(FIXED_ENTRY_PACKS)
CANARY_PACK_INDEX = 4
CANARY_ENTRY_INDICES = FIXED_ENTRY_PACKS[CANARY_PACK_INDEX]
RESULT_JSON_MARKER = "PD_SUCCESSOR_RESULT_JSON="
PAIRED_CANARY_GPU_MEMORY_BYTES = 32 * 1024**3
PAIRED_CANARY_MAX_PEAK_NUMERATOR = 4
PAIRED_CANARY_MAX_PEAK_DENOMINATOR = 5
PAIRED_CANARY_MAX_PROJECTED_SECONDS = 8 * 60 * 60
_FROZEN_V1_TRAINING_CONTRACT = _hparam_runtime.training_contract
_SUCCESSOR_TRAINING_CONTRACT_LOCK = threading.RLock()
EXPECTED_ENTRY_ORDER = tuple(
    (architecture, scheme, optimizer)
    for architecture in ARCHITECTURE_ORDER
    for scheme in SCHEME_ORDER
    for optimizer in OPTIMIZER_ORDER
)
EXPECTED_ARCHITECTURES = {
    "conv1": {
        "input_gain": 40.0,
        "T": 4,
        "K": 4,
        "reference_T": 64,
        "reference_K": 64,
        "epochs": 10,
        "steps_per_epoch": 3_438,
        "expected_steps": 34_380,
    },
    "conv2": {
        "input_gain": 100.0,
        "T": 6,
        "K": 6,
        "reference_T": 64,
        "reference_K": 64,
        "epochs": 20,
        "steps_per_epoch": 3_438,
        "expected_steps": 68_760,
    },
}


def _successor_training_contract(
    *,
    architecture: str,
    mode: str,
) -> Any:
    """Resolve successor-only horizons without editing the frozen v1 engine."""

    if architecture not in EXPECTED_ARCHITECTURES:
        raise ValueError(
            "Expected architecture to be 'conv1' or 'conv2'. "
            f"Provided value: {architecture!r}."
        )
    if mode == "successor_confirmation":
        contract = EXPECTED_ARCHITECTURES[architecture]
        epochs = int(contract["epochs"])
        total_steps = int(contract["expected_steps"])
    elif mode == "successor_canary":
        epochs = 1
        total_steps = 3_438
    else:
        return _FROZEN_V1_TRAINING_CONTRACT(
            architecture=architecture,
            mode=mode,
        )
    return _hparam_runtime.TrainingContract(
        mode=mode,
        architecture=architecture,
        epochs=epochs,
        steps_per_epoch=3_438,
        total_steps=total_steps,
        restart_from_shared_initialization=True,
        validate_after_every_epoch=True,
        save_best_validation_loss=True,
        save_final=True,
    )


@contextmanager
def _successor_training_contract_scope():
    """Temporarily extend the v1 engine inside one isolated worker process."""

    with _SUCCESSOR_TRAINING_CONTRACT_LOCK:
        current = _hparam_runtime.training_contract
        if current is not _FROZEN_V1_TRAINING_CONTRACT:
            raise RuntimeError(
                "Expected the frozen v1 training contract function to remain "
                f"unmodified. Provided value: {current!r}."
            )
        _hparam_runtime.training_contract = _successor_training_contract
        try:
            yield
        finally:
            _hparam_runtime.training_contract = (
                _FROZEN_V1_TRAINING_CONTRACT
            )


def execute_successor_stage_entry(
    study: Mapping[str, Any] | PerfectDiodeHparamStudySpec,
    shard_dir: str | Path,
    stage: str,
    payload: Mapping[str, Any],
    *,
    data_root: str | Path,
    device: str,
    download: bool = False,
    entry_id: str | None = None,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run successor stages through an adapter around the frozen v1 engine."""

    data, spec = _hparam_runtime._study_and_spec(study)
    if not isinstance(payload, Mapping):
        raise TypeError(
            "Expected payload to be a mapping. "
            f"Provided value: {type(payload).__name__}."
        )
    normalized = dict(payload)
    payload_entry_id = normalized.get("entry_id")
    if entry_id is not None:
        if payload_entry_id is not None and payload_entry_id != entry_id:
            raise ValueError(
                "Expected entry_id keyword to match payload.entry_id. "
                f"Provided value: keyword={entry_id!r}, "
                f"payload={payload_entry_id!r}."
            )
        normalized["entry_id"] = entry_id
    resolved_entry_id = normalized.get("entry_id")
    if not isinstance(resolved_entry_id, str) or not resolved_entry_id:
        raise ValueError(
            "Expected payload.entry_id to be a non-empty string. "
            f"Provided value: {resolved_entry_id!r}."
        )
    root = Path(shard_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Expected successor output_dir to remain outside the immutable "
            f"input bundle. Provided value: {output}."
        )
    output.mkdir(parents=True, exist_ok=True)
    if stage == "fixed_tk_gradient_security":
        return _hparam_runtime._execute_security_entry(
            data,
            spec,
            root,
            output,
            normalized,
            data_root=data_root,
            download=download,
            device=device,
        )
    mode_by_stage = {
        "successor_confirm": "successor_confirmation",
        "successor_canary": "successor_canary",
    }
    mode = mode_by_stage.get(stage)
    if mode is None:
        raise ValueError(
            "Expected successor stage to be "
            "'fixed_tk_gradient_security', 'successor_confirm', or "
            f"'successor_canary'. Provided value: {stage!r}."
        )
    with _successor_training_contract_scope():
        return _hparam_runtime._execute_training_entry(
            data,
            spec,
            root,
            output,
            normalized,
            data_root=data_root,
            download=download,
            device=device,
            mode=mode,
        )
EXPECTED_ROWS = {
    ("conv1", "baseline"): ("conv1_baseline_v1_c1", 1.0, 1.0),
    ("conv1", "ours"): ("conv1_ours_v4_c1", 4.0, 1.0),
    ("conv1", "legacy"): ("conv1_legacy_v4_c0p25", 4.0, 0.25),
    ("conv2", "baseline"): ("conv2_baseline_v1_c1", 1.0, 1.0),
    ("conv2", "ours"): ("conv2_ours_v4_c1", 4.0, 1.0),
    ("conv2", "legacy"): ("conv2_legacy_v4_c0p25", 4.0, 0.25),
}
EXPECTED_RHOS = {
    ("conv1", "baseline", "sgd"): (0.009, 0.03),
    ("conv1", "baseline", "adam"): (0.009, 0.03),
    ("conv1", "ours", "sgd"): (0.009, 0.03),
    ("conv1", "ours", "adam"): (0.009, 0.03),
    ("conv1", "legacy", "sgd"): (0.003, 0.03),
    ("conv1", "legacy", "adam"): (0.003, 0.01),
    ("conv2", "baseline", "sgd"): (0.027, 0.09),
    ("conv2", "baseline", "adam"): (0.027, 0.09),
    ("conv2", "ours", "sgd"): (0.009, 0.03),
    ("conv2", "ours", "adam"): (0.027, 0.03),
    ("conv2", "legacy", "sgd"): (0.003, 0.03),
    ("conv2", "legacy", "adam"): (0.009, 0.01),
}
EXPECTED_BOUNDARY_STATUS = {
    ("conv1", "baseline", "sgd"): "core_upper_edge_unresolved",
    ("conv1", "baseline", "adam"): "core_upper_edge_unresolved",
    ("conv1", "ours", "sgd"): "core_upper_edge_unresolved",
    ("conv1", "ours", "adam"): "core_upper_edge_unresolved",
    ("conv1", "legacy", "sgd"): "core_upper_edge_unresolved",
    ("conv1", "legacy", "adam"): "core_bracketed_unpublished_selector",
    ("conv2", "baseline", "sgd"): "partial_corner_or_core_non_frozen",
    ("conv2", "baseline", "adam"): "partial_corner_or_core_non_frozen",
    ("conv2", "ours", "sgd"): "partial_corner_or_core_non_frozen",
    ("conv2", "ours", "adam"): "partial_corner_or_core_non_frozen",
    ("conv2", "legacy", "sgd"): "partial_corner_or_core_non_frozen",
    (
        "conv2",
        "legacy",
        "adam",
    ): "core_upper_edge_unresolved_not_expanded",
}

PRODUCTION_OUTPUT_NAMES = (
    "best_validation.pt",
    "final.pt",
    "result.json",
    "run_spec.json",
    "step_log.csv",
    "validation.json",
)


class PerfectDiodeSuccessorError(ValueError):
    """A successor config, bundle, preflight, or output failed closed."""


def _error(path: str, expected: str, provided: Any) -> PerfectDiodeSuccessorError:
    return PerfectDiodeSuccessorError(
        f"Expected {path} to be {expected}. Provided value: {provided!r}."
    )


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(path, "a JSON object", value)
    return copy.deepcopy(dict(value))


def _list(value: Any, path: str, *, length: int | None = None) -> list[Any]:
    if not isinstance(value, list):
        raise _error(path, "a JSON array", value)
    if length is not None and len(value) != length:
        raise _error(path, f"an array of length {length}", value)
    return copy.deepcopy(value)


def _sha256(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(path, "a lowercase SHA-256 digest", value)
    return value


def _commit(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(path, "a lowercase 40- or 64-character commit id", value)
    return value


def _positive_number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise _error(path, "a positive finite number", value)
    return float(value)


def _portable_relative(value: Any, path: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
        or Path(value).as_posix() != value
    ):
        raise _error(path, "a normalized repo-root-relative path", value)
    return value


def _repo_path(value: Any, path: str) -> Path:
    relative = _portable_relative(value, path)
    result = (REPO_ROOT / relative).resolve()
    try:
        result.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise _error(path, "a path inside the repository", value) from exc
    return result


def _require_equal(value: Any, expected: Any, path: str) -> None:
    if value != expected:
        raise _error(path, f"exactly {expected!r}", value)


def _require_official_test_false_recursive(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if (
                key == "official_test_read"
                and not path.endswith(".prohibitions")
                and item is not False
            ):
                raise _error(f"{path}.{key}", "exactly false", item)
            _require_official_test_false_recursive(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_official_test_false_recursive(item, f"{path}[{index}]")


def _approval_state(config: Mapping[str, Any]) -> dict[str, bool]:
    status = _mapping(config.get("status"), "config.status")
    override = _mapping(
        config.get("selection_override"), "config.selection_override"
    )
    authorities = _mapping(config.get("authorities"), "config.authorities")
    execution = _mapping(config.get("execution"), "config.execution")
    plan = _mapping(
        authorities.get("experiment_plan"),
        "config.authorities.experiment_plan",
    )
    return {
        "approved_plan": (
            status.get("plan") == "approved"
            and isinstance(plan.get("path"), str)
            and bool(plan["path"].strip())
        ),
        "launch_authorized": status.get("launch_authorized") is True,
        "selection_override_approved": (
            override.get("state") == "approved"
            and override.get("approved") is True
            and isinstance(override.get("approved_by"), str)
            and bool(override["approved_by"].strip())
            and isinstance(override.get("approved_at"), str)
            and bool(override["approved_at"].strip())
        ),
        "implementation_ready": (
            status.get("implementation") == "reviewed_ready_for_launch"
            and execution.get("implementation_state")
            == "reviewed_ready_for_launch"
        ),
    }


def validate_config(
    value: Mapping[str, Any],
    *,
    verify_authority_files: bool = False,
    require_production_approval: bool = False,
) -> dict[str, Any]:
    """Validate the rich successor config without changing any files."""

    config = _mapping(value, "config")
    _require_equal(
        config.get("schema_version"),
        CONFIG_SCHEMA_VERSION,
        "config.schema_version",
    )
    if not isinstance(config.get("experiment_id"), str) or not config[
        "experiment_id"
    ].strip():
        raise _error("config.experiment_id", "a non-empty string", config.get("experiment_id"))
    status = _mapping(config.get("status"), "config.status")
    if status.get("implementation") not in {
        "implemented_preflight_pending_approval",
        "reviewed_ready_for_launch",
    }:
        raise _error(
            "config.status.implementation",
            "implemented_preflight_pending_approval or "
            "reviewed_ready_for_launch",
            status.get("implementation"),
        )

    evidence = _mapping(config.get("evidence_scope"), "config.evidence_scope")
    exact_evidence = {
        "class": "ordinary_mnist_optimization_diagnostic",
        "paper_facing": False,
        "selection_role": "diagnostic_best_observed_override",
        "freezes_learning_rate": False,
        "cross_scheme_superiority_allowed": False,
        "replaces_medium_affine_handoff": False,
    }
    for key, expected in exact_evidence.items():
        _require_equal(evidence.get(key), expected, f"config.evidence_scope.{key}")

    dataset = _mapping(config.get("dataset"), "config.dataset")
    official = _mapping(
        dataset.get("official_test"), "config.dataset.official_test"
    )
    _require_equal(official.get("enabled"), False, "config.dataset.official_test.enabled")
    _require_equal(
        official.get("read_allowed"),
        False,
        "config.dataset.official_test.read_allowed",
    )
    for key, expected in (
        ("name", "mnist"),
        ("variant", "ordinary"),
        ("affine_enabled", False),
        ("train_size", 55_000),
        ("train_batch_size", 16),
        ("train_shuffle_seed", 0),
        ("steps_per_epoch", 3_438),
        ("validation_size", 5_000),
        ("validation_split_seed", 0),
    ):
        _require_equal(dataset.get(key), expected, f"config.dataset.{key}")
    _sha256(dataset.get("train_indices_sha256"), "config.dataset.train_indices_sha256")
    _sha256(
        dataset.get("validation_indices_sha256"),
        "config.dataset.validation_indices_sha256",
    )

    model = _mapping(config.get("model"), "config.model")
    _require_equal(model.get("model_seed"), 0, "config.model.model_seed")
    _require_equal(
        model.get("nonlinearity"), "perfect_diode", "config.model.nonlinearity"
    )
    architectures = _mapping(
        model.get("architectures"), "config.model.architectures"
    )
    contracts = _mapping(
        config.get("training_contracts"), "config.training_contracts"
    )
    shared = _mapping(contracts.get("shared"), "config.training_contracts.shared")
    for key, expected in (
        ("restart_from_shared_initialization", True),
        ("continue_from_three_epoch_candidate", False),
        ("schedule", "constant_parameter_specific_lr_vector"),
        ("warmup_steps", 0),
        ("scheduler_enabled", False),
        ("official_test_evaluation", False),
    ):
        _require_equal(shared.get(key), expected, f"config.training_contracts.shared.{key}")
    for architecture, expected in EXPECTED_ARCHITECTURES.items():
        arch = _mapping(
            architectures.get(architecture),
            f"config.model.architectures.{architecture}",
        )
        contract = _mapping(
            contracts.get(architecture),
            f"config.training_contracts.{architecture}",
        )
        for key in ("input_gain", "T", "K", "reference_T", "reference_K"):
            _require_equal(
                arch.get(key),
                expected[key],
                f"config.model.architectures.{architecture}.{key}",
            )
        for key in ("epochs", "steps_per_epoch", "expected_steps"):
            config_key = "total_steps" if key == "expected_steps" else key
            _require_equal(
                contract.get(config_key),
                expected[key],
                f"config.training_contracts.{architecture}.{config_key}",
            )
        _sha256(
            arch.get("initialization_checkpoint_sha256"),
            f"config.model.architectures.{architecture}.initialization_checkpoint_sha256",
        )
        _sha256(
            arch.get("initialization_tensor_sha256"),
            f"config.model.architectures.{architecture}.initialization_tensor_sha256",
        )

    authorities = _mapping(config.get("authorities"), "config.authorities")
    parent = _mapping(
        authorities.get("parent_study"), "config.authorities.parent_study"
    )
    _require_equal(
        parent.get("study_id"),
        PARENT_STUDY_ID,
        "config.authorities.parent_study.study_id",
    )
    _require_equal(
        parent.get("config_sha256"),
        PARENT_CONFIG_FILE_SHA256,
        "config.authorities.parent_study.config_sha256",
    )
    _require_equal(
        parent.get("resolved_config_sha256"),
        PARENT_CONFIG_SHA256,
        "config.authorities.parent_study.resolved_config_sha256",
    )
    parent_config_path = _repo_path(
        parent.get("config_path"), "config.authorities.parent_study.config_path"
    )
    shards = _mapping(
        parent.get("shards"), "config.authorities.parent_study.shards"
    )
    shard_paths = {
        architecture: _repo_path(
            shards.get(architecture),
            f"config.authorities.parent_study.shards.{architecture}",
        )
        for architecture in ARCHITECTURE_ORDER
    }
    continuation = _mapping(
        authorities.get("conv2_high_rho_continuation"),
        "config.authorities.conv2_high_rho_continuation",
    )
    _require_equal(
        continuation.get("continuation_id"),
        HIGH_RHO_CONTINUATION_ID,
        "config.authorities.conv2_high_rho_continuation.continuation_id",
    )
    for key in (
        "config_sha256",
        "manifest_sha256",
        "summary_sha256",
        "card_sha256",
        "review_sha256",
    ):
        _sha256(
            continuation.get(key),
            f"config.authorities.conv2_high_rho_continuation.{key}",
        )
    for key in (
        "config_path",
        "manifest_path",
        "summary_path",
        "card_path",
        "review_path",
    ):
        _repo_path(
            continuation.get(key),
            f"config.authorities.conv2_high_rho_continuation.{key}",
        )

    entries = _list(config.get("entries"), "config.entries", length=12)
    seen_ids: set[str] = set()
    for index, expected_identity in enumerate(EXPECTED_ENTRY_ORDER):
        entry = _mapping(entries[index], f"config.entries[{index}]")
        architecture, scheme, optimizer = expected_identity
        for key, expected in (
            ("architecture", architecture),
            ("scheme", scheme),
            ("optimizer", optimizer),
        ):
            _require_equal(entry.get(key), expected, f"config.entries[{index}].{key}")
        entry_id = entry.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id or entry_id in seen_ids:
            raise _error(
                f"config.entries[{index}].entry_id",
                "a unique non-empty string",
                entry_id,
            )
        seen_ids.add(entry_id)
        row_id, voltage_amp, current_amp = EXPECTED_ROWS[(architecture, scheme)]
        for key, expected in (
            ("row_id", row_id),
            ("voltage_amp", voltage_amp),
            ("current_amp", current_amp),
            ("epochs", EXPECTED_ARCHITECTURES[architecture]["epochs"]),
            (
                "expected_steps",
                EXPECTED_ARCHITECTURES[architecture]["expected_steps"],
            ),
            ("boundary_status", EXPECTED_BOUNDARY_STATUS[expected_identity]),
        ):
            _require_equal(entry.get(key), expected, f"config.entries[{index}].{key}")
        rho_conv, rho_dense = EXPECTED_RHOS[expected_identity]
        _require_equal(float(entry.get("rho_conv")), rho_conv, f"config.entries[{index}].rho_conv")
        _require_equal(
            float(entry.get("rho_dense")), rho_dense, f"config.entries[{index}].rho_dense"
        )
        rates = _mapping(
            entry.get("raw_learning_rates_by_parameter"),
            f"config.entries[{index}].raw_learning_rates_by_parameter",
        )
        if not rates:
            raise _error(
                f"config.entries[{index}].raw_learning_rates_by_parameter",
                "a non-empty parameter LR mapping",
                rates,
            )
        for name, rate in rates.items():
            if not isinstance(name, str) or not name:
                raise _error(
                    f"config.entries[{index}].raw_learning_rates_by_parameter key",
                    "a non-empty parameter name",
                    name,
                )
            _positive_number(
                rate,
                f"config.entries[{index}].raw_learning_rates_by_parameter[{name!r}]",
            )
        source = _mapping(
            entry.get("source_candidate"),
            f"config.entries[{index}].source_candidate",
        )
        source_dir = _repo_path(
            source.get("directory"),
            f"config.entries[{index}].source_candidate.directory",
        )
        _require_equal(
            source_dir.name,
            f"{row_id}--{optimizer}--{source.get('cell_id')}",
            f"config.entries[{index}].source_candidate.directory basename",
        )
        hashes = _mapping(
            source.get("artifact_sha256"),
            f"config.entries[{index}].source_candidate.artifact_sha256",
        )
        required_hash_names = {
            "result.json",
            "run_spec.json",
            "validation.json",
            "complete.json",
        }
        if set(hashes) != required_hash_names:
            raise _error(
                f"config.entries[{index}].source_candidate.artifact_sha256",
                f"exactly keys {sorted(required_hash_names)!r}",
                sorted(hashes),
            )
        for name in sorted(hashes):
            _sha256(
                hashes[name],
                f"config.entries[{index}].source_candidate.artifact_sha256[{name!r}]",
            )

    execution = _mapping(config.get("execution"), "config.execution")
    for key, expected in (
        ("runs_per_gpu", RUNS_PER_GPU),
        ("concurrent_within_pack", True),
        (
            "fixed_entry_packs",
            [list(pack) for pack in FIXED_ENTRY_PACKS],
        ),
        ("incomplete_entry_policy", "restart_from_shared_initialization"),
        ("completion_marker_written_last", True),
    ):
        _require_equal(
            execution.get(key), expected, f"config.execution.{key}"
        )
    _require_equal(
        execution.get("implementation_state"),
        status["implementation"],
        "config.execution.implementation_state",
    )
    _require_equal(
        execution.get("planned_manifest_job_count"),
        12,
        "config.execution.planned_manifest_job_count",
    )
    _require_equal(
        execution.get("planned_slurm_pack_count"),
        PRODUCTION_PACK_COUNT,
        "config.execution.planned_slurm_pack_count",
    )
    if "one_process_per_gpu" in execution:
        raise _error(
            "config.execution",
            "no stale one-process-per-GPU contract",
            {"one_process_per_gpu": execution["one_process_per_gpu"]},
        )
    _require_equal(
        execution.get("resume_policy"),
        "skip_only_hash_verified_complete_entries",
        "config.execution.resume_policy",
    )
    jean_zay = _mapping(
        execution.get("jean_zay"), "config.execution.jean_zay"
    )
    exact_jean_zay = {
        "dossier": "AD011016471R1",
        "project": "umg",
        "account": "umg@v100",
        "partition": "gpu_p13",
        "qos": "qos_gpu-t3",
        "constraint": "v100-32g",
        "gpus_per_task": 1,
        "cpus_per_task": 16,
        "memory_mb": 64_000,
        "hint": "nomultithread",
        "production_time_limit": "08:00:00",
        "canary_time_limit": "02:00:00",
        "module": "pytorch-gpu/py3/2.5.0",
        "python_executable": (
            "/lustre/fshomisc/sup/hpe/pub/miniforge/24.9.0/envs/"
            "pytorch-gpu-2.5.0+py3.12.7/bin/python"
        ),
        "production_array": "0-5%6",
        "production_task_count": 6,
        "entries_per_task": 2,
        "concurrent_training_processes_per_gpu": 2,
        "live_canary_array": "0-0",
        "slurm_parent_submission_count": 2,
    }
    for key, expected in exact_jean_zay.items():
        _require_equal(
            jean_zay.get(key),
            expected,
            f"config.execution.jean_zay.{key}",
        )
    if not isinstance(jean_zay.get("walltime_basis"), str) or not jean_zay[
        "walltime_basis"
    ].strip():
        raise _error(
            "config.execution.jean_zay.walltime_basis",
            "a non-empty measurement rationale",
            jean_zay.get("walltime_basis"),
        )
    preflight = _mapping(
        execution.get("preflight"), "config.execution.preflight"
    )
    for key in (
        "same_launcher_config_environment_device_and_output_path_required",
        "scheduled_runner_static_preflight_required",
        "sbatch_test_only_required",
        "live_singleton_canary_required",
        "fresh_tk_reference_gate_required",
        "semantic_completion_marker_and_hash_required",
        "full_array_blocked_until_all_gates_pass",
    ):
        _require_equal(preflight.get(key), True, f"config.execution.preflight.{key}")
    _require_equal(
        preflight.get("fresh_tk_reference_gate_scope"),
        "six_unique_architecture_x_scheme_rows",
        "config.execution.preflight.fresh_tk_reference_gate_scope",
    )
    if "worst_case_real_optimizer_step_required" in preflight:
        raise _error(
            "config.execution.preflight",
            "no stale one-step canary requirement",
            preflight["worst_case_real_optimizer_step_required"],
        )
    _require_equal(
        preflight.get("full_epoch_training_canary"),
        {
            "pack_index": CANARY_PACK_INDEX,
            "entry_indices": list(CANARY_ENTRY_INDICES),
            "architecture": "conv2",
            "schemes": ["ours", "ours"],
            "optimizers": ["sgd", "adam"],
            "mode": "successor_canary_pack",
            "epochs": 1,
            "expected_steps_per_entry": 3_438,
            "concurrent": True,
            "same_training_entrypoint_as_production": True,
        },
        "config.execution.preflight.full_epoch_training_canary",
    )

    artifacts = _mapping(config.get("artifacts"), "config.artifacts")
    _require_equal(
        artifacts.get("official_test_read"),
        False,
        "config.artifacts.official_test_read",
    )
    prohibitions = _mapping(config.get("prohibitions"), "config.prohibitions")
    for key in (
        "launch_before_explicit_plan_approval",
        "change_T_or_K",
        "change_selected_rho",
        "recompute_or_substitute_learning_rate_vector",
        "continue_from_candidate_checkpoint",
        "official_test_read",
        "paper_facing_evidence",
    ):
        _require_equal(prohibitions.get(key), True, f"config.prohibitions.{key}")

    approvals = _approval_state(config)
    if require_production_approval and not all(approvals.values()):
        raise _error(
            "config production approval",
            "an approved plan, launch_authorized=true, and an approved "
            "selection override with actor and timestamp, plus "
            "reviewed_ready_for_launch implementation state",
            approvals,
        )

    _require_official_test_false_recursive(config, "config")
    if verify_authority_files:
        if not parent_config_path.is_file():
            raise _error(
                "config.authorities.parent_study.config_path",
                "an existing regular file",
                str(parent_config_path),
            )
        _require_equal(
            sha256_file(parent_config_path),
            PARENT_CONFIG_FILE_SHA256,
            "parent config file SHA-256",
        )
        parent_data = read_json(parent_config_path)
        _require_equal(
            sha256_json(parent_data),
            PARENT_CONFIG_SHA256,
            "parent config canonical SHA-256",
        )
        PerfectDiodeHparamStudySpec.from_dict(parent_data)
        for architecture, shard in shard_paths.items():
            if not shard.is_dir():
                raise _error(
                    f"config.authorities.parent_study.shards.{architecture}",
                    "an existing directory",
                    str(shard),
                )
        for authority, fields in (
            (continuation, ("config_path", "manifest_path", "summary_path", "card_path", "review_path")),
        ):
            for field in fields:
                path = _repo_path(
                    authority[field],
                    f"config.authorities.conv2_high_rho_continuation.{field}",
                )
                if not path.is_file():
                    raise _error(
                        f"config.authorities.conv2_high_rho_continuation.{field}",
                        "an existing regular file",
                        str(path),
                    )
                digest_key = field.removesuffix("_path") + "_sha256"
                _require_equal(
                    sha256_file(path),
                    authority[digest_key],
                    f"config.authorities.conv2_high_rho_continuation.{digest_key}",
                )
        for index, entry in enumerate(entries):
            source = entry["source_candidate"]
            source_dir = _repo_path(
                source["directory"],
                f"config.entries[{index}].source_candidate.directory",
            )
            for name, expected in source["artifact_sha256"].items():
                path = source_dir / name
                if not path.is_file():
                    raise _error(
                        f"config.entries[{index}].source_candidate.{name}",
                        "an existing regular file",
                        str(path),
                    )
                _require_equal(
                    sha256_file(path),
                    expected,
                    f"config.entries[{index}].source_candidate.artifact_sha256[{name!r}]",
                )
        plan = _mapping(
            authorities.get("experiment_plan"),
            "config.authorities.experiment_plan",
        )
        plan_path = _repo_path(
            plan.get("path"), "config.authorities.experiment_plan.path"
        )
        if _approval_state(config)["approved_plan"] and not plan_path.is_file():
            raise _error(
                "config.authorities.experiment_plan.path",
                "an existing plan whose SHA-256 will be bound externally by "
                "the generated bundle",
                str(plan_path),
            )
    return config


def load_config(
    path: str | Path,
    *,
    verify_authority_files: bool = False,
    require_production_approval: bool = False,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise _error("config path", "an existing regular JSON file", str(source))
    return validate_config(
        read_json(source),
        verify_authority_files=verify_authority_files,
        require_production_approval=require_production_approval,
    )


def plan_config(path: str | Path) -> dict[str, Any]:
    """Return a side-effect-free plan summary."""

    source = Path(path).expanduser().resolve()
    config = load_config(source, verify_authority_files=True)
    approvals = _approval_state(config)
    return {
        "schema_version": "mnist-conv-perfectdiode-successor-plan/v1",
        "status": "ready" if all(approvals.values()) else "draft",
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_json(config),
        "config_file_sha256": sha256_file(source),
        "official_test_read": False,
        "expected_entry_count": 12,
        "approvals": approvals,
        "production_ready": all(approvals.values()),
        "entries": [
            {
                "entry_index": index,
                "entry_id": entry["entry_id"],
                "architecture": entry["architecture"],
                "scheme": entry["scheme"],
                "optimizer": entry["optimizer"],
                "rho_conv": entry["rho_conv"],
                "rho_dense": entry["rho_dense"],
                "epochs": entry["epochs"],
                "expected_steps": entry["expected_steps"],
                "boundary_status": entry["boundary_status"],
                "source_candidate_directory": entry["source_candidate"]["directory"],
            }
            for index, entry in enumerate(config["entries"])
        ],
    }


def _entry_from_manifest(
    manifest: Mapping[str, Any], entry_id: str, *, path: str
) -> dict[str, Any]:
    entries = [
        entry
        for entry in manifest.get("entries", [])
        if entry.get("entry_id") == entry_id
    ]
    if len(entries) != 1:
        raise _error(path, "exactly one declared stage entry", entry_id)
    return copy.deepcopy(dict(entries[0]))


def _verified_stage_entry(
    root: Path,
    stage: str,
    entry_id: str,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = root / "stages" / stage / "manifest.json"
    manifest = load_stage_manifest(
        manifest_path,
        study_dir=root,
        expected_study_id=PARENT_STUDY_ID,
    )
    entry = _entry_from_manifest(
        manifest,
        entry_id,
        path=f"{manifest_path}.entries",
    )
    completion = validate_entry_completion(
        study_dir=root,
        manifest_path=manifest_path,
        entry_id=entry_id,
    )
    return manifest_path, manifest, entry, completion


def _source_root_and_stage(source_dir: Path) -> tuple[Path, str]:
    if (
        source_dir.parent.name != "entries"
        or source_dir.parent.parent.parent.name != "stages"
    ):
        raise _error(
            "source_candidate.directory",
            "a path matching <study-root>/stages/<stage>/entries/<entry-id>",
            str(source_dir),
        )
    return source_dir.parents[3], source_dir.parent.parent.name


def _relative_to_repo(path: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise _error(label, "a path inside the repository", str(path)) from exc


def _add_copy(
    copies: dict[str, Path],
    destination: str,
    source: Path,
) -> None:
    relative = _portable_relative(destination, "bundle artifact destination")
    if not source.is_file() or source.is_symlink():
        raise _error(
            f"bundle source for {relative}",
            "an existing non-symlink regular file",
            str(source),
        )
    previous = copies.get(relative)
    if previous is not None and previous.resolve() != source.resolve():
        raise _error(
            f"bundle artifact destination {relative!r}",
            "one unambiguous source",
            (str(previous), str(source)),
        )
    copies[relative] = source


def _validate_asset_source(
    *,
    architecture: str,
    shard: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    entry_id = f"{architecture}-assets"
    manifest_path, _manifest, entry, _completion = _verified_stage_entry(
        shard, "assets", entry_id
    )
    entry_dir = shard / "stages" / "assets" / "entries" / entry_id
    required = (
        "initialization.json",
        "initialization.pt",
        "result.json",
        "split_indices.json",
        "split_provenance.json",
    )
    for name in required:
        if not (entry_dir / name).is_file():
            raise _error(
                f"parent {architecture} assets",
                f"the required file {name!r}",
                str(entry_dir),
            )
    result = _mapping(read_json(entry_dir / "result.json"), "asset result")
    initialization = _mapping(
        read_json(entry_dir / "initialization.json"), "asset initialization"
    )
    split_indices = _mapping(
        read_json(entry_dir / "split_indices.json"), "asset split indices"
    )
    require_official_test_excluded(result, require_result_marker=True)
    require_official_test_excluded(initialization, require_result_marker=True)
    exact_result = {
        "status": "complete",
        "study_id": PARENT_STUDY_ID,
        "config_sha256": PARENT_CONFIG_SHA256,
        "architecture": architecture,
    }
    for key, expected in exact_result.items():
        _require_equal(result.get(key), expected, f"asset result.{key}")
    arch = config["model"]["architectures"][architecture]
    for key, expected in (
        ("model_seed", 0),
        ("checkpoint_sha256", arch["initialization_checkpoint_sha256"]),
        ("parameter_tensor_sha256", arch["initialization_tensor_sha256"]),
        ("shared_across_schemes_and_optimizers", True),
    ):
        _require_equal(
            initialization.get(key), expected, f"asset initialization.{key}"
        )
    checkpoint = entry_dir / "initialization.pt"
    _require_equal(
        sha256_file(checkpoint),
        initialization["checkpoint_sha256"],
        f"{architecture} initialization checkpoint SHA-256",
    )
    for key, expected in (
        ("train_indices_sha256", config["dataset"]["train_indices_sha256"]),
        (
            "validation_indices_sha256",
            config["dataset"]["validation_indices_sha256"],
        ),
    ):
        _require_equal(split_indices.get(key), expected, f"asset split_indices.{key}")
        _require_equal(result.get(key), expected, f"asset result.{key}")
    declared_outputs = {Path(path).name for path in entry["outputs"]}
    if not set(required).issubset(declared_outputs):
        raise _error(
            f"{architecture} asset manifest outputs",
            f"all required names {sorted(required)!r}",
            sorted(declared_outputs),
        )
    return {
        "entry_id": entry_id,
        "entry_dir": entry_dir,
        "manifest_path": manifest_path,
        "completion_path": shard / entry["completion_path"],
        "result": result,
        "initialization": initialization,
        "split_indices": split_indices,
    }


def _validate_security_source(
    *,
    architecture: str,
    scheme: str,
    row_id: str,
    shard: Path,
    asset: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path, _manifest, entry, _completion = _verified_stage_entry(
        shard, "fixed_tk_gradient_security", row_id
    )
    entry_dir = (
        shard / "stages" / "fixed_tk_gradient_security" / "entries" / row_id
    )
    result = _mapping(read_json(entry_dir / "result.json"), "security result")
    require_official_test_excluded(result, require_result_marker=True)
    expected_arch = EXPECTED_ARCHITECTURES[architecture]
    exact = {
        "status": "complete",
        "security_passed": True,
        "study_id": PARENT_STUDY_ID,
        "config_sha256": PARENT_CONFIG_SHA256,
        "row_id": row_id,
        "architecture": architecture,
        "scheme": scheme,
        "initialization_checkpoint_sha256": asset["initialization"][
            "checkpoint_sha256"
        ],
        "initialization_tensor_sha256": asset["initialization"][
            "parameter_tensor_sha256"
        ],
    }
    for key, expected in exact.items():
        _require_equal(result.get(key), expected, f"security result.{key}")
    _require_equal(
        result.get("operational"),
        {"T": expected_arch["T"], "K": expected_arch["K"]},
        "security result.operational",
    )
    _require_equal(
        result.get("reference"),
        {"T": 64, "K": 64},
        "security result.reference",
    )
    return {
        "entry_id": row_id,
        "entry_dir": entry_dir,
        "manifest_path": manifest_path,
        "completion_path": shard / entry["completion_path"],
        "result": result,
    }


def _validate_probe_source(
    *,
    architecture: str,
    scheme: str,
    optimizer: str,
    row_id: str,
    shard: Path,
    asset: Mapping[str, Any],
) -> dict[str, Any]:
    entry_id = f"{row_id}--{optimizer}"
    manifest_path, _manifest, entry, _completion = _verified_stage_entry(
        shard, "optimizer_probe", entry_id
    )
    entry_dir = shard / "stages" / "optimizer_probe" / "entries" / entry_id
    result = _mapping(read_json(entry_dir / "result.json"), "optimizer probe")
    require_official_test_excluded(result, require_result_marker=True)
    exact = {
        "status": "complete",
        "probe_stable": True,
        "study_id": PARENT_STUDY_ID,
        "config_sha256": PARENT_CONFIG_SHA256,
        "entry_id": entry_id,
        "row_id": row_id,
        "architecture": architecture,
        "scheme": scheme,
        "optimizer": optimizer,
        "initialization_checkpoint_sha256": asset["initialization"][
            "checkpoint_sha256"
        ],
        "initialization_tensor_sha256": asset["initialization"][
            "parameter_tensor_sha256"
        ],
    }
    for key, expected in exact.items():
        _require_equal(result.get(key), expected, f"optimizer probe.{key}")
    return {
        "entry_id": entry_id,
        "entry_dir": entry_dir,
        "manifest_path": manifest_path,
        "completion_path": shard / entry["completion_path"],
        "result": result,
    }


def _validate_candidate_source(
    *,
    config_entry: Mapping[str, Any],
    probe: Mapping[str, Any],
    asset: Mapping[str, Any],
) -> dict[str, Any]:
    source = config_entry["source_candidate"]
    source_dir = _repo_path(
        source["directory"],
        f"source candidate for {config_entry['entry_id']}",
    )
    root, stage = _source_root_and_stage(source_dir)
    if stage not in {"rho_core_candidates", "rho_extension_candidates"}:
        raise _error(
            f"source candidate stage for {config_entry['entry_id']}",
            "'rho_core_candidates' or 'rho_extension_candidates'",
            stage,
        )
    manifest_path, _manifest, stage_entry, _completion = _verified_stage_entry(
        root, stage, source_dir.name
    )
    if source_dir != root / "stages" / stage / "entries" / stage_entry["entry_id"]:
        raise _error(
            f"source candidate directory for {config_entry['entry_id']}",
            "the manifest-declared entry directory",
            str(source_dir),
        )
    for name, expected in source["artifact_sha256"].items():
        _require_equal(
            sha256_file(source_dir / name),
            expected,
            f"{config_entry['entry_id']} source {name} SHA-256",
        )
    result = _mapping(read_json(source_dir / "result.json"), "candidate result")
    run_spec = _mapping(read_json(source_dir / "run_spec.json"), "candidate run spec")
    validation = _mapping(
        read_json(source_dir / "validation.json"), "candidate validation"
    )
    for name, value in (
        ("candidate result", result),
        ("candidate run spec", run_spec),
        ("candidate validation", validation),
    ):
        require_official_test_excluded(value, require_result_marker=True)
        _require_official_test_false_recursive(value, name)
    exact = {
        "status": "complete",
        "training_completed": True,
        "safety_admissible": True,
        "passing_candidate": True,
        "completed_steps": CANDIDATE_TOTAL_STEPS,
        "expected_steps": CANDIDATE_TOTAL_STEPS,
        "epochs_completed": CANDIDATE_EPOCHS,
        "expected_epochs": CANDIDATE_EPOCHS,
        "study_id": PARENT_STUDY_ID,
        "config_sha256": PARENT_CONFIG_SHA256,
        "cell_id": source["cell_id"],
        "surface_id": f"{config_entry['row_id']}--{config_entry['optimizer']}",
        "row_id": config_entry["row_id"],
        "scheme": config_entry["scheme"],
        "optimizer": config_entry["optimizer"],
        "probe_entry_id": probe["entry_id"],
        "initialization_checkpoint_sha256": asset["initialization"][
            "checkpoint_sha256"
        ],
        "initialization_tensor_sha256": asset["initialization"][
            "parameter_tensor_sha256"
        ],
        "train_indices_sha256": asset["split_indices"]["train_indices_sha256"],
        "validation_indices_sha256": asset["split_indices"][
            "validation_indices_sha256"
        ],
    }
    for key, expected in exact.items():
        _require_equal(result.get(key), expected, f"candidate result.{key}")
    for key in ("rho_conv", "rho_dense"):
        provided = _positive_number(result.get(key), f"candidate result.{key}")
        selected = _positive_number(config_entry[key], f"config entry.{key}")
        if not math.isclose(provided, selected, rel_tol=1.0e-15, abs_tol=0.0):
            raise _error(
                f"candidate result.{key}",
                f"the selected target {selected!r} up to historical JSON "
                "floating-point serialization",
                provided,
            )
    derived_rates = derive_raw_learning_rates(
        probe["result"],
        rho_conv=float(result["rho_conv"]),
        rho_dense=float(result["rho_dense"]),
    )
    configured_rates = {
        str(name): float(value)
        for name, value in config_entry[
            "raw_learning_rates_by_parameter"
        ].items()
    }
    _require_equal(
        configured_rates,
        derived_rates,
        f"{config_entry['entry_id']} exact probe-derived raw LR vector",
    )
    _require_equal(
        result.get("raw_learning_rates_by_parameter"),
        derived_rates,
        "candidate result.raw_learning_rates_by_parameter",
    )
    training = _mapping(run_spec.get("training"), "candidate run_spec.training")
    for key, expected in (
        ("mode", "candidate"),
        ("epochs", CANDIDATE_EPOCHS),
        ("steps_per_epoch", 3_438),
        ("total_steps", CANDIDATE_TOTAL_STEPS),
        ("constant_lr_vector", True),
        ("restart_from_shared_initialization", True),
        ("continue_from_canary_or_candidate", False),
    ):
        _require_equal(training.get(key), expected, f"candidate run_spec.training.{key}")
    _require_equal(
        run_spec.get("raw_learning_rates_by_parameter"),
        derived_rates,
        "candidate run_spec.raw_learning_rates_by_parameter",
    )
    records = _list(
        validation.get("records"), "candidate validation.records", length=3
    )
    final = source["three_epoch_final_validation"]
    for key, expected in (
        ("accuracy", final["accuracy"]),
        ("loss", final["loss"]),
    ):
        _require_equal(records[-1].get(key), expected, f"candidate final validation.{key}")
    _require_equal(
        result.get("median_projection_efficiency"),
        final["median_projection_efficiency"],
        "candidate result.median_projection_efficiency",
    )
    output_names = {Path(path).name for path in stage_entry["outputs"]}
    if not {"result.json", "run_spec.json", "validation.json"}.issubset(
        output_names
    ):
        raise _error(
            f"{config_entry['entry_id']} source stage outputs",
            "result.json, run_spec.json, and validation.json",
            sorted(output_names),
        )
    return {
        "root": root,
        "stage": stage,
        "entry_id": stage_entry["entry_id"],
        "entry_dir": source_dir,
        "manifest_path": manifest_path,
        "completion_path": root / stage_entry["completion_path"],
        "result": result,
        "run_spec": run_spec,
        "validation": validation,
        "derived_rates": derived_rates,
        "runtime_rho_conv": float(result["rho_conv"]),
        "runtime_rho_dense": float(result["rho_dense"]),
    }


def _resolve_bundle_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    authorities = config["authorities"]
    parent = authorities["parent_study"]
    shards = {
        architecture: _repo_path(
            parent["shards"][architecture],
            f"config.authorities.parent_study.shards.{architecture}",
        )
        for architecture in ARCHITECTURE_ORDER
    }
    parent_studies = []
    for architecture, shard in shards.items():
        path = shard / "study.resolved.json"
        if not path.is_file():
            raise _error(
                f"{architecture} parent resolved study",
                "an existing study.resolved.json",
                str(path),
            )
        data = read_json(path)
        spec = PerfectDiodeHparamStudySpec.from_dict(data)
        _require_equal(spec.study_id, PARENT_STUDY_ID, f"{architecture} parent study id")
        _require_equal(
            spec.config_sha256,
            PARENT_CONFIG_SHA256,
            f"{architecture} parent config SHA-256",
        )
        parent_studies.append((path, data))
    _require_equal(
        parent_studies[0][1],
        parent_studies[1][1],
        "Conv1 and Conv2 parent resolved configs",
    )

    assets = {
        architecture: _validate_asset_source(
            architecture=architecture,
            shard=shards[architecture],
            config=config,
        )
        for architecture in ARCHITECTURE_ORDER
    }
    securities: dict[str, dict[str, Any]] = {}
    probes: dict[str, dict[str, Any]] = {}
    for architecture in ARCHITECTURE_ORDER:
        for scheme in SCHEME_ORDER:
            row_id = EXPECTED_ROWS[(architecture, scheme)][0]
            securities[row_id] = _validate_security_source(
                architecture=architecture,
                scheme=scheme,
                row_id=row_id,
                shard=shards[architecture],
                asset=assets[architecture],
            )
            for optimizer in OPTIMIZER_ORDER:
                surface = f"{row_id}--{optimizer}"
                probes[surface] = _validate_probe_source(
                    architecture=architecture,
                    scheme=scheme,
                    optimizer=optimizer,
                    row_id=row_id,
                    shard=shards[architecture],
                    asset=assets[architecture],
                )
    candidates = []
    for entry in config["entries"]:
        surface = f"{entry['row_id']}--{entry['optimizer']}"
        candidates.append(
            _validate_candidate_source(
                config_entry=entry,
                probe=probes[surface],
                asset=assets[entry["architecture"]],
            )
        )
    return {
        "parent_study_path": parent_studies[0][0],
        "parent_study": parent_studies[0][1],
        "shards": shards,
        "assets": assets,
        "securities": securities,
        "probes": probes,
        "candidates": candidates,
    }


def _artifact_record(root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise _error("bundle artifact", "a non-symlink regular file", str(path))
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise _error("bundle artifact", "a file inside the bundle", str(path)) from exc
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _directory_tree_sha256(root: Path) -> str:
    """Match the immutable bundle-tree digest used by the submitter/wrapper."""

    source = root.resolve()
    if not source.is_dir() or source.is_symlink():
        raise _error("directory tree root", "a non-symlink directory", str(root))
    digest = hashlib.sha256()
    files: list[Path] = []
    for candidate in source.rglob("*"):
        if candidate.is_symlink():
            raise _error("directory tree", "no symlinks", str(candidate))
        if candidate.is_file():
            files.append(candidate)
        elif not candidate.is_dir():
            raise _error(
                "directory tree",
                "only regular files and directories",
                str(candidate),
            )
    for candidate in sorted(
        files, key=lambda item: item.relative_to(source).as_posix()
    ):
        relative = candidate.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(candidate.stat().st_size.to_bytes(8, "big"))
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    try:
        return _commit(value, "source checkout HEAD")
    except PerfectDiodeSuccessorError:
        return None


@contextmanager
def _validated_source_archive_checkout(archive: Path):
    """Safely extract a source tarball and yield its unique repository root."""

    if not archive.is_file() or archive.is_symlink():
        raise _error(
            "source archive",
            "an existing non-symlink tar archive",
            str(archive),
        )
    temporary = Path(tempfile.mkdtemp(prefix="pd-successor-source-"))
    extracted = temporary / "checkout"
    extracted.mkdir()
    try:
        try:
            handle = tarfile.open(archive, mode="r:*")
        except (tarfile.TarError, OSError) as exc:
            raise _error(
                "source archive",
                "a readable tar or compressed-tar archive",
                str(archive),
            ) from exc
        with handle:
            members = handle.getmembers()
            if not members:
                raise _error("source archive", "a non-empty tar archive", str(archive))
            seen: set[str] = set()
            total_bytes = 0
            normalized: list[tuple[tarfile.TarInfo, Path]] = []
            for index, member in enumerate(members):
                name = member.name
                relative = Path(name)
                if (
                    not name
                    or "\\" in name
                    or relative.is_absolute()
                    or ".." in relative.parts
                ):
                    raise _error(
                        f"source archive member[{index}]",
                        "a safe relative POSIX path",
                        name,
                    )
                portable = relative.as_posix()
                if portable in {"", "."}:
                    if member.isdir():
                        continue
                    raise _error(
                        f"source archive member[{index}]",
                        "a named file or directory",
                        name,
                    )
                if portable in seen:
                    raise _error(
                        "source archive members",
                        "unique normalized paths",
                        portable,
                    )
                seen.add(portable)
                if member.issym() or member.islnk() or member.isdev():
                    raise _error(
                        f"source archive member[{index}]",
                        "a regular file or directory (no links/devices)",
                        name,
                    )
                if not (member.isfile() or member.isdir()):
                    raise _error(
                        f"source archive member[{index}]",
                        "a regular file or directory",
                        name,
                    )
                if member.isfile():
                    total_bytes += int(member.size)
                    if total_bytes > 2 * 1024 * 1024 * 1024:
                        raise _error(
                            "source archive expanded size",
                            "at most 2 GiB",
                            total_bytes,
                        )
                target = extracted / relative
                try:
                    target.resolve().relative_to(extracted.resolve())
                except ValueError as exc:
                    raise _error(
                        f"source archive member[{index}]",
                        "a path inside the extraction root",
                        name,
                    ) from exc
                normalized.append((member, target))
            for member, target in normalized:
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise _error(
                        "source archive member",
                        "readable regular-file content",
                        member.name,
                    )
                with source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
        identity_files = list(
            extracted.rglob("experiments/mnist_conv/identity.py")
        )
        roots = {
            identity_file.parents[2].resolve()
            for identity_file in identity_files
            if identity_file.is_file() and not identity_file.is_symlink()
        }
        if len(roots) != 1:
            raise _error(
                "source archive repository root",
                "exactly one checkout containing "
                "experiments/mnist_conv/identity.py",
                sorted(str(path) for path in roots),
            )
        yield next(iter(roots))
    finally:
        shutil.rmtree(temporary)


def _source_archive_fingerprint(archive: Path) -> str:
    with _validated_source_archive_checkout(archive) as checkout:
        return code_fingerprint(checkout)


def _execution_source_binding(
    *,
    source_archive: str | Path,
    source_commit: str,
    effective_code_fingerprint: str,
    environment_contract: str | Path,
) -> dict[str, Any]:
    archive = Path(source_archive).expanduser().resolve()
    environment = Path(environment_contract).expanduser().resolve()
    for path, label in (
        (archive, "source archive"),
        (environment, "environment contract"),
        (WORKER_LAUNCHER, "successor worker launcher"),
        (RUNTIME_MODULE, "successor runtime module"),
    ):
        if not path.is_file() or path.is_symlink():
            raise _error(label, "an existing non-symlink regular file", str(path))
    expected_commit = _commit(source_commit, "source_commit")
    observed_commit = _git_head(REPO_ROOT)
    if observed_commit is None or observed_commit != expected_commit:
        raise _error(
            "source_commit",
            f"the current repository HEAD {observed_commit!r}",
            expected_commit,
        )
    expected_fingerprint = _sha256(
        effective_code_fingerprint, "effective_code_fingerprint"
    )
    observed_fingerprint = _source_archive_fingerprint(archive)
    if observed_fingerprint != expected_fingerprint:
        raise _error(
            "effective_code_fingerprint",
            "the fingerprint recomputed from the safely extracted source archive",
            {
                "declared": expected_fingerprint,
                "observed": observed_fingerprint,
            },
        )
    return {
        "source_commit": expected_commit,
        "source_archive_sha256": sha256_file(archive),
        "effective_code_fingerprint": observed_fingerprint,
        "environment_contract_sha256": sha256_file(environment),
        "worker_launcher_sha256": sha256_file(WORKER_LAUNCHER),
        "runtime_module_sha256": sha256_file(RUNTIME_MODULE),
    }


def _copy_config_verbatim(
    source: Path, destination: Path, expected: Mapping[str, Any]
) -> None:
    if not source.is_file() or source.is_symlink():
        raise _error(
            "config source",
            "an existing non-symlink regular file",
            str(source),
        )
    shutil.copyfile(source, destination)
    _require_equal(
        read_json(destination),
        dict(expected),
        "verbatim bundled config semantics",
    )
    _require_equal(
        sha256_file(destination),
        sha256_file(source),
        "verbatim bundled config file SHA-256",
    )


def _copy_bundle_inputs(
    *,
    staging: Path,
    config_source: Path,
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
) -> dict[str, Any]:
    copies: dict[str, Path] = {}
    _add_copy(copies, "study.resolved.json", resolved["parent_study_path"])
    source_manifests: dict[Path, str] = {}

    def add_manifest(source: Path, label: str) -> str:
        if source not in source_manifests:
            destination = f"evidence/source_manifests/{label}.json"
            _add_copy(copies, destination, source)
            source_manifests[source] = destination
        return source_manifests[source]

    for architecture, asset in resolved["assets"].items():
        destination = f"stages/assets/entries/{architecture}-assets"
        for name in (
            "initialization.json",
            "initialization.pt",
            "result.json",
            "split_indices.json",
            "split_provenance.json",
        ):
            _add_copy(copies, f"{destination}/{name}", asset["entry_dir"] / name)
        _add_copy(
            copies,
            f"evidence/source_completions/{architecture}-assets.json",
            asset["completion_path"],
        )
        add_manifest(asset["manifest_path"], f"{architecture}-assets")

    for row_id, security in resolved["securities"].items():
        _add_copy(
            copies,
            f"evidence/security/{row_id}/result.json",
            security["entry_dir"] / "result.json",
        )
        _add_copy(
            copies,
            f"evidence/security/{row_id}/source_completion.json",
            security["completion_path"],
        )
        architecture = security["result"]["architecture"]
        add_manifest(
            security["manifest_path"], f"{architecture}-fixed-tk-security"
        )

    for surface, probe in resolved["probes"].items():
        _add_copy(
            copies,
            f"stages/optimizer_probe/entries/{surface}/result.json",
            probe["entry_dir"] / "result.json",
        )
        _add_copy(
            copies,
            f"evidence/probes/{surface}/source_completion.json",
            probe["completion_path"],
        )
        architecture = probe["result"]["architecture"]
        add_manifest(probe["manifest_path"], f"{architecture}-optimizer-probe")

    entry_evidence = []
    for index, (entry, candidate) in enumerate(
        zip(config["entries"], resolved["candidates"])
    ):
        base = f"evidence/candidates/{entry['entry_id']}"
        for name in ("result.json", "run_spec.json", "validation.json"):
            _add_copy(copies, f"{base}/{name}", candidate["entry_dir"] / name)
        _add_copy(
            copies,
            f"{base}/source_completion.json",
            candidate["completion_path"],
        )
        source_kind = (
            "high_rho_extension"
            if candidate["stage"] == "rho_extension_candidates"
            else "parent_core"
        )
        source_manifest = add_manifest(
            candidate["manifest_path"],
            (
                f"{source_kind}-{entry['architecture']}-"
                f"{entry['row_id']}-{entry['optimizer']}"
            ),
        )
        entry_evidence.append(
            {
                "entry_index": index,
                "entry_id": entry["entry_id"],
                "source_kind": source_kind,
                "original_directory": entry["source_candidate"]["directory"],
                "original_entry_id": candidate["entry_id"],
                "runtime_rho_conv": candidate["runtime_rho_conv"],
                "runtime_rho_dense": candidate["runtime_rho_dense"],
                "source_stage_manifest": source_manifest,
                "candidate_result": f"{base}/result.json",
                "candidate_run_spec": f"{base}/run_spec.json",
                "candidate_validation": f"{base}/validation.json",
                "candidate_completion": f"{base}/source_completion.json",
            }
        )

    _copy_config_verbatim(
        config_source,
        staging / "config.json",
        config,
    )
    for destination, source in sorted(copies.items()):
        target = staging / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    artifacts = [
        _artifact_record(staging, path)
        for path in sorted(
            (
                item
                for item in staging.rglob("*")
                if item.is_file() and item.name != "manifest.json"
            ),
            key=lambda item: item.relative_to(staging).as_posix(),
        )
    ]
    return {
        "entry_evidence": entry_evidence,
        "input_artifacts": artifacts,
    }


def build_input_bundle(
    *,
    config_path: str | Path,
    bundle_dir: str | Path,
    source_archive: str | Path,
    source_commit: str,
    effective_code_fingerprint: str,
    environment_contract: str | Path,
) -> dict[str, Any]:
    """Build one new immutable, self-contained scientific input bundle."""

    config_source = Path(config_path).expanduser().resolve()
    config = load_config(config_source, verify_authority_files=True)
    resolved = _resolve_bundle_inputs(config)
    execution_source = _execution_source_binding(
        source_archive=source_archive,
        source_commit=source_commit,
        effective_code_fingerprint=effective_code_fingerprint,
        environment_contract=environment_contract,
    )
    destination = Path(bundle_dir).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise _error(
            "bundle_dir",
            "a new path that does not already exist",
            str(destination),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        copied = _copy_bundle_inputs(
            staging=temporary,
            config_source=config_source,
            config=config,
            resolved=resolved,
        )
        entries = []
        for index, (entry, evidence) in enumerate(
            zip(config["entries"], copied["entry_evidence"])
        ):
            entries.append(
                {
                    "entry_index": index,
                    "entry_id": entry["entry_id"],
                    "architecture": entry["architecture"],
                    "row_id": entry["row_id"],
                    "scheme": entry["scheme"],
                    "voltage_amp": entry["voltage_amp"],
                    "current_amp": entry["current_amp"],
                    "optimizer": entry["optimizer"],
                    "rho_conv": entry["rho_conv"],
                    "rho_dense": entry["rho_dense"],
                    "runtime_rho_conv": evidence["runtime_rho_conv"],
                    "runtime_rho_dense": evidence["runtime_rho_dense"],
                    "raw_learning_rates_by_parameter": copy.deepcopy(
                        entry["raw_learning_rates_by_parameter"]
                    ),
                    "epochs": entry["epochs"],
                    "steps_per_epoch": 3_438,
                    "expected_steps": entry["expected_steps"],
                    "model_seed": 0,
                    "fresh_restart_from_shared_initialization": True,
                    "continue_from_candidate": False,
                    "selection_basis": entry["selection_basis"],
                    "boundary_status": entry["boundary_status"],
                    "cell_id": entry["source_candidate"]["cell_id"],
                    "asset_entry_id": f"{entry['architecture']}-assets",
                    "probe_result": (
                        "stages/optimizer_probe/entries/"
                        f"{entry['row_id']}--{entry['optimizer']}/result.json"
                    ),
                    "security_result": (
                        f"evidence/security/{entry['row_id']}/result.json"
                    ),
                    "candidate_evidence": evidence,
                    "official_test_read": False,
                }
            )
        body = {
            "schema_version": BUNDLE_MANIFEST_SCHEMA_VERSION,
            "experiment_id": config["experiment_id"],
            "config_path": "config.json",
            "config_sha256": sha256_json(config),
            "config_file_sha256": sha256_file(temporary / "config.json"),
            "parent_study_id": PARENT_STUDY_ID,
            "parent_config_sha256": PARENT_CONFIG_SHA256,
            "approvals": _approval_state(config),
            "production_ready": False,
            "external_launch_authorization_required": True,
            "execution_source": execution_source,
            "official_test_read": False,
            "entry_count": 12,
            "input_artifacts": copied["input_artifacts"],
            "entries": entries,
        }
        manifest = {
            **body,
            "bundle_id": "pdconfirmbundle_" + sha256_json(body),
        }
        atomic_write_json(temporary / "manifest.json", manifest, canonical=True)
        validate_input_bundle(temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_input_bundle(destination)


def _validate_artifact_records(
    root: Path, records: Any, *, expected_files: set[str]
) -> list[dict[str, Any]]:
    values = _list(records, "manifest.input_artifacts")
    normalized = []
    observed_paths: set[str] = set()
    for index, raw in enumerate(values):
        record = _mapping(raw, f"manifest.input_artifacts[{index}]")
        if set(record) != {"path", "sha256", "bytes"}:
            raise _error(
                f"manifest.input_artifacts[{index}]",
                "exactly path, sha256, and bytes",
                sorted(record),
            )
        relative = _portable_relative(
            record.get("path"), f"manifest.input_artifacts[{index}].path"
        )
        digest = _sha256(
            record.get("sha256"), f"manifest.input_artifacts[{index}].sha256"
        )
        byte_count = record.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise _error(
                f"manifest.input_artifacts[{index}].bytes",
                "a non-negative integer",
                byte_count,
            )
        if relative in observed_paths:
            raise _error(
                "manifest.input_artifacts",
                "unique artifact paths",
                relative,
            )
        observed_paths.add(relative)
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != byte_count
            or sha256_file(path) != digest
        ):
            raise _error(
                f"manifest.input_artifacts[{index}]",
                "the current hash- and size-verified bundle artifact",
                str(path),
            )
        normalized.append(
            {"path": relative, "sha256": digest, "bytes": byte_count}
        )
    if observed_paths != expected_files:
        raise _error(
            "manifest.input_artifacts paths",
            "exactly every non-manifest file in the bundle",
            {
                "missing": sorted(expected_files - observed_paths),
                "extra": sorted(observed_paths - expected_files),
            },
        )
    if [item["path"] for item in normalized] != sorted(observed_paths):
        raise _error(
            "manifest.input_artifacts",
            "records sorted by unique path",
            [item["path"] for item in normalized],
        )
    return normalized


def validate_input_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Revalidate a relocated bundle without consulting parent result trees."""

    root = Path(bundle_dir).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise _error("bundle_dir", "an existing non-symlink directory", str(root))
    for item in root.rglob("*"):
        if item.is_symlink():
            raise _error("bundle", "no symlink components", str(item))
    manifest_path = root / "manifest.json"
    manifest = _mapping(read_json(manifest_path), "manifest")
    if manifest.get("schema_version") != BUNDLE_MANIFEST_SCHEMA_VERSION:
        raise _error(
            "manifest.schema_version",
            f"exactly {BUNDLE_MANIFEST_SCHEMA_VERSION!r}",
            manifest.get("schema_version"),
        )
    bundle_id = manifest.get("bundle_id")
    body = {key: value for key, value in manifest.items() if key != "bundle_id"}
    _require_equal(
        bundle_id,
        "pdconfirmbundle_" + sha256_json(body),
        "manifest.bundle_id",
    )
    _require_equal(manifest.get("entry_count"), 12, "manifest.entry_count")
    _require_equal(
        manifest.get("official_test_read"), False, "manifest.official_test_read"
    )
    config = validate_config(read_json(root / "config.json"))
    _require_equal(
        sha256_json(config), manifest.get("config_sha256"), "manifest.config_sha256"
    )
    _require_equal(
        sha256_file(root / "config.json"),
        manifest.get("config_file_sha256"),
        "manifest.config_file_sha256",
    )
    _require_equal(
        manifest.get("parent_study_id"),
        PARENT_STUDY_ID,
        "manifest.parent_study_id",
    )
    _require_equal(
        manifest.get("parent_config_sha256"),
        PARENT_CONFIG_SHA256,
        "manifest.parent_config_sha256",
    )
    expected_files = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item != manifest_path
    }
    artifacts = _validate_artifact_records(
        root, manifest.get("input_artifacts"), expected_files=expected_files
    )
    artifact_paths = {record["path"] for record in artifacts}
    entries = _list(manifest.get("entries"), "manifest.entries", length=12)
    for index, (entry, configured) in enumerate(zip(entries, config["entries"])):
        value = _mapping(entry, f"manifest.entries[{index}]")
        exact = {
            "entry_index": index,
            "entry_id": configured["entry_id"],
            "architecture": configured["architecture"],
            "row_id": configured["row_id"],
            "scheme": configured["scheme"],
            "optimizer": configured["optimizer"],
            "rho_conv": configured["rho_conv"],
            "rho_dense": configured["rho_dense"],
            "runtime_rho_conv": read_json(
                root
                / value["candidate_evidence"]["candidate_result"]
            )["rho_conv"],
            "runtime_rho_dense": read_json(
                root
                / value["candidate_evidence"]["candidate_result"]
            )["rho_dense"],
            "epochs": configured["epochs"],
            "steps_per_epoch": 3_438,
            "expected_steps": configured["expected_steps"],
            "model_seed": 0,
            "fresh_restart_from_shared_initialization": True,
            "continue_from_candidate": False,
            "selection_basis": configured["selection_basis"],
            "boundary_status": configured["boundary_status"],
            "cell_id": configured["source_candidate"]["cell_id"],
            "official_test_read": False,
        }
        for key, expected in exact.items():
            _require_equal(
                value.get(key), expected, f"manifest.entries[{index}].{key}"
            )
        probe_path = _portable_relative(
            value.get("probe_result"), f"manifest.entries[{index}].probe_result"
        )
        security_path = _portable_relative(
            value.get("security_result"), f"manifest.entries[{index}].security_result"
        )
        if probe_path not in artifact_paths or security_path not in artifact_paths:
            raise _error(
                f"manifest.entries[{index}] evidence",
                "hash-bound probe and security paths",
                (probe_path, security_path),
            )
        probe = _mapping(read_json(root / probe_path), "bundled optimizer probe")
        security = _mapping(read_json(root / security_path), "bundled security result")
        require_official_test_excluded(probe, require_result_marker=True)
        require_official_test_excluded(security, require_result_marker=True)
        _require_equal(probe.get("probe_stable"), True, "bundled probe.probe_stable")
        _require_equal(
            security.get("security_passed"),
            True,
            "bundled security.security_passed",
        )
        derived = derive_raw_learning_rates(
            probe,
            rho_conv=float(value["runtime_rho_conv"]),
            rho_dense=float(value["runtime_rho_dense"]),
        )
        _require_equal(
            value.get("raw_learning_rates_by_parameter"),
            derived,
            f"manifest.entries[{index}].raw_learning_rates_by_parameter",
        )
        _require_equal(
            configured["raw_learning_rates_by_parameter"],
            derived,
            f"config.entries[{index}].raw_learning_rates_by_parameter",
        )
        evidence = _mapping(
            value.get("candidate_evidence"),
            f"manifest.entries[{index}].candidate_evidence",
        )
        for key in (
            "source_stage_manifest",
            "candidate_result",
            "candidate_run_spec",
            "candidate_validation",
            "candidate_completion",
        ):
            relative = _portable_relative(
                evidence.get(key),
                f"manifest.entries[{index}].candidate_evidence.{key}",
            )
            if relative not in artifact_paths:
                raise _error(
                    f"manifest.entries[{index}].candidate_evidence.{key}",
                    "a hash-bound bundle artifact",
                    relative,
                )
        candidate = _mapping(
            read_json(root / evidence["candidate_result"]),
            "bundled candidate result",
        )
        require_official_test_excluded(candidate, require_result_marker=True)
        for key, expected in (
            ("status", "complete"),
            ("training_completed", True),
            ("passing_candidate", True),
            ("completed_steps", CANDIDATE_TOTAL_STEPS),
            ("epochs_completed", CANDIDATE_EPOCHS),
            ("raw_learning_rates_by_parameter", derived),
        ):
            _require_equal(candidate.get(key), expected, f"bundled candidate.{key}")
    _require_equal(
        manifest.get("production_ready"),
        False,
        "manifest.production_ready",
    )
    _require_equal(
        manifest.get("external_launch_authorization_required"),
        True,
        "manifest.external_launch_authorization_required",
    )
    _require_official_test_false_recursive(manifest, "manifest")
    return manifest


def _validate_external_execution_inputs(
    manifest: Mapping[str, Any],
    *,
    source_archive: str | Path,
    environment_contract: str | Path,
) -> dict[str, bool]:
    source = _mapping(
        manifest.get("execution_source"), "manifest.execution_source"
    )
    archive = Path(source_archive).expanduser().resolve()
    environment = Path(environment_contract).expanduser().resolve()
    archive_hash_valid = (
        archive.is_file()
        and not archive.is_symlink()
        and sha256_file(archive) == source.get("source_archive_sha256")
    )
    archive_fingerprint_valid = False
    if archive_hash_valid:
        try:
            archive_fingerprint_valid = (
                _source_archive_fingerprint(archive)
                == source.get("effective_code_fingerprint")
            )
        except (OSError, PerfectDiodeSuccessorError, tarfile.TarError):
            archive_fingerprint_valid = False
    try:
        checkout_fingerprint_valid = (
            code_fingerprint(REPO_ROOT)
            == source.get("effective_code_fingerprint")
        )
    except (OSError, ValueError):
        checkout_fingerprint_valid = False
    checks = {
        "source_archive_valid": archive_hash_valid,
        "source_archive_fingerprint_valid": archive_fingerprint_valid,
        "source_checkout_fingerprint_valid": checkout_fingerprint_valid,
        "environment_contract_valid": (
            environment.is_file()
            and not environment.is_symlink()
            and sha256_file(environment)
            == source.get("environment_contract_sha256")
        ),
        "worker_launcher_valid": (
            WORKER_LAUNCHER.is_file()
            and sha256_file(WORKER_LAUNCHER)
            == source.get("worker_launcher_sha256")
        ),
        "runtime_module_valid": (
            RUNTIME_MODULE.is_file()
            and sha256_file(RUNTIME_MODULE)
            == source.get("runtime_module_sha256")
        ),
    }
    for key in (
        "source_archive_sha256",
        "effective_code_fingerprint",
        "environment_contract_sha256",
        "worker_launcher_sha256",
        "runtime_module_sha256",
    ):
        _sha256(source.get(key), f"manifest.execution_source.{key}")
    _commit(source.get("source_commit"), "manifest.execution_source.source_commit")
    return checks


def preflight_bundle(
    *,
    bundle_dir: str | Path,
    source_archive: str | Path,
    environment_contract: str | Path,
) -> dict[str, Any]:
    """Perform the complete read-only production-readiness check."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest = validate_input_bundle(root)
    config = validate_config(read_json(root / "config.json"))
    approvals = _approval_state(config)
    external = _validate_external_execution_inputs(
        manifest,
        source_archive=source_archive,
        environment_contract=environment_contract,
    )
    checks = {
        # This is the config's explicit approval state.  The final plan hash
        # is deliberately external to avoid a config/plan/manifest hash
        # cycle and is verified by the launch-authorization receipt.
        "approved_plan": approvals["approved_plan"],
        "launch_authorized": approvals["launch_authorized"],
        "selection_override_approved": approvals[
            "selection_override_approved"
        ],
        "implementation_ready": approvals["implementation_ready"],
        "config_valid": True,
        "manifest_valid": True,
        "official_test_prohibited": (
            config["dataset"]["official_test"]
            == {"enabled": False, "read_allowed": False}
            and config["artifacts"]["official_test_read"] is False
            and config["prohibitions"]["official_test_read"] is True
        ),
        "expected_entry_count": manifest["entry_count"] == 12,
        **external,
    }
    passed = all(checks.values())
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "official_test_read": False,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(root / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "external_launch_authorization_required": True,
        "expected_entry_count": 12,
        "checks": checks,
    }


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], path: str
) -> None:
    if set(value) != expected:
        raise _error(path, f"exactly keys {sorted(expected)!r}", sorted(value))


def _read_hash_verified_file_from_environment(
    path_name: str, sha_name: str
) -> tuple[Path, str, dict[str, Any]]:
    raw_path = os.environ.get(path_name)
    raw_sha = os.environ.get(sha_name)
    if not raw_path or not raw_sha:
        raise _error(
            f"{path_name}/{sha_name}",
            "both environment variables to be set",
            {"path": raw_path, "sha256": raw_sha},
        )
    expected_sha = _sha256(raw_sha, sha_name)
    supplied = Path(raw_path).expanduser().absolute()
    if supplied.is_symlink():
        raise _error(path_name, "a non-symlink regular file", str(supplied))
    path = supplied.resolve()
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise _error(
            path_name,
            "the declared hash-verified regular file",
            str(path),
        )
    return path, expected_sha, _mapping(read_json(path), path_name)


def _validate_launch_authorization_payload(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    bundle_root: Path,
    input_hashes: Mapping[str, Any],
    verify_plan_file: bool,
) -> dict[str, Any]:
    value = _mapping(payload, "launch authorization")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "status",
            "launch_authorized",
            "approved_plan",
            "bundle",
            "execution",
        },
        "launch authorization",
    )
    for key, expected in (
        ("schema_version", LAUNCH_AUTHORIZATION_SCHEMA_VERSION),
        ("status", "approved"),
        ("launch_authorized", True),
    ):
        _require_equal(value.get(key), expected, f"launch authorization.{key}")
    plan = _mapping(
        value.get("approved_plan"), "launch authorization.approved_plan"
    )
    _require_exact_keys(plan, {"path", "sha256"}, "launch authorization.approved_plan")
    plan_digest = _sha256(
        plan.get("sha256"), "launch authorization.approved_plan.sha256"
    )
    if not isinstance(plan.get("path"), str) or not plan["path"]:
        raise _error(
            "launch authorization.approved_plan.path",
            "a non-empty absolute path",
            plan.get("path"),
        )
    if verify_plan_file:
        supplied_plan = Path(plan["path"]).expanduser().absolute()
        if supplied_plan.is_symlink():
            raise _error(
                "launch authorization.approved_plan.path",
                "a non-symlink file",
                str(supplied_plan),
            )
        plan_path = supplied_plan.resolve()
        if not plan_path.is_file() or sha256_file(plan_path) != plan_digest:
            raise _error(
                "launch authorization.approved_plan",
                "an existing hash-verified approved plan",
                plan,
            )
    bundle = _mapping(value.get("bundle"), "launch authorization.bundle")
    expected_bundle = {
        "bundle_id": manifest["bundle_id"],
        "manifest_sha256": sha256_file(bundle_root / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "bundle_tree_sha256": input_hashes.get("bundle_sha256"),
    }
    _require_equal(bundle, expected_bundle, "launch authorization.bundle")
    execution = _mapping(
        value.get("execution"), "launch authorization.execution"
    )
    expected_execution = {
        "source_archive_sha256": input_hashes.get("source_archive_sha256"),
        "environment_contract_sha256": input_hashes.get(
            "environment_contract_sha256"
        ),
        "runtime_cli_sha256": input_hashes.get("runtime_cli_sha256"),
        "runtime_module_sha256": input_hashes.get("runtime_module_sha256"),
        "submitter_sha256": input_hashes.get("submitter_sha256"),
        "supervisor_sha256": input_hashes.get("supervisor_sha256"),
        "semantic_canary_verifier_sha256": input_hashes.get(
            "semantic_canary_verifier_sha256"
        ),
        "experiment_plan_validator_sha256": input_hashes.get(
            "experiment_plan_validator_sha256"
        ),
        "wrapper_sha256": input_hashes.get("wrapper_sha256"),
        "scheduled_preflight_script_sha256": input_hashes.get(
            "scheduled_preflight_script_sha256"
        ),
        "official_canary_verifier_sha256": input_hashes.get(
            "official_canary_verifier_sha256"
        ),
    }
    for key, digest in expected_execution.items():
        _sha256(digest, f"launch authorization.execution.{key}")
    _require_equal(execution, expected_execution, "launch authorization.execution")
    return value


def _validate_launch_contract(
    raw: Any,
    *,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _mapping(raw, "input binding.launch_contract")
    _require_exact_keys(
        contract,
        {
            "schema_version",
            "self_reference_placeholders",
            "python_executable",
            "output_root",
            "common_inputs",
            "resources",
            "commands",
        },
        "input binding.launch_contract",
    )
    _require_equal(
        contract.get("schema_version"),
        "perfectdiode-successor-launch-contract/v1",
        "input binding.launch_contract.schema_version",
    )
    _require_equal(
        contract.get("self_reference_placeholders"),
        {
            "preflight_receipt_path": (
                "__PD_SUCCESSOR_PREFLIGHT_RECEIPT_PATH__"
            ),
            "preflight_receipt_sha256": (
                "__PD_SUCCESSOR_PREFLIGHT_RECEIPT_SHA256__"
            ),
            "canary_receipt_path": (
                "__PD_SUCCESSOR_CANARY_RECEIPT_PATH__"
            ),
            "canary_receipt_sha256": (
                "__PD_SUCCESSOR_CANARY_RECEIPT_SHA256__"
            ),
        },
        "input binding.launch_contract.self_reference_placeholders",
    )
    _require_equal(
        contract.get("python_executable"),
        JEAN_ZAY_PYTHON,
        "input binding.launch_contract.python_executable",
    )
    output_root = contract.get("output_root")
    if not isinstance(output_root, str) or not Path(output_root).is_absolute():
        raise _error(
            "input binding.launch_contract.output_root",
            "an absolute path",
            output_root,
        )
    output_root = str(Path(output_root).resolve())
    _require_equal(
        contract.get("output_root"),
        output_root,
        "input binding.launch_contract.output_root",
    )
    common_fields = (
        "repo_root",
        "bundle_dir",
        "data_root",
        "source_archive",
        "environment_contract",
        "runtime_cli",
        "runtime_module",
        "submitter",
        "supervisor",
        "semantic_canary_verifier",
        "experiment_plan_validator",
        "wrapper",
        "scheduled_preflight_script",
        "official_canary_verifier",
        "supervisor_state",
        "official_canary_receipt",
        "successor_canary_receipt",
        "preflight_receipt",
        "hashes",
        "canary_pack_index",
        "canary_entry_indices",
        "remote_user",
    )
    provided_common = _mapping(
        contract.get("common_inputs"),
        "input binding.launch_contract.common_inputs",
    )
    expected_common = {
        key: binding[key] if key in binding else provided_common.get(key)
        for key in common_fields
    }
    for key in (
        "scheduled_preflight_script",
        "official_canary_verifier",
        "supervisor_state",
        "official_canary_receipt",
        "successor_canary_receipt",
        "preflight_receipt",
    ):
        path = expected_common[key]
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise _error(
                f"input binding.launch_contract.common_inputs.{key}",
                "an absolute path",
                path,
            )
    remote_user = expected_common["remote_user"]
    if (
        not isinstance(remote_user, str)
        or not remote_user
        or any(
            character
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in remote_user
        )
    ):
        raise _error(
            "input binding.launch_contract.common_inputs.remote_user",
            "a safe non-empty remote login",
            remote_user,
        )
    expected_common.update(
        {
            "launch_authorization_receipt_path": binding[
                "launch_authorization_receipt_path"
            ],
            "launch_authorization_receipt_sha256": binding[
                "launch_authorization_receipt_sha256"
            ],
            "scheduled_preflight_receipt_path": binding[
                "scheduled_preflight_receipt_path"
            ],
            "scheduled_preflight_receipt_sha256": binding[
                "scheduled_preflight_receipt_sha256"
            ],
        }
    )
    _require_equal(
        provided_common,
        expected_common,
        "input binding.launch_contract.common_inputs",
    )
    expected_resources = {
        "allocation_id": "AD011016471R1",
        "project": "umg",
        "account": "umg@v100",
        "partition": "gpu_p13",
        "qos": "qos_gpu-t3",
        "constraint": "v100-32g",
        "nodes": 1,
        "tasks": 1,
        "gpus_per_task": 1,
        "cpus_per_task": 16,
        "memory_mb": 64_000,
        "hint": "nomultithread",
        "module": "pytorch-gpu/py3/2.5.0",
        "python_executable": JEAN_ZAY_PYTHON,
    }
    _require_equal(
        contract.get("resources"),
        expected_resources,
        "input binding.launch_contract.resources",
    )
    commands = _mapping(
        contract.get("commands"), "input binding.launch_contract.commands"
    )
    _require_exact_keys(commands, {"canary", "production"}, "launch contract.commands")
    command_expectations = {
        "canary": {
            "array": "0-0",
            "task_count": 1,
            "walltime": "02:00:00",
            "output_root": output_root,
        },
        "production": {
            "array": "0-5%6",
            "task_count": 6,
            "walltime": "08:00:00",
            "output_root": output_root,
        },
    }
    preflight_placeholders = (
        "__PD_SUCCESSOR_PREFLIGHT_RECEIPT_PATH__",
        "__PD_SUCCESSOR_PREFLIGHT_RECEIPT_SHA256__",
    )
    canary_placeholders = (
        "__PD_SUCCESSOR_CANARY_RECEIPT_PATH__",
        "__PD_SUCCESSOR_CANARY_RECEIPT_SHA256__",
    )
    for kind, expected in command_expectations.items():
        command = _mapping(commands.get(kind), f"launch contract.commands.{kind}")
        _require_exact_keys(
            command,
            {
                "kind",
                "array",
                "task_count",
                "walltime",
                "output_root",
                "template",
                "template_sha256",
            },
            f"launch contract.commands.{kind}",
        )
        for key, value in {"kind": kind, **expected}.items():
            _require_equal(
                command.get(key), value, f"launch contract.commands.{kind}.{key}"
            )
        template = _list(
            command.get("template"), f"launch contract.commands.{kind}.template"
        )
        if not template or any(not isinstance(item, str) for item in template):
            raise _error(
                f"launch contract.commands.{kind}.template",
                "a non-empty string list",
                template,
            )
        _require_equal(
            command.get("template_sha256"),
            sha256_json(template),
            f"launch contract.commands.{kind}.template_sha256",
        )
        required_placeholders = (
            preflight_placeholders
            if kind == "canary"
            else preflight_placeholders + canary_placeholders
        )
        for placeholder in required_placeholders:
            _require_equal(
                sum(item.count(placeholder) for item in template),
                1,
                f"launch contract.commands.{kind} placeholder {placeholder}",
            )
        if kind == "canary":
            for placeholder in canary_placeholders:
                _require_equal(
                    sum(item.count(placeholder) for item in template),
                    0,
                    "launch contract.commands.canary forbidden placeholder "
                    f"{placeholder}",
                )
        required_flags = {
            f"--array={expected['array']}",
            f"--time={expected['walltime']}",
            "--account=umg@v100",
            "--partition=gpu_p13",
            "--qos=qos_gpu-t3",
            "--constraint=v100-32g",
            "--gres=gpu:1",
            "--cpus-per-task=16",
            "--mem=64000M",
            "--hint=nomultithread",
        }
        if not required_flags.issubset(template):
            raise _error(
                f"launch contract.commands.{kind}.template",
                f"all flags {sorted(required_flags)!r}",
                template,
            )
    return contract


def _validate_receipt_payload_structure(
    receipt: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    bundle_root: Path,
    launch_authorization: Mapping[str, Any],
    launch_authorization_path: str,
    launch_authorization_sha256: str,
    verify_plan_file: bool,
) -> dict[str, Any]:
    value = _mapping(receipt, "preflight receipt")
    _require_exact_keys(
        value,
        {
            "schema_version",
            "status",
            "passed",
            "runtime_preflight",
            "runtime_preflight_sha256",
            "input_binding",
            "input_binding_sha256",
            "scheduled_run_preflight",
        },
        "preflight receipt",
    )
    for key, expected in (
        ("schema_version", CUSTOM_PREFLIGHT_RECEIPT_SCHEMA_VERSION),
        ("status", "passed"),
        ("passed", True),
    ):
        _require_equal(value.get(key), expected, f"preflight receipt.{key}")
    runtime = _mapping(
        value.get("runtime_preflight"), "preflight receipt.runtime_preflight"
    )
    _require_equal(
        value.get("runtime_preflight_sha256"),
        sha256_json(runtime),
        "preflight receipt.runtime_preflight_sha256",
    )
    for key, expected in (
        ("schema_version", PREFLIGHT_SCHEMA_VERSION),
        ("status", "passed"),
        ("passed", True),
        ("official_test_read", False),
        ("bundle_id", manifest["bundle_id"]),
        ("bundle_manifest_sha256", sha256_file(bundle_root / "manifest.json")),
        ("config_sha256", manifest["config_sha256"]),
        ("config_file_sha256", manifest["config_file_sha256"]),
    ):
        _require_equal(runtime.get(key), expected, f"runtime preflight.{key}")
    checks = _mapping(runtime.get("checks"), "runtime preflight.checks")
    if not checks or any(item is not True for item in checks.values()):
        raise _error(
            "runtime preflight.checks",
            "a non-empty map whose every gate is true",
            checks,
        )
    binding = _mapping(
        value.get("input_binding"), "preflight receipt.input_binding"
    )
    _require_exact_keys(
        binding,
        {
            "schema_version",
            "bundle_id",
            "bundle_manifest_sha256",
            "config_sha256",
            "config_file_sha256",
            "repo_root",
            "bundle_dir",
            "data_root",
            "source_archive",
            "environment_contract",
            "runtime_cli",
            "runtime_module",
            "submitter",
            "supervisor",
            "semantic_canary_verifier",
            "experiment_plan_validator",
            "wrapper",
            "scheduled_preflight_script",
            "official_canary_verifier",
            "canary_pack_index",
            "canary_entry_indices",
            "hashes",
            "runtime_preflight_sha256",
            "scheduled_preflight_receipt_path",
            "scheduled_preflight_receipt_sha256",
            "launch_authorization_receipt_path",
            "launch_authorization_receipt_sha256",
            "approved_plan",
            "launch_contract",
            "launch_contract_sha256",
        },
        "preflight receipt.input_binding",
    )
    _require_equal(
        value.get("input_binding_sha256"),
        sha256_json(binding),
        "preflight receipt.input_binding_sha256",
    )
    launch_contract = _validate_launch_contract(
        binding.get("launch_contract"), binding=binding
    )
    _require_equal(
        binding.get("launch_contract_sha256"),
        sha256_json(launch_contract),
        "input binding.launch_contract_sha256",
    )
    for key, expected in (
        ("schema_version", GATE_INPUT_SCHEMA_VERSION),
        ("bundle_id", manifest["bundle_id"]),
        ("bundle_manifest_sha256", sha256_file(bundle_root / "manifest.json")),
        ("config_sha256", manifest["config_sha256"]),
        ("config_file_sha256", manifest["config_file_sha256"]),
        ("canary_pack_index", CANARY_PACK_INDEX),
        ("canary_entry_indices", list(CANARY_ENTRY_INDICES)),
        ("runtime_preflight_sha256", sha256_json(runtime)),
        ("launch_authorization_receipt_path", launch_authorization_path),
        (
            "launch_authorization_receipt_sha256",
            launch_authorization_sha256,
        ),
    ):
        _require_equal(binding.get(key), expected, f"input binding.{key}")
    hashes = _mapping(binding.get("hashes"), "input binding.hashes")
    _require_exact_keys(
        hashes,
        {
            "bundle_sha256",
            "source_archive_sha256",
            "environment_contract_sha256",
            "runtime_cli_sha256",
            "runtime_module_sha256",
            "submitter_sha256",
            "supervisor_sha256",
            "semantic_canary_verifier_sha256",
            "experiment_plan_validator_sha256",
            "wrapper_sha256",
            "scheduled_preflight_script_sha256",
            "official_canary_verifier_sha256",
        },
        "input binding.hashes",
    )
    for key, digest in hashes.items():
        _sha256(digest, f"input binding.hashes.{key}")
    for key, expected in (
        (
            "source_archive_sha256",
            manifest["execution_source"]["source_archive_sha256"],
        ),
        (
            "environment_contract_sha256",
            manifest["execution_source"]["environment_contract_sha256"],
        ),
        (
            "runtime_cli_sha256",
            manifest["execution_source"]["worker_launcher_sha256"],
        ),
        (
            "runtime_module_sha256",
            manifest["execution_source"]["runtime_module_sha256"],
        ),
    ):
        _require_equal(
            hashes.get(key), expected, f"input binding.hashes.{key}"
        )
    authorization = _validate_launch_authorization_payload(
        launch_authorization,
        manifest=manifest,
        bundle_root=bundle_root,
        input_hashes=hashes,
        verify_plan_file=verify_plan_file,
    )
    _require_equal(
        binding.get("approved_plan"),
        authorization["approved_plan"],
        "input binding.approved_plan",
    )
    scheduled = _mapping(
        value.get("scheduled_run_preflight"),
        "preflight receipt.scheduled_run_preflight",
    )
    _require_exact_keys(
        scheduled,
        {
            "path",
            "sha256",
            "schema_version",
            "status",
            "runner",
            "runner_sha256",
        },
        "preflight receipt.scheduled_run_preflight",
    )
    for key, expected in (
        ("path", binding.get("scheduled_preflight_receipt_path")),
        ("sha256", binding.get("scheduled_preflight_receipt_sha256")),
        ("schema_version", SCHEDULED_PREFLIGHT_RECEIPT_SCHEMA_VERSION),
        ("status", "passed"),
        ("runner", binding.get("wrapper")),
        ("runner_sha256", hashes.get("wrapper_sha256")),
    ):
        _require_equal(scheduled.get(key), expected, f"scheduled preflight.{key}")
    return value


def _preflight_receipt_from_environment(
    *,
    manifest: Mapping[str, Any],
    bundle_root: Path,
    required: bool,
    source_archive: str | Path,
    environment_contract: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    expected_kind: str,
) -> dict[str, Any] | None:
    raw_path = os.environ.get("PD_SUCCESSOR_PREFLIGHT_RECEIPT")
    raw_sha = os.environ.get("PD_SUCCESSOR_PREFLIGHT_RECEIPT_SHA256")
    if raw_path is None and raw_sha is None and not required:
        return None
    if not raw_path or not raw_sha:
        raise _error(
            "preflight receipt environment",
            "both PD_SUCCESSOR_PREFLIGHT_RECEIPT and "
            "PD_SUCCESSOR_PREFLIGHT_RECEIPT_SHA256",
            {"path": raw_path, "sha256": raw_sha},
        )
    path, expected_sha, receipt = _read_hash_verified_file_from_environment(
        "PD_SUCCESSOR_PREFLIGHT_RECEIPT",
        "PD_SUCCESSOR_PREFLIGHT_RECEIPT_SHA256",
    )
    authorization_path, authorization_sha, authorization = (
        _read_hash_verified_file_from_environment(
            "PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT",
            "PD_SUCCESSOR_LAUNCH_AUTHORIZATION_RECEIPT_SHA256",
        )
    )
    value = _validate_receipt_payload_structure(
        receipt,
        manifest=manifest,
        bundle_root=bundle_root,
        launch_authorization=authorization,
        launch_authorization_path=str(authorization_path),
        launch_authorization_sha256=authorization_sha,
        verify_plan_file=True,
    )
    binding = value["input_binding"]
    expected_runtime = preflight_bundle(
        bundle_dir=bundle_root,
        source_archive=source_archive,
        environment_contract=environment_contract,
    )
    _require_equal(
        value["runtime_preflight"],
        expected_runtime,
        "preflight receipt.runtime_preflight",
    )
    archive = Path(source_archive).expanduser().resolve()
    environment = Path(environment_contract).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    contract_common = value["input_binding"]["launch_contract"][
        "common_inputs"
    ]
    _require_equal(
        contract_common["preflight_receipt"],
        str(path),
        "launch contract preflight_receipt",
    )
    official_verifier = Path(
        contract_common["official_canary_verifier"]
    ).expanduser().resolve()
    _require_equal(
        contract_common["scheduled_preflight_script"],
        str(SCHEDULED_PREFLIGHT_SCRIPT.resolve()),
        "launch contract scheduled_preflight_script",
    )
    _require_equal(
        value["input_binding"]["scheduled_preflight_script"],
        str(SCHEDULED_PREFLIGHT_SCRIPT.resolve()),
        "input binding.scheduled_preflight_script",
    )
    _require_equal(
        value["input_binding"]["official_canary_verifier"],
        str(official_verifier),
        "input binding.official_canary_verifier",
    )
    live_hashes = {
        "bundle_sha256": _directory_tree_sha256(bundle_root),
        "source_archive_sha256": sha256_file(archive),
        "environment_contract_sha256": sha256_file(environment),
        "runtime_cli_sha256": sha256_file(WORKER_LAUNCHER),
        "runtime_module_sha256": sha256_file(RUNTIME_MODULE),
        "submitter_sha256": sha256_file(JEAN_ZAY_SUBMITTER),
        "supervisor_sha256": sha256_file(JEAN_ZAY_SUPERVISOR),
        "semantic_canary_verifier_sha256": sha256_file(
            SEMANTIC_CANARY_VERIFIER
        ),
        "experiment_plan_validator_sha256": sha256_file(
            EXPERIMENT_PLAN_VALIDATOR
        ),
        "wrapper_sha256": sha256_file(JEAN_ZAY_WRAPPER),
        "scheduled_preflight_script_sha256": sha256_file(
            SCHEDULED_PREFLIGHT_SCRIPT
        ),
        "official_canary_verifier_sha256": sha256_file(official_verifier),
    }
    for key, expected in (
        ("repo_root", str(REPO_ROOT.resolve())),
        ("bundle_dir", str(bundle_root)),
        ("data_root", str(data)),
        ("source_archive", str(archive)),
        ("environment_contract", str(environment)),
        ("runtime_cli", str(WORKER_LAUNCHER.resolve())),
        ("runtime_module", str(RUNTIME_MODULE.resolve())),
        ("submitter", str(JEAN_ZAY_SUBMITTER.resolve())),
        ("supervisor", str(JEAN_ZAY_SUPERVISOR.resolve())),
        (
            "semantic_canary_verifier",
            str(SEMANTIC_CANARY_VERIFIER.resolve()),
        ),
        (
            "experiment_plan_validator",
            str(EXPERIMENT_PLAN_VALIDATOR.resolve()),
        ),
        ("wrapper", str(JEAN_ZAY_WRAPPER.resolve())),
        (
            "scheduled_preflight_script",
            str(SCHEDULED_PREFLIGHT_SCRIPT.resolve()),
        ),
        ("official_canary_verifier", str(official_verifier)),
    ):
        _require_equal(binding.get(key), expected, f"input binding.{key}")
    for key, expected in live_hashes.items():
        _require_equal(
            binding["hashes"].get(key),
            expected,
            f"input binding.hashes.{key}",
        )
    if expected_kind not in {"canary", "production"}:
        raise _error(
            "expected_kind", "'canary' or 'production'", expected_kind
        )
    _require_equal(
        value["input_binding"]["launch_contract"]["output_root"],
        str(Path(output_root).expanduser().resolve()),
        f"launch contract shared {expected_kind} output root",
    )
    _require_equal(
        os.environ.get("PD_SUCCESSOR_PYTHON"),
        JEAN_ZAY_PYTHON,
        "PD_SUCCESSOR_PYTHON",
    )
    _require_equal(
        str(Path(sys.executable).resolve()),
        str(Path(JEAN_ZAY_PYTHON).resolve()),
        "running Python executable",
    )
    scheduled_path = Path(
        binding["scheduled_preflight_receipt_path"]
    ).expanduser().resolve()
    scheduled_sha = _sha256(
        binding["scheduled_preflight_receipt_sha256"],
        "input binding.scheduled_preflight_receipt_sha256",
    )
    if (
        not scheduled_path.is_file()
        or scheduled_path.is_symlink()
        or sha256_file(scheduled_path) != scheduled_sha
    ):
        raise _error(
            "input binding scheduled preflight receipt",
            "the live hash-verified regular file",
            str(scheduled_path),
        )
    scheduled_payload = _mapping(
        read_json(scheduled_path), "scheduled-run preflight receipt"
    )
    for key, expected in (
        ("schema_version", SCHEDULED_PREFLIGHT_RECEIPT_SCHEMA_VERSION),
        ("status", "passed"),
        ("runner", binding["wrapper"]),
        ("runner_sha256", binding["hashes"]["wrapper_sha256"]),
    ):
        _require_equal(
            scheduled_payload.get(key),
            expected,
            f"scheduled-run preflight receipt.{key}",
        )
    embedded_authorization = {
        "source_path": str(authorization_path),
        "source_file_sha256": authorization_sha,
        "payload_sha256": sha256_json(authorization),
        "payload": authorization,
    }
    return {
        "source_path": str(path),
        "source_file_sha256": expected_sha,
        "payload_sha256": sha256_json(value),
        "payload": value,
        "launch_authorization": embedded_authorization,
    }


def _validate_embedded_preflight_binding(
    value: Any,
    *,
    manifest: Mapping[str, Any],
    bundle_root: Path,
    path: str,
) -> dict[str, Any]:
    binding = _mapping(value, path)
    _require_exact_keys(
        binding,
        {
            "source_path",
            "source_file_sha256",
            "payload_sha256",
            "payload",
            "launch_authorization",
        },
        path,
    )
    if not isinstance(binding.get("source_path"), str) or not binding[
        "source_path"
    ]:
        raise _error(f"{path}.source_path", "a non-empty string", binding.get("source_path"))
    _sha256(binding.get("source_file_sha256"), f"{path}.source_file_sha256")
    payload = _mapping(binding.get("payload"), f"{path}.payload")
    _require_equal(
        binding.get("payload_sha256"),
        sha256_json(payload),
        f"{path}.payload_sha256",
    )
    authorization_binding = _mapping(
        binding.get("launch_authorization"), f"{path}.launch_authorization"
    )
    _require_exact_keys(
        authorization_binding,
        {"source_path", "source_file_sha256", "payload_sha256", "payload"},
        f"{path}.launch_authorization",
    )
    if (
        not isinstance(authorization_binding.get("source_path"), str)
        or not authorization_binding["source_path"]
    ):
        raise _error(
            f"{path}.launch_authorization.source_path",
            "a non-empty string",
            authorization_binding.get("source_path"),
        )
    authorization_sha = _sha256(
        authorization_binding.get("source_file_sha256"),
        f"{path}.launch_authorization.source_file_sha256",
    )
    authorization = _mapping(
        authorization_binding.get("payload"),
        f"{path}.launch_authorization.payload",
    )
    _require_equal(
        authorization_binding.get("payload_sha256"),
        sha256_json(authorization),
        f"{path}.launch_authorization.payload_sha256",
    )
    _validate_receipt_payload_structure(
        payload,
        manifest=manifest,
        bundle_root=bundle_root,
        launch_authorization=authorization,
        launch_authorization_path=authorization_binding["source_path"],
        launch_authorization_sha256=authorization_sha,
        verify_plan_file=False,
    )
    return binding


def _expected_canary_gate_binding(
    preflight_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the exact post-canary gate identity from the live preflight."""

    embedded = _mapping(preflight_receipt, "embedded preflight receipt")
    payload = _mapping(
        embedded.get("payload"), "embedded preflight receipt.payload"
    )
    input_binding = _mapping(
        payload.get("input_binding"),
        "embedded preflight receipt.payload.input_binding",
    )
    return {
        **input_binding,
        "preflight_receipt_path": embedded.get("source_path"),
        "preflight_receipt_sha256": embedded.get("source_file_sha256"),
    }


def _validate_live_canary_gate_receipt(
    path: Path,
    *,
    expected_gate_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the separately hash-bound semantic verifier at worker start."""

    from experiments.verify_mnist_conv_perfectdiode_successor_canary import (
        validate_canary_gate_receipt,
    )

    return _mapping(
        validate_canary_gate_receipt(
            path,
            expected_gate_binding=expected_gate_binding,
        ),
        "successor canary gate receipt",
    )


def _canary_gate_receipt_from_environment(
    *,
    preflight_receipt: Mapping[str, Any] | None,
    required: bool,
) -> dict[str, Any] | None:
    """Hash-verify and semantically revalidate the production canary gate."""

    raw_path = os.environ.get("PD_SUCCESSOR_CANARY_RECEIPT")
    raw_sha = os.environ.get("PD_SUCCESSOR_CANARY_RECEIPT_SHA256")
    if raw_path is None and raw_sha is None and not required:
        return None
    if not raw_path or not raw_sha:
        raise _error(
            "PD_SUCCESSOR_CANARY_RECEIPT/"
            "PD_SUCCESSOR_CANARY_RECEIPT_SHA256",
            "both environment variables to be set for production",
            {"path": raw_path, "sha256": raw_sha},
        )
    if preflight_receipt is None:
        raise _error(
            "production canary gate",
            "an already validated embedded preflight receipt",
            preflight_receipt,
        )
    path, expected_sha, payload = _read_hash_verified_file_from_environment(
        "PD_SUCCESSOR_CANARY_RECEIPT",
        "PD_SUCCESSOR_CANARY_RECEIPT_SHA256",
    )
    expected_gate = _expected_canary_gate_binding(preflight_receipt)
    validated = _validate_live_canary_gate_receipt(
        path,
        expected_gate_binding=expected_gate,
    )
    _require_equal(
        validated,
        payload,
        "successor canary gate receipt live validation",
    )
    for key, expected in (
        ("schema_version", CANARY_GATE_SCHEMA_VERSION),
        ("status", "passed"),
        ("gate_passed", True),
        ("gate_binding", expected_gate),
        ("gate_binding_sha256", sha256_json(expected_gate)),
    ):
        _require_equal(
            payload.get(key),
            expected,
            f"successor canary gate receipt.{key}",
        )
    return {
        "source_path": str(path),
        "source_file_sha256": expected_sha,
        "payload_sha256": sha256_json(payload),
        "payload": payload,
    }


def _validate_embedded_canary_gate_binding(
    value: Any,
    *,
    preflight_receipt: Mapping[str, Any],
    path: str,
) -> dict[str, Any]:
    """Validate archived canary provenance without requiring remote files."""

    binding = _mapping(value, path)
    _require_exact_keys(
        binding,
        {
            "source_path",
            "source_file_sha256",
            "payload_sha256",
            "payload",
        },
        path,
    )
    if not isinstance(binding.get("source_path"), str) or not binding[
        "source_path"
    ]:
        raise _error(
            f"{path}.source_path",
            "a non-empty string",
            binding.get("source_path"),
        )
    _sha256(binding.get("source_file_sha256"), f"{path}.source_file_sha256")
    payload = _mapping(binding.get("payload"), f"{path}.payload")
    _require_equal(
        binding.get("payload_sha256"),
        sha256_json(payload),
        f"{path}.payload_sha256",
    )
    expected_gate = _expected_canary_gate_binding(preflight_receipt)
    for key, expected in (
        ("schema_version", CANARY_GATE_SCHEMA_VERSION),
        ("status", "passed"),
        ("gate_passed", True),
        ("gate_binding", expected_gate),
        ("gate_binding_sha256", sha256_json(expected_gate)),
    ):
        _require_equal(payload.get(key), expected, f"{path}.payload.{key}")
    return binding


def _output_roots_are_distinct(bundle_root: Path, output_root: Path) -> None:
    try:
        output_root.relative_to(bundle_root)
    except ValueError:
        pass
    else:
        raise _error(
            "output_root",
            "a directory outside the immutable input bundle",
            str(output_root),
        )
    try:
        bundle_root.relative_to(output_root)
    except ValueError:
        return
    raise _error(
        "output_root",
        "a directory that does not contain the immutable input bundle",
        str(output_root),
    )


def _output_records(output_root: Path, entry_dir: Path) -> list[dict[str, Any]]:
    records = []
    for name in PRODUCTION_OUTPUT_NAMES:
        path = entry_dir / name
        if not path.is_file() or path.is_symlink():
            raise _error(
                f"production output {name}",
                "an existing non-symlink regular file",
                str(path),
            )
        records.append(_artifact_record(output_root, path))
    return sorted(records, key=lambda record: record["path"])


@contextmanager
def _entry_execution_lock(output_root: Path, entry_id: str):
    lock_dir = output_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{entry_id}.lock"
    handle = lock_path.open("a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PerfectDiodeSuccessorError(
                "Expected no concurrent execution for the same manifest entry. "
                f"Provided value: active lock {lock_path}."
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _validate_entry_completion(
    *,
    bundle_root: Path,
    manifest: Mapping[str, Any],
    output_root: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    entry_dir = output_root / "entries" / str(entry["entry_id"])
    path = entry_dir / "completion.json"
    completion = _mapping(read_json(path), "entry completion")
    expected_scalars = {
        "schema_version": ENTRY_COMPLETION_SCHEMA_VERSION,
        "state": "complete",
        "official_test_read": False,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(bundle_root / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "entry_index": entry["entry_index"],
        "entry_id": entry["entry_id"],
        "entry_payload_sha256": sha256_json(entry),
        "execution_source": manifest["execution_source"],
    }
    for key, expected in expected_scalars.items():
        _require_equal(completion.get(key), expected, f"entry completion.{key}")
    _require_equal(
        completion.get("outputs"),
        _output_records(output_root, entry_dir),
        "entry completion.outputs",
    )
    receipt = completion.get("preflight_receipt")
    if receipt is None:
        raise _error(
            "entry completion.preflight_receipt",
            "the embedded production preflight provenance",
            receipt,
        )
    _validate_embedded_preflight_binding(
        receipt,
        manifest=manifest,
        bundle_root=bundle_root,
        path="entry completion.preflight_receipt",
    )
    canary_receipt = completion.get("canary_gate_receipt")
    if canary_receipt is None:
        raise _error(
            "entry completion.canary_gate_receipt",
            "the embedded production canary-gate provenance",
            canary_receipt,
        )
    _validate_embedded_canary_gate_binding(
        canary_receipt,
        preflight_receipt=receipt,
        path="entry completion.canary_gate_receipt",
    )
    return completion


def _enrich_successor_outputs(
    *,
    bundle_root: Path,
    manifest: Mapping[str, Any],
    entry: Mapping[str, Any],
    entry_dir: Path,
    preflight_receipt: Mapping[str, Any],
    canary_gate_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    result_path = entry_dir / "result.json"
    run_spec_path = entry_dir / "run_spec.json"
    result = _mapping(read_json(result_path), "successor training result")
    run_spec = _mapping(read_json(run_spec_path), "successor run spec")
    validation = _mapping(
        read_json(entry_dir / "validation.json"), "successor validation"
    )
    for label, value in (
        ("successor training result", result),
        ("successor run spec", run_spec),
        ("successor validation", validation),
    ):
        require_official_test_excluded(value, require_result_marker=True)
        _require_official_test_false_recursive(value, label)
    exact_result = {
        "mode": "successor_confirmation",
        "architecture": entry["architecture"],
        "status": "complete",
        "training_completed": True,
        "completed_steps": entry["expected_steps"],
        "expected_steps": entry["expected_steps"],
        "epochs_completed": entry["epochs"],
        "expected_epochs": entry["epochs"],
        "entry_id": entry["entry_id"],
        "cell_id": entry["cell_id"],
        "surface_id": f"{entry['row_id']}--{entry['optimizer']}",
        "row_id": entry["row_id"],
        "scheme": entry["scheme"],
        "optimizer": entry["optimizer"],
        "rho_conv": entry["runtime_rho_conv"],
        "rho_dense": entry["runtime_rho_dense"],
        "raw_learning_rates_by_parameter": entry[
            "raw_learning_rates_by_parameter"
        ],
        "official_test_read": False,
    }
    for key, expected in exact_result.items():
        _require_equal(result.get(key), expected, f"successor result.{key}")
    training = _mapping(run_spec.get("training"), "successor run_spec.training")
    for key, expected in (
        ("mode", "successor_confirmation"),
        ("epochs", entry["epochs"]),
        ("steps_per_epoch", entry["steps_per_epoch"]),
        ("total_steps", entry["expected_steps"]),
        ("restart_from_shared_initialization", True),
        ("continue_from_canary_or_candidate", False),
    ):
        _require_equal(training.get(key), expected, f"successor run_spec.training.{key}")
    records = _list(
        validation.get("records"),
        "successor validation.records",
        length=int(entry["epochs"]),
    )
    _require_equal(
        records[-1].get("step"),
        entry["expected_steps"],
        "successor final validation step",
    )
    provenance = {
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(bundle_root / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "entry_index": entry["entry_index"],
        "entry_payload_sha256": sha256_json(entry),
        "selection_basis": entry["selection_basis"],
        "boundary_status": entry["boundary_status"],
        "candidate_evidence": copy.deepcopy(entry["candidate_evidence"]),
        "preflight_receipt": dict(preflight_receipt),
        "canary_gate_receipt": dict(canary_gate_receipt),
    }
    result["successor_confirmation"] = copy.deepcopy(provenance)
    run_spec["successor_confirmation"] = copy.deepcopy(provenance)
    atomic_write_json(run_spec_path, run_spec, canonical=True)
    artifacts = _list(result.get("artifacts"), "successor result.artifacts")
    expected_artifacts = {
        "run_spec.json",
        "validation.json",
        "step_log.csv",
        "best_validation.pt",
        "final.pt",
    }
    observed_artifacts: set[str] = set()
    for index, record in enumerate(artifacts):
        value = _mapping(record, f"successor result.artifacts[{index}]")
        _require_exact_keys(
            value,
            {"path", "sha256", "bytes"},
            f"successor result.artifacts[{index}]",
        )
        relative = _portable_relative(
            value.get("path"), f"successor result.artifacts[{index}].path"
        )
        if relative not in expected_artifacts or relative in observed_artifacts:
            raise _error(
                "successor result.artifacts",
                f"each of {sorted(expected_artifacts)!r} exactly once",
                relative,
            )
        observed_artifacts.add(relative)
        artifact_path = entry_dir / relative
        if relative == "run_spec.json":
            record["sha256"] = sha256_file(run_spec_path)
            record["bytes"] = run_spec_path.stat().st_size
        else:
            _require_equal(
                value,
                {
                    "path": relative,
                    "sha256": sha256_file(artifact_path),
                    "bytes": artifact_path.stat().st_size,
                },
                f"successor result.artifacts[{index}]",
            )
    _require_equal(
        observed_artifacts,
        expected_artifacts,
        "successor result.artifacts paths",
    )
    result["artifacts"] = artifacts
    atomic_write_json(result_path, result, canonical=True)
    return result


def _training_payload(
    manifest: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "entry_id": entry["entry_id"],
        "row_id": entry["row_id"],
        "architecture": entry["architecture"],
        "asset_entry_id": entry["asset_entry_id"],
        "surface_id": f"{entry['row_id']}--{entry['optimizer']}",
        "optimizer": entry["optimizer"],
        "rho_conv": entry["runtime_rho_conv"],
        "rho_dense": entry["runtime_rho_dense"],
        "cell_id": entry["cell_id"],
        "probe_result_path": entry["probe_result"],
        "raw_learning_rates_by_parameter": entry[
            "raw_learning_rates_by_parameter"
        ],
        "source_commit": manifest["execution_source"]["source_commit"],
        "source_archive_sha256": manifest["execution_source"][
            "source_archive_sha256"
        ],
        "environment_sha256": manifest["execution_source"][
            "environment_contract_sha256"
        ],
        "fresh_restart_from_shared_initialization": True,
        "official_test_read": False,
    }


def fixed_pack_entries(pack_index: int) -> tuple[int, int]:
    """Return the immutable pair for one of the six Slurm pack indices."""

    if (
        isinstance(pack_index, bool)
        or not isinstance(pack_index, int)
        or not 0 <= pack_index < PRODUCTION_PACK_COUNT
    ):
        raise _error(
            "pack_index",
            f"an integer in [0, {PRODUCTION_PACK_COUNT - 1}]",
            pack_index,
        )
    return FIXED_ENTRY_PACKS[pack_index]


def _runtime_child_command(
    mode: str,
    *,
    bundle_dir: Path,
    output_root: Path,
    data_root: Path,
    source_archive: Path,
    environment_contract: Path,
    entry_index: int,
    device: str,
    download: bool,
    python_executable: str | Path | None,
) -> list[str]:
    if mode not in {"run-entry", "run-canary-entry"}:
        raise _error(
            "runtime child mode",
            "'run-entry' or 'run-canary-entry'",
            mode,
        )
    python = Path(
        sys.executable if python_executable is None else python_executable
    ).expanduser()
    if not python.is_absolute():
        raise _error(
            "runtime child Python",
            "an absolute interpreter path",
            str(python),
        )
    command = [
        str(python.resolve()),
        str(WORKER_LAUNCHER.resolve()),
        mode,
        "--bundle-dir",
        str(bundle_dir),
        "--output-root",
        str(output_root),
        "--data-root",
        str(data_root),
        "--source-archive",
        str(source_archive),
        "--environment-contract",
        str(environment_contract),
        "--entry-index",
        str(entry_index),
        "--device",
        device,
    ]
    if download:
        command.append("--download")
    return command


def _terminal_result_from_log(path: Path, *, label: str) -> dict[str, Any]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    marked = [
        line[len(RESULT_JSON_MARKER) :]
        for line in lines
        if line.startswith(RESULT_JSON_MARKER)
    ]
    if (
        len(marked) != 1
        or not lines
        or not lines[-1].startswith(RESULT_JSON_MARKER)
    ):
        raise _error(
            label,
            f"exactly one terminal {RESULT_JSON_MARKER!r} line",
            lines[-20:],
        )
    try:
        value = json.loads(marked[0])
    except json.JSONDecodeError as exc:
        raise _error(label, "a terminal marker containing strict JSON", marked[0]) from exc
    return _mapping(value, label)


def _run_concurrent_children(
    specs: Sequence[Mapping[str, Any]],
    *,
    log_root: Path,
    artifact_root: Path,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> tuple[list[dict[str, Any]], float]:
    """Start every child before waiting and bind its terminal result/logs."""

    if len(specs) != RUNS_PER_GPU:
        raise _error(
            "concurrent child specs",
            f"exactly {RUNS_PER_GPU} children",
            len(specs),
        )
    log_root.mkdir(parents=True, exist_ok=False)
    active: list[dict[str, Any]] = []
    handles: list[Any] = []
    window_started = time.monotonic()
    try:
        for raw in specs:
            spec = _mapping(raw, "concurrent child spec")
            label = spec.get("label")
            command = spec.get("command")
            if (
                not isinstance(label, str)
                or not label
                or not isinstance(command, list)
                or not command
                or any(not isinstance(item, str) for item in command)
            ):
                raise _error(
                    "concurrent child spec",
                    "a non-empty label and command string list",
                    spec,
                )
            stdout_path = log_root / f"{label}.stdout.log"
            stderr_path = log_root / f"{label}.stderr.log"
            stdout_handle = stdout_path.open("x", encoding="utf-8")
            stderr_handle = stderr_path.open("x", encoding="utf-8")
            handles.extend((stdout_handle, stderr_handle))
            started = time.monotonic()
            process = popen_factory(
                command,
                cwd=str(REPO_ROOT.resolve()),
                env=dict(os.environ),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            active.append(
                {
                    "label": label,
                    "entry_index": spec.get("entry_index"),
                    "command": list(command),
                    "process": process,
                    "started": started,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                }
            )
        for record in active:
            record["returncode"] = int(record["process"].wait())
            record["process_elapsed_seconds"] = max(
                time.monotonic() - float(record["started"]),
                1.0e-12,
            )
    except BaseException:
        for record in active:
            process = record["process"]
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            except BaseException:
                pass
        raise
    finally:
        for handle in handles:
            handle.close()
    wall_elapsed = max(time.monotonic() - window_started, 1.0e-12)
    failures = [
        {
            "label": record["label"],
            "entry_index": record["entry_index"],
            "returncode": record["returncode"],
            "stdout": str(record["stdout_path"]),
            "stderr": str(record["stderr_path"]),
        }
        for record in active
        if record["returncode"] != 0
    ]
    if failures:
        raise RuntimeError(
            "Expected both concurrent packed children to exit successfully. "
            f"Provided failures: {failures!r}."
        )
    results = []
    for record in active:
        terminal = _terminal_result_from_log(
            record["stdout_path"],
            label=f"{record['label']} stdout",
        )
        results.append(
            {
                "label": record["label"],
                "entry_index": record["entry_index"],
                "command": record["command"],
                "returncode": record["returncode"],
                "process_elapsed_seconds": record[
                    "process_elapsed_seconds"
                ],
                "terminal_result": terminal,
                "stdout_log": _artifact_record(
                    artifact_root, record["stdout_path"]
                ),
                "stderr_log": _artifact_record(
                    artifact_root, record["stderr_path"]
                ),
            }
        )
    return results, wall_elapsed


def run_indexed_entry(
    *,
    bundle_dir: str | Path,
    entry_index: int,
    output_root: str | Path,
    data_root: str | Path,
    source_archive: str | Path,
    environment_contract: str | Path,
    device: str = "cuda",
    download: bool = False,
    executor: Callable[..., Mapping[str, Any]] = execute_successor_stage_entry,
    require_preflight_receipt: bool = True,
) -> dict[str, Any]:
    """Execute exactly one manifest index and publish its completion last."""

    if (
        isinstance(entry_index, bool)
        or not isinstance(entry_index, int)
        or not 0 <= entry_index < 12
    ):
        raise _error("entry_index", "an integer in [0, 11]", entry_index)
    bundle_root = Path(bundle_dir).expanduser().resolve()
    manifest = validate_input_bundle(bundle_root)
    config = validate_config(
        read_json(bundle_root / "config.json"),
        require_production_approval=True,
    )
    preflight = preflight_bundle(
        bundle_dir=bundle_root,
        source_archive=source_archive,
        environment_contract=environment_contract,
    )
    if preflight["passed"] is not True:
        raise _error("production preflight", "status='passed'", preflight)
    receipt_binding = _preflight_receipt_from_environment(
        manifest=manifest,
        bundle_root=bundle_root,
        required=require_preflight_receipt,
        source_archive=source_archive,
        environment_contract=environment_contract,
        data_root=data_root,
        output_root=output_root,
        expected_kind="production",
    )
    canary_gate_binding = _canary_gate_receipt_from_environment(
        preflight_receipt=receipt_binding,
        required=True,
    )
    if receipt_binding is None or canary_gate_binding is None:
        raise AssertionError("production receipts are required")
    output = Path(output_root).expanduser().resolve()
    _output_roots_are_distinct(bundle_root, output)
    entry = manifest["entries"][entry_index]
    entry_dir = output / "entries" / entry["entry_id"]
    completion_path = entry_dir / "completion.json"
    with _entry_execution_lock(output, entry["entry_id"]):
        if completion_path.exists() or completion_path.is_symlink():
            completion = _validate_entry_completion(
                bundle_root=bundle_root,
                manifest=manifest,
                output_root=output,
                entry=entry,
            )
            return {
                "status": "already_complete",
                "entry_index": entry_index,
                "entry_id": entry["entry_id"],
                "completion_path": str(completion_path),
                "completion": completion,
            }
        attempt_parent = output / "attempts" / entry["entry_id"]
        attempt_parent.mkdir(parents=True, exist_ok=True)
        if entry_dir.exists() or entry_dir.is_symlink():
            if entry_dir.is_symlink() or not entry_dir.is_dir():
                raise _error(
                    "incomplete canonical entry output",
                    "a real directory recoverable as a prior attempt",
                    str(entry_dir),
                )
            recovered = Path(
                tempfile.mkdtemp(prefix="recovered-", dir=attempt_parent)
            )
            recovered.rmdir()
            os.replace(entry_dir, recovered)
        attempt_dir = Path(
            tempfile.mkdtemp(prefix="attempt-", dir=attempt_parent)
        )
        study = read_json(bundle_root / "study.resolved.json")
        payload = _training_payload(manifest, entry)
        executor(
            study=study,
            shard_dir=bundle_root,
            stage="successor_confirm",
            payload=payload,
            data_root=Path(data_root).expanduser().resolve(),
            device=device,
            download=download,
            output_dir=attempt_dir,
        )
        result = _enrich_successor_outputs(
            bundle_root=bundle_root,
            manifest=manifest,
            entry=entry,
            entry_dir=attempt_dir,
            preflight_receipt=receipt_binding,
            canary_gate_receipt=canary_gate_binding,
        )
        # Validate the complete payload before making any canonical entry
        # output visible.  A failed attempt remains isolated and recoverable.
        for name in PRODUCTION_OUTPUT_NAMES:
            path = attempt_dir / name
            if not path.is_file() or path.is_symlink():
                raise _error(
                    f"attempt output {name}",
                    "an existing non-symlink regular file",
                    str(path),
                )
        entry_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(attempt_dir, entry_dir)
        completion = {
            "schema_version": ENTRY_COMPLETION_SCHEMA_VERSION,
            "state": "complete",
            "official_test_read": False,
            "bundle_id": manifest["bundle_id"],
            "bundle_manifest_sha256": sha256_file(
                bundle_root / "manifest.json"
            ),
            "config_sha256": manifest["config_sha256"],
            "config_file_sha256": manifest["config_file_sha256"],
            "entry_index": entry_index,
            "entry_id": entry["entry_id"],
            "entry_payload_sha256": sha256_json(entry),
            "execution_source": copy.deepcopy(manifest["execution_source"]),
            "preflight_receipt": dict(receipt_binding),
            "canary_gate_receipt": dict(canary_gate_binding),
            "outputs": _output_records(output, entry_dir),
        }
        # The semantic completion marker is the final canonical write.
        atomic_write_json(completion_path, completion, canonical=True)
        _validate_entry_completion(
            bundle_root=bundle_root,
            manifest=manifest,
            output_root=output,
            entry=entry,
        )
    return {
        "status": "complete",
        "entry_index": entry_index,
        "entry_id": entry["entry_id"],
        "completed_steps": result["completed_steps"],
        "completion_path": str(completion_path),
        "completion_sha256": sha256_file(completion_path),
    }


def _validate_production_pack_receipt(
    *,
    bundle_root: Path,
    manifest: Mapping[str, Any],
    output_root: Path,
    pack_dir: Path,
    pack_index: int,
) -> dict[str, Any]:
    receipt = _mapping(
        read_json(pack_dir / "receipt.json"), "production pack receipt"
    )
    body = {key: value for key, value in receipt.items() if key != "pack_id"}
    entry_indices = fixed_pack_entries(pack_index)
    for key, expected in (
        ("schema_version", PRODUCTION_PACK_RECEIPT_SCHEMA_VERSION),
        ("status", "passed"),
        ("passed", True),
        ("official_test_read", False),
        ("bundle_id", manifest["bundle_id"]),
        (
            "bundle_manifest_sha256",
            sha256_file(bundle_root / "manifest.json"),
        ),
        ("config_sha256", manifest["config_sha256"]),
        ("config_file_sha256", manifest["config_file_sha256"]),
        ("pack_index", pack_index),
        ("entry_indices", list(entry_indices)),
        ("runs_per_gpu", RUNS_PER_GPU),
        ("concurrent", True),
    ):
        _require_equal(receipt.get(key), expected, f"production pack.{key}")
    _require_equal(
        receipt.get("pack_id"),
        "pdconfirmpack_" + sha256_json(body),
        "production pack.pack_id",
    )
    children = _list(
        receipt.get("children"),
        "production pack.children",
        length=RUNS_PER_GPU,
    )
    for offset, raw in enumerate(children):
        child = _mapping(raw, f"production pack.children[{offset}]")
        entry_index = entry_indices[offset]
        entry = manifest["entries"][entry_index]
        for key, expected in (
            ("entry_index", entry_index),
            ("entry_id", entry["entry_id"]),
            ("returncode", 0),
        ):
            _require_equal(
                child.get(key),
                expected,
                f"production pack.children[{offset}].{key}",
            )
        terminal = _mapping(
            child.get("terminal_result"),
            f"production pack.children[{offset}].terminal_result",
        )
        if terminal.get("status") not in {"complete", "already_complete"}:
            raise _error(
                f"production pack.children[{offset}].terminal_result.status",
                "'complete' or 'already_complete'",
                terminal.get("status"),
            )
        _require_equal(
            terminal.get("entry_index"),
            entry_index,
            f"production pack.children[{offset}].terminal_result.entry_index",
        )
        completion = output_root / "entries" / entry["entry_id"] / "completion.json"
        _require_equal(
            child.get("completion_artifact"),
            _artifact_record(output_root, completion),
            f"production pack.children[{offset}].completion_artifact",
        )
        for name in ("stdout_log", "stderr_log"):
            record = _mapping(
                child.get(name),
                f"production pack.children[{offset}].{name}",
            )
            relative = _portable_relative(
                record.get("path"),
                f"production pack.children[{offset}].{name}.path",
            )
            _require_equal(
                record,
                _artifact_record(pack_dir, pack_dir / relative),
                f"production pack.children[{offset}].{name}",
            )
        _validate_entry_completion(
            bundle_root=bundle_root,
            manifest=manifest,
            output_root=output_root,
            entry=entry,
        )
    expected_logs = sorted(
        (
            child[name]
            for child in children
            for name in ("stdout_log", "stderr_log")
        ),
        key=lambda record: record["path"],
    )
    _require_equal(
        receipt.get("output_artifacts"),
        expected_logs,
        "production pack.output_artifacts",
    )
    _validate_embedded_preflight_binding(
        receipt.get("preflight_receipt"),
        manifest=manifest,
        bundle_root=bundle_root,
        path="production pack.preflight_receipt",
    )
    _validate_embedded_canary_gate_binding(
        receipt.get("canary_gate_receipt"),
        preflight_receipt=receipt["preflight_receipt"],
        path="production pack.canary_gate_receipt",
    )
    return receipt


def run_pack(
    *,
    bundle_dir: str | Path,
    pack_index: int,
    output_root: str | Path,
    data_root: str | Path,
    source_archive: str | Path,
    environment_contract: str | Path,
    device: str = "cuda",
    download: bool = False,
    python_executable: str | Path | None = None,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    """Run one immutable two-entry pack as concurrent ordinary processes."""

    entry_indices = fixed_pack_entries(pack_index)
    if download:
        raise _error(
            "concurrent production-pack download",
            "false so paired children never race dataset writes",
            download,
        )
    bundle_root = Path(bundle_dir).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    archive = Path(source_archive).expanduser().resolve()
    environment = Path(environment_contract).expanduser().resolve()
    manifest = validate_input_bundle(bundle_root)
    validate_config(
        read_json(bundle_root / "config.json"),
        require_production_approval=True,
    )
    preflight = preflight_bundle(
        bundle_dir=bundle_root,
        source_archive=archive,
        environment_contract=environment,
    )
    if preflight["passed"] is not True:
        raise _error("production pack preflight", "status='passed'", preflight)
    preflight_binding = _preflight_receipt_from_environment(
        manifest=manifest,
        bundle_root=bundle_root,
        required=True,
        source_archive=archive,
        environment_contract=environment,
        data_root=data,
        output_root=output,
        expected_kind="production",
    )
    canary_binding = _canary_gate_receipt_from_environment(
        preflight_receipt=preflight_binding,
        required=True,
    )
    if preflight_binding is None or canary_binding is None:
        raise AssertionError("production pack receipts are required")
    _output_roots_are_distinct(bundle_root, output)
    slurm_identity = _slurm_pack_identity(pack_index)
    attempt_id = (
        f"{slurm_identity['array_job_id']}_"
        f"{slurm_identity['job_id']}_"
        f"{slurm_identity['array_task_id']}"
    )
    pack_dir = (
        output
        / "pack_runs"
        / manifest["bundle_id"]
        / "jobs"
        / attempt_id
        / f"pack_{pack_index}"
    )
    receipt_path = pack_dir / "receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _validate_production_pack_receipt(
            bundle_root=bundle_root,
            manifest=manifest,
            output_root=output,
            pack_dir=pack_dir,
            pack_index=pack_index,
        )
        return {
            "status": "already_passed",
            "pack_index": pack_index,
            "entry_indices": list(entry_indices),
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "receipt": receipt,
        }
    if pack_dir.exists() or pack_dir.is_symlink():
        raise _error(
            "production pack output",
            "a new isolated job/pack path or a verified passed receipt",
            str(pack_dir),
        )
    pack_dir.mkdir(parents=True)
    specs = []
    for entry_index in entry_indices:
        specs.append(
            {
                "label": f"entry_{entry_index:02d}",
                "entry_index": entry_index,
                "command": _runtime_child_command(
                    "run-entry",
                    bundle_dir=bundle_root,
                    output_root=output,
                    data_root=data,
                    source_archive=archive,
                    environment_contract=environment,
                    entry_index=entry_index,
                    device=device,
                    download=False,
                    python_executable=python_executable,
                ),
            }
        )
    children, wall_elapsed = _run_concurrent_children(
        specs,
        log_root=pack_dir / "child_logs",
        artifact_root=pack_dir,
        popen_factory=popen_factory,
    )
    enriched_children = []
    for offset, child in enumerate(children):
        entry_index = entry_indices[offset]
        entry = manifest["entries"][entry_index]
        terminal = child["terminal_result"]
        if (
            terminal.get("status") not in {"complete", "already_complete"}
            or terminal.get("entry_index") != entry_index
        ):
            raise _error(
                f"production pack child {entry_index} terminal result",
                "the exact successful entry identity",
                terminal,
            )
        completion = output / "entries" / entry["entry_id"] / "completion.json"
        _validate_entry_completion(
            bundle_root=bundle_root,
            manifest=manifest,
            output_root=output,
            entry=entry,
        )
        enriched_children.append(
            {
                **child,
                "entry_id": entry["entry_id"],
                "completion_artifact": _artifact_record(output, completion),
            }
        )
    body = {
        "schema_version": PRODUCTION_PACK_RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "passed": True,
        "official_test_read": False,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(bundle_root / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "pack_index": pack_index,
        "entry_indices": list(entry_indices),
        "runs_per_gpu": RUNS_PER_GPU,
        "concurrent": True,
        "slurm_identity": slurm_identity,
        "concurrent_wall_elapsed_seconds": wall_elapsed,
        "children": enriched_children,
        "preflight_receipt": dict(preflight_binding),
        "canary_gate_receipt": dict(canary_binding),
        "output_artifacts": sorted(
            (
                child[name]
                for child in enriched_children
                for name in ("stdout_log", "stderr_log")
            ),
            key=lambda record: record["path"],
        ),
    }
    receipt = {
        **body,
        "pack_id": "pdconfirmpack_" + sha256_json(body),
    }
    atomic_write_json(receipt_path, receipt, canonical=True)
    _validate_production_pack_receipt(
        bundle_root=bundle_root,
        manifest=manifest,
        output_root=output,
        pack_dir=pack_dir,
        pack_index=pack_index,
    )
    return {
        "status": "passed",
        "pack_index": pack_index,
        "entry_indices": list(entry_indices),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "concurrent_wall_elapsed_seconds": wall_elapsed,
    }


def output_status(
    *,
    bundle_dir: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Return side-effect-free, hash-verified status for all twelve entries."""

    bundle_root = Path(bundle_dir).expanduser().resolve()
    manifest = validate_input_bundle(bundle_root)
    output = Path(output_root).expanduser().resolve()
    _output_roots_are_distinct(bundle_root, output)
    entries = []
    errors = []
    completed = 0
    for entry in manifest["entries"]:
        entry_dir = output / "entries" / entry["entry_id"]
        completion_path = entry_dir / "completion.json"
        state = "pending"
        steps = 0
        if completion_path.exists() or completion_path.is_symlink():
            try:
                _validate_entry_completion(
                    bundle_root=bundle_root,
                    manifest=manifest,
                    output_root=output,
                    entry=entry,
                )
                result = _mapping(read_json(entry_dir / "result.json"), "result")
                state = "complete"
                steps = int(result.get("completed_steps", 0))
                completed += 1
            except Exception as exc:
                state = "invalid"
                errors.append(
                    {
                        "entry_index": entry["entry_index"],
                        "entry_id": entry["entry_id"],
                        "error": str(exc),
                    }
                )
        elif entry_dir.exists() or entry_dir.is_symlink():
            state = "invalid"
            errors.append(
                {
                    "entry_index": entry["entry_index"],
                    "entry_id": entry["entry_id"],
                    "error": "incomplete output exists without completion.json",
                }
            )
        entries.append(
            {
                "entry_index": entry["entry_index"],
                "entry_id": entry["entry_id"],
                "status": state,
                "mode": "successor_confirmation",
                "optimizer_steps_completed": steps,
                "completion_path": str(completion_path),
                "result_path": str(entry_dir / "result.json"),
                "checkpoint_paths": [
                    str(entry_dir / "best_validation.pt"),
                    str(entry_dir / "final.pt"),
                ],
            }
        )
    status = (
        "invalid"
        if errors
        else "complete"
        if completed == 12
        else "partial"
        if completed
        else "pending"
    )
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "status": status,
        "official_test_read": False,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(bundle_root / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "expected_entry_count": 12,
        "completed_entry_count": completed,
        "pending_entry_count": 12 - completed,
        "entries": entries,
        "errors": errors,
    }


def _validate_training_canary_outputs(
    *,
    smoke_root: Path,
    entry: Mapping[str, Any],
    output_dir: Path,
    require_admission_evidence: bool = False,
) -> dict[str, Any]:
    result = _mapping(read_json(output_dir / "result.json"), "training canary result")
    run_spec = _mapping(
        read_json(output_dir / "run_spec.json"), "training canary run spec"
    )
    validation = _mapping(
        read_json(output_dir / "validation.json"), "training canary validation"
    )
    for label, value in (
        ("training canary result", result),
        ("training canary run spec", run_spec),
        ("training canary validation", validation),
    ):
        require_official_test_excluded(value, require_result_marker=True)
        _require_official_test_false_recursive(value, label)
    for key, expected in (
        ("mode", "successor_canary"),
        ("status", "complete"),
        ("training_completed", True),
        ("completed_steps", 3_438),
        ("expected_steps", 3_438),
        ("epochs_completed", 1),
        ("expected_epochs", 1),
        ("entry_id", entry["entry_id"]),
        ("cell_id", entry["cell_id"]),
        ("row_id", entry["row_id"]),
        ("architecture", entry["architecture"]),
        ("scheme", entry["scheme"]),
        ("optimizer", entry["optimizer"]),
        ("rho_conv", entry["runtime_rho_conv"]),
        ("rho_dense", entry["runtime_rho_dense"]),
        (
            "raw_learning_rates_by_parameter",
            entry["raw_learning_rates_by_parameter"],
        ),
        ("official_test_read", False),
    ):
        _require_equal(result.get(key), expected, f"training canary result.{key}")
    training = _mapping(
        run_spec.get("training"), "training canary run_spec.training"
    )
    for key, expected in (
        ("mode", "successor_canary"),
        ("epochs", 1),
        ("steps_per_epoch", 3_438),
        ("total_steps", 3_438),
        ("restart_from_shared_initialization", True),
        ("continue_from_canary_or_candidate", False),
    ):
        _require_equal(
            training.get(key),
            expected,
            f"training canary run_spec.training.{key}",
        )
    records = _list(
        validation.get("records"),
        "training canary validation.records",
        length=1,
    )
    _require_equal(
        records[0].get("step"), 3_438, "training canary final validation step"
    )
    result_artifacts = _list(
        result.get("artifacts"), "training canary result.artifacts", length=5
    )
    expected_result_artifacts = {
        "run_spec.json",
        "validation.json",
        "step_log.csv",
        "best_validation.pt",
        "final.pt",
    }
    observed: set[str] = set()
    for index, raw in enumerate(result_artifacts):
        record = _mapping(
            raw, f"training canary result.artifacts[{index}]"
        )
        _require_exact_keys(
            record,
            {"path", "sha256", "bytes"},
            f"training canary result.artifacts[{index}]",
        )
        relative = _portable_relative(
            record.get("path"),
            f"training canary result.artifacts[{index}].path",
        )
        if relative not in expected_result_artifacts or relative in observed:
            raise _error(
                "training canary result.artifacts paths",
                f"each of {sorted(expected_result_artifacts)!r} exactly once",
                relative,
            )
        observed.add(relative)
        artifact = output_dir / relative
        _require_equal(
            record,
            {
                "path": relative,
                "sha256": sha256_file(artifact),
                "bytes": artifact.stat().st_size,
            },
            f"training canary result.artifacts[{index}]",
        )
    _require_equal(
        observed,
        expected_result_artifacts,
        "training canary result.artifacts paths",
    )
    benchmark_value = result.get("benchmark")
    benchmark: dict[str, Any] | None = None
    if benchmark_value is not None:
        benchmark = _mapping(
            benchmark_value, "training canary result.benchmark"
        )
        elapsed = _positive_number(
            benchmark.get("elapsed_seconds"),
            "training canary result.benchmark.elapsed_seconds",
        )
        steps_per_second = _positive_number(
            benchmark.get("successful_steps_per_second"),
            "training canary result.benchmark.successful_steps_per_second",
        )
        peak_allocated = benchmark.get(
            "cuda_peak_memory_allocated_bytes"
        )
        peak_reserved = benchmark.get("cuda_peak_memory_reserved_bytes")
        for key, raw in (
            ("cuda_peak_memory_allocated_bytes", peak_allocated),
            ("cuda_peak_memory_reserved_bytes", peak_reserved),
        ):
            if raw is not None and (
                type(raw) is not int or raw < 0
            ):
                raise _error(
                    f"training canary result.benchmark.{key}",
                    "null or a non-negative integer",
                    raw,
                )
        benchmark = {
            "elapsed_seconds": elapsed,
            "successful_steps_per_second": steps_per_second,
            "cuda_peak_memory_allocated_bytes": peak_allocated,
            "cuda_peak_memory_reserved_bytes": peak_reserved,
        }
    if require_admission_evidence and (
        benchmark is None
        or type(benchmark["cuda_peak_memory_allocated_bytes"]) is not int
        or benchmark["cuda_peak_memory_allocated_bytes"] <= 0
        or type(benchmark["cuda_peak_memory_reserved_bytes"]) is not int
        or benchmark["cuda_peak_memory_reserved_bytes"] <= 0
    ):
        raise _error(
            "training canary result.benchmark",
            "positive finite elapsed_seconds and successful_steps_per_second "
            "plus positive CUDA allocated and reserved peak bytes",
            benchmark_value,
        )
    output_artifacts = [
        _artifact_record(smoke_root, output_dir / name)
        for name in PRODUCTION_OUTPUT_NAMES
    ]
    return {
        "entry_index": entry["entry_index"],
        "entry_id": entry["entry_id"],
        "architecture": entry["architecture"],
        "scheme": entry["scheme"],
        "optimizer": entry["optimizer"],
        "mode": "successor_canary",
        "status": "complete",
        "training_completed": True,
        "completed_steps": 3_438,
        "expected_steps": 3_438,
        "epochs_completed": 1,
        "expected_epochs": 1,
        "official_test_read": False,
        "fresh_restart_from_shared_initialization": True,
        "raw_learning_rates_by_parameter": copy.deepcopy(
            entry["raw_learning_rates_by_parameter"]
        ),
        "benchmark": benchmark,
        "output_artifacts": sorted(
            output_artifacts, key=lambda record: record["path"]
        ),
    }


def _canary_smoke_dir(
    output_root: Path,
    manifest: Mapping[str, Any],
    slurm_identity: Mapping[str, str],
) -> Path:
    attempt_id = (
        f"{slurm_identity['array_job_id']}_"
        f"{slurm_identity['job_id']}_"
        f"{slurm_identity['array_task_id']}"
    )
    return (
        output_root
        / "smoke"
        / str(manifest["bundle_id"])
        / "jobs"
        / attempt_id
    )


def _validate_canary_child_receipt(
    *,
    bundle_root: Path,
    manifest: Mapping[str, Any],
    smoke_dir: Path,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    child_dir = smoke_dir / "training_canaries" / str(entry["entry_id"])
    receipt = _mapping(
        read_json(child_dir / "receipt.json"), "canary child receipt"
    )
    body = {key: value for key, value in receipt.items() if key != "child_id"}
    for key, expected in (
        ("schema_version", CANARY_CHILD_RECEIPT_SCHEMA_VERSION),
        ("status", "passed"),
        ("passed", True),
        ("official_test_read", False),
        ("bundle_id", manifest["bundle_id"]),
        (
            "bundle_manifest_sha256",
            sha256_file(bundle_root / "manifest.json"),
        ),
        ("config_sha256", manifest["config_sha256"]),
        ("config_file_sha256", manifest["config_file_sha256"]),
        ("entry_index", entry["entry_index"]),
        ("entry_id", entry["entry_id"]),
    ):
        _require_equal(receipt.get(key), expected, f"canary child.{key}")
    _require_equal(
        receipt.get("child_id"),
        "pdconfirmcanarychild_" + sha256_json(body),
        "canary child.child_id",
    )
    record = _mapping(
        receipt.get("training_canary"), "canary child.training_canary"
    )
    expected_record = _validate_training_canary_outputs(
        smoke_root=smoke_dir,
        entry=entry,
        output_dir=child_dir / "outputs",
        require_admission_evidence=True,
    )
    _require_equal(
        record, expected_record, "canary child.training_canary"
    )
    _require_equal(
        receipt.get("output_artifacts"),
        expected_record["output_artifacts"],
        "canary child.output_artifacts",
    )
    _validate_embedded_preflight_binding(
        receipt.get("preflight_receipt"),
        manifest=manifest,
        bundle_root=bundle_root,
        path="canary child.preflight_receipt",
    )
    return receipt


def run_canary_entry(
    *,
    bundle_dir: str | Path,
    entry_index: int,
    output_root: str | Path,
    data_root: str | Path,
    source_archive: str | Path,
    environment_contract: str | Path,
    device: str = "cuda",
    download: bool = False,
    stage_executor: Callable[..., Mapping[str, Any]] = execute_successor_stage_entry,
) -> dict[str, Any]:
    """Run one strict member of the concurrent pack-4 training canary."""

    if entry_index not in CANARY_ENTRY_INDICES or isinstance(entry_index, bool):
        raise _error(
            "canary entry_index",
            f"one of {list(CANARY_ENTRY_INDICES)!r}",
            entry_index,
        )
    if download:
        raise _error(
            "concurrent canary download",
            "false so paired children never race dataset writes",
            download,
        )
    bundle_root = Path(bundle_dir).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    archive = Path(source_archive).expanduser().resolve()
    environment = Path(environment_contract).expanduser().resolve()
    manifest = validate_input_bundle(bundle_root)
    validate_config(
        read_json(bundle_root / "config.json"),
        require_production_approval=True,
    )
    preflight = preflight_bundle(
        bundle_dir=bundle_root,
        source_archive=archive,
        environment_contract=environment,
    )
    if preflight["passed"] is not True:
        raise _error("canary child preflight", "status='passed'", preflight)
    preflight_binding = _preflight_receipt_from_environment(
        manifest=manifest,
        bundle_root=bundle_root,
        required=True,
        source_archive=archive,
        environment_contract=environment,
        data_root=data,
        output_root=output,
        expected_kind="canary",
    )
    if preflight_binding is None:
        raise AssertionError("canary child preflight receipt is required")
    _output_roots_are_distinct(bundle_root, output)
    slurm_identity = _slurm_canary_identity()
    smoke_dir = _canary_smoke_dir(output, manifest, slurm_identity)
    entry = manifest["entries"][entry_index]
    child_dir = smoke_dir / "training_canaries" / str(entry["entry_id"])
    receipt_path = child_dir / "receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _validate_canary_child_receipt(
            bundle_root=bundle_root,
            manifest=manifest,
            smoke_dir=smoke_dir,
            entry=entry,
        )
        record = receipt["training_canary"]
        return {
            "status": "already_passed",
            "entry_index": entry_index,
            "entry_id": entry["entry_id"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "completed_steps": record["completed_steps"],
            "benchmark": record["benchmark"],
        }
    if child_dir.exists() or child_dir.is_symlink():
        raise _error(
            "canary child output",
            "a new isolated entry path or a verified passed receipt",
            str(child_dir),
        )
    child_dir.mkdir(parents=True)
    output_dir = child_dir / "outputs"
    study = read_json(bundle_root / "study.resolved.json")
    stage_executor(
        study=study,
        shard_dir=bundle_root,
        stage="successor_canary",
        payload=_training_payload(manifest, entry),
        data_root=data,
        device=device,
        download=False,
        output_dir=output_dir,
    )
    record = _validate_training_canary_outputs(
        smoke_root=smoke_dir,
        entry=entry,
        output_dir=output_dir,
        require_admission_evidence=True,
    )
    body = {
        "schema_version": CANARY_CHILD_RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "passed": True,
        "official_test_read": False,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(bundle_root / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "entry_index": entry_index,
        "entry_id": entry["entry_id"],
        "slurm_identity": slurm_identity,
        "training_canary": record,
        "execution_source": copy.deepcopy(manifest["execution_source"]),
        "preflight_receipt": dict(preflight_binding),
        "output_artifacts": record["output_artifacts"],
    }
    receipt = {
        **body,
        "child_id": "pdconfirmcanarychild_" + sha256_json(body),
    }
    atomic_write_json(receipt_path, receipt, canonical=True)
    _validate_canary_child_receipt(
        bundle_root=bundle_root,
        manifest=manifest,
        smoke_dir=smoke_dir,
        entry=entry,
    )
    return {
        "status": "passed",
        "entry_index": entry_index,
        "entry_id": entry["entry_id"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "completed_steps": record["completed_steps"],
        "benchmark": record["benchmark"],
    }


def _smoke_artifacts(smoke_dir: Path) -> list[dict[str, Any]]:
    return [
        _artifact_record(smoke_dir, path)
        for path in sorted(
            (
                item
                for item in smoke_dir.rglob("*")
                if item.is_file() and item.name != "receipt.json"
            ),
            key=lambda item: item.relative_to(smoke_dir).as_posix(),
        )
    ]


def _slurm_identity_for_task(
    expected_task_id: int, *, label: str
) -> dict[str, str]:
    names = {
        "job_id": "SLURM_JOB_ID",
        "array_job_id": "SLURM_ARRAY_JOB_ID",
        "array_task_id": "SLURM_ARRAY_TASK_ID",
        "account": "SLURM_JOB_ACCOUNT",
        "partition": "SLURM_JOB_PARTITION",
        "qos": "SLURM_JOB_QOS",
        "constraint": "PD_SUCCESSOR_CONSTRAINT",
    }
    identity = {key: os.environ.get(name) for key, name in names.items()}
    invalid = {
        key: value
        for key, value in identity.items()
        if not isinstance(value, str)
        or not value
        or (
            key in {"job_id", "array_job_id", "array_task_id"}
            and not value.isdigit()
        )
    }
    if invalid:
        raise _error(
            f"live Slurm {label} identity",
            "decimal Slurm job/array/task ids and non-empty live "
            "account/partition/qos/constraint values",
            identity,
        )
    _require_equal(
        identity["array_task_id"],
        str(expected_task_id),
        f"live Slurm {label} array task id",
    )
    for key, expected in (
        ("account", "umg@v100"),
        ("partition", "gpu_p13"),
        ("qos", "qos_gpu-t3"),
        ("constraint", "v100-32g"),
    ):
        _require_equal(identity[key], expected, f"live Slurm {label} {key}")
    return {key: str(value) for key, value in identity.items()}


def _slurm_canary_identity() -> dict[str, str]:
    return _slurm_identity_for_task(0, label="canary")


def _slurm_pack_identity(pack_index: int) -> dict[str, str]:
    fixed_pack_entries(pack_index)
    return _slurm_identity_for_task(pack_index, label="production pack")


def _run_fresh_tk_security(
    *,
    bundle_root: Path,
    manifest: Mapping[str, Any],
    smoke_dir: Path,
    data_root: Path,
    device: str,
    download: bool,
    stage_executor: Callable[..., Mapping[str, Any]],
) -> list[dict[str, Any]]:
    study = read_json(bundle_root / "study.resolved.json")
    records = []
    for architecture in ARCHITECTURE_ORDER:
        expected_arch = EXPECTED_ARCHITECTURES[architecture]
        for scheme in SCHEME_ORDER:
            row_id = EXPECTED_ROWS[(architecture, scheme)][0]
            result_dir = smoke_dir / "tk_security" / row_id
            result = stage_executor(
                study=study,
                shard_dir=bundle_root,
                stage="fixed_tk_gradient_security",
                payload={
                    "entry_id": row_id,
                    "row_id": row_id,
                    "architecture": architecture,
                    "asset_entry_id": f"{architecture}-assets",
                    "source_commit": manifest["execution_source"][
                        "source_commit"
                    ],
                    "source_archive_sha256": manifest["execution_source"][
                        "source_archive_sha256"
                    ],
                    "environment_sha256": manifest["execution_source"][
                        "environment_contract_sha256"
                    ],
                    "official_test_read": False,
                },
                data_root=data_root,
                device=device,
                download=download,
                output_dir=result_dir,
            )
            if (
                result.get("status") != "complete"
                or result.get("security_passed") is not True
                or result.get("official_test_read") is not False
            ):
                raise _error(
                    f"fresh T/K security result for {row_id}",
                    "status='complete', security_passed=true, and "
                    "official_test_read=false",
                    result,
                )
            _require_equal(
                result.get("operational"),
                {"T": expected_arch["T"], "K": expected_arch["K"]},
                f"fresh T/K security operational identity for {row_id}",
            )
            _require_equal(
                result.get("reference"),
                {"T": 64, "K": 64},
                f"fresh T/K security reference identity for {row_id}",
            )
            records.append(
                {
                    "row_id": row_id,
                    "architecture": architecture,
                    "scheme": scheme,
                    "T": expected_arch["T"],
                    "K": expected_arch["K"],
                    "reference_T": 64,
                    "reference_K": 64,
                    "security_passed": True,
                    "fresh_replay": True,
                    "result_artifact": _artifact_record(
                        smoke_dir, result_dir / "result.json"
                    ),
                }
            )
    return records


def _paired_canary_admission(
    training_canaries: Sequence[Mapping[str, Any]],
    *,
    concurrent_wall_elapsed_seconds: float,
) -> dict[str, Any]:
    records = _list(
        list(training_canaries),
        "paired canary training records",
        length=RUNS_PER_GPU,
    )
    wall_elapsed = _positive_number(
        concurrent_wall_elapsed_seconds,
        "paired canary concurrent wall elapsed seconds",
    )
    child_evidence = []
    combined_allocated_peak = 0
    combined_reserved_peak = 0
    aggregate_steps = 0
    for offset, raw in enumerate(records):
        record = _mapping(raw, f"paired canary record[{offset}]")
        entry_index = CANARY_ENTRY_INDICES[offset]
        _require_equal(
            record.get("entry_index"),
            entry_index,
            f"paired canary record[{offset}].entry_index",
        )
        benchmark = _mapping(
            record.get("benchmark"),
            f"paired canary record[{offset}].benchmark",
        )
        elapsed = _positive_number(
            benchmark.get("elapsed_seconds"),
            f"paired canary record[{offset}].benchmark.elapsed_seconds",
        )
        steps_per_second = _positive_number(
            benchmark.get("successful_steps_per_second"),
            "paired canary record"
            f"[{offset}].benchmark.successful_steps_per_second",
        )
        peak = benchmark.get("cuda_peak_memory_allocated_bytes")
        reserved_peak = benchmark.get("cuda_peak_memory_reserved_bytes")
        if type(peak) is not int or peak <= 0:
            raise _error(
                "paired canary CUDA peak allocated memory",
                "a positive integer for each child",
                peak,
            )
        if type(reserved_peak) is not int or reserved_peak <= 0:
            raise _error(
                "paired canary CUDA peak reserved memory",
                "a positive integer for each child",
                reserved_peak,
            )
        completed_steps = record.get("completed_steps")
        if completed_steps != 3_438:
            raise _error(
                f"paired canary record[{offset}].completed_steps",
                "exactly 3438",
                completed_steps,
            )
        projected = (
            EXPECTED_ARCHITECTURES["conv2"]["expected_steps"]
            / steps_per_second
        )
        combined_allocated_peak += peak
        combined_reserved_peak += reserved_peak
        aggregate_steps += int(completed_steps)
        child_evidence.append(
            {
                "entry_index": entry_index,
                "entry_id": record.get("entry_id"),
                "completed_steps": completed_steps,
                "elapsed_seconds": elapsed,
                "successful_steps_per_second": steps_per_second,
                "cuda_peak_memory_allocated_bytes": peak,
                "cuda_peak_memory_reserved_bytes": reserved_peak,
                "projected_20_epoch_duration_seconds": projected,
            }
        )
    maximum_combined_peak = (
        PAIRED_CANARY_GPU_MEMORY_BYTES
        * PAIRED_CANARY_MAX_PEAK_NUMERATOR
        // PAIRED_CANARY_MAX_PEAK_DENOMINATOR
    )
    maximum_projected = max(
        record["projected_20_epoch_duration_seconds"]
        for record in child_evidence
    )
    checks = {
        "child_evidence_positive": True,
        "combined_peak_within_limit": (
            combined_reserved_peak * PAIRED_CANARY_MAX_PEAK_DENOMINATOR
            <= PAIRED_CANARY_GPU_MEMORY_BYTES
            * PAIRED_CANARY_MAX_PEAK_NUMERATOR
        ),
        "projected_duration_below_limit": (
            maximum_projected < PAIRED_CANARY_MAX_PROJECTED_SECONDS
        ),
    }
    evidence = {
        "thresholds": {
            "gpu_memory_capacity_bytes": PAIRED_CANARY_GPU_MEMORY_BYTES,
            "maximum_combined_peak_fraction": (
                PAIRED_CANARY_MAX_PEAK_NUMERATOR
                / PAIRED_CANARY_MAX_PEAK_DENOMINATOR
            ),
            "maximum_combined_peak_reserved_bytes": maximum_combined_peak,
            "maximum_projected_child_duration_seconds": (
                PAIRED_CANARY_MAX_PROJECTED_SECONDS
            ),
        },
        "children": child_evidence,
        "combined_peak_memory_allocated_bytes": combined_allocated_peak,
        "combined_peak_memory_reserved_bytes": combined_reserved_peak,
        "combined_peak_memory_fraction": (
            combined_reserved_peak / PAIRED_CANARY_GPU_MEMORY_BYTES
        ),
        "concurrent_training_wall_elapsed_seconds": wall_elapsed,
        "aggregate_completed_steps": aggregate_steps,
        "aggregate_successful_steps_per_second": (
            aggregate_steps / wall_elapsed
        ),
        "maximum_projected_20_epoch_duration_seconds": maximum_projected,
        "checks": checks,
        "passed": all(checks.values()),
    }
    if evidence["passed"] is not True:
        raise _error(
            "paired canary admission",
            "positive child evidence, combined peak at most 80% of 32 GiB, "
            "and projected max child duration below 28800 seconds",
            evidence,
        )
    return evidence


def _validate_smoke_pack_receipt(
    *,
    bundle_root: Path,
    manifest: Mapping[str, Any],
    smoke_dir: Path,
) -> dict[str, Any]:
    receipt = _mapping(
        read_json(smoke_dir / "receipt.json"), "smoke pack receipt"
    )
    expected_top_keys = {
        "schema_version",
        "status",
        "passed",
        "official_test_read",
        "bundle_id",
        "bundle_manifest_sha256",
        "config_sha256",
        "config_file_sha256",
        "expected_tk_security_count",
        "slurm_identity",
        "pack_index",
        "entry_indices",
        "runs_per_gpu",
        "concurrent",
        "tk_security",
        "training_canaries",
        "child_processes",
        "timing_memory_evidence",
        "execution_source",
        "preflight_receipt",
        "input_artifacts",
        "output_artifacts",
        "smoke_id",
    }
    _require_exact_keys(receipt, expected_top_keys, "smoke pack receipt")
    body = {key: value for key, value in receipt.items() if key != "smoke_id"}
    for key, expected in (
        ("schema_version", SMOKE_PACK_RECEIPT_SCHEMA_VERSION),
        ("status", "passed"),
        ("passed", True),
        ("official_test_read", False),
        ("bundle_id", manifest["bundle_id"]),
        (
            "bundle_manifest_sha256",
            sha256_file(bundle_root / "manifest.json"),
        ),
        ("config_sha256", manifest["config_sha256"]),
        ("config_file_sha256", manifest["config_file_sha256"]),
        ("expected_tk_security_count", 6),
        ("slurm_identity", _slurm_canary_identity()),
        ("pack_index", CANARY_PACK_INDEX),
        ("entry_indices", list(CANARY_ENTRY_INDICES)),
        ("runs_per_gpu", RUNS_PER_GPU),
        ("concurrent", True),
        ("execution_source", manifest["execution_source"]),
    ):
        _require_equal(receipt.get(key), expected, f"smoke pack.{key}")
    _require_equal(
        receipt.get("smoke_id"),
        "pdconfirmsmokepack_" + sha256_json(body),
        "smoke pack.smoke_id",
    )
    security = _list(
        receipt.get("tk_security"), "smoke pack.tk_security", length=6
    )
    for index, record in enumerate(security):
        _require_equal(
            record.get("security_passed"),
            True,
            f"smoke pack.tk_security[{index}].security_passed",
        )
        _require_equal(
            record.get("fresh_replay"),
            True,
            f"smoke pack.tk_security[{index}].fresh_replay",
        )
    training = _list(
        receipt.get("training_canaries"),
        "smoke pack.training_canaries",
        length=RUNS_PER_GPU,
    )
    children = _list(
        receipt.get("child_processes"),
        "smoke pack.child_processes",
        length=RUNS_PER_GPU,
    )
    for offset, entry_index in enumerate(CANARY_ENTRY_INDICES):
        entry = manifest["entries"][entry_index]
        child_receipt = _validate_canary_child_receipt(
            bundle_root=bundle_root,
            manifest=manifest,
            smoke_dir=smoke_dir,
            entry=entry,
        )
        _require_equal(
            training[offset],
            child_receipt["training_canary"],
            f"smoke pack.training_canaries[{offset}]",
        )
        child = _mapping(
            children[offset], f"smoke pack.child_processes[{offset}]"
        )
        _require_exact_keys(
            child,
            {
                "label",
                "entry_index",
                "entry_id",
                "command",
                "returncode",
                "process_elapsed_seconds",
                "terminal_result",
                "stdout_log",
                "stderr_log",
                "receipt_artifact",
            },
            f"smoke pack.child_processes[{offset}]",
        )
        for key, expected in (
            ("label", f"entry_{entry_index:02d}"),
            ("entry_index", entry_index),
            ("entry_id", entry["entry_id"]),
            ("returncode", 0),
        ):
            _require_equal(
                child.get(key),
                expected,
                f"smoke pack.child_processes[{offset}].{key}",
            )
        terminal = _mapping(
            child.get("terminal_result"),
            f"smoke pack.child_processes[{offset}].terminal_result",
        )
        _require_exact_keys(
            terminal,
            {
                "status",
                "entry_index",
                "entry_id",
                "receipt_path",
                "receipt_sha256",
                "completed_steps",
                "benchmark",
            },
            f"smoke pack.child_processes[{offset}].terminal_result",
        )
        if terminal["status"] not in {"passed", "already_passed"}:
            raise _error(
                f"smoke pack child {entry_index} status",
                "'passed' or 'already_passed'",
                terminal["status"],
            )
        child_path = (
            smoke_dir / "training_canaries" / entry["entry_id"] / "receipt.json"
        )
        for key, expected in (
            ("entry_index", entry_index),
            ("entry_id", entry["entry_id"]),
            ("receipt_path", str(child_path)),
            ("receipt_sha256", sha256_file(child_path)),
            ("completed_steps", 3_438),
            ("benchmark", training[offset]["benchmark"]),
        ):
            _require_equal(
                terminal.get(key),
                expected,
                f"smoke pack child {entry_index} terminal.{key}",
            )
        _require_equal(
            child.get("receipt_artifact"),
            _artifact_record(smoke_dir, child_path),
            f"smoke pack child {entry_index}.receipt_artifact",
        )
        for name in ("stdout_log", "stderr_log"):
            record = _mapping(
                child.get(name),
                f"smoke pack child {entry_index}.{name}",
            )
            relative = _portable_relative(
                record.get("path"),
                f"smoke pack child {entry_index}.{name}.path",
            )
            _require_equal(
                record,
                _artifact_record(smoke_dir, smoke_dir / relative),
                f"smoke pack child {entry_index}.{name}",
            )
    expected_evidence = _paired_canary_admission(
        training,
        concurrent_wall_elapsed_seconds=receipt[
            "timing_memory_evidence"
        ]["concurrent_training_wall_elapsed_seconds"],
    )
    _require_equal(
        receipt.get("timing_memory_evidence"),
        expected_evidence,
        "smoke pack.timing_memory_evidence",
    )
    _validate_embedded_preflight_binding(
        receipt.get("preflight_receipt"),
        manifest=manifest,
        bundle_root=bundle_root,
        path="smoke pack.preflight_receipt",
    )
    _require_equal(
        receipt.get("output_artifacts"),
        _smoke_artifacts(smoke_dir),
        "smoke pack.output_artifacts",
    )
    return receipt


def run_smoke_pack(
    *,
    bundle_dir: str | Path,
    output_root: str | Path,
    data_root: str | Path,
    source_archive: str | Path,
    environment_contract: str | Path,
    pack_index: int = CANARY_PACK_INDEX,
    device: str = "cuda",
    download: bool = False,
    python_executable: str | Path | None = None,
    stage_executor: Callable[..., Mapping[str, Any]] = execute_successor_stage_entry,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    """Run six T/K gates once, then concurrent one-epoch entries 8 and 9."""

    if pack_index != CANARY_PACK_INDEX or isinstance(pack_index, bool):
        raise _error(
            "smoke pack_index",
            f"exactly {CANARY_PACK_INDEX}",
            pack_index,
        )
    if download:
        raise _error(
            "concurrent smoke-pack download",
            "false so paired children never race dataset writes",
            download,
        )
    bundle_root = Path(bundle_dir).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    archive = Path(source_archive).expanduser().resolve()
    environment = Path(environment_contract).expanduser().resolve()
    manifest = validate_input_bundle(bundle_root)
    validate_config(
        read_json(bundle_root / "config.json"),
        require_production_approval=True,
    )
    preflight = preflight_bundle(
        bundle_dir=bundle_root,
        source_archive=archive,
        environment_contract=environment,
    )
    if preflight["passed"] is not True:
        raise _error("smoke-pack preflight", "status='passed'", preflight)
    preflight_binding = _preflight_receipt_from_environment(
        manifest=manifest,
        bundle_root=bundle_root,
        required=True,
        source_archive=archive,
        environment_contract=environment,
        data_root=data,
        output_root=output,
        expected_kind="canary",
    )
    if preflight_binding is None:
        raise AssertionError("smoke-pack preflight receipt is required")
    _output_roots_are_distinct(bundle_root, output)
    slurm_identity = _slurm_canary_identity()
    smoke_dir = _canary_smoke_dir(output, manifest, slurm_identity)
    receipt_path = smoke_dir / "receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _validate_smoke_pack_receipt(
            bundle_root=bundle_root,
            manifest=manifest,
            smoke_dir=smoke_dir,
        )
        return {
            "status": "already_passed",
            "pack_index": CANARY_PACK_INDEX,
            "entry_indices": list(CANARY_ENTRY_INDICES),
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "completed_steps": 2 * 3_438,
            "timing_memory_evidence": receipt["timing_memory_evidence"],
        }
    if smoke_dir.exists() or smoke_dir.is_symlink():
        raise _error(
            "smoke-pack output",
            "a new isolated path or a verified aggregate receipt",
            str(smoke_dir),
        )
    smoke_dir.mkdir(parents=True)
    security_records = _run_fresh_tk_security(
        bundle_root=bundle_root,
        manifest=manifest,
        smoke_dir=smoke_dir,
        data_root=data,
        device=device,
        download=False,
        stage_executor=stage_executor,
    )
    specs = [
        {
            "label": f"entry_{entry_index:02d}",
            "entry_index": entry_index,
            "command": _runtime_child_command(
                "run-canary-entry",
                bundle_dir=bundle_root,
                output_root=output,
                data_root=data,
                source_archive=archive,
                environment_contract=environment,
                entry_index=entry_index,
                device=device,
                download=False,
                python_executable=python_executable,
            ),
        }
        for entry_index in CANARY_ENTRY_INDICES
    ]
    child_processes, wall_elapsed = _run_concurrent_children(
        specs,
        log_root=smoke_dir / "child_logs",
        artifact_root=smoke_dir,
        popen_factory=popen_factory,
    )
    training_canaries = []
    enriched_children = []
    for offset, entry_index in enumerate(CANARY_ENTRY_INDICES):
        entry = manifest["entries"][entry_index]
        child_path = (
            smoke_dir / "training_canaries" / entry["entry_id"] / "receipt.json"
        )
        child_receipt = _validate_canary_child_receipt(
            bundle_root=bundle_root,
            manifest=manifest,
            smoke_dir=smoke_dir,
            entry=entry,
        )
        terminal = child_processes[offset]["terminal_result"]
        if (
            terminal.get("status") not in {"passed", "already_passed"}
            or terminal.get("entry_index") != entry_index
            or terminal.get("entry_id") != entry["entry_id"]
            or terminal.get("receipt_path") != str(child_path)
            or terminal.get("receipt_sha256") != sha256_file(child_path)
        ):
            raise _error(
                f"smoke-pack child {entry_index} terminal result",
                "the exact successful child receipt identity",
                terminal,
            )
        training_canaries.append(child_receipt["training_canary"])
        enriched_children.append(
            {
                **child_processes[offset],
                "entry_id": entry["entry_id"],
                "receipt_artifact": _artifact_record(
                    smoke_dir, child_path
                ),
            }
        )
    admission = _paired_canary_admission(
        training_canaries,
        concurrent_wall_elapsed_seconds=wall_elapsed,
    )
    body = {
        "schema_version": SMOKE_PACK_RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "passed": True,
        "official_test_read": False,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(bundle_root / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "expected_tk_security_count": 6,
        "slurm_identity": slurm_identity,
        "pack_index": CANARY_PACK_INDEX,
        "entry_indices": list(CANARY_ENTRY_INDICES),
        "runs_per_gpu": RUNS_PER_GPU,
        "concurrent": True,
        "tk_security": security_records,
        "training_canaries": training_canaries,
        "child_processes": enriched_children,
        "timing_memory_evidence": admission,
        "execution_source": copy.deepcopy(manifest["execution_source"]),
        "preflight_receipt": dict(preflight_binding),
        "input_artifacts": [
            _artifact_record(bundle_root, bundle_root / "manifest.json"),
            _artifact_record(bundle_root, bundle_root / "config.json"),
        ],
        "output_artifacts": _smoke_artifacts(smoke_dir),
    }
    receipt = {
        **body,
        "smoke_id": "pdconfirmsmokepack_" + sha256_json(body),
    }
    atomic_write_json(receipt_path, receipt, canonical=True)
    _validate_smoke_pack_receipt(
        bundle_root=bundle_root,
        manifest=manifest,
        smoke_dir=smoke_dir,
    )
    return {
        "status": "passed",
        "pack_index": CANARY_PACK_INDEX,
        "entry_indices": list(CANARY_ENTRY_INDICES),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "completed_steps": 2 * 3_438,
        "timing_memory_evidence": admission,
    }


def _validate_smoke_receipt(
    *,
    bundle_root: Path,
    manifest: Mapping[str, Any],
    smoke_dir: Path,
) -> dict[str, Any]:
    receipt = _mapping(read_json(smoke_dir / "receipt.json"), "smoke receipt")
    body = {key: value for key, value in receipt.items() if key != "smoke_id"}
    exact = {
        "schema_version": SMOKE_RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "passed": True,
        "official_test_read": False,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(bundle_root / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "expected_tk_security_count": 6,
        "slurm_identity": _slurm_canary_identity(),
    }
    for key, expected in exact.items():
        _require_equal(receipt.get(key), expected, f"smoke receipt.{key}")
    _require_equal(
        receipt.get("smoke_id"),
        "pdconfirmsmoke_" + sha256_json(body),
        "smoke receipt.smoke_id",
    )
    security = _list(
        receipt.get("tk_security"), "smoke receipt.tk_security", length=6
    )
    for index, record in enumerate(security):
        _require_equal(
            record.get("security_passed"),
            True,
            f"smoke receipt.tk_security[{index}].security_passed",
        )
        _require_equal(
            record.get("fresh_replay"),
            True,
            f"smoke receipt.tk_security[{index}].fresh_replay",
        )
    canary = _mapping(
        receipt.get("training_canary"), "smoke receipt.training_canary"
    )
    for key, expected in (
        ("entry_index", 9),
        ("mode", "successor_canary"),
        ("status", "complete"),
        ("training_completed", True),
        ("completed_steps", 3_438),
        ("expected_steps", 3_438),
        ("epochs_completed", 1),
        ("expected_epochs", 1),
        ("official_test_read", False),
        ("fresh_restart_from_shared_initialization", True),
    ):
        _require_equal(
            canary.get(key), expected, f"smoke receipt.training_canary.{key}"
        )
    _validate_embedded_preflight_binding(
        receipt.get("preflight_receipt"),
        manifest=manifest,
        bundle_root=bundle_root,
        path="smoke receipt.preflight_receipt",
    )
    _require_equal(
        receipt.get("output_artifacts"),
        _smoke_artifacts(smoke_dir),
        "smoke receipt.output_artifacts",
    )
    return receipt


def run_smoke(
    *,
    bundle_dir: str | Path,
    output_root: str | Path,
    data_root: str | Path,
    source_archive: str | Path,
    environment_contract: str | Path,
    entry_index: int = 9,
    device: str = "cuda",
    download: bool = False,
    stage_executor: Callable[..., Mapping[str, Any]] = execute_successor_stage_entry,
) -> dict[str, Any]:
    """Run six fresh T/K gates plus a full one-epoch production-path canary."""

    if entry_index != 9:
        raise _error(
            "smoke entry_index",
            "exactly 9 (Conv2 ours Adam, the selected worst-case arm)",
            entry_index,
        )
    bundle_root = Path(bundle_dir).expanduser().resolve()
    manifest = validate_input_bundle(bundle_root)
    validate_config(
        read_json(bundle_root / "config.json"),
        require_production_approval=True,
    )
    preflight = preflight_bundle(
        bundle_dir=bundle_root,
        source_archive=source_archive,
        environment_contract=environment_contract,
    )
    if preflight["passed"] is not True:
        raise _error("smoke preflight", "status='passed'", preflight)
    preflight_binding = _preflight_receipt_from_environment(
        manifest=manifest,
        bundle_root=bundle_root,
        required=True,
        source_archive=source_archive,
        environment_contract=environment_contract,
        data_root=data_root,
        output_root=output_root,
        expected_kind="canary",
    )
    output = Path(output_root).expanduser().resolve()
    _output_roots_are_distinct(bundle_root, output)
    slurm_canary = _slurm_canary_identity()
    attempt_id = (
        f"{slurm_canary['array_job_id']}_"
        f"{slurm_canary['job_id']}_"
        f"{slurm_canary['array_task_id']}"
    )
    smoke_dir = (
        output
        / "smoke"
        / manifest["bundle_id"]
        / "jobs"
        / attempt_id
    )
    receipt_path = smoke_dir / "receipt.json"
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt = _validate_smoke_receipt(
            bundle_root=bundle_root,
            manifest=manifest,
            smoke_dir=smoke_dir,
        )
        return {
            "status": "already_passed",
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "receipt": receipt,
        }
    if smoke_dir.exists() or smoke_dir.is_symlink():
        raise _error(
            "smoke output",
            "a new isolated path, or a hash-verified passed receipt",
            str(smoke_dir),
        )
    smoke_dir.mkdir(parents=True)
    study = read_json(bundle_root / "study.resolved.json")
    security_records = []
    for architecture in ARCHITECTURE_ORDER:
        expected_arch = EXPECTED_ARCHITECTURES[architecture]
        for scheme in SCHEME_ORDER:
            row_id = EXPECTED_ROWS[(architecture, scheme)][0]
            result_dir = smoke_dir / "tk_security" / row_id
            result = stage_executor(
                study=study,
                shard_dir=bundle_root,
                stage="fixed_tk_gradient_security",
                payload={
                    "entry_id": row_id,
                    "row_id": row_id,
                    "architecture": architecture,
                    "asset_entry_id": f"{architecture}-assets",
                    "source_commit": manifest["execution_source"][
                        "source_commit"
                    ],
                    "source_archive_sha256": manifest["execution_source"][
                        "source_archive_sha256"
                    ],
                    "environment_sha256": manifest["execution_source"][
                        "environment_contract_sha256"
                    ],
                    "official_test_read": False,
                },
                data_root=Path(data_root).expanduser().resolve(),
                device=device,
                download=download,
                output_dir=result_dir,
            )
            if (
                result.get("status") != "complete"
                or result.get("security_passed") is not True
                or result.get("official_test_read") is not False
            ):
                raise _error(
                    f"fresh T/K security result for {row_id}",
                    "status='complete', security_passed=true, and "
                    "official_test_read=false",
                    result,
                )
            _require_equal(
                result.get("operational"),
                {"T": expected_arch["T"], "K": expected_arch["K"]},
                f"fresh T/K security operational identity for {row_id}",
            )
            _require_equal(
                result.get("reference"),
                {"T": 64, "K": 64},
                f"fresh T/K security reference identity for {row_id}",
            )
            security_records.append(
                {
                    "row_id": row_id,
                    "architecture": architecture,
                    "scheme": scheme,
                    "T": expected_arch["T"],
                    "K": expected_arch["K"],
                    "reference_T": 64,
                    "reference_K": 64,
                    "security_passed": True,
                    "fresh_replay": True,
                    "result_artifact": _artifact_record(
                        smoke_dir, result_dir / "result.json"
                    ),
                }
            )
    smoke_entry = manifest["entries"][entry_index]
    canary_dir = smoke_dir / "training_canary"
    stage_executor(
        study=study,
        shard_dir=bundle_root,
        stage="successor_canary",
        payload=_training_payload(manifest, smoke_entry),
        data_root=Path(data_root).expanduser().resolve(),
        device=device,
        download=download,
        output_dir=canary_dir,
    )
    canary_record = _validate_training_canary_outputs(
        smoke_root=smoke_dir,
        entry=smoke_entry,
        output_dir=canary_dir,
    )
    body = {
        "schema_version": SMOKE_RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "passed": True,
        "official_test_read": False,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(bundle_root / "manifest.json"),
        "config_sha256": manifest["config_sha256"],
        "config_file_sha256": manifest["config_file_sha256"],
        "expected_tk_security_count": 6,
        "slurm_identity": slurm_canary,
        "tk_security": security_records,
        "training_canary": canary_record,
        "execution_source": copy.deepcopy(manifest["execution_source"]),
        "preflight_receipt": (
            None if preflight_binding is None else dict(preflight_binding)
        ),
        "input_artifacts": [
            _artifact_record(bundle_root, bundle_root / "manifest.json"),
            _artifact_record(bundle_root, bundle_root / "config.json"),
        ],
        "output_artifacts": _smoke_artifacts(smoke_dir),
    }
    receipt = {
        **body,
        "smoke_id": "pdconfirmsmoke_" + sha256_json(body),
    }
    atomic_write_json(receipt_path, receipt, canonical=True)
    _validate_smoke_receipt(
        bundle_root=bundle_root,
        manifest=manifest,
        smoke_dir=smoke_dir,
    )
    return {
        "status": "passed",
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "fresh_tk_security_count": 6,
        "training_canary_steps": 3_438,
    }


__all__ = [
    "BUNDLE_MANIFEST_SCHEMA_VERSION",
    "CANARY_ENTRY_INDICES",
    "CANARY_PACK_INDEX",
    "CONFIG_SCHEMA_VERSION",
    "ENTRY_COMPLETION_SCHEMA_VERSION",
    "FIXED_ENTRY_PACKS",
    "PREFLIGHT_SCHEMA_VERSION",
    "PRODUCTION_PACK_COUNT",
    "RESULT_JSON_MARKER",
    "RUNS_PER_GPU",
    "SMOKE_RECEIPT_SCHEMA_VERSION",
    "SMOKE_PACK_RECEIPT_SCHEMA_VERSION",
    "STATUS_SCHEMA_VERSION",
    "PerfectDiodeSuccessorError",
    "build_input_bundle",
    "execute_successor_stage_entry",
    "load_config",
    "output_status",
    "plan_config",
    "preflight_bundle",
    "run_canary_entry",
    "run_indexed_entry",
    "run_pack",
    "run_smoke",
    "run_smoke_pack",
    "validate_config",
    "validate_input_bundle",
]
