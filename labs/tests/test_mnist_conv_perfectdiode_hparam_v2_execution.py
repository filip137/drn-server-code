from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from experiments.mnist_conv.identity import sha256_file
from experiments.mnist_conv.perfectdiode_conv3_hparam_v2_spec import (
    FROZEN_TK_CONFIG,
    PUBLIC_STAGE_SEQUENCE,
    load_frozen_template,
)
from experiments.mnist_conv.perfectdiode_hparam_v2_execution import (
    EPOCH_K_VIABILITY_SCHEMA_VERSION,
    ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
    POST_TK_AUDIT_SCHEMA_VERSION,
    ExecutionContext,
    LoadedV2Assets,
    PerfectDiodeConv3ExecutionError,
    V1NumericalBackend,
    _contract_parameter_tensor_digest,
    _validate_epoch_k_viability_evidence,
    _validate_surface,
    build_environment_contract,
    execute_fixed_operating_point_preflight_gate,
    execute_representative_preflight_canary,
    execute_surface_stage,
    observe_execution_environment,
    run_builtin_post_training_tk_audit,
    validate_environment_contract,
    validate_environment_receipt,
    verify_current_execution_environment,
)
from experiments.mnist_conv.perfectdiode_tk_runtime import (
    TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION,
    LoadedTKAssets,
    parameters_in_contract_order,
)
from experiments.mnist_conv.lr_engine import (
    build_model_runtime,
    parameter_tensor_digest,
)
from experiments.mnist_conv.perfectdiode_tk_spec import (
    PerfectDiodeTKStudySpec,
)
from experiments.run_mnist_conv_perfectdiode_hparam_v2_worker import (
    _parser as _worker_parser,
    main as _worker_main,
)


class _Spec:
    def __init__(self, row: Mapping[str, Any]) -> None:
        self.study_id = "lrstudy_" + "1" * 64
        self.config_sha256 = "2" * 64
        self._row = copy.deepcopy(dict(row))
        self._data = {
            "upstream_tk": {
                "study_id": "tkstudy_" + "5" * 64,
                "config_sha256": "6" * 64,
                "shared_assets_artifact": "upstream_tk/shared_assets",
            },
            "candidate_training": {
                "minimum_final_validation_accuracy": 0.90,
            },
            "selection": {
                "plateau_relative_to_minimum": 0.02,
                "selected_status": (
                    "selected_seed0_ordinary_mnist_three_epoch_screen"
                ),
            },
        }

    @property
    def rows(self) -> tuple[dict[str, Any], ...]:
        return (copy.deepcopy(self._row),)

    @property
    def data(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def _row(
    *,
    scheme: str = "baseline",
    eligible: bool = True,
    host: str = "main",
) -> dict[str, Any]:
    row_id = f"conv3_{scheme}"
    return {
        "row_id": row_id,
        "architecture": "conv3",
        "scheme": scheme,
        "inference_iterations": 16,
        "training_iterations": 32,
        "input_gain": 360.0,
        "lr_eligible": eligible,
        "zero_work_reason": None if eligible else "upstream_tk_unresolved_k",
        "upstream_tk": {
            "execution_host": host,
            "t_extension_used": False,
            "k128_sentinel_used": False,
        },
    }


def _fixed_policy() -> dict[str, Any]:
    return {
        "core_mode": "fixed_high_3x3",
        "rho_conv": [0.009, 0.027, 0.081],
        "rho_dense": [0.03, 0.09, 0.27],
        "automatic_lower_center_search": False,
        "core_cells": [
            {"rho_conv": rho_conv, "rho_dense": rho_dense}
            for rho_conv in (0.009, 0.027, 0.081)
            for rho_dense in (0.03, 0.09, 0.27)
        ],
    }


def _legacy_policy() -> dict[str, Any]:
    return {
        "core_mode": "adaptive_safe_center_factor_three_3x3",
        "initial_rho_conv": 0.003,
        "initial_rho_dense": 0.01,
        "failure_scale_divisor": 3.0,
        "maximum_center_attempts": 6,
        "first_safe_center_wins": True,
    }


def _context(
    tmp_path: Path,
    *,
    scheme: str = "baseline",
    eligible: bool = True,
    optimizer: str = "sgd",
    host: str = "main",
) -> ExecutionContext:
    row = _row(scheme=scheme, eligible=eligible, host=host)
    spec = _Spec(row)
    surface_id = f"{row['row_id']}--{optimizer}"
    shared = {
        "train_indices_sha256": "3" * 64,
        "validation_indices_sha256": "4" * 64,
        "initialization_checkpoint_sha256": "5" * 64,
        "initialization_tensor_sha256": "6" * 64,
        "t_cohort_indices_sha256": "7" * 64,
        "k_cohort_indices_sha256": "8" * 64,
        "train_batch_order_epoch_1_sha256": "9" * 64,
        "train_batch_order_epoch_2_sha256": "a" * 64,
        "train_batch_order_epoch_3_sha256": "b" * 64,
    }
    source = {
        "source_commit": "c" * 40,
        "source_archive_sha256": "d" * 64,
        "effective_code_fingerprint": "e" * 64,
        "worker_launcher_sha256": "f" * 64,
        "environment_contract_sha256": "0" * 64,
        "host_environment_sha256s": {
            "main": "1" * 64,
            "akibscomputer": "2" * 64,
        },
        "resolved_study_path": "study.resolved.json",
        "resolved_study_sha256": "3" * 64,
        "output_root": "surfaces",
    }
    surface = {
        "surface_index": 0,
        "surface_id": surface_id,
        "row_id": row["row_id"],
        "architecture": "conv3",
        "scheme": scheme,
        "optimizer": optimizer,
        "host": host,
        "lr_eligible": eligible,
        "zero_work": not eligible,
        "zero_work_reason": row["zero_work_reason"],
        "inference_iterations": 16 if eligible else None,
        "training_iterations": 32 if eligible else None,
        "input_gain": 360.0,
        "upstream_tk": copy.deepcopy(row["upstream_tk"]),
        "shared_asset_hashes": shared,
        "execution_source": source,
        "execution_environment_sha256": (
            "1" * 64 if host == "main" else "2" * 64
        ),
        "output_path": f"surfaces/{surface_id}",
        "completion_path": f"surfaces/{surface_id}/completion.json",
        "finalization_path": (
            f"surfaces/{surface_id}/stages/finalize_lr/result.json"
        ),
        "rho_policy": (
            _legacy_policy()
            if scheme == "legacy" and eligible
            else _fixed_policy()
            if eligible
            else None
        ),
        "canary_steps_per_cell": 640,
        "candidate_total_steps": 10_314,
        "maximum_cells": 16,
        "maximum_expansion_waves": 1,
        "long_confirmation": False,
    }
    manifest = {
        "schema_version": (
            "mnist-conv-perfectdiode-hparam-surface-manifest/v2"
        ),
        "study_id": spec.study_id,
        "config_sha256": spec.config_sha256,
        "manifest_id": "pdlrmanifest_" + "4" * 64,
        "execution_source": source,
        "shared_asset_hashes": shared,
        "surfaces": [surface],
        "public_stage_sequence": list(PUBLIC_STAGE_SEQUENCE),
        "official_test_read": False,
    }
    root = tmp_path / "study"
    return ExecutionContext(
        spec=spec,  # type: ignore[arg-type]
        manifest=manifest,
        manifest_path=root / "surface_manifest.json",
        surface=surface,
        row=row,
        surface_root=root / "surfaces" / surface_id,
        host=host,
        observed_authority={
            "source_commit": "c" * 40,
            "source_archive_sha256": "d" * 64,
            "effective_code_fingerprint": "e" * 64,
            "environment_contract_sha256": "0" * 64,
            "execution_environment_sha256": (
                "1" * 64 if host == "main" else "2" * 64
            ),
        },
    )


def test_contract_parameter_digest_accepts_model_native_conv3_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    native_names = (
        "ConvWeight_0",
        "ConvWeight_1",
        "ConvWeight_2",
        "DenseWeight_0",
        "Bias_0",
        "Bias_1",
        "Bias_2",
    )
    parameters = tuple(
        SimpleNamespace(
            name=name,
            state=torch.tensor([float(index)], dtype=torch.float32),
        )
        for index, name in enumerate(native_names)
    )
    expected = parameter_tensor_digest(
        parameters_in_contract_order(parameters)
    )
    assert parameter_tensor_digest(parameters) != expected
    assert _contract_parameter_tensor_digest(parameters) == expected

    runtime = SimpleNamespace(parameters=parameters)
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "build_model_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "_optimizer_contract",
        lambda *args, **kwargs: {},
    )
    assets = SimpleNamespace(
        checkpoint_path=tmp_path / "initialization.pt",
        parameter_tensor_sha256=expected,
    )
    observed = V1NumericalBackend()._fresh_runtime(
        _context(tmp_path),
        assets,  # type: ignore[arg-type]
        learning_rate=1.0,
        device="cpu",
    )
    assert observed is runtime


def test_real_conv3_checkpoint_reload_uses_contract_parameter_order_at_t8_k8(
    tmp_path: Path,
) -> None:
    tk_spec = PerfectDiodeTKStudySpec.from_path(FROZEN_TK_CONFIG)
    row = copy.deepcopy(tk_spec.rows[0])
    row.update(
        {
            "inference_iterations": 8,
            "training_iterations": 8,
            "reference_inference_iterations": 64,
            "reference_training_iterations": 64,
        }
    )

    initialization = build_model_runtime(
        tk_spec.data,
        row,
        device="cpu",
        learning_rate=1.0,
    )
    checkpoint = tmp_path / "initialization.pt"
    initialization.save(checkpoint)
    expected = _contract_parameter_tensor_digest(initialization.parameters)
    assert parameter_tensor_digest(initialization.parameters) != expected

    template, template_path = load_frozen_template()
    base_path = template_path.parent / template["base_contract"]["path"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    study = copy.deepcopy(tk_spec.data)
    study["optimizer_arms"] = copy.deepcopy(base["optimizer_arms"])
    context = replace(
        _context(tmp_path, optimizer="adam"),
        spec=SimpleNamespace(data=study),  # type: ignore[arg-type]
        row=row,
    )
    assets = SimpleNamespace(
        checkpoint_path=checkpoint,
        parameter_tensor_sha256=expected,
    )

    reloaded = V1NumericalBackend()._fresh_runtime(
        context,
        assets,  # type: ignore[arg-type]
        learning_rate=1.0,
        device="cpu",
    )

    assert context.row["inference_iterations"] == 8
    assert context.row["training_iterations"] == 8
    assert parameter_tensor_digest(reloaded.parameters) != expected
    assert _contract_parameter_tensor_digest(reloaded.parameters) == expected


def _asset_dir(context: ExecutionContext) -> Path:
    return context.manifest_path.parent / "upstream_tk" / "shared_assets"


def _environment_receipt(
    host: str,
    *,
    torch_version: str,
) -> dict[str, Any]:
    delegated = host == "akibscomputer"
    return {
        "schema_version": ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
        "execution_backend": "tmux",
        "execution_host": host,
        "tmux": {
            "attestation_mode": (
                "delegated_remote_tmux_session"
                if delegated
                else "direct_tmux_session"
            ),
            "session_name": host,
            "pane_token": "%7" if delegated else "%3",
        },
        "python": {
            "implementation": "CPython",
            "version": "3.12.9 (test build)",
            "executable": (
                "/home/filiposana/miniconda3/envs/py312/bin/python"
                if delegated
                else "/home/filip/miniconda3/envs/py312/bin/python"
            ),
        },
        "torch": {
            "version": torch_version,
            "cuda_build_version": "12.1",
            "cudnn_version": 90100,
        },
        "cuda": {
            "available": True,
            "visible_devices": "0",
            "device_count": 1,
            "devices": [
                {
                    "index": 0,
                    "name": "Test GPU",
                    "compute_capability": [8, 6],
                    "total_memory_bytes": 24_000_000_000,
                }
            ],
        },
        "official_test_read": False,
    }


def test_environment_contract_binds_distinct_exact_host_runtimes() -> None:
    main_receipt = _environment_receipt(
        "main", torch_version="2.5.1+cu121"
    )
    akib_receipt = _environment_receipt(
        "akibscomputer", torch_version="2.5.1"
    )

    contract = build_environment_contract(main_receipt, akib_receipt)

    assert contract["host_runtime_policy"] == {
        "mode": "host_specific_exact_receipts",
        "cross_host_software_equality_required": False,
        "same_surface_stays_on_bound_host": True,
    }
    assert contract["host_receipts"]["main"] == main_receipt
    assert contract["host_receipts"]["akibscomputer"] == akib_receipt
    assert validate_environment_contract(contract) == contract


def test_environment_receipt_requires_direct_main_and_delegated_akib() -> None:
    main_receipt = _environment_receipt(
        "main", torch_version="2.5.1+cu121"
    )
    akib_receipt = _environment_receipt(
        "akibscomputer", torch_version="2.5.1"
    )
    assert validate_environment_receipt(
        main_receipt, expected_host="main"
    ) == main_receipt
    assert validate_environment_receipt(
        akib_receipt, expected_host="akibscomputer"
    ) == akib_receipt

    false_local_akib = copy.deepcopy(akib_receipt)
    false_local_akib["tmux"]["attestation_mode"] = "direct_tmux_session"
    with pytest.raises(ValueError, match="delegated_remote_tmux_session"):
        validate_environment_receipt(
            false_local_akib, expected_host="akibscomputer"
        )

    delegated_main = copy.deepcopy(main_receipt)
    delegated_main["tmux"][
        "attestation_mode"
    ] = "delegated_remote_tmux_session"
    with pytest.raises(ValueError, match="direct_tmux_session"):
        validate_environment_receipt(
            delegated_main, expected_host="main"
        )


def test_runtime_semantically_verifies_contract_receipt_and_live_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_receipt = _environment_receipt(
        "main", torch_version="2.5.1+cu121"
    )
    akib_receipt = _environment_receipt(
        "akibscomputer", torch_version="2.5.1"
    )
    contract = build_environment_contract(main_receipt, akib_receipt)
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "observe_execution_environment",
        lambda host: copy.deepcopy(main_receipt),
    )
    assert (
        verify_current_execution_environment(
            contract=contract,
            receipt=main_receipt,
            host="main",
        )
        == main_receipt
    )

    receipt_tamper = copy.deepcopy(main_receipt)
    receipt_tamper["torch"]["version"] = "9.9.9"
    with pytest.raises(
        PerfectDiodeConv3ExecutionError,
        match="contract-bound host receipt",
    ):
        verify_current_execution_environment(
            contract=contract,
            receipt=receipt_tamper,
            host="main",
        )

    self_consistent_tamper = build_environment_contract(
        receipt_tamper, akib_receipt
    )
    with pytest.raises(
        PerfectDiodeConv3ExecutionError,
        match="freshly observed",
    ):
        verify_current_execution_environment(
            contract=self_consistent_tamper,
            receipt=receipt_tamper,
            host="main",
        )


def test_runtime_rejects_tampered_contract_policy() -> None:
    contract = build_environment_contract(
        _environment_receipt("main", torch_version="2.5.1+cu121"),
        _environment_receipt("akibscomputer", torch_version="2.5.1"),
    )
    contract["host_runtime_policy"][
        "cross_host_software_equality_required"
    ] = True

    with pytest.raises(ValueError, match="canonical contract"):
        validate_environment_contract(contract)


def test_live_observer_supports_direct_main_and_delegated_remote_akib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.5.1+cu121",
        version=SimpleNamespace(cuda="12.1"),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(version=lambda: 90100)
        ),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: SimpleNamespace(
                name="Test GPU", total_memory=24_000_000_000
            ),
            get_device_capability=lambda index: (8, 6),
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("TMUX", "/tmp/tmux-test/default,1,0")
    monkeypatch.setenv("TMUX_PANE", "%3")
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="main\n"),
    )

    main_receipt = observe_execution_environment("main")
    assert main_receipt["tmux"] == {
        "attestation_mode": "direct_tmux_session",
        "session_name": "main",
        "pane_token": "%3",
    }

    monkeypatch.delenv("TMUX")
    monkeypatch.delenv("TMUX_PANE")
    monkeypatch.setenv("PD_LR_TMUX_SESSION", "akibscomputer")
    monkeypatch.setenv("PD_LR_TMUX_PANE_TOKEN", "%8")
    akib_receipt = observe_execution_environment("akibscomputer")
    assert akib_receipt["tmux"] == {
        "attestation_mode": "delegated_remote_tmux_session",
        "session_name": "akibscomputer",
        "pane_token": "%8",
    }


def _valid_k_measurement(
    *,
    checkpoint_sha256: str,
    tensor_sha256: str,
    selected_t: int,
    candidate_k: int = 64,
    reference_k: int = 64,
    viable: bool = True,
) -> dict[str, Any]:
    gradient_l2 = 1.0 if viable else 0.0
    zero_fraction = 0.0 if viable else 1.0
    cosine = 1.0 if viable else None
    gradient_rms = 1.0 if viable else 0.0
    parameters = []
    for name in ("ConvWeight_0", "ConvWeight_1", "ConvWeight_2"):
        batches = [
            {
                "batch_index": batch,
                "gradient_l2": gradient_l2,
                "reference_gradient_l2": gradient_l2,
                "relative_gradient_l2_norm_delta": 0.0,
                "gradient_vector_relative_error": 0.0,
                "gradient_vector_cosine": cosine,
                "gradient_zero_fraction": zero_fraction,
                "reference_gradient_zero_fraction": zero_fraction,
                "absolute_zero_fraction_delta": 0.0,
                "reference_gradient_rms": gradient_rms,
            }
            for batch in range(8)
        ]
        parameters.append(
            {
                "parameter": name,
                "batch_count": 8,
                "batches": batches,
                "gradient_l2": gradient_l2,
                "reference_gradient_l2": gradient_l2,
                "relative_gradient_l2_norm_delta": 0.0,
                "gradient_vector_relative_error": 0.0,
                "gradient_vector_cosine": cosine,
                "gradient_zero_fraction": zero_fraction,
                "reference_gradient_zero_fraction": zero_fraction,
                "absolute_zero_fraction_delta": 0.0,
                "reference_median_gradient_rms": gradient_rms,
                "reference_q90_zero_fraction": zero_fraction,
                "initial_weight_rms": 1.0,
                "reference_nominal_update_unit": gradient_rms,
                "reference_viable": viable,
                "passed": viable,
            }
        )
    return {
        "candidate_k": candidate_k,
        "reference_k": reference_k,
        "batch_count": 8,
        "cohort_examples": 256,
        "batch_size": 32,
        "official_test_read": False,
        "cohort_source_indices_sha256": "8" * 64,
        "initialization_checkpoint_sha256": checkpoint_sha256,
        "initialization_tensor_sha256": tensor_sha256,
        "selected_t": selected_t,
        "fixed_step_minimization": True,
        "batch_state_policy": "reset_each_batch",
        "reference_viable": viable,
        "passed": viable,
        "parameter_diagnostics": parameters,
        "free_equilibrium_sha256_by_batch": ["f" * 64] * 8,
    }


def _epoch_k_evidence(
    checkpoint: Mapping[str, Any],
    *,
    selected_t: int,
    viable: bool = True,
) -> dict[str, Any]:
    epoch = int(checkpoint["epoch"])
    tensor_sha = f"{epoch:x}" * 64
    value = {
        "schema_version": EPOCH_K_VIABILITY_SCHEMA_VERSION,
        "epoch": epoch,
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_tensor_sha256": tensor_sha,
        "high_k_reference": 64,
        "official_test_read": False,
        "measurement": _valid_k_measurement(
            checkpoint_sha256=str(checkpoint["sha256"]),
            tensor_sha256=tensor_sha,
            selected_t=selected_t,
            viable=viable,
        ),
    }
    passed, reasons = _validate_epoch_k_viability_evidence(
        value, checkpoint=checkpoint, expected_epoch=epoch
    )
    value.update(
        {
            "status": "passed" if passed else "failed",
            "passed": passed,
            "validation_reasons": reasons,
        }
    )
    return value


def _valid_t_measurement(
    *,
    iteration_count: int,
    checkpoint_sha256: str,
    tensor_sha256: str,
) -> dict[str, Any]:
    statistic = {
        "mean": 1.0e-3,
        "median": 1.0e-3,
        "p90": 1.0e-3,
        "p99": 1.0e-3,
        "max": 1.0e-3,
    }
    layers = []
    for role in ("hidden_0", "hidden_1", "hidden_2", "output"):
        layer = {
            "role": role,
            "name": role,
            "selection_residual_mode": (
                "raw" if role == "output" else "projected_kkt"
            ),
            "selection_residual_p90": 1.0e-3,
            "selection_residual": copy.deepcopy(statistic),
            "raw_residual": copy.deepcopy(statistic),
        }
        if role != "output":
            layer["clamped_occupancy"] = {
                "fraction": 0.0,
                "excitation_fraction": 0.0,
                "inhibition_fraction": 0.0,
            }
        layers.append(layer)
    return {
        "iteration_count": iteration_count,
        "num_examples": 1_024,
        "batch_size": 64,
        "official_test_read": False,
        "cohort_source_indices_sha256": "7" * 64,
        "initialization_checkpoint_sha256": checkpoint_sha256,
        "initialization_tensor_sha256": tensor_sha256,
        "fixed_step_minimization": True,
        "batch_state_policy": "reset_each_batch",
        "layers": layers,
    }


def _fixed_tk_gate_context(tmp_path: Path) -> ExecutionContext:
    context = _context(tmp_path, optimizer="adam")
    context.row.update(
        {
            "inference_iterations": 8,
            "training_iterations": 8,
        }
    )
    context.row["upstream_tk"].update(
        {
            "selected_t": 4,
            "selected_k": 4,
        }
    )
    context.surface.update(
        {
            "inference_iterations": 8,
            "training_iterations": 8,
            "upstream_tk": copy.deepcopy(context.row["upstream_tk"]),
        }
    )
    context.spec._row = copy.deepcopy(context.row)
    context.spec._data["operating_point"] = {
        "mode": "user_fixed_after_residual_gradient_review",
        "inference_iterations": 8,
        "training_iterations": 8,
        "reference_inference_iterations": 64,
        "reference_training_iterations": 64,
        "shared_across_schemes": True,
        "retain_upstream_diagnostic_selection": True,
        "require_fresh_manifest_bound_preflight_gate": True,
    }
    context.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    context.manifest_path.write_text(
        json.dumps(context.manifest, sort_keys=True),
        encoding="utf-8",
    )
    return context


def _fixed_tk_gate_assets(context: ExecutionContext) -> LoadedV2Assets:
    asset_dir = _asset_dir(context)
    asset_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = asset_dir / "initialization.pt"
    checkpoint.write_bytes(b"fixed-tk-gate-initialization")
    shared = context.surface["shared_asset_hashes"]
    tk_assets = LoadedTKAssets(
        root=asset_dir,
        checkpoint=checkpoint,
        checkpoint_sha256=shared["initialization_checkpoint_sha256"],
        parameter_tensor_sha256=shared["initialization_tensor_sha256"],
        train_indices_sha256=shared["train_indices_sha256"],
        validation_indices_sha256=shared["validation_indices_sha256"],
        t_indices_sha256=shared["t_cohort_indices_sha256"],
        k_indices_sha256=shared["k_cohort_indices_sha256"],
        t_batches=(),
        k_batches=(),
    )
    tk_spec = SimpleNamespace(
        rows=(
            {
                "row_id": context.row["row_id"],
                "scheme": context.surface["scheme"],
            },
        )
    )
    return LoadedV2Assets(
        asset_dir=asset_dir,
        checkpoint_path=checkpoint,
        checkpoint_sha256=shared["initialization_checkpoint_sha256"],
        parameter_tensor_sha256=shared["initialization_tensor_sha256"],
        bundle=SimpleNamespace(),
        asset_metadata={},
        epoch_order_sha256s=("9" * 64, "a" * 64, "b" * 64),
        tk_spec=tk_spec,  # type: ignore[arg-type]
        tk_assets=tk_assets,
    )


def _fixed_tk_gate_audit(
    context: ExecutionContext,
    *,
    passed: bool,
) -> dict[str, Any]:
    shared = context.surface["shared_asset_hashes"]
    checkpoint_sha = shared["initialization_checkpoint_sha256"]
    tensor_sha = shared["initialization_tensor_sha256"]
    upstream = context.spec.data["upstream_tk"]
    return {
        "schema_version": TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION,
        "study_id": upstream["study_id"],
        "config_sha256": upstream["config_sha256"],
        "entry_id": f"{context.row['row_id']}--lr-fixed-tk-gate",
        "row_id": context.row["row_id"],
        "architecture": "conv3",
        "scheme": context.surface["scheme"],
        "execution_backend": "tmux",
        "execution_host": context.host,
        "execution_environment_sha256": context.surface[
            "execution_environment_sha256"
        ],
        "selected_t": 8,
        "selected_k": 8,
        "k_audit_reference": 64,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "failure_reasons": (
            [] if passed else ["selected_k_gradient_gate_failed"]
        ),
        "fresh_replay_after_selection": True,
        "official_test_read": False,
        "t_measurement": _valid_t_measurement(
            iteration_count=8,
            checkpoint_sha256=checkpoint_sha,
            tensor_sha256=tensor_sha,
        ),
        "t64_sentinel_measurement": _valid_t_measurement(
            iteration_count=64,
            checkpoint_sha256=checkpoint_sha,
            tensor_sha256=tensor_sha,
        ),
        "t64_sentinel_passed": True,
        "t256_extension_sentinel_measurement": None,
        "t256_extension_sentinel_passed": None,
        "t_extension_used": False,
        "k_measurement": _valid_k_measurement(
            checkpoint_sha256=checkpoint_sha,
            tensor_sha256=tensor_sha,
            selected_t=8,
            candidate_k=8,
            reference_k=64,
            viable=passed,
        ),
    }


class _Backend:
    def __init__(
        self, *, legacy: bool = False, dead_best_candidate: bool = False
    ) -> None:
        self.legacy = legacy
        self.dead_best_candidate = dead_best_candidate
        self.load_calls = 0
        self.probe_calls = 0
        self.canary_calls: list[tuple[float, float]] = []
        self.candidate_calls: list[tuple[float, float]] = []

    def load_assets(self, context: ExecutionContext, **kwargs: Any) -> Any:
        del context, kwargs
        self.load_calls += 1
        return {"status": "verified", "official_test_read": False}

    def optimizer_probe(
        self,
        context: ExecutionContext,
        assets: Any,
        *,
        output_dir: Path,
        device: str,
    ) -> Mapping[str, Any]:
        del context, assets, output_dir, device
        self.probe_calls += 1
        return {
            "schema_version": "fake-probe/v1",
            "status": "complete",
            "probe_stable": True,
            "official_test_read": False,
        }

    def canary(
        self,
        context: ExecutionContext,
        assets: Any,
        probe: Mapping[str, Any],
        *,
        rho_conv: float,
        rho_dense: float,
        output_dir: Path,
        device: str,
    ) -> Mapping[str, Any]:
        del context, assets, probe, output_dir, device
        self.canary_calls.append((rho_conv, rho_dense))
        first_legacy_center = (
            self.legacy
            and rho_conv == pytest.approx(0.003)
            and rho_dense == pytest.approx(0.01)
        )
        return {
            "schema_version": "fake-canary/v1",
            "status": (
                "safety_failure" if first_legacy_center else "complete"
            ),
            "safety_clean": not first_legacy_center,
            "safety_failure": (
                {"kind": "projection_efficiency"}
                if first_legacy_center
                else None
            ),
            "completed_steps": 32 if first_legacy_center else 640,
            "median_projection_efficiency": (
                0.2 if first_legacy_center else 0.8
            ),
            "official_test_read": False,
        }

    def candidate(
        self,
        context: ExecutionContext,
        assets: Any,
        probe: Mapping[str, Any],
        *,
        rho_conv: float,
        rho_dense: float,
        output_dir: Path,
        device: str,
    ) -> Mapping[str, Any]:
        del assets, probe, device
        self.candidate_calls.append((rho_conv, rho_dense))
        output_dir.mkdir(parents=True, exist_ok=True)
        epoch_checkpoints = []
        for epoch in range(1, 4):
            checkpoint = output_dir / f"epoch_{epoch:03d}.pt"
            checkpoint.write_bytes(
                f"{rho_conv}:{rho_dense}:{epoch}".encode("ascii")
            )
            epoch_checkpoints.append(
                {
                    "epoch": epoch,
                    "path": checkpoint.name,
                    "sha256": sha256_file(checkpoint),
                }
            )
        if self.legacy:
            target = rho_conv == pytest.approx(0.003) and rho_dense == pytest.approx(
                0.01 / 3.0
            )
        else:
            target = rho_conv == pytest.approx(0.027) and rho_dense == pytest.approx(
                0.09
            )
        dead_target = (
            self.dead_best_candidate
            and rho_conv == pytest.approx(0.009)
            and rho_dense == pytest.approx(0.09)
        )
        epoch_k_viability = [
            _epoch_k_evidence(
                checkpoint,
                selected_t=int(context.surface["inference_iterations"]),
                viable=not (dead_target and checkpoint["epoch"] == 2),
            )
            for checkpoint in epoch_checkpoints
        ]
        if dead_target:
            validation_loss = 0.10
            validation_accuracy = 0.96
        elif self.dead_best_candidate and target:
            validation_loss = 0.11
            validation_accuracy = 0.94
        else:
            validation_loss = 0.10 if target else 0.30
            validation_accuracy = 0.94 if target else 0.91
        return {
            "schema_version": "fake-candidate/v1",
            "status": "complete",
            "training_completed": True,
            "safety_admissible": True,
            "completed_steps": 10_314,
            "final_validation_loss": validation_loss,
            "final_validation_accuracy": validation_accuracy,
            "median_projection_efficiency": 0.80,
            "raw_learning_rates_by_parameter": {
                "ConvWeight_0": rho_conv,
                "DenseWeight_0": rho_dense,
            },
            "epoch_checkpoints": epoch_checkpoints,
            "epoch_k_reference_viability": epoch_k_viability,
            "epoch_k_reference_viability_passed": all(
                record["passed"] for record in epoch_k_viability
            ),
            "official_test_read": False,
        }


def _run_through_select_final(
    context: ExecutionContext, backend: _Backend
) -> dict[str, Any]:
    kwargs = {
        "backend": backend,
        "asset_dir": _asset_dir(context),
        "data_root": context.surface_root.parent / "mnist",
        "device": "cpu",
    }
    final = None
    for stage in PUBLIC_STAGE_SEQUENCE[:9]:
        final = execute_surface_stage(context, stage, **kwargs)
    assert final is not None
    return final


def _passing_audit(
    context: ExecutionContext,
    *,
    selected_candidate: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    del kwargs
    epoch_audits = []
    for checkpoint, viability in zip(
        selected_candidate["epoch_checkpoints"],
        selected_candidate["epoch_k_reference_viability"],
    ):
        tensor_sha = viability["checkpoint_tensor_sha256"]
        selected_t = _valid_t_measurement(
            iteration_count=int(context.surface["inference_iterations"]),
            checkpoint_sha256=str(checkpoint["sha256"]),
            tensor_sha256=tensor_sha,
        )
        t64 = _valid_t_measurement(
            iteration_count=64,
            checkpoint_sha256=str(checkpoint["sha256"]),
            tensor_sha256=tensor_sha,
        )
        k_measurement = _valid_k_measurement(
            checkpoint_sha256=str(checkpoint["sha256"]),
            tensor_sha256=tensor_sha,
            selected_t=int(context.surface["inference_iterations"]),
            candidate_k=int(context.surface["training_iterations"]),
            reference_k=64,
        )
        operating_audit = {
            "schema_version": TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION,
            "study_id": context.spec.data["upstream_tk"]["study_id"],
            "config_sha256": context.spec.data["upstream_tk"][
                "config_sha256"
            ],
            "entry_id": (
                f"{context.surface_id}--epoch-{checkpoint['epoch']:03d}"
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
            "k_audit_reference": 64,
            "fresh_replay_after_selection": True,
            "t_extension_used": False,
            "status": "passed",
            "passed": True,
            "t_measurement": selected_t,
            "t64_sentinel_measurement": t64,
            "t64_sentinel_passed": True,
            "t256_extension_sentinel_measurement": None,
            "t256_extension_sentinel_passed": None,
            "k_measurement": k_measurement,
        }
        epoch_audits.append(
            {
                "epoch": checkpoint["epoch"],
                "checkpoint_path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["sha256"],
                "checkpoint_tensor_sha256": tensor_sha,
                "selected_t_residual_gate_passed": True,
                "reference_t64_residual_gate_passed": True,
                "selected_k_gradient_gate_passed": True,
                "reference_gradient_viable": True,
                "k128_sentinel_gate_passed": None,
                "t256_extension_residual_gate_passed": None,
                "passed": True,
                "official_test_read": False,
                "selected_operating_point_audit": operating_audit,
                "reference_t64_measurement": t64,
                "selected_k_vs_k64_measurement": k_measurement,
                "k128_sentinel_measurement": None,
            }
        )
    return {
        "schema_version": POST_TK_AUDIT_SCHEMA_VERSION,
        "surface_id": context.surface_id,
        "row_id": context.row["row_id"],
        "scheme": context.surface["scheme"],
        "selected_cell_id": selected_candidate["cell_id"],
        "selected_t": context.surface["inference_iterations"],
        "selected_k": context.surface["training_iterations"],
        "status": "passed",
        "passed": True,
        "official_test_read": False,
        "epoch_audits": epoch_audits,
    }


def test_complete_fixed_surface_lifecycle_and_hash_verified_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    backend = _Backend()
    select_final = _run_through_select_final(context, backend)

    assert select_final["status"] == "selected"
    assert select_final["expansion_used"] is False
    assert len(backend.canary_calls) == 9
    assert len(backend.candidate_calls) == 9

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "run_builtin_post_training_tk_audit",
        _passing_audit,
    )
    audit = execute_surface_stage(
        context,
        "post_training_tk",
        backend=backend,
        asset_dir=_asset_dir(context),
        data_root=tmp_path / "mnist",
        device="cpu",
    )
    final = execute_surface_stage(context, "finalize_lr", backend=backend)
    assert audit["status"] == "complete"
    assert final["status"] == (
        "selected_seed0_ordinary_mnist_three_epoch_screen"
    )
    assert final["selected_rho"] == {
        "rho_conv": pytest.approx(0.027),
        "rho_dense": pytest.approx(0.09),
    }
    assert len(final["epoch_checkpoints"]) == 3
    assert (context.surface_root / "completion.json").is_file()

    before = (
        backend.load_calls,
        backend.probe_calls,
        len(backend.canary_calls),
        len(backend.candidate_calls),
    )
    resumed = execute_surface_stage(context, "finalize_lr", backend=backend)
    assert resumed == final
    assert (
        backend.load_calls,
        backend.probe_calls,
        len(backend.canary_calls),
        len(backend.candidate_calls),
    ) == before


def test_legacy_adaptive_center_and_single_upper_expansion(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, scheme="legacy")
    backend = _Backend(legacy=True)
    select_final = _run_through_select_final(context, backend)

    core = execute_surface_stage(
        context, "select_core", backend=backend
    )
    assert core["status"] == "needs_expansion"
    assert core["selection"]["expansion"]["directions"] == [
        {"axis": "rho_conv", "direction": "upper"}
    ]
    assert select_final["status"] == "selected"
    assert select_final["expansion_used"] is True
    # The failed initial center is reused as a core cell and never promoted.
    assert len(backend.canary_calls) == 12
    assert len(backend.candidate_calls) == 11


def test_dead_best_candidate_cannot_hide_valid_second_choice(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    backend = _Backend(dead_best_candidate=True)
    selected = _run_through_select_final(context, backend)

    assert selected["status"] == "selected"
    assert selected["selection"]["selected"]["rho_conv"] == pytest.approx(
        0.027
    )
    assert selected["selection"]["selected"]["rho_dense"] == pytest.approx(
        0.09
    )
    candidates = execute_surface_stage(
        context, "rho_core_candidates", backend=backend
    )["candidate_records"]
    dead = next(
        record
        for record in candidates
        if record["rho_conv"] == pytest.approx(0.009)
        and record["rho_dense"] == pytest.approx(0.09)
    )
    assert dead["admissible"] is False
    assert dead["inadmissible_reason"].startswith(
        "epoch_k_reference_viability_failed:"
    )


def test_missing_post_training_audit_is_explicit_unresolved_zero_work(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    backend = _Backend()
    _run_through_select_final(context, backend)

    audit = execute_surface_stage(
        context, "post_training_tk", backend=backend
    )
    final = execute_surface_stage(context, "finalize_lr", backend=backend)
    assert audit["status"] == "unresolved_post_training_tk"
    assert audit["zero_work"] is True
    assert final["status"] == "unresolved_post_training_tk"
    assert final["zero_work"] is True


def test_upstream_tk_ineligible_surface_stays_zero_work_without_backend(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, eligible=False)
    backend = _Backend()
    audit = execute_surface_stage(context, "audit", backend=backend)
    assets = execute_surface_stage(context, "import_tk", backend=backend)
    probe = execute_surface_stage(
        context, "optimizer_probe", backend=backend
    )

    assert audit["status"] == "complete"
    assert assets["zero_work"] is True
    assert probe["zero_work"] is True
    assert backend.load_calls == 0
    assert backend.probe_calls == 0


def test_stage_resume_rejects_tampered_nested_candidate_artifact(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    backend = _Backend()
    _run_through_select_final(context, backend)
    checkpoint = next(
        (
            context.surface_root
            / "stages"
            / "rho_core_candidates"
            / "cells"
        ).glob("*/epoch_001.pt")
    )
    checkpoint.write_bytes(b"tampered")

    with pytest.raises(
        PerfectDiodeConv3ExecutionError,
        match="hash- and size-verified artifact",
    ):
        execute_surface_stage(
            context, "rho_core_candidates", backend=backend
        )


def test_post_training_audit_requires_every_defining_epoch_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    backend = _Backend()
    _run_through_select_final(context, backend)

    def bad_audit(**kwargs: Any) -> dict[str, Any]:
        result = _passing_audit(**kwargs)
        result["epoch_audits"][1]["reference_gradient_viable"] = False
        return result

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "run_builtin_post_training_tk_audit",
        bad_audit,
    )
    with pytest.raises(ValueError, match=r"epoch_audits\[1\]"):
        execute_surface_stage(
            context,
            "post_training_tk",
            backend=backend,
            asset_dir=_asset_dir(context),
            data_root=tmp_path / "mnist",
            device="cpu",
        )


def test_builtin_post_training_audit_replays_all_epoch_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    checkpoints = []
    for epoch in range(1, 4):
        path = candidate_dir / f"epoch_{epoch:03d}.pt"
        path.write_bytes(f"epoch-{epoch}".encode("ascii"))
        checkpoints.append(
            {
                "epoch": epoch,
                "path": path.name,
                "sha256": sha256_file(path),
            }
        )

    class _TKSpec:
        study_id = "tkstudy_" + "5" * 64
        config_sha256 = "6" * 64
        data: dict[str, Any] = {}
        rows = ({"row_id": "tk-baseline", "scheme": "baseline"},)

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "PerfectDiodeTKStudySpec.from_path",
        lambda path: _TKSpec(),
    )
    shared = context.surface["shared_asset_hashes"]
    base_assets = LoadedTKAssets(
        root=tmp_path,
        checkpoint=tmp_path / "initialization.pt",
        checkpoint_sha256=shared["initialization_checkpoint_sha256"],
        parameter_tensor_sha256=shared["initialization_tensor_sha256"],
        train_indices_sha256=shared["train_indices_sha256"],
        validation_indices_sha256=shared["validation_indices_sha256"],
        t_indices_sha256=shared["t_cohort_indices_sha256"],
        k_indices_sha256=shared["k_cohort_indices_sha256"],
        t_batches=(),
        k_batches=(),
    )
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "load_tk_shared_assets",
        lambda *args, **kwargs: base_assets,
    )

    class _Runtime:
        parameters = ()

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "build_model_runtime",
        lambda *args, **kwargs: _Runtime(),
    )
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "_contract_parameter_tensor_digest",
        lambda parameters: "7" * 64,
    )

    calls: list[tuple[str, bool, str]] = []

    def operating_audit(
        *args: Any,
        entry_id: str,
        selected_t: int,
        selected_k: int,
        t_extension_used: bool,
        execution_environment_sha256: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del args
        trained_assets = kwargs["assets"]
        selected_t_measurement = _valid_t_measurement(
            iteration_count=selected_t,
            checkpoint_sha256=trained_assets.checkpoint_sha256,
            tensor_sha256=trained_assets.parameter_tensor_sha256,
        )
        t64_measurement = _valid_t_measurement(
            iteration_count=64,
            checkpoint_sha256=trained_assets.checkpoint_sha256,
            tensor_sha256=trained_assets.parameter_tensor_sha256,
        )
        calls.append(
            (
                entry_id,
                t_extension_used,
                execution_environment_sha256,
            )
        )
        return {
            "schema_version": TK_OPERATING_POINT_AUDIT_SCHEMA_VERSION,
            "study_id": _TKSpec.study_id,
            "config_sha256": _TKSpec.config_sha256,
            "entry_id": entry_id,
            "row_id": context.row["row_id"],
            "architecture": "conv3",
            "scheme": context.surface["scheme"],
            "execution_backend": "tmux",
            "execution_host": context.host,
            "execution_environment_sha256": (
                execution_environment_sha256
            ),
            "selected_t": selected_t,
            "selected_k": selected_k,
            "k_audit_reference": 64,
            "status": "passed",
            "passed": True,
            "failure_reasons": [],
            "fresh_replay_after_selection": True,
            "official_test_read": False,
            "t_measurement": selected_t_measurement,
            "t64_sentinel_measurement": t64_measurement,
            "t64_sentinel_passed": True,
            "t256_extension_sentinel_measurement": None,
            "t256_extension_sentinel_passed": None,
            "t_extension_used": t_extension_used,
            "k_measurement": _valid_k_measurement(
                checkpoint_sha256=trained_assets.checkpoint_sha256,
                tensor_sha256=trained_assets.parameter_tensor_sha256,
                selected_t=selected_t,
                candidate_k=selected_k,
                reference_k=64,
            ),
        }

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "run_operating_point_audit",
        operating_audit,
    )
    result = run_builtin_post_training_tk_audit(
        context=context,
        selected_candidate={"cell_id": "cell"},
        candidate_dir=candidate_dir,
        epoch_checkpoints=checkpoints,
        asset_dir=_asset_dir(context),
        data_root=tmp_path,
        device="cpu",
        tk_config_path=tmp_path / "tk.json",
    )

    assert result["passed"] is True
    assert len(result["epoch_audits"]) == 3
    assert len(calls) == 3
    assert all(call[1:] == (False, "1" * 64) for call in calls)
    assert all(
        record["reference_t64_residual_gate_passed"] is True
        for record in result["epoch_audits"]
    )


def test_public_stage_and_worker_contract_expose_only_authoritative_names(
    tmp_path: Path,
) -> None:
    assert PUBLIC_STAGE_SEQUENCE == (
        "audit",
        "import_tk",
        "optimizer_probe",
        "rho_canary_core",
        "rho_core_candidates",
        "select_core",
        "rho_canary_expansion",
        "rho_expansion_candidates",
        "select_expanded",
        "post_training_tk",
        "finalize_lr",
    )
    parser = _worker_parser()
    stage_action = next(
        action for action in parser._actions if action.dest == "stage"
    )
    assert tuple(stage_action.choices) == PUBLIC_STAGE_SEQUENCE
    assert "--post-tk-audit-result" not in parser._option_string_actions
    assert "--preflight-canary" in parser._option_string_actions
    assert "--preflight-tk-gate" in parser._option_string_actions
    assert "--validate-preflight-canary" in parser._option_string_actions
    assert "--validate-preflight-tk-gate" in parser._option_string_actions
    assert {
        "--t",
        "--k",
        "--selected-t",
        "--selected-k",
        "--inference-iterations",
        "--training-iterations",
    }.isdisjoint(parser._option_string_actions)

    context = _context(tmp_path)
    with pytest.raises(TypeError, match="tk_audit_result_path"):
        execute_surface_stage(
            context,
            "post_training_tk",
            tk_audit_result_path=tmp_path / "unbound.json",  # type: ignore[call-arg]
        )


def test_fixed_tk_preflight_calls_exact_t8_k8_and_hash_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _fixed_tk_gate_context(tmp_path)
    assets = _fixed_tk_gate_assets(context)

    class _GateBackend:
        def __init__(self) -> None:
            self.load_calls = 0

        def load_assets(
            self,
            observed_context: ExecutionContext,
            **kwargs: Any,
        ) -> LoadedV2Assets:
            assert observed_context is context
            assert kwargs["asset_dir"] == assets.asset_dir
            assert kwargs["device"] == "cpu"
            self.load_calls += 1
            return assets

    backend = _GateBackend()
    audit_calls: list[dict[str, Any]] = []

    def operating_point_audit(
        tk_spec: Any,
        row: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert tk_spec is assets.tk_spec
        assert row == assets.tk_spec.rows[0]
        assert kwargs["assets"] is assets.tk_assets
        audit_calls.append(dict(kwargs))
        return _fixed_tk_gate_audit(context, passed=True)

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "run_operating_point_audit",
        operating_point_audit,
    )
    result = execute_fixed_operating_point_preflight_gate(
        context,
        backend=backend,
        asset_dir=assets.asset_dir,
        data_root=tmp_path / "mnist",
        device="cpu",
    )

    assert result["status"] == "passed"
    assert result["passed"] is True
    assert result["selected_t"] == 8
    assert result["selected_k"] == 8
    assert result["diagnostic_selected_t"] == 4
    assert result["diagnostic_selected_k"] == 4
    assert backend.load_calls == 1
    assert len(audit_calls) == 1
    assert audit_calls[0]["selected_t"] == 8
    assert audit_calls[0]["selected_k"] == 8
    assert audit_calls[0]["entry_id"].endswith("--lr-fixed-tk-gate")

    resumed = execute_fixed_operating_point_preflight_gate(
        context,
        backend=backend,
        asset_dir=assets.asset_dir,
        data_root=tmp_path / "mnist",
        device="cpu",
    )
    assert resumed == result
    assert backend.load_calls == 1
    assert len(audit_calls) == 1

    result_path = (
        context.manifest_path.parent
        / "preflight"
        / "fixed_tk_gate"
        / context.host
        / context.row["row_id"]
        / "result.json"
    )
    tampered = json.loads(result_path.read_text(encoding="utf-8"))
    tampered["selected_t"] = 4
    result_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        PerfectDiodeConv3ExecutionError,
        match="hash- and size-verified artifact",
    ):
        execute_fixed_operating_point_preflight_gate(
            context,
            backend=backend,
            asset_dir=assets.asset_dir,
            data_root=tmp_path / "mnist",
            device="cpu",
        )


def test_failed_fixed_tk_preflight_is_complete_but_nonpassing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _fixed_tk_gate_context(tmp_path)
    assets = _fixed_tk_gate_assets(context)
    backend = SimpleNamespace(
        load_assets=lambda observed_context, **kwargs: assets
    )
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "run_operating_point_audit",
        lambda *args, **kwargs: _fixed_tk_gate_audit(
            context, passed=False
        ),
    )

    result = execute_fixed_operating_point_preflight_gate(
        context,
        backend=backend,  # type: ignore[arg-type]
        asset_dir=assets.asset_dir,
        data_root=tmp_path / "mnist",
        device="cpu",
    )

    assert result["status"] == "failed"
    assert result["passed"] is False
    assert result["reason"] == "fixed_tk_residual_or_gradient_gate_failed"
    assert result["operating_point_audit"]["passed"] is False


def test_isolated_preflight_runs_one_exact_canary_and_hash_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, optimizer="adam")
    backend = _Backend()
    plan = {
        "status": "required",
        "stage": "preflight_canary",
        "same_public_runner": True,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "row_id": context.row["row_id"],
        "optimizer": "adam",
        "host": context.host,
        "rho_conv": 0.027,
        "rho_dense": 0.09,
        "steps": 640,
        "restart_from_shared_initialization": True,
        "shared_asset_hashes": copy.deepcopy(
            context.surface["shared_asset_hashes"]
        ),
        "execution_source": copy.deepcopy(
            context.surface["execution_source"]
        ),
        "execution_environment_sha256": context.surface[
            "execution_environment_sha256"
        ],
        "official_test_read": False,
    }
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "representative_preflight_plan",
        lambda manifest, *, spec, host: copy.deepcopy(plan),
    )

    result = execute_representative_preflight_canary(
        context,
        backend=backend,
        asset_dir=_asset_dir(context),
        data_root=tmp_path / "mnist",
        device="cpu",
    )
    assert result["status"] == "complete"
    assert result["passed"] is True
    assert backend.canary_calls == [(0.027, 0.09)]
    assert (
        context.manifest_path.parent
        / "preflight"
        / "representative_canary"
        / "main"
        / "completion.json"
    ).is_file()
    assert not (context.surface_root / "stages").exists()

    resumed = execute_representative_preflight_canary(
        context,
        backend=backend,
        asset_dir=_asset_dir(context),
        data_root=tmp_path / "mnist",
        device="cpu",
    )
    assert resumed == result
    assert backend.canary_calls == [(0.027, 0.09)]


def test_failed_isolated_preflight_is_complete_evidence_but_not_passing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, optimizer="adam")
    plan = {
        "status": "required",
        "stage": "preflight_canary",
        "same_public_runner": True,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "row_id": context.row["row_id"],
        "optimizer": "adam",
        "host": context.host,
        "rho_conv": 0.027,
        "rho_dense": 0.09,
        "steps": 640,
        "restart_from_shared_initialization": True,
        "shared_asset_hashes": copy.deepcopy(
            context.surface["shared_asset_hashes"]
        ),
        "execution_source": copy.deepcopy(
            context.surface["execution_source"]
        ),
        "execution_environment_sha256": context.surface[
            "execution_environment_sha256"
        ],
        "official_test_read": False,
    }
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "representative_preflight_plan",
        lambda manifest, *, spec, host: copy.deepcopy(plan),
    )

    class _FailedBackend(_Backend):
        def canary(self, *args: Any, **kwargs: Any) -> Mapping[str, Any]:
            result = dict(super().canary(*args, **kwargs))
            result.update(
                {
                    "status": "safety_failure",
                    "safety_clean": False,
                    "completed_steps": 17,
                    "safety_failure": {"kind": "non_finite_loss"},
                }
            )
            return result

    result = execute_representative_preflight_canary(
        context,
        backend=_FailedBackend(),
        asset_dir=_asset_dir(context),
        data_root=tmp_path / "mnist",
        device="cpu",
    )
    assert result["status"] == "failed_preflight_canary"
    assert result["passed"] is False


def test_preflight_receipts_do_not_collide_across_manifest_hosts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = {
        host: _context(tmp_path, optimizer="adam", host=host)
        for host in ("main", "akibscomputer")
    }

    def plan_for_host(
        manifest: Mapping[str, Any],
        *,
        spec: Any,
        host: str,
    ) -> dict[str, Any]:
        del manifest, spec
        context = contexts[host]
        return {
            "status": "required",
            "stage": "preflight_canary",
            "same_public_runner": True,
            "manifest_id": context.manifest_id,
            "surface_id": context.surface_id,
            "row_id": context.row["row_id"],
            "optimizer": "adam",
            "host": host,
            "rho_conv": 0.027,
            "rho_dense": 0.09,
            "steps": 640,
            "restart_from_shared_initialization": True,
            "shared_asset_hashes": copy.deepcopy(
                context.surface["shared_asset_hashes"]
            ),
            "execution_source": copy.deepcopy(
                context.surface["execution_source"]
            ),
            "execution_environment_sha256": context.surface[
                "execution_environment_sha256"
            ],
            "official_test_read": False,
        }

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "representative_preflight_plan",
        plan_for_host,
    )
    for host, context in contexts.items():
        result = execute_representative_preflight_canary(
            context,
            backend=_Backend(),
            asset_dir=_asset_dir(context),
            data_root=tmp_path / "mnist",
            device="cpu",
        )
        assert result["passed"] is True
        assert result["output_namespace"].endswith(f"/{host}")

    root = contexts["main"].manifest_path.parent
    assert (
        root / "preflight" / "representative_canary" / "main" / "completion.json"
    ).is_file()
    assert (
        root
        / "preflight"
        / "representative_canary"
        / "akibscomputer"
        / "completion.json"
    ).is_file()


def test_numerical_entry_rejects_asset_dir_outside_copied_study_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, optimizer="adam")
    plan = {
        "status": "required",
        "stage": "preflight_canary",
        "same_public_runner": True,
        "manifest_id": context.manifest_id,
        "surface_id": context.surface_id,
        "row_id": context.row["row_id"],
        "optimizer": "adam",
        "host": "main",
        "rho_conv": 0.027,
        "rho_dense": 0.09,
        "steps": 640,
        "restart_from_shared_initialization": True,
        "shared_asset_hashes": context.surface["shared_asset_hashes"],
        "execution_source": context.surface["execution_source"],
        "execution_environment_sha256": context.surface[
            "execution_environment_sha256"
        ],
        "official_test_read": False,
    }
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "representative_preflight_plan",
        lambda manifest, *, spec, host: copy.deepcopy(plan),
    )
    backend = _Backend()
    with pytest.raises(
        ValueError,
        match="copied, resolved-study-bound asset directory",
    ):
        execute_representative_preflight_canary(
            context,
            backend=backend,
            asset_dir=tmp_path / "external-assets",
            data_root=tmp_path / "mnist",
            device="cpu",
        )
    assert backend.load_calls == 0


def test_worker_returns_nonzero_for_failed_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path, optimizer="adam")
    monkeypatch.setattr(
        "experiments.run_mnist_conv_perfectdiode_hparam_v2_worker."
        "load_execution_context",
        lambda **kwargs: context,
    )
    monkeypatch.setattr(
        "experiments.run_mnist_conv_perfectdiode_hparam_v2_worker."
        "execute_representative_preflight_canary",
        lambda *args, **kwargs: {
            "status": "failed_preflight_canary",
            "passed": False,
        },
    )
    exit_code = _worker_main(
        [
            "--study",
            str(tmp_path / "study.json"),
            "--surface-manifest",
            str(tmp_path / "manifest.json"),
            "--surface-id",
            context.surface_id,
            "--preflight-canary",
            "--host",
            "main",
            "--source-archive",
            str(tmp_path / "source.tar"),
            "--environment-contract",
            str(tmp_path / "environment.json"),
            "--execution-environment",
            str(tmp_path / "main-environment.json"),
        ]
    )
    assert exit_code == 2


@pytest.mark.parametrize(
    ("action", "validator_name"),
    (
        (
            "--validate-preflight-canary",
            "validate_existing_representative_preflight_canary",
        ),
        (
            "--validate-preflight-tk-gate",
            "validate_existing_fixed_operating_point_preflight_gate",
        ),
    ),
)
def test_worker_read_only_preflight_validation_skips_live_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    validator_name: str,
) -> None:
    context = _context(tmp_path, optimizer="adam")
    load_calls: list[dict[str, Any]] = []
    validator_calls: list[ExecutionContext] = []

    def load_context(**kwargs: Any) -> ExecutionContext:
        load_calls.append(dict(kwargs))
        return context

    def validate(
        observed_context: ExecutionContext,
    ) -> dict[str, Any]:
        validator_calls.append(observed_context)
        return {"status": "complete", "passed": True}

    monkeypatch.setattr(
        "experiments.run_mnist_conv_perfectdiode_hparam_v2_worker."
        "load_execution_context",
        load_context,
    )
    monkeypatch.setattr(
        "experiments.run_mnist_conv_perfectdiode_hparam_v2_worker."
        f"{validator_name}",
        validate,
    )

    exit_code = _worker_main(
        [
            "--study",
            str(tmp_path / "study.json"),
            "--surface-manifest",
            str(tmp_path / "manifest.json"),
            "--surface-id",
            context.surface_id,
            action,
            "--host",
            "main",
            "--source-archive",
            str(tmp_path / "source.tar"),
            "--environment-contract",
            str(tmp_path / "environment.json"),
            "--execution-environment",
            str(tmp_path / "main-environment.json"),
        ]
    )

    assert exit_code == 0
    assert len(load_calls) == 1
    assert load_calls[0]["verify_live_environment"] is False
    assert validator_calls == [context]


def test_worker_returns_nonzero_for_failed_fixed_tk_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = _fixed_tk_gate_context(tmp_path)
    monkeypatch.setattr(
        "experiments.run_mnist_conv_perfectdiode_hparam_v2_worker."
        "load_execution_context",
        lambda **kwargs: context,
    )
    gate_calls: list[tuple[ExecutionContext, dict[str, Any]]] = []

    def failed_gate(
        observed_context: ExecutionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        gate_calls.append((observed_context, dict(kwargs)))
        return {
            "status": "failed",
            "passed": False,
        }

    monkeypatch.setattr(
        "experiments.run_mnist_conv_perfectdiode_hparam_v2_worker."
        "execute_fixed_operating_point_preflight_gate",
        failed_gate,
    )
    exit_code = _worker_main(
        [
            "--study",
            str(tmp_path / "study.json"),
            "--surface-manifest",
            str(tmp_path / "manifest.json"),
            "--surface-id",
            context.surface_id,
            "--preflight-tk-gate",
            "--host",
            "main",
            "--source-archive",
            str(tmp_path / "source.tar"),
            "--environment-contract",
            str(tmp_path / "environment.json"),
            "--execution-environment",
            str(tmp_path / "main-environment.json"),
            "--assets-dir",
            str(_asset_dir(context)),
            "--data-root",
            str(tmp_path / "mnist"),
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 2
    assert len(gate_calls) == 1
    observed_context, kwargs = gate_calls[0]
    assert observed_context is context
    assert {"selected_t", "selected_k", "t", "k"}.isdisjoint(kwargs)
    assert kwargs["device"] == "cpu"
    assert json.loads(capsys.readouterr().out) == {
        "passed": False,
        "status": "failed",
    }


def test_runtime_rejects_manifest_rho_policy_not_exactly_template_derived(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    expected_policy = _fixed_policy()
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "surface_rho_policy",
        lambda spec, scheme: copy.deepcopy(expected_policy),
    )
    assert _validate_surface(
        context.spec, context.manifest, context.surface
    )["rho_policy"] == expected_policy

    tampered = copy.deepcopy(context.surface)
    tampered["rho_policy"]["rho_conv"][0] = 0.008
    with pytest.raises(ValueError, match=r"surface\.rho_policy"):
        _validate_surface(context.spec, context.manifest, tampered)


def test_epoch_k_evidence_is_bound_to_frozen_cohort_and_selected_t() -> None:
    checkpoint = {
        "epoch": 1,
        "path": "epoch_001.pt",
        "sha256": "a" * 64,
    }
    evidence = _epoch_k_evidence(
        checkpoint,
        selected_t=16,
    )
    passed, reasons = _validate_epoch_k_viability_evidence(
        evidence,
        checkpoint=checkpoint,
        expected_epoch=1,
        expected_selected_t=16,
        expected_k_cohort_sha256="8" * 64,
    )
    assert passed is True
    assert reasons == []

    evidence["measurement"]["cohort_source_indices_sha256"] = "9" * 64
    passed, reasons = _validate_epoch_k_viability_evidence(
        evidence,
        checkpoint=checkpoint,
        expected_epoch=1,
        expected_selected_t=16,
        expected_k_cohort_sha256="8" * 64,
    )
    assert passed is False
    assert "wrong_k_cohort_sha256" in reasons


def test_v1_candidate_replays_high_k_reference_after_each_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    backend = V1NumericalBackend()

    class _Runtime:
        parameters = ("parameter",)

    class _TKSpec:
        data: dict[str, Any] = {}
        rows = ({"row_id": context.row["row_id"], "scheme": "baseline"},)

    initialization = tmp_path / "initialization.pt"
    initialization.write_bytes(b"initialization")
    tk_assets = LoadedTKAssets(
        root=tmp_path,
        checkpoint=initialization,
        checkpoint_sha256=sha256_file(initialization),
        parameter_tensor_sha256="6" * 64,
        train_indices_sha256="3" * 64,
        validation_indices_sha256="4" * 64,
        t_indices_sha256="7" * 64,
        k_indices_sha256="8" * 64,
        t_batches=(),
        k_batches=(),
    )
    assets = LoadedV2Assets(
        asset_dir=tmp_path,
        checkpoint_path=initialization,
        checkpoint_sha256=sha256_file(initialization),
        parameter_tensor_sha256="6" * 64,
        bundle=object(),
        asset_metadata={},
        epoch_order_sha256s=("9" * 64, "a" * 64, "b" * 64),
        tk_spec=_TKSpec(),  # type: ignore[arg-type]
        tk_assets=tk_assets,
    )
    runtime = _Runtime()
    monkeypatch.setattr(
        backend,
        "_fresh_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "derive_raw_learning_rates",
        lambda *args, **kwargs: {"ConvWeight_0": 1.0e-3},
    )
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "evaluate_validation",
        lambda *args, **kwargs: {
            "sample_count": 5_000,
            "loss": 0.2,
            "accuracy": 0.91,
            "source_indices": [],
        },
    )

    def save_checkpoint(runtime_value: Any, path: Path) -> None:
        del runtime_value
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("ascii"))

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "_save_checkpoint_atomic",
        save_checkpoint,
    )
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "_contract_parameter_tensor_digest",
        lambda parameters: "c" * 64,
    )
    replay_calls: list[tuple[str, int, int, str]] = []

    def collect_gradients(
        spec: Any,
        row: Mapping[str, Any],
        *,
        selected_t: int,
        training_iterations: int,
        assets: LoadedTKAssets,
        device: str,
    ) -> dict[str, Any]:
        del spec, row
        replay_calls.append(
            (
                assets.checkpoint.name,
                selected_t,
                training_iterations,
                device,
            )
        )
        return {"epoch_checkpoint": assets.checkpoint.name}

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "_collect_gradients_for_k",
        collect_gradients,
    )

    def compare_gradients(
        candidate: Mapping[str, Any],
        reference: Mapping[str, Any],
        *,
        candidate_k: int,
        reference_k: int,
    ) -> dict[str, Any]:
        assert candidate is reference
        assert candidate_k == reference_k == 64
        return _valid_k_measurement(
            checkpoint_sha256="0" * 64,
            tensor_sha256="0" * 64,
            selected_t=16,
        )

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "compare_k_gradients",
        compare_gradients,
    )

    def train_three_epochs(
        runtime_value: Any,
        bundle: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del bundle
        validate = kwargs["validation_function"]
        for _epoch in range(3):
            validate(runtime_value, object())
        return {
            "step_rows": [],
            "training_completed": True,
            "safety_admissible": True,
            "completed_steps": 10_314,
            "final_validation_loss": 0.2,
            "final_validation_accuracy": 0.91,
            "median_projection_efficiency": 0.8,
            "official_test_read": False,
        }

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "run_epoch_training",
        train_three_epochs,
    )
    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "_write_step_log",
        lambda *args, **kwargs: None,
    )
    output_dir = tmp_path / "candidate"
    output_dir.mkdir()
    result = backend.candidate(
        context,
        assets,
        {},
        rho_conv=0.027,
        rho_dense=0.09,
        output_dir=output_dir,
        device="cpu",
    )

    assert replay_calls == [
        ("epoch_001.pt", 16, 64, "cpu"),
        ("epoch_002.pt", 16, 64, "cpu"),
        ("epoch_003.pt", 16, 64, "cpu"),
    ]
    assert result["epoch_k_reference_viability_passed"] is True
    assert len(result["epoch_k_reference_viability"]) == 3


def test_post_training_validation_recomputes_raw_gradient_aggregates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    backend = _Backend()
    _run_through_select_final(context, backend)

    def tampered_audit(**kwargs: Any) -> dict[str, Any]:
        result = _passing_audit(**kwargs)
        batch = result["epoch_audits"][0][
            "selected_operating_point_audit"
        ]["k_measurement"]["parameter_diagnostics"][0]["batches"][0]
        batch["gradient_l2"] = 2.0
        return result

    monkeypatch.setattr(
        "experiments.mnist_conv.perfectdiode_hparam_v2_execution."
        "run_builtin_post_training_tk_audit",
        tampered_audit,
    )
    with pytest.raises(
        ValueError,
        match="complete internally consistent T/K measurements",
    ):
        execute_surface_stage(
            context,
            "post_training_tk",
            backend=backend,
            asset_dir=_asset_dir(context),
            data_root=tmp_path / "mnist",
            device="cpu",
        )
