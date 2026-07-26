from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_conv import lr_optimizer_boundary_runtime as runtime_module
from experiments.mnist_conv.lr_optimizer_boundary_runtime import (
    BIAS_TO_WEIGHT,
    BoundaryAssets,
    BoundaryRuntimeError,
    PARAMETER_NAMES,
    build_boundary_run_payload,
    compose_effective_optimizer_probe,
    derive_raw_learning_rates,
    optimizer_state_digest,
    replay_shadow_updates,
)
from experiments.mnist_conv.lr_optimizer_boundary import (
    entry_id,
    validate_result,
)
from experiments.mnist_conv.lr_optimizer_boundary_spec import (
    _EXPECTED_REUSE_SOURCE_BY_COORDINATE,
    optimizer_boundary_study_template,
)
from experiments.mnist_conv.lr_step import (
    capture_parameter_optimizer_state,
    restore_parameter_optimizer_state,
)


class _ScientificParameter:
    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.state = torch.tensor([value], dtype=torch.float64)
        self.state.requires_grad_(False)
        self.min_cond = 0.0
        self.max_cond = 100.0


def _probe() -> dict:
    return {
        "normalization_unit_by_weight": {
            "ConvWeight_0": 2.0,
            "ConvWeight_1": 4.0,
            "DenseWeight_0": 5.0,
        },
        "bias_q90_unit_by_parameter": {
            "Bias_0": 10.0,
            "Bias_1": 0.01,
        },
    }


def test_derive_raw_learning_rates_attached_and_capped() -> None:
    attached = derive_raw_learning_rates(
        _probe(),
        rho_conv=0.03,
        rho_dense=0.01,
        bias_policy="attached",
    )
    assert attached == {
        "ConvWeight_0": 0.015,
        "Bias_0": 0.015,
        "ConvWeight_1": 0.0075,
        "Bias_1": 0.0075,
        "DenseWeight_0": 0.002,
    }

    capped = derive_raw_learning_rates(
        _probe(),
        rho_conv=0.03,
        rho_dense=0.01,
        bias_policy="capped",
    )
    assert capped["ConvWeight_0"] == attached["ConvWeight_0"]
    assert capped["ConvWeight_1"] == attached["ConvWeight_1"]
    assert capped["DenseWeight_0"] == attached["DenseWeight_0"]
    assert capped["Bias_0"] == pytest.approx(1.0e-4)
    assert capped["Bias_1"] == attached["Bias_1"]
    for bias, weight in BIAS_TO_WEIGHT.items():
        assert capped[bias] <= capped[weight]


def test_effective_sgd_probe_preserves_historical_weight_provenance() -> None:
    measured = {
        "optimizer_name": "SGD",
        "artifact_sha256": "b" * 64,
        "shadow_weight_unit_by_parameter": {
            "ConvWeight_0": 2.0,
            "ConvWeight_1": 4.0,
            "DenseWeight_0": 5.0,
        },
        "normalization_unit_by_weight": {
            "ConvWeight_0": 2.0,
            "ConvWeight_1": 4.0,
            "DenseWeight_0": 5.0,
        },
        "bias_q90_unit_by_parameter": {"Bias_0": 10.0, "Bias_1": 0.01},
    }
    effective = compose_effective_optimizer_probe(
        measured,
        source_weight_units_by_parameter={
            "ConvWeight_0": 2.0,
            "ConvWeight_1": 4.0,
            "DenseWeight_0": 5.0,
        },
        source_weight_probe_sha256="a" * 64,
    )
    assert effective["normalization_unit_by_weight"] == {
        "ConvWeight_0": 2.0,
        "ConvWeight_1": 4.0,
        "DenseWeight_0": 5.0,
    }
    assert effective["weight_normalization_probe_sha256"] == "a" * 64
    assert effective["bias_q90_probe_sha256"] == "b" * 64
    assert (
        effective["weight_normalization_source"]
        == "hash_verified_v5_v6_source_run_spec"
    )
    assert effective["effective_probe_composed"] is True

    mismatched = copy.deepcopy(measured)
    mismatched["shadow_weight_unit_by_parameter"]["ConvWeight_0"] = 3.0
    diagnostic = compose_effective_optimizer_probe(
        mismatched,
        source_weight_units_by_parameter={
            "ConvWeight_0": 2.0,
            "ConvWeight_1": 4.0,
            "DenseWeight_0": 5.0,
        },
        source_weight_probe_sha256="a" * 64,
    )
    assert diagnostic["normalization_unit_by_weight"]["ConvWeight_0"] == 2.0
    assert diagnostic["shadow_vs_authoritative_weight_unit_diagnostics"][
        "ConvWeight_0"
    ] == {"absolute_difference": 1.0, "ratio": 1.5}


def test_effective_adam_probe_uses_fresh_shadow_for_both_statistics() -> None:
    measured = {
        "optimizer_name": "Adam",
        "artifact_sha256": "c" * 64,
        "shadow_weight_unit_by_parameter": {
            "ConvWeight_0": 1.0,
            "ConvWeight_1": 1.0,
            "DenseWeight_0": 1.0,
        },
        "bias_q90_unit_by_parameter": {"Bias_0": 2.0, "Bias_1": 3.0},
    }
    effective = compose_effective_optimizer_probe(measured)
    assert effective["weight_normalization_probe_sha256"] == "c" * 64
    assert effective["bias_q90_probe_sha256"] == "c" * 64
    assert (
        effective["weight_normalization_source"]
        == "fresh_adam_32_shadow_proposals"
    )


def test_complete_optimizer_state_digest_round_trip_includes_adam_state() -> None:
    parameters = [
        _ScientificParameter("ConvWeight_0", 1.0),
        _ScientificParameter("Bias_0", 0.0),
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter.state], "lr": 0.1}
            for parameter in parameters
        ],
        lr=1.0,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        fused=False,
        maximize=False,
        capturable=False,
        differentiable=False,
    )
    before = optimizer_state_digest(optimizer)
    snapshot = capture_parameter_optimizer_state(parameters, optimizer)
    for parameter in parameters:
        parameter.state.grad = torch.ones_like(parameter.state)
    optimizer.step()
    after_step = optimizer_state_digest(optimizer)
    assert after_step != before
    restore_parameter_optimizer_state(parameters, optimizer, snapshot)
    assert optimizer_state_digest(optimizer) == before
    assert parameters[0].state.item() == 1.0
    assert parameters[1].state.item() == 0.0


def test_optimizer_state_digest_fails_on_nonfinite_state() -> None:
    parameter = torch.tensor([1.0], requires_grad=False)
    optimizer = torch.optim.Adam([{"params": [parameter], "lr": 0.1}])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="finite"):
        optimizer_state_digest(optimizer)


def test_replay_emits_explicit_median_keys_and_plot_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = {
        "ConvWeight_0": SimpleNamespace(
            proposed_update_rms=0.02, projection_efficiency=0.9
        ),
        "ConvWeight_1": SimpleNamespace(
            proposed_update_rms=0.04, projection_efficiency=0.8
        ),
        "DenseWeight_0": SimpleNamespace(
            proposed_update_rms=0.08, projection_efficiency=1.0
        ),
        "Bias_0": SimpleNamespace(
            proposed_update_rms=0.01, projection_efficiency=1.0
        ),
        "Bias_1": SimpleNamespace(
            proposed_update_rms=0.01, projection_efficiency=1.0
        ),
    }
    result = SimpleNamespace(
        source_indices=tuple(range(16)),
        loss=0.5,
        accuracy=0.75,
        transition=SimpleNamespace(by_name=items),
    )
    monkeypatch.setattr(runtime_module, "training_step", lambda *args, **kwargs: result)
    monkeypatch.setattr(
        runtime_module, "parameter_tensor_digest", lambda parameters: "p" * 64
    )
    monkeypatch.setattr(
        runtime_module, "optimizer_state_digest", lambda optimizer: "o" * 64
    )
    monkeypatch.setattr(
        runtime_module, "_hidden_saturation", lambda *args, **kwargs: [0.1, 0.2]
    )
    monkeypatch.setattr(
        runtime_module,
        "_displacement",
        lambda *args, **kwargs: {name: 0.0 for name in PARAMETER_NAMES},
    )
    runtime = SimpleNamespace(parameters=[], optimizer=object())
    replay = replay_shadow_updates(
        runtime,
        [object() for _ in range(8)],
        learning_rates_by_parameter={name: 1.0 for name in PARAMETER_NAMES},
        initial_states={},
        initial_weight_rms={
            "ConvWeight_0": 2.0,
            "ConvWeight_1": 4.0,
            "DenseWeight_0": 8.0,
        },
        checkpoint_label="initialization",
        v_off=4.0,
    )
    expected_rho = {
        "ConvWeight_0": 0.01,
        "ConvWeight_1": 0.01,
        "DenseWeight_0": 0.01,
    }
    assert replay["achieved_weight_rho_median_by_parameter"] == expected_rho
    assert replay["achieved_weight_rho_by_parameter"] == expected_rho
    assert replay["bias_weight_step_ratio_median_by_parameter"] == {
        "Bias_0": 0.5,
        "Bias_1": 0.25,
    }
    assert replay["bias_weight_step_ratio_by_bias"] == {
        "Bias_0": 0.5,
        "Bias_1": 0.25,
    }
    assert replay["hidden_saturation_mean_by_layer"] == [0.1, 0.2]
    assert replay["restoration_verified_after_every_shadow_step"] is True


def _fake_reuse_cells() -> list[dict]:
    rows = {
        "conv2_baseline_v1_c1": (0.01, 0.03),
        "conv2_ours_v4_c1": (0.003, 0.01),
        "conv2_legacy_v4_c0p25": (0.003, 0.01),
    }
    cells = []
    for row_id, dense_values in rows.items():
        for rho_dense in dense_values:
            coordinate = (row_id, 0.01, rho_dense)
            source_entry_id, source_collection_id = (
                _EXPECTED_REUSE_SOURCE_BY_COORDINATE[coordinate]
            )
            run_spec_name = (
                "run_spec.v5.json"
                if row_id == "conv2_baseline_v1_c1"
                else "run_spec.v6.json"
            )
            names = [
                "best_validation.pt",
                "final.pt",
                "minibatches.json",
                "parameter_diagnostics.csv",
                run_spec_name,
                "step_log.csv",
                "summary.json",
                "validation.json",
            ]
            outputs = [
                {"name": name, "bytes": 1, "sha256": "a" * 64}
                for name in names
            ]
            cells.append(
                {
                    "row_id": row_id,
                    "rho_conv": 0.01,
                    "rho_dense": rho_dense,
                    "source_entry_id": source_entry_id,
                    "source_collection_id": source_collection_id,
                    "expected_completed_steps": 17_190,
                    "run_spec_sha256": "a" * 64,
                    "candidate_summary_sha256": "a" * 64,
                    "completion_sha256": "b" * 64,
                    "completion_bytes": 1,
                    "outputs": outputs,
                }
            )
    return cells


def test_v7_run_payload_is_accepted_by_runspec(tmp_path: Path) -> None:
    study = optimizer_boundary_study_template(_fake_reuse_cells())
    bundle = SimpleNamespace(
        batch_size=16,
        validation_indices_hash=(
            "4801b7805d54dbe2197b98320cfe8a34d5d853c4ab568bcf1d7f69e136c94bb4"
        ),
    )
    assets = BoundaryAssets(
        study_dir=tmp_path,
        checkpoint_path=tmp_path / "initialization/conv2.pt",
        initialization={
            "checkpoint_sha256": (
                "dc16ef2bebd2c9e6afe307b41555683c3eec29b9481ab64ea3f412382c837c7e"
            ),
            "parameter_tensor_sha256": (
                "e9aa471dc5558e525758b345de12c53f0a086c5a1b022024c2fc349ec138e2e3"
            ),
        },
        bundle=bundle,
        probe_batches=(),
        minibatches={
            "normalization_batch_order_sha256": (
                "1c65076487a87e0badfe19b80edc8263f50c4b765dc317299dec4a8f1fad460b"
            ),
            "replay_batch_order_sha256": (
                "1d67f374dea1d58ed7d779ed5eb14471fce3b6c64e614c47f47fda5526bc5673"
            ),
        },
        minibatches_path=tmp_path / "probes/minibatches.json",
    )
    probe = {
        "normalization_unit_by_weight": {
            "ConvWeight_0": 1.0,
            "ConvWeight_1": 1.0,
            "DenseWeight_0": 1.0,
        },
        "bias_q90_unit_by_parameter": {
            "Bias_0": 1.0,
            "Bias_1": 1.0,
        },
        "artifact_sha256": "c" * 64,
    }
    entry = {
        "entry_id": "main_grid--ours--adam--c0p015--d0p003",
        "row_id": "conv2_ours_v4_c1",
        "scheme": "ours",
        "optimizer": "adam",
        "stage": "main_grid",
        "bias_policy": "attached",
        "mode": "new_five_epoch_training",
        "rho_conv": 0.015,
        "rho_dense": 0.003,
        "raw_learning_rates_by_parameter": {
            "ConvWeight_0": 0.015,
            "Bias_0": 0.015,
            "ConvWeight_1": 0.015,
            "Bias_1": 0.015,
            "DenseWeight_0": 0.003,
        },
        "probe_sha256": "c" * 64,
        "parent_sha256": "d" * 64,
        "code_fingerprint": "e" * 64,
    }
    payload = build_boundary_run_payload(
        study,
        assets,
        entry,
        probe,
        training_batch_order_sha256=(
            "e774de39fe6f9529dafded0b14356727b3e8510bd42f3a92f7416dcab0592d14"
        ),
    )
    assert payload["schema_version"] == "mnist-conv-run/v7"
    assert payload["run"]["training"]["optimizer"]["name"] == "Adam"
    assert payload["run"]["lr_provenance"]["official_test_read"] is False


def test_immutable_json_rejects_conflicting_republish(tmp_path: Path) -> None:
    path = tmp_path / "immutable.json"
    runtime_module._publish_immutable_json(path, {"value": 1})
    runtime_module._publish_immutable_json(path, {"value": 1})
    with pytest.raises(BoundaryRuntimeError, match="conflicting"):
        runtime_module._publish_immutable_json(path, {"value": 2})


def test_incomplete_adam_restart_allocates_fresh_attempt_without_touching_stale_base(
    tmp_path: Path,
) -> None:
    base = tmp_path / "stages/main_grid/entries/example"
    base.mkdir(parents=True)
    stale_result = base / "result.json"
    stale_checkpoint = base / "final.pt"
    stale_result.write_bytes(b"stale-result\n")
    stale_checkpoint.write_bytes(b"stale-checkpoint\n")
    entry = {
        "optimizer": "adam",
        "resume_action": "restart_from_initialization",
    }

    first, first_relative = runtime_module._allocate_attempt_output_dir(
        base, entry
    )
    second, second_relative = runtime_module._allocate_attempt_output_dir(
        base, entry
    )

    assert first_relative == "attempts/attempt-000001"
    assert second_relative == "attempts/attempt-000002"
    assert first.is_dir()
    assert second.is_dir()
    assert stale_result.read_bytes() == b"stale-result\n"
    assert stale_checkpoint.read_bytes() == b"stale-checkpoint\n"


def _post_training_safety_result() -> dict:
    return {
        "entry_id": entry_id(
            scheme="ours",
            optimizer="adam",
            stage="main_grid",
            rho_conv=0.015,
            rho_dense=0.003,
        ),
        "scheme": "ours",
        "optimizer": "adam",
        "stage": "main_grid",
        "rho_conv": 0.015,
        "rho_dense": 0.003,
        "status": "safety_failure",
        "completed_steps": 17_190,
        "admissible": False,
        "safety_failure": {
            "kind": "non_finite_epoch_validation_or_replay",
            "attempted_step": 17_190,
            "epoch": 5,
            "detail": "non-finite replay",
        },
        "optimizer_parameters": {
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
        },
        "raw_learning_rates_by_parameter": {
            name: 0.001 for name in PARAMETER_NAMES
        },
        "normalization_unit_by_weight": {
            "ConvWeight_0": 1.0,
            "ConvWeight_1": 1.0,
            "DenseWeight_0": 1.0,
        },
        "bias_q90_unit_by_parameter": {
            "Bias_0": 1.0,
            "Bias_1": 1.0,
        },
        "probe_sha256": "1" * 64,
        "bias_q90_probe_sha256": "2" * 64,
        "parent_sha256": "3" * 64,
        "parent_study_config_sha256": "7" * 64,
        "parent_entry_completion_sha256": None,
        "parent_decision_sha256": None,
        "minibatch_sha256": "4" * 64,
        "checkpoint_sha256": "5" * 64,
        "code_fingerprint": "6" * 64,
        "official_test_read": False,
    }


def test_post_training_epoch5_replay_failure_is_terminal_but_inadmissible() -> None:
    normalized = validate_result(_post_training_safety_result())
    assert normalized["completed_steps"] == 17_190
    assert normalized["status"] == "safety_failure"
    assert normalized["admissible"] is False


def test_other_safety_failure_cannot_claim_all_optimizer_steps() -> None:
    result = _post_training_safety_result()
    result["safety_failure"] = {
        "kind": "non_finite_optimizer_transition",
        "attempted_step": 17_190,
        "epoch": 5,
    }
    with pytest.raises(ValueError, match="explicit non-finite"):
        validate_result(result)
