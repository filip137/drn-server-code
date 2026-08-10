from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import experiments.conv3_operating_point_gate as conv3_gate
import experiments.run_conv12_bounded_rho as runner
from experiments.reporting import complete_run, start_run


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_PATH = (
    REPO_ROOT
    / "configs"
    / "conv"
    / "perfectdiode_bounded_uniform_zero_bias_lr_search_conv123_seed0_20260810_v1.json"
)


def test_study_declares_exact_eighteen_zero_bias_surfaces() -> None:
    _path, study = runner.load_study(STUDY_PATH)
    surfaces = runner.surface_specs(study)

    assert len(surfaces) == 18
    assert {row["architecture"] for row in surfaces} == {
        "conv1",
        "conv2",
        "conv3",
    }
    assert {row["scheme"] for row in surfaces} == {
        "baseline",
        "ours",
        "legacy",
    }
    assert {row["optimizer"] for row in surfaces} == {"SGD", "Adam"}
    assert {row["initializer"] for row in surfaces} == {"bounded_uniform"}
    assert study["bias_contract"] == {
        "initialization": "default_zero",
        "learning_rate": 0.0,
        "conductance_projection": False,
    }
    assert study["rho_search"]["bias_policy"] == "zero"
    assert study["rho_search"]["select_best_safe_below_accuracy"] is False


def test_surface_policies_keep_conv3_high_grid_and_hard_boundary_gates() -> None:
    _path, study = runner.load_study(STUDY_PATH)
    surfaces = runner.surface_specs(study)
    by_key = {
        (row["architecture"], row["scheme"], row["optimizer"]): row
        for row in surfaces
    }

    baseline = runner._surface_search(
        study, by_key[("conv3", "baseline", "SGD")]
    )
    legacy = runner._surface_search(study, by_key[("conv3", "legacy", "SGD")])
    conv2 = runner._surface_search(study, by_key[("conv2", "baseline", "SGD")])

    assert baseline["core_mode"] == "fixed_grid"
    assert baseline["fixed_core"] == {
        "rho_conv": [0.009, 0.027, 0.081],
        "rho_dense": [0.03, 0.09, 0.27],
    }
    assert baseline["safety"] == {
        **study["rho_search"]["safety"],
        "bound_occupancy": "reject_persistent_increase",
        "projection_efficiency": "reject_persistent_low_efficiency",
        "bound_occupancy_increase_maximum": 0.2,
        "projection_efficiency_minimum": 0.5,
        "zero_proposal_epsilon": 1e-12,
        "boundary_persistence_steps": 16,
    }
    assert legacy["core_mode"] == "adaptive_safe_center"
    assert legacy["safety"] == baseline["safety"]
    assert conv2["core_mode"] == "adaptive_safe_center"
    assert "bound_occupancy_increase_maximum" not in conv2["safety"]


@pytest.mark.parametrize(
    ("architecture", "expected_rates"),
    (("conv1", [1.0, 1.0, 0.0]), ("conv2", [1.0, 1.0, 1.0, 0.0, 0.0])),
)
def test_source_config_starts_with_zero_bias_rates(
    tmp_path: Path,
    architecture: str,
    expected_rates: list[float],
) -> None:
    _path, study = runner.load_study(STUDY_PATH)
    config = runner.build_source_config(
        study,
        initializer="bounded_uniform",
        architecture=architecture,
        scheme="legacy",
        optimizer="Adam",
        init_checkpoint_path=tmp_path / "initial.pt",
        dataset_root=tmp_path / "mnist",
    )

    assert config["lr"] == expected_rates
    assert config["optimizer"]["learning_rate"] == expected_rates
    assert config["model_base"]["voltage_amp"] == 4.0
    assert config["model_base"]["current_amp"] == 0.25
    assert config["model_base"]["weight_init_mode"] == "bounded_uniform"
    assert config["model_base"]["weight_min"] == 1e-5
    assert config["model_base"]["weight_max"] == 1e-4


def test_conv3_gate_is_cached_across_optimizers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, study = runner.load_study(STUDY_PATH)
    checkpoint = tmp_path / "initial.pt"
    checkpoint.write_bytes(b"shared-conv3-checkpoint")
    calls: list[tuple[Path, Path, bool]] = []

    def fake_gate(
        source_config_path,
        checkpoint_path,
        *,
        device,
        output_path,
        smoke,
    ):
        del device, output_path
        source = Path(source_config_path)
        checkpoint_value = Path(checkpoint_path)
        calls.append((source, checkpoint_value, smoke))
        return {
            "schema_version": "test-conv3-gate/v1",
            "status": "complete",
            "security_passed": True,
            "scientifically_complete": not smoke,
            "smoke": smoke,
            "source_config_sha256": runner._sha256_file(source),
            "checkpoint_sha256": runner._sha256_file(checkpoint_value),
            "official_test_read": False,
        }

    monkeypatch.setattr(conv3_gate, "run_conv3_operating_point_gate", fake_gate)
    surface = {
        "initializer": "bounded_uniform",
        "architecture": "conv3",
        "scheme": "ours",
        "optimizer": "SGD",
    }
    first = runner.run_conv3_tk_operating_point_gate(
        study,
        tmp_path / "results",
        surface,
        checkpoint,
        device="cpu",
        smoke=False,
        dataset_root=tmp_path / "mnist",
    )
    surface["optimizer"] = "Adam"
    second = runner.run_conv3_tk_operating_point_gate(
        study,
        tmp_path / "results",
        surface,
        checkpoint,
        device="cpu",
        smoke=False,
        dataset_root=tmp_path / "mnist",
    )

    assert first == second
    assert len(calls) == 1
    source_config = json.loads(calls[0][0].read_text(encoding="utf-8"))
    assert source_config["optimizer"]["name"] == "SGD"
    assert all(rate == 0.0 for rate in source_config["lr"])


def _conv3_cohort(*, smoke: bool = False) -> dict:
    gradient_examples = 32 if smoke else 256
    residual_examples = 64 if smoke else 1024
    return {
        "source_split": "mnist_train_55000_subset",
        "split_seed": 0,
        "shuffle_seed": 0,
        "train_indices_sha256": "train",
        "validation_indices_sha256": "validation",
        "examples": residual_examples,
        "batch_size": 64,
        "cohort_original_indices_sha256": "residual-indices",
        "cohort_tensor_sha256": "residual-tensors",
        "gradient_prefix_examples": gradient_examples,
        "gradient_batch_size": 32,
        "gradient_prefix_original_indices_sha256": "gradient-indices",
        "gradient_prefix_tensor_sha256": "gradient-tensors",
    }


def _conv3_operating_gate(*, smoke: bool = False) -> dict:
    return {
        "schema_version": (
            "perfectdiode-conv3-bounded-uniform-operating-point-gate/v1"
        ),
        "status": "complete",
        "security_passed": True,
        "scientifically_complete": not smoke,
        "checkpoint_unchanged": True,
        "checkpoint_sha256": "init",
        "checkpoint_sha256_after_replay": "init",
        "dataset_cohort": _conv3_cohort(smoke=smoke),
        "smoke": smoke,
        "official_test_read": False,
    }


def _write_candidate_epoch_bundle(cell_dir: Path, epochs: int = 3) -> dict:
    cell_dir.mkdir(parents=True)
    (cell_dir / "source_config.json").write_text("{}\n", encoding="utf-8")
    checkpoint_dir = cell_dir / "checkpoints"
    checkpoint_dir.mkdir()
    index_rows = []
    for epoch in range(epochs + 1):
        model = checkpoint_dir / f"epoch_{epoch:03d}_model.pt"
        optimizer = checkpoint_dir / f"epoch_{epoch:03d}_optimizer.pt"
        model.write_bytes(f"model-{epoch}".encode())
        optimizer.write_bytes(f"optimizer-{epoch}".encode())
        index_rows.append(
            {
                "epoch": epoch,
                "model_path": str(model.relative_to(cell_dir)),
                "model_size_bytes": model.stat().st_size,
                "model_sha256": runner._sha256_file(model),
                "optimizer_path": str(optimizer.relative_to(cell_dir)),
                "optimizer_size_bytes": optimizer.stat().st_size,
                "optimizer_sha256": runner._sha256_file(optimizer),
            }
        )
    (cell_dir / "epoch_checkpoint_index.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    np.save(cell_dir / "loss_test.npy", np.linspace(0.3, 0.1, epochs))
    np.save(cell_dir / "accuracy_test.npy", np.linspace(0.91, 0.95, epochs))
    metrics = {
        "checkpoint_every_epoch": True,
        "epoch_checkpoint_count": epochs + 1,
        "final_test_loss": 0.1,
        "final_test_accuracy": 0.95,
    }
    runner._write_json(cell_dir / "metrics.json", metrics)
    cell = {
        "index": 0,
        "rho_conv": 0.009,
        "rho_dense": 0.03,
        "status": "complete",
        "selection_eligible": True,
        "training_accuracy_selection_eligible": True,
        "signature": {"cell": "signature"},
    }
    runner._write_json(cell_dir / "cell.json", cell)
    return cell


def _complete_candidate_bundle(
    cell_dir: Path, cell: dict, gate_summary: dict
) -> dict:
    gate_record = {
        "name": "conv3_epochwise_k64_viability",
        "status": gate_summary["status"],
        "passed": gate_summary["passed"],
        "artifact": gate_summary["summary_path"],
        "artifact_sha256": gate_summary["summary_sha256"],
    }
    cell = dict(cell)
    cell.update(
        status="complete",
        selection_eligible=True,
        training_accuracy_selection_eligible=True,
        post_candidate_gate=gate_record,
    )
    runner._write_json(cell_dir / "cell.json", cell)
    start_run(
        cell_dir,
        {
            "study_id": "test-conv3-post-tk",
            "run_id": cell_dir.name,
            "arm_id": cell_dir.name,
            "evidence_class": "test",
            "dataset": {"official_test_read": False},
        },
    )
    complete_run(
        cell_dir,
        terminal_metrics={"final_validation_accuracy": 0.95},
        completion={
            "criteria_met": True,
            "rho_cell_complete": True,
            "safety_admissible": True,
            "official_test_read": False,
            "post_candidate_gate": gate_record,
            "post_training_tk_admissible": True,
        },
    )
    return cell


def _passing_k64_gate(source, checkpoint, *, device, output_path, smoke):
    del device
    result = {
        "schema_version": (
            "perfectdiode-conv3-bounded-uniform-k64-gradient-viability/v1"
        ),
        "status": "complete",
        "viability_passed": True,
        "scientifically_complete": not smoke,
        "checkpoint_unchanged": True,
        "checkpoint_sha256": runner._sha256_file(Path(checkpoint)),
        "checkpoint_sha256_after_replay": runner._sha256_file(Path(checkpoint)),
        "source_config_sha256": runner._sha256_file(Path(source)),
        "dataset_cohort": _conv3_cohort(smoke=smoke),
        "smoke": smoke,
        "official_test_read": False,
    }
    runner._write_json(Path(output_path), result)
    return result


def _passing_full_gate(source, checkpoint, *, device, output_path, smoke):
    del device
    result = {
        "schema_version": (
            "perfectdiode-conv3-bounded-uniform-operating-point-gate/v1"
        ),
        "status": "complete",
        "security_passed": True,
        "scientifically_complete": not smoke,
        "checkpoint_unchanged": True,
        "checkpoint_sha256": runner._sha256_file(Path(checkpoint)),
        "checkpoint_sha256_after_replay": runner._sha256_file(Path(checkpoint)),
        "source_config_sha256": runner._sha256_file(Path(source)),
        "dataset_cohort": _conv3_cohort(smoke=smoke),
        "smoke": smoke,
        "official_test_read": False,
    }
    runner._write_json(Path(output_path), result)
    return result


def _prepare_complete_conv3_post_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    failed_full_epoch: int | None = None,
) -> tuple[dict, dict, Path, dict, dict, Path, Path]:
    _path, study = runner.load_study(STUDY_PATH)
    surface = next(
        row
        for row in runner.surface_specs(study)
        if row["architecture"] == "conv3"
        and row["scheme"] == "baseline"
        and row["optimizer"] == "SGD"
    )
    surface_dir = tmp_path / "surfaces" / surface["surface_id"]
    cell_dir = surface_dir / "rho" / "cells" / "cell"
    cell = _write_candidate_epoch_bundle(cell_dir)
    operating_gate = _conv3_operating_gate()
    monkeypatch.setattr(
        conv3_gate,
        "run_conv3_k64_gradient_viability_gate",
        _passing_k64_gate,
    )
    viability = runner.ensure_conv3_candidate_epochwise_viability(
        study,
        surface,
        cell_dir,
        cell,
        device="cpu",
        smoke=False,
        operating_point_gate=operating_gate,
    )
    cell = _complete_candidate_bundle(cell_dir, cell, viability)
    selected = runner._candidate_record(cell_dir, cell)
    if failed_full_epoch is None:
        full_gate = _passing_full_gate
    else:
        def full_gate(source, checkpoint, *, device, output_path, smoke):
            result = _passing_full_gate(
                source,
                checkpoint,
                device=device,
                output_path=output_path,
                smoke=smoke,
            )
            epoch = int(Path(checkpoint).stem.split("_")[1])
            if epoch == failed_full_epoch:
                result.update(status="unresolved_tk_residual", security_passed=False)
                runner._write_json(Path(output_path), result)
            return result

    monkeypatch.setattr(conv3_gate, "run_conv3_operating_point_gate", full_gate)
    post_path = surface_dir / "selected_post_training_tk.json"
    post = runner.run_conv3_selected_post_training_tk(
        study,
        surface,
        selected,
        operating_gate,
        output_path=post_path,
        device="cpu",
    )
    post_passed = post["passed"] is True
    selection = {
        "schema_version": runner._artifact_schema(study, "surface"),
        **surface,
        "status": "complete" if post_passed else "unresolved_post_training_tk",
        "selection": {
            "selected": selected if post_passed else None,
            "post_training_tk_passed": post_passed,
            **(
                {}
                if post_passed
                else {"pre_post_training_tk_selected": selected}
            ),
        },
        "candidates": [selected],
        "selected": selected if post_passed else None,
        "pre_post_training_tk_selected": selected,
        "post_training_tk": post,
        "official_test_read": False,
    }
    selection_path = surface_dir / "selection.json"
    runner._write_json(selection_path, selection)
    operating_gate_path = tmp_path / "fixed_tk" / "result.json"
    runner._write_json(operating_gate_path, operating_gate)
    return (
        study,
        surface,
        cell_dir,
        selected,
        operating_gate,
        selection_path,
        operating_gate_path,
    )


def test_conv3_candidate_requires_k64_viability_at_all_three_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, study = runner.load_study(STUDY_PATH)
    surface = next(
        row
        for row in runner.surface_specs(study)
        if row["architecture"] == "conv3"
        and row["scheme"] == "baseline"
        and row["optimizer"] == "SGD"
    )
    cell_dir = tmp_path / "cell"
    cell = _write_candidate_epoch_bundle(cell_dir)
    operating_gate = _conv3_operating_gate()
    calls = []

    def fake_viability(source, checkpoint, *, device, output_path, smoke):
        del device
        calls.append(Path(checkpoint).name)
        result = {
            "schema_version": (
                "perfectdiode-conv3-bounded-uniform-k64-gradient-viability/v1"
            ),
            "status": "complete",
            "viability_passed": True,
            "scientifically_complete": True,
            "checkpoint_unchanged": True,
            "checkpoint_sha256": runner._sha256_file(Path(checkpoint)),
            "checkpoint_sha256_after_replay": runner._sha256_file(Path(checkpoint)),
            "source_config_sha256": runner._sha256_file(Path(source)),
            "dataset_cohort": _conv3_cohort(),
            "smoke": smoke,
            "official_test_read": False,
        }
        runner._write_json(Path(output_path), result)
        return result

    monkeypatch.setattr(
        conv3_gate, "run_conv3_k64_gradient_viability_gate", fake_viability
    )

    result = runner.ensure_conv3_candidate_epochwise_viability(
        study,
        surface,
        cell_dir,
        cell,
        device="cpu",
        smoke=False,
        operating_point_gate=operating_gate,
    )

    assert result["passed"] is True
    assert calls == [
        "epoch_001_model.pt",
        "epoch_002_model.pt",
        "epoch_003_model.pt",
    ]
    assert [row["epoch"] for row in result["epochs"]] == [1, 2, 3]
    assert all(row["cohort_matches_operating_point"] for row in result["epochs"])
    summary = cell_dir / "epochwise_k64_viability.json"
    assert result["summary_sha256"] == runner._sha256_file(summary)
    candidate = runner._candidate_record(cell_dir, cell)
    assert candidate["selection_eligible"] is True
    assert candidate["epochwise_conv_gradient_viability"]["checked_epochs"] == [
        1,
        2,
        3,
    ]


def test_selected_conv3_replays_every_epoch_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, study = runner.load_study(STUDY_PATH)
    surface = next(
        row
        for row in runner.surface_specs(study)
        if row["architecture"] == "conv3"
        and row["scheme"] == "baseline"
        and row["optimizer"] == "SGD"
    )
    cell_dir = tmp_path / "cell"
    cell = _write_candidate_epoch_bundle(cell_dir)
    operating_gate = _conv3_operating_gate()

    def fake_viability(source, checkpoint, *, device, output_path, smoke):
        del device
        result = {
            "schema_version": (
                "perfectdiode-conv3-bounded-uniform-k64-gradient-viability/v1"
            ),
            "status": "complete",
            "viability_passed": True,
            "scientifically_complete": True,
            "checkpoint_unchanged": True,
            "checkpoint_sha256": runner._sha256_file(Path(checkpoint)),
            "checkpoint_sha256_after_replay": runner._sha256_file(Path(checkpoint)),
            "source_config_sha256": runner._sha256_file(Path(source)),
            "dataset_cohort": _conv3_cohort(),
            "smoke": smoke,
            "official_test_read": False,
        }
        runner._write_json(Path(output_path), result)
        return result

    monkeypatch.setattr(
        conv3_gate, "run_conv3_k64_gradient_viability_gate", fake_viability
    )
    viability = runner.ensure_conv3_candidate_epochwise_viability(
        study,
        surface,
        cell_dir,
        cell,
        device="cpu",
        smoke=False,
        operating_point_gate=operating_gate,
    )
    cell = _complete_candidate_bundle(cell_dir, cell, viability)
    calls = []

    def fake_full_gate(source, checkpoint, *, device, output_path, smoke):
        del device
        epoch = int(Path(checkpoint).stem.split("_")[1])
        calls.append(epoch)
        passed = epoch != 2
        result = {
            "schema_version": (
                "perfectdiode-conv3-bounded-uniform-operating-point-gate/v1"
            ),
            "status": "complete" if passed else "unresolved_tk_residual",
            "security_passed": passed,
            "scientifically_complete": True,
            "checkpoint_unchanged": True,
            "checkpoint_sha256": runner._sha256_file(Path(checkpoint)),
            "checkpoint_sha256_after_replay": runner._sha256_file(Path(checkpoint)),
            "source_config_sha256": runner._sha256_file(Path(source)),
            "dataset_cohort": _conv3_cohort(),
            "smoke": smoke,
            "official_test_read": False,
        }
        runner._write_json(Path(output_path), result)
        return result

    monkeypatch.setattr(conv3_gate, "run_conv3_operating_point_gate", fake_full_gate)
    selected = {
        "cell_id": "cell",
        "path": str(cell_dir),
        "rho_conv": 0.009,
        "rho_dense": 0.03,
    }
    output = tmp_path / "surface" / "selected_post_training_tk.json"

    result = runner.run_conv3_selected_post_training_tk(
        study,
        surface,
        selected,
        operating_gate,
        output_path=output,
        device="cpu",
    )

    assert calls == [1, 2, 3]
    assert result["status"] == "unresolved_post_training_tk"
    assert result["passed"] is False
    assert [row["passed"] for row in result["epochs"]] == [True, False, True]
    assert all(Path(row["gate_artifact"]).is_file() for row in result["epochs"])
    assert all(
        row["gate_artifact_sha256"]
        == runner._sha256_file(Path(row["gate_artifact"]))
        for row in result["epochs"]
    )
    assert result["summary_sha256"] == runner._sha256_file(output)


def test_conv3_candidate_gate_runtime_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, study = runner.load_study(STUDY_PATH)
    surface = next(
        row
        for row in runner.surface_specs(study)
        if row["architecture"] == "conv3"
        and row["scheme"] == "baseline"
        and row["optimizer"] == "SGD"
    )
    cell_dir = tmp_path / "cell"
    cell = _write_candidate_epoch_bundle(cell_dir)

    def broken_gate(*_args, **_kwargs):
        raise OSError("simulated checkpoint I/O failure")

    monkeypatch.setattr(
        conv3_gate, "run_conv3_k64_gradient_viability_gate", broken_gate
    )
    with pytest.raises(OSError, match="checkpoint I/O failure"):
        runner.ensure_conv3_candidate_epochwise_viability(
            study,
            surface,
            cell_dir,
            cell,
            device="cpu",
            smoke=False,
            operating_point_gate=_conv3_operating_gate(),
        )
    assert not (cell_dir / "epochwise_k64_viability.json").exists()


def test_conv3_candidate_accepts_well_formed_negative_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, study = runner.load_study(STUDY_PATH)
    surface = next(
        row
        for row in runner.surface_specs(study)
        if row["architecture"] == "conv3"
        and row["scheme"] == "baseline"
        and row["optimizer"] == "SGD"
    )
    cell_dir = tmp_path / "cell"
    cell = _write_candidate_epoch_bundle(cell_dir)

    def negative_gate(source, checkpoint, *, device, output_path, smoke):
        result = _passing_k64_gate(
            source,
            checkpoint,
            device=device,
            output_path=output_path,
            smoke=smoke,
        )
        result.update(
            status="unresolved_tk_gradient_viability",
            viability_passed=False,
        )
        runner._write_json(Path(output_path), result)
        return result

    monkeypatch.setattr(
        conv3_gate, "run_conv3_k64_gradient_viability_gate", negative_gate
    )
    result = runner.ensure_conv3_candidate_epochwise_viability(
        study,
        surface,
        cell_dir,
        cell,
        device="cpu",
        smoke=False,
        operating_point_gate=_conv3_operating_gate(),
    )
    assert result["status"] == "unresolved_tk_gradient_viability"
    assert result["passed"] is False
    assert all(row["passed"] is False for row in result["epochs"])
    assert all(row["error"] is None for row in result["epochs"])


def test_conv3_existing_candidate_summary_rejects_tampered_detailed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, study = runner.load_study(STUDY_PATH)
    surface = next(
        row
        for row in runner.surface_specs(study)
        if row["architecture"] == "conv3"
        and row["scheme"] == "baseline"
        and row["optimizer"] == "SGD"
    )
    cell_dir = tmp_path / "cell"
    cell = _write_candidate_epoch_bundle(cell_dir)
    monkeypatch.setattr(
        conv3_gate,
        "run_conv3_k64_gradient_viability_gate",
        _passing_k64_gate,
    )
    runner.ensure_conv3_candidate_epochwise_viability(
        study,
        surface,
        cell_dir,
        cell,
        device="cpu",
        smoke=False,
        operating_point_gate=_conv3_operating_gate(),
    )
    receipt = (
        cell_dir
        / "post_training_tk"
        / "epochwise_k64_viability"
        / "epoch_002.json"
    )
    receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt hash mismatch"):
        runner.ensure_conv3_candidate_epochwise_viability(
            study,
            surface,
            cell_dir,
            cell,
            device="cpu",
            smoke=False,
            operating_point_gate=_conv3_operating_gate(),
        )


def test_run_rho_cell_strictly_revalidates_reused_conv3_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, study = runner.load_study(STUDY_PATH)
    surface = next(
        row
        for row in runner.surface_specs(study)
        if row["architecture"] == "conv3"
        and row["scheme"] == "baseline"
        and row["optimizer"] == "SGD"
    )
    rho_conv = 0.009
    rho_dense = 0.03
    rho_root = tmp_path / "rho"
    cell_dir = (
        rho_root
        / "cells"
        / (
            f"000_rc_{runner._float_token(rho_conv)}_"
            f"rd_{runner._float_token(rho_dense)}"
        )
    )
    cell = _write_candidate_epoch_bundle(cell_dir)
    operating_gate = _conv3_operating_gate()
    monkeypatch.setattr(
        conv3_gate,
        "run_conv3_k64_gradient_viability_gate",
        _passing_k64_gate,
    )
    viability = runner.ensure_conv3_candidate_epochwise_viability(
        study,
        surface,
        cell_dir,
        cell,
        device="cpu",
        smoke=False,
        operating_point_gate=operating_gate,
    )
    _complete_candidate_bundle(cell_dir, cell, viability)
    receipt = (
        cell_dir
        / "post_training_tk"
        / "epochwise_k64_viability"
        / "epoch_003.json"
    )
    receipt.write_text(receipt.read_text(encoding="utf-8") + " ", encoding="utf-8")
    monkeypatch.setattr(
        runner, "run_rho_search", lambda _args: {"status": "complete"}
    )
    with pytest.raises(RuntimeError, match="receipt hash mismatch"):
        runner._run_rho_cell(
            study,
            surface,
            cell_dir / "source_config.json",
            rho_root,
            [rho_conv],
            [rho_dense],
            rho_conv,
            rho_dense,
            device="cpu",
            canary_only=False,
            operating_point_gate=operating_gate,
        )


def test_selected_conv3_gate_runtime_error_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, surface, cell_dir, selected, operating_gate, _selection, _gate = (
        _prepare_complete_conv3_post_bundle(tmp_path, monkeypatch)
    )
    output = tmp_path / "fresh_surface" / "selected_post_training_tk.json"

    def broken_gate(*_args, **_kwargs):
        raise RuntimeError("simulated CUDA replay failure")

    monkeypatch.setattr(conv3_gate, "run_conv3_operating_point_gate", broken_gate)
    with pytest.raises(RuntimeError, match="CUDA replay failure"):
        runner.run_conv3_selected_post_training_tk(
            study,
            surface,
            selected,
            operating_gate,
            output_path=output,
            device="cpu",
        )
    assert not output.exists()
    assert cell_dir.is_dir()


def test_conv3_public_post_tk_validator_accepts_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        _study,
        surface,
        _cell_dir,
        _selected,
        _operating_gate,
        selection_path,
        operating_gate_path,
    ) = _prepare_complete_conv3_post_bundle(tmp_path, monkeypatch)
    result = runner.validate_conv3_selected_post_training_tk_evidence(
        STUDY_PATH,
        selection_path,
        operating_gate_path,
    )
    assert result["status"] == "valid"
    assert result["surface_id"] == surface["surface_id"]
    assert result["post_training_tk_passed"] is True
    assert result["epoch_count"] == 3


def test_conv3_public_post_tk_validator_accepts_well_formed_negative_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        _study,
        _surface,
        _cell_dir,
        _selected,
        _operating_gate,
        selection_path,
        operating_gate_path,
    ) = _prepare_complete_conv3_post_bundle(
        tmp_path,
        monkeypatch,
        failed_full_epoch=2,
    )
    result = runner.validate_conv3_selected_post_training_tk_evidence(
        STUDY_PATH,
        selection_path,
        operating_gate_path,
    )
    assert result["status"] == "valid"
    assert result["post_training_tk_passed"] is False
    assert result["epoch_count"] == 3


@pytest.mark.parametrize(
    "tamper",
    [
        "candidate_receipt",
        "selected_receipt",
        "checkpoint",
        "result_index",
        "selected_path",
        "candidate_membership",
    ],
)
def test_conv3_public_post_tk_validator_rejects_missing_or_tampered_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    (
        _study,
        _surface,
        cell_dir,
        _selected,
        _operating_gate,
        selection_path,
        operating_gate_path,
    ) = _prepare_complete_conv3_post_bundle(tmp_path, monkeypatch)
    if tamper == "candidate_receipt":
        target = (
            cell_dir
            / "post_training_tk"
            / "epochwise_k64_viability"
            / "epoch_001.json"
        )
        target.unlink()
    elif tamper == "selected_receipt":
        target = (
            selection_path.parent
            / "post_training_tk"
            / "selected_epochs"
            / "epoch_002.json"
        )
        target.write_text(target.read_text(encoding="utf-8") + " ", encoding="utf-8")
    elif tamper == "checkpoint":
        target = cell_dir / "checkpoints" / "epoch_003_model.pt"
        target.write_bytes(target.read_bytes() + b"tampered")
    elif tamper == "result_index":
        result_path = cell_dir / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["artifacts"] = [
            row
            for row in result["artifacts"]
            if row.get("path") != "artifacts/epochwise_k64_viability.json"
        ]
        runner._write_json(result_path, result)
        status_path = cell_dir / "status.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        status["result_sha256"] = runner._sha256_file(result_path)
        runner._write_json(status_path, status)
    elif tamper == "selected_path":
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        wrong_path = str(tmp_path / "untrusted-other-run")
        selection["selected"]["path"] = wrong_path
        selection["pre_post_training_tk_selected"]["path"] = wrong_path
        selection["selection"]["selected"]["path"] = wrong_path
        selection["candidates"][0]["path"] = wrong_path
        runner._write_json(selection_path, selection)
    else:
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selection["candidates"] = []
        runner._write_json(selection_path, selection)
    with pytest.raises(RuntimeError):
        runner.validate_conv3_selected_post_training_tk_evidence(
            STUDY_PATH,
            selection_path,
            operating_gate_path,
        )


def test_conv3_surface_with_failed_selected_replay_has_no_final_lr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _path, study = runner.load_study(STUDY_PATH)
    surface = next(
        row
        for row in runner.surface_specs(study)
        if row["architecture"] == "conv3"
        and row["scheme"] == "baseline"
        and row["optimizer"] == "SGD"
    )
    monkeypatch.setattr(
        runner, "run_rho_search", lambda _args: {"status": "complete"}
    )
    calls = []

    def fake_cell(
        _study,
        _surface,
        _source,
        rho_root,
        _conv_axis,
        _dense_axis,
        rho_conv,
        rho_dense,
        **_kwargs,
    ):
        calls.append((float(rho_conv), float(rho_dense)))
        cell_dir = rho_root / "cells" / f"cell-{len(calls)}"
        return cell_dir, {
            "index": len(calls) - 1,
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "status": "complete",
            "selection_eligible": True,
        }

    def fake_candidate(cell_dir, cell):
        center = abs(float(cell["rho_conv"]) - 0.027) < 1e-12 and abs(
            float(cell["rho_dense"]) - 0.09
        ) < 1e-12
        return {
            "cell_id": cell_dir.name,
            "index": cell["index"],
            "rho_conv": float(cell["rho_conv"]),
            "rho_dense": float(cell["rho_dense"]),
            "status": "complete",
            "selection_eligible": True,
            "final_validation_loss": 1.0 if center else 2.0,
            "final_validation_accuracy": 0.95,
            "median_projection_efficiency": 1.0,
            "path": str(cell_dir),
        }

    def fake_selected_gate(
        _study,
        _surface,
        selected,
        _operating_gate,
        *,
        output_path,
        device,
        smoke=False,
    ):
        del device, smoke
        result = {
            "status": "unresolved_post_training_tk",
            "passed": False,
            "selected_cell_id": selected["cell_id"],
            "epochs": [{"epoch": 1, "passed": False}],
        }
        runner._write_json(output_path, result)
        return {
            **result,
            "summary_path": str(output_path.resolve()),
            "summary_sha256": runner._sha256_file(output_path),
        }

    monkeypatch.setattr(runner, "_run_rho_cell", fake_cell)
    monkeypatch.setattr(runner, "_candidate_record", fake_candidate)
    monkeypatch.setattr(
        runner, "run_conv3_selected_post_training_tk", fake_selected_gate
    )
    operating_gate = {
        "dataset_cohort": _conv3_cohort(),
        "checkpoint_sha256": "initial",
    }

    result = runner.run_rho_surface(
        study,
        tmp_path,
        surface,
        tmp_path / "source.json",
        device="cpu",
        operating_point_gate=operating_gate,
    )

    assert len(calls) == 9
    assert result["status"] == "unresolved_post_training_tk"
    assert result["selected"] is None
    assert result["selection"]["selected"] is None
    assert result["pre_post_training_tk_selected"]["rho_conv"] == 0.027
    assert result["post_training_tk"]["passed"] is False
