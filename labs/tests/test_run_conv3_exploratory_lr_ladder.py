from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.run_conv3_exploratory_lr_ladder as ladder


STUDY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "conv"
    / "perfectdiode_conv3_exploratory_lr_ladder_seed0_20260811_v1.json"
)


def _loaded():
    return ladder.load_study(STUDY_PATH)


def _fake_asset(
    _parent,
    output_root,
    *,
    initializer,
    architecture,
    device,
    dataset_root,
):
    del initializer, architecture, device, dataset_root
    checkpoint = Path(output_root) / "assets" / "bounded_uniform" / "conv3" / "final_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fake pinned checkpoint")
    return checkpoint, {
        "checkpoint_sha256": ladder.EXPECTED_INITIALIZER_SHA256,
    }


def _fake_rho_runner(study, calls):
    by_index = {row["index"]: row for row in ladder.ordered_cells(study)}

    def run(args):
        calls.append(args)
        if args.probe_only:
            return {"status": "complete", "probe": "probe.json"}
        if args.collect_only:
            return {"status": "complete", "cells": 64}
        cell = by_index[args.index]
        rho_root = Path(args.output_root)
        cell_dir = ladder._cell_dir(rho_root, cell)
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "cell.json").write_text(
            json.dumps(
                {
                    "index": args.index,
                    "rho_conv": cell["rho_conv"],
                    "rho_dense": cell["rho_dense"],
                    "status": "complete",
                    "learning_rates_by_parameter": {
                        "ConvWeight_0": cell["rho_conv"],
                        "DenseWeight_0": cell["rho_dense"],
                    },
                }
            ),
            encoding="utf-8",
        )
        (cell_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "final_test_loss": 1.0 - args.index / 1000.0,
                    "final_test_accuracy": 0.1 + args.index / 1000.0,
                }
            ),
            encoding="utf-8",
        )
        (cell_dir / "safety_diagnostics.json").write_text(
            json.dumps({"median_projection_efficiency": 0.75}),
            encoding="utf-8",
        )
        return {"status": "complete", "cells": 1}

    return run


def test_contract_has_pinned_parent_four_surfaces_and_correct_axes() -> None:
    _study_path, study, _parent_path, parent = _loaded()

    assert parent["study_id"] == ladder.EXPECTED_PARENT_STUDY_ID
    assert parent["initialization_reference"]["checkpoint_sha256_by_architecture"][
        "conv3"
    ] == ladder.EXPECTED_INITIALIZER_SHA256
    assert [row["surface_id"] for row in ladder.surface_specs(study)] == [
        "bounded_uniform__conv3__baseline__sgd",
        "bounded_uniform__conv3__baseline__adam",
        "bounded_uniform__conv3__ours__sgd",
        "bounded_uniform__conv3__ours__adam",
    ]
    expected_conv = [0.009 / (3**power) for power in range(7, -1, -1)]
    expected_dense = [0.03 / (3**power) for power in range(7, -1, -1)]
    assert study["rho_search"]["rho_conv"] == pytest.approx(expected_conv)
    assert study["rho_search"]["rho_dense"] == pytest.approx(expected_dense)
    assert study["rho_search"]["rho_conv"][-1] == 0.009
    assert study["rho_search"]["rho_dense"][-1] == 0.03
    assert study["rho_search"]["rho_conv"][3] == pytest.approx(1.1111111111111112e-4)
    assert study["rho_search"]["rho_dense"][5] == pytest.approx(
        3.3333333333333335e-3
    )


def test_cell_order_expands_by_max_then_sum_and_preserves_row_major_ids() -> None:
    _study_path, study, _parent_path, _parent = _loaded()
    cells = ladder.ordered_cells(study)

    assert len(cells) == 64
    assert len({row["index"] for row in cells}) == 64
    assert [row["index"] for row in cells[:9]] == [0, 1, 8, 9, 2, 16, 10, 17, 18]
    assert cells[-1] == {
        "index": 63,
        "conv_axis_index": 7,
        "dense_axis_index": 7,
        "rho_conv": 0.009,
        "rho_dense": 0.03,
        "execution_rank": 63,
    }
    keys = [
        (
            max(row["conv_axis_index"], row["dense_axis_index"]),
            row["conv_axis_index"] + row["dense_axis_index"],
            row["conv_axis_index"],
            row["dense_axis_index"],
        )
        for row in cells
    ]
    assert keys == sorted(keys)


def test_production_rho_args_disable_canary_safety_and_post_tk(tmp_path: Path) -> None:
    _study_path, study, _parent_path, _parent = _loaded()
    surface = ladder.surface_specs(study)[0]
    args = ladder._rho_args(
        study,
        surface,
        tmp_path / "source.json",
        tmp_path / "rho",
        device="cuda",
        target="main",
        index=0,
    )

    assert args.probe_batches == [128]
    assert args.stability_tolerance == 0.5
    assert args.epochs == 3
    assert args.evidence_class == "ordinary_mnist_exploratory"
    assert args.bias_policy == "zero"
    assert args.skip_canary is True
    assert args.restart_interrupted_cells is True
    assert args.canary_steps == 0
    assert args.disable_safety_rejections is True
    assert args.safety_bound_occupancy_increase_maximum is None
    assert args.safety_projection_efficiency_minimum is None
    assert args.post_candidate_gate_name is None
    assert args.post_candidate_callback is None
    assert args.checkpoint_every_epoch is False
    assert args.max_batches is None
    assert args.expected_candidate_steps == 10314
    assert args.minimum_validation_accuracy == 0.0


def test_source_config_inherits_exact_parent_initializer_and_zero_bias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _study_path, study, _parent_path, parent = _loaded()
    monkeypatch.setattr(ladder, "ensure_asset", _fake_asset)
    surface = ladder.surface_specs(study)[2]

    surface_dir, source_path = ladder._prepare_surface(
        study,
        parent,
        tmp_path,
        surface,
        device="cpu",
        dataset_root=tmp_path / "mnist",
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))

    assert surface_dir.name == "bounded_uniform__conv3__ours__sgd"
    assert source["model_base"]["weight_init_mode"] == "bounded_uniform"
    assert source["model_base"]["weight_min"] == 1e-5
    assert source["model_base"]["weight_max"] == 1e-4
    assert source["model_base"]["num_iterations_inference"] == 8
    assert source["model_base"]["num_iterations_training"] == 8
    assert source["model_base"]["voltage_amp"] == 4.0
    assert source["model_base"]["current_amp"] == 1.0
    assert source["lr"] == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert source["bias_contract"]["learning_rate"] == 0.0
    assert source["init_checkpoint_path"].endswith("/final_model.pt")


def test_surface_executes_all_cells_in_declared_order_and_writes_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _study_path, study, _parent_path, parent = _loaded()
    surface = ladder.surface_specs(study)[0]
    calls = []
    monkeypatch.setattr(ladder, "ensure_asset", _fake_asset)
    monkeypatch.setattr(ladder, "run_rho_search", _fake_rho_runner(study, calls))

    result = ladder.run_surface(
        study,
        parent,
        tmp_path,
        surface,
        device="cpu",
        target="test",
        dataset_root=tmp_path / "mnist",
    )

    indexed = [call.index for call in calls if call.index is not None]
    assert indexed == [row["index"] for row in ladder.ordered_cells(study)]
    assert calls[0].probe_only is True
    assert calls[-1].collect_only is True
    assert result["status"] == "complete"
    assert result["complete"] is True
    assert result["terminal_cells"] == 64
    assert result["status_counts"] == {"complete": 64}
    assert result["best_observation"]["index"] == 63
    surface_dir = tmp_path / "surfaces" / surface["surface_id"]
    progress = json.loads((surface_dir / "progress.json").read_text(encoding="utf-8"))
    status = json.loads((surface_dir / "status.json").read_text(encoding="utf-8"))
    assert progress["terminal_cells"] == 64
    assert progress["next_cell"] is None
    assert status["status"] == "complete"
    assert status["terminal"] is True


def test_smoke_runs_only_lowest_cell_for_one_training_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _study_path, study, _parent_path, parent = _loaded()
    surface = ladder.surface_specs(study)[1]
    calls = []
    monkeypatch.setattr(ladder, "ensure_asset", _fake_asset)
    monkeypatch.setattr(ladder, "run_rho_search", _fake_rho_runner(study, calls))

    result = ladder.run_smoke(
        study,
        parent,
        tmp_path,
        surface,
        device="cpu",
        target="test",
        dataset_root=tmp_path / "mnist",
    )

    assert result["status"] == "complete"
    assert [call.index for call in calls if call.index is not None] == [0]
    candidate_call = next(call for call in calls if call.index is not None)
    assert candidate_call.max_batches == 1
    assert candidate_call.max_validation_batches == 1
    assert candidate_call.epochs == 1
    assert candidate_call.expected_candidate_steps == 1
    assert candidate_call.skip_canary is True
    assert candidate_call.disable_safety_rejections is True
    assert candidate_call.rho_conv[0] == pytest.approx(0.009 / (3**7))
    assert candidate_call.rho_dense[0] == pytest.approx(0.03 / (3**7))


def test_collect_reports_partial_then_complete(tmp_path: Path) -> None:
    _study_path, study, _parent_path, _parent = _loaded()
    partial = ladder.collect(study, tmp_path, write=False)
    assert partial["status"] == "partial"
    assert partial["counts"] == {"pending": 4}

    for surface in ladder.surface_specs(study):
        surface_dir = tmp_path / "surfaces" / surface["surface_id"]
        surface_dir.mkdir(parents=True)
        (surface_dir / "status.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
        (surface_dir / "summary.json").write_text("{}", encoding="utf-8")
    complete = ladder.collect(study, tmp_path, write=True)
    assert complete["status"] == "complete"
    assert complete["counts"] == {"complete": 4}
    assert (tmp_path / "summary.json").is_file()


def test_only_nonfinite_safety_rejection_is_an_allowed_terminal() -> None:
    assert ladder._is_allowed_terminal({"status": "complete"}) is True
    assert (
        ladder._is_allowed_terminal(
            {
                "status": "candidate_rejected_safety",
                "safety_failure": {"kind": "nonfinite_diagnostic"},
            }
        )
        is True
    )
    assert (
        ladder._is_allowed_terminal(
            {
                "status": "candidate_rejected_safety",
                "safety_failure": {"kind": "gradient_rms_explosion"},
            }
        )
        is False
    )
