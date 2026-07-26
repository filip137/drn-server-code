from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from experiments.mnist_conv.identity import code_provenance
from experiments.mnist_conv.io import atomic_write_bytes, atomic_write_json, read_json
from experiments.mnist_conv.lr_study import create_study
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments import run_mnist_conv_lr_v6_exploratory_transfer as transfer


REPO_ROOT = Path(__file__).resolve().parents[2]
V6_CONFIG = REPO_ROOT / "configs/conv/hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json"


def _source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    spec = LRStudySpec.from_path(V6_CONFIG)
    root, _ = create_study(spec, tmp_path / "source-results")
    monkeypatch.setattr(transfer, "validate_stage_completion", lambda **_kwargs: {})
    for relative in (
        "stages/probe/complete.json",
        "stages/baseline_candidates/complete.json",
        "stages/select_baseline/complete.json",
        "split/indices.json",
    ):
        atomic_write_json(root / relative, {"complete": relative}, canonical=True)
    atomic_write_bytes(root / "initialization/conv2.pt", b"conv2-seed0\n")
    groups = {
        "ConvWeight_0": ["ConvWeight_0", "Bias_0"],
        "ConvWeight_1": ["ConvWeight_1", "Bias_1"],
        "DenseWeight_0": ["DenseWeight_0"],
    }
    scales = {"conv2_ours_v4_c1": 2.0, "conv2_legacy_v4_c0p25": 4.0}
    for row_id, scale in scales.items():
        atomic_write_json(
            root / f"stages/probe/entries/{row_id}/summary.json",
            {
                "median_units_by_weight": {
                    "ConvWeight_0": 0.1 * scale,
                    "ConvWeight_1": 0.2 * scale,
                    "DenseWeight_0": 1.0 * scale,
                },
                "bias_weight_lr_groups": groups,
            },
            canonical=True,
        )
    diagnostic_id = "conv2_baseline_v1_c1--grid-c03-d02"
    atomic_write_json(
        root / "stages/select_baseline/entries/selection/selection.json",
        {
            "status": "unbracketed",
            "reason": "passing_plateau_confined_to_outer_boundary",
            "diagnostic_best_entry_id": diagnostic_id,
            "candidates": [
                {
                    "entry_id": diagnostic_id,
                    "rho_conv": transfer.RHO_CONV,
                    "rho_dense": transfer.RHO_DENSE,
                }
            ],
        },
        canonical=True,
    )
    return root


def test_manifest_is_separate_content_addressed_transfer_with_row_specific_lrs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    manifest, path = transfer.create_manifest(
        source_study=source,
        results_root=tmp_path / "transfer-results",
        provenance=code_provenance(),
    )

    assert manifest["transfer_id"].startswith("lrtransfer_")
    assert path.parent != source
    assert manifest["source_selection_status"] == "unbracketed"
    assert manifest["v6_pair_frozen"] is False
    assert [entry["scheme"] for entry in manifest["entries"]] == ["ours", "legacy"]
    ours, legacy = [entry["payload"] for entry in manifest["entries"]]
    assert ours["rho_conv"] == legacy["rho_conv"] == 1e-2
    assert ours["rho_dense"] == legacy["rho_dense"] == 3e-2
    assert ours["learning_rates_by_parameter"]["Bias_0"] == ours[
        "learning_rates_by_parameter"
    ]["ConvWeight_0"]
    assert ours["learning_rates_by_parameter"]["Bias_1"] == ours[
        "learning_rates_by_parameter"
    ]["ConvWeight_1"]
    assert ours["learning_rates_by_weight"] != legacy["learning_rates_by_weight"]

    loaded, loaded_path = transfer.load_manifest(path)
    assert loaded == manifest
    assert loaded_path == path


def test_transfer_executes_outside_v6_and_finalizes_two_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    manifest, path = transfer.create_manifest(
        source_study=source,
        results_root=tmp_path / "transfer-results",
        provenance=code_provenance(),
    )

    def fake_candidate(
        study: dict[str, Any],
        study_dir: str | Path,
        row_id: str,
        role: str,
        *,
        candidate_payload: dict[str, Any],
        output_stage: str,
        output_root: str | Path,
        data_root: str | Path,
        download: bool,
        device: str,
    ) -> dict[str, Any]:
        del study, study_dir, data_root, download, device
        assert role == transfer.ROLE
        root = Path(output_root)
        entry = root / "stages" / output_stage / "entries" / f"{row_id}--{role}"
        entry.mkdir(parents=True, exist_ok=True)
        for filename in transfer.OUTPUT_FILENAMES:
            if filename == "summary.json":
                atomic_write_json(
                    entry / filename,
                    {
                        "rho_conv": candidate_payload["rho_conv"],
                        "rho_dense": candidate_payload["rho_dense"],
                        "status": "complete",
                        "admissible": True,
                        "inadmissible_reason": None,
                        "final_validation_accuracy": 0.93,
                        "final_validation_loss": 0.12,
                        "median_projection_efficiency": 0.99,
                        "learning_rates_by_weight": candidate_payload[
                            "learning_rates_by_weight"
                        ],
                        "observed_rho_relative_q90_complete_run_by_parameter": {
                            "ConvWeight_0": 0.01,
                            "ConvWeight_1": 0.01,
                            "DenseWeight_0": 0.03,
                        },
                        "maximum_bound_occupancy_by_parameter": {
                            "ConvWeight_0": 0.1,
                            "ConvWeight_1": 0.1,
                            "DenseWeight_0": 0.1,
                        },
                    },
                    canonical=True,
                )
            elif filename.endswith(".json"):
                atomic_write_json(entry / filename, {"stub": filename}, canonical=True)
            else:
                atomic_write_bytes(entry / filename, filename.encode())
        return {"status": "complete"}

    monkeypatch.setattr(transfer, "execute_candidate_entry", fake_candidate)
    for index in (0, 1):
        result = transfer.run_entry(
            manifest_path=path,
            source_study=source,
            entry_index=index,
            data_root=tmp_path / "data",
            device="cpu",
        )
        assert result["status"] == "complete"

    summary = transfer.finalize_transfer(path)
    assert summary["all_rows_passed"] is True
    assert len(summary["records"]) == 2
    assert (path.parent / "summary.csv").is_file()
    assert read_json(path.parent / "summary.json") == summary


def test_transfer_refuses_to_relabel_a_selected_v6_result_as_unbracketed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    selection = source / "stages/select_baseline/entries/selection/selection.json"
    value = read_json(selection)
    value["status"] = "selected"
    atomic_write_json(selection, value, canonical=True)

    with pytest.raises(RuntimeError, match="unbracketed diagnostic-best"):
        transfer.create_manifest(
            source_study=source,
            results_root=tmp_path / "transfer-results",
            provenance=code_provenance(),
        )
