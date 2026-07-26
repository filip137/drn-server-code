from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from experiments import run_mnist_conv_lr_conv1_scheme_rho_sweep as sweep
from experiments.run_mnist_conv_lr_conv1_scheme_rho_split_pack import scheme_entries
from experiments.mnist_conv.identity import code_provenance
from experiments.mnist_conv.io import atomic_write_bytes, atomic_write_json, read_json
from experiments.mnist_conv.lr_protocol import (
    architecture_relative_learning_rates,
    two_rho_learning_rates,
)
from experiments.mnist_conv.lr_stages import candidate_run_spec_v6
from experiments.mnist_conv.lr_study import create_study
from experiments.mnist_conv.lr_study_spec import LRStudySpec


REPO_ROOT = Path(__file__).resolve().parents[2]
V5_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_architecture_relative_rho_constant_sgd_bs16_v5.json"
)


class _FakeBundle:
    validation_indices_hash = "a" * 64

    def train_batch_indices(self, *, num_epochs: int = 1):
        return tuple(((epoch, epoch + 1),) for epoch in range(num_epochs))


def _targets(spec: LRStudySpec, role: str) -> tuple[float, dict[str, float]]:
    arm, alpha_role = role.split("--")
    roles = spec.data["target_policies"]["candidate_roles"]
    factors = spec.data["target_policies"]["candidate_factors"]
    factor = factors[roles.index(alpha_role)]
    alpha = (
        spec.data["target_policies"]["centers_by_architecture_and_arm"]["conv1"][arm]
        * factor
    )
    multipliers = (
        {"ConvWeight_0": 1.0, "DenseWeight_0": 30.0}
        if arm == "historical_profile"
        else {"ConvWeight_0": 1.0, "DenseWeight_0": 1.0}
    )
    return alpha, multipliers


def _source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    spec = LRStudySpec.from_path(V5_CONFIG)
    root, _ = create_study(spec, tmp_path / "source-results")
    monkeypatch.setattr(sweep, "validate_stage_completion", lambda **_kwargs: {})
    for relative in (
        "stages/audit/complete.json",
        "stages/probe/complete.json",
        "stages/candidates/complete.json",
        "stages/select/complete.json",
        "stages/select/entries/selection/selection.json",
        "split/indices.json",
    ):
        atomic_write_json(root / relative, {"complete": relative}, canonical=True)
    atomic_write_bytes(root / "initialization/conv1.pt", b"conv1-seed0\n")
    atomic_write_json(
        root / "initialization/conv1.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )

    rows = {
        row["row_id"]: row
        for row in spec.rows
        if row["architecture"] == "conv1"
    }
    units_by_scheme = {
        "baseline": {"ConvWeight_0": 0.1, "DenseWeight_0": 0.5},
        "ours": {"ConvWeight_0": 0.2, "DenseWeight_0": 1.0},
        "legacy": {"ConvWeight_0": 0.4, "DenseWeight_0": 2.0},
    }
    groups = {
        "ConvWeight_0": ["ConvWeight_0", "Bias_0"],
        "DenseWeight_0": ["DenseWeight_0"],
    }
    for row in rows.values():
        units = units_by_scheme[row["scheme"]]
        atomic_write_json(
            root / f"stages/probe/entries/{row['row_id']}/summary.json",
            {
                "rho_unit_relative_q50_by_parameter": units,
                "bias_weight_lr_groups": groups,
            },
            canonical=True,
        )
        for role in sweep.V5_ROLES:
            alpha, multipliers = _targets(spec, role)
            rates = architecture_relative_learning_rates(
                units,
                alpha,
                ("ConvWeight_0", "Bias_0", "DenseWeight_0"),
                target_multipliers=multipliers,
            )
            rho_conv = alpha * multipliers["ConvWeight_0"]
            rho_dense = alpha * multipliers["DenseWeight_0"]
            loss = 0.11 + abs(__import__("math").log10(rho_conv / 1e-3)) * 0.01
            summary = {
                "row": row,
                "alpha_arch": alpha,
                "target_multipliers_by_weight": multipliers,
                "status": "complete",
                "admissible": True,
                "inadmissible_reason": None,
                "final_validation_accuracy": 0.94,
                "final_validation_loss": loss,
                "median_projection_efficiency": 0.99,
                "learning_rates_by_weight": {
                    "ConvWeight_0": rates["ConvWeight_0"],
                    "DenseWeight_0": rates["DenseWeight_0"],
                },
                "observed_peak_rho_relative_by_parameter": {
                    "ConvWeight_0": rho_conv,
                    "DenseWeight_0": rho_dense,
                },
                "observed_rho_relative_q90_complete_run_by_parameter": {
                    "ConvWeight_0": rho_conv / 2,
                    "DenseWeight_0": rho_dense / 2,
                },
                "maximum_bound_occupancy_by_parameter": {
                    "ConvWeight_0": 0.1,
                    "DenseWeight_0": 0.1,
                },
                "median_projection_efficiency_by_parameter": {
                    "ConvWeight_0": 0.99,
                    "DenseWeight_0": 0.99,
                },
            }
            atomic_write_json(
                root
                / f"stages/candidates/entries/{row['row_id']}--{role}/summary.json",
                summary,
                canonical=True,
            )
    return root


def test_manifest_uses_smaller_common_grid_and_reuses_six_v5_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    manifest, path = sweep.create_manifest(
        source_study=source,
        results_root=tmp_path / "sweep-results",
        provenance=code_provenance(),
    )

    assert manifest["rho_conv_grid"] == list(sweep.RHO_CONV_GRID)
    assert manifest["rho_dense_grid"] == list(sweep.RHO_DENSE_GRID)
    assert manifest["new_training_count"] == 18
    assert manifest["reused_candidate_count"] == 6
    assert len(manifest["entries"]) == 24
    assert {entry["scheme"] for entry in manifest["entries"]} == {"ours", "legacy"}
    for entry in manifest["entries"]:
        rates = entry["payload"]["learning_rates_by_parameter"]
        assert rates["Bias_0"] == rates["ConvWeight_0"]
        assert entry["payload"]["candidate_coordinate"] == "conv1_scheme_direct_two_rho"
    loaded, loaded_path = sweep.load_manifest(path)
    assert loaded == manifest
    assert loaded_path == path


def test_split_pack_selects_nine_disjoint_new_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    manifest, _path = sweep.create_manifest(
        source_study=source,
        results_root=tmp_path / "sweep-results",
        provenance=code_provenance(),
    )

    ours = scheme_entries(manifest, "ours")
    legacy = scheme_entries(manifest, "legacy")
    ours_indices = {entry["entry_index"] for entry in ours}
    legacy_indices = {entry["entry_index"] for entry in legacy}

    assert len(ours) == len(legacy) == 9
    assert {entry["scheme"] for entry in ours} == {"ours"}
    assert {entry["scheme"] for entry in legacy} == {"legacy"}
    assert ours_indices.isdisjoint(legacy_indices)
    assert ours_indices | legacy_indices == {
        entry["entry_index"]
        for entry in manifest["entries"]
        if entry["mode"] == "new_five_epoch_training"
    }


def test_direct_conv1_run_spec_has_no_alpha_coordinate(tmp_path: Path) -> None:
    study = LRStudySpec.from_path(V5_CONFIG)
    row = next(item for item in study.rows if item["row_id"] == "conv1_ours_v4_c1")
    root = tmp_path / "study"
    atomic_write_bytes(root / "initialization/conv1.pt", b"checkpoint")
    atomic_write_json(
        root / "initialization/conv1.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )
    atomic_write_json(
        root / f"stages/probe/entries/{row['row_id']}/summary.json",
        {"probe": True},
        canonical=True,
    )
    units = {"ConvWeight_0": 0.2, "DenseWeight_0": 1.0}
    rates = two_rho_learning_rates(
        units,
        rho_conv=1e-3,
        rho_dense=1e-2,
        bias_weight_lr_groups={
            "ConvWeight_0": ["ConvWeight_0", "Bias_0"],
            "DenseWeight_0": ["DenseWeight_0"],
        },
    )
    result = candidate_run_spec_v6(
        study.data,
        root,
        row,
        "scheme-rho--c01-d01",
        rho_conv=1e-3,
        rho_dense=1e-2,
        median_units_by_weight=units,
        learning_rates_by_parameter=rates,
        bundle=_FakeBundle(),
    )

    assert result.data["schema_version"] == "mnist-conv-run/v6"
    provenance = result.data["run"]["lr_provenance"]
    assert provenance["rho_conv"] == pytest.approx(1e-3)
    assert provenance["rho_dense"] == pytest.approx(1e-2)
    assert "alpha_arch" not in provenance
    assert result.data["run"]["training"]["learning_rates_by_parameter"]["Bias_0"] == result.data["run"]["training"]["learning_rates_by_parameter"]["ConvWeight_0"]


def test_new_entries_execute_resume_and_finalize_per_scheme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path, monkeypatch)
    manifest, path = sweep.create_manifest(
        source_study=source,
        results_root=tmp_path / "sweep-results",
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
        entry = (
            Path(output_root)
            / "stages"
            / output_stage
            / "entries"
            / f"{row_id}--{role}"
        )
        entry.mkdir(parents=True, exist_ok=True)
        rc = candidate_payload["rho_conv"]
        rd = candidate_payload["rho_dense"]
        loss = 0.10 + abs(__import__("math").log10(rc / 1e-3)) * 0.02 + abs(__import__("math").log10(rd / 1e-2)) * 0.02
        for filename in sweep.OUTPUT_FILENAMES:
            if filename == "summary.json":
                atomic_write_json(
                    entry / filename,
                    {
                        "status": "complete",
                        "admissible": True,
                        "inadmissible_reason": None,
                        "final_validation_accuracy": 0.95,
                        "final_validation_loss": loss,
                        "median_projection_efficiency": 0.99,
                        "learning_rates_by_weight": candidate_payload["learning_rates_by_weight"],
                        "observed_peak_rho_relative_by_parameter": {
                            "ConvWeight_0": rc,
                            "DenseWeight_0": rd,
                        },
                        "observed_rho_relative_q90_complete_run_by_parameter": {
                            "ConvWeight_0": rc / 2,
                            "DenseWeight_0": rd / 2,
                        },
                        "maximum_bound_occupancy_by_parameter": {
                            "ConvWeight_0": 0.1,
                            "DenseWeight_0": 0.1,
                        },
                        "median_projection_efficiency_by_parameter": {
                            "ConvWeight_0": 0.99,
                            "DenseWeight_0": 0.99,
                        },
                    },
                    canonical=True,
                )
            elif filename.endswith(".json"):
                atomic_write_json(entry / filename, {"stub": filename}, canonical=True)
            else:
                atomic_write_bytes(entry / filename, filename.encode())
        return {"status": "complete"}

    monkeypatch.setattr(sweep, "execute_candidate_entry", fake_candidate)
    monkeypatch.setattr(
        sweep,
        "_plot_heatmaps",
        lambda _records, root, *, field, filename: atomic_write_bytes(
            root / filename, field.encode()
        ),
    )
    new_entries = [
        entry for entry in manifest["entries"]
        if entry["mode"] == "new_five_epoch_training"
    ]
    for entry in new_entries:
        result = sweep.run_entry(
            manifest_path=path,
            source_study=source,
            entry_index=entry["entry_index"],
            data_root=tmp_path / "data",
            device="cpu",
        )
        assert result["status"] == "complete"
    resumed = sweep.run_entry(
        manifest_path=path,
        source_study=source,
        entry_index=new_entries[0]["entry_index"],
        data_root=tmp_path / "data",
        device="cpu",
    )
    assert resumed["status"] == "resumed_complete"

    result = sweep.finalize_sweep(path, source_study=source)
    assert result["status"] == "complete"
    assert set(result["selections_by_scheme"]) == {"ours", "legacy"}
    assert len(result["records"]) == 24
    assert (path.parent / "summary.csv").is_file()
    assert read_json(path.parent / "summary.json") == result
    assert read_json(path.parent / "complete.json")["state"] == "complete"
