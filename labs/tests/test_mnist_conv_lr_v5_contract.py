from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mnist_conv import lr_stages
from experiments.mnist_conv.io import atomic_write_json
from experiments.mnist_conv.identity import code_provenance, sha256_file
from experiments.mnist_conv.lr_engine import build_model_runtime
from experiments.mnist_conv.lr_artifacts import (
    entry_is_complete,
    publish_entry_completion,
)
from experiments.mnist_conv.lr_protocol import architecture_relative_learning_rates
from experiments.mnist_conv.lr_stages import (
    candidate_run_spec_v4,
    execute_anchor_audit,
)
from experiments.mnist_conv.lr_study import (
    _entries,
    execute_manifest_entry,
    finalize_stage,
    plan_only_result,
    publish_lr_stage_manifest,
    submit_slurm_stage,
)
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments.mnist_conv.lr_v5_packing import build_v5_pack_manifest
from experiments.mnist_conv.specs import RunSpec, SpecValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
V5_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_architecture_relative_rho_constant_sgd_bs16_v5.json"
)
V4_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_layerwise_relative_rho_constant_sgd_bs16_v4.json"
)


class _FakeBundle:
    validation_indices_hash = "a" * 64

    def train_batch_indices(self, *, num_epochs=1):
        return tuple(((epoch, epoch + 1),) for epoch in range(num_epochs))


def test_v5_is_distinct_ordinary_mnist_contract_and_v4_is_unchanged(tmp_path: Path) -> None:
    v5 = LRStudySpec.from_path(V5_CONFIG)
    v4 = LRStudySpec.from_path(V4_CONFIG)

    assert v5.study_id == "lrstudy_5da3a7452c00d42325bfe930d80f037791125d3ce7b5ddd79be13ff8b96b49e2"
    assert v4.study_id == "lrstudy_b345477de5ff64c828a9d31602314a00046bc9728039562d3e5011132805e14d"
    assert v5.data["dataset"]["variant"] == "ordinary"
    assert v5.data["dataset"]["affine"]["enabled"] is False
    assert v5.data["dataset"]["official_test"] == {
        "enabled": False,
        "read_allowed": False,
    }
    assert v5.data["probe"]["normalization_statistic"] == "median"
    assert v5.data["optimizer"]["alpha_scope"] == "architecture"
    assert v5.data["candidate_training"]["batch_size"] == 16
    assert v5.data["candidate_training"]["steps_per_epoch"] == 3438
    assert v5.data["candidate_training"]["total_steps"] == 17190
    assert v5.data["candidate_training"]["schedule"]["scheduler_enabled"] is False
    assert v5.data["candidate_training"]["minimum_final_validation_accuracy"] == 0.9

    study_dir = tmp_path / f"study--{v5.study_id}"
    assert plan_only_result(v5, study_dir, "audit")["entry_count"] == 1
    assert plan_only_result(v5, study_dir, "probe")["entry_count"] == 6
    assert plan_only_result(v5, study_dir, "candidates")["entry_count"] == 36
    assert plan_only_result(v5, study_dir, "select")["entry_count"] == 1
    with pytest.raises(ValueError, match="skip the increasing-LR range"):
        plan_only_result(v5, study_dir, "range")


def test_v5_weight_specific_optimizer_contract_builds_real_runtime() -> None:
    study = LRStudySpec.from_path(V5_CONFIG)

    runtime = build_model_runtime(
        study.data,
        study.rows[0],
        device="cpu",
        learning_rate={
            "ConvWeight_0": 0.1,
            "Bias_0": 0.1,
            "DenseWeight_0": 0.01,
        },
    )

    assert {parameter.name.strip() for parameter in runtime.parameters} == {
        "ConvWeight_0",
        "Bias_0",
        "DenseWeight_0",
    }


def test_v5_rejects_generic_slurm_submission_path(tmp_path: Path) -> None:
    study = LRStudySpec.from_path(V5_CONFIG)
    root = tmp_path / "results/lr_studies" / f"study--{study.study_id}"
    root.mkdir(parents=True)
    atomic_write_json(root / "study.resolved.json", study.data, canonical=True)

    with pytest.raises(
        ValueError,
        match="dedicated measured-memory pack launcher and login-node collection",
    ):
        submit_slurm_stage(
            study_dir=root,
            manifest_path=root / "stages/candidates/manifest.json",
            data_root=tmp_path / "mnist",
            device="cuda",
            profile_path=REPO_ROOT / "configs/executors/jeanzay-v100-32g-v5.json",
            dry_run=True,
        )


def test_existing_initialization_is_hash_verified_without_device_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = LRStudySpec.from_path(V5_CONFIG)

    class AssetBundle:
        train_indices = (0, 1, 2)
        validation_indices = (3,)

        def provenance(self, *, num_epochs):
            return {
                "num_epochs": num_epochs,
                "train_indices": list(self.train_indices),
                "validation_indices": list(self.validation_indices),
                "official_test_read": False,
            }

    monkeypatch.setattr(
        lr_stages, "build_loader_bundle", lambda *_args, **_kwargs: AssetBundle()
    )
    initialization = tmp_path / "initialization"
    initialization.mkdir()
    for architecture in ("conv1", "conv2"):
        checkpoint = initialization / f"{architecture}.pt"
        checkpoint.write_bytes(f"{architecture}-checkpoint".encode())
        atomic_write_json(
            initialization / f"{architecture}.json",
            {
                "schema_version": "mnist-conv-lr-initialization/v1",
                "architecture": architecture,
                "model_seed": 0,
                "checkpoint_path": checkpoint.name,
                "checkpoint_sha256": sha256_file(checkpoint),
            },
            canonical=True,
        )

    monkeypatch.setattr(
        lr_stages,
        "_checkpoint_contract",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("immutable checkpoint must not be rebuilt")
        ),
    )

    result = lr_stages.prepare_study_assets(
        study.data,
        tmp_path,
        data_root=tmp_path,
        download=False,
        device="cuda:0",
    )

    assert set(result["checkpoints"]) == {"conv1", "conv2"}


def test_anchor_audit_materializes_predeclared_centers_without_test_data(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V5_CONFIG)
    summary = execute_anchor_audit(study.data, tmp_path)

    assert summary["derivation_status"] == "fallback_predeclared"
    assert summary["official_test_read"] is False
    assert summary["historical_test_metrics_used"] is False
    assert summary["candidate_grids_by_architecture_and_arm"]["conv2"][
        "historical_profile"
    ] == pytest.approx([0.0005 / 3.0, 0.0005, 0.0015])
    assert (
        tmp_path / "stages/audit/entries/anchor-audit/anchors.csv"
    ).is_file()


def test_candidate_manifest_has_36_row_specific_vectors_but_shared_alpha(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V5_CONFIG)
    for row_index, row in enumerate(study.rows, start=1):
        if row["architecture"] == "conv1":
            units = {
                "ConvWeight_0": 0.1 * row_index,
                "DenseWeight_0": 1.0 * row_index,
            }
            groups = {
                "ConvWeight_0": ["ConvWeight_0", "Bias_0"],
                "DenseWeight_0": ["DenseWeight_0"],
            }
        else:
            units = {
                "ConvWeight_0": 0.1 * row_index,
                "ConvWeight_1": 0.2 * row_index,
                "DenseWeight_0": 1.0 * row_index,
            }
            groups = {
                "ConvWeight_0": ["ConvWeight_0", "Bias_0"],
                "ConvWeight_1": ["ConvWeight_1", "Bias_1"],
                "DenseWeight_0": ["DenseWeight_0"],
            }
        directory = tmp_path / "stages/probe/entries" / row["row_id"]
        directory.mkdir(parents=True)
        atomic_write_json(
            directory / "summary.json",
            {
                "rho_unit_relative_by_parameter": units,
                "bias_weight_lr_groups": groups,
            },
            canonical=True,
        )

    entries = _entries(study, tmp_path, "candidates")
    assert len(entries) == 36
    for start in range(0, 36, 3):
        pack = entries[start : start + 3]
        assert {entry["payload"]["scheme"] for entry in pack} == {
            "baseline",
            "ours",
            "legacy",
        }
        assert len({entry["payload"]["alpha"] for entry in pack}) == 1
        assert len(
            {
                tuple(sorted(entry["payload"]["learning_rates_by_weight"].items()))
                for entry in pack
            }
        ) == 3


def test_v4_run_bundle_records_profile_targets_and_rejects_bias_drift(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V5_CONFIG)
    row = study.rows[0]
    root = tmp_path / "results/lr_studies" / f"study--{study.study_id}"
    (root / "initialization").mkdir(parents=True)
    (root / "initialization/conv1.pt").write_bytes(b"checkpoint")
    atomic_write_json(
        root / "initialization/conv1.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )
    for stage, entry in (("probe", row["row_id"]), ("audit", "anchor-audit")):
        directory = root / "stages" / stage / "entries" / entry
        directory.mkdir(parents=True)
        atomic_write_json(directory / "summary.json", {"stage": stage}, canonical=True)

    units = {"ConvWeight_0": 0.2, "DenseWeight_0": 2.0}
    multipliers = {"ConvWeight_0": 1.0, "DenseWeight_0": 30.0}
    rates = architecture_relative_learning_rates(
        units,
        0.001,
        ("ConvWeight_0", "Bias_0", "DenseWeight_0"),
        target_multipliers=multipliers,
    )
    spec = candidate_run_spec_v4(
        study.data,
        root,
        row,
        "historical_profile--center",
        arm="historical_profile",
        alpha_role="center",
        alpha=0.001,
        median_units_by_weight=units,
        target_multipliers_by_weight=multipliers,
        learning_rates_by_parameter=rates,
        bundle=_FakeBundle(),
    )
    assert spec.data["schema_version"] == "mnist-conv-run/v4"
    assert spec.data["run"]["training"]["schedule"] == {
        "name": "constant",
        "interval": "optimizer_step",
        "total_steps": 17190,
        "scheduler_enabled": False,
    }
    assert spec.data["run"]["lr_provenance"]["alpha_arch"] == 0.001

    value = spec.to_dict()
    value["run"]["training"]["learning_rates_by_parameter"]["Bias_0"] *= 2
    with pytest.raises(SpecValidationError, match="associated ConvWeight_0 rate"):
        RunSpec.from_dict(value)


def test_v5_end_to_end_artifact_smoke_audit_probe_pack_resume_select(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V5_CONFIG)
    root = tmp_path / "results/lr_studies" / f"study--{study.study_id}"
    root.mkdir(parents=True)
    atomic_write_json(root / "study.resolved.json", study.data, canonical=True)
    atomic_write_json(root / "split/indices.json", {"indices": []}, canonical=True)
    atomic_write_json(
        root / "split/provenance.json",
        {"official_test_read": False},
        canonical=True,
    )
    for architecture in ("conv1", "conv2"):
        checkpoint = root / "initialization" / f"{architecture}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"{architecture}-initialization".encode())
        atomic_write_json(
            root / "initialization" / f"{architecture}.json",
            {"architecture": architecture},
            canonical=True,
        )

    provenance = code_provenance()
    audit_manifest, audit_manifest_path = publish_lr_stage_manifest(
        root, study, "audit", provenance
    )
    assert execute_manifest_entry(
        study_dir=root,
        manifest_path=audit_manifest_path,
        entry_index=0,
        data_root=tmp_path,
        download=False,
        device="cpu",
        current_provenance=provenance,
    )["status"] == "complete"
    assert audit_manifest["stage_name"] == "audit"
    finalize_stage(root, audit_manifest_path)

    probe_manifest, probe_manifest_path = publish_lr_stage_manifest(
        root, study, "probe", provenance
    )

    for row_index, row in enumerate(study.rows, start=1):
        weight_names = (
            ["ConvWeight_0", "DenseWeight_0"]
            if row["architecture"] == "conv1"
            else ["ConvWeight_0", "ConvWeight_1", "DenseWeight_0"]
        )
        groups = {
            name: (
                [name, f"Bias_{name.removeprefix('ConvWeight_')}"]
                if name.startswith("ConvWeight_")
                else [name]
            )
            for name in weight_names
        }
        directory = root / "stages/probe/entries" / row["row_id"]
        directory.mkdir(parents=True)
        atomic_write_json(
            directory / "summary.json",
            {
                "rho_unit_relative_by_parameter": {
                    name: row_index * (0.1 if name.startswith("Conv") else 1.0)
                    for name in weight_names
                },
                "bias_weight_lr_groups": groups,
                "normalization_statistic": "median",
                "official_test_read": False,
            },
            canonical=True,
        )
        atomic_write_json(directory / "minibatches.json", {"batches": []}, canonical=True)
        (directory / "parameter_diagnostics.csv").write_text("placeholder\n")
        (directory / "step_log.csv").write_text("placeholder\n")
        publish_entry_completion(
            study_dir=root,
            manifest_path=probe_manifest_path,
            entry_id=row["row_id"],
        )
    assert len(probe_manifest["entries"]) == 6
    finalize_stage(root, probe_manifest_path)

    candidate_manifest, candidate_manifest_path = publish_lr_stage_manifest(
        root, study, "candidates", provenance
    )
    candidate_entries = candidate_manifest["entries"]
    pack_manifest = build_v5_pack_manifest(
        candidate_manifest,
        stage_manifest_sha256=sha256_file(candidate_manifest_path),
        peak_gpu_memory_mib_by_entry_index={index: 9000.0 for index in range(36)},
    )
    assert len(pack_manifest["packs"]) == 12

    for entry in candidate_entries:
        payload = entry["payload"]
        directory = root / "stages/candidates/entries" / entry["entry_id"]
        directory.mkdir(parents=True)
        role_loss = {"lower": 0.20, "center": 0.10, "upper": 0.13}[
            payload["alpha_role"]
        ]
        arm_offset = 0.0 if payload["arm"] == "strict_equal" else 0.004
        atomic_write_json(
            directory / "summary.json",
            {
                "admissible": True,
                "inadmissible_reason": None,
                "final_validation_loss": role_loss + arm_offset,
                "final_validation_accuracy": 0.92,
                "median_projection_efficiency": 0.99,
                "learning_rates_by_weight": payload["learning_rates_by_weight"],
                "observed_peak_rho_relative_by_parameter": {
                    name: 99.0
                    for name in payload["median_units_by_weight"]
                },
                "observed_rho_relative_q90_complete_run_by_parameter": {
                    name: float(payload["alpha"])
                    * float(payload["target_multipliers_by_weight"][name])
                    for name in payload["median_units_by_weight"]
                },
                "maximum_bound_occupancy_by_parameter": {
                    name: 0.01 for name in payload["median_units_by_weight"]
                },
                "median_projection_efficiency_by_parameter": {
                    name: 0.99 for name in payload["median_units_by_weight"]
                },
            },
            canonical=True,
        )
        for relative in entry["outputs"]:
            path = root / relative
            if path == directory / "summary.json":
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                atomic_write_json(path, {}, canonical=True)
            elif path.suffix == ".pt":
                path.write_bytes(b"smoke-checkpoint")
            else:
                path.write_text("placeholder\n")
        publish_entry_completion(
            study_dir=root,
            manifest_path=candidate_manifest_path,
            entry_id=entry["entry_id"],
        )
    assert entry_is_complete(
        study_dir=root,
        manifest_path=candidate_manifest_path,
        entry_id=candidate_entries[0]["entry_id"],
    )
    finalize_stage(root, candidate_manifest_path)

    assert not (root / "stages/range").exists()
    select_manifest, select_manifest_path = publish_lr_stage_manifest(
        root, study, "select", provenance
    )
    first = execute_manifest_entry(
        study_dir=root,
        manifest_path=select_manifest_path,
        entry_index=0,
        data_root=tmp_path,
        download=False,
        device="cpu",
        current_provenance=provenance,
    )
    resumed = execute_manifest_entry(
        study_dir=root,
        manifest_path=select_manifest_path,
        entry_index=0,
        data_root=tmp_path,
        download=False,
        device="cpu",
        current_provenance=provenance,
    )
    assert first["status"] == "complete"
    assert resumed["status"] == "resumed_complete"
    assert select_manifest["stage_name"] == "select"
    assert finalize_stage(root, select_manifest_path)["status"] == "complete"

    selection_path = root / "stages/select/entries/selection/selection.json"
    selection = json.loads(selection_path.read_text())
    assert selection["status"] == "selected"
    assert selection["architectures"]["conv1"]["selected_arm"] == "strict_equal"
    assert selection["architectures"]["conv2"]["selected_arm"] == "strict_equal"
    assert all(
        set(candidate["achieved_relative_updates_by_weight"].values()) != {99.0}
        for candidate in selection["candidates"]
    )
    for filename in (
        "selection.json",
        "selection.csv",
        "epoch5_accuracy_loss.png",
        "achieved_relative_updates.png",
        "raw_learning_rates.png",
        "occupancy.png",
        "projection_efficiency.png",
    ):
        assert (root / "stages/select/entries/selection" / filename).is_file()
    with (root / "stages/select/entries/selection/selection.csv").open(
        newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows
    assert {row["study_role"] for row in csv_rows} == {
        "ordinary_mnist_optimization_diagnostic"
    }
    assert {row["final_paper_training_authorized"] for row in csv_rows} == {
        "False"
    }
    assert {row["medium_affine_handoff_replaced"] for row in csv_rows} == {
        "False"
    }


def test_v5_candidate_preserves_projection_safety_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = LRStudySpec.from_path(V5_CONFIG)
    row = study.rows[0]
    root = tmp_path / "results/lr_studies" / f"study--{study.study_id}"
    checkpoint = root / "initialization/conv1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    atomic_write_json(
        root / "initialization/conv1.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )
    probe_dir = root / "stages/probe/entries" / row["row_id"]
    probe_dir.mkdir(parents=True)
    atomic_write_json(probe_dir / "summary.json", {}, canonical=True)

    class FakeRuntime:
        parameters = (object(),)

        @staticmethod
        def save(path: str | Path) -> None:
            Path(path).write_bytes(b"checkpoint")

    class FakeBundle:
        train_loader = (object(),) * 20
        validation_loader = object()
        train_indices_hash = "c" * 64
        validation_indices_hash = "d" * 64

        @staticmethod
        def reset_train_shuffle() -> None:
            return None

    diagnostics = {
        "ConvWeight_0": {
            "bounded_gate": True,
            "rms": 1.0,
            "combined_bound_occupancy": 0.0,
        },
        "DenseWeight_0": {
            "bounded_gate": True,
            "rms": 1.0,
            "combined_bound_occupancy": 0.0,
        },
    }
    monkeypatch.setattr(
        lr_stages,
        "build_model_runtime",
        lambda *_args, **_kwargs: FakeRuntime(),
    )
    monkeypatch.setattr(
        lr_stages,
        "build_loader_bundle",
        lambda *_args, **_kwargs: FakeBundle(),
    )
    monkeypatch.setattr(
        lr_stages,
        "parameter_state_diagnostics",
        lambda *_args, **_kwargs: diagnostics,
    )
    monkeypatch.setattr(
        lr_stages,
        "parameter_tensor_digest",
        lambda *_args, **_kwargs: "e" * 64,
    )
    monkeypatch.setattr(
        lr_stages,
        "candidate_run_spec_v4",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {"schema_version": "mnist-conv-run/v4"}
        ),
    )
    step = 0

    def fake_training_step(*_args, **_kwargs):
        nonlocal step
        step += 1
        return SimpleNamespace(
            loss=1.0,
            accuracy=0.5,
            sample_count=16,
            source_indices=(step,),
            transition=object(),
        )

    def fake_transition_rows(*_args, **_kwargs):
        return [
            {
                "parameter": name,
                "bounded_gate": True,
                "gradient_rms": 1.0,
                "normalized_update": 0.001,
                "parameter_relative_update": 0.001,
                "projection_gate_eligible": True,
                "projection_efficiency": 0.4,
                "combined_bound_occupancy": 0.0,
            }
            for name in ("ConvWeight_0", "DenseWeight_0")
        ]

    monkeypatch.setattr(lr_stages, "training_step", fake_training_step)
    monkeypatch.setattr(lr_stages, "transition_rows", fake_transition_rows)

    payload = {
        "arm": "strict_equal",
        "alpha_role": "center",
        "alpha": 0.01,
        "peak_learning_rate": 0.1,
        "learning_rates_by_parameter": {
            "ConvWeight_0": 0.1,
            "Bias_0": 0.1,
            "DenseWeight_0": 0.01,
        },
        "learning_rates_by_weight": {
            "ConvWeight_0": 0.1,
            "DenseWeight_0": 0.01,
        },
        "median_units_by_weight": {
            "ConvWeight_0": 0.1,
            "DenseWeight_0": 1.0,
        },
        "target_multipliers_by_weight": {
            "ConvWeight_0": 1.0,
            "DenseWeight_0": 1.0,
        },
    }
    summary = lr_stages.execute_candidate_entry(
        study.data,
        root,
        row["row_id"],
        "strict_equal--center",
        candidate_payload=payload,
        data_root=tmp_path,
        download=False,
        device="cpu",
    )

    assert summary["status"] == "failed"
    assert summary["training_completed"] is False
    assert summary["attempted_steps"] == 16
    assert summary["completed_steps"] == 16
    assert summary["inadmissible_reason"].startswith(
        "early_projection_domination:"
    )
    assert summary["safety_gate_failure"]["kind"] == "projection_efficiency"
    assert summary["safety_gate_failure"]["confirmed_step"] == 16
