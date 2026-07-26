from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.mnist_conv import lr_stages
from experiments.mnist_conv.io import atomic_write_json, read_json
from experiments.mnist_conv.identity import code_provenance, sha256_file
from experiments.mnist_conv.lr_artifacts import (
    LRArtifactError,
    build_stage_manifest,
    load_stage_manifest,
    publish_entry_completion,
    publish_stage_completion,
    publish_stage_manifest,
    stage_entry,
)
from experiments.mnist_conv.lr_engine import build_model_runtime
from experiments.mnist_conv.lr_stages import (
    candidate_run_spec_v5,
    execute_v6_baseline_selection,
    execute_v6_finalization,
    two_rho_learning_rates,
)
from experiments.mnist_conv.lr_study import _entries
from experiments.mnist_conv.lr_study_spec import LRStudySpec


REPO_ROOT = Path(__file__).resolve().parents[2]
V6_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json"
)


def _probe_summary(scale: float) -> dict[str, object]:
    return {
        "median_units_by_weight": {
            "ConvWeight_0": 0.1 * scale,
            "ConvWeight_1": 0.2 * scale,
            "DenseWeight_0": 1.0 * scale,
        },
        "bias_weight_lr_groups": {
            "ConvWeight_0": ["ConvWeight_0", "Bias_0"],
            "ConvWeight_1": ["ConvWeight_1", "Bias_1"],
            "DenseWeight_0": ["DenseWeight_0"],
        },
    }


def _write_probes(root: Path, study: LRStudySpec) -> None:
    for index, row in enumerate(study.rows, start=1):
        path = root / "stages/probe/entries" / row["row_id"] / "summary.json"
        atomic_write_json(path, _probe_summary(float(index)), canonical=True)


def test_v6_manifest_grid_and_confirmations_use_direct_row_specific_targets(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V6_CONFIG)
    _write_probes(tmp_path, study)

    baseline = _entries(study, tmp_path, "baseline_candidates")

    assert len(baseline) == 16
    assert [
        (entry["payload"]["rho_conv"], entry["payload"]["rho_dense"])
        for entry in baseline
    ] == [
        (rho_conv, rho_dense)
        for rho_conv in study.data["rho_grid"]["rho_conv"]
        for rho_dense in study.data["rho_grid"]["rho_dense"]
    ]
    payload = baseline[0]["payload"]
    assert payload["candidate_stage"] == "baseline_grid"
    assert payload["learning_rates_by_parameter"]["Bias_0"] == payload[
        "learning_rates_by_parameter"
    ]["ConvWeight_0"]
    assert payload["learning_rates_by_parameter"]["Bias_1"] == payload[
        "learning_rates_by_parameter"
    ]["ConvWeight_1"]
    encoded = json.dumps([entry["payload"] for entry in baseline]).lower()
    assert "alpha" not in encoded
    assert "strict" not in encoded
    assert "profile" not in encoded

    selection_dir = tmp_path / "stages/select_baseline/entries/selection"
    atomic_write_json(
        selection_dir / "selection.json",
        {
            "status": "selected",
            "selected_rho_conv": 0.003,
            "selected_rho_dense": 0.03,
        },
        canonical=True,
    )
    confirmations = _entries(study, tmp_path, "confirmations")

    assert len(confirmations) == 2
    assert {entry["payload"]["scheme"] for entry in confirmations} == {
        "ours",
        "legacy",
    }
    assert all(
        entry["payload"]["candidate_stage"] == "confirmation"
        for entry in confirmations
    )
    assert (
        confirmations[0]["payload"]["learning_rates_by_weight"]
        != confirmations[1]["payload"]["learning_rates_by_weight"]
    )


def test_v6_confirmations_are_not_generated_for_unbracketed_selection(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V6_CONFIG)
    _write_probes(tmp_path, study)
    atomic_write_json(
        tmp_path / "stages/select_baseline/entries/selection/selection.json",
        {
            "status": "unbracketed",
            "selected_rho_conv": None,
            "selected_rho_dense": None,
        },
        canonical=True,
    )

    with pytest.raises(RuntimeError, match="bracketed v6 baseline selection"):
        _entries(study, tmp_path, "confirmations")


def test_v6_probe_reuse_survives_parent_disappearing_after_audit(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V6_CONFIG)
    row = study.rows[0]
    cache_dir = tmp_path / "reuse/probes" / row["row_id"]
    summary = {"schema_version": "mnist-conv-lr-probe-result/v5", "status": "complete"}
    atomic_write_json(cache_dir / "summary.json", summary, canonical=True)
    atomic_write_json(
        cache_dir / "minibatches.json",
        {"batch_order_sha256": "a" * 64},
        canonical=True,
    )
    for filename in ("parameter_diagnostics.csv", "step_log.csv"):
        path = cache_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("header\n")
    cached_outputs = [
        {
            "path": path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(cache_dir.iterdir())
    ]
    atomic_write_json(
        tmp_path / "reuse/import_receipt.json",
        {
            "source_study_id": study.data["reuse"]["source_study_id"],
            "probes": {
                row["row_id"]: {
                    "mode": "reused_measurement",
                    "cached_outputs": cached_outputs,
                }
            },
        },
        canonical=True,
    )

    reused = lr_stages._reuse_v6_probe_if_available(study.data, tmp_path, row)

    assert reused == summary
    output = tmp_path / "stages/probe/entries" / row["row_id"]
    assert {path.name for path in output.iterdir()} == {
        "minibatches.json",
        "parameter_diagnostics.csv",
        "step_log.csv",
        "summary.json",
    }


class _FakeBundle:
    validation_indices_hash = "a" * 64

    def train_batch_indices(self, *, num_epochs: int = 1):
        return tuple(((epoch, epoch + 1),) for epoch in range(num_epochs))


def test_v6_run_spec_is_constructed_without_a_range_stage(tmp_path: Path) -> None:
    study = LRStudySpec.from_path(V6_CONFIG)
    row = study.rows[0]
    root = (
        tmp_path
        / "results/lr_studies"
        / f"{study.data['name']}--{study.study_id}"
    )
    checkpoint = root / "initialization/conv2.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"v6-checkpoint")
    atomic_write_json(
        root / "initialization/conv2.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )
    atomic_write_json(
        root / "stages/probe/entries" / row["row_id"] / "summary.json",
        _probe_summary(1.0),
        canonical=True,
    )
    units = _probe_summary(1.0)["median_units_by_weight"]
    groups = _probe_summary(1.0)["bias_weight_lr_groups"]
    rates = two_rho_learning_rates(
        units,
        0.003,
        0.03,
        bias_weight_lr_groups=groups,
    )

    result = candidate_run_spec_v5(
        study.data,
        root,
        row,
        "grid-c02-d02",
        candidate_stage="baseline_grid",
        rho_conv=0.003,
        rho_dense=0.03,
        median_units_by_weight=units,
        learning_rates_by_parameter=rates,
        bundle=_FakeBundle(),
    )

    assert result.data["schema_version"] == "mnist-conv-run/v5"
    assert not (root / "stages/range").exists()
    provenance = result.data["run"]["lr_provenance"]
    assert provenance["rho_conv"] == pytest.approx(0.003)
    assert provenance["rho_dense"] == pytest.approx(0.03)
    assert provenance["bias_weight_mapping"] == {
        "Bias_0": "ConvWeight_0",
        "Bias_1": "ConvWeight_1",
    }


def _candidate_summary(
    *,
    row: dict[str, object],
    candidate_stage: str,
    rho_conv: float,
    rho_dense: float,
    loss: float,
    accuracy: float,
    admissible: bool = True,
) -> dict[str, object]:
    return {
        "row": row,
        "candidate_stage": candidate_stage,
        "rho_conv": rho_conv,
        "rho_dense": rho_dense,
        "admissible": admissible,
        "inadmissible_reason": None if admissible else "synthetic_safety_gate",
        "final_validation_loss": loss,
        "final_validation_accuracy": accuracy,
        "median_projection_efficiency": 0.9,
        "learning_rates_by_weight": {
            "ConvWeight_0": 0.1,
            "ConvWeight_1": 0.05,
            "DenseWeight_0": 0.01,
        },
        "observed_rho_relative_q90_complete_run_by_parameter": {
            "ConvWeight_0": rho_conv,
            "ConvWeight_1": rho_conv,
            "DenseWeight_0": rho_dense,
        },
        "maximum_bound_occupancy_by_parameter": {
            "ConvWeight_0": 0.1,
            "ConvWeight_1": 0.1,
            "DenseWeight_0": 0.1,
        },
        "median_projection_efficiency_by_parameter": {
            "ConvWeight_0": 0.9,
            "ConvWeight_1": 0.9,
            "DenseWeight_0": 0.9,
        },
    }


def test_v6_selection_writes_json_csv_and_five_heatmaps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = LRStudySpec.from_path(V6_CONFIG)
    row = study.rows[0]
    entries = []
    for conv_index, rho_conv in enumerate(study.data["rho_grid"]["rho_conv"]):
        for dense_index, rho_dense in enumerate(study.data["rho_grid"]["rho_dense"]):
            role = f"grid-c{conv_index:02d}-d{dense_index:02d}"
            entry_id = f"{row['row_id']}--{role}"
            entries.append(
                {
                    "entry_id": entry_id,
                    "payload": {
                        "row_id": row["row_id"],
                        "scheme": "baseline",
                        "rho_conv": rho_conv,
                        "rho_dense": rho_dense,
                    },
                }
            )
            loss = 0.1 if (rho_conv, rho_dense) == (0.003, 0.03) else 1.0
            accuracy = 0.90 if (rho_conv, rho_dense) == (0.003, 0.03) else 0.91
            atomic_write_json(
                tmp_path
                / "stages/baseline_candidates/entries"
                / entry_id
                / "summary.json",
                _candidate_summary(
                    row=row,
                    candidate_stage="baseline_grid",
                    rho_conv=rho_conv,
                    rho_dense=rho_dense,
                    loss=loss,
                    accuracy=accuracy,
                ),
                canonical=True,
            )
    monkeypatch.setattr(
        lr_stages,
        "load_stage_manifest",
        lambda *_args, **_kwargs: {"entries": entries},
    )

    result = execute_v6_baseline_selection(study.data, tmp_path)

    assert result["status"] == "selected"
    assert result["selected_rho_conv"] == pytest.approx(0.003)
    assert result["selected_rho_dense"] == pytest.approx(0.03)
    output = tmp_path / "stages/select_baseline/entries/selection"
    assert (output / "selection.json").is_file()
    assert (output / "selection.csv").is_file()
    assert sorted(path.name for path in output.glob("*.png")) == [
        "achieved_relative_updates_heatmap.png",
        "epoch5_accuracy_loss_heatmap.png",
        "occupancy_heatmap.png",
        "projection_efficiency_heatmap.png",
        "raw_learning_rates_heatmap.png",
    ]
    encoded = (output / "selection.json").read_text().lower()
    assert "alpha" not in encoded
    assert "strict" not in encoded
    assert "profile" not in encoded


@pytest.mark.parametrize(
    ("confirmation_accuracy", "expected_frozen"),
    [(0.90, True), (0.8999, False)],
)
def test_v6_finalization_requires_all_three_rows_to_pass(
    tmp_path: Path,
    confirmation_accuracy: float,
    expected_frozen: bool,
) -> None:
    study = LRStudySpec.from_path(V6_CONFIG)
    pair = (0.003, 0.03)
    baseline_entry = f"{study.rows[0]['row_id']}--grid-c02-d02"
    atomic_write_json(
        tmp_path / "stages/select_baseline/entries/selection/selection.json",
        {
            "status": "selected",
            "reason": None,
            "selected_entry_id": baseline_entry,
            "selected_rho_conv": pair[0],
            "selected_rho_dense": pair[1],
            "candidates": [
                {
                    "entry_id": baseline_entry,
                    "row_id": study.rows[0]["row_id"],
                    "scheme": "baseline",
                    "rho_conv": pair[0],
                    "rho_dense": pair[1],
                    "selected_for_confirmation": True,
                }
            ],
        },
        canonical=True,
    )
    atomic_write_json(
        tmp_path
        / "stages/baseline_candidates/entries"
        / baseline_entry
        / "summary.json",
        _candidate_summary(
            row=study.rows[0],
            candidate_stage="baseline_grid",
            rho_conv=pair[0],
            rho_dense=pair[1],
            loss=0.1,
            accuracy=0.90,
        ),
        canonical=True,
    )
    for row in study.rows[1:]:
        atomic_write_json(
            tmp_path
            / "stages/confirmations/entries"
            / f"{row['row_id']}--confirmation"
            / "summary.json",
            _candidate_summary(
                row=row,
                candidate_stage="confirmation",
                rho_conv=pair[0],
                rho_dense=pair[1],
                loss=0.2,
                accuracy=confirmation_accuracy,
                admissible=confirmation_accuracy >= 0.90,
            ),
            canonical=True,
        )

    result = execute_v6_finalization(study.data, tmp_path)

    assert result["frozen"] is expected_frozen
    assert (result["rho_conv"], result["rho_dense"]) == (
        pair if expected_frozen else (None, None)
    )
    saved = read_json(
        tmp_path / "stages/finalize/entries/finalization/finalization.json"
    )
    assert saved == result


def test_v6_optimizer_contract_builds_a_real_conv2_runtime() -> None:
    study = LRStudySpec.from_path(V6_CONFIG)
    baseline = next(row for row in study.rows if row["scheme"] == "baseline")

    runtime = build_model_runtime(
        study.data,
        baseline,
        device="cpu",
        learning_rate={
            "ConvWeight_0": 0.1,
            "Bias_0": 0.1,
            "ConvWeight_1": 0.01,
            "Bias_1": 0.01,
            "DenseWeight_0": 0.001,
        },
    )

    assert {parameter.name.strip() for parameter in runtime.parameters} == {
        "ConvWeight_0",
        "Bias_0",
        "ConvWeight_1",
        "Bias_1",
        "DenseWeight_0",
    }


def test_v6_shadow_runs_exact_production_loop_for_32_plus_256_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = LRStudySpec.from_path(V6_CONFIG)
    baseline = next(row for row in study.rows if row["scheme"] == "baseline")
    root = tmp_path / "study"
    checkpoint = root / "initialization/conv2.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"initial")
    start = tmp_path / "barrier/start"
    start.parent.mkdir(parents=True)
    start.write_text("start\n")
    observed = {"steps": 0, "transitions": 0}

    class FakeRuntime:
        parameters = (object(),)

        @staticmethod
        def save(path: str | Path) -> None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"state")

    class FakeBundle:
        train_loader = (object(),) * 288
        train_indices_hash = "a" * 64
        validation_indices_hash = "b" * 64

        @staticmethod
        def reset_train_shuffle() -> None:
            return None

    bounded = {
        name: {
            "bounded_gate": True,
            "rms": 1.0,
            "combined_bound_occupancy": 0.0,
        }
        for name in ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
    }
    monkeypatch.setenv(
        lr_stages.V6_R3_AUTHORIZATION_ENV,
        lr_stages.V6_R3_AUTHORIZATION_TOKEN,
    )
    monkeypatch.setattr(
        lr_stages, "build_model_runtime", lambda *_args, **_kwargs: FakeRuntime()
    )
    monkeypatch.setattr(
        lr_stages, "build_loader_bundle", lambda *_args, **_kwargs: FakeBundle()
    )
    monkeypatch.setattr(
        lr_stages, "parameter_state_diagnostics", lambda *_args: bounded
    )
    monkeypatch.setattr(
        lr_stages,
        "candidate_run_spec_v5",
        lambda *_args, **_kwargs: SimpleNamespace(
            to_dict=lambda: {"schema_version": "mnist-conv-run/v5"}
        ),
    )

    def fake_step(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        observed["steps"] += 1
        step = observed["steps"]
        return SimpleNamespace(
            loss=1.0,
            accuracy=0.5,
            sample_count=16,
            source_indices=(step,),
            transition=object(),
        )

    def fake_rows(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        observed["transitions"] += 1
        return [
            {
                "parameter": name,
                "bounded_gate": True,
                "gradient_rms": 1.0,
                "normalized_update": 0.001,
                "parameter_relative_update": 0.001,
                "projection_gate_eligible": True,
                "projection_efficiency": 1.0,
                "combined_bound_occupancy": 0.0,
            }
            for name in bounded
        ]

    monkeypatch.setattr(lr_stages, "training_step", fake_step)
    monkeypatch.setattr(lr_stages, "transition_rows", fake_rows)
    rates = {
        "ConvWeight_0": 0.003,
        "Bias_0": 0.003,
        "ConvWeight_1": 0.003,
        "Bias_1": 0.003,
        "DenseWeight_0": 0.003,
    }
    payload = {
        "row_id": baseline["row_id"],
        "candidate_role": "rho-conv-0p003--rho-dense-0p003",
        "candidate_stage": "baseline_grid",
        "rho_conv": 0.003,
        "rho_dense": 0.003,
        "median_units_by_weight": {
            "ConvWeight_0": 1.0,
            "ConvWeight_1": 1.0,
            "DenseWeight_0": 1.0,
        },
        "learning_rates_by_parameter": rates,
        "peak_learning_rate": 0.003,
    }

    report = lr_stages.execute_v6_shadow_benchmark_entry(
        study.data,
        root,
        baseline["row_id"],
        payload["candidate_role"],
        candidate_payload=payload,
        data_root=tmp_path / "mnist",
        download=False,
        device="cpu",
        ready_path=tmp_path / "barrier/ready.json",
        start_path=start,
        scratch_output_dir=tmp_path / "shadow",
    )

    assert observed == {"steps": 288, "transitions": 288}
    assert report["completed_steps"] == 288
    assert report["diagnostic_row_count"] == 3 * 288
    assert report["production_loop"] is True
    assert report["canonical_completion_published"] is False
    assert not list(root.glob("stages/**/complete.json"))
    assert "_CandidateStepLoop(" in inspect.getsource(
        lr_stages.execute_candidate_entry
    )
    assert "_CandidateStepLoop(" in inspect.getsource(
        lr_stages.execute_v6_shadow_benchmark_entry
    )


def test_v6_copy_and_checkpoint_publication_resume_after_partial_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "cache/output.bin"
    source.write_bytes(b"complete artifact")
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"partial")
    digest = sha256_file(source)

    lr_stages._copy_verified_artifact_to(
        source, destination, digest, "resume artifact"
    )
    assert destination.read_bytes() == source.read_bytes()
    destination.write_bytes(b"corrupt after publication")
    owner = tmp_path / "cache/complete.json"
    owner.write_text("{}\n")
    with pytest.raises(RuntimeError, match="immutable owners"):
        lr_stages._copy_verified_artifact_to(
            source,
            destination,
            digest,
            "owned artifact",
            immutable_owner_paths=(owner,),
        )

    rows = [
        {"architecture": "conv2", "scheme": scheme}
        for scheme in ("baseline", "ours", "legacy")
    ]

    class FakeRuntime:
        parameters = (object(),)

        @staticmethod
        def save(path: str | Path) -> None:
            Path(path).write_bytes(b"complete checkpoint")

    def fake_runtime(
        _study: Any,
        _row: Any,
        *,
        initialization_checkpoint: str | Path | None = None,
        **_kwargs: Any,
    ) -> FakeRuntime:
        if initialization_checkpoint is not None and Path(
            initialization_checkpoint
        ).read_bytes() != b"complete checkpoint":
            raise RuntimeError("partial checkpoint")
        return FakeRuntime()

    monkeypatch.setattr(lr_stages, "build_model_runtime", fake_runtime)
    monkeypatch.setattr(
        lr_stages, "parameter_tensor_digest", lambda *_args: "f" * 64
    )
    monkeypatch.setattr(
        lr_stages, "parameter_state_diagnostics", lambda *_args: {}
    )
    checkpoint = tmp_path / "initialization/conv2.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"interrupted checkpoint")
    record = lr_stages._checkpoint_contract(
        {"rows": rows}, "conv2", checkpoint_path=checkpoint, device="cpu"
    )
    assert checkpoint.read_bytes() == b"complete checkpoint"
    assert record["parameter_tensor_sha256"] == "f" * 64

    checkpoint.with_suffix(".json").write_text("{}\n")
    checkpoint.write_bytes(b"corrupt owned checkpoint")
    with pytest.raises(RuntimeError, match="partial checkpoint"):
        lr_stages._checkpoint_contract(
            {"rows": rows}, "conv2", checkpoint_path=checkpoint, device="cpu"
        )
    assert checkpoint.read_bytes() == b"corrupt owned checkpoint"


def test_v6_downstream_manifest_binds_every_consumed_output_and_completion(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(V6_CONFIG)
    atomic_write_json(tmp_path / "study.resolved.json", study.data, canonical=True)
    base = "stages/select_baseline/entries/selection"
    outputs = sorted(
        f"{base}/{name}"
        for name in (
            "achieved_relative_updates_heatmap.png",
            "epoch5_accuracy_loss_heatmap.png",
            "occupancy_heatmap.png",
            "projection_efficiency_heatmap.png",
            "raw_learning_rates_heatmap.png",
            "selection.csv",
            "selection.json",
        )
    )
    prior_entry = stage_entry(
        0,
        "selection",
        completion_path=f"{base}/complete.json",
        outputs=outputs,
    )
    prior_manifest = build_stage_manifest(
        study_dir=tmp_path,
        study_id=study.study_id,
        study_config_path=tmp_path / "study.resolved.json",
        code_provenance=code_provenance(),
        stage_name="select_baseline",
        entries=[prior_entry],
    )
    prior_path = tmp_path / "stages/select_baseline/manifest.json"
    publish_stage_manifest(prior_path, prior_manifest, study_dir=tmp_path)
    for output in outputs:
        path = tmp_path / output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(output.encode())
    publish_entry_completion(
        study_dir=tmp_path, manifest_path=prior_path, entry_id="selection"
    )
    publish_stage_completion(study_dir=tmp_path, manifest_path=prior_path)

    from experiments.mnist_conv import lr_study

    bound = lr_study._v6_completed_stage_upstreams(
        tmp_path, "select_baseline"
    )
    assert tmp_path / prior_entry["completion_path"] in bound
    assert all(tmp_path / output in bound for output in outputs)
    downstream_entry = stage_entry(
        0,
        "confirmation",
        completion_path="stages/confirmations/entries/confirmation/complete.json",
        outputs=["stages/confirmations/entries/confirmation/summary.json"],
    )
    downstream = build_stage_manifest(
        study_dir=tmp_path,
        study_id=study.study_id,
        study_config_path=tmp_path / "study.resolved.json",
        code_provenance=code_provenance(),
        stage_name="confirmations",
        entries=[downstream_entry],
        upstream_paths=bound,
    )
    downstream_path = tmp_path / "stages/confirmations/manifest.json"
    publish_stage_manifest(downstream_path, downstream, study_dir=tmp_path)
    (tmp_path / f"{base}/selection.json").write_text("tampered\n")
    with pytest.raises(LRArtifactError, match="current artifact hashes"):
        load_stage_manifest(downstream_path, study_dir=tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rho_dense", 0.1),
        ("candidate_stage", "baseline_grid"),
        ("scheme", "baseline"),
        ("row_id", "duplicate-row"),
    ],
)
def test_v6_finalization_rejects_confirmation_contract_mismatch(
    tmp_path: Path, field: str, value: object
) -> None:
    study = LRStudySpec.from_path(V6_CONFIG)
    pair = (0.003, 0.03)
    baseline = next(row for row in study.rows if row["scheme"] == "baseline")
    confirmations = [row for row in study.rows if row["scheme"] != "baseline"]
    entry = f"{baseline['row_id']}--grid-c02-d02"
    atomic_write_json(
        tmp_path / "stages/select_baseline/entries/selection/selection.json",
        {
            "status": "selected",
            "reason": None,
            "selected_entry_id": entry,
            "selected_rho_conv": pair[0],
            "selected_rho_dense": pair[1],
            "candidates": [
                {
                    "entry_id": entry,
                    "row_id": baseline["row_id"],
                    "scheme": "baseline",
                    "rho_conv": pair[0],
                    "rho_dense": pair[1],
                    "selected_for_confirmation": True,
                }
            ],
        },
        canonical=True,
    )
    atomic_write_json(
        tmp_path / f"stages/baseline_candidates/entries/{entry}/summary.json",
        _candidate_summary(
            row=baseline,
            candidate_stage="baseline_grid",
            rho_conv=pair[0],
            rho_dense=pair[1],
            loss=0.1,
            accuracy=0.91,
        ),
        canonical=True,
    )
    for index, row in enumerate(confirmations):
        configured_row_id = row["row_id"]
        summary = _candidate_summary(
            row=row,
            candidate_stage="confirmation",
            rho_conv=pair[0],
            rho_dense=pair[1],
            loss=0.1,
            accuracy=0.91,
        )
        if index == 0:
            if field in {"row_id", "scheme"}:
                summary["row"][field] = value
            else:
                summary[field] = value
        atomic_write_json(
            tmp_path
            / f"stages/confirmations/entries/{configured_row_id}--confirmation/summary.json",
            summary,
            canonical=True,
        )
    with pytest.raises(RuntimeError, match="Expected every finalization input"):
        execute_v6_finalization(study.data, tmp_path)


def _v6_cli_args(stage: str, *, executor: str = "local") -> SimpleNamespace:
    return SimpleNamespace(
        command="lr-study",
        stage=stage,
        config=None,
        study="/synthetic/v6-study",
        results_root=None,
        data_root="/synthetic/mnist",
        device="cpu",
        download=False,
        plan_only=False,
        executor=executor,
        workers=1,
        profile="profile.json" if executor == "slurm" else None,
        dry_run=False,
        manifest=None,
        entry_index=None,
        finalize_stage=False,
        shadow_benchmark=False,
        shadow_ready=None,
        shadow_start=None,
        shadow_scratch_output=None,
        shadow_start_timeout_seconds=900.0,
    )


def test_v6_cli_rejects_top_level_compute_even_with_auth_but_allows_login_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.mnist_conv import cli, lr_study

    study = LRStudySpec.from_path(V6_CONFIG)
    root = tmp_path / "study"
    monkeypatch.setenv(
        lr_stages.V6_R3_AUTHORIZATION_ENV,
        lr_stages.V6_R3_AUTHORIZATION_TOKEN,
    )
    monkeypatch.setattr(lr_study, "load_study", lambda _path: (root, study))
    with pytest.raises(ValueError, match="top-level local and generic Slurm"):
        cli._lr_study_command(_v6_cli_args("baseline_candidates"), provenance={})

    monkeypatch.setattr(
        lr_study,
        "publish_lr_stage_manifest",
        lambda *_args, **_kwargs: (
            {"entries": []},
            root / "stages/synthetic/manifest.json",
        ),
    )
    monkeypatch.setattr(
        lr_study,
        "execute_local_stage",
        lambda **_kwargs: {"status": "complete", "executor": "local"},
    )
    for stage in ("select_baseline", "finalize"):
        result = cli._lr_study_command(_v6_cli_args(stage), provenance={})
        assert result["planned"] is False
        assert result["executor"] == "local"

    with pytest.raises(ValueError, match="dedicated measured-memory pack launcher"):
        cli._lr_study_command(
            _v6_cli_args("select_baseline", executor="slurm"), provenance={}
        )


def test_v6_r3_authorization_is_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(lr_stages.V6_R3_AUTHORIZATION_ENV, raising=False)
    with pytest.raises(RuntimeError, match="exactly authorize"):
        lr_stages.require_v6_r3_authorization()
    monkeypatch.setenv(lr_stages.V6_R3_AUTHORIZATION_ENV, "umg@v100")
    with pytest.raises(RuntimeError, match="exactly authorize"):
        lr_stages.require_v6_r3_authorization()
    monkeypatch.setenv(
        lr_stages.V6_R3_AUTHORIZATION_ENV,
        lr_stages.V6_R3_AUTHORIZATION_TOKEN,
    )
    lr_stages.require_v6_r3_authorization()
