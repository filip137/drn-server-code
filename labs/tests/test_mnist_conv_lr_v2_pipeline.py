from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.mnist_conv import lr_stages
from experiments.mnist_conv.cli import main
from experiments.mnist_conv.collection import (
    _case_linked_fields,
    _comparison_contract_fields,
    _summary_row,
    collect_sweep,
)
from experiments.mnist_conv.io import atomic_write_json
from experiments.mnist_conv.layout import ResultLayout
from experiments.mnist_conv.lr_stages import (
    candidate_run_spec_v2,
    execute_no_candidate_entry,
    execute_selection,
)
from experiments.mnist_conv.lr_study import (
    _entries,
    _require_current_source,
    plan_only_result,
)
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments.mnist_conv.manifest import publish_manifest
from experiments.mnist_conv.specs import (
    LR_STAGE_RUN_SCHEMA_VERSIONS,
    RUN_SCHEMA_VERSION,
    RunSpec,
    SpecValidationError,
    SweepSpec,
    require_generic_run_schema,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_CONFIG = REPO_ROOT / "configs/conv/hardsigmoid_lr_study_sgd_bs16_v1.json"
V1_EXAMPLE = REPO_ROOT / "configs/conv/run_v1.diagnostic.example.json"
V3_STUDY_CONFIG = (
    REPO_ROOT / "configs/conv/hardsigmoid_lr_relative_rho_sgd_bs16_v3.json"
)
GENERIC_PROVENANCE = {
    "git_revision": "1" * 40,
    "dirty_source_digest": "2" * 64,
    "effective_code_fingerprint": "3" * 64,
}


class _FakeBundle:
    validation_indices_hash = "a" * 64

    def train_batch_indices(self, *, num_epochs=1):
        return tuple(((epoch, epoch + 1),) for epoch in range(num_epochs))


def test_candidate_retry_overwrites_stale_best_checkpoint_before_first_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = json.loads(V3_STUDY_CONFIG.read_text(encoding="utf-8"))
    row = study["rows"][0]
    role = "fast"
    root = tmp_path / "study"
    checkpoint = root / "initialization" / "conv1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"initialization")
    range_dir = root / "stages" / "range" / "entries" / row["row_id"]
    range_dir.mkdir(parents=True)
    atomic_write_json(
        range_dir / "summary.json",
        {
            "rho_unit": 1.0,
            "candidates": {
                "status": "resolved",
                role: {"rho_target": 1e-3, "learning_rate": 0.25},
            },
        },
        canonical=True,
    )
    output_dir = (
        root
        / "stages"
        / "candidates"
        / "entries"
        / f"{row['row_id']}--{role}"
    )
    output_dir.mkdir(parents=True)
    best_checkpoint = output_dir / "best_validation.pt"
    best_checkpoint.write_bytes(b"stale-prior-attempt")

    saved_paths: list[Path] = []

    class FakeRuntime:
        parameters = object()

        def save(self, path: str | Path) -> Path:
            target = Path(path)
            target.write_bytes(b"current-runtime-state")
            saved_paths.append(target)
            return target

    bundle = SimpleNamespace(
        train_loader=[object()],
        validation_loader=object(),
        train_indices_hash="train-indices",
        validation_indices_hash="validation-indices",
        reset_train_shuffle=lambda: None,
    )
    monkeypatch.setattr(
        lr_stages, "build_model_runtime", lambda *args, **kwargs: FakeRuntime()
    )
    monkeypatch.setattr(
        lr_stages, "build_loader_bundle", lambda *args, **kwargs: bundle
    )
    monkeypatch.setattr(
        lr_stages,
        "parameter_state_diagnostics",
        lambda parameters: {
            "ConvWeight_0": {
                "bounded_gate": True,
                "combined_bound_occupancy": 0.5,
                "rms": 2.0,
            }
        },
    )
    monkeypatch.setattr(
        lr_stages,
        "candidate_run_spec_v2",
        lambda *args, **kwargs: SimpleNamespace(
            to_dict=lambda: {"schema_version": "test-run-spec"}
        ),
    )
    monkeypatch.setattr(
        lr_stages, "parameter_tensor_digest", lambda parameters: "tensor-digest"
    )

    def fail_before_validation(*args, **kwargs):
        raise lr_stages.LRStudyNumericalError("synthetic numerical failure")

    monkeypatch.setattr(lr_stages, "training_step", fail_before_validation)

    summary = lr_stages.execute_candidate_entry(
        study,
        root,
        row["row_id"],
        role,
        data_root=tmp_path / "unused-data",
        download=False,
        device="cpu",
    )

    assert summary["best_validation_epoch"] is None
    assert summary["inadmissible_reason"] == "numerical_failure"
    assert best_checkpoint.read_bytes() == b"current-runtime-state"
    assert saved_paths[-1] == best_checkpoint


def _study_tree(tmp_path: Path, study: LRStudySpec, row: dict) -> Path:
    root = (
        tmp_path
        / "results"
        / "lr_studies"
        / f"study--{study.study_id}"
    )
    (root / "initialization").mkdir(parents=True)
    (root / "initialization" / f"{row['architecture']}.pt").write_bytes(b"checkpoint")
    atomic_write_json(
        root / "initialization" / f"{row['architecture']}.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )
    for stage in ("probe", "range"):
        directory = root / "stages" / stage / "entries" / row["row_id"]
        directory.mkdir(parents=True)
        atomic_write_json(directory / "summary.json", {"stage": stage}, canonical=True)
    return root


def _candidate_spec(tmp_path: Path, row_index: int, peak: float) -> RunSpec:
    study = LRStudySpec.from_path(STUDY_CONFIG)
    row = study.rows[row_index]
    root = _study_tree(tmp_path, study, row)
    return candidate_run_spec_v2(
        study.data,
        root,
        row,
        "middle",
        rho_target=1e-3,
        rho_unit=0.02,
        peak_learning_rate=peak,
        bundle=_FakeBundle(),
    )


@pytest.mark.parametrize("schema_version", sorted(LR_STAGE_RUN_SCHEMA_VERSIONS))
def test_generic_surface_guard_rejects_every_lr_candidate_schema(
    schema_version: str,
) -> None:
    with pytest.raises(
        SpecValidationError,
        match=r"LR-stage candidate schemas .* must use .*lr-study",
    ):
        require_generic_run_schema(schema_version, surface="run")


def test_generic_surface_guard_preserves_v1() -> None:
    require_generic_run_schema(RUN_SCHEMA_VERSION, surface="run")


def test_generic_run_plan_rejects_lr_candidate_bundle_before_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "candidate.v2.json"
    config.write_text(
        json.dumps(_candidate_spec(tmp_path / "candidate", 0, 0.05).to_dict()),
        encoding="utf-8",
    )
    results_root = tmp_path / "results"

    with pytest.raises(SystemExit) as failure:
        main(
            [
                "run",
                "--config",
                str(config),
                "--results-root",
                str(results_root),
                "--plan-only",
            ],
            source_provenance=GENERIC_PROVENANCE,
        )

    assert failure.value.code == 2
    assert "generic MNIST Conv run surface" in capsys.readouterr().err
    assert not results_root.exists()


def test_generic_sweep_plan_rejects_lr_candidate_bundle_before_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate = _candidate_spec(tmp_path / "candidate", 0, 0.05)
    config = tmp_path / "sweep.v2.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "mnist-conv-sweep/v1",
                "name": "invalid-generic-lr-sweep",
                "base_run": candidate.to_dict(),
                "axes": [],
                "cases": [{"id": "baseline", "set": {}}],
                "varying_fields": [],
                "collection": {
                    "expected_seeds": [0],
                    "required_cases": ["baseline"],
                    "group_by": ["/run/architecture/profile"],
                },
            }
        ),
        encoding="utf-8",
    )
    results_root = tmp_path / "results"

    with pytest.raises(SystemExit) as failure:
        main(
            [
                "sweep",
                "--config",
                str(config),
                "--results-root",
                str(results_root),
                "--plan-only",
            ],
            source_provenance=GENERIC_PROVENANCE,
        )

    assert failure.value.code == 2
    assert "generic MNIST Conv sweep surface" in capsys.readouterr().err
    assert not results_root.exists()


def test_generic_v1_run_plan_remains_supported(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "run",
                "--config",
                str(V1_EXAMPLE),
                "--results-root",
                str(tmp_path / "results"),
                "--plan-only",
            ],
            source_provenance=GENERIC_PROVENANCE,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["planned"] is True


def test_v1_remains_readable_and_v2_records_validation_schedule_and_diagnostics(
    tmp_path: Path,
) -> None:
    assert RunSpec.from_path(V1_EXAMPLE).data["schema_version"] == "mnist-conv-run/v1"

    spec = _candidate_spec(tmp_path, 0, 0.05)
    run = spec.data["run"]
    assert spec.data["schema_version"] == "mnist-conv-run/v2"
    assert run["dataset"]["batch_size"] == 16
    assert run["dataset"]["train_shuffle_seed"] == 0
    assert run["dataset"]["validation"] == {
        "source": "mnist_train",
        "size": 5000,
        "samples_per_class": 500,
        "split_seed": 0,
        "batch_size": 128,
        "stratified": True,
        "official_test_enabled": False,
        "indices_sha256": "a" * 64,
    }
    assert run["training"]["peak_learning_rate"] == 0.05
    assert run["training"]["schedule"]["warmup_steps"] == 860
    assert run["training"]["schedule"]["total_steps"] == 17190
    assert run["training"]["checkpoint_rule"] == "min_validation_loss"
    assert run["training"]["diagnostics"]["bias_global_gate"] is False
    assert run["lr_provenance"]["candidate_role"] == "middle"


def test_v2_rejects_batch_or_schedule_drift(tmp_path: Path) -> None:
    value = _candidate_spec(tmp_path, 0, 0.05).to_dict()
    value["run"]["dataset"]["batch_size"] = 32
    with pytest.raises(SpecValidationError, match="exactly 16"):
        RunSpec.from_dict(value)

    value = _candidate_spec(tmp_path / "second", 0, 0.05).to_dict()
    value["run"]["training"]["schedule"]["warmup_steps"] = 859
    with pytest.raises(SpecValidationError):
        RunSpec.from_dict(value)


def test_plan_only_reports_six_probe_six_range_and_eighteen_candidates(tmp_path: Path) -> None:
    study = LRStudySpec.from_path(STUDY_CONFIG)
    root = tmp_path / f"study--{study.study_id}"
    assert plan_only_result(study, root, "probe")["entry_count"] == 6
    assert plan_only_result(study, root, "range")["entry_count"] == 6
    assert plan_only_result(study, root, "candidates")["entry_count"] == 18
    assert plan_only_result(study, root, "select")["entry_count"] == 1


def test_v2_collection_contract_allows_row_specific_gain_tk_and_peak_lr(
    tmp_path: Path,
) -> None:
    baseline = _candidate_spec(tmp_path / "baseline", 0, 0.01)
    ours = _candidate_spec(tmp_path / "ours", 1, 0.04)

    assert _comparison_contract_fields(baseline) == _comparison_contract_fields(ours)
    baseline_case = _case_linked_fields(baseline)
    ours_case = _case_linked_fields(ours)
    assert baseline_case["case_input_gain_mismatch"] != ours_case["case_input_gain_mismatch"]
    assert baseline_case["case_learning_rate_mismatch"] != ours_case["case_learning_rate_mismatch"]
    # Conv1 currently happens to share T/K; Conv2 proves row-specific T/K is
    # represented as a case-linked value rather than a table-wide contract.
    conv2_baseline = _candidate_spec(tmp_path / "conv2-base", 3, 0.01)
    conv2_ours = _candidate_spec(tmp_path / "conv2-ours", 4, 0.04)
    assert _comparison_contract_fields(conv2_baseline) == _comparison_contract_fields(conv2_ours)
    assert (
        _case_linked_fields(conv2_baseline)["case_solver_tk_mismatch"]
        != _case_linked_fields(conv2_ours)["case_solver_tk_mismatch"]
    )


def test_v2_collection_summary_uses_peak_lr_and_validation_metrics(
    tmp_path: Path,
) -> None:
    spec = _candidate_spec(tmp_path, 0, 0.05)
    row = _summary_row(
        {
            "job_index": 0,
            "logical_key": "candidate",
            "case_id": "baseline",
            "run_id": "run_" + "1" * 64,
        },
        spec,
        "complete",
        {
            "best_validation_epoch": 4,
            "best_validation_loss": 0.125,
            "final_validation_loss": 0.13,
            "final_validation_accuracy": 0.96,
        },
        "runs/candidate",
        ["category_not_final"],
        {"git_revision": None, "dirty_source_digest": None},
        "2" * 64,
        None,
    )

    assert json.loads(row["learning_rate"]) == 0.05
    assert row["peak_learning_rate"] == 0.05
    assert json.loads(row["lr_schedule"])["warmup_steps"] == 860
    assert row["train_shuffle_seed"] == 0
    assert row["validation_indices_sha256"] == "a" * 64
    assert row["best_validation_epoch"] == 4
    assert row["best_validation_loss"] == 0.125
    assert row["final_validation_loss"] == 0.13
    assert row["final_validation_accuracy"] == 0.96
    assert row["best_test_accuracy"] == ""


def test_generic_collection_rejects_v2_sweep_before_writing_summary(
    tmp_path: Path,
) -> None:
    spec = _candidate_spec(tmp_path / "candidate", 0, 0.05)
    sweep = SweepSpec.from_dict(
        {
            "schema_version": "mnist-conv-sweep/v1",
            "name": "v2-collection-smoke",
            "base_run": spec.to_dict(),
            "axes": [],
            "cases": [{"id": "baseline", "set": {}}],
            "varying_fields": [],
            "collection": {
                "expected_seeds": [0],
                "required_cases": ["baseline"],
                "group_by": ["/run/architecture/profile"],
            },
        }
    )
    layout = ResultLayout(tmp_path / "results")
    _, manifest_path = publish_manifest(
        sweep,
        layout,
        GENERIC_PROVENANCE,
    )

    with pytest.raises(
        SpecValidationError,
        match="generic MNIST Conv collection surface",
    ):
        collect_sweep(manifest_path, layout, allow_incomplete=True)

    assert not (manifest_path.parent / "collection.json").exists()
    assert not (manifest_path.parent / "summary.partial.csv").exists()


def test_all_unresolved_rows_complete_zero_work_candidate_handoff(
    tmp_path: Path,
) -> None:
    study = LRStudySpec.from_path(STUDY_CONFIG)
    root = tmp_path / f"study--{study.study_id}"
    for row in study.rows:
        directory = root / "stages" / "range" / "entries" / row["row_id"]
        directory.mkdir(parents=True)
        atomic_write_json(
            directory / "summary.json",
            {
                "candidates": {
                    "status": "unresolved",
                    "reason": "loss_never_decreased",
                }
            },
            canonical=True,
        )

    entries = _entries(study, root, "candidates")
    assert len(entries) == 1
    assert entries[0]["entry_id"] == "no-resolved-candidates"
    assert entries[0]["payload"]["no_op"] is True

    no_op = execute_no_candidate_entry(study.data, root)
    assert no_op["candidate_count"] == 0
    assert no_op["reason"] == "all_range_rows_unresolved"
    assert len(no_op["rows"]) == 6

    selection = execute_selection(study.data, root)
    assert selection["status"] == "unresolved"
    assert len(selection["rows"]) == 6
    assert all(row["reason"] == "loss_never_decreased" for row in selection["rows"])
    select_dir = root / "stages" / "select" / "entries" / "selection"
    assert (select_dir / "selection.json").is_file()
    assert (select_dir / "selected_raw_peak_lr_vs_input_gain.png").is_file()


def test_worker_source_gate_is_portable_but_content_strict() -> None:
    planned = {
        "git_revision": "1" * 40,
        "dirty_source_digest": "2" * 64,
        "effective_code_fingerprint": "3" * 64,
    }
    copied_checkout = {
        "git_revision": None,
        "dirty_source_digest": "3" * 64,
        "effective_code_fingerprint": "3" * 64,
    }
    _require_current_source(planned, copied_checkout)

    copied_checkout["effective_code_fingerprint"] = "4" * 64
    with pytest.raises(RuntimeError, match="effective source fingerprint"):
        _require_current_source(planned, copied_checkout)
