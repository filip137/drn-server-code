from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from experiments.mnist_conv import lr_stages
from experiments.mnist_conv.identity import code_provenance
from experiments.mnist_conv.io import atomic_write_json, read_json
from experiments.mnist_conv.lr_artifacts import (
    build_stage_manifest,
    entry_is_complete,
    publish_stage_manifest,
    stage_entry,
)
from experiments.mnist_conv.lr_study import execute_manifest_entry
from experiments.mnist_conv.lr_study_spec import LRStudySpec


REPO_ROOT = Path(__file__).resolve().parents[2]
V5_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_architecture_relative_rho_constant_sgd_bs16_v5.json"
)

_PARAMETERS = ("ConvWeight_0", "DenseWeight_0")
_ROLE = "strict_equal--center"


def _candidate_payload(row_id: str) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "architecture": "conv1",
        "scheme": "baseline",
        "candidate_role": _ROLE,
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


def _study_root(tmp_path: Path, study: LRStudySpec) -> Path:
    root = tmp_path / "results/lr_studies" / f"study--{study.study_id}"
    checkpoint = root / "initialization/conv1.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixed-initialization")
    atomic_write_json(
        root / "initialization/conv1.json",
        {"parameter_tensor_sha256": "b" * 64},
        canonical=True,
    )
    probe_dir = root / "stages/probe/entries" / study.rows[0]["row_id"]
    probe_dir.mkdir(parents=True)
    atomic_write_json(probe_dir / "summary.json", {}, canonical=True)
    atomic_write_json(root / "study.resolved.json", study.data, canonical=True)
    return root


def _install_candidate_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loss_at_step: Callable[[int], float] = lambda _step: 1.0,
    gradient_at_step: Callable[[str, int], float] = lambda _name, _step: 1.0,
    occupancy_at_step: Callable[[str, int], float] = lambda _name, _step: 0.0,
    numerical_failure_step: int | None = None,
) -> dict[str, Any]:
    observed: dict[str, Any] = {"attempts": 0, "saves": []}

    class FakeRuntime:
        parameters = (object(),)

        @staticmethod
        def save(path: str | Path) -> None:
            target = Path(path)
            target.write_bytes(b"synthetic-runtime-state")
            observed["saves"].append(target)

    class FakeBundle:
        train_loader = (object(),) * 64
        validation_loader = object()
        train_indices_hash = "c" * 64
        validation_indices_hash = "d" * 64

        @staticmethod
        def reset_train_shuffle() -> None:
            return None

    diagnostics = {
        name: {
            "bounded_gate": True,
            "rms": 1.0,
            "combined_bound_occupancy": 0.0,
        }
        for name in _PARAMETERS
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
    monkeypatch.setattr(
        lr_stages,
        "evaluate_validation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a failed synthetic candidate must stop before validation")
        ),
    )

    def fake_training_step(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        observed["attempts"] += 1
        step = observed["attempts"]
        if numerical_failure_step == step:
            raise lr_stages.LRStudyNumericalError(
                f"synthetic non-finite update at step {step}"
            )
        return SimpleNamespace(
            loss=loss_at_step(step),
            accuracy=0.5,
            sample_count=16,
            source_indices=(step,),
            transition=object(),
        )

    def fake_transition_rows(*_args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        step = int(kwargs["step"])
        return [
            {
                "parameter": name,
                "bounded_gate": True,
                "gradient_rms": gradient_at_step(name, step),
                "normalized_update": 0.001,
                "parameter_relative_update": 0.001,
                "projection_gate_eligible": True,
                "projection_efficiency": 1.0,
                "combined_bound_occupancy": occupancy_at_step(name, step),
            }
            for name in _PARAMETERS
        ]

    monkeypatch.setattr(lr_stages, "training_step", fake_training_step)
    monkeypatch.setattr(lr_stages, "transition_rows", fake_transition_rows)
    return observed


def _execute_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    **harness_options: Any,
) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    study = LRStudySpec.from_path(V5_CONFIG)
    row_id = study.rows[0]["row_id"]
    root = _study_root(tmp_path, study)
    observed = _install_candidate_harness(monkeypatch, **harness_options)
    summary = lr_stages.execute_candidate_entry(
        study.data,
        root,
        row_id,
        _ROLE,
        candidate_payload=_candidate_payload(row_id),
        data_root=tmp_path / "unused-mnist",
        download=False,
        device="cpu",
    )
    return root, row_id, observed, summary


@pytest.mark.parametrize(
    ("kind", "harness_options", "confirmed_step", "parameter"),
    [
        (
            "loss_ema_explosion",
            {"loss_at_step": lambda step: 1_000.0 if step >= 33 else 1.0},
            40,
            None,
        ),
        (
            "gradient_rms_explosion",
            {
                "gradient_at_step": lambda name, step: (
                    101.0 if name == "ConvWeight_0" and step >= 33 else 1.0
                )
            },
            40,
            "ConvWeight_0",
        ),
        (
            "bound_occupancy_increase",
            {
                "occupancy_at_step": lambda name, step: (
                    0.21 if name == "ConvWeight_0" and step >= 33 else 0.0
                )
            },
            48,
            "ConvWeight_0",
        ),
    ],
)
def test_v5_candidate_sustained_safety_gates_backdate_to_first_bad_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    harness_options: dict[str, Any],
    confirmed_step: int,
    parameter: str | None,
) -> None:
    root, row_id, observed, summary = _execute_candidate(
        tmp_path, monkeypatch, **harness_options
    )

    expected_failure = {
        "kind": kind,
        "parameter": parameter,
        "onset_step": 33,
        "confirmed_step": confirmed_step,
    }
    expected_parameter = parameter or "global"
    assert observed["attempts"] == confirmed_step
    assert summary["status"] == "failed"
    assert summary["training_completed"] is False
    assert summary["admissible"] is False
    assert summary["accuracy_gate_passed"] is False
    assert summary["attempted_steps"] == confirmed_step
    assert summary["completed_steps"] == confirmed_step
    assert summary["inadmissible_reason"] == (
        f"{kind}:{expected_parameter}:onset=33"
    )
    assert summary["safety_gate_failure"] == expected_failure
    assert summary["numerical_failure_detail"] is None
    assert summary["validation_metrics"] == []
    assert summary["final_validation_loss"] is None
    assert summary["final_validation_accuracy"] is None

    output_dir = root / "stages/candidates/entries" / f"{row_id}--{_ROLE}"
    assert read_json(output_dir / "summary.json") == summary
    assert read_json(output_dir / "validation.json") == []
    assert (output_dir / "best_validation.pt").is_file()
    assert (output_dir / "final.pt").is_file()
    with (output_dir / "step_log.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == confirmed_step
    assert rows[-1]["status"] == "complete"


def test_v5_numerical_failure_is_an_immutable_completed_attempt_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study = LRStudySpec.from_path(V5_CONFIG)
    row_id = study.rows[0]["row_id"]
    entry_id = f"{row_id}--{_ROLE}"
    root = _study_root(tmp_path, study)
    observed = _install_candidate_harness(
        monkeypatch,
        numerical_failure_step=33,
    )
    output_base = f"stages/candidates/entries/{entry_id}"
    outputs = sorted(
        f"{output_base}/{name}"
        for name in (
            "best_validation.pt",
            "final.pt",
            "minibatches.json",
            "parameter_diagnostics.csv",
            "run_spec.v4.json",
            "step_log.csv",
            "summary.json",
            "validation.json",
        )
    )
    provenance = code_provenance()
    manifest = build_stage_manifest(
        study_dir=root,
        study_id=study.study_id,
        study_config_path=root / "study.resolved.json",
        code_provenance=provenance,
        stage_name="candidates",
        entries=[
            stage_entry(
                0,
                entry_id,
                completion_path=f"{output_base}/complete.json",
                outputs=outputs,
                payload=_candidate_payload(row_id),
            )
        ],
    )
    manifest_path = root / "stages/candidates/manifest.json"
    publish_stage_manifest(manifest_path, manifest, study_dir=root)

    result = execute_manifest_entry(
        study_dir=root,
        manifest_path=manifest_path,
        entry_index=0,
        data_root=tmp_path / "unused-mnist",
        download=False,
        device="cpu",
        current_provenance=provenance,
    )

    output_dir = root / output_base
    summary = read_json(output_dir / "summary.json")
    assert result["status"] == "complete"
    assert result["result_status"] == "failed"
    assert observed["attempts"] == 33
    assert summary["status"] == "failed"
    assert summary["training_completed"] is False
    assert summary["admissible"] is False
    assert summary["attempted_steps"] == 33
    assert summary["completed_steps"] == 32
    assert summary["inadmissible_reason"] == "numerical_failure"
    assert summary["numerical_failure_detail"] == (
        "synthetic non-finite update at step 33"
    )
    assert summary["safety_gate_failure"] is None
    assert summary["validation_metrics"] == []
    with (output_dir / "step_log.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 33
    assert rows[-1]["status"] == "failed"
    assert rows[-1]["failure_reason"] == (
        "synthetic non-finite update at step 33"
    )
    assert entry_is_complete(
        study_dir=root,
        manifest_path=manifest_path,
        entry_id=entry_id,
    )

    resumed = execute_manifest_entry(
        study_dir=root,
        manifest_path=manifest_path,
        entry_index=0,
        data_root=tmp_path / "unused-mnist",
        download=False,
        device="cpu",
        current_provenance=provenance,
    )
    assert resumed["status"] == "resumed_complete"
    assert observed["attempts"] == 33
