from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from experiments.mnist_conv import lr_stages, lr_study
from experiments.mnist_conv.identity import code_provenance, sha256_file
from experiments.mnist_conv.io import atomic_write_json, read_json
from experiments.mnist_conv.lr_artifacts import entry_is_complete
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments.mnist_conv.lr_v5_packing import (
    MEMORY_PREFLIGHT_SCHEMA_VERSION,
    create_v5_pack_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
V5_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_architecture_relative_rho_constant_sgd_bs16_v5.json"
)


class _RuntimeStudy:
    """Mutable fake-runtime view with the immutable study's real identity."""

    def __init__(self, source: LRStudySpec) -> None:
        self.data = copy.deepcopy(source.data)
        self.study_id = source.study_id

    @property
    def rows(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.data["rows"])


class _FakeBundle:
    train_indices = (0, 1)
    validation_indices = (2,)
    train_indices_hash = "c" * 64
    validation_indices_hash = "d" * 64
    validation_loader = object()

    def __init__(self, steps_per_epoch: int) -> None:
        self.train_loader = range(steps_per_epoch)

    @staticmethod
    def reset_train_shuffle() -> None:
        return None

    def train_batch_indices(self, *, num_epochs: int = 1):
        epoch = tuple((index,) for index in self.train_loader)
        return tuple(epoch for _ in range(num_epochs))

    def provenance(self, *, num_epochs: int) -> dict[str, Any]:
        return {
            "schema_version": "fake-mnist-split/v1",
            "num_epochs": num_epochs,
            "train_indices": list(self.train_indices),
            "validation_indices": list(self.validation_indices),
            "official_test_read": False,
        }


def _weight_units(row: Mapping[str, Any]) -> dict[str, float]:
    row_index = next(
        index
        for index, candidate in enumerate(
            LRStudySpec.from_path(V5_CONFIG).rows, start=1
        )
        if candidate["row_id"] == row["row_id"]
    )
    units = {
        "ConvWeight_0": 0.1 * row_index,
        "DenseWeight_0": 1.0 * row_index,
    }
    if row["architecture"] == "conv2":
        units["ConvWeight_1"] = 0.2 * row_index
    return units


class _FakeRuntime:
    def __init__(self, row: Mapping[str, Any], learning_rate: Any) -> None:
        self.row = dict(row)
        self.learning_rate = learning_rate
        self.units = _weight_units(row)
        names = ["ConvWeight_0", "Bias_0"]
        if row["architecture"] == "conv2":
            names.extend(("ConvWeight_1", "Bias_1"))
        names.append("DenseWeight_0")
        self.parameters = tuple(SimpleNamespace(name=name) for name in names)
        self.step = 0

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(
            f"{self.row['row_id']}:{self.step}".encode("utf-8")
        )


def test_v5_fake_runtime_executes_and_resumes_every_packed_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise audit -> probe -> packed candidates -> selection without MNIST/GPU.

    The production stage executors create every summary and completion marker.
    Only model/data primitives and the run length are replaced with a tiny,
    deterministic fake runtime.
    """

    study = LRStudySpec.from_path(V5_CONFIG)
    runtime_study = _RuntimeStudy(study)
    root = (
        tmp_path
        / "results/lr_studies"
        / f"{study.data['name']}--{study.study_id}"
    )
    root.mkdir(parents=True)
    atomic_write_json(root / "study.resolved.json", study.data, canonical=True)

    def fake_load_study(study_dir: str | Path):
        assert Path(study_dir).resolve() == root.resolve()
        return root.resolve(), runtime_study

    monkeypatch.setattr(lr_study, "load_study", fake_load_study)

    loader_calls: list[dict[str, Any]] = []

    def fake_loader(
        study_value: Mapping[str, Any],
        *_args: Any,
        **_kwargs: Any,
    ) -> _FakeBundle:
        assert study_value["dataset"]["official_test"] == {
            "enabled": False,
            "read_allowed": False,
        }
        loader_calls.append(dict(study_value["dataset"]["official_test"]))
        return _FakeBundle(
            int(study_value["candidate_training"]["steps_per_epoch"])
        )

    monkeypatch.setattr(lr_stages, "build_loader_bundle", fake_loader)

    def fake_checkpoint_contract(
        _study_value: Mapping[str, Any],
        architecture: str,
        *,
        checkpoint_path: Path,
        device: str,
    ) -> dict[str, Any]:
        assert device == "cpu"
        checkpoint_path.write_bytes(f"fake-{architecture}-init".encode("utf-8"))
        return {
            "schema_version": "mnist-conv-lr-initialization/v1",
            "architecture": architecture,
            "model_seed": 0,
            "checkpoint_path": checkpoint_path.name,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "parameter_tensor_sha256": ("1" if architecture == "conv1" else "2")
            * 64,
        }

    monkeypatch.setattr(lr_stages, "_checkpoint_contract", fake_checkpoint_contract)

    built_runtimes: list[_FakeRuntime] = []

    def fake_build_runtime(
        _study_value: Mapping[str, Any],
        row: Mapping[str, Any],
        *,
        learning_rate: Any,
        **_kwargs: Any,
    ) -> _FakeRuntime:
        runtime = _FakeRuntime(row, learning_rate)
        built_runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(lr_stages, "build_model_runtime", fake_build_runtime)

    def fake_parameter_diagnostics(parameters: Any) -> dict[str, dict[str, Any]]:
        return {
            parameter.name: {
                "bounded_gate": not parameter.name.startswith("Bias_"),
                "rms": 1.0,
                "combined_bound_occupancy": 0.0,
            }
            for parameter in parameters
        }

    monkeypatch.setattr(
        lr_stages, "parameter_state_diagnostics", fake_parameter_diagnostics
    )
    monkeypatch.setattr(
        lr_stages,
        "parameter_tensor_digest",
        lambda parameters: "e" * 64,
    )

    def fake_training_step(
        runtime: _FakeRuntime,
        batch: int,
        *,
        learning_rate: Any,
        **_kwargs: Any,
    ) -> SimpleNamespace:
        runtime.step += 1
        runtime.learning_rate = learning_rate
        return SimpleNamespace(
            loss=1.0,
            accuracy=0.75,
            sample_count=16,
            source_indices=(int(batch),),
            transition=runtime,
        )

    monkeypatch.setattr(lr_stages, "training_step", fake_training_step)

    def fake_transition_rows(
        runtime: _FakeRuntime,
        *,
        step: int,
        learning_rate: Any,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        rows = []
        for parameter in runtime.parameters:
            name = parameter.name
            bounded = name in runtime.units
            rate = (
                float(learning_rate[name])
                if isinstance(learning_rate, Mapping)
                else float(learning_rate)
            )
            relative_update = rate * runtime.units.get(name, 1.0)
            rows.append(
                {
                    "step": step,
                    "learning_rate": rate,
                    "parameter": name,
                    "parameter_kind": "bias" if name.startswith("Bias_") else "weight",
                    "bounded_gate": bounded,
                    "report_only": not bounded,
                    "gradient_rms": runtime.units.get(name, 0.01),
                    "proposed_update_rms": relative_update,
                    "normalized_update": relative_update / 100.0,
                    "span_normalized_update": relative_update / 100.0,
                    "initial_parameter_rms": 1.0 if bounded else None,
                    "parameter_relative_update": relative_update,
                    "proposed_bound_crossing_fraction": 0.0,
                    "lower_bound_occupancy": 0.0,
                    "upper_bound_occupancy": 0.0,
                    "combined_bound_occupancy": 0.0,
                    "projection_efficiency": 0.99,
                    "proposal_is_numerically_zero": False,
                    "projection_gate_eligible": bounded,
                }
            )
        return rows

    monkeypatch.setattr(lr_stages, "transition_rows", fake_transition_rows)

    def fake_validation(
        runtime: _FakeRuntime, _loader: object
    ) -> dict[str, Any]:
        rates = runtime.learning_rate
        assert isinstance(rates, Mapping)
        conv_alpha = float(rates["ConvWeight_0"]) * runtime.units["ConvWeight_0"]
        dense_alpha = float(rates["DenseWeight_0"]) * runtime.units["DenseWeight_0"]
        arm = (
            "historical_profile"
            if dense_alpha / conv_alpha > 5.0
            else "strict_equal"
        )
        center = float(
            study.data["target_policies"]["centers_by_architecture_and_arm"]
            [runtime.row["architecture"]][arm]
        )
        factor = conv_alpha / center
        role = min(
            {"lower": 1.0 / 3.0, "center": 1.0, "upper": 3.0},
            key=lambda name: abs(
                math.log(factor)
                - math.log(
                    {"lower": 1.0 / 3.0, "center": 1.0, "upper": 3.0}[name]
                )
            ),
        )
        loss = {"lower": 0.20, "center": 0.10, "upper": 0.13}[role]
        loss += 0.004 if arm == "historical_profile" else 0.0
        loss += {"baseline": 0.0, "ours": 0.001, "legacy": 0.002}[
            runtime.row["scheme"]
        ]
        # Decrease across the five fake epochs so the final checkpoint is best.
        loss += (5 - runtime.step) * 0.001
        return {
            "loss": loss,
            "accuracy": 0.92,
            "sample_count": 5000,
            "source_indices": (),
        }

    monkeypatch.setattr(lr_stages, "evaluate_validation", fake_validation)

    # Candidate diagnostics hold thousands of rows in production.  The fake
    # smoke needs only the declared CSV artifact, not a second serialization
    # test for rows already inspected through the JSON summaries.
    def fast_csv(
        path: str | Path,
        fieldnames: list[str],
        _rows: Any,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(",".join(fieldnames) + "\n", encoding="utf-8")
        return target

    monkeypatch.setattr(lr_stages, "atomic_write_csv", fast_csv)

    lr_stages.prepare_study_assets(
        study.data,
        root,
        data_root=tmp_path / "unused-mnist",
        download=False,
        device="cpu",
    )
    provenance = code_provenance()

    audit_manifest, audit_path = lr_study.publish_lr_stage_manifest(
        root, study, "audit", provenance
    )
    assert audit_manifest["stage_name"] == "audit"
    assert lr_study.execute_manifest_entry(
        study_dir=root,
        manifest_path=audit_path,
        entry_index=0,
        data_root=tmp_path,
        download=False,
        device="cpu",
        current_provenance=provenance,
    )["status"] == "complete"
    lr_study.finalize_stage(root, audit_path)

    probe_manifest, probe_path = lr_study.publish_lr_stage_manifest(
        root, study, "probe", provenance
    )
    probe_executions = [
        lr_study.execute_manifest_entry(
            study_dir=root,
            manifest_path=probe_path,
            entry_index=entry["entry_index"],
            data_root=tmp_path,
            download=False,
            device="cpu",
            current_provenance=provenance,
        )
        for entry in probe_manifest["entries"]
    ]
    assert {result["status"] for result in probe_executions} == {"complete"}
    lr_study.finalize_stage(root, probe_path)

    candidate_manifest, candidate_path = lr_study.publish_lr_stage_manifest(
        root, study, "candidates", provenance
    )
    preflight_path = root / "stages/candidates/fake-memory-preflight.json"
    atomic_write_json(
        preflight_path,
        {
            "schema_version": MEMORY_PREFLIGHT_SCHEMA_VERSION,
            "stage_manifest_sha256": sha256_file(candidate_path),
            "peak_gpu_memory_mib_by_entry_index": {
                str(index): 9000.0 for index in range(36)
            },
        },
        canonical=True,
    )
    pack_path = root / "stages/candidates/pack-manifest.json"
    pack_manifest = create_v5_pack_manifest(
        candidate_path, preflight_path, pack_path
    )
    assert len(pack_manifest["packs"]) == 12
    assert read_json(pack_path) == pack_manifest

    # Scale only the fake executor.  The published study and candidate
    # manifest above retain the immutable 17,190-step v5 contract.
    runtime_study.data["candidate_training"]["training_examples"] = 16
    runtime_study.data["candidate_training"]["steps_per_epoch"] = 1
    runtime_study.data["candidate_training"]["total_steps"] = 5
    runtime_study.data["candidate_training"]["schedule"]["final_step"] = 5
    monkeypatch.setattr(lr_stages, "CANDIDATE_TOTAL_STEPS", 5)

    run_spec_calls: list[str] = []

    def fake_run_spec(
        _study_value: Mapping[str, Any],
        _root: Path,
        row: Mapping[str, Any],
        role: str,
        **kwargs: Any,
    ) -> SimpleNamespace:
        run_spec_calls.append(f"{row['row_id']}--{role}")
        return SimpleNamespace(
            to_dict=lambda: {
                "schema_version": "mnist-conv-run/v4",
                "runtime": "deterministic-fake",
                "row_id": row["row_id"],
                "candidate_role": role,
                "alpha_arch": kwargs["alpha"],
            }
        )

    monkeypatch.setattr(lr_stages, "candidate_run_spec_v4", fake_run_spec)

    first_pass = []
    for pack in pack_manifest["packs"]:
        assert {
            candidate_manifest["entries"][index]["payload"]["architecture"]
            for index in pack["entry_indices"]
        } == {pack["architecture"]}
        for index in pack["entry_indices"]:
            first_pass.append(
                lr_study.execute_manifest_entry(
                    study_dir=root,
                    manifest_path=candidate_path,
                    entry_index=index,
                    data_root=tmp_path,
                    download=False,
                    device="cpu",
                    current_provenance=provenance,
                )
            )

    assert len(first_pass) == 36
    assert {result["status"] for result in first_pass} == {"complete"}
    assert {result["result_status"] for result in first_pass} == {"complete"}
    assert len(run_spec_calls) == 36
    candidate_runtime_count = sum(
        isinstance(runtime.learning_rate, Mapping) for runtime in built_runtimes
    )
    assert candidate_runtime_count == 36

    # Reusing the same immutable packed manifest must skip all completed work.
    second_pass = [
        lr_study.execute_manifest_entry(
            study_dir=root,
            manifest_path=candidate_path,
            entry_index=index,
            data_root=tmp_path,
            download=False,
            device="cpu",
            current_provenance=provenance,
        )
        for pack in pack_manifest["packs"]
        for index in pack["entry_indices"]
    ]
    assert len(second_pass) == 36
    assert {result["status"] for result in second_pass} == {"resumed_complete"}
    assert len(run_spec_calls) == 36

    for entry in candidate_manifest["entries"]:
        assert entry_is_complete(
            study_dir=root,
            manifest_path=candidate_path,
            entry_id=entry["entry_id"],
        )
        summary = read_json(
            root / "stages/candidates/entries" / entry["entry_id"] / "summary.json"
        )
        assert summary["schema_version"] == "mnist-conv-lr-candidate-result/v4"
        assert summary["training_completed"] is True
        assert summary["completed_steps"] == 5
        assert summary["admissible"] is True

    candidate_final = lr_study.finalize_stage(root, candidate_path)
    assert candidate_final["stage"] == "candidates"
    assert candidate_final["status"] == "complete"

    select_manifest, select_path = lr_study.publish_lr_stage_manifest(
        root, study, "select", provenance
    )
    selected = lr_study.execute_manifest_entry(
        study_dir=root,
        manifest_path=select_path,
        entry_index=0,
        data_root=tmp_path,
        download=False,
        device="cpu",
        current_provenance=provenance,
    )
    assert selected == {
        "entry_index": 0,
        "entry_id": "selection",
        "status": "complete",
        "result_status": "selected",
    }
    assert lr_study.execute_manifest_entry(
        study_dir=root,
        manifest_path=select_path,
        entry_index=0,
        data_root=tmp_path,
        download=False,
        device="cpu",
        current_provenance=provenance,
    )["status"] == "resumed_complete"
    assert lr_study.finalize_stage(root, select_path)["status"] == "complete"

    selection = read_json(
        root / "stages/select/entries/selection/selection.json"
    )
    assert selection["status"] == "selected"
    assert selection["architectures"]["conv1"]["selected_arm"] == "strict_equal"
    assert selection["architectures"]["conv2"]["selected_arm"] == "strict_equal"
    assert all(call == {"enabled": False, "read_allowed": False} for call in loader_calls)
    assert not (root / "stages/range").exists()
