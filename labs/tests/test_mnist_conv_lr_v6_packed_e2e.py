from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from experiments.mnist_conv import lr_stages as lr_stages_module
from experiments.mnist_conv import lr_study as lr_study_module
from experiments.mnist_conv.identity import code_provenance
from experiments.mnist_conv.io import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
)
from experiments.mnist_conv.lr_artifacts import validate_stage_completion
from experiments.mnist_conv.lr_stages import execute_v6_shadow_benchmark_entry
from experiments.mnist_conv.lr_study import (
    create_study,
    execute_manifest_entry,
    finalize_stage,
    publish_lr_stage_manifest,
)
from experiments.mnist_conv.lr_study_spec import LRStudySpec
from experiments.mnist_conv.lr_v6_packing import (
    CONCURRENCY_LEVELS,
    MEASURED_STEPS,
    R3_AUTHORIZATION_TOKEN,
    build_benchmark_report,
    create_v6_pack_manifest,
)
from experiments.mnist_conv.lr_v6_wave_executor import execute_packed_waves


REPO_ROOT = Path(__file__).resolve().parents[2]
V6_CONFIG = (
    REPO_ROOT
    / "configs/conv/hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json"
)


def _benchmark_levels() -> list[dict[str, Any]]:
    # Four is deliberately the largest viable level, giving four deterministic
    # baseline waves while keeping this orchestration smoke tiny.
    throughput = {1: 10.0, 2: 19.0, 4: 35.0, 8: 0.0, 12: 0.0, 16: 0.0}
    levels = []
    for concurrency in CONCURRENCY_LEVELS:
        failed = 0 if concurrency <= 4 else concurrency
        completed = concurrency - failed
        levels.append(
            {
                "concurrency": concurrency,
                "completed_children": completed,
                "failed_children": failed,
                "combined_peak_gpu_memory_mib": 1000.0 * concurrency,
                "peak_host_rss_mib": 500.0 * concurrency,
                "mean_gpu_utilization_percent": 75.0,
                "aggregate_successful_steps_per_second": throughput[concurrency],
                "successful_measured_steps": completed * MEASURED_STEPS,
                "wall_seconds": 10.0,
                "child_failures": [
                    {"child_index": child, "returncode": 1}
                    for child in range(failed)
                ],
            }
        )
    return levels


def _write_stub_outputs(directory: Path, names: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = directory / name
        if name.endswith(".json"):
            atomic_write_json(path, {"stub": name}, canonical=True)
        else:
            atomic_write_bytes(path, f"reduced-runtime:{name}\n".encode())


def _install_reduced_science(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, str]],
) -> None:
    """Replace only expensive model/data work; retain real stage orchestration."""

    def audit(
        study: dict[str, Any],
        study_dir: str | Path,
        *,
        data_root: str | Path,
        download: bool,
        device: str,
    ) -> dict[str, Any]:
        del study, data_root, download, device
        root = Path(study_dir)
        calls.append(("audit", "reuse-audit"))
        atomic_write_json(
            root / "split/indices.json",
            {
                "train_indices": [0, 1],
                "validation_indices": [2],
                "official_test_read": False,
            },
            canonical=True,
        )
        atomic_write_json(
            root / "split/provenance.json",
            {
                "train_indices_sha256": "train-reduced",
                "validation_indices_sha256": "validation-reduced",
                "official_test_read": False,
            },
            canonical=True,
        )
        atomic_write_bytes(root / "initialization/conv2.pt", b"conv2-seed0\n")
        atomic_write_json(
            root / "initialization/conv2.json",
            {
                "architecture": "conv2",
                "model_seed": 0,
                "parameter_tensor_sha256": "tensor-reduced",
                "checkpoint_sha256": "checkpoint-reduced",
            },
            canonical=True,
        )
        atomic_write_json(
            root / "reuse/import_receipt.json",
            {
                "schema_version": "mnist-conv-lr-v6-reuse/v1",
                "status": "complete",
                "official_test_read": False,
            },
            canonical=True,
        )
        output = root / "stages/audit/entries/reuse-audit"
        atomic_write_bytes(output / "reuse_audit.csv", b"status\ncomplete\n")
        summary = {
            "schema_version": "mnist-conv-lr-v6-reuse-audit/v1",
            "status": "complete",
            "official_test_read": False,
        }
        atomic_write_json(output / "summary.json", summary, canonical=True)
        return summary

    def probe(
        study: dict[str, Any],
        study_dir: str | Path,
        row_id: str,
        *,
        data_root: str | Path,
        download: bool,
        device: str,
    ) -> dict[str, Any]:
        del data_root, download, device
        root = Path(study_dir)
        row = next(item for item in study["rows"] if item["row_id"] == row_id)
        calls.append(("probe", row_id))
        scheme_scale = {"baseline": 1.0, "ours": 2.0, "legacy": 4.0}[
            row["scheme"]
        ]
        units = {
            "ConvWeight_0": 1.0 * scheme_scale,
            "ConvWeight_1": 2.0 * scheme_scale,
            "DenseWeight_0": 4.0 * scheme_scale,
        }
        groups = {
            "ConvWeight_0": ["ConvWeight_0", "Bias_0"],
            "ConvWeight_1": ["ConvWeight_1", "Bias_1"],
            "DenseWeight_0": ["DenseWeight_0"],
        }
        output = root / "stages/probe/entries" / row_id
        _write_stub_outputs(
            output,
            ("minibatches.json", "parameter_diagnostics.csv", "step_log.csv"),
        )
        summary = {
            "schema_version": "mnist-conv-lr-probe-result/v6-smoke",
            "status": "complete",
            "row": row,
            "median_units_by_weight": units,
            "bias_weight_lr_groups": groups,
            "official_test_read": False,
        }
        atomic_write_json(output / "summary.json", summary, canonical=True)
        return summary

    def candidate(
        study: dict[str, Any],
        study_dir: str | Path,
        row_id: str,
        role: str,
        *,
        candidate_payload: dict[str, Any] | None,
        output_stage: str,
        data_root: str | Path,
        download: bool,
        device: str,
    ) -> dict[str, Any]:
        del data_root, download, device
        assert candidate_payload is not None
        root = Path(study_dir)
        row = next(item for item in study["rows"] if item["row_id"] == row_id)
        calls.append((output_stage, f"{row_id}--{role}"))
        rho_conv = float(candidate_payload["rho_conv"])
        rho_dense = float(candidate_payload["rho_dense"])
        if output_stage == "baseline_candidates":
            conv_index = [float(value) for value in study["rho_grid"]["rho_conv"]].index(
                rho_conv
            )
            dense_index = [
                float(value) for value in study["rho_grid"]["rho_dense"]
            ].index(rho_dense)
            distance = abs(conv_index - 2) + abs(dense_index - 2)
            final_loss = 0.1 + 0.01 * distance
            final_accuracy = 0.95 if distance == 0 else 0.91
        else:
            final_loss = 0.105 if row["scheme"] == "ours" else 0.11
            final_accuracy = 0.93
        relative_updates = {
            "ConvWeight_0": rho_conv,
            "ConvWeight_1": rho_conv,
            "DenseWeight_0": rho_dense,
        }
        occupancy = {name: 0.01 for name in relative_updates}
        efficiency = {name: 0.99 for name in relative_updates}
        output = root / "stages" / output_stage / "entries" / f"{row_id}--{role}"
        _write_stub_outputs(
            output,
            (
                "best_validation.pt",
                "final.pt",
                "minibatches.json",
                "parameter_diagnostics.csv",
                "run_spec.v5.json",
                "step_log.csv",
                "validation.json",
            ),
        )
        summary = {
            "schema_version": "mnist-conv-lr-candidate-result/v5",
            "status": "complete",
            "training_completed": True,
            "row": row,
            "candidate_stage": candidate_payload["candidate_stage"],
            "rho_conv": rho_conv,
            "rho_dense": rho_dense,
            "admissible": True,
            "inadmissible_reason": None,
            "final_validation_loss": final_loss,
            "final_validation_accuracy": final_accuracy,
            "median_projection_efficiency": 0.99,
            "learning_rates_by_weight": candidate_payload[
                "learning_rates_by_weight"
            ],
            "observed_rho_relative_q90_complete_run_by_parameter": relative_updates,
            "maximum_bound_occupancy_by_parameter": occupancy,
            "median_projection_efficiency_by_parameter": efficiency,
            "official_test_read": False,
        }
        atomic_write_json(output / "summary.json", summary, canonical=True)
        return summary

    monkeypatch.setattr(lr_study_module, "execute_v6_reuse_audit", audit)
    monkeypatch.setattr(lr_study_module, "execute_probe_entry", probe)
    monkeypatch.setattr(lr_study_module, "execute_candidate_entry", candidate)


class _InProcessWorkers:
    """Popen-compatible adapter that executes each immutable entry in-process."""

    def __init__(self, *, provenance: dict[str, Any]) -> None:
        self.provenance = provenance
        self.active = 0
        self.max_active_by_stage: dict[str, int] = {}
        self.results: list[dict[str, Any]] = []

    def factory(self, command: list[str], **unused: object) -> "_InProcessWorker":
        stage = command[command.index("--stage") + 1]
        self.active += 1
        self.max_active_by_stage[stage] = max(
            self.max_active_by_stage.get(stage, 0), self.active
        )
        return _InProcessWorker(self, command)


class _InProcessWorker:
    def __init__(self, owner: _InProcessWorkers, command: list[str]) -> None:
        self.owner = owner
        self.command = command
        self.returncode: int | None = None

    def _argument(self, name: str) -> str:
        return self.command[self.command.index(name) + 1]

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            result = execute_manifest_entry(
                study_dir=self._argument("--study"),
                manifest_path=self._argument("--manifest"),
                entry_index=int(self._argument("--entry-index")),
                data_root=self._argument("--data-root"),
                download=False,
                device=self._argument("--device"),
                current_provenance=self.owner.provenance,
            )
            self.owner.results.append(result)
            self.owner.active -= 1
            self.returncode = 0
        return self.returncode

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        if self.returncode is None:
            self.owner.active -= 1
            self.returncode = -15

    def kill(self) -> None:
        if self.returncode is None:
            self.owner.active -= 1
            self.returncode = -9


def _run_packed_stage(
    *,
    study: Path,
    manifest_path: Path,
    benchmark_path: Path | None,
    data_root: Path,
    workers: _InProcessWorkers,
) -> dict[str, Any]:
    stage = read_json(manifest_path)["stage_name"]
    pack_path = manifest_path.parent / "pack_manifest.v2.json"
    create_v6_pack_manifest(
        manifest_path,
        pack_path,
        benchmark_path=benchmark_path,
    )
    result = execute_packed_waves(
        study_dir=study,
        manifest_path=manifest_path,
        pack_manifest_path=pack_path,
        benchmark_path=benchmark_path,
        stage=stage,
        data_root=data_root,
        device="cuda",
        python="python",
        log_dir=study / "slurm" / stage,
        process_factory=workers.factory,
    )
    finalize_stage(study, manifest_path)
    validate_stage_completion(study_dir=study, manifest_path=manifest_path)
    return result


def _exercise_production_shadow_seam(
    monkeypatch: pytest.MonkeyPatch,
    *,
    spec: LRStudySpec,
    study: Path,
    payload: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    class Runtime:
        parameters: list[Any] = []

        def save(self, path: str | Path) -> Path:
            return atomic_write_bytes(path, b"shadow-checkpoint\n")

    class Bundle:
        train_loader = [object()] * (32 + 256)

        def reset_train_shuffle(self) -> None:
            return None

    class RunSpec:
        def to_dict(self) -> dict[str, Any]:
            return {"schema_version": "mnist-conv-run/v5", "shadow": True}

    initial = {
        name: {
            "bounded_gate": True,
            "combined_bound_occupancy": 0.0,
            "rms": 1.0,
        }
        for name in ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0")
    }

    def fake_training_step(*unused: Any, **kwargs: Any) -> SimpleNamespace:
        del kwargs
        return SimpleNamespace(
            loss=1.0,
            accuracy=0.5,
            sample_count=16,
            source_indices=(0,),
            transition=object(),
        )

    def fake_transition_rows(
        unused_transition: object,
        *,
        step: int,
        learning_rate: dict[str, float],
        rho_schedule: float | None,
        bounded_initial_rms_scales: dict[str, float],
        **unused: Any,
    ) -> list[dict[str, Any]]:
        del unused_transition, rho_schedule, bounded_initial_rms_scales, unused
        records = []
        for name in ("ConvWeight_0", "ConvWeight_1", "DenseWeight_0"):
            target = float(
                payload["rho_dense"]
                if name == "DenseWeight_0"
                else payload["rho_conv"]
            )
            records.append(
                {
                    "step": step,
                    "learning_rate": float(learning_rate[name]),
                    "parameter": name,
                    "parameter_kind": "bounded_weight",
                    "bounded_gate": True,
                    "report_only": False,
                    "gradient_rms": 1.0,
                    "proposed_update_rms": target,
                    "normalized_update": target,
                    "span_normalized_update": target,
                    "initial_parameter_rms": 1.0,
                    "parameter_relative_update": target,
                    "proposed_bound_crossing_fraction": 0.0,
                    "lower_bound_occupancy": 0.0,
                    "upper_bound_occupancy": 0.0,
                    "combined_bound_occupancy": 0.0,
                    "projection_efficiency": 1.0,
                    "proposal_is_numerically_zero": False,
                    "projection_gate_eligible": True,
                }
            )
        return records

    with monkeypatch.context() as patch:
        patch.setattr(lr_stages_module, "build_model_runtime", lambda *a, **k: Runtime())
        patch.setattr(lr_stages_module, "build_loader_bundle", lambda *a, **k: Bundle())
        patch.setattr(
            lr_stages_module,
            "parameter_state_diagnostics",
            lambda unused: initial,
        )
        patch.setattr(
            lr_stages_module,
            "candidate_run_spec_v5",
            lambda *a, **k: RunSpec(),
        )
        patch.setattr(lr_stages_module, "training_step", fake_training_step)
        patch.setattr(lr_stages_module, "transition_rows", fake_transition_rows)
        start = study / "shadow/start"
        start.parent.mkdir(parents=True, exist_ok=True)
        start.touch()
        return execute_v6_shadow_benchmark_entry(
            spec.data,
            study,
            payload["row_id"],
            payload["candidate_role"],
            candidate_payload=payload,
            data_root=data_root,
            download=False,
            device="cpu",
            ready_path=study / "shadow/ready.json",
            start_path=start,
            scratch_output_dir=study / "shadow/production-artifacts",
        )


def test_reduced_v6_real_publisher_to_finalization_packed_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MNIST_CONV_V6_R3_AUTHORIZATION", R3_AUTHORIZATION_TOKEN)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    spec = LRStudySpec.from_path(V6_CONFIG)
    study, _contract = create_study(spec, tmp_path / "results" / spec.study_id)
    data_root = tmp_path / "mnist"
    data_root.mkdir()
    provenance = code_provenance()
    calls: list[tuple[str, str]] = []
    _install_reduced_science(monkeypatch, calls)
    workers = _InProcessWorkers(provenance=provenance)

    audit_manifest, audit_path = publish_lr_stage_manifest(
        study, spec, "audit", provenance
    )
    assert len(audit_manifest["entries"]) == 1
    assert _run_packed_stage(
        study=study,
        manifest_path=audit_path,
        benchmark_path=None,
        data_root=data_root,
        workers=workers,
    )["entry_count"] == 1

    probe_manifest, probe_path = publish_lr_stage_manifest(
        study, spec, "probe", provenance
    )
    assert len(probe_manifest["entries"]) == 3
    assert _run_packed_stage(
        study=study,
        manifest_path=probe_path,
        benchmark_path=None,
        data_root=data_root,
        workers=workers,
    )["entry_count"] == 3
    assert workers.max_active_by_stage["probe"] == 3

    baseline_manifest, baseline_path = publish_lr_stage_manifest(
        study, spec, "baseline_candidates", provenance
    )
    safe_payload = baseline_manifest["entries"][8]["payload"]
    shadow = _exercise_production_shadow_seam(
        monkeypatch,
        spec=spec,
        study=study,
        payload=safe_payload,
        data_root=data_root,
    )
    assert shadow["status"] == "complete"
    assert shadow["production_loop"] is True
    assert shadow["attempted_steps"] == 288
    assert shadow["completed_steps"] == 288
    assert shadow["canonical_completion_published"] is False

    baseline_digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    benchmark_path = baseline_path.parent / "concurrency_benchmark.v1.json"
    benchmark = build_benchmark_report(
        baseline_manifest,
        stage_manifest_sha256=baseline_digest,
        levels=_benchmark_levels(),
        device={"name": "Tesla V100-SXM2-32GB", "total_memory_mib": 32768.0},
    )
    atomic_write_json(benchmark_path, benchmark, canonical=True)
    baseline_result = _run_packed_stage(
        study=study,
        manifest_path=baseline_path,
        benchmark_path=benchmark_path,
        data_root=data_root,
        workers=workers,
    )
    assert baseline_result["wave_count"] == 4
    assert workers.max_active_by_stage["baseline_candidates"] == 4
    assert len([call for call in calls if call[0] == "baseline_candidates"]) == 16

    # Re-run the same immutable pack. The real entry-completion check must
    # resume all entries without re-invoking the reduced scientific seam.
    before_resume = len(calls)
    resumed = execute_packed_waves(
        study_dir=study,
        manifest_path=baseline_path,
        pack_manifest_path=baseline_path.parent / "pack_manifest.v2.json",
        benchmark_path=benchmark_path,
        stage="baseline_candidates",
        data_root=data_root,
        device="cuda",
        python="python",
        log_dir=study / "slurm/baseline-resume",
        process_factory=workers.factory,
    )
    assert resumed["entry_count"] == 16
    assert len(calls) == before_resume
    assert [item["status"] for item in workers.results[-16:]] == [
        "resumed_complete"
    ] * 16

    selection_manifest, selection_path = publish_lr_stage_manifest(
        study, spec, "select_baseline", provenance
    )
    selection_execution = execute_manifest_entry(
        study_dir=study,
        manifest_path=selection_path,
        entry_index=0,
        data_root=data_root,
        download=False,
        device="cpu",
        current_provenance=provenance,
    )
    assert selection_execution["result_status"] == "selected"
    finalize_stage(study, selection_path)
    validate_stage_completion(study_dir=study, manifest_path=selection_path)
    selection = read_json(
        study / "stages/select_baseline/entries/selection/selection.json"
    )
    assert selection["selected_rho_conv"] == pytest.approx(0.003)
    assert selection["selected_rho_dense"] == pytest.approx(0.03)
    assert sum(
        bool(candidate["selected_for_confirmation"])
        for candidate in selection["candidates"]
    ) == 1
    assert len(selection_manifest["entries"]) == 1

    confirmation_manifest, confirmation_path = publish_lr_stage_manifest(
        study, spec, "confirmations", provenance
    )
    assert {
        (entry["payload"]["rho_conv"], entry["payload"]["rho_dense"])
        for entry in confirmation_manifest["entries"]
    } == {(0.003, 0.03)}
    confirmation_result = _run_packed_stage(
        study=study,
        manifest_path=confirmation_path,
        benchmark_path=benchmark_path,
        data_root=data_root,
        workers=workers,
    )
    assert confirmation_result["wave_count"] == 1
    assert workers.max_active_by_stage["confirmations"] == 2

    final_manifest, final_path = publish_lr_stage_manifest(
        study, spec, "finalize", provenance
    )
    final_execution = execute_manifest_entry(
        study_dir=study,
        manifest_path=final_path,
        entry_index=0,
        data_root=data_root,
        download=False,
        device="cpu",
        current_provenance=provenance,
    )
    assert final_execution["result_status"] == spec.data["selection"][
        "selected_status"
    ]
    finalize_stage(study, final_path)
    validate_stage_completion(study_dir=study, manifest_path=final_path)
    finalization = read_json(
        study / "stages/finalize/entries/finalization/finalization.json"
    )
    assert finalization["frozen"] is True
    assert finalization["rho_conv"] == pytest.approx(0.003)
    assert finalization["rho_dense"] == pytest.approx(0.03)
    assert finalization["canonical_training_run_count"] == 18
    assert finalization["official_test_read"] is False
    assert len(final_manifest["entries"]) == 1

    assert [call[0] for call in calls].count("audit") == 1
    assert [call[0] for call in calls].count("probe") == 3
    assert [call[0] for call in calls].count("baseline_candidates") == 16
    assert [call[0] for call in calls].count("confirmations") == 2
