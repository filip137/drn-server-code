from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import experiments.mnist_conv.lr_v6_benchmark as benchmark_runtime
from experiments.mnist_conv.io import atomic_write_json
from experiments.mnist_conv.lr_v6_packing import (
    CANONICAL_STEPS_PER_ENTRY,
    CONCURRENCY_LEVELS,
    MEASURED_STEPS,
    build_benchmark_report,
    build_v6_pack_manifest,
    find_safe_baseline_entry,
    select_concurrency,
    validate_r3_allocation_contract,
    validate_v6_pack_manifest_files,
)


STUDY_ID = "lrstudy_" + "a" * 64


def _stage(stage: str) -> dict:
    counts = {"audit": 1, "probe": 3, "baseline_candidates": 16, "confirmations": 2}
    entries = []
    for index in range(counts[stage]):
        if stage == "audit":
            payload = {"mode": "verified_v5_reuse_or_rebuild"}
        elif stage == "probe":
            payload = {
                "architecture": "conv2",
                "row": {"architecture": "conv2", "row_id": f"row_{index}"},
            }
        else:
            payload = {
                "architecture": "conv2",
                "row_id": "conv2_baseline_v1_c1",
                "scheme": "baseline" if stage == "baseline_candidates" else ("ours", "legacy")[index],
                "rho_conv": (0.0005, 0.001, 0.003, 0.01)[index // 4]
                if stage == "baseline_candidates"
                else 0.003,
                "rho_dense": (0.003, 0.01, 0.03, 0.1)[index % 4]
                if stage == "baseline_candidates"
                else 0.01,
                "learning_rates_by_parameter": {
                    "ConvWeight_0": 0.1,
                    "Bias_0": 0.1,
                    "ConvWeight_1": 0.2,
                    "Bias_1": 0.2,
                    "DenseWeight_0": 0.3,
                },
            }
        entries.append(
            {
                "entry_index": index,
                "entry_id": f"{stage}-{index}",
                "payload": payload,
            }
        )
    return {
        "schema_version": "mnist-conv-lr-stage-manifest/v1",
        "study_id": STUDY_ID,
        "stage_name": stage,
        "entries": entries,
    }


def _levels(*, maximum_safe_concurrency: int = 16) -> list[dict]:
    throughput = {1: 10.0, 2: 19.0, 4: 36.0, 8: 62.0, 12: 70.0, 16: 68.0}
    result = []
    for concurrency in CONCURRENCY_LEVELS:
        failed = 0 if concurrency <= maximum_safe_concurrency else concurrency
        completed = concurrency - failed
        result.append(
            {
                "concurrency": concurrency,
                "completed_children": completed,
                "failed_children": failed,
                "combined_peak_gpu_memory_mib": 1000.0 * concurrency,
                "peak_host_rss_mib": 500.0 * concurrency,
                "mean_gpu_utilization_percent": 75.0,
                "aggregate_successful_steps_per_second": throughput[concurrency]
                if not failed
                else 0.0,
                "successful_measured_steps": completed * MEASURED_STEPS,
                "wall_seconds": 30.0,
                "child_failures": [
                    {"child_index": index, "returncode": 1}
                    for index in range(failed)
                ],
            }
        )
    return result


def _benchmark(
    maximum_safe_concurrency: int = 16,
    *,
    stage_manifest_sha256: str = "b" * 64,
) -> dict:
    return build_benchmark_report(
        _stage("baseline_candidates"),
        stage_manifest_sha256=stage_manifest_sha256,
        levels=_levels(maximum_safe_concurrency=maximum_safe_concurrency),
        device={"name": "Tesla V100-SXM2-32GB", "total_memory_mib": 32768.0},
    )


def test_safe_shadow_entry_is_exact_two_rho_baseline_point() -> None:
    stage = _stage("baseline_candidates")
    assert find_safe_baseline_entry(stage) == 8

    stage["entries"][8]["payload"]["rho_dense"] = 0.01
    with pytest.raises(ValueError, match="exactly one baseline benchmark entry"):
        find_safe_baseline_entry(stage)


def test_concurrency_selection_uses_largest_level_with_all_three_gates() -> None:
    selected = select_concurrency(_levels(), candidate_count=16)
    assert selected["selected_concurrency"] == 16
    assert selected["projected_full_sweep_seconds"] < 18 * 3600

    selected = select_concurrency(
        _levels(maximum_safe_concurrency=4), candidate_count=16
    )
    assert selected["selected_concurrency"] == 4


def test_projection_sums_measured_durations_for_twelve_plus_four_waves() -> None:
    levels = _levels(maximum_safe_concurrency=12)
    selected = select_concurrency(levels, candidate_count=16)

    assert selected["selected_concurrency"] == 12
    chosen = next(
        item
        for item in selected["evaluated_levels"]
        if item["concurrency"] == 12
    )
    assert chosen["deterministic_wave_widths"] == [12, 4]
    expected = (
        12 * CANONICAL_STEPS_PER_ENTRY / 70.0
        + 4 * CANONICAL_STEPS_PER_ENTRY / 36.0
    )
    assert chosen["projected_full_sweep_seconds"] == pytest.approx(expected)
    assert selected["projected_full_sweep_seconds"] == pytest.approx(expected)


def test_pack_v2_creates_sequential_waves_and_runs_confirmations_together() -> None:
    benchmark = _benchmark(
        maximum_safe_concurrency=4, stage_manifest_sha256="c" * 64
    )
    baseline = build_v6_pack_manifest(
        _stage("baseline_candidates"),
        stage_manifest_sha256="c" * 64,
        benchmark=benchmark,
        benchmark_sha256="d" * 64,
    )
    assert baseline["schema_version"] == "mnist-conv-lr-pack-manifest/v2"
    assert baseline["execution_contract"]["selected_concurrency"] == 4
    assert [wave["entry_indices"] for wave in baseline["waves"]] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]

    confirmations = build_v6_pack_manifest(
        _stage("confirmations"),
        stage_manifest_sha256="e" * 64,
        benchmark=benchmark,
        benchmark_sha256="d" * 64,
    )
    assert confirmations["waves"] == [
        {"wave_index": 0, "entry_indices": [0, 1], "concurrency": 2}
    ]


def test_baseline_pack_rejects_benchmark_from_a_different_manifest() -> None:
    with pytest.raises(ValueError, match="bind the current stage manifest"):
        build_v6_pack_manifest(
            _stage("baseline_candidates"),
            stage_manifest_sha256="c" * 64,
            benchmark=_benchmark(stage_manifest_sha256="b" * 64),
            benchmark_sha256="d" * 64,
        )


def test_stage_manifest_requires_full_content_addressed_study_id() -> None:
    stage = _stage("audit")
    stage["study_id"] = "lrstudy_short"
    with pytest.raises(ValueError, match="64 lowercase hex"):
        build_v6_pack_manifest(stage, stage_manifest_sha256="c" * 64)


def test_audit_and_probe_have_fixed_compute_node_waves_without_benchmark() -> None:
    audit = build_v6_pack_manifest(
        _stage("audit"), stage_manifest_sha256="c" * 64
    )
    probe = build_v6_pack_manifest(
        _stage("probe"), stage_manifest_sha256="c" * 64
    )
    assert audit["waves"] == [
        {"wave_index": 0, "entry_indices": [0], "concurrency": 1}
    ]
    assert probe["waves"] == [
        {"wave_index": 0, "entry_indices": [0, 1, 2], "concurrency": 3}
    ]
    assert audit["benchmark_binding"] is None
    assert probe["benchmark_binding"] is None


def test_pack_file_validation_binds_stage_and_benchmark_bytes(tmp_path: Path) -> None:
    stage = _stage("baseline_candidates")
    stage_path = tmp_path / "manifest.json"
    benchmark_path = tmp_path / "benchmark.json"
    pack_path = tmp_path / "pack.json"
    atomic_write_json(stage_path, stage, canonical=True)
    stage_digest = hashlib.sha256(stage_path.read_bytes()).hexdigest()
    benchmark = build_benchmark_report(
        stage,
        stage_manifest_sha256=stage_digest,
        levels=_levels(),
        device={"name": "Tesla V100-SXM2-32GB", "total_memory_mib": 32768.0},
    )
    atomic_write_json(benchmark_path, benchmark, canonical=True)
    benchmark_digest = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    pack = build_v6_pack_manifest(
        stage,
        stage_manifest_sha256=stage_digest,
        benchmark=benchmark,
        benchmark_sha256=benchmark_digest,
    )
    atomic_write_json(pack_path, pack, canonical=True)

    assert validate_v6_pack_manifest_files(
        stage_path, pack_path, benchmark_path=benchmark_path
    ) == pack
    pack["waves"][0]["entry_indices"].reverse()
    atomic_write_json(pack_path, pack, canonical=True)
    with pytest.raises(RuntimeError, match="immutable stage and benchmark"):
        validate_v6_pack_manifest_files(
            stage_path, pack_path, benchmark_path=benchmark_path
        )


def test_r3_contract_rejects_historical_or_non_r3_allocation() -> None:
    good = _benchmark()["r3_allocation"]
    assert validate_r3_allocation_contract(good)["allocation_id"].endswith("R3")
    bad = dict(good)
    bad["allocation_id"] = "AD010913993"
    bad["slurm_account"] = "umg@v100"
    with pytest.raises(ValueError, match="frozen R3/fmu V100"):
        validate_r3_allocation_contract(bad)


def test_shadow_child_delegates_to_exact_production_candidate_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "results" / STUDY_ID / "lr_studies" / ("v6--" + STUDY_ID)
    manifest_path = root / "stages/baseline_candidates/manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = _stage("baseline_candidates")
    manifest["code_provenance"] = {}
    safe = manifest["entries"][8]["payload"]
    safe.update(
        {
            "candidate_role": "rho-conv-0p003--rho-dense-0p003",
            "candidate_stage": "baseline_grid",
            "median_units_by_weight": {
                "ConvWeight_0": 1.0,
                "ConvWeight_1": 1.0,
                "DenseWeight_0": 1.0,
            },
            "peak_learning_rate": 0.3,
        }
    )
    atomic_write_json(manifest_path, manifest, canonical=True)
    spec = SimpleNamespace(
        study_id=STUDY_ID,
        data={
            "schema_version": "mnist-conv-lr-study/v6",
            "dataset": {
                "official_test": {"enabled": False, "read_allowed": False}
            },
        },
    )
    monkeypatch.setattr(
        benchmark_runtime, "load_study", lambda unused: (root, spec)
    )
    monkeypatch.setattr(
        benchmark_runtime,
        "load_stage_manifest",
        lambda *unused_args, **unused_kwargs: manifest,
    )
    monkeypatch.setattr(
        benchmark_runtime, "_require_current_source", lambda *unused: None
    )
    monkeypatch.setattr(benchmark_runtime, "code_provenance", lambda: {})
    observed: dict = {}

    def fake_production_loop(*args: object, **kwargs: object) -> dict:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {
            "status": "complete",
            "warmup_steps": 32,
            "measured_steps": 256,
            "attempted_steps": 288,
            "completed_steps": 288,
            "measured_started_unix_s": 10.0,
            "measured_finished_unix_s": 20.0,
            "measured_seconds": 10.0,
            "official_test_read": False,
            "production_loop": True,
            "transition_row_count": 1440,
            "canonical_completion_published": False,
        }

    monkeypatch.setattr(
        benchmark_runtime,
        "execute_v6_shadow_benchmark_entry",
        fake_production_loop,
    )
    result = benchmark_runtime.run_shadow_child(
        study_dir=root,
        candidate_manifest_path=manifest_path,
        entry_index=8,
        data_root=tmp_path / "mnist",
        ready_path=tmp_path / "scratch/ready.json",
        start_path=tmp_path / "scratch/start",
        output_path=tmp_path / "scratch/result.json",
        child_index=3,
    )

    assert result["production_loop"] is True
    assert result["child_index"] == 3
    assert observed["args"][:4] == (
        spec.data,
        root,
        safe["row_id"],
        safe["candidate_role"],
    )
    assert observed["kwargs"]["warmup_steps"] == 32
    assert observed["kwargs"]["measured_steps"] == 256
    assert observed["kwargs"]["candidate_payload"] is safe


def test_concurrency_orchestrator_launches_internal_production_cli_route() -> None:
    source = inspect.getsource(benchmark_runtime._run_level)
    assert '"experiments.mnist_conv"' in source
    assert '"lr-study"' in source
    assert '"--shadow-benchmark"' in source
    assert '"--shadow-ready"' in source
    assert '"--shadow-start"' in source
    assert '"--shadow-scratch-output"' in source
    assert '"--child"' not in source
