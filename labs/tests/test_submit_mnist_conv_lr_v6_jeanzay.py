from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

import pytest

import experiments.submit_mnist_conv_lr_v6_benchmark_jeanzay as benchmark_submit
import experiments.submit_mnist_conv_lr_v6_stage_jeanzay as stage_submit
from experiments.mnist_conv.io import atomic_write_json
from experiments.mnist_conv.lr_v6_jeanzay import (
    publish_v6_stage_from_login,
    verify_login_r3_allocation,
)
from experiments.mnist_conv.lr_v6_packing import (
    CONCURRENCY_LEVELS,
    MEASURED_STEPS,
    build_benchmark_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "lrstudy_" + "a" * 64
CANONICAL_V6_ID = (
    "lrstudy_c735d2beead9fcbadda89a65ffe86257da53f15e6cbfab7b4f04b14b08ff6c2f"
)


def _stage(stage: str) -> dict:
    count = {"audit": 1, "probe": 3, "baseline_candidates": 16, "confirmations": 2}[stage]
    entries = []
    for index in range(count):
        if stage == "audit":
            payload = {"mode": "v5_reuse"}
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
                "learning_rates_by_parameter": {"ConvWeight_0": 1.0},
            }
        entries.append(
            {"entry_index": index, "entry_id": f"{stage}-{index}", "payload": payload}
        )
    return {
        "schema_version": "mnist-conv-lr-stage-manifest/v1",
        "study_id": STUDY_ID,
        "stage_name": stage,
        "entries": entries,
    }


def _levels() -> list[dict]:
    throughput = {1: 10.0, 2: 19.0, 4: 35.0, 8: 60.0, 12: 68.0, 16: 66.0}
    return [
        {
            "concurrency": concurrency,
            "completed_children": concurrency,
            "failed_children": 0,
            "combined_peak_gpu_memory_mib": 1000.0 * concurrency,
            "peak_host_rss_mib": 500.0 * concurrency,
            "mean_gpu_utilization_percent": 80.0,
            "aggregate_successful_steps_per_second": throughput[concurrency],
            "successful_measured_steps": concurrency * MEASURED_STEPS,
            "wall_seconds": 30.0,
            "child_failures": [],
        }
        for concurrency in CONCURRENCY_LEVELS
    ]


def _fixture(tmp_path: Path, stage: str) -> tuple[Path, Path, dict]:
    study = tmp_path / "results" / STUDY_ID / "lr_studies" / ("v6--" + STUDY_ID)
    manifest_path = study / "stages" / stage / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = _stage(stage)
    atomic_write_json(manifest_path, manifest, canonical=True)
    return study, manifest_path, manifest


def _stage_args(tmp_path: Path, stage: str) -> argparse.Namespace:
    data_root = tmp_path / "mnist"
    data_root.mkdir(exist_ok=True)
    return argparse.Namespace(
        stage=stage,
        config=None,
        results_root=None,
        study=str(tmp_path / "placeholder"),
        v5_study=(
            str(stage_submit.frozen_v5_parent_study())
            if stage == "audit"
            else None
        ),
        benchmark=None,
        pack_manifest=None,
        repo_root=str(REPO_ROOT),
        python="python",
        data_root=str(data_root),
        module="pytorch-gpu/py3/2.5.0",
        submit=False,
    )


def test_audit_submission_is_one_gpu_one_job_and_exports_exact_v5_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, manifest_path, manifest = _fixture(tmp_path, "audit")
    monkeypatch.setattr(
        stage_submit,
        "publish_v6_stage_from_login",
        lambda **unused: (study, manifest_path, manifest),
    )
    args = _stage_args(tmp_path, "audit")
    command, metadata = stage_submit._prepare(args)

    joined = "\n".join(command)
    assert command[0] == "sbatch"
    assert "--account=fmu@v100" in command
    assert "--partition=gpu_p13" in command
    assert "--qos=qos_gpu-t3" in command
    assert "--constraint=v100-32g" in command
    assert "--gres=gpu:1" in command
    assert "--cpus-per-task=16" in command
    assert "--hint=nomultithread" in command
    assert "--time=20:00:00" in command
    assert not any(item.startswith("--array") for item in command)
    assert "AD010913993R3" in joined
    assert "umg@" not in joined
    assert str(stage_submit.frozen_v5_parent_study()) in joined
    assert metadata["wave_count"] == 1
    assert metadata["login_finalizer"]["execution_host"] == "jean_zay_login_node"
    assert "--finalize-stage" in metadata["login_finalizer"]["command"]


def test_login_publisher_requires_results_study_id_wrapper(tmp_path: Path) -> None:
    config = (
        REPO_ROOT
        / "configs/conv/hardsigmoid_lr_conv2_two_rho_constant_sgd_bs16_v6.json"
    )
    with pytest.raises(ValueError, match="study_id wrapper"):
        publish_v6_stage_from_login(
            stage="audit",
            config_path=config,
            results_root=tmp_path / "results",
        )

    study, manifest_path, manifest = publish_v6_stage_from_login(
        stage="audit",
        config_path=config,
        results_root=tmp_path / "results" / CANONICAL_V6_ID,
    )
    assert study.parent.parent.name == CANONICAL_V6_ID
    assert manifest_path.is_file()
    assert manifest["study_id"] == CANONICAL_V6_ID


def test_frozen_v5_parent_uses_current_remote_user_component() -> None:
    parent = stage_submit.frozen_v5_parent_study("newuser")
    assert str(parent).startswith(
        "/lustre/fsn1/projects/rech/fmu/newuser/server_code/results/"
    )
    assert parent.name.endswith(stage_submit.V5_STUDY_ID)


def test_baseline_submission_binds_benchmark_and_uses_one_sixteen_entry_wave(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, manifest_path, manifest = _fixture(tmp_path, "baseline_candidates")
    benchmark_path = manifest_path.parent / "benchmark.json"
    stage_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    atomic_write_json(
        benchmark_path,
        build_benchmark_report(
            manifest,
            stage_manifest_sha256=stage_digest,
            levels=_levels(),
            device={"name": "Tesla V100-SXM2-32GB", "total_memory_mib": 32768.0},
        ),
        canonical=True,
    )
    monkeypatch.setattr(
        stage_submit,
        "publish_v6_stage_from_login",
        lambda **unused: (study, manifest_path, manifest),
    )
    args = _stage_args(tmp_path, "baseline_candidates")
    args.benchmark = str(benchmark_path)
    command, metadata = stage_submit._prepare(args)

    assert metadata["wave_count"] == 1
    assert metadata["selected_concurrency"] == 16
    assert str(benchmark_path) in "\n".join(command)


def test_benchmark_submitter_publishes_manifest_but_returns_login_next_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, manifest_path, manifest = _fixture(tmp_path, "baseline_candidates")
    monkeypatch.setattr(
        benchmark_submit,
        "publish_v6_stage_from_login",
        lambda **unused: (study, manifest_path, manifest),
    )
    data_root = tmp_path / "mnist"
    data_root.mkdir(exist_ok=True)
    args = argparse.Namespace(
        config=None,
        study=str(study),
        results_root=None,
        repo_root=str(REPO_ROOT),
        python="python",
        data_root=str(data_root),
        module="pytorch-gpu/py3/2.5.0",
        output=None,
        submit=False,
    )
    command, metadata = benchmark_submit._prepare(args)

    assert command[0] == "sbatch"
    assert "--cpus-per-task=16" in command
    assert not any(item.startswith("--array") for item in command)
    assert metadata["next_login_step"]["execution_host"] == "jean_zay_login_node"
    assert "experiments.create_mnist_conv_lr_v6_pack_manifest" in metadata[
        "next_login_step"
    ]["command"]


def test_submit_runs_login_audit_then_exactly_one_sbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study, manifest_path, manifest = _fixture(tmp_path, "probe")
    monkeypatch.setattr(
        stage_submit,
        "publish_v6_stage_from_login",
        lambda **unused: (study, manifest_path, manifest),
    )
    monkeypatch.setattr(
        stage_submit,
        "verify_login_r3_allocation",
        lambda: {"verified": True, "allocation_id": "AD010913993R3"},
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs == {"check": True, "text": True, "capture_output": True}
        return subprocess.CompletedProcess(command, 0, stdout="2122001\n", stderr="")

    monkeypatch.setattr(stage_submit.subprocess, "run", fake_run)
    args = _stage_args(tmp_path, "probe")
    args.submit = True
    result = stage_submit.submit(args)

    assert len(calls) == 1
    assert result["worker_job_id"] == "2122001"
    assert result["login_audit"]["verified"] is True
    assert result["login_finalizer"]["execution_host"] == "jean_zay_login_node"


def test_live_login_audit_requires_r3_allocation_and_fmu_account() -> None:
    outputs = iter(
        [
            "fmu | AD010913993R3 | active\n",
            "export IDRPROJ=fmu\n",
            "fmu@v100||qos_gpu-dev,qos_gpu-t3,qos_gpu-t4\n",
            "PartitionName=gpu_p13 State=UP AllowAccounts=ALL\n",
            "gpu_p13|Tesla,v100,mps,v100-32g,prof,mongpu\n",
        ]
    )

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=next(outputs), stderr="")

    result = verify_login_r3_allocation(fake_runner)
    assert result["allocation_id"].endswith("R3")
    assert result["slurm_account"] == "fmu@v100"
    assert result["partition"] == "gpu_p13"
    assert result["constraint"] == "v100-32g"


def test_live_login_audit_rejects_r3_belonging_to_another_project() -> None:
    outputs = iter(
        [
            "other | AD010913993R3 | active\nfmu | AD999999999R3 | active\n",
            "export IDRPROJ=fmu\n",
            "fmu@v100||qos_gpu-t3\n",
            "PartitionName=gpu_p13 State=UP AllowAccounts=ALL\n",
            "gpu_p13|Tesla,v100,v100-32g\n",
        ]
    )

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=next(outputs), stderr="")

    with pytest.raises(RuntimeError, match="map project 'fmu'"):
        verify_login_r3_allocation(fake_runner)


def test_live_login_audit_requires_qos_on_the_fmu_v100_assoc_line() -> None:
    outputs = iter(
        [
            "fmu | AD010913993R3 | active\n",
            "export IDRPROJ=fmu\n",
            "fmu@v100||qos_gpu-dev\nother|gpu_p13|qos_gpu-t3\n",
            "PartitionName=gpu_p13 State=UP AllowAccounts=ALL\n",
            "gpu_p13|Tesla,v100,v100-32g\n",
        ]
    )

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=next(outputs), stderr="")

    with pytest.raises(RuntimeError, match="one live Slurm association line"):
        verify_login_r3_allocation(fake_runner)


@pytest.mark.parametrize(
    ("partition_output", "feature_output", "message"),
    [
        (
            "PartitionName=gpu_p13 State=DOWN AllowAccounts=ALL\n",
            "gpu_p13|Tesla,v100,v100-32g\n",
            "State=UP",
        ),
        (
            "PartitionName=gpu_p13 State=UP AllowAccounts=ALL\n",
            "gpu_p13|Tesla,v100,v100-16g\n",
            "v100-32g",
        ),
    ],
)
def test_live_login_audit_requires_up_v100_32g_partition(
    partition_output: str,
    feature_output: str,
    message: str,
) -> None:
    outputs = iter(
        [
            "fmu | AD010913993R3 | active\n",
            "export IDRPROJ=fmu\n",
            "fmu@v100||qos_gpu-dev,qos_gpu-t3\n",
            partition_output,
            feature_output,
        ]
    )

    def fake_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=next(outputs), stderr="")

    with pytest.raises(RuntimeError, match=message):
        verify_login_r3_allocation(fake_runner)


def test_wrappers_encode_barrier_benchmark_and_sequential_wave_contract() -> None:
    benchmark = (REPO_ROOT / "experiments/run_mnist_conv_lr_v6_benchmark_jeanzay.slurm").read_text()
    packed = (REPO_ROOT / "experiments/run_mnist_conv_lr_v6_pack_jeanzay.slurm").read_text()

    for script in (benchmark, packed):
        assert "#SBATCH --cpus-per-task=16" in script
        assert "#SBATCH --hint=nomultithread" in script
        assert "#SBATCH --time=20:00:00" in script
        assert "#SBATCH --account=fmu@v100" in script
        assert "#SBATCH --constraint=v100-32g" in script
        assert "AD010913993R3" in script
        assert "<fmu-results>/<study-id>/lr_studies" in script
        assert "OMP_NUM_THREADS=1" in script
        assert 'runtime_account="${SLURM_JOB_ACCOUNT}"' in script
        assert 'runtime_partition="${SLURM_JOB_PARTITION}"' in script
        assert 'runtime_qos="${SLURM_JOB_QOS}"' in script
        assert 'runtime_gpu_count="${SLURM_GPUS_ON_NODE}"' in script
        assert 'scontrol show job "${SLURM_JOB_ID}" -o' in script
        assert 'Features=v100-32g' in script
    assert "concurrency=1,2,4,8,12,16 warmup=32 measured=256" in benchmark
    assert 'benchmark_scratch_base="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"' in benchmark
    assert 'require_env SLURM_TMPDIR' not in benchmark
    assert 'failure-evidence-${SLURM_JOB_ID}.tar.gz' in benchmark
    assert 'exit "${benchmark_status}"' in benchmark
    assert "experiments.run_mnist_conv_lr_v6_pack" in packed
    assert "--pack-manifest" in packed
