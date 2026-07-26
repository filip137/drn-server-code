from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import experiments.submit_mnist_conv_lr_v6_stage_jeanzay_umg_override as override


REPO_ROOT = Path(__file__).resolve().parents[2]


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        stage="baseline_candidates",
        study=str(tmp_path / "study"),
        benchmark=str(tmp_path / "study/benchmark.json"),
        pack_manifest=str(tmp_path / "study/pack.json"),
        repo_root=str(REPO_ROOT),
        python="python",
        data_root=str(tmp_path / "mnist"),
        module="pytorch-gpu/py3/2.5.0",
        submit=False,
        config=None,
        results_root=None,
        v5_study=None,
    )


def test_umg_prepare_changes_only_runtime_dispatch_contract(
    tmp_path: Path, monkeypatch
) -> None:
    wrapper = REPO_ROOT / "experiments/run_mnist_conv_lr_v6_pack_jeanzay.slurm"
    command = [
        "sbatch",
        "--account=fmu@v100",
        "--partition=gpu_p13",
        "--qos=qos_gpu-t3",
        "--constraint=v100-32g",
        "--export=ALL,MNIST_CONV_R3_ALLOCATION=AD010913993R3",
        str(wrapper),
    ]
    metadata = {
        "study_id": "lrstudy_" + "a" * 64,
        "study": str(tmp_path / "study"),
        "stage": "baseline_candidates",
        "log_dir": str(tmp_path / "logs"),
    }
    monkeypatch.setattr(
        override.frozen_submit,
        "_prepare",
        lambda unused: (list(command), dict(metadata)),
    )

    prepared, result = override._prepare(_args(tmp_path))
    joined = "\n".join(prepared)
    assert "--account=umg@v100" in prepared
    assert "--account=fmu@v100" not in prepared
    assert "MNIST_CONV_R3_ALLOCATION=AD010913993R3" in joined
    assert "MNIST_CONV_EXECUTION_ALLOCATION_OVERRIDE=AD011016471R1" in joined
    assert "MNIST_CONV_EXECUTION_PROJECT_OVERRIDE=umg" in joined
    assert "MNIST_CONV_EXECUTION_ACCOUNT_OVERRIDE=umg@v100" in joined
    assert override.OVERRIDE_AUTHORIZATION in joined
    assert result["execution_override"]["frozen_study_execution"]["account"] == (
        "fmu@v100"
    )
    assert result["execution_override"]["actual_execution"]["account"] == (
        "umg@v100"
    )


def test_umg_login_audit_requires_exact_project_account_and_v100_32g() -> None:
    outputs = iter(
        [
            "umg (104552/AD011016471R1) [default]\n"
            "fmu (102871/AD010913993R3)\n",
            "export IDRPROJ=umg\n",
            "umg@v100||qos_gpu-dev,qos_gpu-t3,qos_gpu-t4\n",
            "PartitionName=gpu_p13 State=UP AllowAccounts=ALL\n",
            "gpu_p13|Tesla,v100,mps,v100-32g\n",
        ]
    )

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs == {"check": True, "text": True, "capture_output": True}
        return subprocess.CompletedProcess(command, 0, stdout=next(outputs), stderr="")

    result = override.verify_login_umg_v100_allocation(runner)
    assert result["allocation_id"] == "AD011016471R1"
    assert result["slurm_account"] == "umg@v100"
    assert result["constraint"] == "v100-32g"
    assert result["user_authorized_override"] is True


def test_umg_login_audit_rejects_wrong_allocation() -> None:
    outputs = iter(
        [
            "umg (104552/AD999999999R1) [default]\n",
            "export IDRPROJ=umg\n",
            "umg@v100||qos_gpu-t3\n",
            "PartitionName=gpu_p13 State=UP AllowAccounts=ALL\n",
            "gpu_p13|Tesla,v100,v100-32g\n",
        ]
    )

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout=next(outputs), stderr="")

    try:
        override.verify_login_umg_v100_allocation(runner)
    except RuntimeError as exc:
        assert "AD011016471R1" in str(exc)
    else:
        raise AssertionError("wrong UMG allocation was accepted")


def test_pack_wrapper_fail_closes_and_records_umg_override() -> None:
    source = (
        REPO_ROOT / "experiments/run_mnist_conv_lr_v6_pack_jeanzay.slurm"
    ).read_text()
    assert "user-approved-umg-v100-20260721" in source
    assert "AD011016471R1:umg@v100:gpu_p13:qos_gpu-t3" in source
    assert 'execution_account_expected="fmu@v100"' in source
    assert 'execution_account_expected="${MNIST_CONV_EXECUTION_ACCOUNT_OVERRIDE}"' in source
    assert 'idrenv -d "${execution_project}"' in source
    assert "mnist-conv-lr-execution-override/v1" in source
    assert '"${log_dir}/execution_override.v1.json"' in source
