import os
import sys
from pathlib import Path
import subprocess

import pytest

WRAPPER = Path(__file__).resolve().parents[2] / "experiments/run_conv3_p90_read_noise_continuation_a100.sh"


@pytest.fixture(autouse=True, params=["a100", "v100"])
def wrapper(request, monkeypatch):
    path = Path(__file__).resolve().parents[2] / f"experiments/run_conv3_p90_read_noise_continuation_{request.param}.sh"
    monkeypatch.setattr(sys.modules[__name__], "WRAPPER", path)


def invoke(tmp_path, chunk, task="0"):
    config = tmp_path / "configs"
    config.mkdir(exist_ok=True)
    (config / "config-names.txt").write_text("\n".join(f"{i}.json" for i in range(18)) + "\n")
    env = os.environ.copy()
    env.pop("SLURM_ARRAY_TASK_ID", None)
    if task is not None:
        env["SLURM_ARRAY_TASK_ID"] = task
    return subprocess.run(["bash", str(WRAPPER), str(tmp_path / "source"), str(config),
                           str(tmp_path / "out"), str(tmp_path / "data"), str(chunk)],
                          env=env, capture_output=True, text=True)


@pytest.mark.parametrize("chunk,task", [(3,"0"), (-1,"0"), (0,None), (0,"18")])
def test_rejects_invalid_chunk_or_case_before_allocation_work(tmp_path, chunk, task):
    assert invoke(tmp_path, chunk, task).returncode != 0
    assert not (tmp_path / "out").exists()


def test_duplicate_chunk_preserves_original_receipt(tmp_path):
    receipt = tmp_path / "out/task_0/segments/chunk_1/worker_started"
    receipt.mkdir(parents=True)
    (receipt / "pid").write_text("original")
    assert invoke(tmp_path, 1).returncode != 0
    assert (receipt / "pid").read_text() == "original"
    assert not (receipt / "finished_at").exists()
