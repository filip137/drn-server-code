"""Exercise Slurm's spool-script relocation without loading CUDA or training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.parametrize("task,mode", [(0, "production"), (5, "production"), (0, "smoke")])
def test_relocated_wrapper_uses_explicit_frozen_source(tmp_path, task, mode):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "frozen-source"
    configs = source / "configs/conv/target_scale_conv3_20260929"
    configs.mkdir(parents=True)
    for index in range(6):
        (configs / f"{index:02d}_case.json").write_text("{}")
    spool = tmp_path / "slurm-spool"
    spool.mkdir()
    script = spool / "slurm_script"
    shutil.copy2(root / "experiments/run_conv3_target_scale_jeanzay.sh", script)
    commands = tmp_path / "bin"
    commands.mkdir()
    capture = tmp_path / "capture.json"
    (commands / "module").write_text("#!/bin/sh\nexit 0\n")
    (commands / "python").write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['WRAPPER_CAPTURE']).write_text(json.dumps("
        "dict(cwd=os.getcwd(),argv=sys.argv[1:],task=os.environ['SLURM_ARRAY_TASK_ID'])))\n"
    )
    for name in ("module", "python"):
        (commands / name).chmod(0o755)
    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ["PATH"],
               WRAPPER_CAPTURE=str(capture), SLURM_ARRAY_TASK_ID=str(task),
               EXPERIMENT_SOURCE_COMMIT="test", EXPERIMENT_SOURCE_ARCHIVE_SHA256="test")
    subprocess.run(["bash", str(script), str(tmp_path / "output"), str(source), mode],
                   cwd=spool, env=env, check=True)
    observed = json.loads(capture.read_text())
    assert observed["cwd"] == str(source)
    assert observed["task"] == str(task)
    assert len([a for a in observed["argv"] if a.endswith("_case.json")]) == 6
    assert ("--smoke" in observed["argv"]) == (mode == "smoke")
    assert "--skip-terminal-official-test" in observed["argv"]
    assert "--index" not in observed["argv"]
    missing = subprocess.run(["bash", str(script), str(tmp_path / "output")],
                             cwd=spool, env=env, capture_output=True, text=True)
    assert missing.returncode != 0
    assert "absolute frozen source root" in missing.stderr
