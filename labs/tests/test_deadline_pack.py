"""The overnight transport must reject late work and stop its whole job group."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

import pytest


def transport():
    path = Path(__file__).resolve().parents[2] / "experiments/run_deadline_pack.py"
    spec = importlib.util.spec_from_file_location("deadline_transport_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parent_modules(monkeypatch, config):
    exact = types.ModuleType("experiments.exact_run")
    exact.load_exact_config = lambda path: config
    reporting = types.ModuleType("experiments.reporting")
    reporting.validate_run = lambda path: []
    monkeypatch.setitem(sys.modules, "experiments.exact_run", exact)
    monkeypatch.setitem(sys.modules, "experiments.reporting", reporting)


def test_past_deadline_does_not_launch(monkeypatch, tmp_path):
    module = transport()
    monkeypatch.chdir(tmp_path)
    parent_modules(monkeypatch, {})
    monkeypatch.setattr(sys, "argv", ["pack", "--source", str(tmp_path), "--config", "config.json",
        "--output", str(tmp_path / "output"), "--target", "test", "--mode", "production",
        "--deadline", str(time.time() - 1), "--expected-seconds", "100"])
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("Late launch"))
    with pytest.raises(ValueError, match="Insufficient time"):
        module.main()
    assert not (tmp_path / "output").exists()


def test_deadline_stops_child_and_grandchild(monkeypatch, tmp_path):
    import hashlib

    module = transport()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments/__init__.py").write_text("")
    # This fake scientific runner uses no GPU, but creates a real process tree.
    (tmp_path / "experiments/exact_run.py").write_text(
        "import os,sys,time,subprocess\nfrom pathlib import Path\n"
        "out=Path(sys.argv[sys.argv.index('--output-root')+1]); out.mkdir(parents=True)\n"
        "(out/'dataset_root').write_text(sys.argv[sys.argv.index('--dataset-root')+1])\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
        "(out/'pids').write_text(str(os.getpid())+' '+str(child.pid))\n"
        "time.sleep(60)\n")
    asset = tmp_path / "init.pt"
    asset.write_bytes(b"test initializer")
    config = {"study_id": "test", "evaluation": {"official_test": {"policy": "disabled"}},
              "init_checkpoint_path": str(asset),
              "initialization": {"checkpoint_sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    for name in ("SOURCE_COMMIT", "SOURCE_ARCHIVE_SHA256"):
        (tmp_path / name).write_text("test\n")
    parent_modules(monkeypatch, config)
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: "")
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: None)
    start_real = time.monotonic()
    start_wall = time.time()
    # Accelerate only the absolute admission clock; process waits stay real.
    monkeypatch.setattr(module.time, "time", lambda: start_wall + 100 * (time.monotonic() - start_real))
    monkeypatch.setattr(sys, "argv", ["pack", "--source", str(tmp_path), "--config", "config.json",
        "--output", str(tmp_path / "output"), "--target", "test", "--mode", "timing",
        "--dataset-root", str(tmp_path / "custom_dataset"),
        "--deadline", str(start_wall + 130), "--max-seconds", "130"])
    assert module.main() == 1
    status = json.loads((tmp_path / "output/pack_status.json").read_text())
    assert status["state"] == "failed" and "deadline" in status["error"].lower()
    assert (tmp_path / "output/config/dataset_root").read_text() == str(tmp_path / "custom_dataset")
    for pid in (tmp_path / "output/config/pids").read_text().split():
        proc = Path(f"/proc/{pid}/stat")
        assert not proc.exists() or proc.read_text().split()[2] == "Z"
    assert (tmp_path / "output/pack_exit_code").read_text().strip() == "1"
