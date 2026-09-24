import copy
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from experiments.compute_budget import committed_hours, extra_5090_deadline, reserve
from experiments.check_training_pilots import (
    stable_metrics, check_block, check_bounded_groups, validate_bounded_replication,
)
from experiments.prepare_paper_training_completion import resolved_config

ROOT = Path(__file__).resolve().parents[2]


def test_budget_counts_outstanding_full_reservations_and_settled_actuals():
    rows = [{"name": "done", "state": "settled", "reserved_gpu_hours": 100., "actual_gpu_hours": 90.}]
    rows = reserve(rows, "wave", 200., 300.)
    assert committed_hours(rows) == 290
    assert committed_hours(reserve(rows, "last", 10., 300.)) == 300
    with pytest.raises(ValueError, match="limit"):
        reserve(rows, "over", 10.001, 300.)
    with pytest.raises(ValueError, match="already exists"):
        reserve(rows, "wave", 1., 300.)
    with pytest.raises(ValueError):
        reserve(rows, "nan", float("nan"), 300.)


def test_extra_5090_weekday_boundary_and_one_hour_margin():
    # Friday UTC can already be Saturday in Paris.
    now = datetime(2026, 9, 11, 22, 0, tzinfo=timezone.utc)
    assert extra_5090_deadline(now, 8*3600).isoformat() == "2026-09-14T00:00:00+02:00"
    with pytest.raises(ValueError, match="weekdays"):
        extra_5090_deadline(datetime(2026,9,11,12,tzinfo=timezone.utc), 3600)
    with pytest.raises(ValueError, match="margin"):
        extra_5090_deadline(datetime(2026,9,13,14,tzinfo=timezone.utc), 8*3600)
    with pytest.raises(ValueError, match="weekdays"):
        extra_5090_deadline(datetime(2026,9,13,22,tzinfo=timezone.utc), 3600)


def test_pilot_requires_full_budget_and_strict_five_point_drop_without_floor():
    assert stable_metrics(.21, .20, 30, 30)
    assert not stable_metrics(.95, .90, 30, 30)
    assert not stable_metrics(.9, .85, 30, 30)
    assert not stable_metrics(.99, .99, 1, 30)
    assert not stable_metrics(.99, float("nan"), 30, 30)


def test_pilot_block_rejects_missing_or_duplicated_conditions(monkeypatch):
    import experiments.check_training_pilots as module
    rows = [{"architecture":f"conv{a}", "voltage_amp":av, "current_amp":ai,
             "weight_max":100., "stable":True, "base_beta":1.}
            for a in (1,2,3) for av,ai in ((1.,1.),(4.,1.),(4.,.25))]
    monkeypatch.setattr(module, "check_pilot", lambda index: rows[index])
    assert check_block(list(range(9)), False)["stable"]
    with pytest.raises(ValueError, match="missing, duplicate"):
        check_block(list(range(8)), False)
    with pytest.raises(ValueError, match="missing, duplicate"):
        check_block([*range(8),0], False)
    rows[-1]["stable"] = False
    with pytest.raises(ValueError, match="stability"):
        check_block(list(range(9)), False)


def test_seed_transform_preserves_full_precision_rates_and_both_initializer_paths():
    parent = json.loads((ROOT / "configs/conv/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv1/03_wmax_5em4_baseline_adam.json").read_text())
    original = copy.deepcopy(parent)
    ep = {"eqprop": {"variant":"centered", "current_scale":"auto", "nudging_mode":"current", "normalize_current_scale":True}}
    row = dict(model_seed="2", epochs="10", algorithm="EP", architecture="conv1", scheme="baseline", block="T3_EP", cell_id="bounded2", template_or_parent_config="parent.json", T="4", K="4")
    asset = {"checkpoint_path":"shared/conv1/seed2.pt", "checkpoint_sha256":"f"*64}
    config = resolved_config(row, parent, asset, ep)
    assert parent == original
    assert config["lr"] == parent["lr"] == config["optimizer"]["learning_rate"]
    assert config["seed"] == config["datasets"]["mnist"]["params"]["shuffle_seed"] == 2
    assert config["datasets"]["mnist"]["params"]["split_seed"] == 0
    assert config["init_checkpoint_path"] == config["weight_ceiling_sweep"]["initializer_checkpoint_path"] == asset["checkpoint_path"]
    assert config["weight_ceiling_sweep"]["initializer_checkpoint_sha256"] == asset["checkpoint_sha256"]
    assert config["completion_plan"]["replication_gate"] == "qualified_group_seed0_pilots"
    assert config["evaluation"]["official_test"]["policy"] == "disabled"


def bounded_replication_fixture():
    config = json.loads((ROOT / "configs/conv/paper_training_completion_20260911_v1/training/T3_EP_conv1_ours_gmax0.0001_seed1.json").read_text())
    rows = [{"architecture": "conv1", "voltage_amp": 4., "current_amp": 1.,
             "weight_max": ceiling, "stable": True, "base_beta": config["beta"]}
            for ceiling in (1e-4, 5e-4, 1e-3)]
    gate = {"scope": "complete_bounded_groups", "stable": True, "bounded": True,
            "official_test_read": False, "pilots": rows}
    return config, gate


def test_qualified_bounded_group_releases_both_seeds_without_unqualified_groups(monkeypatch):
    import experiments.check_training_pilots as module
    config, gate = bounded_replication_fixture()
    monkeypatch.setattr(module, "check_pilot", lambda index: gate["pilots"][index])
    report = check_bounded_groups([0, 1, 2])
    for seed in (1, 2):
        config["seed"] = seed
        validate_bounded_replication(config, report)
    # The original complete-table gate remains strict for callers using it.
    with pytest.raises(ValueError, match="missing, duplicate"):
        check_block([0, 1, 2], True)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "unstable", "mixed_beta",
                                     "wrong_beta", "unqualified", "other_group", "short_budget"])
def test_bounded_replication_rejects_incomplete_or_mismatched_pilot_contract(problem):
    config, gate = bounded_replication_fixture()
    if problem == "missing":
        gate["pilots"].pop()
    elif problem == "duplicate":
        gate["pilots"].append(copy.deepcopy(gate["pilots"][0]))
    elif problem == "unstable":
        gate["pilots"][0]["stable"] = False
    elif problem == "mixed_beta":
        gate["pilots"][0]["base_beta"] *= .1
    elif problem == "wrong_beta":
        config["beta"] *= .1
    elif problem == "unqualified":
        config["completion_plan"]["qualification"] = "bounded_beta_pending"
    elif problem == "other_group":
        config["model_base"]["voltage_amp"] = 1.
    elif problem == "short_budget":
        config["lab"]["epochs"] = 1
    with pytest.raises(ValueError):
        validate_bounded_replication(config, gate)


def test_revised_operating_point_cannot_reuse_native_or_mixed_tk_pilots():
    config, gate = bounded_replication_fixture()
    config['completion_plan']['operating_point_revision'] = 'test_revision'
    config['model_base'].update(num_iterations_inference=8, num_iterations_training=4)
    with pytest.raises(ValueError, match='Revised T/K'):
        validate_bounded_replication(config, gate)
    for row in gate['pilots']:
        row.update(T=8, K=4)
    validate_bounded_replication(config, gate)
    gate['pilots'][0]['T'] = 4
    with pytest.raises(ValueError, match='Revised T/K'):
        validate_bounded_replication(config, gate)


def test_gradient_replay_loads_explicit_initializer_and_rejects_wrong_hash(tmp_path):
    # Isolate this historical analyzer's import-time --config bootstrap.
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text(json.dumps({"source_contract":{"runtime_source_root":str(ROOT)}}))
    program = r'''
import sys, json, hashlib
from pathlib import Path
sys.argv = ['replay', '--config', sys.argv[1]]
from experiments import analyze_conv_eqprop_bptt_checkpoint_gradients as base
from experiments.prepare_paper_training_completion import build_energy
source = Path('configs/conv/perfectdiode-conv123-fixed-uniform-init-wmax-adam-10-30-30ep-seed0-20260814-v1/conv1/03_wmax_5em4_baseline_adam.json')
config = json.loads(source.read_text())
config['model_base']['weight_max'] = 1e-4
energy = build_energy(config)
checkpoint = Path(sys.argv[-1]).with_name('init.pt')
energy.save(checkpoint)
expected = [p.state.clone() for p in energy.params()]
config['model_base']['weight_max'] = 5e-4
config['init_checkpoint_path'] = str(checkpoint)
sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
config['weight_ceiling_sweep']['initializer_checkpoint_sha256'] = sha
runtime = base._build_runtime(config, device=base.torch.device('cpu'), gradient_iterations=4)
assert all(base.torch.equal(a,p.state) for a,p in zip(expected,runtime['parameters']))
config['weight_ceiling_sweep']['initializer_checkpoint_sha256'] = '0'*64
try:
    base._build_runtime(config, device=base.torch.device('cpu'), gradient_iterations=4)
except ValueError as e:
    assert 'hash mismatch' in str(e)
else:
    raise AssertionError('Corrupt initializer accepted')
'''
    result = subprocess.run([sys.executable,"-c",program,str(bootstrap)], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("array_index", [None, "0", "3"])
def test_serial_worker_and_array_select_one_config_without_double_indexing(tmp_path, array_index):
    source = tmp_path / "source"
    source.mkdir()
    (source / "SOURCE_COMMIT").write_text("test-source\n")
    (source / "SOURCE_ARCHIVE_SHA256").write_text("a"*64+"\n")
    (source / "asset.pt").write_bytes(b"dry-run initializer")
    config = json.loads((ROOT / "configs/conv/paper_training_completion_20260911_v1/training/T1_BPTT_conv1_baseline_gmax100_seed1.json").read_text())
    config["init_checkpoint_path"] = "asset.pt"
    config["initialization"]["checkpoint_sha256"] = hashlib.sha256(b"dry-run initializer").hexdigest()
    names = [f"case{index}.json" for index in range(4)]
    for name in names:
        (source / name).write_text(json.dumps(config))
    (source / "configs.txt").write_text("\n".join(names)+"\n")
    (source / "INPUT_SHA256SUMS").write_text("".join(
        hashlib.sha256((source/name).read_bytes()).hexdigest()+"  "+name+"\n" for name in names))
    fake_python = tmp_path / "python-dry-run"
    fake_python.write_text("#!/usr/bin/env bash\nset -e\n"+
        f"export PYTHONPATH={shlex.quote(str(ROOT))}\n"+
        'if [[ $1 == -m ]]; then\n'+
        f'  exec {shlex.quote(sys.executable)} "$@" --dry-run\n'+
        'fi\n'+f'exec {shlex.quote(sys.executable)} "$@"\n')
    fake_python.chmod(0o755)
    fake_nvidia = tmp_path / "nvidia-smi"
    fake_nvidia.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_nvidia.chmod(0o755)
    environment = os.environ.copy()
    environment.pop("SLURM_ARRAY_TASK_ID", None)
    environment.pop("SLURM_JOB_ID", None)
    environment["PATH"] = str(tmp_path) + os.pathsep + environment["PATH"]
    if array_index is not None:
        environment["SLURM_ARRAY_TASK_ID"] = array_index
    command = ["bash", str(ROOT / "experiments/run_exact_training_list.sh"), str(source),
               str(source/"configs.txt"), str(tmp_path/"output"), str(tmp_path/"data"),
               "test", str(fake_python)]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    summaries = list((tmp_path/"output").rglob("*.summary.json"))
    assert len(summaries) == (4 if array_index is None else 1)
    if array_index is not None:
        assert summaries[0].name == f"case{array_index}.summary.json"
    repeated = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert repeated.returncode != 0
    if array_index is None:
        # A newly occupied lane must not admit even the first dry-run case.
        fake_nvidia.write_text("#!/usr/bin/env bash\necho 12345\n")
        command[4] = str(tmp_path / "occupied-output")
        occupied = subprocess.run(command, env=environment, capture_output=True, text=True)
        assert occupied.returncode == 75
        assert "GPU_OCCUPIED" in occupied.stderr
        assert not list((tmp_path / "occupied-output").rglob("*.summary.json"))
