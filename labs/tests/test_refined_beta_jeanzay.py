"""Admission/duplicate checks must fail before the cluster module or GPU is used."""
import os
from pathlib import Path
import re
import subprocess
import pytest
from experiments.exact_run import selected_configs

WRAPPER = Path(__file__).resolve().parents[2] / 'experiments/run_refined_beta_jeanzay.sh'


def invoke(tmp_path, task):
    env = os.environ.copy()
    env.pop('SLURM_ARRAY_JOB_ID', None)
    env.pop('SLURM_ARRAY_TASK_ID', None)
    if task is not None:
        env['SLURM_ARRAY_TASK_ID'] = str(task)
    return subprocess.run(['bash', str(WRAPPER), str(tmp_path / 'source'),
                           str(tmp_path / 'configs'), str(tmp_path / 'out'),
                           str(tmp_path / 'data')], env=env, capture_output=True, text=True)


def test_missing_array_index_stops_before_output_or_modules(tmp_path):
    result = invoke(tmp_path, None)
    assert result.returncode != 0
    assert 'Expected a Slurm array task index' in result.stderr
    assert not (tmp_path / 'out').exists()


def test_unexpected_task_stops_before_output_or_modules(tmp_path):
    result = invoke(tmp_path, 6)
    assert result.returncode == 2
    assert not (tmp_path / 'out').exists()


def test_duplicate_task_preserves_existing_attempt(tmp_path):
    receipt = tmp_path / 'out/task_0/worker_started'
    receipt.mkdir(parents=True)
    (receipt / 'pid').write_text('existing-attempt')
    result = invoke(tmp_path, 0)
    assert result.returncode != 0
    assert 'File exists' in result.stderr
    assert (receipt / 'pid').read_text() == 'existing-attempt'
    assert not (receipt / 'finished_at').exists()


def test_array_maps_each_declared_config_once():
    block = WRAPPER.read_text().split('configs=(', 1)[1].split(')', 1)[0]
    names = re.findall(r'conv3_\w+\.json', block)
    expected = [f'conv3_{scheme}_cos{threshold}_seed0.json'
                for scheme in ('baseline', 'legacy', 'ours') for threshold in (95, 90)]
    assert names == expected
    config_root = WRAPPER.parents[1] / 'configs/conv/eqprop_conv3_refined_beta_training_20260919_v1'
    assert all((config_root / name).exists() for name in names)


@pytest.mark.parametrize('task', range(6))
def test_preselected_single_config_does_not_inherit_slurm_index(tmp_path, task):
    config = tmp_path / 'one.json'
    config.write_text('{}')
    # The shell already selects the case; exact_run must receive an explicit0.
    assert selected_configs([config], 0, environ={'SLURM_ARRAY_TASK_ID': str(task)}) == [(0, config.resolve())]
    command = WRAPPER.read_text().split('timeout --signal=', 1)[1]
    assert 'experiments.exact_run "$config" --index 0' in command
