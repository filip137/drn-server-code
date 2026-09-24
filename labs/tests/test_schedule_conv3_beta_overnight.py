"""Admission must preserve unrelated GPU work and the overnight window."""
import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[2] / 'experiments/schedule_conv3_beta_overnight.py'
    spec = importlib.util.spec_from_file_location('overnight_beta', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_only_idle_requested_gpu_classes_are_admitted():
    m = module()
    assert m.idle_snapshot('NVIDIA GeForce RTX 5090, 2, 0\n', '')
    assert m.idle_snapshot('NVIDIA GeForce RTX 3090, 28, 0\n', '123, nvidia-cuda-mps-server\n')
    assert not m.idle_snapshot('NVIDIA GeForce RTX 3090, 28, 0\n', '123, nvidia-cuda-mps-server\n124, python\n')
    assert not m.idle_snapshot('NVIDIA GeForce RTX 3090, 28, 99\n', '123, nvidia-cuda-mps-server\n')
    assert not m.idle_snapshot('NVIDIA GeForce RTX 5090, 1000, 0\n', '')
    assert not m.idle_snapshot('NVIDIA GeForce RTX 3080, 2, 0\n', '')
    assert not m.idle_snapshot('NVIDIA GeForce RTX 5090, 2, 0\n', '124, python\n')


def test_no_early_or_late_production_admission():
    m = module()
    assert not m.fits(m.START - .1)
    assert m.fits(m.START)
    assert m.fits(m.DEADLINE - m.CASE_SECONDS - 60)
    assert not m.fits(m.DEADLINE - m.CASE_SECONDS - 59)
    assert m.fits(m.DEADLINE - 7260, 'fifi')
    assert not m.fits(m.DEADLINE - 7259, 'fifi')
    assert not m.fits(m.DEADLINE - 7260, 'nom-cool-1')


def test_pairs_keep_one_host_and_reserve_both_run_budgets():
    m = module()
    rows = [dict(injected_beta=b, scheme=s) for b in m.PRIORITY for s in ('legacy', 'ours')]
    pending = m.ordered_cases(list(reversed(rows)))
    hosts = {}
    assert m.choose_case(pending, hosts, 'fifi', m.START) == rows[0]
    assert m.choose_case(pending, hosts, 'local', m.START) == rows[2]
    assert m.choose_case(pending, hosts, 'fifi', m.START + 100) == rows[1]
    assert m.choose_case(pending, hosts, 'local', m.START + 100) == rows[3]
    # Two hours remain: enough for one run, insufficient for a new pair.
    assert m.choose_case(pending, hosts, 'trex', m.DEADLINE - 7260) is None
    assert len(pending) == 2
    assert m.choose_case(pending, hosts, 'loulou', m.START) == rows[4]
    assert m.choose_case(pending, hosts, 'loulou', m.START + 100) == rows[5]
    assert not pending and len(hosts) == 3


def test_ours_and_legacy_use_their_own_beta_conversion():
    import copy
    import json
    import pytest
    m = module()
    configs = Path(__file__).resolve().parents[2] / 'configs/conv/eqprop_conv3_beta_sweep_5em4_10ep_20260922_v2'
    for path in configs.glob('*.json'):
        data = json.loads(path.read_text())
        m.check_config(data)
        wrong = copy.deepcopy(data)
        wrong['beta'] *= 64
        with pytest.raises(AssertionError):
            m.check_config(wrong)
    assert len(list(configs.glob('*.json'))) == 6
