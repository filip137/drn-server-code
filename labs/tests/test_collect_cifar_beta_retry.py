"""Retry collection is atomic, idempotent and cannot overwrite other evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import collect_cifar_beta_retry as collector


def bundle(path):
    (path/'checkpoints').mkdir(parents=True)
    (path/'checkpoints/final_model.pt').write_bytes(b'exact checkpoint bytes')
    (path/'result.json').write_text('{"complete": true}\n')
    return path


def test_atomic_copy_accepts_empty_canonical_and_identical_repeat(tmp_path):
    source = bundle(tmp_path/'retry')
    destination = tmp_path/'original'
    destination.mkdir()
    assert collector.copy_bundle(source, destination)
    assert collector.tree_identity(source) == collector.tree_identity(destination)
    assert not collector.copy_bundle(source, destination)
    assert not list(tmp_path.glob('.original-collect-*'))


def test_differing_canonical_is_preserved(tmp_path):
    source = bundle(tmp_path/'retry')
    destination = bundle(tmp_path/'original')
    (destination/'result.json').write_text('irreplaceable earlier evidence')
    before = collector.tree_identity(destination)
    with pytest.raises(ValueError, match='refusing overwrite'):
        collector.copy_bundle(source, destination)
    assert collector.tree_identity(destination) == before


def test_failed_copy_leaves_original_empty_and_cleans_staging(tmp_path, monkeypatch):
    source = bundle(tmp_path/'retry')
    destination = tmp_path/'original'
    destination.mkdir()

    def partial_copy(source, temporary, **kwargs):
        (temporary/'partial').write_text('incomplete')
        raise OSError('copy interrupted')

    monkeypatch.setattr(collector.shutil, 'copytree', partial_copy)
    with pytest.raises(OSError, match='interrupted'):
        collector.copy_bundle(source, destination)
    assert list(destination.iterdir()) == []
    assert not list(tmp_path.glob('.original-collect-*'))


def test_collection_receipts_then_original_policy_are_idempotent(tmp_path, monkeypatch):
    original, retry = tmp_path/'original', tmp_path/'retry'
    source = bundle(retry/'cells/1/beta_0p01')
    monkeypatch.setenv('RUN_WATCH_JOB_ID', 'retry-beta001')
    monkeypatch.setattr(collector.subprocess, 'check_output', lambda *a, **kw:
                        json.dumps({'attempts': [{'target': 'local'}]}))
    calls = []

    def atomic_json(path, data):
        path.write_text(json.dumps(data))

    wrapper = SimpleNamespace(ROOT=retry, reporting=SimpleNamespace(atomic_write_json=atomic_json))

    def validate(stage, case):
        assert wrapper.ROOT == retry
        return dict(validated=True, rejected=False, stage=stage, case=case)

    def advance():
        assert wrapper.ROOT == original
        assert (original/'analysis/beta_0p01-1-collection.json').exists()
        calls.append('advance')

    def load(root):
        wrapper.ROOT = root
        return wrapper

    wrapper.validate, wrapper.advance = validate, advance
    monkeypatch.setattr(collector, 'load_wrapper', load)
    first = collector.collect(original, retry, 'beta_0p01', '1')
    second = collector.collect(original, retry, 'beta_0p01', '1')
    assert first == second and calls == ['advance', 'advance']
    assert collector.tree_identity(source) == collector.tree_identity(original/'cells/1/beta_0p01')
    for root in (original, retry):
        receipt = json.loads((root/'analysis/beta_0p01-1-collection.json').read_text())
        assert receipt['retry_root'] == str(retry) and receipt['job_id'] == 'retry-beta001'


def test_unvalidated_retry_never_installs_or_advances(tmp_path, monkeypatch):
    original, retry = tmp_path/'original', tmp_path/'retry'
    bundle(retry/'cells/1/beta_0p01')
    monkeypatch.setenv('RUN_WATCH_JOB_ID', 'retry-beta001')
    monkeypatch.setattr(collector.subprocess, 'check_output', lambda *a, **kw:
                        json.dumps({'attempts': [{'target': 'local'}]}))
    monkeypatch.setattr(collector, 'load_wrapper', lambda root:
                        SimpleNamespace(validate=lambda *a: {'validated': False}))
    with pytest.raises(ValueError, match='did not validate'):
        collector.collect(original, retry, 'beta_0p01', '1')
    assert not (original/'cells/1/beta_0p01').exists()
