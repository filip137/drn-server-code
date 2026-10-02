"""Protect exp020 control identity, fixed horizons and incomplete outcomes."""
import copy
import json

import pytest

from experiments.analyze_current_amp2 import (
    ARCHITECTURES, ETAS, SCIENTIFIC_FIELDS, SCHEMES, comparisons,
    epoch_history, reused, scientific_config, select_attempt,
)


def fixture_config():
    config = {field: None for field in SCIENTIFIC_FIELDS}
    config.update(lab={'dataset_key': 'mnist', 'epochs': 30},
                  datasets={'mnist': {'params': {'root': '/old/path', 'batch_size': 16}}},
                  initialization={'checkpoint_sha256': 'accepted', 'optimizer_state': 'fresh',
                                  'recipe': 'wide_kaiming', 'seed': 0},
                  evaluation={'checkpoint_selection': 'maximum_validation_accuracy',
                              'epoch_split': 'validation', 'official_test': {'policy': 'disabled'}},
                  relative_output_calibration={'target': 4}, lr=[.01, 0],
                  optimizer={'name': 'Adam', 'betas': [.9, .999]},
                  model_base={'current_amp': 1}, eqprop={'endpoint_read_noise_eta': 1e-5})
    return config


def test_historical_reuse_ignores_provenance_but_rejects_numerical_changes():
    old = fixture_config()
    new = copy.deepcopy(old)
    new['datasets']['mnist']['params']['root'] = '/staged/path'
    new['initialization']['optimizer_state'] = 'fresh_at_initialization'
    new['relative_output_calibration']['source'] = 'different-record'
    new['study_id'] = 'different-study'
    new['evaluation']['inclusion_rule'] = 'different-prose'
    assert scientific_config(old) == scientific_config(new)
    for field, value in [('lr', [.02, 0]), ('optimizer', {'name': 'SGD'}),
                         ('model_base', {'current_amp': 2}),
                         ('eqprop', {'endpoint_read_noise_eta': 1e-4})]:
        changed = copy.deepcopy(new)
        changed[field] = value
        assert scientific_config(old) != scientific_config(changed)
    new['initialization']['checkpoint_sha256'] = 'different'
    assert scientific_config(old) != scientific_config(new)


def test_exact_coverage_and_failed_outcome_is_not_imputed():
    rows = [dict(architecture=a, scheme=s, eta=e, state='complete',
                 accuracy_percent={'amp2': 96., 'ours': 94., 'legacy': 93.}[s]-(e > 0))
            for a in ARCHITECTURES for s in SCHEMES for e in ETAS]
    assert sum(reused((r['architecture'], r['scheme'], r['eta'])) for r in rows) == 10
    failed = next(r for r in rows if (r['architecture'], r['scheme'], r['eta']) == ('conv3', 'amp2', 1e-4))
    failed.update(state='failed', accuracy_percent=None)
    pairs, degradation = comparisons(rows)
    assert len(pairs) == 18 and len(degradation) == 27
    assert sum(p['amp2_minus_comparator_pp'] is None for p in pairs) == 2
    assert next(d for d in degradation if (d['architecture'], d['scheme'], d['eta']) == ('conv2', 'amp2', 1e-5))['degradation_pp'] == 1


def test_only_identical_collected_copies_are_deduplicated(tmp_path):
    attempt = dict(identity='same-manifest:same-result', run=str(tmp_path/'first'))
    copied = dict(attempt, run=str(tmp_path/'second'))
    chosen, copies = select_attempt([attempt, copied])
    assert chosen == attempt and copies == [copied['run']]
    with pytest.raises(ValueError, match='Multiple distinct attempts'):
        select_attempt([attempt, dict(copied, identity='other-attempt')])


def test_fixed_horizon_rejects_partial_sequence_and_keeps_lower_final(tmp_path):
    epochs = [dict(kind='epoch', epoch=e, evaluation_split='validation', metrics=dict(
        validation_accuracy=.99 if e == 1 else .91, validation_loss=.1,
        train_accuracy=.95, train_loss=.1)) for e in range(1, 11)]
    metrics = tmp_path/'metrics.jsonl'
    metrics.write_text('\n'.join(json.dumps(e) for e in epochs))
    assert epoch_history(tmp_path, 10)[-1]['metrics']['validation_accuracy'] == .91
    with pytest.raises(ValueError, match='Incomplete/unordered'):
        epoch_history(tmp_path, 30)
    metrics.write_text('\n'.join(json.dumps(e) for e in epochs[:-1]))
    with pytest.raises(ValueError, match='Incomplete/unordered'):
        epoch_history(tmp_path, 10)


def test_adam_epoch10_reset_cannot_pass_uninterrupted_control(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from experiments.analyze_current_amp2 import optimizer_history
    from experiments.reporting import sha256_file
    config = fixture_config()
    path = tmp_path/'optimizer.pt'
    path.write_bytes(b'fixture optimizer checkpoint')
    (tmp_path/'exact_run.json').write_text('{}')
    (tmp_path/'epoch_checkpoint_index.jsonl').write_text(json.dumps(dict(
        epoch=30, optimizer_path=path.name, optimizer_sha256=sha256_file(path))))
    checkpoint = dict(epoch=30, optimizer_class='Adam', rng_state={'torch': 'saved'},
                      optimizer_state_dict={'state': {0: {'step': 3438*30}}})
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(load=lambda *a, **k: checkpoint))
    assert 'verified' in optimizer_history(tmp_path, config, 30)
    checkpoint['optimizer_state_dict']['state'][0]['step'] = 3438*20
    with pytest.raises(ValueError, match='Adam step count'):
        optimizer_history(tmp_path, config, 30)


def test_authorized_checkpoint_continuation_is_labeled_and_hash_checked(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from experiments.analyze_current_amp2 import optimizer_history
    from experiments.reporting import sha256_file
    source = tmp_path/'source'
    source.mkdir()
    (source/'manifest.json').write_text('{}')
    saved = {'epoch': 28}
    for kind in ('model', 'optimizer'):
        path = tmp_path/f'epoch28_{kind}.pt'
        path.write_bytes(kind.encode())
        saved[kind+'_path'] = path.name
        saved[kind+'_sha256'] = sha256_file(path)
    final = tmp_path/'final_optimizer.pt'
    final.write_bytes(b'final optimizer')
    (tmp_path/'epoch_checkpoint_index.jsonl').write_text(json.dumps(dict(
        epoch=30, optimizer_path=final.name, optimizer_sha256=sha256_file(final))))
    (tmp_path/'exact_run.json').write_text(json.dumps({'diagnostics': {'resume_diagnostic_run': str(source)}}))
    checkpoint = dict(epoch=30, optimizer_class='Adam', rng_state={'torch': 'saved'},
                      optimizer_state_dict={'state': {0: {'step': 3438*30}}})
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(load=lambda *a, **k: checkpoint))
    with pytest.raises(ValueError, match='lacks recovery receipt'):
        optimizer_history(tmp_path, fixture_config(), 30)
    receipt = dict(schema='diagnostic-epoch-recovery/v1', optimizer_state_restored=True,
        uninterrupted_trajectory=False, continuation_quality='approximate_stochastic_continuation',
        completed_epochs=28, total_epochs=30, remaining_epochs=2, next_epoch=29,
        source_run=str(source), source_manifest={'sha256': sha256_file(source/'manifest.json')}, checkpoint=saved)
    (tmp_path/'diagnostic_resume.json').write_text(json.dumps(receipt))
    history = optimizer_history(tmp_path, fixture_config(), 30)
    assert 'continued from epoch 28' in history and 'approximate stochastic' in history
    assert 'uninterrupted Adam' not in history
    checkpoint['optimizer_state_dict']['state'][0]['step'] = 3438*2
    with pytest.raises(ValueError, match='Adam step count'):
        optimizer_history(tmp_path, fixture_config(), 30)
    checkpoint['optimizer_state_dict']['state'][0]['step'] = 3438*30
    (tmp_path/saved['model_path']).write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Recovery checkpoint hash changed'):
        optimizer_history(tmp_path, fixture_config(), 30)
