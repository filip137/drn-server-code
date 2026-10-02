"""Frozen coverage, scientific identity, and fixed-final reporting checks."""
import copy
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from experiments import analyze_rescaled_target_noise as analysis
from experiments.reporting import append_metric, complete_run, fail_run, sha256_file, start_run


def config(depth=1, scheme='baseline', eta=0.):
    a, b = analysis.AMPLIFICATION[scheme]
    return dict(study_id=analysis.STUDY, arm_id=f'case_{depth}_{scheme}_{eta}',
        seed=0, target_scale=analysis.TARGETS[depth], beta=.001,
        training_algorithm='EP', runtime_dtype='float64', max_batches=None, max_test_batches=None,
        lab=dict(model_key='model', dataset_key='mnist', epochs=analysis.HORIZONS[depth]),
        model_overrides=dict(model=dict(conv_pipeline=[{}] * depth)),
        model_base=dict(voltage_amp=a, current_amp=b),
        lr=[.01] * (depth + 1) + [0.] * depth,
        parameter_order=[f'ConvWeight_{i}' for i in range(depth)] + ['DenseWeight_0'] + [f'Bias_{i}' for i in range(depth)],
        optimizer=dict(name='Adam', learning_rate=[.01] * (depth + 1) + [0.] * depth),
        datasets=dict(mnist=dict(params=dict(batch_size=16, split_seed=0, shuffle_seed=0, root='/data'))),
        eqprop=dict(endpoint_read_noise_eta=eta, endpoint_read_noise_std=0,
            endpoint_read_noise_mode='nodewise_relative', input_read_noise=False,
            validation_read_noise=False, variant='centered', nudging_mode='current',
            injected_beta_B=.001 * (a / b) ** depth),
        evaluation=dict(epoch_split='validation', official_test=dict(policy='disabled'),
                        checkpoint_selection='maximum_validation_accuracy'))


def checkpoint_fixture(run, cfg, mutate=None, start=0):
    directory = run / 'checkpoints'
    directory.mkdir(exist_ok=True)
    records = []
    for epoch in range(start, cfg['lab']['epochs'] + 1):
        state = torch.Generator().manual_seed(epoch).get_state()
        model = dict(format='drn.function.parameters', version=1,
            schema=[dict(name=name, type='fixture', shape=[1], dtype='torch.float64') for name in cfg['parameter_order']],
            states=[torch.tensor([0. if name.startswith('Bias_') else 1.], dtype=torch.float64) for name in cfg['parameter_order']])
        saved = dict(schema_version='mnist-conv-diagnostic-optimizer-checkpoint/v1', epoch=epoch,
            optimizer_class='Adam', parameter_names=cfg['parameter_order'],
            optimizer_parameter_groups=[dict(group_index=i, learning_rate=rate, parameter_names=[name])
                for i, (name, rate) in enumerate(zip(cfg['parameter_order'], cfg['lr']))],
            optimizer_state_dict=dict(param_groups=[dict(params=[i], lr=rate) for i, rate in enumerate(cfg['lr'])],
                state={i: dict(step=torch.tensor(3438 * epoch), exp_avg=torch.zeros(1, dtype=torch.float64),
                               exp_avg_sq=torch.zeros(1, dtype=torch.float64)) for i in range(len(cfg['lr']))} if epoch else {}),
            rng_state=dict(python=random.Random(epoch).getstate(), numpy=np.random.RandomState(epoch).get_state(),
                           torch_cpu=state, torch_cuda=[state]),
            continuation_state=dict(sampler_rng=state, worker_rng=state,
                contract=dict(epochs=cfg['lab']['epochs'], max_batches=None, max_test_batches=None,
                    source='source', config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()),
                endpoint_noise=dict(draw_count=epoch if cfg['eqprop']['endpoint_read_noise_eta'] else 0,
                    generators={'cuda:0': state} if epoch and cfg['eqprop']['endpoint_read_noise_eta'] else {})))
        if mutate:
            mutate(epoch, model, saved)
        model_path = directory / f'epoch_{epoch:03d}_model.pt'
        optimizer_path = directory / f'epoch_{epoch:03d}_optimizer.pt'
        torch.save(model, model_path)
        saved.update(model_checkpoint=model_path.name, model_checkpoint_sha256=sha256_file(model_path))
        torch.save(saved, optimizer_path)
        record = dict(schema_version='mnist-conv-diagnostic-epoch-checkpoint/v1', epoch=epoch)
        for kind, path in [('model', model_path), ('optimizer', optimizer_path)]:
            record.update({kind + '_path': str(path.relative_to(run)), kind + '_sha256': sha256_file(path),
                           kind + '_size_bytes': path.stat().st_size})
        records.append(record)
    (run / 'epoch_checkpoint_index.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))


def bundle(root, cfg, *, accuracy=.9, mutate=None, failed=False, checkpoint_mutation=None, checkpoint_start=0):
    run = root / cfg['arm_id']
    start_run(run, dict(study_id=analysis.STUDY, run_id=run.name, arm_id=cfg['arm_id'],
        configuration=dict(resolved=cfg), runtime=dict(device='cuda', host='fixture'),
        git=dict(source_archive_sha256='source'), command=['python', 'runner'],
        dataset=dict(official_test_read=False, evaluation_split='validation')))
    if failed:
        fail_run(run, error=RuntimeError('fixture failure'))
        return run
    (run / 'config.used.json').write_text(json.dumps(cfg))
    for epoch in range(1, cfg['lab']['epochs'] + 1):
        append_metric(run / 'metrics.jsonl', dict(kind='epoch', epoch=epoch, evaluation_split='validation',
            metrics=dict(train_accuracy=.9, train_loss=.2, validation_loss=.3,
                         validation_accuracy=.99 if epoch == 1 else accuracy)))
    metrics = dict(official_test_evaluations=0, official_test_examples=0, seed=0,
        training_algorithm='EP', runtime_dtype='float64', final_learning_rate=cfg['lr'],
        eqprop=cfg['eqprop'], target_scale=cfg['target_scale'], beta=cfg['beta'],
        best_epoch=1, final_validation_accuracy=accuracy, best_validation_accuracy=.99,
        initial_parameter_state_sha256='initial', dataset_provenance=dict(source_split='train',
            split_seed=0, shuffle_seed=0, train_batch_order_sha256=['batch'] * cfg['lab']['epochs'],
            train_indices_sha256='train', validation_indices_sha256='validation'))
    if mutate:
        mutate(metrics)
    (run / 'metrics.json').write_text(json.dumps(metrics))
    for role in ('best', 'final'):
        (run / f'{role}_model.pt').write_bytes(b'fixture checkpoint')
    checkpoint_fixture(run, cfg, checkpoint_mutation, checkpoint_start)
    complete_run(run, terminal_metrics=dict(validation=dict(final_accuracy=accuracy, best_accuracy=.99)))
    return run


def test_all_48_scientific_keys_and_target_schedule():
    assert len(analysis.expected_keys()) == 48
    for depth, scheme, eta in analysis.expected_keys():
        cfg = config(depth, scheme, eta)
        assert analysis.validate_config(cfg) == (depth, scheme, eta)
        # Arm names deliberately do not encode scientific metadata.
        cfg['arm_id'] = 'arbitrary'
        assert analysis.config_key(cfg) == (depth, scheme, eta)


@pytest.mark.parametrize('field,value,match', [
    ('target_scale', 1., 'target scale'), ('training_algorithm', 'BP', 'algorithm'),
    ('seed', 1, 'seed'), ('beta', -1., 'beta'), ('max_batches', 2, 'shortened')])
def test_wrong_scientific_contract_rejected(field, value, match):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError, match=match):
        analysis.validate_config(cfg)


def test_final_is_primary_best_secondary_and_loss_unscaled(tmp_path):
    cfg = config()
    run = bundle(tmp_path, cfg)
    row, history = analysis.validate_complete(run, cfg)
    assert row['final_accuracy_pct'] == 90
    assert row['best_accuracy_pct'] == 99
    assert row['best_epoch'] == 1
    assert row['final_validation_loss'] == .3
    assert len(history) == 10


@pytest.mark.parametrize('mutation,match', [
    (lambda m: m.update(official_test_evaluations=1), 'official test'),
    (lambda m: m.update(target_scale=1.), 'target_scale'),
    (lambda m: m.update(beta=.02), 'beta'),
    (lambda m: m.update(final_learning_rate=[1., 1., 0.]), 'learning rates'),
    (lambda m: m.update(final_validation_accuracy=.99), 'terminal accuracy'),
    (lambda m: m['eqprop'].update(endpoint_read_noise_eta=1e-3), 'EqProp')])
def test_hash_valid_runtime_science_changes_fail(tmp_path, mutation, match):
    cfg = config()
    run = bundle(tmp_path, copy.deepcopy(cfg), mutate=mutation)
    with pytest.raises(ValueError, match=match):
        analysis.validate_complete(run, cfg)


def test_used_config_transport_does_not_hide_lr_change(tmp_path):
    cfg = config()
    run = bundle(tmp_path, cfg)
    changed = copy.deepcopy(cfg)
    changed['lr'][0] = .3
    (run / 'config.used.json').write_text(json.dumps(changed))
    # Reseal a fixture to distinguish scientific checks from artifact hashing.
    status = analysis.read(run / 'status.json')
    status['state'] = 'running'
    (run / 'status.json').write_text(json.dumps(status))
    complete_run(run, terminal_metrics=dict(validation=dict(final_accuracy=.9, best_accuracy=.99)))
    with pytest.raises(ValueError, match='used config differs'):
        analysis.validate_complete(run, cfg)


def test_full_coverage_and_all_six_contrasts_per_depth_eta(tmp_path):
    config_root = tmp_path / 'configs'
    config_root.mkdir()
    runs = tmp_path / 'production'
    for depth, scheme, eta in analysis.expected_keys():
        cfg = config(depth, scheme, eta)
        (config_root / (cfg['arm_id'] + '.json')).write_text(json.dumps(cfg))
        accuracy = .90 + .01 * analysis.SCHEMES.index(scheme) - .005 * analysis.ETAS.index(eta)
        bundle(runs, cfg, accuracy=accuracy)
    summary = analysis.analyze(tmp_path, config_root, [runs], plots=False)
    assert summary['complete']
    assert summary['completed_runs'] == 48
    assert summary['completed_pairs'] == 72
    noisy = next(r for r in summary['clean_degradation'] if r['depth'] == 1 and r['scheme'] == 'baseline' and r['eta'] == 1e-4)
    assert noisy['clean_minus_noisy_pp'] == pytest.approx(1.5)
    pair = summary['pairwise_differences'][0]
    assert (pair['first'], pair['second']) == ('baseline', 'ours')
    assert pair['first_minus_second_pp'] == pytest.approx(-1.)


def test_missing_failed_ambiguous_and_excluded_attempts_stay_visible(tmp_path):
    config_root = tmp_path / 'configs'
    config_root.mkdir()
    cfg = config()
    (config_root / 'case.json').write_text(json.dumps(cfg))
    failed = bundle(tmp_path / 'attempt1', cfg, failed=True)
    result = analysis.analyze(tmp_path, config_root, [tmp_path / 'attempt1'], plots=False)
    assert result['states'] == {'failed': 1, 'missing_config': 47}
    assert len(result['coverage']) == 48 and not result['complete']
    bundle(tmp_path / 'attempt2', cfg)
    roots = [tmp_path / 'attempt1', tmp_path / 'attempt2']
    assert analysis.analyze(tmp_path, config_root, roots, plots=False)['states']['ambiguous'] == 1
    result = analysis.analyze(tmp_path, config_root, roots, exclude_runs=[failed], plots=False)
    assert result['completed_runs'] == 1 and not result['complete']
    assert any(a['excluded'] for a in result['attempts'])


def test_source_or_initialization_mismatch_prevents_pairing():
    row = dict(source_archive_sha256='source', train_indices_sha256='train', validation_indices_sha256='val',
               train_batch_order_sha256=['batch'], initial_parameter_state_sha256='initial')
    assert analysis.matching_issue(row, row) == ''
    for name in row:
        changed = dict(row, **{name: 'different'})
        assert name in analysis.matching_issue(row, changed)


def test_default_cli_fails_incomplete_allow_partial_only_changes_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, 'write_plots', lambda *args: None)
    argv = ['--study-root', str(tmp_path)]
    assert analysis.main(argv) == 1
    assert analysis.main(argv + ['--allow-partial']) == 0
    assert not analysis.read(tmp_path / 'analysis/summary.json')['complete']


@pytest.mark.parametrize('fault,match', [
    ('private_rng', 'private RNG'), ('noise_rng', 'private noise RNG'), ('adam_nan', 'Invalid Adam tensor'),
    ('adam_reset', 'Invalid Adam step'), ('bias', 'nonzero final bias'), ('global_rng', 'global CUDA RNG'),
    ('approximate_resume', 'not exact full-state')])
def test_full_state_damage_rejected_even_when_hashes_are_valid(tmp_path, fault, match):
    cfg = config(eta=1e-5)
    def damage(epoch, model, saved):
        if epoch != cfg['lab']['epochs']:
            return
        if fault == 'private_rng':
            saved['continuation_state'].pop('sampler_rng')
        elif fault == 'noise_rng':
            saved['continuation_state']['endpoint_noise']['generators'] = {}
        elif fault == 'adam_nan':
            saved['optimizer_state_dict']['state'][0]['exp_avg'][0] = float('nan')
        elif fault == 'adam_reset':
            saved['optimizer_state_dict']['state'][0]['step'] = torch.tensor(3438)
        elif fault == 'bias':
            model['states'][-1][0] = .1
        elif fault == 'global_rng':
            saved['rng_state']['torch_cuda'] = []
        elif fault == 'approximate_resume':
            saved['diagnostic_resume'] = dict(continuation_quality='approximate_stochastic_continuation',
                rng_missing=['endpoint_read_noise_generators'], rng_reconstructed=['sampler'])
    run = bundle(tmp_path, cfg, checkpoint_mutation=damage)
    assert analysis.validate_run(run) == []
    with pytest.raises(ValueError, match=match):
        analysis.validate_complete(run, cfg)


@pytest.mark.parametrize('start,expected', [(0, 11), (1, 10)])
def test_native_epoch_checkpoint_coverage_accepts_zero_or_one_start(tmp_path, start, expected):
    cfg = config()
    run = bundle(tmp_path, cfg, checkpoint_start=start)
    assert analysis.validate_complete(run, cfg)[0]['checkpoints'] == expected
