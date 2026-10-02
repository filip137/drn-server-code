"""Scientific integrity and failure accounting for exp022's paired new seeds."""
import copy
import hashlib
import json
import math
import random

import numpy as np
import pytest
import torch

from experiments import analyze_relative_noise_seeds as analysis
from experiments.reporting import append_metric, complete_run, sha256_file, start_run


def config_record(tmp_path):
    config = dict(
        arm_id='conv1_baseline_relative1_eta1em5_seed1', study_id=analysis.STUDY,
        seed=1, initialization=dict(seed=1, checkpoint_sha256='a' * 64, optimizer_state='fresh_at_initialization'),
        lab=dict(epochs=10, dataset_key='mnist'),
        datasets=dict(mnist=dict(params=dict(batch_size=16, shuffle_seed=1, split_seed=0))),
        max_batches=None, max_test_batches=None, optimizer=dict(name='Adam', learning_rate=[.01, .01, 0.]),
        lr=[.01, .01, 0.], parameter_order=['ConvWeight_0', 'DenseWeight_0', 'Bias_0'],
        learning_rates_by_parameter=dict(ConvWeight_0=.01, DenseWeight_0=.01, Bias_0=0.),
        training_algorithm='EP', runtime_dtype='float64',
        eqprop=dict(endpoint_read_noise_eta=1e-5, endpoint_read_noise_std=0.,
                    endpoint_read_noise_mode='nodewise_relative', endpoint_read_noise_seed=123,
                    input_read_noise=False, validation_read_noise=False),
        relative_noise_training=dict(relative_target=1),
        read_noise_seed_replication=dict(optimizer_reset_at_epoch10=False, rng_reset_at_epoch10=False),
        evaluation=dict(official_test=dict(policy='disabled'), epoch_split='validation'))
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    record = dict(arm_id=config['arm_id'], config=str(path), config_sha256=sha256_file(path),
                  checkpoint_sha256='a' * 64, seed=1, depth=1, scheme='baseline', eta=1e-5, epochs=10)
    return config, record


def rng_payload(config, epoch):
    state = torch.Generator().manual_seed(epoch).get_state()
    return dict(
        rng_state=dict(python=random.Random(epoch).getstate(), numpy=np.random.RandomState(epoch).get_state(),
                       torch_cpu=state, torch_cuda=[state]),
        continuation_state=dict(
            contract=dict(source='b' * 64, epochs=10, max_batches=None, max_test_batches=None,
                          config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()),
            sampler_rng=state, worker_rng=state,
            endpoint_noise=dict(generators={'cuda:0': state} if epoch else {}, draw_count=epoch * 100)))


def complete_fixture(tmp_path, mutation=None, *, official_test_policy='disabled',
                     official_test_evaluations=0, official_test_examples=0, official_test_read=False,
                     runtime_config_mutation=None):
    config, record = config_record(tmp_path)
    config['evaluation']['official_test']['policy'] = official_test_policy
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    record['config_sha256'] = sha256_file(config_path)
    run = tmp_path / '000_case'
    used_config = copy.deepcopy(config)
    used_config['datasets']['mnist']['params']['root'] = '/staged/mnist'
    if runtime_config_mutation:
        runtime_config_mutation(used_config)
    dataset = dict(evaluation_split='validation', official_test_policy=official_test_policy,
                   official_test_read=official_test_read, official_test_read_during_training=False)
    manifest = dict(study_id=analysis.STUDY, run_id=run.name, arm_id=config['arm_id'],
                    evidence_class='ordinary_mnist_selection', git=dict(source_archive_sha256='b' * 64), configuration=dict(
                        resolved=config, sha256=record['config_sha256'], epochs=10, configured_epochs=10,
                        diagnostic_overrides=dict(checkpoint_every_epoch=True)),
                    dataset=dataset, runtime=dict(device='cuda', host='fixture'),
                    transport_overrides=dict(dataset_root='/staged/mnist'),
                    command=['python', '/frozen/source/labs/mnist_train.py', '--config', str(config_path),
                             '--output-dir', str(run), '--reporting-run-dir', str(run),
                             '--device', 'cuda', '--dataset-root', '/staged/mnist',
                             '--checkpoint-every-epoch', '--epoch-chunk-size', '10'])
    start_run(run, manifest)
    (run / 'config.used.json').write_text(json.dumps(used_config))
    for epoch in range(1, 11):
        append_metric(run / 'metrics.jsonl', dict(kind='epoch', epoch=epoch, evaluation_split='validation',
            metrics=dict(train_accuracy=.95, train_loss=.2, validation_loss=.3,
                         validation_accuracy=.99 if epoch == 1 else .9)))
    directory = run / 'checkpoints'; directory.mkdir(exist_ok=True)
    records = []
    initial_hash = None
    for epoch in range(11):
        states = [torch.tensor([1.], dtype=torch.float64), torch.tensor([2.], dtype=torch.float64), torch.zeros(1, dtype=torch.float64)]
        model = dict(format='drn.function.parameters', version=1,
                     schema=[dict(name=n + (' ' if n.startswith('ConvWeight') else ''), type='fixture', shape=[1], dtype='torch.float64') for n in config['parameter_order']],
                     states=states)
        saved = dict(epoch=epoch, optimizer_class='Adam', parameter_names=config['parameter_order'],
                     optimizer_parameter_groups=[dict(learning_rate=rate, parameter_names=[name])
                        for name, rate in config['learning_rates_by_parameter'].items()],
                     optimizer_state_dict=dict(param_groups=[dict(params=[i], lr=rate) for i, rate in enumerate(config['lr'])],
                        state={i: dict(step=torch.tensor(math.ceil(55000/16)*epoch), exp_avg=torch.zeros(1),
                                       exp_avg_sq=torch.zeros(1)) for i in range(3)} if epoch else {}),
                     **rng_payload(used_config, epoch))
        if mutation:
            mutation(epoch, model, saved)
        model_path = directory / f'epoch_{epoch:03d}_model.pt'
        torch.save(model, model_path)
        saved['model_checkpoint_sha256'] = sha256_file(model_path)
        optimizer_path = directory / f'epoch_{epoch:03d}_optimizer.pt'
        torch.save(saved, optimizer_path)
        if epoch == 0:
            initial_hash = analysis.parameter_hash(model)
        index = dict(epoch=epoch)
        for name, path in [('model', model_path), ('optimizer', optimizer_path)]:
            index.update({name + '_path': str(path.relative_to(run)), name + '_sha256': sha256_file(path),
                          name + '_size_bytes': path.stat().st_size})
        records.append(index)
    (run / 'epoch_checkpoint_index.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    metrics = dict(seed=1, official_test_evaluations=official_test_evaluations,
                   official_test_examples=official_test_examples,
                   training_algorithm='EP', runtime_dtype='float64', final_learning_rate=config['lr'], eqprop=config['eqprop'],
                   final_validation_accuracy=.9, best_validation_accuracy=.99,
                   initial_parameter_state_sha256=initial_hash,
                   dataset_provenance=dict(source_split='train', split_seed=0, shuffle_seed=1,
                       train_batch_order_sha256=['order' + str(i) for i in range(10)],
                       train_indices_sha256='train', validation_indices_sha256='validation'))
    (run / 'metrics.json').write_text(json.dumps(metrics))
    complete_run(run, terminal_metrics=dict(validation=dict(final_accuracy=.9, best_accuracy=.99)))
    return run, config, record


def test_complete_real_canonical_bundle_uses_final_not_best(tmp_path):
    run, config, record = complete_fixture(tmp_path)
    analysis.validate_config(config, record)
    row = analysis.validate_complete(run, config, record)
    assert row['final_accuracy_pct'] == 90
    assert row['best_accuracy_pct'] == 99
    assert row['best_epoch'] == 1
    assert row['checkpoints'] == 11


def test_production_command_needs_no_redundant_official_test_skip_flag(tmp_path):
    run, config, record = complete_fixture(tmp_path)
    command = analysis.read(run / 'manifest.json')['command']
    assert command[-3:] == ['--checkpoint-every-epoch', '--epoch-chunk-size', '10']
    assert '--skip-terminal-official-test' not in command
    assert analysis.validate_complete(run, config, record)['checkpoints'] == 11


def test_canonical_runtime_config_cannot_change_science_under_transport_override(tmp_path):
    run, config, record = complete_fixture(tmp_path,
        runtime_config_mutation=lambda used: used['lr'].__setitem__(0, .02))
    assert analysis.validate_run(run) == []
    with pytest.raises(ValueError, match='used config differs beyond declared dataset-root'):
        analysis.validate_complete(run, config, record)


def test_dataset_root_override_requires_matching_command(tmp_path):
    run, config, _ = complete_fixture(tmp_path)
    manifest = analysis.read(run / 'manifest.json')
    manifest['transport_overrides']['dataset_root'] = '/different/path'
    with pytest.raises(ValueError, match='dataset-root override differs from command'):
        analysis.runtime_configuration(run, config, manifest, analysis.read(run / 'result.json'))


@pytest.mark.parametrize('overrides,match', [
    ({'official_test_policy': 'terminal_once'}, 'official-test config policy'),
    ({'official_test_evaluations': 1}, 'official-test runtime mismatch'),
    ({'official_test_examples': 10000}, 'official-test runtime mismatch'),
    ({'official_test_read': True}, 'official-test access or dataset mismatch'),
])
def test_missing_redundant_flag_does_not_relax_official_test_guards(tmp_path, overrides, match):
    run, config, record = complete_fixture(tmp_path, **overrides)
    with pytest.raises(ValueError, match=match):
        analysis.validate_complete(run, config, record)


@pytest.mark.parametrize('fault,match', [
    ('reset', 'Adam steps'), ('bias', 'bias not zero'), ('private', 'continuation_state'),
    ('noise', 'noise RNG missing or reset'), ('global', 'CUDA RNG state missing'),
    ('nonfinite', 'nonfinite model'), ('optimizer_nonfinite', 'nonfinite Adam'),
])
def test_hash_valid_but_scientifically_invalid_checkpoints_are_rejected(tmp_path, fault, match):
    def mutate(epoch, model, saved):
        if epoch != 10:
            return
        if fault == 'reset':
            saved['optimizer_state_dict']['state'][0]['step'] = torch.tensor(3438)
        elif fault == 'bias':
            model['states'][-1][0] = .01
        elif fault == 'private':
            del saved['continuation_state']
        elif fault == 'noise':
            saved['continuation_state']['endpoint_noise']['draw_count'] = 0
        elif fault == 'global':
            saved['rng_state']['torch_cuda'] = []
        elif fault == 'nonfinite':
            model['states'][0][0] = float('nan')
        elif fault == 'optimizer_nonfinite':
            saved['optimizer_state_dict']['state'][0]['exp_avg'][0] = float('inf')
    run, config, record = complete_fixture(tmp_path, mutate)
    with pytest.raises((ValueError, KeyError), match=match):
        analysis.validate_complete(run, config, record)


def test_changed_actual_checkpoint_hash_rejected_before_deserialization(tmp_path):
    run, config, record = complete_fixture(tmp_path)
    (run / 'checkpoints/epoch_010_optimizer.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        analysis.validate_complete(run, config, record)


def test_missing_epoch_and_nonfinite_metrics_fail(tmp_path):
    run, _, _ = complete_fixture(tmp_path)
    path = run / 'metrics.jsonl'; rows = analysis.read_lines(path)
    path.write_text('\n'.join(json.dumps(r) for r in rows[:-1]))
    with pytest.raises(ValueError, match='exactly 1..N'):
        analysis.epoch_history(run, 10)
    rows[-1]['metrics']['validation_accuracy'] = float('nan')
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='nonfinite'):
        analysis.epoch_history(run, 10)


def test_72_cases_and_paired_sample_sd_with_failed_seed_visible():
    rows = [dict(depth=d, scheme=s, eta=eta, seed=seed, state='complete',
                 final_accuracy_pct={'ours': 97, 'baseline': 92, 'legacy': 95}[s] + (seed-1)*(2 if s == 'ours' else 0),
                 best_accuracy_pct=99) for d, s, eta, seed in analysis.expected_keys()]
    assert len(rows) == 72
    groups, pairs = analysis.aggregate(rows)
    group = groups[1]['final_accuracy_pct']
    assert group['values_by_seed'] == {'1': 97, '2': 99}
    assert group['mean'] == 98 and group['sample_sd'] == pytest.approx(math.sqrt(2))
    gap = pairs[0]['ours_minus_comparator_pp']
    assert gap['values_by_seed'] == {'1': 5, '2': 7}
    assert gap['mean'] == 6 and gap['sample_sd'] == pytest.approx(math.sqrt(2))
    rows[1]['state'] = 'failed'  # baseline seed 2
    _, pairs = analysis.aggregate(rows)
    gap = pairs[0]['ours_minus_comparator_pp']
    assert gap['n'] == 1 and gap['expected_n'] == 2 and not gap['complete']
    assert gap['values_by_seed'] == {'1': 5, '2': None} and gap['sample_sd'] is None


def test_pairing_rejects_mismatch_within_seed_but_not_between_seeds():
    rows = [dict(depth=1, seed=seed, scheme=scheme, state='complete',
                 **{field: f'seed{seed}' for field in ('initial_parameter_state_sha256', 'initializer_checkpoint_sha256',
                    'train_indices_sha256', 'validation_indices_sha256', 'train_batch_order_sha256')})
            for seed in (1, 2) for scheme in analysis.SCHEMES]
    assert not analysis.matched_identity(rows)
    rows[0]['initial_parameter_state_sha256'] = 'different'
    assert analysis.matched_identity(rows)
    assert [r['state'] for r in rows] == ['invalid'] * 3 + ['complete'] * 3


def test_seed_zero_and_nonzero_bias_rates_are_rejected(tmp_path):
    config, record = config_record(tmp_path)
    with pytest.raises(ValueError, match='seed 0'):
        analysis.validate_config(config, dict(record, seed=0))
    config['lr'][-1] = .01
    config['optimizer']['learning_rate'][-1] = .01
    config['learning_rates_by_parameter']['Bias_0'] = .01
    with pytest.raises(ValueError, match='frozen-bias'):
        analysis.validate_config(config, record)


def test_partial_analysis_records_all_cells_and_never_accepts_failure(tmp_path, monkeypatch):
    config, _ = config_record(tmp_path)
    records = []
    for i, (depth, scheme, eta, seed) in enumerate(analysis.expected_keys()):
        candidate = copy.deepcopy(config)
        arm = f'case{i}'
        candidate['arm_id'] = arm
        path = tmp_path / f'{arm}.json'; path.write_text(json.dumps(candidate))
        records.append(dict(depth=depth, scheme=scheme, eta=eta, seed=seed, arm_id=arm,
                            epochs=analysis.HORIZONS[depth], config=str(path), config_sha256=sha256_file(path)))
    (tmp_path / 'preparation.json').write_text(json.dumps(dict(records=records)))
    # Numerical config checking is independently exercised above. Here isolate
    # coverage/output accounting, including a genuine canonical failed bundle.
    monkeypatch.setattr(analysis, 'validate_config', lambda *_: None)
    monkeypatch.setattr(analysis, 'write_plots', lambda *_: None)
    assert analysis.main(['--study-root', str(tmp_path), '--allow-partial']) == 0
    summary = analysis.read(tmp_path / 'analysis/summary.json')
    assert summary['states'] == {'not_collected': 72}
    assert len((tmp_path / 'analysis/coverage.csv').read_text().splitlines()) == 73
    assert not summary['complete']
    from experiments.reporting import fail_run
    record = records[0]; actual = analysis.read(record['config'])
    run = tmp_path / 'collected/production' / record['arm_id'] / 'runs/000_failed'
    start_run(run, dict(study_id=analysis.STUDY, arm_id=record['arm_id'], run_id='000_failed',
        evidence_class='ordinary_mnist_selection', configuration=dict(resolved=actual, sha256=record['config_sha256']),
        runtime=dict(host='fixture'), command=['python'], dataset=dict(evaluation_split='validation')))
    fail_run(run, error=ValueError('nonfinite gradient'))
    assert analysis.main(['--study-root', str(tmp_path), '--allow-partial']) == 1
    summary = analysis.read(tmp_path / 'analysis/summary.json')
    assert summary['states'] == {'failed': 1, 'not_collected': 71}
    assert 'nonfinite gradient' in summary['outcomes'][0]['issue']
    assert 'nonfinite gradient' in (tmp_path / 'analysis/report.md').read_text()


def test_effective_source_identity_must_match_frozen_archive_and_rng_contract(tmp_path):
    run, config, record = complete_fixture(tmp_path)
    with pytest.raises(ValueError, match='effective source archive hash mismatch'):
        analysis.validate_complete(run, config, record, expected_source_sha256='c' * 64)
    assert analysis.validate_complete(run, config, record, expected_source_sha256='b' * 64)['source_archive_sha256'] == 'b' * 64


def test_supplementary_uses_only_24_approved_seed0_cells_and_common_seed_pairs():
    primary = [dict(depth=d, scheme=s, eta=e, seed=seed, state='complete',
                    final_accuracy_pct=95 + (s == 'ours') + seed)
               for d, s, e, seed in analysis.expected_keys()]
    historical = [dict(depth=d, scheme=s, eta=e, seed=0, state='complete', final_accuracy_pct=95 + (s == 'ours'))
                  for d in analysis.HORIZONS for s in analysis.SCHEMES for e in analysis.ETAS
                  if analysis.seed0_source(d, s, e) is not None]
    assert len(historical) == 24
    assert all(r['depth'] == 1 or r['scheme'] == 'baseline' or r['eta'] == 5e-4 for r in historical)
    groups, pairs = analysis.supplement_statistics(primary, historical)
    assert len(groups) == 24 and all(r['final_accuracy_pct']['n'] == 3 for r in groups)
    for pair in pairs:
        common = [0, 1, 2] if pair['depth'] == 1 or pair['eta'] == 5e-4 else [1, 2]
        assert pair['common_seeds'] == common
        assert pair['ours_minus_comparator_pp']['n'] == len(common)
        assert pair['ours_minus_comparator_pp']['mean'] == 1
    # A bad/missing seed0 remains a missing declared third observation.
    historical[0]['state'] = 'invalid'
    _, pairs = analysis.supplement_statistics(primary, historical)
    assert pairs[0]['common_seeds'] == [0, 1, 2]
    assert pairs[0]['ours_minus_comparator_pp']['n'] == 2
    assert pairs[0]['ours_minus_comparator_pp']['values_by_seed']['0'] is None
