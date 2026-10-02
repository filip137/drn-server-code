"""Validate exp022 seed-1/2 bundles and summarize fixed-final validation accuracy.

Run after rsync with ``python -m experiments.analyze_relative_noise_seeds
--study-root RESULTS [--allow-partial]``. Preparation records freeze the 72 cases;
collected/production/ARM/runs/000_* contains each production bundle. No historical
seed-0 evidence is pooled. This command reads checkpoints but never trains.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

from experiments.reporting import sha256_file, validate_run

REPO = Path(__file__).resolve().parents[1]
STUDY = 'conv123-relative-noise-seeds12-jeanzay-20260930-v1'
SCHEMES = ('baseline', 'ours', 'legacy')
ETAS = (1e-5, 1e-4, 3e-4, 5e-4)
SEEDS = (1, 2)
HORIZONS = {1: 10, 2: 30, 3: 30}
TARGETS = {1: 1, 2: 4, 3: 6}
QUALIFICATION = (
    'Ordinary MNIST validation only; seeds 1 and 2, paired within seed. Primary: '
    'final epoch 10/30/30 for Conv1/2/3; best accuracy is secondary. '
    'Mean and sample SD are descriptive (n=2), without significance or optimum claims. '
    'Historical seed 0 is excluded from the primary analysis, including the Conv2/3 ours/legacy Adam/RNG-reset '
    'runs at eta 1e-5, 1e-4 and 3e-4. Failed or missing seeds are never imputed.'
)


def read(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(record):
    return int(record['depth']), record['scheme'], float(record['eta']), int(record['seed'])


def expected_keys():
    return [(d, s, e, seed) for d in HORIZONS for s in SCHEMES for e in ETAS for seed in SEEDS]


def config_path(root, record):
    path = Path(record['config'])
    candidates = [path] if path.is_absolute() else [root / 'source' / path, root / path, REPO / path]
    return next((p for p in candidates if p.is_file()), candidates[0])


def validate_config(config, record):
    depth, scheme, eta, seed = key(record)
    require(key(record) in expected_keys(), 'undeclared cell (seed 0 is never eligible)')
    require(config['arm_id'] == record['arm_id'] and config['study_id'] == STUDY, 'config identifiers')
    require(config['seed'] == config['initialization']['seed'] == seed, 'initialization/model seed')
    require(config['initialization']['checkpoint_sha256'] == record['checkpoint_sha256'], 'initializer hash')
    require(config['initialization']['optimizer_state'] in ('fresh', 'fresh_at_initialization'), 'fresh Adam required')
    require(config['lab']['epochs'] == record['epochs'] == HORIZONS[depth], 'configured horizon')
    params = config['datasets'][config['lab']['dataset_key']]['params']
    require(params['batch_size'] == 16 and params['shuffle_seed'] == seed and params['split_seed'] == 0,
            'batch size, shuffle seed or fixed validation split')
    require(config['max_batches'] is None and config['max_test_batches'] is None, 'shortened data')
    require(config['optimizer']['name'] == 'Adam', 'optimizer must be Adam')
    require(config['optimizer']['learning_rate'] == config['lr'], 'optimizer learning rates')
    require(config['training_algorithm'] == 'EP' and config['runtime_dtype'] == 'float64', 'algorithm/dtype')
    noise = config['eqprop']
    require(noise['endpoint_read_noise_eta'] == eta and noise['endpoint_read_noise_std'] == 0
            and noise['endpoint_read_noise_mode'] == 'nodewise_relative', 'relative read noise')
    require(noise['input_read_noise'] is False and noise['validation_read_noise'] is False, 'noise scope')
    require(config['relative_noise_training']['relative_target'] == TARGETS[depth], 'D/F target')
    require(config['read_noise_seed_replication']['optimizer_reset_at_epoch10'] is False
            and config['read_noise_seed_replication']['rng_reset_at_epoch10'] is False, 'optimizer/RNG reset metadata')
    require(config['evaluation']['official_test']['policy'] == 'disabled'
            and config['evaluation']['epoch_split'] == 'validation', 'official test config')
    rates = dict(zip(config['parameter_order'], config['lr'], strict=True))
    require(rates == config['learning_rates_by_parameter'], 'named learning rates')
    biases = [name for name in rates if name.startswith('Bias_')]
    require(len(biases) == depth and all(rates[name] == 0 for name in biases), 'frozen-bias learning rates')


def epoch_history(run, epochs):
    rows = [r for r in read_lines(run / 'metrics.jsonl') if r.get('kind') == 'epoch']
    require([r['epoch'] for r in rows] == list(range(1, epochs + 1)), 'epoch sequence must be exactly 1..N')
    for row in rows:
        require(row['evaluation_split'] == 'validation', 'epoch split')
        for name in ('train_accuracy', 'train_loss', 'validation_accuracy', 'validation_loss'):
            value = row['metrics'][name]
            require(isinstance(value, (int, float)) and math.isfinite(value), f'nonfinite {name} epoch {row["epoch"]}')
            require(not name.endswith('accuracy') or 0 <= value <= 1, 'accuracy outside [0,1]')
    return rows


def parameter_hash(model):
    digest = hashlib.sha256(b'mnist-train-ordered-parameter-state/v1\0')
    for entry, tensor in zip(model['schema'], model['states'], strict=True):
        for value in (entry['name'].strip(), str(tensor.dtype), json.dumps(list(tensor.shape))):
            digest.update(value.encode()); digest.update(b'\0')
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes(order='C'))
    return digest.hexdigest()


def validate_rng(saved, epoch, previous_draw_count):
    """Validate restorable global/private state, including lazy epoch-zero noise."""
    import numpy as np
    import torch
    rng, private = saved['rng_state'], saved['continuation_state']
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.Generator().set_state(rng['torch_cpu'])
    require(bool(rng['torch_cuda']), 'CUDA RNG state missing')
    for state in rng['torch_cuda']:
        require(torch.is_tensor(state) and state.dtype == torch.uint8 and state.numel() > 0, 'invalid CUDA RNG')
    for name in ('sampler_rng', 'worker_rng'):
        torch.Generator().set_state(private[name])
    require(isinstance(private['contract'], dict) and bool(private['contract']), 'RNG continuation contract missing')
    noise = private['endpoint_noise']
    require(isinstance(noise, dict), 'endpoint noise RNG missing')
    count = noise['draw_count']
    require(isinstance(count, int) and count >= 0, 'invalid noise draw count')
    if epoch == 0:
        require(count == 0, 'noise already advanced at initialization')
    else:
        require(count > previous_draw_count and bool(noise['generators']), 'noise RNG missing or reset')
    for name, state in noise['generators'].items():
        require(name.startswith('cuda') and torch.is_tensor(state) and state.dtype == torch.uint8
                and state.numel() > 0, 'invalid endpoint noise generator state')
    return count


def runtime_configuration(run, config, manifest, result):
    """Only the declared data-location override may alter the trainer config."""
    path = 'artifacts/config.used.json'
    require(any(item['path'] == path for item in result['artifacts']), 'used config not canonically hashed')
    used = read(run / path)
    expected = copy.deepcopy(config)
    root = manifest.get('transport_overrides', {}).get('dataset_root')
    command = manifest['command']
    if root is not None:
        require(command.count('--dataset-root') == 1, 'dataset-root override missing or repeated in command')
        index = command.index('--dataset-root')
        require(index + 1 < len(command) and command[index + 1] == root, 'dataset-root override differs from command')
        expected['datasets'][config['lab']['dataset_key']]['params']['root'] = root
    else:
        require('--dataset-root' not in command, 'undeclared dataset-root command override')
    require(used == expected, 'used config differs beyond declared dataset-root transport override')
    return used


def checkpoint_history(run, config, result, initial_hash, source_sha256, runtime_config):
    """Canonical hashes are checked before loading locally produced checkpoints."""
    import torch
    epochs = config['lab']['epochs']
    artifacts = {a['path']: a for a in result['artifacts']}
    require(len(artifacts) == len(result['artifacts']), 'duplicate artifact paths')
    index = 'artifacts/epoch_checkpoint_index.jsonl'
    require(index in artifacts, 'checkpoint index not canonically hashed')
    records = read_lines(run / index)
    require([r['epoch'] for r in records] == list(range(epochs + 1)), 'checkpoint epochs must be exactly 0..N')
    draw_count, first_contract = 0, None
    for record in records:
        epoch = record['epoch']
        for kind in ('model', 'optimizer'):
            name = f'checkpoints/epoch_{epoch:03d}_{kind}.pt'
            require(record[f'{kind}_path'] == name and name in artifacts, 'missing indexed epoch checkpoint')
            item = artifacts[name]
            require(item['sha256'] == record[f'{kind}_sha256'], 'checkpoint index hash mismatch')
            require((run / name).stat().st_size == item['size_bytes'] == record[f'{kind}_size_bytes'] > 0,
                    'checkpoint size mismatch')
        model = torch.load(run / record['model_path'], map_location='cpu', weights_only=True)
        names = [p['name'].strip() for p in model['schema']]
        require(names == config['parameter_order'], 'model parameter schema mismatch')
        require(len(model['states']) == len(names), 'model states mismatch')
        for name, state in zip(names, model['states'], strict=True):
            require(torch.isfinite(state).all().item(), f'nonfinite model {name} epoch {epoch}')
            require(not name.startswith('Bias_') or torch.count_nonzero(state).item() == 0,
                    f'bias not zero: {name} epoch {epoch}')
        if epoch == 0:
            require(parameter_hash(model) == initial_hash, 'actual initial parameter hash differs from metrics')
        saved = torch.load(run / record['optimizer_path'], map_location='cpu', weights_only=False)
        require(saved['epoch'] == epoch and saved['optimizer_class'] == 'Adam', 'optimizer epoch/class')
        require(saved['model_checkpoint_sha256'] == record['model_sha256'], 'optimizer/model linkage')
        require(saved['parameter_names'] == names, 'optimizer parameter order')
        groups = saved['optimizer_parameter_groups']
        observed_rates = {name: group['learning_rate'] for group in groups for name in group['parameter_names']}
        require(observed_rates == config['learning_rates_by_parameter'], 'checkpoint learning rates/frozen biases')
        state_dict = saved['optimizer_state_dict']
        require(len(state_dict['param_groups']) == len(groups), 'optimizer group count')
        for group, description in zip(state_dict['param_groups'], groups, strict=True):
            require(group['lr'] == description['learning_rate'], 'actual optimizer learning rates differ')
            ids = group['params']
            require(bool(ids), 'empty optimizer parameter group')
        states = state_dict['state']
        if epoch == 0:
            require(not states, 'Adam not fresh at epoch zero')
        else:
            require(bool(states) and set(states) == {p for g in state_dict['param_groups'] for p in g['params']},
                    'incomplete Adam states')
            for state in states.values():
                require(float(state['step']) == math.ceil(55000 / 16) * epoch, 'Adam steps indicate reset or shortened training')
                for name in ('exp_avg', 'exp_avg_sq'):
                    require(torch.isfinite(state[name]).all().item(), 'nonfinite Adam moments')
        draw_count = validate_rng(saved, epoch, draw_count)
        contract = saved['continuation_state']['contract']
        if first_contract is None:
            first_contract = contract
        require(contract == first_contract, 'RNG continuation contract changed')
        require(contract['source'] == source_sha256, 'RNG continuation source identity mismatch')
        require(contract['epochs'] == epochs and contract['max_batches'] is None and contract['max_test_batches'] is None,
                'RNG continuation budget mismatch')
        require(contract['config_sha256'] == hashlib.sha256(json.dumps(runtime_config, sort_keys=True).encode()).hexdigest(),
                'RNG continuation scientific config mismatch')
    return len(records)


def validate_complete(run, config, record, expected_source_sha256=None):
    problems = validate_run(run)
    require(not problems, 'canonical bundle: ' + '; '.join(problems))
    manifest, result, metrics = [read(run / name) for name in ('manifest.json', 'result.json', 'metrics.json')]
    require(read(run / 'status.json')['state'] == 'complete' and result['completion']['criteria_met'], 'completion criteria')
    require(not manifest.get('smoke') and not result.get('smoke'), 'smoke is not production')
    require(manifest['arm_id'] == record['arm_id'] and manifest['study_id'] == STUDY, 'manifest identifiers')
    source_sha256 = manifest['git']['source_archive_sha256']
    require(isinstance(source_sha256, str) and len(source_sha256) == 64, 'effective source archive hash missing')
    require(expected_source_sha256 is None or source_sha256 == expected_source_sha256, 'effective source archive hash mismatch')
    recorded = manifest['configuration']
    require(recorded['resolved'] == config and recorded['sha256'] == record['config_sha256'], 'scientific config mismatch')
    epochs = config['lab']['epochs']
    require(recorded['epochs'] == recorded['configured_epochs'] == epochs, 'manifest epochs')
    require(recorded['diagnostic_overrides']['checkpoint_every_epoch'] is True, 'per-epoch checkpoints disabled')
    require(str(manifest['runtime']['device']).startswith('cuda'), 'production not CUDA')
    # exact_run omits this redundant flag when the frozen config disables the
    # official test. Check the scientific policy and actual reads below instead.
    require(config['evaluation']['official_test']['policy'] == 'disabled', 'official-test config policy')
    require(not any(flag in manifest['command'] for flag in ('--max-batches', '--max-test-batches')), 'shortened invocation')
    for owner in (manifest, result):
        data = owner['dataset']
        require(data['evaluation_split'] == 'validation' and data['official_test_policy'] == 'disabled'
                and data['official_test_read'] is False and data['official_test_read_during_training'] is False,
                'official-test access or dataset mismatch')
    require(metrics['seed'] == record['seed'] and metrics['official_test_evaluations'] == metrics['official_test_examples'] == 0,
            'seed or official-test runtime mismatch')
    require(metrics['training_algorithm'] == 'EP' and metrics['runtime_dtype'] == 'float64', 'runtime algorithm/dtype')
    require(metrics['final_learning_rate'] == config['lr'], 'runtime learning rates')
    for name in ('endpoint_read_noise_eta', 'endpoint_read_noise_std', 'endpoint_read_noise_mode', 'endpoint_read_noise_seed'):
        require(metrics['eqprop'][name] == config['eqprop'][name], 'runtime noise mismatch: ' + name)
    provenance = metrics['dataset_provenance']
    require(provenance['source_split'] == 'train' and provenance['split_seed'] == 0
            and provenance['shuffle_seed'] == record['seed'], 'split or shuffle seed mismatch')
    require(len(provenance['train_batch_order_sha256']) == epochs, 'incomplete minibatch history')
    history = epoch_history(run, epochs)
    accuracy = [r['metrics']['validation_accuracy'] for r in history]
    for which, value in (('final', accuracy[-1]), ('best', max(accuracy))):
        require(math.isclose(result['terminal_metrics']['validation'][which + '_accuracy'], value, rel_tol=0, abs_tol=1e-12)
                and math.isclose(metrics[which + '_validation_accuracy'], value, rel_tol=0, abs_tol=1e-12),
                'terminal/' + which + ' epoch mismatch')
    runtime_config = runtime_configuration(run, config, manifest, result)
    checkpoints = checkpoint_history(run, config, result, metrics['initial_parameter_state_sha256'], source_sha256,
                                     runtime_config)
    return dict(source_archive_sha256=source_sha256, final_accuracy_pct=100 * accuracy[-1], best_accuracy_pct=100 * max(accuracy),
                best_epoch=accuracy.index(max(accuracy)) + 1, checkpoints=checkpoints,
                initial_parameter_state_sha256=metrics['initial_parameter_state_sha256'],
                initializer_checkpoint_sha256=config['initialization']['checkpoint_sha256'],
                train_indices_sha256=provenance['train_indices_sha256'],
                validation_indices_sha256=provenance['validation_indices_sha256'],
                train_batch_order_sha256=provenance['train_batch_order_sha256'],
                result_sha256=sha256_file(run / 'result.json'))


def matched_identity(rows, seeds=SEEDS):
    errors = []
    for depth in HORIZONS:
        for seed in seeds:
            group = [r for r in rows if r['depth'] == depth and r['seed'] == seed and r['state'] == 'complete']
            for field in ('initial_parameter_state_sha256', 'initializer_checkpoint_sha256',
                          'train_indices_sha256', 'validation_indices_sha256', 'train_batch_order_sha256'):
                if len({json.dumps(r[field], sort_keys=True) for r in group}) > 1:
                    issue = f'Conv{depth} seed{seed}: unmatched {field}'
                    errors.append(issue)
                    for row in group:
                        row.update(state='invalid', issue=issue)
    return errors


def describe(values):
    present = [value for value in values.values() if value is not None]
    return dict(values_by_seed=values, n=len(present), expected_n=len(values), complete=len(present) == len(values),
                mean=statistics.mean(present) if present else None,
                sample_sd=statistics.stdev(present) if len(present) > 1 else None)


def aggregate(rows):
    lookup = {key(r): r for r in rows}
    groups, pairs = [], []
    def accuracy(depth, scheme, eta, seed, metric='final_accuracy_pct'):
        row = lookup.get((depth, scheme, eta, seed), {})
        return row.get(metric) if row.get('state') == 'complete' else None
    for depth in HORIZONS:
        for eta in ETAS:
            for scheme in SCHEMES:
                groups.append(dict(depth=depth, eta=eta, scheme=scheme, epoch=HORIZONS[depth],
                    final_accuracy_pct=describe({str(seed): accuracy(depth, scheme, eta, seed) for seed in SEEDS}),
                    best_accuracy_pct_secondary=describe({str(seed): accuracy(depth, scheme, eta, seed, 'best_accuracy_pct') for seed in SEEDS})))
            for comparator in ('baseline', 'legacy'):
                values = {}
                for seed in SEEDS:
                    ours, other = accuracy(depth, 'ours', eta, seed), accuracy(depth, comparator, eta, seed)
                    values[str(seed)] = ours - other if ours is not None and other is not None else None
                pairs.append(dict(depth=depth, eta=eta, comparator=comparator, epoch=HORIZONS[depth],
                                  ours_minus_comparator_pp=describe(values)))
    return groups, pairs


def write_plots(out, groups, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename, records, label_key, values_key, labels, ylabel in (
        ('final_accuracy.jpg', groups, 'scheme', 'final_accuracy_pct', SCHEMES, 'Final validation accuracy (%)'),
        ('paired_gaps.jpg', pairs, 'comparator', 'ours_minus_comparator_pp', ('baseline', 'legacy'), 'Ours − comparator (pp)'),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), squeeze=False)
        for depth, ax in zip(HORIZONS, axes[0]):
            for label in labels:
                cells = [next(r for r in records if r['depth'] == depth and r['eta'] == eta and r[label_key] == label)[values_key] for eta in ETAS]
                # Draw means/SD only when both declared seeds are valid; individuals remain visible.
                line, = ax.plot(range(4), [c['mean'] if c['complete'] else float('nan') for c in cells], '-o', label=label)
                for i, cell in enumerate(cells):
                    for seed, value in cell['values_by_seed'].items():
                        if value is not None:
                            ax.scatter(i + (-.04 if seed == '1' else .04), value, color=line.get_color(), marker='x', s=22)
                    if cell['complete']:
                        ax.errorbar(i, cell['mean'], yerr=cell['sample_sd'], color=line.get_color(), capsize=3)
            ax.set(title=f'Conv{depth} · epoch {HORIZONS[depth]}', xticks=range(4),
                   xticklabels=['1e−5', '1e−4', '3e−4', '5e−4'], xlabel='Relative read noise η')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
            if filename == 'paired_gaps.jpg':
                ax.axhline(0, color='black', linewidth=.6)
        axes[0, 0].set_ylabel(ylabel)
        fig.suptitle('Seeds 1/2 · mean ± sample SD (n=2) · × individual seeds; missing outcomes blank')
        fig.tight_layout(); fig.savefig(out / filename, dpi=180); plt.close(fig)


# These exact historical sources are the approved supplementary inclusion set.
SEED0_SOURCES = {
    'exp014': 'conv123-relative-noise-training-20260925-v1',
    'exp017': 'conv23-baseline-relative-noise-20260927-v1',
    'exp018': 'conv1-clean-and-baseline-noise-20260927-v1',
    'exp019': 'conv123-relative-noise-eta5em4-jeanzay-20260928-v1',
}


def seed0_source(depth, scheme, eta):
    if eta == 5e-4:
        return 'exp019'
    if depth == 1:
        return 'exp018' if scheme == 'baseline' else 'exp014'
    return 'exp017' if scheme == 'baseline' else None


def seed_independent_config(config):
    """Same numerical comparison, changing only intended replicate seed inputs."""
    from experiments.analyze_current_amp2 import scientific_config
    projected = scientific_config(config)
    projected['seed'] = 0
    projected['initialization']['seed'] = 0
    projected['initialization']['checkpoint_sha256'] = '<seed-specific-initializer>'
    params = projected['datasets'][config['lab']['dataset_key']]['params']
    params['shuffle_seed'] = 0
    projected['eqprop']['endpoint_read_noise_seed'] = '<seed-specific-acquisition>'
    return projected


def validate_seed0(run, expected, source):
    from experiments.analyze_current_amp2 import optimizer_history
    problems = validate_run(run)
    require(not problems, 'historical canonical bundle: ' + '; '.join(problems))
    manifest, result, metrics = [read(run / name) for name in ('manifest.json', 'result.json', 'metrics.json')]
    config = manifest['configuration']['resolved']
    require(manifest['study_id'] == SEED0_SOURCES[source], 'historical study mismatch')
    require(read(run / 'status.json')['state'] == 'complete' and result['completion']['criteria_met'], 'historical completion')
    require(not manifest.get('smoke') and not result.get('smoke'), 'historical smoke')
    require(config['seed'] == metrics['seed'] == 0, 'historical seed is not zero')
    require(seed_independent_config(config) == seed_independent_config(expected), 'historical numerical contract differs')
    require(config['relative_noise_training'].get('optimizer_reset_at_epoch10', False) is False,
            'historical optimizer reset')
    require(config['initialization']['optimizer_state'] in ('fresh', 'fresh_at_initialization'), 'historical warm restart')
    require(not (run / 'diagnostic_resume.json').exists() and not (run / 'continuation.json').exists(),
            'historical recovery needs separate compatibility review')
    record = read(run / 'exact_run.json')
    require('continued_from' not in record, 'historical continuation is not fresh training')
    for owner in (manifest, result):
        data = owner['dataset']
        require(data['evaluation_split'] == 'validation' and data['official_test_read'] is False
                and data['official_test_read_during_training'] is False, 'historical official-test access')
    require(metrics['official_test_evaluations'] == metrics['official_test_examples'] == 0, 'historical test count')
    require(metrics['final_learning_rate'] == config['lr'], 'historical LR mismatch')
    epochs = config['lab']['epochs']
    history = epoch_history(run, epochs)
    accuracy = [row['metrics']['validation_accuracy'] for row in history]
    for which, value in (('final', accuracy[-1]), ('best', max(accuracy))):
        require(math.isclose(result['terminal_metrics']['validation'][which + '_accuracy'], value, rel_tol=0, abs_tol=1e-12),
                'historical terminal metric mismatch')
    # Older runs lack private RNG checkpoints. Preserve that weaker historical
    # evidence explicitly; never apply its reduced evidence contract to exp022.
    history_quality = optimizer_history(run, config, epochs)
    import torch
    paths = [run / 'checkpoints/best_model.pt', run / 'checkpoints/final_model.pt']
    if (run / 'epoch_checkpoint_index.jsonl').exists():
        index = read_lines(run / 'epoch_checkpoint_index.jsonl')
        require([r['epoch'] for r in index] == list(range(epochs + 1)), 'historical checkpoint epochs')
        paths.extend(run / row['model_path'] for row in index)
    for path in paths:
        model = torch.load(path, map_location='cpu', weights_only=True)
        names = [entry['name'].strip() for entry in model['schema']]
        require(names == config['parameter_order'], 'historical parameter schema')
        for name, tensor in zip(names, model['states'], strict=True):
            require(torch.isfinite(tensor).all().item(), 'historical nonfinite parameters')
            require(not name.startswith('Bias_') or torch.count_nonzero(tensor).item() == 0, 'historical nonzero bias')
    provenance = metrics['dataset_provenance']
    require(provenance['source_split'] == 'train' and provenance['split_seed'] == provenance['shuffle_seed'] == 0,
            'historical dataset split or minibatch seed')
    require(len(provenance['train_batch_order_sha256']) == epochs, 'historical minibatch history')
    return dict(state='complete', final_accuracy_pct=100 * accuracy[-1], best_accuracy_pct=100 * max(accuracy),
                best_epoch=accuracy.index(max(accuracy)) + 1,
                initial_parameter_state_sha256=metrics['initial_parameter_state_sha256'],
                initializer_checkpoint_sha256=config['initialization']['checkpoint_sha256'],
                train_indices_sha256=provenance['train_indices_sha256'],
                validation_indices_sha256=provenance['validation_indices_sha256'],
                train_batch_order_sha256=provenance['train_batch_order_sha256'],
                result_sha256=sha256_file(run / 'result.json'), config_sha256=manifest['configuration']['sha256'],
                source_identity=manifest.get('git'), host=manifest['runtime']['host'],
                optimizer_history=history_quality,
                rng_evidence='Historical uninterrupted provenance; full private RNG coverage not available.')


def supplement_statistics(primary, historical):
    lookup = {key(row): row for row in [*primary, *historical]}
    groups, pairs = [], []
    def value(depth, scheme, eta, seed):
        row = lookup.get((depth, scheme, eta, seed), {})
        return row.get('final_accuracy_pct') if row.get('state') == 'complete' else None
    for depth in HORIZONS:
        for eta in ETAS:
            for scheme in SCHEMES:
                if seed0_source(depth, scheme, eta) is not None:
                    groups.append(dict(depth=depth, scheme=scheme, eta=eta,
                        final_accuracy_pct=describe({str(seed): value(depth, scheme, eta, seed) for seed in (0, 1, 2)})))
            for comparator in ('baseline', 'legacy'):
                # Use exactly the common eligible seed set; this prevents an
                # n=3 baseline being compared with n=2 ours by unpaired means.
                common = (0, 1, 2) if all(seed0_source(depth, scheme, eta) is not None
                                         for scheme in ('ours', comparator)) else (1, 2)
                values = {}
                for seed in common:
                    ours, other = value(depth, 'ours', eta, seed), value(depth, comparator, eta, seed)
                    values[str(seed)] = ours - other if ours is not None and other is not None else None
                pairs.append(dict(depth=depth, eta=eta, comparator=comparator, common_seeds=list(common),
                                  ours_minus_comparator_pp=describe(values)))
    return groups, pairs


def seed0_supplement(root, records, primary):
    expected = {(d, s, eta): record for record in records for d, s, eta, seed in [key(record)] if seed == 1}
    paths, discovery_errors = {}, []
    for source, name in SEED0_SOURCES.items():
        table = root.parent / name / 'analysis' / ('coverage.csv' if source == 'exp014' else 'accuracy.csv')
        try:
            with table.open() as handle:
                entries = list(csv.DictReader(handle))
            for entry in entries:
                run = Path(entry.get('run') or entry.get('path') or entry['run_path'])
                if not run.is_absolute():
                    run = REPO / run
                manifest = read(run / 'manifest.json'); config = manifest['configuration']['resolved']
                arm = manifest['arm_id']; depth, scheme = int(arm[4]), arm.split('_')[1]
                eta = float(config['eqprop']['endpoint_read_noise_eta'])
                if (depth, scheme, eta) not in expected or config['seed'] != 0:
                    continue
                if config['relative_output_calibration']['target'] != TARGETS[depth] or seed0_source(depth, scheme, eta) != source:
                    continue
                cell = depth, scheme, eta
                require(cell not in paths, 'duplicate historical cell; explicit reconciliation required')
                paths[cell] = run, source
        except (OSError, ValueError, KeyError, TypeError) as error:
            discovery_errors.append(f'{source}: {error}')
    rows, errors = [], list(discovery_errors)
    for cell, record in expected.items():
        depth, scheme, eta = cell
        source = seed0_source(depth, scheme, eta)
        if source is None:
            continue
        row = dict(depth=depth, scheme=scheme, eta=eta, seed=0, source=source, state='missing', issue='', run='')
        try:
            require(cell in paths, 'approved historical bundle missing')
            run, source = paths[cell]; row['run'] = str(run)
            row.update(validate_seed0(run, read(config_path(root, record)), source))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            row.update(state='invalid', issue=str(error)); errors.append(f'{cell}: {error}')
        rows.append(row)
    errors.extend(matched_identity(rows, seeds=(0,)))
    groups, pairs = supplement_statistics(primary, rows)
    return dict(included=True, expected_cases=24, validated_cases=sum(r['state'] == 'complete' for r in rows),
                complete=not errors and len(rows) == 24, errors=errors, outcomes=rows,
                aggregates=groups, paired_differences=pairs,
                qualification='Separate historical supplement. Source/hardware differ; no full historical private-RNG audit. '
                'Only designated fresh-initialization sources are eligible. Conv2/3 ours/legacy eta<5e-4 seed0 reset runs are excluded. '
                'Paired differences use common eligible seeds; primary seeds1/2 remain unchanged.')


def supplementary_report(supplement):
    def fmt(value):
        return '—' if value is None else f'{value:.3f}'
    lines = ['', '## Compatible historical seed-0 supplement', '', supplement['qualification'], '',
             f'Historical validated cases: {supplement["validated_cases"]}/24. See summary.json for hashes, source identity and RNG evidence.', '',
             '| Model | η | Scheme | Seed 0 (%) | Seed 1 (%) | Seed 2 (%) | Mean (%) | Sample SD | n |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in supplement['aggregates']:
        stats = row['final_accuracy_pct']; values = stats['values_by_seed']
        lines.append(f'| Conv{row["depth"]} | {row["eta"]:g} | {row["scheme"]} | {fmt(values["0"])} | {fmt(values["1"])} | {fmt(values["2"])} | {fmt(stats["mean"])} | {fmt(stats["sample_sd"])} | {stats["n"]} |')
    lines += ['', '| Model | η | Ours minus | Common eligible seeds | Mean gap (pp) | Sample SD | n |',
              '|---|---:|---|---|---:|---:|---:|']
    for row in supplement['paired_differences']:
        stats = row['ours_minus_comparator_pp']
        lines.append(f'| Conv{row["depth"]} | {row["eta"]:g} | {row["comparator"]} | {row["common_seeds"]} | {fmt(stats["mean"])} | {fmt(stats["sample_sd"])} | {stats["n"]} |')
    lines += ['', 'Supplementary validation errors: ' + ('; '.join(supplement['errors']) or 'none')]
    return lines


def analyze(root):
    root = Path(root).resolve()
    source_identity_path = root / 'transport/source-identity.json'
    source_sha256 = read(source_identity_path)['sha256'] if source_identity_path.exists() else None
    preparation = read(root / 'preparation.json')
    records = preparation['records'] if isinstance(preparation, dict) else preparation
    by_key, errors = {}, []
    for record in records:
        cell = key(record)
        require(cell not in by_key, f'duplicate preparation cell: {cell}')
        by_key[cell] = record
    require(set(by_key) == set(expected_keys()), 'preparation must declare exactly all 72 seed-1/2 cells')
    require(len({r['arm_id'] for r in records}) == 72, 'duplicate preparation arm IDs')
    expected_arms = {r['arm_id'] for r in records}
    production = root / 'collected/production'
    if production.exists():
        extras = sorted(p.name for p in production.iterdir() if p.is_dir() and p.name not in expected_arms)
        if extras:
            errors.append('undeclared collected arms: ' + ', '.join(extras))
    rows = []
    for cell in expected_keys():
        record = by_key[cell]
        depth, scheme, eta, seed = cell
        row = dict(depth=depth, scheme=scheme, eta=eta, seed=seed, arm_id=record['arm_id'],
                   epochs=record['epochs'], state='not_collected', issue='', run='',
                   final_accuracy_pct=None, best_accuracy_pct=None)
        try:
            path = config_path(root, record)
            require(sha256_file(path) == record['config_sha256'], 'prepared config hash mismatch')
            config = read(path); validate_config(config, record)
            candidates = sorted(p for p in (production / record['arm_id'] / 'runs').glob('*') if p.is_dir())
            require(len(candidates) <= 1, 'multiple attempts require explicit reconciliation; never select by accuracy')
            if candidates:
                run = candidates[0]; row['run'] = str(run)
                require(run.name.startswith('000_'), 'unexpected run index')
                status = read(run / 'status.json'); row['state'] = status['state']
                require(row['state'] in ('running', 'paused', 'failed', 'complete'), 'unknown bundle state')
                if row['state'] == 'complete':
                    require(source_sha256 is not None, 'frozen transport/source-identity.json missing')
                    row.update(validate_complete(run, config, record, source_sha256))
                else:
                    row['issue'] = json.dumps(status.get('error') or status.get('pause') or {}, sort_keys=True)
                    problems = validate_run(run)
                    require(not problems, 'canonical incomplete/failed bundle: ' + '; '.join(problems))
                    manifest = read(run / 'manifest.json')
                    require(manifest['arm_id'] == record['arm_id'] and manifest['study_id'] == STUDY
                            and manifest['configuration']['resolved'] == config
                            and manifest['configuration']['sha256'] == record['config_sha256'], 'incomplete/failed config mismatch')
        except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as error:
            row.update(state='invalid', issue=str(error)); errors.append(f'{record["arm_id"]}: {error}')
        rows.append(row)
    errors.extend(matched_identity(rows))
    groups, pairs = aggregate(rows)
    supplementary = seed0_supplement(root, records, rows)
    states = {state: sum(r['state'] == state for r in rows) for state in sorted({r['state'] for r in rows})}
    complete = states.get('complete', 0) == 72 and not errors
    summary = dict(study_id=STUDY, complete=complete, expected_cases=72,
                   validated_cases=states.get('complete', 0), states=states, errors=errors,
                   ready_for_review=not errors and all(r['state'] in ('complete', 'failed') for r in rows),
                   qualification=QUALIFICATION, outcomes=rows, aggregates=groups, paired_differences=pairs,
                   supplementary_seed0=supplementary)
    out = root / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    columns = ('depth', 'scheme', 'eta', 'seed', 'arm_id', 'epochs', 'state', 'final_accuracy_pct',
               'best_accuracy_pct', 'best_epoch', 'result_sha256', 'run', 'issue')
    with (out / 'coverage.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    write_plots(out, groups, pairs)
    def fmt(value):
        return '—' if value is None else f'{value:.3f}'
    lines = ['# Exp022 fixed-final validation evidence', '', QUALIFICATION, '',
             f'Validated complete cases: {summary["validated_cases"]}/72. Study complete: {complete}. States: {states}.', '',
             'Incomplete groups report available values with n; only n=2 groups have a sample SD.', '',
             '| Model | η | Scheme | Seed 1 (%) | Seed 2 (%) | Mean (%) | Sample SD | n | Best mean (secondary) |',
             '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for group in groups:
        stats = group['final_accuracy_pct']; values = stats['values_by_seed']
        lines.append(f'| Conv{group["depth"]} | {group["eta"]:g} | {group["scheme"]} | {fmt(values["1"])} | {fmt(values["2"])} | {fmt(stats["mean"])} | {fmt(stats["sample_sd"])} | {stats["n"]} | {fmt(group["best_accuracy_pct_secondary"]["mean"])} |')
    lines += ['', '| Model | η | Ours minus | Seed 1 (pp) | Seed 2 (pp) | Mean (pp) | Sample SD | n |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    for pair in pairs:
        stats = pair['ours_minus_comparator_pp']; values = stats['values_by_seed']
        lines.append(f'| Conv{pair["depth"]} | {pair["eta"]:g} | {pair["comparator"]} | {fmt(values["1"])} | {fmt(values["2"])} | {fmt(stats["mean"])} | {fmt(stats["sample_sd"])} | {stats["n"]} |')
    lines += ['', '[Final accuracy](final_accuracy.jpg) · [Paired gaps](paired_gaps.jpg) · [All case coverage](coverage.csv)', '',
              'Exceptions and unfinished cases:', '']
    lines += [f'- {r["arm_id"]}: {r["state"]}; {r["issue"]}' for r in rows if r['state'] != 'complete'] or ['- None.']
    lines += supplementary_report(supplementary)
    lines += ['', 'Validation errors: ' + ('; '.join(errors) or 'none'), '',
              'Scientific interpretation remains with the campaign Reviewer.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=REPO / 'results' / STUDY)
    parser.add_argument('--allow-partial', action='store_true', help='Permit missing/running/paused cases, never failed or invalid cases.')
    args = parser.parse_args(argv)
    try:
        summary = analyze(args.study_root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'complete': False, 'error': str(error)})); return 1
    print(json.dumps({k: summary[k] for k in ('complete', 'expected_cases', 'validated_cases', 'states', 'errors')}))
    return 0 if summary['complete'] or (args.allow_partial and not summary['errors']
                                      and not summary['states'].get('failed')) else 1


if __name__ == '__main__':
    raise SystemExit(main())
