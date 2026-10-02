"""Collect the 48 rescaled-target EqProp cases without selecting successful arms.

Use --config-root for the frozen JSON configs and --production-root for collected
exact-run bundles. Missing and failed cells remain in coverage; --allow-partial
permits an interim report but never marks an incomplete study complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

from experiments.analyze_relative_noise_seeds import epoch_history, runtime_configuration
from experiments.reporting import sha256_file, validate_run

STUDY = 'conv123-rescaled-target-relative-noise-jeanzay-20261001-v1'
SCHEMES = ('baseline', 'ours', 'legacy', 'amp2')
AMPLIFICATION = dict(baseline=(1, 1), ours=(4, 1), legacy=(4, .25), amp2=(4, 2))
TARGETS = {1: .25, 2: .0625, 3: .015625}
HORIZONS = {1: 10, 2: 30, 3: 30}
ETAS = (0., 1e-6, 1e-5, 1e-4)
QUALIFICATION = ('Ordinary MNIST validation only, seed 0. Fixed final epoch 10/30/30 '
                 'is primary; best validation accuracy is secondary. Single-seed '
                 'differences are descriptive. Missing/failed outcomes are never imputed.')


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_keys():
    return list(itertools.product(HORIZONS, SCHEMES, ETAS))


def config_key(config):
    """Identify science from the model and noise settings, independent of arm spelling."""
    model = config['model_overrides'][config['lab']['model_key']]
    depth = len(model['conv_pipeline'])
    amps = tuple(config['model_base'][k] for k in ('voltage_amp', 'current_amp'))
    scheme = next((s for s, a in AMPLIFICATION.items() if a == amps), None)
    return depth, scheme, float(config['eqprop']['endpoint_read_noise_eta'])


def validate_config(config):
    depth, scheme, eta = config_key(config)
    require((depth, scheme, eta) in expected_keys(), 'undeclared depth/amplification/noise case')
    require(config['study_id'] == STUDY and bool(config['arm_id']), 'study/arm identity')
    require(config['target_scale'] == TARGETS[depth], 'target scale mismatch')
    require(config['seed'] == 0 and config['lab']['epochs'] == HORIZONS[depth], 'seed/horizon mismatch')
    require(config['training_algorithm'] == 'EP' and config['runtime_dtype'] == 'float64', 'algorithm/dtype mismatch')
    require(config['max_batches'] is None and config['max_test_batches'] is None, 'shortened data budget')
    require(config['optimizer']['name'] == 'Adam' and config['optimizer']['learning_rate'] == config['lr'], 'optimizer/rates mismatch')
    rates = config['lr']
    require(len(rates) == 2 * depth + 1 and all(math.isfinite(v) and v > 0 for v in rates[:depth + 1])
            and rates[depth + 1:] == [0.] * depth, 'weight rates/frozen bias rates')
    if 'learning_rates_by_parameter' in config:
        require(dict(zip(config['parameter_order'], rates, strict=True)) == config['learning_rates_by_parameter'], 'named rates mismatch')
    params = config['datasets'][config['lab']['dataset_key']]['params']
    require(params['batch_size'] == 16 and params['split_seed'] == params['shuffle_seed'] == 0,
            'batch/split/shuffle contract')
    noise = config['eqprop']
    require(noise['endpoint_read_noise_mode'] == 'nodewise_relative' and noise['endpoint_read_noise_std'] == 0,
            'relative noise mode/std mismatch')
    require(noise['input_read_noise'] is False and noise['validation_read_noise'] is False, 'noise scope mismatch')
    require(noise['variant'] == 'centered' and noise['nudging_mode'] == 'current', 'EqProp variant/nudging mismatch')
    require(math.isfinite(config['beta']) and config['beta'] > 0, 'invalid beta')
    if 'injected_beta_B' in noise:
        a, b = AMPLIFICATION[scheme]
        require(math.isclose(config['beta'] * (a / b) ** depth, noise['injected_beta_B'], rel_tol=1e-12), 'beta convention mismatch')
    evaluation = config['evaluation']
    require(evaluation['official_test']['policy'] == 'disabled' and evaluation['epoch_split'] == 'validation', 'official-test policy/split')
    require(evaluation['checkpoint_selection'] == 'maximum_validation_accuracy', 'checkpoint selection rule')
    return depth, scheme, eta


def load_configs(config_root):
    configs, errors = {}, []
    for path in sorted(Path(config_root).glob('*.json')):
        try:
            config = read(path)
            key = validate_config(config)
            require(key not in configs, f'duplicate case {key}')
            require(config['arm_id'] not in [c['arm_id'] for _, c in configs.values()], 'duplicate arm id')
            configs[key] = path, config
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(f'{path}: {error}')
    return configs, errors


def discover(roots, excluded):
    attempts, ignored, errors, seen = {}, [], [], set()
    for root in roots:
        for path in sorted(Path(root).rglob('manifest.json')):
            run = path.parent.resolve()
            if run in seen:
                continue
            seen.add(run)
            try:
                manifest = read(path)
                if manifest.get('smoke') or manifest.get('study_id') != STUDY:
                    ignored.append(str(run))
                    continue
                key = config_key(manifest['configuration']['resolved'])
                require(key in expected_keys(), 'unexpected production cell')
                status = read(run / 'status.json') if (run / 'status.json').exists() else {'state': 'missing_status'}
                identity = sha256_file(path)
                # Only terminal, identical bundles can be duplicate collected copies.
                identity += ':' + (sha256_file(run / 'result.json') if (run / 'result.json').exists() else str(run))
                attempts.setdefault(key, []).append(dict(run=str(run), state=status['state'],
                    issue=status.get('error', ''), excluded=run in excluded, identity=identity))
            except (ValueError, KeyError, TypeError, OSError) as error:
                errors.append(f'{run}: {error}')
    return attempts, ignored, errors


def checkpoint_evidence(run, config, used, manifest, result):
    """Check complete checkpoint coverage and the final restorable Adam/RNG state."""
    import numpy as np
    import torch
    from experiments.resume_diagnostic_epoch import _optimizer_payload

    artifacts = {item['path']: item for item in result['artifacts']}
    require(len(artifacts) == len(result['artifacts']), 'duplicate canonical artifact paths')
    index = 'artifacts/epoch_checkpoint_index.jsonl'
    require(index in artifacts, 'missing canonical epoch checkpoint index')
    records = [json.loads(line) for line in (run / index).read_text().splitlines() if line.strip()]
    horizon = config['lab']['epochs']
    epochs = [row['epoch'] for row in records]
    require(epochs in (list(range(horizon + 1)), list(range(1, horizon + 1))),
            'checkpoint index must cover unique ordered epochs 0..N or 1..N')
    for record in records:
        for kind in ('model', 'optimizer'):
            name = record[kind + '_path']
            require(name in artifacts, 'indexed checkpoint not canonically hashed')
            item = artifacts[name]
            require(item['sha256'] == record[kind + '_sha256']
                    and item['size_bytes'] == record[kind + '_size_bytes'] > 0,
                    'checkpoint index hash/size differs from canonical evidence')
    saved, tensors = _optimizer_payload(run.resolve(), records[-1], config)
    for name, tensor in zip(config['parameter_order'], tensors, strict=True):
        require(bool(torch.isfinite(tensor).all()), f'nonfinite final model parameter: {name}')
        require(not name.startswith('Bias_') or torch.count_nonzero(tensor).item() == 0,
                f'nonzero final bias: {name}')
    rng = saved['rng_state']
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.Generator().set_state(rng['torch_cpu'])
    require(bool(rng['torch_cuda']), 'missing global CUDA RNG state')
    for state in rng['torch_cuda']:
        require(torch.is_tensor(state) and state.dtype == torch.uint8 and state.numel() > 0,
                'invalid global CUDA RNG state')
    private = saved.get('continuation_state')
    require(isinstance(private, dict), 'missing private RNG continuation state')
    for name in ('sampler_rng', 'worker_rng'):
        require(name in private, f'missing private RNG: {name}')
        torch.Generator().set_state(private[name])
    contract = private['contract']
    require(contract['epochs'] == horizon and contract['max_batches'] is None
            and contract['max_test_batches'] is None, 'continuation budget changed')
    require(contract['source'] == manifest['git']['source_archive_sha256'], 'continuation source changed')
    require(contract['config_sha256'] == hashlib.sha256(json.dumps(used, sort_keys=True).encode()).hexdigest(),
            'continuation scientific config changed')
    noise = private['endpoint_noise']
    require(type(noise['draw_count']) is int and noise['draw_count'] >= 0, 'invalid private noise draw count')
    require(isinstance(noise['generators'], dict), 'missing private noise generators')
    if config['eqprop']['endpoint_read_noise_eta'] > 0:
        require(noise['draw_count'] > 0 and bool(noise['generators']), 'missing private noise RNG state')
    for device, state in noise['generators'].items():
        require(device.startswith('cuda') and torch.is_tensor(state) and state.dtype == torch.uint8
                and state.numel() > 0, 'invalid private noise RNG state')
    recoveries = [owner['diagnostic_resume'] for owner in (manifest, saved) if 'diagnostic_resume' in owner]
    receipt = run / 'diagnostic_resume.json'
    if receipt.exists():
        require('artifacts/diagnostic_resume.json' in artifacts, 'unhashed diagnostic resume receipt')
        recoveries.append(read(receipt))
    for recovery in recoveries:
        require(recovery['continuation_quality'] == 'exact_epoch_state_continuation'
                and recovery['rng_missing'] == [] and recovery['rng_reconstructed'] == []
                and recovery['optimizer_state_restored'] is True
                and recovery['uninterrupted_trajectory'] is True,
                'diagnostic resume is not exact full-state continuation')
    return len(records)


def validate_complete(run, config):
    errors = validate_run(run)
    require(not errors, f'invalid reporting bundle: {errors}')
    manifest, status, result, metrics = (read(run / name) for name in
        ('manifest.json', 'status.json', 'result.json', 'metrics.json'))
    require(status['state'] == 'complete' and result['completion']['criteria_met'], 'completion not met')
    require(not manifest.get('smoke') and not result.get('smoke'), 'smoke bundle')
    require(manifest['study_id'] == result['study_id'] == STUDY and manifest['arm_id'] == config['arm_id'], 'bundle identity mismatch')
    require(manifest['configuration']['resolved'] == config, 'resolved scientific config changed')
    require(manifest['runtime']['device'] == 'cuda', 'non-CUDA execution')
    used = runtime_configuration(run, config, manifest, result)
    for owner in (manifest, result):
        require(owner['dataset']['official_test_read'] is False
                and owner['dataset']['evaluation_split'] == 'validation', 'official test access/wrong split')
        require(owner['dataset'].get('official_test_read_during_training', False) is False, 'official test during training')
    require(metrics['official_test_evaluations'] == metrics['official_test_examples'] == 0, 'official test counts')
    require(metrics['seed'] == 0 and metrics['training_algorithm'] == 'EP'
            and metrics['runtime_dtype'] == 'float64', 'runtime seed/algorithm/dtype')
    require(metrics['final_learning_rate'] == config['lr'], 'runtime learning rates changed')
    # Runtime metric dictionaries may contain extra derived fields, but recorded
    # scientific values must agree with the frozen configuration.
    for name, value in metrics.get('eqprop', {}).items():
        if name in config['eqprop']:
            require(value == config['eqprop'][name], f'runtime EqProp mismatch: {name}')
    for name in ('target_scale', 'beta'):
        if name in metrics:
            require(metrics[name] == config[name], f'runtime {name} mismatch')
    metadata_path = run / 'run_metadata.json'
    if metadata_path.exists():
        metadata = read(metadata_path)
        for name in ('target_scale', 'beta', 'seed'):
            if name in metadata:
                require(metadata[name] == config[name], f'metadata {name} mismatch')
    history = epoch_history(run, config['lab']['epochs'])
    accuracies = [r['metrics']['validation_accuracy'] for r in history]
    for role, value in (('final', accuracies[-1]), ('best', max(accuracies))):
        require(math.isclose(metrics[role + '_validation_accuracy'], value, rel_tol=0, abs_tol=1e-12)
                and math.isclose(result['terminal_metrics']['validation'][role + '_accuracy'], value, rel_tol=0, abs_tol=1e-12),
                f'{role} terminal accuracy disagrees with epochs')
        require((run / (role + '_model.pt')).stat().st_size > 0, f'missing {role} checkpoint')
    require(metrics['best_epoch'] == accuracies.index(max(accuracies)) + 1, 'best epoch mismatch')
    provenance = metrics['dataset_provenance']
    require(provenance['source_split'] == 'train' and provenance['split_seed'] == provenance['shuffle_seed'] == 0, 'dataset provenance mismatch')
    require(len(provenance['train_batch_order_sha256']) == config['lab']['epochs'], 'minibatch history coverage')
    source = manifest['git'].get('source_archive_sha256')
    require(bool(source), 'missing frozen source identity')
    checkpoints = checkpoint_evidence(run, config, used, manifest, result)
    return dict(final_accuracy_pct=100 * accuracies[-1], best_accuracy_pct=100 * max(accuracies),
        checkpoints=checkpoints,
        best_epoch=metrics['best_epoch'], final_validation_loss=history[-1]['metrics']['validation_loss'],
        source_archive_sha256=source, train_indices_sha256=provenance['train_indices_sha256'],
        validation_indices_sha256=provenance['validation_indices_sha256'],
        train_batch_order_sha256=provenance['train_batch_order_sha256'],
        initial_parameter_state_sha256=metrics['initial_parameter_state_sha256'],
        host=manifest['runtime'].get('host', '')), history


def matching_issue(left, right):
    for name in ('source_archive_sha256', 'train_indices_sha256', 'validation_indices_sha256',
                 'train_batch_order_sha256', 'initial_parameter_state_sha256'):
        if not left.get(name) or left.get(name) != right.get(name):
            return f'unmatched {name}'
    return ''


def comparisons(rows):
    lookup = {(r['depth'], r['scheme'], r['eta']): r for r in rows}
    clean, pairs = [], []
    for row in rows:
        control = lookup[row['depth'], row['scheme'], 0.]
        issue = 'incomplete' if row['state'] != 'complete' or control['state'] != 'complete' else matching_issue(row, control)
        clean.append(dict(depth=row['depth'], scheme=row['scheme'], eta=row['eta'],
            state='complete' if not issue else issue, clean_minus_noisy_pp=None if issue else
            control['final_accuracy_pct'] - row['final_accuracy_pct']))
    for depth, eta in itertools.product(HORIZONS, ETAS):
        for first, second in itertools.combinations(SCHEMES, 2):
            left, right = lookup[depth, first, eta], lookup[depth, second, eta]
            issue = 'incomplete' if left['state'] != 'complete' or right['state'] != 'complete' else matching_issue(left, right)
            pairs.append(dict(depth=depth, eta=eta, first=first, second=second,
                state='complete' if not issue else issue, first_minus_second_pp=None if issue else
                left['final_accuracy_pct'] - right['final_accuracy_pct']))
    return clean, pairs


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def write_plots(out, rows, clean, pairs, histories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename, records, labels, label_key, metric, ylabel in (
        ('final_accuracy_vs_noise.jpg', rows, SCHEMES, 'scheme', 'final_accuracy_pct', 'Final validation accuracy (%)'),
        ('clean_degradation.jpg', clean, SCHEMES, 'scheme', 'clean_minus_noisy_pp', 'Clean − noisy accuracy (pp)'),
        ('scheme_gaps.jpg', pairs, list(itertools.combinations(SCHEMES, 2)), None, 'first_minus_second_pp', 'First − second accuracy (pp)')):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for depth, ax in zip(HORIZONS, axes):
            for label in labels:
                selected = [r for r in records if r['depth'] == depth and
                            (r[label_key] == label if label_key else (r['first'], r['second']) == label)]
                values = {r['eta']: r[metric] if r['state'] == 'complete' else None for r in selected}
                ax.plot(range(len(ETAS)), [values.get(e) if values.get(e) is not None else float('nan') for e in ETAS],
                        '-o', label=label if label_key else ' − '.join(label))
            ax.set(title=f'Conv{depth} · target {TARGETS[depth]:g} · epoch {HORIZONS[depth]}',
                   xticks=range(len(ETAS)), xticklabels=['0 (clean)', '1e−6', '1e−5', '1e−4'], xlabel='Relative read noise η (categorical)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
            if filename != 'final_accuracy_vs_noise.jpg':
                ax.axhline(0, color='black', linewidth=.6)
        axes[0].set_ylabel(ylabel)
        fig.suptitle('Seed 0 · validation only · missing/ineligible outcomes blank')
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
    for depth in HORIZONS:
        fig, axes = plt.subplots(1, 4, figsize=(17, 4), sharey=True)
        for eta, ax in zip(ETAS, axes):
            for scheme in SCHEMES:
                history = histories.get((depth, scheme, eta), [])
                if history:
                    ax.plot([r['epoch'] for r in history], [100 * r['metrics']['validation_accuracy'] for r in history], label=scheme)
            ax.set(title=f'η={eta:g}', xlabel='Epoch')
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=8)
        axes[0].set_ylabel('Validation accuracy (%)')
        fig.suptitle(f'Conv{depth} · target {TARGETS[depth]:g} · seed 0')
        fig.tight_layout()
        fig.savefig(out / f'conv{depth}_learning_curves.jpg', dpi=180)
        plt.close(fig)


def analyze(study_root, config_root, production_roots, *, exclude_runs=(), plots=True):
    configs, errors = load_configs(config_root)
    attempts, ignored, discovery_errors = discover(production_roots, {Path(p).resolve() for p in exclude_runs})
    errors.extend(discovery_errors)
    rows, histories = [], {}
    for key in expected_keys():
        depth, scheme, eta = key
        row = dict(depth=depth, scheme=scheme, eta=eta, seed=0, target_scale=TARGETS[depth],
                   final_epoch=HORIZONS[depth], arm_id='', state='missing_config', issue='', run='',
                   final_accuracy_pct=None, best_accuracy_pct=None, best_epoch=None, final_validation_loss=None)
        if key in configs:
            _, config = configs[key]
            row.update(arm_id=config['arm_id'], state='not_collected')
            candidates = {a['identity']: a for a in attempts.get(key, []) if not a['excluded']}
            if len(candidates) > 1:
                row.update(state='ambiguous', issue='Multiple attempts; explicitly exclude superseded runs by path')
            elif candidates:
                attempt = next(iter(candidates.values()))
                row.update(state=attempt['state'], run=attempt['run'], issue=attempt['issue'])
                if row['state'] == 'complete':
                    try:
                        values, history = validate_complete(Path(row['run']), config)
                        row.update(values)
                        histories[key] = history
                    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
                        row.update(state='invalid', issue=str(error))
        rows.append(row)
    clean, pairs = comparisons(rows)
    complete = (all(r['state'] == 'complete' for r in rows) and not errors
                and all(p['state'] == 'complete' for p in pairs + clean))
    summary = dict(study_id=STUDY, complete=complete, expected_runs=48,
        completed_runs=sum(r['state'] == 'complete' for r in rows), expected_pairs=72,
        completed_pairs=sum(p['state'] == 'complete' for p in pairs), qualification=QUALIFICATION,
        states={state: sum(r['state'] == state for r in rows) for state in sorted({r['state'] for r in rows})},
        errors=errors, ignored_bundles=ignored, attempts=[a for group in attempts.values() for a in group],
        coverage=rows, clean_degradation=clean, pairwise_differences=pairs)
    out = Path(study_root) / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    for name, records in (('coverage', rows), ('clean_degradation', clean), ('pairwise_differences', pairs)):
        write_csv(out / (name + '.csv'), records)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    if plots:
        write_plots(out, rows, clean, pairs, histories)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--config-root', type=Path)
    parser.add_argument('--production-root', type=Path, action='append')
    parser.add_argument('--exclude-run', type=Path, action='append', default=[])
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    config_root = args.config_root or args.study_root / 'source/configs/conv/rescaled_target_noise_20261001'
    roots = args.production_root or [args.study_root / 'collected/production', args.study_root / 'production']
    summary = analyze(args.study_root, config_root, roots, exclude_runs=args.exclude_run)
    print(json.dumps({k: summary[k] for k in ('complete', 'expected_runs', 'completed_runs', 'completed_pairs', 'states', 'errors')}))
    return 0 if summary['complete'] or args.allow_partial else 1


if __name__ == '__main__':
    raise SystemExit(main())
