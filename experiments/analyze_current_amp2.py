"""Reconcile exp020 fixed-horizon validation outcomes, including matched controls.

Usage: python -m experiments.analyze_current_amp2 --study-root RESULTS [--allow-partial]
Only canonical production bundles count. Repeated collected copies are deduplicated
by immutable manifest and result identity; distinct attempts are never selected by
accuracy. Historical controls must match the prepared comparator configuration.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

from experiments.reporting import sha256_file, validate_run

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURES = ('conv1', 'conv2', 'conv3')
SCHEMES = ('amp2', 'ours', 'legacy')
ETAS = (0., 1e-5, 1e-4)
HORIZONS = {'conv1': 10, 'conv2': 30, 'conv3': 30}
TARGETS = {'conv1': 1., 'conv2': 4., 'conv3': 6.}
CONFIG_SUBDIR = 'current_amp2_20260928'
HISTORICAL = {
    'conv1_clean': 'conv1-clean-and-baseline-noise-20260927-v1',
    'conv23_clean': 'conv23-noiseless-three-seed-20260927-v1',
    'conv1_noisy': 'conv123-relative-noise-training-20260925-v1',
}
# Explicit executable fields; descriptive calibration proofs, study IDs and
# handoff prose vary across studies and are deliberately not compared.
SCIENTIFIC_FIELDS = (
    'batch_state_policy', 'beta', 'datasets', 'energy_minimizer', 'eqprop',
    'initialization', 'input_mode', 'lab', 'learning_rates_by_parameter', 'lr',
    'max_batches', 'max_test_batches', 'model_base', 'model_overrides', 'optimizer',
    'parameter_order', 'runtime_dtype', 'seed', 'training_algorithm',
)


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def scientific_config(config):
    """Preserve numerical settings and initializer hash, normalize data transport."""
    projected = {key: copy.deepcopy(config[key]) for key in SCIENTIFIC_FIELDS}
    params = projected['datasets'][config['lab']['dataset_key']]['params']
    for key in ('root', 'download', 'num_workers', 'pin_memory'):
        params.pop(key, None)
    projected['initialization'] = {key: config['initialization'][key]
        for key in ('checkpoint_sha256', 'optimizer_state', 'recipe', 'seed')}
    if projected['initialization']['optimizer_state'] == 'fresh':
        projected['initialization']['optimizer_state'] = 'fresh_at_initialization'
    projected['evaluation'] = {key: config['evaluation'][key]
        for key in ('checkpoint_selection', 'epoch_split', 'official_test')}
    projected['relative_target'] = config['relative_output_calibration']['target']
    return projected


def cell_key(config):
    architecture, scheme = config['arm_id'].split('_')[:2]
    return architecture, scheme, float(config['eqprop']['endpoint_read_noise_eta'])


def reused(key):
    architecture, scheme, eta = key
    return scheme != 'amp2' and (eta == 0 or architecture == 'conv1')


def historical_root(root, key):
    architecture, _, eta = key
    name = ('conv1_noisy' if eta else 'conv1_clean') if architecture == 'conv1' else 'conv23_clean'
    return root.parent / HISTORICAL[name]


def production_roots(root):
    return [root / 'production', root / 'collected']


def load_configs(root):
    candidates = [root / 'source-training-v1/configs/conv' / CONFIG_SUBDIR,
                  ROOT / 'configs/conv' / CONFIG_SUBDIR]
    config_root = next((p for p in candidates if p.is_dir()), candidates[-1])
    configs = {}
    for path in sorted(config_root.rglob('*.json')):
        config = read(path)
        if not isinstance(config, dict) or 'arm_id' not in config or 'lab' not in config:
            continue
        key = cell_key(config)
        if key[0] not in ARCHITECTURES or key[1] not in SCHEMES or key[2] not in ETAS:
            continue
        require(key not in configs, f'Duplicate prepared config for {key}: {path}')
        require(config['seed'] == 0 and config['lab']['epochs'] == HORIZONS[key[0]],
                f'Wrong seed/horizon: {path}')
        require(config['relative_output_calibration']['target'] == TARGETS[key[0]],
                f'Wrong displacement target: {path}')
        require(config['initialization']['optimizer_state'] in ('fresh', 'fresh_at_initialization'),
                f'Weights-only continuation is ineligible: {path}')
        require(config['evaluation']['official_test']['policy'] == 'disabled', 'Official test enabled')
        configs[key] = (path, config)
    return config_root, configs


def discover(root, expected_arms):
    grouped, ignored, seen = {}, [], set()
    for production in production_roots(root):
        if not production.exists():
            continue
        for path in sorted(production.rglob('manifest.json')):
            run = path.parent.resolve()
            if run in seen:
                continue
            seen.add(run)
            manifest = read(path)
            if manifest.get('smoke'):
                ignored.append(str(run))
                continue
            if manifest.get('arm_id') not in expected_arms:
                continue
            status = read(run/'status.json') if (run/'status.json').exists() else {'state': 'missing_status'}
            result_hash = sha256_file(run/'result.json') if (run/'result.json').exists() else ''
            attempt = dict(run=str(run), manifest=manifest, state=status['state'],
                           error=status.get('error'), identity=sha256_file(path)+':'+result_hash)
            grouped.setdefault(manifest['arm_id'], []).append(attempt)
    return grouped, ignored


def select_attempt(attempts):
    """Identical copied successes count once; distinct attempts require review."""
    unique = {}
    for attempt in attempts:
        unique.setdefault(attempt['identity'], []).append(attempt)
    require(len(unique) <= 1, 'Multiple distinct attempts; resolve operational replacement explicitly')
    if not unique:
        return None, []
    copies = next(iter(unique.values()))
    # Prefer a copy with checkpoints present; all copies remain recorded.
    copies.sort(key=lambda a: not (Path(a['run'])/'checkpoints/final_model.pt').exists())
    return copies[0], [a['run'] for a in copies[1:]]


def epoch_history(run, horizon):
    epochs = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    epochs = [r for r in epochs if r.get('kind') == 'epoch']
    require([r['epoch'] for r in epochs] == list(range(1, horizon+1)), 'Incomplete/unordered epoch history')
    for row in epochs:
        require(row.get('evaluation_split') == 'validation', 'Non-validation epoch')
        require(all(math.isfinite(row['metrics'][k]) for k in
                    ('validation_accuracy', 'validation_loss', 'train_accuracy', 'train_loss')), 'Nonfinite epoch')
        require(0 <= row['metrics']['validation_accuracy'] <= 1, 'Invalid accuracy')
    return epochs


def optimizer_history(run, config, horizon):
    """Verify final Adam steps when saved; record weaker historical evidence."""
    record = read(run/'exact_run.json')
    recovery = None
    if (run/'diagnostic_resume.json').exists():
        recovery = read(run/'diagnostic_resume.json')
        require(recovery.get('schema') == 'diagnostic-epoch-recovery/v1', 'Unknown recovery receipt')
        require(recovery.get('optimizer_state_restored') is True and
                recovery.get('uninterrupted_trajectory') is False and
                recovery.get('continuation_quality') == 'approximate_stochastic_continuation',
                'Recovery must preserve Adam and disclose stochastic continuation')
        epoch = recovery['completed_epochs']
        require(0 < epoch < horizon and recovery['total_epochs'] == horizon and
                recovery['remaining_epochs'] == horizon-epoch and recovery['next_epoch'] == epoch+1,
                'Recovery horizon mismatch')
        source = Path(recovery['source_run'])
        require(sha256_file(source/'manifest.json') == recovery['source_manifest']['sha256'],
                'Recovery source manifest changed')
        saved = recovery['checkpoint']
        require(saved['epoch'] == epoch, 'Recovery checkpoint epoch mismatch')
        for kind in ('model', 'optimizer'):
            require(sha256_file(run/saved[kind+'_path']) == saved[kind+'_sha256'],
                    'Recovery checkpoint hash changed: '+kind)
    else:
        require(not any('resume' in str(x) for x in record.get('diagnostics', {})),
                'Diagnostic resume lacks recovery receipt')
    require(config['initialization']['optimizer_state'] in ('fresh', 'fresh_at_initialization'), 'Not initialized with fresh Adam')
    index = run/'epoch_checkpoint_index.jsonl'
    if not index.exists():
        require('continued_from' not in record, 'Continuation lacks optimizer evidence')
        return 'from-initialization full epoch history; historical optimizer checkpoints unavailable'
    entries = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
    final = [e for e in entries if e['epoch'] == horizon]
    require(len(final) == 1, 'Missing unique final optimizer checkpoint')
    import torch
    checkpoint_path = run/final[0]['optimizer_path']
    require(sha256_file(checkpoint_path) == final[0]['optimizer_sha256'], 'Optimizer checkpoint hash changed')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    require(checkpoint['epoch'] == horizon and checkpoint['optimizer_class'] == 'Adam', 'Optimizer checkpoint epoch/class')
    steps = [float(state['step']) for state in checkpoint['optimizer_state_dict']['state'].values() if 'step' in state]
    batch = config['datasets'][config['lab']['dataset_key']]['params']['batch_size']
    require(steps and all(step == math.ceil(55000/batch)*horizon for step in steps),
            'Adam step count indicates reset, shortening, or different train cohort')
    require(bool(checkpoint.get('rng_state')), 'Missing RNG checkpoint')
    if recovery is not None:
        return (f'Adam continued from epoch {recovery["completed_epochs"]}; final step counts verified; '
                'approximate stochastic continuation (missing/reconstructed RNG recorded)')
    return 'uninterrupted Adam verified by final step counts and saved RNG state'


def validate_complete(run, expected):
    errors = validate_run(run)
    require(not errors, 'Bundle validation: '+ '; '.join(errors))
    manifest, result, metrics, metadata = [read(run/name) for name in
        ('manifest.json', 'result.json', 'metrics.json', 'run_metadata.json')]
    actual = manifest['configuration']['resolved']
    want = scientific_config(expected)
    for label, observed in [('manifest', actual), ('used config', read(run/'config.used.json'))]:
        got = scientific_config(observed)
        require(got == want, label+' scientific mismatch: '+', '.join(k for k in want if got[k] != want[k]))
    horizon = expected['lab']['epochs']
    require(not manifest.get('smoke') and not result.get('smoke'), 'Smoke is not training')
    require(result['completion']['criteria_met'], 'Completion criteria not met')
    require(manifest['configuration']['epochs'] == metadata['epochs'] == horizon, 'Wrong horizon')
    require(manifest['runtime']['device'] == 'cuda', 'Non-CUDA production')
    require(metadata['seed'] == metrics['seed'] == expected['seed'] == 0, 'Seed mismatch')
    require(metadata['beta'] == expected['beta'], 'Actual beta mismatch')
    require(metrics['final_learning_rate'] == expected['lr'], 'Actual LR mismatch')
    require(metrics['training_algorithm'] == 'EP' and metrics['runtime_dtype'] == 'float64', 'Algorithm/dtype mismatch')
    for owner in (manifest, result):
        require(owner['dataset']['official_test_read'] is False and
                owner['dataset']['official_test_read_during_training'] is False and
                owner['dataset']['evaluation_split'] == 'validation', 'Official-test or split violation')
    require(metrics['official_test_evaluations'] == metrics['official_test_examples'] == 0, 'Official test read')
    require('--skip-terminal-official-test' in manifest['command'], 'Missing official-test skip')
    require(not any(flag in manifest['command'] for flag in ('--max-batches', '--max-test-batches')), 'Shortened batches')
    provenance = metrics['dataset_provenance']
    require(provenance['source_split'] == 'train' and provenance['split_seed'] == provenance['shuffle_seed'] == 0,
            'Wrong train/validation split or shuffle seed')
    require(len(provenance['train_batch_order_sha256']) == horizon, 'Incomplete minibatch history')
    for name in ('endpoint_read_noise_eta', 'endpoint_read_noise_std', 'endpoint_read_noise_mode',
                 'endpoint_read_noise_seed', 'variant', 'nudging_mode', 'input_read_noise'):
        require(metrics['eqprop'][name] == expected['eqprop'][name], 'Actual EqProp mismatch: '+name)
    history = epoch_history(run, horizon)
    require(math.isclose(history[-1]['metrics']['validation_accuracy'], metrics['final_validation_accuracy'],
                         abs_tol=1e-12), 'Final accuracy differs from history')
    return dict(accuracy_percent=100*history[-1]['metrics']['validation_accuracy'],
                initial_parameter_state_sha256=metrics['initial_parameter_state_sha256'],
                validation_indices_sha256=provenance['validation_indices_sha256'],
                train_batch_order_sha256=provenance['train_batch_order_sha256'],
                optimizer_history=optimizer_history(run, actual, horizon),
                source=manifest['git'], host=manifest['runtime']['host'],
                result_sha256=sha256_file(run/'result.json'),
                curve=[dict(epoch=e['epoch'], **e['metrics']) for e in history])


def comparisons(rows):
    lookup = {(r['architecture'], r['scheme'], r['eta']): r for r in rows}
    pairs, degradation = [], []
    for architecture in ARCHITECTURES:
        for scheme in SCHEMES:
            clean = lookup[architecture, scheme, 0.]
            for eta in ETAS:
                row = lookup[architecture, scheme, eta]
                drop = clean['accuracy_percent']-row['accuracy_percent'] if clean['state'] == row['state'] == 'complete' else None
                degradation.append(dict(architecture=architecture, scheme=scheme, eta=eta, degradation_pp=drop))
        for eta in ETAS:
            amp = lookup[architecture, 'amp2', eta]
            for scheme in ('ours', 'legacy'):
                other = lookup[architecture, scheme, eta]
                gap = amp['accuracy_percent']-other['accuracy_percent'] if amp['state'] == other['state'] == 'complete' else None
                pairs.append(dict(architecture=architecture, eta=eta, comparator=scheme, amp2_minus_comparator_pp=gap))
    return pairs, degradation


def write_plot(out, rows, pairs, degradation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(amp2='tab:blue', ours='tab:orange', legacy='gray')
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    for column, architecture in enumerate(ARCHITECTURES):
        for scheme in SCHEMES:
            rr = [next(r for r in rows if (r['architecture'], r['scheme'], r['eta']) == (architecture, scheme, eta)) for eta in ETAS]
            axes[0, column].plot(range(3), [r['accuracy_percent'] if r['state'] == 'complete' else float('nan') for r in rr], '-o', color=colors[scheme], label=scheme)
            dd = [d for d in degradation if d['architecture'] == architecture and d['scheme'] == scheme]
            axes[2, column].plot(range(3), [d['degradation_pp'] if d['degradation_pp'] is not None else float('nan') for d in dd], '-o', color=colors[scheme], label=scheme)
        for scheme in ('ours', 'legacy'):
            pp = [p for p in pairs if p['architecture'] == architecture and p['comparator'] == scheme]
            axes[1, column].plot(range(3), [p['amp2_minus_comparator_pp'] if p['amp2_minus_comparator_pp'] is not None else float('nan') for p in pp], '-o', color=colors[scheme], label='amp2 − '+scheme)
        axes[0, column].set_title(f'{architecture.upper()} · epoch {HORIZONS[architecture]}')
        for ax in axes[:, column]:
            ax.set_xticks(range(3), ['clean', '1e−5', '1e−4'])
            ax.set_xlabel('Relative read noise η'); ax.grid(alpha=.2); ax.legend(fontsize=8)
        for ax in axes[1:, column]:
            ax.axhline(0, color='black', lw=.7)
    for ax, label in zip(axes[:, 0], ['Validation accuracy (%)', 'Amp2 gap (pp)', 'Clean minus noisy accuracy (pp)']):
        ax.set_ylabel(label)
    fig.suptitle('Seed 0 · fixed horizons · missing outcomes left blank')
    fig.tight_layout(); fig.savefig(out/'fixed_horizon_comparison.jpg', dpi=180); plt.close(fig)


def analyze(root):
    root = root.resolve()
    config_root, configs = load_configs(root)
    expected = [(a, s, eta) for a in ARCHITECTURES for s in SCHEMES for eta in ETAS]
    roots = {root, *(historical_root(root, key) for key in expected if reused(key))}
    arm_ids = {c['arm_id'] for _, c in configs.values()}
    discovered = {r: discover(r, arm_ids) for r in roots}
    rows, errors, duplicates = [], [], []
    for key in expected:
        architecture, scheme, eta = key
        row = dict(architecture=architecture, scheme=scheme, eta=eta, seed=0,
                   horizon=HORIZONS[architecture], reused=reused(key), state='missing_config',
                   arm_id='', accuracy_percent=None, run='', issue='')
        if key not in configs:
            row['issue'] = 'Prepared canonical comparator/training config missing'
            rows.append(row); continue
        path, config = configs[key]
        row.update(arm_id=config['arm_id'], config=str(path), beta=config['beta'],
                   injected_beta=config['eqprop']['injected_beta_B'], learning_rates=config['lr'],
                   initializer_checkpoint_sha256=config['initialization']['checkpoint_sha256'])
        source_root = historical_root(root, key) if reused(key) else root
        try:
            attempt, copies = select_attempt(discovered[source_root][0].get(config['arm_id'], []))
            duplicates.extend(copies)
            if attempt is None:
                row['state'] = 'not_collected'
            else:
                row.update(state=attempt['state'], run=attempt['run'], issue=attempt['error'] or '')
                if row['state'] == 'complete':
                    row.update(validate_complete(Path(attempt['run']), config))
                elif row['state'] == 'failed':
                    require(scientific_config(attempt['manifest']['configuration']['resolved']) ==
                            scientific_config(config), 'Failed attempt scientific config mismatch')
                    validation_errors = validate_run(Path(attempt['run']))
                    require(not validation_errors, 'Invalid failed bundle: '+str(validation_errors))
        except (OSError, ValueError, KeyError, TypeError) as error:
            row.update(state='invalid', issue=str(error)); errors.append(f'{key}: {error}')
        rows.append(row)
    # Actual initializer and deterministic cohort/order must match across all schemes.
    for architecture in ARCHITECTURES:
        complete = [r for r in rows if r['architecture'] == architecture and r['state'] == 'complete']
        for field in ('initial_parameter_state_sha256', 'initializer_checkpoint_sha256',
                      'validation_indices_sha256', 'train_batch_order_sha256'):
            if len({json.dumps(r[field], sort_keys=True) for r in complete}) > 1:
                errors.append(f'{architecture}: unmatched {field}')
                for row in complete:
                    row.update(state='invalid', issue='Cross-scheme identity mismatch: '+field)
    pairs, degradation = comparisons(rows)
    new = [r for r in rows if not r['reused']]
    old = [r for r in rows if r['reused']]
    complete = sum(r['state'] == 'complete' for r in rows)
    caveats = [
        'Single seed, exploratory ordinary-MNIST validation; no official-test or best-epoch comparison.',
        'The amp2 arm changes A/B and doubles ours weight learning rates; gains cannot isolate amplification from LR.',
        'Historical and new source archives/hosts may differ. Identities are recorded; scientific config matching does not establish numerical source equivalence.',
        'Historical Conv1 noisy controls have no saved optimizer checkpoints; uninterrupted history is supported by from-initialization provenance and complete epochs, not Adam-step replay.',
        'User-authorized diagnostic continuations preserve model/Adam state and completed epochs; their missing noise RNG state is disclosed per outcome and they are not bit-exact uninterrupted trajectories.',
    ]
    summary = dict(complete=complete == 27 and not errors, ready_for_review=not errors and
        all(r['state'] in ('complete', 'failed') for r in new) and all(r['state'] == 'complete' for r in old),
        expected_outcomes=27, expected_new=17, expected_reused=10, completed=complete,
        new_completed=sum(r['state'] == 'complete' for r in new), reused_completed=sum(r['state'] == 'complete' for r in old),
        states={state: sum(r['state'] == state for r in rows) for state in sorted({r['state'] for r in rows})},
        config_root=str(config_root), errors=errors, duplicate_copies=duplicates,
        ignored_smokes=[p for _, ignored in discovered.values() for p in ignored], caveats=caveats,
        outcomes=rows, comparisons=pairs, degradation=degradation)
    out = root/'analysis'; out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    columns = ['architecture', 'scheme', 'eta', 'seed', 'horizon', 'reused', 'state', 'accuracy_percent',
               'beta', 'injected_beta', 'learning_rates', 'initializer_checkpoint_sha256',
               'initial_parameter_state_sha256', 'optimizer_history', 'host', 'source', 'result_sha256', 'run', 'issue']
    with (out/'accuracy.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    write_plot(out, rows, pairs, degradation)
    lines = ['# Exp020 fixed-horizon evidence', '',
             f'Validated outcomes: {complete}/27; new: {summary["new_completed"]}/17; reused: {summary["reused_completed"]}/10.',
             '', '| Model | Scheme | η | Epoch | Accuracy (%) | State |', '|---|---|---:|---:|---:|---|']
    for r in rows:
        acc = f'{r["accuracy_percent"]:.2f}' if r['state'] == 'complete' else '—'
        lines.append(f'| {r["architecture"]} | {r["scheme"]} | {r["eta"]:g} | {r["horizon"]} | {acc} | {r["state"]} |')
    lines += ['', '| Model | η | Comparator | Amp2 gap (pp) |', '|---|---:|---|---:|']
    for pair in pairs:
        gap = pair['amp2_minus_comparator_pp']
        lines.append(f'| {pair["architecture"]} | {pair["eta"]:g} | {pair["comparator"]} | '+(f'{gap:+.2f}' if gap is not None else '—')+' |')
    lines += ['', '[Accuracy, gaps and degradation](fixed_horizon_comparison.jpg)', '', *['- '+c for c in caveats],
              '', 'Validation errors: '+('; '.join(errors) or 'none'),
              '', 'Scientific interpretation and next decision belong in the campaign result note.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', required=True, type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(argv)
    summary = analyze(args.study_root)
    print(json.dumps({k: summary[k] for k in ('complete', 'ready_for_review', 'completed', 'new_completed', 'reused_completed', 'states', 'errors')}))
    return 0 if summary['complete'] or (args.allow_partial and not summary['errors']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
