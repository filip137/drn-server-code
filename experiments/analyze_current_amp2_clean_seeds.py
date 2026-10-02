"""Validate the six clean replicates and compare 27 fixed-horizon seed outcomes.

python -m experiments.analyze_current_amp2_clean_seeds --study-root RESULTS --allow-partial
Only production/collected bundles count. Parent seed0 outcomes are references,
not additional executions. Means and sample SDs explicitly retain their seeds.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from experiments.analyze_current_amp2 import (
    ARCHITECTURES, HORIZONS, SCHEMES, TARGETS, discover, epoch_history,
    optimizer_history, read, require, scientific_config, select_attempt,
)
from experiments.reporting import sha256_file as sha, validate_run

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SUBDIR = 'current_amp2_clean_seeds12_20260928'
PARENT = 'conv123-current-amp2-20260928-v1'
HISTORICAL = {1: 'conv1-clean-and-baseline-noise-20260927-v1',
              2: 'conv23-noiseless-three-seed-20260927-v1',
              3: 'conv23-noiseless-three-seed-20260927-v1'}


def config_path(study, subdir, arm):
    choices = [study/'source-training-v1/configs/conv'/subdir/(arm+'.json'),
               ROOT/'configs/conv'/subdir/(arm+'.json')]
    return next((p for p in choices if p.is_file()), choices[0])


def validate_complete(run, expected):
    require(not (errors := validate_run(run)), 'Bundle validation: '+ '; '.join(errors))
    manifest, result, metrics, metadata = [read(run/name) for name in
        ('manifest.json', 'result.json', 'metrics.json', 'run_metadata.json')]
    actual = manifest['configuration']['resolved']
    want = scientific_config(expected)
    for label, observed in [('manifest', actual), ('used config', read(run/'config.used.json'))]:
        got = scientific_config(observed)
        require(got == want, label+' scientific mismatch: '+', '.join(k for k in want if got[k] != want[k]))
    horizon, seed = expected['lab']['epochs'], expected['seed']
    require(not manifest.get('smoke') and not result.get('smoke'), 'Smoke is not training')
    require(result['completion']['criteria_met'], 'Completion criteria not met')
    require(manifest['configuration']['epochs'] == metadata['epochs'] == horizon, 'Wrong horizon')
    require(manifest['runtime']['device'] == 'cuda', 'Non-CUDA production')
    require(metadata['seed'] == metrics['seed'] == seed, 'Seed mismatch')
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
    require(provenance['source_split'] == 'train' and provenance['split_seed'] == 0 and
            provenance['shuffle_seed'] == seed, 'Wrong split or shuffle seed')
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
                manifest_sha256=sha(run/'manifest.json'), result_sha256=sha(run/'result.json'))


def summarize(values):
    return dict(n=len(values), mean=statistics.mean(values) if values else None,
                sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                all_three_seeds=len(values) == 3)


def validate_initializer_digest(study, config, check, observed):
    """Verify the preparation and trainer schemas against identical CPU tensors."""
    import torch

    checkpoint = study/'inputs/initializers'/Path(check['accepted_checkpoint']).name
    require(sha(checkpoint) == check['checkpoint_sha256'] ==
            config['initialization']['checkpoint_sha256'], 'Initializer checkpoint identity changed')
    states = torch.load(checkpoint, map_location='cpu', weights_only=False)['states']
    names = config['parameter_order']
    require(len(names) == len(states), 'Initializer parameter count mismatch')
    preparation = hashlib.sha256(b'drn-ordered-parameter-state/v1\0')
    trainer = hashlib.sha256(b'mnist-train-ordered-parameter-state/v1\0')
    for name, state in zip(names, states):
        require(state.dtype == torch.float32, 'Unexpected accepted initializer dtype')
        tensor = state.to(torch.float64).contiguous()
        for digest in (preparation, trainer):
            for part in (name.encode('utf-8'), b'\0', str(tensor.dtype).encode('ascii'),
                         b'\0', json.dumps(list(tensor.shape)).encode('ascii'), b'\0',
                         tensor.numpy().tobytes(order='C')):
                digest.update(part)
    require(preparation.hexdigest() == check['initial_parameter_state_sha256'],
            'Accepted initializer differs from preparation CPU check')
    require(trainer.hexdigest() == observed, 'Actual initializer differs from accepted CPU check')


def write_plot(out, rows, pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(amp2='tab:blue', ours='tab:orange', legacy='gray')
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for col, architecture in enumerate(ARCHITECTURES):
        for scheme in SCHEMES:
            selected = [r for r in rows if r['architecture'] == architecture and r['scheme'] == scheme]
            axes[0, col].plot([r['seed'] for r in selected],
                [r['accuracy_percent'] if r['state'] == 'complete' else float('nan') for r in selected],
                '-o', label=scheme, color=colors[scheme])
        for comparator in ('ours', 'legacy'):
            selected = [r for r in pairs if r['architecture'] == architecture and r['comparator'] == comparator]
            axes[1, col].plot([r['seed'] for r in selected],
                [r['gap_pp'] if r['gap_pp'] is not None else float('nan') for r in selected],
                '-o', label='amp2 − '+comparator, color=colors[comparator])
        axes[0, col].set_title(f'{architecture.upper()} · epoch {HORIZONS[architecture]}')
        axes[1, col].axhline(0, color='black', lw=.7)
        for ax in axes[:, col]:
            ax.set_xticks([0, 1, 2]); ax.set_xlabel('Seed'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    axes[0, 0].set_ylabel('Validation accuracy (%)')
    axes[1, 0].set_ylabel('Paired accuracy difference (pp)')
    fig.suptitle('Clean A4/B2 · fixed seed0 beta · missing outcomes left blank')
    fig.tight_layout(); fig.savefig(out/'clean_three_seed_comparison.jpg', dpi=180); plt.close(fig)


def analyze(root):
    root = root.resolve()
    preparation_path = config_path(root, CONFIG_SUBDIR, 'provenance/preparation')
    preparation = read(preparation_path)
    prepared = {r['arm_id']: r for r in preparation['configs']}
    checks = {(r['architecture'], r['seed']): r for r in preparation['initializer_checks']}
    rows, errors, duplicates = [], [], []
    for depth, architecture in enumerate(ARCHITECTURES, 1):
        for scheme in SCHEMES:
            for seed in (0, 1, 2):
                new = scheme == 'amp2' and seed != 0
                source = root if new else root.parent/(PARENT if scheme == 'amp2' else HISTORICAL[depth])
                subdir = CONFIG_SUBDIR if new else ('current_amp2_20260928' if scheme == 'amp2' else
                    'conv1_clean_baseline_noise_20260927' if depth == 1 else 'noiseless_three_seed_20260927')
                arm = f'{architecture}_{scheme}_relative{int(TARGETS[architecture])}_eta0_seed{seed}'
                row = dict(architecture=architecture, scheme=scheme, seed=seed, arm_id=arm,
                    horizon=HORIZONS[architecture], new=new, parent_seed0=scheme == 'amp2' and seed == 0,
                    state='not_collected', accuracy_percent=None, run='', issue='')
                try:
                    path = config_path(source, subdir, arm)
                    config = read(path)
                    require(config['seed'] == config['initialization']['seed'] == seed, 'Config seed mismatch')
                    require(config['datasets']['mnist']['params']['shuffle_seed'] == seed and
                            config['datasets']['mnist']['params']['split_seed'] == 0, 'Config cohort mismatch')
                    require(config['lab']['epochs'] == HORIZONS[architecture], 'Config horizon mismatch')
                    require(config['eqprop']['endpoint_read_noise_eta'] == config['eqprop']['endpoint_read_noise_std'] == 0,
                            'Not a clean config')
                    require(config['evaluation']['official_test']['policy'] == 'disabled', 'Official test enabled')
                    if new:
                        require(sha(path) == prepared[arm]['sha256'], 'Prepared config hash changed')
                        parent_path = config_path(root.parent/PARENT, 'current_amp2_20260928', arm[:-1]+'0')
                        parent = read(parent_path)
                        require(sha(parent_path) == config['current_amp2_comparison']['parent_seed0_config_sha256'],
                                'Parent config identity changed')
                        require(config['lr'] == parent['lr'] and config['beta'] == parent['beta'] and
                                config['eqprop'] == parent['eqprop'], 'Fixed LR/beta/EqProp contract changed')
                        require(config['relative_output_calibration']['calibrated_for_this_seed'] is False,
                                'Seed-specific calibration incorrectly claimed')
                    row.update(config=str(path), beta=config['beta'], learning_rates=config['lr'],
                        initializer_checkpoint_sha256=config['initialization']['checkpoint_sha256'])
                    discovered, _ = discover(source, {arm})
                    attempt, copies = select_attempt(discovered.get(arm, [])); duplicates.extend(copies)
                    if attempt:
                        row.update(state=attempt['state'], run=attempt['run'], issue=attempt['error'] or '')
                        if row['state'] == 'complete':
                            row.update(validate_complete(Path(attempt['run']), config))
                            if new:
                                validate_initializer_digest(root, config, checks[architecture, seed],
                                    row['initial_parameter_state_sha256'])
                        elif row['state'] == 'failed':
                            require(scientific_config(attempt['manifest']['configuration']['resolved']) ==
                                    scientific_config(config), 'Failed attempt config mismatch')
                            require(not validate_run(Path(attempt['run'])), 'Invalid failed bundle')
                    if row['parent_seed0'] and row['state'] in ('not_collected', 'running', 'starting'):
                        row['issue'] = 'Parent seed0 pending; supplemental replication is not complete'
                except (OSError, ValueError, KeyError, TypeError) as error:
                    row.update(state='invalid', issue=str(error)); errors.append(f'{arm}: {error}')
                rows.append(row)
    for architecture in ARCHITECTURES:
        for seed in (0, 1, 2):
            complete = [r for r in rows if r['architecture'] == architecture and r['seed'] == seed and r['state'] == 'complete']
            for field in ('initial_parameter_state_sha256', 'initializer_checkpoint_sha256',
                          'validation_indices_sha256', 'train_batch_order_sha256'):
                if len({json.dumps(r[field], sort_keys=True) for r in complete}) > 1:
                    issue = f'{architecture} seed{seed}: unmatched {field}'
                    errors.append(issue)
                    for row in complete:
                        row.update(state='invalid', issue=issue)
    lookup = {(r['architecture'], r['scheme'], r['seed']): r for r in rows}
    stats, pairs, paired_stats = [], [], []
    for architecture in ARCHITECTURES:
        for scheme in SCHEMES:
            selected = [r for r in rows if r['architecture'] == architecture and r['scheme'] == scheme and r['state'] == 'complete']
            stats.append(dict(architecture=architecture, scheme=scheme, seeds=[r['seed'] for r in selected],
                              **summarize([r['accuracy_percent'] for r in selected])))
        for comparator in ('ours', 'legacy'):
            for seed in (0, 1, 2):
                amp, other = lookup[architecture, 'amp2', seed], lookup[architecture, comparator, seed]
                gap = amp['accuracy_percent']-other['accuracy_percent'] if amp['state'] == other['state'] == 'complete' else None
                pairs.append(dict(architecture=architecture, comparator=comparator, seed=seed, gap_pp=gap))
            selected = [r for r in pairs if r['architecture'] == architecture and r['comparator'] == comparator and r['gap_pp'] is not None]
            paired_stats.append(dict(architecture=architecture, comparator=comparator, seeds=[r['seed'] for r in selected],
                                     **summarize([r['gap_pp'] for r in selected])))
    completed = sum(r['state'] == 'complete' for r in rows)
    caveats = ['Ordinary-MNIST validation only; fixed epochs 10/30/30, no official-test or best-epoch comparison.',
        'Seeds1/2 retain seed0 beta; their initial D/F is unmeasured and may differ.',
        'Amp2 combines A4/B2 with twice ours learning rates; this does not isolate amplification from learning rate.',
        'Sample SD uses n−1; partial summaries report only observed seeds and are not three-seed results.',
        'Historical numerical source and hosts can differ; config and paired initializers/cohorts are validated.']
    summary = dict(complete=completed == 27 and not errors, completed=completed,
        expected_new=6, expected_parent_seed0=3, expected_reused_controls=18,
        new_completed=sum(r['state'] == 'complete' and r['new'] for r in rows),
        parent_seed0_completed=sum(r['state'] == 'complete' and r['parent_seed0'] for r in rows),
        states={s: sum(r['state'] == s for r in rows) for s in sorted({r['state'] for r in rows})},
        errors=errors, duplicate_copies=duplicates, outcomes=rows, statistics=stats,
        paired_seed_gaps=pairs, paired_statistics=paired_stats, caveats=caveats)
    out = root/'analysis'; out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    with (out/'accuracy.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['architecture', 'scheme', 'seed', 'horizon', 'state',
            'accuracy_percent', 'beta', 'learning_rates', 'run', 'issue'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    write_plot(out, rows, pairs)
    lines = ['# Clean A4/B2 seed replication', '', f'Validated outcomes: {completed}/27; new: {summary["new_completed"]}/6; parent seed0: {summary["parent_seed0_completed"]}/3.',
        '', '| Model | Scheme | Seeds | Mean accuracy (%) | Sample SD (pp) |', '|---|---|---|---:|---:|']
    fmt = lambda value: '—' if value is None else f'{value:.3f}'
    for r in stats:
        lines.append(f'| {r["architecture"]} | {r["scheme"]} | {r["seeds"]} | {fmt(r["mean"])} | {fmt(r["sample_sd"])} |')
    lines += ['', '| Model | Paired difference | Seeds | Mean gap (pp) | Sample SD (pp) |', '|---|---|---|---:|---:|']
    for r in paired_stats:
        lines.append(f'| {r["architecture"]} | amp2 − {r["comparator"]} | {r["seeds"]} | {fmt(r["mean"])} | {fmt(r["sample_sd"])} |')
    lines += ['', '[Fixed-horizon seed comparisons](clean_three_seed_comparison.jpg)', '', *['- '+c for c in caveats],
              '', 'Validation errors: '+('; '.join(errors) or 'none')]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', required=True, type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    summary = analyze(args.study_root)
    print(json.dumps({key: summary[key] for key in ('complete', 'completed', 'new_completed', 'parent_seed0_completed', 'states', 'errors')}))
    return 0 if summary['complete'] or (args.allow_partial and not summary['errors']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
