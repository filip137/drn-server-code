"""Reconcile this completed study and plot its frozen ten-epoch comparison.

Run from the repository root after both phase analyzers report full coverage.
Scientific interpretation is written separately in report.md and the manifest.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from experiments.reporting import atomic_write_json, validate_run

SCHEMES = ('baseline', 'ours', 'legacy')
LABELS = {'baseline': 'Baseline (v1/c1)', 'ours': 'Proposed (v4/c1)', 'legacy': 'Legacy (v4/c0.25)'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--configs', type=Path, default=Path('configs/cifar/lr_search_20260921'))
    args = parser.parse_args()
    ROOT = args.root
    OUT = ROOT / 'analysis'
    CONFIGS = args.configs
    grid = read(OUT / 'grid-plus-boundary/summary.json')
    confirmation = read(OUT / 'confirmation/summary.json')
    assert grid['ready_for_selection'] and grid['declared'] == grid['terminal'] == 29
    assert confirmation['ready_for_selection'] and confirmation['declared'] == confirmation['terminal'] == 6
    rows = grid['rows'] + confirmation['rows']
    assert len({r['run_id'] for r in rows}) == 35
    assert all(r['state'] == 'complete' and r['eligible'] and r['native_exit_code'] == 0 for r in rows)
    configs = {p.stem: read(p) for d in (CONFIGS, CONFIGS / 'boundary', CONFIGS / 'confirmation') for p in d.glob('*.json')}
    assert len(configs) == 35 and set(configs) == {r['run_id'] for r in rows}
    assert {p.parent.name for p in (ROOT / 'cells').glob('*/status.json')} == set(configs)

    receipts = {}
    for p in sorted((ROOT / 'launcher').glob('*.json')):
        for run in read(p).get('runs', []):
            if 'exit_code' not in run:
                continue
            assert run['run_id'] not in receipts, run['run_id']
            receipts[run['run_id']] = run
    assert set(receipts) == set(configs)
    assert all(r['exit_code'] == 0 for r in receipts.values())

    bundles = sorted([*(ROOT / 'cells').glob('*/manifest.json'),
                      *ROOT.glob('prepare-*/manifest.json'), *ROOT.glob('smoke-*/manifest.json'),
                      *ROOT.glob('operational_failures/*/*/manifest.json')])
    validation = []
    for p in bundles:
        errors = validate_run(p.parent)
        assert not errors, (str(p.parent), errors)
        validation.append({'path': str(p.parent.relative_to(ROOT)),
                           'state': read(p.parent / 'status.json')['state'], 'errors': errors})
    assert len(validation) == 58
    assert Counter(r['state'] for r in validation) == {'complete': 57, 'failed': 1}
    release = read(ROOT / 'worker_release_checks.json')
    assert release['all_study_workers_exited'] and len(release['hosts']) == 6

    source = read(ROOT / 'source-v2/source_identity.json')
    assert all(sha(ROOT / 'source-v2' / p) == h for p, h in source['source_sha256'].items())
    assert sha(ROOT / 'shared_initial_model.pt') == rows[0]['initial_sha256']
    initial_models = {}
    for row in rows:
        model = torch.load(ROOT / 'cells' / row['run_id'] / 'checkpoints/initial_model.pt', map_location='cpu', weights_only=False)['model']
        h = hashlib.sha256()
        for section in sorted(model):
            for name in sorted(model[section]):
                t = model[section][name].detach().cpu().contiguous()
                h.update(f'{section}.{name}:{t.dtype}:{tuple(t.shape)}'.encode())
                h.update(t.numpy().tobytes())
        initial_models[row['run_id']] = h.hexdigest()
    assert len(set(initial_models.values())) == 1

    replays = []
    fields = ('train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy',
              'first_batch_gradient_norms', 'clipping', 'learning_rates', 'steps')
    for row in confirmation['rows']:
        cfg = configs[row['run_id']]
        original = cfg['lr_search']['promoted_from']
        a = [json.loads(line) for line in (ROOT / 'cells' / original / 'metrics.jsonl').read_text().splitlines()]
        b = [json.loads(line) for line in (ROOT / 'cells' / row['run_id'] / 'metrics.jsonl').read_text().splitlines()]
        differences = [{'epoch': x['epoch'], 'field': k} for x, y in zip(a, b) for k in fields if x[k] != y[k]]
        replays.append({'original': original, 'confirmation': row['run_id'],
                        'epochs_compared': len(a), 'exact': not differences, 'differing_fields': differences})

    selected = {scheme: next(r for r in confirmation['rows'] if r['run_id'] == confirmation['top_two'][scheme][0]) for scheme in SCHEMES}
    cost = sum(r['elapsed_seconds'] for r in receipts.values()) / 3600
    failed_cost = sum(r.get('elapsed_seconds', 0) for p in (ROOT / 'operational_failures').glob('*/launcher.json') for r in read(p).get('runs', [])) / 3600
    support_seconds = sum(read(p)['terminal_metrics']['elapsed_seconds'] for p in [*ROOT.glob('prepare-*/result.json'), *ROOT.glob('smoke-*/result.json')])
    summary = {'study_id': ROOT.name, 'generated_at': datetime.now(timezone.utc).isoformat(),
               'scientific_cases': 35, 'five_epoch_cases': 29, 'ten_epoch_cases': 6,
               'selected': selected, 'confirmation_rows': confirmation['rows'],
               'official_test_read': False, 'evidence_class': 'exploratory_single_seed_validation',
               'ranking_rule': 'epoch10 validation accuracy, then lower CE, then lower conv/head LR multipliers',
               'native_candidate_worker_gpu_hours': cost, 'failed_startup_worker_gpu_hours': failed_cost,
               'reported_support_worker_gpu_hours': support_seconds / 3600,
               'measured_total_worker_gpu_hours': cost + failed_cost + support_seconds / 3600,
               'checks_reserve_gpu_hours': 1, 'study_cap_gpu_hours': 72,
               'frozen_admission_bound_gpu_hours': 60.447653479653695}
    assert cost + failed_cost + 1 < 72
    atomic_write_json(OUT / 'summary.json', summary)
    atomic_write_json(OUT / 'confirmation_replay_checks.json', {'fields': fields, 'comparisons': replays})
    atomic_write_json(ROOT / 'closeout_validation.json', {
        'generated_at': summary['generated_at'], 'ready_for_review': True,
        'expected_scientific_cases': sorted(configs), 'validated_bundle_count': len(validation),
        'bundles': validation, 'all_scientific_native_exits_zero': True,
        'all_final_solver_gates_passed': True, 'source_snapshot_hashes_match': True,
        'all_study_workers_exited': True,
        'analysis_source': {'path': str(Path(__file__)), 'sha256': sha(Path(__file__))},
        'identical_initial_model_tensor_hashes': initial_models,
        'phase_summary_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [OUT / 'grid-plus-boundary/summary.json', OUT / 'confirmation/summary.json']},
        'exclusions': ['operational_failures/local-rebalanced/legacy_c2_d2: failed before first optimizer step; unchanged replacement cells/legacy_c2_d2 is included with receipt launcher/local-retry1.json'],
        'superseded_transport_receipts': ['launcher/local.json', 'launcher/nom-cool-1.json', 'launcher/riri.json']})

    fields_csv = ('run_id', 'scheme', 'epochs', 'target', 'conv_multiplier', 'head_multiplier', 'validation_accuracy', 'cross_entropy', 'final_solver_audit_passed', 'accounted_seconds')
    with (OUT / 'comparison.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields_csv, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    with (OUT / 'selected_learning_rates.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['layer', *SCHEMES])
        vectors = {s: [v for block in selected[s]['conv_learning_rates'] for v in block] for s in SCHEMES}
        for i in range(8):
            w.writerow([f'conv{i+1}', *[vectors[s][i] for s in SCHEMES]])
        w.writerow(['analog_dense_head', *[selected[s]['head_learning_rate'] for s in SCHEMES]])
        for key in ('bn_learning_rate', 'gain_learning_rate'):
            w.writerow([key, *[configs[selected[s]['run_id']]['optimizer'][key] for s in SCHEMES]])

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey='row', constrained_layout=True)
    for j, scheme in enumerate(SCHEMES):
        for row in [r for r in confirmation['rows'] if r['scheme'] == scheme]:
            chosen = row['run_id'] == selected[scheme]['run_id']
            label = f"conv x{row['conv_multiplier']:g}, head x{row['head_multiplier']:g}" + (' (selected)' if chosen else '')
            epochs = [x['epoch'] for x in row['curve']]
            style = {'linewidth': 2.1 if chosen else 1.5, 'linestyle': '-' if chosen else '--', 'color': '#1764ab' if chosen else '#777777', 'marker': 'o', 'markersize': 3}
            axes[0, j].plot(epochs, [100*x['accuracy'] for x in row['curve']], label=label, **style)
            axes[1, j].plot(epochs, [x['ce'] for x in row['curve']], **style)
        axes[0, j].set_title(LABELS[scheme]); axes[0, j].legend(fontsize=8)
        axes[1, j].set_xlabel('Epoch'); axes[1, j].set_xticks(range(1, 11))
        for ax in axes[:, j]:
            ax.grid(alpha=.25)
    axes[0, 0].set_ylabel('Validation accuracy (%)'); axes[1, 0].set_ylabel('Validation cross-entropy')
    fig.suptitle('CIFAR-10 analog L8: frozen ten-epoch finalists, seed 0\nLR multipliers are relative to each scheme’s MNIST-derived prior')
    fig.savefig(OUT / 'ten_epoch_learning_curves.png', dpi=180); plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('selected', 'confirmation_rows')}, indent=2))
    print(json.dumps({s: {k: r[k] for k in ('run_id', 'validation_accuracy', 'cross_entropy', 'head_learning_rate')} for s, r in selected.items()}, indent=2))


if __name__ == '__main__':
    main()
