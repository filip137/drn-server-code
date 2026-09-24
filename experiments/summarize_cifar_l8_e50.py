"""Compare the two requested epoch50 continuations with completed proposed."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments import reporting
from experiments.summarize_cifar_l8_continuation import verify_checkpoints


def read_json(path):
    return json.loads(path.read_text())


def ancestry(run):
    """Follow declared parents, retaining absolute epoch numbers."""
    config = read_json(run / 'manifest.json')['resolved_config']
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()] if (run / 'metrics.jsonl').exists() else []
    if 'continuation' in config:
        parent = run.parents[2] / config['continuation']['parent_study'] / 'cells' / config['continuation']['parent_cell']
        assert reporting.sha256_file(parent / 'checkpoints/final_model.pt') == config['continuation']['checkpoint_sha256']
        rows = ancestry(parent) + rows
    assert [r['epoch'] for r in rows] == list(range(1, len(rows) + 1))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--verify-checkpoints', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    runs = {'baseline': root / 'cells/baseline_e30_e50',
            'ours': root.parent / 'cifar10-l8-analog-ours-e30-e50-seed0-20260922-v1/cells/ours_e30_e50',
            'legacy': root / 'cells/legacy_e30_e50'}
    out = root / 'analysis'; out.mkdir(exist_ok=True)
    summary = {'complete': True, 'official_test_read': False, 'schemes': {}}
    checks = {}
    lines = ['# CIFAR L8 comparison through epoch50', '',
             'Seed0 validation, Adam, batch32, trainable affine BN, crop/flip',
             'augmentation, cross-entropy and the unchanged50epoch cosine schedule.', '',
             '| Scheme | Epoch10 | Epoch30 | Epoch50 | CE50 | Latest epoch | Latest accuracy |',
             '|---|---:|---:|---:|---:|---:|---:|']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for scheme, run in runs.items():
        manifest = read_json(run / 'manifest.json'); config = manifest['resolved_config']
        state = read_json(run / 'status.json')['state']
        parent = root.parent / config['continuation']['parent_study'] / 'cells' / config['continuation']['parent_cell']
        prior = read_json(parent / 'manifest.json')['resolved_config']
        permitted = {'study_id', 'arm_id', 'evidence_class', 'epochs', 'maximum_runtime_hours', 'continuation'}
        assert all(config.get(k) == prior.get(k) for k in set(config) | set(prior) if k not in permitted)
        assert config['epochs'] == 50 and config['continuation']['from_epoch'] == 30
        provenance = read_json(run / 'artifacts/continuation.json')
        assert provenance['parent_steps'] == 42210 and provenance['scheduler_last_epoch'] == 30
        rows = ancestry(run)
        assert all((r['train_examples'], r['validation_examples'], r['steps']) == (45000, 5000, 1407*r['epoch']) for r in rows)
        if state == 'complete':
            assert not reporting.validate_run(run)
            result = read_json(run / 'result.json')
            assert len(rows) == 50 and result['terminal_metrics']['final_solver_audit_passed']
            if args.verify_checkpoints:
                checks[scheme] = verify_checkpoints(run, parent, config, rows[30:])
        else:
            summary['complete'] = False
            assert args.allow_partial, (scheme, state)
        by_epoch = {r['epoch']: r for r in rows}
        best = max(rows, key=lambda r: (r['validation_accuracy'], -r['validation_loss']))
        item = {'run': str(run), 'state': state, 'latest_epoch': rows[-1]['epoch'],
                'latest_accuracy': rows[-1]['validation_accuracy'], 'latest_loss': rows[-1]['validation_loss'],
                'epoch30_accuracy': by_epoch[30]['validation_accuracy'],
                'best_epoch': best['epoch'], 'best_accuracy': best['validation_accuracy']}
        summary['schemes'][scheme] = item
        values = [f"{100*by_epoch[e]['validation_accuracy']:.2f}%" if e in by_epoch else 'pending' for e in (10, 30, 50)]
        ce = f"{by_epoch[50]['validation_loss']:.4f}" if 50 in by_epoch else 'pending'
        lines.append(f"| {scheme} | {' | '.join(values)} | {ce} | {item['latest_epoch']} | {100*item['latest_accuracy']:.2f}% |")
        for ax, metric in zip(axes, ('validation_accuracy', 'validation_loss')):
            scale = 100 if metric.endswith('accuracy') else 1
            ax.plot([r['epoch'] for r in rows], [scale*r[metric] for r in rows], label=scheme)
    for ax, label in zip(axes, ('Validation accuracy (%)', 'Validation cross-entropy')):
        for epoch in (10, 30): ax.axvline(epoch, color='gray', linestyle='--', linewidth=1)
        ax.set(xlabel='Absolute epoch', ylabel=label, xlim=(1, 50)); ax.grid(alpha=.2); ax.legend()
    fig.savefig(out / 'learning_curves.png', dpi=180); plt.close(fig)
    lines += ['', 'Proposed is the separately completed continuation, included as a reference.',
              'One seed and different GPU/software stacks limit scheme-ranking claims.',
              'No official test is read. Endpoint accuracy is distinct from best-epoch accuracy.',
              '', '![Learning curves](learning_curves.png)', '']
    reporting.atomic_write_json(out / 'summary.json', summary)
    if args.verify_checkpoints: reporting.atomic_write_json(out / 'checkpoint_verification.json', checks)
    (out / 'report.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
