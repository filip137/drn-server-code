"""Validate local p95 first-level outcomes and compare matching V100 controls."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np

from experiments.analyze_conv3_refined_beta_training import without_dataset_root
from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'results/eqprop-conv3-p95-read-noise-1em3-20260921-v1'
REPORT = ROOT / 'paper_ready_results/conv3_p95_read_noise_1em3_20260921'
SOURCE_SHA = '74f41a03ce4718eebcd227bb38e4bbd2309e6b30bb9dd5515c099bb880068819'


def read(path):
    return json.loads(path.read_text())


def passes_stability_screen(values):
    """Use exact validation counts so an exactly 5pp drop cannot round down."""
    values = np.asarray(values, dtype=float)
    if len(values) != 30 or not np.isfinite(values).all():
        return False
    counts = np.rint(values * 5000)
    assert np.allclose(values * 5000, counts, rtol=0, atol=1e-6)
    return bool(counts.max() - counts[-1] < 250)


def collect(case):
    row = dict(case, state='pending', epochs_completed=0, epochs_data=[])
    task = STUDY / 'jz/production' / f"task_{case['index']}"
    bundles = list((task / 'runs').glob(f"000_{case['case']}_*"))
    assert len(bundles) <= 1, f"Ambiguous bundle: {case['case']}"
    if not bundles:
        return row
    run = bundles[0]
    assert not validate_run(run), run
    cfg = ROOT / case['config']
    assert hashlib.sha256(cfg.read_bytes()).hexdigest() == case['config_sha256']
    config, status, manifest = read(cfg), read(run / 'status.json'), read(run / 'manifest.json')
    assert manifest['git']['source_archive_sha256'] == SOURCE_SHA
    assert manifest['configuration']['sha256'] == case['config_sha256']
    assert not manifest['smoke'] and not manifest['dataset']['official_test_read']
    assert config['lab']['epochs'] == 30 and config['seed'] == 0
    assert config['evaluation']['official_test']['policy'] == 'disabled'
    used = run / 'config.used.json'
    if used.exists():
        assert without_dataset_root(read(used)) == without_dataset_root(config)
    epochs = [r for line in (run / 'metrics.jsonl').read_text().splitlines()
              if (r := json.loads(line)).get('kind') == 'epoch'] if (run / 'metrics.jsonl').exists() else []
    assert [e['epoch'] for e in epochs] == list(range(1, len(epochs) + 1))
    row.update(state=status['state'], epochs_completed=len(epochs), epochs_data=epochs,
               bundle=str(run.relative_to(ROOT)))
    if status.get('progress', {}).get('stage') == 'checkpointed_pause':
        row['state'] = 'paused'
    values = np.asarray([e['metrics']['validation_accuracy'] for e in epochs])
    assert np.isfinite(values).all()
    if len(values):
        row.update(latest_validation_pct=100 * float(values[-1]),
                   best_observed_validation_pct=100 * float(values.max()),
                   max_running_best_drawdown_pp=100 * float((np.maximum.accumulate(values) - values).max()))
    if row['state'] == 'failed':
        assert not (run / 'result.json').exists()
        errors = [line for log in sorted(task.glob('segments/chunk_*/training.log'))
                  for line in log.read_text().splitlines() if line.startswith('NonFiniteTrainingError:')]
        if errors:
            match = re.search(r'epoch=(\d+), batch=(\d+)', errors[-1])
            assert match, errors[-1]
            row.update(failure_kind='scientific_nonfinite', failure_detail=errors[-1],
                       failure_epoch=int(match[1]), failure_batch=int(match[2]))
        else:
            row.update(failure_kind='operational_failure', failure_detail=status.get('error'))
        return row
    if row['state'] != 'complete':
        return row
    assert len(epochs) == 30
    result, metrics = read(run / 'result.json'), read(run / 'metrics.json')
    assert result['completion']['criteria_met'] and metrics['official_test_evaluations'] == 0
    assert metrics['eqprop_endpoint_read_noise_draw_count'] == (825120 if case['sigma'] else 0)
    assert metrics['eqprop']['endpoint_read_noise_std'] == case['sigma']
    assert metrics['eqprop']['endpoint_read_noise_seed'] == 2026081601
    environments = []
    for chunk, code in enumerate((75, 75, 0)):
        segment = task / f'segments/chunk_{chunk}'
        summary = read(segment / 'summary.json')
        assert len(summary) == 1 and summary[0]['returncode'] == code
        assert summary[0]['epochs'] == 30 and not summary[0]['smoke']
        assert summary[0]['config_sha256'] == case['config_sha256']
        assert (segment / 'worker_started/exit_code').read_text().strip() == '0'
        environments.append(read(segment / 'gpu.json'))
    gpu = environments[0]
    assert 'V100' in gpu['name']
    assert all(all(e[k] == gpu[k] for k in ('name', 'torch', 'cuda')) for e in environments)
    for name in ('accuracy_train', 'accuracy_test', 'loss_train', 'loss_test'):
        data = np.load(run / f'{name}.npy', allow_pickle=False)
        assert len(data) == 30 and np.isfinite(data).all()
        if name == 'accuracy_test':
            assert np.allclose(data, values, rtol=0, atol=1e-12)
    import torch
    for role in ('best', 'final'):
        ck = torch.load(run / f'{role}_model.pt', map_location='cpu', weights_only=True)
        with np.load(run / f'weights_{role}.npz', allow_pickle=False) as weights:
            for spec, tensor in zip(ck['schema'], ck['states'], strict=True):
                name = spec['name'].strip()
                assert tensor.dtype == torch.float64 and torch.isfinite(tensor).all()
                assert np.array_equal(tensor.numpy(), weights[name])
                assert (torch.count_nonzero(tensor) == 0 if name.startswith('Bias_')
                        else ((tensor >= 0) & (tensor <= 100)).all())
    drop = 100 * float(values.max() - values[-1])
    row.update(final_validation_pct=100 * float(values[-1]), best_validation_pct=100 * float(values.max()),
               best_to_final_drop_pp=drop, passes_final_drop_screen=passes_stability_screen(values),
               dataset_provenance=metrics['dataset_provenance'], gpu_environment=gpu,
               initial_parameter_state_sha256=metrics['initial_parameter_state_sha256'])
    return row


def analyze():
    cases = read(STUDY / 'cases.json')
    assert len(cases) == 6 and [c['index'] for c in cases] == list(range(6))
    assert {(c['scheme'], c['sigma']) for c in cases} == {
        (s, sigma) for s in ('baseline', 'ours', 'legacy') for sigma in (0., .001)}
    rows = [collect(c) for c in cases]
    for row in rows[:3]:
        clean = rows[row['clean_index']]
        row['clean_state'] = clean['state']
        if row['state'] == clean['state'] == 'complete':
            for key in ('dataset_provenance', 'initial_parameter_state_sha256', 'gpu_environment'):
                assert row[key] == clean[key], key
            row['clean_relative_drop_pp'] = clean['final_validation_pct'] - row['final_validation_pct']
    payload = dict(updated_at=datetime.now(timezone.utc).isoformat(), expected=6,
                   completed=sum(r['state'] == 'complete' for r in rows),
                   terminal=sum(r['state'] in ('complete', 'failed') for r in rows), rows=rows)
    (STUDY / 'analysis.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    fields = ['scheme', 'beta', 'base_beta', 'sigma', 'state', 'epochs_completed',
              'final_validation_pct', 'best_validation_pct', 'clean_relative_drop_pp',
              'best_to_final_drop_pp', 'passes_final_drop_screen', 'max_running_best_drawdown_pp',
              'latest_validation_pct', 'best_observed_validation_pct', 'failure_kind', 'failure_epoch',
              'failure_batch', 'failure_detail', 'bundle']
    with REPORT.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    lines = ['# Conv3 p95: read noise 1e-3 versus clean', '',
             f"{payload['completed']}/6 completed; {payload['terminal']}/6 terminal. Updated {payload['updated_at']}.", '',
             'Thirty epochs, seed 0, fixed per-matrix cosine >.95 betas, T=K=8, float64 centered EqProp, '
             'unchanged Adam rates and matched V100 controls. MNIST 55k/5k validation; official test disabled. '
             'Endpoint noise leaves relaxation and validation clean. Finite-T residual caveat retained.', '',
             '| Scheme | Sigma | State | Epochs | Final / best validation (%) | Clean-relative drop (pp) |',
             '|---|---:|---|---:|---:|---:|']
    for r in rows:
        score = f"{r['final_validation_pct']:.2f} / {r['best_validation_pct']:.2f}" if r['state'] == 'complete' else '—'
        drop = f"{r['clean_relative_drop_pp']:+.2f}" if 'clean_relative_drop_pp' in r else '—'
        lines.append(f"| {r['scheme']} | {r['sigma']:g} | {r['state']} | {r['epochs_completed']} | {score} | {drop} |")
        if r.get('failure_detail'):
            lines += ['', f"{r['scheme']}, sigma {r['sigma']:g}: {r['failure_detail']}", '']
    lines += ['', 'The stability screen requires 30 finite epochs and final accuracy strictly less than 5pp '
              'below its own best. Temporary drawdown is reported separately; passing does not imply smooth '
              'training. Scientific failures remain included. If a clean control is unstable, its paired '
              'noisy failure cannot be attributed solely to noise. Single-seed validation diagnostics only.', '',
              'Historical p90/p99 runs used other GPU/runtime contracts and are contextual comparisons, '
              'not a controlled beta-only effect. No further noise level is scheduled.', '',
              '![Accuracy and loss trajectories](conv3_p95_read_noise_1em3_20260921_epochs.png)']
    REPORT.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey='row')
    for column, scheme in enumerate(('baseline', 'ours', 'legacy')):
        for r in [r for r in rows if r['scheme'] == scheme]:
            label = 'Clean' if r['sigma'] == 0 else 'Noise 1e-3'
            color = '.45' if r['sigma'] == 0 else 'tab:blue'
            epochs = r['epochs_data']
            for ax, key, scale in ((axes[0, column], 'validation_accuracy', 100),
                                   (axes[1, column], 'validation_loss', 1)):
                data = [(e['epoch'], e['metrics'][key] * scale) for e in epochs if key in e['metrics']]
                if data: ax.plot(*zip(*data), color=color, label=label)
                if r.get('failure_epoch'): ax.axvline(r['failure_epoch'], color=color, linestyle=':')
                ax.grid(alpha=.2); ax.set_xlim(1, 30)
        axes[0, column].set_title(scheme.capitalize()); axes[1, column].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Validation accuracy (%)'); axes[1, 0].set_ylabel('Validation loss')
    if axes[0, 0].lines: axes[0, 0].legend()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(REPORT.with_name(REPORT.name + '_epochs.' + ext), dpi=180)
    plt.close(fig)
    print(json.dumps({k: v for k, v in payload.items() if k != 'rows'}))


if __name__ == '__main__':
    analyze()
