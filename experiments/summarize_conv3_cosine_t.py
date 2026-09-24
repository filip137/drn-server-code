"""Summarize the requested four-T trained-checkpoint cosine diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'results/eqprop-conv3-cosine-t-transition-20260922-v1'
CELL = 'A100_ours_sigma0.0005_factor_1'
PARAMETERS = ['ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0']
LABELS = ['Conv1', 'Conv2', 'Conv3', 'Readout']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=STUDY)
    args = parser.parse_args()
    study = args.study_root.resolve()
    sources = {
        8: ROOT / 'results/eqprop-conv3-trained-beta-noise-20260922-v1/collected/runs' / CELL,
        **{t: study / 'collected' / f'T{t}' / 'runs' / CELL for t in [12, 16, 20, 32]},
        64: ROOT / 'results/eqprop-conv3-trained-beta-noise-t64-20260922-v1/collected/runs' / CELL,
    }
    analysis = study / 'analysis'
    analysis.mkdir(exist_ok=True)
    records, batches, measurements, provenance = [], {}, {}, []
    source_hash = None
    for t, run in sources.items():
        assert not validate_run(run), run
        rows = list(csv.DictReader((run / 'layer_metrics.csv').open()))
        assert len(rows) == 720
        identities = set()
        for row in rows:
            assert int(row['T']) == t and int(row['K']) == 8
            assert float(row['beta_factor']) == 1
            assert float(row['injected_beta']) == 5.26875648112
            assert row['checkpoint'] == 'A100_ours_sigma0.0005'
            assert int(row['checkpoint_epoch']) == 30
            source_hash = source_hash or row['checkpoint_sha256']
            assert row['checkpoint_sha256'] == source_hash
            key = int(row['batch_index'])
            identity = (row['batch_payload_sha256'], row['batch_source_indices_sha256'])
            assert batches.setdefault(key, identity) == identity
            draw = int(row['noise_draw_index'])
            condition = 'clean' if draw == 0 else 'noisy'
            assert float(row['endpoint_read_noise_std']) == (0 if draw == 0 else .0005)
            expected_seed = '' if draw == 0 else str([2026092101, 2026092111, 2026092121, 2026092131][draw - 1])
            assert row['endpoint_read_noise_seed'] == expected_seed
            cell_identity = (key, draw, row['parameter_name'])
            assert cell_identity not in identities
            identities.add(cell_identity)
            value = float(row['cosine'])
            assert np.isfinite(value)
            measurements.setdefault((t, row['parameter_name'], condition), []).append(value)
        provenance.append(dict(T=t, run=str(run.relative_to(ROOT)), rows=len(rows),
                               result_sha256=hashlib.sha256((run / 'result.json').read_bytes()).hexdigest()))
    assert len(batches) == 36
    for t in sources:
        for parameter, label in zip(PARAMETERS, LABELS):
            for condition in ['clean', 'noisy']:
                values = np.asarray(measurements[t, parameter, condition])
                assert len(values) == (36 if condition == 'clean' else 144)
                ref = np.median(measurements[64, parameter, condition])
                records.append(dict(T=t, K=8, injected_beta=5.26875648112,
                    read_noise_std=0 if condition == 'clean' else .0005,
                    parameter_name=parameter, layer=label, condition=condition,
                    comparisons=len(values), median_cosine=float(np.median(values)),
                    p10_cosine=float(np.quantile(values, .1)), p90_cosine=float(np.quantile(values, .9)),
                    minimum_cosine=float(values.min()), positive_fraction=float((values > 0).mean()),
                    median_difference_from_T64=float(np.median(values) - ref)))
    with (analysis / 'cosine_vs_T.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    for t in [12, 16, 20, 32]:
        assert not validate_run(study / 'collected' / f'T{t}' / 'smoke' / CELL)
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    for ax, parameter, label in zip(axes.flat, PARAMETERS, LABELS):
        for condition, color, linestyle in [('noisy', '#1f77b4', '-'), ('clean', '#e1812c', '--')]:
            selected = [r for r in records if r['parameter_name'] == parameter and r['condition'] == condition]
            ts = [r['T'] for r in selected]
            ax.plot(ts, [r['median_cosine'] for r in selected], color=color, linestyle=linestyle,
                    marker='o', label='Noisy median' if condition == 'noisy' else 'Clean median')
            if condition == 'noisy':
                ax.fill_between(ts, [r['p10_cosine'] for r in selected],
                                [r['p90_cosine'] for r in selected], color=color, alpha=.13,
                                label='Noisy 10–90% range')
        ax.axhline(0, color='gray', linewidth=.8)
        ax.set(title=label, ylabel='EP–BPTT cosine', xticks=list(sources), ylim=(-1.03, 1.03))
        ax.grid(alpha=.18)
    for ax in axes[-1]:
        ax.set_xlabel('Free-phase iterations T')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    figure.suptitle('Ours · epoch 30 · training/read σ = 5×10⁻⁴\n'
                   'Training β = 5.26876 · K = 8 · same 576 examples', y=.94)
    figure.tight_layout(rect=[0, 0, 1, .87])
    for suffix in ['png', 'pdf']:
        figure.savefig(analysis / f'cosine_vs_T.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(figure)
    def selected(t, parameter, condition='noisy'):
        return next(r for r in records if r['T'] == t and r['parameter_name'] == parameter
                    and r['condition'] == condition)
    report = ['# Conv3 cosine changes across T', '',
        'Ours, final epoch-30 seed-0 checkpoint, training/read sigma 5e-4. '
        'Injected beta stays at its training value 5.26875648112; K stays at 8.', '',
        'The four requested new measurements use T=12/16/20/32. T8 and T64 are '
        'validated cached anchors from the same Akib environment and identical cohort.', '',
        '| T | Conv1 | Conv2 | Conv3 | Readout | Readout positive fraction |',
        '|---:|---:|---:|---:|---:|---:|']
    for t in sources:
        values = [selected(t, p)['median_cosine'] for p in PARAMETERS]
        fraction = selected(t, 'DenseWeight_0')['positive_fraction']
        report.append(f'| {t} | ' + ' | '.join(f'{v:.6f}' for v in values) + f' | {fraction:.1%} |')
    report += ['', 'The table shows median noisy layer cosines across 36 batches × four matched '
        'noise draws. The positive fraction is over those 144 readout comparisons. '
        'Clean controls use the same 36 batches. Shading shows the 10–90% distribution, '
        'not a confidence interval. Connecting lines do not add measurements at intermediate T.', '',
        '![Cosine versus T](cosine_vs_T.png)', '',
        '[Measurements](cosine_vs_T.csv) · [PDF](cosine_vs_T.pdf) · [Validation](validation.json)', '',
        'All checkpoint, cohort, seed, beta and K identities match. New coverage is '
        'four production cells (2,880 layer comparisons) and four separate smokes. '
        'No optimizer step, accuracy evaluation or official-test read occurs. '
        'This diagnostic does not establish an exact minimum T between sampled values.']
    (analysis / 'report.md').write_text('\n'.join(report) + '\n')
    validation = dict(state='complete', new_production_cells=4, new_smoke_cells=4,
        new_layer_comparisons=2880, cached_anchor_cells=2, total_layer_comparisons=4320,
        matched_batches=len(batches), checkpoint_sha256=source_hash, sources=provenance,
        official_test_read=False, optimizer_steps_applied=False, accuracy_evaluation=False)
    (analysis / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print(json.dumps({'rows': len(records), 'noisy_medians': {
        t: [selected(t, p)['median_cosine'] for p in PARAMETERS] for t in sources}}, indent=2))


if __name__ == '__main__':
    main()
