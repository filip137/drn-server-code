"""Summarize final-checkpoint gradient replays, retaining all hardware controls."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.reporting import validate_run, sha256_file

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'results/eqprop-conv3-p90-final-noise-cosine-20260921-v1'
OUT = ROOT / 'paper_ready_results'
STEM = 'conv3_p90_final_noise_cosine_20260921'
PARAMETERS = ('ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0')
SCHEMES = ('baseline', 'legacy', 'ours')


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = json.loads((ROOT / 'configs/conv/eqprop_conv3_p90_final_noise_cosine_20260921.json').read_text())
    all_rows, sources = [], []
    for c in cfg['cases']:
        p = STUDY / 'runs' / c['name']
        assert not validate_run(p), p
        result = json.loads((p / 'result.json').read_text())
        m = result['terminal_metrics']
        assert m['replay_batches'] == 36 and m['checkpoint_epoch'] == 30
        assert not m['official_test_read'] and not m['optimizer_steps_applied']
        assert m['source_bytes_unchanged'] and m['all_parameter_tensors_unchanged']
        rows = read_csv(p / 'layer_metrics.csv')
        assert len(rows) == 36 * 4 * (5 if c['sigma'] > 0 else 1)
        all_rows.extend(rows)
        sources.append(dict(run=str(p.relative_to(ROOT)), result_sha256=sha256_file(p / 'result.json'),
                            source_bundle=c['bundle'], checkpoint_sha256=c['source_hashes']['final_model.pt'],
                            elapsed_seconds=m['elapsed_seconds']))
    assert len(sources) == 25
    groups = defaultdict(list)
    for r in all_rows:
        groups[r['scheme'], r['training_gpu'], float(r['training_sigma']),
               float(r['endpoint_read_noise_std']), r['parameter_name']].append(r)
    summary = []
    for (scheme, gpu, sigma, read_sigma, parameter), rows in sorted(groups.items()):
        n = 144 if read_sigma else 36
        assert len(rows) == n
        row = dict(scheme=scheme, training_gpu=gpu, training_sigma=sigma,
                   readout_sigma=read_sigma, parameter=parameter, comparisons=n, batches=36,
                   injected_beta=float(rows[0]['injected_beta']), checkpoint_epoch=30,
                   source_bundle=rows[0]['source_bundle'])
        for metric in ('cosine', 'clean_eqprop_bptt_cosine', 'noisy_clean_eqprop_cosine',
                       'eqprop_over_bptt_norm_ratio', 'relative_l2_difference_over_bptt', 'bptt_rms', 'eqprop_rms'):
            values = [float(r[metric]) for r in rows if r[metric] != '']
            row[metric + '_defined_count'] = len(values)
            for label, q in (('min', 0), ('p10', .1), ('median', .5), ('p90', .9), ('max', 1)):
                row[metric + '_' + label] = float(np.quantile(values, q)) if values else None
        row['all_endpoint_residual_gates_passed'] = all(r['all_endpoint_residual_gates_passed'] == 'True' for r in rows)
        for threshold in (.90, .99):
            row[f'fraction_cosine_above_{threshold:.2f}'] = sum(
                r['cosine'] != '' and float(r['cosine']) > threshold for r in rows) / len(rows)
        summary.append(row)
    write_csv(OUT / f'{STEM}_summary.csv', summary)
    write_csv(STUDY / 'all_layer_metrics.csv', all_rows)
    # The displayed zero-noise point uses the existing RTX5090 control;
    # all A100/V100 zero controls are retained in the full summary.
    primary = [r for r in summary if r['readout_sigma'] == r['training_sigma']
               and (r['training_sigma'] > 0 or r['training_gpu'] == 'RTX5090')]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    colors = plt.get_cmap('tab10').colors
    for ax, scheme in zip(axes, SCHEMES):
        for i, parameter in enumerate(PARAMETERS):
            points = sorted((r for r in primary if r['scheme'] == scheme and r['parameter'] == parameter),
                            key=lambda r:r['training_sigma'])
            x = [r['training_sigma'] for r in points]
            ax.plot(x, [r['cosine_median'] for r in points], 'o-', color=colors[i], label=['Conv 1','Conv 2','Conv 3','Dense'][i])
            ax.fill_between(x, [r['cosine_p10'] for r in points], [r['cosine_p90'] for r in points], color=colors[i], alpha=.12)
            ax.plot(x, [r['clean_eqprop_bptt_cosine_median'] for r in points], ':', color=colors[i], alpha=.8)
        ax.set_xscale('symlog', linthresh=1e-5)
        ax.set_xticks([0,1e-5,3e-5,1e-4,3e-4,5e-4,1e-3], ['0','1e−5','3e−5','1e−4','3e−4','5e−4','1e−3'], rotation=45)
        ax.set_title(scheme.capitalize())
        ax.set_xlabel('Training and replay noise σ')
        ax.grid(alpha=.2)
        if scheme != 'baseline':
            ax.text(.97,.06,'σ=1e−3: no final checkpoint',transform=ax.transAxes,ha='right',fontsize=8)
    axes[0].set_ylabel('EP–BPTT cosine at epoch 30')
    axes[0].legend(fontsize=8)
    fig.suptitle('Conv3 p90 final checkpoints: median cosine, shaded 10–90% range\nSolid: noisy EP; dotted: clean EP at the same checkpoint. All replays on RTX3090.')
    fig.tight_layout()
    fig.savefig(OUT / f'{STEM}.png', dpi=180)
    fig.savefig(OUT / f'{STEM}.pdf')
    lines = ['# Conv3 p90: final-checkpoint gradient cosine versus read noise', '',
        'Measured by read-only replay of 25 existing epoch-30 checkpoints; no training or official-test evaluation. '
        'Seed 0; beta baseline404.141105702, legacy4.42250110273, ours5.26875648112; T=K=8, '
        'float64 centered frozen-current EP. The BPTT reference differentiates exactly K zero-nudge iterations '
        'from the same post-T state. Gradients use the training amplification normalization, with no Adam or LR transformation.', '',
        'Every checkpoint uses the same 36 validation batches of16. At nonzero sigma, four independent endpoint-noise '
        'draws per batch give144 comparisons per weight matrix; zero noise gives36. Standard-normal draws are '
        'matched across cases, while positive/negative phases and non-input layers use independent draws. '
        'Noise changes endpoint readout only. Checkpoint bytes, native float64 loading against the saved NPZ, '
        'zero biases and unchanged parameters are verified. All replays use the local RTX3090.', '',
        '## Per-layer median cosine: noisy EP versus clean BPTT', '',
        '| Training σ | Scheme | Training GPU | Conv1 | Conv2 | Conv3 | Dense |',
        '|---:|---|---|---:|---:|---:|---:|']
    for sigma in (0,1e-5,3e-5,1e-4,3e-4,5e-4,1e-3):
        for scheme in SCHEMES:
            points = [next((r for r in primary if r['training_sigma']==sigma and r['scheme']==scheme and r['parameter']==p), None) for p in PARAMETERS]
            if points[0] is None:
                lines.append(f'| {sigma:g} | {scheme} | RTX5090 | — | — | — | — |')
            else:
                lines.append(f"| {sigma:g} | {scheme} | {points[0]['training_gpu']} | " +
                             ' | '.join(f"{r['cosine_median']:.6f}" if r['cosine_median'] is not None else 'undefined' for r in points) + ' |')
    lines += ['', 'Legacy and ours at sigma1e-3 became nonfinite during epochs8 and18 respectively. '
              'They have no epoch-30 checkpoint, so no end-of-training gradient is substituted from a best or earlier checkpoint.', '',
              '## Clean EP at the same trained checkpoints', '',
              '| Training σ | Scheme | Conv1 | Conv2 | Conv3 | Dense |',
              '|---:|---|---:|---:|---:|---:|']
    for sigma in (0,1e-5,3e-5,1e-4,3e-4,5e-4,1e-3):
        for scheme in SCHEMES:
            points = [next((r for r in primary if r['training_sigma']==sigma and r['scheme']==scheme and r['parameter']==p), None) for p in PARAMETERS]
            if points[0] is not None:
                lines.append(f"| {sigma:g} | {scheme} | " + ' | '.join(f"{r['clean_eqprop_bptt_cosine_median']:.6f}" for r in points) + ' |')
    lines += ['', 'These are medians, not all-batch minima. The CSV includes minima, p10/p90, norm ratios, '
              'relative errors, gradient RMS and defined-cosine counts. Zero directions remain undefined. '
              'The original beta qualification tested initialization and a saved BPTT checkpoint, not these final '
              'EP-trained checkpoints. Its >.90 threshold is not guaranteed along training.', '',
              'Training noise changes the learned weights; thus the solid curves combine trajectory changes and '
              'readout noise. The dotted controls isolate readout noise at each fixed checkpoint. Training used '
              'V100 at1e-5, RTX5090 at3e-5/1e-3, and A100 at1e-4/3e-4/5e-4. The displayed zero point is RTX5090; '
              'all nine GPU-specific clean controls remain in the CSV. Lines guide the eye and do not establish '
              'a causal hardware-independent noise curve. Four draws quantify only limited Monte Carlo variability. '
              'Finite T/K residuals are reported separately; this is not an exact-equilibrium gradient guarantee.', '',
              f'![Cosine versus noise]({STEM}.png)', '',
              f'[Summary CSV]({STEM}_summary.csv) · [PDF]({STEM}.pdf) · '
              '[Source bundles and raw replay](../results/eqprop-conv3-p90-final-noise-cosine-20260921-v1/)']
    (OUT / f'{STEM}.md').write_text('\n'.join(lines)+'\n')
    (STUDY / 'analysis.json').write_text(json.dumps(dict(expected=25,completed=25,summary_cells=len(summary),
        comparisons=len(all_rows),sources=sources,production_gpu_hours=sum(s['elapsed_seconds'] for s in sources)/3600,
        exclusions=['legacy sigma1e-3: nonfinite epoch8','ours sigma1e-3: nonfinite epoch18']),indent=2)+'\n')
    print('\n'.join(lines[:45]))


if __name__ == '__main__':
    main()
