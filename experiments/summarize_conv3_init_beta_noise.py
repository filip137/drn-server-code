"""Validate and plot the fixed-initialization Conv3 beta/read-noise replay."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.reporting import sha256_file, validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'results/eqprop-conv3-init-beta-noise-20260921-v1'
CONFIG = ROOT / 'configs/conv/eqprop_conv3_init_beta_noise_20260921.json'
STEM = 'conv3_init_beta_noise_20260921'
PARAMETERS = ('ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0')
LABELS = ('Conv layer 1', 'Conv layer 2', 'Conv layer 3', 'Dense output')
SCHEMES = ('baseline', 'legacy', 'ours')
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7')


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_values(row, name, values):
    values = [float(v) for v in values if v not in ('', None)]
    assert all(np.isfinite(values))
    row[name + '_defined_count'] = len(values)
    for label, q in (('min', 0), ('p10', .1), ('median', .5), ('p90', .9), ('max', 1)):
        row[name + '_' + label] = float(np.quantile(values, q)) if values else None


def main():
    cfg = json.loads(CONFIG.read_text())
    execution = json.loads((STUDY / 'summary.json').read_text())
    assert execution['state'] == 'complete' and len(execution['cases']) == 6
    assert execution['config_sha256'] == sha256_file(CONFIG)
    assert execution['runner_sha256'] == sha256_file(STUDY / 'source/replay_conv3_init_beta_noise.py')
    analysis = STUDY / 'analysis'
    analysis.mkdir(exist_ok=True)
    plots = analysis / 'plots'
    plots.mkdir(exist_ok=True)
    paper = ROOT / 'paper_ready_results'
    all_rows, signal_rows, residual_rows, sources = [], [], [], []
    noise_signatures = {}
    for declared in cfg['cases']:
        run = STUDY / 'runs' / declared['name']
        assert not validate_run(run), run
        result = json.loads((run / 'result.json').read_text())
        m = result['terminal_metrics']
        assert m['replay_batches'] == 36 and m['layer_comparisons'] == 1008
        assert m['noise_tensor_draws'] == 1728 and m['checkpoint_epoch'] == 0
        for key in ('official_test_read', 'optimizer_steps_applied', 'accuracy_evaluation'):
            assert not m[key]
        for key in ('source_bytes_unchanged', 'all_parameter_tensors_unchanged', 'all_bias_tensors_exact_zero'):
            assert m[key]
        assert sha256_file(run / 'checkpoints/initialization.pt') == cfg['initializer_checkpoint_sha256']
        rows = read_csv(run / 'layer_metrics.csv')
        expected = {(i, sigma, p) for i in range(36) for sigma in cfg['noise_sigmas'] for p in PARAMETERS}
        assert len(rows) == len(expected) == 1008
        assert {(int(r['batch_index']), float(r['readout_sigma']), r['parameter_name']) for r in rows} == expected
        assert {int(r['noise_draw_index']) for r in rows} == {0}
        assert {r['checkpoint_role'] for r in rows} == {'reconstructed_initialization'}
        assert {float(r['injected_beta']) for r in rows} == {declared['injected_beta']}
        for r in rows:
            if float(r['readout_sigma']) == 0:
                assert float(r['added_noise_rms']) == 0
                if r['noisy_clean_eqprop_cosine']:
                    assert abs(float(r['noisy_clean_eqprop_cosine']) - 1) < 1e-12
        all_rows.extend(rows)
        signal_rows.extend(read_csv(run / 'state_signal.csv'))
        residual_rows.extend(dict(r, injected_beta=declared['injected_beta'])
                             for r in read_csv(run / 'equilibrium_residuals.csv'))
        for r in read_csv(run / 'endpoint_read_noise.csv'):
            key = (r['batch_index'], r['endpoint_read_noise_std'], r['phase'], r['state_layer_index'])
            assert noise_signatures.setdefault(key, r['standard_normal_sha256']) == r['standard_normal_sha256']
        sources.append(dict(case=declared['name'], run=str(run.relative_to(ROOT)),
                            result_sha256=sha256_file(run / 'result.json'), elapsed_seconds=m['elapsed_seconds']))
    assert len(all_rows) == cfg['expected_layer_comparisons'] == 6048
    bptt_hashes = {}
    groups = defaultdict(list)
    for r in all_rows:
        key = (r['scheme'], r['batch_index'], r['parameter_name'])
        assert bptt_hashes.setdefault(key, r['bptt_gradient_sha256']) == r['bptt_gradient_sha256']
        groups[r['scheme'], float(r['injected_beta']), float(r['readout_sigma']), r['parameter_name']].append(r)
    summary = []
    metrics = ('cosine', 'clean_eqprop_bptt_cosine', 'noisy_clean_eqprop_cosine',
               'eqprop_over_bptt_norm_ratio', 'relative_l2_difference_over_bptt',
               'bptt_rms', 'clean_eqprop_rms', 'eqprop_rms', 'added_noise_rms', 'noise_over_clean_norm',
               'bptt_near_zero_fraction', 'clean_eqprop_near_zero_fraction', 'eqprop_near_zero_fraction')
    for (scheme, beta, sigma, parameter), rows in sorted(groups.items()):
        assert len(rows) == 36
        row = dict(scheme=scheme, injected_beta=beta, base_beta=float(rows[0]['base_beta']),
                   readout_sigma=sigma, parameter=parameter, batches=36, draws_per_batch=1, checkpoint_epoch=0)
        for metric in metrics:
            summarize_values(row, metric, [r[metric] for r in rows])
        row['fraction_cosine_above_0.90'] = sum(r['cosine'] != '' and float(r['cosine']) > .90 for r in rows) / 36
        row['all_endpoint_residual_gates_passed'] = all(r['all_endpoint_residual_gates_passed'] == 'True' for r in rows)
        summary.append(row)
    assert len(summary) == 168
    signal_groups = defaultdict(list)
    for row in signal_rows:
        signal_groups[row['scheme'], float(row['injected_beta']), int(row['state_layer_index'])].append(row)
    signals = []
    for (scheme, beta, layer), rows in sorted(signal_groups.items()):
        assert len(rows) == 36
        row = dict(scheme=scheme, injected_beta=beta, state_layer_index=layer,
                   state_layer_name=rows[0]['state_layer_name'], batches=36)
        count = sum(int(r['element_count']) for r in rows)
        for metric in ('free_rms', 'positive_free_rms', 'negative_free_rms', 'centered_half_difference_rms'):
            row[metric + '_pooled'] = float(np.sqrt(sum(float(r[metric])**2*int(r['element_count']) for r in rows) / count))
        signals.append(row)
    residuals = []
    for scheme in SCHEMES:
        for beta in (.01, .1):
            rr = [r for r in residual_rows if r['scheme'] == scheme and r['injected_beta'] == beta]
            residuals.append(dict(scheme=scheme, injected_beta=beta, residual_rows=len(rr),
                failed_rows=sum(r['gate_passed'] != 'True' for r in rr),
                maximum_batch_layer_p90=max(float(r['selected_max_p90']) for r in rr)))
    write_csv(analysis / 'all_layer_metrics.csv', all_rows)
    write_csv(analysis / 'summary.csv', summary)
    write_csv(analysis / 'state_signal_summary.csv', signals)
    write_csv(analysis / 'residual_summary.csv', residuals)
    for name, rows in [('summary', summary), ('state_signals', signals), ('residuals', residuals)]:
        write_csv(paper / f'{STEM}_{name}.csv', rows)

    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    sigmas = cfg['noise_sigmas']
    tick_labels = ['0', '$10^{-5}$', '$3\\times10^{-5}$', '$10^{-4}$',
                   '$3\\times10^{-4}$', '$5\\times10^{-4}$', '$10^{-3}$']

    def make_panels(metric, suffix, title, ylabel, clean_controls=False):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.8), sharex=True, sharey=True)
        for i, beta in enumerate((.01, .1)):
            for j, scheme in enumerate(SCHEMES):
                ax = axes[i, j]
                for parameter, label, color in zip(PARAMETERS, LABELS, COLORS):
                    rr = sorted((r for r in summary if r['scheme'] == scheme and r['injected_beta'] == beta
                                 and r['parameter'] == parameter), key=lambda r:r['readout_sigma'])
                    ax.plot(sigmas, [r[metric+'_median'] for r in rr], 'o-', lw=2, ms=4.5, color=color, label=label)
                    ax.fill_between(sigmas, [r[metric+'_p10'] for r in rr], [r[metric+'_p90'] for r in rr], color=color, alpha=.14)
                    if clean_controls:
                        ax.axhline(rr[0]['clean_eqprop_bptt_cosine_median'], color=color, ls=':', lw=1, alpha=.75)
                ax.axhline(.9, color='#777777', ls='--', lw=1)
                ax.axhline(0, color='#777777', lw=.8)
                ax.set_title(f'{scheme.capitalize()}  ·  injected β = {beta:g}')
                ax.set_xscale('symlog', linthresh=1e-5)
                ax.set_xticks(sigmas, tick_labels, rotation=48, ha='right')
                ax.set_ylim(-1.02, 1.055)
                ax.set_yticks([-1, -.5, 0, .5, .9, 1])
                ax.grid(alpha=.12)
                if i == 1:
                    ax.set_xlabel('Endpoint read-noise σ')
                if j == 0:
                    ax.set_ylabel(ylabel)
        handles = [Line2D([], [], color=c, marker='o', lw=2, label=l) for c,l in zip(COLORS, LABELS)]
        handles.append(Line2D([], [], color='#777777', ls='--', label='Cosine 0.90'))
        fig.suptitle(title, fontsize=19, y=.988)
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .95), ncol=5, frameon=False)
        footer = ('Shared seed-0 initialization · 36 matched batches of 16 · one matched noise draw per batch and σ\n'
                  'Median and 10–90% batch range · T = K = 8 · float64 · RTX 3090 · no training or accuracy evaluation')
        if clean_controls:
            footer += '\nDotted horizontal lines: clean EP–BPTT median at the same initialization.'
        fig.text(.5, .008, footer, ha='center', va='bottom', fontsize=10, color='#555555')
        fig.tight_layout(rect=(0, .087 if clean_controls else .073, 1, .895))
        for extension in ('png', 'pdf', 'jpg'):
            fig.savefig(plots / f'{STEM}{suffix}.{extension}', dpi=180)
            fig.savefig(paper / f'{STEM}{suffix}.{extension}', dpi=180)
        plt.close(fig)

    make_panels('cosine', '', 'Conv3 at initialization: gradient alignment versus read noise',
                'Noisy EP–BPTT cosine', clean_controls=True)
    make_panels('noisy_clean_eqprop_cosine', '_readout_only',
                'Conv3 at initialization: effect of readout noise on EP gradients', 'Noisy–clean EP cosine')

    lines = ['# Conv3 initialization: beta and endpoint read noise', '',
        'Read-only diagnostic requested September21,2026. All six cases completed at the identical saved '
        'seed-0 initialization (epoch0). No trained checkpoint, optimizer step, accuracy evaluation or official-test access.', '',
        'Injected beta B is0.01/0.1 for baseline(v1/c1),legacy(v4/c0.25),ours(v4/c1). Base beta=B/(v/c)^3. '
        'T=K=8; centered frozen-current EP; float64; explicit perfect diodes; input gain360; zero biases; '
        'wide[0,100] weights. The initializer was stored in float32 and promoted exactly to float64, '
        'matching the prior initialization replay. The BPTT reference differentiates K zero-nudge steps from the same post-T state.', '',
        'The exact prior36 batches of16 validation examples are reused (576 examples from the55k/5k MNIST split). '
        'One endpoint-noise draw per batch/sigma uses seed2026092101, matched across schemes and beta; the same '
        'underlying standard normals are reused across sigma. Noise affects copied endpoint voltages only; '
        'relaxation and inputs are clean. Positive/negative phases and state layers have independent samples.', '',
        f'![Gradient cosine]({STEM}.png)', '',
        '## Per-layer median noisy EP–BPTT cosine', '',
        '| Injected beta | Scheme | Readout sigma | Conv1 | Conv2 | Conv3 | Dense |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for beta in (.01, .1):
        for sigma in sigmas:
            for scheme in SCHEMES:
                rr = [next(r for r in summary if r['scheme']==scheme and r['injected_beta']==beta
                           and r['readout_sigma']==sigma and r['parameter']==p) for p in PARAMETERS]
                lines.append(f'| {beta:g} | {scheme} | {sigma:g} | ' + ' | '.join(
                    f"{r['cosine_median']:.6f}" if r['cosine_median'] is not None else 'undefined' for r in rr) + ' |')
    lines += ['', '## Physical centered phase response at initialization', '',
        'Pooled RMS((v_plus-v_minus)/2) in simulator voltage units. These physical states are unchanged across the readout-noise sweep.', '',
        '| Injected beta | Scheme | Hidden1 | Hidden2 | Hidden3 | Output |',
        '|---:|---|---:|---:|---:|---:|']
    for beta in (.01, .1):
        for scheme in SCHEMES:
            rr = sorted((r for r in signals if r['scheme']==scheme and r['injected_beta']==beta), key=lambda r:r['state_layer_index'])
            lines.append(f'| {beta:g} | {scheme} | ' + ' | '.join(f"{r['centered_half_difference_rms_pooled']:.6g}" for r in rr) + ' |')
    lines += ['', '## Coverage and limits', '',
        f"All6 bundles validate, with6048 comparisons and10368 endpoint-noise tensor draws. Checkpoint bytes, "
        f"in-memory parameters, zero biases, shared cohorts, BPTT invariance across beta and matched noise samples pass. "
        f"Replay including smoke took{execution['total_including_smoke_seconds']/60:.2f} minutes on one local RTX3090.", '',
        f"Undefined cosines: {sum(r['cosine']=='' for r in all_rows)}. "
        f"Residual failures: {sum(r['failed_rows'] for r in residuals)}/{len(residual_rows)} batch/layer/phase rows. "
        'Residuals are separate from gradient fidelity; finite-T/K BPTT is not an exact-equilibrium guarantee.', '',
        'Medians and10–90% ranges summarize batches with one draw each; they are not confidence intervals '
        'or within-batch Monte Carlo variance estimates. This is one initializer and two beta values, '
        'not a training-performance or universal scheme-ranking result. Equal injected beta does not '
        'equalize physical output displacement. No accuracy result is inferred.', '',
        f"Initializer SHA256: `{cfg['initializer_checkpoint_sha256']}`. "
        f"Cohort SHA256: `{execution['cohort_sha256']}`.", '',
        f'[Layer summary CSV]({STEM}_summary.csv) · [PDF]({STEM}.pdf) · '
        f'[Readout-only figure]({STEM}_readout_only.png) · [Voltage response CSV]({STEM}_state_signals.csv) · '
        '[Raw bundles](../results/eqprop-conv3-init-beta-noise-20260921-v1/)']
    (paper / f'{STEM}.md').write_text('\n'.join(lines)+'\n')
    (analysis / 'report.md').write_text('\n'.join(lines).replace(
        f']({STEM}', f'](../../../paper_ready_results/{STEM}').replace(
        '](../results/eqprop-conv3-init-beta-noise-20260921-v1/)', '](../)')+'\n')
    validation = dict(state='complete', expected_cases=6, complete_cases=6, expected_comparisons=6048,
        measured_comparisons=len(all_rows), summary_rows=len(summary), all_bundles_valid=True,
        all_initializer_hashes_match=True, bptt_invariant_across_beta=True, noise_draws_matched=True,
        undefined_cosines=sum(r['cosine']=='' for r in all_rows), sources=sources,
        cohort_sha256=execution['cohort_sha256'], elapsed_seconds=execution['total_including_smoke_seconds'],
        failures=[], exclusions=['six one-batch operational smokes excluded from scientific summaries'],
        official_test_read=False, optimizer_steps_applied=False, accuracy_evaluation=False)
    (analysis / 'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
