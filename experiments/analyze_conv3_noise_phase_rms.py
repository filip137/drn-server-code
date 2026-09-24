"""Relate saved endpoint-read gradient noise to the measured phase-voltage RMS."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import (append_metric, atomic_write_json, complete_run,
                                   fail_run, sha256_file, start_run, validate_run)
from experiments.summarize_conv3_trained_beta_noise import read_csv, write_csv, quantiles

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Conv1', 'Conv2', 'Conv3', 'Readout']
COLORS = ['#3366aa', '#dd7722', '#559944']


def crossover(points):
    """Interpolate only a bracketed median noise/clean-gradient ratio of one."""
    points = sorted(points, key=lambda r: r['beta'])
    for low, high in zip(points, points[1:]):
        a, b = low['noise_ratio_median'], high['noise_ratio_median']
        if a >= 1 >= b and a != b:
            fraction = math.log(a)/(math.log(a)-math.log(b))
            lerp = lambda key: math.exp(math.log(low[key]) + fraction*math.log(high[key]/low[key]))
            return dict(status='bracketed_interpolation', beta_low=low['beta'], beta_high=high['beta'],
                        ratio_low=a, ratio_high=b, beta_estimate=lerp('beta'),
                        phase_rms_low=low['centered_half_difference_rms_median'],
                        phase_rms_high=high['centered_half_difference_rms_median'],
                        phase_rms_estimate=lerp('centered_half_difference_rms_median'),
                        positive_free_rms_estimate=lerp('positive_free_rms_median'),
                        negative_free_rms_estimate=lerp('negative_free_rms_median'))
    return dict(status='noise_dominates_entire_grid' if all(r['noise_ratio_median'] > 1 for r in points)
                else 'signal_dominates_entire_grid', beta_low=points[0]['beta'], beta_high=points[-1]['beta'],
                ratio_low=points[0]['noise_ratio_median'], ratio_high=points[-1]['noise_ratio_median'],
                beta_estimate=None, phase_rms_low=points[0]['centered_half_difference_rms_median'],
                phase_rms_high=points[-1]['centered_half_difference_rms_median'],
                phase_rms_estimate=None, positive_free_rms_estimate=None, negative_free_rms_estimate=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    study = ROOT/cfg['output_root']
    out = study/'analysis/measurements'
    assert cfg['study_id'] in (ROOT/'docs/current_simulations.md').read_text()
    started = time.monotonic()
    sources, cases = [], None
    for path, k in zip(cfg['source_configs'], [cfg['main_K'], cfg['control_K']]):
        source = json.loads((ROOT/path).read_text())
        cases = source['cases'] if cases is None else cases
        assert source['cases'] == cases
        for case in source['cases']:
            for beta in source['injected_betas']:
                name = f'{case["name"]}_T{cfg["T"]}K{k}_beta_'+format(beta, '.12g').replace('.', 'p')
                run = ROOT/source['output_root']/'collected/runs'/name
                assert not validate_run(run), run
                sources.append(dict(case=case, beta=beta, K=k, run=str(run.relative_to(ROOT)),
                                    result_sha256=sha256_file(run/'result.json')))
    assert len(sources) == 30
    start_run(out, dict(study_id=cfg['study_id'], run_id='saved_rms_analysis', arm_id='saved_rms_analysis',
        evidence_class=cfg['evidence_class'], evidence_tier=cfg['evidence_tier'],
        dataset='ordinary_mnist_validation', configuration=dict(resolved=cfg, sha256=sha256_file(args.config)),
        inputs=sources, command=[sys.executable, *sys.argv],
        git=dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), dirty_worktree=True,
                 analyzer_sha256=sha256_file(Path(__file__))),
        runtime=dict(target='local:CPU', host=socket.gethostname(), python=platform.python_version()),
        official_test_read=False, optimizer_steps_applied=False, new_model_replay=False))
    try:
        summary, joined, cohort_hashes, batch_hashes = [], [], set(), {}
        for source in sources:
            assert time.monotonic()-started < cfg['budget_seconds']
            run = ROOT/source['run']
            case, beta, k = source['case'], source['beta'], source['K']
            manifest = json.loads((run/'manifest.json').read_text())
            cohort_hashes.add(manifest['cohort_sha256'])
            assert manifest['replay_T'] == cfg['T'] and manifest['replay_K'] == k
            gradients, states = read_csv(run/'layer_metrics.csv'), read_csv(run/'state_signals.csv')
            assert len(gradients) == 720 and len(states) == 144
            state_map = {(r['batch_index'], int(r['layer_index'])): r for r in states}
            assert len(state_map) == 144
            for i, parameter in enumerate(cfg['parameter_order']):
                group = [r for r in gradients if r['parameter_name'] == parameter and
                         float(r['endpoint_read_noise_std']) == cfg['sigma']]
                assert len(group) == 144
                voltage = [r for r in states if int(r['layer_index']) == i]
                common = dict(checkpoint=case['name'], epoch=case['checkpoint_epoch'], scheme=case['scheme'],
                              T=cfg['T'], K=k, beta=beta, training_beta=case['beta'], sigma=cfg['sigma'],
                              parameter=parameter, state_layer_index=i, layer=LABELS[i])
                ratios = []
                for r in group:
                    identity = (r['batch_payload_sha256'], r['batch_source_indices_sha256'])
                    assert batch_hashes.setdefault(r['batch_index'], identity) == identity
                    v = state_map[r['batch_index'], i]
                    ratio = float(r['added_noise_rms'])/float(r['clean_eqprop_rms'])
                    assert math.isclose(ratio, float(r['noise_over_clean_norm']), rel_tol=1e-12)
                    ratios.append(ratio)
                    joined.append(dict(common, batch_index=int(r['batch_index']), draw=int(r['noise_draw_index']),
                        free_rms=float(v['free_rms']), positive_free_rms=float(v['positive_free_rms']),
                        negative_free_rms=float(v['negative_free_rms']),
                        centered_half_difference_rms=float(v['centered_half_difference_rms']),
                        clean_gradient_rms=float(r['clean_eqprop_rms']), noise_gradient_rms=float(r['added_noise_rms']),
                        noise_ratio=ratio, noise_dominated=ratio >= 1,
                        noisy_clean_cosine=float(r['noisy_clean_eqprop_cosine']),
                        noisy_bptt_cosine=float(r['cosine'])))
                row = dict(common, batch_draw_pairs=len(group), noise_dominated_fraction=float(np.mean(np.array(ratios) >= 1)),
                           half_difference_noise_rms=cfg['sigma']/math.sqrt(2))
                quantiles(row, 'noise_ratio', ratios)
                for field in ['clean_eqprop_rms', 'added_noise_rms', 'noisy_clean_eqprop_cosine', 'cosine']:
                    quantiles(row, field, [r[field] for r in group])
                row['pooled_gradient_noise_ratio'] = math.sqrt(sum(float(r['added_noise_rms'])**2 for r in group)/
                                                              sum(float(r['clean_eqprop_rms'])**2 for r in group))
                for field in ['free_rms', 'positive_free_rms', 'negative_free_rms', 'centered_half_difference_rms']:
                    quantiles(row, field, [r[field] for r in voltage])
                    row[field+'_pooled'] = math.sqrt(sum(int(r['element_count'])*float(r[field])**2 for r in voltage)/
                                                     sum(int(r['element_count']) for r in voltage))
                row['phase_over_free_rms_median'] = float(np.median([float(r['centered_half_difference_rms'])/float(r['free_rms']) for r in voltage]))
                summary.append(row)
            append_metric(out/'metrics.jsonl', dict(stage='aggregate_saved_replay', dataset_split='validation',
                checkpoint=case['name'], K=k, beta=beta, source_cells_complete=len(summary)//4))
        assert len(cohort_hashes) == 1 and len(batch_hashes) == 36 and len(joined) == 17280
        primary = [r for r in summary if r['K'] == cfg['main_K']]
        thresholds = []
        for case in cases:
            for parameter, layer in zip(cfg['parameter_order'], LABELS):
                points = [r for r in primary if r['checkpoint'] == case['name'] and r['parameter'] == parameter]
                threshold = dict(checkpoint=case['name'], layer=layer, sigma=cfg['sigma'], **crossover(points))
                x = threshold['phase_rms_estimate']
                threshold['phase_rms_over_sigma'] = x/cfg['sigma'] if x else None
                threshold['phase_rms_over_half_difference_noise'] = x/(cfg['sigma']/math.sqrt(2)) if x else None
                thresholds.append(threshold)
        selections = []
        for case in cases:
            for label, beta in [('training_beta', case['beta']),
                                ('best_conv3_beta', .987333678708 if case['scheme'] == 'legacy' else 10.),
                                ('same_beta_10', 10.)]:
                selections.extend([dict(selection=label, **r) for r in primary
                                   if r['checkpoint'] == case['name'] and r['beta'] == beta])
        controls = []
        for r in summary:
            if r['K'] != cfg['control_K']:
                continue
            a = next(x for x in primary if (x['checkpoint'], x['beta'], x['parameter']) ==
                     (r['checkpoint'], r['beta'], r['parameter']))
            controls.append(dict(checkpoint=r['checkpoint'], beta=r['beta'], layer=r['layer'],
                noise_ratio_relative_change=r['noise_ratio_median']/a['noise_ratio_median']-1,
                phase_rms_relative_change=r['centered_half_difference_rms_median']/a['centered_half_difference_rms_median']-1))
        for name, rows in [('summary', summary), ('batch_draw_measurements', joined), ('crossovers', thresholds),
                           ('selected_comparisons', selections), ('k64_controls', controls)]:
            write_csv(out/(name+'.csv'), rows)
        for xfield, xlabel, filename in [('beta', 'Injected β', 'noise_vs_beta'),
                ('centered_half_difference_rms_median', 'Median RMS((v₊ − v₋)/2)', 'noise_vs_phase_rms')]:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
            for ax, layer in zip(axes, LABELS):
                for case, color in zip(cases, COLORS):
                    rows = sorted([r for r in primary if r['checkpoint'] == case['name'] and r['layer'] == layer], key=lambda r:r['beta'])
                    x = [r[xfield] for r in rows]
                    ax.plot(x, [r['noise_ratio_median'] for r in rows], 'o-', color=color, label=case['name'])
                    ax.fill_between(x, [r['noise_ratio_p10'] for r in rows], [r['noise_ratio_p90'] for r in rows], color=color, alpha=.12)
                ax.axhline(1, color='black', linestyle='--', linewidth=1)
                ax.set(xscale='log', yscale='log', title=layer, xlabel=xlabel)
                ax.grid(alpha=.2)
            axes[0].set_ylabel('RMS(gradient noise) / RMS(clean EqProp gradient)')
            axes[-1].legend(fontsize=8)
            fig.suptitle('σ=5×10⁻⁴ · T=16, K=8 · above dashed line: noise dominates · bands: 10–90%')
            for extension in ['png', 'pdf']:
                fig.savefig(out/(filename+'.'+extension), dpi=180)
            plt.close(fig)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
        for ax, layer in zip(axes, LABELS):
            for case, color in zip(cases, COLORS):
                rows = sorted([r for r in primary if r['checkpoint'] == case['name'] and r['layer'] == layer], key=lambda r:r['beta'])
                x = [r['beta'] for r in rows]
                ax.plot(x, [r['centered_half_difference_rms_median'] for r in rows], 'o-', color=color, label=case['name'])
                ax.plot(x, [r['positive_free_rms_median'] for r in rows], '--', color=color, alpha=.6)
                ax.plot(x, [r['negative_free_rms_median'] for r in rows], ':', color=color, alpha=.6)
            ax.axhline(cfg['sigma']/math.sqrt(2), color='black', linestyle='-.', linewidth=1)
            ax.set(xscale='log', yscale='log', title=layer, xlabel='Injected β', ylabel='Median voltage-difference RMS')
            ax.grid(alpha=.2)
        axes[-1].legend(fontsize=8)
        fig.suptitle('Solid: centered half-difference · dashed/dotted: +/− phase minus free · black: σ/√2')
        for extension in ['png', 'pdf']:
            fig.savefig(out/('phase_rms_vs_beta.'+extension), dpi=180)
        plt.close(fig)
        for source in sources:
            assert sha256_file(ROOT/source['run']/'result.json') == source['result_sha256']
        verification = dict(source_cells=30, primary_cells=21, convergence_control_cells=9,
            matched_batches=36, noise_batch_layer_draw_rows=len(joined), source_result_hashes_unchanged=True,
            cohort_sha256=next(iter(cohort_hashes)), official_test_read=False, optimizer_steps_applied=False,
            new_model_replay=False, undefined_thresholds=sum(r['phase_rms_estimate'] is None for r in thresholds),
            max_k64_phase_rms_relative_change=max(abs(r['phase_rms_relative_change']) for r in controls),
            max_k64_noise_ratio_relative_change=max(abs(r['noise_ratio_relative_change']) for r in controls),
            elapsed_seconds=time.monotonic()-started)
        atomic_write_json(out/'summary.json', verification)
        complete_run(out, terminal_metrics=verification, completion=dict(criteria_met=True, coverage_complete=True))
        assert not validate_run(out)
        print(json.dumps(verification, indent=2))
    except BaseException as exc:
        fail_run(out, error=exc)
        raise


if __name__ == '__main__':
    main()
