"""Pool fixed-T/K clean checkpoint replays and plot displacement and cosine."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.reporting import atomic_write_json, sha256_file, validate_run
from experiments.replay_conv3_trained_beta_noise import beta_points, cell_identity, replay_contexts

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ['ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0']
LABELS = ['Conv1', 'Conv2', 'Conv3', 'Readout']
COLORS = {'baseline': '#4c566a', 'ours': '#0072b2', 'legacy': '#d55e00'}
RMS_FIELDS = ['free_rms', 'positive_rms', 'negative_rms', 'zero_rms',
              'positive_free_rms', 'negative_free_rms',
              'centered_half_difference_rms', 'zero_free_rms']


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    if value is None or value == '':
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def pooled_rms(rows, field):
    """Root of pooled squared displacement, never the mean of batch RMS."""
    values = [finite(row.get(field)) for row in rows]
    counts = [int(row['element_count']) for row in rows]
    if not rows or any(v is None or v < 0 for v in values):
        return None
    if any(n <= 0 for n in counts):
        raise ValueError('Element counts must be positive')
    return math.sqrt(math.fsum(n*v*v for n, v in zip(counts, values))/sum(counts))


def cosine_summary(rows):
    values = [v for row in rows if (v := finite(row.get('cosine'))) is not None]
    if any(abs(v) > 1 + 1e-10 for v in values):
        raise ValueError('Cosine outside [-1,1]')
    q = np.quantile(np.clip(values, -1, 1), [.1, .5, .9]) if values else [None]*3
    return dict(cosine_defined_batches=len(values), cosine_p10=q[0],
                cosine_median=q[1], cosine_p90=q[2])


def local_slopes(rows):
    output = []
    keys = sorted({(r['scheme'], r['epoch'], r['layer_index']) for r in rows})
    for key in keys:
        group = sorted((r for r in rows if (r['scheme'], r['epoch'], r['layer_index']) == key),
                       key=lambda r: r['beta'])
        for left, right in zip(group, group[1:]):
            record = dict(scheme=key[0], epoch=key[1], layer_index=key[2], layer=LABELS[key[2]],
                          beta_low=left['beta'], beta_high=right['beta'], slope=None,
                          endpoints_residuals_pass=bool(left.get('residuals_pass') and right.get('residuals_pass')))
            values = [finite(r.get('centered_half_difference_rms')) for r in (left, right)]
            if all(r.get('centered_resolved') for r in (left, right)) and all(v is not None and v > 0 for v in values):
                record['slope'] = math.log(values[1]/values[0])/math.log(right['beta']/left['beta'])
            output.append(record)
    lookup = {(r['scheme'], r['epoch'], r['layer_index'], r['beta_low'], r['beta_high']): r for r in output}
    for row in output:
        if row['scheme'] != 'legacy':
            continue
        for other in ['baseline', 'ours']:
            ref = lookup[(other, row['epoch'], row['layer_index'], row['beta_low'], row['beta_high'])]
            row['slope_minus_'+other] = (row['slope']-ref['slope']
                                        if row['slope'] is not None and ref['slope'] is not None else None)
    return output


def collect(cfg, config_path, study, *, allow_partial=False):
    summaries, coverage, raw_states, raw_gradients = [], [], [], []
    cohorts, targets, batch_identity, bptt_identity = set(), set(), {}, {}
    config_hash = sha256_file(config_path)
    batches = cfg['expected_batches']
    for case in replay_contexts(cfg):
        for beta in beta_points(cfg, case=case):
            name = cell_identity(case, beta, absolute=True)
            run = study / 'runs' / name
            status_path = run / 'status.json'
            status = json.loads(status_path.read_text()).get('state') if status_path.exists() else 'unstarted'
            item = dict(cell=name, scheme=case['scheme'], epoch=case['checkpoint_epoch'], beta=beta,
                        state=status, run_dir=str(run.relative_to(ROOT)))
            if status != 'complete':
                if not allow_partial and status != 'failed':
                    raise ValueError(f'Nonterminal cell: {name}: {status}')
                coverage.append(item)
                for layer in range(4):
                    summaries.append(dict(scheme=case['scheme'], epoch=case['checkpoint_epoch'], beta=beta,
                                          layer_index=layer, layer=LABELS[layer], cell_state=status))
                continue
            errors = validate_run(run)
            if errors:
                raise ValueError(f'Invalid bundle {name}: {errors}')
            manifest = json.loads((run/'manifest.json').read_text())
            metrics = json.loads((run/'result.json').read_text())['terminal_metrics']
            assert manifest['configuration']['sha256'] == config_hash, name
            assert metrics['replay_batches'] == batches and metrics['layer_comparisons'] == batches*4, name
            assert metrics['source_bytes_unchanged'] and metrics['all_parameter_tensors_unchanged'], name
            assert metrics['all_bias_tensors_exact_zero'] and metrics['bptt_invariant_across_beta'], name
            assert not metrics['official_test_read'] and not metrics['optimizer_steps_applied'], name
            assert manifest['replay_T'] == manifest['replay_K'] == 8, name
            cohorts.add(manifest['cohort_sha256'])
            targets.add(manifest['runtime']['target'])
            expected_sha = (cfg['initializer_checkpoint_sha256'] if case['checkpoint_epoch'] == 0
                            else case['source_hashes']['final_model.pt'])
            states = read_csv(run/'state_signals.csv')
            gradients = read_csv(run/'layer_metrics.csv')
            residuals = read_csv(run/'equilibrium_residuals.csv')
            assert len(states) == len(gradients) == batches*4
            assert {(int(r['batch_index']), int(r['layer_index'])) for r in states} == {
                (b, layer) for b in range(batches) for layer in range(4)}, name
            assert {(int(r['batch_index']), r['parameter_name']) for r in gradients} == {
                (b, p) for b in range(batches) for p in PARAMETERS}, name
            for row in states + gradients:
                assert row['scheme'] == case['scheme'] and float(row['injected_beta']) == beta, name
                assert int(row['T']) == int(row['K']) == 8, name
                row['epoch'] = case['checkpoint_epoch']
                row['cell'] = name
            for row in gradients:
                assert row['checkpoint_sha256'] == expected_sha, name
                assert int(row['noise_draw_index']) == 0 and float(row['endpoint_read_noise_std']) == 0, name
                identity = (row['batch_payload_sha256'], row['batch_source_indices_sha256'])
                assert batch_identity.setdefault(int(row['batch_index']), identity) == identity, name
                reference_key = (case['name'], int(row['batch_index']), row['parameter_name'])
                assert bptt_identity.setdefault(reference_key, row['bptt_gradient_sha256']) == row['bptt_gradient_sha256'], name
            assert residuals, name
            item.update(residual_failed_rows=sum(r['gate_passed'] != 'True' for r in residuals),
                        result_sha256=sha256_file(run/'result.json'), checkpoint_sha256=expected_sha)
            coverage.append(item)
            raw_states.extend(states)
            raw_gradients.extend(gradients)
            for layer, parameter in enumerate(PARAMETERS):
                sr = [r for r in states if int(r['layer_index']) == layer]
                gr = [r for r in gradients if r['parameter_name'] == parameter]
                rr = [r for r in residuals if int(r['layer_index']) == layer]
                assert len(sr) == len(gr) == batches and rr
                row = dict(scheme=case['scheme'], epoch=case['checkpoint_epoch'], beta=beta,
                           layer_index=layer, layer=LABELS[layer], parameter=parameter, cell_state='complete',
                           batches=batches, element_count=sum(int(r['element_count']) for r in sr),
                           residuals_pass=all(r['gate_passed'] == 'True' for r in rr),
                           residual_failed_rows=sum(r['gate_passed'] != 'True' for r in rr))
                row.update({field: pooled_rms(sr, field) for field in RMS_FIELDS})
                row['relative_centered_displacement'] = (
                    row['centered_half_difference_rms']/row['free_rms']
                    if row['free_rms'] is not None and row['free_rms'] > 0 else None)
                for field in ['free_mean', 'positive_mean', 'negative_mean', 'zero_mean']:
                    values = [finite(r.get(field)) for r in sr]
                    row[field] = (math.fsum(int(r['element_count'])*v for r, v in zip(sr, values))/row['element_count']
                                  if all(v is not None for v in values) else None)
                scale = max(row['positive_rms'], row['negative_rms'])
                row['resolution_indicator'] = 100*np.finfo(np.float64).eps*scale
                row['centered_resolved'] = row['centered_half_difference_rms'] > row['resolution_indicator']
                row.update(cosine_summary(gr))
                for field in ['bptt_rms', 'clean_eqprop_rms', 'relative_l2_difference_over_bptt']:
                    values = [v for r in gr if (v := finite(r.get(field))) is not None]
                    row[field+'_median'] = float(np.median(values)) if values else None
                summaries.append(row)
    assert len(coverage) == cfg['expected_production_cells'] == 114
    assert len(cohorts) <= 1 and len(targets) <= 1, (cohorts, targets)
    if raw_gradients:
        assert len(batch_identity) == batches
    validation = dict(expected_cells=len(coverage), complete_cells=sum(r['state'] == 'complete' for r in coverage),
                      failed_cells=sum(r['state'] == 'failed' for r in coverage),
                      pending_cells=sum(r['state'] not in ['complete', 'failed'] for r in coverage),
                      state_rows=len(raw_states), gradient_rows=len(raw_gradients),
                      cohort_sha256=sorted(cohorts), targets=sorted(targets), config_sha256=config_hash,
                      T=8, K=8, noisy_draws=0, official_test_read=False, optimizer_steps_applied=False,
                      source_bundles_validated=True, bptt_reference_beta_invariant=True)
    return summaries, coverage, raw_states, raw_gradients, validation


def plot(rows, field, title, stem, out, *, cosine=False, partial=False):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.7), sharex=True, sharey='col', squeeze=False)
    for ri, epoch in enumerate([0, 30]):
        for layer in range(4):
            ax = axes[ri, layer]
            for scheme, color in COLORS.items():
                group = sorted((r for r in rows if (r['scheme'], r['epoch'], r['layer_index']) == (scheme, epoch, layer)),
                               key=lambda r: r['beta'])
                x = np.array([r['beta'] for r in group])
                y = np.array([finite(r.get(field)) if finite(r.get(field)) is not None else np.nan for r in group])
                if not cosine:
                    y[y <= 0] = np.nan
                ax.plot(x, y, '.-', color=color, label=scheme, linewidth=1.5, markersize=4)
                if cosine:
                    lo = np.array([r.get('cosine_p10') if r.get('cosine_p10') is not None else np.nan for r in group])
                    hi = np.array([r.get('cosine_p90') if r.get('cosine_p90') is not None else np.nan for r in group])
                    ax.fill_between(x, lo, hi, color=color, alpha=.12, linewidth=0)
                failed_residuals = np.array([r.get('cell_state') == 'complete' and not r.get('residuals_pass') for r in group])
                ax.scatter(x[failed_residuals], y[failed_residuals], marker='x', color=color, s=32, zorder=4)
                if field == 'centered_half_difference_rms':
                    unresolved = np.array([r.get('cell_state') == 'complete' and not r.get('centered_resolved') for r in group])
                    ax.scatter(x[unresolved], y[unresolved], marker='o', edgecolor=color, facecolor='white', s=22, zorder=5)
            ax.set_xscale('log')
            if cosine:
                ax.set_ylim(-1.03, 1.03)
                ax.axhline(0, color='.7', lw=.5)
                ax.axhline(.99, color='.6', lw=.5, ls=':')
            else:
                ax.set_yscale('log')
            ax.grid(True, alpha=.2, which='major')
            ax.set_title(f'{LABELS[layer]} · '+('Initialization' if epoch == 0 else 'Clean EqProp epoch 30'), fontsize=10)
            if ri == 1:
                ax.set_xlabel('Injected beta')
            if layer == 0:
                ax.set_ylabel('Gradient cosine' if cosine else 'RMS displacement [voltage units]')
    handles = [Line2D([0], [0], color=c, marker='.', label=s) for s, c in COLORS.items()]
    handles.append(Line2D([0], [0], color='.3', marker='x', linestyle='none', label='Residual flag (fixed T/K)'))
    if field == 'centered_half_difference_rms':
        handles.append(Line2D([0], [0], color='.3', marker='o', markerfacecolor='white',
                              linestyle='none', label='Below resolution indicator'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False, bbox_to_anchor=(.5, .003))
    subtitle = 'T=K=8 · float64 · 576 matched validation examples · clean replay · one seed'
    if cosine:
        subtitle += '\nBPTT: same post-T start, K=8 unroll; band: batch 10th–90th percentiles'
    else:
        subtitle += '\nElement-weighted pooled RMS; missing/nonfinite points are gaps'
    if partial:
        title = 'PARTIAL — '+title
    fig.suptitle(title+'\n'+subtitle, fontsize=12)
    fig.subplots_adjust(left=.065, right=.99, bottom=.13, top=.85, hspace=.25, wspace=.26)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'{stem}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_relative_output(rows, out, *, partial=False):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for ax, epoch in zip(axes, [0, 30]):
        ax.axhspan(.8, 2., color='.85', alpha=.6)
        for scheme, color in COLORS.items():
            group = sorted((r for r in rows if (r['scheme'], r['epoch'], r['layer_index']) ==
                            (scheme, epoch, 3)), key=lambda r: r['beta'])
            x = np.array([r['beta'] for r in group])
            y = np.array([finite(r.get('relative_centered_displacement'))
                          if finite(r.get('relative_centered_displacement')) is not None
                          else np.nan for r in group])
            y[y <= 0] = np.nan
            ax.plot(x, y, '.-', color=color, label=scheme, linewidth=1.5)
            flagged = np.array([r.get('cell_state') == 'complete' and
                                not r.get('residuals_pass') for r in group])
            ax.scatter(x[flagged], y[flagged], marker='x', color=color, s=32)
        ax.set(xscale='log', yscale='log', xlabel='Injected beta',
               title='Initialization' if epoch == 0 else 'Clean EqProp epoch 30')
        ax.grid(True, alpha=.2)
    axes[0].set_ylabel('Relative output displacement D/F')
    axes[1].legend(frameon=False)
    title = 'Conv3 relative output displacement versus beta'
    if partial:
        title = 'PARTIAL — '+title
    fig.suptitle(title+'\nD = pooled RMS((v+ − v−)/2); F = pooled post-T free-output RMS', fontsize=12)
    fig.text(.5, .015, 'T=K=8 · 576 matched examples · shaded: new initial targets 0.8–2.0 · ×: residual flag',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.10, right=.98, bottom=.16, top=.80, wspace=.15)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'relative_output_beta.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    assert cfg['T'] == cfg['K'] == 8 and not cfg['read_noise_seeds']
    study = ROOT/cfg['output_root']
    out = study/('preview' if args.allow_partial else 'analysis')
    rows, coverage, states, gradients, validation = collect(cfg, args.config, study, allow_partial=args.allow_partial)
    out.mkdir(parents=True, exist_ok=True)
    for filename, data in [('layer_summary', rows), ('coverage', coverage), ('state_batches', states),
                           ('gradient_batches', gradients), ('local_slopes', local_slopes(rows))]:
        write_csv(out/(filename+'.csv'), data)
    partial = validation['pending_cells'] > 0
    plot(rows, 'positive_free_rms', 'Conv3 positive-free RMS displacement versus beta', 'rms_displacement_beta', out, partial=partial)
    plot(rows, 'centered_half_difference_rms', 'Conv3 centered phase RMS displacement versus beta', 'centered_rms_beta', out, partial=partial)
    plot(rows, 'cosine_median', 'Conv3 clean EqProp–BPTT gradient cosine versus beta', 'cosine_beta', out, cosine=True, partial=partial)
    plot_relative_output(rows, out, partial=partial)
    atomic_write_json(out/'validation.json', validation)
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
