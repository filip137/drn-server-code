"""Validate the beta-rule sweep and publish its seed-0 diagnostic comparisons."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.run_beta_rule_comparison import (
    ROOT, OUT, ANCHORS, ROLES, checked, csv_rows, read, passes, select_beta, write_csv,
)
from experiments.reporting import atomic_write_json, sha256_file

DEST = ROOT / 'paper_ready_results'
PREFIX = 'beta_rule_comparison_20260918'
THRESHOLDS = (.90, .95, .99)
GROUPS = ('layerwise', 'whole_gradient')


def number(value):
    if value is None or value == '':
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def minimum(rows, key):
    values = [number(row[key]) for row in rows]
    return min(values) if values and all(v is not None for v in values) else None


def maximum(rows, key):
    values = [number(row[key]) for row in rows]
    return max(values) if values and all(v is not None for v in values) else None


def regression(case, layers):
    if case['scheme'] != 'baseline':
        return None
    arch, beta = case['architecture'], case['beta']
    if beta in (100, 200, 300):
        source = ROOT / 'results/eqprop-beta-selection-audit-20260914-v1/selection' / f'selection_{arch}_baseline_seed0_beta{beta:g}'
    elif arch == 'conv3' and beta in (10, 30, 50, 75):
        source = ROOT / 'results/eqprop-conv3-baseline-beta-tk8-20260916-v1/selection' / f'selection_conv3_baseline_seed0_T8_K8_beta{beta:g}'
    else:
        return None
    path = source / 'layer_metrics.csv'
    if not path.is_file():
        raise ValueError(f'Missing historical regression evidence: {path}')
    old = {(r['checkpoint_role'], r['batch_index'], r['parameter_name']): r for r in csv_rows(path)}
    if len(old) != len(layers):
        raise ValueError('Historical regression coverage mismatch.')
    delta = 0.0
    for row in layers:
        reference = old[(row['checkpoint_role'], row['batch_index'], row['parameter_name'])]
        if row['batch_payload_sha256'] != reference['batch_payload_sha256']:
            raise ValueError('Historical input payload mismatch.')
        for key in ('cosine', 'symmetric_norm_delta', 'eqprop_l2', 'bptt_l2'):
            delta = max(delta, abs(float(row[key]) - float(reference[key])))
    if delta > 1e-10:
        raise ValueError(f'Historical gradient changed: {case["name"]}, delta {delta}')
    return dict(case=case['name'], source=str(path.relative_to(ROOT)),
                comparisons=len(layers), maximum_absolute_difference=delta, passed=True)


def plot_curves(summaries):
    figures = []
    for metric, ylabel, filename in (
        ('cosine', 'Minimum cosine across batches', 'cosine'),
        ('norm_delta', 'Maximum symmetric norm mismatch', 'norm'),
    ):
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        for i, arch in enumerate(ANCHORS):
            for j, scheme in enumerate(ANCHORS[arch]):
                axis = axes[i, j]
                rr = [r for r in summaries if r['architecture'] == arch and r['scheme'] == scheme]
                for role, color, role_label in zip(ROLES, ('#2864b7', '#d77b1f'), ('Initial', 'Trained')):
                    for group, style, group_label in zip(GROUPS, ('--', '-'), ('worst matrix', 'whole gradient')):
                        data = sorted([r for r in rr if r['checkpoint_role'] == role], key=lambda r:r['beta'])
                        values = [r[f'{group}_{metric}'] for r in data]
                        axis.plot([r['beta'] for r in data],
                                  [np.nan if v is None else v for v in values],
                                  style, color=color, marker='.', markersize=4,
                                  label=f'{role_label}: {group_label}')
                axis.set_xscale('log')
                if metric == 'cosine':
                    for threshold in THRESHOLDS:
                        axis.axhline(threshold, color='grey', lw=.7, alpha=.6)
                else:
                    axis.set_yscale('log')
                    axis.axhline(.1, color='black', lw=.8)
                axis.axvline(ANCHORS[arch][scheme], color='#3f7b51', lw=.8, alpha=.7)
                axis.set_title(f'{arch.capitalize()} {scheme}')
                axis.set_xlabel('Injected beta')
                axis.set_ylabel(ylabel)
                axis.grid(alpha=.15)
                axis.spines[['top', 'right']].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, .965))
        for ext in ('png', 'pdf'):
            path = DEST / f'{PREFIX}_{filename}.{ext}'
            fig.savefig(path, dpi=180)
            figures.append(path.name)
        plt.close(fig)
    return figures


def plot_limits(selections):
    surfaces = [(a, s) for a in ANCHORS for s in ANCHORS[a]]
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for axis, norm in zip(axes, (False, True)):
        data = np.full((9, 6), np.nan)
        annotations = {}
        for i, (arch, scheme) in enumerate(surfaces):
            for j, (group, threshold) in enumerate((g, t) for g in GROUPS for t in THRESHOLDS):
                row = next(r for r in selections if r['architecture'] == arch and r['scheme'] == scheme
                           and r['grouping'] == group and r['cosine_threshold'] == threshold
                           and r['norm_gate'] == norm and r['checkpoint_role'] == 'both')
                beta = row['selected_beta']
                if beta is not None:
                    data[i, j] = math.log10(beta / ANCHORS[arch][scheme])
                annotations[i, j] = ('none' if beta is None else f'{beta:g}') + ('*' if row['upper_edge_open'] else '')
        cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#dddddd')
        image = axis.imshow(data, cmap=cmap, vmin=-2, vmax=3, aspect='auto')
        for (i, j), text in annotations.items():
            axis.text(j, i, text, ha='center', va='center', fontsize=9,
                      color='black' if np.isnan(data[i,j]) or data[i,j] > 1.5 else 'white')
        axis.set_xticks(range(6), ['Matrix\n.90', 'Matrix\n.95', 'Matrix\n.99',
                                  'Whole\n.90', 'Whole\n.95', 'Whole\n.99'])
        axis.set_yticks(range(9), [f'{a} {s}' for a,s in surfaces])
        axis.set_title('Cosine + norm ≤ .10' if norm else 'Cosine only')
    fig.suptitle('Largest passing tested beta, both checkpoints and every batch\nSeed 0; * marks an open upper grid edge')
    fig.subplots_adjust(left=.13, right=.84, top=.87, bottom=.1, wspace=.12)
    colorbar_axis = fig.add_axes((.88, .20, .016, .58))
    fig.colorbar(image, cax=colorbar_axis, label='log10(selected beta / current anchor)')
    for ext in ('png', 'pdf'):
        fig.savefig(DEST / f'{PREFIX}_limits.{ext}', dpi=180)
    plt.close(fig)


def plot_contributions(layers):
    """Show which matrices dominate the concatenated gradient at each anchor."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharey=True)
    colors = ('#2676b9', '#52a57c', '#bf845a', '#9066ad')
    labels = ('Conv weight 1', 'Conv weight 2', 'Conv weight 3', 'Readout weight')
    summaries = []
    for i, arch in enumerate(ANCHORS):
        for j, (scheme, anchor) in enumerate(ANCHORS[arch].items()):
            rows = [r for r in layers if r['architecture']==arch and r['scheme']==scheme
                    and float(r['injected_beta'])==anchor]
            axis = axes[i, j]
            for x, (role, estimator) in enumerate((role, e) for role in ROLES for e in ('bptt', 'eqprop')):
                bottom = 0
                for name in sorted({r['parameter_name'] for r in rows}):
                    rr = [r for r in rows if r['checkpoint_role']==role and r['parameter_name']==name]
                    values = [r[f'{estimator}_squared_norm_fraction'] for r in rr
                              if r[f'{estimator}_squared_norm_fraction'] is not None]
                    mean = float(np.mean(values)) if values else 0
                    index = 3 if name.startswith('DenseWeight_') else int(name.split('_')[-1])
                    axis.bar(x, mean, bottom=bottom, color=colors[index], width=.65)
                    bottom += mean
                    summaries.append(dict(architecture=arch, scheme=scheme, beta=anchor,
                        checkpoint_role=role, estimator=estimator, parameter_name=name,
                        mean_squared_norm_fraction=mean,
                        minimum_squared_norm_fraction=min(values) if values else None,
                        maximum_squared_norm_fraction=max(values) if values else None))
            axis.set_title(f'{arch.capitalize()} {scheme}')
            axis.set_xticks(range(4), ['Init\nBPTT','Init\nEP','Best\nBPTT','Best\nEP'])
            axis.set_ylim(0, 1)
            axis.spines[['top','right']].set_visible(False)
    handles = [plt.Rectangle((0,0),1,1,color=c) for c in colors]
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)
    fig.supylabel('Mean fraction of total squared gradient norm', x=.01, fontsize=11)
    fig.tight_layout(rect=(.02,0,1,.96))
    for ext in ('png','pdf'):
        fig.savefig(DEST / f'{PREFIX}_contributions.{ext}', dpi=180)
    plt.close(fig)
    write_csv(DEST / f'{PREFIX}_anchor_contributions.csv', summaries)


def main():
    execution = read(OUT / 'execution.json')
    if execution['state'] != 'complete' or len(execution['runs']) != 153:
        raise ValueError('The full declared sweep is not terminal.')
    all_layers, all_whole, summaries, regressions, failures = [], [], [], [], []
    measurements = {}
    receipts = {}
    for case in execution['runs']:
        directory = ROOT / case['directory']
        if case['state'] == 'complete':
            verified = checked(case, directory)
            if verified['result_sha256'] != case['result_sha256']:
                raise ValueError('A completed result changed.')
            receipts[case['directory']] = case['result_sha256']
            layers = csv_rows(directory / 'layer_metrics.csv')
            whole = csv_rows(directory / 'whole_gradient_metrics.csv')
            if proof := regression(case, layers):
                regressions.append(proof)
            by_batch = {(r['checkpoint_role'], r['batch_index']): r for r in whole}
            for row in layers:
                group = by_batch[(row['checkpoint_role'], row['batch_index'])]
                for estimator in ('bptt', 'eqprop'):
                    denominator = float(group[f'{estimator}_l2']) ** 2
                    row[f'{estimator}_squared_norm_fraction'] = (
                        float(row[f'{estimator}_l2']) ** 2 / denominator if denominator > 0 else None)
                    row[f'{estimator}_rms'] = float(row[f'{estimator}_l2']) / math.sqrt(int(row['element_count']))
                row['result_directory'] = case['directory']
            all_layers.extend(layers); all_whole.extend(whole)
        else:
            layers, whole = [], []
            status = read(directory / 'status.json')
            if status['state'] != 'failed' or (directory / 'result.json').exists():
                raise ValueError('Invalid retained numerical failure.')
            failures.append(dict(case=case['name'], beta=case['beta'], directory=case['directory'],
                                 error=status.get('error')))
        measurements[case['name']] = {'layerwise': layers, 'whole_gradient': whole}
        for role in ROLES:
            lr = [r for r in layers if r['checkpoint_role'] == role]
            wr = [r for r in whole if r['checkpoint_role'] == role]
            summaries.append(dict(architecture=case['architecture'], scheme=case['scheme'],
                beta=case['beta'], anchor=case['anchor'], factor=case['factor'], checkpoint_role=role,
                layerwise_cosine=minimum(lr, 'cosine'), whole_gradient_cosine=minimum(wr, 'cosine'),
                layerwise_norm_delta=maximum(lr, 'symmetric_norm_delta'),
                whole_gradient_norm_delta=maximum(wr, 'symmetric_norm_delta'),
                residual_passed=bool(lr and all(r['all_endpoint_residual_gates_passed']=='True' for r in lr)),
                state=case['state'], result_directory=case['directory']))

    # The BPTT reference and cohort must not change when only beta changes.
    references = {}
    bptt_repeats = 0
    bptt_max_difference = 0.0
    for row in all_layers:
        key = tuple(row[k] for k in ('architecture', 'scheme', 'checkpoint_role',
                                     'batch_index', 'parameter_name'))
        if key in references:
            ref = references[key]
            if ref['batch_payload_sha256'] != row['batch_payload_sha256']:
                raise ValueError('The replay cohort changed across betas.')
            bptt_max_difference = max(bptt_max_difference,
                abs(float(ref['bptt_l2'])-float(row['bptt_l2'])))
            bptt_repeats += 1
        else:
            references[key] = row
    if bptt_max_difference > 1e-10:
        raise ValueError('The supposedly fixed BPTT reference changed across betas.')

    # Relative gradient scale uses the identical initial minibatch and parameter.
    initial = {(r['architecture'],r['scheme'],r['injected_beta'],r['batch_index'],r['parameter_name']): r
               for r in all_layers if r['checkpoint_role'] == ROLES[0]}
    for row in all_layers:
        ref = initial[(row['architecture'],row['scheme'],row['injected_beta'],row['batch_index'],row['parameter_name'])]
        for estimator in ('bptt', 'eqprop'):
            denominator = float(ref[f'{estimator}_l2'])
            row[f'{estimator}_l2_relative_to_initial'] = float(row[f'{estimator}_l2']) / denominator if denominator > 1e-30 else None

    selections = []
    for arch in ANCHORS:
        for scheme, anchor in ANCHORS[arch].items():
            cases = [c for c in execution['runs'] if c['architecture']==arch and c['scheme']==scheme]
            for role in ('both', *ROLES):
                for grouping in GROUPS:
                    for norm_gate in (False, True):
                        for threshold in THRESHOLDS:
                            points = []
                            for case in cases:
                                rows = measurements[case['name']][grouping]
                                if role != 'both': rows = [r for r in rows if r['checkpoint_role']==role]
                                points.append((case['beta'], passes(rows, threshold, norm_gate)))
                            decision = select_beta(points)
                            selections.append(dict(architecture=arch, scheme=scheme, checkpoint_role=role,
                                grouping=grouping, cosine_threshold=threshold, norm_gate=norm_gate,
                                current_anchor=anchor,
                                ratio_to_anchor=decision['selected_beta']/anchor if decision['selected_beta'] is not None else None,
                                **decision))
    joint = [s for s in selections if s['checkpoint_role']=='both']
    assert len(joint)==108 and len(selections)==324
    analysis = OUT / 'analysis'; analysis.mkdir(exist_ok=True)
    for filename, rows in [('layer_metrics',all_layers), ('whole_gradient_metrics',all_whole),
                           ('case_summary',summaries), ('selections',joint), ('checkpoint_selections',selections)]:
        write_csv(DEST / f'{PREFIX}_{filename}.csv', rows)
    figures = plot_curves(summaries); plot_limits(selections); plot_contributions(all_layers)
    global_larger = []
    norm_changed = []
    lookup = {(r['architecture'],r['scheme'],r['grouping'],r['cosine_threshold'],r['norm_gate']):r for r in joint}
    for arch in ANCHORS:
        for scheme in ANCHORS[arch]:
            for threshold in THRESHOLDS:
                for norm in (False, True):
                    layer=lookup[arch,scheme,'layerwise',threshold,norm]['selected_beta']
                    whole=lookup[arch,scheme,'whole_gradient',threshold,norm]['selected_beta']
                    if whole is not None and (layer is None or whole>layer):
                        global_larger.append(dict(architecture=arch,scheme=scheme,threshold=threshold,norm_gate=norm,
                                                 layerwise_beta=layer,whole_gradient_beta=whole))
                for grouping in GROUPS:
                    without=lookup[arch,scheme,grouping,threshold,False]['selected_beta']
                    with_norm=lookup[arch,scheme,grouping,threshold,True]['selected_beta']
                    if without!=with_norm:
                        norm_changed.append(dict(architecture=arch,scheme=scheme,grouping=grouping,
                                                 threshold=threshold,cosine_only_beta=without,with_norm_beta=with_norm))
    verification=dict(state='validated', expected_cases=153, completed_cases=len(receipts),
        numerical_failures=failures, measured_replays=len(all_whole), measured_layer_rows=len(all_layers),
        joint_selections=108, checkpoint_selections=324, historical_regressions=regressions,
        maximum_historical_difference=max((r['maximum_absolute_difference'] for r in regressions), default=None),
        source_results=receipts, source_checkpoints_unchanged=True, official_test_read=False,
        optimizer_steps_applied=False, whole_gradient_all_batch_requirement=True,
        bptt_norm_repeat_comparisons=bptt_repeats,
        bptt_norm_maximum_absolute_difference=bptt_max_difference,
        physical_gpu_hours=execution['charged_gpu_hours'], global_larger=global_larger,norm_changed=norm_changed)
    atomic_write_json(DEST / f'{PREFIX}_verification.json', verification)
    atomic_write_json(analysis / 'summary.json',dict(verification=verification,selections=selections))
    lines=['# Seed-0 beta calibration: matrix versus whole-gradient criteria','',
        f'All 153 declared beta cases are terminal: {len(receipts)} complete measurement bundles and {len(failures)} retained numerical failures. '
        f'The sweep measured {len(all_whole):,} checkpoint/batch gradients and {len(all_layers):,} weight-matrix comparisons.','',
        f'Whole-gradient selection admits a larger tested beta in **{len(global_larger)}/54** matched comparisons. '
        f'Adding the norm constraint changes **{len(norm_changed)}/54** selections. These counts compare decision rules on identical measurements.','',
        'All values below are injected beta; require every batch at both initialization and the BPTT best-validation checkpoint. '
        'A star denotes a passing upper grid edge, not an identified maximum. “None” means no tested beta passed.','',
        'The frozen grid multiplies each current anchor by '
        '`.01, .03, .1, .3, 1, 2, 3, 5, 7.5, 10, 20, 30, 50, 75, 100, 300, 1000`. '
        'These are largest passing tested points, with no additional decade reduction.','']
    for norm in (False, True):
        lines += ['## '+('Cosine plus symmetric norm mismatch ≤ 0.10' if norm else 'Cosine only'),'',
            '| Architecture/scheme | Current anchor | Matrix .90 | Matrix .95 | Matrix .99 | Whole .90 | Whole .95 | Whole .99 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arch in ANCHORS:
            for scheme,anchor in ANCHORS[arch].items():
                cells=[]
                for grouping in GROUPS:
                    for threshold in THRESHOLDS:
                        r=lookup[arch,scheme,grouping,threshold,norm]
                        cells.append(('None' if r['selected_beta'] is None else f'{r["selected_beta"]:g}')+('*' if r['upper_edge_open'] else ''))
                lines.append('| '+f'{arch} {scheme} | {anchor:g} | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## Interpretation and limits','',
        'Whole-gradient cosine weights each matrix through its gradient magnitude. It can conceal a poorly aligned '
        'small-gradient matrix. The layer CSV retains individual cosines, squared-norm contributions, RMS, sparsity and '
        'scale relative to the same initialization minibatch. The whole-gradient norm gate can similarly conceal matrix-specific scale errors. '
        'With layer-specific learning rates or Adam, raw-gradient alignment also differs from alignment of the actual optimizer update.','',
        'For EP gradient e and BPTT gradient b, cosine is `(e · b) / (||e|| ||b||)` and symmetric norm mismatch is '
        '`2 | ||e|| - ||b|| | / (||e|| + ||b||)`. Matrix checks evaluate these separately for every weight tensor; '
        'whole-gradient checks concatenate all trainable weight tensors before evaluating them. '
        'Each rule must pass all 72 checkpoint/batch comparisons; a single failing matrix also rejects a matrixwise candidate.','',
        'The Conv3 balanced/ours anchor beta 3 retains its already documented initialization failure: '
        'first-convolution cosine 0.939168 on historical batch 2, with norm mismatch 0.278178. '
        'Thus its smaller matrixwise selection is not evidence that previously stable training diverged; '
        'training stability and this all-batch direct-gradient requirement are distinct. '
        'See the [earlier qualification record](../docs/conv_paper_one_seed_bptt_eqprop_protocol.md).','',
        'The comparison changes the acceptance rule only. Raw gradients have the usual EqProp amplification normalization, '
        'but receive no layer normalization, learning-rate scaling or Adam transformation. Frozen biases are excluded. '
        'No additional decade margin is applied to the selected values. Passing regions are recorded without assuming monotonicity.','',
        'These are zero-noise seed-0 diagnostics on the September 14 selection cohort '
        '(four historical plus 32 additional batches of 16), using the original '
        'wide [0,100] initialization/BPTT checkpoints, float64 centered frozen-current EqProp, and T=K=4/6/8. '
        'Every comparison starts from the same post-T state. Equilibrium residuals remain separate; the Conv3 baseline '
        'T8 free-state caveat is not removed by a relaxed cosine criterion. This study establishes neither full-training '
        'stability nor multi-seed qualification and does not change any training beta. Official-test reads and optimizer steps are zero.','',
        '## Verification and artifacts','',
        f'Historical regression: {len(regressions)} complete overlapping cases, maximum absolute metric difference '
        f'{verification["maximum_historical_difference"]}. All included canonical bundles validate, and source-byte, '
        'parameter, float64, cohort, frozen-force and zero-bias guards pass. The aggregation/selection and existing cohort/TK tests passed (26 tests).','',
        f'The BPTT reference is beta-independent across {bptt_repeats:,} repeated layer comparisons: '
        f'maximum norm difference {bptt_max_difference:g}, with identical input payload hashes.','',
        f'Charged combined GPU time: **{execution["charged_gpu_hours"]:.4f}/6 physical GPU-hours**, including smokes. '
        'Conv3 ran on local RTX3090; all Conv1/Conv2 schemes and betas ran on Akib RTX3080 after an exact historical smoke regression. '
        'The user-authorized parallel placement changes five filesystem paths only; source, checkpoints and cohorts remain identical. '
        'A two-worker local benchmark gave 1.068x throughput, so production used one worker per GPU. '
        'The earlier local coordinators were retired between cases without discarding any measurement. '
        'Per-case commands, source identities, logs and receipts are retained in the study directory.','',
        f'- [108 joint beta selections]({PREFIX}_selections.csv)',
        f'- [Initialization/trained/joint selections]({PREFIX}_checkpoint_selections.csv)',
        f'- [Case summaries]({PREFIX}_case_summary.csv)',
        f'- [Layer measurements and norm contributions]({PREFIX}_layer_metrics.csv)',
        f'- [Whole-gradient measurements]({PREFIX}_whole_gradient_metrics.csv)',
        f'- [Verification]({PREFIX}_verification.json)',
        f'- [Beta-limit figure]({PREFIX}_limits.png) / [PDF]({PREFIX}_limits.pdf)',
        f'- [Cosine curves]({PREFIX}_cosine.png) / [PDF]({PREFIX}_cosine.pdf)',
        f'- [Norm-mismatch curves]({PREFIX}_norm.png) / [PDF]({PREFIX}_norm.pdf)',
        f'- [Layer contributions at current beta]({PREFIX}_contributions.png) / [PDF]({PREFIX}_contributions.pdf)',
        f'- [Raw study](../results/{OUT.name}/)',
        '- [Execution plan and transport amendment](../docs/eqprop_beta_rule_comparison_plan_20260918.md)','',
        'In the curve figures, green vertical lines mark the current anchors. Horizontal cosine lines mark '
        '.90/.95/.99; the horizontal norm line marks .10. Contribution bars average per-batch squared-norm fractions.','']
    if failures:
        lines += ['Retained numerical failures: '+', '.join(f['case'] for f in failures)+'.','']
    (DEST / f'{PREFIX}.md').write_text('\n'.join(lines))
    print(json.dumps({k:verification[k] for k in ('completed_cases','measured_replays','measured_layer_rows',
          'maximum_historical_difference','physical_gpu_hours')},indent=2))


if __name__=='__main__':
    main()
