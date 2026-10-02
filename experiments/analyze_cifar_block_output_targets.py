"""Find sampled block-output displacement bands preserving every layer's gradient."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.analyze_cifar_precision_response import ARMS, OUTPUT_LAYERS


CRITERIA = ['pooled_cos95', 'pooled_cos99', 'every_batch_cos95',
            'pooled_cos95_norm10', 'every_batch_cos95_norm10']
CANDIDATES = {1: 1e-3, 2: 1e-3, 3: 1e-6}


def minimum_cosine(rows):
    if any(r['native_bptt']['cosine'] is None for r in rows):
        return None
    return min(r['native_bptt']['cosine'] for r in rows)


def sampled_spans(rows, criterion):
    """Endpoints of consecutive passing samples, without bridging failed samples."""
    spans, current = [], []
    for row in rows:
        if row[criterion]:
            current.append(row)
        elif current:
            spans.append(current); current = []
    if current: spans.append(current)
    return [dict(lower=min(r['output_relative_rms'] for r in group),
                 upper=max(r['output_relative_rms'] for r in group),
                 multipliers=[r['beta_multiplier'] for r in group],
                 sampled_displacements=[r['output_relative_rms'] for r in group])
            for group in spans]


def analyze(root):
    out = root / 'analysis'
    previous = json.loads((out / 'summary.json').read_text())
    assert previous['complete_coverage']
    hashes = {r['epoch']: r['artifact_sha256'] for r in previous['summaries']}
    cases, bands, common, candidates, per_layer_oracles = [], [], [], [], []
    for epoch in [0, 1, 5]:
        path = root / f'cells/e{epoch}/artifacts/comparison.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[epoch]
        data = json.loads(path.read_text())
        for arm in ARMS:
            for block, output_layer in OUTPUT_LAYERS.items():
                group = []
                for multiplier in sorted({r['beta_multiplier'] for r in data['layers']}):
                    rows = [r for r in data['layers'] if (r['arm'], r['block'], r['beta_multiplier']) ==
                            (arm, block, multiplier)]
                    batches = [r for r in data['batches'] if (r['arm'], r['block'], r['beta_multiplier']) ==
                               (arm, block, multiplier)]
                    assert len(rows) == output_layer and len(batches) == 8 * output_layer
                    output, = [r for r in rows if r['layer'] == output_layer]
                    cosine, batch_cosine = minimum_cosine(rows), minimum_cosine(batches)
                    norm = all(.9 <= r['native_bptt']['norm_ratio'] <= 1.1 for r in rows)
                    batch_norm = all(.9 <= r['native_bptt']['norm_ratio'] <= 1.1 for r in batches)
                    worst = min(rows, key=lambda r: r['native_bptt']['cosine']
                                if r['native_bptt']['cosine'] is not None else -2)
                    record = dict(epoch=epoch, arm=arm, block=block, beta_multiplier=multiplier,
                        beta_injected_B=output['beta_injected_B'],
                        output_relative_rms=output['state_displacement']['relative_rms'],
                        minimum_layer_cosine=cosine, limiting_layer=worst['layer'],
                        minimum_batch_layer_cosine=batch_cosine,
                        pooled_norm_min=min(r['native_bptt']['norm_ratio'] for r in rows),
                        pooled_norm_max=max(r['native_bptt']['norm_ratio'] for r in rows),
                        pooled_cos95=cosine is not None and cosine >= .95,
                        pooled_cos99=cosine is not None and cosine >= .99,
                        every_batch_cos95=batch_cosine is not None and batch_cosine >= .95)
                    record['pooled_cos95_norm10'] = record['pooled_cos95'] and norm
                    record['every_batch_cos95_norm10'] = record['every_batch_cos95'] and batch_norm
                    group.append(record)
                # Reconcile the stricter criterion with the previously validated analysis.
                old, = [r for r in previous['passing_multipliers_by_epoch']
                         if (r['epoch'], r['arm'], r['block']) == (epoch, arm, block)]
                assert [r['beta_multiplier'] for r in group if r['pooled_cos95_norm10']] == old['pooled']
                assert [r['beta_multiplier'] for r in group if r['every_batch_cos95_norm10']] == old['every_batch']
                cases.extend(group)
                best = max((r for r in group if r['minimum_layer_cosine'] is not None),
                           key=lambda r: r['minimum_layer_cosine'])
                selected_layers = []
                for layer in range(1, output_layer + 1):
                    layer_rows = [r for r in data['layers'] if
                        (r['arm'], r['block'], r['layer']) == (arm, block, layer)
                        and r['native_bptt']['cosine'] is not None]
                    optimum = max(layer_rows, key=lambda r: r['native_bptt']['cosine'])
                    shared, = [r for r in layer_rows if r['beta_multiplier'] == best['beta_multiplier']]
                    optimum_case, = [r for r in group if r['beta_multiplier'] == optimum['beta_multiplier']]
                    selected_layers.append(dict(layer=layer,
                        beta_multiplier=optimum['beta_multiplier'],
                        output_relative_rms=optimum_case['output_relative_rms'],
                        cosine=optimum['native_bptt']['cosine'], norm_ratio=optimum['native_bptt']['norm_ratio'],
                        cosine_at_shared_beta=shared['native_bptt']['cosine']))
                oracle_minimum = min(r['cosine'] for r in selected_layers)
                per_layer_oracles.append(dict(epoch=epoch, arm=arm, block=block,
                    shared_beta_multiplier=best['beta_multiplier'],
                    shared_output_relative_rms=best['output_relative_rms'],
                    shared_minimum_cosine=best['minimum_layer_cosine'],
                    per_layer_minimum_cosine=oracle_minimum,
                    improvement=oracle_minimum-best['minimum_layer_cosine'],
                    selected_layers=selected_layers,
                    distinct_phase_pairs=len({r['beta_multiplier'] for r in selected_layers})))
                bands.append(dict(epoch=epoch, arm=arm, block=block, best_observed=best,
                    spans={criterion: sampled_spans(group, criterion) for criterion in CRITERIA}))
                if arm == 'float64':
                    target = CANDIDATES[block]
                    lower = max((r for r in group if r['output_relative_rms'] <= target),
                                key=lambda r: r['output_relative_rms'])
                    upper = min((r for r in group if r['output_relative_rms'] >= target),
                                key=lambda r: r['output_relative_rms'])
                    assert all(r['pooled_cos99'] and r['every_batch_cos95_norm10'] for r in [lower, upper])
                    candidates.append(dict(epoch=epoch, arm=arm, block=block,
                        proposed_output_relative_rms=target, exact_target_validated=False,
                        lower_sample=lower, upper_sample=upper))
    assert len(cases) == 540 and len(bands) == 27 and len(candidates) == 9
    for arm in ARMS:
        for block in OUTPUT_LAYERS:
            selected = [r for r in bands if r['arm'] == arm and r['block'] == block]
            for criterion in CRITERIA:
                overlaps = [(r['lower'], r['upper']) for r in selected[0]['spans'][criterion]]
                for epoch_band in selected[1:]:
                    overlaps = [(max(a, r['lower']), min(b, r['upper']))
                        for a, b in overlaps for r in epoch_band['spans'][criterion]
                        if max(a, r['lower']) <= min(b, r['upper'])]
                common.append(dict(arm=arm, block=block, criterion=criterion,
                                   overlap_of_sampled_spans=overlaps))
    summary = dict(epochs=[0,1,5], input_hashes=hashes, cases=len(cases), bands=bands,
        common=common, float64_candidate_brackets=candidates,
        per_layer_beta_oracles=per_layer_oracles,
        oracle_caveat='Retrospective best-beta selection on the same 256-image cohort and sampled grid. Separate phase solves per distinct beta; not a trained or validated selection policy.',
        interval_caveat='Spans join consecutive passing samples. Interiors and common targets are not directly validated.',
        primary_criterion='Minimum pooled native-BPTT cosine across all convolutions in the block >= 0.95',
        displacement='RMS((v_plus-v_minus)/2) / RMS(v_free) at the last analog layer, before pooling and BN')
    (out / 'block_output_targets.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (out / 'block_output_targets.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cases[0])); writer.writeheader(); writer.writerows(cases)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True, layout='constrained')
    colors = {0: '#377eb8', 1: '#e08024', 5: '#398b54'}
    ymin = min(-.04, min(r['minimum_layer_cosine'] for r in cases
        if r['minimum_layer_cosine'] is not None and 1e-10 <= r['output_relative_rms'] <= 1) - .03)
    for i, arm in enumerate(['float32', 'float64']):
        for j, block in enumerate(OUTPUT_LAYERS):
            ax = axes[i, j]
            for epoch in [0,1,5]:
                group = [r for r in cases if (r['epoch'],r['arm'],r['block']) == (epoch,arm,block)
                         and r['minimum_layer_cosine'] is not None and r['output_relative_rms'] > 0]
                ax.plot([r['output_relative_rms'] for r in group],
                        [r['minimum_layer_cosine'] for r in group], color=colors[epoch],
                        marker='o', ms=3, lw=1.5, label=f'Epoch {epoch}')
            ax.axhline(.95, color='#555', ls='--', lw=.9)
            ax.axhline(.99, color='#999', ls=':', lw=.8)
            ax.set(xscale='log', xlim=(1e-10,1), ylim=(ymin,1.015),
                   title=f'{"Float32" if i == 0 else "Full float64"} · Block {block}')
            ax.grid(alpha=.16)
            if j == 0: ax.set_ylabel('Minimum cosine across all layers')
            if i == 1:
                ax.set_xlabel('Block output relative RMS displacement')
                ax.axvline(CANDIDATES[block], color='#654c99', ls='--', lw=.9)
                ax.text(CANDIDATES[block], .12, f' Candidate {CANDIDATES[block]:.0e}',
                        color='#654c99', rotation=90, va='bottom', ha='right', fontsize=9)
    axes[0,0].legend(frameon=False, loc='lower right', fontsize=9)
    fig.suptitle('Worst-layer gradient agreement at a shared block-output displacement\n'
                 'Dashed horizontal: 0.95 · Dotted: 0.99 · Candidates require exact-target validation', fontsize=13)
    fig.savefig(out / 'block_output_worst_layer_cosine.jpg', dpi=170); plt.close(fig)
    print(json.dumps(dict(cases=len(cases), validated_prior_input_hashes=True,
                         prior_joint_criteria_reconciled=True, common=common), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
