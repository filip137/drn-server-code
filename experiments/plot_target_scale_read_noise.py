"""Plot layerwise BPTT agreement for the saved rescaled-target noise replay."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiments.reporting import validate_run


PANELS = [('conv2', 1/16), ('conv3', 1/16), ('conv3', 1/64)]


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_production(production, schemes, *, allow_subset=False):
    """Reject incomplete or incompatible scientific cells before aggregation."""
    config = json.loads((production/'config.used.json').read_text())
    expected = {c['name'] for c in config['cases']}
    cells = {(c['architecture'], c['target_scale'], c['scheme']) for c in config['cases']}
    assert len(expected) == len(cells) == len(config['cases']), 'Duplicate cases'
    expected_cells = {(a, t, s) for a, t in PANELS for s in schemes}
    assert cells and (cells <= expected_cells if allow_subset else cells == expected_cells), 'Unexpected case coverage'
    assert {p.name for p in (production/'cases').iterdir()} == expected
    errors = {str(p):validate_run(p) for p in [production,*sorted((production/'cases').iterdir())]}
    assert not any(errors.values()), errors
    cohort = json.loads((production/'cohort.json').read_text())
    assert cohort['cohort_sha256'] == config['expected_cohort_sha256'], 'Cohort hash mismatch'
    assert cohort['official_test_read'] is False
    assert cohort['replayed_batches'] == list(range(config['batch_count'])), 'Replay batch mismatch'
    assert cohort['batch_size'] == config['dataset']['batch_size']
    assert cohort['validation_indices_sha256'] == config['dataset']['validation_indices_sha256']
    for actual, expected_guard in zip(cohort['batches'], config['dataset']['expected_batch_guards'], strict=True):
        assert all(actual[k] == v for k, v in expected_guard.items()), 'Cohort batch mismatch'
    seeds, levels = config['read_noise_seeds'], config['noise_levels']
    assert len(seeds) == len(set(seeds)) == 3, 'Expected three distinct paired draws'
    assert len(levels) == len(set(levels)) and all(math.isfinite(v) and v > 0 for v in levels)

    def check_manifest(path, inputs):
        manifest = json.loads((path/'manifest.json').read_text())
        assert manifest['configuration'] == config, 'Manifest/config mismatch'
        assert manifest['inputs'] == inputs, 'Manifest case mismatch'
        for field, value in dict(smoke=False, official_test_read=False,
                optimizer_steps_applied=False, checkpoint_role='best_validation',
                runtime_dtype='float64', seed=0, dataset='ordinary_mnist_validation',
                evidence_class='ordinary_mnist_selection').items():
            assert manifest[field] == value, f'Manifest mismatch: {field}'

    check_manifest(production, {})
    raw = []
    for case in config['cases']:
        path = production/'cases'/case['name']
        check_manifest(path, case)
        assert case['checkpoint_role'] == 'best_validation'
        depth = int(case['architecture'][-1])
        assert (case['T'], case['K'], case['relative_target']) == (2*depth+2, 2*depth+2, 2*depth)
        with (path/'layer_metrics.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        layers = [f'ConvWeight_{i}' for i in range(depth)] + ['DenseWeight_0']
        expected_rows = {(layer, batch, 'clean', 0., -1, '')
                         for layer in layers for batch in cohort['replayed_batches']}
        expected_rows.update((layer, batch, 'nodewise_relative', eta, draw, str(seed))
                             for layer in layers for batch in cohort['replayed_batches']
                             for eta in levels for draw, seed in enumerate(seeds))
        seen = set()
        for row in rows:
            for field in ('cosine','target_scale','noise_level','clean_eqprop_bptt_cosine','relative_target'):
                row[field] = float(row[field]); assert math.isfinite(row[field])
            row['noise_draw_index'] = int(row['noise_draw_index'])
            row['batch_index'] = int(row['batch_index'])
            for field in ('name','architecture','scheme','target_scale','relative_target'):
                assert row[field] == case[field], f'CSV case mismatch: {field}'
            assert int(row['checkpoint_epoch']) == case['checkpoint_epoch']
            assert -1.000001 <= row['cosine'] <= 1.000001
            key = tuple(row[k] for k in ('parameter_name','batch_index','noise_model','noise_level','noise_draw_index','noise_seed'))
            assert key not in seen, f'Duplicate CSV cell: {key}'
            seen.add(key)
        assert seen == expected_rows, f'Clean/noise coverage mismatch: {case["name"]}'
        raw.extend(rows)
    return config, cohort, raw, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--additional-production-root', type=Path, action='append', default=[],
                        help='Validated amp2 production bundle; repeat for separate workers (three cases total)')
    parser.add_argument('--output-dir', type=Path,
                        help='Plot directory; required and distinct from historical analysis when merging')
    args = parser.parse_args()
    root = args.study_root.resolve()
    out = args.output_dir.resolve() if args.output_dir else root/'analysis'
    if args.additional_production_root and (args.output_dir is None or out == root/'analysis'):
        parser.error('Merging requires --output-dir distinct from the historical analysis directory')
    config, cohort, raw, errors = load_production(root/'production', ('ours', 'legacy'))
    schemes = ['ours', 'legacy']
    case_count = len(config['cases'])
    extra_cells = set()
    for additional_root in args.additional_production_root:
        extra_config, extra_cohort, extra_rows, extra_errors = load_production(
            additional_root.resolve(), ('amp2',), allow_subset=True)
        for field in ('dataset','batch_count','expected_cohort_sha256','noise_levels',
                      'read_noise_seeds','matching_tolerance','max_matching_corrections'):
            assert extra_config[field] == config[field], f'Cannot merge different {field}'
        assert extra_cohort == cohort, 'Cannot merge different replay cohorts'
        cells = {(c['architecture'], c['target_scale']) for c in extra_config['cases']}
        assert not extra_cells.intersection(cells), 'Duplicate additional cases'
        extra_cells.update(cells)
        raw.extend(extra_rows)
        errors.update(extra_errors)
        case_count += len(extra_config['cases'])
    if args.additional_production_root:
        assert extra_cells == set(PANELS), 'Expected exactly the three additional amp2 cases'
        schemes.append('amp2')
    groups = defaultdict(list)
    for row in raw:
        key = tuple(row[k] for k in ('architecture','target_scale','scheme','parameter_name','noise_level'))
        groups[key].append(row)
    summaries = []
    for (arch,scale,scheme,layer,eta), rows in sorted(groups.items()):
        draws = defaultdict(list)
        for row in rows: draws[row['noise_draw_index']].append(row['cosine'])
        assert all(len(v)==config['batch_count'] for v in draws.values())
        draw_means = [mean(v) for v in draws.values()]
        summaries.append(dict(architecture=arch,target_scale=scale,scheme=scheme,layer=layer,
            eta=eta,cosine=mean(draw_means),draw_min=min(draw_means),draw_max=max(draw_means),
            batches=config['batch_count'],draws=len(draw_means)))
    lookup = {(r['architecture'],r['target_scale'],r['scheme'],r['layer'],r['eta']):r for r in summaries}
    paired = []
    for key, rows in sorted(groups.items()):
        arch,scale,scheme,layer,eta=key
        if scheme != 'ours': continue
        legacy = {(int(r['batch_index']),r['noise_draw_index']):r for r in groups[(arch,scale,'legacy',layer,eta)]}
        by_draw = defaultdict(list)
        for row in rows:
            ref = legacy[int(row['batch_index']),row['noise_draw_index']]
            by_draw[row['noise_draw_index']].append(row['cosine']-ref['cosine'])
        differences = [mean(v) for v in by_draw.values()]
        paired.append(dict(architecture=arch,target_scale=scale,layer=layer,eta=eta,
            ours=lookup[(arch,scale,'ours',layer,eta)]['cosine'],
            legacy=lookup[(arch,scale,'legacy',layer,eta)]['cosine'],
            gap=mean(differences),gap_draw_min=min(differences),gap_draw_max=max(differences)))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'layer_summary.csv',summaries)
    write_csv(out/'paired_gaps.csv',paired)
    colors={'ours':'#ff7f0e','legacy':'#1f77b4'}
    labels={'ours':'Ours','legacy':'Legacy','amp2':'A=4 B=2'}
    if 'amp2' in schemes:
        colors['amp2'] = '#9467bd'
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    panels=PANELS

    def draw(ax,arch,scale,layer):
        for scheme in schemes:
            data=sorted([r for r in summaries if (r['architecture'],r['target_scale'],r['scheme'],r['layer'])==(arch,scale,scheme,layer) and r['eta']>0],key=lambda r:r['eta'])
            assert len(data)==len(config['noise_levels']), 'Missing plotted noise cells'
            xs=[r['eta'] for r in data]
            ax.plot(xs,[r['cosine'] for r in data],'-o',ms=3,lw=1.8,color=colors[scheme])
            ax.fill_between(xs,[r['draw_min'] for r in data],[r['draw_max'] for r in data],color=colors[scheme],alpha=.17)
            clean=lookup[arch,scale,scheme,layer,0.]['cosine']
            ax.axhline(clean,color=colors[scheme],ls=':',lw=.8,alpha=.65)
        ax.set_xscale('log');ax.set_xlim(1.7e-9,1.7e-2);ax.set_ylim(-.06,1.045)
        ax.set_xticks([1e-8,1e-6,1e-4,1e-2]);ax.grid(alpha=.17)
        ax.set_title('Readout' if layer.startswith('Dense') else 'Convolution '+str(int(layer.split('_')[-1])+1))
    handles=[Line2D([0],[0],color=colors[s],lw=2,label=labels[s]) for s in schemes]
    fig,axes=plt.subplots(3,4,figsize=(14,9),sharex=True,sharey=True)
    for row,(arch,scale) in enumerate(panels):
        depth=int(arch[-1]);layers=[f'ConvWeight_{i}' for i in range(depth)]+['DenseWeight_0']
        columns=list(range(depth))+[3]
        for col,layer in zip(columns,layers):draw(axes[row,col],arch,scale,layer)
        for col in set(range(4))-set(columns):axes[row,col].set_visible(False)
        axes[row,0].set_ylabel(f'{arch.capitalize()} · target 1/{int(1/scale)}\nCosine to BPTT')
    for ax in axes[-1]:ax.set_xlabel('Relative read noise η')
    fig.suptitle('Rescaled-target trained checkpoints: noisy EqProp agreement with BPTT',y=.99,fontsize=15)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.957),ncol=len(schemes),frameon=False)
    fig.text(.5,.012,'Best-validation checkpoints · seed 0 · 192 validation images · bands: range over 3 paired noise draws\nDotted: clean EqProp. Matched output D/F = 4 (Conv2), 6 (Conv3); native T = K = 6 / 8.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.923))
    fig.savefig(out/'conv23_rescaled_targets_noise_vs_bptt.jpg',dpi=200)
    plt.close(fig)
    for arch,scale in panels:
        depth=int(arch[-1]);layers=[f'ConvWeight_{i}' for i in range(depth)]+['DenseWeight_0']
        fig,axes=plt.subplots(1,len(layers),figsize=(3.6*len(layers),3.6),sharey=True)
        for ax,layer in zip(axes,layers):draw(ax,arch,scale,layer);ax.set_xlabel('Relative read noise η')
        axes[0].set_ylabel('Cosine to BPTT')
        fig.suptitle(f'{arch.capitalize()} · target 1/{int(1/scale)} · best-validation checkpoints',fontsize=13)
        fig.legend(handles=handles,loc='lower center',ncol=len(schemes),frameon=False)
        fig.tight_layout(rect=(0,.08,1,.96))
        fig.savefig(out/f'{arch}_target1over{int(1/scale)}_noise_vs_bptt.jpg',dpi=200);plt.close(fig)
    validation=dict(cases=case_count,layer_comparisons=len(raw),summary_cells=len(summaries),
                    bundle_errors=errors,clean_cosine_range=[min(r['cosine'] for r in summaries if r['eta']==0),max(r['cosine'] for r in summaries if r['eta']==0)])
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation,indent=2))
    print('Comparisons at eta=1e-4:')
    for row in paired:
        if row['eta']==1e-4:print(row)


if __name__=='__main__':main()
