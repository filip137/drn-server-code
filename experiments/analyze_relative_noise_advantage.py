"""Summarize exp013 and freeze its common fresh-noise refinement coordinates.

Exploratory coverage checks only: no source-hash or cross-device parity audit.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from experiments.plot_relative_endpoint_noise import number, write_csv, scale_controls

SCHEMES = ('baseline', 'ours', 'legacy')
ARCHES = ('conv1', 'conv2', 'conv3')
METRICS = ('cosine', 'noisy_clean_eqprop_cosine', 'clean_eqprop_bptt_cosine',
           'noise_over_clean_norm', 'relative_gradient_error_norm')
ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def layers(arch):
    return [f'ConvWeight_{i}' for i in range(int(arch[-1]))] + ['DenseWeight_0']


def safe_mean(values):
    return mean(values) if values and all(v is not None for v in values) else None


def load(study, stage):
    rows, configs, matching, coverage = [], {}, {}, []
    for arch in ARCHES:
        lane = study / stage / arch
        if not (lane / 'result.json').exists():
            raise RuntimeError(f'Lane has no terminal result: {lane}')
        cfg = json.loads((lane / 'config.used.json').read_text())
        configs[arch] = cfg
        case_rows = []
        paths = sorted((lane / 'cases').glob('*/layer_metrics.csv'))
        if len(paths) != len(cfg['cases']):
            raise RuntimeError(f'Case coverage differs for {arch}: {len(paths)}')
        for path in paths:
            if not (path.parent / 'result.json').exists():
                raise RuntimeError(f'Case has no terminal result: {path.parent}')
            sample = read_csv(path)
            case_rows.extend(sample)
            if not sample:
                raise RuntimeError(f'Empty case: {path}')
            key = (arch, sample[0]['scheme'], float(sample[0]['relative_target']))
            matching[key] = json.loads((path.parent / 'matching.json').read_text())
        expected = set()
        for case in cfg['cases']:
            r = float(case['relative_target'])
            for param in layers(arch):
                expected.update((case['scheme'], r, param, 0., -1, b) for b in range(36))
                for eta in case.get('noise_levels', cfg['noise_levels']):
                    expected.update((case['scheme'], r, param, float(eta), d, b)
                                    for d in range(len(cfg['read_noise_seeds']))
                                    for b in range(cfg['noisy_batches']))
        seen = set()
        undefined = 0
        for row in case_rows:
            if row['architecture'] != arch:
                raise RuntimeError(f'Mixed architecture in {lane}')
            eta, draw = float(row['noise_level']), int(row['noise_draw_index'])
            key = (row['scheme'], float(row['relative_target']), row['parameter_name'],
                   eta, draw, int(row['batch_index']))
            if key in seen or key not in expected:
                raise RuntimeError(f'Duplicate/unexpected comparison: {arch}/{key}')
            seen.add(key)
            seed = number(row['noise_seed'])
            if seed != (cfg['read_noise_seeds'][draw] if eta else None):
                raise RuntimeError(f'Unexpected acquisition seed: {arch}/{key}')
            if eta and row['noise_model'] != 'nodewise_relative':
                raise RuntimeError('The assigned noise model is nodewise relative')
            if str(row.get('matched', 'true')).lower() not in ('true', '1'):
                raise RuntimeError(f'Unmatched displacement in {arch}/{key}')
            for field in METRICS:
                value = number(row.get(field))
                if field in ('cosine', 'noisy_clean_eqprop_cosine'):
                    undefined += value is None
                    if value is not None and abs(value) > 1.00000001:
                        raise RuntimeError(f'Invalid cosine: {arch}/{key}')
        if seen != expected:
            raise RuntimeError(f'Incomplete {arch}: {len(seen)}/{len(expected)} comparisons')
        coverage.append(dict(architecture=arch, contexts=len(paths), rows=len(seen),
                             undefined_alignment_values=undefined,
                             clean_batches=36, noisy_batches=cfg['noisy_batches']))
        rows.extend(case_rows)
    coords = [{(float(c['relative_target']), float(e)) for c in configs[a]['cases']
               for e in c.get('noise_levels', configs[a]['noise_levels'])} for a in ARCHES]
    if not coords[0] == coords[1] == coords[2]:
        raise RuntimeError('Architectures do not share identical (D/F, eta) coordinates')
    return rows, configs, matching, coverage


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row['architecture'], row['scheme'], float(row['relative_target']),
               float(row['noise_level']), row['parameter_name'], int(row['noise_draw_index']))
        groups[key].append(row)
    per_draw = []
    for key, samples in sorted(groups.items()):
        entry = dict(zip(('architecture', 'scheme', 'relative_target', 'noise_level',
                          'parameter_name', 'noise_draw_index'), key))
        entry['batches'] = len(samples)
        for field in METRICS:
            entry[field] = safe_mean([number(r.get(field)) for r in samples])
        per_draw.append(entry)
    grouped = defaultdict(list)
    for row in per_draw:
        key = tuple(row[k] for k in ('architecture', 'scheme', 'relative_target',
                                    'noise_level', 'parameter_name'))
        grouped[key].append(row)
    cells = []
    for key, draws in sorted(grouped.items()):
        entry = dict(zip(('architecture', 'scheme', 'relative_target', 'noise_level',
                          'parameter_name'), key))
        entry['draws'] = len(draws)
        for field in METRICS:
            values = [r[field] for r in draws]
            entry[field] = safe_mean(values)
            entry[field + '_min'] = min(values) if entry[field] is not None else None
            entry[field + '_max'] = max(values) if entry[field] is not None else None
        cells.append(entry)
    lookup = {(r['architecture'], r['scheme'], r['relative_target'], r['noise_level'],
               r['parameter_name'], r['noise_draw_index']): r for r in per_draw}
    contrasts = []
    for key, ours in lookup.items():
        arch, scheme, r, eta, param, draw = key
        if scheme != 'ours' or not eta:
            continue
        legacy = lookup[(arch, 'legacy', r, eta, param, draw)]
        baseline = lookup[(arch, 'baseline', r, eta, param, draw)]
        entry = {k: ours[k] for k in ('architecture', 'relative_target', 'noise_level',
                                      'parameter_name', 'noise_draw_index')}
        for metric in ('cosine', 'noisy_clean_eqprop_cosine'):
            o, l, b = ours[metric], legacy[metric], baseline[metric]
            entry['ours_' + metric] = o
            entry['legacy_' + metric] = l
            entry['baseline_' + metric] = b
            entry['delta_' + metric] = o-l if o is not None and l is not None else None
            entry['delta_baseline_' + metric] = o-b if o is not None and b is not None else None
            entry['delta_best_other_' + metric] = o-max(l, b) if all(x is not None for x in (o,l,b)) else None
        contrasts.append(entry)
    grouped = defaultdict(list)
    for row in contrasts:
        key = tuple(row[k] for k in ('architecture', 'relative_target', 'noise_level', 'parameter_name'))
        grouped[key].append(row)
    differences = []
    for key, draws in sorted(grouped.items()):
        entry = dict(zip(('architecture', 'relative_target', 'noise_level', 'parameter_name'), key))
        for metric in contrasts[0]:
            if metric.startswith(('ours_', 'legacy_', 'baseline_', 'delta_')):
                values = [r[metric] for r in draws]
                entry[metric] = safe_mean(values)
                entry[metric+'_min'] = min(values) if entry[metric] is not None else None
                entry[metric+'_max'] = max(values) if entry[metric] is not None else None
        differences.append(entry)
    return cells, per_draw, contrasts, differences


def peak_rows(differences):
    peaks, near = [], []
    for arch in ARCHES:
        for layer in layers(arch):
            eligible = [r for r in differences if r['architecture']==arch and
                        r['parameter_name']==layer and r['delta_cosine'] is not None]
            if not eligible:
                continue
            best = min(eligible, key=lambda r: (-r['delta_cosine'], r['noise_level'], r['relative_target']))
            peaks.append(best)
            if best['delta_cosine'] > 0:
                near.extend(r for r in eligible if r['delta_cosine'] >= .95*best['delta_cosine'])
    return peaks, near


def select(differences):
    peaks, _ = peak_rows(differences)
    patches, union = [], set()
    for best in peaks:
        if not best['parameter_name'].startswith('ConvWeight_'):
            continue
        arch, layer = best['architecture'], best['parameter_name']
        surface = {(r['relative_target'], r['noise_level']):r['delta_cosine'] for r in differences
                   if r['architecture']==arch and r['parameter_name']==layer}
        xs = sorted({r for r, eta in surface})
        ys = sorted({eta for r, eta in surface})
        r, eta = best['relative_target'], best['noise_level']
        def adjacent(values, value, coordinate):
            i = values.index(value)
            neighbors = [values[j] for j in (i-1, i+1) if 0 <= j < len(values)]
            def priority(v):
                score = surface[coordinate(v)]
                return (math.inf if score is None else -score, v)
            return min(neighbors, key=priority)
        if r == xs[0]:
            rnew = .5
        elif r == xs[-1]:
            rnew = 10.
        else:
            rnew = (r + adjacent(xs, r, lambda v:(v,eta))) / 2
        if eta == ys[0]:
            enew = max(1e-9, eta/math.sqrt(ys[1]/eta))
        elif eta == ys[-1]:
            enew = min(2e-2, eta*math.sqrt(eta/ys[-2]))
        else:
            enew = math.sqrt(eta*adjacent(ys, eta, lambda v:(r,v)))
        coords = sorted({(x,y) for x in (r,rnew) for y in (eta,enew)})
        if len(coords) != 4:
            raise RuntimeError('Refinement patch must have four distinct coordinates')
        union.update(coords)
        patches.append(dict(architecture=arch, parameter_name=layer,
                            screen_peak=best, coordinates=[list(c) for c in coords]))
    if len(patches) != 6:
        raise RuntimeError('Cannot select a defined peak for every convolution')
    return dict(patches=patches, shared_coordinates=[list(c) for c in sorted(union)],
                read_noise_seeds=[121000000,122000000,123000000],
                noisy_batches=36, conditional_on_one_initializer=True)


def refinement_configs(study, selection, configs, matching):
    by_target = defaultdict(list)
    for r, eta in selection['shared_coordinates']:
        by_target[r].append(eta)
    outputs = []
    for arch in ARCHES:
        cfg = copy.deepcopy(configs[arch])
        cfg.update(stage='refinement', output_root=str(study / 'stage_b' / arch),
                   read_noise_seeds=selection['read_noise_seeds'], noisy_batches=36,
                   relative_targets=sorted(by_target),
                   noise_levels=sorted({e for r,e in selection['shared_coordinates']}))
        cfg['prior_gpu_seconds'] = stage_seconds(study / 'stage_a' / arch)
        cases = []
        for scheme in SCHEMES:
            refs = [c for c in configs[arch]['cases'] if c['scheme']==scheme]
            for r, etas in sorted(by_target.items()):
                ref = min(refs, key=lambda c:abs(float(c['relative_target'])-r))
                match = matching[(arch,scheme,float(ref['relative_target']))]
                beta = match.get('injected_beta', match.get('actual_injected_beta'))
                if beta is None:
                    raise RuntimeError('matching.json must identify attained injected_beta')
                c = copy.deepcopy(ref)
                c.update(relative_target=r, injected_beta=float(beta)*r/float(ref['relative_target']),
                         beta_reference_target=r, noise_levels=sorted(etas),
                         name=f'{arch}_{scheme}_relative{r:g}'.replace('.', 'p'))
                cases.append(c)
        cfg['cases'] = cases
        cfg['expected_case_count'] = len(cases)
        cfg['expected_layer_comparisons'] = sum((36+36*3*len(c['noise_levels']))*(int(arch[-1])+1) for c in cases)
        path = study / 'inputs' / f'stage_b_{arch}.json'
        if path.exists() and json.loads(path.read_text()) != cfg:
            raise RuntimeError(f'Refusing to change already frozen refinement config: {path}')
        write_json(path,cfg)
        outputs.append(str(path))
    return outputs


def stage_seconds(lane):
    # Launcher wall time includes interpreter startup and teardown as well as replay.
    study = lane.parent.parent
    candidates = [study/'launch'/lane.parent.name/lane.name/'production/measurement.json',
                  study/'collected'/lane.name/'launch'/lane.parent.name/lane.name/'production/measurement.json']
    for receipt in candidates:
        if receipt.exists():
            return float(json.loads(receipt.read_text())['wall_seconds'])
    result = json.loads((lane/'result.json').read_text())
    summary = json.loads((lane/'summary.json').read_text()) if (lane/'summary.json').exists() else {}
    sources = [summary, result.get('terminal_metrics',{}), result]
    for source in sources:
        for key in ('elapsed_seconds','wall_seconds','gpu_seconds','runtime_seconds'):
            if key in source:
                return float(source[key])
    raise RuntimeError(f'No elapsed runtime for remaining-budget calculation: {lane}')


def worst_layer(cells):
    lookup = {(r['architecture'],r['scheme'],r['relative_target'],r['noise_level'],r['parameter_name']):r for r in cells}
    points = sorted({(r['architecture'],r['relative_target'],r['noise_level']) for r in cells if r['noise_level']})
    out = []
    for arch, target, eta in points:
        entry = dict(architecture=arch,relative_target=target,noise_level=eta)
        for scheme in SCHEMES:
            values=[lookup[(arch,scheme,target,eta,layer)]['cosine'] for layer in layers(arch)]
            entry[scheme]=min(values) if all(v is not None for v in values) else None
        entry['ours_minus_legacy'] = entry['ours']-entry['legacy'] if entry['ours'] is not None and entry['legacy'] is not None else None
        out.append(entry)
    return out


def physical_controls(study, stage, destination):
    physical, noises = [], []
    for arch in ARCHES:
        states, acquisitions = [], []
        for path in sorted((study/stage/arch/'cases').glob('*/result.json')):
            states.extend(read_csv(path.parent/'state_signal.csv'))
            acquisitions.extend(read_csv(path.parent/'endpoint_read_noise.csv'))
        state_rows, noise_rows = scale_controls(states, acquisitions)
        physical.extend(dict(architecture=arch, **r) for r in state_rows)
        noises.extend(dict(architecture=arch, **r) for r in noise_rows)
    write_csv(destination/'physical_state_controls.csv',physical)
    write_csv(destination/'noise_scale_controls.csv',noises)


def plot(cells, differences, out, stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for arch in ARCHES:
        params = layers(arch)
        fig, axes = plt.subplots(5,len(params),figsize=(4*len(params),14),squeeze=False,
                                 sharex=True,sharey=True,constrained_layout=True)
        for col,param in enumerate(params):
            for i,scheme in enumerate(SCHEMES):
                rows=[r for r in cells if r['architecture']==arch and r['parameter_name']==param
                      and r['scheme']==scheme and r['noise_level'] and r['cosine'] is not None]
                dots=axes[i,col].scatter([r['relative_target'] for r in rows],[r['noise_level'] for r in rows],
                    c=[r['cosine'] for r in rows],cmap='viridis',vmin=0,vmax=1,marker='s',s=85)
                if col==len(params)-1:
                    fig.colorbar(dots,ax=axes[i,:],label='Cosine to finite-K BPTT')
                if col==0:
                    axes[i,col].set_ylabel(f'{scheme}\nGlobal eta')
            rows=[r for r in differences if r['architecture']==arch and r['parameter_name']==param]
            for i,metric in ((3,'delta_cosine'),(4,'delta_baseline_cosine')):
                valid=[r for r in rows if r[metric] is not None]
                dots=axes[i,col].scatter([r['relative_target'] for r in valid],[r['noise_level'] for r in valid],
                    c=[r[metric] for r in valid],cmap='coolwarm',vmin=-1,vmax=1,marker='s',s=85)
                if col==len(params)-1:
                    fig.colorbar(dots,ax=axes[i,:],label='Paired cosine difference')
                if col==0:
                    axes[i,col].set_ylabel(('ours − legacy' if i==3 else 'ours − baseline')+'\nGlobal eta')
            axes[0,col].set_title(param)
            axes[-1,col].set_xlabel('Matched output D/F')
        for ax in axes.flat:
            ax.set_yscale('log')
            ax.set_ylim(7e-10,3e-2)
            ax.set_xlim(0,10.5)
            ax.grid(alpha=.15)
        fig.suptitle(f'{arch}: {stage}, initialization only\nOne eta for every noisy layer; identical coordinates across architectures. Measured points only.')
        for ext in ('png','pdf'):
            fig.savefig(out/f'{arch}_cosine_map.{ext}',dpi=170)
        plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,arch in zip(axes,ARCHES):
        for param in layers(arch)[:-1]:
            rows=sorted([r for r in differences if r['architecture']==arch and r['parameter_name']==param and r['delta_cosine'] is not None],
                        key=lambda r:r['noise_level']/r['relative_target'])
            ax.scatter([r['noise_level']/r['relative_target'] for r in rows],
                       [r['delta_cosine'] for r in rows],s=15,label=param)
        ax.set_xscale('log');ax.set_title(arch);ax.set_xlabel('eta / (output D/F)')
        ax.set_ylabel('ours − legacy cosine');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(out/'noise_displacement_ridge.png',dpi=170)
    fig.savefig(out/'noise_displacement_ridge.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--stage',choices=('stage_a','stage_b'),required=True)
    parser.add_argument('--prepare-refinement',action='store_true')
    args=parser.parse_args()
    study=args.study_root
    rows,configs,matching,coverage=load(study,args.stage)
    cells,draws,contrasts,differences=summarize(rows)
    peaks,near=peak_rows(differences)
    out=study/'analysis'/args.stage
    out.mkdir(parents=True,exist_ok=True)
    for name,data in (('layer_summary',cells),('per_draw_summary',draws),('per_draw_differences',contrasts),
                      ('layer_differences',differences),('peaks',peaks),('near_peak_cells',near),
                      ('worst_layer',worst_layer(cells))):
        write_csv(out/f'{name}.csv',data)
    write_json(out/'coverage.json',dict(stage=args.stage,lanes=coverage,total_rows=len(rows),
               evidence='exploratory',canonical_source_revalidation=False,cross_device_bptt_equivalence_checked=False))
    physical_controls(study,args.stage,out)
    if args.prepare_refinement:
        if args.stage!='stage_a':
            raise RuntimeError('Refinement selection uses Stage A only')
        selection=select(differences)
        path=out/'selection.json'
        if path.exists() and json.loads(path.read_text())!=selection:
            raise RuntimeError('Refusing to change frozen refinement selection')
        write_json(path,selection)
        write_json(out/'refinement_configs.json',refinement_configs(study,selection,configs,matching))
    plot(cells,differences,out,args.stage)
    lines=[f'# Exp013 {args.stage}: uniform relative-noise map','',
           f'Collected {len(rows):,} comparisons across all three architectures. '
           'Each measurement uses one global eta across all noninput layers; all architectures share coordinates.','',
           'Exploratory initialization-only evidence, one model seed and one validation cohort. '
           'Canaries/tests and additional qualification were waived. Cross-device BPTT equivalence is unchecked. '
           'Noise draws are not model seeds; selected maxima are conditional on the sampled domain.','',
           '| Architecture | Layer | Output D/F | eta | Ours cosine | Legacy cosine | Difference |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in peaks:
        if r['parameter_name'].startswith('ConvWeight_'):
            lines.append(f"| {r['architecture']} | {r['parameter_name']} | {r['relative_target']:g} | {r['noise_level']:.5g} | {r['ours_cosine']:.4f} | {r['legacy_cosine']:.4f} | {r['delta_cosine']:+.4f} |")
    lines+=['','The 95%-of-peak table is descriptive, not a confidence interval. '
            'A large gap in one layer does not establish whole-network alignment or training benefit.','']
    if args.stage=='stage_b':
        selection=json.loads((study/'analysis/stage_a/selection.json').read_text())
        decision=[]
        for patch in selection['patches']:
            selected=[r for r in contrasts if r['architecture']==patch['architecture'] and
                      r['parameter_name']==patch['parameter_name'] and
                      [r['relative_target'],r['noise_level']] in patch['coordinates']]
            values=[r['delta_cosine'] for r in selected]
            passed=len(values)==12 and all(v is not None and v>0 for v in values)
            decision.append(dict(architecture=patch['architecture'],parameter_name=patch['parameter_name'],
                                 observed_draw_cells=len(values),positive_draw_cells=sum(v is not None and v>0 for v in values),
                                 neighborhood_supported=passed))
        write_json(out/'neighborhood_decisions.json',decision)
        lines+=['Fresh-draw neighborhood criterion:']+[f"- {r['architecture']} {r['parameter_name']}: {r['positive_draw_cells']}/{r['observed_draw_cells']} positive cell/draw differences; supported={r['neighborhood_supported']}." for r in decision]+['']
    else:
        lines+=['Screen noisy estimates use 192 examples; clean matching uses 576. '
                'These peaks require the predeclared fresh-noise/full-cohort refinement.','']
    lines += [f'[{a} map]({a}_cosine_map.png)' for a in ARCHES]
    lines += ['','[Noise/displacement ridge](noise_displacement_ridge.png) · [All layer differences](layer_differences.csv) · [Worst-layer scores](worst_layer.csv)']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(stage=args.stage,rows=len(rows),report=str(out/'report.md'))))


if __name__=='__main__':
    main()
