"""Offline, complete-case analysis for the series005 initialization screen.

Run only on locally collected canonical bundles; no simulation or optimizer use.
Partial evidence remains visible, but cannot generate confirmed region claims.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from experiments.reporting import validate_run

ARCHITECTURES = ('conv1', 'conv2', 'conv3')
SCHEMES = ('baseline', 'ours', 'legacy')
TARGETS = (.2, .8, 2., 6., 10.)
SIGMAS = (0., .0001, .0005, .001, .002, .005)
SEEDS = (91000000, 92000000, 93000000)
RUNNER_SHA256 = 'cacca7852b8949164c65d19761ec7541acc3fd8ec18f0e77facae72131d06e0a'
CONFIG_SHA256 = {
    'conv1':'1e65781a41bf17c1cc50d2f28e7db3609452ae16b05e1157ee4f421a8a12b20a',
    'conv2':'9e999b1fd5d32b1d9df79258de4d3a5e26649a3431c9d973dbb6d391d01a2eac',
    'conv3':'2ab9189772b0617c8ad96af04a1724bbe88a0990a84d20c6a0d099026d93b4a4',
}


def layers(architecture):
    return tuple(f'ConvWeight_{i}' for i in range(int(architecture[-1]))) + ('DenseWeight_0',)


def number(value):
    if value is None or value == '' or str(value).lower() in ('none', 'null'):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*q
    low = int(position)
    high = min(low+1, len(values)-1)
    return values[low] + (position-low)*(values[high]-values[low])


def summarize(rows):
    """Every expected layer/cell is emitted; incomplete cosines never become a win."""
    grouped = defaultdict(dict)
    for row in rows:
        a, s = row['architecture'], row['scheme']
        r, sigma = float(row['relative_target']), float(row['readout_sigma'])
        p = row['parameter_name'].strip()
        if a not in ARCHITECTURES or s not in SCHEMES or r not in TARGETS or sigma not in SIGMAS or p not in layers(a):
            raise ValueError(f'Unexpected scientific cell: {(a,s,r,sigma,p)}')
        sample = (int(row['batch_index']), int(row['noise_draw_index']))
        expected_draws = (-1,) if sigma == 0 else (0, 1, 2)
        if sample[0] not in range(36) or sample[1] not in expected_draws:
            raise ValueError(f'Unexpected batch/draw: {sample}')
        seed = number(row.get('noise_seed'))
        if sigma and seed != SEEDS[sample[1]]:
            raise ValueError('Noise seed does not match the frozen screen draw')
        if not sigma and seed is not None:
            raise ValueError('Clean reference must not have a noise seed')
        key = (a,s,r,sigma,p)
        if sample in grouped[key]:
            raise ValueError(f'Duplicate scientific sample: {key}/{sample}')
        value = number(row['cosine'])
        if value is not None and not -1.0000000001 <= value <= 1.0000000001:
            raise ValueError('Cosine is outside [-1,1]')
        grouped[key][sample] = value
    means, worst, differences = [], [], []
    lookup = {}
    for a in ARCHITECTURES:
        for s in SCHEMES:
            for r in TARGETS:
                for sigma in SIGMAS:
                    count = 36*(1 if sigma == 0 else 3)
                    expected = {(b,d) for b in range(36) for d in ((-1,) if sigma == 0 else (0,1,2))}
                    layer_means = []
                    for p in layers(a):
                        samples = grouped[(a,s,r,sigma,p)]
                        values = [v for v in samples.values() if v is not None]
                        complete = set(samples) == expected and len(values) == count
                        measured_mean = sum(values)/len(values) if values else None
                        entry = dict(architecture=a,scheme=s,relative_target=r,readout_sigma=sigma,
                            parameter_name=p,expected_samples=count,observed_samples=len(samples),
                            defined_samples=len(values),complete=complete,
                            cosine_mean=measured_mean if complete else None,
                            observed_defined_mean=measured_mean,cosine_median=quantile(values,.5),
                            cosine_p10=quantile(values,.1),cosine_p90=quantile(values,.9))
                        means.append(entry)
                        lookup[(a,s,r,sigma,p)] = entry
                        layer_means.append(entry['cosine_mean'])
                    valid = all(v is not None for v in layer_means)
                    worst.append(dict(architecture=a,scheme=s,relative_target=r,readout_sigma=sigma,
                                      complete=valid,worst_layer_cosine=min(layer_means) if valid else None))
    for a in ARCHITECTURES:
        for s in ('baseline','ours'):
            for r in TARGETS:
                for sigma in SIGMAS:
                    for p in layers(a):
                        left, right = lookup[(a,s,r,sigma,p)], lookup[(a,'legacy',r,sigma,p)]
                        valid = left['complete'] and right['complete']
                        delta = None
                        if valid:
                            x,y = grouped[(a,s,r,sigma,p)],grouped[(a,'legacy',r,sigma,p)]
                            delta = sum(x[k]-y[k] for k in sorted(x))/len(x)
                        differences.append(dict(architecture=a,scheme=s,relative_target=r,
                            readout_sigma=sigma,parameter_name=p,complete=valid,cosine_delta=delta))
    return means, differences, worst


def select_candidates(worst):
    """Predeclared largest-minimum-margin adjacent pair; never a best-scheme beta."""
    lookup = {(x['architecture'],x['scheme'],x['relative_target'],x['readout_sigma']):
              x['worst_layer_cosine'] if x['complete'] else None for x in worst}
    output = []
    nonzero = SIGMAS[1:]
    for a in ARCHITECTURES:
        options = []
        for s in ('baseline','ours'):
            margins = {}
            for r in TARGETS:
                for sigma in nonzero:
                    left,right = lookup.get((a,s,r,sigma)),lookup.get((a,'legacy',r,sigma))
                    margins[(r,sigma)] = left-right if left is not None and right is not None else None
            for i,r in enumerate(TARGETS):
                for j,sigma in enumerate(nonzero):
                    first = (r,sigma)
                    neighbors = []
                    if i+1 < len(TARGETS): neighbors.append((TARGETS[i+1],sigma))
                    if j+1 < len(nonzero): neighbors.append((r,nonzero[j+1]))
                    for second in neighbors:
                        x,y = margins[first],margins[second]
                        if x is not None and y is not None and x > 0 and y > 0:
                            options.append(dict(architecture=a,competitor=s,cells=[list(first),list(second)],
                                margins=[x,y],minimum_margin=min(x,y)))
        if options:
            output.append(sorted(options,key=lambda x:(-x['minimum_margin'],x['cells'][0],
                0 if x['competitor']=='baseline' else 1,x['cells'][1]))[0])
    return output


def per_draw_summaries(rows):
    groups=defaultdict(dict)
    for row in rows:
        key=(row['architecture'],row['scheme'],float(row['relative_target']),float(row['readout_sigma']),
             int(row['noise_draw_index']),row['parameter_name'].strip())
        groups[key][int(row['batch_index'])]=number(row['cosine'])
    layers_out=[]; scores=defaultdict(dict)
    for (a,s,r,sigma,d,p),batches in sorted(groups.items()):
        valid=set(batches)==set(range(36)) and all(v is not None for v in batches.values())
        mean=sum(batches.values())/36 if valid else None
        layers_out.append(dict(architecture=a,scheme=s,relative_target=r,readout_sigma=sigma,
                               noise_draw_index=d,parameter_name=p,complete=valid,cosine_mean=mean))
        scores[(a,s,r,sigma,d)][p]=mean
    worst=[]
    for (a,s,r,sigma,d),means in sorted(scores.items()):
        valid=set(means)==set(layers(a)) and all(v is not None for v in means.values())
        worst.append(dict(architecture=a,scheme=s,relative_target=r,readout_sigma=sigma,
                          noise_draw_index=d,complete=valid,worst_layer_cosine=min(means.values()) if valid else None))
    return layers_out,worst


def dominance(differences):
    groups=defaultdict(dict)
    for row in differences:
        key=(row['architecture'],row['scheme'],row['relative_target'],row['readout_sigma'])
        groups[key][row['parameter_name']]=row['cosine_delta']
    rows=[]
    for (a,s,r,sigma),delta in sorted(groups.items()):
        valid=set(delta)==set(layers(a)) and all(v is not None for v in delta.values())
        rows.append(dict(architecture=a,scheme=s,relative_target=r,readout_sigma=sigma,complete=valid,
                         all_layer_dominance=all(v>0 for v in delta.values()) if valid else None,
                         minimum_layer_delta=min(delta.values()) if valid else None))
    return rows


def state_summary(rows):
    """Pool physical RMS by squared values/counts; never average batch ratios."""
    groups = defaultdict(dict)
    for row in rows:
        key = (row['architecture'], row['scheme'], float(row['relative_target']),
               int(row['state_layer_index']))
        batch = int(row['batch_index'])
        a,s,r,index=key
        if a not in ARCHITECTURES or s not in SCHEMES or r not in TARGETS or index not in range(int(a[-1])+1):
            raise ValueError('Unexpected state context/layer')
        count=int(row['element_count'])
        if count <= 0 or batch not in range(36): raise ValueError('Invalid state count/batch')
        if index==int(a[-1]) and count != 320: raise ValueError('Output must contain16×20 elements')
        if batch in groups[key]:
            raise ValueError(f'Duplicate state sample: {key}/{batch}')
        groups[key][batch] = row
    summary = []
    for (a,s,r,index), batches in sorted(groups.items()):
        complete = set(batches) == set(range(36))
        values = list(batches.values())
        count = sum(int(x['element_count']) for x in values)
        if count <= 0: raise ValueError('State RMS element count must be positive')
        def pooled(field):
            numbers = [number(x[field]) for x in values]
            if any(x is None or x < 0 for x in numbers): return None
            return math.sqrt(sum(int(row['element_count'])*x*x for row,x in zip(values,numbers))/count)
        free = pooled('free_rms'); displacement = pooled('centered_half_difference_rms')
        ratio = displacement/free if displacement is not None and free is not None and free > 0 else None
        is_output = index == int(a[-1])
        match = complete and ratio is not None and abs(ratio/r-1) <= .01
        entry = dict(architecture=a,scheme=s,relative_target=r,state_layer_index=index,
            state_layer_name=values[0]['state_layer_name'],is_output=is_output,complete=complete,
            free_rms=free,centered_displacement_rms=displacement,measured_relative_displacement=ratio,
            output_match_passed=match if is_output else None,element_count=count)
        for sigma in SIGMAS[1:]:
            entry[f'displacement_over_sigma_{sigma:g}'] = displacement/sigma if displacement is not None else None
        summary.append(entry)
    return summary



def gradient_scale_summary(rows):
    """Arithmetic means of per-sample norms; no gradient averaging or rescaling."""
    fields=('bptt_rms','clean_eqprop_rms','eqprop_rms','added_noise_rms',
            'noise_over_clean_norm','relative_l2_difference_over_bptt')
    grouped=defaultdict(lambda:defaultdict(list))
    keys=('architecture','scheme','relative_target','readout_sigma','parameter_name')
    for row in rows:
        key=(row['architecture'],row['scheme'],float(row['relative_target']),
             float(row['readout_sigma']),row['parameter_name'].strip())
        for field in fields:
            value=number(row.get(field))
            if value is not None: grouped[key][field].append(value)
    output=[]
    for key,data in sorted(grouped.items()):
        record=dict(zip(keys,key))
        for field in fields:
            values=data[field]
            record[field+'_mean']=sum(values)/len(values) if values else None
            record[field+'_defined_samples']=len(values)
        output.append(record)
    return output

def write_csv(path, rows):
    with Path(path).open('w',newline='') as stream:
        if not rows: return
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def audit_bundle(run, manifest, rows, shared):
    """Verify the declared initialization experiment as well as reporting syntax."""
    root = Path(__file__).resolve().parents[1]
    result = json.loads((run/'result.json').read_text())
    terminal = result['terminal_metrics']
    cfg, case = manifest['configuration']['resolved'], manifest['inputs']
    a,s,r = case['architecture'],case['scheme'],float(case['relative_target'])
    depth = int(a[-1]); horizon = {1:4,2:6,3:8}[depth]
    def require(ok, message):
        if not ok: raise ValueError(f'{run}: {message}')
    def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    require(manifest['checkpoint_epoch']==0 and manifest['model_seed']==0,'Wrong checkpoint/seed')
    require(manifest['checkpoint_role']=='reconstructed_initialization','Wrong checkpoint role')
    require(manifest['runtime']['device']=='cuda','Replay did not use CUDA')
    require(tuple(cfg['noise_sigmas'])==SIGMAS and tuple(cfg['read_noise_seeds'])==SEEDS,'Noise grid changed')
    require(manifest['cohort_sha256']==cfg['expected_cohort_sha256']==
        '95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd','Wrong cohort')
    for flag in ('official_test_read','optimizer_steps_applied','accuracy_evaluation'):
        require(manifest[flag] is False and terminal[flag] is False,'Unexpected '+flag)
    for flag in ('source_bytes_unchanged','all_parameter_tensors_unchanged','all_bias_tensors_exact_zero'):
        require(terminal[flag] is True,'Failed '+flag)
    require(terminal['replay_batches']==36 and terminal['layer_comparisons']==36*16*(depth+1),'Short replay')
    require(len(rows)==36*16*(depth+1),'CSV coverage differs from result')
    require(tuple(case['parameter_order'])==layers(a),'Unexpected parameter ordering')
    require(case['T']==case['K']==horizon,'Wrong horizon')
    require(digest(run/'checkpoints/initialization.pt')==case['initializer_checkpoint_sha256'],'Initializer bytes changed')
    require(digest(root/case['source_config'])==case['source_config_sha256'],'Source config changed')
    proof=case['calibration_proof']
    require(digest(root/proof['path'])==proof['sha256'],'Calibration proof changed')
    calibration=case['calibration']
    require(abs(calibration['relative_output_rms']/r-1)<=.01,'Unmatched declared calibration')
    require(abs(terminal['actual_relative_output_rms']/r-1)<=.01,'Unmatched actual output')
    require(shared.setdefault(('initializer',a),case['initializer_checkpoint_sha256'])==case['initializer_checkpoint_sha256'],'Unmatched scheme initialization')
    require(shared.setdefault(('host',a),manifest['runtime']['host'])==manifest['runtime']['host'],'Architecture split across hosts')
    require(shared.setdefault(('runner',),manifest['git']['runner_sha256'])==manifest['git']['runner_sha256'],'Replay implementation differs')
    canonical_path=root/f'configs/conv/initial_displacement_noise_{a}_20260925.json'
    require(digest(canonical_path)==CONFIG_SHA256[a],'Frozen canonical config changed')
    canonical=json.loads(canonical_path.read_text())
    expected=copy.deepcopy(canonical)
    expected['dataset']=json.loads((root/canonical['cohort_config']).read_text())['dataset']
    actual=copy.deepcopy(cfg)
    expected['dataset']['root']=actual['dataset']['root']='<dataset-location>'
    require(actual==expected,'Resolved scientific config differs from frozen config')
    require(manifest['configuration']['sha256']==CONFIG_SHA256[a],'Unexpected config digest')
    require(manifest['git']['runner_sha256']==RUNNER_SHA256,'Unexpected replay implementation')
    expected_case=next((x for x in canonical['cases'] if x['name']==manifest['arm_id']),None)
    require(expected_case is not None and all(case.get(k)==v for k,v in expected_case.items()),'Declared case differs from frozen case')
    voltage,current={'baseline':(1.,1.),'ours':(4.,1.),'legacy':(4.,.25)}[s]
    for row in rows:
        require((row['architecture'],row['scheme'],float(row['relative_target']))==(a,s,r),'Row/context mismatch')
        require(int(row['T'])==int(row['K'])==horizon and int(row['checkpoint_epoch'])==0,'Row horizon/state mismatch')
        require(float(row['injected_beta'])==case['injected_beta'],'Injected beta changed')
        require(math.isclose(float(row['base_beta'])*(voltage/current)**depth,case['injected_beta'],rel_tol=1e-13),'Beta conversion mismatch')
        b=int(row['batch_index']); p=row['parameter_name'].strip()
        for key,value in [(('batch_indices',b),row['batch_source_indices_sha256']),
                          (('batch_payload',a,b),row['batch_payload_sha256']),
                          (('BPTT',a,s,b,p),row['bptt_gradient_sha256'])]:
            require(shared.setdefault(key,value)==value,'Unmatched batch or BPTT reference')
    guards=json.loads((run/'read_only_guards.json').read_text())
    require(len(guards)==36,'Missing read-only batch guards')
    for guard in guards:
        for key in ('iteration_contract_passed','parameter_tensors_unchanged_after_cast',
                    'frozen_current_force_matches_post_T_cost_gradient','frozen_current_force_unchanged_across_phases'):
            require(guard[key] is True,'Failed '+key)
        require(guard['precision']=='float64','Wrong precision')
        require(guard['reconstructed_initialization_tensor_sha256']==case['reconstructed_initialization_tensor_sha256'],'Wrong initialized tensors')
    with (run/'endpoint_read_noise.csv').open(newline='') as stream: noise=list(csv.DictReader(stream))
    require(len(noise)==36*15*2*(depth+1),'Missing endpoint draws')
    noise_keys=set()
    for draw in noise:
        b=int(draw['batch_index']); seed=int(draw['noise_seed']); sigma=float(draw['endpoint_read_noise_std'])
        phase=draw['phase']; layer=int(draw['state_layer_index']); index=int(draw['noise_draw_index'])
        require(index in range(3) and b in range(36) and layer in range(depth+1) and phase in ('negative','positive'),'Invalid endpoint noise coordinates')
        require(seed==SEEDS[index] and sigma in SIGMAS[1:],'Unassigned noise draw')
        require(int(draw['seed'])==seed+1000*b+100*{'negative':1,'positive':2}[phase]+layer,'Noise RNG changed')
        key=(a,sigma,seed,b,phase,layer)
        require(key not in noise_keys,'Duplicate endpoint draw');noise_keys.add(key)
        key=('noise',*key)
        require(shared.setdefault(key,draw['standard_normal_sha256'])==draw['standard_normal_sha256'],'Endpoint noise not paired across schemes/targets')
    expected_noise={(a,sigma,seed,b,phase,layer) for sigma in SIGMAS[1:] for seed in SEEDS
                    for b in range(36) for phase in ('negative','positive') for layer in range(depth+1)}
    require(noise_keys==expected_noise,'Endpoint noise Cartesian coverage mismatch')


def collect(roots):
    rows, states, attempts, seen = [], [], [], set()
    shared = {}
    for root in roots:
        for path in sorted(Path(root).rglob('manifest.json')):
            run = path.parent.resolve()
            if run in seen: continue
            seen.add(run)
            manifest = json.loads(path.read_text())
            if manifest.get('smoke'): continue
            if manifest.get('study_id') != 'conv123-initial-displacement-noise-map-20260925-v1': continue
            metric_path = run/'layer_metrics.csv'
            status = json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {'state':'missing_status'}
            entry = dict(run=str(run),state=status['state'],errors=[],has_layer_metrics=metric_path.exists(),
                         reason=status.get('error'))
            if status['state'] == 'complete':
                entry['errors'] = validate_run(run)
                if not metric_path.exists(): entry['errors'].append('Successful run lacks layer_metrics.csv')
                if not entry['errors']:
                    with metric_path.open(newline='') as stream:
                        case_rows = list(csv.DictReader(stream))
                        audit_bundle(run,manifest,case_rows,shared)
                        rows.extend(case_rows)
                    state_path = run/'state_signal.csv'
                    if state_path.exists() and case_rows:
                        context = {k:case_rows[0][k] for k in ('architecture','scheme','relative_target')}
                        with state_path.open(newline='') as stream:
                            states.extend(dict(row,**context) for row in csv.DictReader(stream))
            attempts.append(entry)
    if any(x['errors'] for x in attempts):
        raise ValueError('Invalid local reporting bundles: '+json.dumps(attempts))
    return rows, states, attempts


def plots(output, means, differences, worst, *, partial):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    mean_lookup = {(r['architecture'],r['scheme'],r['relative_target'],r['readout_sigma'],r['parameter_name']):
                   r['cosine_mean'] for r in means}
    delta_lookup = {(r['architecture'],r['scheme'],r['relative_target'],r['readout_sigma'],r['parameter_name']):
                    r['cosine_delta'] for r in differences}
    prefix = 'PARTIAL — ' if partial else ''
    for a in ARCHITECTURES:
        ps = layers(a)
        for kind,schemes,lookup,limit,cmap in [('cosine',SCHEMES,mean_lookup,1.,'RdBu_r'),
                                              ('difference',('baseline','ours'),delta_lookup,2.,'RdBu_r')]:
            fig,axes = plt.subplots(len(schemes),len(ps),figsize=(4.1*len(ps),3.2*len(schemes)),squeeze=False,
                                    constrained_layout=True)
            for i,s in enumerate(schemes):
                for j,p in enumerate(ps):
                    matrix=np.array([[lookup.get((a,s,r,sigma,p)) for r in TARGETS] for sigma in SIGMAS[1:]],dtype=float)
                    ax=axes[i,j]
                    im=ax.imshow(np.ma.masked_invalid(matrix),origin='lower',aspect='auto',vmin=-limit,vmax=limit,cmap=cmap)
                    ax.set_xticks(range(len(TARGETS)),[f'{x:g}' for x in TARGETS]);ax.set_xlabel('Output D/F')
                    ax.set_yticks(range(len(SIGMAS)-1),[f'{x:g}' for x in SIGMAS[1:]]);ax.set_ylabel('Endpoint noise σ')
                    ax.set_title(f'{s}'+(' − legacy' if kind=='difference' else '')+f' · {p}')
                    for y in range(matrix.shape[0]):
                        for x in range(matrix.shape[1]):
                            if np.isfinite(matrix[y,x]): ax.text(x,y,f'{matrix[y,x]:.3f}',ha='center',va='center',fontsize=7)
            fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.75,label='Cosine difference' if kind=='difference' else 'EqProp/BPTT cosine')
            fig.suptitle(prefix+f'{a}: initialization; means of individual batch/draw cosines')
            for suffix in ('png','pdf'): fig.savefig(output/f'{a}_{kind}.{suffix}',dpi=180)
            plt.close(fig)
        fig,axes=plt.subplots(1,len(ps),figsize=(4.1*len(ps),3.4),squeeze=False,constrained_layout=True)
        for j,p in enumerate(ps):
            ax=axes[0,j]
            for s in SCHEMES: ax.plot(TARGETS,[mean_lookup.get((a,s,r,0.,p)) for r in TARGETS],marker='o',label=s)
            ax.set_xscale('log');ax.set_ylim(-1.02,1.02);ax.set_title(p);ax.set_xlabel('Output D/F');ax.set_ylabel('Clean cosine');ax.legend()
        fig.suptitle(prefix+f'{a}: clean reference')
        for suffix in ('png','pdf'):fig.savefig(output/f'{a}_clean.{suffix}',dpi=180)
        plt.close(fig)
    w={(x['architecture'],x['scheme'],x['relative_target'],x['readout_sigma']):x['worst_layer_cosine'] for x in worst}
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    for j,a in enumerate(ARCHITECTURES):
        for i,s in enumerate(('baseline','ours')):
            data=[]
            for sigma in SIGMAS[1:]:
                line=[]
                for r in TARGETS:
                    l,h=w[(a,s,r,sigma)],w[(a,'legacy',r,sigma)]
                    line.append(l-h if l is not None and h is not None else float('nan'))
                data.append(line)
            im=axes[i,j].imshow(np.ma.masked_invalid(data),origin='lower',aspect='auto',vmin=-2,vmax=2,cmap='RdBu_r')
            axes[i,j].set_title(f'{a}: {s} − legacy, worst layer')
            axes[i,j].set_xticks(range(5),[f'{r:g}' for r in TARGETS]);axes[i,j].set_xlabel('Output D/F')
            axes[i,j].set_yticks(range(5),[f'{sigma:g}' for sigma in SIGMAS[1:]]);axes[i,j].set_ylabel('Endpoint noise σ')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Difference of minimum layer means');fig.suptitle(prefix+'Worst-layer alignment')
    for suffix in ('png','pdf'):fig.savefig(output/f'worst_layer_difference.{suffix}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production-root',type=Path,action='append',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    rows,states,attempts=collect(args.production_root)
    means,differences,worst=summarize(rows)
    physical=state_summary(states)
    outputs=[x for x in physical if x['is_output']]
    complete=all(x['complete'] for x in means) and len(outputs)==45 and all(x['output_match_passed'] for x in outputs)
    candidates=select_candidates(worst) if complete else []
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'layer_summary.csv',means);write_csv(out/'layer_differences.csv',differences)
    write_csv(out/'gradient_scale_controls.csv',gradient_scale_summary(rows))
    write_csv(out/'worst_layer.csv',worst);write_csv(out/'physical_state_controls.csv',physical)
    draw_layers,draw_worst=per_draw_summaries(rows)
    write_csv(out/'per_draw_layer_summary.csv',draw_layers);write_csv(out/'per_draw_worst_layer.csv',draw_worst)
    write_csv(out/'all_layer_dominance.csv',dominance(differences))
    (out/'candidate_regions.json').write_text(json.dumps(dict(screen_complete=complete,candidates=candidates,
        interpretation='Candidates only; independent-noise confirmation required.'),indent=2)+'\n')
    validation=dict(screen_complete=complete,layer_rows=len(rows),expected_layer_rows=77760,
        matched_output_contexts=sum(x['output_match_passed'] for x in outputs),expected_output_contexts=45,
        complete_layer_cells=sum(x['complete'] for x in means),expected_layer_cells=810,attempts=attempts,
        scientific_bundle_audit='Canonical bundles, source/calibration/initializer identities, paired noise/BPTT/cohort, read-only guards and matched output verified.')
    (out/'coverage.json').write_text(json.dumps(validation,indent=2)+'\n')
    plots(out,means,differences,worst,partial=not complete)
    print(json.dumps({k:v for k,v in validation.items() if k!='attempts'}))


if __name__=='__main__':
    main()
