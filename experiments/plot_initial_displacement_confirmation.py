"""Validate and summarize the preselected five-draw initialization confirmation."""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from experiments.reporting import validate_run
from experiments.plot_initial_displacement_noise_map import layers,number,state_summary,write_csv

ROOT=Path(__file__).resolve().parents[1]
RUNNER='2525967925b75c566537ea3e0fae67921220f93444964b4c31a01a6f50e01529'
CONFIGS={
 'conv1':'a49c213df6ffcfe62d568078b0f6f40182dcf04dca72cfe527db837ae9ee3138',
 'conv2':'22461b3e6a49efca3c9de59aba689135055e846637b6bb48b12dc905ee7603a5',
 'conv3':'299431f1a98d482fb7920cb93e46357f5b6e878822c0364d4365d8fa047ed66f'}
SEEDS=(94000000,95000000,96000000,97000000,98000000)
COHORT='95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_csv(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))

def summarize(rows,configs):
    """Each fresh draw gets its own min-of-layer-means statistic over36 batches."""
    grouped=defaultdict(dict)
    for row in rows:
        a,s=row['architecture'],row['scheme'];r,n=float(row['relative_target']),float(row['readout_sigma'])
        b,d=int(row['batch_index']),int(row['noise_draw_index']);p=row['parameter_name'].strip()
        cfg=configs[a]
        assert (r,n) in {(c['relative_target'],n) for c in cfg['cases'] for n in cfg['noise_sigmas']}
        assert s in ('baseline','ours','legacy') and p in layers(a) and b in range(36)
        assert d in (range(5) if n else (-1,))
        assert number(row['noise_seed']) == (SEEDS[d] if n else None)
        key=(a,s,r,n,d,p);assert b not in grouped[key], 'Duplicate confirmation sample'
        value=number(row['cosine']);assert value is None or -1.0000000001<=value<=1.0000000001
        grouped[key][b]=value
    means=[];lookup={}
    for a,cfg in configs.items():
        for case in cfg['cases']:
            for n in cfg['noise_sigmas']:
                for d in (range(5) if n else (-1,)):
                    for p in layers(a):
                        key=(a,case['scheme'],case['relative_target'],n,d,p);samples=grouped[key]
                        finite=[v for v in samples.values() if v is not None]
                        complete=set(samples)==set(range(36)) and len(finite)==36
                        value=sum(finite)/36 if complete else None
                        record=dict(architecture=a,scheme=case['scheme'],relative_target=case['relative_target'],
                            readout_sigma=n,noise_draw_index=d,noise_seed=SEEDS[d] if n else None,
                            parameter_name=p,complete=complete,observed_samples=len(samples),defined_samples=len(finite),
                            cosine_mean=value)
                        means.append(record);lookup[key]=value
    contrasts=[];judgments=[];layer_differences=[]
    for a,cfg in configs.items():
        selected=cfg['selected_pair'];selected_contrasts=[]
        for r,n in selected['cells']:
            for d in range(5):
                weights={s:[lookup[(a,s,r,n,d,p)] for p in layers(a)] for s in ('baseline','ours','legacy')}
                ws={s:min(vs) if all(v is not None for v in vs) else None for s,vs in weights.items()}
                for s in ('baseline','ours'):
                    complete=ws[s] is not None and ws['legacy'] is not None
                    delta=ws[s]-ws['legacy'] if complete else None
                    differences=[left-right if left is not None and right is not None else None
                                 for left,right in zip(weights[s],weights['legacy'])]
                    for p,value in zip(layers(a),differences):
                        layer_differences.append(dict(architecture=a,scheme=s,relative_target=r,readout_sigma=n,
                            noise_draw_index=d,noise_seed=SEEDS[d],parameter_name=p,cosine_delta=value))
                    record=dict(architecture=a,scheme=s,relative_target=r,readout_sigma=n,
                        noise_draw_index=d,noise_seed=SEEDS[d],complete=complete,worst_layer_competitor=ws[s],
                        worst_layer_legacy=ws['legacy'],worst_layer_delta=delta,
                        all_layer_dominance=all(v is not None and v>0 for v in differences))
                    contrasts.append(record)
                    if s==selected['competitor']:selected_contrasts.append(record)
        assert len(selected_contrasts)==10
        complete=all(x['complete'] for x in selected_contrasts)
        positive=sum(x['worst_layer_delta'] is not None and x['worst_layer_delta']>0 for x in selected_contrasts)
        judgments.append(dict(architecture=a,competitor=selected['competitor'],cells=selected['cells'],
            complete=complete,positive_cell_draws=positive,expected_cell_draws=10,
            confirmed=complete and positive==10,
            minimum_margin=min(x['worst_layer_delta'] for x in selected_contrasts) if complete else None,
            maximum_margin=max(x['worst_layer_delta'] for x in selected_contrasts) if complete else None))
    return means,contrasts,layer_differences,judgments


def collect(roots,configs):
    rows=[];states=[];attempts=[];seen=set();case_keys=set();shared_noise={}
    for root in roots:
        for path in sorted(Path(root).rglob('manifest.json')):
            run=path.parent.resolve()
            if run in seen:continue
            seen.add(run);m=json.loads(path.read_text())
            if m.get('smoke') or m.get('git',{}).get('runner_sha256')!=RUNNER:continue
            status=json.loads((run/'status.json').read_text());attempts.append(dict(run=str(run),state=status['state'],error=status.get('error')))
            if status['state']!='complete':continue
            assert not validate_run(run),run
            case=m['inputs'];a=case['architecture'];cfg=configs[a];depth=int(a[-1])
            canonical=next(c for c in cfg['cases'] if c['name']==m['arm_id'])
            assert all(case.get(k)==v for k,v in canonical.items())
            assert (a,case['name']) not in case_keys,'Duplicate completed attempt';case_keys.add((a,case['name']))
            expected=copy.deepcopy(cfg);expected['dataset']=json.loads((ROOT/cfg['cohort_config']).read_text())['dataset']
            actual=copy.deepcopy(m['configuration']['resolved']);actual['dataset']['root']=expected['dataset']['root']='<data-location>'
            assert expected==actual and m['configuration']['sha256']==CONFIGS[a]
            assert m['cohort_sha256']==COHORT and m['checkpoint_role']=='reconstructed_initialization' and m['checkpoint_epoch']==m['model_seed']==0
            assert m['runtime']['device']=='cuda'
            terminal=json.loads((run/'result.json').read_text())['terminal_metrics']
            for flag in ('official_test_read','optimizer_steps_applied','accuracy_evaluation'):assert m[flag] is False and terminal[flag] is False
            for flag in ('source_bytes_unchanged','all_parameter_tensors_unchanged','all_bias_tensors_exact_zero'):assert terminal[flag] is True
            assert sha(run/'checkpoints/initialization.pt')==case['initializer_checkpoint_sha256']
            assert sha(ROOT/case['source_config'])==case['source_config_sha256']
            for proof in (case['calibration_proof'],*case['screen_references'].values()):assert sha(ROOT/proof['path'])==proof['sha256']
            screen_manifest=json.loads((ROOT/case['screen_references']['manifest']['path']).read_text())
            for field in ('host','torch_version','cuda_version'):
                assert m['runtime'][field]==screen_manifest['runtime'][field],('Changed screen runtime',field)
            guards=json.loads((run/'read_only_guards.json').read_text());screen_guards=json.loads((ROOT/case['screen_references']['guards']['path']).read_text());assert len(guards)==len(screen_guards)==36
            for guard,reference in zip(guards,screen_guards):
                for key in ('common_post_T_state_sha256','positive_state_sha256','negative_state_sha256','zero_state_sha256','frozen_current_force_sha256','parameter_state_sha256_after_cast','reconstructed_initialization_tensor_sha256'):assert guard[key]==reference[key]
                assert guard['precision']=='float64'
                for key in ('iteration_contract_passed','parameter_tensors_unchanged_after_cast','frozen_current_force_matches_post_T_cost_gradient','frozen_current_force_unchanged_across_phases'):assert guard[key] is True
            rr=read_csv(run/'layer_metrics.csv');expected_rows=36*(1+5*(len(cfg['noise_sigmas'])-1))*(depth+1)
            assert len(rr)==terminal['layer_comparisons']==expected_rows and terminal['replay_batches']==36
            screen_rows={(int(x['batch_index']),x['parameter_name'].strip()):x for x in read_csv(ROOT/case['screen_references']['layer_metrics']['path']) if float(x['readout_sigma'])==0}
            for row in rr:
                assert (row['architecture'],row['scheme'],float(row['relative_target']))==(a,case['scheme'],case['relative_target'])
                assert int(row['T'])==int(row['K'])==case['T']==case['K'] and int(row['checkpoint_epoch'])==0
                assert float(row['injected_beta'])==case['injected_beta']
                scale={'baseline':1.,'ours':4.,'legacy':16.}[case['scheme']]**depth
                assert math.isclose(float(row['base_beta'])*scale,case['injected_beta'],rel_tol=1e-13)
                ref=screen_rows[(int(row['batch_index']),row['parameter_name'].strip())]
                for field in ('bptt_gradient_sha256','batch_payload_sha256','batch_source_indices_sha256'):assert row[field]==ref[field]
                assert math.isclose(float(row['clean_eqprop_rms']),float(ref['clean_eqprop_rms']),rel_tol=1e-12,abs_tol=1e-15)
            noise=read_csv(run/'endpoint_read_noise.csv');keys=set()
            for draw in noise:
                b=int(draw['batch_index']);d=int(draw['noise_draw_index']);sigma=float(draw['endpoint_read_noise_std']);layer=int(draw['state_layer_index']);phase=draw['phase']
                assert b in range(36) and d in range(5) and layer in range(depth+1) and phase in ('negative','positive')
                seed=SEEDS[d];assert int(draw['noise_seed'])==seed and sigma in cfg['noise_sigmas'][1:]
                assert int(draw['seed'])==seed+1000*b+100*{'negative':1,'positive':2}[phase]+layer
                key=(sigma,d,b,phase,layer);assert key not in keys;keys.add(key)
                paired=(a,*key);assert shared_noise.setdefault(paired,draw['standard_normal_sha256'])==draw['standard_normal_sha256']
            assert keys=={(n,d,b,p,l) for n in cfg['noise_sigmas'][1:] for d in range(5) for b in range(36) for p in ('negative','positive') for l in range(depth+1)}
            rows.extend(rr);states.extend(read_csv(run/'state_signal.csv'))
    return rows,states,attempts,case_keys


def plot(output,contrasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,a in zip(axes,CONFIGS):
        selected=[x for x in contrasts if x['architecture']==a and x['scheme']=='ours']
        cells=sorted({(x['relative_target'],x['readout_sigma']) for x in selected})
        for r,n in cells:
            points=sorted([x for x in selected if (x['relative_target'],x['readout_sigma'])==(r,n)],key=lambda x:x['noise_draw_index'])
            ax.plot(range(1,6),[x['worst_layer_delta'] for x in points],marker='o',label=f'D/F={r:g}, σ={n:g}')
        ax.axhline(0,color='black',linewidth=.8);ax.set_title(a);ax.set_xticks(range(1,6));ax.set_xlabel('Fresh noise draw');ax.set_ylabel('W(ours) − W(legacy)');ax.legend(fontsize=8)
    fig.suptitle('Selected pairs: five independent-noise confirmations (per-panel y scales)')
    for suffix in ('png','pdf'):fig.savefig(output/f'confirmation_margins.{suffix}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--production-root',type=Path,action='append',required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    configs={}
    for a,digest in CONFIGS.items():
        path=ROOT/f'configs/conv/confirmation_initial_displacement_{a}_20260925.json';assert sha(path)==digest;configs[a]=json.loads(path.read_text())
    rows,states,attempts,cases=collect(args.production_root,configs)
    means,contrasts,differences,judgments=summarize(rows,configs);physical=state_summary(states)
    outputs=[x for x in physical if x['is_output']]
    expected={(a,c['name']) for a,cfg in configs.items() for c in cfg['cases']}
    complete=cases==expected and all(x['complete'] for x in means) and len(outputs)==12 and all(x['output_match_passed'] for x in outputs)
    for judgment in judgments:
        a=judgment['architecture'];wanted={(a,c['name']) for c in configs[a]['cases']}
        state_outputs=[x for x in outputs if x['architecture']==a]
        valid=(wanted <= cases and len(state_outputs)==len(wanted) and all(x['output_match_passed'] for x in state_outputs)
               and all(x['complete'] for x in means if x['architecture']==a))
        judgment['scientific_coverage_complete']=valid
        judgment['confirmed']=judgment['confirmed'] and valid
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    for name,data in [('per_draw_layer_summary',means),('per_draw_worst_layer_contrasts',contrasts),('per_draw_layer_differences',differences),('physical_state_controls',physical)]:write_csv(out/(name+'.csv'),data)
    result=dict(complete=complete,completed_contexts=len(cases),expected_contexts=12,layer_rows=len(rows),expected_layer_rows=11016,judgments=judgments,attempts=attempts)
    (out/'confirmation.json').write_text(json.dumps(result,indent=2)+'\n');plot(out,contrasts)
    print(json.dumps({k:v for k,v in result.items() if k!='attempts'}))
if __name__=='__main__':main()
