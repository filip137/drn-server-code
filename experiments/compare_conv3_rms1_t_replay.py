"""Pair T8/T16 gradient quality at fixed initial-output-displacement betas."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import atomic_write_json, sha256_file, validate_run
from experiments.replay_conv3_trained_beta_noise import beta_points, cell_identity, replay_contexts
from experiments.summarize_conv3_trained_beta_noise import read_csv, write_csv, quantiles

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ['ConvWeight_0','ConvWeight_1','ConvWeight_2','DenseWeight_0']
LABELS = ['Conv 1','Conv 2','Conv 3','Readout']
METRICS = ['cosine','relative_l2_difference_over_bptt','eqprop_over_bptt_norm_ratio',
           'bptt_rms','clean_eqprop_rms','eqprop_rms','noise_over_clean_norm']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg8=json.loads(args.config.read_text())
    cfg16=json.loads((ROOT/cfg8['comparison_T16_config']).read_text())
    assert cfg8['T_values']==[8] and cfg16['T_values']==[16]
    ignored={'study_id','output_root','T_values','analysis_objective','comparison_T16_config'}
    assert {k:v for k,v in cfg8.items() if k not in ignored}=={k:v for k,v in cfg16.items() if k not in ignored}
    study=ROOT/cfg8['output_root']; out=study/'analysis'; out.mkdir(exist_ok=True)
    data,phases,residuals,sources={},{},[],[]
    manifests={}
    for T,cfg in [(8,cfg8),(16,cfg16)]:
        for case in replay_contexts(cfg):
            beta,=beta_points(cfg,case=case)
            name=cell_identity(case,beta,absolute=True)
            run=ROOT/cfg['output_root']/'collected/runs'/name
            assert not validate_run(run),run
            manifest=json.loads((run/'manifest.json').read_text())
            manifests[(T,case['name'])]=manifest
            assert manifest['replay_T']==T and manifest['replay_K']==8
            rows=read_csv(run/'layer_metrics.csv')
            assert len(rows)==720
            for r in rows:
                key=(case['name'],r['parameter_name'],int(r['batch_index']),int(r['noise_draw_index']))
                assert (T,key) not in data
                data[(T,key)]=r
            for r in read_csv(run/'state_signals.csv'):
                key=(case['name'],int(r['layer_index']),int(r['batch_index']))
                assert (T,key) not in phases
                phases[(T,key)]=r
            rr=read_csv(run/'equilibrium_residuals.csv')
            assert len(rr)==576 and all(r['gate_passed']=='True' for r in rr)
            for phase in sorted({r['phase'] for r in rr}):
                rr_phase=[r for r in rr if r['phase']==phase]
                residuals.append(dict(T=T,checkpoint=case['name'],phase=phase,rows=len(rr_phase),
                    max_selected_residual=max(float(r['selected_maximum']) for r in rr_phase),failures=0))
            sources.append(dict(T=T,checkpoint=case['name'],run_dir=str(run.relative_to(ROOT)),
                result_sha256=sha256_file(run/'result.json'),runner_sha256=manifest['git']['runner_sha256']))
    for case in cfg8['cases']:
        a,b=manifests[(8,case['name'])],manifests[(16,case['name'])]
        for key in ['cohort_sha256','checkpoint_role','checkpoint_epoch','injected_beta','base_beta']:
            assert a[key]==b[key],(case['name'],key)
        assert a['inputs']==dict(b['inputs'],replay_T=8,context_name=b['inputs']['context_name'].replace('_T16K8','_T8K8'))
        assert a['git']['runner_sha256']==b['git']['runner_sha256']
        for key in ['host','target','device','torch_version','cuda_version','python']:
            assert a['runtime'][key]==b['runtime'][key],(case['name'],key)
    keys={key for T,key in data if T==8}
    assert keys=={key for T,key in data if T==16} and len(keys)==2880
    paired,groups=[],defaultdict(list)
    for key in sorted(keys):
        a,b=data[(8,key)],data[(16,key)]
        for field in ['batch_payload_sha256','batch_source_indices_sha256','checkpoint_sha256',
                      'parameter_name','injected_beta','endpoint_read_noise_std','endpoint_read_noise_seed']:
            assert a[field]==b[field],(key,field)
        checkpoint,parameter,batch,draw=key
        r=dict(checkpoint=checkpoint,scheme=a['scheme'],epoch=int(a['checkpoint_epoch']),
               parameter=parameter,batch=batch,draw=draw,sigma=float(a['endpoint_read_noise_std']))
        for metric in METRICS:
            x,y=float(a[metric]),float(b[metric])
            r[metric+'_T8']=x;r[metric+'_T16']=y;r[metric+'_T8_minus_T16']=x-y
            r[metric+'_relative_change']=(x-y)/abs(y) if y else None
        paired.append(r);groups[(checkpoint,r['sigma'],parameter)].append(r)
    effects,paired_batches=[],[]
    for (checkpoint,sigma,parameter),group in sorted(groups.items()):
        batches=[]
        for batch in range(36):
            sample=[r for r in group if r['batch']==batch]
            assert len(sample)==(4 if sigma else 1)
            r=dict(checkpoint=checkpoint,sigma=sigma,parameter=parameter,batch=batch)
            for metric in METRICS:
                r[metric+'_T8_minus_T16']=float(np.mean([x[metric+'_T8_minus_T16'] for x in sample]))
            batches.append(r)
        paired_batches.extend(batches)
        r=dict(checkpoint=checkpoint,sigma=sigma,parameter=parameter,comparisons=len(group))
        for metric in METRICS:
            r[metric+'_T8_median']=float(np.median([x[metric+'_T8'] for x in group]))
            r[metric+'_T16_median']=float(np.median([x[metric+'_T16'] for x in group]))
            r[metric+'_median_difference']=r[metric+'_T8_median']-r[metric+'_T16_median']
            r[metric+'_maximum_absolute_paired_draw_change']=max(abs(x[metric+'_T8_minus_T16']) for x in group)
            r[metric+'_maximum_absolute_relative_change']=max((abs(x[metric+'_relative_change']) for x in group if x[metric+'_relative_change'] is not None),default=0.)
            quantiles(r,metric+'_paired_batch_delta',[x[metric+'_T8_minus_T16'] for x in batches])
        r['fraction_paired_batches_cosine_change_above_01']=float(np.mean([abs(x['cosine_T8_minus_T16'])>.01 for x in batches]))
        effects.append(r)
    phase_groups=defaultdict(list)
    for T,key in sorted(phases):
        if T!=8:continue
        a,b=phases[(8,key)],phases[(16,key)]
        r=dict(checkpoint=key[0],layer=key[1],batch=key[2],count=int(a['element_count']))
        assert a['element_count']==b['element_count']
        for metric in ['free_rms','positive_free_rms','negative_free_rms','centered_half_difference_rms','zero_free_rms']:
            r[metric+'_T8']=float(a[metric]);r[metric+'_T16']=float(b[metric])
        phase_groups[key[:2]].append(r)
    phase_effects=[]
    for (checkpoint,layer),group in sorted(phase_groups.items()):
        assert len(group)==36
        r=dict(checkpoint=checkpoint,layer=layer)
        for metric in ['free_rms','positive_free_rms','negative_free_rms','centered_half_difference_rms','zero_free_rms']:
            for T in [8,16]:
                k=f'{metric}_T{T}'
                r[k+'_pooled']=float(np.sqrt(sum(x[k]**2*x['count'] for x in group)/sum(x['count'] for x in group)))
            denominator=r[metric+'_T16_pooled']
            r[metric+'_pooled_relative_change']=(r[metric+'_T8_pooled']/denominator-1
                                                  if denominator else None)
        phase_effects.append(r)
    for name,rows in [('T_paired_draws',paired),('T_paired_batches',paired_batches),('T_effects',effects),
                      ('T_phase_effects',phase_effects),('T_residuals',residuals)]:
        write_csv(out/(name+'.csv'),rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,epoch in zip(axes,[0,30]):
        for scheme,color in [('legacy','#276FBF'),('ours','#D46A21')]:
            for T,marker,style in [(8,'o','-'),(16,'x','--')]:
                cells=[next(r for r in effects if (r['checkpoint'],r['sigma'],r['parameter'])==(f'{scheme}_epoch{epoch}',.0005,p)) for p in PARAMETERS]
                ax.plot(range(4),[r[f'cosine_T{T}_median'] for r in cells],marker=marker,linestyle=style,color=color,label=f'{scheme} T={T}')
        ax.set_xticks(range(4),LABELS);ax.set_ylim(-.05,1.05);ax.grid(alpha=.2)
        ax.set_title('Initialization' if epoch==0 else 'P99-trained epoch 30');ax.set_ylabel('Noisy EqProp–BPTT cosine');ax.legend(frameon=False)
    fig.suptitle('Fixed β: ours 1.385, legacy 0.02173 · K8 · σ=5×10⁻⁴')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'T_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    summary=dict(complete=True,paired_layer_measurements=len(paired),new_cases=4,reused_T16_cases=4,
        maximum_noisy_median_cosine_change=max(abs(r['cosine_median_difference']) for r in effects if r['sigma']),
        maximum_clean_median_cosine_change=max(abs(r['cosine_median_difference']) for r in effects if not r['sigma']),
        maximum_absolute_paired_draw_cosine_change=max(abs(r['cosine_T8_minus_T16']) for r in paired),
        maximum_absolute_paired_batch_cosine_change=max(abs(r['cosine_T8_minus_T16']) for r in paired_batches),
        matched_checkpoints_cohort_noise_runtime_and_runner=True,sources=sources,
        official_test_read=False,optimizer_steps_applied=False)
    atomic_write_json(out/'T_validation.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='sources'},indent=2))
    for r in effects:
        if r['sigma']:print(r['checkpoint'],r['parameter'],r['cosine_T8_median'],r['cosine_T16_median'],r['cosine_median_difference'])


if __name__=='__main__':main()
