"""Pool phase displacements by element count and compare all nine Conv schemes."""
from __future__ import annotations

from collections import defaultdict
import csv
import math
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.run_phase_displacement import ROOT, OUT, checked, read, sha, write


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def pooled(rows):
    n=sum(int(r['element_count']) for r in rows)
    dsq=sum(float(r['delta_squared_sum']) for r in rows)
    rsq=sum(float(r['reference_squared_sum']) for r in rows)
    csq=sum(float(r['current_squared_sum']) for r in rows)
    dm=sum(float(r['delta_sum']) for r in rows)/n
    rm=sum(float(r['reference_sum']) for r in rows)/n
    cm=sum(float(r['current_sum']) for r in rows)/n
    constrained=sum(int(r['constrained_element_count']) for r in rows)
    transitions=sum(int(r['active_set_transition_count']) for r in rows)
    rms=[float(r['delta_rms']) for r in rows]
    relative=[float(r['relative_displacement']) for r in rows]
    return dict(batch_count=len(rows),element_count=n,delta_mean=dm,delta_rms=math.sqrt(dsq/n),
        delta_std=math.sqrt(max(dsq/n-dm*dm,0)),reference_mean=rm,reference_rms=math.sqrt(rsq/n),
        reference_std=math.sqrt(max(rsq/n-rm*rm,0)),current_mean=cm,current_rms=math.sqrt(csq/n),
        current_std=math.sqrt(max(csq/n-cm*cm,0)),relative_displacement=math.sqrt(dsq/rsq) if rsq>0 else None,
        reference_norm_zero=rsq==0,max_abs_delta=max(float(r['max_abs_delta']) for r in rows),
        reference_min=min(float(r['reference_min']) for r in rows),reference_max=max(float(r['reference_max']) for r in rows),
        current_min=min(float(r['current_min']) for r in rows),current_max=max(float(r['current_max']) for r in rows),
        active_set_transition_count=transitions,constrained_element_count=constrained,
        active_set_transition_fraction=transitions/constrained if constrained else None,
        batch_p05_delta_rms=float(np.quantile(rms,.05)),batch_median_delta_rms=float(np.median(rms)),
        batch_p95_delta_rms=float(np.quantile(rms,.95)),batch_p05_relative=float(np.quantile(relative,.05)),
        batch_median_relative=float(np.median(relative)),batch_p95_relative=float(np.quantile(relative,.95)))


COLORS={'baseline':'#2676b9','ours':'#d77c18','legacy':'#399363'}
ROLES=('reconstructed_initialization','best_validation')


def phase_figure(rows,reference,metric,stem,ylabel):
    fig,axes=plt.subplots(2,3,figsize=(12,6.8))
    for column,architecture in enumerate(('conv1','conv2','conv3')):
        depth=int(architecture[-1])
        for i,role in enumerate(ROLES):
            ax=axes[i,column]
            for scheme,color in COLORS.items():
                for phase,style,marker in [('positive','-','o'),('negative','--','v')]:
                    rr=sorted([r for r in rows if r['architecture']==architecture and r['scheme']==scheme
                        and r['checkpoint_role']==role and r['reference_kind']==reference and r['phase']==phase],key=lambda r:r['layer_index'])
                    ax.plot([r['layer_index'] for r in rr],[r[metric] for r in rr],style,marker=marker,color=color,linewidth=1.4,markersize=4)
            ax.set(yscale='log',xticks=list(range(1,depth+2)),xticklabels=[f'H{k}' for k in range(1,depth+1)]+['Out'])
            ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            if column==0:ax.set_ylabel(('Initialization\n' if i==0 else 'Best BPTT checkpoint\n')+ylabel)
            if i==0:
                values={r['scheme']:r['injected_beta'] for r in rows if r['architecture']==architecture}
                ax.set_title(f'{architecture.capitalize()} | beta '+ '/'.join(f'{values[s]:g}' for s in COLORS),fontsize=10)
    handles=[Line2D([0],[0],color=c,label=s) for s,c in COLORS.items()]
    handles += [Line2D([0],[0],color='black',linestyle='-',marker='o',label='positive nudge'),Line2D([0],[0],color='black',linestyle='--',marker='v',label='negative nudge')]
    fig.legend(handles=handles,loc='upper center',ncol=5,bbox_to_anchor=(.5,.925),fontsize=9)
    label='post-T free state' if reference=='post_T_free' else 'zero-nudge endpoint after K steps'
    fig.suptitle(f'Per-layer nudged displacement relative to {label}\nSeed 0; 576 matched validation examples; column betas: baseline / ours / legacy',fontsize=11)
    fig.text(.5,.012,'Conv3 baseline beta 10: unconfirmed; T8 residual caveat. Conv3 ours: gradient exception. Different betas/contracts.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.94))
    for suffix in ('png','pdf'):fig.savefig(OUT/'analysis'/f'{stem}.{suffix}',dpi=180)
    plt.close(fig)


def main():
    execution=read(OUT/'execution.json')
    if execution['state']!='complete':raise ValueError('Wait for all nine cases.')
    cases=execution['runs']
    assert {(c['architecture'],c['scheme']) for c in cases}=={(a,s) for a in ('conv1','conv2','conv3') for s in COLORS}
    assert len(cases)==9
    checked(execution['smoke'],smoke=True)
    assert read(OUT/'smoke_regression.json')['shared_measurements_identical']
    analysis=OUT/'analysis';analysis.mkdir(exist_ok=True)
    all_rows=[];sources=[];expected_payloads=None
    for case in cases:
        assert checked(case)['result_sha256']==case['result_sha256']
        directory=(ROOT/case['result']).parent;config=read(ROOT/case['path']);source=config['cases'][0]
        source_dir=ROOT/source['run_dir'];metadata=read(source_dir/'metrics.json')
        raw=list(csv.DictReader((directory/'state_displacement.csv').open()))
        rows=[r for r in raw if r['state_layer_name']!='__all__']
        assert len(rows)==72*(int(case['architecture'][-1])+1)*5
        payloads={(r['checkpoint_role'],r['batch_index']):(r['batch_source_indices_sha256'],r['batch_payload_sha256']) for r in rows}
        if expected_payloads is None:expected_payloads=payloads
        else:assert payloads==expected_payloads,'Unmatched cohort across schemes/architectures.'
        for r in rows:
            assert r['outcome']=='ok'
            r.update(injected_beta=case['beta'],base_beta=case['base_beta'],model_seed=0,
                layer_index=int(re.search(r'(\d+)$',r['state_layer_name']).group(1)),
                source_best_epoch=metadata['best_epoch'],source_checkpoint_role=r['checkpoint_role'],
                source_checkpoint_epoch=0 if r['checkpoint_role']=='reconstructed_initialization' else metadata['best_epoch'])
            all_rows.append(r)
        sources.append(dict(architecture=case['architecture'],scheme=case['scheme'],injected_beta=case['beta'],base_beta=case['base_beta'],T=case['T'],K=case['K'],
            source_run=str(source_dir.relative_to(ROOT)),best_epoch=metadata['best_epoch'],initializer=source['initializer_checkpoint_path'],
            initializer_sha256=source['initializer_checkpoint_sha256'],best_checkpoint=str((source_dir/'best_model.pt').relative_to(ROOT)),
            best_checkpoint_sha256=source['source_file_sha256']['best_model.pt'],result=case['result'],result_sha256=case['result_sha256'],
            gradient_pass=case['gradient_fidelity_all_passed'],equilibrium_pass=case['equilibrium_residual_all_passed']))
    assert len(all_rows)==9720
    keys=('architecture','scheme','checkpoint_role','phase','reference_kind','state_layer_name')
    groups=defaultdict(list)
    for row in all_rows:groups[tuple(row[k] for k in keys)].append(row)
    summaries=[]
    for group,rows in groups.items():
        assert len(rows)==36 and len({r['batch_index'] for r in rows})==36
        r=rows[0]
        summary=dict(zip(keys,group),layer_index=r['layer_index'],layer_type=r['state_layer_type'],state_shape=r['state_shape'],
            T=int(r['T']),K=int(r['K']),injected_beta=r['injected_beta'],base_beta=r['base_beta'],model_seed=0,
            source_checkpoint_epoch=r['source_checkpoint_epoch'],**pooled(rows))
        summary['half_span_rms']=summary['delta_rms']/2 if r['phase']=='positive_minus_negative' else None
        summaries.append(summary)
    assert len(summaries)==270
    # The two phase views must give identical statistics for the same reference state.
    lookup={(r['architecture'],r['scheme'],r['checkpoint_role'],r['reference_kind'],r['layer_index'],r['phase']):r for r in summaries}
    for r in summaries:
        if r['phase']=='positive':
            other=lookup[(r['architecture'],r['scheme'],r['checkpoint_role'],r['reference_kind'],r['layer_index'],'negative')]
            assert r['reference_rms']==other['reference_rms'] and r['reference_mean']==other['reference_mean']
    csv_write(analysis/'layer_batch_displacement.csv',all_rows)
    csv_write(analysis/'layer_displacement_summary.csv',summaries)
    csv_write(analysis/'sources.csv',sources)
    phase_figure(summaries,'post_T_free','delta_rms','phase_displacement_absolute','RMS displacement [voltage units]')
    phase_figure(summaries,'post_T_free','relative_displacement','phase_displacement_relative','RMS displacement / free RMS')
    phase_figure(summaries,'matched_zero_K','delta_rms','phase_displacement_matched_zero','RMS displacement [voltage units]')
    fig,axes=plt.subplots(4,3,figsize=(12,10))
    voltage=[r for r in summaries if r['phase']=='positive' and r['reference_kind']=='post_T_free']
    for col,architecture in enumerate(('conv1','conv2','conv3')):
        depth=int(architecture[-1])
        for role_index,role in enumerate(ROLES):
            for metric_index,metric in enumerate(('reference_mean','reference_rms')):
                ax=axes[2*role_index+metric_index,col]
                for scheme,color in COLORS.items():
                    rr=sorted([r for r in voltage if r['architecture']==architecture and r['scheme']==scheme and r['checkpoint_role']==role],key=lambda r:r['layer_index'])
                    ax.plot([r['layer_index'] for r in rr],[r[metric] for r in rr],'o-',color=color,label=scheme)
                ax.set(xticks=list(range(1,depth+2)),xticklabels=[f'H{k}' for k in range(1,depth+1)]+['Out'])
                if metric_index==1:ax.set_yscale('log')
                ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
                if col==0:ax.set_ylabel(('Initialization' if role_index==0 else 'Best BPTT checkpoint')+'\n'+('Signed mean voltage' if metric_index==0 else 'RMS voltage'))
                if role_index==metric_index==0:ax.set_title(architecture.capitalize())
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Free-state voltage scale: signed mean and RMS reported separately\nSeed 0; element-weighted statistics across 576 validation examples',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95))
    for suffix in ('png','pdf'):fig.savefig(analysis/f'phase_free_voltage.{suffix}',dpi=180)
    plt.close(fig)
    summary=dict(study_id=OUT.name,execution=execution,source_cases=sources,pooled_rows=summaries,
        formal_cases=9,checkpoint_batch_replays=648,gradient_comparisons=1944,individual_layer_rows=9720,pooled_layer_context_rows=270,
        cohort_payloads_match=True,smoke_measurements_match=True,analysis_source_sha256=sha(__file__),official_test_read=False,training_started=False)
    write(analysis/'summary.json',summary)
    for name in ('layer_displacement_summary.csv','sources.csv','summary.json'):
        shutil.copy2(analysis/name,ROOT/'paper_ready_results'/f'phase_displacement_20260916_{name}')
    for name in ('phase_displacement_absolute','phase_displacement_relative','phase_displacement_matched_zero','phase_free_voltage'):
        for suffix in ('png','pdf'):shutil.copy2(analysis/f'{name}.{suffix}',ROOT/'paper_ready_results'/f'{name}_20260916.{suffix}')
    print('Validated all nine cases; pooled270 layer/context rows from9,720 batch rows.')


if __name__=='__main__':main()
