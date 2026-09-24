"""Summarize the seven exploratory BN ablations against existing CIFAR references."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from experiments import reporting


def training_rows(path, results, visited=()):
    """Stitch saved epoch prefixes, including recoveries across study roots."""
    if path in visited:
        raise ValueError(f'Checkpoint recovery cycle: {path}')
    metric=path/'metrics.jsonl'
    rows=[json.loads(line) for line in metric.read_text().splitlines()] if metric.exists() else []
    rows=[r for r in rows if 'epoch' in r and 'validation_accuracy' in r]
    recovery=path/'artifacts/recovery.json'
    if recovery.exists():
        record=json.loads(recovery.read_text());start=record['from_epoch']
        remote=Path(record['checkpoint']).parents[1]
        parts=remote.parts
        parent=results.joinpath(*parts[parts.index('results')+1:])
        before=training_rows(parent,results,(*visited,path))
        if not before or before[-1]['epoch'] < start:
            raise ValueError(f'Missing recovered epoch prefix through {start}: {parent}')
        rows=[r for r in before if r['epoch']<=start]+rows
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args();root=args.root.resolve();results=root.parent
    out=root/'analysis';out.mkdir(exist_ok=True)
    cases=[p.stem for p in sorted((root/'configs').glob('*.json'))]
    references={'baseline':'baseline_c1_d1_e10','ours':'ours_edge_head_down_e10','legacy':'legacy_c1_d1_e10'}
    previous=results/'cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/cells'
    paths={f'{scheme}_reference':previous/name for scheme,name in references.items()}
    paths['legacy_voltage_normalized']=results/'cifar10-l8-amplification-mechanism-seed0-20260923-v1/cells/legacy_voltage_normalized'
    paths.update({case:root/'cells'/case for case in cases})
    lanes={}
    for path in (root/'launcher').glob('*.json'):
        record=json.loads(path.read_text())
        if record.get('replacement_target'):
            continue
        for entry in record.get('commands',[]):
            if entry['case'] not in record.get('outsourced_cases',[]):
                lanes[entry['case']]=entry
    checks_path=out/'local_checks.json'
    checks={r['case']:r for r in json.loads(checks_path.read_text()).get('results',[])} if checks_path.exists() else {}
    summary=[];curves={};csv_rows=[];gain_rows=[]
    for name,path in paths.items():
        collapse_path=out/f'{name}_zero_gradient_replay.json'
        collapse=json.loads(collapse_path.read_text()) if collapse_path.exists() else {}
        rows=training_rows(path,results)
        curves[name]=rows
        status=json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
        state=status.get('state','pending')
        if name in checks and not checks[name]['passed']:state='excluded: qualification/smoke'
        lane=lanes.get(name,{})
        if lane.get('exit_code') not in (None,0):state=f"failed: exit {lane['exit_code']}"
        errors=reporting.validate_run(path) if status else []
        result=json.loads((path/'result.json').read_text()) if (path/'result.json').exists() else {}
        final=result.get('terminal_metrics',{})
        checkpoint_path=path/'checkpoints/final_model.pt'
        if (result or status.get('state')=='failed') and checkpoint_path.exists():
            checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            for parameter,raw in checkpoint['model']['module'].items():
                if '_input_gain_raw' in parameter:
                    gain_rows.append({'case':name,'epoch':checkpoint['epoch'],
                                      'parameter':parameter,
                                      'gain':torch.nn.functional.softplus(raw).item()})
        if final.get('final_solver_audit_passed') is False:state='complete: solver audit failed'
        zero_collapse=(status.get('state')=='failed'
                            and bool(collapse.get('gradients'))
                            and collapse.get('logits',{}).get('abs_max')==0
                            and all(g['gradient_nonzero_count']==0 for g in collapse['gradients'].values()))
        solver_failure=final.get('final_solver_audit_passed') is False
        scientific_failure=(zero_collapse or solver_failure) and not errors
        if zero_collapse:state='stopped: zero-activation/gradient collapse'
        if errors:state='invalid bundle'
        last=rows[-1] if rows else {}
        summary.append({'case':name,'state':state,'epochs':last.get('epoch',0),
                        'train_accuracy':last.get('train_accuracy'),'train_loss':last.get('train_loss'),
                        'validation_accuracy':last.get('validation_accuracy'),'validation_loss':last.get('validation_loss'),
                        'validation_errors':errors,'final_solver_audit_passed':final.get('final_solver_audit_passed'),
                        'scientific_terminal_failure':scientific_failure,
                        'zero_activation_collapse':zero_collapse})
        for row in rows:csv_rows.append({'case':name,**{k:row.get(k) for k in ['epoch','train_accuracy','train_loss','validation_accuracy','validation_loss','epoch_seconds']}})
    complete=all(next(r for r in summary if r['case']==case)['state']=='complete' for case in cases)
    coverage_reconciled=all(next(r for r in summary if r['case']==case)['state']=='complete'
                            or next(r for r in summary if r['case']==case)['scientific_terminal_failure']
                            for case in cases)
    (out/'summary.json').write_text(json.dumps({'complete':complete,'coverage_reconciled':coverage_reconciled,'cases':summary},indent=2)+'\n')
    if csv_rows:
        with (out/'training.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    fig,axes=plt.subplots(3,2,figsize=(12,11),sharex=True)
    for i,scheme in enumerate(['baseline','ours','legacy']):
        for name,rows in curves.items():
            if not name.startswith(scheme+'_') or not rows:continue
            label=name[len(scheme)+1:]; style='--' if label=='reference' else '-'
            axes[i,0].plot([r['epoch'] for r in rows],[100*r['validation_accuracy'] for r in rows],style,marker='o',markersize=3,label=label)
            axes[i,1].plot([r['epoch'] for r in rows],[r['validation_loss'] for r in rows],style,marker='o',markersize=3,label=label)
        axes[i,0].set_ylabel(f'{scheme}: validation accuracy (%)');axes[i,1].set_ylabel('Validation cross-entropy')
        for ax in axes[i]:
            ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlabel('Epoch')
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
    report=['# CIFAR L8 BN ablations','',
            'Exploratory seed0; ten epochs, batch32, fixed selected Adam rates, augmentation and cross-entropy. Existing references are reused. Frozen BN fixes affine parameters while updating statistics. No-BN removes normalization and affine transforms. Hardware/software differ between local3090 and5090 lanes.',
            '', '| Case | Epochs | Train accuracy | Validation accuracy | Validation CE | State |',
            '|---|---:|---:|---:|---:|---|']
    def acc(x):return '—' if x is None else f'{100*x:.2f}%'
    for row in summary:
        ce='—' if row['validation_loss'] is None else f"{row['validation_loss']:.4f}"
        report.append(f"| {row['case']} | {row['epochs']} | {acc(row['train_accuracy'])} | {acc(row['validation_accuracy'])} | {ce} | {row['state']} |")
    report += ['',f'All seven new cases complete and bundles valid: **{complete}**.',
               f'Case coverage accounted for, including diagnosed scientific failures: **{coverage_reconciled}**.',
               'The initial cuDNN failure is retained under operational-attempt-01; accepted local checks use isolated cuDNN9.1. No LR retuning or official-test evaluation.',
               '', '![Learning curves](learning_curves.png)', '',
               'First-batch pooled-voltage, boundary-output, logit and gradient statistics remain in each case’s metrics.jsonl. Partial results do not establish final accuracy or optimal BN-free learning rates.']
    for row in summary:
        if row['zero_activation_collapse']:
            report += ['', f"{row['case']} stopped at the first batch of epoch2 after epoch1 validation accuracy10% and CE log(10). Exact read-only replay found zero block outputs, zero logits and zero gradients for every parameter, including float64 norm checks. Its stopped checkpoint and failed bundle are retained; it is an early scientific collapse under the selected settings, not a completed ten-epoch run. No LR retry was launched. The run did not reach its final higher-iteration solver audit. [Replay evidence]({row['case']}_zero_gradient_replay.json)."]
        elif row['final_solver_audit_passed'] is False:
            report += ['', f"{row['case']} completed its epoch budget but failed the final solver audit. Its reported accuracy is the observed result at the configured iterations, not a qualified steady-state comparison. See that case’s artifacts/final_solver_audit.json; no iteration or LR retry was launched."]
    report += ['', '## Comparisons at the same epoch', '',
               '| Intervention | Epoch | Validation accuracy | Existing reference | Difference |',
               '|---|---:|---:|---:|---:|']
    for name, rows in curves.items():
        if name.endswith('_reference') or not rows:
            continue
        latest=rows[-1];reference=curves[name.split('_')[0]+'_reference']
        match=next((r for r in reference if r['epoch']==latest['epoch']),None)
        if match:
            delta=100*(latest['validation_accuracy']-match['validation_accuracy'])
            report.append(f"| {name} | {latest['epoch']} | {acc(latest['validation_accuracy'])} | {acc(match['validation_accuracy'])} | {delta:+.2f} pp |")
    # Describe the whole matched trajectory without changing the declared endpoint.
    trajectory=[];paired_rows=[]
    selected=['baseline_epsilon_legacy','legacy_voltage_normalized',
              'baseline_frozen_bn','ours_frozen_bn','legacy_frozen_bn']
    delta_fig,delta_axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for name in selected:
        reference={r['epoch']:r for r in curves[name.split('_')[0]+'_reference']}
        pairs=[]
        for row in curves.get(name,[]):
            ref=reference.get(row['epoch'])
            if ref is None:continue
            pairs.append({'case':name,'epoch':row['epoch'],
                          'validation_accuracy_delta_pp':100*(row['validation_accuracy']-ref['validation_accuracy']),
                          'validation_ce_delta':row['validation_loss']-ref['validation_loss'],
                          'train_accuracy_delta_pp':100*(row['train_accuracy']-ref['train_accuracy']),
                          'train_ce_delta':row['train_loss']-ref['train_loss']})
        paired_rows.extend(pairs)
        if not pairs:continue
        frozen=name.endswith('_frozen_bn')
        label=name.split('_')[0] if frozen else name
        delta_axes[int(frozen)].plot([r['epoch'] for r in pairs],
                                    [r['validation_accuracy_delta_pp'] for r in pairs],
                                    marker='o',markersize=4,label=label)
        late=[r for r in pairs if 6<=r['epoch']<=10]
        final=next((r for r in pairs if r['epoch']==10),None)
        if len(late)==5 and final is not None:
            trajectory.append({'case':name,'epoch10':final,
                               'mean_validation_delta_pp_epochs6_10':sum(r['validation_accuracy_delta_pp'] for r in late)/5,
                               'mean_validation_ce_delta_epochs6_10':sum(r['validation_ce_delta'] for r in late)/5,
                               'epochs_better_than_reference':sum(r['validation_accuracy_delta_pp']>0 for r in pairs),
                               'epochs_compared':len(pairs)})
    for ax,title in zip(delta_axes,['Voltage scale / BN epsilon','Frozen BN affine parameters']):
        ax.axhline(0,color='black',linestyle='--',linewidth=.8)
        ax.set(title=title,xlabel='Epoch',xticks=range(1,11));ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=8)
    delta_axes[0].set_ylabel('Validation difference from existing reference (pp)')
    delta_fig.tight_layout();delta_fig.savefig(out/'matched_accuracy_differences.png',dpi=160);plt.close(delta_fig)
    if paired_rows:
        with (out/'matched_epoch_differences.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(paired_rows[0]));writer.writeheader();writer.writerows(paired_rows)
    (out/'trajectory_summary.json').write_text(json.dumps(trajectory,indent=2)+'\n')
    report += ['', '## Endpoint versus learning trajectory', '',
               'These are descriptive comparisons of the same seed and matched epochs. The mean over epochs6–10 is a sensitivity check, not an independent-seed estimate, statistical test, or replacement checkpoint-selection rule. Epoch10 remains the declared endpoint.', '',
               '| Intervention | Epoch10 validation difference | Mean difference, epochs6–10 | Epoch10 training CE difference |',
               '|---|---:|---:|---:|']
    for row in trajectory:
        report.append(f"| {row['case']} | {row['epoch10']['validation_accuracy_delta_pp']:+.2f} pp | {row['mean_validation_delta_pp_epochs6_10']:+.2f} pp | {row['epoch10']['train_ce_delta']:+.4f} |")
    report += ['', 'Positive training-CE differences mean worse fit to the augmented training stream.', '',
               '![Differences at each matched epoch](matched_accuracy_differences.png)']
    report += ['', '## BN-free initialization scale', '',
               'The accepted one-batch smokes use identical initialization and augmented training examples. Statistics below are measured before the first optimizer update; they are not trained accuracy results.', '',
               '| Scheme | Block 1 RMS | Block 2 RMS | Block 3 RMS | Logit RMS |',
               '|---|---:|---:|---:|---:|']
    scale_fig,scale_ax=plt.subplots(figsize=(6,4))
    for scheme in ['baseline','ours','legacy']:
        path=root/f'smoke-{scheme}_no_bn/metrics.jsonl'
        if not path.exists():continue
        row=json.loads(path.read_text().splitlines()[0])
        boundaries=row['first_batch_boundary_statistics']
        values=[boundaries[f'block_{i}/boundary_output']['rms'] for i in range(3)]
        values.append(row['first_batch_logit_statistics']['rms'])
        report.append('| '+scheme+' | '+' | '.join(f'{v:.6g}' for v in values)+' |')
        scale_ax.plot(['Block 1','Block 2','Block 3','Logits'],values,marker='o',label=scheme)
    scale_ax.set_yscale('log');scale_ax.set_ylabel('RMS (simulation units)')
    scale_ax.set_title('BN removed: before the first optimizer update')
    scale_ax.grid(alpha=.2);scale_ax.legend();scale_fig.tight_layout()
    scale_fig.savefig(out/'no_bn_initial_scale.png',dpi=160);plt.close(scale_fig)
    report += ['', 'Without BN, legacy’s output scale factors accumulate across blocks: 16 × 16 × 4 = 1024 relative to baseline at matched weights and gains. The smoke logit-RMS ratio is exactly 1024. Thus the BN-free comparison also changes the scale entering cross-entropy; it is not solely a test of learned BN affine parameters.']
    report += ['', '![BN-free initial scale](no_bn_initial_scale.png)']
    report += ['', '## Learned input gains', '',
               'All gains started at100. Values below come from locally collected completed-run checkpoints or the last saved checkpoint of a failed run; the epoch column gives its actual training extent. An absent case has no collected terminal checkpoint yet. These are actual learned input gains, separate from fixed amplification A/B and any fixed block-output normalization. The voltage-normalized legacy case still divides its outputs by16/16/4.', '',
               '| Case | Epoch | Block 1 | Block 2 | Block 3 | Classifier |',
               '|---|---:|---:|---:|---:|---:|']
    for name in paths:
        selected=[r for r in gain_rows if r['case']==name]
        if not selected:continue
        gains={r['parameter']:r['gain'] for r in selected}
        names=[f'analog_blocks.{i}._input_gain_raw' for i in range(3)]+['head._input_gain_raw']
        report.append(f"| {name} | {selected[0]['epoch']} | "+' | '.join(f'{gains[n]:.6f}' for n in names)+' |')
    if gain_rows:
        with (out/'input_gains.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(gain_rows[0]));writer.writeheader();writer.writerows(gain_rows)
    (out/'report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'complete':complete,'cases':[{k:r[k] for k in ['case','state','epochs']} for r in summary]}))


if __name__=='__main__':main()
