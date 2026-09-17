"""Check paired coverage and render KL comparisons from completed outputs."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'analysis'
OUT.mkdir(exist_ok=True)
plan = json.loads((ROOT/'plan.json').read_text())
results, rows, checks = {}, [], []
for condition in ('healthy','faulted'):
    for schedule in ('constant','exponential'):
        pair = []
        for writer in ('open_loop','closed_loop_pv'):
            name = f'{condition}-{schedule}-{writer}'
            r = json.loads((ROOT/'runs'/name/'result.json').read_text())
            assert r['status']=='complete' and r['selected_replay_passed']
            assert len(r['epochs'])==30
            assert [e['epoch'] for e in r['epochs']]==list(range(1,31))
            assert all(e['examples']==55000 and e['batches']==3438 for e in r['epochs'])
            assert all(e['invariants']['bounds_respected'] and e['invariants']['faults_immutable'] for e in r['epochs'])
            assert r['manifest']['plan_sha256']==hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest()
            for source,digest in r['manifest']['code_sha256'].items():
                path = ROOT/source if '/' not in source else ROOT/'crossbar_code'/source
                assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
            for e in r['epochs']:
                s=plan['schedules'][schedule]
                expected=plan['crossbar']['learning_rates'][condition]*s['final_factor']**min((e['epoch']-1)/(s['decay_epochs']-1),1)
                assert e['effective_learning_rate']==expected
            row = dict(condition=condition,schedule=schedule,writer=writer,
                       initial_learning_rate=r['manifest']['learning_rate'],
                       selected_epoch=r['selected']['epoch'],selected_pulses=r['selected']['pulses'],
                       total_pulses=r['total_pulses'],pulsed_cells=r['epochs'][-1]['pulsed_cells'],
                       verify_reads=r['verify_reads'],runtime_seconds=r['total_seconds'])
            row['pulses_per_cell']=row['total_pulses']/203264
            for stage in ('initial','selected','final'):
                for state,prefix in (('apparent',''),('persistent_secondary_diagnostic','persistent_')):
                    m=r[f'{stage}_test'][state]
                    assert m['examples']==10000
                    row[f'{stage}_{prefix}test_KL']=m['kl_teacher_student']
                    row[f'{stage}_{prefix}test_accuracy_percent']=100*m['student_accuracy']
            if writer=='closed_loop_pv':
                row['first_write_epoch']=next((e['epoch'] for e in r['epochs'] if e['pulses']),None)
                row['final_target_error_max']=r['epochs'][-1]['target_error_max']
            else:
                row['first_write_epoch']=1 if r['total_pulses'] else None
                row['final_target_error_max']=None
            pair.append(r)
            rows.append(row)
            results[condition,schedule,writer]=r
        a,b=pair
        for field in ('initial_state','inputs','seeds','learning_rate','epochs','objective','code_sha256','plan_sha256','learning_rate_schedule'):
            assert a['manifest'][field]==b['manifest'][field],(condition,schedule,field)
        assert a['initial']==b['initial'] and a['initial_test']==b['initial_test']
        for key in ('data_stream_sha256','ordered_model_inputs_sha256','ordered_labels_sha256','effective_learning_rate'):
            assert [e[key] for e in a['epochs']]==[e[key] for e in b['epochs']]
        checks.append(dict(condition=condition,schedule=schedule,exact_P0_data_schedule=True,
                           physical_invariants=True,full_coverage=True))

with (OUT/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(OUT/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
(OUT/'verification.json').write_text(json.dumps(dict(status='complete',arms=8,epochs=240,pairs=checks),indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
colors={'open_loop':'#be5c2b','closed_loop_pv':'#246b99'}
labels={'open_loop':'Open-loop pulse','closed_loop_pv':'Closed-loop P&V'}
for metric,filename,ylabel in [('kl','validation_kl','Validation KL(teacher || student) ↓'),('pulses','pulse_exposure','Cumulative recovery pulses per cell')]:
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    fig.subplots_adjust(left=.11,right=.98,top=.86,bottom=.14,hspace=.4,wspace=.25)
    for row,schedule in enumerate(('constant','exponential')):
        for col,condition in enumerate(('healthy','faulted')):
            ax=axes[row,col]
            for writer in colors:
                r=results[condition,schedule,writer]
                if metric=='kl':
                    values=[r['initial']['apparent']['kl_teacher_student']]+[e['validation']['apparent']['kl_teacher_student'] for e in r['epochs']]
                else:
                    values=[0]+[e['pulses']/203264 for e in r['epochs']]
                ax.plot(range(31),values,color=colors[writer],label=labels[writer],lw=1.8)
            ax.set_title(f'{"Healthy" if condition=="healthy" else "Corrupt"} · {"constant rate" if schedule=="constant" else "exponential decay"}')
            ax.set_xlabel('Recovery epoch'); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=9)
    fig.suptitle('Crossbar recovery with the September 6 learning-rate choices',fontsize=14,y=.965)
    fig.text(.5,.02,'Healthy: 3e-6 · Corrupt: 1e-5 · Decay reaches 1% at epoch 10\nSame HWA + P&V P0, data order and schedule within each pair · One development array/write',ha='center',fontsize=9)
    for ext in ('png','svg','pdf'): fig.savefig(OUT/f'{filename}.{ext}',dpi=200)
    plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(9,6.5))
fig.subplots_adjust(left=.1,right=.98,top=.85,bottom=.14,hspace=.42,wspace=.27)
for col,schedule in enumerate(('constant','exponential')):
    for row,condition in enumerate(('healthy','faulted')):
        ax=axes[row,col]
        a,b=[results[condition,schedule,w] for w in colors]
        values=[a['initial_test']['apparent']['kl_teacher_student'],a['selected_test']['apparent']['kl_teacher_student'],b['selected_test']['apparent']['kl_teacher_student']]
        ax.bar(range(3),values,color=['#999999',colors['open_loop'],colors['closed_loop_pv']],width=.65)
        for i,v in enumerate(values): ax.text(i,v+max(values)*.025,f'{v:.5f}',ha='center',fontsize=9)
        ax.set_ylim(0,max(values)*1.22); ax.set_xticks(range(3),['P0','Open-loop','P&V'])
        ax.set_title(f'{"Healthy" if condition=="healthy" else "Corrupt"} · {schedule}')
        ax.set_ylabel('Test KL(teacher || student) ↓'); ax.grid(axis='y',alpha=.18); ax.set_axisbelow(True)
fig.suptitle('Test KL at validation-selected crossbar checkpoints',fontsize=14,y=.96)
fig.text(.5,.025,'Selection from epochs 0–30 by validation KL · Held apparent state\nOne development array/write · Test measurements do not select schedules or epochs',ha='center',fontsize=9)
for ext in ('png','svg','pdf'): fig.savefig(OUT/f'selected_test_kl.{ext}',dpi=200)
plt.close(fig)

constant_corrupt=[r for r in rows if r['condition']=='faulted' and r['schedule']=='constant']
open_corrupt=next(r for r in constant_corrupt if r['writer']=='open_loop')
pv_corrupt=next(r for r in constant_corrupt if r['writer']=='closed_loop_pv')
reduction=100*(1-pv_corrupt['selected_test_KL']/open_corrupt['selected_test_KL'])
no_write_decay=[r['condition'] for r in rows if r['schedule']=='exponential' and r['writer']=='closed_loop_pv' and r['total_pulses']==0]
headline=f"With the newer constant rate and 30 epochs, corrupt crossbar selected test KL is {open_corrupt['selected_test_KL']:.6f} for open-loop recovery and {pv_corrupt['selected_test_KL']:.6f} for P&V ({reduction:.1f}% lower). This changes both the rate and training budget relative to the original three-epoch comparison; their separate effects are not isolated."
if no_write_decay:
    headline+=f" The P&V decay schedule makes no physical writes in the following conditions: {', '.join(no_write_decay)}. Its unchanged KL reflects the verify tolerance and accumulated target movement."
boundary=[r['condition'] for r in rows if r['writer']=='closed_loop_pv' and r['schedule']=='constant' and r['selected_epoch']==30]
if boundary:
    headline+=f" Constant-rate P&V selects the epoch-30 budget boundary for {', '.join(boundary)} devices; convergence has not been established."
lines=['# Crossbar recovery with the newer schedules','',headline,'',
       'Completed exploratory comparison of eight arms from the same original development-array HWA + P&V states. Both writers use 30 full epochs, the September 6 rates (3e-6 healthy; 1e-5 corrupt), and the same constant or exponential schedule. Decay reaches 1% of the initial rate at epoch 10 and holds it through epoch 30. Checkpoint selection minimizes held-apparent validation KL, including P0.','',
       '| Devices | Schedule | Writer | Selected epoch | P0 test KL | Selected test KL | Final test KL | Total pulses |','|---|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['condition']} | {r['schedule']} | {labels[r['writer']]} | {r['selected_epoch']} | {r['initial_test_KL']:.6f} | {r['selected_test_KL']:.6f} | {r['final_test_KL']:.6f} | {r['total_pulses']:,} |")
lines+=['','![Selected test KL](selected_test_kl.png)','','![Validation trajectories](validation_kl.png)','',
        'The controller retains the previous verify tolerance of 0.04745 in q=a−r. It accumulates Adam targets, observes held apparent state, and issues SET/RESET pulses until within tolerance, with limits of 128 pulses per cell per minibatch and 640 per cell during recovery. Hidden bounds and immobile native OM faults are enforced only by the plant. Verify reads do not resample write noise.','',
        '| Devices | Schedule | First P&V write epoch | Cells pulsed over 30 epochs | Final maximum target error |','|---|---|---:|---:|---:|']
for r in rows:
    if r['writer']=='closed_loop_pv':
        lines.append(f"| {r['condition']} | {r['schedule']} | {r['first_write_epoch'] if r['first_write_epoch'] is not None else 'No writes'} | {100*r['pulsed_cells']/203264:.2f}% | {r['final_target_error_max']:.6f} |")
lines+=['','![Pulse exposure](pulse_exposure.png)','',
        'The schedules were selected for open-loop recovery in the recent study. This transfer tests those exact choices with P&V; it does not establish an optimized P&V learning rate or tolerance. A decay schedule can stop useful target accumulation before any verify threshold is crossed. No-write outcomes must be interpreted through the controller activity, not as an architecture limit.','',
        'Both writers were rerun on the same local RTX 3090. Full canaries matched the remote schedule and data order but did not reproduce its pulse trajectories. A separate local replay verified exact gradients and physical states against the native update loop. The prior remote outcomes remain unchanged and are not substituted into these matched pairs.','',
        'The original DRN results used three epochs, a different teacher and a different device/fault model. Keep them distinct from this thirty-epoch crossbar figure. Held apparent results include the conditioned post-write noise state; persistent-state KL and accuracy are secondary diagnostics in the CSV. One development array/write does not establish population uncertainty or an architecture ranking.','',
        '[Full metrics](comparison.csv) · [Pairing and coverage checks](verification.json) · [Native-loop replay](../smoke/native_loop_parity.json) · [Plan](../plan.json)']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows,indent=2))
