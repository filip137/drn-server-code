"""Validate and plot the healthy control at the corrupt arm's constant rate."""
import csv
import hashlib
import json
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
CONTROL=ROOT/'healthy_lr1e5_control'
OUT=CONTROL/'analysis'
plan=json.loads((CONTROL/'plan.json').read_text())
paths=[Path(plan['comparison_sources']['healthy']['path']),
       CONTROL/'run/result.json',
       Path(plan['comparison_sources']['faulted']['path'])]
labels=['Healthy · 3e-6','Healthy · 1e-5','Corrupt · 1e-5']
colors=['#888888','#246b99','#be5c2b']
results=[json.loads(p.read_text()) for p in paths]
old,new,corrupt=results
assert json.loads((CONTROL/'terminal.json').read_text())['semantic_complete']
for source in plan['comparison_sources'].values():
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
for r in results:
    assert r['status']=='complete' and r['selected_replay_passed']
    assert len(r['epochs'])==30
    assert all(e['examples']==55000 and e['batches']==3438 for e in r['epochs'])
    assert all(e['invariants']['bounds_respected'] and e['invariants']['faults_immutable'] for e in r['epochs'])
    assert r['manifest']['code_sha256']==new['manifest']['code_sha256']
    assert [e['data_stream_sha256'] for e in r['epochs']]==[e['data_stream_sha256'] for e in new['epochs']]
    for key in ('seeds','objective','verify_tolerance','total_pulse_cap','maximum_verify_pulses','learning_rate_schedule'):
        assert r['manifest'][key]==new['manifest'][key],key
assert old['manifest']['initial_state']==new['manifest']['initial_state']
assert old['manifest']['inputs']==new['manifest']['inputs']
assert old['initial']==new['initial'] and old['initial_test']==new['initial_test']
assert new['manifest']['learning_rate']==corrupt['manifest']['learning_rate']==1e-5
assert all(e['effective_learning_rate']==1e-5 for e in new['epochs'])

rows=[]
for label,r in zip(labels,results):
    row=dict(case=label,learning_rate=r['manifest']['learning_rate'],selected_epoch=r['selected']['epoch'],
             first_write_epoch=next((e['epoch'] for e in r['epochs'] if e['pulses']),None),
             total_pulses=r['total_pulses'],pulsed_cells=r['epochs'][-1]['pulsed_cells'],
             runtime_seconds=r['total_seconds'])
    for stage in ('initial','selected','final'):
        for state,prefix in (('apparent',''),('persistent_secondary_diagnostic','persistent_')):
            m=r[f'{stage}_test'][state]
            assert m['examples']==10000
            row[f'{stage}_{prefix}test_KL']=m['kl_teacher_student']
            row[f'{stage}_{prefix}test_accuracy_percent']=100*m['student_accuracy']
    rows.append(row)
with (OUT/'comparison.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(OUT/'comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
checks=dict(status='complete',new_epochs=30,unchanged_healthy_P0=True,identical_data_streams=True,
            same_teacher_source_and_controller=True,rate_matched_to_corrupt=True,
            physical_invariants=True,selected_replay_passed=True,reference_hashes_unchanged=True)
(OUT/'verification.json').write_text(json.dumps(checks,indent=2)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
fig.subplots_adjust(left=.08,right=.985,bottom=.23,top=.82,wspace=.29)
for label,color,r in zip(labels,colors,results):
    ys=[r['initial']['apparent']['kl_teacher_student']]+[e['validation']['apparent']['kl_teacher_student'] for e in r['epochs']]
    axes[0].plot(range(31),ys,color=color,label=label,lw=1.8)
axes[0].set(xlabel='Recovery epoch',ylabel='Validation KL(teacher || student) ↓',title='P&V recovery trajectories')
axes[0].legend(frameon=False,fontsize=9)
values=[r['selected_test_KL'] for r in rows]
axes[1].bar(range(3),values,color=colors,width=.65)
for x,v in enumerate(values):axes[1].text(x,v+max(values)*.025,f'{v:.6f}',ha='center',fontsize=10)
axes[1].set_xticks(range(3),['Healthy\n3e-6','Healthy\n1e-5','Corrupt\n1e-5'])
axes[1].set(ylabel='Test KL(teacher || student) ↓',title='Validation-selected checkpoints',ylim=(0,max(values)*1.25))
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('Healthy crossbar P&V at the matched learning rate',fontsize=15,y=.95)
fig.text(.5,.035,'Constant rates · 30 epochs · Same teacher, data order and verify tolerance (0.04745)\nHeld apparent state · One paired array/write · P0 eligible for checkpoint selection',ha='center',fontsize=9)
for ext in ('png','svg','pdf'):fig.savefig(OUT/f'healthy_matched_rate.{ext}',dpi=200)
plt.close(fig)

kl_old,kl_new,kl_corrupt=[r['selected_test_KL'] for r in rows]
relation='lower' if kl_new < kl_corrupt else 'higher'
text=f'The healthy P&V rerun at LR=1e-5 completed all 30 epochs. Its selected test KL is {kl_new:.6f}, compared with {kl_old:.6f} at LR=3e-6 and {kl_corrupt:.6f} for the corrupt P&V arm at LR=1e-5. Healthy KL is {relation} than corrupt KL after matching the rate. The new healthy run changes only the rate relative to its previous healthy control.'
lines=['# Healthy crossbar P&V with matched learning rate','',text,'',
       '| Case | First write epoch | Selected epoch | P0 test KL | Selected test KL | Test accuracy | Total pulses |',
       '|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['case']} | {r['first_write_epoch']} | {r['selected_epoch']} | {r['initial_test_KL']:.6f} | {r['selected_test_KL']:.6f} | {r['selected_test_accuracy_percent']:.2f}% | {r['total_pulses']:,} |")
lines+=['','![Matched-rate comparison](healthy_matched_rate.png)','',
        'The source teacher, healthy starting tensors, dataset split/order, 55,000 training examples per epoch, batch size 16, 5,000 validation examples, optimizer/controller implementation, noise model, verification tolerance and pulse caps are unchanged. The corrupt arm is the preserved completed LR=1e-5 run. Sources, all 30 minibatch streams and physical invariants were verified. The new test readout uses the full official 10,000-example cohort after validation selected the checkpoint.','',
        'This one-array follow-up tests the learning-rate explanation. It does not establish a global optimum, population uncertainty, or convergence. Held apparent metrics include the retained write-noise state; persistent diagnostics and final-epoch outcomes are in the CSV.','',
        '[Full metrics](comparison.csv) · [Verification](verification.json) · [Control plan](../plan.json)']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')
receipt=dict(status='complete',finished=time.time(),checks=checks,artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file() and p.suffix in ('.csv','.json','.md','.png','.svg','.pdf')})
(OUT/'completion_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(rows,indent=2))
