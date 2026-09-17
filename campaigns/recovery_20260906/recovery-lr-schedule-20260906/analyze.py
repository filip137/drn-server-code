"""Summarize saved recovery trajectories; never reevaluate or reselect models."""
import argparse
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def read(path):return json.loads(Path(path).read_text())

def screen_results(root,architecture):
    root=Path(root)
    paths=root.glob('runs/*/result.json') if architecture=='drn' else root.glob('results/*/runs/*/*/result.json')
    rows=[]
    for p in paths:
        d=read(p)
        if d.get('status')!='complete':continue
        if architecture=='drn':
            arm=d['arm']; initial=d['initial']['apparent']; selected=d['selected']['validation']['apparent']; final=d['final']['validation']['apparent']
            epoch=d['selected']['epoch'];pulses=d['selected']['pulses']; total=d['total_pulses'];history=d['candidates']
            trajectory=[dict(epoch=r['epoch'],kl=r['validation']['apparent']['kl_teacher_student'],pulses=r['pulses']) for r in history]
        else:
            m=d['metrics']; arm_id=p.parent.parent.name
            arm=next(a for a in read(HERE/'plan.json')['crossbar']['arms'] if a['arm_id']==arm_id)
            initial=m['initial']['validation']['apparent_forward'];selected=m['selected_validation']['apparent_forward'];final=m['training_final_validation']['apparent_forward']
            epoch=m['selected_epoch'];pulses=m['selected_optimizer']['applied_pulses'];total=m['training_final_optimizer']['applied_pulses']
            trajectory=[dict(epoch=0,kl=initial['kl_teacher_student'],pulses=0)]+[dict(epoch=r['epoch'],kl=r['validation']['apparent_forward']['kl_teacher_student'],pulses=r['optimizer_cumulative']['applied_pulses']) for r in m['epochs']]
        previous=[r['kl'] for r in trajectory if 20<=r['epoch']<25]
        last=[r['kl'] for r in trajectory if 25<=r['epoch']<=30]
        relative=(min(previous)-min(last))/min(previous) if previous and last else 0
        rows.append(dict(architecture=architecture,arm_id=arm['arm_id'],condition=arm['condition'],schedule_kind=arm['schedule']['kind'],learning_rate=arm['learning_rate'],selected_epoch=epoch,initial_kl=initial['kl_teacher_student'],selected_kl=selected['kl_teacher_student'],final_kl=final['kl_teacher_student'],initial_accuracy=initial['student_accuracy'],selected_accuracy=selected['student_accuracy'],final_accuracy=final['student_accuracy'],selected_pulses=pulses,total_pulses=total,relative_kl_reduction=(initial['kl_teacher_student']-selected['kl_teacher_student'])/initial['kl_teacher_student'],extension_eligible=epoch>=25 and relative>=.01,late_window_relative_improvement=relative,result_path=str(p),trajectory=trajectory))
    return rows

def choose(rows):
    winners=[]
    for condition in ['healthy','faulted']:
        for kind in ['constant','exponential']:
            candidates=[r for r in rows if r['condition']==condition and r['schedule_kind']==kind]
            if candidates:
                winners.append(min(candidates,key=lambda r:(r['selected_kl'],-r['selected_accuracy'],r['selected_pulses'],r['learning_rate'])))
    return winners

def main(args):
    rows=screen_results(HERE/'drn_screen','drn')+screen_results(HERE/'crossbar_collected','crossbar')
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    active_plan=read(HERE/('plan.with_lr_boundary.json' if (HERE/'plan.with_lr_boundary.json').exists() else 'plan.json'))
    expected=sum(len(active_plan[a]['arms']) for a in ['drn','crossbar'])
    if not rows:
        print('No completed arms yet.');return
    fields=[k for k in rows[0] if k!='trajectory']
    with (out/'screen_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    (out/'screen_summary.json').write_text(json.dumps(dict(status='complete' if len(rows)==expected else 'partial',completed_arms=len(rows),expected_arms=expected,rows=rows),indent=2)+'\n')
    for r in rows:print(f"{r['architecture']} {r['arm_id']}: {r['initial_kl']:.5f} -> {r['selected_kl']:.5f} @ {r['selected_epoch']:.3g}; final {r['final_kl']:.5f}")
    if not args.plot:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    colors={1e-6:'#6a4c93',3e-6:'#0077b6',1e-5:'#e07a17',3e-5:'#208a56',1e-4:'#aa3377'}
    for xkey,xlabel,filename in [('epoch','Recovery epochs','kl_vs_epochs'),('pulses','Cumulative update pulses','kl_vs_pulses')]:
        fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
        for ai,architecture in enumerate(['drn','crossbar']):
            for ci,condition in enumerate(['healthy','faulted']):
                ax=axes[ai,ci]; subset=[r for r in rows if r['architecture']==architecture and r['condition']==condition]
                for r in subset:
                    curve=r['trajectory']; lr=r['learning_rate']; decay=r['schedule_kind']=='exponential'
                    label=f'{lr:.0e}'+(' decay to 1% by epoch 10' if decay else ' constant')
                    ax.plot([v[xkey] for v in curve],[v['kl'] for v in curve],label=label,color=colors.get(lr),linestyle='--' if decay else '-',linewidth=1.8)
                    ix=min(range(len(curve)),key=lambda i:curve[i]['kl'])
                    ax.scatter(curve[ix][xkey],curve[ix]['kl'],s=25,color=colors.get(lr),zorder=4)
                if subset:ax.axhline(subset[0]['initial_kl'],color='.4',linestyle=':',linewidth=1,label='Original deployment')
                ax.set_title(f"{'DRN' if architecture=='drn' else 'Crossbar'} — {condition}")
                ax.set_xlabel(xlabel);ax.set_ylabel('Held-apparent validation KL (log scale)')
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(ticker.LogLocator(base=10,subs=(1,2,3,4,6,8)))
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,_:f'{value:g}'))
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
                ax.grid(alpha=.2,which='both')
                if xkey=='pulses':ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
                if subset:ax.legend(fontsize=7,frameon=False)
        fig.suptitle(f"Lower-rate recovery screen — {'complete' if len(rows)==expected else 'partial: '+str(len(rows))+'/'+str(expected)+' arms'}\nOne saved array/write per architecture; different teachers and device models",fontsize=12)
        for extension in ['png','pdf','svg']:fig.savefig(out/(filename+'.'+extension),dpi=170)
        plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plot',action='store_true');main(parser.parse_args())
