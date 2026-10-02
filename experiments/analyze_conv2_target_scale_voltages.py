"""Validate and plot trained Conv2 target-encoding voltage measurements."""
import csv
import json
import math
from pathlib import Path

from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'results/conv2-target-scale-voltage-20260929-v1'


def main():
    production = STUDY / 'production'
    config = json.loads((ROOT / 'configs/conv/conv2_target_scale_voltage_20260929.json').read_text())
    summary = json.loads((production / 'summary.json').read_text())
    assert summary['cases'] == 6 and summary['profile_rows'] == 18
    assert summary['examples_per_case'] == 576 and summary['full_cohort']
    assert summary['checkpoint_role'] == 'final' and summary['T'] == 6
    assert not validate_run(production)
    cases = sorted((production / 'cases').iterdir())
    assert {p.name for p in cases} == {c['name'] for c in config['cases']}
    for case in cases:
        errors = validate_run(case)
        assert not errors, (case, errors)
        guard = json.loads((case / 'read_only_guards.json').read_text())
        assert all(guard[k] for k in ('exact_restore','parameters_unchanged','source_bytes_unchanged','finite_states','zero_biases','native_float32'))
        assert guard['examples'] == 576 and guard['cohort_sha256'] == config['expected_cohort_sha256']
    with (production / 'voltage_profiles.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 18
    assert {(r['scheme'],float(r['target_scale']),int(r['layer_index'])) for r in rows} == {
        (s,t,i) for s in ('baseline','ours','legacy') for t in (.0625,.25) for i in range(3)}
    for r in rows:
        assert int(r['element_count']) == 576 * [64*14*14,128*7*7,20][int(r['layer_index'])]
        assert math.isclose(float(r['rms'])**2,float(r['signed_mean'])**2+float(r['std'])**2,rel_tol=1e-10)
        assert float(r['mean_absolute']) <= float(r['rms'])
    out = STUDY / 'analysis'
    out.mkdir(exist_ok=True)
    (out / 'voltage_epoch_comparison.csv').write_bytes((production / 'voltage_profiles.csv').read_bytes())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'baseline':'#333333','ours':'#d95f02','legacy':'#9b59b6'}
    labels = {'baseline':'Baseline (1, 1)','ours':'Ours (4, 1)','legacy':'Legacy (4, 0.25)'}
    for metric, label in [('rms','RMS voltage (V)'),('signed_mean','Signed mean voltage (V)')]:
        fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
        for ax,target,encoding in zip(axes,(.0625,.25),('1/16','1/4')):
            for scheme in colors:
                group=sorted((r for r in rows if r['scheme']==scheme and float(r['target_scale'])==target),key=lambda r:int(r['layer_index']))
                ax.plot(range(3),[float(r[metric]) for r in group],marker='o',color=colors[scheme],label=labels[scheme])
            if metric=='rms': ax.set_yscale('log')
            else: ax.axhline(0,color='gray',lw=.7)
            ax.set_xticks(range(3),['Conv1','Conv2','Readout'])
            ax.set_title(f'Target amplitude {encoding}')
            ax.set_xlabel('Layer'); ax.grid(alpha=.2)
        axes[0].set_ylabel(label)
        axes[1].legend(title='(voltage gain, current gain)',fontsize=9)
        fig.suptitle('Trained Conv2 · clean BPTT · final epoch 30 · seed 0\n576 shared validation examples · native T=6 · physical node voltages')
        fig.tight_layout()
        fig.savefig(out/f'{metric}_trained.jpg',dpi=180,bbox_inches='tight')
        plt.close(fig)
    (out/'validation.json').write_text(json.dumps(dict(complete=True,cases=6,layer_rows=18,examples_per_case=576,checkpoint_role='final',checkpoint_epoch=30,cohort_sha256=config['expected_cohort_sha256'],official_test_read=False),indent=2)+'\n')
    for r in rows:
        print(r['scheme'],r['target_scale'],r['layer'],'RMS',round(float(r['rms']),7))


if __name__ == '__main__':
    main()
