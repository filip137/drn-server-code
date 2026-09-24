"""Collect and summarize the completed larger-T/K validation-only diagnostic."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'results/paper-training-completion-20260911-v1'
CHECKS = STUDY/'checks/baseline_tk_20260913'
PAPER = ROOT/'paper_ready_results'


def main():
    selection = json.loads((CHECKS/'selection.json').read_text())
    assert len(selection['groups']) == 2
    assert all(group['status'] != 'running' for group in selection['groups'])
    assert (CHECKS/'exit_code').read_text().strip() == '0'
    assert len(selection['axis_diagnostics']) == 4
    records = [row for group in selection['groups'] for attempt in group['attempts'] for row in attempt['ceilings']]
    records += selection['axis_diagnostics']
    refinement_path = CHECKS/'free_phase_refinement.json'
    refinement = json.loads(refinement_path.read_text()) if refinement_path.exists() else None
    if refinement is not None:
        assert (CHECKS/'refinement_exit_code').read_text().strip() == '0'
        assert all(g['status'] != 'running' for g in refinement['groups'])
        extra = [row for g in refinement['groups'] for a in g['attempts'] for row in a['ceilings']]
        records = list({row['run']: row for row in records+extra}.values())
    assert len({row['run'] for row in records}) == len(records)
    summary = []
    for row in records:
        run = ROOT/row['run']
        assert not validate_run(run), str(run)
        assert hashlib.sha256((run/'result.json').read_bytes()).hexdigest() == row['result_sha256']
        inventory = json.loads((run/'source_inventory.json').read_text())['cases'][0]
        assert (inventory['T'],inventory['K']) == (row['T'],row['K'])
        assert inventory['native_T'] == inventory['native_K'] == (6 if row['architecture']=='conv2' else 8)
        assert row['replay_count'] == 8
        assert row['layer_comparison_count'] == (24 if row['architecture']=='conv2' else 32)
        assert row['official_test_read'] is False and row['optimizer_steps_applied'] is False
        assert row['source_bytes_unchanged'] and row['all_bias_tensors_exact_zero']
        residuals = list(csv.DictReader((run/'equilibrium_residuals.csv').open()))
        summary.append({key:row[key] for key in ('architecture','ceiling','T','K','injected_beta','minimum_cosine',
                       'maximum_symmetric_norm_delta','gradient_fidelity_all_passed','equilibrium_residual_all_passed',
                       'unqualified_launch_gate_all_passed','run','result_sha256')})
    target = PAPER/'provenance/baseline_tk_20260913'
    target.mkdir(parents=True,exist_ok=True)
    copied = []
    for row in records:
        run = ROOT/row['run']; dest = target/'runs'/run.name
        if dest.exists():
            assert (dest/'result.json').read_bytes() == (run/'result.json').read_bytes()
        else:
            shutil.copytree(run,dest,symlinks=True)
        assert not validate_run(dest), str(dest)
        for p in run.rglob('*'):
            if p.is_file():
                d=dest/p.relative_to(run); digest=hashlib.sha256(p.read_bytes()).hexdigest()
                assert hashlib.sha256(d.read_bytes()).hexdigest()==digest
                copied.append({'file':str(d.relative_to(PAPER)),'source':str(p.relative_to(ROOT)),'sha256':digest})
    for p in (CHECKS/'selection.json',CHECKS/'plan.json',CHECKS/'source/experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py',
              CHECKS/'prepare_bounded_ep_tk_checks.py',Path(__file__)):
        shutil.copy2(p,target/p.name)
    if refinement is not None:
        shutil.copy2(refinement_path,target/refinement_path.name)
        shutil.copy2(STUDY/'launch/refine_baseline_free_phase_tk.py',target/'refine_baseline_free_phase_tk.py')
    (target/'collection_validation.json').write_text(json.dumps({'all_canonical_valid':True,'read_only_guards_valid':True,'run_count':len(records),'files':copied},indent=2)+'\n')
    csv_path=PAPER/'baseline_tk_beta_qualification.csv'
    with csv_path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),constrained_layout=True)
    for i,arch in enumerate(('conv2','conv3')):
        for ceiling,color in zip(('1em4','5em4','1em3'),('#1274b8','#bd681c','#3b8b55')):
            rows=sorted([r for r in summary if r['architecture']==arch and r['ceiling']==ceiling and r['T']==r['K'] and r['injected_beta']==.01],key=lambda x:x['T'])
            for j,key in enumerate(('minimum_cosine','maximum_symmetric_norm_delta')):
                axes[i,j].plot([r['T'] for r in rows],[r[key] for r in rows],'-o',label=ceiling.replace('em','e-'),color=color)
                axes[i,j].set_xlabel('T = K (iterations)');axes[i,j].grid(alpha=.2)
                axes[i,j].set_xticks(sorted({r['T'] for r in rows}))
        axes[i,0].axhline(.99,color='black',ls='--',lw=1,label='Required ≥ 0.99')
        axes[i,1].axhline(.10,color='black',ls='--',lw=1,label='Required ≤ 0.10')
        axes[i,0].set_ylabel(f'{arch}: minimum cosine');axes[i,1].set_ylabel(f'{arch}: maximum symmetric\nnorm difference')
        for ax in axes[i]:ax.legend(fontsize=8)
    fig.suptitle('Baseline: additional relaxation resolves small-beta mismatch (β = 0.01)')
    for ext in ('png','pdf'):fig.savefig(PAPER/f'figures/baseline_tk_beta_qualification.{ext}',dpi=180)
    lines=['# Baseline EqProp: larger-T/K qualification','',
           'The original Conv2/Conv3 baseline failures are reproduced at T=K=6/8 on the local RTX 3090. Larger iteration counts resolve the gradient and equilibrium gates on the same checkpoints and fixed validation batches. No training or official-test evaluation is included.','',
           '| Architecture | Qualified T/K | Largest passing tested beta | Worst cosine | Maximum symmetric norm difference |',
           '|---|---:|---:|---:|---:|']
    for group in selection['groups']:
        if group['selected_beta'] is None:
            lines.append(f"| {group['architecture']} | unresolved | — | — | — |")
            continue
        rows=next(a['ceilings'] for a in group['attempts'] if a['T']==group['selected_T'] and a['K']==group['selected_K'] and a['beta']==group['selected_beta'])
        lines.append(f"| {group['architecture']} | {group['selected_T']}/{group['selected_K']} | {group['selected_beta']:g} | {min(r['minimum_cosine'] for r in rows):.6f} | {max(r['maximum_symmetric_norm_delta'] for r in rows):.6f} |")
    if refinement is not None:
        lines += ['', 'The follow-up keeps native K and reduces the iteration cost. These are the selected settings for the newly authorized matched BPTT/EqProp baseline training:', '',
                  '| Architecture | Selected T/K | Common beta | Worst cosine | Maximum symmetric norm difference |',
                  '|---|---:|---:|---:|---:|']
        for g in refinement['groups']:
            assert g['status']=='qualified_native_K_refinement'
            rows=next(a['ceilings'] for a in g['attempts'] if (a['T'],a['K'],a['beta'])==(g['selected_T'],g['selected_K'],g['selected_beta']))
            lines.append(f"| {g['architecture']} | {g['selected_T']}/{g['selected_K']} | {g['selected_beta']:g} | {min(r['minimum_cosine'] for r in rows):.6f} | {max(r['maximum_symmetric_norm_delta'] for r in rows):.6f} |")
        lines += ['', 'Conv3 T=20/K=8 still fails the norm threshold at the largest ceiling; T=24/K=8 passes everywhere. Larger beta candidates 100, 10 and 1 are rejected once a ceiling fails; those deliberately short-circuited failed candidates do not have all-three-ceiling coverage. Every selected candidate does.', '',
                  'Filip requested the new T/K for both algorithms. First train new seed-0 BPTT references and repeat the beta qualification at their best checkpoints, then run the three full EqProp pilots before releasing seeds 1/2. The previous 18 baseline BPTT results remain intact as original-operating-point evidence.']
    lines += ['',f"All {len(records)} full diagnostic replays validate locally: both checkpoint roles, all four batches, finite float64 calculations, exact-zero biases and unchanged source/checkpoint bytes. The native GPU controls reproduce the prior CPU failure measurements. No layer or batch is excluded.",'',
              '| Architecture | T/K | Previously failing ceiling | Worst cosine | Maximum symmetric norm difference | Combined gate |',
              '|---|---:|---:|---:|---:|---|']
    for row in selection['axis_diagnostics']:
        lines.append(f"| {row['architecture']} | {row['T']}/{row['K']} | {row['ceiling']} | {row['minimum_cosine']:.6f} | {row['maximum_symmetric_norm_delta']:.6f} | {'pass' if row['unqualified_launch_gate_all_passed'] else 'fail'} |")
    lines += ['', 'For Conv2 at the tight ceiling, increasing only the free-phase T fixes the mismatch; increasing only the gradient-phase K does not. This supports insufficient free-phase relaxation as the cause in that case. The Conv3 controls show what doubling one phase achieves, without establishing a globally optimal T/K choice.', '',
              'The gate requires every weight-layer/batch cosine ≥ .99, symmetric norm difference ≤ .10, and all projected-residual p90 checks ≤ .01. The selected beta is shared across all three ceilings. Selection tested larger betas in descending order at the first passing equal-T/K setting; higher iteration counts were not searched once a setting passed.', '',
              'The source BPTT checkpoints were trained at T/K=6/6 and 8/8; these replays evaluate both gradients at the explicitly larger diagnostic T/K. New full training would change the baseline operating-point contract and still requires three full seed-0 pilots before releasing seeds 1/2. Existing BPTT runs cannot be relabeled as trained at these larger counts.', '',
              'Individual replay reports inherit a generic “shared paper T/K” introductory phrase from the frozen analyzer. Their T/K tables, execution manifests and source-native T/K fields record the actual override; this collection is diagnostic evidence, not a replacement of the original paper contract.', '',
              '![Gradient agreement versus relaxation iterations](figures/baseline_tk_beta_qualification.png)', '',
              '[All diagnostic measurements](baseline_tk_beta_qualification.csv) · [Selection and source provenance](provenance/baseline_tk_20260913/selection.json)', '']
    (PAPER/'baseline_tk_beta_qualification.md').write_text('\n'.join(lines))
    print('VALIDATED_AND_COLLECTED',len(records),'diagnostic replays')


if __name__=='__main__':
    main()
