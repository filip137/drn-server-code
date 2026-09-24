"""Prepare matched baseline replacement configs without changing original evidence."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

from experiments.prepare_paper_training_completion import resolved_config, write_json, digest
from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'results/paper-training-completion-20260911-v1'
CONFIGS = ROOT/'configs/conv/paper_baseline_tk_revision_20260913_v1'
COLLECTION = ROOT/'paper_ready_results/baseline_tk_revision'
REVISION = 'baseline_tk_20260913_v1'


def main():
    selection = json.loads((STUDY/'checks/baseline_tk_20260913/free_phase_refinement.json').read_text())
    groups = {g['architecture']:g for g in selection['groups']}
    assert set(groups)=={'conv2','conv3'}
    assert all(g['status']=='qualified_native_K_refinement' for g in groups.values())
    with (ROOT/'paper_ready_results/run_status.csv').open() as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames; rows=list(reader)
    rows=[r for r in rows if r['block'] in ('T3_BPTT','T3_EP') and r['architecture'] in groups and r['scheme']=='baseline']
    assert len(rows)==36
    assets=json.loads((STUDY/'assets/initializers.json').read_text())
    ep_template=json.loads((ROOT/'paper_ready_results/bundles/table2_wide_ep/conv1/baseline/seed0/config.used.json').read_text())
    lists={}; prior=[]
    for row in rows:
        group=groups[row['architecture']]
        t,k=group['selected_T'],group['selected_K']
        selected=next(a for a in group['attempts'] if (a['T'],a['K'],a['beta'])==(t,k,group['selected_beta']))
        assert len(selected['ceilings'])==3 and all(r['unqualified_launch_gate_all_passed'] for r in selected['ceilings'])
        for r in selected['ceilings']:
            assert not validate_run(ROOT/r['run']) and digest(ROOT/r['run']/'result.json')==r['result_sha256']
        prior.append(copy.deepcopy(row))
        row.update(T=str(t),K=str(k))
        parent_path=ROOT/row['template_or_parent_config']
        parent=json.loads(parent_path.read_text())
        asset=assets[f"bounded_uniform/{row['architecture']}/seed{row['model_seed']}"]
        config=resolved_config(row,parent,asset,ep_template)
        config['model_base'].update(num_iterations_inference=t,num_iterations_training=k)
        config['completion_plan'].update(operating_point_revision=REVISION,
            resolved_parent_path=str(parent_path.relative_to(ROOT)),resolved_parent_sha256=digest(parent_path),
            numerical_tk_references=[{'run':r['run'],'result_sha256':r['result_sha256']} for r in selected['ceilings']])
        config['reporting']['evidence_class']='ordinary_mnist_matched_baseline_tk_revision'
        if row['algorithm']=='EP':
            config['beta']=group['selected_beta']
            config['eqprop'].update(injected_beta_B=group['selected_beta'],beta_tier='pending_new_matched_bptt_checkpoint_recheck')
            # The original-reference diagnostic selects T/K, but new BPTT
            # checkpoint qualification must precede the full EqProp pilots.
            config['completion_plan']['qualification']='bounded_beta_pending'
        branch='candidates' if row['algorithm']=='EP' else 'training'
        path=CONFIGS/branch/(row['cell_id']+'.json')
        write_json(path,config)
        row['template_or_parent_config']=str(path.relative_to(ROOT))
        for key in ('collected_result','source_result_sha256','best_validation_accuracy_percent',
                    'final_validation_accuracy_percent','completed_epochs','source_evidence_class','source_target',
                    'future_target_and_job','source_commit','source_archive_sha256','source_result_or_completion_record'):
            row[key]=''
        row.update(training_status='awaiting_new_matched_bptt_beta_check' if row['algorithm']=='EP' else 'planned_replacement',
                   paper_status='revised_operating_point_training_pending',official_test_read='false',
                   planned_action='matched_baseline_TK_replacement' if row['algorithm']=='BPTT' else 'new_matched_baseline_EP',
                   prelaunch_dependency='new_matched_BPTT_best_beta_gate_and_full_seed0_pilots' if row['algorithm']=='EP' else 'same_config_smoke_and_budget',last_checked='2026-09-13')
        if branch=='training':
            lists.setdefault(f"{row['architecture']}_bptt_seed{row['model_seed']}",[]).append(str(path.relative_to(ROOT)))
    for name,paths in lists.items():
        path=CONFIGS/'lists'/(name+'.txt');path.parent.mkdir(parents=True,exist_ok=True)
        content='\n'.join(sorted(paths))+'\n'
        if path.exists():assert path.read_text()==content
        else:path.write_text(content)
    COLLECTION.mkdir(parents=True,exist_ok=True)
    path=COLLECTION/'run_status.csv';assert not path.exists()
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write_json(COLLECTION/'original_native_tk_rows.json',prior)
    write_json(CONFIGS/'selection.json',selection)
    print('Prepared 18 BPTT configs, 18 held EqProp candidates, six BPTT seed lists and a separate 36-cell revision ledger')


if __name__=='__main__':
    main()
