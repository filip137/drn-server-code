"""Prepare Conv1 matched clean seeds and baseline-noise controls, without launch."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import torch
from experiments.prepare_paper_training_completion import build_energy,state_hash

STUDY=Path('results/conv1-clean-and-baseline-noise-20260927-v1')
CONFIGS=Path('configs/conv/conv1_clean_baseline_noise_20260927')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    STUDY.mkdir(parents=True,exist_ok=True);CONFIGS.mkdir(parents=True,exist_ok=True)
    replay_path=Path('results/conv123-relative-noise-advantage-map-20260925-v1/stage_a/conv1/config.used.json')
    replay=json.loads(replay_path.read_text());templates={};proofs={}
    for scheme in ['baseline','ours','legacy']:
        case=next(c for c in replay['cases'] if c['scheme']==scheme and c['relative_target']==1)
        source=Path(case['source_config']);assert sha(source)==case['source_config_sha256']
        b=json.loads(source.read_text())
        template=Path(f'configs/conv/relative_noise_training_20260925/conv1_{scheme if scheme != "baseline" else "ours"}_relative1_eta1em4_seed0.json')
        c=json.loads(template.read_text())
        # Restore every scheme's own frozen optimizer and model contract.
        for key in ['model_base','model_overrides','optimizer','lr','learning_rates_by_parameter','handoff','parameter_order']:
            c[key]=copy.deepcopy(b[key])
        match_path=replay_path.parent/'cases'/f'conv1_{scheme}_relative1'/'matching.json'
        m=json.loads(match_path.read_text())
        assert m['matched'] and m['relative_target']==1 and m['relative_error']<=.01
        assert m['matching_examples']==576 and m['cohort_sha256']==replay['expected_cohort_sha256']
        assert case['T']==case['K']==4
        c['model_base'].update(num_iterations_inference=4,num_iterations_training=4)
        scale=c['model_base']['voltage_amp']/c['model_base']['current_amp']
        c['beta']=m['base_beta'];c['eqprop'].update(amplification_factor=scale,injected_beta_B=m['injected_beta'])
        assert abs(m['injected_beta']/scale-c['beta'])<1e-12
        proof=dict(source_config=str(source),source_config_sha256=sha(source),matching_path=str(match_path),matching_sha256=sha(match_path),replay_path=str(replay_path),replay_sha256=sha(replay_path),initializer_checkpoint_sha256=case['initializer_checkpoint_sha256'],base_beta=m['base_beta'],injected_beta=m['injected_beta'],measured=m,lr=b['lr'])
        c['relative_output_calibration']=dict(target=1.0,T=4,K=4,base_beta=m['base_beta'],injected_beta=m['injected_beta'],definition=c['relative_output_calibration']['definition'],reference_seed=0,beta_frozen_across_seeds=True,seed0_reference=proof)
        templates[scheme]=c;proofs[scheme]=proof
    assets=[];records=[];hashes=[]
    for seed in [0,1,2]:
        source=Path(f'results/paper-training-completion-20260911-v1/assets/wide_kaiming/conv1/seed{seed}.pt')
        state_identity=None
        for scheme,template in templates.items():
            cfg=copy.deepcopy(template);cfg['seed']=seed
            energy=build_energy(cfg);expected=state_hash(energy);energy.load(source)
            assert state_hash(energy)==expected
            if state_identity is None:state_identity=expected
            assert expected==state_identity
            assert all(torch.isfinite(p.state).all() and p.state.dtype==torch.float32 for p in energy.params())
            assert all(torch.count_nonzero(p.state)==0 for p in energy.params() if str(p.name).startswith('Bias'))
            if seed==0:assert sha(source)==proofs[scheme]['initializer_checkpoint_sha256']
        hashes.append(state_identity)
        checkpoint=STUDY/'initializers'/f'conv1_seed{seed}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,checkpoint)
        assets.append(dict(seed=seed,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),tensor_sha256=state_identity,reproduced_normal_seed_draw=True,matched_across_three_schemes=True))
        for scheme,template in templates.items():
            noises=[('0',0.0)]
            if scheme=='baseline' and seed==0:noises += [('1em6',1e-6),('1em5',1e-5),('1em4',1e-4),('3em4',3e-4),('1em3',1e-3)]
            for tag,eta in noises:
                c=copy.deepcopy(template);arm=f'conv1_{scheme}_relative1_eta{tag}_seed{seed}'
                c.update(study_id=STUDY.name,arm_id=arm,seed=seed,init_checkpoint_path=f'inputs/initializers/conv1_seed{seed}.pt')
                c['initialization'].update(seed=seed,checkpoint_sha256=sha(checkpoint),recipe='wide_kaiming',optimizer_state='fresh_at_initialization')
                c['lab']['epochs']=10;c['datasets']['mnist']['params'].update(shuffle_seed=seed,split_seed=0)
                c['eqprop'].update(endpoint_read_noise_eta=eta,endpoint_read_noise_std=0.0,endpoint_read_noise_mode='nodewise_relative',endpoint_read_noise_seed=2026081601)
                c['relative_output_calibration'].update(calibrated_for_this_seed=(seed==0),actual_displacement_for_this_seed='accepted_seed0_reference' if seed==0 else 'unmeasured_may_differ')
                c['relative_noise_training'].update(eta=eta,primary_metric='epoch_10_validation_accuracy',calibration_reused=(seed==0),reference_displacement_seed=0,fixed_seed0_beta=True)
                c['reporting'].update(study_id=STUDY.name,arm_id=arm)
                c['evaluation']['inclusion_rule']='Retain all outcomes; compare fixed epoch10 validation accuracy with paired clean seeds or seed0 noisy controls.'
                c['conv1_control_contract']=dict(model_seed=seed,shuffle_seed=seed,split_seed=0,noise_eta=eta,trained_from_initialization=True,uninterrupted_epochs=10,optimizer_reset_at_epoch10=False,rng_reset_at_epoch10=False,displacement_label='seed0-reference D/F=1',scheme_specific_learning_rates=True,scheme_source_proof=proofs[scheme],comparison_limitation='Existing Conv1 ours/legacy noisy seed0 results use the same10epoch horizon.')
                assert c['lr']==c['optimizer']['learning_rate']==proofs[scheme]['lr']
                assert c['evaluation']['official_test']['policy']=='disabled'
                dest=CONFIGS/(arm+'.json');dest.write_text(json.dumps(c,indent=2,sort_keys=True)+'\n')
                records.append(dict(arm_id=arm,config=str(dest),config_sha256=sha(dest),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),staged_checkpoint=c['init_checkpoint_path'],base_beta=c['beta'],eta=eta,seed=seed,epochs=10))
    assert len(set(hashes))==3 and len(records)==14
    (STUDY/'preparation.json').write_text(json.dumps(dict(expected_cases=14,ready_cases=14,assets=assets,proofs=proofs,records=records),indent=2,sort_keys=True)+'\n')
    snapshot=STUDY/'source-training-v1'
    if snapshot.exists():
        for r in records:
            for src,rel in [(Path(r['checkpoint']),Path(r['staged_checkpoint'])),(Path(r['config']),Path(r['config']))]:
                out=snapshot/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
    print(json.dumps(dict(ready_cases=14,staged=snapshot.exists(),independent_seeds=3)))


if __name__=='__main__': main()
