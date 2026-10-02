"""Add baseline clean controls using its own frozen optimizer and accepted beta."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil

STUDY=Path('results/conv23-noiseless-baseline-three-seed-20260927-v1')
CONFIGS=Path('configs/conv/noiseless_baseline_three_seed_20260927')
PARENT=Path('results/conv23-noiseless-three-seed-20260927-v1')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    STUDY.mkdir(parents=True,exist_ok=True);CONFIGS.mkdir(parents=True,exist_ok=True)
    records=[];proofs=[]
    for depth,target in [(2,4),(3,6)]:
        replay_path=Path(f'results/conv123-relative-noise-advantage-map-20260925-v1/stage_a/conv{depth}/config.used.json')
        replay=json.loads(replay_path.read_text())
        case=next(c for c in replay['cases'] if c['scheme']=='baseline' and c['relative_target']==target)
        source=Path(case['source_config']);assert sha(source)==case['source_config_sha256']
        b=json.loads(source.read_text())
        match_path=replay_path.parent/'cases'/f'conv{depth}_baseline_relative{target}'/'matching.json'
        match=json.loads(match_path.read_text())
        assert match['matched'] and match['relative_target']==target and match['relative_error']<=.01
        assert match['matching_examples']==576 and match['cohort_sha256']==replay['expected_cohort_sha256']
        assert b['model_base']['voltage_amp']==b['model_base']['current_amp']==1
        assert b['model_base']['num_iterations_inference']==case['T']==(6 if depth==2 else 8)
        assert b['model_base']['num_iterations_training']==case['K']
        proof=dict(source_config=str(source),source_config_sha256=sha(source),matching_path=str(match_path),matching_sha256=sha(match_path),replay_config=str(replay_path),replay_sha256=sha(replay_path),initializer_checkpoint_sha256=case['initializer_checkpoint_sha256'],base_beta=match['base_beta'],matched_relative_output=match['actual_relative_output_rms'],T=case['T'],K=case['K'],lr=b['lr'])
        proofs.append(proof)
        for seed in [0,1,2]:
            template=Path(f'configs/conv/noiseless_three_seed_20260927/conv{depth}_ours_relative{target}_eta0_seed{seed}.json')
            c=json.loads(template.read_text())
            compare=copy.deepcopy(c['model_base']);compare['voltage_amp']=1.0
            assert compare==b['model_base'] and c['model_overrides']==b['model_overrides']
            assert c['parameter_order']==b['parameter_order']
            c['baseline_source_proof']=proof
            for key in ['model_base','model_overrides','optimizer','lr','learning_rates_by_parameter','handoff']:
                c[key]=copy.deepcopy(b[key])
            arm=f'conv{depth}_baseline_relative{target}_eta0_seed{seed}'
            c.update(arm_id=arm,study_id=STUDY.name,beta=match['base_beta'])
            c['reporting'].update(arm_id=arm,study_id=STUDY.name)
            c['eqprop'].update(amplification_factor=1.0,injected_beta_B=match['injected_beta'])
            checkpoint=PARENT/'initializers'/f'conv{depth}'/f'seed{seed}.pt'
            assert sha(checkpoint)==c['initialization']['checkpoint_sha256']
            if seed==0: assert sha(checkpoint)==case['initializer_checkpoint_sha256']
            out=STUDY/'initializers'/f'conv{depth}'/f'seed{seed}.pt';out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(checkpoint,out)
            ref=dict(target=float(target),T=case['T'],K=case['K'],base_beta=match['base_beta'],injected_beta=match['injected_beta'],definition=c['relative_output_calibration']['definition'],measured=match,matching_proof=proof)
            c['relative_output_calibration'].update(base_beta=match['base_beta'],injected_beta=match['injected_beta'],seed0_reference=ref)
            assert c['lr']==c['optimizer']['learning_rate']==b['lr']
            assert all(x==0 for k,x in c['learning_rates_by_parameter'].items() if k.startswith('Bias_'))
            dest=CONFIGS/(arm+'.json');dest.write_text(json.dumps(c,indent=2,sort_keys=True)+'\n')
            records.append(dict(arm_id=arm,config=str(dest),config_sha256=sha(dest),checkpoint=str(out),checkpoint_sha256=sha(out),staged_checkpoint=c['init_checkpoint_path'],base_beta=c['beta'],seed=seed))
    report=dict(expected_cases=6,ready_cases=len(records),epochs=30,eta=0,proofs=proofs,records=records)
    (STUDY/'preparation.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    snapshot=STUDY/'source-training-v1'
    if snapshot.exists():
        for r in records:
            for src,rel in [(Path(r['checkpoint']),Path(r['staged_checkpoint'])),(Path(r['config']),Path(r['config']))]:
                out=snapshot/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
    print(json.dumps(dict(ready_cases=len(records),staged=snapshot.exists(),betas=[p['base_beta'] for p in proofs])))


if __name__=='__main__': main()
