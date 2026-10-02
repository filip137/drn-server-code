"""Prepare baseline noisy runs: uninterrupted 30 epochs, as explicitly requested."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil

STUDY=Path('results/conv23-baseline-relative-noise-20260927-v1')
CONFIGS=Path('configs/conv/baseline_relative_noise_20260927')
PARENT=Path('results/conv23-noiseless-baseline-three-seed-20260927-v1')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    STUDY.mkdir(parents=True,exist_ok=True);CONFIGS.mkdir(parents=True,exist_ok=True)
    records=[]
    for depth,target in [(2,4),(3,6)]:
        source=Path(f'configs/conv/noiseless_baseline_three_seed_20260927/conv{depth}_baseline_relative{target}_eta0_seed0.json')
        template=json.loads(source.read_text())
        checkpoint=PARENT/'initializers'/f'conv{depth}'/'seed0.pt'
        assert sha(checkpoint)==template['initialization']['checkpoint_sha256']
        copy_path=STUDY/'initializers'/f'conv{depth}'/'seed0.pt';copy_path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(checkpoint,copy_path)
        for tag,eta in [('1em6',1e-6),('1em5',1e-5),('1em4',1e-4),('3em4',3e-4),('1em3',1e-3)]:
            c=copy.deepcopy(template)
            arm=f'conv{depth}_baseline_relative{target}_eta{tag}_seed0'
            c.update(study_id=STUDY.name,arm_id=arm)
            c['reporting'].update(study_id=STUDY.name,arm_id=arm)
            c['eqprop']['endpoint_read_noise_eta']=eta
            c['relative_noise_training'].update(eta=eta,primary_metric='epoch_30_validation_accuracy')
            c.pop('noiseless_replication')
            c['noisy_baseline_comparison']=dict(source_baseline_template=str(source),source_baseline_template_sha256=sha(source),model_seed=0,shuffle_seed=0,split_seed=0,noise_eta=eta,trained_from_initialization=True,uninterrupted_epochs=30,optimizer_reset_at_epoch10=False,rng_reset_at_epoch10=False,user_requested_uninterrupted_training=True,comparison_limitation='Ours/legacy comparison runs restart Adam and RNG after epoch10; these baseline runs use uninterrupted Adam/RNG for30epochs.',matching='Same seed0 initializer, data split/shuffle seed, read-noise mode and seed, iteration counts, initial relative output displacement target; scheme-specific frozen learning rates and beta retained.')
            c['evaluation']['inclusion_rule']='Retain all outcomes; compare epoch30 validation accuracy with explicit optimizer-history caveat against restarted ours/legacy.'
            assert c['lab']['epochs']==30 and c['seed']==0
            assert c['eqprop']['endpoint_read_noise_seed']==2026081601
            assert c['eqprop']['endpoint_read_noise_std']==0
            assert c['eqprop']['endpoint_read_noise_mode']=='nodewise_relative'
            assert c['lr']==template['lr']==c['optimizer']['learning_rate']
            assert c['beta']==template['beta']
            assert c['evaluation']['official_test']['policy']=='disabled'
            dest=CONFIGS/(arm+'.json');dest.write_text(json.dumps(c,indent=2,sort_keys=True)+'\n')
            records.append(dict(arm_id=arm,config=str(dest),config_sha256=sha(dest),checkpoint=str(copy_path),checkpoint_sha256=sha(copy_path),staged_checkpoint=c['init_checkpoint_path'],base_beta=c['beta'],eta=eta,epochs=30))
    report=dict(expected_cases=10,ready_cases=len(records),epochs=30,optimizer_reset_at_epoch10=False,records=records)
    (STUDY/'preparation.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    snapshot=STUDY/'source-training-v1'
    if snapshot.exists():
        for r in records:
            for src,rel in [(Path(r['checkpoint']),Path(r['staged_checkpoint'])),(Path(r['config']),Path(r['config']))]:
                out=snapshot/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
    print(json.dumps(dict(ready_cases=len(records),staged=snapshot.exists(),epochs=30)))


if __name__=='__main__': main()
