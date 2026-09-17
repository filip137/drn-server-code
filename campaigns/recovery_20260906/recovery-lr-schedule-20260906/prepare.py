import copy
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
DRN=BASE/'ibm-om-cell-aware-quantized'
ROOT=DRN/'simulation_results/exploratory_noncanonical_figure6_om_784_256_10_postpv_fault'
CB=HERE/'crossbar_code'
SID='mnist-ibm-om-crossbar-long-kl-schedules-20260906-v1'

def single(pattern):
    paths=list(ROOT.glob(pattern))
    assert len(paths)==1,(pattern,paths)
    return str(paths[0].resolve())

def schedules(rates,decays):
    result=[]
    for rate in rates:
        result.append(dict(name=f'constant-{rate:.0e}'.replace('-0','-'), learning_rate=rate, schedule=dict(kind='constant',final_factor=1.0,decay_epochs=10),epochs=30))
    for rate in decays:
        result.append(dict(name=f'decay-{rate:.0e}'.replace('-0','-'),learning_rate=rate,schedule=dict(kind='exponential',final_factor=0.01,decay_epochs=10),epochs=30))
    return result

def arms(grid):
    return [dict(**item, condition=condition, arm_id=condition+'-'+item['name']) for item in grid for condition in ['healthy','faulted']]

drn_arms=arms(schedules([3e-6,1e-5,3e-5,1e-4],[1e-5,3e-5]))
crossbar_arms=arms(schedules([1e-6,3e-6,1e-5,3e-5],[3e-6,1e-5]))
plan=dict(evidence_class='exploratory_noncanonical',question='Can longer lower-rate or decaying pulse-Adam recovery improve teacher KL beyond previously accuracy-selected one-epoch states?',screen_epochs=30,selection='minimum held-apparent validation teacher KL, then higher accuracy, then earlier checkpoint; include P0',confirmation='Freeze the best constant and best decaying schedule per architecture and condition; evaluate on two additional saved arrays, one write per array. Report per-array results.',extension='If a screen winner is still improving near the 30-epoch boundary (best at epoch >=25 and >=1% KL improvement over the preceding five-epoch window), extend that schedule to 60 epochs with the same decay floor; report any unresolved boundary.',data='Same original teacher, P0, batch size 16, 55000 examples/epoch, full 5000 validation examples; test remains closed during screening.',drn=dict(source_revision=subprocess.check_output(['git','-C',str(DRN),'rev-parse','HEAD'],text=True).strip(),original_config=str(DRN/'examples/mnist_relu_drn/figure6_om_784_256_10_postpv_fault/full.json'),prepared=single('stages/prepare/main/*/artifacts/prepared.pt'),arms=drn_arms,replicas=[dict(id=f'array-{a}-write-1',deployment=single(f'stages/deploy/array_{a}__write_1/*/checkpoints/deployment.pt'),repaired_population=single(f'stages/prepare/main/*/artifacts/om_heldout_array_{a}_counterfactual_repaired.npz')) for a in [1,2,3]]),crossbar=dict(source_revision='207a8f6b25105b6ceaee85f00bf5f5826b9821b2 plus recorded recovery extension',study_id=SID,arms=crossbar_arms,input_dir='/tmp/ibm-om-crossbar-hwa-canary-20260905/data/mnist_ibm_om_crossbar_hwa_canary_1f4ef1b8',screen_assignment=2090402,screen_endpoint=2091402))
(HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
template=json.loads((CB/'examples/mnist_analog_relu/ibm_om_crossbar_hwa_recovery_canary/adam-faulted-kl-lr3em6.json').read_text())
cfgdir=CB/'examples/mnist_analog_relu/long_kl_schedules_20260906'
cfgdir.mkdir(exist_ok=True)
study=json.loads((CB/'studies/mnist-ibm-om-crossbar-hwa-recovery-canary-20260905-v1.json').read_text())
study.update(study_id=SID,title='Thirty-epoch low-rate and decaying KL recovery from exact HWA deployments',hypothesis=plan['question'],motivation='Prior low-rate crossbar runs still improved at epoch 3, while DRN rates had been screened over only 8192 examples.',evidence_class='exploratory_model_based_exact_p0_schedule_screen',arms=[],completion_criteria=['All 12 declared schedule/condition arms complete 30 full epochs on CUDA.','Every selected checkpoint replays minimum held-apparent validation KL including P0.','Preserve the two architecture/device models; make no causal cross-architecture claim.'],analysis_plan=[plan['selection'],plan['confirmation'],plan['extension'],'Plot KL versus epochs and cumulative physical pulses; report accuracy and persistent-state diagnostics separately.'])
for arm in crossbar_arms:
    cfg=copy.deepcopy(template)
    cfg['stage'].update(epochs=arm['epochs'],start_state='hwa_healthy_p0' if arm['condition']=='healthy' else 'hwa_published_fault',checkpoint_policy='best_held_apparent_validation_objective_then_accuracy_then_earlier_epoch_including_epoch0',learning_rate_schedule=arm['schedule'])
    cfg['stage']['hyperparameters']['learning_rate']=arm['learning_rate']
    path=cfgdir/(arm['arm_id']+'.json')
    path.write_text(json.dumps(cfg,indent=2)+'\n')
    study['arms'].append(dict(arm_id=arm['arm_id'],configs=['../'+str(path.relative_to(CB))],description='Thirty full epochs of teacher-KL recovery: '+arm['name'],experiment_id='mnist_ibm_om_crossbar_relu.v2',mode='train'))
canary=copy.deepcopy(template)
canary['stage'].update(epochs=2,checkpoint_policy='best_held_apparent_validation_objective_then_accuracy_then_earlier_epoch_including_epoch0',learning_rate_schedule=dict(kind='exponential',final_factor=0.1,decay_epochs=2))
(cfgdir/'canary.json').write_text(json.dumps(canary,indent=2)+'\n')
(CB/'studies'/f'{SID}.json').write_text(json.dumps(study,indent=2)+'\n')
print(HERE/'plan.json')
