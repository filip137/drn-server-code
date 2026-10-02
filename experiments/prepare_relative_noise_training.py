"""Emit exp014 exact training configs from inherited training and measured beta files."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STUDY='conv123-relative-noise-training-20260925-v1'
ETAS=[1e-8,1e-7,1e-6,1e-5,1e-4,3e-4,1e-3]
TARGETS={1:[1.],2:[1.,2.,4.],3:[4.,6.]}
COHORT='95af3eebd9416ec643b7d698d061b7dac3fe0d52311dd888293609bada629acd'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(path):return str(path.resolve().relative_to(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'configs/conv/relative_noise_training_20260925')
    parser.add_argument('--calibration-root',type=Path,default=ROOT/'results/conv123-relative-noise-advantage-map-20260925-v1/stage_a')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    records=[];proofs=[]
    for depth,targets in TARGETS.items():
        replay_path=args.calibration_root/f'conv{depth}'/'config.used.json'
        replay=json.loads(replay_path.read_text())
        assert replay['expected_cohort_sha256']==COHORT and replay['expected_batches']==36
        for scheme in ('ours','legacy'):
            template_path=ROOT/f'configs/conv/eqprop_conv123_relative_output_20260924/conv{depth}_{scheme}_relative2_sigma5em4_seed0.json'
            template=json.loads(template_path.read_text())
            for target in targets:
                name=f'conv{depth}_{scheme}_relative{target:g}'
                case=next(c for c in replay['cases'] if c['name']==name)
                matching_path=args.calibration_root/f'conv{depth}'/'cases'/name/'matching.json'
                measured=json.loads(matching_path.read_text())
                assert measured['matched'] and measured['relative_error']<=.01
                assert measured['relative_target']==target and measured['matching_examples']==576 and measured['cohort_sha256']==COHORT
                source_path=ROOT/case['source_config'];source=json.loads(source_path.read_text())
                assert digest(source_path)==case['source_config_sha256']
                assert template['relative_output_calibration']['source_config_sha256']==case['source_config_sha256']
                for key in ('model_base','model_overrides','optimizer','parameter_order','learning_rates_by_parameter','lr'):
                    assert template[key]==source[key],f'Source training contract differs: {name}/{key}'
                assert template['initialization']['checkpoint_sha256']==case['initializer_checkpoint_sha256']
                model=template['model_base'];T,gain={1:(4,40),2:(6,100),3:(8,360)}[depth]
                assert model['num_iterations_inference']==model['num_iterations_training']==case['T']==case['K']==T
                assert model['input_gain']==gain and model['weight_min']==0 and model['weight_max']==100
                assert template['runtime_dtype']=='float64' and template['seed']==0
                assert template['lab']['epochs']==10 and template['batch_state_policy']=='reset_each_batch'
                dataset=template['datasets']['mnist']['params']
                assert dataset['batch_size']==16 and dataset['validation_batch_size']==64 and dataset['split_seed']==dataset['shuffle_seed']==0
                assert all(rate==0 for param,rate in template['learning_rates_by_parameter'].items() if param.startswith('Bias_'))
                amplification=(float(model['voltage_amp'])/float(model['current_amp']))**depth
                assert abs(measured['base_beta']-measured['injected_beta']/amplification)<=abs(measured['base_beta'])*1e-14
                proof=dict(architecture=f'conv{depth}',scheme=scheme,target=target,
                    matching_path=relative(matching_path),matching_sha256=digest(matching_path),
                    replay_config=relative(replay_path),replay_config_sha256=digest(replay_path),
                    training_template=relative(template_path),training_template_sha256=digest(template_path),
                    source_config=case['source_config'],source_config_sha256=case['source_config_sha256'],
                    initializer_checkpoint_path=case['initializer_checkpoint_path'],initializer_checkpoint_sha256=case['initializer_checkpoint_sha256'],
                    cohort_sha256=COHORT,measured=measured)
                proofs.append(proof)
                for eta in ETAS:
                    tag=f'{eta:.0e}'.replace('e-0','em').replace('e-','em')
                    arm=f'{name}_eta{tag}_seed0'
                    cfg=copy.deepcopy(template)
                    cfg.update(study_id=STUDY,arm_id=arm,beta=measured['base_beta'])
                    cfg['reporting'].update(study_id=STUDY,arm_id=arm)
                    cfg['init_checkpoint_path']=case['initializer_checkpoint_path']
                    cfg['initialization']['checkpoint_sha256']=case['initializer_checkpoint_sha256']
                    cfg['eqprop'].update(endpoint_read_noise_mode='nodewise_relative',endpoint_read_noise_eta=eta,
                        endpoint_read_noise_std=0.,injected_beta_B=measured['injected_beta'],
                        endpoint_read_noise_floor=0.,endpoint_read_noise_units='dimensionless',
                        same_eta_all_noninput_layers=True,validation_read_noise=False,
                        beta_policy='initial_centered_physical_output_relative_rms_fixed_during_training')
                    cfg['relative_output_calibration']=dict(target=target,T=T,K=T,base_beta=measured['base_beta'],
                        injected_beta=measured['injected_beta'],definition='D/F: pooled physical output half-difference RMS / post-T free RMS over 576x20 nodes',
                        measured=measured,source_config=case['source_config'],source_config_sha256=case['source_config_sha256'],
                        matching_proof=proof)
                    cfg['relative_noise_training']=dict(stage='exploratory',relative_target=target,eta=eta,
                        primary_metric='epoch_10_validation_accuracy',beta_frozen=True,
                        calibration_reused=True,tests_and_canaries_skipped_by_user=True)
                    destination=args.output_dir/f'{arm}.json'
                    destination.write_text(json.dumps(cfg,indent=2)+'\n')
                    records.append(dict(path=relative(destination),sha256=digest(destination),architecture=f'conv{depth}',scheme=scheme,
                                        relative_target=target,eta=eta,beta=measured['base_beta'],injected_beta=measured['injected_beta']))
    assert len(records)==84
    proof_dir=args.output_dir/'provenance';proof_dir.mkdir(exist_ok=True)
    (proof_dir/'preparation.json').write_text(json.dumps(dict(study_id=STUDY,cases=records,matching_contexts=proofs,
        counts={'conv1':14,'conv2':42,'conv3':28},training_noise_seed=2026081601),indent=2)+'\n')
    print(json.dumps(dict(configs=len(records),matching_contexts=len(proofs),output_dir=str(args.output_dir))))


if __name__=='__main__':main()
