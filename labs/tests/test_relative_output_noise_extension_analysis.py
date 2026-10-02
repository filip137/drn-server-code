"""Protect noise separation, fixed-target means and per-study source identities."""
from pathlib import Path

import pytest

from experiments import plot_initial_relative_output_training as original
from experiments.plot_relative_output_noise_extension import (
    NEW_ID, OLD_ID, TARGETS_BY_ARCH, active_keys, combined_configs, contrasts,
)


def rows():
    return [dict(architecture=a,scheme=s,target=t,sigma=n,state='complete',
                 validation_correct=4700+(10 if s=='ours' else 0),
                 placement_group='host',gpu_class='gpu',initial_state_sha256='init',
                 dataset_provenance={'orders':list(range(10))},source_archive_sha256='source')
            for a,s,t,n in sorted(active_keys())]


def test_noise_cells_stay_separate_and_one_missing_pair_blocks_only_its_mean():
    data=rows()
    missing=next(r for r in data if (r['architecture'],r['scheme'],r['target'],r['sigma'])==('conv2','legacy',6.,.002))
    missing['state']='failed'
    pairs,means=contrasts(data)
    assert len(pairs)==45 and len(means)==9
    assert sum(p['state']=='complete' for p in pairs)==44
    assert [m['expected_pairs'] for m in means]==[7,7,7,4,4,4,4,4,4]
    incomplete=next(m for m in means if (m['architecture'],m['sigma'])==('conv2',.002))
    assert incomplete['fixed_target_mean_pp'] is None and incomplete['complete_pairs']==3
    assert all(m['fixed_target_mean_pp']==pytest.approx(.2) for m in means if m!=incomplete)


def test_seven_target_exact_tie_uses_integer_validation_counts():
    data=rows()
    for target,delta in zip(TARGETS_BY_ARCH['conv1'],(5,-3,-2,0,0,0,0)):
        ours=next(r for r in data if (r['architecture'],r['scheme'],r['target'],r['sigma'])==('conv1','ours',target,.001))
        ours['validation_correct']=4700+delta
    _,means=contrasts(data)
    assert next(m for m in means if (m['architecture'],m['sigma'])==('conv1',.001))['fixed_target_mean_pp']==0.


def test_active_configs_do_not_repeat_inherited_or_include_pruned_targets():
    root=Path(__file__).resolve().parents[2]
    configs=combined_configs(root/'configs/conv/eqprop_conv123_relative_output_noise_extension_20260924',
                             root/'configs/conv/eqprop_conv123_relative_output_20260924')
    assert len(configs)==153
    assert sum(key in active_keys() for key in configs)==135
    assert sum(c['study_id']==NEW_ID for _,c in configs.values())==117
    assert sum(c['study_id']==OLD_ID and key in active_keys() for key,(_,c) in configs.items())==18


def test_collect_routes_source_identity_and_compares_hosts_within_noise(monkeypatch,tmp_path):
    configs={};attempts={};seen=[]
    for sigma,study,host in ((.0005,OLD_ID,'old-host'),(.001,NEW_ID,'new-host')):
        for scheme in ('ours','legacy'):
            arm=f'{scheme}-{sigma}'
            config=dict(arm_id=arm,study_id=study,eqprop={'injected_beta_B':1.},beta=1.,
                        initialization={'checkpoint_sha256':'init'},
                        relative_output_calibration={'measured':{'relative_output_rms':2.}})
            configs[('conv1',scheme,2.,sigma)]=(tmp_path/(arm+'.json'),config)
            attempts[arm]=[dict(run=str(tmp_path/host/arm),state='complete',identity=arm,excluded=False)]
    monkeypatch.setattr(original,'initializer_identity',lambda c:'init')
    def validate(run,config,archive,initial,cohort):
        seen.append((config['study_id'],archive))
        return dict(placement_group=run.parent.name,gpu_class='gpu',dataset_provenance={'orders':[1]},
                    initial_state_sha256=initial,source_archive_sha256=archive)
    monkeypatch.setattr(original,'validate_complete',validate)
    collected=original.collect(configs,attempts,{OLD_ID:'old-archive',NEW_ID:'new-archive'},'cohort')
    assert all(r['state']=='complete' for r in collected)
    assert set(seen)=={(OLD_ID,'old-archive'),(NEW_ID,'new-archive')}
    attempts['ours-0.001'][0]['run']=str(tmp_path/'wrong-host'/'ours-0.001')
    collected=original.collect(configs,attempts,{OLD_ID:'old-archive',NEW_ID:'new-archive'},'cohort')
    assert all(r['state']=='invalid' for r in collected if r['sigma']==.001)
    assert all(r['state']=='complete' for r in collected if r['sigma']==.0005)
