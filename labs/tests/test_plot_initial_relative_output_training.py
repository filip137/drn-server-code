"""Protect fixed endpoint contrasts, missing coverage and transport identity."""
import copy
import json

import pytest

from experiments.plot_initial_relative_output_training import (
    ARCHITECTURES, SCHEMES, TARGETS, apply_cancellations, collect, contrasts, discover, epoch_history,
    placement, scientific_config, validate_v100_wrappers,
)


def complete_rows():
    return [dict(architecture=a,scheme=s,target=t,state='complete',
        final_validation_percent={'baseline':91.,'ours':94.,'legacy':92.}[s]+t,
        validation_correct=round(({'baseline':91.,'ours':94.,'legacy':92.}[s]+t)*50),
        placement_group='host-'+a,gpu_class='class-'+a,initial_state_sha256='init-'+a,
        dataset_provenance={'orders':list(range(10))},source_archive_sha256='source')
        for a in ARCHITECTURES for s in SCHEMES for t in TARGETS]


def test_fixed_pairs_do_not_choose_best_targets_or_impute_missing_arm():
    rows=complete_rows()
    # A large baseline endpoint must never enter the paired comparison.
    rows[0]['final_validation_percent']=99.9
    missing=next(r for r in rows if (r['architecture'],r['scheme'],r['target'])==('conv1','legacy',2.))
    missing.update(state='failed',final_validation_percent=None)
    pairs,means=contrasts(rows)
    assert len(pairs)==12 and sum(p['state']=='complete' for p in pairs)==11
    assert all(p['ours_minus_legacy_pp']==pytest.approx(2) for p in pairs if p['state']=='complete')
    assert means[0]['four_target_mean_pp'] is None
    assert means[0]['complete_pairs']==3
    assert 'available_complete_pair_mean_pp' not in means[0]
    assert means[1]['four_target_mean_pp']==pytest.approx(2)


def test_host_or_data_order_mismatch_blocks_pair():
    rows=complete_rows()
    ours=next(r for r in rows if (r['architecture'],r['scheme'],r['target'])==('conv2','ours',.8))
    ours['placement_group']='other-host'
    pairs,means=contrasts(rows)
    pair=next(p for p in pairs if (p['architecture'],p['target'])==('conv2',.8))
    assert pair['state']=='mismatched' and pair['ours_minus_legacy_pp'] is None
    assert means[1]['four_target_mean_pp'] is None


def test_exact_four_target_tie_is_not_a_floating_point_advantage():
    rows=complete_rows()
    for target,difference in zip(TARGETS,(5,-3,-2,0)):
        ours=next(r for r in rows if (r['architecture'],r['scheme'],r['target'])==('conv1','ours',target))
        legacy=next(r for r in rows if (r['architecture'],r['scheme'],r['target'])==('conv1','legacy',target))
        ours['validation_correct']=legacy['validation_correct']+difference
        ours['final_validation_percent']=ours['validation_correct']/50
    pairs,means=contrasts(rows)
    assert [p['ours_minus_legacy_pp'] for p in pairs[:4]]==[.1,-.06,-.04,0.]
    assert means[0]['four_target_mean_pp']==0.


def test_only_paths_are_ignored_in_config_comparison():
    c={'init_checkpoint_path':'original.pt','lab':{'dataset_key':'mnist'},
       'datasets':{'mnist':{'params':{'root':'/data','shuffle_seed':0}}},'beta':.1}
    transport=copy.deepcopy(c)
    transport['init_checkpoint_path']='assets/init.pt'
    transport['datasets']['mnist']['params']['root']='/remote/data'
    assert scientific_config(c)==scientific_config(transport)
    transport['beta']=.2
    assert scientific_config(c)!=scientific_config(transport)


def test_smoke_is_excluded_and_failed_attempt_is_retained(tmp_path):
    for name,smoke,state in [('smoke',True,'complete'),('production',False,'failed')]:
        run=tmp_path/name
        run.mkdir()
        (run/'manifest.json').write_text(json.dumps({'arm_id':'arm','smoke':smoke}))
        (run/'status.json').write_text(json.dumps({'state':state,'error':'numerical failure'}))
    grouped,attempts,ignored=discover([tmp_path],{'arm'})
    assert ignored==1 and len(attempts)==1 and grouped['arm'][0]['state']=='failed'


def test_missing_or_duplicate_epoch_cannot_count_as_ten_epoch_result(tmp_path):
    rows=[dict(kind='epoch',epoch=i,evaluation_split='validation',
               metrics=dict(validation_accuracy=.9,validation_loss=.1,train_accuracy=.9,train_loss=.1))
          for i in range(1,11)]
    path=tmp_path/'metrics.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    assert len(epoch_history(tmp_path))==10
    rows[-1]['epoch']=9
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='exactly epochs'):
        epoch_history(tmp_path)


def test_distinct_failed_and_successful_attempts_are_not_silently_selected(tmp_path):
    c={'arm_id':'arm','eqprop':{'injected_beta_B':1.},'beta':1.,
       'relative_output_calibration':{'measured':{'relative_output_rms':.8}}}
    attempts=[dict(run=str(tmp_path/name),state=state,identity=name,excluded=False)
              for name,state in [('old','failed'),('retry','complete')]]
    rows=collect({('conv1','ours',.8):(tmp_path/'config.json',c)}, {'arm':attempts},'archive')
    assert rows[0]['state']=='ambiguous'
    assert rows[0]['final_validation_percent'] is None
    assert len(rows[0]['runs'])==2


def test_user_cancellation_preserves_completed_and_observed_failure_evidence():
    rows=[dict(arm_id='finished',state='complete',issue=''),
          dict(arm_id='stopped',state='failed',issue='SIGTERM'),
          dict(arm_id='unobserved',state='not_collected',issue='')]
    apply_cancellations(rows,['finished','stopped','unobserved'])
    assert rows[0]['state']=='complete'
    assert rows[1]['state']=='cancelled_by_user' and rows[1]['observed_bundle_state']=='failed'
    assert 'SIGTERM' in rows[1]['issue']
    assert rows[2]['state']=='cancelled_by_user' and rows[2]['observed_bundle_state']=='not_collected'


def test_declared_v100_recovery_alias_requires_slurm_and_its_own_wrapper(tmp_path):
    manifest=dict(study_id=tmp_path.name,runtime=dict(target='jean-zay-task6-recovery',
                  host='r10i4n3',device='cuda',slurm_job_id='159309'))
    assert placement(manifest)[2]=='Jean_Zay_V100'
    transport=tmp_path/'transport';transport.mkdir()
    (transport/'conv3.slurm').write_text('#SBATCH --constraint=v100-16g\n')
    recovery=transport/'conv3-task6-recovery.slurm'
    recovery.write_text('#SBATCH --constraint=a100\n')
    with pytest.raises(ValueError,match='V100 execution contract'):
        validate_v100_wrappers(tmp_path,[{'manifest':manifest}])
    recovery.write_text('#SBATCH --constraint=v100-16g\n')
    validate_v100_wrappers(tmp_path,[{'manifest':manifest}])
    del manifest['runtime']['slurm_job_id']
    with pytest.raises(ValueError,match='Unrecognized'):
        placement(manifest)
