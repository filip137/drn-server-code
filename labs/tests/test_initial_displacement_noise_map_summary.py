import pytest

from experiments.plot_initial_displacement_noise_map import summarize, select_candidates, TARGETS, SIGMAS


def sample(batch,draw=0,cosine=.5):
    return dict(architecture='conv1',scheme='ours',relative_target=.2,readout_sigma=.0001,
                parameter_name='ConvWeight_0',batch_index=batch,noise_draw_index=draw,
                noise_seed=(91000000,92000000,93000000)[draw],cosine=cosine)


def test_partial_and_undefined_do_not_create_complete_mean():
    rows=[sample(b,d) for b in range(36) for d in range(3)]
    rows[0]['cosine']=''
    means,_,_=summarize(rows)
    row=next(r for r in means if r['observed_samples'])
    assert not row['complete'] and row['cosine_mean'] is None
    assert row['defined_samples']==107 and row['observed_defined_mean']==.5


def test_duplicate_and_wrong_rng_are_rejected():
    with pytest.raises(ValueError,match='Duplicate'):summarize([sample(0),sample(0)])
    row=sample(0);row['noise_seed']=92000000
    with pytest.raises(ValueError,match='Noise seed'):summarize([row])


def test_worst_layer_is_minimum_of_layer_means_not_mean_of_minima():
    rows=[]
    for p in ('ConvWeight_0','DenseWeight_0'):
        for b in range(36):
            for d in range(3):
                r=sample(b,d,1 if (b%2==0)==(p=='ConvWeight_0') else 0)
                r['parameter_name']=p;rows.append(r)
    _,_,worst=summarize(rows)
    row=next(r for r in worst if r['complete'])
    assert row['worst_layer_cosine']==.5


def test_isolated_positive_and_clean_only_do_not_select_region():
    w=[]
    for s in ('ours','legacy'):
        for r in TARGETS:
            for sigma in SIGMAS:
                value=.8 if s=='legacy' else (.9 if (r==.2 and sigma==.0001) or sigma==0 else .7)
                w.append(dict(architecture='conv1',scheme=s,relative_target=r,readout_sigma=sigma,
                              complete=True,worst_layer_cosine=value))
    assert select_candidates(w)==[]
    next(x for x in w if x['scheme']=='ours' and x['relative_target']==.8 and x['readout_sigma']==.0001)['worst_layer_cosine']=.95
    result=select_candidates(w)
    assert len(result)==1 and result[0]['cells']==[[.2,.0001],[.8,.0001]]
    assert result[0]['minimum_margin']==pytest.approx(.1)


def test_physical_normalization_pools_squares_not_ratios():
    from experiments.plot_initial_displacement_noise_map import state_summary
    rows=[dict(architecture='conv1',scheme='ours',relative_target=2,state_layer_index=0,
               state_layer_name='hidden',batch_index=i,element_count=1 if i%2==0 else 3,
               free_rms=1,centered_half_difference_rms=1 if i%2==0 else 3) for i in range(36)]
    value=state_summary(rows)[0]
    assert value['centered_displacement_rms']==pytest.approx(7**.5)
    assert value['measured_relative_displacement']==pytest.approx(7**.5)
    assert value['complete'] and value['output_match_passed'] is None


def test_failed_attempt_without_metrics_is_retained(tmp_path):
    import json
    from experiments.plot_initial_displacement_noise_map import collect
    run=tmp_path/'failed';run.mkdir()
    (run/'manifest.json').write_text(json.dumps({'study_id':'conv123-initial-displacement-noise-map-20260925-v1','smoke':False}))
    (run/'status.json').write_text(json.dumps({'state':'failed','error':'fixture'}))
    rows,states,attempts=collect([tmp_path])
    assert not rows and not states
    assert len(attempts)==1 and attempts[0]['state']=='failed' and not attempts[0]['has_layer_metrics']


def test_output_counts_and_each_state_count_are_checked():
    from experiments.plot_initial_displacement_noise_map import state_summary
    row=dict(architecture='conv1',scheme='ours',relative_target=2,state_layer_index=1,
             state_layer_name='output',batch_index=0,element_count=319,free_rms=1,centered_half_difference_rms=2)
    with pytest.raises(ValueError,match='16×20'):state_summary([row])
    row.update(state_layer_index=0,element_count=-1)
    with pytest.raises(ValueError,match='count'):state_summary([row])


def test_draw_means_preserve_variation_and_all_layer_dominance_is_separate():
    from experiments.plot_initial_displacement_noise_map import per_draw_summaries,dominance
    rows=[sample(b,d,.1*d) for b in range(36) for d in range(3)]
    means,worst=per_draw_summaries(rows)
    assert [r['cosine_mean'] for r in means]==pytest.approx([0,.1,.2])
    assert all(not r['complete'] for r in worst)  # Readout is absent.
    common=dict(architecture='conv1',scheme='ours',relative_target=.2,readout_sigma=.0001)
    result=dominance([dict(common,parameter_name='ConvWeight_0',cosine_delta=.2),
                      dict(common,parameter_name='DenseWeight_0',cosine_delta=-.1)])
    assert result[0]['complete'] and not result[0]['all_layer_dominance']
