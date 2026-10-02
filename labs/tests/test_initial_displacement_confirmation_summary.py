import pytest
from experiments.plot_initial_displacement_confirmation import summarize,SEEDS


def example_rows(one_failed_draw=False):
    cfg={'conv1':dict(cases=[dict(scheme=s,relative_target=.8) for s in ('baseline','ours','legacy')],
        noise_sigmas=[0.,.001,.002],selected_pair=dict(competitor='ours',cells=[[.8,.001],[.8,.002]]))}
    rows=[]
    for s in ('baseline','ours','legacy'):
        for n in (0.,.001,.002):
            for d in (range(5) if n else (-1,)):
                for b in range(36):
                    for p in ('ConvWeight_0','DenseWeight_0'):
                        value={'baseline':(.1,.8),'ours':(.6,.7),'legacy':(.4,.9)}[s][p=='DenseWeight_0']
                        if one_failed_draw and (s,n,d,p)==('ours',.002,4,'ConvWeight_0'):value=.3
                        rows.append(dict(architecture='conv1',scheme=s,relative_target=.8,readout_sigma=n,
                            noise_draw_index=d,noise_seed=SEEDS[d] if n else None,batch_index=b,parameter_name=p,cosine=value))
    return rows,cfg


def test_every_draw_and_both_cells_must_confirm():
    rows,cfg=example_rows();_,contrasts,_,judgments=summarize(rows,cfg)
    assert judgments[0]['confirmed'] and judgments[0]['positive_cell_draws']==10
    assert not any(x['all_layer_dominance'] for x in contrasts if x['scheme']=='ours')
    rows,cfg=example_rows(one_failed_draw=True);_,_,_,judgments=summarize(rows,cfg)
    assert judgments[0]['positive_cell_draws']==9 and not judgments[0]['confirmed']
    assert judgments[0]['minimum_margin']==pytest.approx(-.1)


def test_undefined_comparison_cannot_confirm_or_be_imputed():
    rows,cfg=example_rows()
    next(r for r in rows if r['scheme']=='ours' and r['readout_sigma']==.001)['cosine']=''
    _,_,_,judgments=summarize(rows,cfg)
    assert not judgments[0]['complete'] and not judgments[0]['confirmed']


def test_duplicate_or_wrong_draw_seed_rejected():
    rows,cfg=example_rows()
    with pytest.raises(AssertionError,match='Duplicate'):summarize(rows+[rows[0]],cfg)
    next(r for r in rows if r['readout_sigma']==.001)['noise_seed']=91000000
    with pytest.raises(AssertionError):summarize(rows,cfg)
