import copy
import torch
import pytest
from experiments.probe_cifar_l8_bn_adam import ChannelMoments, set_probe_parameters
from experiments.probe_cifar_l8_bn_adam import select_checkpoint_records


def test_checkpoint_selector_exact_subset_and_default_coverage():
    records = [{'id': name} for name in ('baseline_e50', 'ours_e50', 'legacy_e50')]
    assert select_checkpoint_records(records) == records
    assert select_checkpoint_records(records, ['legacy_e50']) == records[2:]
    assert select_checkpoint_records(records, ['legacy_e50', 'baseline_e50']) == [records[0], records[2]]


@pytest.mark.parametrize('records, requested', [
    ([{'id': 'legacy_e50'}], ['legacy_e10']),
    ([{'id': 'legacy_e50'}], []),
    ([], None),
    ([{'id': 'legacy_e50'}], ['legacy_e50', 'legacy_e50']),
])
def test_checkpoint_selector_rejects_unavailable_empty_or_duplicate(records, requested):
    with pytest.raises(ValueError):
        select_checkpoint_records(records, requested)


def test_pooled_unbiased_moments_match_direct_and_are_batch_partition_invariant():
    torch.manual_seed(6)
    x = torch.randn(17, 3, 5, 7, dtype=torch.float64) + 1000
    stats = ChannelMoments()
    for part in x.split(4): stats.add(part)
    mean, variance = stats.finish()
    torch.testing.assert_close(mean, x.mean((0, 2, 3)), atol=1e-10, rtol=1e-12)
    torch.testing.assert_close(variance, x.var((0, 2, 3), unbiased=True), atol=1e-10, rtol=1e-12)
    assert stats.count == 17*5*7


def test_saved_adam_matches_analytic_bias_correction_and_projection_endpoints():
    p=torch.tensor([.01,.4,.99], dtype=torch.float64, requires_grad=True)
    opt=torch.optim.Adam([p],lr=.04,betas=(.8,.9),eps=1e-8,foreach=False,fused=False)
    p.grad=torch.tensor([.2,-.3,.5],dtype=p.dtype);opt.step()
    with torch.no_grad(): p.clamp_(0,1)
    saved=copy.deepcopy(opt.state_dict());before=p.detach().clone()
    gradient=torch.tensor([.7,.4,-.9],dtype=p.dtype)
    state=saved['state'][0]; first=.8*state['exp_avg']+.2*gradient
    second=.9*state['exp_avg_sq']+.1*gradient.square()
    expected=before-.04*(first/(1-.8**2))/((second/(1-.9**2)).sqrt()+1e-8)
    q=before.clone().requires_grad_();other=torch.optim.Adam([q],lr=.04,foreach=False,fused=False)
    other.load_state_dict(copy.deepcopy(saved));q.grad=gradient;other.step()
    torch.testing.assert_close(q,expected,rtol=1e-12,atol=1e-12)
    nominal=q.detach().clamp(0,1)
    class Model:
        def trainable_tensors(self):return {'p':q}
        def project_(self):
            with torch.no_grad():q.clamp_(0,1)
    model=Model()
    for multiplier in (0,.5,1,2):
        set_probe_parameters(model,{'p':before},{'p':nominal},multiplier)
        expected=before if multiplier==0 else nominal if multiplier==1 else (before+multiplier*(nominal-before)).clamp(0,1)
        assert torch.equal(q,expected)
    assert torch.equal(saved['state'][0]['exp_avg'],state['exp_avg'])
