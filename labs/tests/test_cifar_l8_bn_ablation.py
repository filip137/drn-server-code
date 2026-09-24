import copy
import pytest
import torch
from torch.nn import functional as F
from labs.cifar_l8_analog import CifarL8Analog
from labs.tests.test_cifar_l8_amplification import config
from experiments.train_cifar_l8_analog import (
    check_boundary_compatibility, make_optimizer, gradient_norms,
    capture_boundary_statistics, restore_checkpoint,
)


def test_frozen_bn_updates_statistics_once_and_keeps_affine_fixed():
    torch.manual_seed(57)
    c = config(4., .25)
    c['bn_affine_trainable'] = False
    model = CifarL8Analog(c)
    optimizer, _ = make_optimizer(model, c)
    assert not any(g['name'].startswith('bridges.') for g in optimizer.param_groups)
    x, y = torch.randn(4,3,8,8), torch.arange(4)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with capture_boundary_statistics(model) as stats:
            _, out = model.bptt(x, [6,6,4])
        assert len(stats) == 6
        F.cross_entropy(out,y).backward()
        assert gradient_norms(model)
        optimizer.step(); model.project_(); model.detach_state_()
        for bridge in model.bridges:
            bn = bridge[1]
            assert int(bn.num_batches_tracked) == step+1
            assert torch.equal(bn.weight, torch.ones_like(bn.weight))
            assert torch.equal(bn.bias, torch.zeros_like(bn.bias))
            assert bn.weight.grad is None and bn.bias.grad is None
            assert not torch.equal(bn.running_var, torch.ones_like(bn.running_var))


def test_no_bn_imports_only_surviving_initializer_state_and_remains_strict():
    torch.manual_seed(58)
    c = config(1.,1.)
    source = CifarL8Analog(c).snapshot()
    model = CifarL8Analog(dict(c,boundary_normalization='none'))
    removed = model.restore_initializer(source)
    assert len(removed) == 15
    assert not any(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    saved = model.snapshot()
    for kind, values in saved.items():
        for key,value in values.items():
            assert torch.equal(value,source[kind][key])
    with pytest.raises(RuntimeError): model.restore(source)
    invalid = copy.deepcopy(source); invalid['module']['unrelated.weight'] = torch.ones(1)
    with pytest.raises(RuntimeError): model.restore_initializer(invalid)
    opt,_ = make_optimizer(model,c)
    _,out=model.bptt(torch.randn(4,3,8,8),[6,6,4])
    F.cross_entropy(out,torch.arange(4)).backward()
    assert gradient_norms(model)
    opt.step()


def test_default_contract_and_explicit_contract_match():
    c=config(1.,1.)
    explicit=dict(c,boundary_normalization='batch_norm',batch_norm_eps=[1e-5]*3,bn_affine_trainable=True)
    check_boundary_compatibility(c,explicit)
    for field,value in [('bn_affine_trainable',False),('boundary_normalization','none'),('batch_norm_eps',[1e-6]*3)]:
        changed=dict(c,**{field:value})
        with pytest.raises(ValueError,match='normalization'): check_boundary_compatibility(c,changed)
        with pytest.raises(ValueError,match='normalization'):
            restore_checkpoint({'config':c},None,None,None,changed)


def test_epsilon_identity_with_consistent_evaluation_buffers():
    torch.manual_seed(59)
    for scale in [16.,4.]:
        a=torch.nn.BatchNorm2d(3,eps=1e-5).double()
        b=torch.nn.BatchNorm2d(3,eps=1e-5/scale**2).double()
        x=torch.randn(8,3,4,4,dtype=torch.float64)*.002
        torch.testing.assert_close(a(scale*x),b(x),atol=1e-12,rtol=1e-12)
        b.running_mean.copy_(a.running_mean/scale)
        b.running_var.copy_(a.running_var/scale**2)
        a.eval();b.eval()
        torch.testing.assert_close(a(scale*x),b(x),atol=1e-12,rtol=1e-12)


def test_frozen_bn_checkpoint_resume_preserves_next_adam_step(tmp_path):
    from experiments.train_cifar_l8_analog import save_checkpoint
    torch.manual_seed(60)
    c=dict(config(1.,1.),bn_affine_trainable=False,epochs=3)
    a=CifarL8Analog(c);oa,sa=make_optimizer(a,c)
    x,y=torch.randn(4,3,8,8),torch.arange(4)
    def step(m,o,s):
        o.zero_grad(set_to_none=True)
        _,out=m.bptt(x,[6,6,4]);F.cross_entropy(out,y).backward()
        o.step();m.project_();m.detach_state_();s.step()
    step(a,oa,sa)
    path=tmp_path/'saved.pt';save_checkpoint(path,a,oa,sa,1,c)
    step(a,oa,sa)
    b=CifarL8Analog(c);ob,sb=make_optimizer(b,c)
    checkpoint=torch.load(path,weights_only=False)
    assert restore_checkpoint(checkpoint,b,ob,sb,c)==1
    step(b,ob,sb)
    for group,values in a.snapshot().items():
        for name,value in values.items():torch.testing.assert_close(value,b.snapshot()[group][name],rtol=0,atol=0)
    assert oa.state_dict()['param_groups']==ob.state_dict()['param_groups']
    assert sa.state_dict()==sb.state_dict()
