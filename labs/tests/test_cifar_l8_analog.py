import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytest
from contextlib import nullcontext
from labs.cifar_l8_analog import CifarL8Analog
from experiments.train_cifar_l8_analog import Cohort, make_split, make_optimizer
from experiments.train_cifar_l8_analog import save_checkpoint, restore_checkpoint, deterministic_setup
from experiments.train_cifar_l8_analog import check_gpu_resident_execution


def config():
    c=json.loads(Path('configs/cifar/cifar10_l8_analog_bs16_seed0.json').read_text())
    c['blocks']=[[2,2,2],[3,3,3],[4,4]]
    c['input_size']=8
    return c


def test_gpu_resident_guard_rejects_cpu_and_offload(monkeypatch):
    with pytest.raises(ValueError,match='forbids CPU offloading'):
        check_gpu_resident_execution('cpu',False)
    with pytest.raises(ValueError,match='forbids CPU offloading'):
        check_gpu_resident_execution('cuda',True)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CPU fallback is forbidden'):
        check_gpu_resident_execution('cuda',False)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    check_gpu_resident_execution('cuda',False)


def test_dense_equilibrium_and_gradient():
    torch.manual_seed(0);m=CifarL8Analog(config());head=m.head
    x=torch.randn(2,4,requires_grad=True)
    actual=head(x);weight=head.weight.state
    voltage=F.softplus(head._input_gain_raw)*torch.cat([x,-x],1)
    expected=(voltage@weight)/weight.sum(0)
    torch.testing.assert_close(actual,expected)
    residual=head.interaction.b_coef_fn(head.output)()+2*head.interaction.a_coef_fn(head.output)()*actual
    torch.testing.assert_close(residual,torch.zeros_like(residual),atol=1e-5,rtol=0)
    actual.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm()>0
    assert head.weight.state.grad.norm()>0
    assert not any(isinstance(module,nn.Linear) for module in m.modules())


def test_bptt_bn_updates_and_checkpoint():
    torch.manual_seed(1);c=config();m=CifarL8Analog(c);m.train()
    x=torch.randn(3,3,8,8);y=torch.tensor([0,1,2])
    opt,sched=make_optimizer(m,c)
    before=m.snapshot();free,tracked=m.bptt(x,[6,6,4])
    assert free.shape==tracked.shape==(3,10)
    F.cross_entropy(tracked,y).backward()
    tensors=m.trainable_tensors();assert len(tensors)==19
    for name,p in tensors.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0,name
    for bn in [b[1] for b in m.bridges]:assert bn.num_batches_tracked.item()==1
    opt.step();m.project_();sched.step();state=m.snapshot()
    for name,p in m.named_conductances():
        assert p.min()>=c['weight_min'] and p.max()<=c['weight_max']
        assert not torch.equal(before['conductances'][name],state['conductances'][name])
    fresh=CifarL8Analog(c);fresh.restore(state);m.eval();fresh.eval()
    with torch.no_grad():torch.testing.assert_close(m(x,[6,6,4]),fresh(x,[6,6,4]),rtol=0,atol=0)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in tensors.values()}


def test_split_and_augmentation_are_batch_independent():
    labels=np.repeat(np.arange(10),5000);split=make_split(labels,0)
    assert len(split['train'])==45000 and len(split['validation'])==5000
    assert not set(split['train'])&set(split['validation'])
    assert np.bincount(labels[split['validation']]).tolist()==[500]*10
    images=np.random.default_rng(0).integers(0,256,(7,32,32,3),dtype=np.uint8)
    c=config();a=Cohort(images,labels[:7],list(range(7)),c,epoch=3,augment=True)
    b=Cohort(images,labels[:7],[6,2,0],c,epoch=3,augment=True)
    torch.testing.assert_close(a[2][0],b[1][0],rtol=0,atol=0)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_epoch_resume_matches_uninterrupted_adam(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    deterministic_setup(23);c=config();c['epochs']=30
    a=CifarL8Analog(c,device);oa,sa=make_optimizer(a,c)
    def step(model,opt,sched,epoch):
        generator=torch.Generator().manual_seed(1000000*epoch)
        x=torch.randn(3,3,8,8,generator=generator).to(device)
        y=torch.tensor([0,1,2],device=device)
        model.train();opt.zero_grad(set_to_none=True)
        _,tracked=model.bptt(x,[6,6,4]);F.cross_entropy(tracked,y).backward()
        opt.step();model.project_();model.detach_state_();sched.step()
    for epoch in range(1,11):step(a,oa,sa,epoch)
    path=tmp_path/'epoch10.pt';save_checkpoint(path,a,oa,sa,10,c)
    for epoch in range(11,13):step(a,oa,sa,epoch)
    b=CifarL8Analog(c,device);ob,sb=make_optimizer(b,c)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert restore_checkpoint(saved,b,ob,sb,c)==10
    for epoch in range(11,13):step(b,ob,sb,epoch)
    def exact(x,y):
        if isinstance(x,torch.Tensor):torch.testing.assert_close(x,y,rtol=0,atol=0)
        elif isinstance(x,dict):
            assert x.keys()==y.keys()
            for key in x:exact(x[key],y[key])
        elif isinstance(x,(list,tuple)):
            assert len(x)==len(y)
            for xx,yy in zip(x,y):exact(xx,yy)
        else:assert x==y
    exact(a.snapshot(),b.snapshot());exact(oa.state_dict(),ob.state_dict());exact(sa.state_dict(),sb.state_dict())
    changed=copy.deepcopy(c);changed['scheduler']['horizon_epochs']=30
    with pytest.raises(ValueError,match='scheduler'):restore_checkpoint(saved,b,ob,sb,changed)
    changed=copy.deepcopy(c);changed['optimizer']['head_learning_rate']*=3
    with pytest.raises(ValueError,match='optimizer'):restore_checkpoint(saved,b,ob,sb,changed)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_saved_tensor_cpu_offload_preserves_gradients(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(17);c=config();a=CifarL8Analog(c,device)
    b=CifarL8Analog(c,device);b.restore(a.snapshot())
    x=torch.randn(3,3,8,8,device=device);y=torch.tensor([0,1,2],device=device)
    outputs=[]
    for model,offload in [(a,False),(b,True)]:
        model.train()
        context=torch.autograd.graph.save_on_cpu(pin_memory=False) if offload else nullcontext()
        with context:
            free,tracked=model.bptt(x,[6,6,4]);loss=F.cross_entropy(tracked,y)
        loss.backward();outputs.append((free,tracked))
    for aa,bb in zip(*outputs):torch.testing.assert_close(aa,bb,rtol=0,atol=0)
    for name,p in a.trainable_tensors().items():
        other=b.trainable_tensors()[name].grad
        torch.testing.assert_close(p.grad,other,rtol=0,atol=0)
    for aa,bb in zip(a.bridges,b.bridges):
        assert aa[1].num_batches_tracked.item()==bb[1].num_batches_tracked.item()==1
