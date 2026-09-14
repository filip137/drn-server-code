"""Export Torch fixtures, then compare the authors' actual MLP/JAX routines.

Run --export under the repository Python, then --check under an isolated
JAX/Flax Python. No training, input-data upload, or parameter modification.
Only unavailable, unused CNN/constraint imports are omitted when loading source.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def export(output, checkpoint=None, data=None):
    import torch
    from labs.fmnist_homeostasis import PaperNet, free_state, feedback, homeostasis
    from labs.directed_eqprop import local_gradient
    torch.set_num_threads(1)
    if checkpoint:
        f=np.load(checkpoint)
        model=PaperNet(hidden=f['forward1'].shape[0]).double()
        model.load_state_dict({k:torch.tensor(f[k]) for k in model.state_dict()})
        x=torch.tensor(np.load(data/'test_x.npy',mmap_mode='r')[:500]).double()
        y=torch.tensor(np.load(data/'test_y.npy',mmap_mode='r')[:500])
        u,drive=free_state(model,x)
        worst=model.force(u,drive).norm(dim=-1).topk(3).indices
        x,y=x[worst],y[worst]
    else:
        model=PaperNet(inputs=4,hidden=3,outputs=2,seed=3,dtype=torch.float64)
        x=torch.tensor([[.2,.8,.4,.1],[.9,.2,.3,.6]],dtype=torch.float64)
        y=torch.tensor([0,1])
    ell,u,_,_=feedback(model,x,y,'vf_ad',torch.Generator())
    grads=local_gradient(model,x,u,y,ell)
    eps=torch.randn(5*len(x),model.n,dtype=torch.float64,generator=torch.Generator().manual_seed(5))
    reg,reg_loss=homeostasis(model,len(x),None,eps=eps)
    values={k:v.detach().numpy() for k,v in model.state_dict().items()}
    values.update({"grad_"+k:v.numpy() for k,v in grads.items()})
    values.update({"homeo_"+k:v.numpy() for k,v in reg.items()})
    np.savez(output,**values,x=x.numpy(),y=y.numpy(),u=u.numpy(),ell=ell.numpy(),
             eps=eps.numpy(),homeo_loss=reg_loss,force=model.force(u,model.drive(x)).detach().numpy())


def check(fixture, source, output):
    import sys,types,importlib.util
    import jax
    import jax.numpy as jnp
    from flax.core import freeze
    jax.config.update('jax_enable_x64',True)
    def load(name,path,omit=()):
        text=path.read_text()
        for line in omit:
            assert line in text
            text=text.replace(line,'')
        module=types.ModuleType(name)
        module.__file__=str(path)
        sys.modules[name]=module
        exec(compile(text,str(path),'exec'),module.__dict__)
        return module
    for package in ('utils','models'):
        module=types.ModuleType(package);module.__path__=[];sys.modules[package]=module
    load('models.act',source/'act.py')
    funcs=load('utils.funcs',source/'utils/funcs.py')
    # CNN pooling is never instantiated in this MLP-only test.
    vfs=load('models.vfs',source/'vfs.py',('from utils.pool import SfmPool\n',))
    # The public funcs.py lacks these two unused constraint helpers.
    dyn=load('models.dyn',source/'dyn.py',('from utils.funcs import dalify, ff_exc_fb_inh, batch_keys, concat_flat_leaves\n',))
    dyn.batch_keys=funcs.batch_keys;dyn.concat_flat_leaves=funcs.concat_flat_leaves
    f=np.load(fixture)
    h=f['forward1'].shape[0];o=f['readout'].shape[0]
    sizes={'l1':h,'l2':h,'out':o}
    connections={'in-l1':'dense','l1-l2':'dense','l2-l1':'dense_nobias','l2-out':'dense','out-l2':'dense_nobias'}
    vf=vfs.mlp_discrete_vf(shapes={k:(v,) for k,v in sizes.items()},sizes=sizes,
                          connections=connections,loss='xent',act=funcs.sigmoid)
    model=dyn.DiscreteDynamics(vf=vf,seed=0,jac_reg_coef=1.)
    pairs={'in-l1':'input','l1-l2':'forward1','l2-l1':'backward1','l2-out':'forward2','out-l2':'backward2'}
    p={k:{'kernel':jnp.asarray(f[name].T)} for k,name in pairs.items()}
    p['in-l1']['bias']=jnp.asarray(f['bias'][:h])
    p['l1-l2']['bias']=jnp.asarray(f['bias'][h:2*h])
    p['l2-out']['bias']=jnp.asarray(f['bias'][2*h:])
    p['readout']={'Dense_0':{'kernel':jnp.asarray(f['readout'].T),'bias':jnp.asarray(f['readout_bias'])}}
    params=freeze({'params':p})
    x=jnp.asarray(f['x']);y=jax.nn.one_hot(f['y'],o,dtype=jnp.float64)
    concat=lambda tree:jnp.concatenate([tree[k] for k in ('l1','l2','out')],axis=-1)
    u=model.fwd(params,model.batch_nrn(x),x,y,0.,150)
    du=model.du_dbeta_autodiff(params,u,x,y,20)
    grad=model.hep_vf(params,u,x,y,20,.01,0)
    diffs={}
    def compare(name,a,b):
        a,b=np.asarray(a),np.asarray(b)
        diffs[name]=dict(max_absolute=float(np.max(np.abs(a-b))),
                        relative=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30)))
        np.testing.assert_allclose(a,b,rtol=2e-8,atol=2e-9,err_msg=name)
    compare('free_state',concat(u),f['u'])
    compare('force',concat(model.vfield(params,u,x,y,0.)),f['force'])
    compare('VF_response_sign_translated',-concat(du),f['ell'])
    def compare_grads(prefix,g):
        g=g['params']
        for k,name in pairs.items():
            compare(prefix+name,g[k]['kernel'].T,f[prefix+name])
        compare(prefix+'bias',jnp.concatenate((g['in-l1']['bias'],g['l1-l2']['bias'],g['l2-out']['bias'])),f[prefix+'bias'])
        compare(prefix+'readout',g['readout']['Dense_0']['kernel'].T,f[prefix+'readout'])
        compare(prefix+'readout_bias',g['readout']['Dense_0']['bias'],f[prefix+'readout_bias'])
    compare_grads('grad_',grad)
    # Supply the same Gaussian draws to the actual homeo_loss routine, while
    # leaving its JVPs, loss and parameter differentiation unchanged.
    eps=jnp.asarray(f['eps']).reshape(5,len(x),2*h+o)
    saved_normal=jax.random.normal
    calls=[]
    def fixed_normal(key,shape,**kwargs):
        index=len(calls);calls.append(index)
        probe,layer=divmod(index,3)
        start=(0,h,2*h)[layer];end=(h,2*h,2*h+o)[layer]
        result=eps[probe,:,start:end]
        assert result.shape==shape
        return result
    jax.random.normal=fixed_normal
    try:
        def loss(p):
            return sum(model.homeo_loss(p,u,x,y,jax.random.PRNGKey(k)) for k in range(5))/5
        value,g=jax.value_and_grad(loss)(params)
    finally:
        jax.random.normal=saved_normal
    assert len(calls)==15
    compare('homeo_loss',value,f['homeo_loss'])
    compare_grads('homeo_',g)
    hashes={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (source/'act.py',source/'vfs.py',source/'dyn.py',source/'utils/funcs.py')}
    record=dict(status='passed',jax=jax.__version__,fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        public_source_sha256=hashes,checks=diffs,
        import_adaptations=['unused CNN SfmPool import omitted','missing unused dalify/ff_exc_fb_inh imports omitted'],
        same_gaussian_draws_injected_for_homeostasis=True)
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--export',action='store_true');mode.add_argument('--check',action='store_true')
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--data',type=Path)
    args=p.parse_args()
    if args.export: export(args.fixture,args.checkpoint,args.data)
    else: check(args.fixture,args.source,args.output)


if __name__=='__main__':
    main()
