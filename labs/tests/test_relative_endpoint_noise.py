"""CPU tests for the new acquisition law; full CUDA estimator parity is a smoke gate."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from experiments.replay_relative_endpoint_noise import noise_draws,noisy_endpoint,check_noise_record,offloaded_call


def test_relative_formula_preserves_zeros_and_signed_clean_endpoint():
    v=torch.tensor([-3.,0.,2.,0.],dtype=torch.float64)
    xi=torch.tensor([2.,9.,-4.,-7.],dtype=torch.float64)
    original=v.clone()
    acquired,noise,expected=noisy_endpoint(torch,v,xi,model='nodewise_relative',level=.001)
    assert torch.equal(noise,.001*v.abs()*xi)
    assert torch.equal(acquired,v+noise)
    assert torch.equal(v,original)
    assert torch.equal(noise[v==0],torch.zeros(2,dtype=torch.float64))
    assert expected==pytest.approx(.001*(13/4)**.5)


def test_absolute_and_relative_share_xi_without_matching_power():
    v=torch.tensor([0.,20.,-20.],dtype=torch.float64)
    xi=torch.tensor([1.,2.,3.],dtype=torch.float64)
    _,absolute,expected_absolute=noisy_endpoint(torch,v,xi,model='absolute',level=.0001)
    _,relative,expected_relative=noisy_endpoint(torch,v,xi,model='nodewise_relative',level=.0001)
    assert torch.equal(absolute,xi*.0001)
    assert relative[0]==0 and absolute[0]!=0
    assert expected_relative>10*expected_absolute


def test_raw_pairing_spans_models_levels_and_rejects_changed_xi():
    seen={}
    row=dict(phase='positive',state_layer_index=2,seed=101000202,element_count=16,standard_normal_sha256='raw')
    for model in ('absolute','nodewise_relative'):
        for level in (1e-4,1e-3):
            check_noise_record(seen,dict(row,noise_model=model,noise_level=level),architecture='conv3',seed=101000000,batch_index=0)
    with pytest.raises(AssertionError,match='Gaussian'):
        check_noise_record(seen,dict(row,standard_normal_sha256='changed'),architecture='conv3',seed=101000000,batch_index=0)


def test_six_contexts_thirteen_draws_and_preserved_parent_cases():
    count=0
    for lane,target in [('local',2.),('akib',6.)]:
        cfg=json.loads(Path(f'configs/conv/relative_endpoint_noise_conv3_{lane}_20260925.json').read_text())
        draws=noise_draws(cfg);assert len(draws)==13 and draws[0]==('clean',0.,None,-1)
        assert len({(model,level,seed) for model,level,seed,_ in draws})==13
        assert cfg['saved_activation_cpu_offload_scope']=='physical_replay_and_acquisition'
        assert {c['relative_target'] for c in cfg['cases']}=={target}
        parent=json.loads(Path(cfg['parent_config']['path']).read_text())
        for case in cfg['cases']:
            original=next(c for c in parent['cases'] if c['name']==case['name'])
            assert all(case[k]==v for k,v in original.items())
            assert case['T']==case['K']==8
        count+=len(cfg['cases'])*36*4*len(draws)
    assert count==11232


def test_saved_tensor_cpu_roundtrip_preserves_double_autograd():
    # CPU-only arithmetic test of the exact saved-tensor packing API; CUDA smoke
    # additionally requires old BPTT/state hashes, proving device arithmetic parity.
    x=torch.tensor([-.5,2.,3.],dtype=torch.float64,requires_grad=True)
    reference=torch.autograd.grad((x.sin()*x.square()).sum(),x)[0]
    with torch.autograd.graph.save_on_cpu(pin_memory=False):
        value=(x.sin()*x.square()).sum()
        offloaded=torch.autograd.grad(value,x)[0]
    assert torch.equal(reference,offloaded)


def test_cuda_limit_is_bytes_not_host_dependent_fraction():
    from types import SimpleNamespace
    from experiments.replay_relative_endpoint_noise import configure_cuda,assert_cuda_runtime
    calls=[]
    cuda=SimpleNamespace(is_available=lambda:True,get_device_properties=lambda d:SimpleNamespace(total_memory=10*1024**3),
                         set_per_process_memory_fraction=lambda fraction,device:calls.append((fraction,device)))
    cap,total=configure_cuda(SimpleNamespace(cuda=cuda),{'cuda_allocator_cap_bytes':int(7.5*1024**3)})
    assert calls==[(.75,0)] and cap/total==.75
    cuda.is_available=lambda:False
    with pytest.raises(RuntimeError,match='CUDA is required'):
        configure_cuda(SimpleNamespace(cuda=cuda),{'cuda_allocator_cap_bytes':cap})
    tensor=SimpleNamespace(device=SimpleNamespace(type='cuda'))
    layer=SimpleNamespace(state=tensor)
    runtime=dict(energy_fn=SimpleNamespace(layers=lambda:[layer]),free_layers=[layer],parameters=[layer])
    assert_cuda_runtime(runtime)
    tensor.device.type='cpu'
    with pytest.raises(AssertionError,match='CPU'):assert_cuda_runtime(runtime)


def test_offload_covers_residual_autograd_before_bptt(monkeypatch):
    from contextlib import contextmanager
    events=[]
    original=torch.autograd.graph.save_on_cpu
    @contextmanager
    def recording_offload(*,pin_memory):
        assert pin_memory is True
        events.append('enter')
        with original(pin_memory=False):
            yield
        events.append('exit')
    monkeypatch.setattr(torch.autograd.graph,'save_on_cpu',recording_offload)
    x=torch.tensor([-.5,2.,3.],dtype=torch.float64,requires_grad=True)
    reference=torch.autograd.grad((x.sin()*x.square()).sum(),x)[0]
    def physical_replay():
        events.append('residual')
        residual=torch.autograd.grad((x.sin()*x.square()).sum(),x)[0]
        events.append('bptt')
        bptt=torch.autograd.grad((x.sin()*x.square()).sum(),x)[0]
        return residual,bptt
    residual,bptt=offloaded_call(torch,physical_replay)
    assert events==['enter','residual','bptt','exit']
    assert torch.equal(reference,residual) and torch.equal(reference,bptt)


def test_residual_reuse_requires_full_coverage_and_identical_states(tmp_path):
    import csv
    from types import SimpleNamespace
    from experiments.replay_relative_endpoint_noise import load_residual_reference,ReusedResidualAccumulator
    rows=[dict(batch_index=b,phase=p,layer_index=l,coverage_complete=True,example_count=16,
               expected_example_count=16,unique_source_index_count=16,beta=0.,gate_threshold=.01)
          for b in range(36) for p in ('post_T_free','zero','negative','positive') for l in range(4)]
    path=tmp_path/'residuals.csv'
    def write(items):
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(items)
    write(rows)
    loaded=load_residual_reference(path)
    assert loaded==rows
    write(rows[:-1]+[rows[0]])
    with pytest.raises(AssertionError,match='coverage'):load_residual_reference(path)
    base=SimpleNamespace(_layer_state_sha256=lambda layers:'state',_parameter_state_sha256=lambda params:'params')
    guards={k:'state' for k in ('common_post_T_state_sha256','zero_state_sha256','negative_state_sha256','positive_state_sha256')}
    guards['parameter_state_sha256_after_cast']='params'
    accumulator=ReusedResidualAccumulator(loaded[:16],base,guards)
    runtime=dict(free_layers=[],parameters=[])
    for phase in ('post_T_free','zero','negative','positive'):
        accumulator.add_endpoint(runtime,gradient_source=None,phase=phase,beta=0.,source_indices=list(range(16)))
    assert accumulator.summary_rows(context={'batch_index':0},expected_examples=16,p90_threshold=.01)==loaded[:16]
    guards['positive_state_sha256']='wrong'
    with pytest.raises(AssertionError,match='endpoint differs'):
        ReusedResidualAccumulator(loaded[:16],base,guards).add_endpoint(runtime,gradient_source=None,phase='positive',beta=0.,source_indices=list(range(16)))


def test_phase_residual_reuse_requires_state_beta_parameters_and_force():
    from types import SimpleNamespace
    from experiments.replay_relative_endpoint_noise import reused_phase_residual
    base=SimpleNamespace(_layer_state_sha256=lambda layers:'state',_parameter_state_sha256=lambda params:'params',_tensor_sha256=lambda force:force)
    runtime=dict(free_layers=[],parameters=[],augmented_fn=SimpleNamespace(nudging=2.,_nudging=SimpleNamespace(_force='force')))
    row=dict(state_sha256='state',actual_beta=2.,projected_kkt_rms=.001,projected_kkt_max_abs=.002,active_fraction=.5)
    guard=dict(parameter_state_sha256_after_cast='params',frozen_current_force_sha256='force')
    assert reused_phase_residual(runtime,base,[row],guard)=={k:row[k] for k in ('projected_kkt_rms','projected_kkt_max_abs','active_fraction')}
    runtime['augmented_fn'].nudging=3.
    with pytest.raises(AssertionError,match='endpoint/beta'):reused_phase_residual(runtime,base,[row],guard)
    runtime['augmented_fn'].nudging=2.;runtime['augmented_fn']._nudging._force='changed'
    with pytest.raises(AssertionError):reused_phase_residual(runtime,base,[row],guard)
