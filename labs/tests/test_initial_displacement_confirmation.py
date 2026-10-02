import json
import math
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.replay_initial_displacement_confirmation import noise_draws,verify_screen_reference

class Scalar:
    def __init__(self,value): self.value=value
    def square(self): return Scalar(self.value**2)
    def mean(self): return self
    def sqrt(self): return Scalar(math.sqrt(self.value))
    def __float__(self): return float(self.value)


def test_confirmation_has_exact_selected_cells_and_fresh_draws():
    total=0
    for depth in (1,2,3):
        cfg=json.loads(Path(f'configs/conv/confirmation_initial_displacement_conv{depth}_20260925.json').read_text())
        draws=noise_draws(cfg)
        assert draws[0]==(0.,None,-1)
        assert {seed for sigma,seed,i in draws if sigma}==set(range(94000000,99000000,1000000))
        cells={(c['relative_target'],sigma) for c in cfg['cases'] for sigma in cfg['noise_sigmas'][1:]}
        assert cells=={tuple(x) for x in cfg['selected_pair']['cells']}
        rows=len(cfg['cases'])*36*len(draws)*(depth+1)
        assert rows==cfg['expected_layer_comparisons']
        total+=rows
    assert total==11016


def test_confirmation_rejects_changed_screen_trajectory_or_bptt():
    keys=('common_post_T_state_sha256','positive_state_sha256','negative_state_sha256','zero_state_sha256',
          'frozen_current_force_sha256','parameter_state_sha256_after_cast','reconstructed_initialization_tensor_sha256')
    guard={k:k for k in keys}
    output={'guard':dict(guard),'bptt_gradients':{'weight':'gradienthash'},'gradients':{'weight':Scalar(2)}}
    rows={(0,'weight'):{'bptt_gradient_sha256':'gradienthash','clean_eqprop_rms':'2'}}
    base=SimpleNamespace(_tensor_sha256=lambda x:x)
    verify_screen_reference(base,output,guard,rows,0)
    output['guard']['positive_state_sha256']='changed'
    with pytest.raises(AssertionError,match='Screen reconstruction'):
        verify_screen_reference(base,output,guard,rows,0)
    output['guard']=dict(guard);output['bptt_gradients']['weight']='wrong'
    with pytest.raises(AssertionError,match='BPTT'):
        verify_screen_reference(base,output,guard,rows,0)
