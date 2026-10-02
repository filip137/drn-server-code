"""Cheap CPU contracts for matched order, restoration and partial checkpoints."""
import copy
import json
import random

import numpy as np
import pytest
import torch

from experiments.screen_cifar_beta_half_epoch import (
    EXAMPLES, UPDATES, backward, partial_metadata, restore_parent, save_partial,
    selected_order, validate_case, verify_hashes,
)
from experiments.train_cifar_l8_analog import save_checkpoint


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, dtype=torch.float64))
        self.register_buffer('bn_count', torch.tensor(0))

    def snapshot(self):
        return {'module': {n:p.detach().clone() for n,p in self.state_dict().items()}}

    def restore(self, state):
        self.load_state_dict(state['module'])


def setup():
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: 1 - e / 50)
    config = dict(epochs=11, input_gain_init=1., blocks=[[1]])
    return model, optimizer, scheduler, config


def update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    noise = random.random() + np.random.random() + torch.rand(1).item()
    (model.weight.square().sum() * noise).backward()
    optimizer.step()
    model.bn_count.add_(1)


def test_selected_order_exact_half_and_same_epoch11_prefix():
    train = list(range(45000))
    expected = torch.randperm(45000, generator=torch.Generator().manual_seed(11000007)).tolist()
    actual = selected_order(train, 7)
    assert actual == expected[:22500] == selected_order(train, 7)
    assert len(set(actual)) == EXAMPLES
    sizes = [len(actual[i:i+32]) for i in range(0, EXAMPLES, 32)]
    assert len(sizes) == UPDATES == 704 and sizes[-1] == 4


def test_parent_restore_reproduces_next_update_and_preserves_partial_state(tmp_path):
    torch.manual_seed(17)
    random.seed(19)
    np.random.seed(23)
    a, oa, sa, config = setup()
    for _ in range(10):
        update(a, oa)
        sa.step()
    path = tmp_path / 'parent.pt'
    save_checkpoint(path, a, oa, sa, 10, config)
    checkpoint = torch.load(path, weights_only=False)
    update(a, oa)
    b, ob, sb, _ = setup()
    restore_parent(checkpoint, b, ob, sb, config, parent_steps=10)
    update(b, ob)
    torch.testing.assert_close(a.weight, b.weight, rtol=0, atol=0)
    assert torch.equal(a.bn_count, b.bn_count)
    assert sa.state_dict() == sb.state_dict()
    for key, value in oa.state[a.weight].items():
        torch.testing.assert_close(value, ob.state[b.weight][key], rtol=0, atol=0)
    progress = partial_metadata(1, 32, 10, 'order')
    save_partial(tmp_path / 'partial.pt', b, ob, sb, config, progress)
    saved = torch.load(tmp_path / 'partial.pt', weights_only=False)
    assert saved['epoch'] == saved['scheduler']['last_epoch'] == 10
    assert saved['partial_epoch']['adam_steps'] == 11
    assert saved['partial_epoch']['examples_in_epoch'] == 32
    assert saved['native_epoch_resume_safe'] is False
    assert {'torch_rng', 'numpy_rng', 'python_rng', 'cuda_rng'} <= saved.keys()
    with pytest.raises(ValueError, match='complete epoch-10'):
        restore_parent(saved, b, ob, sb, config, parent_steps=10)


def test_partial_final_metadata_and_coverage_validation():
    progress = partial_metadata(704, 22500, 14070, 'hash')
    assert progress['adam_steps'] == 14774
    assert progress['scheduler_last_epoch'] == 10
    assert progress['native_epoch_resume_safe'] is False
    with pytest.raises(ValueError, match='progress'):
        partial_metadata(704, 22528, 14070, 'hash')


def test_free_t_control_matches_ordinary_mean_ce_and_updates_bn_once():
    class FreeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(3)
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, x, iterations, reset):
            assert reset is True
            return self.linear(self.bn(x))

        def bptt(self, *args):
            raise AssertionError('Free-T control called native BPTT')

    torch.manual_seed(9)
    a = FreeModel()
    b = copy.deepcopy(a)
    x, y = torch.randn(4, 3), torch.tensor([0, 1, 0, 1])
    expected = torch.nn.functional.cross_entropy(a(x, [1], True), y)
    expected.backward()
    _, actual = backward(b, x, y, [1], {'method': 'free_t_bptt'})
    torch.testing.assert_close(actual, expected.detach())
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert int(b.bn.num_batches_tracked) == 1


def test_hash_verification_detects_tampering_and_missing_required_file(tmp_path):
    from experiments.reporting import sha256_file
    path = tmp_path / 'case.json'
    path.write_text(json.dumps({'method': 'native_bptt'}))
    hashes = {'case.json': sha256_file(path)}
    verify_hashes(tmp_path, hashes, ['case.json'])
    with pytest.raises(ValueError, match='Incomplete'):
        verify_hashes(tmp_path, hashes, ['parent.pt'])
    path.write_text('{}')
    with pytest.raises(ValueError, match='Identity mismatch'):
        verify_hashes(tmp_path, hashes)


def test_case_requires_finite_positive_beta_vector():
    validate_case({'method': 'hybrid_eqprop', 'betas': [.1, 1., 10.]})
    for betas in ([1., 2.], [1., 2., float('nan')], [0., 1., 2.]):
        with pytest.raises(ValueError, match='betas'):
            validate_case({'method': 'hybrid_eqprop', 'betas': betas})
