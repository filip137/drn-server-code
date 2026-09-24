"""Physical gain equivalence, optimizer separation, and resumability controls."""
import copy
import pytest
import torch
from torch.nn import functional as F
from labs.cifar_l8_analog import CifarL8Analog
from labs.tests.test_cifar_l8_amplification import config
from experiments.train_cifar_l8_analog import make_optimizer, restore_checkpoint, save_checkpoint


def gain_config(mode):
    c = dict(config(4., .25), conv_input_gain_mode=mode, bn_affine_trainable=False)
    c['optimizer']['conv_gain_learning_rate'] = 1e-3
    c['optimizer']['gain_learning_rate'] = 5e-5
    c['epochs'] = 10
    return c


@pytest.mark.parametrize('mode', ['fixed', 'log'])
def test_original_initializer_preserves_physical_model(mode):
    torch.manual_seed(74)
    old = CifarL8Analog(gain_config('softplus'))
    new = CifarL8Analog(gain_config(mode))
    new.restore_initializer(old.snapshot())
    for name, value in old.named_conductances():
        assert torch.equal(value, dict(new.named_conductances())[name])
    for a, b in zip(old.analog_blocks, new.analog_blocks):
        assert float(a.input_gain) == float(b.input_gain) == 100.
    x = torch.randn(4, 3, 8, 8)
    torch.testing.assert_close(old(x, [6, 6, 4]), new(x, [6, 6, 4]), rtol=0, atol=0)
    invalid = copy.deepcopy(old.snapshot())
    invalid['module']['analog_blocks.0._input_gain_raw'] += 1
    with pytest.raises(ValueError, match='original softplus'):
        new.restore_initializer(invalid)


@pytest.mark.parametrize('mode', ['fixed', 'log'])
def test_gain_optimizer_and_resumed_next_step(mode, tmp_path):
    torch.manual_seed(75)
    c = gain_config(mode)
    model = CifarL8Analog(c)
    opt, sched = make_optimizer(model, c)
    groups = {g['name']: g for g in opt.param_groups}
    assert groups['head._input_gain_raw']['lr'] == 5e-5
    assert not any(name.startswith('bridges.') for name in groups)
    gains = [g for n, g in groups.items() if n.startswith('analog_blocks.')]
    assert len(gains) == (3 if mode == 'log' else 0)
    assert all(g['lr'] == 1e-3 for g in gains)
    x, y = torch.randn(4, 3, 8, 8), torch.arange(4)
    def step(m, o, s):
        o.zero_grad(set_to_none=True)
        _, out = m.bptt(x, [6, 6, 4])
        F.cross_entropy(out, y).backward()
        o.step(); m.project_(); m.detach_state_(); s.step()
    step(model, opt, sched)
    if mode == 'fixed':
        assert all(float(b.input_gain) == 100. and b._input_gain_raw.grad is None
                   for b in model.analog_blocks)
    else:
        assert any(abs(float(b.input_gain) / 100. - 1) > 1e-4 for b in model.analog_blocks)
        for b in model.analog_blocks:
            torch.testing.assert_close(b.input_gain, 100. * b._input_gain_raw.exp())
    for bridge in model.bridges:
        assert int(bridge[1].num_batches_tracked) == 1
        assert torch.equal(bridge[1].weight, torch.ones_like(bridge[1].weight))
        assert torch.equal(bridge[1].bias, torch.zeros_like(bridge[1].bias))
    path = tmp_path / 'epoch1.pt'
    save_checkpoint(path, model, opt, sched, 1, c)
    step(model, opt, sched)
    restored = CifarL8Analog(c); ro, rs = make_optimizer(restored, c)
    assert restore_checkpoint(torch.load(path, weights_only=False), restored, ro, rs, c) == 1
    step(restored, ro, rs)
    for kind, values in model.snapshot().items():
        for name, value in values.items():
            torch.testing.assert_close(value, restored.snapshot()[kind][name], rtol=0, atol=0)
