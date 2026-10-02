import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from experiments import train_cifar_digital_l12 as digital


def config(tiny=True):
    value = json.loads(Path('configs/cifar/digital_l8_linear_outputs_20261002.json').read_text())
    if tiny:
        value.update(blocks=[[2, 2, 2], [3, 3, 3], [4, 4]],
                     input_shape=[3, 8, 8], readout_features=4, num_workers=0)
    return value


def test_linear_outputs_preserve_topology_initialization_and_parameter_keys():
    c = config(False)
    torch.manual_seed(0)
    linear = digital.DigitalL12(c)
    c['block_output_activation'] = 'relu'
    torch.manual_seed(0)
    original = digital.DigitalL12(c)
    assert sum(p.numel() for p in linear.parameters()) == 5395594
    assert sum(isinstance(m, nn.Conv2d) for m in linear.modules()) == 8
    assert sum(isinstance(m, nn.ReLU) for m in linear.modules()) == 5
    assert [i for i, m in enumerate(linear.features) if isinstance(m, nn.Identity)] == [5, 13, 19]
    assert linear.state_dict().keys() == original.state_dict().keys()
    for key, value in linear.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)


def test_default_relu_behavior_and_invalid_option():
    c = config()
    c.pop('block_output_activation')
    torch.manual_seed(4)
    implicit = digital.DigitalL12(c).eval()
    c['block_output_activation'] = 'relu'
    torch.manual_seed(4)
    explicit = digital.DigitalL12(c).eval()
    x = torch.randn(2, 3, 8, 8)
    torch.testing.assert_close(implicit(x), explicit(x), rtol=0, atol=0)
    c['block_output_activation'] = 'typo'
    with pytest.raises(ValueError, match='block_output_activation'):
        digital.DigitalL12(c)


def test_negative_block_outputs_have_unclamped_gradients():
    model = digital.DigitalL12(config())
    with torch.no_grad():
        for layer in (model.features[0], model.features[2]):
            layer.weight.fill_(0.1)
        model.features[4].weight.fill_(-0.1)
    output = model.features[:6](torch.ones(2, 3, 8, 8))
    assert (output < 0).all()
    output.sum().backward()
    assert (model.features[4].weight.grad > 0).all()
    assert model(torch.randn(2, 3, 8, 8)).shape == (2, 10)


def test_zero_decay_applies_to_every_parameter():
    c = config()
    model = digital.DigitalL12(c)
    optimizer = digital.make_optimizer(model, c)
    assert all(group['weight_decay'] == 0 for group in optimizer.param_groups)
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {
        id(p) for p in model.parameters()}


def test_validation_split_never_constructs_official_dataset(tmp_path, monkeypatch):
    from experiments import train_cifar_l8_analog as analog

    c = config(False)
    c['num_workers'] = 0
    labels = np.repeat(np.arange(10), 5000)
    images = np.broadcast_to(np.zeros((1, 32, 32, 3), dtype=np.uint8), (50000, 32, 32, 3))
    monkeypatch.setattr(analog, 'read_training_data', lambda root: (images, labels))

    def forbidden(*args, **kwargs):
        raise AssertionError('Official CIFAR dataset must never be constructed')

    monkeypatch.setattr(digital.datasets, 'CIFAR10', forbidden)
    for i in range(1, 6):
        (tmp_path / f'data_batch_{i}').write_bytes(b'training-only')
    train, validation = digital.build_loaders(c, tmp_path, False, torch.device('cpu'), tmp_path)
    saved = json.loads((tmp_path / 'split.json').read_text())
    assert len(train.dataset) == 45000 and len(validation.dataset) == 5000
    assert set(saved['train']).isdisjoint(saved['validation'])
    assert len(set(saved['train']) | set(saved['validation'])) == 50000
    assert np.array_equal(np.bincount(labels[saved['validation']]), np.full(10, 500))
    assert {k: saved[k] for k in ('train', 'validation')} == analog.make_split(labels, 0)
    assert not saved['official_test_read']
    assert len(saved['training_files_sha256']) == 5
    first_images, _ = next(iter(validation))
    second_images, _ = next(iter(validation))
    torch.testing.assert_close(first_images, second_images, rtol=0, atol=0)
    smoke_train, smoke_val = digital.build_loaders(c, tmp_path, True, torch.device('cpu'))
    assert len(smoke_train) == 3 and len(smoke_val) == 1


def test_legacy_loader_keeps_official_test_contract(monkeypatch):
    c = json.loads(Path('configs/cifar_digital_l8_wide_affinebn_20260921.json').read_text())
    c['num_workers'] = 0
    calls = []

    def fake_dataset(root, *, train, transform, download):
        calls.append(train)
        return range(50000 if train else 10000)

    monkeypatch.setattr(digital.datasets, 'CIFAR10', fake_dataset)
    train, test = digital.build_loaders(c, '/unused', False, torch.device('cpu'))
    assert calls == [True, False]
    assert len(train.dataset) == 50000 and len(test.dataset) == 10000
