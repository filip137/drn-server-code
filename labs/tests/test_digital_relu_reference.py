import json
from pathlib import Path

import pytest
import torch

from experiments.train_digital_relu_reference import build_model, build_optimizer, loss_value, check_state


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("loss", ["mse", "cross_entropy"])
def test_requested_geometry_and_frozen_bias_training(depth, loss):
    config = json.loads((Path(__file__).resolve().parents[2] / "configs/digital_relu_table12_20260917.json").read_text())
    torch.manual_seed(0)
    model = build_model(config, depth)
    optimizer = build_optimizer(model, config)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    assert [m.out_channels for m in convs] == [32, 64, 128][:depth]
    assert [m.stride for m in convs] == [(2, 2), (2, 2), (1, 1)][:depth]
    x = torch.randn(2, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (2, 10)
    value = loss_value(logits, torch.tensor([1, 4]), loss)
    value.backward()
    optimizer.step()
    check_state(model)
    for name, parameter in model.named_parameters():
        if name.endswith("bias"):
            assert torch.count_nonzero(parameter) == 0
        else:
            assert not torch.equal(parameter, before[name])
    assert all(g["lr"] == 0 for g in optimizer.param_groups if g["name"].endswith("bias"))


def test_losses_use_logits_and_one_hot_targets():
    logits = torch.tensor([[2., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    labels = torch.tensor([0])
    assert loss_value(logits, labels, "mse").item() == pytest.approx(.1)
    assert loss_value(logits, labels, "cross_entropy").item() == pytest.approx(torch.logsumexp(logits, 1).item() - 2)
