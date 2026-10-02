import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from experiments.replay_cifar_digital_to_legacy import analog_features, split_convolution, transfer
from experiments.train_cifar_digital_l12 import DigitalL12
from labs.cifar_l8_analog import CifarL8Analog


def test_sign_split_preserves_signed_drive_and_relu_pair():
    torch.manual_seed(37)
    weight = torch.randn(4, 3, 3, 3)
    x = torch.randn(2, 3, 8, 8)
    mapped = split_convolution(weight, 1e-7, 10.)
    drive = F.conv2d(torch.cat((x, -x), 1), mapped, padding=1)
    expected = F.conv2d(x, weight, padding=1)
    torch.testing.assert_close(drive[:, :4], expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(drive[:, 4:], -expected, atol=1e-5, rtol=1e-5)
    decoded = drive[:, :4].clamp_min(0) - drive[:, 4:].clamp_max(0)
    torch.testing.assert_close(decoded, 2 * expected.relu(), atol=2e-5, rtol=1e-5)
    with pytest.raises(ValueError, match='bounds'):
        split_convolution(weight, 1e-7, .1)


def test_transfer_preserves_bn_and_native_forward_with_explicit_head_limit():
    torch.manual_seed(41)
    config = json.loads(Path('configs/cifar/cifar10_l8_analog_bs32_seed0.json').read_text())
    config.update(blocks=[[2, 2, 2], [3, 3, 3], [4, 4]], input_size=8)
    dc = json.loads(Path('configs/cifar_digital_l8_wide_affinebn_20260921.json').read_text())
    dc.update(blocks=config['blocks'], input_shape=[3, 8, 8], readout_features=4)
    digital = DigitalL12(dc).eval()
    analog = CifarL8Analog(config).eval()
    audit = transfer(digital, analog, config)
    assert not audit['native_head_bias_representable']
    norms = [m for m in digital.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    for source, bridge in zip(norms, analog.bridges):
        for key, value in source.state_dict().items():
            assert torch.equal(value, bridge[1].state_dict()[key])
    with torch.no_grad():
        x = torch.randn(2, 3, 8, 8)
        features = analog_features(analog, x, config['iterations'])
        native = analog.head(features)
        torch.testing.assert_close(native, analog(x, config['iterations']), atol=0, rtol=0)
        head_gain = F.softplus(analog.head._input_gain_raw)
        expected = head_gain * F.linear(features, digital.head.weight) / analog.head.weight.state.sum(0)
        torch.testing.assert_close(native, expected, atol=1e-4, rtol=1e-5)
