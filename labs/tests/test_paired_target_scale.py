import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from model.function.cost import SquaredErrorPairedOutputs
from model.variable.layer import LinearLayer


def tiny_config():
    root = Path(__file__).resolve().parents[2]
    source = root / "configs/conv/perfectdiode_conv123_zero_bias_ordinary_mnist_seed0_20260805_v1/conv1/01_baseline_adam_bias_zero_seed0.json"
    config = json.loads(source.read_text())
    config["model_overrides"]["mnist_bp_conv_amp"]["layer_shapes"][1][0] = 2
    config["model_base"].update(input_gain=1.0, num_iterations_training=2, num_iterations_inference=2)
    config["datasets"]["mnist"]["params"].update(batch_size=2, validation_batch_size=2)
    return config


def make_cost(**kwargs):
    layer = LinearLayer((12,), batch_size=3, device="cpu")
    layer.state = torch.linspace(-0.7, 0.9, 36, dtype=torch.float64).reshape(3, 12)
    layer.state.requires_grad_(True)
    cost = SquaredErrorPairedOutputs(layer, num_classes=6, **kwargs)
    cost.set_target(torch.tensor([0, 4, 5]))
    return layer, cost


@pytest.mark.parametrize("scale", [1.0, 1 / 64, 1 / 16, 1 / 4])
def test_scaled_loss_gradients_and_coordinate_coefficients(scale):
    layer, cost = make_cost(target_scale=scale)
    y = layer.state
    target = scale * F.one_hot(torch.tensor([0, 4, 5]), 6).to(y)
    loss = 0.5 * ((y[:, 0::2] - y[:, 1::2] - target) ** 2).sum(dim=1)
    expected_grad = torch.autograd.grad(loss.sum(), y)[0]

    torch.testing.assert_close(cost.eval(), loss, rtol=0, atol=0)
    torch.testing.assert_close(cost._grad(layer, mean=False), expected_grad)
    torch.testing.assert_close(cost._grad(layer, mean=True), expected_grad.mean(0))
    torch.testing.assert_close(2 * cost._a_coef() * y + cost._b_coef(), expected_grad)
    torch.testing.assert_close(cost._target, target.float())


def test_default_is_bitwise_compatible_with_unscaled_one_hot():
    layer, cost = make_cost()
    y = layer.state
    target = F.one_hot(torch.tensor([0, 4, 5]), 6).to(y)
    diff = y[:, 0::2] - y[:, 1::2] - target
    assert torch.equal(cost.eval(), 0.5 * (diff * diff).sum(dim=1))
    expected = torch.stack((diff, -diff), dim=-1).flatten(1)
    assert torch.equal(cost._grad(layer, mean=False), expected)


def test_predictions_do_not_depend_on_target_amplitude():
    _, original = make_cost()
    for scale in [1 / 64, 1 / 16, 1 / 4]:
        _, scaled = make_cost(target_scale=scale)
        assert torch.equal(original.error_fn(), scaled.error_fn())
        assert torch.equal(original.top_five_error_fn(), scaled.top_five_error_fn())


@pytest.mark.parametrize("scale", [0, -0.25, float("nan"), float("inf"), -float("inf")])
def test_invalid_target_scale_rejected(scale):
    with pytest.raises(ValueError, match="finite and strictly positive"):
        make_cost(target_scale=scale)


@pytest.mark.parametrize("scale", [1 / 64, 1 / 16, 1 / 4])
def test_image_training_uses_configured_targets(tmp_path, monkeypatch, scale):
    from labs import mnist_train

    config = tiny_config()
    config["target_scale"] = scale
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    x = torch.linspace(-1, 1, 2 * 28 * 28).reshape(2, 1, 28, 28)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, torch.tensor([2, 7])), batch_size=2,
    )

    class SyntheticData:
        def __init__(self, **kwargs):
            pass

        def build(self):
            return loader, loader

    monkeypatch.setattr(mnist_train, "_resolve_callable", lambda name: SyntheticData)
    monkeypatch.setattr(mnist_train, "SummaryWriter", None)
    original_eval = SquaredErrorPairedOutputs.eval
    observed = []

    def checked_eval(self):
        observed.append(self._target_scale)
        y = self._layer.state
        expected = 0.5 * ((y[:, 0::2] - y[:, 1::2]
                          - scale * F.one_hot(self._label, 10).to(y)) ** 2).sum(1)
        actual = original_eval(self)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return actual

    monkeypatch.setattr(SquaredErrorPairedOutputs, "eval", checked_eval)
    run_dir = tmp_path / "run"
    mnist_train.train_mnist_conv(
        config_path, epochs=1, lr=config["lr"], beta=1.0, log_interval=1,
        max_batches=1, max_test_batches=1, device="cpu", output_dir=str(run_dir),
        training_algorithm="BP", seed=0, skip_terminal_official_test=True,
    )
    assert observed and set(observed) == {scale}
    resolved = json.loads((run_dir / "config.json").read_text())
    assert resolved["target_scale"] == scale
    assert resolved["training"]["optimizer_steps_applied"] is True
    assert (run_dir / "final_model.pt").is_file()


def test_training_rejects_scaled_unpaired_output(tmp_path):
    from labs import mnist_train

    config = tiny_config()
    config["target_scale"] = 0.25
    config["model_overrides"]["mnist_bp_conv_amp"]["layer_shapes"][-1] = [10]
    path = tmp_path / "unpaired.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="requires paired outputs"):
        mnist_train.train_mnist_conv(
            path, epochs=1, lr=config["lr"], beta=1.0, log_interval=1,
            max_batches=1, device="cpu", output_dir=str(tmp_path / "run"),
            training_algorithm="BP", seed=0,
        )
