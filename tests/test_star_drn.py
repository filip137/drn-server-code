from unittest.mock import patch

import pytest
import torch

from model.function.interaction import QFunction
from model.resistive.layer import NonlinearResistiveLayer
from model.variable.layer import LinearLayer
from training.sgd import AugmentedFunction
from training.star_drn import (
    DRN_LOGICAL_RAIL_DIFFERENCE_ABLATION,
    DrnLocalStarNudge,
    convert_raw_rails_to_logical_difference_ablation,
)


def _layers() -> tuple[NonlinearResistiveLayer, LinearLayer]:
    hidden = NonlinearResistiveLayer(
        (4,),
        batch_size=2,
        device="cpu",
        non_linearity="perfect_diode",
    )
    output = LinearLayer((4,), batch_size=2, device="cpu")
    hidden.state = torch.tensor(
        [[0.8, 0.2, -0.3, -0.7], [0.4, 0.9, -0.6, -0.1]],
        dtype=torch.float32,
    )
    output.state = torch.tensor(
        [[0.5, -0.2, 0.1, -0.4], [-0.1, 0.3, 0.7, -0.5]],
        dtype=torch.float32,
    )
    return hidden, output


def _target_tables() -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.tensor(
        [
            [0.1, 0.2, -0.3, -0.4],
            [0.5, 0.6, -0.7, -0.8],
            [0.9, 1.0, -1.1, -1.2],
        ],
        dtype=torch.float64,
    )
    output = torch.tensor(
        [
            [0.2, -0.1, 0.4, -0.3],
            [0.6, -0.5, 0.8, -0.7],
            [1.0, -0.9, 1.2, -1.1],
        ],
        dtype=torch.float64,
    )
    return hidden, output


def test_raw_rail_nudge_has_exact_local_quadratic_interface() -> None:
    hidden, output = _layers()
    hidden_means, output_means = _target_tables()
    nudge = DrnLocalStarNudge(
        hidden,
        output,
        hidden_means,
        output_means,
        hidden_gain=0.25,
        output_gain=1.5,
    )
    labels = torch.tensor([2, 0], dtype=torch.int64)
    nudge.set_target(labels)

    hidden_target = hidden_means[labels].to(torch.float32)
    output_target = output_means[labels].to(torch.float32)
    expected_hidden_gradient = 0.25 * (hidden.state - hidden_target)
    expected_output_gradient = 1.5 * (output.state - output_target)
    expected_loss = (
        0.5 * 0.25 * (hidden.state - hidden_target).square().sum(dim=1)
        + 0.5 * 1.5 * (output.state - output_target).square().sum(dim=1)
    )

    # The analytic interface must remain independent of autograd/BPTT.
    with patch("torch.autograd.grad", side_effect=AssertionError("autograd used")):
        torch.testing.assert_close(nudge.eval(), expected_loss)
        torch.testing.assert_close(
            nudge.grad_layer_fn(hidden)(),
            expected_hidden_gradient,
        )
        torch.testing.assert_close(
            nudge.grad_layer_fn(output)(),
            expected_output_gradient,
        )
        assert nudge.a_coef_fn(hidden)() == pytest.approx(0.125)
        assert nudge.a_coef_fn(output)() == pytest.approx(0.75)
        torch.testing.assert_close(
            nudge.b_coef_fn(hidden)(),
            -0.25 * hidden_target,
        )
        torch.testing.assert_close(
            nudge.b_coef_fn(output)(),
            -1.5 * output_target,
        )

    torch.testing.assert_close(
        2.0 * nudge.a_coef_fn(hidden)() * hidden.state
        + nudge.b_coef_fn(hidden)(),
        expected_hidden_gradient,
    )
    torch.testing.assert_close(
        2.0 * nudge.a_coef_fn(output)() * output.state
        + nudge.b_coef_fn(output)(),
        expected_output_gradient,
    )
    assert nudge.layers() == [hidden, output]
    assert nudge.params() == []
    assert nudge.num_classes == 3


class _IndependentQuadraticEnergy(QFunction):
    def __init__(self, hidden, output):
        self.hidden = hidden
        self.output = output
        super().__init__([hidden, output], [])

    def eval(self):
        return 0.5 * (
            self.hidden.state.square().sum(dim=1)
            + self.output.state.square().sum(dim=1)
        )

    def grad_layer_fn(self, layer):
        if layer not in (self.hidden, self.output):
            raise ValueError
        return lambda: layer.state

    def a_coef_fn(self, layer):
        if layer not in (self.hidden, self.output):
            raise ValueError
        return lambda: 0.5

    def b_coef_fn(self, layer):
        if layer not in (self.hidden, self.output):
            raise ValueError
        return lambda: torch.zeros_like(layer.state)


def test_nudge_composes_with_augmented_function_for_centered_ep() -> None:
    hidden, output = _layers()
    hidden_means, output_means = _target_tables()
    nudge = DrnLocalStarNudge(
        hidden,
        output,
        hidden_means,
        output_means,
        hidden_gain=0.4,
        output_gain=0.8,
    )
    labels = torch.tensor([1, 2], dtype=torch.int64)
    nudge.set_target(labels)
    energy = _IndependentQuadraticEnergy(hidden, output)
    augmented = AugmentedFunction(energy, nudge, nudging_mode="cost")
    augmented.nudging = -0.2

    expected_eval = energy.eval() - 0.2 * nudge.eval()
    torch.testing.assert_close(augmented.eval(), expected_eval)
    for layer, gain in ((hidden, 0.4), (output, 0.8)):
        expected_gradient = layer.state - 0.2 * nudge.grad_layer_fn(layer)()
        torch.testing.assert_close(
            augmented.grad_layer_fn(layer)(),
            expected_gradient,
        )
        assert augmented.a_coef_fn(layer)() == pytest.approx(
            0.5 - 0.2 * 0.5 * gain
        )
        torch.testing.assert_close(
            augmented.b_coef_fn(layer)(),
            -0.2 * nudge.b_coef_fn(layer)(),
        )


def test_target_rows_are_label_addressed_and_copied() -> None:
    hidden, output = _layers()
    hidden_means, output_means = _target_tables()
    nudge = DrnLocalStarNudge(
        hidden,
        output,
        hidden_means,
        output_means,
        hidden_gain=1.0,
        output_gain=1.0,
    )
    hidden_means.zero_()
    output_means.zero_()

    nudge.set_target(torch.tensor([0, 1], dtype=torch.int64))
    first = nudge.b_coef_fn(hidden)().clone()
    nudge.set_target(torch.tensor([2, 0], dtype=torch.int64))
    second = nudge.b_coef_fn(hidden)()

    original_hidden, _ = _target_tables()
    torch.testing.assert_close(first, -original_hidden[[0, 1]].to(torch.float32))
    torch.testing.assert_close(second, -original_hidden[[2, 0]].to(torch.float32))
    assert not torch.equal(first, second)


def test_logical_difference_is_explicit_named_ablation() -> None:
    hidden = torch.tensor(
        [[1.0, 2.0, 3.0, -4.0, -5.0, -6.0]],
        dtype=torch.float32,
    )
    output = torch.tensor(
        [[0.7, -0.2, 1.1, 0.4]],
        dtype=torch.float32,
    )
    converted = convert_raw_rails_to_logical_difference_ablation(hidden, output)

    assert converted.representation_id == DRN_LOGICAL_RAIL_DIFFERENCE_ABLATION
    torch.testing.assert_close(
        converted.hidden_logical,
        torch.tensor([[5.0, 7.0, 9.0]]),
    )
    torch.testing.assert_close(
        converted.output_logical,
        torch.tensor([[0.9, 0.7]]),
    )


def test_nudge_fails_closed_on_unset_or_invalid_labels_and_layers() -> None:
    hidden, output = _layers()
    hidden_means, output_means = _target_tables()
    nudge = DrnLocalStarNudge(
        hidden,
        output,
        hidden_means,
        output_means,
        hidden_gain=1.0,
        output_gain=1.0,
    )

    with pytest.raises(RuntimeError, match="call set_target"):
        nudge.eval()
    with pytest.raises(ValueError, match="int64"):
        nudge.set_target(torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        nudge.set_target(torch.tensor([0, 3], dtype=torch.int64))
    other = LinearLayer((4,), batch_size=2, device="cpu")
    with pytest.raises(ValueError, match="declared hidden or output"):
        nudge.grad_layer_fn(other)


@pytest.mark.parametrize(
    "hidden, output, message",
    [
        (torch.zeros(2, 3), torch.zeros(2, 4), "hidden raw rails"),
        (torch.zeros(2, 4), torch.zeros(3, 4), "share a batch size"),
        (
            torch.tensor([[0.0, float("nan")]]),
            torch.zeros(1, 2),
            "hidden raw rails",
        ),
    ],
)
def test_logical_difference_ablation_validates_raw_rails(
    hidden: torch.Tensor,
    output: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        convert_raw_rails_to_logical_difference_ablation(hidden, output)
