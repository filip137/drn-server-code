from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from experiments.mnist_relu.model import BiasFreeReluTeacher
from experiments.mnist_relu_drn.components import TeacherKLDivergence
from experiments.mnist_relu_drn.star_hidden_objective import (
    HEALTHY_CLASS_HIDDEN_KL,
    HEALTHY_CLASS_HIDDEN_MSE,
    RELU_SAMPLE_HIDDEN_KL,
    StarInspiredHiddenObjective,
    accumulate_class_prototypes,
    fit_positive_hidden_gain,
    halves_scores,
    paired_scores,
    teacher_to_student_kl,
)
from model.variable.layer import LinearLayer


def _layers(*, batch_size: int = 2) -> tuple[LinearLayer, LinearLayer]:
    return (
        LinearLayer((4,), batch_size=batch_size, device="cpu"),
        LinearLayer((6,), batch_size=batch_size, device="cpu"),
    )


def test_bias_free_teacher_exposes_the_exact_post_relu_representation() -> None:
    teacher = BiasFreeReluTeacher(device=torch.device("cpu"))
    inputs = torch.zeros((2, 784), dtype=torch.float32)
    inputs[0, :2] = torch.tensor([2.0, -1.0])
    inputs[1, :2] = torch.tensor([-0.5, 3.0])
    with torch.no_grad():
        teacher.input_weight.state.zero_()
        teacher.output_weight.state.zero_()
        teacher.input_weight.state[:2, :2] = torch.tensor(
            [[1.0, -2.0], [0.5, 1.0]]
        )
        teacher.output_weight.state[:2, :2] = torch.tensor(
            [[1.5, -0.5], [-2.0, 3.0]]
        )

    expected_hidden = torch.relu(inputs @ teacher.input_weight.state)
    torch.testing.assert_close(teacher.hidden(inputs), expected_hidden)
    torch.testing.assert_close(teacher.representations(inputs), expected_hidden)
    torch.testing.assert_close(
        teacher.logits(inputs),
        expected_hidden @ teacher.output_weight.state,
    )


def test_hidden_halves_and_output_pairs_have_distinct_decoders() -> None:
    hidden = torch.tensor([[3.0, 5.0, 1.0, 2.0]])
    output = torch.tensor([[3.0, 1.0, 5.0, 2.0, 4.0, 9.0]])
    torch.testing.assert_close(halves_scores(hidden), torch.tensor([[2.0, 3.0]]))
    torch.testing.assert_close(
        paired_scores(output),
        torch.tensor([[2.0, 3.0, -5.0]]),
    )


def test_hidden_kl_is_teacher_to_student_and_includes_temperature_squared() -> None:
    teacher = torch.tensor([[2.0, -1.0, 0.5], [0.3, 1.2, -0.7]])
    student = torch.tensor([[0.1, 1.0, -0.4], [-0.5, 0.4, 1.4]])
    temperature = 2.5
    expected = temperature**2 * F.kl_div(
        F.log_softmax(student / temperature, dim=1),
        F.softmax(teacher / temperature, dim=1),
        reduction="none",
    ).sum(dim=1)
    reverse = temperature**2 * F.kl_div(
        F.log_softmax(teacher / temperature, dim=1),
        F.softmax(student / temperature, dim=1),
        reduction="none",
    ).sum(dim=1)

    actual = teacher_to_student_kl(
        teacher,
        student,
        temperature=temperature,
    )
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, reverse)


def test_positive_hidden_gain_fit_recovers_known_scale() -> None:
    generator = torch.Generator().manual_seed(71)
    student = torch.randn((96, 8), generator=generator)
    teacher = 3.0 * student
    result = fit_positive_hidden_gain(
        student,
        teacher,
        temperature=1.7,
        gain_min=0.25,
        gain_max=8.0,
        steps=121,
    )
    assert result["gain"] == pytest.approx(3.0, rel=0.025)
    assert result["calibrated_kl"] < result["raw_kl"]
    assert result["temperature"] == 1.7


def test_class_prototypes_use_train_split_float64_and_are_batch_order_stable() -> None:
    first = (
        torch.tensor(
            [[2.0, 4.0, 1.0, 2.0], [10.0, 8.0, 3.0, 1.0]],
            dtype=torch.float32,
        ),
        torch.tensor([0, 1]),
    )
    second = (
        torch.tensor(
            [[4.0, 8.0, 1.0, 4.0], [14.0, 6.0, 5.0, 1.0]],
            dtype=torch.float32,
        ),
        torch.tensor([0, 1]),
    )
    forward = accumulate_class_prototypes(
        [first, second],
        num_classes=2,
        split="train",
        expected_examples=4,
    )
    reverse = accumulate_class_prototypes(
        [second, first],
        num_classes=2,
        split="train",
        expected_examples=4,
    )

    assert forward.raw_means.dtype == torch.float64
    assert forward.raw_means.device.type == "cpu"
    assert torch.equal(forward.class_counts, torch.tensor([2, 2]))
    assert forward.examples == 4
    torch.testing.assert_close(forward.raw_means, reverse.raw_means)
    torch.testing.assert_close(
        forward.decoded_means,
        halves_scores(forward.raw_means),
    )
    with pytest.raises(ValueError, match="split='train'"):
        accumulate_class_prototypes(
            [first, second],
            num_classes=2,
            split="validation",
        )


def test_lambda_zero_is_exact_output_kl_value_and_output_gradient_parity() -> None:
    hidden, output = _layers()
    hidden.state = torch.tensor(
        [[0.5, 0.7, 0.1, 0.2], [0.3, 0.9, 0.4, 0.2]],
        requires_grad=True,
    )
    output.state = torch.tensor(
        [[0.3, -0.2, 0.1, 0.0, -0.4, 0.2], [0.2, 0.1, -0.1, 0.5, 0.4, -0.3]],
        requires_grad=True,
    )
    teacher_logits = torch.tensor([[1.0, -0.5, 0.2], [-0.3, 0.7, 0.1]])
    labels = torch.tensor([0, 1])
    baseline = TeacherKLDivergence(output, gain=2.5)
    baseline.set_teacher(teacher_logits, labels)
    composite = StarInspiredHiddenObjective(
        hidden,
        output,
        hidden_objective=RELU_SAMPLE_HIDDEN_KL,
        hidden_weight=0.0,
        output_gain=2.5,
        hidden_gain=4.0,
        hidden_temperature=3.0,
    )
    # Deliberately do not bind a hidden target: lambda=0 must remain a pure
    # TeacherKLDivergence control.
    composite.set_teacher(teacher_logits, labels)

    baseline_value = baseline.eval()
    composite_value = composite.eval()
    assert torch.equal(composite_value, baseline_value)
    baseline_gradient = torch.autograd.grad(
        baseline_value.mean(),
        output.state,
        retain_graph=True,
    )[0]
    composite_gradient = torch.autograd.grad(
        composite_value.mean(),
        output.state,
    )[0]
    assert torch.equal(composite_gradient, baseline_gradient)


def test_raw_prototype_mse_uses_label_lookup_and_half_mean_normalization() -> None:
    hidden, output = _layers()
    hidden.state = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
        requires_grad=True,
    )
    output.state = torch.zeros((2, 6), requires_grad=True)
    prototypes = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, 13.0],
            [2.0, 3.0, 4.0, 5.0],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([2, 0])
    objective = StarInspiredHiddenObjective(
        hidden,
        output,
        hidden_objective=HEALTHY_CLASS_HIDDEN_MSE,
        hidden_weight=1.0,
        class_prototypes=prototypes,
    )
    objective.set_teacher(torch.zeros((2, 3)), labels)

    expected_targets = prototypes[labels].to(torch.float32)
    expected = 0.5 * (hidden.state - expected_targets).square().mean(dim=1)
    torch.testing.assert_close(objective.hidden_loss(), expected)


def test_relu_sample_objective_applies_frozen_gain_and_temperature() -> None:
    hidden, output = _layers()
    hidden.state = torch.tensor(
        [[2.0, 4.0, 1.0, 1.0], [3.0, 2.0, 1.0, 0.5]],
        requires_grad=True,
    )
    output.state = torch.zeros((2, 6), requires_grad=True)
    teacher_hidden = torch.tensor([[0.5, 2.5], [1.0, -0.5]])
    objective = StarInspiredHiddenObjective(
        hidden,
        output,
        hidden_objective=RELU_SAMPLE_HIDDEN_KL,
        hidden_weight=0.8,
        hidden_gain=0.4,
        hidden_temperature=2.0,
    )
    objective.set_teacher(
        torch.zeros((2, 3)),
        torch.tensor([0, 1]),
        teacher_hidden,
    )
    expected = teacher_to_student_kl(
        teacher_hidden,
        0.4 * halves_scores(hidden.state),
        temperature=2.0,
    )
    torch.testing.assert_close(objective.hidden_loss(), expected)


def test_prototype_kl_decodes_class_targets_and_summed_loss_reaches_both_layers() -> None:
    hidden, output = _layers()
    hidden.state = torch.tensor(
        [[2.0, 4.0, 1.0, 1.0], [1.0, 3.0, 2.0, 1.0]],
        requires_grad=True,
    )
    output.state = torch.tensor(
        [[0.5, 0.0, 0.2, -0.1, 0.4, -0.2], [0.2, 0.1, 0.7, 0.0, -0.1, 0.3]],
        requires_grad=True,
    )
    prototypes = torch.tensor(
        [
            [3.0, 5.0, 1.0, 2.0],
            [7.0, 2.0, 4.0, 1.0],
            [2.0, 6.0, 1.0, 3.0],
        ]
    )
    labels = torch.tensor([2, 0])
    objective = StarInspiredHiddenObjective(
        hidden,
        output,
        hidden_objective=HEALTHY_CLASS_HIDDEN_KL,
        hidden_weight=0.7,
        output_gain=1.3,
        class_prototypes=prototypes,
    )
    objective.set_teacher(
        torch.tensor([[1.0, 0.0, -0.5], [-0.2, 0.8, 0.1]]),
        labels,
    )

    expected_hidden = teacher_to_student_kl(
        halves_scores(prototypes.index_select(0, labels)),
        halves_scores(hidden.state),
        temperature=1.0,
    )
    torch.testing.assert_close(objective.hidden_loss(), expected_hidden)
    components = objective.component_losses()
    torch.testing.assert_close(
        components["total"],
        components["output"] + 0.7 * components["hidden"],
    )
    hidden_gradient, output_gradient = torch.autograd.grad(
        components["total"].mean(),
        (hidden.state, output.state),
    )
    assert float(hidden_gradient.abs().sum()) > 0.0
    assert float(output_gradient.abs().sum()) > 0.0
