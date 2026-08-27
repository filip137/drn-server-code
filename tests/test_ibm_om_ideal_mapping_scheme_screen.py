from __future__ import annotations

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_ideal_mapping_scheme_screen import (
    Scheme,
    build_scheme_targets,
)


def _targets(
    weight: torch.Tensor,
    *,
    devices: int,
    reference: bool,
    reference_fraction: float,
):
    return build_scheme_targets(
        (weight, weight),
        scheme=Scheme(
            "test",
            devices,
            reference,
            reference_fraction,
        ),
        scale_fractions=(0.5, 0.5),
        conductance_min=0.0,
        conductance_max=1.0,
    )[0]


@pytest.mark.parametrize(
    ("weight", "expected"),
    [
        (torch.tensor([[1.0]]), torch.tensor([[0.625, 0.25], [0.25, 0.625]])),
        (torch.tensor([[-1.0]]), torch.tensor([[0.25, 0.625], [0.625, 0.25]])),
    ],
)
def test_four_device_fixed_reference_has_requested_a_r_pattern(
    weight: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    first, second = _targets(
        weight,
        devices=4,
        reference=True,
        reference_fraction=0.25,
    )
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, expected)


def test_eight_device_uses_difference_for_transfer_and_sum_for_loading() -> None:
    weight = torch.tensor([[1.0]])
    plus_w1, minus_w1, plus_w2, minus_w2 = _targets(
        weight,
        devices=8,
        reference=True,
        reference_fraction=0.25,
    )
    expected_difference = torch.tensor([[0.375, 0.0], [0.0, 0.375]])
    expected_sum = torch.tensor([[0.875, 0.5], [0.5, 0.875]])
    for plus, minus in ((plus_w1, minus_w1), (plus_w2, minus_w2)):
        torch.testing.assert_close(plus - minus, expected_difference)
        torch.testing.assert_close(plus + minus, expected_sum)


def test_zero_reference_collapses_to_lower_bound_control_targets() -> None:
    weight = torch.tensor([[0.3, -0.2], [-0.4, 0.1]])
    for devices in (4, 8):
        control = _targets(
            weight,
            devices=devices,
            reference=False,
            reference_fraction=0.0,
        )
        zero_reference = _targets(
            weight,
            devices=devices,
            reference=True,
            reference_fraction=0.0,
        )
        assert len(control) == len(zero_reference)
        for left, right in zip(control, zero_reference):
            torch.testing.assert_close(left, right)


def test_no_reference_control_rejects_nonzero_reference_fraction() -> None:
    with pytest.raises(ValueError, match="no-reference control"):
        _targets(
            torch.tensor([[1.0]]),
            devices=4,
            reference=False,
            reference_fraction=0.25,
        )
