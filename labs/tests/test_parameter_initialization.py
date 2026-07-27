import math

import pytest
import torch

from model.variable.parameter import ConvWeight, DenseWeight


def _make_weight(kind, *, gain=1.0, lower=179e-6, upper=180e-6, mode="bounded_uniform"):
    if kind == "dense":
        return DenseWeight(
            (256,),
            (256,),
            gain=gain,
            device="cpu",
            clamp=True,
            clamp_min=lower,
            clamp_max=upper,
            init_mode=mode,
        )
    if kind == "conv":
        return ConvWeight(
            (128, 64, 3, 3),
            gain=gain,
            device="cpu",
            clamp=True,
            clamp_min=lower,
            clamp_max=upper,
            init_mode=mode,
        )
    raise AssertionError(f"Unexpected test weight kind: {kind!r}")


def _make_unclamped_kaiming_weight(kind, gain):
    if kind == "dense":
        return DenseWeight(
            (64,),
            (32,),
            gain=gain,
            device="cpu",
            clamp=False,
            init_mode="kaiming_uniform",
        )
    if kind == "conv":
        return ConvWeight(
            (32, 2, 3, 3),
            gain=gain,
            device="cpu",
            clamp=False,
            clamp_min=None,
            clamp_max=None,
            init_mode="kaiming_uniform",
        )
    raise AssertionError(f"Unexpected test weight kind: {kind!r}")


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_bounded_uniform_samples_directly_inside_narrow_nonzero_bounds(kind):
    lower = 179e-6
    upper = 180e-6
    span = upper - lower
    torch.manual_seed(1234)

    values = _make_weight(kind, lower=lower, upper=upper).state

    assert bool((values >= lower).all())
    assert bool((values <= upper).all())
    assert float(values.min()) > lower
    assert float(values.max()) < upper
    assert not bool((values == lower).any())
    assert not bool((values == upper).any())

    expected_mean = 0.5 * (lower + upper)
    expected_std = span / math.sqrt(12.0)
    assert float(values.mean()) == pytest.approx(expected_mean, abs=0.01 * span)
    assert float(values.std(unbiased=False)) == pytest.approx(
        expected_std,
        rel=0.02,
    )


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_bounded_uniform_uses_bounds_instead_of_gain(kind):
    torch.manual_seed(7)
    small_gain = _make_weight(kind, gain=1e-6).state.clone()
    torch.manual_seed(7)
    large_gain = _make_weight(kind, gain=1e6).state.clone()

    assert torch.equal(small_gain, large_gain)


@pytest.mark.parametrize(
    ("kind", "fan_in"),
    [
        ("dense", 256),
        ("conv", 64 * 3 * 3),
    ],
)
def test_bounded_kaiming_uniform_has_expected_centered_he_moments(
    kind,
    fan_in,
):
    lower = 1e-5
    upper = 1e-4
    midpoint = 0.5 * (lower + upper)
    half_span = 0.5 * (upper - lower)
    torch.manual_seed(2026)

    values = _make_weight(
        kind,
        lower=lower,
        upper=upper,
        mode="bounded_kaiming_uniform",
    ).state

    expected_half_width = half_span * math.sqrt(6.0 / fan_in)
    expected_std = half_span * math.sqrt(2.0 / fan_in)
    assert float(values.min()) > midpoint - expected_half_width
    assert float(values.max()) < midpoint + expected_half_width
    assert not bool((values == lower).any())
    assert not bool((values == upper).any())
    assert float(values.mean()) == pytest.approx(
        midpoint,
        abs=0.01 * expected_half_width,
    )
    assert float(values.std(unbiased=False)) == pytest.approx(
        expected_std,
        rel=0.02,
    )


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_bounded_kaiming_uniform_gain_scales_deviation_from_midpoint(kind):
    lower = 1e-5
    upper = 1e-4
    midpoint = 0.5 * (lower + upper)
    scale = 0.25
    torch.manual_seed(73)
    unit_gain = _make_weight(
        kind,
        gain=1.0,
        lower=lower,
        upper=upper,
        mode="bounded_kaiming_uniform",
    ).state.clone()
    torch.manual_seed(73)
    scaled_gain = _make_weight(
        kind,
        gain=scale,
        lower=lower,
        upper=upper,
        mode="bounded_kaiming_uniform",
    ).state.clone()

    assert torch.allclose(
        scaled_gain - midpoint,
        scale * (unit_gain - midpoint),
        rtol=1e-5,
        atol=1e-11,
    )


@pytest.mark.parametrize("kind", ["dense", "conv"])
@pytest.mark.parametrize(
    ("lower", "upper", "gain"),
    [
        (None, 1.0, 1.0),
        (0.0, None, 1.0),
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (0.0, float("inf"), 1.0),
        (float("nan"), 1.0, 1.0),
        (0.0, 1.0, float("nan")),
        (0.0, 1.0, -1.0),
        (0.0, 1.0, 100.0),
    ],
)
def test_bounded_kaiming_uniform_rejects_invalid_contract(
    kind,
    lower,
    upper,
    gain,
):
    with pytest.raises(
        ValueError,
        match=(
            r"Expected bounded_kaiming_uniform initialization.*Provided value"
        ),
    ):
        _make_weight(
            kind,
            lower=lower,
            upper=upper,
            gain=gain,
            mode="bounded_kaiming_uniform",
        )


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_kaiming_uniform_gain_scales_the_same_random_draw(kind):
    scale = 0.03125
    torch.manual_seed(19)
    unit_gain = _make_unclamped_kaiming_weight(kind, gain=1.0).state.clone()
    torch.manual_seed(19)
    scaled_gain = _make_unclamped_kaiming_weight(kind, gain=scale).state.clone()

    assert torch.allclose(scaled_gain, scale * unit_gain, rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("kind", ["dense", "conv"])
@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (None, 1.0),
        (0.0, None),
        (1.0, 1.0),
        (2.0, 1.0),
        (0.0, float("inf")),
        (float("nan"), 1.0),
    ],
)
def test_bounded_uniform_rejects_missing_nonfinite_or_degenerate_bounds(
    kind,
    lower,
    upper,
):
    with pytest.raises(
        ValueError,
        match=r"Expected bounded_uniform initialization bounds.*Provided value",
    ):
        _make_weight(kind, lower=lower, upper=upper)


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_weight_initialization_rejects_unknown_mode(kind):
    with pytest.raises(
        ValueError,
        match=r"Expected init_mode to be one of.*Provided value: 'kaiming_unifrom'",
    ):
        _make_weight(kind, mode="kaiming_unifrom")
