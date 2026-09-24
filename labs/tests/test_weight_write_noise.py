from __future__ import annotations

import json
import math

import pytest
import torch

import experiments.weight_write_noise as write_noise
from model.variable.parameter import Bias, ConvWeight, DenseWeight, PoolWeight


class _FakeParameter:
    def __init__(
        self,
        name,
        values,
        *,
        lower=0.0,
        upper=1.0,
        dtype=torch.float64,
        device="cpu",
        requires_grad=False,
    ):
        self.name = name
        self.min_cond = lower
        self.max_cond = upper
        self.state = (
            torch.as_tensor(values, dtype=dtype, device=device)
            .clone()
            .detach()
            .requires_grad_(requires_grad)
        )


def _receipt_by_name(receipt):
    return {row["canonical_name"]: row for row in receipt["parameters"]}


def test_zero_eta_is_bitwise_base_identity_and_excludes_nonweights():
    weight = _FakeParameter(
        "DenseWeight_7",
        [0.9, 0.8, 0.7],
        requires_grad=True,
    )
    excluded = _FakeParameter("Bias_0", [0.77, 0.66], lower=None, upper=None)
    weight_base = torch.tensor([0.1, 0.25, 1.0], dtype=torch.float64)
    excluded_base = torch.zeros(2, dtype=torch.float64)
    weight_state_identity = weight.state
    excluded_before = excluded.state.clone()

    torch.manual_seed(991)
    global_rng_before = torch.random.get_rng_state().clone()
    receipt = write_noise.apply_weight_write_noise(
        [weight, excluded],
        [weight_base, excluded_base],
        eta=0.0,
        programming_seed=19,
    )

    assert weight.state is weight_state_identity
    assert weight.state.requires_grad
    assert torch.equal(weight.state, weight_base)
    assert torch.equal(
        weight.state.contiguous().view(torch.uint8),
        weight_base.contiguous().view(torch.uint8),
    )
    assert torch.equal(excluded.state, excluded_before)
    assert torch.equal(torch.random.get_rng_state(), global_rng_before)
    assert receipt["selected_parameter_count"] == 1
    assert receipt["input_parameter_count"] == 2
    assert receipt["eta"] == 0.0
    assert receipt["rms_error"] == 0.0
    assert receipt["mean_absolute_error"] == 0.0
    assert receipt["signed_mean_error"] == 0.0
    assert receipt["rms_error_fraction"] == 0.0
    assert receipt["mean_absolute_error_fraction"] == 0.0
    assert receipt["signed_mean_error_fraction"] == 0.0
    assert receipt["clipping_fraction"] == 0.0
    assert receipt["new_clipping_fraction"] == 0.0
    assert receipt["parameters"][0]["canonical_name"] == "DenseWeight_7"
    assert receipt["parameters"][0]["written_sha256"] == receipt[
        "parameters"
    ][0]["target_sha256"]
    assert receipt["written_sha256"] == receipt["target_sha256"]
    json.dumps(receipt, sort_keys=True, allow_nan=False)


def test_keyed_draws_are_common_across_order_eta_dtype_bounds_and_targets():
    left_a = _FakeParameter(
        "ConvWeight_2",
        [0.2, 0.3, 0.4, 0.5],
        lower=0.0,
        upper=1.0,
        dtype=torch.float32,
    )
    left_b = _FakeParameter(
        "DenseWeight_1",
        [0.5, 0.75, 1.0],
        lower=0.0,
        upper=2.0,
        dtype=torch.float64,
    )
    left_receipt = write_noise.apply_weight_write_noise(
        [left_a, left_b],
        [left_a.state.clone(), left_b.state.clone()],
        eta=0.05,
        programming_seed=73,
    )

    excluded = _FakeParameter("PoolWeight_0", [0.4], lower=0.0, upper=1.0)
    right_a = _FakeParameter(
        "ConvWeight_2",
        [2.0, 3.0, 4.0, 5.0],
        lower=0.0,
        upper=10.0,
        dtype=torch.float64,
    )
    right_b = _FakeParameter(
        "DenseWeight_1",
        [0.2, 0.3, 0.4],
        lower=0.0,
        upper=1.0,
        dtype=torch.float32,
    )
    right_receipt = write_noise.apply_weight_write_noise(
        [right_b, excluded, right_a],
        [right_b.state.clone(), excluded.state.clone(), right_a.state.clone()],
        eta=0.2,
        programming_seed=73,
    )

    left = _receipt_by_name(left_receipt)
    right = _receipt_by_name(right_receipt)
    assert left["ConvWeight_2"]["draw_sha256"] == right["ConvWeight_2"][
        "draw_sha256"
    ]
    assert left["DenseWeight_1"]["draw_sha256"] == right["DenseWeight_1"][
        "draw_sha256"
    ]
    assert left_receipt["draw_sha256"] == right_receipt["draw_sha256"]
    assert [row["canonical_name"] for row in right_receipt["parameters"]] == [
        "ConvWeight_2",
        "DenseWeight_1",
    ]
    assert left_receipt["written_sha256"] != right_receipt["written_sha256"]


def test_key_components_change_the_draw():
    def draw_hash(name, shape, seed):
        parameter = _FakeParameter(name, torch.full(shape, 0.5))
        receipt = write_noise.apply_weight_write_noise(
            [parameter],
            [parameter.state.clone()],
            eta=0.1,
            programming_seed=seed,
        )
        return receipt["parameters"][0]["draw_sha256"]

    reference = draw_hash("DenseWeight_0", (5,), 11)
    assert (
        reference
        == "6b027fe3790d6adaa94f536daee336c58e8632434a91d18662ce7def291e6555"
    )
    assert draw_hash("DenseWeight_0", (5,), 11) == reference
    assert draw_hash("DenseWeight_1", (5,), 11) != reference
    assert draw_hash("DenseWeight_0", (6,), 11) != reference
    assert draw_hash("DenseWeight_0", (5,), 12) != reference


def test_formula_clipping_rail_and_error_receipts(monkeypatch):
    draw = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64)

    def fixed_draw(*, programming_seed, canonical_name, shape):
        assert programming_seed == 5
        assert canonical_name == "DenseWeight_3"
        assert shape == (4,)
        return draw.clone(), "a" * 64, 123

    monkeypatch.setattr(write_noise, "_parameter_draw", fixed_draw)
    parameter = _FakeParameter("DenseWeight_3", [0.8, 0.8, 0.8, 0.8])
    base = torch.tensor([0.0, 0.1, 0.9, 1.0], dtype=torch.float64)

    receipt = write_noise.apply_weight_write_noise(
        [parameter],
        [base],
        eta=0.5,
        programming_seed=5,
    )
    row = receipt["parameters"][0]

    assert torch.equal(parameter.state, torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert row["requested_sigma"] == 0.5
    assert row["pre_existing_lower_rail_count"] == 1
    assert row["pre_existing_upper_rail_count"] == 1
    assert row["pre_existing_rail_fraction"] == 0.5
    assert row["pre_existing_interior_count"] == 2
    assert row["pre_existing_interior_fraction"] == 0.5
    assert row["final_lower_rail_count"] == 2
    assert row["final_upper_rail_count"] == 2
    assert row["final_rail_fraction"] == 1.0
    assert row["lower_clipping_count"] == 2
    assert row["upper_clipping_count"] == 2
    assert row["lower_clipping_fraction"] == 0.5
    assert row["upper_clipping_fraction"] == 0.5
    assert row["clipping_count"] == 4
    assert row["clipping_fraction"] == 1.0
    assert row["new_lower_clipping_count"] == 1
    assert row["new_upper_clipping_count"] == 1
    assert row["new_lower_clipping_fraction"] == 0.25
    assert row["new_upper_clipping_fraction"] == 0.25
    assert row["new_clipping_count"] == 2
    assert row["new_clipping_fraction"] == 0.5
    assert row["new_lower_clipping_fraction_of_interior"] == 0.5
    assert row["new_upper_clipping_fraction_of_interior"] == 0.5
    assert row["new_clipping_fraction_of_interior"] == 1.0
    assert row["signed_mean_error"] == pytest.approx(0.0, abs=1e-15)
    assert row["mean_absolute_error"] == pytest.approx(0.05)
    assert row["rms_error"] == pytest.approx(math.sqrt(0.005))
    assert row["signed_mean_error_fraction"] == pytest.approx(0.0, abs=1e-15)
    assert row["mean_absolute_error_fraction"] == pytest.approx(0.05)
    assert row["rms_error_fraction"] == pytest.approx(math.sqrt(0.005))
    for key in (
        "requested_sigma",
        "pre_existing_rail_fraction",
        "pre_existing_interior_fraction",
        "final_rail_fraction",
        "lower_clipping_fraction",
        "upper_clipping_fraction",
        "clipping_fraction",
        "new_lower_clipping_fraction",
        "new_upper_clipping_fraction",
        "new_clipping_fraction",
        "new_lower_clipping_fraction_of_interior",
        "new_upper_clipping_fraction_of_interior",
        "new_clipping_fraction_of_interior",
        "signed_mean_error",
        "mean_absolute_error",
        "rms_error",
        "signed_mean_error_fraction",
        "mean_absolute_error_fraction",
        "rms_error_fraction",
    ):
        assert receipt[key] == row[key]


def test_crossing_from_one_rail_to_the_other_is_not_new_interior_clipping(
    monkeypatch,
):
    draw = torch.tensor([3.0, -3.0], dtype=torch.float64)

    def fixed_draw(*, programming_seed, canonical_name, shape):
        return draw.clone(), "b" * 64, 456

    monkeypatch.setattr(write_noise, "_parameter_draw", fixed_draw)
    parameter = _FakeParameter("DenseWeight_4", [0.5, 0.5])
    base = torch.tensor([0.0, 1.0], dtype=torch.float64)

    receipt = write_noise.apply_weight_write_noise(
        [parameter],
        [base],
        eta=1.0,
        programming_seed=2,
    )

    assert torch.equal(parameter.state, torch.tensor([1.0, 0.0]))
    assert receipt["clipping_count"] == 2
    assert receipt["pre_existing_interior_count"] == 0
    assert receipt["new_clipping_count"] == 0
    assert receipt["new_clipping_fraction_of_interior"] == 0.0


def test_aggregate_error_fractions_are_element_weighted_across_spans(monkeypatch):
    def unit_draw(*, programming_seed, canonical_name, shape):
        return torch.ones(shape, dtype=torch.float64), canonical_name * 4, 17

    monkeypatch.setattr(write_noise, "_parameter_draw", unit_draw)
    narrow = _FakeParameter(
        "DenseWeight_0",
        [0.5],
        lower=0.0,
        upper=1.0,
    )
    wide = _FakeParameter(
        "ConvWeight_0",
        [1.0, 1.0, 1.0],
        lower=0.0,
        upper=2.0,
    )
    bases = [narrow.state.clone(), wide.state.clone()]

    receipt = write_noise.apply_weight_write_noise(
        [narrow, wide],
        bases,
        eta=0.1,
        programming_seed=9,
    )

    expected_physical_rms = math.sqrt((0.1**2 + 3 * 0.2**2) / 4)
    assert receipt["signed_mean_error"] == pytest.approx(0.175)
    assert receipt["mean_absolute_error"] == pytest.approx(0.175)
    assert receipt["rms_error"] == pytest.approx(expected_physical_rms)
    assert receipt["requested_sigma"] == pytest.approx(expected_physical_rms)
    assert receipt["signed_mean_error_fraction"] == pytest.approx(0.1)
    assert receipt["mean_absolute_error_fraction"] == pytest.approx(0.1)
    assert receipt["rms_error_fraction"] == pytest.approx(0.1)


def test_actual_weight_types_are_selected_and_bias_pool_are_untouched():
    conv = ConvWeight(
        (1, 1, 1, 3),
        gain=1.0,
        device="cpu",
        clamp=True,
        clamp_min=0.0,
        clamp_max=1.0,
        init_mode="bounded_uniform",
    )
    dense = DenseWeight(
        (2,),
        (2,),
        gain=1.0,
        device="cpu",
        clamp=True,
        clamp_min=0.0,
        clamp_max=1.0,
        init_mode="bounded_uniform",
    )
    bias = Bias((2,), gain=0.1, device="cpu")
    pool = PoolWeight(
        (1, 1, 1, 1),
        gain=0.5,
        device="cpu",
        clamp=True,
        clamp_min=0.0,
        clamp_max=1.0,
    )
    # Even an accidentally misleading bias name must not make it eligible.
    bias.name = "DenseWeight_999"

    parameters = [conv, bias, dense, pool]
    bases = [parameter.state.detach().clone() for parameter in parameters]
    bias_before = bias.state.clone()
    pool_before = pool.state.clone()
    receipt = write_noise.apply_weight_write_noise(
        parameters,
        bases,
        eta=0.0,
        programming_seed=0,
    )

    assert receipt["selected_parameter_count"] == 2
    assert {
        row["canonical_name"] for row in receipt["parameters"]
    } == {conv.name.strip(), dense.name.strip()}
    assert torch.equal(bias.state, bias_before)
    assert torch.equal(pool.state, pool_before)


def test_replay_always_uses_clean_base_and_preserves_state_identity():
    parameter = _FakeParameter(
        "ConvWeight_4",
        [0.2, 0.4, 0.6, 0.8],
        dtype=torch.float32,
        requires_grad=True,
    )
    base = parameter.state.detach().clone()
    state_identity = parameter.state

    first = write_noise.apply_weight_write_noise(
        [parameter],
        [base],
        eta=0.3,
        programming_seed=101,
    )
    first_written = parameter.state.detach().clone()
    second = write_noise.apply_weight_write_noise(
        [parameter],
        [base],
        eta=0.3,
        programming_seed=101,
    )

    assert parameter.state is state_identity
    assert parameter.state.requires_grad
    assert torch.equal(parameter.state, first_written)
    assert second["written_sha256"] == first["written_sha256"]
    assert second["draw_sha256"] == first["draw_sha256"]


@pytest.mark.parametrize("eta", [-1.0, float("nan"), float("inf"), True])
def test_invalid_eta_fails_without_mutation(eta):
    parameter = _FakeParameter("DenseWeight_0", [0.9])
    before = parameter.state.clone()
    with pytest.raises(ValueError, match="eta"):
        write_noise.apply_weight_write_noise(
            [parameter],
            [torch.tensor([0.5], dtype=torch.float64)],
            eta=eta,
            programming_seed=0,
        )
    assert torch.equal(parameter.state, before)


@pytest.mark.parametrize(
    ("lower", "upper"),
    [
        (None, 1.0),
        (0.0, None),
        (0.0, float("inf")),
        (float("nan"), 1.0),
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_invalid_or_implicit_bounds_fail(lower, upper):
    parameter = _FakeParameter(
        "DenseWeight_0",
        [0.5],
        lower=lower,
        upper=upper,
    )
    with pytest.raises(ValueError, match="explicit finite bounds"):
        write_noise.apply_weight_write_noise(
            [parameter],
            [parameter.state.clone()],
            eta=0.1,
            programming_seed=0,
        )


def test_late_validation_failure_and_duplicate_names_are_fail_atomic():
    first = _FakeParameter("DenseWeight_0", [0.9])
    invalid = _FakeParameter(
        "ConvWeight_0",
        [0.8],
        lower=0.0,
        upper=float("inf"),
    )
    first_before = first.state.clone()
    invalid_before = invalid.state.clone()
    with pytest.raises(ValueError, match="explicit finite bounds"):
        write_noise.apply_weight_write_noise(
            [first, invalid],
            [
                torch.tensor([0.5], dtype=torch.float64),
                torch.tensor([0.5], dtype=torch.float64),
            ],
            eta=0.2,
            programming_seed=3,
        )
    assert torch.equal(first.state, first_before)
    assert torch.equal(invalid.state, invalid_before)

    duplicate = _FakeParameter(" DenseWeight_0 ", [0.7])
    with pytest.raises(ValueError, match="must be unique"):
        write_noise.apply_weight_write_noise(
            [first, duplicate],
            [first.state.clone(), duplicate.state.clone()],
            eta=0.2,
            programming_seed=3,
        )
    assert torch.equal(first.state, first_before)


@pytest.mark.parametrize(
    "base",
    [
        torch.tensor([float("nan")], dtype=torch.float64),
        torch.tensor([-0.1], dtype=torch.float64),
        torch.tensor([1.1], dtype=torch.float64),
    ],
)
def test_nonfinite_or_out_of_bounds_base_state_fails(base):
    parameter = _FakeParameter("ConvWeight_0", [0.5])
    before = parameter.state.clone()
    with pytest.raises(ValueError, match="Base state"):
        write_noise.apply_weight_write_noise(
            [parameter],
            [base],
            eta=0.1,
            programming_seed=0,
        )
    assert torch.equal(parameter.state, before)


def test_no_eligible_weights_and_invalid_seed_fail():
    bias = _FakeParameter("Bias_0", [0.0], lower=None, upper=None)
    with pytest.raises(ValueError, match="No eligible conductance weights"):
        write_noise.apply_weight_write_noise(
            [bias],
            [bias.state.clone()],
            eta=0.1,
            programming_seed=0,
        )

    weight = _FakeParameter("DenseWeight_0", [0.5])
    for seed in (-1, 1.5, True):
        with pytest.raises(ValueError, match="programming_seed"):
            write_noise.apply_weight_write_noise(
                [weight],
                [weight.state.clone()],
                eta=0.1,
                programming_seed=seed,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_draw_is_device_independent_on_cuda():
    cpu = _FakeParameter(
        "DenseWeight_8",
        [0.2, 0.4, 0.6],
        dtype=torch.float32,
        device="cpu",
    )
    cuda = _FakeParameter(
        "DenseWeight_8",
        [0.2, 0.4, 0.6],
        dtype=torch.float32,
        device="cuda",
    )
    cpu_receipt = write_noise.apply_weight_write_noise(
        [cpu],
        [cpu.state.clone()],
        eta=0.1,
        programming_seed=27,
    )
    cuda_receipt = write_noise.apply_weight_write_noise(
        [cuda],
        [cuda.state.clone()],
        eta=0.1,
        programming_seed=27,
    )

    assert cpu_receipt["draw_sha256"] == cuda_receipt["draw_sha256"]
    assert cpu_receipt["written_sha256"] == cuda_receipt["written_sha256"]
    assert torch.equal(cpu.state, cuda.state.cpu())
