import math

import pytest

from experiments.analyze_reduced_current_layerwise import (
    cosine_summary, exact_cells, pooled_signal, validate_cosine,
)


def state(count, d, p, f=1):
    return dict(element_count=count, centered_half_difference_rms=d,
                endpoint_rms_P=p, free_rms=f)


def test_pool_squares_before_dividing_not_batch_ratios():
    result = pooled_signal([state(1, 1, 2), state(3, 2, 8)])
    assert result["D"] == pytest.approx(math.sqrt(13/4))
    assert result["P"] == pytest.approx(7)
    assert result["D_over_P"] == pytest.approx(math.sqrt(13)/14)
    assert result["D_over_P"] != pytest.approx((.5+.25)/2)
    assert result["element_count"] == 4


def test_zero_signal_preserves_undefined_ratio():
    result = pooled_signal([state(16, 0, 0, 0)])
    assert result["D"] == 0
    assert result["D_over_P"] is None
    assert result["D_over_F"] is None


@pytest.mark.parametrize("bad", [float("nan"),float("inf"),-1])
def test_invalid_rms_fails(bad):
    with pytest.raises(ValueError):
        pooled_signal([state(16,bad,1)])


def gradient_rows():
    return [dict(batch_index=b,noise_draw_index=d,cosine=v)
            for d,v in enumerate((-.9,.3,.6)) for b in range(12)]


def test_cosine_draw_mean_band_preserves_negative_values():
    result = cosine_summary(gradient_rows(),"cosine",range(3))
    assert result["mean"] == pytest.approx(0,abs=1e-15)
    assert result["draw_min"] == pytest.approx(-.9)
    assert result["draw_max"] == pytest.approx(.6)
    assert result["defined_count"] == 36


def test_undefined_batch_is_a_gap_with_count_not_renormalized():
    rows = gradient_rows(); rows[0]["cosine"] = ""
    result = cosine_summary(rows,"cosine",range(3))
    assert result["mean"] is None and result["draw_min"] is None
    assert result["defined_count"] == 35 and result["undefined_count"] == 1


def test_missing_gradient_batch_rejected():
    with pytest.raises(ValueError,match="Missing/duplicate"):
        cosine_summary(gradient_rows()[1:],"cosine",range(3))


@pytest.mark.parametrize("rows", [[0,1],[0,1,1,2],[0,1,2,3]])
def test_missing_duplicate_or_extra_cells_rejected(rows):
    with pytest.raises(ValueError):
        exact_cells(rows,{0,1,2},lambda x:x,"fixture")


def test_zero_norm_requires_undefined_cosine():
    row = dict(eqprop_rms=0,bptt_rms=1,cosine="")
    validate_cosine(row,"cosine","bptt_rms")
    row["cosine"] = 0
    with pytest.raises(ValueError,match="zero norm"):
        validate_cosine(row,"cosine","bptt_rms")
    row.update(eqprop_rms=1,cosine="")
    with pytest.raises(ValueError,match="zero norm"):
        validate_cosine(row,"cosine","bptt_rms")


def test_nonfinite_cosine_is_not_treated_as_undefined():
    with pytest.raises(ValueError,match="Nonfinite"):
        validate_cosine(dict(eqprop_rms=1,bptt_rms=1,cosine="nan"),"cosine","bptt_rms")
