"""Numerical aggregation checks for the fixed-T/K beta-response figures."""
import math

import pytest

from experiments.plot_conv3_clean_beta_response import cosine_summary, local_slopes, pooled_rms


def test_pooled_rms_uses_element_counts_and_squared_signal():
    rows = [dict(element_count=1, delta=1.), dict(element_count=3, delta=3.)]
    assert pooled_rms(rows, 'delta') == pytest.approx(math.sqrt(7))
    assert pooled_rms([dict(element_count=1, delta='')], 'delta') is None


def test_undefined_cosine_does_not_become_zero():
    summary = cosine_summary([dict(cosine=''), dict(cosine=.8), dict(cosine=1.)])
    assert summary['cosine_defined_batches'] == 2
    assert summary['cosine_median'] == pytest.approx(.9)
    assert cosine_summary([dict(cosine='')])['cosine_median'] is None


def test_slopes_distinguish_amplitude_from_exponent_and_preserve_gaps():
    rows = []
    for scheme, scale, exponent in [('baseline', 1., 1.), ('ours', 100., 1.), ('legacy', 2., 2.)]:
        for beta in [.001, .01, .1, 1.]:
            rows.append(dict(scheme=scheme, epoch=0, layer_index=0, beta=beta,
                             centered_half_difference_rms=scale*beta**exponent,
                             centered_resolved=beta != .1, residuals_pass=True))
    slopes = local_slopes(rows)
    first = [r for r in slopes if r['beta_low'] == .001]
    assert [r['slope'] for r in first] == pytest.approx([1., 2., 1.])
    legacy = next(r for r in first if r['scheme'] == 'legacy')
    assert legacy['slope_minus_baseline'] == pytest.approx(1.)
    assert legacy['slope_minus_ours'] == pytest.approx(1.)
    assert all(r['slope'] is None for r in slopes if r['beta_low'] != .001)
