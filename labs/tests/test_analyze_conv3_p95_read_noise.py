import numpy as np
import pytest

from experiments.analyze_conv3_p95_read_noise import passes_stability_screen


@pytest.mark.parametrize('final_correct,expected', [(4651, True), (4650, False), (4649, False)])
def test_strict_five_percentage_point_boundary(final_correct, expected):
    # 98% -> 93% can round to a floating-point drop just below 5pp.
    values = np.full(30, 4900 / 5000)
    values[-1] = final_correct / 5000
    assert passes_stability_screen(values) is expected


def test_incomplete_or_nonfinite_history_never_passes():
    assert not passes_stability_screen([.98] * 29)
    assert not passes_stability_screen([.98] * 29 + [float('nan')])
