"""Scientific normalization and finite bounded correction checks."""
import json
from pathlib import Path

import pytest

from experiments.calibrate_initial_relative_output import matched_beta, pooled_ratio, training_config


def test_pool_squares_before_dividing_not_batch_ratios():
    result = pooled_ratio([
        dict(free_sum_squares=100., centered_sum_squares=4., elements=20),
        dict(free_sum_squares=1., centered_sum_squares=9., elements=20),
    ])
    assert result['relative_output_rms'] == pytest.approx((13/101)**.5)
    assert result['elements'] == 40


def test_nonlinear_correction_is_measured_and_budget_bounded():
    calls = []
    def measure(beta):
        calls.append(beta)
        return {'relative_output_rms': beta**1.1}
    beta, history = matched_beta(measure, 1., 1.6, .01, 3)
    assert len(calls) <= 4
    assert abs(beta**1.1/1.6-1) <= .01
    assert history[-1]['injected_beta'] == beta
    calls.clear()
    with pytest.raises(RuntimeError, match='missed after 3 corrections'):
        matched_beta(lambda beta: calls.append(beta) or {'relative_output_rms': .1}, 1., 2., .01, 3)
    assert len(calls) == 4


def test_all_configs_preserve_accepted_rates_iterations_and_disable_test():
    root = Path(__file__).resolve().parents[2]
    spec = json.loads((root/'configs/conv/eqprop_conv123_relative_output_calibration_20260924.json').read_text())
    assert len(spec['cases']) == 9
    assert spec['relative_targets'] == [.8,1.2,1.6,2.]
    for declared in spec['cases']:
        source = json.loads((root/declared['source_config']).read_text())
        for target in spec['relative_targets']:
            c = training_config(source, declared, spec, beta=.17, target=target, calibration={'relative_output_rms':target})
            assert c['lr'] == source['lr'] == c['optimizer']['learning_rate']
            assert c['learning_rates_by_parameter'] == source['learning_rates_by_parameter']
            assert c['model_base'] == source['model_base']
            assert c['lab']['epochs'] == 10
            assert c['evaluation']['official_test']['policy'] == 'disabled'
            assert c['eqprop']['endpoint_read_noise_std'] == .0005
            assert c['eqprop']['injected_beta_B'] == .17
            assert c['runtime_dtype'] == 'float64'
            assert c['eqprop']['injected_beta_B'] == pytest.approx(c['beta']*c['eqprop']['amplification_factor'])
