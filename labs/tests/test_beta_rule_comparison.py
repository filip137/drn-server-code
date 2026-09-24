"""Scientific checks for changing gradient grouping and beta decision rules."""
import ast
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.run_beta_rule_comparison import passes, select_beta


@pytest.fixture
def aggregate():
    path = Path(__file__).resolve().parents[2] / 'experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_whole_gradient_metrics')
    namespace = dict(math=math, Sequence=list, Mapping=dict, Any=object, NORM_EPSILON=1e-30)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_whole_gradient_metrics']


def rows_for(ep, bp):
    result = []
    for i, (e, b) in enumerate(zip(ep, bp)):
        e, b = np.asarray(e, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
        en, bn = np.linalg.norm(e), np.linalg.norm(b)
        dot = float(e @ b)
        result.append(dict(parameter_name=f'ConvWeight_{i}', element_count=e.size,
            eqprop_l2=float(en), bptt_l2=float(bn), dot_product=dot,
            difference_l2=float(np.linalg.norm(e-b)), cosine=dot/(en*bn) if en*bn > 1e-30 else None,
            symmetric_norm_delta=2*abs(en-bn)/max(en+bn, 1e-30)))
    return result


def test_aggregate_matches_explicit_concatenation(aggregate):
    rng = np.random.default_rng(20260918)
    ep = [rng.normal(size=(3, 4)), rng.normal(size=(20,)), rng.normal(size=(7, 2))]
    bp = [rng.normal(size=e.shape) for e in ep]
    measured = aggregate(rows_for(ep, bp))
    direct = rows_for([np.concatenate([v.ravel() for v in ep])],
                      [np.concatenate([v.ravel() for v in bp])])[0]
    for key in ('eqprop_l2', 'bptt_l2', 'dot_product', 'difference_l2', 'cosine', 'symmetric_norm_delta'):
        assert measured[key] == pytest.approx(direct[key], abs=1e-14)


def test_global_alignment_can_hide_reversed_small_layer(aggregate):
    layers = rows_for([[1000], [-1]], [[1000], [1]])
    assert aggregate(layers)['cosine'] > .99
    assert passes([aggregate(layers)], .99, True)
    assert not passes(layers, .90, False)


def test_norm_gate_uses_requested_grouping(aggregate):
    layers = rows_for([[1000], [10]], [[1000], [1]])
    assert passes(layers, .99, False)
    assert not passes(layers, .99, True)
    assert passes([aggregate(layers)], .99, True)


@pytest.mark.parametrize('value', [None, '', float('nan'), float('inf')])
def test_undefined_and_nonfinite_never_pass(value):
    assert not passes([dict(cosine=value, symmetric_norm_delta=0)], .90, False)


def test_zero_gradient_and_bias_exclusion(aggregate):
    layers = rows_for([[0, 0]], [[0, 0]])
    assert aggregate(layers)['cosine'] is None
    assert not passes([aggregate(layers)], .90, False)
    layers[0]['parameter_name'] = 'Bias_0'
    with pytest.raises(ValueError, match='weights only'):
        aggregate(layers)


def test_nonfinite_statistics_are_rejected(aggregate):
    layers = rows_for([[1]], [[1]])
    layers[0]['dot_product'] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        aggregate(layers)


def test_nonmonotonic_selection_and_open_boundary():
    result = select_beta([(1, True), (3, False), (10, True), (30, True)])
    assert result['selected_beta'] == 30 and result['upper_edge_open']
    assert result['passing_regions'] == [dict(low=1, high=1), dict(low=10, high=30)]
    assert select_beta([(1, False), (3, False)])['selected_beta'] is None


def test_each_batch_must_pass_and_threshold_is_inclusive():
    good = dict(cosine=.95, symmetric_norm_delta=.10)
    assert passes([good], .95, True)
    assert not passes([good], .99, True)
    assert not passes([good, dict(cosine=.94, symmetric_norm_delta=0)], .95, False)
