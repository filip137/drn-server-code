"""Guard the distinction between checkpoint provenance and diagnostic iteration counts."""
import ast
from pathlib import Path

import pytest


@pytest.fixture
def gate_functions():
    # These pure helpers should be testable without bootstrapping historical datasets.
    tree = ast.parse((Path(__file__).resolve().parents[2] / 'experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py').read_text())
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'_replay_iterations', '_selected'}]
    namespace = {'Mapping': dict, 'Any': object}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), '<gate helpers>', 'exec'), namespace)
    return namespace


def test_override_retains_native_checkpoint_contract(gate_functions):
    declared = dict(T=6, K=6, replay_T=12, replay_K=24)
    replay = gate_functions['_replay_iterations'](declared)
    case = dict(architecture='conv2', scheme='baseline', native_T=6, native_K=6,
                base_beta=.01, injected_beta=.01, **replay)
    selected = gate_functions['_selected'](case, 'best_validation')
    assert (selected['T'], selected['K']) == (12, 24)
    assert (selected['source_native_T'], selected['source_native_K']) == (6, 6)
    assert not selected['native_context']
    assert declared['T'] == declared['K'] == 6


@pytest.mark.parametrize('extra', [{'replay_T': 12}, {'replay_K': 12}, {'replay_T': 0, 'replay_K': 12}, {'replay_T': 12.5, 'replay_K': 12}, {'replay_T': True, 'replay_K': 12}])
def test_invalid_or_partial_override_is_rejected(gate_functions, extra):
    with pytest.raises(ValueError):
        gate_functions['_replay_iterations'](dict(T=6, K=6, **extra))


def test_native_replay_is_unchanged(gate_functions):
    assert gate_functions['_replay_iterations']({'T': 8, 'K': 8}) == {'T': 8, 'K': 8}
