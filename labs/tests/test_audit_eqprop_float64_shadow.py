import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "experiments/audit_eqprop_float64_shadow.py"
CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv123_positive_onesided_small_beta_native_seed0_20260810_v1.json"
)


def _run_helper_assertions(body):
    program = f"""
import importlib.util
import pathlib
import sys

runner = pathlib.Path({str(RUNNER)!r})
config_path = pathlib.Path({str(CONFIG)!r})
sys.argv = [str(runner), '--config', str(config_path)]
spec = importlib.util.spec_from_file_location('eqprop_float64_shadow_test', runner)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

{body}
"""
    environment = os.environ.copy()
    environment.update(
        {
            "KMP_DISABLE_SHM": "1",
            "KMP_SHM_DISABLE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLCONFIGDIR": "/tmp/mplconfig-eqprop-float64-shadow-test",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_case_selection_defaults_cover_deep_trained_cases_and_all18():
    _run_helper_assertions(
        """
defaults = module._requested_case_specs([], all_18=False)
assert len(defaults) == 6
assert {
    (row['architecture'], row['scheme'], row['checkpoint_role'])
    for row in defaults
} == {
    (architecture, scheme, 'best_validation')
    for architecture in ('conv2', 'conv3')
    for scheme in ('baseline', 'ours', 'legacy')
}

all_cases = module._requested_case_specs([], all_18=True)
assert len(all_cases) == 18
assert len({
    (row['architecture'], row['scheme'], row['checkpoint_role'])
    for row in all_cases
}) == 18

parsed = module._parse_case_spec(
    'conv3:legacy:best_validation:1e-8'
)
assert parsed == {
    'architecture': 'conv3',
    'scheme': 'legacy',
    'checkpoint_role': 'best_validation',
    'explicit_beta': 1e-8,
}
"""
    )


def test_stage_selection_uses_actual_beta_and_native_coordinate():
    _run_helper_assertions(
        """
rows = [
    {
        'architecture': 'conv2',
        'scheme': 'ours',
        'checkpoint_role': 'best_validation',
        'T': '6',
        'K': '6',
        'native_T': '6',
        'native_K': '6',
        'native_context': 'True',
        'selected_beta': '2.5e-7',
        'selected_beta_hat': '3e-5',
        'selection_rule': 'lowest_uncapped_adjacent_consistent_pair_choose_larger',
    },
    {
        'architecture': 'conv2',
        'scheme': 'ours',
        'checkpoint_role': 'best_validation',
        'T': '12',
        'K': '6',
        'native_T': '6',
        'native_K': '6',
        'native_context': 'False',
        'selected_beta': '9e-7',
        'selection_rule': 'lowest_uncapped_adjacent_consistent_pair_choose_larger',
    },
]
spec = module._parse_case_spec('conv2:ours:best_validation')
resolved = module._resolve_selected_cases(rows, [spec])
assert len(resolved) == 1
assert resolved[0]['T'] == 6
assert resolved[0]['K'] == 6
assert resolved[0]['actual_beta'] == 2.5e-7
assert resolved[0]['beta_source'] == 'stage_selected'
assert len(resolved[0]['source_selection_row_sha256']) == 64

explicit = module._resolve_selected_cases(
    rows,
    [module._parse_case_spec('conv2:ours:best_validation:1e-8')],
    explicit_beta_label='common_effective_beta_shadow',
)
assert explicit[0]['actual_beta'] == 1e-8
assert explicit[0]['beta_source'] == 'common_effective_beta_shadow'
"""
    )


def test_tk_override_is_atomic_explicit_and_architecture_generic():
    _run_helper_assertions(
        """
import copy

inventory = {
    ('conv3', 'baseline'): {
        'architecture': 'conv3', 'scheme': 'baseline', 'native_T': 12, 'native_K': 8,
    },
    ('conv2', 'baseline'): {
        'architecture': 'conv2', 'scheme': 'baseline', 'native_T': 6, 'native_K': 6,
    },
}
source = {
    'architecture': 'conv3', 'scheme': 'baseline', 'T': 12, 'K': 8,
    'beta_explicit_cli': True,
}
native = [copy.deepcopy(source)]
module._apply_tk_override(native, inventory, None)
assert native[0]['T'] == 12 and native[0]['K'] == 8
assert native[0]['source_native_T'] == 12 and native[0]['source_native_K'] == 8
assert native[0]['native_context'] is True
assert native[0]['tk_source'] == 'source_native'

overridden = [copy.deepcopy(source)]
module._apply_tk_override(overridden, inventory, [64, 128])
assert overridden[0]['T'] == 64 and overridden[0]['K'] == 128
assert overridden[0]['source_native_T'] == 12 and overridden[0]['source_native_K'] == 8
assert overridden[0]['native_context'] is False
assert overridden[0]['tk_source'] == 'explicit_cli_override'

for bad in ([0, 64], [64, -1], [64]):
    try:
        module._apply_tk_override([copy.deepcopy(source)], inventory, bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f'Expected invalid override to fail: {bad}')

implicit = [copy.deepcopy({**source, 'beta_explicit_cli': False})]
try:
    module._apply_tk_override(implicit, inventory, [64, 64])
except ValueError as error:
    assert 'explicit beta' in str(error)
else:
    raise AssertionError('Non-native T/K accepted a stage-selected beta.')

conv2 = [{
    'architecture': 'conv2', 'scheme': 'baseline', 'T': 6, 'K': 6,
    'beta_explicit_cli': True,
}]
module._apply_tk_override(conv2, inventory, [64, 64])
assert conv2[0]['T'] == 64 and conv2[0]['K'] == 64
assert conv2[0]['source_native_T'] == 6 and conv2[0]['source_native_K'] == 6
assert conv2[0]['native_context'] is False
"""
    )


def test_out_of_grid_beta_records_uncapped_injected_extension():
    _run_helper_assertions(
        """
module.extended._beta_grid_for_case = lambda contract, case: []
config = {
    'gradient_contract': {
        'nudging_scale_power_by_architecture': {'conv1': 1, 'conv2': 2, 'conv3': 3},
        'baseline_init_output_curvature_scale_b0_by_architecture': {
            'conv1': 7.0, 'conv2': 14.0, 'conv3': 28.046421,
        },
        'actual_beta_cap': 0.01,
    },
}
case = {
    'architecture': 'conv3', 'scheme': 'ours',
    'voltage_amplification': 4.0, 'current_amplification': 1.0,
}
row = module._resolve_beta_row(
    config=config,
    case=case,
    requested_actual_beta=0.0015625,
    allow_out_of_grid=True,
)
assert row['source_grid_member'] is False
assert row['source_grid_cap_bypassed'] is True
assert row['capped'] is False
assert abs(row['beta_effective'] - 0.1) < 1e-15
assert abs(row['beta_hat_requested'] - 0.0015625 / 28.046421) < 1e-15
assert abs(row['beta_hat'] - 0.1 / 28.046421) < 1e-15

try:
    module._resolve_beta_row(
        config=config,
        case=case,
        requested_actual_beta=0.0015625,
        allow_out_of_grid=False,
    )
except ValueError as error:
    assert 'absent from the resolved source grid' in str(error)
else:
    raise AssertionError('Out-of-grid beta was accepted without the explicit flag.')

conv1_ours = {
    'architecture': 'conv1', 'scheme': 'ours',
    'voltage_amplification': 4.0, 'current_amplification': 1.0,
}
conv1_row = module._resolve_beta_row(
    config=config,
    case=conv1_ours,
    requested_actual_beta=25.0,
    allow_out_of_grid=True,
)
assert conv1_row['beta_effective'] == 100.0

conv2_ours = {
    'architecture': 'conv2', 'scheme': 'ours',
    'voltage_amplification': 4.0, 'current_amplification': 1.0,
}
conv2_row = module._resolve_beta_row(
    config=config,
    case=conv2_ours,
    requested_actual_beta=6.25,
    allow_out_of_grid=True,
)
assert conv2_row['beta_effective'] == 100.0
"""
    )


def test_iteration_contract_checks_free_and_all_gradient_iterations():
    _run_helper_assertions(
        """
class Minimizer:
    def __init__(self, iterations):
        self.num_iterations = iterations

runtime = {
    'inference_iterations': 64,
    'gradient_iterations': 128,
    'minimizer_inference': Minimizer(64),
    'minimizer_gradient': Minimizer(128),
    'minimizer_augmented': Minimizer(128),
}
contract = module._iteration_contract(runtime, {'T': 64, 'K': 128})
assert contract['iteration_contract_passed'] is True
assert contract['runtime_inference_iterations_T'] == 64
assert contract['augmented_minimizer_iterations_K'] == 128

for key in ('minimizer_inference', 'minimizer_gradient', 'minimizer_augmented'):
    broken = dict(runtime)
    broken[key] = Minimizer(7)
    try:
        module._iteration_contract(broken, {'T': 64, 'K': 128})
    except RuntimeError:
        pass
    else:
        raise AssertionError(f'Iteration mismatch in {key} was accepted.')
"""
    )


def test_native_dtype_finite_difference_exposes_float32_cancellation():
    _run_helper_assertions(
        """
import torch

zero32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
plus32 = torch.tensor([1.0 + 1e-8, 2.0 + 2e-8], dtype=torch.float32)
delta32, gradient32 = module._native_one_sided_estimate(
    beta=1e-8, zero=zero32, positive=plus32
)
assert delta32.dtype == torch.float32
assert gradient32.dtype == torch.float32
assert torch.equal(delta32, torch.zeros_like(delta32))

zero64 = torch.tensor([1.0, 2.0], dtype=torch.float64)
plus64 = torch.tensor([1.0 + 1e-8, 2.0 + 2e-8], dtype=torch.float64)
delta64, gradient64 = module._native_one_sided_estimate(
    beta=1e-8, zero=zero64, positive=plus64
)
_, scaled64 = module._native_one_sided_estimate(
    beta=1e-8, denominator_scale=4.0, zero=zero64, positive=plus64
)
assert delta64.dtype == torch.float64
assert gradient64.dtype == torch.float64
assert torch.all(delta64 != 0.0)
assert torch.allclose(gradient64, torch.tensor([1.0, 2.0], dtype=torch.float64))
assert torch.allclose(scaled64, gradient64 / 4.0)

metrics = module._precision_metrics(gradient32, gradient64)
assert metrics['cosine'] is None
assert metrics['precision_gate_passed'] is False
"""
    )


def test_native_centered_difference_uses_signed_endpoints_and_two_beta_denominator():
    _run_helper_assertions(
        """
import torch

zero = torch.tensor([99.0, 101.0], dtype=torch.float64)
negative = torch.tensor([1.0, 2.0], dtype=torch.float64)
positive = torch.tensor([5.0, 8.0], dtype=torch.float64)
numerator, estimate = module._native_eqprop_estimate(
    eqprop_variant='centered',
    beta=0.5,
    denominator_scale=2.0,
    zero=zero,
    negative=negative,
    positive=positive,
)
assert numerator.dtype == estimate.dtype == torch.float64
assert torch.equal(numerator, torch.tensor([4.0, 6.0], dtype=torch.float64))
assert torch.equal(estimate, torch.tensor([2.0, 3.0], dtype=torch.float64))

try:
    module._native_eqprop_estimate(
        eqprop_variant='centered', beta=0.5, zero=zero, positive=positive
    )
except ValueError as error:
    assert 'negative endpoint' in str(error)
else:
    raise AssertionError('Centered estimate accepted no negative endpoint.')

try:
    module._native_eqprop_estimate(
        eqprop_variant='centered',
        beta=0.5,
        zero=zero,
        negative=negative.to(torch.float32),
        positive=positive,
    )
except ValueError as error:
    assert 'dtypes' in str(error)
else:
    raise AssertionError('Centered estimate accepted mismatched endpoint dtypes.')
"""
    )


def test_full_runtime_variable_conversion_survives_explicit_state_reset():
    _run_helper_assertions(
        """
import torch

class Variable:
    def __init__(self, name, state, shape=None):
        self.name = name
        self.state = state
        self.shape = tuple(state.shape[1:]) if shape is None else shape

class Input(Variable):
    def set_input(self, images):
        self.state = 3.0 * torch.cat((images, -images), dim=1)

class Energy:
    def __init__(self, layers, params):
        self._layers = layers
        self._all_params = params
    def layers(self):
        return self._layers

input_layer = Input('Layer_0', torch.zeros(1, 2))
hidden = Variable('Layer_1', torch.zeros(1, 4), shape=(4,))
output = Variable('Layer_2', torch.zeros(1, 2), shape=(2,))
weight = Variable('DenseWeight_0', torch.tensor([[1.25]], dtype=torch.float32))
bias = Variable('Bias_0', torch.tensor([0.5], dtype=torch.float32))
runtime = {
    'energy_fn': Energy([input_layer, hidden, output], [weight, bias]),
    'free_layers': [hidden, output],
    'parameters': [weight, bias],
}
proof = module._convert_runtime_dtype(runtime, torch.float64)
assert proof['all_runtime_variables_target_dtype'] is True
assert all(row['dtype'] == 'torch.float64' for row in proof['after_conversion_before_state_reset'])
images = torch.ones((3, 1), dtype=torch.float64)
reset = module._set_input_and_reset(runtime, images, dtype=torch.float64)
assert reset['all_runtime_variables_target_dtype_after_reset'] is True
assert input_layer.state.dtype == torch.float64
assert hidden.state.shape == (3, 4)
assert output.state.shape == (3, 2)
assert hidden.state.dtype == output.state.dtype == torch.float64
assert weight.state.dtype == bias.state.dtype == torch.float64
"""
    )


def test_amplification_scaling_and_empirical_beta_over_b0_are_explicit():
    _run_helper_assertions(
        """
config = {
    'gradient_contract': {
        'amplification_exponent_convention': 'output_bias_current_row',
        'nudging_scale_power_by_architecture': {'conv3': 3},
        'baseline_init_output_curvature_scale_b0_by_architecture': {
            'conv3': 20.0,
        },
    },
}
case = {
    'architecture': 'conv3',
    'voltage_amplification': 4.0,
    'current_amplification': 0.25,
    'scored_interaction_count_L': 4,
    'init_output_curvature_scale_b0': 0.5,
    'best_output_curvature_scale_b0_reference': 0.75,
}
metadata = module._scaling_metadata(
    config=config,
    case=case,
    actual_beta=1e-8,
    checkpoint_b0=0.8,
    scored_weight_count=4,
)
assert metadata['amplification_depth_L'] == 3
assert metadata['scored_interaction_count'] == 4
assert metadata['amplification_factor'] == 16.0 ** 3
assert metadata['effective_beta'] == 1e-8 * 16.0 ** 3
assert metadata['beta_hat'] == metadata['effective_beta'] / 20.0
assert metadata['actual_beta_over_case_init_b0'] == 2e-8
assert metadata['actual_beta_over_checkpoint_b0'] == 1.25e-8
assert 'effective_beta=actual_beta' in metadata['scaling_formula']
"""
    )


def test_precision_gate_uses_inclusive_cosine_and_symmetric_norm_limits():
    _run_helper_assertions(
        """
import math
import torch

reference = torch.tensor([1.0, 2.0], dtype=torch.float32)
same_direction = torch.tensor([1.05, 2.10], dtype=torch.float64)
passing = module._precision_metrics(reference, same_direction)
assert math.isclose(passing['cosine'], 1.0)
assert passing['symmetric_norm_delta'] < 0.10
assert passing['precision_gate_passed'] is True

wrong_norm = torch.tensor([2.0, 4.0], dtype=torch.float64)
failing = module._precision_metrics(reference, wrong_norm)
assert math.isclose(failing['cosine'], 1.0)
assert failing['symmetric_norm_delta'] > 0.10
assert failing['precision_gate_passed'] is False
"""
    )


def test_eqprop_bptt_metrics_keep_direction_and_norm_errors_separate():
    _run_helper_assertions(
        """
import math
import torch

bptt = torch.tensor([1.0, 0.0], dtype=torch.float32)
scaled_eqprop = torch.tensor([2.0, 0.0], dtype=torch.float32)
scaled = module._eqprop_bptt_metrics(
    eqprop_gradient=scaled_eqprop,
    bptt_gradient=bptt,
)
assert scaled['element_count'] == 2
assert math.isclose(scaled['cosine'], 1.0)
assert math.isclose(scaled['bptt_l2'], 1.0)
assert math.isclose(scaled['eqprop_l2'], 2.0)
assert math.isclose(scaled['eqprop_over_bptt_norm_ratio'], 2.0)
assert math.isclose(scaled['relative_l2_difference_over_bptt'], 1.0)
assert scaled['symmetric_norm_delta'] > 0.0

orthogonal = module._eqprop_bptt_metrics(
    eqprop_gradient=torch.tensor([0.0, 3.0], dtype=torch.float64),
    bptt_gradient=bptt,
)
assert math.isclose(orthogonal['cosine'], 0.0, abs_tol=1e-15)
assert math.isclose(orthogonal['eqprop_over_bptt_norm_ratio'], 3.0)
"""
    )


def test_native_bptt_restores_post_t_uses_selected_k_and_excludes_bias():
    _run_helper_assertions(
        """
import torch

class Layer:
    def __init__(self, name, state):
        self.name = name
        self.state = state

class ConvWeight:
    def __init__(self, name, state):
        self.name = name
        self.state = state

class Bias:
    def __init__(self, name, state):
        self.name = name
        self.state = state

layer = Layer('Layer_1', torch.tensor([[99.0]], dtype=torch.float32))
weight = ConvWeight('ConvWeight_0', torch.tensor([[1.0]], dtype=torch.float32))
bias = Bias('Bias_0', torch.tensor([0.5], dtype=torch.float32))
post_t_states = [torch.tensor([[4.0]], dtype=torch.float32)]
module.base._restore_states([layer], post_t_states)
post_t_hash = module.base._layer_state_sha256([layer])
layer.state = torch.tensor([[-7.0]], dtype=torch.float32)

class Backprop:
    def compute_gradient(self):
        assert module.base._layer_state_sha256([layer]) == post_t_hash
        layer.state = layer.state + 1.0
        return [
            torch.tensor([[2.0]], dtype=torch.float32),
            torch.tensor([3.0], dtype=torch.float32),
        ]

class Minimizer:
    num_iterations = 7

runtime = {
    'free_layers': [layer],
    'parameters': [weight, bias],
    'weight_indices': [0],
    'weight_names': ['ConvWeight_0'],
    'gradient_iterations': 7,
    'minimizer_gradient': Minimizer(),
    'backprop': Backprop(),
}
gradients, guard = module._native_bptt_weight_gradients(
    runtime,
    post_t_states=post_t_states,
    post_t_hash=post_t_hash,
    expected_k=7,
)
assert list(gradients) == ['ConvWeight_0']
assert gradients['ConvWeight_0'].dtype == torch.float32
assert torch.equal(gradients['ConvWeight_0'], torch.tensor([[2.0]]))
assert guard['bptt_start_state_sha256'] == post_t_hash
assert guard['bptt_gradient_iterations_K'] == 7
assert guard['bptt_minimizer_iterations_K'] == 7
assert guard['bptt_all_parameter_gradient_count'] == 2
assert guard['bptt_scored_weight_gradient_count'] == 1
assert guard['bptt_bias_gradients_excluded'] is True
"""
    )


def test_gradient_archive_contains_native_bptt_tensors_for_both_precisions():
    _run_helper_assertions(
        """
from pathlib import Path
import tempfile
import torch

outputs = {
    'float32': {
        'gradients': {'ConvWeight_0': torch.tensor([1.0], dtype=torch.float32)},
        'bptt_gradients': {'ConvWeight_0': torch.tensor([2.0], dtype=torch.float32)},
        'numerators': {'ConvWeight_0': torch.tensor([3.0], dtype=torch.float32)},
    },
    'float64': {
        'gradients': {'ConvWeight_0': torch.tensor([4.0], dtype=torch.float64)},
        'bptt_gradients': {'ConvWeight_0': torch.tensor([5.0], dtype=torch.float64)},
        'numerators': {'ConvWeight_0': torch.tensor([6.0], dtype=torch.float64)},
    },
}
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / 'gradients.npz'
    module._write_gradient_archive(
        path,
        selected={'architecture': 'conv1', 'source_selection_row': {}},
        precision_outputs=outputs,
        scaling={'actual_beta': 1e-8},
    )
    with module.np.load(path, allow_pickle=False) as archive:
        assert 'bptt_float32__ConvWeight_0' in archive.files
        assert 'bptt_float64__ConvWeight_0' in archive.files
        assert archive['bptt_float32__ConvWeight_0'].dtype.name == 'float32'
        assert archive['bptt_float64__ConvWeight_0'].dtype.name == 'float64'
        assert not any('Bias' in key for key in archive.files)
        metadata = __import__('json').loads(str(archive['metadata_json']))
        assert metadata['bptt_reference'] == 'same_post_T_state_exactly_K_iterations'
        assert metadata['scored_gradients'] == 'weights_only_biases_excluded'
"""
    )


def test_completion_allows_a_recorded_scientific_precision_failure():
    _run_helper_assertions(
        """
completion = {
    'criteria_met': True,
    'all_comparisons_present': True,
    'scientific_precision_gate_outcome_recorded': True,
    'optimizer_steps_applied': False,
    'official_test_read': False,
}
assert module._completion_criteria_met(completion) is True
assert module._completion_criteria_met({
    **completion,
    'official_test_read': True,
}) is False
"""
    )
