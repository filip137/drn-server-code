import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    REPOSITORY_ROOT
    / "experiments/analyze_conv_eqprop_bptt_beta_tk_displacement.py"
)
CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv123_all_amplification_adam_eqprop_bptt_beta_tk_displacement_seed0_20260810_v1.json"
)
POSITIVE_CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_conv123_positive_onesided_small_beta_native_seed0_20260810_v1.json"
)
SLURM_WRAPPER = (
    REPOSITORY_ROOT
    / "experiments/run_conv_eqprop_bptt_beta_tk_displacement_jeanzay.slurm"
)


def _run_helper_assertions(body):
    program = f"""
import importlib.util
import json
import pathlib
import sys

runner = pathlib.Path({str(RUNNER)!r})
config_path = pathlib.Path({str(CONFIG)!r})
sys.argv = [str(runner), '--config', str(config_path)]
spec = importlib.util.spec_from_file_location('eqprop_beta_tk_test', runner)
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


def test_config_inventory_aliases_and_shared_tk_grids():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert {
        item["scheme"]: (
            item["voltage_amplification"],
            item["current_amplification"],
        )
        for item in config["amplification_schemes"]
    } == {
        "baseline": (1.0, 1.0),
        "ours": (4.0, 1.0),
        "legacy": (4.0, 0.25),
    }
    assert {
        (case["scheme"], case["architecture"]) for case in config["cases"]
    } == {
        (scheme, architecture)
        for scheme in ("baseline", "ours", "legacy")
        for architecture in ("conv1", "conv2", "conv3")
    }
    conv3_native = {
        case["scheme"]: (
            case["native_inference_iterations"],
            case["native_gradient_iterations"],
        )
        for case in config["cases"]
        if case["architecture"] == "conv3"
    }
    assert conv3_native == {
        "baseline": (12, 8),
        "ours": (8, 8),
        "legacy": (8, 8),
    }

    _run_helper_assertions(
        """
config = json.loads(config_path.read_text(encoding='utf-8'))

alias_only = {
    'native_T': 12,
    'native_K': 8,
    'best_model_pt_sha256': 'best-alias',
    'manifest_sha256': 'manifest-alias',
    'reconstructed_init_tensor_sha256': 'init-alias',
}
translated_alias = module._case_for_base(alias_only)
assert translated_alias['inference_iterations'] == 12
assert translated_alias['gradient_iterations'] == 8
assert translated_alias['best_checkpoint_sha256'] == 'best-alias'
assert translated_alias['source_manifest_sha256'] == 'manifest-alias'
assert translated_alias['reconstructed_initialization_tensor_sha256'] == 'init-alias'

raw_baseline = next(
    case for case in config['cases']
    if case['architecture'] == 'conv3' and case['scheme'] == 'baseline'
)
translated_canonical = module._case_for_base(raw_baseline)
assert translated_canonical['inference_iterations'] == 12
assert translated_canonical['gradient_iterations'] == 8
assert translated_canonical['best_checkpoint_sha256'] == raw_baseline['best_checkpoint_sha256']
assert translated_canonical['source_manifest_sha256'] == raw_baseline['source_manifest_sha256']
assert translated_canonical['reconstructed_initialization_tensor_sha256'] == raw_baseline['reconstructed_initialization_tensor_sha256']

baseline_case = dict(raw_baseline, native_T=12, native_K=8)
ours_case = dict(
    next(
        case for case in config['cases']
        if case['architecture'] == 'conv3' and case['scheme'] == 'ours'
    ),
    native_T=8,
    native_K=8,
)
production_grid = module._grid_for_case(config, baseline_case, smoke=False)
assert len(production_grid) == 20
assert len(set(production_grid)) == 20
assert (12, 8) in production_grid
assert (8, 8) in production_grid
assert (64, 64) in production_grid
assert module._grid_for_case(config, baseline_case, smoke=True) == [(12, 8), (8, 8), (64, 64)]
assert module._grid_for_case(config, ours_case, smoke=True) == [(8, 8), (64, 64)]
assert module._shared_anchor('conv3') == (8, 8)
"""
    )


def test_positive_one_sided_variant_uses_zero_and_positive_only():
    _run_helper_assertions(
        """
import torch

assert module._eqprop_variant({
    'eqprop_variant_primary': 'centered',
    'eqprop_variants_reported': ['centered'],
}) == 'centered'
assert module._eqprop_variant({
    'eqprop_variant_primary': 'positive_one_sided',
    'eqprop_variants_reported': ['positive_one_sided'],
}) == 'positive_one_sided'
assert module._nudged_phase_plan('centered') == (
    ('negative', -1.0), ('positive', 1.0)
)
assert module._nudged_phase_plan('positive_one_sided') == (('positive', 1.0),)

zero = torch.tensor([1.0, -2.0])
negative = torch.tensor([0.8, -2.4])
positive = torch.tensor([1.6, -1.2])
one_sided = module._eqprop_estimate(
    eqprop_variant='positive_one_sided',
    beta=0.2,
    zero_gradient=zero,
    endpoint_gradients={'positive': positive},
)
scaled_current = module._eqprop_estimate(
    eqprop_variant='positive_one_sided',
    beta=0.2,
    denominator_scale=4.0,
    zero_gradient=zero,
    endpoint_gradients={'positive': positive},
)
centered = module._eqprop_estimate(
    eqprop_variant='centered',
    beta=0.2,
    zero_gradient=torch.tensor([999.0, 999.0]),
    endpoint_gradients={'negative': negative, 'positive': positive},
)
assert one_sided.dtype == torch.float64
assert torch.allclose(one_sided, torch.tensor([3.0, 4.0], dtype=torch.float64))
assert torch.allclose(scaled_current, one_sided / 4.0)
assert torch.allclose(centered, torch.tensor([2.0, 3.0], dtype=torch.float64))

try:
    module._eqprop_variant({
        'eqprop_variant_primary': 'positive_one_sided',
        'eqprop_variants_reported': ['centered'],
    })
except ValueError:
    pass
else:
    raise AssertionError('A mismatched declared variant must fail closed.')
"""
    )


def test_amplification_aware_normalized_beta_grid_caps_and_deduplicates():
    _run_helper_assertions(
        """
import math

contract = {
    'normalized_betas': [1e-6, 3e-3, 1e-2],
    'actual_beta_cap': 0.03,
    'amplification_exponent_convention': 'output_bias_current_row',
    'nudging_scale_power_by_architecture': {'conv1': 1},
    'baseline_init_output_curvature_scale_b0_by_architecture': {
        'conv1': 28.0,
    },
}
baseline = {
    'architecture': 'conv1',
    'voltage_amplification': 1.0,
    'current_amplification': 1.0,
    'scored_interaction_count_L': 2,
    'init_output_curvature_scale_b0': 28.0,
}
ours = {
    **baseline,
    'voltage_amplification': 4.0,
    'init_output_curvature_scale_b0': 7.0,
}
legacy = {
    **ours,
    'current_amplification': 0.25,
    'init_output_curvature_scale_b0': 28.0 / 16.0,
}
baseline_grid = module._beta_grid_for_case(contract, baseline)
ours_grid = module._beta_grid_for_case(contract, ours)
legacy_grid = module._beta_grid_for_case(contract, legacy)
assert len(baseline_grid) == 2
assert baseline_grid[-1]['beta_hat_requested'] == 3e-3
assert baseline_grid[-1]['beta'] == 0.03
assert baseline_grid[-1]['capped'] is True
assert len(ours_grid) == len(legacy_grid) == 3
assert math.isclose(ours_grid[-2]['beta'], 0.003 * 28.0 / 4.0)
assert ours_grid[-1]['beta'] == 0.03
assert ours_grid[-1]['capped'] is True
assert math.isclose(legacy_grid[-1]['beta'], 0.01 * 28.0 / 16.0)
assert ours_grid[-1]['amplification_depth_L'] == 1
assert ours_grid[-1]['scored_interaction_count'] == 2
for grid in (baseline_grid, ours_grid, legacy_grid):
    for row in grid:
        assert math.isclose(
            row['beta_hat'],
            row['beta_effective'] / 28.0,
        )

literal_contract = {
    **contract,
    'normalized_betas': [1e-4, 1e-3],
    'actual_beta_cap': 0.01,
    'normalized_beta_parameterization': 'common_base',
}
literal_grid = module._beta_grid_for_case(literal_contract, ours)
assert math.isclose(literal_grid[0]['beta'], 0.0025)
assert math.isclose(literal_grid[0]['beta_effective'], 0.01)
assert literal_grid[0]['capped'] is True
assert literal_grid[0]['cap_semantics'] == 'injected_beta'
assert len(literal_grid) == 1

config = {
    'tk_coordinate_mode': 'native_only',
    'tk_factorial': {
        'conv1': {
            'inference_iterations': [4, 8, 64],
            'gradient_iterations': [4, 8, 64],
        },
    },
}
case = {'architecture': 'conv1', 'native_T': 4, 'native_K': 4}
assert module._grid_for_case(config, case, smoke=False) == [(4, 4)]

class Layer:
    pass

class Energy:
    def a_coef_fn(self, _layer):
        return lambda: torch.tensor([1.0, 2.0, 3.0, 4.0])

import torch
curvature = module._output_curvature_diagnostic({
    'free_layers': [Layer()],
    'energy_fn': Energy(),
})
assert curvature['output_curvature_median'] == 2.5
assert curvature['output_curvature_scale_b0'] == 5.0
"""
    )


def test_positive_stage_a_config_resolves_declared_native_case_counts():
    config = json.loads(POSITIVE_CONFIG.read_text(encoding="utf-8"))
    assert config["tk_coordinate_mode"] == "native_only"
    assert config["gradient_contract"]["eqprop_variant_primary"] == (
        "positive_one_sided"
    )
    expected_counts = config["completion"]["expected_beta_values_by_case"]
    _run_helper_assertions(
        f"""
positive_config = json.loads(pathlib.Path({str(POSITIVE_CONFIG)!r}).read_text())
assert module._eqprop_variant(positive_config['gradient_contract']) == 'positive_one_sided'
for case in positive_config['cases']:
    translated = module._case_for_base(case)
    translated.update({{
        'scheme': case['scheme'],
        'native_T': case['native_inference_iterations'],
        'native_K': case['native_gradient_iterations'],
        'voltage_amp': case['voltage_amplification'],
        'current_amp': case['current_amplification'],
    }})
    grid = module._beta_grid_for_case(
        positive_config['gradient_contract'], translated
    )
    key = f"{{case['scheme']}}/{{case['architecture']}}"
    assert len(grid) == positive_config['completion']['expected_beta_values_by_case'][key]
    assert module._grid_for_case(
        positive_config, translated, smoke=False
    ) == [(translated['native_T'], translated['native_K'])]
"""
    )
    assert set(expected_counts) == {
        f"{case['scheme']}/{case['architecture']}" for case in config["cases"]
    }


def test_positive_selector_uses_lowest_consistent_pair_and_reports_bptt_oracle():
    _run_helper_assertions(
        """
import torch

name = 'ConvWeight_0'
betas = [1e-3, 3e-3, 1e-2]
beta_grid = [
    {
        'beta': beta,
        'beta_hat': beta,
        'beta_hat_requested': beta,
        'beta_effective': beta,
        'capped': False,
        'beta_mode': 'amplification_aware_baseline_output_curvature',
        'baseline_init_output_curvature_scale_b0': 1.0,
        'init_output_curvature_scale_b0': 1.0,
        'amplification_depth_L': 1,
        'amplification_factor': 1.0,
    }
    for beta in betas
]
metadata = module._beta_metadata_by_actual(beta_grid)
accumulator = module._ContextAccumulator()
bptt = torch.tensor([0.0, 1.0], dtype=torch.float64)
accumulator.add(('bptt', name), bptt, batch_size=2)
eqprop = {
    1e-3: torch.tensor([1.0, 0.0], dtype=torch.float64),
    3e-3: torch.tensor([1.01, 0.0], dtype=torch.float64),
    1e-2: torch.tensor([0.0, 1.0], dtype=torch.float64),
}
zero = torch.tensor([1.0, 1.0], dtype=torch.float64)
for beta in betas:
    accumulator.add(('eqprop', beta, name), eqprop[beta], batch_size=2)
    accumulator.add(
        ('eqprop_zero_endpoint', beta, name), zero, batch_size=2
    )
    accumulator.add(
        ('eqprop_positive_endpoint', beta, name),
        zero + beta * eqprop[beta],
        batch_size=2,
    )
precision = module._finite_difference_precision_gates(
    accumulator,
    names=[name],
    betas=betas,
    expected_examples=2,
    ratio_threshold=100.0,
)
assert all(row['passed'] for row in precision.values())
context = {
    'architecture': 'conv1',
    'scheme': 'baseline',
    'checkpoint_role': 'reconstructed_initialization',
    'T': 4,
    'K': 4,
    'native_context': True,
    'shared_anchor_context': True,
    'init_output_curvature_scale_b0': 1.0,
    'checkpoint_output_curvature_scale_b0': 1.0,
}
residual = {
    beta: {'passed': True, 'failed_layer_phases': []} for beta in betas
}
rows = module._parameter_rows_for_context(
    accumulator,
    context=context,
    names=[name],
    types={name: 'ConvWeight'},
    betas=betas,
    expected_examples=2,
    zero_epsilon=1e-12,
    eqprop_variant='positive_one_sided',
    residual_gates=residual,
    residual_gate_required=True,
    beta_metadata=metadata,
    precision_gates=precision,
    precision_gate_required=True,
)
module._attach_adjacent_beta_consistency(
    rows,
    accumulator,
    names=[name],
    beta_grid=beta_grid,
    expected_examples=2,
)
assert rows[1]['adjacent_consistency_passed'] is True
assert rows[2]['adjacent_consistency_passed'] is False
selected = module._select_context_beta(
    rows,
    context,
    eqprop_variant='positive_one_sided',
)
assert selected['selected_beta'] == 3e-3
assert selected['selected_pair_lower_beta'] == 1e-3
assert selected['selection_rule'] == 'lowest_uncapped_adjacent_consistent_pair_choose_larger'
assert selected['oracle_selected_beta'] == 1e-2
assert selected['worst_layer_positive_one_sided_cosine'] == 0.0

# A capped point is diagnostic/oracle-only and cannot complete a pair.
beta_grid[-1] = {**beta_grid[-1], 'capped': True}
module._attach_adjacent_beta_consistency(
    rows,
    accumulator,
    names=[name],
    beta_grid=beta_grid,
    expected_examples=2,
)
assert rows[-1]['adjacent_pair_uncapped_consecutive'] is False
assert rows[-1]['adjacent_consistency_passed'] is False
"""
    )


def test_exact_per_sample_projected_kkt_residual_and_one_sided_gate():
    _run_helper_assertions(
        """
import math
import torch

state = torch.tensor([
    [[0.0], [0.0]],
    [[1.0], [-1.0]],
])
gradient = torch.tensor([
    [[2.0], [-3.0]],
    [[-4.0], [5.0]],
])
projected = module._projected_perfect_diode_residual(
    state, gradient, epsilon=1e-8
)
assert torch.equal(
    projected,
    torch.tensor([[[0.0], [0.0]], [[4.0], [5.0]]]),
)

class PerfectLayer:
    name = 'hidden'
    non_linearity = 'perfect_diode'
    state = torch.tensor([
        [[0.0], [0.0]],
        [[1.0], [-1.0]],
    ])

class OutputLayer:
    name = 'output'
    non_linearity = 'identity'
    state = torch.zeros((2, 1))

class GradientSource:
    def __init__(self, gradients):
        self.gradients = gradients

    def grad_layer_fn(self, layer):
        return lambda: self.gradients[layer.name]

layers = [PerfectLayer(), OutputLayer()]
source = GradientSource({
    'hidden': torch.tensor([
        [[1e-4], [-1e-4]],
        [[2e-4], [-2e-4]],
    ]),
    'output': torch.tensor([[1e-4], [2e-4]]),
})
runtime = {'free_layers': layers}
accumulator = module._ResidualAccumulator()
for phase, beta in (('post_T_free', 0.0), ('zero', 0.0), ('positive', 1e-4)):
    accumulator.add_endpoint(
        runtime,
        gradient_source=source,
        phase=phase,
        beta=beta,
        source_indices=[7, 11],
    )
context = {
    'architecture': 'conv1',
    'scheme': 'baseline',
    'checkpoint_role': 'reconstructed_initialization',
    'eqprop_variant': 'positive_one_sided',
    'T': 4,
    'K': 4,
    'native_T': 4,
    'native_K': 4,
    'native_context': True,
    'shared_anchor_context': True,
}
summaries = accumulator.summary_rows(
    context=context,
    expected_examples=2,
    p90_threshold=1e-2,
)
assert len(summaries) == 6
assert all(row['coverage_complete'] for row in summaries)
assert all(row['gate_passed'] for row in summaries)
assert all(row['unique_source_index_count'] == 2 for row in summaries)
assert all(len(row['per_sample_selected_max_sha256']) == 64 for row in summaries)
gates = module._residual_gate_by_beta(
    summaries,
    betas=[1e-4],
    eqprop_variant='positive_one_sided',
)
assert gates[1e-4]['passed'] is True
assert gates[1e-4]['required_layer_phase_count'] == 6
archive = accumulator.archive_records(context=context)
assert len(archive) == 6
assert all(values.dtype.names == (
    'source_index', 'selected_max', 'raw_max', 'clamp_occupancy'
) for _, values in archive)
"""
    )


def test_state_displacement_references_and_squared_sum_pooling():
    _run_helper_assertions(
        """
import math
import torch

class PerfectLayer:
    name = 'hidden'
    non_linearity = 'perfect_diode'

class LinearLayer:
    name = 'output'
    non_linearity = 'identity'

layers = [PerfectLayer(), LinearLayer()]
current = [torch.tensor([[0.0, 0.0]]), torch.tensor([[4.0]])]
post_t_free = [torch.tensor([[1.0, -1.0]]), torch.tensor([[3.0]])]
matched_zero_k = [torch.tensor([[0.5, -0.5]]), torch.tensor([[3.5]])]

post_rows = module._state_displacement_rows(
    layers,
    current,
    post_t_free,
    reference_kind='post_T_free',
    reference_norm_epsilon=1e-12,
)
matched_rows = module._state_displacement_rows(
    layers,
    current,
    matched_zero_k,
    reference_kind='matched_zero_K',
    reference_norm_epsilon=1e-12,
)
assert len(post_rows) == len(matched_rows) == 3
assert {row['reference_kind'] for row in post_rows} == {'post_T_free'}
assert {row['reference_kind'] for row in matched_rows} == {'matched_zero_K'}

# Signed cancellation must not erase the free state's nonzero RMS.
hidden = next(row for row in post_rows if row['state_layer_name'] == 'hidden')
assert hidden['reference_sum'] == 0.0
assert hidden['reference_squared_sum'] == 2.0
assert hidden['reference_rms'] == 1.0
assert hidden['reference_min'] == -1.0 and hidden['reference_max'] == 1.0
output = next(row for row in post_rows if row['state_layer_name'] == 'output')
assert output['current_sum'] == 4.0 and output['reference_sum'] == 3.0
assert output['delta_sum'] == 1.0 and output['current_squared_sum'] == 16.0

post_aggregate = next(row for row in post_rows if row['state_layer_name'] == '__all__')
matched_aggregate = next(row for row in matched_rows if row['state_layer_name'] == '__all__')
assert math.isclose(post_aggregate['delta_squared_sum'], 3.0)
assert math.isclose(post_aggregate['reference_squared_sum'], 11.0)
assert math.isclose(post_aggregate['relative_displacement'], math.sqrt(3.0 / 11.0))
assert post_aggregate['active_set_transition_count'] == 2
assert post_aggregate['constrained_element_count'] == 2
assert math.isclose(matched_aggregate['delta_squared_sum'], 0.75)
assert math.isclose(matched_aggregate['reference_squared_sum'], 12.75)
assert math.isclose(
    matched_aggregate['relative_displacement'], math.sqrt(0.75 / 12.75)
)

second_current = [torch.tensor([[2.0, -2.0]]), torch.tensor([[13.0]])]
second_post_rows = module._state_displacement_rows(
    layers,
    second_current,
    post_t_free,
    reference_kind='post_T_free',
    reference_norm_epsilon=1e-12,
)
context = {
    'architecture': 'conv1',
    'scheme': 'ours',
    'checkpoint_role': 'best_validation',
    'T': 4,
    'K': 4,
    'native_T': 4,
    'native_K': 4,
    'native_context': True,
    'shared_anchor_context': True,
    'beta': 0.1,
    'phase': 'positive',
}
batch_rows = [
    {**context, **row}
    for row in post_rows + second_post_rows + matched_rows
]
summaries = module._summarize_state_rows(batch_rows)
pooled_post = next(
    row for row in summaries
    if row['reference_kind'] == 'post_T_free'
    and row['state_layer_name'] == '__all__'
)
pooled_matched = next(
    row for row in summaries
    if row['reference_kind'] == 'matched_zero_K'
    and row['state_layer_name'] == '__all__'
)
assert pooled_post['batch_count'] == 2
assert math.isclose(pooled_post['pooled_delta_squared_sum'], 105.0)
assert math.isclose(pooled_post['pooled_reference_squared_sum'], 22.0)
assert math.isclose(
    pooled_post['pooled_relative_displacement'], math.sqrt(105.0 / 22.0)
)
assert pooled_matched['batch_count'] == 1
assert math.isclose(
    pooled_matched['pooled_relative_displacement'], math.sqrt(0.75 / 12.75)
)
"""
    )


def test_maximin_beta_selection_excludes_dead_and_incomplete_then_breaks_ties_small():
    _run_helper_assertions(
        """
context = {
    'architecture': 'conv2',
    'scheme': 'legacy',
    'checkpoint_role': 'best_validation',
    'T': 6,
    'K': 6,
}
rows = []
for beta, cosines, complete in (
    (0.003, (None, None), (True, True)),
    (0.01, (0.8, 0.9), (True, True)),
    (0.03, (0.8, 0.95), (True, True)),
    (0.1, (0.99, 0.99), (True, False)),
):
    for index, (cosine, coverage) in enumerate(zip(cosines, complete)):
        rows.append({
            'beta': beta,
            'parameter_name': f'weight_{index}',
            'cosine': cosine,
            'coverage_complete': coverage,
        })

selected = module._select_context_beta(rows, context)
assert selected['selected_beta'] == 0.01
assert selected['worst_layer_centered_cosine'] == 0.8
assert selected['worst_layer'] == 'weight_0'
assert selected['selectable_beta_count'] == 2
assert selected['tested_beta_count'] == 4
assert selected['selection_rule'] == 'maximize_worst_layer_then_smaller_beta'

no_selection = module._select_context_beta(
    [
        {
            'beta': 0.5,
            'parameter_name': 'weight_0',
            'cosine': None,
            'coverage_complete': True,
        },
        {
            'beta': 1.0,
            'parameter_name': 'weight_0',
            'cosine': 1.0,
            'coverage_complete': False,
        },
    ],
    context,
)
assert no_selection['selected_beta'] is None
assert no_selection['selectable_beta_count'] == 0
assert no_selection['outcome'] == 'no_finite_all_layer_beta'
"""
    )


def test_fixed_beta_tk_summary_freezes_shared_anchor_beta_without_retuning():
    _run_helper_assertions(
        """
import math

architecture = 'conv3'
scheme = 'baseline'
role = 'best_validation'
context_specs = [
    # T, K, native, shared, independently selected beta, retuned maximin cosine
    (8, 8, False, True, 0.1, 0.8),
    (12, 8, True, False, 0.003, 0.95),
    (8, 16, False, False, 0.25, 0.9),
    (12, 16, False, False, 0.5, 0.99),
]
context_rows = [
    {
        'architecture': architecture,
        'scheme': scheme,
        'checkpoint_role': role,
        'T': t_value,
        'K': k_value,
        'native_context': native,
        'shared_anchor_context': shared,
        'selected_beta': selected_beta,
        'worst_layer_centered_cosine': maximin,
    }
    for t_value, k_value, native, shared, selected_beta, maximin in context_specs
]

fixed_cosines = {
    (8, 8): (0.8, 0.9),
    (12, 8): (0.4, 0.5),
    (8, 16): (0.6, 0.7),
    (12, 16): (0.2, 0.3),
}
parameter_rows = []
for t_value, k_value, native, shared, _, _ in context_specs:
    for index, cosine in enumerate(fixed_cosines[(t_value, k_value)]):
        parameter_rows.append({
            'architecture': architecture,
            'scheme': scheme,
            'checkpoint_role': role,
            'T': t_value,
            'K': k_value,
            'native_context': native,
            'shared_anchor_context': shared,
            'beta': 0.1,
            'parameter_name': f'weight_{index}',
            'coverage_complete': True,
            'cosine': cosine,
        })

# The native coordinate has much better values at its independently selected
# beta. They must not leak into the fixed-beta T/K diagnostic.
for index, cosine in enumerate((0.95, 0.96)):
    parameter_rows.append({
        'architecture': architecture,
        'scheme': scheme,
        'checkpoint_role': role,
        'T': 12,
        'K': 8,
        'native_context': True,
        'shared_anchor_context': False,
        'beta': 0.003,
        'parameter_name': f'weight_{index}',
        'coverage_complete': True,
        'cosine': cosine,
    })

beta, fixed_layer_rows = module._fixed_beta_layer_rows(
    parameter_rows,
    context_rows,
    architecture=architecture,
    scheme=scheme,
    role=role,
)
assert beta == 0.1
assert len(fixed_layer_rows) == 8
assert {row['beta'] for row in fixed_layer_rows} == {0.1}

fixed_rows = module._fixed_beta_context_rows(parameter_rows, context_rows)
assert len(fixed_rows) == 4
by_coordinate = {(row['T'], row['K']): row for row in fixed_rows}
assert by_coordinate[(8, 8)]['shared_anchor_context'] is True
assert by_coordinate[(12, 8)]['native_context'] is True
assert by_coordinate[(12, 8)]['fixed_beta'] == 0.1
assert by_coordinate[(12, 8)]['worst_layer_centered_cosine'] == 0.4
assert by_coordinate[(12, 8)]['worst_layer'] == 'weight_0'
assert by_coordinate[(12, 16)]['worst_layer_centered_cosine'] == 0.2
assert all(row['coverage_complete'] for row in fixed_rows)

study = module._study_summary_rows(context_rows, fixed_rows)
assert len(study) == 1
summary = study[0]
assert (summary['native_T'], summary['native_K']) == (12, 8)
assert (summary['shared_T'], summary['shared_K']) == (8, 8)
assert summary['native_selected_beta'] == 0.003
assert summary['shared_selected_beta'] == 0.1
assert summary['fixed_beta'] == 0.1
assert math.isclose(summary['fixed_beta_vary_T_worst_cosine_range'], 0.4)
assert math.isclose(summary['fixed_beta_vary_K_worst_cosine_range'], 0.2)
assert math.isclose(summary['fixed_beta_full_grid_worst_cosine_range'], 0.6)

# A purpose-built one-beta Stage-B run has no adjacent-beta selection.  Its
# sole declared/measured beta must still drive the fixed-beta T/K summary.
single_context_rows = [
    {
        **row,
        'eqprop_variant': 'positive_one_sided',
        'selected_beta': None,
        'selected_beta_hat': None,
        'selected_beta_hat_requested': None,
    }
    for row in context_rows
]
single_parameter_rows = [
    {
        **row,
        'beta_hat': 1e-3,
        'beta_hat_requested': 1e-3,
    }
    for row in parameter_rows
    if row['beta'] == 0.1
]
single_beta, single_layers = module._fixed_beta_layer_rows(
    single_parameter_rows,
    single_context_rows,
    architecture=architecture,
    scheme=scheme,
    role=role,
)
assert single_beta == 0.1
assert len(single_layers) == 8
single_fixed = module._fixed_beta_context_rows(
    single_parameter_rows, single_context_rows
)
assert len(single_fixed) == 4
assert all(row['fixed_beta_source'] == 'single_declared_fixed_beta' for row in single_fixed)
assert all(row['fixed_beta_hat'] == 1e-3 for row in single_fixed)
assert all(row['coverage_complete'] for row in single_fixed)
single_study = module._study_summary_rows(single_context_rows, single_fixed)
assert single_study[0]['fixed_beta'] == 0.1
assert single_study[0]['fixed_beta_hat'] == 1e-3
"""
    )


def test_jeanzay_wrapper_is_single_job_fail_closed_and_uses_transport_overrides():
    syntax = subprocess.run(
        ["bash", "-n", str(SLURM_WRAPPER)],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    text = SLURM_WRAPPER.read_text(encoding="utf-8")
    assert "#SBATCH --account=fmu@v100" in text
    assert "#SBATCH --partition=gpu_p13" in text
    assert "#SBATCH --qos=qos_gpu-dev" in text
    assert "#SBATCH --constraint=v100-32g" in text
    assert "#SBATCH --time=02:00:00" in text
    assert "#SBATCH --array" not in text
    assert "--source-study-root" in text
    assert "--mechanism-initialization-hash-artifact" in text
    assert "--runtime-source-root" in text
    assert "--run-id analysis" in text
    assert '"${EQ_WRAPPER_SHA256:?Set the frozen Slurm-wrapper SHA-256.}"' in text
    assert 'check_sha256 "${wrapper}" "${EQ_WRAPPER_SHA256}"' in text
    assert 'echo "wrapper_sha256=${EQ_WRAPPER_SHA256}"' in text

    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
    }
    missing_contract = subprocess.run(
        ["bash", str(SLURM_WRAPPER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_contract.returncode != 0
    assert "EQ_ANALYSIS_ROOT" in missing_contract.stderr


def test_completion_gate_accepts_required_false_read_only_sentinels():
    _run_helper_assertions(
        """
passing = {
    'criteria_met': True,
    'all_source_bytes_unchanged': True,
    'all_parameter_tensors_unchanged': True,
    'optimizer_steps_applied': False,
    'official_test_read': False,
}
assert module._completion_criteria_met(passing) is True
assert module._completion_criteria_met({**passing, 'official_test_read': True}) is False
assert module._completion_criteria_met({**passing, 'optimizer_steps_applied': True}) is False
assert module._completion_criteria_met({**passing, 'all_source_bytes_unchanged': False}) is False
"""
    )
