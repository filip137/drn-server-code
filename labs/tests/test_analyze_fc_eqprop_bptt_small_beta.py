import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY_ROOT / "experiments/analyze_fc_eqprop_bptt_small_beta.py"
CONFIG = (
    REPOSITORY_ROOT
    / "configs/conv/perfectdiode_fc123_initialization_positive_eqprop_small_beta_seed0_20260810_v1.json"
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
spec = importlib.util.spec_from_file_location('fc_eqprop_small_beta_test', runner)
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
            "MPLCONFIGDIR": "/tmp/matplotlib-fc-eqprop-unit-tests",
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


def test_fc_config_is_matched_dense_initialization_control():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["paper_facing"] is False
    assert config["comparison_scope"]["checkpoint_roles"] == [
        "reconstructed_initialization"
    ]
    assert config["comparison_scope"]["trained_fc_checkpoint_available"] is False
    assert {
        row["scheme"]: (
            row["voltage_amplification"],
            row["current_amplification"],
        )
        for row in config["amplification_schemes"]
    } == {
        "baseline": (1.0, 1.0),
        "ours": (4.0, 1.0),
        "legacy": (4.0, 0.25),
    }
    architectures = {
        row["architecture"]: row for row in config["architecture_controls"]
    }
    assert architectures["fc1"]["hidden_layer_shapes"] == [[64]]
    assert architectures["fc2"]["hidden_layer_shapes"] == [[64], [128]]
    assert architectures["fc3"]["hidden_layer_shapes"] == [[64], [128], [256]]
    assert architectures["fc3"]["native_tk_by_scheme"] == {
        "baseline": [12, 8],
        "ours": [8, 8],
        "legacy": [8, 8],
    }
    assert all(row["long_tk"] == [64, 64] for row in architectures.values())
    gradient = config["gradient_contract"]
    assert gradient["eqprop_variant_primary"] == "positive_one_sided"
    assert gradient["normalized_betas_requested"] == [
        1e-10,
        3e-10,
        1e-9,
        3e-9,
        1e-8,
        3e-8,
        1e-7,
        3e-7,
        1e-6,
        3e-6,
        1e-5,
        3e-5,
        1e-4,
        3e-4,
    ]
    assert gradient["actual_beta_cap"] == 0.01
    assert gradient["nudging_mode"] == "current"
    assert gradient["normalized_beta_parameterization"] == "common_base"
    assert gradient["precision_gate"]["minimum_snr"] == 100.0
    assert gradient["adjacent_beta_consistency_gate"] == {
        "eqprop_cosine_minimum": 0.99,
        "symmetric_norm_delta_maximum": 0.1,
        "symmetric_norm_delta_definition": "2*abs(||g_hi||-||g_lo||)/(||g_hi||+||g_lo||)",
        "selection_rule": "Among consecutive uncapped normalized-beta requests, select the larger beta of the lowest adjacent pair passing residual, precision, and consistency on every scored layer, independently of BPTT cosine. Capped rows remain diagnostic/oracle-only.",
    }
    assert config["equilibrium_residual_contract"]["p90_threshold"] == 0.01
    assert config["equilibrium_residual_contract"]["hard_failure_threshold"] == 0.1
    assert config["model_contract"]["conv_pipeline"] == []
    assert config["model_contract"]["output_shape"] == [20]
    for key in (
        "quadratic_diode_param",
        "hard_sigmoid_param",
        "exponential_diode_param",
    ):
        assert config["model_contract"][key]


def test_source_config_and_curvature_guards_construct_expected_dense_models():
    _run_helper_assertions(
        """
config = json.loads(config_path.read_text(encoding='utf-8'))
module._validate_config(config)

expected_shapes = {
    'fc1': [[2, 28, 28], [64], [20]],
    'fc2': [[2, 28, 28], [64], [128], [20]],
    'fc3': [[2, 28, 28], [64], [128], [256], [20]],
}
for architecture, shapes in expected_shapes.items():
    hashes = set()
    for scheme in ('baseline', 'ours', 'legacy'):
        source = module._source_config(
            config, architecture=architecture, scheme=scheme
        )
        model = module.base._model_config(source)
        assert model['layer_shapes'] == shapes
        assert model['conv_pipeline'] == []
        assert model['weight_gains'] == [1.0] * (len(shapes) - 1)
        runtime = module.base._build_runtime(
            source,
            device=module.torch.device('cpu'),
            gradient_iterations=source['model_base']['num_iterations_training'],
        )
        assert runtime['weight_names'] == [
            f'DenseWeight_{index}' for index in range(len(shapes) - 1)
        ]
        assert all(kind == 'DenseWeight' for kind in runtime['weight_types'])
        hashes.add(module.base._historical_initialization_sha256(runtime['parameters']))
        curvature = module._output_curvature_scale(runtime)
        expected = config['expected_reconstructed_initialization_guards'][architecture]
        assert curvature['b0_scheme_init'] == expected['b0_by_scheme'][scheme]
        assert next(iter(hashes)) == expected['tensor_sha256']
    assert len(hashes) == 1
"""
    )


def test_amplification_normalized_beta_mapping_caps_and_deduplicates():
    _run_helper_assertions(
        """
import math

all_rows, retained = module._derive_beta_points(
    normalized_requests=[1e-3, 1e-2],
    b0_baseline_init=2.0,
    b0_scheme_init=0.5,
    voltage_amplification=4.0,
    current_amplification=1.0,
    amplification_exponent=1,
    actual_beta_cap=0.03,
)
assert len(all_rows) == len(retained) == 2
assert math.isclose(retained[0]['beta'], 1e-3 * 2.0 / 4.0)
assert math.isclose(retained[1]['beta'], 1e-2 * 2.0 / 4.0)
assert math.isclose(retained[1]['effective_beta'], 0.02)
assert math.isclose(retained[1]['beta_hat_effective'], 1e-2)
assert math.isclose(retained[1]['actual_beta_over_b0_scheme_init'], 1e-2)

all_rows, retained = module._derive_beta_points(
    normalized_requests=[0.01, 0.02, 0.03],
    b0_baseline_init=10.0,
    b0_scheme_init=10.0,
    voltage_amplification=1.0,
    current_amplification=1.0,
    amplification_exponent=2,
    actual_beta_cap=0.03,
)
assert [row['beta'] for row in all_rows] == [0.03, 0.03, 0.03]
assert [row['retained_after_actual_beta_deduplication'] for row in all_rows] == [
    True, False, False
]
assert len(retained) == 1
assert retained[0]['beta_was_capped'] is True

all_rows, retained = module._derive_beta_points(
    normalized_requests=[1e-3, 1e-2],
    b0_baseline_init=2.0,
    b0_scheme_init=0.5,
    voltage_amplification=4.0,
    current_amplification=1.0,
    amplification_exponent=1,
    actual_beta_cap=0.01,
    normalized_beta_parameterization='common_base',
)
assert math.isclose(all_rows[0]['actual_beta_uncapped'], 0.002)
assert math.isclose(all_rows[0]['beta'], 0.002)
assert math.isclose(all_rows[0]['effective_beta'], 0.008)
assert all_rows[0]['beta_was_capped'] is False
assert math.isclose(all_rows[1]['beta'], 0.0025)
assert math.isclose(all_rows[1]['effective_beta'], 0.01)
assert all_rows[1]['beta_was_capped'] is True
assert all_rows[1]['cap_semantics'] == 'injected_beta'
"""
    )


def test_precision_adjacent_selection_and_bptt_oracle_are_separate():
    _run_helper_assertions(
        """
import torch

eps = module.FLOAT32_EPSILON
low = module._precision_metrics(
    torch.tensor([1.0]),
    torch.tensor([1.0 + 10.0 * eps]),
    minimum_snr=100.0,
)
high = module._precision_metrics(
    torch.tensor([1.0]),
    torch.tensor([1.0 + 400.0 * eps]),
    minimum_snr=100.0,
)
assert low['precision_gate_passed'] is False
assert high['precision_gate_passed'] is True

consistent = module._adjacent_metrics(
    torch.tensor([0.0, 1.0]),
    torch.tensor([0.0, 1.05]),
    cosine_minimum=0.99,
    symmetric_norm_delta_maximum=0.1,
)
inconsistent = module._adjacent_metrics(
    torch.tensor([0.0, 1.0]),
    torch.tensor([1.0, 0.0]),
    cosine_minimum=0.99,
    symmetric_norm_delta_maximum=0.1,
)
assert consistent['adjacent_consistency_passed'] is True
assert inconsistent['adjacent_consistency_passed'] is False

def point(index, beta, capped=False):
    return {
        'request_index': index,
        'beta_hat_requested': beta,
        'actual_beta_uncapped': beta,
        'beta': beta,
        'actual_beta_cap': 0.03,
        'beta_was_capped': capped,
        'retained_after_actual_beta_deduplication': True,
        'deduplicated_to_request_index': index,
        'amplification_depth_L': 2,
        'amplification_exponent_convention': 'output_bias_current_row',
        'nudging_effective_multiplier': 1.0,
        'nudging_physical_multiplier': 1.0,
        'effective_beta': beta,
        'beta_hat_effective': beta,
        'b0_baseline_init': 1.0,
        'b0_scheme_init': 1.0,
        'actual_beta_over_b0_scheme_init': beta,
    }

points = [point(0, 1e-4), point(1, 3e-4), point(2, 1e-3)]
accumulator = module._MeanAccumulator()
name = 'DenseWeight_0'
bptt = torch.tensor([1.0, 0.0])
accumulator.add(('bptt', name), bptt, 2)
accumulator.add(('zero', name), torch.zeros(2), 2)
eqprops = {
    1e-4: torch.tensor([0.0, 1.0]),
    3e-4: torch.tensor([0.0, 1.05]),
    1e-3: torch.tensor([1.0, 0.0]),
}
for beta, estimate in eqprops.items():
    accumulator.add(('eqprop', beta, name), estimate, 2)
    accumulator.add(('positive', beta, name), estimate * beta, 2)
residual_gates = {
    beta: {'passed': True, 'failed_layer_phases': []} for beta in eqprops
}
context = {
    'architecture': 'fc1',
    'matched_conv_architecture': 'conv1',
    'scheme': 'baseline',
    'checkpoint_role': module.CHECKPOINT_ROLE,
    'coordinate_kind': 'native',
    'T': 4,
    'K': 4,
    'native_T': 4,
    'native_K': 4,
    'native_context': True,
    'shared_anchor_context': True,
    'voltage_amplification': 1.0,
    'current_amplification': 1.0,
    'b0_baseline_init': 1.0,
    'b0_scheme_init': 1.0,
}
rows, selection = module._parameter_rows_and_selection(
    accumulator,
    context=context,
    points=points,
    names=[name],
    types={name: 'DenseWeight'},
    expected_examples=2,
    zero_epsilon=1e-12,
    residual_gates=residual_gates,
    minimum_snr=100.0,
    sentinel_max_beta_hat=1e-5,
    adjacent_cosine_minimum=0.99,
    adjacent_norm_delta_maximum=0.1,
)
assert selection['selected_beta'] == 3e-4
assert selection['selection_independent_of_bptt_cosine'] is True
assert selection['oracle_beta'] == 1e-3
assert next(row for row in rows if row['beta'] == 3e-4)[
    'selected_by_adjacent_convergence_rule'
] is True
assert next(row for row in rows if row['beta'] == 1e-3)[
    'oracle_selected_by_bptt_cosine'
] is True

capped_points = [point(0, 1e-4), point(1, 3e-4, capped=True)]
capped_accumulator = module._MeanAccumulator()
capped_accumulator.add(('bptt', name), bptt, 2)
capped_accumulator.add(('zero', name), torch.zeros(2), 2)
for candidate in capped_points:
    beta = candidate['beta']
    estimate = torch.tensor([1.0, 0.0])
    capped_accumulator.add(('eqprop', beta, name), estimate, 2)
    capped_accumulator.add(('positive', beta, name), estimate * beta, 2)
_, capped_selection = module._parameter_rows_and_selection(
    capped_accumulator,
    context=context,
    points=capped_points,
    names=[name],
    types={name: 'DenseWeight'},
    expected_examples=2,
    zero_epsilon=1e-12,
    residual_gates={
        candidate['beta']: {'passed': True, 'failed_layer_phases': []}
        for candidate in capped_points
    },
    minimum_snr=100.0,
    sentinel_max_beta_hat=1e-5,
    adjacent_cosine_minimum=0.99,
    adjacent_norm_delta_maximum=0.1,
)
assert capped_selection['selected_beta'] is None
assert capped_selection['oracle_beta'] in {1e-4, 3e-4}
"""
    )


def test_runner_help_bootstraps_exact_runtime():
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = "/tmp/matplotlib-fc-eqprop-help-test"
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--runtime-source-root" in result.stdout
    assert "--smoke" in result.stdout
