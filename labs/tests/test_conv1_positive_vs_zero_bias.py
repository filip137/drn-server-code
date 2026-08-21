import json
from pathlib import Path

import pytest
import torch

from experiments import analyze_conv1_positive_vs_zero_bias as analysis
from experiments.prepare_conv1_positive_vs_zero_bias import (
    CASES,
    OUTPUT_ROOT,
    PARAMETER_ORDER,
    PRACTICAL_SIMILARITY_THRESHOLD_PP,
    STUDY_ID,
    WEIGHT_RATES,
    prepare,
)
from model.function.interaction import (
    FUNCTION_CHECKPOINT_FORMAT,
    FUNCTION_CHECKPOINT_VERSION,
)
from model.variable.parameter import Bias


def _recursive_differences(left, right, path=""):
    if type(left) is not type(right):
        return {path}
    if isinstance(left, dict):
        differences = set()
        for key in set(left) | set(right):
            child = f"{path}/{key}"
            if key not in left or key not in right:
                differences.add(child)
            else:
                differences |= _recursive_differences(left[key], right[key], child)
        return differences
    if isinstance(left, list):
        if len(left) != len(right):
            return {path}
        differences = set()
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences |= _recursive_differences(
                left_item, right_item, f"{path}/{index}"
            )
        return differences
    return set() if left == right else {path}


def _write_bias_checkpoint(path: Path, values: torch.Tensor) -> None:
    payload = {
        "format": FUNCTION_CHECKPOINT_FORMAT,
        "version": FUNCTION_CHECKPOINT_VERSION,
        "schema": [
            {
                "name": "Bias_0",
                "type": "model.variable.parameter.Bias",
                "shape": list(values.shape),
                "dtype": str(values.dtype),
            }
        ],
        "states": [values],
    }
    torch.save(payload, path)


def test_generated_configs_are_current_and_explicit():
    digests = prepare(check=True)
    assert len(digests) == 4
    for case in CASES:
        config = json.loads((OUTPUT_ROOT / case.filename).read_text(encoding="utf-8"))
        assert config["study_id"] == STUDY_ID
        assert config["parameter_order"] == list(PARAMETER_ORDER)
        assert config["training_algorithm"] == "BP"
        assert config["model_base"]["non_linearity"] == "perfect_diode"
        assert isinstance(config["model_base"]["exponential_diode_param"], dict)
        assert isinstance(config["model_base"]["quadratic_diode_param"], dict)
        assert config["evaluation"]["official_test"]["policy"] == "disabled"
        assert config["learning_rates_by_parameter"] == {
            **WEIGHT_RATES,
            "Bias_0": case.bias_learning_rate,
        }


def test_arms_differ_only_in_bias_identity_contract_and_rate():
    positive = json.loads((OUTPUT_ROOT / CASES[0].filename).read_text(encoding="utf-8"))
    zero = json.loads((OUTPUT_ROOT / CASES[1].filename).read_text(encoding="utf-8"))
    assert _recursive_differences(positive, zero) == {
        "/arm_id",
        "/bias_contract/learning_rate",
        "/bias_contract/role",
        "/evaluation/inclusion_rule",
        "/handoff/diagnostic_bias_ablation/bias_learning_rate",
        "/handoff/diagnostic_bias_ablation/comparison_role",
        "/learning_rates_by_parameter/Bias_0",
        "/lr/2",
        "/optimizer/learning_rate/2",
        "/reporting/arm_id",
    }


def test_current_bias_implementation_is_positive_only():
    bias = Bias((3,), gain=0.0, device="cpu")
    with torch.no_grad():
        bias.state.copy_(torch.tensor([-1.0, 0.0, 2.0]))
        bias.clamp_()
    assert torch.equal(bias.state, torch.tensor([0.0, 0.0, 2.0]))


@pytest.mark.parametrize(
    ("delta", "active", "expected"),
    [
        (
            PRACTICAL_SIMILARITY_THRESHOLD_PP,
            True,
            "no_material_effect_observed",
        ),
        (
            -PRACTICAL_SIMILARITY_THRESHOLD_PP,
            True,
            "no_material_effect_observed",
        ),
        (0.5000001, True, "material_difference_observed"),
        (0.0, False, "inconclusive_no_positive_bias_learned"),
    ],
)
def test_predeclared_classification(delta, active, expected):
    assert (
        analysis._classify(
            delta_best_validation_accuracy_pp=delta,
            positive_bias_active=active,
        )
        == expected
    )


def test_bias_checkpoint_stats_measure_sign_and_exact_zero(tmp_path):
    positive_path = tmp_path / "positive.pt"
    zero_path = tmp_path / "zero.pt"
    _write_bias_checkpoint(
        positive_path, torch.tensor([0.0, 0.25, 0.0, 1.0], dtype=torch.float32)
    )
    _write_bias_checkpoint(zero_path, torch.zeros(4, dtype=torch.float32))

    positive = analysis._bias_stats(
        positive_path, role="positive_only_bias", label="final"
    )
    zero = analysis._bias_stats(zero_path, role="zero_bias", label="final")

    assert positive["negative_count"] == 0
    assert positive["positive_count"] == 2
    assert positive["nonzero_percent"] == 50.0
    assert zero["negative_count"] == zero["positive_count"] == 0
    assert zero["zero_count"] == 4
