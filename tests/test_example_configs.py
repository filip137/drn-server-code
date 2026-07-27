from pathlib import Path

from experiments import RunMode, resolve_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_base_example_resolves_every_documented_mode() -> None:
    path = ROOT / "examples" / "small_drn" / "base.json"

    for mode in (RunMode.TRAIN, RunMode.LINSPACE, RunMode.VALIDATE):
        definition, spec = resolve_experiment_config(path, mode)
        assert definition.experiment_id == "small_drn.v1"
        assert spec.experiment_id == "small_drn.v1"


def test_hardware_aware_example_is_nested_focused_and_runnable() -> None:
    path = ROOT / "examples" / "small_drn" / "hardware_aware.json"

    definition, spec = resolve_experiment_config(path, RunMode.TRAIN)

    assert definition.experiment_id == "small_drn.v1"
    assert spec.common.runtime.device == "cpu"
    assert spec.common.data.dataset == "moons"
    assert spec.common.model.dims == (4, 2)
    assert (
        spec.extensions.model_adapter,
        spec.extensions.weight_modifier,
        spec.extensions.update_backend,
        spec.extensions.algorithm,
    ) == (
        "none",
        "add_normal",
        "direct",
        "ep",
    )
    assert dict(spec.settings.weight_modifier.parameters) == {
        "std_dev": 0.06,
        "seed": 7,
        "noisy_evaluation": True,
    }
