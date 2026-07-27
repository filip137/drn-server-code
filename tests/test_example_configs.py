from pathlib import Path

from experiments import RunMode, resolve_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_base_example_resolves_every_documented_mode() -> None:
    path = ROOT / "examples" / "small_drn" / "base.json"

    for mode in (RunMode.TRAIN, RunMode.LINSPACE, RunMode.VALIDATE):
        definition, spec = resolve_experiment_config(path, mode)
        assert definition.experiment_id == "small_drn.v1"
        assert spec.experiment_id == "small_drn.v1"


def test_passive_low_rank_example_is_nested_and_resolves_every_mode() -> None:
    path = ROOT / "examples" / "small_drn" / "lora.json"

    resolved = {}
    for mode in (RunMode.TRAIN, RunMode.LINSPACE, RunMode.VALIDATE):
        definition, spec = resolve_experiment_config(path, mode)
        assert definition.experiment_id == "small_drn.v1"
        assert spec.experiment_id == "small_drn.v1"
        resolved[mode] = spec

    common = resolved[RunMode.TRAIN].common
    train = resolved[RunMode.TRAIN].settings
    assert common.runtime.seed == 7
    assert common.runtime.data_seed == 11
    assert common.data.batch_size == 8
    assert common.data.num_points == 32
    assert common.model.dims == (4, 2)
    assert common.model.adapter.type == "passive_low_rank"
    assert dict(common.model.adapter.parameters) == {
        "rank": 2,
        "input_factor_gain": 0.01,
        "input_factor_min": 1e-7,
        "conductance_max": 1.0,
        "output_factor_init": "zero",
        "output_off_conductance": None,
    }
    assert train.learning_rates == (0.01, 0.01)
    assert train.bias_learning_rates == ()
    assert train.weight_modifier.type == "none"
    assert train.update_backend.type == "direct"
