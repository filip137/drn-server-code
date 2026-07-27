from pathlib import Path

from experiments import RunMode, resolve_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def test_base_example_resolves_every_documented_mode() -> None:
    path = ROOT / "examples" / "small_drn" / "base.json"

    for mode in (RunMode.TRAIN, RunMode.LINSPACE, RunMode.VALIDATE):
        definition, spec = resolve_experiment_config(path, mode)
        assert definition.experiment_id == "small_drn.v1"
        assert spec.experiment_id == "small_drn.v1"
