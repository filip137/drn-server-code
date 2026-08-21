from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from ebl.cli import CharacterizeRequest, CommandHandlers, main
from experiments.definitions import resolve_experiment_config
from experiments.schema import ConfigError, RunMode
from experiments.study_workflow import prepare_study, summarize_study


_ROOT = Path(__file__).resolve().parents[1]
_SMOKE = _ROOT / "examples" / "reram_program_verify" / "smoke_om.json"
_PRODUCTION = tuple(
    _ROOT / "examples" / "reram_program_verify" / name
    for name in (
        "production_om_continuous.json",
        "production_om_corrupt.json",
        "production_hfo2_continuous.json",
        "production_hfo2_corrupt.json",
    )
)
_STUDY = _ROOT / "studies" / "ibm-reram-program-verify-noise-20260821-v1.json"
_PRODUCTION_SHORT = tuple(
    _ROOT / "examples" / "reram_program_verify" / name
    for name in (
        "production_short_om_continuous.json",
        "production_short_om_corrupt.json",
        "production_short_hfo2_continuous.json",
        "production_short_hfo2_corrupt.json",
    )
)
_SHORT_STUDY = (
    _ROOT / "studies" / "ibm-reram-program-verify-noise-20260821-v2.json"
)


def test_reram_characterization_config_resolves_strictly() -> None:
    definition, spec = resolve_experiment_config(_SMOKE, RunMode.CHARACTERIZE)
    assert definition.experiment_id == "ibm_reram_program_verify.v1"
    assert spec.device.preset == "reram_array_om"
    assert spec.settings.controllers == ("one_pulse", "adaptive")
    assert spec.settings.start_protocols == ("lower_to_target", "upper_to_target")


@pytest.mark.parametrize("path", _PRODUCTION)
def test_every_declared_production_arm_has_the_exact_frozen_geometry(path: Path) -> None:
    _, spec = resolve_experiment_config(path, RunMode.CHARACTERIZE)
    settings = spec.settings
    trajectories = (
        settings.num_devices
        * settings.repeats_per_device
        * settings.target_points
        * len(settings.tolerance_step_ratios)
        * len(settings.start_protocols)
        * len(settings.controllers)
    )

    assert settings.profile == "production"
    assert trajectories == 16_121_856


@pytest.mark.parametrize("path", _PRODUCTION_SHORT)
def test_every_short_production_arm_has_the_exact_frozen_geometry(path: Path) -> None:
    _, spec = resolve_experiment_config(path, RunMode.CHARACTERIZE)
    settings = spec.settings
    trajectories = (
        settings.num_devices
        * settings.repeats_per_device
        * settings.target_points
        * len(settings.tolerance_step_ratios)
        * len(settings.start_protocols)
        * len(settings.controllers)
    )

    assert spec.runtime.device == "cuda"
    assert settings.profile == "production_short"
    assert settings.conditioning.quiet_steps == 4
    assert trajectories == 671_744


def test_declared_study_prepares_and_remains_planned_without_native_runs(
    tmp_path: Path,
) -> None:
    study_dir = prepare_study(_STUDY, tmp_path)
    summary = summarize_study(study_dir)

    assert summary["study_id"] == "ibm-reram-program-verify-noise-20260821-v1"
    assert summary["state"] == "planned"
    assert {arm["arm_id"] for arm in summary["arms"]} == {
        "om-continuous",
        "om-corrupt",
        "hfo2-continuous",
        "hfo2-corrupt",
    }


def test_short_study_prepares_and_remains_planned_without_native_runs(
    tmp_path: Path,
) -> None:
    study_dir = prepare_study(_SHORT_STUDY, tmp_path)
    summary = summarize_study(study_dir)

    assert summary["study_id"] == "ibm-reram-program-verify-noise-20260821-v2"
    assert summary["state"] == "planned"
    assert {arm["arm_id"] for arm in summary["arms"]} == {
        "om-continuous",
        "om-corrupt",
        "hfo2-continuous",
        "hfo2-corrupt",
    }


def test_reram_config_rejects_unknown_keys(tmp_path: Path) -> None:
    payload = json.loads(_SMOKE.read_text(encoding="utf-8"))
    payload["modes"]["characterize"]["undeclared"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown"):
        resolve_experiment_config(path, RunMode.CHARACTERIZE)


def test_production_profile_rejects_a_reduced_smoke_matrix(tmp_path: Path) -> None:
    payload = json.loads(_SMOKE.read_text(encoding="utf-8"))
    payload["modes"]["characterize"]["profile"] = "production"
    path = tmp_path / "undersized-production.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="immutable production characterization contract"):
        resolve_experiment_config(path, RunMode.CHARACTERIZE)


def test_short_production_profile_rejects_an_unreviewed_geometry(
    tmp_path: Path,
) -> None:
    payload = json.loads(_PRODUCTION_SHORT[0].read_text(encoding="utf-8"))
    payload["modes"]["characterize"]["repeats_per_device"] = 3
    path = tmp_path / "undersized-short-production.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        ConfigError, match="immutable short-production characterization contract"
    ):
        resolve_experiment_config(path, RunMode.CHARACTERIZE)


def test_controller_calibration_order_is_immutable(tmp_path: Path) -> None:
    payload = json.loads(_SMOKE.read_text(encoding="utf-8"))
    payload["modes"]["characterize"]["controllers"] = ["adaptive", "one_pulse"]
    path = tmp_path / "reversed-controllers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="calibration precedes adaptation"):
        resolve_experiment_config(path, RunMode.CHARACTERIZE)


def test_stochastic_roles_require_distinct_base_seeds(tmp_path: Path) -> None:
    payload = json.loads(_SMOKE.read_text(encoding="utf-8"))
    payload["runtime"]["pulse_seed"] = payload["runtime"]["conditioning_seed"]
    path = tmp_path / "duplicate-seeds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigError, match="distinct base seeds"):
        resolve_experiment_config(path, RunMode.CHARACTERIZE)


def test_characterize_cli_dispatches_immutable_request(tmp_path: Path) -> None:
    seen: list[CharacterizeRequest] = []

    def characterize(request: CharacterizeRequest) -> int:
        seen.append(request)
        return 9

    result = main(
        ["characterize", "--config", str(_SMOKE), "--output-dir", str(tmp_path)],
        handlers=CommandHandlers(characterize=characterize),
    )
    assert result == 9
    assert len(seen) == 1
    assert seen[0].spec.settings.target_points == 5
    assert seen[0].command[:2] == ("ebl", "characterize")


def test_describe_advertises_characterization() -> None:
    stdout = io.StringIO()
    assert main(
        ["describe", "--experiment", "ibm_reram_program_verify.v1", "--json"],
        stdout=stdout,
    ) == 0
    payload = json.loads(stdout.getvalue())
    assert payload["supported_modes"] == ["characterize"]
    assert payload["commands"]["characterize"]["available"] is True
