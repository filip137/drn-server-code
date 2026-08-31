from __future__ import annotations

import pytest

from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam_config import (
    EXPERIMENT_ID,
    parse_winsorized_onchip_adam_config,
    resolve_winsorized_onchip_adam_spec,
)
from experiments.schema import ConfigError, RunMode


def _payload(spacing: int = 2) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "spacing_delta_x_multiplier": spacing,
        "target_assignment_seed": 87003,
        "p0_endpoint_seed": 89301,
    }


@pytest.mark.parametrize("spacing", (1, 2, 4))
def test_strict_onchip_config_resolves_fixed_cuda_protocol(spacing: int) -> None:
    document = parse_winsorized_onchip_adam_config(_payload(spacing))
    spec = resolve_winsorized_onchip_adam_spec(document, RunMode.TRAIN)
    assert spec.protocol.spacing_delta_x_multiplier == spacing
    assert spec.protocol.p0.assignment_seed == 87003
    assert spec.protocol.p0.endpoint_seed == 89301
    assert spec.protocol.p0.maximum_random_draws == 385
    assert spec.protocol.execution.device == "cuda"
    assert spec.protocol.execution.state_authority == "persistent_raw_active_plant_only"
    assert spec.student.settings.max_batches is None


def test_onchip_config_rejects_target_or_extra_field_changes() -> None:
    payload = _payload()
    payload["target_assignment_seed"] = 87002
    with pytest.raises(ConfigError):
        parse_winsorized_onchip_adam_config(payload)

    payload = _payload()
    payload["unexpected"] = True
    with pytest.raises(ConfigError):
        parse_winsorized_onchip_adam_config(payload)
