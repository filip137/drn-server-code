from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.mnist_relu_drn.figure6_om_256_ladder_config import (
    EXPERIMENT_ID,
    load_ladder_config,
    parse_ladder_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/mnist_relu_drn/figure6_om_784_256_10_postpv_fault/full.json"


def test_reviewed_full_ladder_config_is_exact() -> None:
    config = load_ladder_config(CONFIG)
    assert config["experiment_id"] == EXPERIMENT_ID
    assert config["student"]["physical_dims"] == [1568, 512, 20]
    assert config["adam"]["forward_state"] == "held_apparent"
    assert len(config["arms"]) == 8


def test_ladder_config_rejects_unknown_or_changed_protocol_fields() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["adam"]["forward_state"] = "persistent"
    with pytest.raises(ValueError, match="config.adam.forward_state"):
        parse_ladder_config(payload)

    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["surprise"] = True
    with pytest.raises(ValueError, match="unknown keys"):
        parse_ladder_config(payload)
