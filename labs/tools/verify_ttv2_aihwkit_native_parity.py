#!/usr/bin/env python3
"""Emit a deterministic AIHWKit 1.1.0 ChoppedTransfer parity receipt."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch


REQUIRED_AIHWKIT_VERSION = "1.1.0"


def _constant_step_device(dw_min: float):
    from aihwkit.simulator.configs.devices import ConstantStepDevice

    return ConstantStepDevice(
        dw_min=dw_min,
        w_min=-1.0,
        w_max=1.0,
        dw_min_dtod=0.0,
        w_min_dtod=0.0,
        w_max_dtod=0.0,
        up_down=0.0,
        up_down_dtod=0.0,
        dw_min_std=0.0,
        lifetime=0.0,
        diffusion=0.0,
    )


def _assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    label: str,
) -> None:
    if actual.shape != expected.shape or not torch.allclose(
        actual,
        expected,
        rtol=0.0,
        atol=2e-6,
    ):
        raise RuntimeError(
            f"Native AIHWKit parity failed for {label}: "
            f"actual={actual.tolist()!r}, expected={expected.tolist()!r}."
        )


def _snapshot(tile) -> dict[str, Any]:
    hidden = tile.get_hidden_parameters()
    extra = tile.dump_extra()
    return {
        "fast": hidden["hidden_weights_0"].tolist(),
        "slow": hidden["hidden_weights_1"].tolist(),
        "buffer": hidden["buffered_FP_weight_0"].tolist(),
        "cursor": [
            int(value)
            for value in extra["rpu.rpu_device.current_slice_indices"]
        ],
        "fast_count_lr": float(
            extra["rpu.rpu_device.tmp_count_lr"][0]
        ),
    }


def build_receipt() -> dict[str, Any]:
    """Run two native transfers and fail closed unless the oracle matches."""

    import aihwkit
    from aihwkit.simulator.configs import UnitCellRPUConfig
    from aihwkit.simulator.configs.compounds import ChoppedTransferCompound
    from aihwkit.simulator.parameters.io import IOParameters
    from aihwkit.simulator.parameters.training import UpdateParameters
    from aihwkit.simulator.tiles import AnalogTile

    version = str(getattr(aihwkit, "__version__", "unknown"))
    if version != REQUIRED_AIHWKIT_VERSION:
        raise RuntimeError(
            "Expected the native TTv2 parity oracle to use AIHWKit "
            f"{REQUIRED_AIHWKIT_VERSION}; installed={version!r}."
        )

    out_size = 3
    in_size = 4
    fast_dw = 0.2
    slow_dw = 0.1
    fast_lr = 0.5
    transfer_lr = 1.0
    transfer_every = 1
    buffer_granularity = 1.0
    auto_granularity = 10.0
    b0 = (
        buffer_granularity
        * fast_dw
        * auto_granularity
        / (in_size * transfer_every)
    )
    corrected_b = b0 * slow_dw / fast_dw
    lambda_h = transfer_lr / (fast_lr * corrected_b)

    compound = ChoppedTransferCompound(
        unit_cell_devices=[
            _constant_step_device(fast_dw),
            _constant_step_device(slow_dw),
        ],
        construction_seed=7,
        gamma=0.0,
        transfer_every=transfer_every,
        units_in_mbatch=True,
        n_reads_per_transfer=1,
        transfer_columns=True,
        with_reset_prob=0.0,
        random_selection=False,
        fast_lr=fast_lr,
        transfer_lr=transfer_lr,
        scale_transfer_lr=False,
        transfer_forward=IOParameters(is_perfect=True),
        transfer_update=UpdateParameters(
            desired_bl=1,
            update_bl_management=False,
            update_management=False,
        ),
        in_chop_prob=0.0,
        in_chop_random=False,
        out_chop_prob=0.0,
        buffer_granularity=buffer_granularity,
        auto_granularity=auto_granularity,
        step=1.0,
        momentum=0.0,
        forget_buffer=True,
        no_buffer=False,
        auto_scale=False,
        correct_gradient_magnitudes=True,
    )
    tile = AnalogTile(
        out_size,
        in_size,
        UnitCellRPUConfig(device=compound),
        bias=False,
    )
    tile.set_learning_rate(0.25)

    hidden = tile.get_hidden_parameters()
    for key in ("hidden_weights_0", "hidden_weights_1", "buffered_FP_weight_0"):
        hidden[key].zero_()
    hidden["hidden_weights_0"][:, 0] = torch.tensor([0.1, 0.05, -0.1])
    hidden["buffered_FP_weight_0"][:, 0] = torch.tensor([0.1, 0.1, -0.1])
    hidden["hidden_weights_0"][:, 1] = torch.tensor([0.2, -0.2, 0.15])
    hidden["buffered_FP_weight_0"][:, 1] = torch.tensor([0.1, -0.1, 0.0])
    tile.set_hidden_parameters(hidden)

    initial = _snapshot(tile)
    zero_x = torch.zeros(1, in_size)
    zero_d = torch.zeros(1, out_size)
    tile.update(zero_x, zero_d)
    after_first = _snapshot(tile)
    tile.update(zero_x, zero_d)
    after_second = _snapshot(tile)

    initial_fast = torch.tensor(initial["fast"])
    expected_first_buffer = torch.tensor(initial["buffer"])
    expected_first_buffer[:, 0] = torch.tensor([0.9, 0.5, -0.9])
    expected_second_buffer = expected_first_buffer.clone()
    expected_second_buffer[:, 1] = 0.0
    expected_second_slow = torch.zeros((out_size, in_size))
    expected_second_slow[:, 1] = torch.tensor([0.1, -0.1, 0.1])

    _assert_close(
        torch.tensor(after_first["fast"]),
        initial_fast,
        label="persistent fast A after first read",
    )
    _assert_close(
        torch.tensor(after_first["buffer"]),
        expected_first_buffer,
        label="sub-threshold first-column H",
    )
    _assert_close(
        torch.tensor(after_first["slow"]),
        torch.zeros((out_size, in_size)),
        label="no first-column slow pulses",
    )
    _assert_close(
        torch.tensor(after_second["fast"]),
        initial_fast,
        label="persistent fast A after second read",
    )
    _assert_close(
        torch.tensor(after_second["buffer"]),
        expected_second_buffer,
        label="forget-buffer second-column H",
    )
    _assert_close(
        torch.tensor(after_second["slow"]),
        expected_second_slow,
        label="signed second-column slow pulses",
    )
    if initial["cursor"] != [0, 0]:
        raise RuntimeError(f"Expected zero native cursor; got {initial['cursor']!r}.")
    if after_first["cursor"] != [1, 0] or after_second["cursor"] != [2, 0]:
        raise RuntimeError(
            "Expected native column cursor 0 -> 1 -> 2; "
            f"got {initial['cursor']!r} -> {after_first['cursor']!r} "
            f"-> {after_second['cursor']!r}."
        )
    if abs(after_first["fast_count_lr"] - fast_lr) > 2e-6:
        raise RuntimeError(
            "Expected native fixed fast count LR after update; "
            f"got {after_first['fast_count_lr']!r}."
        )

    return {
        "schema": "ebl.ttv2_aihwkit_native_parity",
        "schema_version": 1,
        "aihwkit_version": version,
        "native_device": "ChoppedTransferCompound",
        "shape": {"out": out_size, "in": in_size},
        "controls": {
            "fast_dw": fast_dw,
            "slow_dw": slow_dw,
            "fast_lr": fast_lr,
            "transfer_lr": transfer_lr,
            "scale_transfer_lr": False,
            "transfer_every": transfer_every,
            "units_in_mbatch": True,
            "transfer_columns": True,
            "buffer_granularity": buffer_granularity,
            "auto_granularity": auto_granularity,
            "correct_gradient_magnitudes": True,
            "desired_bl": 1,
            "momentum": 0.0,
            "forget_buffer": True,
            "in_chop_probability": 0.0,
            "out_chop_probability": 0.0,
        },
        "equation": {
            "buffer_scale_uncorrected": b0,
            "buffer_scale_corrected": corrected_b,
            "lambda_h": lambda_h,
        },
        "initial": initial,
        "after_first_transfer": after_first,
        "after_second_transfer": after_second,
        "checks": {
            "version_exact": True,
            "fast_state_persistent": True,
            "buffer_equation": True,
            "signed_slow_pulses": True,
            "scan_axis_columns": True,
            "cursor_zero_origin_sequential": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_receipt(),
            indent=2 if args.pretty else None,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
