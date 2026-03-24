import json
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_json_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return json.loads(config_path.read_text())


def _require_config_dict(config: dict, name: str) -> dict:
    value = config.get(name)
    if value is None:
        raise SystemExit(f"Config must define '{name}'.")
    if not isinstance(value, dict):
        raise SystemExit(f"Config '{name}' must be an object.")
    return value


def _require_keys(params: dict, keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in params]
    if missing:
        raise SystemExit(f"Config '{label}' missing keys: {', '.join(missing)}")


_NON_LINEARITY_REQUIREMENTS = {
    "double_diode_quadratic": ("quadratic_diode_param", ["diode_conductance", "v_off"]),
    "double_diode_exponential": ("exponential_diode_param", ["I_s", "V_t", "V_off"]),
    "double_diode": ("hard_sigmoid_param", ["g_on", "g_off", "v_min", "v_max"]),
    "hard_sigmoid": ("hard_sigmoid_param", ["g_on", "g_off", "v_min", "v_max"]),
}


def _parse_non_linearity_params(config: dict, non_linearity: str) -> tuple[dict, dict, dict]:
    quadratic_params = config.get("quadratic_diode_param", {})
    exponential_params = config.get("exponential_diode_param", {})
    hard_sigmoid_params = config.get("hard_sigmoid_param", {})

    requirement = _NON_LINEARITY_REQUIREMENTS.get(non_linearity)
    if requirement:
        param_name, keys = requirement
        params = _require_config_dict(config, param_name)
        _require_keys(params, keys, param_name)
        if param_name == "quadratic_diode_param":
            quadratic_params = params
        elif param_name == "exponential_diode_param":
            exponential_params = params
        else:
            hard_sigmoid_params = params

    return quadratic_params, exponential_params, hard_sigmoid_params


def _resolve_config_value(name: str, cli_value, config: dict):
    if cli_value is not None:
        return cli_value
    if name in config:
        return config[name]
    cli_flag = f"--{name.replace('_', '-')}"
    raise SystemExit(f"Missing required value for '{name}'. Provide it in the config or via {cli_flag}.")


def _resolve_optional_config_value(name: str, cli_value, config: dict, default=None):
    if cli_value is not None:
        return cli_value
    if name in config:
        return config[name]
    return default


def _resolve_config_list(name: str, cli_value, config: dict) -> list:
    resolved = _resolve_config_value(name, cli_value, config)
    if resolved is None:
        return []
    if isinstance(resolved, (list, tuple)):
        return list(resolved)
    return [resolved]


def _parse_layer_shapes(layer_shapes) -> tuple[int, list[int], int, list[tuple[int, ...]]]:
    if not isinstance(layer_shapes, (list, tuple)) or len(layer_shapes) < 2:
        raise SystemExit("Config 'layer_shapes' must be a list with at least input and output shapes.")
    parsed = []
    for shape in layer_shapes:
        if isinstance(shape, (list, tuple)):
            if len(shape) != 1 or not isinstance(shape[0], int):
                raise SystemExit("Each entry in 'layer_shapes' must be a single integer list, e.g. [4].")
            parsed.append((shape[0],))
        elif isinstance(shape, int):
            parsed.append((shape,))
        else:
            raise SystemExit("Each entry in 'layer_shapes' must be an int or single-int list.")

    input_size = parsed[0][0]
    if input_size % 2 != 0:
        raise SystemExit("Input layer size must be divisible by 2 to infer input_dim.")
    input_dim = input_size // 2
    hidden_dims = [shape[0] for shape in parsed[1:-1]]
    output_dim = parsed[-1][0]
    return input_dim, hidden_dims, output_dim, parsed
