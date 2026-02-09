
import argparse
import glob
import json
import os
import pickle


def _has_hidden_prefix(path, base, hidden_prefix):
    """Return True if any path segment under base starts with hidden_prefix."""
    if not hidden_prefix:
        return False
    try:
        rel = os.path.relpath(path, base)
    except ValueError:
        rel = path
    parts = [p for p in rel.split(os.sep) if p and p != "."]
    return any(part.startswith(hidden_prefix) for part in parts)


def _error_to_percent(err_values):
    """Return the minimum error in percent, handling 0-1 or 0-100 scales."""
    if not err_values:
        return None
    min_err = min(float(v) for v in err_values)
    if min_err <= 1.0:
        min_err *= 100.0
    return min_err


def _select_model_name(config, run_dir, target_model=None):
    """Return the model name to use for a run, or None if not found."""
    if not config or "models" not in config or not isinstance(config["models"], dict):
        return None
    models = config["models"]
    run_dir_lower = run_dir.lower()

    model_name = None
    if target_model and target_model in models:
        model_name = target_model
    else:
        # Prefer the longest match between model names and the run directory path.
        for name in sorted(models.keys(), key=len, reverse=True):
            if name.lower() in run_dir_lower:
                model_name = name
                break
    if model_name is None and len(models) == 1:
        model_name = next(iter(models))
    return model_name


def _layer_shape_value(layer_shapes, index):
    """Return a comparable layer shape value for the given index."""
    if not isinstance(layer_shapes, list) or index >= len(layer_shapes):
        return None
    layer = layer_shapes[index]
    if isinstance(layer, (list, tuple)):
        if len(layer) == 1:
            return layer[0]
        return list(layer)
    return layer


def _parse_int_list(raw_values):
    """Return list of ints from comma/space-separated inputs."""
    values = []
    for raw in raw_values:
        for part in str(raw).replace(",", " ").split():
            if not part:
                continue
            values.append(int(part))
    return values


def find_highest_accuracy(base, hidden_prefix="hidden", target_model=None, layer_shape_1=None):
    """Find the run with the highest test accuracy under base, skipping hidden runs.

    Returns:
        dict | None: {"accuracy": float, "run_dir": str, "time_series": str} or None if not found.
    """
    best = None
    best_info = None

    pattern = os.path.join(base, "**", "time_series.pkl")
    for ts_path in sorted(glob.glob(pattern, recursive=True)):
        run_dir = os.path.dirname(ts_path)
        if _has_hidden_prefix(run_dir, base, hidden_prefix):
            continue

        try:
            with open(ts_path, "rb") as f:
                data = pickle.load(f)
        except (OSError, pickle.UnpicklingError):
            continue

        err = data.get("Error/test")
        if not err:
            continue

        config, _ = load_config(run_dir)
        model_name = _select_model_name(config, run_dir, target_model=target_model)
        if target_model and model_name != target_model:
            continue
        if layer_shape_1 is not None:
            model_cfg = config.get("models", {}).get(model_name, {}) if config and model_name else {}
            layer_shapes = model_cfg.get("layer_shapes")
            value = _layer_shape_value(layer_shapes, 1)
            if value != layer_shape_1:
                continue

        min_err_percent = _error_to_percent(err)
        if min_err_percent is None:
            continue
        acc = 100.0 - min_err_percent
        if best is None or acc > best:
            best = acc
            best_info = {
                "accuracy": acc,
                "run_dir": run_dir,
                "time_series": ts_path,
            }

    return best_info


def load_config(run_dir):
    """Return (config_dict, path) for the first JSON config found in run_dir."""
    candidates = [os.path.join(run_dir, name) for name in ("config.used.json", "config.override.json")]
    candidates.extend(sorted(glob.glob(os.path.join(run_dir, "*.json"))))

    seen = set()
    for path in candidates:
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        try:
            with open(path, "r") as f:
                return json.load(f), path
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


def extract_model_and_params(config, run_dir, target_model=None):
    """Return (non_linearity, model_name, pooling_mode) for a run.

    If target_model is provided, use it if present; otherwise guess by path match.
    """
    if not config or "models" not in config or not isinstance(config["models"], dict):
        return None, None, None

    model_name = _select_model_name(config, run_dir, target_model=target_model)
    non_linearity = None
    pooling_mode = None
    if model_name:
        model_cfg = config["models"].get(model_name, {})
        non_linearity = model_cfg.get("non_linearity")
        pooling_mode = model_cfg.get("pooling_mode")

    return non_linearity, model_name, pooling_mode


def main():
    parser = argparse.ArgumentParser(description="Find best accuracy across nested run directories.")
    parser.add_argument("base", help="Dir of dirs to analyze")
    parser.add_argument(
        "--highest-accuracy",
        action="store_true",
        help="Report highest test accuracy from time_series.pkl and exit.",
    )
    parser.add_argument(
        "--hidden-prefix",
        default="hidden",
        help="Skip runs if any path segment starts with this prefix.",
    )
    parser.add_argument(
        "--layer-shape-1",
        type=int,
        dest="layer_shape_1",
        help="Filter runs by layer_shapes[1] value in the chosen model.",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        dest="hidden_sizes",
        help="Comma/space-separated list of layer_shapes[1] values to scan.",
    )
    parser.add_argument(
        "--non-linearity",
        "--nl",
        dest="non_linearity",
        help="Filter runs by non_linearity (case-insensitive).",
    )
    parser.add_argument(
        "--model",
        dest="model",
        help="Filter runs by model name (matches config model keys).",
    )
    parser.add_argument(
        "--pooling-mode",
        dest="pooling_mode",
        help="Filter runs by pooling_mode (case-insensitive match against config).",
    )
    args = parser.parse_args()
    base = args.base
    if args.highest_accuracy:
        sizes = _parse_int_list(args.hidden_sizes) if args.hidden_sizes else None
        if sizes:
            found_any = False
            for size in sizes:
                info = find_highest_accuracy(
                    base,
                    hidden_prefix=args.hidden_prefix,
                    target_model=args.model,
                    layer_shape_1=size,
                )
                if not info:
                    print(f"layer_shape_1={size}: no runs with non-empty Error/test under {base}")
                    continue
                found_any = True
                print(
                    "layer_shape_1={}: highest_accuracy={:.2f}%, run_dir={}, time_series={}".format(
                        size,
                        info["accuracy"],
                        info["run_dir"],
                        info["time_series"],
                    )
                )
            if not found_any:
                return
        else:
            info = find_highest_accuracy(
                base,
                hidden_prefix=args.hidden_prefix,
                target_model=args.model,
                layer_shape_1=args.layer_shape_1,
            )
            if not info:
                print(f"No runs with non-empty Error/test found under {base}")
                return
            print(
                "highest_accuracy={:.2f}%, run_dir={}, time_series={}".format(
                    info["accuracy"],
                    info["run_dir"],
                    info["time_series"],
                )
            )
        return

    target_nl = args.non_linearity.lower() if args.non_linearity else None
    target_model = args.model
    target_pool = args.pooling_mode.lower() if args.pooling_mode else None
    target_layer_shape_1 = args.layer_shape_1

    results = []
    seen_dirs = set()

    for events_file in sorted(glob.glob(os.path.join(base, "**", "events.out.tfevents*"), recursive=True)):
        run_dir = os.path.dirname(events_file)
        if run_dir in seen_dirs:
            continue  # already processed this run folder
        seen_dirs.add(run_dir)

        ts = os.path.join(run_dir, "time_series.pkl")
        if not os.path.isfile(ts):
            continue

        with open(ts, "rb") as f:
            data = pickle.load(f)

        err = data.get("Error/test")
        if not err:
            continue

        best = 100 - min(err)
        final = 100 - err[-1]
        name = os.path.basename(run_dir)
        events = os.path.basename(events_file)
        run_dir_display = os.path.relpath(run_dir, base)
        config, _ = load_config(run_dir)
        non_linearity, model_name, pool_mode = extract_model_and_params(config, run_dir, target_model=target_model)
        if target_layer_shape_1 is not None:
            model_cfg = config.get("models", {}).get(model_name, {}) if config and model_name else {}
            layer_shapes = model_cfg.get("layer_shapes")
            value = _layer_shape_value(layer_shapes, 1)
            if value != target_layer_shape_1:
                continue

        if target_model and (model_name != target_model):
            continue
        if target_nl and (not non_linearity or non_linearity.lower() != target_nl):
            continue
        if target_pool and (not pool_mode or str(pool_mode).lower() != target_pool):
            continue

        results.append((best, final, name, events, non_linearity, model_name, pool_mode, run_dir_display))

    results.sort(key=lambda x: x[0], reverse=True)

    if not results:
        if target_nl:
            print(f"No runs with non-empty Error/test and non_linearity '{args.non_linearity}' found under {base}")
        else:
            print(f"No runs with non-empty Error/test found under {base}")
        return

    for best, final, name, events, non_linearity, model_name, pool_mode, run_dir_display in results:
        details = []
        if model_name:
            details.append(f"model={model_name}")
        if non_linearity:
            details.append(f"non_linearity={non_linearity}")
        if pool_mode:
            details.append(f"pooling_mode={pool_mode}")
        detail_str = f", {', '.join(details)}" if details else ""
        print(f"{name}: best={best:.2f}%, final={final:.2f}%, events={events}, dir={run_dir_display}{detail_str}")


if __name__ == "__main__":
    main()
