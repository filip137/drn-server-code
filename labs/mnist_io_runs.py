import json
import shutil
from datetime import datetime
from pathlib import Path


def _unique_run_dir(base_dir: Path, stem: str) -> Path:
    candidate = base_dir / stem
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    for idx in range(1, 1000):
        candidate = base_dir / f"{stem}_{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"Could not create unique run dir under {base_dir}")


def _init_run_dir(args) -> Path:
    base_dir = Path(args.output_dir).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    sim_type = args.cmd or "run"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _unique_run_dir(base_dir, f"{timestamp}_{sim_type}")

    config_src = Path(args.config).expanduser().resolve()
    if not config_src.exists():
        raise FileNotFoundError(f"Config file not found: {config_src}")
    shutil.copy2(config_src, run_dir / config_src.name)

    weights_path = args.weights
    if weights_path:
        weights_path = str(Path(weights_path).expanduser().resolve())
    metadata = {
        "timestamp": timestamp,
        "simulation_type": sim_type,
        "config_path": str(config_src),
        "output_root": str(base_dir),
        "model_key": args.model_key,
        "weights_path": weights_path,
    }
    (run_dir / "run_info.json").write_text(json.dumps(metadata, indent=2))
    return run_dir


def _serialize_stats(stats_lists):
    serialized = []
    for list_idx, stats in enumerate(stats_lists):
        for stat in stats:
            name = getattr(stat, "display_name", None) or getattr(stat, "name", None) or stat.__class__.__name__
            try:
                value = stat.get()
            except Exception:
                value = str(stat)
            serialized.append({"list": list_idx, "name": name, "value": value})
    return serialized
