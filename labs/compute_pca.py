import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

LABS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LABS_DIR.parent
DATASETS_MODULE_PATH = PROJECT_ROOT / "datasets.py"
DEFAULT_PCA_CACHE_DIR = LABS_DIR / "datasets"

for path in (LABS_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from supporting.pca_utils import compute_pca_from_loader  # noqa: E402


def load_json_config(config_path):
    path = Path(config_path).expanduser()
    return json.loads(path.read_text())


def _load_project_datasets_module():
    module_name = "ebl_datasets"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not DATASETS_MODULE_PATH.exists():
        raise FileNotFoundError(f"Expected datasets.py at {DATASETS_MODULE_PATH}")
    spec = importlib.util.spec_from_file_location(module_name, DATASETS_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load datasets module from {DATASETS_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


load_dataloaders = _load_project_datasets_module().load_dataloaders


def compute_pca(config_path, model_key):
    config = load_json_config(config_path)
    training_cfg = config.get("training", {})

    dataset_name = training_cfg.get("dataset", "MNIST")
    batch_size = 1
    normalize = training_cfg.get("normalize")
    normalize_std = training_cfg.get("normalize_std")
    augment_32 = training_cfg.get("augment_32x32")

    train_loader, _ = load_dataloaders(
        dataset_name,
        batch_size,
        augment_32x32=augment_32,
        normalize=normalize,
        normalize_std=normalize_std,
    )
    return compute_pca_from_loader(train_loader)


def _resolve_pca_cache_path(pca_cache_path, model_key, dataset_name):
    if pca_cache_path:
        return Path(pca_cache_path).expanduser()
    safe_dataset = str(dataset_name).lower()
    return DEFAULT_PCA_CACHE_DIR / f"pca_{safe_dataset}_{model_key}.pt"


def _load_pca_cache(path: Path):
    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu")
        mu = data["mu"]
        eigvals = data["eigvals"]
        eigvecs = data["eigvecs"]
        n_total = int(data["n_total"])
        return mu, eigvals, eigvecs, n_total
    # Legacy .npz support
    data = np.load(path)
    mu = data["mu"]
    eigvals = data["eigvals"]
    eigvecs = data["eigvecs"]
    n_total = int(data["n_total"]) if "n_total" in data else int(eigvecs.shape[0])
    return torch.as_tensor(mu), torch.as_tensor(eigvals), torch.as_tensor(eigvecs), n_total


def _save_pca_cache(path: Path, mu, eigvals, eigvecs, n_total):
    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        return torch.as_tensor(x)

    payload = {
        "mu": _to_tensor(mu),
        "eigvals": _to_tensor(eigvals),
        "eigvecs": _to_tensor(eigvecs),
        "n_total": int(n_total),
    }
    torch.save(payload, path)
