from __future__ import annotations

import numpy as np
import torch


def snapshot_stats(stats_lists) -> dict:
    """Convert evaluator stats objects into a serializable name->value mapping."""
    snapshot = {}
    for stats in stats_lists:
        for stat in stats:
            name = getattr(stat, "display_name", None) or getattr(stat, "name", None) or stat.__class__.__name__
            try:
                value = stat.get()
            except Exception:
                continue
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
            elif isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.generic):
                value = value.item()
            snapshot[name] = value
    return snapshot

