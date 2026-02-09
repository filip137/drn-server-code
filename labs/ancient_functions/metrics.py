import numpy as np

def aggregate_gradient_history(history_list):
    """Compute the mean gradient trajectory per layer across all recorded batches."""
    if not history_list:
        return {}

    aggregated = {}
    for history in history_list:
        for layer_name, values in history.items():
            if not values:
                continue
            aggregated.setdefault(layer_name, []).append(np.asarray(values, dtype=np.float64))

    mean_history = {}
    for layer_name, sequences in aggregated.items():
        if not sequences:
            continue
        max_len = max(seq.shape[0] for seq in sequences)
        padded = np.full((len(sequences), max_len), np.nan, dtype=np.float64)
        for idx, seq in enumerate(sequences):
            padded[idx, : seq.shape[0]] = seq
        mean_history[layer_name] = np.nanmean(padded, axis=0)

    return mean_history
