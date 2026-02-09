import matplotlib.pyplot as plt
import numpy as np
def plot_gradient_history(history, title, base_dir, tag):
    """Plot per-layer gradient norms on a semilog scale."""
    if not history:
        return

    base_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    offset = 0
    for layer_name, values in history.items():
        if not values:
            continue
        xs = np.arange(offset, offset + len(values))
        plt.semilogy(xs, values, label=layer_name)
        offset += len(values)
        plt.axvline(offset, color="k", alpha=0.2, ls="--")

    plt.xlabel("Update index")
    plt.ylabel("‖∂E/∂z‖∞")
    plt.title(title)
    plt.grid(True, which="both", axis="y")
    plt.legend(fontsize=7)
    plt.tight_layout()

    path = base_dir / f"{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved gradient plot: {path}")


def plot_layer_state(layer, title, base_dir, tag):
    """Plot the flattened state of a layer for the current batch."""
    state = layer.state.detach().cpu()
    if state.numel() == 0:
        return

    base_dir.mkdir(parents=True, exist_ok=True)
    flat = state.reshape(state.shape[0], -1).numpy()

    plt.figure(figsize=(8, 3))
    for sample_idx, sample in enumerate(flat):
        plt.plot(sample, label=f"sample {sample_idx}")
    plt.xlabel("Unit index")
    plt.ylabel("State value")
    plt.title(title)
    if flat.shape[0] > 1:
        plt.legend(fontsize=7)
    plt.tight_layout()

    path = base_dir / f"{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved layer state plot: {path}")


def plot_average_gradient_history(mean_history, title, base_dir, tag):
    """Plot the averaged gradient norms over updates for each layer."""
    if not mean_history:
        return None

    base_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    for layer_name, values in mean_history.items():
        xs = np.arange(len(values))
        plt.semilogy(xs, values, label=layer_name)

    plt.xlabel("Update index")
    plt.ylabel("‖∂E/∂z‖∞ (mean across batches)")
    plt.title(title)
    plt.grid(True, which="both", axis="y")
    plt.legend(fontsize=7)
    plt.tight_layout()

    path = base_dir / f"{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved mean gradient plot: {path}")
    return path

def plot_layer_state_over_inputs(layer_name, state_history, input_history, base_dir, feature_idx=0):
    """Plot how a layer's units evolve as a function of a selected input feature."""
    if not state_history or not input_history:
        return None

    base_dir.mkdir(parents=True, exist_ok=True)

    inputs = np.asarray(input_history)
    if inputs.ndim == 1:
        primary_input = inputs
    else:
        primary_input = inputs[:, feature_idx]

    sort_idx = np.argsort(primary_input)
    sorted_inputs = primary_input[sort_idx]

    states = np.stack(state_history, axis=0)
    states = states[sort_idx]

    plt.figure(figsize=(8, 3))
    for unit_idx in range(states.shape[1]):
        plt.plot(sorted_inputs, states[:, unit_idx], label=f"unit {unit_idx}")
    plt.xlabel(f"Input[{feature_idx}] value")
    plt.ylabel("State value")
    plt.title(f"{layer_name} responses vs input[{feature_idx}]")
    if states.shape[1] <= 10:
        plt.legend(fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()

    path = base_dir / f"{layer_name}_responses.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved layer response plot: {path}")
    return path

def plot_input_output(inputs, scores, predictions, labels, base_dir, feature_idx=0):
    """Create a combined plot of network scores and discrete predictions vs an input feature."""
    if not len(inputs) or not len(scores):
        return None

    base_dir.mkdir(parents=True, exist_ok=True)

    inputs = np.asarray(inputs)
    if inputs.ndim == 1:
        primary_input = inputs
    else:
        primary_input = inputs[:, feature_idx]
    scores = np.asarray(scores)
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    sort_idx = np.argsort(primary_input)
    sorted_inputs = primary_input[sort_idx]
    sorted_scores = scores[sort_idx]
    sorted_predictions = predictions[sort_idx]
    sorted_labels = labels[sort_idx]

    fig, (ax_scores, ax_classes) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for dim in range(sorted_scores.shape[1]):
        ax_scores.plot(sorted_inputs, sorted_scores[:, dim], label=f"score[{dim}]")
    ax_scores.set_ylabel("Score")
    ax_scores.set_title(f"Network scores vs input[{feature_idx}]")
    ax_scores.grid(True, alpha=0.3)
    if sorted_scores.shape[1] <= 5:
        ax_scores.legend(fontsize=7, ncol=sorted_scores.shape[1])

    ax_classes.scatter(sorted_inputs, sorted_predictions, s=10, alpha=0.6, label="prediction")
    ax_classes.scatter(sorted_inputs, sorted_labels, s=10, alpha=0.6, label="label")
    ax_classes.set_xlabel(f"Input[{feature_idx}] value")
    ax_classes.set_ylabel("Class")
    ax_classes.set_title("Predicted vs true class")
    ax_classes.grid(True, alpha=0.3)
    ax_classes.legend(fontsize=7)

    plt.tight_layout()
    path = base_dir / "input_output_relationship.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved input/output plot: {path}")
    return path
