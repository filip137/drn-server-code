#!/usr/bin/env python3

import os
import sys
import pickle
import matplotlib.pyplot as plt
import shutil
import torch
from datetime import datetime

def create_diode_model_folder(base_path, model, non_linearity, diode_conductance=None, v_min=None, v_max=None):
    """Create a folder with diode type and model name"""
    
    # Create folder name based on non_linearity and model
    if non_linearity == 'perfect_diode':
        folder_name = f"{model}_perfect_diode"
    elif non_linearity == 'lpw_diode':
        folder_name = f"{model}_lpw_diode_cond{diode_conductance}"
    elif non_linearity == 'hard_sigmoid':
        folder_name = f"{model}_hard_sigmoid_cond{diode_conductance}_vmin{v_min}_vmax{v_max}"
    else:
        folder_name = f"{model}_{non_linearity}"
    
    # Add timestamp to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{folder_name}_{timestamp}"
    
    # Create the full path
    plots_dir = os.path.join(base_path, 'plots', folder_name)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Created diode-specific plots directory: {plots_dir}")
    return plots_dir

def plot_all_simulation_data(time_series_path, plots_dir):
    """Plot all simulation data using the existing plot_all_organized.py logic"""
    
    print(f"Loading time series data from: {time_series_path}")
    
    # Load the time series data
    try:
        with open(time_series_path, 'rb') as f:
            series = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Time series file not found at {time_series_path}")
        return False
    except Exception as e:
        print(f"Error loading time series data: {e}")
        return False
    
    # Get epoch count
    some_key = next(iter(series))
    n = len(series[some_key])
    epochs = list(range(1, n + 1))
    
    print(f"Found {n} epochs of data")
    print(f"Available data series: {list(series.keys())}")
    
    # Plot accuracy (1 - error)
    for split in ['train', 'test']:
        err_key = f'Error/{split}'
        if err_key in series:
            error = series[err_key]
            # Convert to accuracy
            error_frac = [e/100.0 if e > 1 else e for e in error]
            accuracy = [1.0 - e for e in error_frac]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, accuracy, 'o-', linewidth=2, markersize=4)
            plt.title(f'{split.capitalize()} Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{plots_dir}/accuracy_{split}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {plots_dir}/accuracy_{split}.png")

    # Plot error
    for split in ['train', 'test']:
        err_key = f'Error/{split}'
        if err_key in series:
            error = series[err_key]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, error, 'o-', linewidth=2, markersize=4)
            plt.title(f'{split.capitalize()} Error')
            plt.xlabel('Epoch')
            plt.ylabel('Error (%)')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{plots_dir}/error_{split}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {plots_dir}/error_{split}.png")

    # Plot cost
    for split in ['train', 'test']:
        cost_key = f'Cost/{split}'
        if cost_key in series:
            cost = series[cost_key]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, cost, 'o-', linewidth=2, markersize=4)
            plt.title(f'{split.capitalize()} Cost')
            plt.xlabel('Epoch')
            plt.ylabel('Cost')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{plots_dir}/cost_{split}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {plots_dir}/cost_{split}.png")

    # Plot energy
    for split in ['train', 'test']:
        energy_key = f'Energy/{split}'
        if energy_key in series:
            energy = series[energy_key]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, energy, 'o-', linewidth=2, markersize=4)
            plt.title(f'{split.capitalize()} Energy')
            plt.xlabel('Epoch')
            plt.ylabel('Energy')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{plots_dir}/energy_{split}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {plots_dir}/energy_{split}.png")

    # Plot top-5 error
    for split in ['train', 'test']:
        top5_key = f'Top5Error/{split}'
        if top5_key in series:
            top5_error = series[top5_key]
            
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, top5_error, 'o-', linewidth=2, markersize=4)
            plt.title(f'{split.capitalize()} Top-5 Error')
            plt.xlabel('Epoch')
            plt.ylabel('Top-5 Error (%)')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'{plots_dir}/top5_error_{split}.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved {plots_dir}/top5_error_{split}.png")

    # Plot saturation for each layer
    for split in ['train', 'test']:
        plt.figure(figsize=(12, 8))
        
        for layer_num in range(3):  # Layer_0, Layer_1, Layer_2
            sat_key = f'Saturation/{split}_Layer_{layer_num}'
            if sat_key in series:
                saturation = series[sat_key]
                plt.plot(epochs, saturation, 'o-', linewidth=2, markersize=4, 
                        label=f'Layer {layer_num}')
        
        plt.title(f'{split.capitalize()} Saturation by Layer')
        plt.xlabel('Epoch')
        plt.ylabel('Saturation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{plots_dir}/saturation_{split}_all_layers.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {plots_dir}/saturation_{split}_all_layers.png")

    # Plot individual layer saturation
    for split in ['train', 'test']:
        for layer_num in range(3):
            sat_key = f'Saturation/{split}_Layer_{layer_num}'
            if sat_key in series:
                saturation = series[sat_key]
                
                plt.figure(figsize=(10, 6))
                plt.plot(epochs, saturation, 'o-', linewidth=2, markersize=4)
                plt.title(f'{split.capitalize()} Saturation - Layer {layer_num}')
                plt.xlabel('Epoch')
                plt.ylabel('Saturation')
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{plots_dir}/saturation_{split}_layer_{layer_num}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved {plots_dir}/saturation_{split}_layer_{layer_num}.png")

    # Plot layer norms
    for split in ['train', 'test']:
        plt.figure(figsize=(12, 8))
        
        for layer_num in range(3):  # Layer_0, Layer_1, Layer_2
            norm_key = f'Norm/{split}_Layer_{layer_num}'
            if norm_key in series:
                norm = series[norm_key]
                plt.plot(epochs, norm, 'o-', linewidth=2, markersize=4, 
                        label=f'Layer {layer_num}')
        
        plt.title(f'{split.capitalize()} Layer Norms')
        plt.xlabel('Epoch')
        plt.ylabel('L2 Norm')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{plots_dir}/norms_{split}_all_layers.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {plots_dir}/norms_{split}_all_layers.png")

    # Plot gradients
    plt.figure(figsize=(12, 8))
    grad_keys = ['Gradient/train_Bias_1', 'Gradient/train_Bias_2', 
                 'Gradient/train_DenseWeight_0', 'Gradient/train_DenseWeight_1']

    for grad_key in grad_keys:
        if grad_key in series:
            gradient = series[grad_key]
            plt.plot(epochs, gradient, 'o-', linewidth=2, markersize=4, 
                    label=grad_key.replace('Gradient/train_', ''))

    plt.title('Training Gradients')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{plots_dir}/gradients_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {plots_dir}/gradients_all.png")

    # Create a summary file with configuration info
    summary_file = os.path.join(plots_dir, 'simulation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Simulation Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Data series available: {len(series)}\n")
        f.write(f"Number of epochs: {n}\n")
        f.write(f"Time series file: {time_series_path}\n")
        f.write(f"Plots directory: {plots_dir}\n\n")
        f.write("Available data series:\n")
        for key in sorted(series.keys()):
            f.write(f"  - {key}\n")
    
    print(f"Saved simulation summary: {summary_file}")
    print(f"\nAll plots saved in '{plots_dir}/' directory!")
    print(f"Total files created: {len(os.listdir(plots_dir))}")
    

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) != 3:
        print("Usage: python plot_simulation_results.py <time_series_path> <plots_dir>")
        sys.exit(1)
    
    time_series_path = sys.argv[1]
    plots_dir = sys.argv[2]
    
    success = plot_all_simulation_data(time_series_path, plots_dir)
    if success:
        print("Plotting completed successfully!")
    else:
        print("Plotting failed!")
        sys.exit(1)

    # Plot weight row sums (normalized by batch size)
    plt.figure(figsize=(12, 8))
    weight_row_sum_keys = ['WeightRowSum/train_DenseWeight_0', 'WeightRowSum/train_DenseWeight_1']

    for weight_key in weight_row_sum_keys:
        if weight_key in series:
            row_sums = series[weight_key]
            plt.plot(epochs, row_sums, 'o-', linewidth=2, markersize=4, 
                    label=weight_key.replace('WeightRowSum/train_', ''))

    plt.title('Weight Row Sums (Normalized by Batch Size)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Row Sum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{plots_dir}/weight_row_sums.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {plots_dir}/weight_row_sums.png")

    # Plot weight column sums (normalized by batch size)
    plt.figure(figsize=(12, 8))
    weight_col_sum_keys = ['WeightColumnSum/train_DenseWeight_0', 'WeightColumnSum/train_DenseWeight_1']

    for weight_key in weight_col_sum_keys:
        if weight_key in series:
            col_sums = series[weight_key]
            plt.plot(epochs, col_sums, 'o-', linewidth=2, markersize=4, 
                    label=weight_key.replace('WeightColumnSum/train_', ''))

    plt.title('Weight Column Sums (Normalized by Batch Size)')
    plt.xlabel('Epoch')
    plt.ylabel('Average Column Sum')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{plots_dir}/weight_column_sums.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {plots_dir}/weight_column_sums.png")


    # Plot weight distributions at the end of training
    plt.figure(figsize=(15, 10))
    
    # Get the final weight values from the saved model
    model_path = os.path.join(os.path.dirname(os.path.dirname(time_series_path)), 'model.pt')
    if os.path.exists(model_path):
        try:
            # Load the model weights
            model_data = torch.load(model_path, map_location='cpu')
            if isinstance(model_data, list) and len(model_data) >= 4:
                # model_data contains [weight_matrices, bias_vectors]
                weight_matrices = model_data[0]
                
                # Plot distribution for each weight matrix
                for i, weight_matrix in enumerate(weight_matrices):
                    plt.subplot(2, 2, i+1)
                    
                    # Flatten the weight matrix for histogram
                    weights_flat = weight_matrix.flatten()
                    
                    # Create histogram
                    plt.hist(weights_flat.numpy(), bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
                    plt.yscale('log')
                    plt.title(f'Weight Distribution - DenseWeight_{i}')
                    plt.xlabel('Weight Value')
                    plt.ylabel('Frequency (log scale)')
                    plt.grid(True, alpha=0.3)
                    
                    # Add statistics text
                    mean_val = weights_flat.mean().item()
                    std_val = weights_flat.std().item()
                    min_val = weights_flat.min().item()
                    max_val = weights_flat.max().item()
                    zero_count = (weights_flat == 0).sum().item()
                    total_count = weights_flat.numel()
                    
                    stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}\nZeros: {zero_count}/{total_count} ({100*zero_count/total_count:.1f}%)'
                    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                # Add overall statistics
                plt.subplot(2, 2, 3)
                all_weights = torch.cat([w.flatten() for w in weight_matrices])
                plt.hist(all_weights.numpy(), bins=100, alpha=0.7, edgecolor='black', linewidth=0.5)
                plt.yscale('log')
                plt.title('Combined Weight Distribution')
                plt.xlabel('Weight Value')
                plt.ylabel('Frequency (log scale)')
                plt.grid(True, alpha=0.3)
                
                # Log scale for better visualization
                plt.subplot(2, 2, 4)
                non_zero_weights = all_weights[all_weights > 0]
                if len(non_zero_weights) > 0:
                    plt.hist(non_zero_weights.numpy(), bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
                    plt.title('Non-Zero Weight Distribution')
                    plt.xlabel('Weight Value')
                    plt.ylabel('Frequency (log scale)')
                    plt.yscale('log')
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, 'No non-zero weights found', ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Non-Zero Weight Distribution')
                
                plt.tight_layout()
                plt.savefig(f'{plots_dir}/weight_distributions.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved {plots_dir}/weight_distributions.png")
                
        except Exception as e:
            print(f"Warning: Could not load model weights for distribution plot: {e}")
            plt.close()
    else:
        print(f"Warning: Model file not found at {model_path}")

    print(f"\nAll plots saved in '{plots_dir}/' directory!")
