#!/usr/bin/env python3
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_simulation_data(time_series_path="./papers/fast-drn/drn-xs/EP/time_series.pkl"):
    """Plot weight distribution and saturation data from simulation time series"""
    
    print(f"Loading time series data from: {time_series_path}")
    
    # Load time series data
    with open(time_series_path, 'rb') as f:
        time_series_data = pickle.load(f)
    
    print(f"Available time series: {list(time_series_data.keys())}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Simulation Data: Weight Distribution and Saturation', fontsize=16)
    
    # Plot 1: Weight Row Sums over time
    if 'WeightRowSum_DenseWeight_0' in time_series_data:
        epochs = time_series_data['WeightRowSum_DenseWeight_0'].epochs
        values = time_series_data['WeightRowSum_DenseWeight_0'].values
        axes[0,0].plot(epochs, values, 'b-', linewidth=2, label='Weight 0')
        axes[0,0].set_title('Weight Row Sums - Layer 0')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Row Sum')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
    
    if 'WeightRowSum_DenseWeight_1' in time_series_data:
        epochs = time_series_data['WeightRowSum_DenseWeight_1'].epochs
        values = time_series_data['WeightRowSum_DenseWeight_1'].values
        axes[0,0].plot(epochs, values, 'r-', linewidth=2, label='Weight 1')
        axes[0,0].legend()
    
    # Plot 2: Weight Column Sums over time
    if 'WeightColumnSum_DenseWeight_0' in time_series_data:
        epochs = time_series_data['WeightColumnSum_DenseWeight_0'].epochs
        values = time_series_data['WeightColumnSum_DenseWeight_0'].values
        axes[0,1].plot(epochs, values, 'b-', linewidth=2, label='Weight 0')
        axes[0,1].set_title('Weight Column Sums - Layer 0')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Column Sum')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].legend()
    
    if 'WeightColumnSum_DenseWeight_1' in time_series_data:
        epochs = time_series_data['WeightColumnSum_DenseWeight_1'].epochs
        values = time_series_data['WeightColumnSum_DenseWeight_1'].values
        axes[0,1].plot(epochs, values, 'r-', linewidth=2, label='Weight 1')
        axes[0,1].legend()
    
    # Plot 3: Weight Distribution Statistics over time
    weight_stats = ['WeightDist_mean_DenseWeight_0', 'WeightDist_std_DenseWeight_0', 
                   'WeightDist_min_DenseWeight_0', 'WeightDist_max_DenseWeight_0']
    
    colors = ['blue', 'green', 'red', 'orange']
    for i, stat in enumerate(weight_stats):
        if stat in time_series_data:
            epochs = time_series_data[stat].epochs
            values = time_series_data[stat].values
            axes[0,2].plot(epochs, values, color=colors[i], linewidth=2, 
                          label=stat.split('_')[-2] + '_' + stat.split('_')[-1])
    
    axes[0,2].set_title('Weight Distribution Stats - Layer 0')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('Value')
    axes[0,2].grid(True, alpha=0.3)
    axes[0,2].legend()
    axes[0,2].set_yscale('log')
    
    # Plot 4: Saturation over time - Layer 0
    if 'Saturation/train_Layer_0' in time_series_data:
        epochs = time_series_data['Saturation/train_Layer_0'].epochs
        values = time_series_data['Saturation/train_Layer_0'].values
        axes[1,0].plot(epochs, values, 'b-', linewidth=2, label='Train')
        axes[1,0].set_title('Saturation - Layer 0')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Saturation %')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
    
    if 'Saturation/test_Layer_0' in time_series_data:
        epochs = time_series_data['Saturation/test_Layer_0'].epochs
        values = time_series_data['Saturation/test_Layer_0'].values
        axes[1,0].plot(epochs, values, 'r-', linewidth=2, label='Test')
        axes[1,0].legend()
    
    # Plot 5: Saturation over time - Layer 1
    if 'Saturation/train_Layer_1' in time_series_data:
        epochs = time_series_data['Saturation/train_Layer_1'].epochs
        values = time_series_data['Saturation/train_Layer_1'].values
        axes[1,1].plot(epochs, values, 'b-', linewidth=2, label='Train')
        axes[1,1].set_title('Saturation - Layer 1')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Saturation %')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].legend()
    
    if 'Saturation/test_Layer_1' in time_series_data:
        epochs = time_series_data['Saturation/test_Layer_1'].epochs
        values = time_series_data['Saturation/test_Layer_1'].values
        axes[1,1].plot(epochs, values, 'r-', linewidth=2, label='Test')
        axes[1,1].legend()
    
    # Plot 6: Error over time
    if 'Error/train' in time_series_data:
        epochs = time_series_data['Error/train'].epochs
        values = time_series_data['Error/train'].values
        axes[1,2].plot(epochs, values, 'b-', linewidth=2, label='Train Error')
        axes[1,2].set_title('Error Over Time')
        axes[1,2].set_xlabel('Epoch')
        axes[1,2].set_ylabel('Error %')
        axes[1,2].grid(True, alpha=0.3)
        axes[1,2].legend()
    
    if 'Error/test' in time_series_data:
        epochs = time_series_data['Error/test'].epochs
        values = time_series_data['Error/test'].values
        axes[1,2].plot(epochs, values, 'r-', linewidth=2, label='Test Error')
        axes[1,2].legend()
    
    plt.tight_layout()
    plt.savefig('simulation_data_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved simulation data analysis: simulation_data_analysis.png")
    
    # Print summary statistics
    print("\n=== Simulation Summary ===")
    for key, series in time_series_data.items():
        if hasattr(series, 'values') and len(series.values) > 0:
            print(f"{key}: {len(series.values)} data points, "
                  f"range: [{series.values[0]:.6f}, {series.values[-1]:.6f}]")

if __name__ == "__main__":
    plot_simulation_data()
