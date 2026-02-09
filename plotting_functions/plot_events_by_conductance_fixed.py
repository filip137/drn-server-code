#!/usr/bin/env python3
import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from collections import defaultdict
import re

def plot_events_by_conductance(base_dir, plots_dir):
    """
    Plot all available metrics from events files in separate conductance folders.
    """
    
    # Find all conductance folders
    conductance_folders = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('cond_'):
            conductance_folders.append(item_path)
    
    conductance_folders.sort()
    print(f"Found {len(conductance_folders)} conductance folders")
    
    # Create organized plots directory
    organized_dir = os.path.join(plots_dir, "organized_by_conductance")
    os.makedirs(organized_dir, exist_ok=True)
    
    # Extract conductance values from folder names and find events files
    conductance_values = []
    events_files = []
    
    for folder in conductance_folders:
        folder_name = os.path.basename(folder)
        print(f"Processing folder: {folder_name}")
        
        # Extract conductance value from folder name like "cond_1.000e-04" or "cond_0.000"
        if folder_name == "cond_0.000":
            conductance = 0.0
        else:
            match = re.search(r'cond_([\d.e-]+)', folder_name)
            if match:
                conductance = float(match.group(1))
            else:
                print(f"Warning: Could not parse conductance from {folder_name}")
                conductance = 0.0
        
        conductance_values.append(conductance)
        
        # Find events file in the folder structure: cond_*/drn-xs/EP/events.out.tfevents*
        events_pattern = os.path.join(folder, "drn-xs", "EP", "events.out.tfevents*")
        events_matches = glob.glob(events_pattern)
        
        if events_matches:
            # If multiple events files, take the latest one
            latest_file = max(events_matches, key=os.path.getctime)
            events_files.append(latest_file)
            print(f"  Found events file: {os.path.basename(latest_file)}")
        else:
            print(f"  Warning: No events file found in {folder}")
            events_files.append(None)
    
    # Filter out None events files
    valid_indices = [i for i, f in enumerate(events_files) if f is not None]
    events_files = [events_files[i] for i in valid_indices]
    conductance_values = [conductance_values[i] for i in valid_indices]
    
    print(f"\nFound {len(events_files)} valid events files")
    for i, (cond, file) in enumerate(zip(conductance_values, events_files)):
        print(f"  {i+1}. Conductance {cond:.2e}: {os.path.basename(file)}")
    
    # First pass: collect all available metrics
    all_metrics = set()
    for events_file in events_files:
        try:
            for event in tf.compat.v1.train.summary_iterator(events_file):
                for value in event.summary.value:
                    all_metrics.add(value.tag)
                if event.step > 5:  # Just check first few steps to get metric names
                    break
        except Exception as e:
            print(f"Error reading {events_file}: {e}")
    
    print(f"\nFound {len(all_metrics)} unique metrics:")
    for metric in sorted(all_metrics):
        print(f"  - {metric}")
    
    # Group metrics by category
    metric_categories = {
        'Performance': ['Error/train', 'Error/test', 'Top5Error/train', 'Top5Error/test'],
        'Energy': ['Energy/train', 'Energy/test'],
        'Cost': ['Cost/train', 'Cost/test'],
        'Layer_Norms': [m for m in all_metrics if 'Norm' in m],
        'Layer_Saturation': [m for m in all_metrics if 'Saturation' in m],
        'Weight_Stats': [m for m in all_metrics if 'Weight' in m],
        'Gradients': [m for m in all_metrics if 'Gradient' in m],
        'Other': []
    }
    
    # Put remaining metrics in 'Other'
    categorized = set()
    for category, metrics in metric_categories.items():
        categorized.update(metrics)
    metric_categories['Other'] = [m for m in all_metrics if m not in categorized]
    
    # Plot each category
    for category, metrics in metric_categories.items():
        if not metrics:
            continue
            
        print(f"\nPlotting {category} metrics...")
        plot_category(events_files, conductance_values, metrics, organized_dir, category)

def plot_category(events_files, conductance_values, metrics, organized_dir, category):
    """Plot a specific category of metrics."""
    
    # Create category directory
    category_dir = os.path.join(organized_dir, category.lower())
    os.makedirs(category_dir, exist_ok=True)
    
    # Extract data for all metrics in this category
    metric_data = defaultdict(lambda: defaultdict(list))
    
    for i, events_file in enumerate(events_files):
        conductance = conductance_values[i]
        
        try:
            for event in tf.compat.v1.train.summary_iterator(events_file):
                epoch = event.step
                
                for value in event.summary.value:
                    if value.tag in metrics:
                        metric_data[value.tag]['epochs'].append(epoch)
                        metric_data[value.tag]['values'].append(value.simple_value)
                        metric_data[value.tag]['conductance'].append(conductance)
                        
        except Exception as e:
            print(f"Error reading {events_file}: {e}")
    
    # Create plots for each metric
    for metric_name, data in metric_data.items():
        if not data['epochs']:
            continue
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{metric_name} vs Conductance', fontsize=16)
        
        # Convert to numpy arrays
        epochs = np.array(data['epochs'])
        values = np.array(data['values'])
        conductances = np.array(data['conductance'])
        
        # Plot 1: All runs overlaid
        ax1 = axes[0, 0]
        unique_conductances = np.unique(conductances)
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_conductances)))
        
        for i, cond in enumerate(unique_conductances):
            mask = conductances == cond
            if np.any(mask):
                ax1.plot(epochs[mask], values[mask], color=colors[i], 
                        label=f'{cond:.2e}', alpha=0.7, linewidth=1)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel(metric_name)
        ax1.set_title(f'{metric_name} - All Runs')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Final values vs conductance
        ax2 = axes[0, 1]
        final_values = []
        final_conductances = []
        
        for cond in unique_conductances:
            mask = conductances == cond
            if np.any(mask):
                # Get the last value for this conductance
                cond_epochs = epochs[mask]
                cond_values = values[mask]
                max_epoch_idx = np.argmax(cond_epochs)
                final_values.append(cond_values[max_epoch_idx])
                final_conductances.append(cond)
        
        if len(final_conductances) > 1:
            ax2.semilogx(final_conductances, final_values, 'bo-', linewidth=2, markersize=6)
        else:
            ax2.plot(final_conductances, final_values, 'bo-', linewidth=2, markersize=6)
        ax2.set_xlabel('Diode Conductance')
        ax2.set_ylabel(f'Final {metric_name}')
        ax2.set_title(f'Final {metric_name} vs Conductance')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Max values vs conductance
        ax3 = axes[1, 0]
        max_values = []
        max_conductances = []
        
        for cond in unique_conductances:
            mask = conductances == cond
            if np.any(mask):
                max_values.append(np.max(values[mask]))
                max_conductances.append(cond)
        
        if len(max_conductances) > 1:
            ax3.semilogx(max_conductances, max_values, 'ro-', linewidth=2, markersize=6)
        else:
            ax3.plot(max_conductances, max_values, 'ro-', linewidth=2, markersize=6)
        ax3.set_xlabel('Diode Conductance')
        ax3.set_ylabel(f'Max {metric_name}')
        ax3.set_title(f'Max {metric_name} vs Conductance')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Min values vs conductance
        ax4 = axes[1, 1]
        min_values = []
        min_conductances = []
        
        for cond in unique_conductances:
            mask = conductances == cond
            if np.any(mask):
                min_values.append(np.min(values[mask]))
                min_conductances.append(cond)
        
        if len(min_conductances) > 1:
            ax4.semilogx(min_conductances, min_values, 'go-', linewidth=2, markersize=6)
        else:
            ax4.plot(min_conductances, min_values, 'go-', linewidth=2, markersize=6)
        ax4.set_xlabel('Diode Conductance')
        ax4.set_ylabel(f'Min {metric_name}')
        ax4.set_title(f'Min {metric_name} vs Conductance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        safe_metric_name = metric_name.replace('/', '_').replace(' ', '_')
        plot_path = os.path.join(category_dir, f'{safe_metric_name}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {plot_path}")

if __name__ == "__main__":
    base_dir = "/home/filip/cursor_python_projects/energy-based-learning/papers/fast-drn/runs/single_runs"
    plots_dir = "/home/filip/cursor_python_projects/energy-based-learning/papers/fast-drn/runs/single_runs"
    
    plot_events_by_conductance(base_dir, plots_dir)
