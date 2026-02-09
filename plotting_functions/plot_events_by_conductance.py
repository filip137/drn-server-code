#!/usr/bin/env python3
import os
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_events_by_conductance(events_dir, plots_dir, num_files=15):
    """
    Plot the last num_files events files, each in a separate folder denoting diode conductance.
    """
    
    # Get the last num_files events files
    events_files = sorted(glob.glob(os.path.join(events_dir, "events.out.tfevents*")))[-num_files:]
    
    print(f"Found {len(events_files)} events files")
    
    # Create organized plots directory
    organized_dir = os.path.join(plots_dir, "organized_by_conductance")
    os.makedirs(organized_dir, exist_ok=True)
    
    # Generate diode conductance values (assuming they're logarithmically spaced)
    # Based on the multiple_runs.py script, they go from 1 to 100
    diode_conductances = np.logspace(np.log10(1), np.log10(100), num_files)
    
    for i, (events_file, conductance) in enumerate(zip(events_files, diode_conductances)):
        print(f"Processing file {i+1}/{num_files}: {os.path.basename(events_file)}")
        
        # Create folder for this conductance value
        folder_name = f"conductance_{conductance:.6g}"
        folder_path = os.path.join(organized_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        try:
            # Read events file
            events_data = []
            for event in tf.compat.v1.train.summary_iterator(events_file):
                for value in event.summary.value:
                    if value.tag == 'Error/test':
                        events_data.append({
                            'step': event.step,
                            'value': value.simple_value,
                            'wall_time': event.wall_time
                        })
            
            if not events_data:
                print(f"  No data found in {events_file}")
                continue
            
            # Convert to numpy arrays
            steps = np.array([d['step'] for d in events_data])
            values = np.array([d['value'] for d in events_data])
            
            # Calculate accuracy (assuming Error/test is error rate, so accuracy = 1 - error)
            accuracy = 1 - values
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(steps, accuracy, 'b-', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Test Accuracy')
            plt.title(f'Test Accuracy vs Epochs (Diode Conductance: {conductance:.6g})')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            # Save plot
            plot_path = os.path.join(folder_path, 'accuracy_vs_epochs.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved plot to {plot_path}")
            
        except Exception as e:
            print(f"  Error processing {events_file}: {e}")
            continue
    
    print(f"\nAll plots saved to: {organized_dir}")
    return organized_dir

if __name__ == "__main__":
    events_dir = "/home/filip/cursor_python_projects/energy-based-learning/papers/fast-drn/papers/fast-drn/drn-xs/EP"
    plots_dir = "/home/filip/cursor_python_projects/energy-based-learning/papers/fast-drn/papers/fast-drn/drn-xs/EP/plots"
    
    plot_events_by_conductance(events_dir, plots_dir, num_files=15)
