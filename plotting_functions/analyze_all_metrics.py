#!/usr/bin/env python3
import tensorflow as tf
import os
import glob
from collections import defaultdict

def analyze_all_metrics(events_dir, num_files=5):
    """
    Analyze all available metrics in the events files.
    """
    
    # Get the first few events files
    events_files = sorted(glob.glob(os.path.join(events_dir, "events.out.tfevents*")))[:num_files]
    
    print(f"Analyzing {len(events_files)} events files...")
    
    all_metrics = defaultdict(list)
    metric_samples = defaultdict(list)
    
    for i, events_file in enumerate(events_files):
        print(f"\nFile {i+1}: {os.path.basename(events_file)}")
        
        try:
            events = tf.compat.v1.train.summary_iterator(events_file)
            file_metrics = set()
            
            for event in events:
                epoch = event.step
                
                for value in event.summary.value:
                    tag = value.tag
                    file_metrics.add(tag)
                    
                    # Store sample values for each metric
                    if len(metric_samples[tag]) < 3:  # Keep first 3 samples
                        metric_samples[tag].append({
                            'epoch': epoch,
                            'value': value.simple_value,
                            'file': os.path.basename(events_file)
                        })
            
            print(f"  Metrics found: {sorted(file_metrics)}")
            
            # Track which files have which metrics
            for metric in file_metrics:
                all_metrics[metric].append(os.path.basename(events_file))
                
        except Exception as e:
            print(f"  Error reading file: {e}")
    
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL METRICS:")
    print(f"{'='*60}")
    
    for metric, files in sorted(all_metrics.items()):
        print(f"\n{metric}:")
        print(f"  Found in {len(files)} files")
        print(f"  Sample values:")
        for sample in metric_samples[metric]:
            print(f"    Epoch {sample['epoch']}: {sample['value']:.6f} (from {sample['file']})")
    
    return all_metrics, metric_samples

if __name__ == "__main__":
    events_dir = "/home/filip/cursor_python_projects/energy-based-learning/papers/fast-drn/papers/fast-drn/drn-xs/EP"
    analyze_all_metrics(events_dir, num_files=3)
