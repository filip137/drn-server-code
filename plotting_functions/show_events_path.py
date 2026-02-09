#!/usr/bin/env python3
import json
import os

def show_events_path_construction():
    print("=== Events File Path Construction ===")
    
    # Load config
    with open("config.json", "r") as f:
        config = json.load(f)
    
    # Get base path from config
    base_path = config["paths"]["base_path"]
    print(f"1. Base path from config: '{base_path}'")
    
    # Example model and algorithm
    model = "drn-xs"
    algorithm = "EP"
    print(f"2. Model: '{model}'")
    print(f"3. Algorithm: '{algorithm}'")
    
    # Construct the path
    path = "/".join([base_path, model, algorithm])
    print(f"4. Constructed path: '{path}'")
    
    # Show full absolute path
    full_path = os.path.abspath(path)
    print(f"5. Full absolute path: '{full_path}'")
    
    # Show where events files are created
    events_dir = full_path
    print(f"6. Events files created in: '{events_dir}'")
    
    # Show example events file name
    example_filename = "events.out.tfevents.1756719739.nom-cool-2.1492202.0"
    full_events_path = os.path.join(events_dir, example_filename)
    print(f"7. Example events file: '{full_events_path}'")
    
    print("\n=== Summary ===")
    print(f"Events files are created in: {full_path}")
    print("The SummaryWriter automatically creates files with names like:")
    print("events.out.tfevents.{timestamp}.{hostname}.{pid}")

if __name__ == "__main__":
    show_events_path_construction()
