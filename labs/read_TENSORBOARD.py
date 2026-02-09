from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

events_path = Path("/home/filip/server_code/plots/small_network/drn-conv_20251127-143535/iters_12/events.out.tfevents.1764250535.nom-cool-2.3544896.0")

# Load the event file
ea = event_accumulator.EventAccumulator(str(events_path))
ea.Reload()

# List scalar tags
print("Scalar tags:", ea.Tags().get("scalars", []))

# Dump counts and last values
for tag in ea.Tags().get("scalars", []):
    events = ea.Scalars(tag)
    print(f"{tag}: {len(events)} points, last={events[-1].value} @ step {events[-1].step}")
