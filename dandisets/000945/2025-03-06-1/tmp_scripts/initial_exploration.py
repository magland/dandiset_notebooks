import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print("\nBasic Information:")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Institution: {nwb.institution}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject Age: {nwb.subject.age}")
print(f"Subject Species: {nwb.subject.species}")

# Get trials info
trials = nwb.intervals["trials"]
trial_starts = trials["start_time"].data[:]
trial_stops = trials["stop_time"].data[:]
trial_durations = trial_stops - trial_starts

print("\nTrial Information:")
print(f"Number of trials: {len(trial_starts)}")
print(f"Mean trial duration: {np.mean(trial_durations):.3f} seconds")
print(f"Mean interval between trials: {np.mean(np.diff(trial_starts)):.3f} seconds")

# Get units info
units = nwb.units
unit_ids = units["id"].data[:]
cell_types = units["celltype_label"].data[:]

print("\nUnit Information:")
print(f"Total number of units: {len(unit_ids)}")
print(f"Number of RSUs (type 1): {np.sum(cell_types == 1)}")
print(f"Number of FSUs (type 2): {np.sum(cell_types == 2)}")

# Create a figure with multiple subplots
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig)

# Plot 1: Trial duration histogram
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(trial_durations, bins=30, range=(0, 5))  # Limit range to 0-5 seconds
ax1.set_title('Trial Duration Distribution')
ax1.set_xlabel('Duration (s)')
ax1.set_ylabel('Count')

# Plot 2: Inter-trial interval histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(np.diff(trial_starts), bins=30)
ax2.set_title('Inter-trial Interval Distribution')
ax2.set_xlabel('Interval (s)')
ax2.set_ylabel('Count')

# Plot 3: Raster plot for first 5 units around first 10 trials
ax3 = fig.add_subplot(gs[1, :])

# Plot horizontal lines to separate units
for i in range(6):  # 6 lines for 5 spaces
    ax3.axhline(y=i, color='gray', linestyle='-', alpha=0.2)

# Plot spikes
colors = ['b', 'r', 'g', 'm', 'c']  # Different color for each unit
for i in range(min(5, len(unit_ids))):
    spike_times = units["spike_times"][i][:]
    cell_type = "RSU" if cell_types[i] == 1 else "FSU"
    unit_id = unit_ids[i]
    
    # For each trial, plot spikes within ±0.5s window
    for trial_idx in range(min(10, len(trial_starts))):
        trial_start = trial_starts[trial_idx]
        mask = (spike_times >= trial_start - 0.5) & (spike_times <= trial_start + 0.5)
        trial_spikes = spike_times[mask] - trial_start
        y_pos = i + trial_idx/20  # Reduced spacing between trials
        ax3.plot(trial_spikes, [y_pos]*len(trial_spikes), '|', color=colors[i],
                label=f"Unit {unit_id} ({cell_type})" if trial_idx == 0 else "")

ax3.set_title('Spike Raster Plot (First 5 Units, First 10 Trials)')
ax3.set_xlabel('Time from trial start (s)')
ax3.set_ylabel('Unit')
ax3.set_ylim(-0.2, 4.8)
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('tmp_scripts/initial_exploration.png', bbox_inches='tight')
plt.close()
