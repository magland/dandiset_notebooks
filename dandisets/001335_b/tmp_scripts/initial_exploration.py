# %% [markdown]
# # Initial Exploration of DANDI:001335
# This script explores the Neuropixels recordings from hippocampus during odor presentation

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# %% Load the data
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# %% Print basic information
print("\nDataset Information:")
print(f"Session ID: {nwb.identifier}")
print(f"Description: {nwb.session_description}")
print(f"Experiment: {nwb.experiment_description}")
print(f"Subject: {nwb.subject.subject_id} ({nwb.subject.species})")
print(f"Age: {nwb.subject.age}")
print(f"Keywords: {', '.join(nwb.keywords)}")

# %% Analyze odor presentations
print("\nAnalyzing odor presentations...")
odors = ['A', 'B', 'C', 'D', 'E', 'F']
for odor in odors:
    odor_intervals = nwb.intervals[f"Odor {odor} ON"]
    start_times = odor_intervals["start_time"].data[:]
    stop_times = odor_intervals["stop_time"].data[:]
    durations = stop_times - start_times
    print(f"\nOdor {odor}:")
    print(f"Number of presentations: {len(start_times)}")
    print(f"Average duration: {np.mean(durations):.2f} seconds")
    print(f"First presentation at: {start_times[0]:.2f} seconds")
    print(f"Last presentation at: {start_times[-1]:.2f} seconds")

# %% Load a sample of LFP data (first 10 seconds, first 5 channels)
print("\nLoading sample LFP data...")
sample_duration = 10  # seconds
sample_channels = 5
lfp = nwb.processing["ecephys"]["LFP"]
n_samples = int(sample_duration * lfp.rate)
lfp_data = lfp.data[0:n_samples, 0:sample_channels]
time = np.arange(n_samples) / lfp.rate

# Create LFP plot
plt.figure(figsize=(15, 8))
for i in range(sample_channels):
    plt.plot(time, lfp_data[:, i] + i*200, label=f'Channel {i}')
plt.xlabel('Time (seconds)')
plt.ylabel('LFP Signal (offset for visibility)')
plt.title('Sample LFP Traces (First 10 seconds)')
plt.legend()
plt.savefig('lfp_example.png')
plt.close()

# %% Create spike raster plot for a subset of neurons
print("\nCreating spike raster plot...")
n_units = 20  # Number of units to plot
time_window = [0, 60]  # First 60 seconds

plt.figure(figsize=(15, 8))
for i in range(n_units):
    spike_times = nwb.units["spike_times"][i]
    mask = (spike_times >= time_window[0]) & (spike_times <= time_window[1])
    plt.plot(spike_times[mask], np.ones_like(spike_times[mask])*i, '|', markersize=4)

plt.xlabel('Time (seconds)')
plt.ylabel('Neuron #')
plt.title(f'Spike Raster Plot (First {n_units} neurons, first {time_window[1]} seconds)')
plt.savefig('spike_raster.png')
plt.close()

# %% Create odor presentation timeline
print("\nCreating odor presentation timeline...")
plt.figure(figsize=(15, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y']
for i, odor in enumerate(odors):
    odor_intervals = nwb.intervals[f"Odor {odor} ON"]
    starts = odor_intervals["start_time"].data[:]
    durations = odor_intervals["stop_time"].data[:] - starts
    
    # Plot each odor presentation as a colored line
    for start, duration in zip(starts, durations):
        plt.plot([start, start + duration], [i, i], 
                 color=colors[i], linewidth=2, label=f'Odor {odor}' if start == starts[0] else "")

plt.ylim(-0.5, len(odors)-0.5)
plt.yticks(range(len(odors)), [f'Odor {o}' for o in odors])
plt.xlabel('Time (seconds)')
plt.title('Odor Presentation Timeline')
plt.legend()
plt.savefig('odor_timeline.png')
plt.close()

print("\nFinished creating plots: lfp_example.png, spike_raster.png, and odor_timeline.png")
