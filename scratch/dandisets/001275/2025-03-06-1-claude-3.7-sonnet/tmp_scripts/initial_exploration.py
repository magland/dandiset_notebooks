#!/usr/bin/env python

"""
Initial exploration of Dandiset 001275: Mental navigation primate PPC
This script examines the basic structure and content of the dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi

# Load a behavior+ecephys file (smaller file with both behavioral and neural data)
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic session information
print("\n=== Session Information ===")
print(f"Subject: {nwb.subject.subject_id} ({nwb.subject.species})")
print(f"Sex: {nwb.subject.sex}, Age: {nwb.subject.age}")
print(f"Session date: {nwb.session_start_time}")
print(f"Lab: {nwb.lab}, Institution: {nwb.institution}")
print(f"Session description: {nwb.session_description}")

# Examine trials information
trials = nwb.intervals["trials"]
print("\n=== Trials Information ===")
print(f"Number of trials: {len(trials['start_time'].data)}")
print(f"Trial columns: {trials.colnames}")

# Get trial durations
trial_starts = trials["start_time"].data[:]
trial_stops = trials["stop_time"].data[:]
trial_durations = trial_stops - trial_starts

print(f"Average trial duration: {np.mean(trial_durations):.2f} seconds")
print(f"Min trial duration: {np.min(trial_durations):.2f} seconds")
print(f"Max trial duration: {np.max(trial_durations):.2f} seconds")

# Plot histogram of trial durations
plt.figure(figsize=(10, 6))
plt.hist(trial_durations, bins=30)
plt.title('Distribution of Trial Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.savefig('scratch/dandisets/001275/2025-03-06-1/tmp_scripts/trial_durations.png')
plt.close()

# Examine behavioral data
print("\n=== Behavioral Data ===")
behavior = nwb.processing["behavior"]
print(f"Behavior data types: {list(behavior.data_interfaces.keys())}")

# Eye position data
eye_position = behavior["eye_position"]
print(f"Eye position data shape: {eye_position.data.shape}")
print(f"Eye position sampling rate: {1.0 / np.mean(np.diff(eye_position.timestamps[:1000])):.2f} Hz")

# Hand position data (joystick)
hand_position = behavior["hand_position"]
print(f"Hand position data shape: {hand_position.data.shape}")
print(f"Hand position sampling rate: {1.0 / np.mean(np.diff(hand_position.timestamps[:1000])):.2f} Hz")

# Plot a sample of hand position data (joystick movement)
sample_start = 1000000  # Start at 1M sample to skip initial period
sample_length = 100000  # Look at 100K samples

hand_times = hand_position.timestamps[sample_start:sample_start+sample_length]
hand_data = hand_position.data[sample_start:sample_start+sample_length]

plt.figure(figsize=(12, 6))
plt.plot(hand_times - hand_times[0], hand_data)
plt.title('Sample Joystick Movement')
plt.xlabel('Time (seconds)')
plt.ylabel('Position')
plt.savefig('scratch/dandisets/001275/2025-03-06-1/tmp_scripts/joystick_sample.png')
plt.close()

# Examine neural data
print("\n=== Neural Data ===")
units = nwb.processing["ecephys"]["units"]
print(f"Number of units (neurons): {len(units['id'].data)}")
print(f"Unit properties: {units.colnames}")

# Get basic statistics on units
quality_counts = {}
for q in units["quality"].data[:]:
    if q in quality_counts:
        quality_counts[q] += 1
    else:
        quality_counts[q] = 1

print(f"Unit quality counts: {quality_counts}")

# Plot firing rates
firing_rates = units["fr"].data[:]
plt.figure(figsize=(10, 6))
plt.hist(firing_rates, bins=20)
plt.title('Distribution of Firing Rates')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.savefig('scratch/dandisets/001275/2025-03-06-1/tmp_scripts/firing_rates.png')
plt.close()

# Examine electrode information
electrodes = nwb.ec_electrodes
print("\n=== Electrode Information ===")
print(f"Number of electrodes: {len(electrodes['id'].data[:])}")
print(f"Electrode properties: {electrodes.colnames}")

# Get electrode locations
locations = electrodes["location"].data[:]
location_counts = {}
for loc in locations:
    if loc in location_counts:
        location_counts[loc] += 1
    else:
        location_counts[loc] = 1

print(f"Electrode locations: {location_counts}")

# Print a summary of what we've learned
print("\n=== Summary ===")
print(f"This dataset contains recordings from {len(units['id'].data)} neurons in {nwb.subject.subject_id}'s brain during a mental navigation task.")
print(f"The task involved {len(trials['start_time'].data)} trials where the subject navigated mentally between landmarks.")
print("Behavioral data includes eye tracking and joystick movements.")
print("Neural data includes spike times and properties for each unit.")
