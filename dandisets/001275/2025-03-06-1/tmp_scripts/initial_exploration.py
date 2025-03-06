import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the behavior+ecephys file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Basic session info
print("Session Info:")
print(f"Subject: {nwb.subject.subject_id}")
print(f"Date: {nwb.session_start_time}")
print(f"Number of trials: {len(nwb.intervals['trials']['start_time'])}")
print(f"Number of units: {len(nwb.processing['ecephys']['units']['id'])}")

print("\nUnit Quality Distribution:")
qualities = nwb.processing['ecephys']['units']['quality'][:]
unique_qualities, counts = np.unique(qualities, return_counts=True)
for q, c in zip(unique_qualities, counts):
    print(f"{q}: {c}")

# Get trial durations for first 100 trials
trial_starts = nwb.intervals['trials']['start_time'][:100]
trial_stops = nwb.intervals['trials']['stop_time'][:100]
trial_durations = trial_stops - trial_starts

print("\nTrial Duration Statistics (first 100 trials, seconds):")
print(f"Mean: {np.mean(trial_durations):.2f}")
print(f"Median: {np.median(trial_durations):.2f}")
print(f"Min: {np.min(trial_durations):.2f}")
print(f"Max: {np.max(trial_durations):.2f}")

# Plot trial duration distribution
plt.figure(figsize=(10, 6))
plt.hist(trial_durations, bins=30)
plt.xlabel('Trial Duration (s)')
plt.ylabel('Count')
plt.title('Distribution of Trial Durations (First 100 Trials)')
plt.grid(True)
plt.savefig('trial_durations.png')
plt.close()

# Print information about first few trials
print("\nFirst 5 trials:")
for i in range(5):
    print(f"Trial {i+1}: duration = {trial_durations[i]:.2f}s")

# Print information about units
units = nwb.processing['ecephys']['units']
good_unit_indices = [i for i, q in enumerate(units['quality'][:]) if q == 'good']

print("\nFiring Rate Statistics (Good Units Only):")
good_firing_rates = units['fr'][:][good_unit_indices]
print(f"Mean firing rate: {np.mean(good_firing_rates):.2f} Hz")
print(f"Median firing rate: {np.median(good_firing_rates):.2f} Hz")
print(f"Min firing rate: {np.min(good_firing_rates):.2f} Hz")
print(f"Max firing rate: {np.max(good_firing_rates):.2f} Hz")

# Get data ranges
hand_pos = nwb.processing['behavior']['hand_position']
print("\nData Shapes:")
print(f"Hand position shape: {hand_pos.data.shape}")
if len(hand_pos.data.shape) > 1:
    print(f"Hand position dimensions: {len(hand_pos.data.shape)}")
print(f"Number of trials: {len(trial_starts)}")
print(f"Number of good units: {len(good_unit_indices)}")

# Print information about electrode locations
print("\nElectrode Locations:")
locations = np.unique(nwb.ec_electrodes['location'][:])
for loc in locations:
    count = np.sum(nwb.ec_electrodes['location'][:] == loc)
    print(f"{loc}: {count} electrodes")
