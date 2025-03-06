#!/usr/bin/env python

"""
Improved analysis of trial structure in Dandiset 001275: Mental navigation primate PPC
This script examines the trial data without loading all timestamps at once.
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Examine trials information in detail
trials = nwb.intervals["trials"]
print("\n=== Trials Information ===")
print(f"Number of trials: {len(trials['start_time'].data)}")
print(f"Trial columns: {trials.colnames}")

# Check for NaN values in start_time and stop_time
start_times = trials["start_time"].data[:]
stop_times = trials["stop_time"].data[:]
print(f"NaN values in start_time: {np.isnan(start_times).sum()}")
print(f"NaN values in stop_time: {np.isnan(stop_times).sum()}")

# Print the first few values to inspect
print("\nFirst 5 start times:", start_times[:5])
print("First 5 stop times:", stop_times[:5])

# If there are valid start and stop times, calculate durations
valid_indices = ~(np.isnan(start_times) | np.isnan(stop_times))
if np.any(valid_indices):
    valid_starts = start_times[valid_indices]
    valid_stops = stop_times[valid_indices]
    durations = valid_stops - valid_starts
    
    print(f"\nValid trials: {np.sum(valid_indices)} out of {len(start_times)}")
    print(f"Average trial duration: {np.mean(durations):.2f} seconds")
    print(f"Min trial duration: {np.min(durations):.2f} seconds")
    print(f"Max trial duration: {np.max(durations):.2f} seconds")
    
    # Plot histogram of trial durations
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=30)
    plt.title('Distribution of Trial Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.savefig('scratch/dandisets/001275/2025-03-06-1/tmp_scripts/trial_durations.png')
    plt.close()
else:
    print("\nNo valid trial durations found.")

# Examine other trial properties
print("\n=== Trial Properties ===")
for col in trials.colnames:
    if col not in ['start_time', 'stop_time']:
        data = trials[col].data[:]
        if data.dtype.kind in 'iuf':  # Integer, unsigned int, or float
            print(f"{col}: min={np.nanmin(data)}, max={np.nanmax(data)}, mean={np.nanmean(data):.2f}")
            # Count NaN values
            nan_count = np.isnan(data).sum()
            if nan_count > 0:
                print(f"  NaN values: {nan_count}")
        else:
            # For non-numeric data, show unique values
            unique_vals = np.unique(data)
            if len(unique_vals) < 10:  # Only show if there aren't too many unique values
                print(f"{col}: unique values = {unique_vals}")
            else:
                print(f"{col}: {len(unique_vals)} unique values")

# Look at specific trial properties that might help understand the task
if 'target' in trials.colnames and 'curr' in trials.colnames:
    targets = trials['target'].data[:]
    currents = trials['curr'].data[:]
    
    # Check for NaN values
    valid_indices = ~(np.isnan(targets) | np.isnan(currents))
    if np.any(valid_indices):
        valid_targets = targets[valid_indices]
        valid_currents = currents[valid_indices]
        
        # Calculate distances between start and target positions
        distances = np.abs(valid_targets - valid_currents)
        
        print("\n=== Navigation Distances ===")
        print(f"Average distance: {np.mean(distances):.2f}")
        print(f"Min distance: {np.min(distances):.2f}")
        print(f"Max distance: {np.max(distances):.2f}")
        
        # Plot histogram of distances
        plt.figure(figsize=(10, 6))
        plt.hist(distances, bins=np.arange(np.min(distances), np.max(distances)+1.5)-0.5)
        plt.title('Distribution of Navigation Distances')
        plt.xlabel('Distance (landmarks)')
        plt.ylabel('Count')
        plt.xticks(np.arange(np.min(distances), np.max(distances)+1))
        plt.savefig('scratch/dandisets/001275/2025-03-06-1/tmp_scripts/navigation_distances.png')
        plt.close()
        
        # Plot start vs. target positions
        plt.figure(figsize=(10, 6))
        plt.scatter(valid_currents, valid_targets, alpha=0.5)
        plt.title('Start vs. Target Positions')
        plt.xlabel('Start Position (Landmark)')
        plt.ylabel('Target Position (Landmark)')
        plt.grid(True)
        plt.savefig('scratch/dandisets/001275/2025-03-06-1/tmp_scripts/start_vs_target.png')
        plt.close()

# Examine behavioral data during trials - only for a few selected trials
print("\n=== Behavioral Data During Trials ===")
hand_position = nwb.processing["behavior"]["hand_position"]

# Select a few trials to plot joystick movement
if np.any(valid_indices):
    # Get a few valid trials
    valid_trial_indices = np.where(valid_indices)[0][:5]  # First 5 valid trials
    
    plt.figure(figsize=(15, 10))
    for i, trial_idx in enumerate(valid_trial_indices):
        trial_start = start_times[trial_idx]
        trial_stop = stop_times[trial_idx]
        
        # Find indices for this time range - more efficient than loading all timestamps
        # We'll use a binary search approach to find approximate indices
        
        # Get a rough estimate of the indices based on sampling rate
        # From the previous script, we know the sampling rate is around 13.5 Hz
        estimated_sampling_rate = 13.5
        
        # Estimate the total number of samples based on the file size and data type
        total_samples = hand_position.data.shape[0]
        
        # Estimate the total duration of the recording
        if hasattr(nwb, 'session_start_time') and hasattr(nwb, 'file_create_date'):
            # This is just a rough estimate and might not be accurate
            total_duration = (nwb.file_create_date - nwb.session_start_time).total_seconds()
        else:
            # If we can't determine the duration, use the last trial time as an approximation
            total_duration = np.max(stop_times)
        
        # Estimate indices based on time
        est_start_idx = int((trial_start / total_duration) * total_samples)
        est_stop_idx = int((trial_stop / total_duration) * total_samples)
        
        # Add a buffer to ensure we capture the full trial
        buffer = 1000  # Add a buffer of 1000 samples
        start_idx = max(0, est_start_idx - buffer)
        stop_idx = min(total_samples, est_stop_idx + buffer)
        
        # Load only the timestamps and data for this range
        trial_hand_times = hand_position.timestamps[start_idx:stop_idx]
        trial_hand_data = hand_position.data[start_idx:stop_idx]
        
        # Further filter to the exact trial time range
        mask = (trial_hand_times >= trial_start) & (trial_hand_times <= trial_stop)
        trial_hand_times = trial_hand_times[mask]
        trial_hand_data = trial_hand_data[mask]
        
        if len(trial_hand_times) > 0:
            plt.subplot(len(valid_trial_indices), 1, i+1)
            plt.plot(trial_hand_times - trial_start, trial_hand_data)
            plt.title(f'Trial {trial_idx+1}: Start={currents[trial_idx]:.0f}, Target={targets[trial_idx]:.0f}')
            plt.ylabel('Joystick Position')
            if i == len(valid_trial_indices) - 1:
                plt.xlabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('scratch/dandisets/001275/2025-03-06-1/tmp_scripts/trial_joystick_movements.png')
    plt.close()

# Examine success rates
if 'succ' in trials.colnames:
    success = trials['succ'].data[:]
    success_rate = np.nanmean(success) * 100
    print(f"\nSuccess rate: {success_rate:.2f}%")
    
    # Success rate by distance
    if np.any(valid_indices) and 'target' in trials.colnames and 'curr' in trials.colnames:
        distances = np.abs(targets - currents)
        unique_distances = np.unique(distances[~np.isnan(distances)])
        
        success_by_distance = []
        for dist in unique_distances:
            dist_mask = distances == dist
            dist_success_rate = np.nanmean(success[dist_mask]) * 100
            success_by_distance.append((dist, dist_success_rate))
            print(f"Success rate for distance {dist}: {dist_success_rate:.2f}%")
        
        # Plot success rate by distance
        dists, rates = zip(*sorted(success_by_distance))
        plt.figure(figsize=(10, 6))
        plt.bar(dists, rates)
        plt.title('Success Rate by Navigation Distance')
        plt.xlabel('Distance (landmarks)')
        plt.ylabel('Success Rate (%)')
        plt.xticks(dists)
        plt.ylim(0, 100)
        plt.savefig('scratch/dandisets/001275/2025-03-06-1/tmp_scripts/success_by_distance.png')
        plt.close()

print("\n=== Summary ===")
print("This script analyzed the trial structure of the mental navigation task.")
print(f"The task involved navigating between landmarks on a linear track.")
if 'succ' in trials.colnames:
    print(f"Overall success rate: {success_rate:.2f}%")
if np.any(valid_indices) and 'target' in trials.colnames and 'curr' in trials.colnames:
    print(f"Navigation distances ranged from {np.min(distances):.0f} to {np.max(distances):.0f} landmarks.")
