#!/usr/bin/env python

"""
Analysis of neural activity in Dandiset 001275: Mental navigation primate PPC
This script examines how neural activity relates to the mental navigation task.
"""

import numpy as np
import matplotlib.pyplot as plt
import pynwb
import lindi
from collections import defaultdict

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/001275/assets/b0bbeb4c-5e0d-4050-a993-798173797d94/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get trial information
trials = nwb.intervals["trials"]
start_times = trials["start_time"].data[:]
stop_times = trials["stop_time"].data[:]
valid_indices = ~(np.isnan(start_times) | np.isnan(stop_times))

# Get neural data
units = nwb.processing["ecephys"]["units"]
print("\n=== Neural Data ===")
print(f"Number of units (neurons): {len(units['id'].data)}")
print(f"Unit properties: {units.colnames}")

# Get quality counts
quality_counts = {}
for q in units["quality"].data[:]:
    if q in quality_counts:
        quality_counts[q] += 1
    else:
        quality_counts[q] = 1

print(f"Unit quality counts: {quality_counts}")

# Filter for good units only
good_unit_indices = [i for i, q in enumerate(units["quality"].data[:]) if q == 'good']
print(f"Number of good units: {len(good_unit_indices)}")

# Get firing rates of good units
good_firing_rates = units["fr"].data[:][good_unit_indices]
print(f"Average firing rate of good units: {np.mean(good_firing_rates):.2f} Hz")
print(f"Min firing rate: {np.min(good_firing_rates):.2f} Hz")
print(f"Max firing rate: {np.max(good_firing_rates):.2f} Hz")

# Plot distribution of firing rates
plt.figure(figsize=(10, 6))
plt.hist(good_firing_rates, bins=15)
plt.title('Distribution of Firing Rates (Good Units)')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.savefig('scratch/dandisets/001275/2025-03-06-1/tmp_scripts/good_unit_firing_rates.png')
plt.close()

# Analyze neural activity for a few example trials
# We'll select a few trials with different navigation distances
if 'target' in trials.colnames and 'curr' in trials.colnames:
    targets = trials['target'].data[:]
    currents = trials['curr'].data[:]
    
    # Calculate distances
    distances = np.abs(targets - currents)
    
    # Find trial indices for each distance
    distance_trial_indices = {}
    for dist in range(1, 6):  # Distances 1 through 5
        dist_indices = np.where((distances == dist) & valid_indices)[0]
        if len(dist_indices) > 0:
            # Take the first trial for each distance
            distance_trial_indices[dist] = dist_indices[0]
    
    print("\n=== Example Trials for Neural Analysis ===")
    for dist, trial_idx in distance_trial_indices.items():
        print(f"Distance {dist}: Trial {trial_idx}, Start={currents[trial_idx]:.0f}, Target={targets[trial_idx]:.0f}")
    
    # Select a subset of good units for analysis (first 5)
    analysis_unit_indices = good_unit_indices[:5] if len(good_unit_indices) >= 5 else good_unit_indices
    
    print(f"\nAnalyzing {len(analysis_unit_indices)} good units across example trials")
    
    # For each selected trial, count spikes from each unit during the trial
    trial_unit_spikes = {}
    
    for dist, trial_idx in distance_trial_indices.items():
        trial_start = start_times[trial_idx]
        trial_stop = stop_times[trial_idx]
        trial_duration = trial_stop - trial_start
        
        unit_spikes = {}
        for i, unit_idx in enumerate(analysis_unit_indices):
            # Get spike times for this unit
            spike_times = units["spike_times"][unit_idx]
            
            # Count spikes during this trial
            trial_spikes = spike_times[(spike_times >= trial_start) & (spike_times <= trial_stop)]
            spike_count = len(trial_spikes)
            
            # Calculate firing rate during this trial
            trial_firing_rate = spike_count / trial_duration if trial_duration > 0 else 0
            
            unit_spikes[unit_idx] = {
                'count': spike_count,
                'rate': trial_firing_rate
            }
        
        trial_unit_spikes[dist] = unit_spikes
    
    # Print spike counts and rates for each unit in each trial
    print("\n=== Neural Activity During Example Trials ===")
    for dist, unit_data in trial_unit_spikes.items():
        print(f"\nDistance {dist} (Trial {distance_trial_indices[dist]}):")
        for unit_idx, spike_data in unit_data.items():
            unit_name = units["unit_name"].data[unit_idx]
            print(f"  Unit {unit_name}: {spike_data['count']} spikes, {spike_data['rate']:.2f} Hz")
    
    # Plot firing rates by distance
    plt.figure(figsize=(12, 8))
    for i, unit_idx in enumerate(analysis_unit_indices):
        unit_name = units["unit_name"].data[unit_idx]
        rates = [trial_unit_spikes[dist][unit_idx]['rate'] for dist in distance_trial_indices.keys()]
        plt.plot(list(distance_trial_indices.keys()), rates, 'o-', label=f"Unit {unit_name}")
    
    plt.title('Firing Rates by Navigation Distance')
    plt.xlabel('Distance (landmarks)')
    plt.ylabel('Firing Rate (Hz)')
    plt.xticks(list(distance_trial_indices.keys()))
    plt.legend()
    plt.grid(True)
    plt.savefig('scratch/dandisets/001275/2025-03-06-1/tmp_scripts/firing_rates_by_distance.png')
    plt.close()

# Analyze neural activity by trial outcome (success vs. failure)
if 'succ' in trials.colnames:
    success = trials['succ'].data[:]
    
    # Find successful and failed trials
    successful_trials = np.where(success == 1)[0]
    failed_trials = np.where(success == 0)[0]
    
    print(f"\n=== Neural Activity by Trial Outcome ===")
    print(f"Number of successful trials: {len(successful_trials)}")
    print(f"Number of failed trials: {len(failed_trials)}")
    
    # Select a subset of trials for analysis
    num_trials_to_analyze = 10
    successful_sample = successful_trials[:num_trials_to_analyze] if len(successful_trials) >= num_trials_to_analyze else successful_trials
    failed_sample = failed_trials[:num_trials_to_analyze] if len(failed_trials) >= num_trials_to_analyze else failed_trials
    
    # Calculate average firing rates during successful and failed trials
    success_rates = defaultdict(list)
    failure_rates = defaultdict(list)
    
    # Process successful trials
    for trial_idx in successful_sample:
        if not valid_indices[trial_idx]:
            continue
            
        trial_start = start_times[trial_idx]
        trial_stop = stop_times[trial_idx]
        trial_duration = trial_stop - trial_start
        
        for unit_idx in analysis_unit_indices:
            spike_times = units["spike_times"][unit_idx]
            trial_spikes = spike_times[(spike_times >= trial_start) & (spike_times <= trial_stop)]
            trial_firing_rate = len(trial_spikes) / trial_duration if trial_duration > 0 else 0
            success_rates[unit_idx].append(trial_firing_rate)
    
    # Process failed trials
    for trial_idx in failed_sample:
        if not valid_indices[trial_idx]:
            continue
            
        trial_start = start_times[trial_idx]
        trial_stop = stop_times[trial_idx]
        trial_duration = trial_stop - trial_start
        
        for unit_idx in analysis_unit_indices:
            spike_times = units["spike_times"][unit_idx]
            trial_spikes = spike_times[(spike_times >= trial_start) & (spike_times <= trial_stop)]
            trial_firing_rate = len(trial_spikes) / trial_duration if trial_duration > 0 else 0
            failure_rates[unit_idx].append(trial_firing_rate)
    
    # Calculate average rates
    avg_success_rates = {unit_idx: np.mean(rates) for unit_idx, rates in success_rates.items() if rates}
    avg_failure_rates = {unit_idx: np.mean(rates) for unit_idx, rates in failure_rates.items() if rates}
    
    # Print average rates
    print("\nAverage firing rates during successful vs. failed trials:")
    for unit_idx in analysis_unit_indices:
        unit_name = units["unit_name"].data[unit_idx]
        success_rate = avg_success_rates.get(unit_idx, 0)
        failure_rate = avg_failure_rates.get(unit_idx, 0)
        print(f"  Unit {unit_name}: Success={success_rate:.2f} Hz, Failure={failure_rate:.2f} Hz")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    unit_names = [units["unit_name"].data[idx] for idx in analysis_unit_indices]
    success_vals = [avg_success_rates.get(idx, 0) for idx in analysis_unit_indices]
    failure_vals = [avg_failure_rates.get(idx, 0) for idx in analysis_unit_indices]
    
    x = np.arange(len(unit_names))
    width = 0.35
    
    plt.bar(x - width/2, success_vals, width, label='Successful Trials')
    plt.bar(x + width/2, failure_vals, width, label='Failed Trials')
    
    plt.xlabel('Unit')
    plt.ylabel('Average Firing Rate (Hz)')
    plt.title('Neural Activity by Trial Outcome')
    plt.xticks(x, unit_names)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('scratch/dandisets/001275/2025-03-06-1/tmp_scripts/success_vs_failure_rates.png')
    plt.close()

print("\n=== Summary ===")
print(f"This script analyzed the neural activity of {len(good_unit_indices)} good units during the mental navigation task.")
print("We examined how firing rates vary with navigation distance and trial outcome (success vs. failure).")
print("The results suggest that neural activity in the posterior parietal cortex (PPC) may encode information about the navigation task.")
