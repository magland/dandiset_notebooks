# %% [markdown]
# # Neural Response Analysis
# This script analyzes neural responses to odor presentations

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# %% Load the data
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001335/assets/aca66db7-4c02-4453-8dcb-a179d44b1c5d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# %% Analysis parameters
window = [-1, 3]  # Time window around odor onset (1s before to 3s after)
selected_odor = 'A'  # We'll focus on responses to odor A
n_units = 10  # Number of units to analyze

# %% Get spike data aligned to odor presentations
print(f"\nAnalyzing neural responses to Odor {selected_odor}...")
odor_starts = nwb.intervals[f"Odor {selected_odor} ON"]["start_time"].data[:]

# Get first 10 presentations
n_presentations = 10
odor_starts = odor_starts[:n_presentations]

# Initialize array to store spike counts
time_bins = np.arange(window[0], window[1], 0.1)  # 100ms bins
spike_counts = np.zeros((n_units, n_presentations, len(time_bins)-1))

# For each unit
for i in range(n_units):
    spike_times = nwb.units["spike_times"][i]
    unit_depth = nwb.units["depth"].data[i]
    
    # For each odor presentation
    for j, start in enumerate(odor_starts):
        # Get spikes in window around odor onset
        window_mask = (spike_times >= start + window[0]) & (spike_times <= start + window[1])
        presentation_spikes = spike_times[window_mask] - start
        
        # Count spikes in bins
        spike_counts[i, j, :], _ = np.histogram(presentation_spikes, bins=time_bins)

# %% Create PSTH plot
print("\nCreating PSTH plot...")
plt.figure(figsize=(15, 10))

# Plot individual unit PSTHs
for i in range(n_units):
    plt.subplot(n_units, 1, i+1)
    mean_rate = np.mean(spike_counts[i], axis=0) / 0.1  # Convert to Hz
    sem_rate = np.std(spike_counts[i], axis=0) / np.sqrt(n_presentations) / 0.1  # Standard error of mean
    
    plt.fill_between(time_bins[:-1], mean_rate - sem_rate, mean_rate + sem_rate, alpha=0.3)
    plt.plot(time_bins[:-1], mean_rate)
    
    # Plot odor presentation period
    plt.axvspan(0, 2, color='gray', alpha=0.2)
    plt.axvline(0, color='k', linestyle='--')
    
    plt.ylabel(f'Unit {i}\nSpikes/s')
    if i == 0:
        plt.title(f'PSTH for first {n_units} units during Odor {selected_odor} presentation')
    if i == n_units-1:
        plt.xlabel('Time from odor onset (s)')

plt.tight_layout()
plt.savefig('odor_psth.png')
plt.close()

# %% Analyze LFP around odor onset
print("\nAnalyzing LFP responses...")
lfp = nwb.processing["ecephys"]["LFP"]
sample_rate = lfp.rate

# Calculate window sizes in samples
pre_samples = int(-window[0] * sample_rate)
post_samples = int(window[1] * sample_rate)
total_samples = pre_samples + post_samples

# Get LFP for first presentation, first 5 channels
n_channels = 5
presentation_start = odor_starts[0]
start_idx = int(presentation_start * sample_rate) - pre_samples
end_idx = start_idx + total_samples

try:
    lfp_snippet = lfp.data[start_idx:end_idx, :n_channels]
except Exception as e:
    print(f"Error loading LFP data: {e}")
    print("Skipping LFP analysis")
    lfp_snippet = None
time = np.linspace(window[0], window[1], total_samples)

# Plot LFP
if lfp_snippet is not None:
    plt.figure(figsize=(15, 8))
    for i in range(n_channels):
        plt.plot(time, lfp_snippet[:, i] + i*200, label=f'Channel {i}')

    plt.axvspan(0, 2, color='gray', alpha=0.2)
    plt.axvline(0, color='k', linestyle='--')
    plt.xlabel('Time from odor onset (s)')
    plt.ylabel('LFP Signal (offset for visibility)')
    plt.title(f'LFP around first Odor {selected_odor} presentation')
    plt.legend()
    plt.savefig('odor_lfp.png')
    plt.close()

print("\nFinished creating plots: odor_psth.png and odor_lfp.png")

# %% Print some statistics about neural responses
print("\nComputing response statistics...")
baseline_period = (window[0], 0)  # Pre-odor period
response_period = (0, 2)  # During odor

for i in range(n_units):
    baseline_idx = (time_bins[:-1] >= baseline_period[0]) & (time_bins[:-1] < baseline_period[1])
    response_idx = (time_bins[:-1] >= response_period[0]) & (time_bins[:-1] < response_period[1])
    
    baseline_rate = np.mean(spike_counts[i, :, baseline_idx]) / 0.1
    response_rate = np.mean(spike_counts[i, :, response_idx]) / 0.1
    
    print(f"\nUnit {i}:")
    print(f"Baseline firing rate: {baseline_rate:.2f} Hz")
    print(f"Response firing rate: {response_rate:.2f} Hz")
    print(f"Modulation: {((response_rate - baseline_rate) / baseline_rate * 100):.1f}%")
