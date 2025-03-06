import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the first NWB file
print('Loading NWB file...')
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001341/assets/e3d421c3-13bf-4b2e-b0f9-b6357a885de9/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print('\nBasic session info:')
print(f"Subject: {nwb.subject.subject_id}, Age: {nwb.subject.age}, Sex: {nwb.subject.sex}")
print(f"Genotype: {nwb.subject.genotype}")

print('\nNeural data summary:')
n_units = len(nwb.units['spike_times'])
print(f"Number of units: {n_units}")

# Get unit types
unit_types = nwb.units['unit_type'].data[:]
exc_units = np.sum(unit_types == 'excitatory')
inh_units = np.sum(unit_types == 'inhibitory')
print(f"Excitatory units: {exc_units}")
print(f"Inhibitory units: {inh_units}")

# Plot 1: Distribution of firing rates
plt.figure(figsize=(10, 6))
plt.hist(nwb.units['firing_rate'].data[:], bins=20)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Number of Units')
plt.title('Distribution of Unit Firing Rates')
plt.savefig('firing_rates.png')
plt.close()

# Trial information
trials = nwb.intervals['trials']
n_trials = len(trials['start_time'].data[:])
print(f"\nNumber of trials: {n_trials}")

# Note: Using file_start_time and file_stop_time as per readme note
trial_durations = trials['file_stop_time'].data[:] - trials['file_start_time'].data[:]
print(f"Average trial duration: {np.mean(trial_durations):.2f} seconds")
print(f"Trial duration range: {np.min(trial_durations):.2f} to {np.max(trial_durations):.2f} seconds")

# Plot 2: Trial durations
plt.figure(figsize=(10, 6))
plt.hist(trial_durations, bins=30)
plt.xlabel('Trial Duration (s)')
plt.ylabel('Number of Trials')
plt.title('Distribution of Trial Durations')
plt.savefig('trial_durations.png')
plt.close()

# Get a sample of movement data
print('\nAnalyzing movement data...')
# Take first 100,000 samples to avoid loading too much data
speed_abs = nwb.acquisition['speed_abs'].data[:100000]
speed_for = nwb.acquisition['speed_for'].data[:100000]
speed_lat = nwb.acquisition['speed_lat'].data[:100000]
timestamps = nwb.acquisition['speed_abs'].timestamps[:100000]

# Plot 3: Movement speeds
plt.figure(figsize=(12, 6))
plt.plot(timestamps[:1000], speed_abs[:1000], label='Absolute')
plt.plot(timestamps[:1000], speed_for[:1000], label='Forward')
plt.plot(timestamps[:1000], speed_lat[:1000], label='Lateral')
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.title('Movement Speeds (First 1000 Samples)')
plt.legend()
plt.savefig('movement_data.png')
plt.close()

# Plot 4: Spike raster for a few units
plt.figure(figsize=(15, 8))
# Plot first 5 units for first 10 seconds
end_time = 10  # seconds
for i in range(min(5, n_units)):
    spike_times = nwb.units['spike_times'][i]
    mask = spike_times < end_time
    plt.plot(spike_times[mask], np.ones_like(spike_times[mask]) * i, '|', markersize=5)
plt.xlabel('Time (s)')
plt.ylabel('Unit #')
plt.title('Spike Raster Plot (First 10 seconds)')
plt.savefig('spike_raster.png')
plt.close()

# Plot 5: Speed modulation of neural activity
print('\nAnalyzing speed modulation of neural activity...')
# Get speed bins
speed_bins = np.linspace(0, 30, 10)  # 0 to 30 cm/s in 10 bins
speed_centers = (speed_bins[1:] + speed_bins[:-1]) / 2

# Calculate firing rate vs speed for first 5 units
firing_vs_speed = np.zeros((min(5, n_units), len(speed_bins)-1))
for i in range(min(5, n_units)):
    spike_times = nwb.units['spike_times'][i]
    for j in range(len(speed_bins)-1):
        # Find timepoints where speed is in this bin
        speed_mask = (speed_abs >= speed_bins[j]) & (speed_abs < speed_bins[j+1])
        time_in_bin = np.sum(speed_mask) / len(speed_abs) * timestamps[-1]  # Total time in this speed bin
        
        if time_in_bin > 0:
            # Count spikes that occurred during these times
            n_spikes = np.sum((spike_times >= timestamps[0]) & 
                            (spike_times <= timestamps[-1]) &
                            (np.interp(spike_times, timestamps, speed_abs) >= speed_bins[j]) &
                            (np.interp(spike_times, timestamps, speed_abs) < speed_bins[j+1]))
            firing_vs_speed[i, j] = n_spikes / time_in_bin

plt.figure(figsize=(10, 6))
for i in range(min(5, n_units)):
    plt.plot(speed_centers, firing_vs_speed[i], '-o', label=f'Unit {i}')
plt.xlabel('Speed (cm/s)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Speed Modulation of Neural Activity')
plt.legend()
plt.savefig('speed_modulation.png')
plt.close()

print('\nAnalysis complete. Check the generated plots for visualization of the results.')
