import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the first NWB file
print('Loading NWB file...')
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001341/assets/e3d421c3-13bf-4b2e-b0f9-b6357a885de9/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

def get_unit_id(index):
    """Helper function to get the unit ID"""
    return nwb.units['id'].data[index]

# Trial information with improved duration analysis
trials = nwb.intervals['trials']
trial_durations = trials['file_stop_time'].data[:] - trials['file_start_time'].data[:]

print('\nPlotting trial duration distribution...')
plt.figure(figsize=(10, 6))
plt.hist(trial_durations[trial_durations > 0], bins=40)  # Filter out negative durations
plt.xlabel('Trial Duration (s)')
plt.ylabel('Number of Trials')
plt.title('Distribution of Trial Durations')
plt.grid(True, alpha=0.3)
plt.savefig('trial_durations_improved.png')
plt.close()

print('\nPlotting spike raster...')
# Improved spike raster with better visibility and proper unit IDs
plt.figure(figsize=(15, 8))
n_example_units = 5
end_time = 10  # seconds

# Create y-axis labels with unit IDs
unit_labels = [f'Unit {get_unit_id(i)}' for i in range(n_example_units)]

for i in range(n_example_units):
    spike_times = nwb.units['spike_times'][i]
    mask = spike_times < end_time
    plt.plot(spike_times[mask], np.ones_like(spike_times[mask]) * i, '|', markersize=8, label=unit_labels[i])

plt.yticks(range(n_example_units), unit_labels)
plt.xlabel('Time (s)')
plt.title('Spike Raster Plot (First 10 seconds)')
plt.grid(True, alpha=0.3)
plt.savefig('spike_raster_improved.png')
plt.close()

print('\nAnalyzing trial-aligned neural activity...')
# Load a subset of the speed data for analysis
chunk_size = 100000  # Analyze data in chunks
sample_size = chunk_size  # Total amount of data to analyze

# Load speed data
speed_abs_data = nwb.acquisition['speed_abs'].data[:sample_size]
speed_for_data = nwb.acquisition['speed_for'].data[:sample_size]
speed_lat_data = nwb.acquisition['speed_lat'].data[:sample_size]
timestamps = np.linspace(0, sample_size/1000, sample_size)  # Approximate timestamps assuming 1kHz

print('\nPlotting movement speeds...')
# Plot movement data
plt.figure(figsize=(12, 6))
plt.plot(timestamps[:1000], speed_abs_data[:1000], label='Absolute')
plt.plot(timestamps[:1000], speed_for_data[:1000], label='Forward')
plt.plot(timestamps[:1000], speed_lat_data[:1000], label='Lateral')
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.title('Movement Speeds (First Second)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('movement_speeds.png')
plt.close()

print('\nAnalyzing speed modulation of neural activity...')
# Improved speed modulation analysis with proper unit IDs
speed_bins = np.linspace(0, 30, 10)
speed_centers = (speed_bins[1:] + speed_bins[:-1]) / 2

firing_vs_speed = np.zeros((n_example_units, len(speed_bins)-1))
for i in range(n_example_units):
    unit_id = get_unit_id(i)
    spike_times = nwb.units['spike_times'][i]
    spike_mask = (spike_times >= timestamps[0]) & (spike_times <= timestamps[-1])
    spike_times_subset = spike_times[spike_mask]
    
    for j in range(len(speed_bins)-1):
        speed_mask = (speed_abs_data >= speed_bins[j]) & (speed_abs_data < speed_bins[j+1])
        time_in_bin = np.sum(speed_mask) / len(speed_abs_data) * (timestamps[-1] - timestamps[0])
        
        if time_in_bin > 0:
            n_spikes = np.sum((spike_times_subset >= timestamps[0]) & 
                            (spike_times_subset <= timestamps[-1]) &
                            (np.interp(spike_times_subset, timestamps, speed_abs_data) >= speed_bins[j]) &
                            (np.interp(spike_times_subset, timestamps, speed_abs_data) < speed_bins[j+1]))
            firing_vs_speed[i, j] = n_spikes / time_in_bin

plt.figure(figsize=(10, 6))
for i in range(n_example_units):
    plt.plot(speed_centers, firing_vs_speed[i], '-o', label=f'Unit {get_unit_id(i)}')
plt.xlabel('Speed (cm/s)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Speed Modulation of Neural Activity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('speed_modulation_improved.png')
plt.close()

print('\nAnalysis complete. Check the generated plots for visualization of the results.')
