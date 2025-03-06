import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the data
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001341/assets/e3d421c3-13bf-4b2e-b0f9-b6357a885de9/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get movement data (first 10000 timepoints for exploration)
print("\nLoading movement data...")
cor_pos = nwb.acquisition['cor_pos'].data[:10000]  # VR feedback signal
speed_for = nwb.acquisition['speed_for'].data[:10000]  # Forward speed
speed_lat = nwb.acquisition['speed_lat'].data[:10000]  # Lateral speed
timestamps = nwb.acquisition['cor_pos'].timestamps[:10000]

# Find a unit with moderate firing rate for clearer visualization
print("\nAnalyzing neural responses...")
units = nwb.units
firing_rates = units['firing_rate'].data[:]
# Get unit with firing rate closest to median
median_fr = np.median(firing_rates)
unit_idx = np.argmin(np.abs(firing_rates - median_fr))
example_unit_spikes = units['spike_times'][unit_idx]
print(f"Selected unit {unit_idx} with firing rate {firing_rates[unit_idx]:.2f} Hz")
print(f"Unit type: {units['unit_type'].data[:][unit_idx]}")
print(f"Unit layer: {units['layer'].data[:][unit_idx]}")

# Plot movement trajectory
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(timestamps, cor_pos)
plt.title('VR Position Feedback')
plt.xlabel('Time (s)')
plt.ylabel('Position')

plt.subplot(1, 2, 2)
plt.plot(timestamps, speed_for, label='Forward')
plt.plot(timestamps, speed_lat, label='Lateral')
plt.title('Movement Speed')
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.legend()
plt.tight_layout()
plt.savefig('001341_b/tmp_scripts/movement_data.png')
plt.close()

# Create spike raster for example neuron
# Get spikes within our timewindow
mask = (example_unit_spikes >= timestamps[0]) & (example_unit_spikes <= timestamps[-1])
spikes_in_window = example_unit_spikes[mask]

plt.figure(figsize=(12, 4))
plt.eventplot([spikes_in_window], lineoffsets=[0], linelengths=[1])
plt.title(f'Spike Raster for Unit {unit_idx}')
plt.xlabel('Time (s)')
plt.yticks([])
plt.savefig('001341_b/tmp_scripts/spike_raster.png')
plt.close()

# Calculate speed modulation with error bars
print("\nCalculating speed modulation...")
print(f"Speed range: {np.min(speed_for):.1f} to {np.max(speed_for):.1f} cm/s")
# Use fewer bins for more robust statistics
bins = np.linspace(0, np.max(speed_for), 10)
bin_centers = (bins[:-1] + bins[1:]) / 2
firing_rates_by_speed = []
firing_rate_errors = []
time_in_bins = []

# Bin size in seconds for spike counting
dt = 0.1

for i in range(len(bin_centers)):
    speed_mask = (speed_for >= bins[i]) & (speed_for < bins[i+1])
    time_in_bin = np.sum(speed_mask) * dt
    time_in_bins.append(time_in_bin)
    
    if time_in_bin > 0:
        # Get all time windows in this speed bin
        speed_times = timestamps[speed_mask]
        rates = []
        
        # Calculate rate for each window
        for t_start in speed_times:
            spike_mask = (example_unit_spikes >= t_start) & (example_unit_spikes < t_start + dt)
            spike_count = np.sum(spike_mask)
            rates.append(spike_count / dt)
        
        # Calculate mean and SEM
        rates = np.array(rates)
        firing_rates_by_speed.append(np.mean(rates))
        firing_rate_errors.append(np.std(rates) / np.sqrt(len(rates)))
        print(f"Speed bin {bins[i]:.1f}-{bins[i+1]:.1f} cm/s: {len(rates)} samples")
    else:
        firing_rates_by_speed.append(0)
        firing_rate_errors.append(0)
        print(f"Speed bin {bins[i]:.1f}-{bins[i+1]:.1f} cm/s: No samples")

firing_rates_by_speed = np.array(firing_rates_by_speed)
firing_rate_errors = np.array(firing_rate_errors)

print("\nSpeed bin statistics:")
for i in range(len(bin_centers)):
    print(f"Speed {bin_centers[i]:.1f} cm/s: {time_in_bins[i]:.2f}s total time, {firing_rates_by_speed[i]:.1f} ± {firing_rate_errors[i]:.1f} Hz")

plt.figure(figsize=(8, 6))
plt.errorbar(bin_centers, firing_rates_by_speed, yerr=firing_rate_errors, fmt='o-')
plt.title(f'Speed Modulation of Unit {unit_idx}')
plt.xlabel('Forward Speed (cm/s)')
plt.ylabel('Firing Rate (Hz)')
plt.savefig('001341_b/tmp_scripts/speed_modulation.png')
plt.close()

print("\nAnalysis complete. Generated plots:")
print("1. movement_data.png - Shows VR position and speed profiles")
print("2. spike_raster.png - Shows spike timing for example neuron")
print("3. speed_modulation.png - Shows how firing rate varies with movement speed")
