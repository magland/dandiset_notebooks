import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import medfilt

# Load the same NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get data streams
pupil_radius = nwb.acquisition["PupilTracking"]["pupil_raw_radius"].data[:]
pupil_times = nwb.acquisition["PupilTracking"]["pupil_raw_radius"].timestamps[:]
velocity = nwb.acquisition["treadmill_velocity"].data[:]
velocity_times = nwb.acquisition["treadmill_velocity"].timestamps[:]
fluorescence = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries1"].data[:]
fluor_times = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries1"].timestamps[:]

# Find periods of valid pupil tracking (non-nan values)
valid_pupil = ~np.isnan(pupil_radius)
print(f"Valid pupil samples: {np.sum(valid_pupil)} out of {len(pupil_radius)}")

# Get active periods based on movement
velocity_thresh = 2.0  # Arbitrary threshold for movement
is_moving = np.abs(velocity) > velocity_thresh
active_periods = []
current_period = []
for i, (t, moving) in enumerate(zip(velocity_times, is_moving)):
    if moving and not current_period:
        current_period = [t]
    elif not moving and current_period:
        current_period.append(velocity_times[i-1])
        active_periods.append(current_period)
        current_period = []

print(f"\nFound {len(active_periods)} active movement periods")

# Plot example period
if active_periods:
    # Take first period that's at least 10 seconds long
    for period in active_periods:
        period_length = period[1] - period[0]
        if period_length > 10:
            example_period = period
            break
    else:
        example_period = active_periods[0]
    
    start_time, end_time = example_period
    margin = 5  # Add 5 seconds before and after
    plot_start = start_time - margin
    plot_end = end_time + margin
    
    # Get data within time window
    def get_time_mask(times):
        return (times >= plot_start) & (times <= plot_end)
    
    pupil_mask = get_time_mask(pupil_times)
    velocity_mask = get_time_mask(velocity_times)
    fluor_mask = get_time_mask(fluor_times)
    
    # Normalize signals for comparison
    pupil_norm = (pupil_radius[pupil_mask] - np.nanmean(pupil_radius)) / np.nanstd(pupil_radius)
    velocity_norm = (velocity[velocity_mask] - np.mean(velocity)) / np.std(velocity)
    fluor_norm = (fluorescence[fluor_mask] - np.mean(fluorescence)) / np.std(fluorescence)
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Plot normalized signals
    ax1.plot(pupil_times[pupil_mask], pupil_norm)
    ax1.set_ylabel('Normalized\nPupil Size')
    ax1.axvspan(start_time, end_time, color='gray', alpha=0.2)
    
    ax2.plot(velocity_times[velocity_mask], velocity_norm)
    ax2.set_ylabel('Normalized\nVelocity')
    ax2.axvspan(start_time, end_time, color='gray', alpha=0.2)
    
    ax3.plot(fluor_times[fluor_mask], fluor_norm.flatten())
    ax3.set_ylabel('Normalized\nACh Signal')
    ax3.set_xlabel('Time (s)')
    ax3.axvspan(start_time, end_time, color='gray', alpha=0.2)
    
    plt.suptitle('Example Active Period with Behavioral and Neural Data')
    plt.tight_layout()
    plt.savefig("scratch/dandisets/001176/2025-03-06-1/tmp_scripts/active_period.png")
    plt.close()

# Analyze relationship between behavioral state and ACh levels
# Interpolate behavioral data to fluorescence timestamps for comparison
valid_velocity_times = ~np.isnan(velocity)
valid_pupil_times = ~np.isnan(pupil_radius)

if np.any(valid_velocity_times) and np.any(valid_pupil_times):
    velocity_interp = interp1d(velocity_times[valid_velocity_times], 
                             velocity[valid_velocity_times], 
                             bounds_error=False)(fluor_times)
    pupil_interp = interp1d(pupil_times[valid_pupil_times],
                           pupil_radius[valid_pupil_times],
                           bounds_error=False)(fluor_times)
    
    # Remove any remaining nans
    valid_samples = ~np.isnan(velocity_interp) & ~np.isnan(pupil_interp)
    
    if np.any(valid_samples):
        # Create state-dependent averages
        high_velocity = velocity_interp > np.nanpercentile(velocity_interp, 75)
        low_velocity = velocity_interp < np.nanpercentile(velocity_interp, 25)
        high_pupil = pupil_interp > np.nanpercentile(pupil_interp, 75)
        low_pupil = pupil_interp < np.nanpercentile(pupil_interp, 25)
        
        # Calculate mean ACh levels in different states
        ach_means = {
            'High Velocity': np.mean(fluorescence[high_velocity]),
            'Low Velocity': np.mean(fluorescence[low_velocity]),
            'Large Pupil': np.mean(fluorescence[high_pupil]),
            'Small Pupil': np.mean(fluorescence[low_pupil])
        }
        
        # Plot state-dependent ACh levels
        plt.figure(figsize=(8, 6))
        x = np.arange(len(ach_means))
        plt.bar(x, list(ach_means.values()))
        plt.xticks(x, list(ach_means.keys()), rotation=45)
        plt.ylabel('Mean ACh Fluorescence')
        plt.title('ACh Levels in Different Behavioral States')
        plt.tight_layout()
        plt.savefig("scratch/dandisets/001176/2025-03-06-1/tmp_scripts/state_dependent_ach.png")
        plt.close()
        
        print("\nMean ACh levels in different states:")
        for state, mean_val in ach_means.items():
            print(f"{state}: {mean_val:.2f}")
