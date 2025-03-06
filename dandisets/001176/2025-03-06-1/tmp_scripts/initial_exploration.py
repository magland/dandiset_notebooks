import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load a sample NWB file from M1
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001176/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print("Dataset Information:")
print(f"Subject: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Description: {nwb.experiment_description}")
print(f"Region: M1")
print()

# Get behavioral data
pupil_radius = nwb.acquisition["PupilTracking"]["pupil_raw_radius"].data[:]
pupil_times = nwb.acquisition["PupilTracking"]["pupil_raw_radius"].timestamps[:]
velocity = nwb.acquisition["treadmill_velocity"].data[:]
velocity_times = nwb.acquisition["treadmill_velocity"].timestamps[:]

# Get fluorescence data
fluorescence = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries1"].data[:]
fluor_times = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries1"].timestamps[:]

# Plot pupil radius over time
plt.figure(figsize=(10, 4))
plt.plot(pupil_times, pupil_radius)
plt.title("Pupil Radius Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Radius (pixels)")
plt.tight_layout()
plt.savefig("scratch/dandisets/001176/2025-03-06-1/tmp_scripts/pupil_radius.png")
plt.close()

# Plot treadmill velocity
plt.figure(figsize=(10, 4))
plt.plot(velocity_times, velocity)
plt.title("Treadmill Velocity Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
plt.tight_layout()
plt.savefig("scratch/dandisets/001176/2025-03-06-1/tmp_scripts/velocity.png")
plt.close()

# Plot fluorescence trace
plt.figure(figsize=(10, 4))
plt.plot(fluor_times, fluorescence)
plt.title("ACh Fluorescence Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Fluorescence")
plt.tight_layout()
plt.savefig("scratch/dandisets/001176/2025-03-06-1/tmp_scripts/fluorescence.png")
plt.close()

# Get correlation between pupil size and fluorescence
# First interpolate pupil data to match fluorescence timestamps
from scipy.interpolate import interp1d
pupil_interp = interp1d(pupil_times, pupil_radius)(fluor_times)
correlation = np.corrcoef(pupil_interp, fluorescence.flatten())[0, 1]
print(f"\nCorrelation between pupil size and ACh fluorescence: {correlation:.3f}")

# Plot summary images
summary_images = nwb.processing["ophys"]["SummaryImages_chan1"]
avg_image = summary_images["average"].data[:]
corr_image = summary_images["correlation"].data[:]

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.imshow(avg_image, cmap='gray')
plt.title("Average Image")
plt.colorbar()
plt.subplot(122)
plt.imshow(corr_image, cmap='viridis')
plt.title("Correlation Image")
plt.colorbar()
plt.tight_layout()
plt.savefig("scratch/dandisets/001176/2025-03-06-1/tmp_scripts/summary_images.png")
plt.close()
