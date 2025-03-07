# %% [markdown]
# # Initial exploration of Dandiset 001174
# This script examines the structure and content of a calcium imaging session from macaque SMA,
# looking at fluorescence traces and basic properties of the data.

# %%
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load a sample NWB file from sub-Q (smaller file size for exploration)
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print(f"Session description: {nwb.session_description}")
print(f"Subject: {nwb.subject.subject_id} ({nwb.subject.species})")
print(f"Age: {nwb.subject.age}")
print(f"Recording start time: {nwb.session_start_time}")

# %%
# Basic properties of the imaging data
ophys = nwb.processing["ophys"]
fluorescence = ophys["Fluorescence"]["RoiResponseSeries"]
event_amplitude = ophys["EventAmplitude"]
segmentation = ophys["ImageSegmentation"]["PlaneSegmentation"]

print(f"\nNumber of ROIs: {fluorescence.data.shape[1]}")
print(f"Number of timepoints: {fluorescence.data.shape[0]}")
print(f"Sampling rate: {fluorescence.rate} Hz")
print(f"Recording duration: {fluorescence.data.shape[0]/fluorescence.rate:.2f} seconds")

# %%
# Plot fluorescence traces for a few example neurons
example_neurons = [0, 1, 2]  # First 3 neurons
time_vec = np.arange(fluorescence.data.shape[0]) / fluorescence.rate

plt.figure(figsize=(12, 8))
for i, n_id in enumerate(example_neurons):
    # Get fluorescence trace
    trace = fluorescence.data[:, n_id]
    # Get event amplitudes
    events = event_amplitude.data[:, n_id]
    
    plt.subplot(len(example_neurons), 1, i+1)
    plt.plot(time_vec, trace, 'b', label='Fluorescence')
    # Mark events with red dots
    event_times = time_vec[events > 0]
    event_values = trace[events > 0]
    plt.plot(event_times, event_values, 'r.', label='Events')
    
    plt.ylabel(f'ROI {n_id}')
    if i == 0:
        plt.title('Example fluorescence traces with detected events')
    if i == len(example_neurons)-1:
        plt.xlabel('Time (seconds)')
    plt.legend()

plt.tight_layout()
plt.savefig('example_traces.png')
plt.close()

# %%
# Plot the spatial footprints of these example neurons
masks = segmentation["image_mask"].data[example_neurons]

plt.figure(figsize=(12, 4))
for i, mask in enumerate(masks):
    plt.subplot(1, len(example_neurons), i+1)
    plt.imshow(mask, cmap='gray')
    plt.title(f'ROI {example_neurons[i]} footprint')
    plt.axis('off')

plt.tight_layout()
plt.savefig('example_footprints.png')
plt.close()

print("\nGenerated plots:")
print("1. example_traces.png - Shows fluorescence traces and detected events for 3 example neurons")
print("2. example_footprints.png - Shows the spatial footprints (locations) of these neurons in the field of view")
