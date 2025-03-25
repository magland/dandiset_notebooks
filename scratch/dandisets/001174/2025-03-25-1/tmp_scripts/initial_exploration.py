# %% [markdown]
# # Initial Exploration of Dandiset 001174
#
# This script explores the calcium imaging data from the SMA/M1 recordings in macaques during a reaching task.
# We'll examine:
# 1. Basic dataset structure and properties
# 2. Example fluorescence traces and detected events
# 3. Spatial distribution of cells (footprints)

# %%
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

# %% [markdown]
# Let's load one example session from subject F:

# %%
# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/17963d0d-362d-40a3-aa7f-645c719f3f4a/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print(f"Session description: {nwb.session_description}")
print(f"Subject: {nwb.subject.subject_id} ({nwb.subject.species})")
print(f"Session start time: {nwb.session_start_time}")

# %% [markdown]
# Let's look at the imaging data properties:

# %%
imaging = nwb.acquisition["OnePhotonSeries"]
print(f"Image dimensions: {imaging.data.shape}")
print(f"Sampling rate: {imaging.rate} Hz")

# Let's look at an example frame
frame = imaging.data[0]  # Get first frame
plt.figure(figsize=(10, 8))
plt.imshow(frame, cmap='gray')
plt.colorbar(label='Intensity')
plt.title('Example calcium imaging frame')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.savefig('example_frame.png')
plt.close()

# %% [markdown]
# Now let's examine the detected cells and their activity:

# %%
# Get cell footprints
segmentation = nwb.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
footprints = segmentation["image_mask"].data[:]
print(f"Number of detected cells: {len(footprints)}")

# Visualize cell footprints
plt.figure(figsize=(10, 8))
# Sum all footprints to show cell locations
footprint_sum = np.sum(footprints, axis=0)
plt.imshow(footprint_sum, cmap='viridis')
plt.colorbar(label='Sum of cell masks')
plt.title('Spatial distribution of detected cells')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.savefig('cell_locations.png')
plt.close()

# %% [markdown]
# Let's look at fluorescence traces and detected events for a few example cells:

# %%
# Get fluorescence and events data
fluorescence = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"].data[:]
events = nwb.processing["ophys"]["EventAmplitude"].data[:]
time = np.arange(len(fluorescence)) / imaging.rate

# Plot first 3 cells for first 1000 timepoints
n_cells = 3
t_end = 1000

plt.figure(figsize=(15, 8))
for i in range(n_cells):
    plt.subplot(n_cells, 1, i+1)
    plt.plot(time[:t_end], fluorescence[:t_end, i], 'b-', label='Fluorescence', alpha=0.7)
    plt.plot(time[:t_end], events[:t_end, i], 'r.', label='Events')
    plt.ylabel(f'Cell {i+1}')
    if i == 0:
        plt.title('Fluorescence traces and detected events')
    if i == n_cells-1:
        plt.xlabel('Time (s)')
    plt.legend()
plt.tight_layout()
plt.savefig('example_traces.png')
plt.close()
