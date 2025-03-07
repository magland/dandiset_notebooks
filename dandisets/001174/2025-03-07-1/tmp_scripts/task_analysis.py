# %% [markdown]
# # Task-related analysis of calcium imaging data
# This script compares neural activity between spontaneous and task conditions,
# looking at differences in event rates and patterns of activity.

# %%
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load one spontaneous and one task session from subject Q for comparison
# First the spontaneous session we looked at before
f_spont = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/de07db56-e7f3-4809-9972-755c51598e8d/nwb.lindi.json")
nwb_spont = pynwb.NWBHDF5IO(file=f_spont, mode='r').read()

# Load a task session
f_task = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/807851a7-ad52-4505-84ee-3b155a5bd2a3/nwb.lindi.json")
nwb_task = pynwb.NWBHDF5IO(file=f_task, mode='r').read()

print("Spontaneous session:", nwb_spont.session_description)
print("Task session:", nwb_task.session_description)

# %%
# Function to compute event rate for each neuron
def compute_event_rates(events, sampling_rate):
    """Compute event rate (events/second) for each neuron"""
    duration = events.shape[0] / sampling_rate
    event_counts = np.sum(events > 0, axis=0)
    return event_counts / duration

# Get event rates for both conditions
spont_events = nwb_spont.processing["ophys"]["EventAmplitude"].data[:]
task_events = nwb_task.processing["ophys"]["EventAmplitude"].data[:]

spont_rates = compute_event_rates(spont_events, nwb_spont.processing["ophys"]["EventAmplitude"].rate)
task_rates = compute_event_rates(task_events, nwb_task.processing["ophys"]["EventAmplitude"].rate)

# %%
# Plot distribution of event rates in both conditions
plt.figure(figsize=(10, 6))
plt.hist(spont_rates, bins=15, alpha=0.5, label='Spontaneous', density=True)
plt.hist(task_rates, bins=15, alpha=0.5, label='Task', density=True)
plt.xlabel('Event rate (events/second)')
plt.ylabel('Density')
plt.title('Distribution of neuronal event rates')
plt.legend()
plt.savefig('event_rate_distribution.png')
plt.close()

print(f"\nMean event rates:")
print(f"Spontaneous: {np.mean(spont_rates):.3f} events/sec")
print(f"Task: {np.mean(task_rates):.3f} events/sec")

# %%
# Look at temporal patterns - compute pairwise correlations between neurons
def compute_correlations(fluorescence):
    """Compute pairwise correlations between all neurons"""
    return np.corrcoef(fluorescence.T)

spont_fluor = nwb_spont.processing["ophys"]["Fluorescence"]["RoiResponseSeries"].data[:]
task_fluor = nwb_task.processing["ophys"]["Fluorescence"]["RoiResponseSeries"].data[:]

spont_corr = compute_correlations(spont_fluor)
task_corr = compute_correlations(task_fluor)

# Plot correlation matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

im1 = ax1.imshow(spont_corr, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax1.set_title('Spontaneous correlations')
plt.colorbar(im1, ax=ax1)

im2 = ax2.imshow(task_corr, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
ax2.set_title('Task correlations')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('correlation_matrices.png')
plt.close()

# %%
# Compare the overall correlation structure
mean_spont_corr = np.mean(np.abs(spont_corr[np.triu_indices_from(spont_corr, k=1)]))
mean_task_corr = np.mean(np.abs(task_corr[np.triu_indices_from(task_corr, k=1)]))

print(f"\nMean absolute correlations:")
print(f"Spontaneous: {mean_spont_corr:.3f}")
print(f"Task: {mean_task_corr:.3f}")

print("\nGenerated plots:")
print("1. event_rate_distribution.png - Distribution of event rates across neurons in both conditions")
print("2. correlation_matrices.png - Pairwise correlations between neurons in both conditions")
