# %% [markdown]
# # Activity Analysis for Dandiset 001174
#
# This script analyzes patterns of neuronal activity in the calcium imaging data:
# 1. Summarize basic activity statistics for each neuron
# 2. Look for correlated activity between neurons
# 3. Examine temporal patterns in the calcium events

# %%
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# %%
# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001174/assets/17963d0d-362d-40a3-aa7f-645c719f3f4a/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get fluorescence and events data
fluorescence = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries"].data[:]
events = nwb.processing["ophys"]["EventAmplitude"].data[:]
sampling_rate = nwb.acquisition["OnePhotonSeries"].rate

print(f"Recording duration: {len(fluorescence)/sampling_rate:.1f} seconds")
print(f"Number of cells: {fluorescence.shape[1]}")

# %% [markdown]
# ## Basic Activity Statistics

# %%
# Calculate statistics for each neuron
def get_neuron_stats(fluor, events, rate):
    n_neurons = fluor.shape[1]
    stats_dict = {
        'mean_fluorescence': [],
        'std_fluorescence': [],
        'event_rate': [],
        'mean_event_amplitude': []
    }

    for i in range(n_neurons):
        f = fluor[:, i]
        e = events[:, i]
        event_times = np.where(e > 0)[0]

        stats_dict['mean_fluorescence'].append(np.mean(f))
        stats_dict['std_fluorescence'].append(np.std(f))
        stats_dict['event_rate'].append(len(event_times) / (len(f) / rate))
        if len(event_times) > 0:
            stats_dict['mean_event_amplitude'].append(np.mean(e[event_times]))
        else:
            stats_dict['mean_event_amplitude'].append(0)

    return stats_dict

neuron_stats = get_neuron_stats(fluorescence, events, sampling_rate)

# Plot summary statistics
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Summary Statistics for Each Neuron')

axs[0,0].bar(range(len(neuron_stats['mean_fluorescence'])), neuron_stats['mean_fluorescence'])
axs[0,0].set_title('Mean Fluorescence')
axs[0,0].set_xlabel('Neuron ID')

axs[0,1].bar(range(len(neuron_stats['std_fluorescence'])), neuron_stats['std_fluorescence'])
axs[0,1].set_title('Fluorescence Std Dev')
axs[0,1].set_xlabel('Neuron ID')

axs[1,0].bar(range(len(neuron_stats['event_rate'])), neuron_stats['event_rate'])
axs[1,0].set_title('Event Rate (events/sec)')
axs[1,0].set_xlabel('Neuron ID')

axs[1,1].bar(range(len(neuron_stats['mean_event_amplitude'])), neuron_stats['mean_event_amplitude'])
axs[1,1].set_title('Mean Event Amplitude')
axs[1,1].set_xlabel('Neuron ID')

plt.tight_layout()
plt.savefig('neuron_stats.png')
plt.close()

# %% [markdown]
# ## Cross-correlations between neurons

# %%
# Calculate correlation matrix
corr_matrix = np.corrcoef(fluorescence.T)

# Plot correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
plt.title('Neuron-to-Neuron Correlation Matrix')
plt.xlabel('Neuron ID')
plt.ylabel('Neuron ID')
plt.savefig('correlation_matrix.png')
plt.close()

# %% [markdown]
# ## Temporal patterns of events

# %%
# Calculate inter-event intervals for each neuron
def get_inter_event_intervals(events, rate):
    event_times = np.where(events > 0)[0]
    if len(event_times) < 2:
        return np.array([])
    return np.diff(event_times) / rate  # Convert to seconds

# Plot inter-event interval distributions
plt.figure(figsize=(12, 4))
for i in range(events.shape[1]):
    intervals = get_inter_event_intervals(events[:, i], sampling_rate)
    if len(intervals) > 0:
        plt.subplot(1, events.shape[1], i+1)
        plt.hist(intervals, bins=20, density=True)
        plt.title(f'Neuron {i+1}')
        plt.xlabel('Inter-event interval (s)')
        if i == 0:
            plt.ylabel('Density')

plt.tight_layout()
plt.savefig('inter_event_intervals.png')
plt.close()

print("\nActivity Analysis Summary:")
print("-------------------------")
mean_event_rate = np.mean(neuron_stats['event_rate'])
mean_amplitude = np.mean(neuron_stats['mean_event_amplitude'])
print(f"Average event rate across neurons: {mean_event_rate:.2f} events/sec")
print(f"Average event amplitude: {mean_amplitude:.2f}")

# Calculate proportion of correlated pairs
sig_threshold = 0.5  # Correlation coefficient threshold
n_neurons = corr_matrix.shape[0]
n_pairs = (n_neurons * (n_neurons - 1)) // 2  # Number of unique pairs
sig_corr_pairs = np.sum(np.abs(corr_matrix[np.triu_indices(n_neurons, k=1)]) > sig_threshold)
print(f"Proportion of strongly correlated neuron pairs (|r| > {sig_threshold}): {sig_corr_pairs/n_pairs:.1%}")
