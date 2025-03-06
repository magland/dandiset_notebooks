import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load the NWB file
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get trials and units data
trials = nwb.intervals["trials"]
trial_starts = trials["start_time"].data[:]
units = nwb.units
unit_ids = units["id"].data[:]
cell_types = units["celltype_label"].data[:]

# Parameters for PSTH
window = [-0.5, 2.0]  # Time window around trial start
bin_size = 0.05  # 50ms bins
bins = np.arange(window[0], window[1] + bin_size, bin_size)
bin_centers = bins[:-1] + bin_size/2

# Functions to compute PSTH
def compute_psth(spike_times, trial_times, window, bin_size):
    psth = np.zeros((len(trial_times), len(bins)-1))
    for i, trial_time in enumerate(trial_times):
        aligned_spikes = spike_times - trial_time
        mask = (aligned_spikes >= window[0]) & (aligned_spikes < window[1])
        if np.sum(mask) > 0:
            psth[i], _ = np.histogram(aligned_spikes[mask], bins=bins)
    return np.mean(psth, axis=0) / bin_size  # Convert to Hz

# Compute average PSTH for RSUs and FSUs
rsu_psth = np.zeros((np.sum(cell_types == 1), len(bins)-1))
fsu_psth = np.zeros((np.sum(cell_types == 2), len(bins)-1))
rsu_idx = 0
fsu_idx = 0

for i in range(len(unit_ids)):
    spike_times = units["spike_times"][i][:]
    if cell_types[i] == 1:  # RSU
        rsu_psth[rsu_idx] = compute_psth(spike_times, trial_starts, window, bin_size)
        rsu_idx += 1
    else:  # FSU
        fsu_psth[fsu_idx] = compute_psth(spike_times, trial_starts, window, bin_size)
        fsu_idx += 1

# Create figure
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig)

# Plot 1: Average PSTH for RSUs with confidence intervals
ax1 = fig.add_subplot(gs[0, :])
mean_rsu = np.mean(rsu_psth, axis=0)
sem_rsu = np.std(rsu_psth, axis=0) / np.sqrt(rsu_psth.shape[0])
ax1.fill_between(bin_centers, mean_rsu - sem_rsu, mean_rsu + sem_rsu, alpha=0.3, color='blue')
ax1.plot(bin_centers, mean_rsu, color='blue', label='RSU (n=32)')
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
ax1.axvline(x=2.2, color='k', linestyle='--', alpha=0.5)  # Approx. trial duration
ax1.set_title('Regular Spiking Units (RSU) Response')
ax1.set_xlabel('Time from trial start (s)')
ax1.set_ylabel('Firing Rate (Hz)')
ax1.legend()

# Plot 2: Average PSTH for FSUs with confidence intervals
ax2 = fig.add_subplot(gs[1, :])
mean_fsu = np.mean(fsu_psth, axis=0)
sem_fsu = np.std(fsu_psth, axis=0) / np.sqrt(fsu_psth.shape[0])
ax2.fill_between(bin_centers, mean_fsu - sem_fsu, mean_fsu + sem_fsu, alpha=0.3, color='red')
ax2.plot(bin_centers, mean_fsu, color='red', label='FSU (n=32)')
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
ax2.axvline(x=2.2, color='k', linestyle='--', alpha=0.5)  # Approx. trial duration
ax2.set_title('Fast Spiking Units (FSU) Response')
ax2.set_xlabel('Time from trial start (s)')
ax2.set_ylabel('Firing Rate (Hz)')
ax2.legend()

# Calculate some statistics
pre_window = [-0.5, 0]
stim_window = [0, 2.2]

def get_mean_rates(psth, window):
    start_idx = int((window[0] - bins[0]) / bin_size)
    end_idx = int((window[1] - bins[0]) / bin_size)
    return np.mean(psth[:, start_idx:end_idx], axis=1)

rsu_pre = get_mean_rates(rsu_psth, pre_window)
rsu_stim = get_mean_rates(rsu_psth, stim_window)
fsu_pre = get_mean_rates(fsu_psth, pre_window)
fsu_stim = get_mean_rates(fsu_psth, stim_window)

print("\nRSU Statistics:")
print(f"Pre-stim rate: {np.mean(rsu_pre):.2f} ± {np.std(rsu_pre)/np.sqrt(len(rsu_pre)):.2f} Hz")
print(f"During-stim rate: {np.mean(rsu_stim):.2f} ± {np.std(rsu_stim)/np.sqrt(len(rsu_stim)):.2f} Hz")
print(f"Change: {((np.mean(rsu_stim) - np.mean(rsu_pre))/np.mean(rsu_pre)*100):.1f}%")

print("\nFSU Statistics:")
print(f"Pre-stim rate: {np.mean(fsu_pre):.2f} ± {np.std(fsu_pre)/np.sqrt(len(fsu_pre)):.2f} Hz")
print(f"During-stim rate: {np.mean(fsu_stim):.2f} ± {np.std(fsu_stim)/np.sqrt(len(fsu_stim)):.2f} Hz")
print(f"Change: {((np.mean(fsu_stim) - np.mean(fsu_pre))/np.mean(fsu_pre)*100):.1f}%")

plt.tight_layout()
plt.savefig('tmp_scripts/neural_responses.png', bbox_inches='tight')
plt.close()
