import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# BH497 recordings with different PRFs
files = [
    {
        'url': "https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json",
        'prf': 30
    },
    {
        'url': "https://lindi.neurosift.org/dandi/dandisets/000945/assets/526c681d-0c50-44e1-92be-9c0134c71fd8/nwb.lindi.json",
        'prf': 1500
    },
    {
        'url': "https://lindi.neurosift.org/dandi/dandisets/000945/assets/f88a9bec-23d6-4444-8b97-8083e45057c9/nwb.lindi.json",
        'prf': 300
    },
    {
        'url': "https://lindi.neurosift.org/dandi/dandisets/000945/assets/a7549e3f-9b14-432a-be65-adb5f6811343/nwb.lindi.json",
        'prf': 3000
    },
    {
        'url': "https://lindi.neurosift.org/dandi/dandisets/000945/assets/02151b40-5064-4ba1-a5b7-d0473ff09262/nwb.lindi.json",
        'prf': 4500
    }
]

# Parameters for PSTH
window = [-0.5, 2.0]  # Time window around trial start
bin_size = 0.05  # 50ms bins
bins = np.arange(window[0], window[1] + bin_size, bin_size)
bin_centers = bins[:-1] + bin_size/2

# Function to compute PSTH
def compute_psth(spike_times, trial_times, window, bin_size):
    psth = np.zeros((len(trial_times), len(bins)-1))
    for i, trial_time in enumerate(trial_times):
        aligned_spikes = spike_times - trial_time
        mask = (aligned_spikes >= window[0]) & (aligned_spikes < window[1])
        if np.sum(mask) > 0:
            psth[i], _ = np.histogram(aligned_spikes[mask], bins=bins)
    return np.mean(psth, axis=0) / bin_size  # Convert to Hz

# Create figure
fig = plt.figure(figsize=(15, 12))
gs = GridSpec(3, 2, figure=fig)

# Store rate changes for summary
prf_changes = {'RSU': [], 'FSU': []}
prfs = []

# Process each file
for idx, file_info in enumerate(files):
    print(f"\nProcessing PRF {file_info['prf']} Hz...")
    f = lindi.LindiH5pyFile.from_lindi_file(file_info['url'])
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
    
    # Get trials and units data
    trials = nwb.intervals["trials"]
    trial_starts = trials["start_time"].data[:]
    units = nwb.units
    cell_types = units["celltype_label"].data[:]
    
    # Compute PSTH for each cell type
    rsu_psth = []
    fsu_psth = []
    
    for i in range(len(units["id"].data)):
        spike_times = units["spike_times"][i][:]
        psth = compute_psth(spike_times, trial_starts, window, bin_size)
        if cell_types[i] == 1:  # RSU
            rsu_psth.append(psth)
        else:  # FSU
            fsu_psth.append(psth)
    
    rsu_psth = np.array(rsu_psth)
    fsu_psth = np.array(fsu_psth)
    
    # Calculate rate changes
    pre_window = [-0.5, 0]
    stim_window = [0, 2.2]
    pre_idx = slice(int((pre_window[0] - window[0])/bin_size), 
                    int((pre_window[1] - window[0])/bin_size))
    stim_idx = slice(int((stim_window[0] - window[0])/bin_size), 
                     int((stim_window[1] - window[0])/bin_size))
    
    rsu_change = (np.mean(rsu_psth[:, stim_idx], axis=1) - 
                  np.mean(rsu_psth[:, pre_idx], axis=1)) / np.mean(rsu_psth[:, pre_idx], axis=1) * 100
    fsu_change = (np.mean(fsu_psth[:, stim_idx], axis=1) - 
                  np.mean(fsu_psth[:, pre_idx], axis=1)) / np.mean(fsu_psth[:, pre_idx], axis=1) * 100
    
    prf_changes['RSU'].append(np.mean(rsu_change))
    prf_changes['FSU'].append(np.mean(fsu_change))
    prfs.append(file_info['prf'])
    
    # Plot PSTHs
    ax = fig.add_subplot(gs[idx//2, idx%2])
    
    # Plot RSUs
    mean_rsu = np.mean(rsu_psth, axis=0)
    sem_rsu = np.std(rsu_psth, axis=0) / np.sqrt(rsu_psth.shape[0])
    ax.fill_between(bin_centers, mean_rsu - sem_rsu, mean_rsu + sem_rsu, 
                   alpha=0.3, color='blue')
    ax.plot(bin_centers, mean_rsu, color='blue', label=f'RSU (n={len(rsu_psth)})')
    
    # Plot FSUs
    mean_fsu = np.mean(fsu_psth, axis=0)
    sem_fsu = np.std(fsu_psth, axis=0) / np.sqrt(fsu_psth.shape[0])
    ax.fill_between(bin_centers, mean_fsu - sem_fsu, mean_fsu + sem_fsu, 
                   alpha=0.3, color='red')
    ax.plot(bin_centers, mean_fsu, color='red', label=f'FSU (n={len(fsu_psth)})')
    
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.axvline(x=2.2, color='k', linestyle='--', alpha=0.5)
    ax.set_title(f'PRF: {file_info["prf"]} Hz')
    ax.set_xlabel('Time from trial start (s)')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.legend()

# Plot summary of rate changes
ax = fig.add_subplot(gs[2, :])
ax.plot(prfs, prf_changes['RSU'], 'bo-', label='RSU')
ax.plot(prfs, prf_changes['FSU'], 'ro-', label='FSU')
ax.set_xscale('log')
ax.set_xlabel('Pulse Repetition Frequency (Hz)')
ax.set_ylabel('Mean Firing Rate Change (%)')
ax.set_title('Response Magnitude vs PRF')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig('tmp_scripts/prf_comparison.png', bbox_inches='tight', dpi=150)
plt.close()

# Print summary statistics
for prf, rsu_change, fsu_change in zip(prfs, prf_changes['RSU'], prf_changes['FSU']):
    print(f"\nPRF {prf} Hz:")
    print(f"RSU change: {rsu_change:.1f}%")
    print(f"FSU change: {fsu_change:.1f}%")
