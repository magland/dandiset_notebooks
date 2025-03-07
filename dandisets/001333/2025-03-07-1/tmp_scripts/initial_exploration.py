"""
Initial exploration of PESD dataset (DANDI:001333)
This script aims to:
1. Load and examine the basic structure of the dataset
2. Plot example LFP signals
3. Perform spectral analysis to examine beta band activity (13-30 Hz)
4. Visualize electrode information
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001333/assets/5409700b-e080-44e6-a6db-1d3e8890cd6c/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic information
print("Dataset Information:")
print(f"Description: {nwb.session_description}")
print(f"Lab: {nwb.lab}")
print(f"Subject: {nwb.subject.subject_id}")
print(f"Keywords: {', '.join(nwb.keywords)}")
print("\nExperiment Description:")
print(nwb.experiment_description)

# Get LFP data
lfp = nwb.processing["ecephys"]["LFP"]["LFP"]
sampling_rate = lfp.rate
print(f"\nLFP sampling rate: {sampling_rate} Hz")

# Get first 5 seconds of data for visualization
n_seconds = 5
n_samples = int(n_seconds * sampling_rate)
data = lfp.data[:n_samples]
time = np.arange(n_samples) / sampling_rate

# Plot LFP signals
plt.figure(figsize=(12, 6))
plt.plot(time, data)
plt.xlabel('Time (s)')
plt.ylabel('LFP Amplitude')
plt.title('LFP Signal - First 5 seconds')
plt.grid(True)
plt.savefig('tmp_scripts/lfp_signal.png')
plt.close()

# Spectral analysis
# Use Welch's method to compute power spectral density
frequencies, psd = signal.welch(data, fs=sampling_rate, nperseg=1024)

# Plot power spectrum with beta band highlighted
plt.figure(figsize=(10, 6))
plt.semilogy(frequencies, psd)
plt.fill_between(frequencies, psd, where=(frequencies >= 13) & (frequencies <= 30),
                alpha=0.3, label='Beta band (13-30 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('LFP Power Spectrum')
plt.grid(True)
plt.legend()
plt.savefig('tmp_scripts/power_spectrum.png')
plt.close()

# Get electrode information
electrodes = nwb.ec_electrodes
locations = electrodes["location"].data[:]
labels = electrodes["label"].data[:]

print("\nElectrode Information:")
print("----------------------")
for i, (loc, label) in enumerate(zip(locations, labels)):
    print(f"Electrode {i+1}: Location={loc}, Label={label}")
