"""
Detailed analysis of beta band activity in the PESD dataset
This script:
1. Calculates beta band power across all electrodes
2. Implements ARV (Average Rectified Value) calculation as described in dataset
3. Compares power in different frequency bands
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt

def get_band_power(data, fs, band, window_sec=1):
    """Calculate average power in a specific frequency band"""
    nperseg = min(int(window_sec * fs), data.shape[0])
    frequencies, psd = signal.welch(data, fs=fs, nperseg=nperseg)
    band_mask = (frequencies >= band[0]) & (frequencies <= band[1])
    return np.mean(psd[band_mask])

def calculate_arv(data, fs, center_freq, bandwidth=8):
    """Calculate Average Rectified Value using Chebyshev bandpass filter"""
    nyquist = fs / 2
    low = (center_freq - bandwidth/2) / nyquist
    high = (center_freq + bandwidth/2) / nyquist
    b, a = signal.cheby1(4, 1, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    return np.mean(np.abs(filtered))

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001333/assets/5409700b-e080-44e6-a6db-1d3e8890cd6c/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get LFP data
lfp = nwb.processing["ecephys"]["LFP"]["LFP"]
sampling_rate = lfp.rate
electrodes = nwb.ec_electrodes
electrode_labels = electrodes["label"].data[:]

# Define frequency bands
bands = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# Calculate power in each band for 10-second segments
segment_duration = 10  # seconds
n_samples = int(segment_duration * sampling_rate)
data = lfp.data[:n_samples]  # First 10 seconds

# Plot power across frequency bands
powers = {band: [] for band in bands.keys()}
for band_name, band_range in bands.items():
    power = get_band_power(data, sampling_rate, band_range)
    powers[band_name].append(power)

plt.figure(figsize=(10, 6))
x = range(len(bands))
plt.bar(x, [powers[band][0] for band in bands.keys()])
plt.xlabel('Frequency Band')
plt.ylabel('Average Power')
plt.title('Power in Different Frequency Bands')
plt.xticks(x, list(bands.keys()))
plt.yscale('log')
plt.grid(True)
plt.savefig('tmp_scripts/band_powers.png')
plt.close()

# Calculate and plot spectrogram
f, t, Sxx = signal.spectrogram(data, fs=sampling_rate, nperseg=1024, noverlap=512)
plt.figure(figsize=(12, 6))
plt.pcolormesh(t, f, 10 * np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram')
plt.colorbar(label='Power/Frequency [dB/Hz]')
# Highlight beta band
plt.axhline(y=13, color='r', linestyle='--', alpha=0.5)
plt.axhline(y=30, color='r', linestyle='--', alpha=0.5)
plt.ylim(0, 100)  # Focus on relevant frequencies
plt.savefig('tmp_scripts/spectrogram.png')
plt.close()

# Calculate and plot ARV values over time
window_size = int(0.5 * sampling_rate)  # 500ms windows
n_windows = len(data) // window_size
arv_values = []
time_points = []

for i in range(n_windows):
    start_idx = i * window_size
    end_idx = start_idx + window_size
    window_data = data[start_idx:end_idx]
    arv = calculate_arv(window_data, sampling_rate, 21.5)  # Center of beta band
    arv_values.append(arv)
    time_points.append(i * 0.5)  # Convert to seconds

plt.figure(figsize=(12, 6))
plt.plot(time_points, arv_values)
plt.xlabel('Time (s)')
plt.ylabel('Beta ARV')
plt.title('Beta Band Average Rectified Value Over Time')
plt.grid(True)
plt.savefig('tmp_scripts/beta_arv_timecourse.png')
plt.close()

# Print summary statistics
print("\nBeta Band Analysis Summary:")
print("---------------------------")
print(f"Analysis duration: {segment_duration} seconds")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Number of samples analyzed: {len(data)}")

print("\nPower Distribution Across Frequency Bands:")
total_power = sum([powers[band][0] for band in bands.keys()])
for band, power_list in powers.items():
    relative_power = (power_list[0] / total_power) * 100
    print(f"{band.capitalize()}: {relative_power:.2f}%")

print("\nBeta Band ARV Statistics:")
print(f"Mean ARV: {np.mean(arv_values):.6f}")
print(f"Max ARV: {np.max(arv_values):.6f}")
print(f"Min ARV: {np.min(arv_values):.6f}")
print(f"ARV Standard Deviation: {np.std(arv_values):.6f}")
