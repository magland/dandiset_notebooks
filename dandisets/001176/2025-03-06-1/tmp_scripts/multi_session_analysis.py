import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

def load_and_analyze_session(url, region):
    """Load session and extract key metrics."""
    f = lindi.LindiH5pyFile.from_lindi_file(url)
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
    
    # Get fluorescence data
    fluorescence = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries1"].data[:]
    fluor_times = nwb.processing["ophys"]["Fluorescence"]["RoiResponseSeries1"].timestamps[:]
    
    # Calculate basic stats
    mean_fluor = np.mean(fluorescence)
    std_fluor = np.std(fluorescence)
    
    # Calculate power spectrum
    fs = 1 / np.mean(np.diff(fluor_times))  # Sampling frequency
    freqs, psd = welch(fluorescence.flatten(), fs=fs, nperseg=1024)
    
    # Find peak frequency
    peak_freq = freqs[np.argmax(psd)]
    
    return {
        'subject_id': nwb.subject.subject_id,
        'region': region,
        'mean_fluor': mean_fluor,
        'std_fluor': std_fluor,
        'peak_freq': peak_freq,
        'freqs': freqs,
        'psd': psd,
        'fluorescence': fluorescence,
        'times': fluor_times
    }

# Analyze one M1 and one V1 session
m1_url = "https://lindi.neurosift.org/dandi/dandisets/001176/assets/4550467f-b94d-406b-8e30-24dd6d4941c1/nwb.lindi.json"
v1_url = "https://lindi.neurosift.org/dandi/dandisets/001176/assets/be84b6ff-7016-4ed8-af63-aa0e07c02530/nwb.lindi.json"

m1_data = load_and_analyze_session(m1_url, 'M1')
v1_data = load_and_analyze_session(v1_url, 'V1')

# Plot example traces and power spectra
plt.figure(figsize=(15, 10))

# Plot M1 trace
plt.subplot(221)
t_start = 100  # Start at 100s to avoid any initial artifacts
t_window = 100  # Show 100s window
t_mask = (m1_data['times'] >= t_start) & (m1_data['times'] <= t_start + t_window)
plt.plot(m1_data['times'][t_mask], m1_data['fluorescence'][t_mask])
plt.title(f"M1 ACh Signal (Subject {m1_data['subject_id']})")
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence')

# Plot V1 trace
plt.subplot(222)
t_mask = (v1_data['times'] >= t_start) & (v1_data['times'] <= t_start + t_window)
plt.plot(v1_data['times'][t_mask], v1_data['fluorescence'][t_mask])
plt.title(f"V1 ACh Signal (Subject {v1_data['subject_id']})")
plt.xlabel('Time (s)')
plt.ylabel('Fluorescence')

# Plot M1 power spectrum
plt.subplot(223)
plt.semilogy(m1_data['freqs'], m1_data['psd'])
plt.title('M1 Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid(True)

# Plot V1 power spectrum
plt.subplot(224)
plt.semilogy(v1_data['freqs'], v1_data['psd'])
plt.title('V1 Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid(True)

plt.tight_layout()
plt.savefig("scratch/dandisets/001176/2025-03-06-1/tmp_scripts/region_comparison.png")
plt.close()

# Print summary statistics
print("\nM1 Summary:")
print(f"Mean fluorescence: {m1_data['mean_fluor']:.2f}")
print(f"Fluorescence std: {m1_data['std_fluor']:.2f}")
print(f"Peak frequency: {m1_data['peak_freq']:.2f} Hz")

print("\nV1 Summary:")
print(f"Mean fluorescence: {v1_data['mean_fluor']:.2f}")
print(f"Fluorescence std: {v1_data['std_fluor']:.2f}")
print(f"Peak frequency: {v1_data['peak_freq']:.2f} Hz")
