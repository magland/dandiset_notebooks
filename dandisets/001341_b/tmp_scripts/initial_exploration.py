import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt

# Load the first subject's data
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001341/assets/e3d421c3-13bf-4b2e-b0f9-b6357a885de9/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print("\nDataset Overview:")
print(f"Subject: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")
print(f"Genotype: {nwb.subject.genotype}")

print("\nNeural Data Summary:")
units = nwb.units
n_units = len(units['id'].data[:])
print(f"Number of units: {n_units}")

# Get unit types
unit_types = units['unit_type'].data[:]
exc_count = sum(t == 'excitatory' for t in unit_types)
inh_count = sum(t == 'inhibitory' for t in unit_types)
print(f"Excitatory units: {exc_count}")
print(f"Inhibitory units: {inh_count}")

# Get layer distribution
layers = units['layer'].data[:]
unique_layers, layer_counts = np.unique(layers, return_counts=True)
print("\nUnits per layer:")
for layer, count in zip(unique_layers, layer_counts):
    print(f"{layer}: {count}")

# Trial information
trials = nwb.intervals['trials']
n_trials = len(trials['start_time'].data[:])
trial_durations = trials['stop_time'].data[:] - trials['start_time'].data[:]
mean_duration = np.mean(trial_durations)
std_duration = np.std(trial_durations)

print(f"\nBehavioral Trials:")
print(f"Number of trials: {n_trials}")
print(f"Mean trial duration: {mean_duration:.2f} seconds")
print(f"Std trial duration: {std_duration:.2f} seconds")

# Plot trial duration distribution
plt.figure(figsize=(10, 6))
plt.hist(trial_durations, bins=30)
plt.title('Distribution of Trial Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Count')
plt.savefig('001341_b/tmp_scripts/trial_durations.png')
plt.close()

# Get firing rates
firing_rates = units['firing_rate'].data[:]
plt.figure(figsize=(10, 6))
plt.hist(firing_rates, bins=30)
plt.title('Distribution of Unit Firing Rates')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.savefig('001341_b/tmp_scripts/firing_rates.png')
plt.close()

print("\nExample unit properties:")
print("Mean firing rate:", np.mean(firing_rates), "Hz")
print("Median firing rate:", np.median(firing_rates), "Hz")
print("Min firing rate:", np.min(firing_rates), "Hz") 
print("Max firing rate:", np.max(firing_rates), "Hz")
