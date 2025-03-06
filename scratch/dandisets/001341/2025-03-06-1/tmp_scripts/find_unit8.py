import pynwb
import lindi
import numpy as np

# Load the NWB file
print('Loading NWB file...')
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/001341/assets/e3d421c3-13bf-4b2e-b0f9-b6357a885de9/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Find index for Unit 8
unit_ids = nwb.units['id'].data[:]
unit8_index = np.where(unit_ids == 8)[0][0]
print(f"\nUnit ID 8 is at index: {unit8_index}")

# Print some info about this unit
print(f"\nUnit 8 characteristics:")
print(f"Firing rate: {nwb.units['firing_rate'].data[unit8_index]:.2f} Hz")
print(f"SNR: {nwb.units['snr'].data[unit8_index]:.2f}")
print(f"Layer: {nwb.units['layer'].data[unit8_index]}")
print(f"Depth: {nwb.units['depth'].data[unit8_index]} µm")
