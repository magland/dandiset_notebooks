# %% [markdown]
# # Improved task-related analysis of calcium imaging data
# This script compares neural activity between spontaneous and task conditions,
# with clearer visualizations and analyses.

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
# Plot all ROI footprints in one image for both sessions
def plot_all_footprints(segmentation, title):
    masks = segmentation["image_mask"].data[:]
    # Create a composite image where each ROI has a different color
    composite = np.zeros(masks[0].shape + (3,))
    for i, mask in enumerate(masks):
        # Cycle through some distinct colors
        color = plt.cm.tab20(i % 20)[:3]  # Get RGB values from colormap
        for c in range(3):
            composite[:,:,c] += mask * color[c]
    
    # Normalize to [0,1]
    composite /= composite.max()
    
    plt.figure(figsize=(8, 8))
    plt.imshow(composite)
    plt.title(f'{title}\n({len(masks)} ROIs)')
    plt.axis('off')
    return composite

spont_seg = nwb_spont.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]
task_seg = nwb_task.processing["ophys"]["ImageSegmentation"]["PlaneSegmentation"]

plot_all_footprints(spont_seg, "ROI locations - Spontaneous")
plt.savefig('all_footprints_spontaneous.png')
plt.close()

plot_all_footprints(task_seg, "ROI locations - Task")
plt.savefig('all_footprints_task.png')
plt.close()

# %%
# Analyze event statistics
def analyze_events(events, sampling_rate):
    """Compute various event statistics"""
    duration = events.shape[0] / sampling_rate
    n_neurons = events.shape[1]
    
    # Count events per neuron
    event_counts = np.sum(events > 0, axis=0)
    event_rates = event_counts / duration
    
    # Count total events and active neurons
    total_events = np.sum(event_counts)
    active_neurons = np.sum(event_counts > 0)
    
    return {
        'total_events': total_events,
        'active_neurons': active_neurons,
        'total_neurons': n_neurons,
        'duration': duration,
        'event_counts': event_counts,
        'event_rates': event_rates,
        'mean_rate': np.mean(event_rates),
        'median_rate': np.median(event_rates)
    }

# Analyze both conditions
spont_events = nwb_spont.processing["ophys"]["EventAmplitude"].data[:]
task_events = nwb_task.processing["ophys"]["EventAmplitude"].data[:]

spont_stats = analyze_events(spont_events, nwb_spont.processing["ophys"]["EventAmplitude"].rate)
task_stats = analyze_events(task_events, nwb_task.processing["ophys"]["EventAmplitude"].rate)

# Print summary statistics
print("\nEvent Statistics:")
print(f"{'':20} {'Spontaneous':15} {'Task':15}")
print("-" * 50)
print(f"Duration:         {spont_stats['duration']:.1f} sec      {task_stats['duration']:.1f} sec")
print(f"Total events:     {spont_stats['total_events']:7d}         {task_stats['total_events']:7d}")
print(f"Active neurons:   {spont_stats['active_neurons']:3d}/{spont_stats['total_neurons']:<3d}         {task_stats['active_neurons']:3d}/{task_stats['total_neurons']:<3d}")
print(f"Mean rate:        {spont_stats['mean_rate']:.3f} Hz      {task_stats['mean_rate']:.3f} Hz")
print(f"Median rate:      {spont_stats['median_rate']:.3f} Hz      {task_stats['median_rate']:.3f} Hz")

# %%
# Plot event count distribution as a bar plot
plt.figure(figsize=(10, 6))
bins = np.arange(0, max(spont_stats['event_counts'].max(), task_stats['event_counts'].max()) + 10, 5)
plt.hist([spont_stats['event_counts'], task_stats['event_counts']], 
         bins=bins, label=['Spontaneous', 'Task'],
         alpha=0.7)
plt.xlabel('Number of events per neuron')
plt.ylabel('Number of neurons')
plt.title('Distribution of event counts')
plt.legend()
plt.savefig('event_count_comparison.png')
plt.close()

print("\nGenerated plots:")
print("1. all_footprints_spontaneous.png - Shows all ROI locations in the spontaneous session")
print("2. all_footprints_task.png - Shows all ROI locations in the task session")
print("3. event_count_comparison.png - Compares the number of events per neuron between conditions")
