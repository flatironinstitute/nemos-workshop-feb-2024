"""
# GLM in practice: Analysis of An Intracellular Recording
After understanding the basics of GLMs, let's see how we can apply these concepts in a real-world example.
In this tutorial, we will load, explore, and model an intracellular recording from the Allen Cell Type.

### Dataset

The dataset consists of whole cell current clamp recordings of individual neurons.
The neuron is stimulated with an injected current following various protocols: short and long square waves,
ramps, white noise, impulses.

The variables we will be working with are:

 - **trial_interval_set**: A dictionary containing the start and end time of each trial in seconds as a
 `pynapple.IntervalSet` object. Dictionary keys refer to the stimulation protocol used for the trial.
 - **injected current**: The trace of the injected current as a `pynapple.Tsd` object.
 - **response**: The membrane voltage of the neuron in response to the current as a `pynapple.Tsd` object.
 - **spike_times**: The spike times of the neuron as a `pynapple.TsGroup` object.
 - **sweep_metadata**: A nested dictionary containing information about the trial. The first level key will be
 the stimulation protocol, the second the sweep number, the third the metadata label.
"""

from utils.load_allen_utils import load_to_pynapple
import pynapple as nap

# load the data
(trial_interval_set,
 injected_current,
 response,
 spike_times,
 sweep_metadata) = load_to_pynapple("/Users/ebalzani/Code/fit_from_allensdk/cell_types/specimen_478498617/ephys.nwb")

# Plot the TsGroup containing the spike times of the neuron.
# Neuron index and the firing rate will be visualized
print(f"Spike times: {type(spike_times)}")
print(f"{spike_times}\n")

# Visualize the interval set for some stimulation protocol
print(f'Noise stimuli trials: {type(trial_interval_set["Noise 1"])}')
print(trial_interval_set["Noise 1"], "\n")

# Injected current and responses are Tsd with the same number of samples
print(f"Injected current and response: {type(injected_current)}")
print("\n\nCurrent:\n", injected_current)
print("\n\nResponse:\n", response, "\n")

# Visualize the metadata
stim_type = "Noise 1"
sweep_num = 50
print(f"Metadata\n Stimulation protocol: '{stim_type}' - Trial: {sweep_num}")
for key, values in sweep_metadata[stim_type][sweep_num].items():
    print(f"\t - {key}: {values}")


# %%
# ### Visualizing the Neural Responses
# Before digging into the GLM setup we should explore our data. As a first step,
# we can select a trial from the white noise stimulation protocol and visualize the activity and to the
# injected current.
# We will make use of the pynapple "bin_average" method to up-sample the current time series from
# a 200KHz to a more manageable 1KHz.
#
# #### A Single Trial

import matplotlib.pyplot as plt
import numpy as np

# get the IntervalSet from the metadata "trial_index";
trial_index = sweep_metadata[stim_type][sweep_num]["trial_index"]
trial_interval = trial_interval_set[stim_type][trial_index: trial_index+1]
# up-sample the current and convert to from A to pA
binned_current = injected_current.bin_average(bin_size=0.001, ep=trial_interval) * 10**12

# plot the trial
plt.figure()
plt.title(f"Sweep {sweep_num} - {stim_type}")
plt.plot(binned_current, label="injected current")
plt.vlines(spike_times[1].restrict(trial_interval).t, 0, 0.1 * np.max(binned_current), 'k', label="spikes")
plt.legend()
plt.xlabel('time [sec]', fontsize=10)
plt.ylabel('current [pA]', fontsize=10)
plt.tight_layout()

# %%
# #### The Whole Recording
# Next we can visualize the whole dataset to get a better sense of what stimulation protocols were
# applied, and how the neuron responded to the stimulation.

# up-sample and rescale to pA
binned_current = injected_current.bin_average(bin_size=0.001) * 10**12

fig, ax = plt.subplots(figsize=(10, 3.5))
plt.plot(binned_current, label="injected current")
plt.vlines(spike_times[1].t, 0, 0.1 * np.max(binned_current), 'k', label="spikes")
ylim = plt.ylim()
for start, end in spike_times.time_support.values:
    rect = plt.Rectangle(
        xy=(start, 0),
        width=end - start,
        height=ylim[1],
        alpha=0.2,
        facecolor="grey",
        label="trial" if start == 0 else None
    )
    ax.add_patch(rect)
plt.xlabel('time [sec]', fontsize=10)
plt.ylabel('current [pA]', fontsize=10)
plt.legend()
plt.tight_layout()

# %%
# #### Tuning Curve
# We can use pynapple to extract the tuning function of the neuron to the injected current during the
# noise stimulation protocol.

# compute and plot the tuning curve with pynapple
tuning_noise = nap.compute_1d_tuning_curves(
    spike_times,
    injected_current * 10**12,
    ep=trial_interval_set["Noise 1"],
    nb_bins=20,
    minmax=(0, 200)
)

plt.figure()
plt.title("tuning curve")
plt.plot(tuning_noise)
plt.ylabel("firing rate [Hz]")
plt.xlabel("current [pA]")

# %%
# ### Setting-up the Poisson GLM
#
# *Predict using the instantaneous injected current.*
#
# *Use the instantaneous injected current as a single scalar input; use nemos to learn the ML parameters (intercept and
# weight), check the model test-set fit accuracy.*
#
#
# ### Model current injection history
# *Use the past history of the injected current as a predictor. Show that this is equivalent to
# convolve the weights with the current; the convolution can be represented as matrix vector product;
# show that for fine time resolution and long history it over-fits (too many params); introduce the basis as a way to
# reduce the parameter-space dimension and smoothen the history effect. show if the basis increase the
# fit accuracy*
#
# ### A better parametrization
#
# ### Can we do better? Model Self-Excitation
