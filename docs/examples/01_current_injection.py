# -*- coding: utf-8 -*-

"""
# Fit injected current

Super simple
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import nemos as nmo
import pynapple as nap
import utils
from typing import Optional

# load the data
(trial_interval_set,
 all_injected_current,
 response,
 all_spike_times,
 sweep_metadata) = utils.data.load_to_pynapple("/home/billbrod/Downloads/data/cell_types/specimen_478498617/ephys.nwb")
all_spike_times.set_info(label=np.array(['neuron 1']))

# pick the trial
stim_type = "Noise 1"
sweep_num = 50
bin_size = .001

# grab data from that trial
trial_index = sweep_metadata[stim_type][sweep_num]["trial_index"]
trial_interval = trial_interval_set[stim_type][trial_index: trial_index+1]
# there's small amount of stimulation at beginning that we want to ignore
trial_interval.start += .1
# make sure this is a tsdframe
all_injected_current = nap.TsdFrame(all_injected_current.t, all_injected_current.d, columns=['injected_current'])
# plot the trial
binned_current = all_injected_current.bin_average(bin_size=bin_size, ep=trial_interval) * 10**12
spikes = all_spike_times.restrict(trial_interval)

def plot_current_injection(current: nap.TsdFrame, spikes: nap.TsGroup,
                           predicted_firing_rate: Optional[nap.TsdFrame] = None,
                           ax=None, figsize=(12, 4), title='',
                           spike_colors=['k', 'r'],
                           add_legend=False) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    artists = []
    ax.set_title(title)
    artists.extend(ax.plot(np.squeeze(current), label="injected current"))
    for i, idx in enumerate(spikes):
        lbl = spikes.get_info('label')[idx]
        y_min = -0.1*i*np.max(current)
        artists.append(ax.vlines(spikes[idx].t, y_min, y_min+.1*np.max(current), spike_colors[i],
                                 label=f"{lbl} spikes"))
    if predicted_firing_rate is not None:
        ax2 = ax.twinx()
        artists.extend(ax2.plot(predicted_firing_rate.t, predicted_firing_rate, 'r--', alpha=.5,
                                label="Predicted firing rate", zorder=0))
        utils.plotting.set_two_y_axes_zeros_equal(ax, ax2)
        ax2.set_ylabel('Predicted firing rate [Hz]')
    if add_legend:
        ax.legend(artists, [a.get_label() for a in artists])
    ax.set_xlabel('time [sec]', fontsize=10)
    ax.set_ylabel('current [pA]', fontsize=10)
    ax.set_xlim((current.t[0], current.t[-1]))
    return fig

# %%
# Look at trial

plot_current_injection(binned_current, spikes, title=f"Stimulus {stim_type}, trial {sweep_num}")

# %%
# We picked a trial with some simple current injection, and we can see that
# injecting current leads to increased firing rate.
#
# Initial impulse then is that this is all we need! let's fit a simple model
# that only takes the injected current as input. Using this model is equivalent
# to saying "the only thing that influences the firing rate of the neuron is
# whether it receives a current injection." (As neuroscientists, we know this
# isn't true, but it does look reasonable from the data above! We'll build in
# complications later.)

# For nemo's GLM, our input must be 3d: (num_time_pts, num_neurons,
# num_features). This will be 2d, so let's add an extra dimension
binned_current = np.expand_dims(binned_current, -1)
# spikes needs to be 2d: (num_time_pts, num_neurons), which this already is.
binned_spikes = spikes.count(bin_size=bin_size)

# To start, we will do the unregularized GLM (see XXX for details on
# regularization)
glm = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized())
glm.fit(binned_current, binned_spikes)

# Now that we've fit our data, let's see simulate to see what it looks like
pred_spikes, pred_fr = glm.simulate(jax.random.PRNGKey(4), binned_current)
# convert units from spikes/bin to spikes/sec
pred_fr /= bin_size
pred_fr = nap.TsdFrame(binned_current.t, np.array(pred_fr))
# get the times of these simulated spikes
pred_spike_times = binned_spikes[np.where(pred_spikes)].times()
spikes_dict = dict(spikes)
spikes_dict[2] = np.asarray(pred_spike_times)
spikes = nap.TsGroup(spikes_dict)
spikes.set_info(label=np.array(['neuron', 'GLM predicted']))

plot_current_injection(binned_current, spikes, pred_fr,
                       title=f"Stimulus {stim_type}, trial {sweep_num}")

# %%
#
# Let's look at the periods when we're injecting current, when we see spikes:

current_intervals = np.squeeze(binned_current).threshold(0).time_support
fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'wspace': .4})
for i, ax in zip(range(len(current_intervals)), axes.flatten()):
    interval = current_intervals.loc[[i]]
    current_spikes = spikes.restrict(interval)
    spike_count = np.squeeze(current_spikes.count().values)
    plot_current_injection(binned_current.restrict(interval),
                           current_spikes,
                           pred_fr.restrict(interval),
                           ax=ax, add_legend=False,
                           title=f"{spike_count[0]} real spikes, {spike_count[1]} simulated spikes")

# %%
# What do we see?
#
# - simulated spike in first current interval
# - spikes spaced weird in second interval
# - way too many spikes in third
#
# why might this be? if you're really observant, you may have noticed that
# firing rate looks like it's basically a shifted copy of the input current.
# And it is!
#
# Remember tutorial 0: GLM models the log firing rate as a linear combination
# of inputs. in this case, the firing rate at time t is only the function of
# the input current *at that time*. we can see that by plotting the tuning
# curve between firing rate and input current:

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(nap.compute_1d_tuning_curves(spikes[[1]], binned_current, 15), 'k', label="Observed")
ax.plot(nap.compute_1d_tuning_curves_continuous(pred_fr, np.squeeze(binned_current), 15), 'r--', label="Predicted")
ax.legend()
ax.set(ylabel="Firing rate [Hz]", xlabel="Input current [pA]")

# %%
#
# Predicted firing rate is a smooth exponential of the input current, while
# observed is not just noisier, but falls off rapidly at high levels of current.
#
# We can also recover this tuning directly from the GLM object:

print(glm.coef_.shape, glm.intercept_.shape)
current = np.linspace(0, 140)
fr = np.exp(glm.coef_ * current + glm.intercept_).squeeze()
# the firing rate calcualted above is in spikes per bin, so convert that to Hz
fr /= bin_size
ax.plot(nap.compute_1d_tuning_curves(spikes[[1]], binned_current, 15), 'k', label="Observed")
ax.plot(nap.compute_1d_tuning_curves_continuous(pred_fr, np.squeeze(binned_current), 15), 'r--', label="Predicted")
ax.set(ylabel="Firing rate [Hz]", xlabel="Input current [pA]")
ax.plot(current, fr, 'r', label="Predicted from GLM directly")
ax.legend()
fig
# %%
# Now:
#
# - compare number of spikes within each current interval
#
# - plot each current interval, show how rate is linear with input and so misses some effects
#
# - show tuning curves, point out that they differ at high input levels


# point out this is two parameters and is only using input at time t, say we
# can get a little fancier by adding history. try with and without bases?
#
# then add self-excitatin
