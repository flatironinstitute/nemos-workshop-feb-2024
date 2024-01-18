# -*- coding: utf-8 -*-

"""
# Fit injected current

Super simple (or not)
"""

import jax
import math
import os


import matplotlib.pyplot as plt
import nemos as nmo
import nemos.glm
import numpy as np
import pynapple as nap
import requests
import tqdm

# required for second order methods (BFGS, Newton-CG)
jax.config.update("jax_enable_x64", True)

# %%
# ## DATA STREAMING
# 
# Here we load the data from OSF. The data is a NWB file from the Allen Institute.
# blblalba say more
# Just run this cell

path = os.path.join(os.getcwd(), "allen_478498617.nwb")
if os.path.basename(path) not in os.listdir(os.getcwd()):
    r = requests.get(f"https://osf.io/um3bj/download", stream=True)
    block_size = 1024*1024
    with open(path, "wb") as f:
        for data in tqdm.tqdm(r.iter_content(block_size), unit="MB", unit_scale=True,
            total=math.ceil(int(r.headers.get("content-length", 0))//block_size)):
            f.write(data)


# %%
# ## PYNAPPLE
# The data have been copied to your local station.
# We are gonna open the NWB file with pynapple

data = nap.load_file(path)

# %%
# What does it look like?
print(data)

# %%
# With pynapple, you can quickly extract time series from the NWB files.

trial_interval_set = data["epochs"]
current = data["stimulus"] * 1e12
response = data["response"]
spikes = data["units"]

# %% 
# First let"s examine what trial_interval_set is.

print(trial_interval_set.keys())

# %%
# In this case, it's a dictionnary of IntervalSet.
# During the recording, multiple types of current were injected to the cell. For this tutorial, we will take "Noise 1".

noise_interval = trial_interval_set["Noise 1"]

print(noise_interval)

# %%
# As you can see there are 3 rows. Each contains the start and end (in seconds) of an epoch.
# To select only one epoch from an IntervalSet, use 2 square brackets.

noise_interval.loc[[0]]

# %%
# We can look at the current.

print(current)

# %%
# As you can see, it's an object called a Tsd (TimeSeriesData) with 2 columns. The first column indicates time and the second column is the current in Ampere.
# A key point of pynapple is that objects can interact. In this case, we want to restrict the current (Tsd) to the noise_interval epochs (IntervalSet). Notice how the timestamps are changing.
current.restrict(noise_interval)

# %%
# The third object we are interacting with is the TsGroup for a group of timestamps. This is typically, a population of neurons. In this case we have only one neuron so there is only one row.

print(spikes)

# %%
# TsGroup is a dictionnary. You can look at the spike times of neuron by indexing it.
print(spikes[1])

# %%
# For the rest of the notebook, we are going to restrict the data to the first of epoch of Noise1.

noise_interval = noise_interval.loc[[0]]
spikes = spikes.restrict(noise_interval)
current = current.restrict(noise_interval)


# %%
# Now we can visualize for the first epoch of `noise_interval` all the data together.

fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.plot(current, "grey")
ax.plot(spikes.to_tsd([-5]), "|", color="k", ms = 10)
ax.set_ylabel("Current (pA)")
ax.set_xlabel("Time (s)")
plt.show()

# %%
# Pynapple can compute a tuning curve (i.e. firing rate as a function of feature). Here the feature is the current.
# In pynapple, it's one line. In this case, we compute the firing rate over 15 bins over the range of the current.
tuning_curve = nap.compute_1d_tuning_curves(spikes, current, 15)

print(tuning_curve)

# %%
# In this case tuning_curve is a pandas DataFrame where each column is a neuron (one neuron in this case) and each row is a bin over the feature. We can plot the tuning curve of the neuron.

fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.plot(tuning_curve, color="k")
ax.set_xlabel("Current (pA)")
ax.set_ylabel("Firing rate (Hz)")
plt.show()

# %%
# We will try to capture this behavior with our GLM model that models spike counts. To fit a GLM model, you need to count the spikes within a particular bin size. You can do the spike count in pynapple in one line.

# bin size in seconds
bin_size = 0.001
count = spikes.count(bin_size, ep=noise_interval)

print(count)

# %%
# The GLM model is going to predict a firing rate. To be able to compare the output, we can compute the neuron firing rate.
firing_rate = count.smooth(50, 1000) / bin_size

# %%
# Let"s plot the firing rate against the spike times.
ex_intervals = current.threshold(0.0).time_support


# define plotting parameters
# colormap, color levels and transparency level
# for the current injection epochs
cmap = plt.get_cmap("autumn")
color_levs = [0.8, 0.5, 0.2]
alpha = 0.4

fig = plt.figure()
# first row subplot: current
ax = plt.subplot2grid((3, 3), loc=(0, 0), rowspan=1, colspan=3, fig=fig)
ax.plot(current, color="grey")
ax.set_ylabel("Current (pA)")
ax.set_title("Injected Current")
ax.axvspan(ex_intervals.loc[1,"start"], ex_intervals.loc[1,"end"], alpha=alpha, color=cmap(color_levs[0]))
ax.axvspan(ex_intervals.loc[2,"start"], ex_intervals.loc[2,"end"], alpha=alpha, color=cmap(color_levs[1]))
ax.axvspan(ex_intervals.loc[3,"start"], ex_intervals.loc[3,"end"], alpha=alpha, color=cmap(color_levs[2]))

# second row subplot: response
ax = plt.subplot2grid((3, 3), loc=(1, 0), rowspan=1, colspan=3, fig=fig)
ax.plot(firing_rate, color="k")
ax.plot(spikes.to_tsd([-1.5]), "|", color="k", ms=10)
ax.set_ylabel("Firing rate (Hz)")
ax.set_xlabel("Time (s)")
ax.set_title("Response")
ax.axvspan(ex_intervals.loc[1,"start"], ex_intervals.loc[1,"end"], alpha=alpha, color=cmap(color_levs[0]))
ax.axvspan(ex_intervals.loc[2,"start"], ex_intervals.loc[2,"end"], alpha=alpha, color=cmap(color_levs[1]))
ax.axvspan(ex_intervals.loc[3,"start"], ex_intervals.loc[3,"end"], alpha=alpha, color=cmap(color_levs[2]))
ylim = ax.get_ylim()

# third subplot: zoomed responses
for i in range(len(ex_intervals)-1):
    interval = ex_intervals.loc[[i+1]]
    ax = plt.subplot2grid((3, 3), loc=(2, i), rowspan=1, colspan=1, fig=fig)
    ax.plot(firing_rate.restrict(interval), color="k")
    ax.plot(spikes.restrict(interval).to_tsd([-1.5]), "|", color="k", ms=10)
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(ylim)
    for spine in ["left", "right", "top", "bottom"]:
        color = cmap(color_levs[i])
        # add transparency
        color = (*color[:-1], alpha)
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(2)

plt.tight_layout()

# %%
# ##NEMOS
#
# We picked a trial with some simple current injection, and we can see that
# injecting current leads to increased firing rate.
#
# Initial impulse then is that this is all we need! let's fit a simple model
# that only takes the injected current as input. Using this model is equivalent
# to saying "the only thing that influences the firing rate of the neuron is
# whether it receives a current injection." (As neuroscientists, we know this
# isn't true, but it does look reasonable from the data above! We'll build in
# complications later.)

# %%
# In this most basic form, the model components will be the following:
#
# 1. The predictor and the observations, i.e. the injected current and the counts
# 2. A liner-non-linear map that maps the predictor into a firing rate.
# 3. The Poisson log-likelihood that quantifies how likely are the observed count for a given firing rate.
#
# Let's define and inspect one component at the time.
#
# ### 1. Predictor & Observations
# The input current is now a Tsd object, of shape (num_time_pts, ) sampled at 20KHz, while counts
# are binned at 1KHz (1ms bin size).

print(f"current shape: {current.shape}")
print(f"current sampling rate: {current.rate/1000.} KHz")

print(f"\ncount shape: {count.shape}")
print(f"count sampling rate: {count.rate/1000} KHz")

# %%
# As a first step, we should down-sample it to the resolution of the spike counts.
# We can use the bin_average method from pynapple to achieve this.
input_feature = current.bin_average(bin_size, ep=noise_interval)

# Additional step, nemos requires predictors of shape (num_time_pts, num_neurons, num_features) and
# counts of shape (num_time_pts, num_neurons). We are modeling
# one neuron with one predictor. pynapple as a class called TsdFrame for 2-dimensional data and TsdTensor
# n-dimensnoal data, n > 2.

input_feature = nap.TsdTensor(t=input_feature.t, d=np.expand_dims(input_feature.d, (1, 2)))
counts = nap.TsdFrame(t=count.t, d=count.d, columns="count")

# check that the dimensionality matches nemos expectation
print(f"feature shape: {input_feature.shape}")
print(f"count shape: {count.shape}")

# %%
# ### 2. Linear-Non-Linear transformation of the current
# Our feature current $i(t)$, is first passed by a linear (affine more precisely) transformation.
# Since it is 1-dimensional, this is equivalent to scaling it by a constant ad adding an intercept
# $$
# L(i(t)) = w \cdot i(t) + c
# $$
# Then a non-linearity is applied to transform this quantity into a positive firing rate.
# $$ r(t) = \exp (w \cdot i(t) + c) $$
# Let's see how this linear-non-linear transformation is implemented in nemos.
# First we need to define a model object.

# this the defines the model object, ignore the regularizer for now, more about it
# in future tutorial, we are just changing the default.
model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(solver_name="LBFGS"))

# %%
# Usually we would fit the weights and intercepts, but for this demonstration we hard code them
# In nemos weights are (num_neurons, num_features) and intercepts (num_neurons,)
w = np.atleast_2d(0.05)
c = np.atleast_1d(-2)

model.coef_ = w
model.intercept_ = c

# linear
linear = np.squeeze(input_feature) * np.squeeze(model.coef_) + model.intercept_
# non-linearity is called inverse-link function, a naming convention from statistics
# it is stored in the observation model attribute
predicted_rate = model.observation_model.inverse_link_function(linear.d)
predicted_rate = nap.Tsd(t=input_feature.t, d=np.asarray(predicted_rate))

# in nemos, this transformation computed in a single step by the model `predict` method.
predicted_rate_nmo = model.predict(input_feature.d)
predicted_rate_nmo = nap.Tsd(t=input_feature.t, d=np.squeeze(np.asarray(predicted_rate_nmo)))


# %%
# let's plot each step for 500ms
interval = nap.IntervalSet(start=471, end=471.5)
plt.figure(figsize=(10, 3.5))
ax = plt.subplot(1, 3, 1)
ax.set_title("Current")
ax.plot(np.squeeze(input_feature.restrict(interval)))
ax = plt.subplot(1, 3, 2)
ax.set_title("Linearly Transformed")
ax.plot(np.squeeze(linear.restrict(interval)))
ax.set_xlabel("Time (s)")
ax = plt.subplot(1, 3, 3)
ax.set_title("LNL Transformed")
ax.plot(np.squeeze(predicted_rate.restrict(interval)))
ax.plot(np.squeeze(predicted_rate_nmo.restrict(interval)), "--k")
plt.tight_layout()


# %%
# !!! info
#     Only the weights $w$ and the intercept $c$ are learned, the non-linearity is kept fixed.
#     In nemos, we default to the exponential, other choices such as soft-plus, are allowed. These
#     choices guarantee convexity, i.e. a single optimal solution.
#     In principle, one could choose a more complex non-linearity, but convexity is not
#     guaranteed in general.
#

# Use a second order method for a more precise estimates of the maximum-likelihood
glm = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(solver_name="LBFGS"))
glm.fit(input_feature, count)

# %%
# !!! warning
#     explain why LBFGS

# %%
# Now that we've fit our data, we can use the model to predict the firing rates and
# convert units from spikes/bin to spikes/sec
predicted_fr = glm.predict(input_feature) / bin_size

# let's reintroduce the time axis by defining a TsdFrame
# convert first to numpy array otherwise to make it pynapple compatible
predicted_fr = nap.TsdFrame(t=count.t, d=np.asarray(predicted_fr))
smooth_predicted_fr = predicted_fr.smooth(50, 1000)

# %%
# We can compare the model predicted rate with the observed ones. The predicted rate
# is small but never 0, even when the injected current is below the firing threshold (yellow subplot); And,
# since the average rate is captured correctly by the model, for higher injected currents the rate
# is over-estimated.

# compare observed mean firing rate with the model predicted one
print(f"Observed mean firing rate: {np.mean(count) / bin_size} Hz")
print(f"Predicted mean firing rate: {np.mean(predicted_fr)} Hz")

fig = plt.figure()
# first row subplot: current
ax = plt.subplot2grid((3, 3), loc=(0, 0), rowspan=1, colspan=3, fig=fig)
ax.plot(current, color="grey")
ax.set_ylabel("Current (pA)")
ax.set_title("Injected Current")
ax.axvspan(ex_intervals.loc[1, "start"], ex_intervals.loc[1, "end"], alpha=alpha, color=cmap(color_levs[0]))
ax.axvspan(ex_intervals.loc[2, "start"], ex_intervals.loc[2, "end"], alpha=alpha, color=cmap(color_levs[1]))
ax.axvspan(ex_intervals.loc[3, "start"], ex_intervals.loc[3, "end"], alpha=alpha, color=cmap(color_levs[2]))

# second row subplot: response
ax = plt.subplot2grid((3, 3), loc=(1, 0), rowspan=1, colspan=3, fig=fig)
ax.plot(firing_rate, color="k", label="observed")
ax.plot(smooth_predicted_fr, color="tomato", label="glm")
ax.plot(spikes.to_tsd([-1.5]), "|", color="k", ms=10)
ax.set_ylabel("Firing rate (Hz)")
ax.set_xlabel("Time (s)")
ax.set_title("Response")
ax.legend()
ax.axvspan(ex_intervals.loc[1, "start"], ex_intervals.loc[1, "end"], alpha=alpha, color=cmap(color_levs[0]))
ax.axvspan(ex_intervals.loc[2, "start"], ex_intervals.loc[2, "end"], alpha=alpha, color=cmap(color_levs[1]))
ax.axvspan(ex_intervals.loc[3, "start"], ex_intervals.loc[3, "end"], alpha=alpha, color=cmap(color_levs[2]))
ylim = ax.get_ylim()

# third subplot: zoomed responses
for i in range(len(ex_intervals)-1):
    interval = ex_intervals.loc[[i+1]]
    ax = plt.subplot2grid((3, 3), loc=(2, i), rowspan=1, colspan=1, fig=fig)
    ax.plot(firing_rate.restrict(interval), color="k")
    ax.plot(smooth_predicted_fr.restrict(interval), color="tomato")
    ax.plot(spikes.restrict(interval).to_tsd([-1.5]), "|", color="k", ms=10)
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_xlabel("Time (s)")
    #ax.set_ylim(ylim)
    for spine in ["left", "right", "top", "bottom"]:
        color = cmap(color_levs[i])
        # add transparency
        color = (*color[:-1], alpha)
        ax.spines[spine].set_color(color)
        ax.spines[spine].set_linewidth(2)

plt.tight_layout()

# We can compare the tuning curves
tuning_curve_model = nap.compute_1d_tuning_curves_continuous(predicted_fr, current, 15)

plt.figure()
plt.plot(tuning_curve, "k", label="observed")
plt.plot(tuning_curve_model, color="tomato", label="glm")
plt.ylabel("Firing rate (Hz)")
plt.xlabel("Current (pA)")
plt.legend()

# Input and predicted rate looks very similar, why?
interval = nap.IntervalSet(start=471, end=472)
fig, axs = plt.subplots(2, 1, figsize=(12, 6))

axs[0].plot(current.restrict(interval), "k")
axs[0].set_ylabel("Current (pA)")
axs[1].plot(predicted_fr.restrict(interval), "tomato")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Firing rate (Hz)")
plt.tight_layout()

# %%
# This is true even if the model

# pass it through the log
rate = np.exp(glm.coef_[0,0] * current + glm.intercept_[0]) / bin_size
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

ax.plot(rate.restrict(interval), "b")
ax.set_ylabel("Firing rate (Hz)")
ax.plot(predicted_fr.restrict(interval), "tomato", ls="--")
ax.set_xlabel("Time (s)")
plt.tight_layout()


# pred_fr = nap.TsdFrame(binned_current.t, np.array(pred_fr))
# # get the times of these simulated spikes
# pred_spike_times = binned_spikes[np.where(pred_spikes)].times()
# spikes_dict = dict(spikes)
# spikes_dict[2] = np.asarray(pred_spike_times)
# spikes = nap.TsGroup(spikes_dict)
# spikes.set_info(label=np.array(['neuron', 'GLM predicted']))

# plot_current_injection(binned_current, spikes, pred_fr,
#                        title=f"Stimulus {stim_type}, trial {sweep_num}")

# # %%
# #
# # Let's look at the periods when we're injecting current, when we see spikes:

# current_intervals = np.squeeze(binned_current).threshold(0).time_support
# fig, axes = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'wspace': .4})
# for i, ax in zip(range(len(current_intervals)), axes.flatten()):
#     interval = current_intervals.loc[[i]]
#     current_spikes = spikes.restrict(interval)
#     spike_count = np.squeeze(current_spikes.count().values)
#     plot_current_injection(binned_current.restrict(interval),
#                            current_spikes,
#                            pred_fr.restrict(interval),
#                            ax=ax, add_legend=False,
#                            title=f"{spike_count[0]} real spikes, {spike_count[1]} simulated spikes")

# # %%
# # What do we see?
# #
# # - simulated spike in first current interval
# # - spikes spaced weird in second interval
# # - way too many spikes in third
# #
# # why might this be? if you're really observant, you may have noticed that
# # firing rate looks like it's basically a shifted copy of the input current.
# # And it is!
# #
# # Remember tutorial 0: GLM models the log firing rate as a linear combination
# # of inputs. in this case, the firing rate at time t is only the function of
# # the input current *at that time*. we can see that by plotting the tuning
# # curve between firing rate and input current:

# fig, ax = plt.subplots(1, 1, figsize=(7, 5))
# ax.plot(nap.compute_1d_tuning_curves(spikes[[1]], binned_current, 15), 'k', label="Observed")
# ax.plot(nap.compute_1d_tuning_curves_continuous(pred_fr, np.squeeze(binned_current), 15), 'r--', label="Predicted")
# ax.legend()
# ax.set(ylabel="Firing rate [Hz]", xlabel="Input current [pA]")

# # %%
# #
# # Predicted firing rate is a smooth exponential of the input current, while
# # observed is not just noisier, but falls off rapidly at high levels of current.
# #
# # We can also recover this tuning directly from the GLM object:

# print(glm.coef_.shape, glm.intercept_.shape)
# current = np.linspace(0, 140)
# fr = np.exp(glm.coef_ * current + glm.intercept_).squeeze()
# # the firing rate calcualted above is in spikes per bin, so convert that to Hz
# fr /= bin_size
# ax.plot(nap.compute_1d_tuning_curves(spikes[[1]], binned_current, 15), 'k', label="Observed")
# ax.plot(nap.compute_1d_tuning_curves_continuous(pred_fr, np.squeeze(binned_current), 15), 'r--', label="Predicted")
# ax.set(ylabel="Firing rate [Hz]", xlabel="Input current [pA]")
# ax.plot(current, fr, 'r', label="Predicted from GLM directly")
# ax.legend()
# fig
# # %%
# # Now:
# #
# # - compare number of spikes within each current interval
# #
# # - plot each current interval, show how rate is linear with input and so misses some effects
# #
# # - show tuning curves, point out that they differ at high input levels


# # point out this is two parameters and is only using input at time t, say we
# # can get a little fancier by adding history. try with and without bases?
# #
# # then add self-excitatin
