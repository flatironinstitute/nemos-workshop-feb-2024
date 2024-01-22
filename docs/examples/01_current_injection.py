# -*- coding: utf-8 -*-

"""# Fit injected current

For our first example, we will look at a very simple dataset: patch-clamp
recordings from a single neuron in layer 4 of rodent primary visual cortex.
This data is from the [Allen Brain
Atlas](https://celltypes.brain-map.org/experiment/electrophysiology/478498617),
and experimenters injected current directly into the cell, while recording the
neuron's membrane potential and spiking behavior. The experiments varied the
shape of the current across many sweeps, mapping the neuron's behavior in
response to a wide range of potential inputs.

!!! warning
    Is this description of the experiment correct?

For our purposes, we will examine only one of these sweeps, "Noise 1", in which
the experimentalists injected three pulses of current. The current is a square
pulse multiplied by a sinusoid of a fixed frequency, with some random noise
riding on top.

![Allen Brain Atlas view of the data we will analyze.](../../assets/allen_data.png)

In the figure above (from the Allen Brain Atlas website), we see the
approximately 22 second sweep, with the input current plotted in the first row,
the intracellular voltage in the second, and the recorded spikes in the third.
(The grey lines and dots in the second and third rows comes from other sweeps
with the same stimulus, which we'll ignore in this exercise.) When fitting the
Generalized Linear Model, we are attempting to model the spiking behavior, and
we generally do not have access to the intracellular voltage, so for the rest
of this notebook, we'll use only the input current and the recorded spikes
displayed in the first and third rows.

First, let us see how to load in the data and reproduce the above figure, which
we'll do using [pynapple](https://pynapple-org.github.io/pynapple/). We will
use pynapple throughout this workshop, as it simplifies handling this type of
data.

"""

# Import everything
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
import utils

# Set the default precision to float64, which is generally a good idea for
# optimization purposes.
jax.config.update("jax_enable_x64", True)

# %%
# ## Data Streaming
#
# While you can download the data directly from the Allen Brain Atlas and
# interact with it using their
# [AllenSDK](https://allensdk.readthedocs.io/en/latest/visual_behavior_neuropixels.html),
# we prefer the burgeoning [Neurodata Without Borders (NWB)
# standard](https://nwb-overview.readthedocs.io/en/latest/). We have converted
# this single dataset to NWB and uploaded it to the [Open Science
# Framework](https://osf.io/5crqj/). This allows us to easily load the data
# using pynapple, and it will immediately be in a format that pynapple understands!
#
# !!! tip
#
#     Pynapple can stream any NWB-formatted dataset! See [their
#     documentation](https://pynapple-org.github.io/pynapple/generated/gallery/tutorial_pynapple_dandi/)
#     for more details, and see the [DANDI Archive](https://dandiarchive.org/)
#     for a repository of compliant datasets.
#
# The first time the following cell is run, it will take a little bit of time
# to download the data, and a progress bar will show the download's progress.
# On subsequent runs, the cell gets skipped: we do not need to redownload the
# data.

path = os.path.join(os.getcwd(), "allen_478498617.nwb")
if os.path.basename(path) not in os.listdir(os.getcwd()):
    r = requests.get(f"https://osf.io/um3bj/download", stream=True)
    block_size = 1024*1024
    with open(path, "wb") as f:
        for data in tqdm.tqdm(r.iter_content(block_size), unit="MB", unit_scale=True,
            total=math.ceil(int(r.headers.get("content-length", 0))//block_size)):
            f.write(data)


# %%
# ## Pynapple
#
# ### Data structures and preparation
#
# Now that we've downloaded the data, let's open it with pynapple and examine
# its contents.

data = nap.load_file(path)
print(data)

# %%
#
# The dataset contains several different pynapple objects, which we discussed
# earlier today. Let's see how these relate to the data we visualized above:
#
# ![Annotated view of the data we will analyze.](../../assets/allen_data_annotated.gif)
# <!-- this gif created with the following imagemagick command: convert -layers OptimizePlus -delay 100 allen_data_annotated-units.svg allen_data_annotated-epochs.svg allen_data_annotated-stimulus.svg allen_data_annotated-response.svg -loop 0 allen_data_annotated.gif -->
#
# - `units`: timestamps of the neuron's spikes.
# - `epochs`: start and end times of different intervals, defining the
#   experimental structure, specifying when each stimulation protocol began and
#   ended.
# - `stimulus`: injected current, in Amperes, sampled at around 12k Hz.
# - `response`: the neuron's intracellular voltage, sampled at around 12k Hz.
#   We will not use this info in this example
#
# Now let's go through the relevant variables in some more detail:

trial_interval_set = data["epochs"]
# convert current from Ampere to pico-amperes, to match the above visualization
# and move the values to a more reasonable range.
current = data["stimulus"] * 1e12
spikes = data["units"]

# %% 
# First, let's examine `trial_interval_set`:

trial_interval_set.keys()

# %%
#
# `trial_interval_set` is a dictionary with strings for keys and
# [`IntervalSets`](https://pynapple-org.github.io/pynapple/reference/core/interval_set/)
# for values. Each key defines the stimulus protocol, with the value defining
# the begining and end of that stimulation protocol.

noise_interval = trial_interval_set["Noise 1"]
noise_interval

# %%
#
# As described above, we will be examining "Noise 1". We can see it contains
# three rows, each defining a separate sweep. We'll just grab the first sweep
# (shown in blue in the pictures above) and ignore the other two (shown in
# gray).
#
# To select only one epoch from an IntervalSet, use 2 square brackets:

noise_interval = noise_interval.loc[[0]]
noise_interval

# %%
#
# Now let's examine `current`:

current

# %%
#
# `current` is a `Tsd`
# ([TimeSeriesData](https://pynapple-org.github.io/pynapple/reference/core/time_series/))
# object with 2 columns. Like all `Tsd` objects, the first column contains the
# time index and the second column contains the data; in this case, the current
# in pA.
#
# Currently `current` contains the entire ~900 second experiment but, as
# discussed above, we only want one of the Noise 1 sweeps. Fortunately,
# `pynapple` makes it easy to grab out the relevant time points by making use
# of the `noise_interval` we defined above:
current = current.restrict(noise_interval)
current

# %%
#
# Notice that the timestamps have changed and our shape is much smaller.
#
# Finally, let's examine the spike times. `spikes` is a
# [`TsGroup`](https://pynapple-org.github.io/pynapple/reference/core/ts_group/),
# a dictionary-like object that holds multiple `Ts` (timeseries) objects with
# potentially different time index:

spikes

# %%
#
# Typically, this is used to hold onto the spike times for a population of
# neurons. In this experiment, we only have recordings from a single neuron, so
# there's only one row.
#
# We can index into the `TsGroup` to see the timestamps for this neuron's
# spikes:
spikes[1]

# %%
#
# Similar to `current`, this object originally contains data from the entire
# experiment. To get only the data we need, we again use
# `restrict(noise_interval)`:

spikes = spikes.restrict(noise_interval)
spikes


# %%
#
# Now, let's visualize the data from this trial, replicating rows 1 and 3
# from the Allen Brain Atlas figure at the beginning of this notebook:

fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.plot(current, "grey")
ax.plot(spikes.to_tsd([-5]), "|", color="k", ms = 10)
ax.set_ylabel("Current (pA)")
ax.set_xlabel("Time (s)")

# %%
#
# ### Basic analyses
#
# Before using the Generalized Linear Model, or any model, it's worth taking
# some time to examine our data and think about what features are interesting
# and worth capturing. As we discussed in tutorial 0, the GLM is a model of the
# neuronal firing rate. However, in our experiments, we do not observe the
# firing rate, only the spikes! Even worse, the spikes are the output of a
# stochastic process, so running the exact same experiment multiple times will
# lead to slightly different spike times. This means that no model can
# perfectly predict spike times. So how do we tell if our model is doing a good
# job?
#
# Our objective function is the log-likelihood of the observed spikes given the
# predicted firing rate. That is, we're trying to find the firing rate, as a
# function of time, for which the observed spikes are likely. Intuitively, this
# makes sense: the firing rate should be high where there are many spikes, and
# vice versa. However, it can be difficult to figure out if your model is doing
# a good job by squinting at the observed spikes and the predicted firing rates
# plotted together. We'd like to compare the predicted firing rates against the
# observed ones, so we'll have to approximate the firing rate from the data in
# a model-free way.
#
# One common way of doing this is to smooth the spikes, convolving them with a
# Gaussian filter. This is equivalent to taking the local average of the number
# of spikes, with the size of the Gaussian determining the size of the
# averaging window. This approximate firing rate can then be compared to our
# model's predictions, in order to visualize its performance.
#
# !!! info
#
#     This is a heuristic for getting the firing rate, and shouldn't be taken
#     as the literal truth (to see why, pass a firing rate through a Poisson
#     process to generate spikes and then smooth the output to approximate the
#     generating firing rate). A model should not be expected to match this
#     approximate firing rate exactly, but visualizing the two firing rates
#     together can help you reason about which phenomena in your data the model
#     is able to adequately capture, and which it is missing.
#
#     For more information, see section 1.2 of [*Theoretical
#     Neuroscience*](https://boulderschool.yale.edu/sites/default/files/files/DayanAbbott.pdf),
#     by Dayan and Abbott.
#
# Pynapple can easily compute this approximate firing rate, and plotting this
# information will help us pull out some phenomena that we think are
# interesting and would like a model to capture.
#
# First, we must convert from our spike times to binned spikes:

# bin size in seconds
bin_size = 0.001
count = spikes.count(bin_size)
count

# %%
#
# Now, let's convert the binned spikes into the firing rate, by smoothing them
# with a gaussian kernel. Pynapple provides a convenience function for this:

# the argument to this method are the standard deviation of the gaussian and
# the full width of the window, given in bins. So std=50 corresponds to a
# standard deviation of 50*.001=.05 seconds
firing_rate = count.smooth(std=50, size=1000)
# convert from spikes per bin to spikes per second (Hz)
firing_rate = firing_rate / bin_size

# %%
#
# Note that this changes the object's type to a
# [`TsdFrame`](https://pynapple-org.github.io/pynapple/reference/core/time_series/)!
print(type(firing_rate))

# %%
# Now let's make a plot to more easily visualize the data:

# we're hiding the details of the plotting function for the purposes of this
# tutorial, but you can find it in the associated github repo if you're
# interested:
# https://github.com/flatironinstitute/nemos-workshop-feb-2024/blob/main/docs/examples/utils/plotting.py
utils.plotting.current_injection_plot(current, spikes, firing_rate)

# %%
# !!! warning
#     clean up plot some
#
# So now that we can view the details of our experiment a little more clearly,
# what do we see?
#
# - We have three intervals of increasing current, and the firing rate
#   increases as the current does.
#
# - While the neuron is receiving the input, it does not fire continuously or
#   at a steady rate; there appears to be some periodicity in the response. The
#   neuron fires for a while, stops, and then starts again. There's periodicity
#   in the input as well, so this pattern in the repsonse might be reflecting
#   that.
#
# - There's some decay in firing rate as the input remains on: there are three
#   four "bumps" of neuronal firing in the second and third intervals, and the
#   first is always the largest.
#
# These give us some good phenomena to try and predict! But there's something
# that's not quite obvious from the above plot: what is the relationship
# between the input and the firing rate? As described in the first bullet point
# above, it looks to be *monotonically increasing*: as the current increases,
# so does the firing rate. But is that exactly true? What form is that
# relationship?
#
# Pynapple can compute a tuning curve to help us answer this question, by
# binning our spikes based on the instantaneous input current and computing the
# firing rate within those bins:

tuning_curve = nap.compute_1d_tuning_curves(spikes, current, nb_bins=15)
tuning_curve

# %%
#
# `tuning_curve` is a pandas DataFrame where each column is a neuron (one
# neuron in this case) and each row is a bin over the feature (here, the input
# current). We can easily plot the tuning curve of the neuron:

fig, ax = plt.subplots(1, 1)
tc_idx = tuning_curve.index.to_numpy()
tc_val = tuning_curve.values.flatten()
width = tc_idx[1]-tc_idx[0]
ax.bar(tc_idx, tc_val, width, facecolor="grey", edgecolor="k", label="observed", alpha=0.4)
ax.set_xlabel("Current (pA)")
# swallow the output of this line
_=ax.set_ylabel("Firing rate (Hz)")

# %%
#
# We can see that, while the firing rate mostly increases with the current,
# it's definitely not a linear relationship, and it might start decreasing as
# the current gets too large.
#
# So this gives us three interesting phenomena we'd like our model to help
# explain: the tuning curve between the firing rate and the current, the firing
# rate's periodicity, and the gradual reduction in firing rate while the
# current remains on.

# %%
# ## Nemos
#
# Now that we've sufficiently explored our data, let's start modeling our
# spikes! How should we begin?
#
# When modeling, it's generally a good idea to start simple and add complexity
# as needed. Simple models are:
#
# - Easier to understand, so you can more easily reason through why a model is
#   capturing or not capturing some feature of your data.
#
# - Easier to fit, so you can more quickly see how you did.
#
# - Surprisingly powerful, so you might not actually need all the bells and
#   whistles you expected.
#
# Therefore, let's start with the simplest possible model: the only input is
# the instantaneous injected current. This is equivalent to saying that the
# only input influencing the firing rate of this neuron at time $t$ is current
# it received at that same time. As neuroscientists, we know this isn't exactly
# true, but given the data exploration we did above, it looks like a reasonable
# starting place. We can always build in more complications later.
#
# ### GLM components
#
# As described in tutorial 0, the Generalized Linear Model in neuroscience can
# also be thought of as a LNP model: a linear-nonlinear-Poisson model.
#
# <figure markdown>
# <!-- note that the src here has an extra ../ compared to other images, necessary when specifying path directly in html -->
# <img src="../../../assets/lnp_model.svg" style="width: 100%"/>
# <figcaption>LNP model schematic. Modified from Pillow et al., 2008.</figcaption>
# </figure>
#
# The model receives some input and then:
#
# - sends it through a linear filter or transformation of some sort.
# - passes that through a nonlinearity to get the *firing rate*.
# - uses the firing rate as the mean of a Poisson process to generate *spikes*.
#
# Let's step through each of those in turn:
#
# #### Linear transformation of input
#
# Our input feature(s) are first passed through a linear transformation, which
# rescales and shifts the input: $\bm{WX}+c$. In the one-dimensional case, as
# in this examine, this is equivalent to scaling it by a constant and adding an
# intercept: $w\dot x + c$.
#
# !!! note
#
#     In geometry, this is more correctly referred to as an [affine
#     transformation](https://en.wikipedia.org/wiki/Affine_transformation),
#     which includes translations, scaling, and rotations. *Linear*
#     transformations are the subset of affine transformations that do not
#     include translations.
#
#     In neuroscience, "linear" is the more common term, and we will use it
#     throughout.
#
# This means that, in the 1d case, we have two knobs to transform the input: we
# can make it bigger or smaller, or we can shift it up or down. Let's visualize
# some possible transformations that our model can make:

# to make this plot work well, keep this to three values
w = np.asarray([0.05, 10, -2]).reshape(3, 1)
c = np.asarray([0, -3, 1]).reshape(3, 1)

# pick a small subset, so we can see what's going on
plotting_interval = nap.IntervalSet(start=470.5, end=471.5)
L = w * np.expand_dims(current.restrict(plotting_interval), 0) + c
print(L.shape)

plt.figure(figsize=(10, 3.5))
ax = plt.subplot(1, 2, 1)
ax.set_title("Current $i(t)$")
ax.plot(current.restrict(plotting_interval), color="grey")
ax.set_xlabel("Time (s)")
ax = plt.subplot(1, 2, 2)
ax.set_title("Linearly Transformed $L(t)$")
ax.plot(L, color="orange")
ax.set_xlabel("Time (s)")
plt.tight_layout()

# %%
#
# ### 1. Predictor & Observations
# We are using the input current as the only predictor.
# Nemos requires predictors and counts to have the same number of samples.
# We can achieve that by down-sampling our current to the spike counts resolution using the bin_average method from
# pynapple.

input_feature = current.bin_average(bin_size, ep=noise_interval)

print(f"current shape: {input_feature.shape}")
print(f"current sampling rate: {input_feature.rate/1000.} KHz")

print(f"\ncount shape: {count.shape}")
print(f"count sampling rate: {count.rate/1000} KHz")


# %%
# Secondly we have to appropriately expand our variable dimensions, because nemos requires features of
# shape (num_time_pts, num_neurons, num_features) and counts of shape (num_time_pts, num_neurons).
# We can expand the dimension of counts and feature using the pynapple TsdFrame for 2-dimensional data and TsdTensor
# n-dimensnoal data, n > 2.

input_feature = nap.TsdTensor(t=input_feature.t, d=np.expand_dims(input_feature.d, (1, 2)))
counts = nap.TsdFrame(t=count.t, d=count.d, columns="count")

# check that the dimensionality matches nemos expectation
print(f"count shape: {count.shape}")
print(f"current shape: {input_feature.shape}")

# %%
# ### 2. Linear-Non-Linear transformation of the current
# Our feature current $i(t)$, is first passed by a linear (affine more precisely) transformation.
# Since it is 1-dimensional, this is equivalent to scaling it by a constant ad adding an intercept
# $$
# L(i(t)) = w \cdot i(t) + c
# $$

w = 0.05
c = -2

L = w * np.squeeze(input_feature) + c

interval = nap.IntervalSet(start=471, end=471.5)
plt.figure(figsize=(10, 3.5))
ax = plt.subplot(1, 2, 1)
ax.set_title("Current $i(t)$")
ax.plot(np.squeeze(input_feature.restrict(interval)), color="grey")
ax.set_xlabel("Time (s)")
ax = plt.subplot(1, 2, 2)
ax.set_title("Linearly Transformed $L(t)$")
ax.plot(np.squeeze(L.restrict(interval)), color="orange")
ax.set_xlabel("Time (s)")
plt.tight_layout()
# %%
# Then a non-linearity is applied to transform this quantity into a positive firing rate.
# $$ \lambda (t) = \exp (w \cdot i(t) + c) \tag{1}$$

# apply an exponential non-linearity to L(t)
predicted_rate = np.exp(L)

# %%
# The same linear-non-linear transformation is implemented by the `predict` method of the nemos.glm.GLM
# object.

# First we need to define a model object: default likelihood is poisson, default non-linearity is exp
# ignore the regularizer for now, more about it later
model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(solver_name="LBFGS"))

# set the weights and intercept (usually learned, but hard-coded for the example)
# in nemos weights are (num_neurons, num_features)
model.coef_ = np.atleast_2d(w)
# and intercepts (num_neurons,)
model.intercept_ = np.atleast_1d(c)

# equivalently in a single step, call the `predict` method passing the current
predicted_rate_nmo = model.predict(input_feature.d)
predicted_rate_nmo = nap.Tsd(t=input_feature.t, d=np.squeeze(np.asarray(predicted_rate_nmo)))

# %%
# let's plot each step for 500ms
interval = nap.IntervalSet(start=471, end=471.5)
plt.figure(figsize=(10, 3.5))
ax = plt.subplot(1, 3, 1)
ax.set_title("Current $i(t)$")
ax.plot(np.squeeze(input_feature.restrict(interval)), color="grey")
ax = plt.subplot(1, 3, 2)
ax.set_title("Linearly Transformed $L(t)$")
ax.plot(np.squeeze(L.restrict(interval)), color="orange")
ax.set_xlabel("Time (s)")
ax = plt.subplot(1, 3, 3)
ax.set_title(r"Rate $\lambda(t)$")
ax.plot(np.squeeze(predicted_rate.restrict(interval)), color="tomato", label=r"$\exp(L(t))$")
ax.plot(np.squeeze(predicted_rate_nmo.restrict(interval)), "--", label="nemos")
plt.legend()
plt.tight_layout()


# %%
# !!! info
#     Only the weights $w$ and the intercept $c$ are learned, the non-linearity is kept fixed.
#     In nemos, we default to the exponential, other choices such as soft-plus, are allowed. These
#     choices guarantee convexity, i.e. a single optimal solution.
#     In principle, one could choose a more complex non-linearity, but convexity is not
#     guaranteed in general.

# %%
# ### 3. The Poisson log-likelihood
# The last component of the model is the poisson log-likelihood that quantifies how likely it is
# to observe certain counts for a given firing rate time series.
# The if $y(t)$ are the spike counts, the equation for the log-likelihood is
# $$ \sum\_t \log P(y(t) | \lambda(t)) = \sum\_t  y(t) \log(\lambda(t)) - \lambda(t) - \log (y(t)!)\tag{2}$$
# In nemos, the log-likelihood can be computed by calling the score method passing the predictors and the counts.
# The method first compute the rate $\lambda(t)$ using (1) and then the likelihood using (2).

log_likelihood_0 = model.score(input_feature, counts, score_type="log-likelihood")
print(f"log-likelihood hard-coded weights: {log_likelihood_0}")

# %%
# We can learn the  maximum-likelihood (ML) weights by calling the fit method. If we compare the likelihood after
# fitting, it increased.
model.fit(input_feature, counts)
log_likelihood_ML = model.score(input_feature, counts, score_type="log-likelihood")
print(f"log-likelihood ML weights: {log_likelihood_ML}")

# %%
# !!! warning
#     explain why LBFGS

# %%
# Now that we've fit our data, we can use the model to predict the firing rates and
# convert units from spikes/bin to spikes/sec
predicted_fr = model.predict(input_feature) / bin_size

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

# fig = plt.figure()
# # first row subplot: current
# ax = plt.subplot2grid((3, 3), loc=(0, 0), rowspan=1, colspan=3, fig=fig)
# ax.plot(current, color="grey")
# ax.set_ylabel("Current (pA)")
# ax.set_title("Injected Current")
# ax.axvspan(ex_intervals.loc[1, "start"], ex_intervals.loc[1, "end"], alpha=alpha, color=cmap(color_levs[0]))
# ax.axvspan(ex_intervals.loc[2, "start"], ex_intervals.loc[2, "end"], alpha=alpha, color=cmap(color_levs[1]))
# ax.axvspan(ex_intervals.loc[3, "start"], ex_intervals.loc[3, "end"], alpha=alpha, color=cmap(color_levs[2]))

# # second row subplot: response
# ax = plt.subplot2grid((3, 3), loc=(1, 0), rowspan=1, colspan=3, fig=fig)
# ax.plot(firing_rate, color="k", label="observed")
# ax.plot(smooth_predicted_fr, color="tomato", label="glm")
# ax.plot(spikes.to_tsd([-1.5]), "|", color="k", ms=10)
# ax.set_ylabel("Firing rate (Hz)")
# ax.set_xlabel("Time (s)")
# ax.set_title("Response")
# ax.legend()
# ax.axvspan(ex_intervals.loc[1, "start"], ex_intervals.loc[1, "end"], alpha=alpha, color=cmap(color_levs[0]))
# ax.axvspan(ex_intervals.loc[2, "start"], ex_intervals.loc[2, "end"], alpha=alpha, color=cmap(color_levs[1]))
# ax.axvspan(ex_intervals.loc[3, "start"], ex_intervals.loc[3, "end"], alpha=alpha, color=cmap(color_levs[2]))
# ylim = ax.get_ylim()

# # third subplot: zoomed responses
# for i in range(len(ex_intervals)-1):
#     interval = ex_intervals.loc[[i+1]]
#     ax = plt.subplot2grid((3, 3), loc=(2, i), rowspan=1, colspan=1, fig=fig)
#     ax.plot(firing_rate.restrict(interval), color="k")
#     ax.plot(smooth_predicted_fr.restrict(interval), color="tomato")
#     ax.plot(spikes.restrict(interval).to_tsd([-1.5]), "|", color="k", ms=10)
#     ax.set_ylabel("Firing rate (Hz)")
#     ax.set_xlabel("Time (s)")
#     #ax.set_ylim(ylim)
#     for spine in ["left", "right", "top", "bottom"]:
#         color = cmap(color_levs[i])
#         # add transparency
#         color = (*color[:-1], alpha)
#         ax.spines[spine].set_color(color)
#         ax.spines[spine].set_linewidth(2)

# plt.tight_layout()

# # We can compare the tuning curves
# tuning_curve_model = nap.compute_1d_tuning_curves_continuous(predicted_fr, current, 15)

# plt.figure()
# tc_idx = tuning_curve.index.to_numpy()
# tc_val = tuning_curve.values.flatten()
# width = tc_idx[1]-tc_idx[0]
# plt.bar(tc_idx, tc_val, width, facecolor="grey", edgecolor="k", label="observed", alpha=0.4)
# plt.plot(tuning_curve_model, color="tomato", label="glm")
# plt.ylabel("Firing rate (Hz)")
# plt.xlabel("Current (pA)")
# plt.legend()

# %%
# below this is old

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
