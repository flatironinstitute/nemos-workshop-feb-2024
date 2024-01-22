# -*- coding: utf-8 -*-

"""
# Fit Head-direction population

C'est la vie
"""

import math
import os

import jax
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap
import requests
import tqdm
import utils

jax.config.update("jax_enable_x64", True)

# %%
# ## DATA STREAMING
# 
# Here we load the data from OSF. The data is a NWB file.
# blblalba say more
# Just run this cell

path = os.path.join(os.getcwd(), "Mouse32-140822.nwb")
if os.path.basename(path) not in os.listdir(os.getcwd()):
    r = requests.get(f"https://osf.io/jb2gd/download", stream=True)
    block_size = 1024*1024
    with open(path, 'wb') as f:
        for data in tqdm.tqdm(r.iter_content(block_size), unit='MB', unit_scale=True,
            total=math.ceil(int(r.headers.get('content-length', 0))//block_size)):
            f.write(data)

# %%
# ## PYNAPPLE
# We are gonna open the NWB file with pynapple
# Since pynapple has been covered in tutorial 0, we are going faster here.

data = nap.load_file(path)

spikes = data["units"]  # Get spike timings
epochs = data["epochs"]  # Get the behavioural epochs (in this case, sleep and wakefulness)
angle = data["ry"]  # Get the tracked orientation of the animal
wake_ep = data["epochs"]["wake"]

# %%
# This cell will restrict the data to what we care about i.e. the activity of head-direction neurons during wakefulness.

spikes = spikes.getby_category("location")["adn"].getby_threshold("rate", 1.0)  # Select only those units that are in ADn

# First let's check that they are head-direction neurons.
tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes,
    feature=angle,
    nb_bins=61,
    minmax=(0, 2 * np.pi)
    )

print(tuning_curves)

# %%
# Each row indicates an angular bin (in radians), and each column corresponds to a single unit. 

# %%
# Let's plot the tuning curve of the first two neurons.

fig, ax = plt.subplots(1, 2, figsize=(12,4))
ax[0].plot(tuning_curves.iloc[:,0])
ax[0].set_xlabel("Angle (rad)")
ax[0].set_ylabel("Firing rate (Hz)")
ax[1].plot(tuning_curves.iloc[:,1])
ax[1].set_xlabel("Angle (rad)")
plt.tight_layout()

# %% 
# Before using Nemos, let's explore the data at the population level.

# Let's compute the preferred angle quickly as follows.
pref_ang = tuning_curves.idxmax()

plot_ep = nap.IntervalSet(8910, 8960)
fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.plot(spikes.restrict(plot_ep).to_tsd(pref_ang), "|")
ax.plot(angle.restrict(plot_ep), label="Animal's HD")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (rad)")
ax.legend()
plt.tight_layout()

# %%
# As we can see, the population activity tracks very well the current head-direction of the animal. 
# **Question : can we predict the spiking activity of each neuron based only on the activity of other neurons?**

# To fit the GLM faster, we will use only the first 10 min of wake
wake_ep = nap.IntervalSet(start=wake_ep.loc[0,'start'], end=wake_ep.loc[0,'start']+3*60)
spikes = spikes.restrict(wake_ep).getby_threshold("rate", 1.0)
angle = angle.restrict(wake_ep)
# throw away those neurons who had a low firing rate on the epoch we're
# examining
pref_ang = pref_ang[spikes.keys()]

# %%
# To use the GLM, we need first to bin the spike trains. Here we use pynapple
bin_size = 0.01
count = spikes.count(bin_size, ep=wake_ep)

# %%
# Here we are going to rearrange neurons order based on their prefered directions.

count = nap.TsdFrame(t=count.t, d=count.values[:,pref_ang.reset_index(drop=True).sort_values().index.values])


# %%
# ## NEMOS
# It's time to use nemos. Our goal is to estimate the pairwise interaction between neurons.
# This can be quantified with a GLM if we use the past neuronal activity to predict the next time step.
# Before seeing how to model an entire neuronal populaiton, let's see how we can model a single neuron in this way.
# The simplest approach to directly use the past spike count history over a fixed length window.
# To visualize what we are going to do, let's zoom in the spike count history time series

# select a neuron
neuron_count = count.loc[[0]]

interval = nap.IntervalSet(start=count.time_support["start"][0], end=count.time_support["start"][0] + 1.2)
plt.figure(figsize=(8, 3.5))
plt.step(neuron_count.restrict(interval).t, neuron_count.restrict(interval).d, where="post")
plt.title("Spike Count Time Series")
plt.xlabel("Time (sec)")
plt.ylabel("Counts")
plt.tight_layout()

# %%
# What we want to achieve is to predict the spike count of a neuron at time $t$ using the spike count history in
# a window of size 1 second preceding "t". Clearly, if $t<1 \text{ sec}$ we won't have a full spike history, and
# this may generate border artifacts. We will restrict our time-series to $t>1 \text{ sec}$ to avoid that.

# select the predictor interval must end before the predicted count interval starts, we subtract an epsilon of 1ms
# to enforce that

# set the duration of the prediction window in sec
prediction_window = 1

input_interval = nap.IntervalSet(
    start=interval["start"][0],
    end=prediction_window+interval["start"][0] - 0.001
)
predicted_interval = nap.IntervalSet(
    start=prediction_window + interval["start"][0],
    end=prediction_window + interval["start"][0] + bin_size
)

plt.figure(figsize=(8, 3.5))
plt.step(neuron_count.restrict(interval).t, neuron_count.restrict(interval).d, where="post")
ylim = plt.ylim()
plt.axvspan(input_interval["start"][0], input_interval["end"][0], *ylim, alpha=0.4, color="orange", label="input")
plt.axvspan(predicted_interval["start"][0], predicted_interval["end"][0], *ylim, alpha=0.4, color="tomato", label="predicted")
plt.ylim(ylim)
plt.title("Spike Count Time Series")
plt.xlabel("Time (sec)")
plt.ylabel("Counts")
plt.legend()
plt.tight_layout()

# %%
# For each time point we shift our window one bin at the time.
# **mark the first one with a rectangle to show tha tit is the same
# time course as the previous fig**
n_shift = 20
fig, axs = plt.subplots(n_shift, 1, figsize=(8, 8))
for shift_bin in range(n_shift):
    ax = axs[shift_bin]
    shift_sec = shift_bin * bin_size
    # select the first bin after one sec
    input_interval = nap.IntervalSet(
        start=interval["start"][0] + shift_sec,
        end=prediction_window + interval["start"][0] + shift_sec - 0.001
    )
    predicted_interval = nap.IntervalSet(
        start=prediction_window + interval["start"][0] + shift_sec,
        end=prediction_window + interval["start"][0] + bin_size + shift_sec
    )

    ax.step(neuron_count.restrict(interval).t, neuron_count.restrict(interval).d, where="post")

    ax.axvspan(
        input_interval["start"][0],
        input_interval["end"][0], *ylim, alpha=0.4, color="orange")
    ax.axvspan(
        predicted_interval["start"][0],
        predicted_interval["end"][0], *ylim, alpha=0.4, color="tomato"
    )

    plt.ylim(ylim)
    if shift_bin == 0:
        ax.set_title("Spike Count Time Series")
    elif shift_bin == n_shift-1:
        ax.set_xlabel("Time (sec)")
    if shift_bin != n_shift-1:
        ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()

# %%
# We can construct a predictor feature matrix by vertically stacking the "orange" chunks of spike history.
# A fast way to do so is by convolving the counts with an identity matrix.

# convert the prediction window to bins (by multiplying with the sampling rate)
window_size = int(prediction_window * neuron_count.rate)

# create an input feature for the history (num_sample_pts, num_neuron, num_features)
# one feature for each time point in the window.

input_feature = nmo.utils.convolve_1d_trials(
    np.eye(window_size),
    np.expand_dims(neuron_count.d, (0, 2))
)[0]
# convert to numpy array (nemos returns jax arrays) and remove the last sample
# because there is nothing left to predict, there are no future counts.
input_feature = np.asarray(input_feature[:-1])

# %%
# !!! info
#     Convolution in mode "valid" always returns  `num_samples - window_size + 1` time points.
#     This is true in general (numpy, scipy, etc.), however, the spike counts will be of size
#     `num_samples - window_size` after we chunk the initial counts. For matching the time axis
#     we need to remove the last time point in feature.
#
# We can visualize the output for a few time bins

plt.figure(figsize=(8, 8))
plt.suptitle("Input feature: Count History")
cmap = plt.get_cmap("Reds_r")

for k in range(n_shift):
    ax = plt.subplot(n_shift, 1, k + 1)
    plt.step(np.arange(0, window_size)/count.rate, input_feature[k, 0], where="post")
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    if k != n_shift-1:

        ax.set_xticks([])
    else:
        ax.set_xlabel("lag (sec)")
    if k in [0, n_shift - 1]:
        ax.set_ylabel("$t_{%d}$" % (window_size+k), rotation=0)

plt.tight_layout()

# %%
# As you can see, the time axis is backward, this happens because convolution flips the time axis.
# This is equivalent, as we can interpret the result as how much a spike will affect the future rate.


# %%
# In the previous tutorial our feature was 1-dimensional (just the current), now
# instead the feature dimension is 100, because our bin size was 0.01 sec and the window size is 1 sec.
# By selecting a vector of 100 weights and by doing matrix multiplication of the features with the weights,
# plus the exponential non-linearity, we can obtain a scalar prediction for the rate.
# These weights are what we are learning.

# define the GLM object
model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))

# predict ML paramametrs. Crop the first window_size (1 sec)
# because we don't have the full count history to predict
# these samples.
model.fit(input_feature, np.expand_dims(neuron_count[window_size:], 1))

plt.figure()
plt.title("spike history weights")
# flip time plot how a spike affects the future rate
plt.plot(np.arange(window_size)/count.rate, model.coef_.flatten())
plt.xlabel("time from spike (sec)")
plt.ylabel("kernel")

# %%
# The response in the previous figure seems noise added to a decay, therefore the response
# can be described with fewer degrees of freedom.
#
# In the GLM framework, tne main way to construct a lower dimensional filter (while preserving convexity), is
# to use a set of basis functions. One such basis has precision decaying
# linearly with time from spike, is using the log-raised cosine basis function.
#
# ## Basis introduction.
#
# NEEDS TRANSITION
#
# How should we add something like spiking history? We could do the simple way:
# treat time t, t-1, ... t-i, all as independent predictors, with a separate
# weight on each one. That feels pretty weird -- that's a whole lot of weights
# and it's odd to treat each of them as independent predictors.
#
# Instead, what is typically done in the GLM framework is to use a set of basis
# functions. Why? Remember in tutorial 0: basis functions allow us to create a
# relatively low-dimensional representation that captures relevant properties
# of our feature while keeping the problem convex. that's pretty nifty.
#
# nemos includes `Basis` objects to handle the construction and use of these
# basis functions.
#
# !!! info
#
#     We provide a handful of different choices for basis functions, and
#     selecting the proper basis function for your input is an important
#     analytical step. We will eventually provide guidance on this choice, but
#     for now we'll give you a decent choice.
#
# ### History-related inputs
# For history-type inputs, whether of the spiking history or of the current
# history, we'll use the raised cosine log-stretched basis first described in
# [Pillow et al., 2005](https://www.jneurosci.org/content/25/47/11003). This
# basis set has the nice property that their precision drops linearly with
# distance from event, which is a nice property for many history-related inputs
# in neuroscience: whether an input happened 1 or 5 msec ago matters a lot,
# whereas whether an input happened 51 or 55 msec ago is less important.
#
# When we instantiate this object, the only argument we need to specify is the
# number of functions we want: with more basis functions, we'll be able to
# represent the effect of the corresponding input with the higher precision, at
# the cost of adding additional parameters.
basis = nmo.basis.RaisedCosineBasisLog(n_basis_funcs=8)
# %%
#
# `basis.evaluate_on_grid` is a convenience method to view all basis functions
# across their whole domain:
time, basis_kernels = basis.evaluate_on_grid(250)
plt.plot(time, basis_kernels)

# %%
#
# The above plot is the response of each of the 8 basis functions to a single
# pulse. This is known as the impulse response function, and is a useful way to
# characterize linear systems like our basis objects.
#
# Our predictor previously was huge: every possible 100 time point chunk of the
# data, for XXX total numbers. By using this basis set we can instead reduce
# the predictor to 8 numbers for every 100 time point window for YYY total
# numbers.
#
# We're approximating, the predicted firing rate will be comparable.
#
# - weights was 100 numbers
# - now will be 8 numbers
# - will be able to capture the main trend (exponential decay) without the noise
#
# We need to convolve  with the basis functions
# in order to generate our inputs
#

# evaluate the basis to get a (window_sizd, n_basis_funcs) matrix
eval_basis = basis.evaluate_on_grid(window_size)[1]

# plot the basis
plt.figure()
plt.plot(eval_basis)
plt.xticks(np.arange(0, window_size+20, 20), np.arange(0, window_size+20, 20)*bin_size)
plt.xlabel("Time from spike (sec)")
plt.ylabel("Weight")

# %%
# We can "compress" input feature by multiplying the matrix with the basis.

compressed_features = np.squeeze(input_feature) @ eval_basis
compressed_features = nap.TsdFrame(t=count[window_size:].t, d=compressed_features)

# compare dimensionality of features
print(f"Raw count history as feature: {np.squeeze(input_feature).shape}")
print(f"Compressed count history as feature: {compressed_features.shape}")


interval = nap.IntervalSet(8820.4, 8821)

plt.figure()
plt.plot(compressed_features.restrict(interval)[:, :3], label=[f"feature {k}" for k in range(3)])
cnt_interval = neuron_count.restrict(interval)
plt.vlines(cnt_interval.t[cnt_interval.d > 0], -1, 1,"k",lw=1.5, label="spikes")
plt.xlabel("time (sec)")
plt.legend()

# %%
# This is equivalent to convolve the basis with the counts (without creating the large input_feature)
# Operation that can be performed in nemos.
conv_spk = nmo.utils.convolve_1d_trials(eval_basis, [neuron_count[:, None]])[0]
conv_spk = nap.TsdTensor(t=count[window_size:].t, d=np.asarray(conv_spk[:-1]))

plt.figure()
plt.plot(compressed_features.restrict(interval)[:, :3], label=[f"feature {k}" for k in range(3)])
plt.plot(conv_spk.restrict(interval)[:, 0, :3], ms=3, marker="o", ls="none", color="tomato")
cnt_interval = neuron_count.restrict(interval)
plt.vlines(cnt_interval.t[cnt_interval.d>0], 1, 1,"k",lw=1.5, label="spikes")
plt.xlabel("time (sec)")
plt.legend()


# %%
# We can fit the ML parameter with the basis and compare the predicted history effect
# from the raw spike history and when using the basis. Note that we might flip the weights
# estimated using the raw count history for comparing it to the convolved basis, because in the former
# model the most recent history corresponds to the last weights, in the latter the opposite holds.

model_basis = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))
model_basis.fit(conv_spk, neuron_count[window_size:, None])

plt.figure()
plt.title("spike history weights")
plt.plot(model.coef_.flatten())
plt.plot(eval_basis @ model_basis.coef_.flatten())

# compare model scores
print(f"full history score: {model.score(input_feature, neuron_count[window_size:, None])}")
print(f"basis score: {model_basis.score(conv_spk, neuron_count[window_size:, None])}")

# %%
# The same approach can be applied to the whole population. Now the firing rate of a neuron
# is predicted not only on the basis of its own firing history, but also that of the
# rest simultaneously recorded population.
_, eval_basis = basis.evaluate_on_grid(window_size)

fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.plot(eval_basis)
plt.tight_layout()

# %%
# Perform the convolution over the whole population (23 neurons)
convolved_count = nmo.utils.convolve_1d_trials(eval_basis, [count.values])[0]
convolved_count = convolved_count[:-1]

# %%
# Check the dimension to make sure it make sense
print(convolved_count.shape)

# %%
#
# Build the right feature matrix. explain why we do it this way: because this
# is all-to-all connectivity with all 8 basis functions
use_tp = 10000
features = convolved_count.reshape(convolved_count.shape[0], -1)
features = np.expand_dims(features[:use_tp], 1)
features = np.repeat(features, len(spikes), 1)

# %%
# Now fit the GLM.
model = nmo.glm.GLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=0.1, solver_name="LBFGS"))
model.fit(features, count.values[window_size:][:use_tp])

# %%
# Now extract the weights
weights = np.asarray(model.coef_)
weights = weights.reshape(len(spikes), len(spikes), -1)

# %%
# 
# Try to plot the glm coupling weights. What do you see?

fig, axs = plt.subplots(2, 4, figsize=(12,4))
for idx, weight in enumerate(np.transpose(weights, (2, 0, 1))):
    row, col = np.unravel_index(idx, axs.shape)
    axs[row, col].imshow(weight)
    axs[row, col].set_xlabel("Weights")
    axs[row, col].set_ylabel("Neurons")

plt.tight_layout()

# predict rate
# model.predict(np.expand_dims(compressed_features, axis=1))

# %%
# ### Exercise
# What would happen if we regressed explicitly the head direction?


