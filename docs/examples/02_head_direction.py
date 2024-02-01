# -*- coding: utf-8 -*-

"""
# Fit Head-direction population

## Learning objectives {.keep-text}

- Learn how to add history-related predictors to nemos GLM
- Learn about nemos `Basis` objects
- Learn how to use `Basis` objects with convolution

"""

import math
import os

import jax
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap
import sys
sys.path.append('..')
import utils

# Set the default precision to float64, which is generally a good idea for
# optimization purposes.
jax.config.update("jax_enable_x64", True)
# configure plots some
plt.style.use('../utils/nemos.mplstyle')

# %%
# ## Data Streaming
#
# Here we load the data from OSF. The data is a NWB file.
# blblalba say more
# Just run this cell

path = utils.data.download_data("Mouse32-140822.nwb", "https://osf.io/jb2gd/download")

# %%
# ## Pynapple
# We are going to open the NWB file with pynapple
# Since pynapple has been covered in tutorial 0, we are going faster here.

data = nap.load_file(path)

data

# %%

# Get spike timings
spikes = data["units"]

spikes

# Get the behavioural epochs (in this case, sleep and wakefulness)
epochs = data["epochs"]
# Get the tracked orientation of the animal
angle = data["ry"]
wake_ep = data["epochs"]["wake"]

# %%
# This cell will restrict the data to what we care about i.e. the activity of head-direction neurons during wakefulness.



# %%
# Select only those units that are in ADn

spikes = spikes.getby_category("location")["adn"].getby_threshold("rate", 1.0)
spikes = spikes.restrict(wake_ep).getby_threshold("rate", 1.0)
angle = angle.restrict(wake_ep)

# First let's check that they are head-direction neurons.
tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes, feature=angle, nb_bins=61, minmax=(0, 2 * np.pi)
)


# %%
# Each row indicates an angular bin (in radians), and each column corresponds to a single unit.

# %%
# Let's plot the tuning curve of the first two neurons.

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(tuning_curves.iloc[:, 0])
ax[0].set_xlabel("Angle (rad)")
ax[0].set_ylabel("Firing rate (Hz)")
ax[1].plot(tuning_curves.iloc[:, 1])
ax[1].set_xlabel("Angle (rad)")
plt.tight_layout()

# %%
# Before using Nemos, let's explore the data at the population level.

# Let's plot the preferred heading
fig = utils.plotting.plot_head_direction_tuning(
    tuning_curves, spikes, angle, threshold_hz=1, start=8910, end=8960
)

# %%
# As we can see, the population activity tracks very well the current head-direction of the animal.
# **Question : can we predict the spiking activity of each neuron based only on the activity of other neurons?**
# @Guillaume: why do we expect that we can predict using activity only?, and why it is interesting?
# maybe add reference to a paper

# To fit the GLM faster, we will use only the first 3 min of wake
wake_ep = nap.IntervalSet(
    start=wake_ep.loc[0, "start"], end=wake_ep.loc[0, "start"] + 3 * 60
)
# Filter the spikes with at least 1hz Rate


# Compute the preferred angle
pref_ang = tuning_curves.idxmax()
# Throw away those neurons who had a low firing rate
pref_ang = pref_ang[spikes.keys()]

# %%
# To use the GLM, we need first to bin the spike trains. Here we use pynapple
bin_size = 0.01
count = spikes.count(bin_size, ep=wake_ep)

# %%
# Here we are going to rearrange neurons order based on their prefered directions.
count = nap.TsdFrame(
    t=count.t,
    d=count.values[:, np.argsort(pref_ang.values)],
)

# %%
# ## Nemos {.strip-code}
# It's time to use nemos. Our goal is to estimate the pairwise interaction between neurons.
# This can be quantified with a GLM if we use the recent population spike history to predict the current time step.
# ### Spike History of a Neuron
# To simplify our life, let's see first how we can model spike history effects in a single neuron.
# The simplest approach is to use counts in fixed length window $i$, $y_{t-i}, \dots, y_{t-1}$ to predict the next
# count $y_{t}$. Let's plot the count history,


# select a neuron's spike count time series
neuron_count = count.loc[[0]]

# restrict to a smaller time interval
epoch_one_spk = nap.IntervalSet(
    start=count.time_support["start"][0], end=count.time_support["start"][0] + 1.2
)
plt.figure(figsize=(8, 3.5))
plt.step(
    neuron_count.restrict(epoch_one_spk).t, neuron_count.restrict(epoch_one_spk).d, where="post"
)
plt.title("Spike Count Time Series")
plt.xlabel("Time (sec)")
plt.ylabel("Counts")
plt.tight_layout()

# %%
# Now let's fix the spike history window size that we will use as predictor.

# set the size of the spike history window in seconds
window_size_sec = 0.8

utils.plotting.plot_history_window(neuron_count, epoch_one_spk, window_size_sec)


# %%
# For each time point, we shift our window one bin at the time and vertically stack the spike count history in a matrix.
# Each row of the matrix will be used as the predictors for the rate in the next bin (red narrow rectangle in
# the figure).

utils.plotting.run_animation(neuron_count, float(epoch_one_spk.start))

# %%
# If $t$ is smaller than the window size, we won't have a full window of spike history for estimating the rate.
# One may think of padding the window (with zeros for example) but this may generate weird border artifacts.
# To avoid that, we can simply restrict our analysis to times $t$ larger than the window.
# In this case, the total number of possible shifts is ("num_samples - window_size + 1").
# We also have to discard the very last shift of the matrix, since we don't have any more counts to predict
# (the red rectangle above is out of range), leaving us with "num_samples - window_size" rows.
#
# A fast way to compute this feature matrix is convolving the counts with the identity matrix, and get
# rid of the last row result.
#
# The binned counts originally have shape "number of samples",

print(f"Count shape: {neuron_count.shape}")

# %%
# Let's apply the convolution and strip the last row of the output.

# convert the prediction window to bins (by multiplying with the sampling rate)
window_size = int(window_size_sec * neuron_count.rate)

# convolve the counts with the identity matrix.
input_feature = nmo.utils.convolve_1d_trials(
    np.eye(window_size), np.expand_dims(neuron_count.d, axis=1)
)

# get rid of the last time point.
input_feature = np.squeeze(input_feature[:-1])

# %%
# We can confirm that the output shape matches the expectation.

print(f"Feature shape: {input_feature.shape}")
print(f"Time bins in counts: {neuron_count.shape[0]}")
print(f"Convolution window size in bins: {window_size}")

# %%
# !!! info
#     The convolution is performed in mode "valid" and always returns `num_samples - window_size + 1` time points.
#     This is true in general (numpy, scipy, etc.).
#
# We can visualize the output for a few time bins

suptitle = "Input feature: Count History"
neuron_id = 0
utils.plotting.plot_features(input_feature, count.rate, suptitle)

# %%
# As you may see, the time axis is backward, this happens because convolution flips the time axis.
# This is equivalent, as we can interpret the result as how much a spike will affect the future rate.
# In the previous tutorial our feature was 1-dimensional (just the current), now
# instead the feature dimension is 80, because our bin size was 0.01 sec and the window size is 0.8 sec.
# We can learn these weights by maximum likelihood by fitting a GLM.
#
# When working a real dataset, it is good practice to train your models on a chunk of the data and
# use the other chunk to assess the model performance. This process is known as "cross-validation".
# There is no unique strategy on how to cross-validate your model; What works best
# depends on the characteristic of your data (time series or independent samples,
# presence or absence of trials...), and that of your model. Here, for simplicity use the first
# half of the wake epochs for training and the second half for testing. This is a reasonable
# choice if the statistics of the neural activity does not change during the course of
# the recording. We will learn about better cross-validation strategy in later
# tutorials.

# convert features to TsdFrame
input_feature = nap.TsdFrame(t=neuron_count.t[window_size:], d=np.asarray(input_feature))

# construct the train and test epochs
duration = input_feature.time_support.tot_length("s")
start = input_feature.time_support["start"]
end = input_feature.time_support["end"]
first_half = nap.IntervalSet(start, start + duration / 2)
second_half = nap.IntervalSet(start + duration / 2, end)

# %%
# Fit the glm to the training set.

# define the GLM object
model = utils.model.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))

# predict ML paramametrs. Crop the first window_size (1 sec)
# because we don't have the full count history to predict
# these samples.

# Fit over the training epochs
model.fit(input_feature.restrict(first_half), neuron_count.restrict(first_half))

plt.figure()
plt.title("Spike History Weights")
plt.plot(np.arange(window_size) / count.rate, model.coef_, lw=2, label="GLM raw history 1st Half")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time From Spike (sec)")
plt.ylabel("Kernel")
plt.legend()

# %%
# The response in the previous figure seems noise added to a decay, therefore the response
# can be described with fewer degrees of freedom. In other words, it looks like we
# are using way too many weights to describe a simple response.
# If we are correct, what would happen if we re-fit the weights on the other half of the data?

# fit on the test set

model_second_half = utils.model.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))
model_second_half.fit(input_feature.restrict(second_half), neuron_count.restrict(second_half))

plt.figure()
plt.title("Spike History Weights")
plt.plot(np.arange(window_size) / count.rate, model.coef_, label="GLM raw history 1st Half", lw=2)
plt.plot(np.arange(window_size) / count.rate, model_second_half.coef_, color="orange", label="GLM raw history 2nd Half", lw=2)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time From Spike (sec)")
plt.ylabel("Kernel")
plt.legend()
# %%
# What can we conclude?
#
# The fast fluctuations are inconsistent across fits, indicating that
# they are probably capturing noise, a phenomenon known as over-fitting;
# On the other hand, the decaying trend is fairly consistent, even if
# our estimate is noisy. You can imagine how things could get
# worst if we needed a finer temporal resolution, such 1ms time bins
# (which would require 800 coefficients instead of 80).
# What can we do to mitigate over-fitting now?
#
# One way to proceed is to find a lower-dimensional representation of the response
# by parametrizing the decay effect. For instance, we could try to model it
# with an exponentially decaying function $f(t) = \exp( - \alpha t)$, with
# $\alpha >0$ a positive scalar. This is not a bad idea, because we would greatly
# simplify the dimensionality our features (from 80 to 1). Unfortunately,
# there is no way to know a-priori what is a good parameterization. More
# importantly, not all the parametrizations guarantee a unique and stable solution
# to the maximum likelihood estimation of the coefficients (convexity).
#
# In the GLM framework, the main way to construct a lower dimensional parametrization
# while preserving convexity, is to use a set of basis functions.
# For history-type inputs, whether of the spiking history or of the current
# history, we'll use the raised cosine log-stretched basis first described in
# [Pillow et al., 2005](https://www.jneurosci.org/content/25/47/11003). This
# basis set has the nice property that their precision drops linearly with
# distance from event, which is a makes sense for many history-related inputs
# in neuroscience: whether an input happened 1 or 5 msec ago matters a lot,
# whereas whether an input happened 51 or 55 msec ago is less important.

utils.plotting.plot_basis()

# %%
# !!! info
#
#     We provide a handful of different choices for basis functions, and
#     selecting the proper basis function for your input is an important
#     analytical step. We will eventually provide guidance on this choice, but
#     for now we'll give you a decent choice.
#
# nemos includes `Basis` objects to handle the construction and use of these
# basis functions.
#
# When we instantiate this object, the only argument we need to specify is the
# number of functions we want: with more basis functions, we'll be able to
# represent the effect of the corresponding input with the higher precision, at
# the cost of adding additional parameters.

basis = nmo.basis.RaisedCosineBasisLog(n_basis_funcs=8)

# `basis.evaluate_on_grid` is a convenience method to view all basis functions
# across their whole domain:
time, basis_kernels = basis.evaluate_on_grid(window_size)

print(basis_kernels.shape)

# time takes equi-spaced values between 0 and 1, we could multiply by the
# duration of our window to scale it to seconds.
time *= window_size_sec

# %%
# To appreciate why the raised-cosine basis can approximate well our response
# we can learn a "good" set of weight for the basis element such that
# a weighted sum of the basis approximates the GLM weights for the count history.
# One way to do so is by minimizing the least-squares.

# compute the least-squares weights
lsq_coef, _, _, _ = np.linalg.lstsq(basis_kernels, model.coef_, rcond=-1)

# plot the basis and the approximation
utils.plotting.plot_weighted_sum_basis(time, model.coef_, basis_kernels, lsq_coef)

# %%
#
# The first plot is the response of each of the 8 basis functions to a single
# pulse. This is known as the impulse response function, and is a useful way to
# characterize linear systems like our basis objects. The second plot are is a
# bar plot representing the least-square coefficients. The third one are the
# impulse responses scaled by the weights. The last plot shows the sum of the
# scaled response overlapped to the original spike count history weights.
#
# Our predictor previously was huge: every possible 80 time point chunk of the
# data, for 716800 total numbers. By using this basis set we can instead reduce
# the predictor to 8 numbers for every 80 time point window for 71680 total
# numbers. Basically an order of magnitude less. With 1ms bins we would have
# achieved 2 order of magnitude reduction in input size. This is a huge benefit
# in terms of memory allocation and, computing time. As an additional benefit,
# we will reduce over-fitting.
#
# Let's see our basis in action. We can "compress" spike history feature by convolving the basis
# with the counts (without creating the large spike history feature matrix).
# This that can be performed in nemos.
# use expand dims instead
# finish pr for convolve with pytree

conv_spk = nmo.utils.convolve_1d_trials(basis_kernels, [neuron_count[:, None]])[0]
conv_spk = nap.TsdFrame(t=count[window_size:].t, d=np.asarray(conv_spk[:-1, 0]))

print(f"Raw count history as feature: {input_feature.shape}")
print(f"Compressed count history as feature: {conv_spk.shape}")


# %%xf

# Visualize the convolution results
epoch_one_spk = nap.IntervalSet(8917.5, 8918.5)
epoch_multi_spk = nap.IntervalSet(8979.2, 8980.2)

utils.plotting.plot_convolved_counts(neuron_count, conv_spk, epoch_one_spk, epoch_multi_spk)

# find interval with two spikes to show the accumulation, in a second row

# %%
# Now that we have our "compressed" history feature matrix, we can fit the ML parameters for a GLM.

# use restrict on interval set training
model_basis = utils.model.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))
model_basis.fit(conv_spk.restrict(first_half), neuron_count.restrict(first_half))

# %%
# We can plot the resulting response, noting that the weights we just learned needs to be "expanded" back
# to the original `window_size` dimension by multiplying them with the basis kernels.
# We have now 8 coefficients,

print(model_basis.coef_)

# %%
# In order to get the response we need to multiply the coefficients by their corresponding
# basis function, and sum them.

self_connection = np.matmul(basis_kernels, model_basis.coef_)

print(self_connection.shape)

# %%

plt.figure()
plt.title("Spike History Weights")
plt.plot(time, model.coef_, alpha=0.3, label="GLM raw history")
plt.plot(time, self_connection, "--k", label="GLM basis", lw=2)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time from spike (sec)")
plt.ylabel("Weight")
plt.legend()

# %%
# Let's check if our new estimate does a better job in terms of over-fitting. We can do that
# by visual comparison, as we did previously.

model_basis_second_half = utils.model.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))
model_basis_second_half.fit(conv_spk.restrict(second_half), neuron_count.restrict(second_half))

# compute responses for the 2nd half fit
self_connection_second_half = np.matmul(basis_kernels, model_basis_second_half.coef_)

plt.figure()
plt.title("Spike History Weights")
plt.plot(time, model.coef_, "k", alpha=0.3, label="GLM raw history 1st half")
plt.plot(time, model_second_half.coef_, alpha=0.3, color="orange", label="GLM raw history 2nd half")
plt.plot(time, self_connection, "--k", lw=2, label="GLM basis 1st half")
plt.plot(time, self_connection_second_half, color="orange", lw=2, ls="--", label="GLM basis 2nd half")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time from spike (sec)")
plt.ylabel("Weight")
plt.legend()


# %%
# Or we can score the model predictions using both one half of the set for training
# and the other half for testing.

# compare model scores, as expected the training score is better with more parameters
# this may could be over-fitting.
print(f"full history train score: {model.score(input_feature.restrict(first_half), neuron_count.restrict(first_half), score_type='pseudo-r2-Cohen')}")
print(f"basis train score: {model_basis.score(conv_spk.restrict(first_half), neuron_count.restrict(first_half), score_type='pseudo-r2-Cohen')}")

# %%
# To check that, let's try to see ho the model perform on unseen data and obtaining a test
# score.
print(f"\nfull history test score: {model.score(input_feature.restrict(second_half), neuron_count.restrict(second_half), score_type='pseudo-r2-Cohen')}")
print(f"basis test score: {model_basis.score(conv_spk.restrict(second_half), neuron_count.restrict(second_half), score_type='pseudo-r2-Cohen')}")

# %%

# Let's extract
rate_basis = nap.Tsd(t=conv_spk.t, d=np.asarray(model_basis.predict(conv_spk.d))) * conv_spk.rate
rate_history = nap.Tsd(t=conv_spk.t, d=np.asarray(model.predict(input_feature))) * conv_spk.rate
ep = nap.IntervalSet(start=8819.4, end=8821)

# plot the rates
utils.plotting.plot_rates_and_smoothed_counts(
    neuron_count,
    {"Self-connection raw history":rate_history, "Self-connection bsais": rate_basis}
)

# %%
# ### All-to-all Connectivity
# The same approach can be applied to the whole population. Now the firing rate of a neuron
# is predicted not only by its own count history, but also by the rest of the
# simultaneously recorded population. We can convolve the basis with the counts of each neuron
# to get an array of predictors of shape, `(num_time_points, num_neurons, num_basis_funcs)`.
# This can be done in nemos with a single call,

convolved_count = nmo.utils.convolve_1d_trials(basis_kernels, count.values)
convolved_count = np.asarray(convolved_count[:-1])

# %%
# Check the dimension to make sure it make sense
print(f"Convolved count shape: {convolved_count.shape}")

# %%
# This is all neuron to one neuron. We can fit a neuron at the time, this is mathematically equivalent
# to fit the population jointly and easier to parallelize.
#
# !!! note
#     Once we condition on past activity, log-likelihood of the population is the sum of the log-likelihood
#     of individual neurons. Maximizing the sum (i.e. the population log-likelihood) is equivalent to
#     maximizing each individual term separately (i.e. fitting one neuron at the time).
#
# Nemos requires an input of shape `(num_time_points, num_features)`. To achieve that we need to concatenate
# the convolved count history in a single feature dimension. This can be done using numpy reshape.

convolved_count = convolved_count.reshape(convolved_count.shape[0], -1)
print(f"Convolved count reshaped: {convolved_count.shape}")
convolved_count = nap.TsdFrame(t=neuron_count.t[window_size:], d=convolved_count)

# %%
# Now fit the GLM for each neuron.

models = []
for neu in range(count.shape[1]):
    print(f"fitting neuron {neu}...")
    count_neu = count[:, neu]
    model = utils.model.GLM(
        regularizer=nmo.regularizer.Ridge(regularizer_strength=0.1, solver_name="LBFGS")
    )
    # models.append(model.fit(convolved_count.restrict(train_epoch), count_neu.restrict(train_epoch)))
    models.append(model.fit(convolved_count, count_neu.restrict(convolved_count.time_support)))


# %%
# Extract the weights

weights = np.zeros((count.shape[1], count.shape[1], basis.n_basis_funcs))
for receiver_neu in range(count.shape[1]):
    weights[receiver_neu] = models[receiver_neu].coef_.reshape(
        count.shape[1], basis.n_basis_funcs
    )

# %%
# Predict the rate (counts are already sorted by tuning prefs)

predicted_firing_rate = np.zeros((count.shape[0] - window_size, count.shape[1]))
for receiver_neu in range(count.shape[1]):
    predicted_firing_rate[:, receiver_neu] = models[receiver_neu].predict(
        convolved_count
    ) * conv_spk.rate

predicted_firing_rate = nap.TsdFrame(t=count[window_size:].t, d=predicted_firing_rate)

# %%
# Plot fit predictions over a short window not used for training.

# use pynapple for time axis for all variables plotted for tick labels in imshow
utils.plotting.plot_head_direction_tuning_model(tuning_curves, predicted_firing_rate, spikes, angle, threshold_hz=1,
                                                start=8910, end=8960, cmap_label="hsv")
# %%
# Let's see if our firing rate predictions improved and in what sense,
utils.plotting.plot_rates_and_smoothed_counts(
    neuron_count,
    {"Self-connection: raw history": rate_history,
     "Self-connection: bsais": rate_basis,
     "All-to-all: basis": predicted_firing_rate[:, 0]}
)

# %%
# Compute the responses by multiplying the coefficients with the basis and adding
# the result. This can be done in a single line of code with numpy.einsum.

responses = np.einsum("ijk, tk->ijt", weights, basis_kernels)

print(responses.shape)

# %%
# Finally, we can visualize the pairwise interactions by plotting
# all the coupling filters.

utils.plotting.plot_coupling(responses)


# %%
# ## Exercise

# What would happen if we regressed explicitly the head direction?
