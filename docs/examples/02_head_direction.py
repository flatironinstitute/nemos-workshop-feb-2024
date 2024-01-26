# -*- coding: utf-8 -*-

"""
# Fit Head-direction population

C'est la vie
"""

import math
import os
import sys
from typing import Optional

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
angle = angle.restrict(wake_ep)

# %%
# First let's check that they are head-direction neurons.
tuning_curves = nap.compute_1d_tuning_curves(
    group=spikes, 
    feature=angle,
    nb_bins=61, 
    ep = wake_ep,
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

ep = nap.IntervalSet(
    start=10717, end=10730
)  # Select an arbitrary interval for plotting

pref_ang = tuning_curves.idxmax() #Let's compute the preferred angle quickly as follows.

fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.plot(spikes.restrict(ep).to_tsd(pref_ang), "|")
ax.plot(angle.restrict(ep), label = "Animal's HD")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Angle (rad)")
ax.legend()
plt.tight_layout()

# %%
# As we can see, the population activity tracks very well the current head-direction of the animal. 
# **Question : can we predict the spiking activity of each neuron based only on the activity of other neurons?**

# %%
# To use the GLM, we need first to bin the spike trains. Here we use pynapple
count = spikes.count(0.01, ep=wake_ep)

# %%
# Here we are going to rearrange neurons order based on their prefered directions.

count = nap.TsdFrame(t=count.t, d=count.values[:,pref_ang.reset_index(drop=True).sort_values().index.values])


# %%
# ## NEMOS
# It's time to use nemos. Our goal is to estimate the pairwise interaction between neurons.
# This can be quantified with a GLM if we use the past neuronal activity to predict the next time step.
# Before seeing how to model an entire neuronal populaiton, let's see how we can model a single neuron in this way.
# The simplest approach to directly use the past spike count history over a fixed length window.

# select a neuron
neuron_count = count.loc[[0]]

# fix a window size of 300ms (rate is in seconds)
window_size = int(1. * neuron_count.rate)

# create an input feature for the history (num_sample_pts, num_neuron, num_features)
# one feature for each time point in the window
input_feature = np.zeros((neuron_count.shape[0] - window_size, 1, window_size))
for i in range(window_size, neuron_count.shape[0]):
    input_feature[i-window_size, 0, :] = neuron_count[i-window_size:i]

plt.figure(figsize=(5, 7))
plt.suptitle("Input feature: Count History")
for k in range(10):
    ax = plt.subplot(10, 1, k+1)
    xvals = np.linspace(0, window_size-1,1000)
    yvals = input_feature[k, 0][np.searchsorted(np.arange(window_size), xvals)]
    plt.plot(xvals/count.rate, yvals, color="k")
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])

    if k != 9 :
        ax.set_xticks([])
    else:
        ax.set_xlabel("window (sec)")
    if k == 5:
        ax.set_ylabel(f"time bin")

plt.tight_layout()

# %%
# Now the feature dim is window_size = 30. If we select some weights, we can convert this feature matrix
# to a rate by the usual linear non-linear transformation.
# weight[i] represent how much the counts 30 - i  bins in the past contributes to the present rate.

# assume some weights (here exp decay)
weights = np.exp(np.linspace(-1, -2, window_size))
intercept = -2

# the predicted rate would be
pred_rate = np.exp(np.squeeze(input_feature) @ weights + intercept)

# or via nemos
model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))
model.coef_ = np.atleast_2d(weights)
model.intercept_ = np.atleast_1d(intercept)
pred_rate_nmo = model.predict(input_feature)


# predict ml param
model.fit(input_feature, neuron_count[window_size:, None])

plt.figure()
plt.title("spike history weights")
# flip time plot how a spike affects the future rate
plt.plot(np.arange(window_size)/count.rate, model.coef_.flatten()[::-1])
plt.xlabel("time from spike (sec)")
plt.ylabel("kernel")

# %%
# The response in the previous figure seems fairly smooth (low dimensional). It may be captured as a
# linear combination of simpler filters.
#
# One such filter, which has precision decaying linearly with time from spike is the log-raised cosine

# define five dim basis
basis = nmo.basis.RaisedCosineBasisLog(n_basis_funcs=10)

# evaluate the basis to get a (window_sizd, n_basis_funcs) matrix
eval_basis = basis.evaluate_on_grid(window_size)[1]

# plot the basis
plt.figure()
plt.plot(eval_basis)

# %%
# We can "compress" input feature by multiplying the matrix with the basis.

# Flip basis time axis, so that the leftward basis (narrowest basis) affects
# most recent history
compressed_features = np.squeeze(input_feature) @ eval_basis[::-1]
compressed_features = nap.TsdFrame(t=count[window_size:].t, d=compressed_features)

# compare dimensionality of features
print(f"Raw count history as feature: {np.squeeze(input_feature).shape}")
print(f"Compressed count history as feature: {compressed_features.shape}")


interval = nap.IntervalSet(8820.4, 8821)

plt.figure()
plt.plot(compressed_features.restrict(interval)[:, :3], label=[f"feature {k}" for k in range(3)])
cnt_interval = neuron_count.restrict(interval)
plt.vlines(cnt_interval.t[cnt_interval.d>0], -1, 1,"k",lw=1.5, label="spikes")
plt.xlabel("time (sec)")
plt.legend()

# %%
# This is equivalent to convolve the basis with the counts (without creating the large input_feature)
# Operation that can be performed in nemos
conv_spk = nmo.utils.convolve_1d_trials(eval_basis, [neuron_count[:, None]])[0]
conv_spk = nap.TsdTensor(t=count[window_size:].t, d=np.asarray(conv_spk[:-1]))

plt.figure()
plt.plot(compressed_features.restrict(interval)[:, :3], label=[f"feature {k}" for k in range(3)])
plt.plot(conv_spk.restrict(interval)[:, 0, :3], ms=3, marker="o", ls="none", color="tomato")
cnt_interval = neuron_count.restrict(interval)
plt.vlines(cnt_interval.t[cnt_interval.d>0], -1, 1,"k",lw=1.5, label="spikes")
plt.xlabel("time (sec)")
plt.legend()


# %%
# We can fit the ML parameter with the basis and compare the predicted history effect
# from the raw spike history and when using the basis

model_basis = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized("LBFGS"))
model_basis.fit(conv_spk, neuron_count[window_size:, None])

plt.figure()
plt.title("spike history weights")
plt.plot(model.coef_.flatten()[::-1])
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

# %%
# Check the dimension to make sure it make sense
print(convolved_count.shape)

# %%
# Build the right feature matrix
use_tp = 50000
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

fig, axs = plt.subplots(2, 5, figsize=(12,4))
for idx, weight in enumerate(np.transpose(weights, (2, 0, 1))):
    row, col = np.unravel_index(idx, axs.shape)
    axs[row, col].imshow(weight)
    axs[row, col].set_xlabel("Weights")
    axs[row, col].set_ylabel("Neurons")

plt.tight_layout()

# %%
# What would happen if we regressed explicitly the head direction?

