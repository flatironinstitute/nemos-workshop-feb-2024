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
# It's time to use nemos. In this case, we want to use the activity of other neurons to predict the activity of a current neurons. We also want to introduce some time lag. We can use the basis of Nemos for this. 
# create three filters
window_size = 30
basis_obj = nmo.basis.RaisedCosineBasisLog(n_basis_funcs=4)
_, eval_basis = basis_obj.evaluate_on_grid(window_size)

fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.plot(eval_basis)
plt.tight_layout()

# %%
# Do the convolution
convolved_count = nmo.utils.convolve_1d_trials(eval_basis, [count.values])[0]

# %%
# Check the dimension to make sure it make sense
print(convolved_count.shape)

# convolved_count = convolved_count

# %%
# Notice the difference in size. To go back to pynapple, you need padding blabla
#padded_conv = np.asarray(nmo.utils.nan_pad_conv(convolved_count, ws, filter_type="causal")[0])

# %% 
# Now you can put the convoluted activity of the first neuron in a TsdFrame :
#padded_conv_0 = nap.TsdFrame(t=count.t, d=padded_conv[:,0,:])


# %%
# Build the right feature matrix
features = convolved_count.reshape(convolved_count.shape[0], -1)
features = np.expand_dims(features, 1)
features = np.repeat(features, len(spikes), 1)

# %%
# Now fit the GLM.
model = nmo.glm.GLM(regularizer=nmo.regularizer.Ridge(regularizer_strength=0.1, solver_name="LBFGS"))
model.fit(features[0:50000], count.values[window_size:][0:50000])

# %%
# Now extract the weights
weights = np.asarray(model.coef_)
weights = weights.reshape(len(spikes), len(spikes), -1)

# %%
# 
# Try to plot the glm coupling weights. What do you see?

fig, axs = plt.subplots(1, 4, figsize=(12,4))
axs[0].imshow(weights[..., 0])
axs[0].set_xlabel("Weights")
axs[0].set_ylabel("Neurons")
axs[1].imshow(weights[..., 1])
axs[1].set_xlabel("Weights")
axs[1].set_ylabel("Neurons")
axs[2].imshow(weights[..., 2])
axs[2].set_xlabel("Weights")
axs[2].set_ylabel("Neurons")
axs[3].imshow(weights[..., 3])
axs[3].set_xlabel("Weights")
axs[3].set_ylabel("Neurons")


# %%
# We can compute the L2 integral norm under the filter squared $f(t)^2$, using the basis
# $$
# \int f(t)^2 dt = \sum _w_{ij} \int b_i(t) b_j(t) dt
# $$
# First define a custom function for computing $\int b_i(x)b_j(x) dx$
# from scipy.integrate import simps
# def integrate_basis(basis):
#     x = np.linspace(0, 1, 10 ** 4)
#     eval = basis.evaluate(x)
#     eval_square = np.einsum("ti,tj->tij", eval, eval)
#     return simps(eval_square, axis=0, dx=x[1]-x[0])
#
# # %%
# # We can compute the integral and plot the results
# # int_basis = integrate_basis(basis_obj)
# # l2_norm = np.zeros(weights.shape[:2])
# # for i in range(l2_norm.shape[0]):
# #     for j in range(l2_norm.shape[1]):
# #         l2_norm[i,j] = np.einsum("j,jk,k->", weights[i,j], int_basis, weights[i,j])

# %%
# Give the feature and plot the predicted firing rate as a matrix
# Overly with spike counts and angle



