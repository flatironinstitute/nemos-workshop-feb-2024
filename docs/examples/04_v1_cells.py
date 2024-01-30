# # -*- coding: utf-8 -*-
#
"""
# Fit V1 cell


"""

import jax
import math
import os


import matplotlib.pyplot as plt
import nemos as nmo
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

path = utils.data.download_data("m691l1.nwb", "https://osf.io/xesdm/download")


# %%
# ## PYNAPPLE
# The data have been copied to your local station.
# We are gonna open the NWB file with pynapple

data = nap.load_file(path)

# %%
# What does it look like?
print(data)

# %%
# Let's extract the data.
epochs = data["epochs"]
spikes = data["units"]
stimulus = data["whitenoise"]

# %%
# stimulus is white noise shown at 40 Hz
fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.imshow(stimulus[0])

# %%
# There are 73 neurons recorded together in V1. To fit the GLM faster, we will focus on one neuron.
spikes = spikes[[34]]

# %%
# First let's compute the response of this neuron to the stimulus by computing a spike trigger average.
sta = nap.compute_event_trigger_average(spikes, stimulus, binsize=0.025, windowsize = (-0.1, 0.0))

# %%
# Let's plot this receptive field
fig, axes = plt.subplots(1, len(sta), figsize=(12,4))
for i,t in enumerate(sta.t):
    axes[i].imshow(sta[i,0], vmin = np.min(sta), vmax = np.max(sta))
    axes[i].set_title(str(t)+" s")
plt.show()


# %%
# Let's get the average response of neuron 34 and plot it
response = np.mean(sta.get(-0.1, -0.025), axis=0)[0]
response = sta.get(-0.075)[0]

fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.imshow(response)

# %%
# Now we can compute the dot-product of the response with the stimulus to create a single 1 dimensional input for the GLM model.

stim_filt = np.dot(
    np.reshape(stimulus, (stimulus.shape[0], np.prod(stimulus.shape[1:]))), # if you know the shortcut let me know
    np.reshape(response, (np.prod(response.shape), 1))
    )

# %%
# And everything stays in pynapple yeah

fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.plot(stim_filt)


# %% 
# Fit the model
 
model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(solver_name="LBFGS"))


