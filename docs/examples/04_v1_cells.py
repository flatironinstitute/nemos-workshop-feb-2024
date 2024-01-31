# # -*- coding: utf-8 -*-
#
"""
# Fit V1 cell

## Learning objectives

 - Learn how to combine GLM with other modeling approach.
 - Review previous tutorials.

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
import sys
sys.path.append('..')
import utils

# required for second order methods (BFGS, Newton-CG)
jax.config.update("jax_enable_x64", True)

# %%
# ## DATA STREAMING
#
# Here we load the data from OSF. This data comes from Sonica Saraf, in Tony
# Movshon's lab.

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
# <div class="notes">
#   - stimulus is white noise shown at 40 Hz
#   - white noise is a good stimulus for mapping basic stimulus properties of
#     V1 simple cells
# </div>
fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.imshow(stimulus[0])
stimulus.shape

# %%
# There are 73 neurons recorded together in V1. To fit the GLM faster, we will focus on one neuron.
print(spikes)
spikes = spikes[[34]]

# %%
#
# <div class="notes">
#   - goal is to predict the neuron's response to this white noise stimuli
#   - several ways we could do this, what do you think?
# </div>
#
# How could we predict neuron's response to white noise stimulus?
# 
# - we could fit the instantaneous spatial response. that is, just predict
#   neuron's response to a given frame of white noise. this will give an x by y
#   filter. implicitly assumes that there's no temporal info: only matters what
#   we've just seen
#
# - could fit spatiotemporal filter. instead of an x by y that we use
#   independently on each frame, fit (x, y, t) over, say 100 msecs. and then
#   fit each of these independently (like in head direction example)
#
# - that's a lot of parameters! can simplify by assumping that the response is
#   separable: fit a single (x, y) filter and then modulate it over time. this
#   wouldn't catch e.g., direction-selectivity because it assumes that phase
#   preference is constant over time
#
# - could mkae use of our knowledge of V1 and try to fit a more complex
#   functional form, e.g., a Gabor.
#
# That last one is very non-linear and thus non-convex. we'll do the third one.
#
# in this example, we'll fit the spatial filter outside of the GLM framework,
# using spike-triggered average, and then we'll use the GLM to fit the temporal
# timecourse.
#
# Spike-triggered average says: every time our neuron spikes, we store the
# stimulus that was on the screen. for the whole recording, we'll have many of
# these, which we then average to get this STA, which is the "optimal stimulus"
# / spatial filter.
#
# In practice, we do not just the stimulus on screen, but in some window of
# time around it. (it takes some time for info to travel through the eye/LGN to
# V1). Pynapple makes this easy:

sta = nap.compute_event_trigger_average(spikes, stimulus, binsize=0.025,
                                        windowsize=(-0.15, 0.0))
# %%
#
# sta is a `TsdTensor`, which gives us the 2d receptive field at each of the
# time points.
sta

# %%
#
# We index into this in a 2d manner: row, column (here we only have 1 column).
print(sta[1])
sta[1, 0]

# %%
# we can easily plot this
fig, axes = plt.subplots(1, len(sta), figsize=(3*len(sta),3))
for i, t in enumerate(sta.t):
    axes[i].imshow(sta[i,0], vmin = np.min(sta), vmax = np.max(sta))
    axes[i].set_title(str(t)+" s")


# %%
#
# that looks pretty reasonable for a V1 simple cell: localized in space,
# orientation, and spatial frequency. that is, looks Gabor-ish
#
# To convert this to the spatial filter we'll use for the GLM, let's take the
# average across the bins that look informative: -.125 to -.05
response = np.mean(sta.get(-0.125, -0.05), axis=0)[0]

fig, ax = plt.subplots(1, 1, figsize=(4,4))
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

# spikes -> restrict to same time support as stim_filt -> count
# stim_filt -> bin_average to same resolution

# %% 
# Fit the model
 
model = nmo.glm.GLM(regularizer=nmo.regularizer.UnRegularized(solver_name="LBFGS"))


