# -*- coding: utf-8 -*-

"""
# Fit Grid Cells population


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
import nemos as nmo
from scipy.ndimage import gaussian_filter

import sys
sys.path.append('..')
import utils

jax.config.update("jax_enable_x64", True)

# %%
# ## DATA STREAMING
# 
# Here we load the data from OSF. The data is a NWB file.
# blblalba say more
# Just run this cell

io = utils.data.download_dandi_data("000582", "sub-11265/sub-11265_ses-07020602_behavior+ecephys.nwb",
)

# %%
# ## PYNAPPLE
# 

data = nap.NWBFile(io.read())

# %%
# Let's see what is in our data

print(data)


# %%
# In this case, the data were used in this publicaton.
# https://www.science.org/doi/full/10.1126/science.1125572
# We thus expect to find neurons tuned to position and head-direction of the animal. 
# Let's verify that with pynapple first.

# %%
# Let's extract the spike times and the position of the animal.
spikes = data["units"]  # Get spike timings
position = data["SpatialSeriesLED1"] # Get the tracked orientation of the animal

# # %%
# # To make computation faster for this workshop, we will cut the data to the first x minutes.
# ep = nap.IntervalSet(start=0, end = 180)

# spikes = spikes.restrict(ep)
# position = position.restrict(ep)

# %%
# Here we compute quickly the head-direction of the animal from the position of the LEDs
diff = data['SpatialSeriesLED1'].values-data['SpatialSeriesLED2'].values
head_dir = (np.arctan2(*diff.T) + (2*np.pi))%(2*np.pi)
head_dir = nap.Tsd(data['SpatialSeriesLED1'].index, head_dir).dropna()


# %%
# Let's quickly compute some tuning curves for head-direction and spatial position 
hd_tuning = nap.compute_1d_tuning_curves(
    group=spikes, 
    feature=head_dir,
    nb_bins=61, 
    minmax=(0, 2 * np.pi)
    )

pos_tuning, binsxy = nap.compute_2d_tuning_curves(
    group=spikes, 
    features=position, 
    nb_bins=12)



# %%
# Let's plot the tuning curves for each neurons.
fig = plt.figure(figsize = (12, 4))
gs = plt.GridSpec(2, len(spikes))
for i in range(len(spikes)):
    ax = plt.subplot(gs[0,i], projection='polar')
    ax.plot(hd_tuning.loc[:,i])
    
    ax = plt.subplot(gs[1,i])
    ax.imshow(gaussian_filter(pos_tuning[i], sigma=1))
plt.tight_layout()


# %%
# ## NEMOS {.strip-code}
# It's time to use nemos. 
# Let's try to predict the spikes as a function of position and see if we can generate better tuning curves
# First we start by binning the spike trains in 5 ms bins.

bin_size = 0.005 # second
counts = spikes.count(bin_size, ep=position.time_support)

# %%
# We need to interpolate the position to the same time resolution.
# We can still use pynapple for this.
position = position.interpolate(counts)

# %%
# It's time to use nemos
# Let's define a multiplicative basis for position in 2 dimensions.

basis_2d = nmo.basis.RaisedCosineBasisLinear(n_basis_funcs=10) * \
            nmo.basis.RaisedCosineBasisLinear(n_basis_funcs=10)

# %%
# Let's ee what a few basis look like 
# Here we evaluate the basis on a 100x100 grid

X, Y, Z = basis_2d.evaluate_on_grid(100, 100)

fig, axs = plt.subplots(2,5, figsize=(10, 4))
for k in range(2):
  for h in range(5):
    axs[k][h].contourf(X, Y, Z[:, :, 50+2*(k+h)], cmap='Blues')

plt.tight_layout()

# %%
# Each basis represent a possible position of the animal in an arena whose borders are between 0 and 1.
# To make sure that we evaluate the true position of the animal, we need to rescale the position between 0 and 1.
position = (position - np.min(position, 0)) / (np.max(position, 0) - np.min(position, 0))

# %%
# Now we can "evaluate" the basis for each position of the animal
position_basis = basis_2d.evaluate(position['x'], position['y'])

# %%
# Now try to make sense of what it is
print(position_basis.shape)

# %%
# THe shape is (T, N_basis). It means for each time point, we evaluated the value of basis at the particular position 
# Let's plot 10 time steps.
fig = plt.figure(figsize = (12, 4))
gs = plt.GridSpec(2, 5)
xt = np.arange(0, 1000, 200)
for i in xt:
    ax = plt.subplot(gs[0,i//200])
    ax.imshow(position_basis[i].reshape(10, 10).T, origin = 'lower')
    plt.title("T "+str(i))

ax = plt.subplot(gs[1,2])

ax.plot(position['x'][0:1000], position['y'][0:1000])
ax.plot(position['x'][xt], position['y'][xt], 'o', color = 'red')

plt.tight_layout()


# %%
# Now we can fit the GLM and see what we get. In this case, we use Ridge for regularization.
model = nmo.glm.GLM(
    #regularizer=nmo.regularizer.Lasso(regularizer_strength=1e-3)
    )



# %%
# Let's fit the model

position_basis = np.repeat(np.expand_dims(position_basis, 1), len(spikes), 1)

model.fit(position_basis, counts)


# %%
# We can look at the tuning curves
rate_pos = model.predict(position_basis)


# %%
# Let's go back to pynapple
rate_pos = nap.TsdFrame(t=counts.t, d=np.asarray(rate_pos))

# %%
# And compute a tuning curves again

model_tuning, binsxy = nap.compute_2d_tuning_curves_continuous(
    tsdframe=rate_pos,
    features=position, 
    nb_bins=12)


# %% 
# Let's compare tuning curves

fig = plt.figure(figsize = (12, 4))
gs = plt.GridSpec(2, len(spikes))
for i in range(len(spikes)):
    ax = plt.subplot(gs[0,i])
    ax.imshow(gaussian_filter(pos_tuning[i], sigma=1))
    
    ax = plt.subplot(gs[1,i])
    ax.imshow(model_tuning[i])
plt.tight_layout()



# %%
# What is the score of the model?
L_model = model.score(position_basis, rate_pos)

print("log-likelihood = ", L_model)

# %%
# It's already very good. Can we improve it by adding regularization?
# We can cross-validate with scikit-learn
model = nmo.glm.GLM(
    regularizer=nmo.regularizer.Lasso(regularizer_strength=1e-3)
    )

from sklearn.model_selection import GridSearchCV
param_grid = dict(regularizer__regularizer_strength=[0.0, 1e-6, 1e-3])
cls = GridSearchCV(model, param_grid=param_grid)
cls.fit(position_basis, counts)

# %%
# Let's get the best estimator and see what we get

best_model = cls.best_estimator_

# %%
# Let's predict and compute new tuning curves
best_rate_pos = best_model.predict(position_basis)
best_rate_pos = nap.TsdFrame(t=counts.t, d=np.asarray(best_rate_pos))

best_model_tuning, binsxy = nap.compute_2d_tuning_curves_continuous(
    tsdframe=best_rate_pos,
    features=position, 
    nb_bins=12)


# %%
# Let's predict and compute new tuning curves
fig = plt.figure(figsize = (12, 4))
gs = plt.GridSpec(3, len(spikes))
for i in range(len(spikes)):
    ax = plt.subplot(gs[0,i])
    ax.imshow(gaussian_filter(pos_tuning[i], sigma=1))
    
    ax = plt.subplot(gs[1,i])
    ax.imshow(model_tuning[i])

    ax = plt.subplot(gs[2,i])
    ax.imshow(best_model_tuning[i])

plt.tight_layout()
plt.show()








