# -*- coding: utf-8 -*-

"""
# Fit Grid Cells population


"""

import math
import os
from typing import Optional

import jax
import matplotlib.pyplot as plt
import nemos as nmo
import numpy as np
import pynapple as nap

from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import h5py
from scipy.ndimage import gaussian_filter

# %%
# ## DATA STREAMING
# 
# Here we load the data from the Allen datset. The data is a NWB file.
# Just run this cell

dandiset_id, filepath = (
    "000582",
    "sub-11265/sub-11265_ses-07020602_behavior+ecephys.nwb",
)

with DandiAPIClient() as client:
    asset = client.get_dandiset(dandiset_id, "draft").get_asset_by_path(filepath)
    s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

# first, create a virtual filesystem based on the http protocol
fs = fsspec.filesystem("http")

# create a cache to save downloaded data to disk (optional)
fs = CachingFileSystem(
    fs=fs,
    cache_storage="nwb-cache",  # Local folder for the cache
)

# next, open the file
file = h5py.File(fs.open(s3_url, "rb"))
io = NWBHDF5IO(file=file, load_namespaces=True)

# %%
# ## PYNAPPLE
# We are gonna open the NWB file with pynapple
# Since pynapple has been covered in tutorial 0, we are going faster here.

data = nap.NWBFile(io.read())

# %%
# Let's extract the data
spikes = data["units"]  # Get spike timings
position = data["SpatialSeriesLED1"] # Get the tracked orientation of the animal

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

pos_tuning, binsxy = nap.compute_2d_tuning_curves(spikes, position, 12)



# %%
# Here we plot quickly the tuning curves for each neurons.
fig = plt.figure(figsize = (12, 4))
gs = plt.GridSpec(2, len(spikes))
for i in range(len(spikes)):
    ax = plt.subplot(gs[0,i], projection='polar')
    ax.plot(hd_tuning.loc[:,i])
    
    ax = plt.subplot(gs[1,i])
    ax.imshow(gaussian_filter(pos_tuning[i], sigma=1))
plt.tight_layout()

plt.show()

# %%
# As we can see neurons have head-direction tuning and position tuning. Can you try to replicate the previous notebook of fitting the glm at the population level and predict the rate.



# %%
# Can you plot the coupling weights?


# %% 
# What happens if we add the position as a feature?



