# # -*- coding: utf-8 -*-
#
"""
# Fit V1 cell


"""
#
# import jax
# import math
# import os
#
#
# import matplotlib.pyplot as plt
# import nemos as nmo
# import numpy as np
# import pynapple as nap
# import requests
# import tqdm
#
# # required for second order methods (BFGS, Newton-CG)
# jax.config.update("jax_enable_x64", True)
#
# # %%
# # ## DATA STREAMING
# #
# # Here we load the data from OSF. The data is a NWB file from the Allen Institute.
# # blblalba say more
# # Just run this cell
#
# path = os.path.join(os.getcwd(), "m691l1.nwb")
# if os.path.basename(path) not in os.listdir(os.getcwd()):
#     r = requests.get(f"https://osf.io/k65wh/download", stream=True)
#     block_size = 1024*1024
#     with open(path, "wb") as f:
#         for data in tqdm.tqdm(r.iter_content(block_size), unit="MB", unit_scale=True,
#             total=math.ceil(int(r.headers.get("content-length", 0))//block_size)):
#             f.write(data)
#
#
# # %%
# # ## PYNAPPLE
# # The data have been copied to your local station.
# # We are gonna open the NWB file with pynapple
#
# data = nap.load_file(path)
#
# # %%
# # What does it look like?
# print(data)
#
# # %%
# # Let's extract the data.
# epochs = data["epochs"]
# spikes = data["units"]
# stimulus = data["whitenoise"]
#
# # %%
# # stimulus is white noise shown at 40 Hz
# fig, ax = plt.subplots(1, 1, figsize=(12,4))
# ax.imshow(stimulus[0])
# plt.show()
#
# # %%
# # There are 73 neurons recorded together in V1. To fit the GLM faster, we will focus on one neuron.
# spikes = spikes[[34]]
#
# # %%
# # First let's compute the response of this neuron to the stimulus by computing a spike trigger average.
#
# try:
#     sta = nap.compute_event_trigger_average(spikes, stimulus, binsize=0.2, edge_offset='left')
# except:
#     sta = np.zeros((stimulus[0].shape))
#
# # %%
# # Let's plot this receptive field
# fig, ax = plt.subplots(1, 1, figsize=(12,4))
# ax.imshow(sta)
# plt.show()
#
# # %%
# # Do more stuffs
