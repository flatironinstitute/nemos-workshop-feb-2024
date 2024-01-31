#!/usr/bin/env python3

import requests
import os
import os.path as op
import math
import click
import tqdm.auto as tqdm

# Dandi stuffs
from pynwb import NWBHDF5IO
from dandi.dandiapi import DandiAPIClient
import fsspec
from fsspec.implementations.cached import CachingFileSystem
import h5py

DATA_DIR = op.dirname(op.realpath(__file__))

def download_data(filename, url):
    filename = op.join(DATA_DIR, filename)
    if not os.path.exists(filename):
        r = requests.get(url, stream=True)
        block_size = 1024*1024
        with open(filename, "wb") as f:
            for data in tqdm.tqdm(r.iter_content(block_size), unit="MB", unit_scale=True,
                                  total=math.ceil(int(r.headers.get("content-length", 0))//block_size)):
                f.write(data)
    return filename

def download_dandi_data(dandiset_id, filepath):
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

    return io



@click.command()
def main():
    download_data("allen_478498617.nwb", "https://osf.io/vf2nj/download")
    download_data("m691l1.nwb", "https://osf.io/xesdm/download")
    download_data("Mouse32-140822.nwb", "https://osf.io/jb2gd/download")


if __name__ == '__main__':
    main()
