#!/usr/bin/env python3

import requests
import numpy as np
import os
import os.path as op
import math
import click
import tqdm.auto as tqdm
from typing import Union
import pynapple as nap

TsdType = Union[nap.Tsd, nap.TsdFrame, nap.TsdTensor]

DATA_DIR = op.join(op.dirname(op.realpath(__file__)), '..', 'data')

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


def fill_forward(time_series, data, ep=None, out_of_range=np.nan):
    """
    Fill a time series forward in time with data.

    Parameters
    ----------
    time_series:
        The time series to match.
    data: Tsd, TsdFrame, or TsdTensor
        The time series with data to be extend.

    Returns
    -------
    : Tsd, TsdFrame, or TsdTensor
        The data time series filled forward.

    """
    assert isinstance(data, TsdType)

    if ep is None:
        ep = time_series.time_support
    else:
        assert isinstance(ep, nap.IntervalSet)
        time_series.restrict(ep)

    data = data.restrict(ep)
    starts = ep.start.values
    ends = ep.end.values

    filled_d = np.full((time_series.t.shape[0], *data.shape[1:]), out_of_range, dtype=data.dtype)
    fill_idx = 0
    for start, end in zip(starts, ends):
        data_ep = data.get(start, end)
        ts_ep = time_series.get(start, end)
        idxs = np.searchsorted(data_ep.t, ts_ep.t, side="right") - 1
        filled_d[fill_idx:fill_idx + ts_ep.t.shape[0]][idxs >= 0] = data_ep.d[idxs[idxs>=0]]
        fill_idx += ts_ep.t.shape[0]
    return type(data)(t=time_series.t, d=filled_d, time_support=ep)


@click.command()
def main():
    download_data("allen_478498617.nwb", "https://osf.io/vf2nj/download")
    download_data("m691l1.nwb", "https://osf.io/xesdm/download")
    download_data("Mouse32-140822.nwb", "https://osf.io/jb2gd/download")


if __name__ == '__main__':
    main()
