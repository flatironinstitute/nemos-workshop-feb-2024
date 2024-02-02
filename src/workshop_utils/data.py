#!/usr/bin/env python3

import requests
import os
import os.path as op
import math
import click
import tqdm.auto as tqdm

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


@click.command()
def main():
    download_data("allen_478498617.nwb", "https://osf.io/vf2nj/download")
    download_data("m691l1.nwb", "https://osf.io/xesdm/download")
    download_data("Mouse32-140822.nwb", "https://osf.io/jb2gd/download")


if __name__ == '__main__':
    main()
