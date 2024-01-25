#!/usr/bin/env python3

import requests
import os
import math
import tqdm.auto as tqdm

def download_data(path, url):
    if os.path.basename(path) not in os.listdir(os.getcwd()):
        r = requests.get(url, stream=True)
        block_size = 1024*1024
        with open(path, "wb") as f:
            for data in tqdm.tqdm(r.iter_content(block_size), unit="MB", unit_scale=True,
                                  total=math.ceil(int(r.headers.get("content-length", 0))//block_size)):
                f.write(data)
