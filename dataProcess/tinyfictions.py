"""
preprocess and serve the tinyfiction dataset as a DataLoader.

训练一些中文小说: https://github.com/shjwudp/shu/tree/master/books
"""


import argparse
import glob
import json
import os, sys
sys.path.insert(0, os.getcwd())
import random
from typing import List

import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data/fictions"


def download():

    """Downloads the dataset to disk."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the fiction dataset, unless it's already downloaded
    filename = "红楼梦.txt"
    if os.path.exists(os.path.join(DATA_CACHE_DIR, filename)):
        print(f"{filename} is existed in {DATA_CACHE_DIR}")
    else:
        os.system(f"wget https://github.com/shjwudp/shu/raw/master/books/{filename} -P {DATA_CACHE_DIR}")

def pretokenize():
    enc = Tokenizer()
    filename = "红楼梦.txt"
    data_file = os.path.join(DATA_CACHE_DIR, filename)
    all_tokens = []
    with open(data_file, "r") as f:
        for line in f:
            text = line.strip()
            tokens = enc.encode(text, bos=True, eos=False)
            all_tokens.extend(tokens)
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens)}")
    with open(data_file.replace(".txt", ".bin"), "wb") as f:
        f.write(all_tokens.tobytes())
    print(f"Saved {data_file.replace('.txt', '.bin')}")
    print("Done.")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        data_file = os.path.join(DATA_CACHE_DIR, "红楼梦.bin")
        m_all = np.memmap(data_file, dtype=np.uint16, mode="r")

        # split out 10% of the data for validation
        split_ix = int(len(m_all) * 0.9)
        if self.split == "train":
            m = m_all[:split_ix]
        else:
            m = m_all[split_ix:]

        num_batches = len(m) // self.max_seq_len
        num_batches -= 1  # drop the last partial batch
        assert num_batches > 0, "this split is way too small? investigate."

        while True:
            ixs = list(range(num_batches))
            rng.shuffle(ixs)
            for ix in ixs:
                start = ix * self.max_seq_len
                end = start + self.max_seq_len + 1
                # calling .astype will copy the data into a new numpy array, now in RAM
                chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                x = chunk[:-1]
                y = chunk[1:]
                yield x, y


class TaskFiction:

    @staticmethod
    def iter_batches(split, batch_size, max_seq_len, device, num_workers=0):
        ds = PretokDataset(split, max_seq_len)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y




if __name__ == "__main__":
    """Usage
        python dataProcess/tinyfictions.py download
        python dataProcess/tinyfictions.py pretokenize
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "train_tokenizer", "pretokenize"])
    args = parser.parse_args()
    fun = {
        "download": download,
        "pretokenize": pretokenize,
    }
    fun[args.stage]()