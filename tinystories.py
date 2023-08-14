"""
Download, preprocess and serve the TinyStories dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
import torch.distributed as dist
from tqdm import tqdm
from config import LANGUAGE

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download():
    def download_with_url(data_url, data_filename):
        if not os.path.exists(data_filename):
            print(f"Downloading {data_url} to {data_filename}...")
            download_file(data_url, data_filename)
        else:
            print(f"{data_filename} already exists, skipping download...")
    
    def unpack(data_dir, data_filename):
        # unpack the tar.gz file into all the data shards (json files)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            print(f"Unpacking {data_filename}...")
            os.system(f"tar -xzf {data_filename} -C {data_dir}")
        else:
            print(f"{data_dir} already exists, skipping unpacking...")

        # print a single example just for debugging and such
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        with open(shard_filenames[0], "r") as f:
            data = json.load(f)
        print("Download done.")
        print(f"Number of shards: {len(shard_filenames)}")
        print(f"Example story:\n{data[0]}")

    """Downloads the dataset to disk."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url_en = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename_en = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    data_dir_en = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data-en")

    data_url_zh = "https://huggingface.co/datasets/52AI/TinyStoriesZh/resolve/main/TinyStories_all_data_zh_1M.tar.gz"
    data_filename_zh = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data_zh_1M.tar.gz")
    data_dir_zh = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data-zh")

    data_urls = [data_url_en, data_url_zh]
    data_filenames = [data_filename_en, data_filename_zh]
    data_dirs = [data_dir_en, data_dir_zh]
    if LANGUAGE=="en":
        download_with_url(data_urls[0], data_filenames[0])
        unpack(data_dirs[0], data_filenames[0])
    elif LANGUAGE=="zh":
        download_with_url(data_urls[1], data_filenames[1])
        unpack(data_dirs[1], data_filenames[1])
    else: #enzh
        for data_url, data_filename, data_dir in zip(data_urls, data_filenames, data_dirs):
            download_with_url(data_url, data_filename)
            unpack(data_dir, data_filename)
    

def pretokenize():
    enc = Tokenizer()

    def process_shard(shard):
        print(shard)
        # open when you not want to pretokenize repeatly.
        # if os.path.exists(shard.replace(".json", ".bin")): return None 
        with open(shard, "r") as f:
            data = json.load(f)
        all_tokens = []
        for example in tqdm(data):
            text = example["story"]
            text = text.strip() # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens) # a list for all tokens
        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        # write to disk
        tokenized_filename = shard.replace(".json", ".bin")
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        print(f"Saved {tokenized_filename}")

    # iterate the shards and tokenize all of them one by one
    data_dir_en = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data-en")
    data_dir_zh = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data-zh")
    shard_filenames_en = sorted(glob.glob(os.path.join(data_dir_en, "*.json")))
    shard_filenames_zh = sorted(glob.glob(os.path.join(data_dir_zh, "*.json")))
    shard_filenames = []
    if LANGUAGE=="en":
        assert len(shard_filenames_en)>0, f"Not found data in {data_dir_en}"
        shard_filenames = shard_filenames_en
    elif LANGUAGE=="zh":
        assert len(shard_filenames_zh) >0, f"Not found data in {data_dir_zh}"
        shard_filenames = shard_filenames_zh
    else:
        assert len(shard_filenames_zh) >0, f"Not found data in {data_dir_zh}"
        assert len(shard_filenames_en)>0, f"Not found data in {data_dir_en}"
        shard_filenames = shard_filenames_en + shard_filenames_zh
    
    # process all the shards in a threadpool
    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(process_shard, shard_filenames)
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

        data_dir_en = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data-en")
        data_dir_zh = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data-zh")
        shard_filenames_en = sorted(glob.glob(os.path.join(data_dir_en, "*.bin")))
        shard_filenames_zh = sorted(glob.glob(os.path.join(data_dir_zh, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        if LANGUAGE == "en":
            print(f"only using english dataset")
            shard_filenames = shard_filenames_en[1:] if self.split == "train" else shard_filenames_en[:1]
        elif LANGUAGE == "zh":
            print(f"only using chinese dataset")
            shard_filenames = shard_filenames_zh[1:] if self.split == "train" else shard_filenames_zh[:1]
        else:
            print(f"Using both in chinese and english dataset")
            shard_filenames = shard_filenames_en[1:]+shard_filenames_zh[1:] if \
                    self.split == "train" else shard_filenames_en[:1] + shard_filenames_zh[:1]
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
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


class Task:

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
        python tinystories.py download
        python tinystories.py pretokenize
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "train_tokenizer", "pretokenize"])
    args = parser.parse_args()
    fun = {
        "download": download,
        "pretokenize": pretokenize,
    }
    fun[args.stage]