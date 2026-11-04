"""Fetch CIFAR-10 (Krizhevsky 2009) from the Hugging Face mirror `uoft-cs/cifar10` (lossless PNGs of the
original images) and save uint8 arrays to <root>/cifar10_{train,test}.npz, which src/mcvr/data.py loads.
Used on Kaggle because the authors' server was too slow there (~90 kB/s) and Kaggle batch sessions
cannot attach Kaggle datasets. Locally, scripts/download_cifar10.py (torchvision) is an alternative.
python scripts/prepare_cifar10.py data/cifar10"""
import io
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
import pyarrow.parquet as pq

root = sys.argv[1] if len(sys.argv) > 1 else "data/cifar10"
os.makedirs(root, exist_ok=True)
files = list_repo_files("uoft-cs/cifar10", repo_type="dataset")
for split in ["train", "test"]:
    dst = os.path.join(root, f"cifar10_{split}.npz")
    if os.path.exists(dst):
        continue
    xs, ys = [], []
    for f in sorted(x for x in files if x.endswith(".parquet") and f"/{split}-" in x):
        t = pq.read_table(hf_hub_download("uoft-cs/cifar10", f, repo_type="dataset"))
        img_col = [c for c in t.column_names if "im" in c][0]
        lab_col = [c for c in t.column_names if "label" in c][0]
        xs += [np.array(Image.open(io.BytesIO(v["bytes"])).convert("RGB")) for v in t.column(img_col).to_pylist()]
        ys += t.column(lab_col).to_pylist()
    x, y = np.stack(xs), np.array(ys, dtype=np.int64)
    assert x.shape == ((50000 if split == "train" else 10000), 32, 32, 3) and x.dtype == np.uint8, x.shape
    np.savez(dst, x=x, y=y)
    print(split, x.shape, "mean", x.mean().round(3), "->", dst)
