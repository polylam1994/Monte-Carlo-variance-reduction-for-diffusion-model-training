"""Datasets held entirely on the training device as uint8 tensors (fast random indexing)."""
import gzip
import os
import numpy as np
import torch


def _fmnist(root, split):
    pre = "train" if split == "train" else "t10k"
    with gzip.open(os.path.join(root, f"{pre}-images-idx3-ubyte.gz"), "rb") as f:
        x = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    return torch.from_numpy(x.copy())


def _cifar10(root, split):
    npz = os.path.join(root, f"cifar10_{split}.npz")  # written by scripts/prepare_cifar10.py
    if os.path.exists(npz):
        x = np.load(npz)["x"]
    else:  # torchvision layout, see scripts/download_cifar10.py
        import torchvision
        x = torchvision.datasets.CIFAR10(root, train=(split == "train"), download=False).data
    return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()


def load(name, root, split, device):
    x = {"fmnist": _fmnist, "cifar10": _cifar10}[name](root, split)
    return x.to(device)


class ImageData:
    """uint8 images on device; `get(idx, flip)` returns float images in [-1, 1].

    Random horizontal flips (if enabled) are drawn per *distinct* x0 by the caller so that an
    antithetic pair shares exactly the same x0."""

    def __init__(self, x, hflip):
        self.x, self.hflip = x, hflip
        self.n, self.shape = x.shape[0], tuple(x.shape[1:])

    def get(self, idx, flip=None):
        x = self.x[idx].float().div_(127.5).sub_(1.0)
        if self.hflip and flip is not None:
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return x
