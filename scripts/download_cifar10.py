"""Download CIFAR-10 (Krizhevsky 2009) for local runs via torchvision (from the authors' server).
On Kaggle the kernels instead attach the dataset pankrzysiu/cifar10-python (identical original files)."""
import sys
import torchvision

root = sys.argv[1] if len(sys.argv) > 1 else "data/cifar10"
for train in (True, False):
    torchvision.datasets.CIFAR10(root, train=train, download=True)
print("done:", root)
