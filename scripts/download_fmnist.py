"""Download Fashion-MNIST (MIT licence) from the official Zalando Research GitHub repository."""
import os
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = ["train-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"]

root = sys.argv[1] if len(sys.argv) > 1 else "data/fmnist"
os.makedirs(root, exist_ok=True)
for f in FILES:
    p = os.path.join(root, f)
    if not os.path.exists(p):
        print("downloading", f)
        urllib.request.urlretrieve(BASE + f, p)
print("done:", root)
