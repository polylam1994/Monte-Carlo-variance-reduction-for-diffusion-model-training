"""CPU-only Kaggle kernel (no GPU quota): check that CIFAR-10 (via kagglehub) and the FID assets can be
fetched quickly inside a kernel, and smoke-test the FID code path used by src/mcvr/fid.py."""
import glob, os, shutil, subprocess, sys, time
os.chdir("/kaggle/working")
t0 = time.time()
subprocess.call("pip install -q clean-fid==0.1.35 2>&1 | tail -1", shell=True)
print("pip", round(time.time() - t0), "s", flush=True)
import kagglehub
t0 = time.time(); root = kagglehub.dataset_download("pankrzysiu/cifar10-python")
print("kagglehub", kagglehub.__version__, root, round(time.time() - t0), "s", flush=True)
subprocess.call(f"ls -la {root} {root}/*", shell=True)
d = None
for dp, dns, fns in os.walk(root, followlinks=True):
    if os.path.basename(dp) == "cifar-10-batches-py":
        d = dp; break
print("cifar dir:", d, flush=True)
subprocess.call(f"md5sum {d}/*", shell=True)
import numpy as np, torch, torchvision, cleanfid
shutil.copytree(d, "data/cifar10/cifar-10-batches-py")
ds = torchvision.datasets.CIFAR10("data/cifar10", train=True, download=False)
print("torchvision ok", ds.data.shape, flush=True)
from cleanfid import fid
from cleanfid.features import build_feature_extractor
t0 = time.time(); build_feature_extractor("clean", torch.device("cpu"), use_dataparallel=False)
print("inception download", round(time.time() - t0), "s", flush=True)
t0 = time.time(); mu, sig = fid.get_reference_statistics("cifar10", 32, mode="clean", split="train")
print("ref stats", round(time.time() - t0), "s", mu.shape, sig.shape, flush=True)
import pickle
x = pickle.load(open(os.path.join(d, "test_batch"), "rb"), encoding="bytes")[b"data"][:1000]
imgs = torch.from_numpy(x.reshape(-1, 3, 32, 32))
state = {"i": 0}
def gen_fn(z):
    b = z.shape[0]; out = imgs[state["i"]: state["i"] + b]; state["i"] += b; return out
t0 = time.time()
s = fid.compute_fid(gen=gen_fn, dataset_name="cifar10", dataset_res=32, dataset_split="train", mode="clean",
                    num_gen=1000, batch_size=500, z_dim=1, device=torch.device("cpu"), verbose=False,
                    use_dataparallel=False)
print("FID(1000 real test images vs train stats) =", s, round(time.time() - t0), "s", flush=True)
shutil.rmtree("data")
