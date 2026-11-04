"""Post-hoc CPU analysis of a saved checkpoint (review fix; exploratory, not pre-registered).
1) Probe-noise check: repeat the gradient-variance probe with several independent probe seeds at the same theta.
2) Decomposition with more repetitions than in-run (tighter per-bin estimates).
3) 'Oracle' gradient-norm importance sampling: proposal q(bin) ∝ sqrt(E||g||^2 | bin) from (2), probed at the
   same theta, to test the predicted variance reduction directly (it is an oracle because q is estimated at the
   same parameters it is evaluated at).
python scripts/probe_checkpoint.py results/raw/fmnist/fmnist_uniform_s0 --repeats 4 --R 24 --reps 8 --bins 10"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mcvr import data as D  # noqa: E402
from mcvr import estimators as E  # noqa: E402
from mcvr import probes as P  # noqa: E402
from mcvr.diffusion import Diffusion  # noqa: E402
from mcvr.unet import UNet  # noqa: E402


class FixedProposal:
    """Stand-in for ISState with a fixed piecewise-constant proposal over K bins."""

    def __init__(self, probs):
        self.p = torch.as_tensor(probs, dtype=torch.float32)
        self.K = len(self.p)

    def probs(self):
        return self.p


ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--repeats", type=int, default=4)
ap.add_argument("--R", type=int, default=24)
ap.add_argument("--reps", type=int, default=8)
ap.add_argument("--bins", type=int, default=10)
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
torch.set_num_threads(a.threads)

meta = json.load(open(os.path.join(a.run_dir, "meta.json"))); cfg = meta["cfg"]
dev = torch.device("cpu")
train_x = D.load(cfg["dataset"], cfg["data_root"], "train", dev)
g0 = torch.Generator(device="cpu"); g0.manual_seed(10_000)  # same probe pool as in training
pool_idx = torch.randperm(train_x.shape[0], generator=g0)[: cfg.get("probe_pool", 4096)]
pool = D.ImageData(train_x[pool_idx], cfg.get("hflip", False))
model = UNet(in_ch=pool.shape[0], **cfg["model"])
ck = torch.load(os.path.join(a.run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"]); step = ck["step"]
diff = Diffusion(cfg["T"])
is_state = E.ISState(cfg["T"], K=cfg.get("is_bins", 50), history=cfg.get("is_history", 100), mix=cfg.get("is_mix", 0.1))
is_state.load_state_dict(ck["is_state"])
N = cfg["batch"]
out = {"run": cfg["run_id"], "step": step, "R": a.R, "repeats": a.repeats, "reps": a.reps, "bins": a.bins}
t0 = time.time()
dec = P.timestep_decomposition(model, diff, pool, a.bins, N // 2, a.reps, seed=90_000, amp=False)
dec["pred"] = P.predict_variances(dec, N, None)
out["decomp"] = dec
second = torch.tensor([b["Vw"] + max(b["mk_sqnorm"], 0.0) for b in dec["bins"]]).clamp(min=1e-12).sqrt()
q_star = (second / second.sum()).tolist()
out["q_star"] = q_star
print("decomposition done", round(time.time() - t0), "s; predicted oracle ratio",
      round(dec["pred"]["oracle_grad_IS"] / dec["pred"]["uniform"], 3), flush=True)

names = ["uniform", "antithetic", "paired_iid", "stratified", "sobol", "importance", "is_strat", "strat_anti"]
reps_out = []
for k in range(a.repeats):
    res = P.variance_probe(model, diff, pool, names, is_state, N, a.R, seed=70_000 + 1000 * k, amp=False)
    # oracle gradient-IS (fixed proposal from the decomposition); reuse the probe machinery
    orc = P.variance_probe(model, diff, pool, ["uniform", "importance"], FixedProposal(q_star), N, a.R,
                           seed=70_000 + 1000 * k, amp=False)
    res["grad_is_oracle"] = orc["importance"]
    reps_out.append(res)
    base = res["uniform"]["trace_cov"]
    print(f"repeat {k}: " + " ".join(f"{n}={v['trace_cov'] / base:.3f}" for n, v in res.items()),
          round(time.time() - t0), "s", flush=True)
out["probes"] = reps_out
dst = os.path.join(os.path.dirname(__file__), "..", "results", "posthoc")
os.makedirs(dst, exist_ok=True)
json.dump(out, open(os.path.join(dst, f"{cfg['run_id']}_step{step}.json"), "w"))
print("saved", flush=True)
