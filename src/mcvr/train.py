"""Train one DDPM with a given (x0, t, eps) estimator.  Resumable from out_dir/ckpt.pt.

Usage: python -m mcvr.train --config configs/cifar10.yaml --set estimator=antithetic seed=1
"""
import argparse
import copy
import json
import os
import platform
import subprocess
import time

import torch
import yaml

from . import data as D
from . import estimators as E
from . import probes as P
from .diffusion import Diffusion
from .unet import UNet


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for o in overrides:
        k, v = o.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    cfg.setdefault("run_id", f"{cfg['dataset']}_{cfg['estimator']}_s{cfg['seed']}")
    cfg["out_dir"] = os.path.join(cfg["out_dir"], cfg["run_id"])
    return cfg


def jdump(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


class Timer:
    """Wall-clock of optimiser steps only (evaluation/probe time is excluded)."""

    def __init__(self, cuda):
        self.cuda, self.total = cuda, 0.0

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def start(self):
        self.sync(); self.t0 = time.perf_counter()

    def stop(self):
        self.sync(); self.total += time.perf_counter() - self.t0


@torch.no_grad()
def heldout_loss(model, diff, test, crn, amp, bs=500):
    """Held-out L_simple with common random numbers: identical (x0, t, eps) for every model."""
    model.eval()
    tot = 0.0
    for i in range(0, len(crn["t"]), bs):
        sl = slice(i, i + bs)
        x0 = test.get(crn["idx"][sl])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            tot += float(diff.per_sample_loss(model, x0, crn["t"][sl], crn["eps"][sl]).sum())
    model.train()
    return tot / len(crn["t"])


def run(cfg):
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    cuda = dev.type == "cuda"
    amp = bool(cfg.get("amp", True)) and cuda
    torch.backends.cudnn.benchmark = True
    if not cuda and cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))

    # ---- data -------------------------------------------------------------------------
    train_x = D.load(cfg["dataset"], cfg["data_root"], "train", dev)
    test_x = D.load(cfg["dataset"], cfg["data_root"], "test", dev)
    train = D.ImageData(train_x, cfg.get("hflip", False))
    test = D.ImageData(test_x, False)
    g0 = torch.Generator(device="cpu"); g0.manual_seed(10_000)  # fixed across all runs (CRN)
    pool_idx = torch.randperm(train.n, generator=g0)[: cfg.get("probe_pool", 4096)].to(dev)
    pool = D.ImageData(train_x[pool_idx], cfg.get("hflip", False))
    n_ho = min(cfg.get("heldout_n", 2000), test.n)
    crn = {"idx": torch.randperm(test.n, generator=g0)[:n_ho].to(dev),
           "t": torch.randint(0, cfg["T"], (n_ho,), generator=g0).to(dev),
           "eps": torch.randn((n_ho,) + train.shape, generator=g0).to(dev)}

    # ---- model ------------------------------------------------------------------------
    torch.manual_seed(cfg["seed"])  # identical init for a given seed across estimators
    model = UNet(in_ch=train.shape[0], **cfg["model"]).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    diff = Diffusion(cfg["T"], device=dev)
    is_state = E.ISState(cfg["T"], K=cfg.get("is_bins", 50), history=cfg.get("is_history", 100),
                         mix=cfg.get("is_mix", 0.1), device=dev)
    est = E.make(cfg["estimator"], cfg["T"], train.n, train.shape, dev, is_state, seed=cfg["seed"])
    gen = torch.Generator(device=dev); gen.manual_seed(cfg["seed"] * 1_000_003 + 7)

    step, timer = 0, Timer(cuda)
    ck = os.path.join(out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        scaler.load_state_dict(s["scaler"]); is_state.load_state_dict(s["is_state"])
        gen.set_state(s["gen"].cpu()); step = s["step"]; timer.total = s["train_time"]
        if hasattr(est, "engine"):
            est.engine.fast_forward(step * cfg["batch"])
        print(f"[{cfg['run_id']}] resumed at step {step}")
    else:
        meta = {"cfg": cfg, "n_params": n_params, "device": str(dev),
                "gpu": torch.cuda.get_device_name(0) if cuda else platform.processor() or "cpu",
                "torch": torch.__version__, "python": platform.python_version(),
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            meta["git"] = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            meta["git"] = cfg.get("git", "unknown")
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)

    def save():
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "is_state": is_state.state_dict(),
                    "gen": gen.get_state(), "step": step, "train_time": timer.total}, ck + ".tmp")
        os.replace(ck + ".tmp", ck)

    def evaluate(step):
        rec = {"step": step, "train_time": timer.total,
               "heldout": heldout_loss(model, diff, test, crn, amp),
               "heldout_ema": heldout_loss(ema, diff, test, crn, amp)}
        jdump(os.path.join(out, "evals.jsonl"), rec)
        print(f"[{cfg['run_id']}] step {step} heldout {rec['heldout']:.5f} ema {rec['heldout_ema']:.5f} t={timer.total:.0f}s", flush=True)

    probe_at, fid_at = set(cfg.get("probe_at", [])), set(cfg.get("fid_at", []))
    decomp_at = set(cfg.get("decomp_at", []))
    eval_every, log_every, ck_every = cfg.get("eval_every", 500), cfg.get("log_every", 50), cfg.get("ckpt_every", 1000)

    def hooks(step):
        if step % eval_every == 0 or step == cfg["steps"]:
            evaluate(step)
        if step in probe_at:
            t0 = time.perf_counter()
            res = P.variance_probe(model, diff, pool, cfg["probe_estimators"], is_state, cfg["batch"],
                                   cfg["probe_R"], seed=50_000 + step, amp=amp)
            jdump(os.path.join(out, "probes.jsonl"), {"step": step, "res": res, "is_probs": is_state.probs().tolist(),
                                                      "seconds": time.perf_counter() - t0})
            base = res["uniform"]["trace_cov"]
            print(f"[{cfg['run_id']}] probe@{step}: " + " ".join(f"{k}={v['trace_cov'] / base:.3f}" for k, v in res.items()), flush=True)
        if step in decomp_at:
            t0 = time.perf_counter()
            dec = P.timestep_decomposition(model, diff, pool, cfg.get("decomp_bins", 20), cfg["batch"] // 2,
                                           cfg.get("decomp_reps", 2), seed=60_000 + step, amp=amp)
            dec["pred"] = P.predict_variances(dec, cfg["batch"], None)
            jdump(os.path.join(out, "decomp.jsonl"), {"step": step, "dec": dec, "seconds": time.perf_counter() - t0})
        if step in fid_at:
            from .fid import fid_eval
            t0 = time.perf_counter()
            r = fid_eval(ema, diff, cfg, dev, amp, os.path.join(out, f"samples_{step}.png"))
            r.update(step=step, train_time=timer.total, seconds=time.perf_counter() - t0)
            jdump(os.path.join(out, "fid.jsonl"), r)
            print(f"[{cfg['run_id']}] FID@{step} = {r['fid']:.2f} ({r['seconds']:.0f}s)", flush=True)

    def boundary(step):
        return (step % log_every == 0 or step % eval_every == 0 or step % ck_every == 0 or step == cfg["steps"]
                or step in probe_at or step in fid_at or step in decomp_at)

    if step == 0:
        hooks(0)
    model.train()
    timer.start()
    while step < cfg["steps"]:
        idx, flip, t, eps, w = est.sample(cfg["batch"], gen)
        x0 = train.get(idx, flip)
        lr = cfg["lr"] * min(1.0, (step + 1) / cfg.get("warmup", 200))
        for gparam in opt.param_groups:
            gparam["lr"] = lr
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            L = diff.per_sample_loss(model, x0, t, eps)
            loss = (w * L).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        scaler.step(opt); scaler.update()
        with torch.no_grad():
            d = cfg.get("ema", 0.999)
            if cfg.get("ema_warmup", False):  # standard warm-up so early checkpoints are not dominated by init
                d = min(d, (1 + step) / (10 + step))
            torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - d)
        is_state.update(t, L)  # every run tracks the IS proposal (used by probes; sampled only by IS)
        step += 1
        if boundary(step):
            timer.stop()  # synchronises; only optimiser-step time is accumulated
            if step % log_every == 0:
                jdump(os.path.join(out, "train.jsonl"), {"step": step, "loss": float(loss.detach()), "unweighted": float(L.detach().mean()),
                                                         "gnorm": float(gnorm), "train_time": timer.total})
            hooks(step)
            if step % ck_every == 0 or step == cfg["steps"]:
                save()
            timer.start()
    json.dump({"finished": True, "step": step, "train_time": timer.total,
               "ms_per_step": 1000 * timer.total / step, "end_time": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(os.path.join(out, "done.json"), "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    run(load_config(a.config, a.set))


if __name__ == "__main__":
    main()
