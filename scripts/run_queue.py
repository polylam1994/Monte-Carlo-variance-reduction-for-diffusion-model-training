"""Run a list of training jobs, one sequential queue per GPU (or a single CPU queue).

plan JSON: {"runs": [{"config": "configs/cifar10.yaml", "set": ["estimator=uniform", "seed=0"]}, ...]}
Jobs whose out_dir already contains done.json are skipped; unfinished jobs resume from ckpt.pt.
No new job is started after --deadline seconds (a guard on the compute budget)."""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mcvr.train import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--gpus", default="")  # e.g. "0,1"; empty -> CPU
ap.add_argument("--deadline", type=float, default=1e9)
ap.add_argument("--extra", nargs="*", default=[])
a = ap.parse_args()

runs = json.load(open(a.plan))["runs"]
T0 = time.time()
slots = a.gpus.split(",") if a.gpus else ["cpu"]
queues = [runs[i::len(slots)] for i in range(len(slots))]
os.makedirs("logs", exist_ok=True)


def worker(slot, q):
    for r in q:
        sets = r["set"] + a.extra
        cfg = load_config(r["config"], sets)
        out = cfg["out_dir"]
        if os.path.exists(os.path.join(out, "done.json")):
            print(f"[{slot}] skip {cfg['run_id']} (done)", flush=True); continue
        if time.time() - T0 > a.deadline:
            print(f"[{slot}] deadline reached, not starting {cfg['run_id']}", flush=True); continue
        env = dict(os.environ, PYTHONPATH=os.path.join(os.getcwd(), "src"))
        if slot != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = slot
        cmd = [sys.executable, "-m", "mcvr.train", "--config", r["config"], "--set", *sets]
        t0 = time.time()
        with open(os.path.join("logs", cfg["run_id"] + ".log"), "a") as lf:
            rc = subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        print(f"[{slot}] {cfg['run_id']} rc={rc} {time.time() - t0:.0f}s", flush=True)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "status.json"), "w") as f:
            json.dump({"rc": rc, "wall_s": time.time() - t0}, f)


ths = [threading.Thread(target=worker, args=(s, q)) for s, q in zip(slots, queues)]
[t.start() for t in ths]
[t.join() for t in ths]
print(f"queue finished in {time.time() - T0:.0f}s")
