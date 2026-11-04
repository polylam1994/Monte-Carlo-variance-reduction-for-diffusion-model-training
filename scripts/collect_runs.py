"""Collect every run directory under results/raw into results/runs.csv (one row per run).

Provenance (platform, Kaggle notebook URL + version) comes from results/sources.json:
{"results/raw/cifar10": {"platform": "kaggle", "url": "...", "version": 3}, ...}
keyed by directory prefix; the longest matching prefix wins."""
import csv
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
os.chdir(ROOT)
sources = json.load(open("results/sources.json")) if os.path.exists("results/sources.json") else {}


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


rows = []
for meta_p in sorted(glob.glob("results/raw/*/*/meta.json")):
    d = os.path.dirname(meta_p)
    meta = json.load(open(meta_p)); cfg = meta["cfg"]
    done = json.load(open(os.path.join(d, "done.json"))) if os.path.exists(os.path.join(d, "done.json")) else None
    status = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
    ev = jl(os.path.join(d, "evals.jsonl")); fid = jl(os.path.join(d, "fid.jsonl"))
    src = {}
    for k in sorted(sources, key=len):
        if d.startswith(k):
            src = sources[k]
    row = {"run_id": cfg["run_id"], "dataset": cfg["dataset"], "estimator": cfg["estimator"], "seed": cfg["seed"],
           "status": "finished" if done else f"incomplete(rc={status.get('rc', '?')})",
           "steps_done": done["step"] if done else (ev[-1]["step"] if ev else 0), "steps_planned": cfg["steps"],
           "batch": cfg["batch"], "n_params": meta["n_params"], "device": meta["gpu"], "torch": meta["torch"],
           "git": meta.get("git", ""), "train_time_s": round(done["train_time"], 1) if done else "",
           "wall_s": round(status.get("wall_s", 0), 1) if status else "",
           "ms_per_step": round(done["ms_per_step"], 2) if done else "",
           "heldout_final": ev[-1]["heldout"] if ev else "", "heldout_ema_final": ev[-1]["heldout_ema"] if ev else "",
           "fid": ";".join(f"{r['step']}:{r['fid']:.3f}" for r in fid),
           "platform": src.get("platform", "local-cpu" if meta["device"] == "cpu" else "unknown"),
           "notebook_url": src.get("url", ""), "notebook_version": src.get("version", ""),
           "run_dir": d}
    rows.append(row)
if rows:
    with open("results/runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
print(f"{len(rows)} runs -> results/runs.csv")
