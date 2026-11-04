"""Shared loaders and statistics for figures/tables. Everything is read from results/raw."""
import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAINED = ["uniform", "antithetic", "stratified", "importance", "strat_anti"]
PROBED = ["uniform", "antithetic", "paired_iid", "stratified", "sobol", "importance", "is_strat", "strat_anti"]
LABEL = {"uniform": "Uniform (baseline)", "antithetic": "Antithetic ε", "paired_iid": "Paired i.i.d. ε (diag.)",
         "stratified": "Stratified t", "sobol": "Sobol t (RQMC)", "importance": "Importance t",
         "is_strat": "IS + stratified t", "strat_anti": "Stratified t + antithetic ε"}
# Colours: baseline in neutral ink, methods in the validated categorical order (dataviz reference palette).
COLOR = {"uniform": "#52514e", "antithetic": "#2a78d6", "stratified": "#eb6834", "importance": "#1baf7a",
         "strat_anti": "#eda100"}
MARKER = {"uniform": "o", "antithetic": "s", "stratified": "^", "importance": "D", "strat_anti": "v"}
LINESTYLE = {"uniform": "-", "antithetic": "--", "stratified": "-.", "importance": ":", "strat_anti": (0, (5, 1, 1, 1))}


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def runs(dataset, include_unfinished=False):
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "results/raw", dataset, "*"))):
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        done = os.path.exists(os.path.join(d, "done.json"))
        if not done and not include_unfinished:
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        r = {"dir": d, "cfg": meta["cfg"], "est": meta["cfg"]["estimator"], "seed": meta["cfg"]["seed"],
             "done": json.load(open(os.path.join(d, "done.json"))) if done else None,
             "evals": jl(os.path.join(d, "evals.jsonl")), "fid": jl(os.path.join(d, "fid.jsonl")),
             "probes": jl(os.path.join(d, "probes.jsonl")), "decomp": jl(os.path.join(d, "decomp.jsonl")),
             "train": jl(os.path.join(d, "train.jsonl"))}
        out.append(r)
    return out


def by_est(rs):
    d = {}
    for r in rs:
        d.setdefault(r["est"], []).append(r)
    return d


def mean_std(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def var_ratios(rs):
    """{estimator: [(step, seed, ratio), ...]} from all probes of the given runs (same theta per probe)."""
    out = {}
    for r in rs:
        for p in r["probes"]:
            base = p["res"]["uniform"]["trace_cov"]
            for k, v in p["res"].items():
                out.setdefault(k, []).append((p["step"], r["seed"], v["trace_cov"] / base))
    return out
LABEL_TEX = {k: v.replace("ε", "$\\epsilon$").replace("+", "+") for k, v in LABEL.items()}
