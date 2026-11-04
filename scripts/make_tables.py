"""Generate every paper table (LaTeX) and results/summary.json from results/raw.
python scripts/make_tables.py -> paper/tables/*.tex, results/summary.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analysis import LABEL, PROBED, ROOT, TRAINED, by_est, mean_std, runs, var_ratios, welch  # noqa: E402

OUT = os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)
summary = {}


def fmt(m, s, d=3):
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}" if np.isfinite(s) else f"{m:.{d}f}"


def pfmt(p):
    return "--" if not np.isfinite(p) else (f"{p:.2f}" if p >= 0.01 else f"{p:.0e}")


def main_table(ds):
    g = by_est(runs(ds))
    if "uniform" not in g:
        return
    base_ms = np.mean([r["done"]["ms_per_step"] for r in g["uniform"]])
    fid_steps = sorted({f["step"] for rs in g.values() for r in rs for f in r["fid"]})
    final_fid_step = fid_steps[-1] if fid_steps else None
    base_ho = [r["evals"][-1]["heldout"] for r in g["uniform"]]
    base_fid = [f["fid"] for r in g["uniform"] for f in r["fid"] if f["step"] == final_fid_step]
    rows, S = [], {}
    for e in TRAINED:
        if e not in g:
            continue
        rs = g[e]
        ms = [r["done"]["ms_per_step"] for r in rs]
        ho = [r["evals"][-1]["heldout"] for r in rs]
        hoe = [r["evals"][-1]["heldout_ema"] for r in rs]
        S[e] = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs], "ms_per_step": mean_std(ms),
                "overhead_pct": 100 * (np.mean(ms) / base_ms - 1), "heldout_final": mean_std(ho),
                "heldout_ema_final": mean_std(hoe), "p_heldout_vs_uniform": welch(ho, base_ho),
                "steps": rs[0]["cfg"]["steps"]}
        cells = [LABEL[e], str(len(rs)), f"{np.mean(ms):.1f}", f"{S[e]['overhead_pct']:+.1f}\\%",
                 fmt(*mean_std(ho), 4), pfmt(S[e]["p_heldout_vs_uniform"]) if e != "uniform" else "--"]
        if final_fid_step:
            fd = {st: [f["fid"] for r in rs for f in r["fid"] if f["step"] == st] for st in fid_steps}
            S[e]["fid"] = {str(st): mean_std(v) for st, v in fd.items() if v}
            S[e]["p_fid_final_vs_uniform"] = welch(fd[final_fid_step], base_fid)
            cells += [fmt(*mean_std(fd[st]), 2) if fd[st] else "--" for st in fid_steps]
            cells += [pfmt(S[e]["p_fid_final_vs_uniform"]) if e != "uniform" else "--"]
        rows.append(" & ".join(cells) + r" \\")
    head = ["Estimator", "seeds", "ms/step", "overhead", "held-out loss", "$p$"]
    if final_fid_step:
        head += [f"FID@{st}" for st in fid_steps] + ["$p$ (FID)"]
    tex = ("\\begin{tabular}{l" + "r" * (len(head) - 1) + "}\n\\toprule\n" + " & ".join(head) + r" \\" +
           "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, f"main_{ds}.tex"), "w").write(tex)
    summary[ds] = {"trained": S, "fid_steps": fid_steps}
    print(f"wrote main_{ds}.tex")


def var_table():
    lines, S = [], {}
    dss = [d for d in ["fmnist", "cifar10"] if any(r["probes"] for r in runs(d))]
    if not dss:
        return
    per = {}
    for ds in dss:
        vr = var_ratios([r for r in runs(ds) if r["probes"]])
        steps = sorted({s for v in vr.values() for s, _, _ in v})
        S[ds] = {}
        for e in PROBED:
            if e not in vr:
                continue
            S[ds][e] = {str(st): mean_std([r for s, _, r in vr[e] if s == st]) for st in steps}
            S[ds][e]["all_below_1"] = bool(all(r < 1 for _, _, r in vr[e]))
            S[ds][e]["n"] = len(vr[e])
        per[ds] = steps
    head = ["Estimator"] + [f"{ds} @{st}" for ds in dss for st in per[ds]]
    for e in PROBED:
        if e == "uniform":
            continue
        cells = [LABEL[e]]
        for ds in dss:
            for st in per[ds]:
                m, s = S[ds].get(e, {}).get(str(st), (np.nan, np.nan))
                cells.append(fmt(m, s, 2))
        lines.append(" & ".join(cells) + r" \\")
    tex = ("\\begin{tabular}{l" + "r" * (len(head) - 1) + "}\n\\toprule\n" + " & ".join(head) + r" \\" +
           "\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(OUT, "var_ratio.tex"), "w").write(tex)
    summary["var_ratio"] = S
    print("wrote var_ratio.tex")


def decomp_table():
    S = {}
    for ds in ["fmnist", "cifar10"]:
        rs = [r for r in runs(ds) if r["decomp"]]
        if not rs:
            continue
        S[ds] = []
        for r in rs:
            for d in r["decomp"]:
                dec = d["dec"]; b = dec["bins"]
                Vw = np.mean([x["Vw"] for x in b]); Ve = np.mean([x["Veps"] for x in b])
                tot = Vw + dec["V_between"]
                rho_w = np.sum([x["rho"] * x["Veps"] for x in b]) / np.sum([x["Veps"] for x in b])
                S[ds].append({"seed": r["seed"], "step": d["step"], "V_total": tot,
                              "frac_between_t": dec["V_between"] / tot, "frac_eps": Ve / tot,
                              "frac_x0_given_t": (Vw - Ve) / tot, "rho_weighted": rho_w,
                              "pred": dec.get("pred", {})})
    summary["decomp"] = S


if __name__ == "__main__":
    for ds in ["fmnist", "cifar10"]:
        main_table(ds)
    var_table(); decomp_table()
    json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1)
    print("wrote results/summary.json")
