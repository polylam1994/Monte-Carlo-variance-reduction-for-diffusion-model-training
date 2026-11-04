"""Write paper/numbers.tex: every number quoted in the paper text, as a LaTeX macro, computed from
results/raw and results/budget.json (run after make_tables.py). Nothing in the text is typed by hand."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analysis import PROBED, ROOT, TRAINED, by_est, mean_std, runs, var_ratios, welch  # noqa: E402

M = {}
TAG = {"uniform": "Unif", "antithetic": "Anti", "stratified": "Strat", "importance": "Imp", "strat_anti": "SA",
       "sobol": "Sobol", "paired_iid": "Paired", "is_strat": "ISS"}
DS = {"cifar10": "", "fmnist": "F"}


def pm(m, s, d):
    return f"{m:.{d}f} $\\pm$ {s:.{d}f}" if np.isfinite(s) else f"{m:.{d}f}"


def p_str(p):
    return "n/a" if not np.isfinite(p) else (f"{p:.2f}" if p >= 0.01 else f"{p:.3f}")


# ---- budget -----------------------------------------------------------------------------------
B = json.load(open(os.path.join(ROOT, "results", "budget.json")))
used = B["readings"][-1]["time_used_s"] - B["baseline_s"]
M["GPUhours"] = f"{used / 3600:.2f}"
M["GPUhoursLost"] = f"{B['lost_to_failed_download_s'] / 3600:.2f}"

for ds, pre in DS.items():
    g = by_est(runs(ds))
    if "uniform" not in g:
        continue
    base_ho = [r["evals"][-1]["heldout"] for r in g["uniform"]]
    base_ms = np.mean([r["done"]["ms_per_step"] for r in g["uniform"]])
    M[pre + "Steps"] = str(g["uniform"][0]["cfg"]["steps"])
    for e in TRAINED:
        if e not in g:
            continue
        rs, t = g[e], pre + TAG[e]
        ho = [r["evals"][-1]["heldout"] for r in rs]
        M[t + "Seeds"] = str(len(rs))
        M[t + "HO"] = pm(*mean_std([1e3 * x for x in ho]), 2)  # x 1e-3
        M[t + "HOdiff"] = f"{100 * (np.mean(ho) / np.mean(base_ho) - 1):+.1f}"
        M[t + "pHO"] = p_str(welch(ho, base_ho))
        ms = np.mean([r["done"]["ms_per_step"] for r in rs])
        M[t + "MS"] = f"{ms:.1f}"
        M[t + "Ovh"] = f"{100 * (ms / base_ms - 1):+.1f}"
        if ds == "cifar10":
            final = max(f["step"] for r in rs for f in r["fid"])
            fd = [f["fid"] for r in rs for f in r["fid"] if f["step"] == final]
            bfd = [f["fid"] for r in g["uniform"] for f in r["fid"] if f["step"] == final]
            M[t + "FID"] = pm(*mean_std(fd), 1)
            M[t + "FIDdiff"] = f"{np.mean(fd) - np.mean(bfd):+.1f}"
            M[t + "pFID"] = p_str(welch(fd, bfd))
    # variance ratios
    vr = var_ratios([r for r in runs(ds) if r["probes"]])
    if vr:
        steps = sorted({s for v in vr.values() for s, _, _ in v})
        M[pre + "ProbeSteps"] = ", ".join(map(str, steps))
        M[pre + "ProbeN"] = str(len({(s, sd) for s, sd, _ in vr["uniform"]}))
        for e in PROBED:
            if e == "uniform" or e not in vr:
                continue
            t = pre + TAG[e]
            allr = [r for _, _, r in vr[e]]
            fin = [r for s, _, r in vr[e] if s == steps[-1]]
            M[t + "VR"] = pm(*mean_std(fin), 2)
            M[t + "VRall"] = f"{np.exp(np.mean(np.log(allr))):.2f}"   # geometric mean over all probes
            M[t + "VRmin"], M[t + "VRmax"] = f"{min(allr):.2f}", f"{max(allr):.2f}"
            M[t + "VRbelow"] = f"{sum(r < 1 for r in allr)}/{len(allr)}"
        ub = [v["dist2_to_uniform_mean"] / v["dist2_expected_if_unbiased"] for r in runs(ds) for p in r["probes"]
              for k, v in p["res"].items() if k != "uniform"]
        M[pre + "UnbiasN"], M[pre + "UnbiasMean"] = str(len(ub)), f"{np.mean(ub):.2f}"
        M[pre + "UnbiasMin"], M[pre + "UnbiasMax"] = f"{min(ub):.2f}", f"{max(ub):.2f}"
    # decomposition at the last step (all seeds)
    decs = [(r["seed"], d) for r in runs(ds) for d in r["decomp"]]
    if decs:
        last = max(d["step"] for _, d in decs)
        L = [d["dec"] for _, d in decs if d["step"] == last]
        fr = {"T": [], "X": [], "E": [], "rho": [], "oracle": [], "pAnti": [], "pStrat": [], "pSA": [], "pPaired": []}
        for dec in L:
            b = dec["bins"]; Vw = np.mean([x["Vw"] for x in b]); Ve = np.mean([x["Veps"] for x in b])
            tot = Vw + dec["V_between"]
            fr["T"].append(dec["V_between"] / tot); fr["E"].append(Ve / tot); fr["X"].append((Vw - Ve) / tot)
            fr["rho"].append(1 - sum(x["Vanti"] for x in b) / sum(x["Veps"] for x in b))
            p = dec["pred"]; u = p["uniform"]
            fr["oracle"].append(p["oracle_grad_IS"] / u); fr["pAnti"].append(p["antithetic"] / u)
            fr["pStrat"].append(p["stratified"] / u); fr["pSA"].append(p["strat_anti"] / u)
            fr["pPaired"].append(p["paired_iid"] / u)
        M[pre + "DecStep"], M[pre + "DecN"] = str(last), str(len(L))
        for k, v in fr.items():
            m, s = mean_std(v)
            M[pre + "Dec" + k] = f"{100 * m:.0f}" if k in "TXE" else f"{m:.2f}"
        # condition for antithetic: V_t + V_x  vs  -rho V_eps (fractions of total)
        lhs = np.mean(np.array(fr["T"]) + np.array(fr["X"])); rhs = np.mean(-np.array(fr["rho"]) * np.array(fr["E"]))
        M[pre + "DecLHS"], M[pre + "DecRHS"] = f"{lhs:.2f}", f"{rhs:.2f}"
        # loss and gradient dynamic range across timesteps (max/min over bins)
        lm = np.mean([[x["loss2_mean"] for x in dec["bins"]] for dec in L], 0)
        gm = np.mean([[x["Vw"] + max(x["mk_sqnorm"], 0) for x in dec["bins"]] for dec in L], 0)
        M[pre + "LossRange"] = f"{np.sqrt(lm.max() / lm.min()):.0f}"
        M[pre + "GradRange"] = f"{np.sqrt(gm.max() / gm.min()):.0f}"

# Merge unbiasedness stats over datasets for the setup paragraph
ubs = [v["dist2_to_uniform_mean"] / v["dist2_expected_if_unbiased"] for ds in DS for r in runs(ds) for p in r["probes"]
       for k, v in p["res"].items() if k != "uniform"]
M["UnbiasN"], M["UnbiasMean"] = str(len(ubs)), f"{np.mean(ubs):.2f}"
M["UnbiasMin"], M["UnbiasMax"] = f"{min(ubs):.2f}", f"{max(ubs):.2f}"

with open(os.path.join(ROOT, "paper", "numbers.tex"), "w") as f:
    f.write("% AUTO-GENERATED by scripts/make_numbers.py from results/ -- do not edit\n")
    for k in sorted(M):
        assert k.isalpha(), k
        f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
json.dump(M, open(os.path.join(ROOT, "results", "numbers.json"), "w"), indent=1)
print(len(M), "macros -> paper/numbers.tex")
