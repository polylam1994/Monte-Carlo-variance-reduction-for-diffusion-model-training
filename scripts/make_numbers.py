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
        # secondary (not pre-registered): paired by seed -- same initialisation and same evaluation noise
        bmap = {r["seed"]: r["evals"][-1]["heldout"] for r in g["uniform"]}
        dd = [100 * (r["evals"][-1]["heldout"] / bmap[r["seed"]] - 1) for r in rs if r["seed"] in bmap]
        if e != "uniform" and len(dd) > 1:
            from scipy import stats as _st
            M[t + "HOpair"] = pm(*mean_std(dd), 1)
            M[t + "pHOpair"] = p_str(float(_st.ttest_1samp(dd, 0).pvalue))
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
            bmapf = {r["seed"]: f["fid"] for r in g["uniform"] for f in r["fid"] if f["step"] == final}
            ddf = [f["fid"] - bmapf[r["seed"]] for r in rs for f in r["fid"] if f["step"] == final and r["seed"] in bmapf]
            if e != "uniform" and len(ddf) > 1:
                from scipy import stats as _st
                M[t + "FIDpair"] = pm(*mean_std(ddf), 1)
                M[t + "pFIDpair"] = p_str(float(_st.ttest_1samp(ddf, 0).pvalue))
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
            # robust summary over independent probes (different seeds/checkpoints/probe RNG): geometric mean with a
            # 95% t-interval on log ratios, and a two-sided sign test of "ratio < 1"
            from scipy import stats as _st
            lr = np.log(allr); k = len(lr)
            half = _st.t.ppf(0.975, k - 1) * lr.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
            M[t + "VRci"] = f"[{np.exp(lr.mean() - half):.2f}, {np.exp(lr.mean() + half):.2f}]"
            M[t + "VRsign"] = p_str(float(_st.binomtest(int(sum(r < 1 for r in allr)), k, 0.5).pvalue))
        # antithetic vs its paired-i.i.d. control at the same probe
        pa = {(s_, sd): r for s_, sd, r in vr.get("paired_iid", [])}
        an = [(r, pa[(s_, sd)]) for s_, sd, r in vr.get("antithetic", []) if (s_, sd) in pa]
        if an:
            M[pre + "AntiBeatsPaired"] = f"{sum(a < b for a, b in an)}/{len(an)}"
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

# Minimum detectable effect (two-sided two-sample t-test, alpha 0.05, power 0.8, n per arm), using the pooled
# within-arm sd across all trained arms. Solved with the non-central t distribution.
from scipy import stats as _st


def mde_d(n, alpha=0.05, power=0.8):
    df = 2 * n - 2; tc = _st.t.ppf(1 - alpha / 2, df)
    lo, hi = 0.01, 20.0
    for _ in range(100):
        d = (lo + hi) / 2
        ncp = d * np.sqrt(n / 2)
        pw = 1 - _st.nct.cdf(tc, df, ncp) + _st.nct.cdf(-tc, df, ncp)
        lo, hi = (d, hi) if pw < power else (lo, d)
    return d


g = by_est(runs("cifar10"))
d3 = mde_d(3)
M["MDEd"] = f"{d3:.1f}"
sd_ho = np.sqrt(np.mean([np.var([r["evals"][-1]["heldout"] for r in rs], ddof=1) for rs in g.values() if len(rs) > 1]))
base_ho = np.mean([r["evals"][-1]["heldout"] for r in g["uniform"]])
M["MDEho"] = f"{100 * d3 * sd_ho / base_ho:.1f}"
sd_fid = np.sqrt(np.mean([np.var([f["fid"] for r in rs for f in r["fid"] if f["step"] == 3000], ddof=1)
                          for rs in g.values() if sum(1 for r in rs for f in r["fid"] if f["step"] == 3000) > 1]))
M["MDEfid"] = f"{d3 * sd_fid:.0f}"

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
