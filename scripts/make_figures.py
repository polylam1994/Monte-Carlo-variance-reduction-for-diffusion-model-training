"""Generate every paper figure from results/raw (never edited by hand).
python scripts/make_figures.py  ->  paper/figures/*.pdf (+ .png copies for the README)"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from analysis import (COLOR, LABEL, LINESTYLE, MARKER, PROBED, ROOT, TRAINED, by_est, runs,  # noqa: E402
                      var_ratios)

OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "lines.linewidth": 1.6, "savefig.dpi": 200, "font.family": "serif"})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def band(ax, groups, key_x, key_y, field, ests=TRAINED):
    """Mean line + min/max-free +/-1 std band across seeds, one line per estimator."""
    for e in ests:
        rs = groups.get(e, [])
        if not rs:
            continue
        series = [{r_[key_x]: r_[key_y] for r_ in r[field]} for r in rs]
        xs = sorted(set.intersection(*[set(s) for s in series]))
        Y = np.array([[s[x] for x in xs] for s in series])
        m = Y.mean(0)
        sd = Y.std(0, ddof=1) if len(rs) > 1 else np.zeros_like(m)
        ax.plot(xs, m, color=COLOR[e], ls=LINESTYLE[e], marker=MARKER[e], ms=3.5, label=f"{LABEL[e]} (n={len(rs)})")
        ax.fill_between(xs, m - sd, m + sd, color=COLOR[e], alpha=0.15, lw=0)


def fig_fid(ds="cifar10"):
    g = by_est(runs(ds))
    if not any(r["fid"] for rs in g.values() for r in rs):
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    band(ax, g, "step", "fid", "fid")
    ax.set_xlabel("optimiser steps (128 evaluations each)")
    ax.set_ylabel("FID-10k (clean-fid, DDIM-50)")
    ax.set_title("CIFAR-10: FID vs training steps (mean ± 1 sd over seeds)")
    ax.legend(frameon=False)
    save(fig, f"fid_vs_steps_{ds}")


def fig_loss(ds):
    g = by_est(runs(ds))
    if not g:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), sharey=True)
    band(axes[0], g, "step", "heldout", "evals")
    band(axes[1], g, "train_time", "heldout", "evals")
    # wall-clock: x differs per seed, so plot each seed thinly instead of a band
    axes[1].cla(); axes[1].grid(True)
    for e in TRAINED:
        for i, r in enumerate(g.get(e, [])):
            ev = [x for x in r["evals"] if x["step"] > 0]
            axes[1].plot([x["train_time"] for x in ev], [x["heldout"] for x in ev], color=COLOR[e],
                         ls=LINESTYLE[e], lw=1.0, label=LABEL[e] if i == 0 else None)
    for ax in axes:
        ax.set_yscale("log")
    axes[0].set_xlabel("optimiser steps"); axes[1].set_xlabel("training wall-clock (s, excl. evaluation)")
    axes[0].set_ylabel("held-out L_simple (CRN)")
    axes[0].set_title(f"{ds}: held-out loss vs steps"); axes[1].set_title("vs wall-clock (each seed)")
    axes[0].legend(frameon=False)
    save(fig, f"loss_curves_{ds}")


def fig_var_ratio():
    dss = [d for d in ["fmnist", "cifar10"] if any(r["probes"] for r in runs(d))]
    if not dss:
        return
    fig, axes = plt.subplots(1, len(dss), figsize=(3.4 * len(dss), 2.6), sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], dss):
        vr = var_ratios([r for r in runs(ds) if r["probes"]])
        steps = sorted({s for v in vr.values() for s, _, _ in v})
        names = [p for p in PROBED if p in vr and p != "uniform"]
        cmap = plt.get_cmap("Blues")
        for j, st in enumerate(steps):
            for i, nme in enumerate(names):
                ys = [r for s, _, r in vr[nme] if s == st]
                ax.scatter(ys, [i + (j - (len(steps) - 1) / 2) * 0.18] * len(ys), s=12,
                           color=cmap(0.4 + 0.6 * j / max(1, len(steps) - 1)), zorder=3,
                           label=f"step {st}" if i == 0 else None, edgecolor="white", lw=0.5)
        ax.axvline(1.0, color=INK, lw=1.0)
        ax.set_xscale("log")
        ax.set_yticks(range(len(names))); ax.set_yticklabels([LABEL[n] for n in names])
        ax.set_xlabel("tr Cov(ĝ) / tr Cov(ĝ_uniform)  (< 1 = less variance)")
        ax.set_title(f"{ds}: gradient-variance ratio at equal N=128")
        ax.legend(frameon=False, loc="lower right", title="checkpoint", title_fontsize=7)
        ax.invert_yaxis()
    save(fig, "var_ratio")


def fig_per_t(ds):
    rs = [r for r in runs(ds) if r["decomp"]]
    if not rs:
        return
    last = max(d["step"] for r in rs for d in r["decomp"])
    decs = [d["dec"] for r in rs for d in r["decomp"] if d["step"] == last]
    K = len(decs[0]["bins"])
    mid = np.array([(b["t_lo"] + b["t_hi"]) / 2 for b in decs[0]["bins"]])

    def arr(key):
        return np.array([[b[key] for b in d["bins"]] for d in decs])

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for key, lab, c in [("loss_mean", "mean", COLOR["antithetic"]), ("loss_var", "variance", COLOR["stratified"])]:
        A = arr(key); axes[0].plot(mid, A.mean(0), color=c, marker="o", ms=3, label=lab)
    axes[0].set_yscale("log"); axes[0].set_title("per-sample loss by timestep"); axes[0].legend(frameon=False)
    Vw, Ve, m2 = arr("Vw"), arr("Veps"), arr("mk_sqnorm")
    for A, lab, c, ls in [(Vw, "within-bin var. V_w(t)", COLOR["antithetic"], "-"),
                          (Ve, "ε-part V_ε(t)", COLOR["stratified"], "--"),
                          (np.clip(m2, 1e-12, None), "‖E[g|t]‖²", COLOR["importance"], ":")]:
        axes[1].plot(mid, A.mean(0), color=c, ls=ls, marker="o", ms=3, label=lab)
    axes[1].set_yscale("log"); axes[1].set_title("per-sample gradient moments"); axes[1].legend(frameon=False)
    rho = arr("rho")
    axes[2].plot(mid, rho.mean(0), color=COLOR["antithetic"], marker="s", ms=3)
    if len(decs) > 1:
        axes[2].fill_between(mid, rho.min(0), rho.max(0), color=COLOR["antithetic"], alpha=0.15, lw=0)
    axes[2].axhline(0, color=INK, lw=0.8); axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_title("antithetic correlation ρ(t)")
    for ax in axes:
        ax.set_xlabel("timestep t")
    fig.suptitle(f"{ds}, step {last}: where the variance lives ({len(decs)} seed(s), {K} bins)", fontsize=8.5, y=1.03)
    fig.tight_layout()
    save(fig, f"per_t_{ds}")


if __name__ == "__main__":
    fig_fid()
    for ds in ["fmnist", "cifar10"]:
        fig_loss(ds); fig_per_t(ds)
    fig_var_ratio()
