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
    """Seed-0 FID curves (all checkpoints) + mean +/- sd over all seeds at the final step (PLAN A4)."""
    g = by_est(runs(ds))
    if not any(r["fid"] for rs in g.values() for r in rs):
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.6, 1]})
    for e in TRAINED:
        for r in g.get(e, []):
            if r["seed"] == 0 and len(r["fid"]) > 1:
                axes[0].plot([f["step"] for f in r["fid"]], [f["fid"] for f in r["fid"]], color=COLOR[e],
                             ls=LINESTYLE[e], marker=MARKER[e], ms=3.5, label=LABEL[e])
    axes[0].set_xlabel("optimiser steps (128 evaluations each)"); axes[0].set_ylabel("FID-10k")
    axes[0].set_title("seed 0, all checkpoints"); axes[0].legend(frameon=False)
    final = max(f["step"] for rs in g.values() for r in rs for f in r["fid"])
    for i, e in enumerate(TRAINED):
        v = [f["fid"] for r in g.get(e, []) for f in r["fid"] if f["step"] == final]
        if not v:
            continue
        m, sd = np.mean(v), (np.std(v, ddof=1) if len(v) > 1 else 0)
        axes[1].errorbar([i], [m], yerr=[sd], color=COLOR[e], marker=MARKER[e], ms=5, capsize=3, lw=1.4)
        axes[1].scatter([i] * len(v), v, color=COLOR[e], s=8, alpha=0.5, zorder=3)
    axes[1].set_xticks(range(len(TRAINED)))
    axes[1].set_xticklabels([LABEL[e].replace(" (baseline)", "").replace(" + ", "+\n") for e in TRAINED], fontsize=6.5)
    n = len(g.get("uniform", []))
    axes[1].set_title(f"step {final}: mean ± sd over {n} seeds (dots = seeds)")
    fig.suptitle("CIFAR-10 FID (clean-fid, 10k samples, DDIM-25, same initial noise for all models)", fontsize=8.5, y=1.02)
    fig.tight_layout()
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
    fig, axes = plt.subplots(1, len(dss), figsize=(3.6 * len(dss), 2.6), sharey=True, squeeze=False)
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
        ax.set_xlabel("variance ratio vs uniform (log; <1 is better)")
        ax.set_title({"fmnist": "Fashion-MNIST", "cifar10": "CIFAR-10"}[ds] + ", N = 128 evaluations")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), title="checkpoint", title_fontsize=7)
        ax.invert_yaxis()
    save(fig, "var_ratio")


def fig_per_t(ds, groups=5):
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
    for key, lab, c, ls in [("loss_mean", "mean", COLOR["antithetic"], "-"), ("loss_var", "variance", COLOR["stratified"], "--")]:
        A = arr(key); axes[0].plot(mid, A.mean(0), color=c, ls=ls, marker="o", ms=3, label=lab)
    axes[0].set_yscale("log"); axes[0].set_title("per-sample loss by timestep"); axes[0].legend(frameon=False)
    Vw, Ve, m2 = arr("Vw"), arr("Veps"), arr("mk_sqnorm")
    for A, lab, c, ls in [(Vw, "within-bin variance", COLOR["antithetic"], "-"),
                          (Ve, "ε-part $V_\\epsilon(t)$", COLOR["stratified"], "--"),
                          (np.clip(m2, 1e-12, None), "$\\|E[g|t]\\|^2$", COLOR["importance"], ":")]:
        axes[1].plot(mid, A.mean(0), color=c, ls=ls, marker="o", ms=3, label=lab)
    axes[1].set_yscale("log"); axes[1].set_title("per-sample gradient moments"); axes[1].legend(frameon=False)
    # antithetic correlation, pooled over bins within a group and over seeds (ratio of sums, less noisy)
    Va = arr("Vanti")
    G = np.array_split(np.arange(K), groups)
    gm = [mid[g].mean() for g in G]
    rho_pool = [1 - Va[:, g].sum() / Ve[:, g].sum() for g in G]
    rho_seed = np.array([[1 - Va[i, g].sum() / Ve[i, g].sum() for g in G] for i in range(len(decs))])
    axes[2].plot(gm, rho_pool, color=COLOR["antithetic"], marker="s", ms=4, label="pooled")
    if len(decs) > 1:
        for i in range(len(decs)):
            axes[2].plot(gm, rho_seed[i], color=COLOR["antithetic"], lw=0.6, alpha=0.5, label="per seed" if i == 0 else None)
    axes[2].axhline(0, color=INK, lw=0.8); axes[2].axhline(-1, color=MUTED, lw=0.6, ls=":")
    axes[2].set_ylim(-1.1, 0.5); axes[2].legend(frameon=False, loc="upper right")
    axes[2].set_title("antithetic correlation ρ(t)")
    for ax in axes:
        ax.set_xlabel("timestep t")
    fig.suptitle(f"{ds}, step {last}: per-timestep variance ({len(decs)} seed(s), {K} bins)", fontsize=8.5, y=1.03)
    fig.tight_layout()
    save(fig, f"per_t_{ds}")


if __name__ == "__main__":
    fig_fid()
    for ds in ["fmnist", "cifar10"]:
        fig_loss(ds); fig_per_t(ds)
    fig_var_ratio()
