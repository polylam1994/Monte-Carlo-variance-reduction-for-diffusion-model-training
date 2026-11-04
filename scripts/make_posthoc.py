"""Summarise results/posthoc/*.json (CPU re-probes of saved Fashion-MNIST checkpoints; exploratory review fix)
into paper/tables/posthoc.tex and extra macros appended to paper/numbers.tex (run after make_numbers.py)."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analysis import LABEL_TEX, ROOT  # noqa: E402

fs = sorted(glob.glob(os.path.join(ROOT, "results", "posthoc", "*.json")))
if not fs:
    sys.exit("no posthoc results")
names = ["antithetic", "paired_iid", "stratified", "sobol", "importance", "is_strat", "strat_anti", "grad_is_oracle"]
lab = dict(LABEL_TEX, grad_is_oracle="Oracle gradient-norm IS")
per = {n: [] for n in names}      # per checkpoint: list of ratios over repeats
pred = {"stratified": [], "antithetic": [], "paired_iid": [], "strat_anti": [], "grad_is_oracle": []}
ub = []
for f in fs:
    d = json.load(open(f))
    for n in names:
        per[n].append([p[n]["trace_cov"] / p["uniform"]["trace_cov"] for p in d["probes"]])
    pr = d["decomp"]["pred"]; u = pr["uniform"]
    for k in pred:
        pred[k].append(pr["oracle_grad_IS" if k == "grad_is_oracle" else k] / u)
    for p in d["probes"]:
        ub += [v["dist2_to_uniform_mean"] / v["dist2_expected_if_unbiased"] for k, v in p.items()
               if k != "uniform" and v.get("dist2_expected_if_unbiased")]
rows, M = [], {}
cv = []
for n in names:
    A = np.array(per[n])                     # checkpoints x repeats
    m = A.mean(1); sd = A.std(1, ddof=1)     # spread across independent probe seeds, same theta
    cv += list(sd / m)
    pstr = " / ".join(f"{x:.2f}" for x in pred[n]) if n in pred else "--"
    rows.append(f"{lab[n]} & " + " & ".join(f"{a:.2f} $\\pm$ {b:.2f}" for a, b in zip(m, sd)) + f" & {pstr} \\\\")
tag = {"grad_is_oracle": "Oracle", "stratified": "Strat", "antithetic": "Anti", "importance": "Imp"}
for n, t in tag.items():
    A = np.array(per[n]); M["PH" + t] = f"{A.mean():.2f}"
    M["PH" + t + "min"], M["PH" + t + "max"] = f"{A.mean(1).min():.2f}", f"{A.mean(1).max():.2f}"
M["PHOraclePred"] = " / ".join(f"{x:.2f}" for x in pred["grad_is_oracle"])
M["PHcv"] = f"{100 * np.median(cv):.0f}"
M["PHcvmax"] = f"{100 * np.max(cv):.0f}"
M["PHn"] = str(len(fs)); M["PHrep"] = str(len(json.load(open(fs[0]))["probes"])); M["PHR"] = str(json.load(open(fs[0]))["R"])
M["PHub"] = f"{np.mean(ub):.2f}"
ckpts = [os.path.basename(f).replace(".json", "").replace("fmnist_uniform_", "seed ").replace("_step", ", step ") for f in fs]
head = "Estimator & " + " & ".join(ckpts) + " & predicted (decomp.) \\\\"
tex = ("\\begin{tabular}{l" + "r" * (len(fs) + 1) + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(rows) +
       "\n\\bottomrule\n\\end{tabular}\n")
open(os.path.join(ROOT, "paper", "tables", "posthoc.tex"), "w").write(tex)
with open(os.path.join(ROOT, "paper", "numbers.tex"), "a") as f:
    f.write("% post-hoc (scripts/make_posthoc.py)\n")
    for k in sorted(M):
        f.write(f"\\newcommand{{\\{k}}}{{{M[k]}}}\n")
nj = os.path.join(ROOT, "results", "numbers.json")
J = json.load(open(nj)); J.update(M); json.dump(J, open(nj, "w"), indent=1)
print(tex); print(M)
