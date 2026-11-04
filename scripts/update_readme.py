"""Rewrite the block between <!-- RESULTS:BEGIN --> and <!-- RESULTS:END --> in README.md from results/numbers.json."""
import json
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
N = json.load(open(os.path.join(ROOT, "results", "numbers.json")))
c = lambda s: s.replace("$\\pm$", "±")
rows = [("Uniform (baseline)", "Unif"), ("Antithetic ε", "Anti"), ("Stratified t", "Strat"),
        ("Importance t (loss-based)", "Imp"), ("Stratified t + antithetic ε", "SA")]
lines = ["| Estimator (CIFAR-10, 3,000 steps) | grad-variance ratio @3000 | held-out loss ×10⁻³ | FID-10k @3000 | ms/step |",
         "|---|---|---|---|---|"]
for name, t in rows:
    vr = "1 (reference)" if t == "Unif" else c(N.get(t + "VR", "–"))
    lines.append(f"| {name} | {vr} | {c(N[t + 'HO'])} | {c(N[t + 'FID'])} | {N[t + 'MS']} |")
lines.append(f"| Sobol t (probe only) | {c(N['SobolVR'])} | – | – | – |")
lines.append(f"| Oracle gradient-norm IS (predicted from decomposition) | {N['Decoracle']} | – | – | – |")
block = "\n".join(lines) + (
    f"\n\nMean ± sd over seeds (strat+anti FID: 2 seeds). No held-out-loss or FID difference is significant "
    f"(pre-registered Welch tests: smallest p = {N['ImppHO']} for held-out loss, {N['ImppFID']} for FID, both importance sampling). "
    f"Variance decomposition at step 3000: between-timestep {N['DecT']}%, data-given-t {N['DecX']}%, noise {N['DecE']}%; "
    f"antithetic correlation ρ = {N['Decrho']}. GPU used: {N['GPUhours']} of 2.0 GPU-h.")
p = os.path.join(ROOT, "README.md")
s = open(p).read()
s = re.sub(r"<!-- RESULTS:BEGIN -->.*<!-- RESULTS:END -->",
           "<!-- RESULTS:BEGIN -->\n" + block + "\n<!-- RESULTS:END -->", s, flags=re.S)
open(p, "w").write(s)
print("README results block updated")
