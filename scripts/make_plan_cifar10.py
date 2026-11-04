"""Write configs/plans/cifar10_main.json: 5 trained estimators x 3 seeds, balanced over 2 GPU queues.
Probes (all 8 estimators at the same parameters) and the per-timestep decomposition run inside the
uniform-baseline runs only, so every estimator is evaluated at identical checkpoints.
python scripts/make_plan_cifar10.py STEPS OUT SEED [SEED ...]"""
import json
import sys

S, OUT, SEEDS = int(sys.argv[1]), sys.argv[2], [int(x) for x in sys.argv[3:]]
fid_at = [S // 3, 2 * S // 3, S]
probe_at = [S // 12, S // 3, 2 * S // 3, S]
runs = []
for seed in SEEDS:
    for e in ["uniform", "antithetic", "stratified", "importance", "strat_anti"]:
        st = [f"estimator={e}", f"seed={seed}", f"steps={S}", f"fid_at={fid_at}", f"eval_every={S // 12}",
              f"ckpt_every={S // 6}"]
        st += [f"probe_at={probe_at}", f"decomp_at=[{S // 3}, {S}]"] if e == "uniform" else []
        runs.append({"config": "configs/cifar10.yaml", "set": st})
# balance two GPU queues: greedy by estimated cost (uniform runs carry the probes, ~1.4x)
load = [0.0, 0.0]
for r in sorted(runs, key=lambda r: -(1.4 if "estimator=uniform" in r["set"] else 1.0)):
    k = load.index(min(load)); r["slot"] = k; load[k] += 1.4 if "estimator=uniform" in r["set"] else 1.0
runs.sort(key=lambda r: (r["slot"], [x for x in r["set"] if x.startswith("seed=")][0]))
json.dump({"runs": runs}, open(OUT, "w"), indent=1)
print(len(runs), "runs; fid_at", fid_at, "probe_at", probe_at)
