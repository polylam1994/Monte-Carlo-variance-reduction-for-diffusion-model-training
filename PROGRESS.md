# Progress log

GPU budget: 2.0 Kaggle GPU-hours (7,200 s of quota). Quota reading before the project: 63,896 s.
"GPU used" below = current quota reading − 63,896 s (the account may be used by other notebooks, so
this is an upper bound on this project's use).

## Phase 0–2 (2026-11-04)
* Literature check → `docs/related_work.md` (15 papers). Stratified/low-discrepancy t (VDM) and
  importance-sampled t (Improved DDPM, Song et al. 2021, Kim et al. 2025) are prior work; antithetic ε
  inside the *training* estimator was not found. Project framed as a controlled equal-compute comparison.
* `PLAN.md` committed before any run.
* CPU pilot (Fashion-MNIST, 60 steps, width 16) ran end to end: training, held-out CRN loss, gradient
  variance probe for 8 estimators, per-timestep decomposition (`results/raw/pilot`).
* Fashion-MNIST comparison (5 estimators × 3 seeds × 800 steps) started on the local CPU.
* Kaggle GPU smoke test (`mcvr-cifar10-smoke`, 3 × 400-step CIFAR-10 runs + one FID-10k) submitted.

## Iteration 1 — Phase 3 experiments (2026-11-04)
**Ran.**
* Kaggle `mcvr-assets` (CPU only, 0 GPU): verified that CIFAR-10 (HF mirror) and FID assets download quickly in a
  kernel and that the clean-fid path works (FID of 1k real test images vs train stats = 31.4).
* Kaggle `mcvr-cifar10-smoke` v1: **failed to train** — the 25-min session was spent downloading CIFAR-10 at
  ~90 kB/s (0.42 GPU-h lost). v2: full pipeline check and timings (200 ms/step at width 64; FID-2k 131 s).
* PLAN amendments A1–A3 (width 32, 3,000 steps, DDIM-25, EMA warm-up), committed before main runs.
* `mcvr-cifar10-seed0`: 4 runs (strat_anti held back by the deadline guard).
* PLAN amendment A4: seeds 1–2 with FID at the final step only (FID-10k costs ~180–200 s/run).
* `mcvr-cifar10-seeds12`: 10 runs (strat_anti seed 2 held back by the deadline guard).
* PLAN amendment A5; `mcvr-cifar10-sa-s2`: strat_anti seed 2, training only (no FID).
* Fashion-MNIST: 5 estimators × 3 seeds × 800 steps on the local CPU (in progress at time of writing).
**GPU used:** 7,092 s = **1.97 / 2.00 GPU-h** (results/budget.json). The GPU budget is exhausted; all further
work is CPU-only or analysis.
**Headline (CIFAR-10, 3 seeds):** stratified/Sobol t reduce gradient variance by ~15–20% at every checkpoint;
antithetic increases it at all but the last checkpoint, despite ρ≈−0.7, because pairs share (x0, t);
loss-based IS halves variance early but increases it late; no estimator changes FID or held-out loss
significantly (IS best on both, p≈0.07–0.24).
**Next:** finish Fashion-MNIST, write the paper, self-review.
