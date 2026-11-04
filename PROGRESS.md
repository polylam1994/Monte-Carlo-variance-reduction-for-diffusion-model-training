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
