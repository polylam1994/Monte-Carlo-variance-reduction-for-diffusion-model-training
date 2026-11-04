# PLAN — Monte Carlo variance reduction for diffusion-model training

Committed before any experiment was run (Phase 1). Changes made after the pilot are
recorded in the *Amendments* section at the bottom. Nothing above that line is edited
after the fact.

## 1. Research question
Can classical Monte Carlo variance-reduction techniques used in computational finance
(antithetic variates, stratification / quasi-Monte Carlo, importance sampling) reduce the
variance of the DDPM denoising score-matching (`L_simple`) loss **gradient estimator**,
and does that turn into faster or better training **at equal compute**?

## 2. Estimators compared (only the estimator changes; objective, model and optimiser are fixed)
Each optimiser step uses exactly N = 128 (x0, t, ε) evaluations (an antithetic pair counts as two).
All estimators are unbiased for the same objective E_{x0,t~U,ε}[‖ε̂_θ(x_t,t) − ε‖²/d].

| id | name | x0 | t | ε |
|----|------|----|---|---|
| `uniform` | baseline | N i.i.d. | N i.i.d. U{0..T−1} | N i.i.d. |
| `antithetic` | antithetic noise | N/2 i.i.d., each used twice | shared within pair | (ε, −ε) |
| `stratified` | stratified t | N i.i.d. | one uniform draw in each of N equal strata of [0,1), mapped to {0..T−1} | i.i.d. |
| `sobol` | randomised QMC t | N i.i.d. | consecutive points of a scrambled 1-D Sobol sequence | i.i.d. |
| `importance` | importance-sampled t | N i.i.d. | p(t) ∝ sqrt(E[L_t²]) (running per-bin estimate, Improved-DDPM rule), mixed 90/10 with uniform; loss weight 1/(T p(t)) | i.i.d. |
| `strat_anti` | combination | N/2 i.i.d. | stratified over N/2 strata, shared within pair | (ε, −ε) |
| `paired_iid` | *diagnostic only* | N/2, each used twice | shared within pair | two independent ε |
| `is_strat` | *probe; trained only if budget allows* | N i.i.d. | stratified uniforms through the IS inverse CDF | i.i.d. |

`paired_iid` isolates the antithetic effect: `antithetic` vs `paired_iid` differs **only** in the
ε coupling, while `antithetic` vs `uniform` also pays for having half as many distinct (x0, t).

Control variates (optional item 5 in the brief): **not planned** in this budget; listed as future work.

## 3. Hypotheses (stated before running)
* **H1 (stratification).** `stratified` and `sobol` reduce total gradient variance relative to
  `uniform` at the same N, by removing the between-timestep component Var_t(E[g|t])/N. In 1-D with
  N = 2^k, scrambled Sobol is essentially a stratified sample, so `sobol ≈ stratified`.
* **H2 (antithetic).** Law of total variance gives, per step,
  Var(uniform) = (V_{x0,t} + V_ε)/N and Var(antithetic) = (2V_{x0,t} + (1+ρ)V_ε)/N,
  with V_{x0,t} = Var(E[g|x0,t]), V_ε = E[Var(g|x0,t)], and ρ the correlation of per-sample
  gradients under ε → −ε. Antithetic helps **iff V_{x0,t} < −ρ V_ε**. If the network is locally
  affine in x_t, ρ ≈ −1 (the odd part of g in ε cancels). We predict ρ < 0, but whether antithetic
  beats uniform depends on V_{x0,t}/V_ε, which we measure; no directional prediction.
* **H3 (importance sampling).** The loss-based proposal is optimal for *loss* variance, not gradient
  variance, and `L_simple` is fairly flat across t, so we expect little or no gradient-variance
  reduction (possibly an increase).
* **H4 (training outcome).** Variance reductions of < 2x will give at most small differences in
  FID / held-out loss at equal steps under Adam (cf. Jeha et al. 2024). A null result is a valid outcome.

## 4. Metrics
1. **Gradient variance** at fixed parameters θ: for each estimator, R resamples of the full N-sample
   gradient with x0 drawn from a fixed probe pool; report tr Cov(ĝ) and the ratio to `uniform` at the
   **same θ** (all estimators evaluated on the same checkpoints). Means are compared to check unbiasedness.
2. **Variance decomposition** per timestep bin: within-bin variance, between-bin variance of E[g|t],
   ε-share V_ε, antithetic correlation ρ(t), and per-timestep per-sample loss mean/variance.
3. **Held-out DSM loss** with common random numbers (fixed test images, fixed t and ε shared by all
   methods and checkpoints), vs steps and vs wall-clock.
4. **FID** (clean-fid, `clean` mode, 10k samples, CIFAR-10 train statistics, DDIM 50 steps η=0 from
   EMA weights, same initial noise for every model) at fixed checkpoints. Absolute FIDs will be high
   (short training, 10k samples); only differences between methods are interpreted.
5. **Overhead**: wall-clock per optimiser step for each estimator.

Statistics: 3 seeds per configuration, mean ± std (sample std, n−1). A difference is called
"significant" only if a two-sided Welch t-test gives p < 0.05; with n = 3 the test has low power and
this is stated wherever it matters. Seeds are fixed in advance (0, 1, 2); no seed is dropped.

## 5. Datasets and licences
* **Fashion-MNIST** (Xiao et al. 2017), MIT licence, downloaded from the official GitHub repository
  by `scripts/download_fmnist.py`. Used for pilot and fast-iteration runs on CPU (no FID; held-out loss
  and gradient-variance only).
* **CIFAR-10** (Krizhevsky 2009), distributed by the authors for research use; downloaded on Kaggle via
  `torchvision`. Main results. Not committed to the repo.
* **Inception-v3 FID weights** via `clean-fid` (MIT licence; weights from the TF FID implementation, Apache-2.0).

## 6. Model and training (identical for every estimator)
DDPM, T = 1000, linear β ∈ [1e-4, 0.02], ε-prediction, `L_simple`. Small U-Net (GroupNorm, SiLU,
sinusoidal time embedding, residual blocks, self-attention at the lowest resolution). Adam,
lr 2e-4, 200-step linear warm-up, grad-clip 1.0, EMA 0.999, batch N = 128, fp16 autocast on GPU.
Fashion-MNIST: base width 32, mult (1,2), CPU. CIFAR-10: base width 64, mult (1,2,2), 1 Kaggle GPU
per run (two runs in parallel on a T4 x2 machine).

## 7. Compute budget and estimate (hard cap: 2 Kaggle GPU hours = 7,200 s of quota)
Kaggle quota used before this project: 63,896 s (read via the API on 2026-11-04). The project stops
before the quota reading exceeds 71,096 s. Estimates below are *pre-pilot guesses* and will be
replaced by measured numbers in the Amendments.

| Experiment | Where | Estimate |
|---|---|---|
| Pilot, Fashion-MNIST, all estimators, 1 seed, ~200 steps | local CPU | 0 GPU-h |
| Fashion-MNIST comparison, 5 trained estimators x 3 seeds | local CPU | 0 GPU-h |
| GPU smoke test (timing, FID pipeline) | Kaggle | ~0.15 GPU-h |
| CIFAR-10 main: 5 estimators x 3 seeds, fixed step budget S (set from pilot timing), FID at 3 checkpoints, gradient-variance probes in the uniform runs | Kaggle T4 x2 | ~1.3 GPU-h |
| Reserve for review-driven re-runs | Kaggle | ~0.5 GPU-h |

Trained estimators: `uniform`, `antithetic`, `stratified`, `importance`, `strat_anti`. `sobol`,
`paired_iid` and `is_strat` are evaluated in the gradient-variance probes only (Sobol is expected to
coincide with stratified in 1-D; this is checked, not assumed).

## 8. Success criteria
* The project succeeds as research if it delivers a correct, reproducible, equal-compute comparison
  with measured gradient variances that explain the training outcomes, whichever way they go.
* A technique is said to "help training" only if it improves final FID **and** held-out loss with
  p < 0.05 at equal steps, and its wall-clock overhead does not cancel the gain.
* A technique is said to "reduce variance" if its variance ratio is < 1 at every probed checkpoint in
  every seed.

## Amendments
All made after the pilot and GPU smoke tests, before any main-result run. Reasons are measured costs.

**A1 (2026-11-04) Compute lost to data download.** The first GPU smoke test spent its whole 25-min
session downloading CIFAR-10 from the authors' server at ~90 kB/s (0.42 GPU-h, no training). CIFAR-10 is
now fetched from the Hugging Face mirror `uoft-cs/cifar10` (identical images, lossless PNG; ~20 s), since
Kaggle batch sessions cannot attach Kaggle datasets. Remaining budget after both smoke tests: 5,308 s.

**A2 Model and schedule sized to the measured cost.** Measured on a Kaggle T4: 200 ms/step for the
width-64 U-Net (6.1M params) and 131 s for a 2k-sample FID, i.e. the planned 15 runs x 3 FID-10k would not
fit. Changes, identical for every estimator:
* U-Net base width 64 -> **32** (1.54M params), same topology.
* Training budget **3,000 steps** per run (384k (x0,t,ε) evaluations; ~7.7 epochs).
* FID-10k at steps **1000/2000/3000** with **DDIM-25** (was DDIM-50) to halve sampling cost.
* EMA keeps decay 0.999 but with the standard warm-up min(0.999, (1+k)/(10+k)); without it the EMA
  at step 1000 would still carry 37% of the random initialisation. (The Fashion-MNIST CPU runs were
  started before this change and use plain EMA; for them only the raw-model held-out loss is reported.)
* Gradient-variance probes at steps 250/1000/2000/3000 (R = 48) and the per-timestep decomposition at
  1000/3000 (20 bins, 3 reps), inside the uniform runs.
* Seeds are run in two sessions (seed 0, then seeds 1-2) so that the measured cost of the first session
  can be checked against the budget before the rest is launched. The step budget is fixed now and will
  not change between sessions.

**A3 Fashion-MNIST** uses 800 steps (CPU cost ~3 s/step/worker), width 32 with one res-block per level,
probes at 200/800 (R = 24) and decomposition at 800 (10 bins, 3 reps).
