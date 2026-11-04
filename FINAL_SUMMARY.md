# Final summary

## What was done
A controlled, equal-compute comparison of Monte Carlo estimators for the DDPM training loss (ε-prediction,
`L_simple`). Only the way the 128 (x0, t, ε) evaluations per step are drawn changes: uniform (baseline),
antithetic ε-pairs, stratified t, scrambled-Sobol t, Improved-DDPM importance-sampled t, and stratified t +
antithetic ε, plus two diagnostics (paired-i.i.d. ε; IS + stratification). Experiments: CIFAR-10 (1.5M-param
U-Net, 3,000 steps, 5 trained estimators × 3 seeds, Kaggle T4) and Fashion-MNIST (0.6M params, 800 steps,
5 × 3 seeds, CPU). Gradient variance was measured directly at fixed parameters for all 8 estimators and
decomposed with the law of total variance. Total GPU: 1.97 of 2.0 h.

## What was found (numbers: `results/numbers.json`, paper Tables 1–5)
1. **Stratified and Sobol timesteps reliably reduce gradient variance, for free.** CIFAR-10 geometric mean
   0.82× [0.79, 0.86] (all 8 probes < 1, sign test p = 0.008); Fashion-MNIST 0.85× [0.74, 0.98]. Sobol ≈
   stratified, as 1-D theory predicts. Overhead is under 3%.
2. **Only ~6% of the per-sample gradient variance is between timesteps.** The rest is data-given-t (~44%)
   and noise (~50%). This caps what any timestep-stratification scheme can do.
3. **Antithetic ε works as a coupling (ρ ≈ −0.7; it beats its paired-i.i.d. control in 8/8 CIFAR probes)
   but not as an estimator at equal compute.** Each pair shares one image and one timestep, so it helps only
   if V_t + V_x < −ρ V_ε; measured 0.50 vs 0.36. Ratio 1.16 [0.98, 1.38] on CIFAR-10 (not significantly
   different from uniform) and 1.41 [1.25, 1.58] on Fashion-MNIST (significantly worse).
4. **The Improved-DDPM loss-based IS proposal is mismatched to the gradient.** Per-sample loss varies by
   ~190× across timesteps; gradient second moments vary by ~8×. A gradient-norm proposal
   q ∝ sqrt(E‖g‖²|t), measured post hoc at fixed parameters, gives **≈0.54×** variance at all three checkpoints, matching the
   decomposition's prediction (0.52–0.55). This is the largest reduction found. It was not trained.
5. **Training outcomes.** No estimator significantly changed CIFAR-10 held-out loss or FID (n = 3;
   detection limit ~1.7% / ~12 FID). Loss-based IS was best on both (held-out −1.5%, p = 0.07; FID 135.3 vs
   141.0, p = 0.24) and nominally significant on Fashion-MNIST (−2.7% held-out, p = 0.04; does not survive
   Bonferroni). This is consistent but not conclusive evidence of a small gain.
6. **Single gradient-variance probes are noisy** (CV ≈ 20% at R = 24 draws). Conclusions must be aggregated
   over many probes. This is a practical warning for anyone who "measures gradient variance".

## What is weak (be upfront about these)
* **Scale.** 3,000 steps of a 1.5M-param model; FID ≈ 140. These are not converged models, and late-training
  behaviour is unknown.
* **Power.** n = 3, and FID at the final checkpoint only for seeds 1–2. strat_anti has 2 FID seeds (PLAN A5).
* **Budget accident.** 0.42 of the 2 GPU-h were lost to a slow CIFAR-10 download in the first smoke test.
  This forced the reduced FID schedule (PLAN A1–A5). Every deviation is in PLAN.md.
* **Only one IS proposal trained.** The gradient-norm "oracle" is an analysis result, not a training result.
* **The decomposition is approximate.** It ranks estimators correctly and nails the oracle-IS ratio, but
  over-predicts the antithetic penalty (CIFAR-10 step 3000; also on Fashion-MNIST with 8 repetitions).
* **Novelty.** Stratified and importance-sampled t are prior work (VDM, Improved DDPM, CARV, Kim et al.).
  Antithetic training noise was not found in a web-only literature search. The contribution is the controlled
  comparison and the variance-decomposition explanation.

## Be ready to explain in an interview
* **Finance → ML mapping:** payoff ↔ per-sample loss/gradient; Brownian increments ↔ (t, ε); antithetic
  variates (Boyle 1977) ↔ (ε, −ε); stratification / QMC (Paskov & Traub) ↔ stratified / Sobol t; importance
  sampling for rare events ↔ over-sampling informative timesteps; common random numbers ↔ the held-out loss
  and the shared FID sampling noise.
* **Why antithetic fails here but works for options:** in option pricing the pair shares nothing but the
  normal draw, so ρ < 0 is enough. Here the pair also shares (x0, t), which doubles the data term. Derive
  Var_anti = [2(V_t+V_x) + (1+ρ)V_ε]/N on a whiteboard.
* **Why ρ ≈ −1:** if the denoiser is locally affine in x_t, the per-sample gradient is affine in ε, and the
  odd part cancels exactly.
* **Optimal IS proposal:** q* ∝ sqrt(E‖g‖² | t) minimises tr Cov (Cauchy–Schwarz). The Improved-DDPM rule
  q ∝ sqrt(E[ℓ²|t]) minimises *loss* variance, which is a different target.
* **Why lower variance did not give better FID:** Adam's normalisation, the small share of variance that is
  reducible, and low power (MDE ≈ 3 sd with n = 3). IS's slight edge is unexplained; the two conjectures are an
  early-training head start, or Adam interacting with the re-allocated timesteps. Clipping was ruled out (it
  was never active after warm-up).
* **The p = 0.04 question:** be clear that it is uncorrected, does not survive Bonferroni, and was not the
  pre-registered success criterion (which needed FID too). Do not oversell it.
* **Rigour choices:** equal evaluations per step; same initialisation per seed across arms; all estimators
  probed at the *same* parameters; unbiasedness check; pre-registration plus amendments; numbers generated by
  script; budget ledger.
* **What I would do with 10× compute:** train the gradient-norm IS proposal with an online estimate; run 10 seeds
  and 50k-sample FID over longer training; try other weightings (ELBO / Min-SNR) where V_t is larger; add a
  control variate.
