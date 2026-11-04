"""Direct measurement of gradient-estimator variance at fixed parameters, and a
law-of-total-variance decomposition of the per-sample gradient by timestep."""
import torch

from . import estimators as E


def flat_grad(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def estimator_grad(model, diff, data, idx, flip, t, eps, w, amp, scale=1024.0):
    """Gradient of mean(w * L_i) at the current parameters, as one flat fp32 vector.
    Under fp16 autocast a fixed loss scale is used and removed (no optimiser state touched)."""
    model.zero_grad(set_to_none=True)
    x0 = data.get(idx, flip)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        loss = (w * diff.per_sample_loss(model, x0, t, eps)).mean()
    (loss * (scale if amp else 1.0)).backward()
    g = flat_grad(model)
    if amp:
        g = g / scale
    return g, loss.detach()


def variance_probe(model, diff, pool, names, is_state, n, R, seed, amp):
    """For each estimator: R independent draws of the n-sample gradient estimator at the same
    parameters.  x0 is drawn from the fixed probe pool `pool` (an ImageData).  Returns trace of
    covariance, squared norm of the mean, and squared distance of each mean to the uniform mean
    (unbiasedness check: E||mean_e - mean_u||^2 = (tr_e + tr_u)/R if both are unbiased)."""
    was_training = model.training
    model.eval()  # no dropout in our configs; eval() only for safety
    dev = pool.x.device
    out, means = {}, {}
    for name in names:
        gen = torch.Generator(device=dev); gen.manual_seed(seed)
        est = E.make(name, diff.T, pool.n, pool.shape, dev, is_state, seed=seed)
        S, SS, nonfinite = None, 0.0, 0
        for r in range(R):
            g, _ = estimator_grad(model, diff, pool, *est.sample(n, gen), amp)
            if not torch.isfinite(g).all():
                nonfinite += 1; continue
            g64 = g.double()
            S = g64.clone() if S is None else S.add_(g64)
            SS += float(g64.pow(2).sum())
        Rv = R - nonfinite
        mean = S / Rv
        tr = (SS - Rv * float(mean.pow(2).sum())) / (Rv - 1)
        means[name] = mean
        out[name] = {"trace_cov": tr, "mean_sqnorm": float(mean.pow(2).sum()), "R": Rv, "nonfinite": nonfinite}
    if "uniform" in means:
        for name in names:
            d = float((means[name] - means["uniform"]).pow(2).sum())
            out[name]["dist2_to_uniform_mean"] = d
            out[name]["dist2_expected_if_unbiased"] = (out[name]["trace_cov"] + out["uniform"]["trace_cov"]) / out[name]["R"] if name != "uniform" else 0.0
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return out


def timestep_decomposition(model, diff, pool, K, n, reps, seed, amp, loss_samples=512):
    """Per timestep bin k (K equal bins of {0..T-1}):
      m_k     : E[g | t in bin k]                            (from two independent batches A, B)
      Vw_k    : per-sample within-bin variance  = n ||gA - gB||^2 / 2
      Veps_k  : per-sample E[Var(g | x0, t)]    = n ||gA - gC||^2 / 2   (C: same x0,t, new eps)
      Vanti_k : (1 - rho_k) Veps_k              = n ||gA - gA'||^2 / 2  (A': same x0,t, -eps)
    plus per-sample loss mean / variance in the bin (forward only).
    Returns per-bin scalars and estimator-level variance predictions for N = 2n... (computed later)."""
    was_training = model.training
    model.eval()
    dev = pool.x.device
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    T, wbin = diff.T, diff.T // K
    rows, mks = [], []
    for k in range(K):
        acc = {"vw": 0.0, "veps": 0.0, "vanti": 0.0}
        msum = None
        for _ in range(reps):
            def draw():
                idx = torch.randint(0, pool.n, (n,), generator=gen, device=dev)
                flip = torch.rand(n, generator=gen, device=dev) < 0.5
                t = k * wbin + torch.randint(0, wbin, (n,), generator=gen, device=dev)
                eps = torch.randn((n,) + pool.shape, generator=gen, device=dev)
                return idx, flip, t, eps
            one = torch.ones(n, device=dev)
            ia, fa, ta, ea = draw()
            gA, _ = estimator_grad(model, diff, pool, ia, fa, ta, ea, one, amp)
            gAm, _ = estimator_grad(model, diff, pool, ia, fa, ta, -ea, one, amp)
            eC = torch.randn(ea.shape, generator=gen, device=dev)
            gC, _ = estimator_grad(model, diff, pool, ia, fa, ta, eC, one, amp)
            gB, _ = estimator_grad(model, diff, pool, *draw(), one, amp)
            acc["vw"] += n * float((gA - gB).double().pow(2).sum()) / 2
            acc["veps"] += n * float((gA - gC).double().pow(2).sum()) / 2
            acc["vanti"] += n * float((gA - gAm).double().pow(2).sum()) / 2
            m = (gA.double() + gB.double()) / 2
            msum = m if msum is None else msum + m
        mk = msum / reps
        mks.append(mk)
        vw, veps, vanti = acc["vw"] / reps, acc["veps"] / reps, acc["vanti"] / reps
        # per-sample loss statistics (no grad)
        with torch.no_grad():
            Ls = []
            for i0 in range(0, loss_samples, n):
                idx = torch.randint(0, pool.n, (n,), generator=gen, device=dev)
                flip = torch.rand(n, generator=gen, device=dev) < 0.5
                t = k * wbin + torch.randint(0, wbin, (n,), generator=gen, device=dev)
                eps = torch.randn((n,) + pool.shape, generator=gen, device=dev)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    Ls.append(diff.per_sample_loss(model, pool.get(idx, flip), t, eps).float())
            L = torch.cat(Ls)
        rows.append({"bin": k, "t_lo": k * wbin, "t_hi": (k + 1) * wbin - 1,
                     "Vw": vw, "Veps": veps, "Vanti": vanti, "rho": 1 - vanti / veps,
                     # var of mk estimate is Vw/(2 n reps); subtract it from ||mk||^2
                     "mk_sqnorm": float(mk.pow(2).sum()) - vw / (2 * n * reps),
                     "loss_mean": float(L.mean()), "loss_var": float(L.var()),
                     "loss2_mean": float(L.pow(2).mean())})
    M = torch.stack(mks)
    mbar = M.mean(0)
    noise = sum(r["Vw"] for r in rows) / len(rows) / (2 * n * reps)
    between = float((M - mbar).pow(2).sum(1).mean()) - noise * (1 - 1 / K)
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return {"bins": rows, "V_between": between, "mbar_sqnorm": float(mbar.pow(2).sum()),
            "K": K, "n": n, "reps": reps}


def predict_variances(dec, N, is_probs=None):
    """Predicted tr Cov of each N-sample estimator from the decomposition (per-sample units / N).
    Uses bin-level quantities only, so 'stratified' is the bin-stratified approximation
    (within-bin t variation is kept in Vw)."""
    bins = dec["bins"]; K = len(bins)
    Vw = torch.tensor([b["Vw"] for b in bins]); Veps = torch.tensor([b["Veps"] for b in bins])
    rho = torch.tensor([b["rho"] for b in bins]); m2 = torch.tensor([b["mk_sqnorm"] for b in bins])
    Vb, mbar2 = dec["V_between"], dec["mbar_sqnorm"]
    Vx0t = Vw - Veps
    pred = {"uniform": (Vw.mean() + Vb) / N,
            "stratified": Vw.mean() / N,
            "paired_iid": (2 * (Vb + Vx0t.mean()) + Veps.mean()) / N,
            "antithetic": (2 * (Vb + Vx0t.mean()) + ((1 + rho) * Veps).mean()) / N,
            "strat_anti": (2 * Vx0t.mean() + ((1 + rho) * Veps).mean()) / N}
    second = Vw + m2  # E||g_i||^2 per bin
    q_opt = second.clamp(min=0).sqrt(); q_opt = q_opt / q_opt.sum()
    pred["oracle_grad_IS"] = ((second / (K ** 2 * q_opt)).sum() - mbar2) / N
    if is_probs is not None:
        q = torch.as_tensor(is_probs, dtype=torch.float64).reshape(K, -1).sum(1)
        pred["importance"] = ((second.double() / (K ** 2 * q)).sum() - mbar2) / N
    return {k: float(v) for k, v in pred.items()}
