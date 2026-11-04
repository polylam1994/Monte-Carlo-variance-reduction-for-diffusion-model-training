"""Monte Carlo estimators of the DDPM L_simple objective.

Every estimator returns, for a budget of n (x0, t, eps) evaluations:
    idx  : LongTensor[n]  indices into the training set (repeated for paired estimators)
    flip : BoolTensor[n]  horizontal-flip flags, drawn per distinct x0 (repeated for pairs)
    t    : LongTensor[n]  timesteps in {0..T-1}
    eps  : FloatTensor[n, C, H, W]
    w    : FloatTensor[n] importance weights (1 for unweighted estimators)
and the loss estimate is mean(w * per_sample_loss).  All are unbiased for
E_{x0 ~ data, t ~ U{0..T-1}, eps ~ N(0,I)} [ per_sample_loss ].
"""
import torch


class ISState:
    """Running per-bin estimate of E[L_t^2] (Improved-DDPM rule: p(t) ∝ sqrt(E[L_t^2])).

    The proposal is piecewise constant over `K` equal bins of timesteps and mixed with the
    uniform distribution with weight `mix` so that importance weights are bounded by 1/mix.
    It is uniform until every bin holds `history` losses (warm-up, as in Improved DDPM)."""

    def __init__(self, T, K=50, history=100, mix=0.1, device="cpu"):
        assert T % K == 0
        self.T, self.K, self.H, self.mix = T, K, history, mix
        self.buf = torch.zeros(K, history, device=device)
        self.cnt = torch.zeros(K, dtype=torch.long, device=device)
        self.warm = False  # becomes True once every bin holds `history` losses (checked until then)

    @torch.no_grad()
    def update(self, t, loss):
        """Vectorised ring-buffer update with the squared per-sample (unweighted) losses."""
        b = (t // (self.T // self.K)).long()
        order = torch.argsort(b, stable=True)
        bs, v = b[order], loss.detach().float()[order] ** 2
        counts = torch.bincount(bs, minlength=self.K)
        starts = torch.cumsum(counts, 0) - counts
        rank = torch.arange(len(bs), device=bs.device) - starts[bs]
        pos = (self.cnt[bs] + rank) % self.H
        self.buf[bs, pos] = v
        self.cnt += counts

    def probs(self):
        if not self.warm:
            self.warm = bool((self.cnt >= self.H).all())
        if not self.warm:
            return torch.full((self.K,), 1.0 / self.K, device=self.buf.device)
        p = self.buf.mean(1).sqrt()
        p = p / p.sum()
        return (1 - self.mix) * p + self.mix / self.K

    def state_dict(self):
        return {"buf": self.buf, "cnt": self.cnt}

    def load_state_dict(self, d):
        self.buf.copy_(d["buf"]); self.cnt.copy_(d["cnt"]); self.warm = False


def _stratified_u(m, gen, device):
    return (torch.arange(m, device=device) + torch.rand(m, generator=gen, device=device)) / m


def _u_to_t(u, T):
    return (u * T).long().clamp_(0, T - 1)


class Estimator:
    name = "base"
    paired = False  # True -> n/2 distinct (x0, t), each used twice

    def __init__(self, T, n_data, shape, device, is_state=None, seed=0):
        self.T, self.n_data, self.shape, self.device = T, n_data, tuple(shape), device
        self.is_state = is_state

    def _t(self, m, gen):
        return torch.randint(0, self.T, (m,), generator=gen, device=self.device)

    def _eps(self, m, gen):
        return torch.randn((m,) + self.shape, generator=gen, device=self.device)

    def _idx(self, m, gen):
        idx = torch.randint(0, self.n_data, (m,), generator=gen, device=self.device)
        flip = torch.rand(m, generator=gen, device=self.device) < 0.5
        return idx, flip

    def sample(self, n, gen):
        (idx, flip), t, eps = self._idx(n, gen), self._t(n, gen), self._eps(n, gen)
        return idx, flip, t, eps, torch.ones(n, device=self.device)


class Uniform(Estimator):
    name = "uniform"


class Stratified(Estimator):
    name = "stratified"

    def _t(self, m, gen):
        return _u_to_t(_stratified_u(m, gen, self.device), self.T)


class Sobol(Estimator):
    """Consecutive points of a scrambled 1-D Sobol sequence (torch SobolEngine: random linear
    matrix scramble + digital shift).  With n a power of two each batch is a (0, k, 1)-net."""
    name = "sobol"

    def __init__(self, *a, seed=0, **k):
        super().__init__(*a, seed=seed, **k)
        self.engine = torch.quasirandom.SobolEngine(1, scramble=True, seed=seed + 12345)

    def _t(self, m, gen):
        return _u_to_t(self.engine.draw(m)[:, 0].to(self.device), self.T)


class Importance(Estimator):
    name = "importance"

    def _sample_t(self, u):
        p = self.is_state.probs()
        K, wbin = self.is_state.K, self.T // self.is_state.K
        cdf = torch.cumsum(p, 0)
        cdf[-1] = 1.0
        b = torch.searchsorted(cdf, u, right=True).clamp_(0, K - 1)
        lo = cdf[b] - p[b]
        pos = ((u - lo) / p[b]).clamp_(0, 1 - 1e-6)  # position inside the bin, uniform
        t = b * wbin + (pos * wbin).long()
        w = 1.0 / (K * p[b])
        return t.clamp_(0, self.T - 1), w

    def sample(self, n, gen):
        u = torch.rand(n, generator=gen, device=self.device)
        t, w = self._sample_t(u)
        return (*self._idx(n, gen), t, self._eps(n, gen), w)


class ISStrat(Importance):
    """Stratified uniforms pushed through the IS inverse CDF (cf. CARV)."""
    name = "is_strat"

    def sample(self, n, gen):
        t, w = self._sample_t(_stratified_u(n, gen, self.device))
        return (*self._idx(n, gen), t, self._eps(n, gen), w)


class Antithetic(Estimator):
    name = "antithetic"
    paired = True

    def _pair_t(self, m, gen):
        return self._t(m, gen)

    def _second_eps(self, eps, gen):
        return -eps

    def sample(self, n, gen):
        assert n % 2 == 0
        m = n // 2
        (idx, flip), t, eps = self._idx(m, gen), self._pair_t(m, gen), self._eps(m, gen)
        return (idx.repeat(2), flip.repeat(2), t.repeat(2),
                torch.cat([eps, self._second_eps(eps, gen)]), torch.ones(n, device=self.device))


class PairedIID(Antithetic):
    """Diagnostic: same (x0, t) pairing as antithetic but two *independent* eps."""
    name = "paired_iid"

    def _second_eps(self, eps, gen):
        return torch.randn(eps.shape, generator=gen, device=self.device)


class StratAnti(Antithetic):
    name = "strat_anti"

    def _pair_t(self, m, gen):
        return _u_to_t(_stratified_u(m, gen, self.device), self.T)


REGISTRY = {c.name: c for c in [Uniform, Stratified, Sobol, Importance, ISStrat, Antithetic, PairedIID, StratAnti]}


def make(name, T, n_data, shape, device, is_state, seed=0):
    return REGISTRY[name](T, n_data, shape, device, is_state=is_state, seed=seed)
