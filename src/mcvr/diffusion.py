"""DDPM forward process, per-sample L_simple loss and a deterministic DDIM sampler."""
import torch


class Diffusion:
    def __init__(self, T=1000, beta1=1e-4, betaT=0.02, device="cpu"):
        self.T = T
        betas = torch.linspace(beta1, betaT, T, dtype=torch.float64)
        abar = torch.cumprod(1 - betas, 0)
        self.sqrt_abar = abar.sqrt().float().to(device)
        self.sqrt_1m_abar = (1 - abar).sqrt().float().to(device)
        self.abar = abar.float().to(device)

    def q_sample(self, x0, t, eps):
        return self.sqrt_abar[t][:, None, None, None] * x0 + self.sqrt_1m_abar[t][:, None, None, None] * eps

    def per_sample_loss(self, model, x0, t, eps):
        """Unweighted L_simple per sample: mean over pixels of (eps_hat - eps)^2."""
        pred = model(self.q_sample(x0, t, eps), t)
        return (pred.float() - eps).pow(2).flatten(1).mean(1)

    @torch.no_grad()
    def ddim_sample(self, model, n, shape, steps=50, generator=None, device="cpu", batch=1000, amp=False):
        """Deterministic DDIM (eta=0). Initial noise comes from `generator` so that every model
        is sampled from the *same* noise (common random numbers)."""
        ts = torch.linspace(self.T - 1, 0, steps, dtype=torch.float64).round().long().tolist()
        out = []
        for i0 in range(0, n, batch):
            b = min(batch, n - i0)
            x = torch.randn((b,) + tuple(shape), generator=generator, device="cpu").to(device)
            for j, t in enumerate(ts):
                tt = torch.full((b,), t, device=device, dtype=torch.long)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    eps = model(x, tt).float()
                a = self.abar[t]
                x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)
                eps = (x - a.sqrt() * x0) / (1 - a).sqrt()  # re-derive eps after clipping
                if j + 1 < len(ts):
                    an = self.abar[ts[j + 1]]
                    x = an.sqrt() * x0 + (1 - an).sqrt() * eps
                else:
                    x = x0
            out.append(x.cpu())
        return torch.cat(out)
