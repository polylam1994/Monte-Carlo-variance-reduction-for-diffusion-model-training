"""Small DDPM U-Net (Ho et al. 2020 style), kept deliberately compact."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def gn(c):
    return nn.GroupNorm(min(32, c // 4), c)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, dropout=0.0):
        super().__init__()
        self.n1, self.c1 = gn(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.n2, self.c2 = gn(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(temb))[:, :, None, None]
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return h + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n, self.qkv, self.o = gn(c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(b, 3, c, h * w).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(1, 2)[:, None], k.transpose(1, 2)[:, None],
                                           v.transpose(1, 2)[:, None])[:, 0]
        return x + self.o(a.transpose(1, 2).reshape(b, c, h, w))


class UNet(nn.Module):
    def __init__(self, in_ch=3, ch=64, mult=(1, 2, 2), nres=2, attn_lowest=True, dropout=0.0):
        super().__init__()
        self.ch = ch
        tdim = ch * 4
        self.temb = nn.Sequential(nn.Linear(ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(in_ch, ch, 3, padding=1)
        self.down = nn.ModuleList()
        chans, c = [ch], ch
        for i, m in enumerate(mult):
            for _ in range(nres):
                self.down.append(ResBlock(c, ch * m, tdim, dropout)); c = ch * m; chans.append(c)
            if i < len(mult) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); chans.append(c)
        self.mid1 = ResBlock(c, c, tdim, dropout)
        self.mida = Attn(c) if attn_lowest else nn.Identity()
        self.mid2 = ResBlock(c, c, tdim, dropout)
        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(mult))):
            for _ in range(nres + 1):
                self.up.append(ResBlock(c + chans.pop(), ch * m, tdim, dropout)); c = ch * m
            if i > 0:
                self.up.append(nn.Upsample(scale_factor=2, mode="nearest"))
                self.up.append(nn.Conv2d(c, c, 3, padding=1))
        self.out = nn.Sequential(gn(c), nn.SiLU(), nn.Conv2d(c, in_ch, 3, padding=1))

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.ch))
        h = self.inp(x)
        hs = [h]
        for m in self.down:
            h = m(h, temb) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        h = self.mid2(self.mida(self.mid1(h, temb)), temb)
        for m in self.up:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, hs.pop()], 1), temb)
            else:
                h = m(h)
        return self.out(h)
