"""FID with clean-fid (Parmar et al. 2022), `clean` mode, against CIFAR-10 *train* statistics.

All models are sampled from the same initial noise (generator seeded with `fid_noise_seed`)
using deterministic DDIM, so FID differences are not inflated by sampler noise."""
import numpy as np
import torch


def save_grid(x, path, nrow=8):
    from PIL import Image
    x = ((x[: nrow * nrow].clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).numpy()
    n, h, w, c = x.shape
    g = x.reshape(nrow, nrow, h, w, c).transpose(0, 2, 1, 3, 4).reshape(nrow * h, nrow * w, c)
    Image.fromarray(g.squeeze(-1) if c == 1 else g).save(path)


def fid_eval(model, diff, cfg, dev, amp, grid_path=None):
    from cleanfid import fid
    gen = torch.Generator(device="cpu"); gen.manual_seed(cfg.get("fid_noise_seed", 2024))
    n = cfg.get("fid_n", 10000)
    x = diff.ddim_sample(model, n, (3, 32, 32), steps=cfg.get("ddim_steps", 50), generator=gen,
                         device=dev, batch=cfg.get("fid_batch", 1000), amp=amp)
    if grid_path:
        save_grid(x, grid_path)
    imgs = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    state = {"i": 0}

    def gen_fn(z):  # clean-fid asks for batches; we serve the pre-generated samples in order
        b = z.shape[0]
        out = imgs[state["i"]: state["i"] + b]
        state["i"] += b
        return out

    score = fid.compute_fid(gen=gen_fn, dataset_name="cifar10", dataset_res=32, dataset_split="train",
                            mode="clean", num_gen=n, batch_size=500, z_dim=1, device=dev, verbose=False,
                            use_dataparallel=False)
    return {"fid": float(score), "n": n, "ddim_steps": cfg.get("ddim_steps", 50)}
