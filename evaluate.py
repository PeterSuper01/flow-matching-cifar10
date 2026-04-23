import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from models.unet import NUM_CLASSES
from sampler import generate


@torch.no_grad()
def compute_fid(model, dataset, n_real=10_000, n_fake=10_000,
                batch_size=256, guidance_scale=3.0, steps=100):
    device = next(model.parameters()).device
    fid    = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    n_done = 0
    for imgs, _ in DataLoader(dataset, batch_size=batch_size, shuffle=True):
        if n_done >= n_real:
            break
        fid.update((imgs.to(device) * 0.5 + 0.5).clamp(0, 1), real=True)
        n_done += len(imgs)

    n_done = 0
    while n_done < n_fake:
        bs   = min(batch_size, n_fake - n_done)
        y    = torch.arange(bs, device=device) % NUM_CLASSES
        imgs = generate(model, y, steps=steps, guidance_scale=guidance_scale)
        fid.update((imgs * 0.5 + 0.5).clamp(0, 1), real=False)
        n_done += bs
        print(f"  generated {n_done}/{n_fake}", end="\r")

    score = fid.compute().item()
    print(f"\nFID ({n_fake} samples, guidance={guidance_scale}): {score:.2f}")
    return score
