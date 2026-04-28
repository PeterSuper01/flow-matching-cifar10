import torch
import torchvision
import matplotlib.pyplot as plt

from models.unet import NUM_CLASSES

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


@torch.no_grad()
def generate(model, y, steps=100, guidance_scale=3.0):
    """
    y: (n,) integer tensor of class indices (0–9).
    Returns (n, 3, 32, 32) images in [-1, 1].
    Uses Heun (2nd-order) integration with batched cond + uncond forward passes.
    """
    model.eval()
    device = next(model.parameters()).device
    y    = y.to(device)
    n    = len(y)
    x    = torch.randn(n, 3, 32, 32, device=device)
    null = torch.full_like(y, NUM_CLASSES)
    dt   = 1.0 / steps

    def guided_v(t_scalar, x_cur):
        t_batch = torch.full((n,), t_scalar, device=device)
        v_both  = model(t_batch.repeat(2), torch.cat([x_cur, x_cur]), torch.cat([y, null]))
        v_cond, v_uncond = v_both.chunk(2)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    for i in range(steps):
        t  = i * dt
        v1 = guided_v(t, x)
        v2 = guided_v(t + dt, x + dt * v1)
        x  = x + dt * 0.5 * (v1 + v2)

    return x.clamp(-1, 1)


def show_images(imgs, nrow=8, title="Generated", fname=None):
    """imgs: (N, 3, H, W) float tensor in [-1, 1]."""
    imgs = (imgs.cpu() * 0.5 + 0.5).clamp(0, 1)
    grid = torchvision.utils.make_grid(imgs[:nrow * nrow], nrow=nrow, padding=2)
    plt.figure(figsize=(nrow * 1.5, nrow * 1.5))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=150)
    plt.show()
