import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from models.unet import UNet, NUM_CLASSES
from data import get_dataset, get_loader

P_UNCOND = 0.15


class EMA:
    """Exponential moving average of model weights for stable inference."""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = {k: v.clone().float() for k, v in state.items()}

    def apply_to(self, model):
        dtype = next(model.parameters()).dtype
        model.load_state_dict({k: v.to(dtype) for k, v in self.shadow.items()})


def train(epochs=100, lr=2e-4, grad_clip=1.0, batch_size=128,
          checkpoint_dir="checkpoints", checkpoint_every=10,
          resume_from=None, device=None, dataset=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset is None:
        dataset = get_dataset()

    os.makedirs(checkpoint_dir, exist_ok=True)
    loader    = get_loader(dataset, batch_size=batch_size)
    model     = UNet(in_ch=3, base_ch=64, ch_mult=(1, 2, 4)).to(device)
    ema       = EMA(model, decay=0.9999)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    FM        = ConditionalFlowMatcher(sigma=0.0)
    print(f"batch_size={batch_size}  |  steps/epoch: {len(loader):,}")

    start_epoch = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=True)
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model_state"])
        ema.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        print(f"Resumed from '{resume_from}'  (epoch {start_epoch}/{epochs})")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        for x1, y in loader:
            x1, y = x1.to(device), y.to(device)
            x0 = torch.randn_like(x1)

            drop = torch.rand(len(y), device=device) < P_UNCOND
            y_in = y.masked_fill(drop, NUM_CLASSES)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            loss = F.mse_loss(model(t, xt, y_in), ut)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update(model)
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        print(f"epoch {epoch+1:3d}/{epochs} | lr {optimizer.param_groups[0]['lr']:.2e} | loss {avg_loss:.4f}")

        if (epoch + 1) % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch+1:04d}.pt")
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "ema_state":       ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }, path)
            print(f"  └─ checkpoint saved → {path}")

    ema.apply_to(model)
    return model
