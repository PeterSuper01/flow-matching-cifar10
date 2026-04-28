import copy
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


@torch.no_grad()
def evaluate(model, val_loader, FM, device):
    """Compute average flow-matching MSE loss on the validation set.

    Notes:
      - No CFG dropout: y_in = y (we evaluate the conditional objective
        the model is asked to fit, not the unconditional branch).
      - Stochasticity in (x0, t) still exists; loss is a Monte-Carlo
        estimate but stable enough across the full 10K test set.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for x1, y in val_loader:
        x1, y = x1.to(device), y.to(device)
        x0 = torch.randn_like(x1)
        y_in = y  # no CFG dropout on val
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        loss = F.mse_loss(model(t, xt, y_in), ut)
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def train(epochs=100, lr=2e-4, grad_clip=1.0, batch_size=128,
          val_batch_size=256,
          checkpoint_dir="checkpoints", checkpoint_every=10,
          resume_from=None, device=None, dataset=None, val_dataset=None,
          model=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset is None:
        dataset = get_dataset()
    if val_dataset is None:
        # CIFAR-10 test split (10K images), no augmentation.
        val_dataset = get_dataset(train=False)

    os.makedirs(checkpoint_dir, exist_ok=True)
    loader     = get_loader(dataset, batch_size=batch_size)
    val_loader = get_loader(val_dataset, batch_size=val_batch_size, shuffle=False)
    if model is None:
        model = UNet(in_ch=3, base_ch=64, ch_mult=(1, 2, 4)).to(device)
    ema       = EMA(model, decay=0.9999)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    FM        = ConditionalFlowMatcher(sigma=0.0)
    print(f"batch_size={batch_size}  |  steps/epoch: {len(loader):,}  |  "
          f"val_batch_size={val_batch_size}  |  val_steps: {len(val_loader):,}")

    train_losses = []
    val_losses = []
    start_epoch = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=True)
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model_state"])
        ema.load_state_dict(ckpt["ema_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        train_losses = ckpt.get("train_losses", [])
        val_losses   = ckpt.get("val_losses",   [])
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

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            ema.update(model)
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)

        # ── Validation pass (no grads, no CFG dropout) ──────────────────
        val_loss = evaluate(model, val_loader, FM, device)
        val_losses.append(val_loss)

        print(f"epoch {epoch+1:3d}/{epochs} | lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"train {avg_loss:.4f} | val {val_loss:.4f}")

        if (epoch + 1) % checkpoint_every == 0:
            path = os.path.join(checkpoint_dir, f"ckpt_epoch{epoch+1:04d}.pt")
            torch.save({
                "epoch":           epoch + 1,
                "model_state":     model.state_dict(),
                "ema_state":       ema.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "train_losses":    train_losses,
                "val_losses":      val_losses,
            }, path)
            print(f"  └─ checkpoint saved → {path}")

    ema_model = copy.deepcopy(model)
    ema.apply_to(ema_model)
    return ema_model, train_losses, val_losses
