# Flow Matching — CIFAR-10

Conditional flow matching with Classifier-Free Guidance (CFG) on CIFAR-10, implemented in PyTorch.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PeterSuper01/flow-matching-cifar10/blob/main/flow_matching_cifar10.ipynb)

## What's inside

| File | Description |
|---|---|
| `models/unet.py` | `UNet` (additive conditioning) and `UNetAdaGN` (AdaGN scale+shift conditioning) |
| `data.py` | CIFAR-10 dataset and dataloader helpers |
| `trainer.py` | `train()` with EMA, cosine LR, CFG dropout |
| `sampler.py` | Heun ODE sampler with CFG guidance |
| `evaluate.py` | FID evaluation via torchmetrics |
| `flow_matching_cifar10.ipynb` | End-to-end notebook: train → sample → compare → FID |

## Quick start

### Google Colab

Click the badge above. The first notebook cell auto-installs dependencies and clones the repo.

### Local (Poetry)

```bash
git clone https://github.com/PeterSuper01/flow-matching-cifar10.git
cd flow-matching-cifar10
poetry install            # core deps
poetry install --with eval  # + torchmetrics for FID
poetry run jupyter notebook
```

## Model comparison

The notebook includes a comparison section that trains both models for the same number of epochs and plots:
- Loss curves side-by-side
- Generated samples (one row per model, all 10 classes)
- FID scores (optional, requires `--with eval`)

| Model | ResBlock conditioning |
|---|---|
| `UNet` | `h = h + Linear(t_emb)` |
| `UNetAdaGN` | `h = GroupNorm(h) * (1 + γ) + β`, γ/β predicted from `t_emb` |
