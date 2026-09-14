"""
utils.py
========
Small, dependency-light helpers shared by train.py and evaluate.py:
reproducibility seeding, checkpoint save/load, a running-average meter for
loss/accuracy, and a CSV logger for per-epoch metrics.
"""

import os
import csv
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Fix every relevant RNG so that Plain-CNN and ResNet training runs are
    driven by identical batch orderings, weight initializations (modulo
    architecture), and augmentation draws -- a prerequisite for the
    comparison in this project to be scientifically meaningful.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels trade a little speed for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Tracks a running average of a scalar (loss or accuracy) over a batch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 classification accuracy for a batch, as a percentage."""
    preds = logits.argmax(dim=1)
    correct = (preds == targets).float().sum().item()
    return 100.0 * correct / targets.size(0)


def save_checkpoint(model, optimizer, epoch: int, best_acc: float,
                     checkpoint_dir: str, tag: str) -> str:
    """Save model/optimizer state so training can resume or be audited later."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{tag}_epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_acc": best_acc,
    }, path)
    return path


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu"):
    """Restore model (and optionally optimizer) state from a checkpoint file."""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


class CSVLogger:
    """
    Appends one row per epoch to a CSV file with columns:
    epoch, train_loss, val_loss, train_acc, val_acc, lr

    Kept intentionally simple (no external logging dependency) so the raw
    numbers behind every plot in plots.py are always inspectable directly.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "lr"]
            )

    def log(self, epoch, train_loss, val_loss, train_acc, val_acc, lr):
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc, lr])