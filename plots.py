"""
plots.py
========
Generates the visualizations requested for Deliverable 2: training/validation
loss and accuracy curves (per model and overlaid plain-vs-resnet, echoing the
style of Fig. 1 / Fig. 4 / Fig. 6 in the paper), confusion matrices, and a
grid of example predictions.
"""

import os
import csv

import numpy as np
import matplotlib.pyplot as plt

from config import cfg
from data import CIFAR10_CLASSES


def _read_csv_log(path):
    epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))
    return map(np.array, (epochs, train_loss, val_loss, train_acc, val_acc))


def plot_curves(plain_csv: str, resnet_csv: str, out_dir: str = None):
    """
    Overlaid train/val loss and accuracy curves for Plain CNN vs. ResNet,
    directly analogous to Fig. 1 (CIFAR-10 20 vs 56-layer) and Fig. 6
    (plain vs. ResNet) in the paper -- the same "does the deeper model
    train worse?" comparison, but for this project's own models.
    """
    out_dir = out_dir or cfg.plot_dir
    os.makedirs(out_dir, exist_ok=True)

    p_epochs, p_tl, p_vl, p_ta, p_va = _read_csv_log(plain_csv)
    r_epochs, r_tl, r_vl, r_ta, r_va = _read_csv_log(resnet_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(p_epochs, p_tl, "--", label="Plain CNN (train)", color="tab:orange")
    axes[0].plot(p_epochs, p_vl, "-", label="Plain CNN (val)", color="tab:orange")
    axes[0].plot(r_epochs, r_tl, "--", label="ResNet (train)", color="tab:blue")
    axes[0].plot(r_epochs, r_vl, "-", label="ResNet (val)", color="tab:blue")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()

    axes[1].plot(p_epochs, p_ta, "--", label="Plain CNN (train)", color="tab:orange")
    axes[1].plot(p_epochs, p_va, "-", label="Plain CNN (val)", color="tab:orange")
    axes[1].plot(r_epochs, r_ta, "--", label="ResNet (train)", color="tab:blue")
    axes[1].plot(r_epochs, r_va, "-", label="ResNet (val)", color="tab:blue")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_title("Training / Validation Accuracy")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "plain_vs_resnet_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(confusion: np.ndarray, title: str, out_dir: str = None):
    """Heatmap confusion matrix for one model's test-set predictions."""
    out_dir = out_dir or cfg.plot_dir
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(CIFAR10_CLASSES)))
    ax.set_yticks(range(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    fname = title.lower().replace(" ", "_") + ".png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_example_predictions(images, targets, preds, out_dir: str = None, n: int = 16):
    """
    Grid of n example test images with their ground-truth and predicted
    labels overlaid -- images are expected already de-normalized to [0,1].
    """
    out_dir = out_dir or cfg.plot_dir
    os.makedirs(out_dir, exist_ok=True)

    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        gt = CIFAR10_CLASSES[targets[i]]
        pred = CIFAR10_CLASSES[preds[i]]
        color = "green" if gt == pred else "red"
        ax.set_title(f"gt: {gt}\npred: {pred}", fontsize=8, color=color)
        ax.axis("off")
    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    path = os.path.join(out_dir, "example_predictions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path