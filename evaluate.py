"""
evaluate.py
===========
Loads trained Plain-CNN and ResNet checkpoints and produces the final
side-by-side comparison: test accuracy, test loss, parameter count,
inference speed, and a confusion matrix per model. This directly supports
Deliverable 1's Section 8 discussion (what evidence demonstrates ResNet's
effectiveness) by generating the concrete numbers for THIS project's own
CIFAR-10 run, analogous to Table 2 / Table 6 in the paper.

Usage
-----
    python evaluate.py --resnet_ckpt checkpoints/resnet_final.pt \
                        --plain_ckpt checkpoints/plain_final.pt
"""

import argparse
import time

import torch
import torch.nn as nn
import numpy as np

from config import cfg
from data import get_dataloaders, CIFAR10_CLASSES
from models.resnet import resnet_cifar
from models.cnn import plain_cnn_cifar
from utils import load_checkpoint, AverageMeter, accuracy


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Full-test-set pass: returns (avg_loss, avg_acc, confusion_matrix, imgs_per_sec)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_meter, acc_meter = AverageMeter(), AverageMeter()
    num_classes = cfg.num_classes
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    total_images, total_time = 0, 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        t0 = time.time()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - t0
        total_images += images.size(0)

        loss = criterion(logits, targets)
        loss_meter.update(loss.item(), targets.size(0))
        acc_meter.update(accuracy(logits, targets), targets.size(0))

        preds = logits.argmax(dim=1).cpu().numpy()
        gts = targets.cpu().numpy()
        for gt, pred in zip(gts, preds):
            confusion[gt, pred] += 1

    imgs_per_sec = total_images / max(total_time, 1e-9)
    return loss_meter.avg, acc_meter.avg, confusion, imgs_per_sec


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resnet_ckpt", type=str, required=True)
    parser.add_argument("--plain_ckpt", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(cfg.device)
    _, test_loader = get_dataloaders()

    results = {}
    for name, ckpt_path, builder in [
        ("resnet", args.resnet_ckpt, resnet_cifar),
        ("plain", args.plain_ckpt, plain_cnn_cifar),
    ]:
        model = builder(cfg.n_blocks_per_stage, cfg.stage_channels, cfg.num_classes).to(device)
        load_checkpoint(ckpt_path, model, map_location=device)
        loss, acc, confusion, speed = evaluate_model(model, test_loader, device)
        results[name] = {
            "loss": loss,
            "acc": acc,
            "params": count_params(model),
            "imgs_per_sec": speed,
            "confusion": confusion,
        }

    print("\n=== Final Comparison (CIFAR-10 test set) ===")
    header = f"{'Metric':<22}{'Plain CNN':>15}{'ResNet':>15}"
    print(header)
    print("-" * len(header))
    print(f"{'Test accuracy (%)':<22}{results['plain']['acc']:>15.2f}{results['resnet']['acc']:>15.2f}")
    print(f"{'Test loss':<22}{results['plain']['loss']:>15.4f}{results['resnet']['loss']:>15.4f}")
    print(f"{'# parameters':<22}{results['plain']['params']:>15,}{results['resnet']['params']:>15,}")
    print(f"{'Inference (img/s)':<22}{results['plain']['imgs_per_sec']:>15.1f}{results['resnet']['imgs_per_sec']:>15.1f}")

    print("\nConfusion matrices saved via plots.py (see plot_confusion_matrix).")
    return results


if __name__ == "__main__":
    main()