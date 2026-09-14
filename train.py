"""
train.py
========
Trains either the Plain CNN or the ResNet on CIFAR-10 using IDENTICAL
settings (data, augmentation, batch size, optimizer, weight decay, LR
schedule, epochs, loss, seed, hardware) so that any accuracy/loss
difference between the two runs can be attributed to the architecture
alone -- the specific comparison the paper itself relies on (Table 2,
Fig. 4, Fig. 6).

Usage
-----
    python train.py --model resnet
    python train.py --model plain
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from config import cfg
from data import get_dataloaders
from models.resnet import resnet_cifar
from models.cnn import plain_cnn_cifar
from utils import set_seed, AverageMeter, accuracy, save_checkpoint, CSVLogger


def build_model(name: str) -> nn.Module:
    if name == "resnet":
        return resnet_cifar(cfg.n_blocks_per_stage, cfg.stage_channels, cfg.num_classes)
    elif name == "plain":
        return plain_cnn_cifar(cfg.n_blocks_per_stage, cfg.stage_channels, cfg.num_classes)
    raise ValueError(f"Unknown model name: {name}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    """
    One pass over `loader`. If train=True, gradients are computed and
    weights updated; if False, this is a plain forward-only evaluation
    pass (used for the validation/test split each epoch).
    """
    model.train(train)
    loss_meter, acc_meter = AverageMeter(), AverageMeter()

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, targets)

            if train:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(accuracy(logits, targets), batch_size)

    return loss_meter.avg, acc_meter.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "plain"], required=True)
    args = parser.parse_args()

    # ---- Fix every RNG so plain/resnet runs are comparable ---------------
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ---- Data (identical for both models) ---------------------------------
    train_loader, test_loader = get_dataloaders()

    # ---- Model, loss, optimizer, schedule (identical hyperparameters) ----
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.model}] trainable parameters: {num_params:,}")

    logger = CSVLogger(os.path.join(cfg.log_dir, f"{args.model}_metrics.csv"))
    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, test_loader, criterion, optimizer, device, train=False
        )
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        logger.log(epoch, train_loss, val_loss, train_acc, val_acc, current_lr)
        print(
            f"[{args.model}] epoch {epoch:3d}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.2f}% val_acc={val_acc:.2f}% "
            f"lr={current_lr:.4f} ({time.time()-t0:.1f}s)"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch, best_acc,
                             cfg.checkpoint_dir, tag=f"{args.model}_best")

    save_checkpoint(model, optimizer, cfg.epochs, best_acc,
                     cfg.checkpoint_dir, tag=f"{args.model}_final")
    print(f"[{args.model}] training complete. Best val_acc={best_acc:.2f}%")


if __name__ == "__main__":
    main()