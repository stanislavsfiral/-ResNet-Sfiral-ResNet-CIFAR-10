"""
config.py
=========
Single source of truth for every hyperparameter used in this project.

Design rationale
-----------------
The paper's central claim is that residual connections improve *optimization*,
not that they change the hypothesis space. To make that claim testable, the
Plain-CNN and ResNet models in this project MUST be trained with exactly the
same dataset, data augmentation, batch size, optimizer, weight decay,
schedule, and random seed. All shared settings live here so both train.py
runs pull from one place and cannot silently drift apart.

The default numeric values mirror the CIFAR-10 recipe described in Sec. 4.2
of He et al. (2016): SGD, momentum 0.9, weight decay 1e-4, batch size 128,
initial LR 0.1 with 10x drops, and simple pad-4 + random-crop + h-flip
augmentation.
"""

import torch


class Config:
    # ---- Reproducibility -----------------------------------------------
    seed: int = 42

    # ---- Hardware ---------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2

    # ---- Data ---------------------------------------------------------
    data_root: str = "./data"
    num_classes: int = 10
    image_size: int = 32
    pad: int = 4  # for random-crop augmentation (paper: pad 4px each side)
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616)

    # ---- Training (identical for both models for a fair comparison) ---
    batch_size: int = 128
    epochs: int = 50  # Увеличили до 50 эпох для глубокого обучения на GPU
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_milestones = [30, 40]  # Скорректировали шаги снижения LR под 50 эпох
    lr_gamma: float = 0.1

    # ---- Model: how many residual blocks per stage (n in the paper's
    #      "6n+2" CIFAR formula). n=3 -> 20-layer ResNet, matching the
    #      smallest CIFAR-10 model in Table 6.
    n_blocks_per_stage: int = 3
    stage_channels = (16, 32, 64)

    # ---- Logging / checkpoints -----------------------------------------
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    plot_dir: str = "./plots"


cfg = Config()