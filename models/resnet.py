"""
models/resnet.py
=================
CIFAR-10 ResNet, following the "6n+2" architecture family described in
Sec. 4.2 of He et al. (2016):

    - One initial 3x3 conv (16 filters).
    - Three stages of feature-map sizes {32, 32}, {16, 16}, {8, 8} for CIFAR
      (32x32 input), with n residual blocks per stage and filter counts
      {16, 32, 64} respectively.
    - Global average pooling -> 10-way fully-connected softmax classifier.
    - Total weighted layers = 6n + 2 (n blocks/stage x 3 stages x 2 conv
      layers/block, + 1 stem conv + 1 fc layer).

n=3 gives the 20-layer ResNet reported in Table 6 of the paper (the
smallest CIFAR-10 model discussed there), and is this project's default
(see config.py: n_blocks_per_stage).

No pretrained weights and no torchvision.models.resnet are used anywhere
in this file -- every layer is defined and connected manually.
"""

import torch
import torch.nn as nn

from models.residual_block import BasicResidualBlock


class CIFARResNet(nn.Module):
    """
    Manually-built CIFAR-style ResNet.

    Parameters
    ----------
    n_blocks_per_stage : int
        n in the paper's "6n+2" formula.
    stage_channels : tuple[int, int, int]
        Filter counts for the three stages (paper default: (16, 32, 64)).
    num_classes : int
        Number of output classes (10 for CIFAR-10).
    """

    def __init__(self, n_blocks_per_stage: int = 3,
                 stage_channels=(16, 32, 64), num_classes: int = 10):
        super().__init__()

        # ---- Stem: single 3x3 conv, as specified in Sec. 4.2 -------------
        self.in_channels = stage_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        # ---- Three stages of residual blocks ------------------------------
        # Stage 1: spatial size stays 32x32 (stride 1 throughout).
        # Stage 2: first block halves spatial size 32->16 (stride 2) while
        #          doubling channels 16->32, per the paper's rule that
        #          filter count doubles whenever feature-map size halves.
        # Stage 3: first block halves spatial size 16->8 (stride 2) while
        #          doubling channels 32->64.
        self.stage1 = self._make_stage(stage_channels[0], n_blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(stage_channels[1], n_blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(stage_channels[2], n_blocks_per_stage, stride=2)

        # ---- Head: global average pool + linear classifier ----------------
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_channels[2], num_classes)

        self._initialize_weights()

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Build one stage of `num_blocks` BasicResidualBlocks. Only the first
        block in the stage may change spatial resolution / channel count
        (stride possibly 2); every subsequent block in the stage uses
        stride=1 with in_channels == out_channels (a pure identity-shortcut
        block, Eqn. 1).
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicResidualBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        He (Kaiming) initialization for conv layers, matching the
        initialization scheme referenced in Sec. 3.4 of the paper ([12]:
        He et al., "Delving Deep into Rectifiers") -- appropriate because
        every conv in this network is followed by a ReLU (or feeds into a
        BN layer whose output then meets a ReLU).
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)          # [B, 16, 32, 32]
        x = self.stage1(x)        # [B, 16, 32, 32]
        x = self.stage2(x)        # [B, 32, 16, 16]
        x = self.stage3(x)        # [B, 64, 8, 8]
        x = self.avgpool(x)       # [B, 64, 1, 1]
        x = torch.flatten(x, 1)   # [B, 64]
        x = self.fc(x)            # [B, num_classes]
        return x


def resnet_cifar(n_blocks_per_stage: int = 3, stage_channels=(16, 32, 64),
                  num_classes: int = 10) -> CIFARResNet:
    """Factory matching the naming convention used in train.py / config.py."""
    return CIFARResNet(n_blocks_per_stage, stage_channels, num_classes)