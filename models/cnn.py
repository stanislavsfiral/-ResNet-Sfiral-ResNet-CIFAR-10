"""
models/cnn.py
=============
Plain CNN baseline -- architecturally identical to models/resnet.py
(same depth, same stage widths, same stem, same head) with ONE difference:
no shortcut connections. This isolates the effect of residual learning
per Sec. 4.1/4.2 of He et al. (2016), which stresses that plain and
residual networks in their experiments are otherwise matched in depth,
width, and (for identity shortcuts) parameter count.

Each "PlainBlock" here mirrors BasicResidualBlock's two 3x3 conv-BN-ReLU
layers, but the addition-with-shortcut step (Eqn. 1) is simply omitted --
the block only computes and returns F(x) directly, i.e. it must learn the
full mapping H(x) from scratch rather than a correction relative to x.
This is exactly the "plain net" the paper trains as its point of
comparison (Fig. 3 middle / Fig. 1).
"""

import torch
import torch.nn as nn


class PlainBlock(nn.Module):
    """Two 3x3 conv-BN-ReLU layers, NO shortcut connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)   # no shortcut addition here -- this is the
                                # single architectural difference vs.
                                # BasicResidualBlock
        return out


class CIFARPlainCNN(nn.Module):
    """
    Plain-CNN counterpart to CIFARResNet: identical stem, stage structure
    (n blocks per stage, same channel progression 16->32->64, same
    stride-2 downsampling points), and head -- but built from PlainBlock
    instead of BasicResidualBlock.
    """

    def __init__(self, n_blocks_per_stage: int = 3,
                 stage_channels=(16, 32, 64), num_classes: int = 10):
        super().__init__()

        self.in_channels = stage_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(stage_channels[0], n_blocks_per_stage, stride=1)
        self.stage2 = self._make_stage(stage_channels[1], n_blocks_per_stage, stride=2)
        self.stage3 = self._make_stage(stage_channels[2], n_blocks_per_stage, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_channels[2], num_classes)

        self._initialize_weights()

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(PlainBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Identical initialization scheme to CIFARResNet, for a fair comparison."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def plain_cnn_cifar(n_blocks_per_stage: int = 3, stage_channels=(16, 32, 64),
                     num_classes: int = 10) -> CIFARPlainCNN:
    """Factory matching resnet_cifar's signature exactly."""
    return CIFARPlainCNN(n_blocks_per_stage, stage_channels, num_classes)