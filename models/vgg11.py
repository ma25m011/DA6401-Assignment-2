"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


def _make_block(in_c, out_c, num_convs, use_bn=True):
    layers = []
    for i in range(num_convs):
        c_in = in_c if i == 0 else out_c
        layers.append(nn.Conv2d(c_in, out_c, 3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.
    """

    def __init__(self, in_channels: int = 3, use_bn: bool = True):
        """Initialize the VGG11Encoder model."""
        super().__init__()
        self.block1 = _make_block(in_channels, 64, 1, use_bn)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.block2 = _make_block(64, 128, 1, use_bn)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.block3 = _make_block(128, 256, 2, use_bn)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.block4 = _make_block(256, 512, 2, use_bn)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.block5 = _make_block(512, 512, 2, use_bn)
        self.pool5 = nn.MaxPool2d(2, 2)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor.
            - if return_features=True: (bottleneck, feature_dict).
        """
        f1 = self.block1(x)
        x = self.pool1(f1)

        f2 = self.block2(x)
        x = self.pool2(f2)

        f3 = self.block3(x)
        x = self.pool3(f3)

        f4 = self.block4(x)
        x = self.pool4(f4)

        f5 = self.block5(x)
        x = self.pool5(f5)

        if return_features:
            return x, {'block1': f1, 'block2': f2, 'block3': f3, 'block4': f4, 'block5': f5}
        return x


VGG11 = VGG11Encoder
