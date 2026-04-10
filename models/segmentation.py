"""Segmentation model
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout


def _dec_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels)
        self.dropout = CustomDropout(dropout_p)

        # up5: 512@7 → 512@14, concat f5(512) → 1024 → 512
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _dec_block(512 + 512, 512)

        # up4: 512@14 → 256@28, concat f4(512) → 768 → 256
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _dec_block(256 + 512, 256)

        # up3: 256@28 → 128@56, concat f3(256) → 384 → 256
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _dec_block(128 + 256, 256)

        # up2: 256@56 → 64@112, concat f2(128) → 192 → 128
        self.up2 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec2 = _dec_block(64 + 128, 128)

        # up1: 128@112 → 32@224, concat f1(64) → 96 → 64
        self.up1 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2)
        self.dec1 = _dec_block(32 + 64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bottleneck, feats = self.encoder(x, return_features=True)

        x = self.up5(bottleneck)
        x = self.dec5(torch.cat([x, feats['block5']], dim=1))
        x = self.dropout(x)

        x = self.up4(x)
        x = self.dec4(torch.cat([x, feats['block4']], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, feats['block3']], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, feats['block2']], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, feats['block1']], dim=1))

        return self.out_conv(x)
