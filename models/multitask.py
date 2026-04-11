"""Unified multi-task model
"""

import torch
import torch.nn as nn

from models.vgg11 import VGG11Encoder
from models.layers import CustomDropout
from models.segmentation import _dec_block


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model."""

    def __init__(self, num_breeds: int = 37, seg_classes: int = 3, in_channels: int = 3,
                 classifier_path: str = "checkpoints/classifier.pth",
                 localizer_path: str = "checkpoints/localizer.pth",
                 unet_path: str = "checkpoints/unet.pth"):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels)

        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), CustomDropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), CustomDropout(0.5),
            nn.Linear(4096, num_breeds),
        )

        self.loc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), CustomDropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), CustomDropout(0.5),
            nn.Linear(4096, 4),
        )

        self.seg_dropout = CustomDropout(0.5)
        self.up5 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec5 = _dec_block(1024, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _dec_block(768, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _dec_block(384, 256)
        self.up2 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.dec2 = _dec_block(192, 128)
        self.up1 = nn.ConvTranspose2d(128, 32, 2, stride=2)
        self.dec1 = _dec_block(96, 64)
        self.out_conv = nn.Conv2d(64, seg_classes, 1)

        import gdown
        gdown.download(id="1P-ScGJasApYLTBDFUxY6aBPRPCKSSIkT", output=classifier_path, quiet=False)
        gdown.download(id="1hhtDbaY8rK_1utx2tLtOInNgSqLQOjro", output=localizer_path, quiet=False)
        gdown.download(id="1yEtWsSbi33Ak4ONIRUFytrSf8Bkg-uHC", output=unet_path, quiet=False)
        self._load_weights(classifier_path, localizer_path, unet_path)

    def _load_weights(self, clf_path, loc_path, unet_path):
        clf_sd = torch.load(clf_path, map_location='cpu')
        cls_sd = {k[5:]: v for k, v in clf_sd.items() if k.startswith('head.')}
        self.cls_head.load_state_dict(cls_sd)

        # load encoder from localizer checkpoint so BN running stats match loc_head
        loc_sd = torch.load(loc_path, map_location='cpu')
        enc_sd = {k[8:]: v for k, v in loc_sd.items() if k.startswith('encoder.')}
        self.encoder.load_state_dict(enc_sd)
        loc_head_sd = {k[5:]: v for k, v in loc_sd.items() if k.startswith('head.')}
        self.loc_head.load_state_dict(loc_head_sd)

        unet_sd = torch.load(unet_path, map_location='cpu')
        dec_sd = {k: v for k, v in unet_sd.items() if not k.startswith('encoder.')}
        self.load_state_dict(dec_sd, strict=False)

    def forward(self, x: torch.Tensor):
        bottleneck, feats = self.encoder(x, return_features=True)

        cls_out = self.cls_head(bottleneck)
        loc_out = self.loc_head(bottleneck)

        s = self.up5(bottleneck)
        s = self.dec5(torch.cat([s, feats['block5']], dim=1))
        s = self.seg_dropout(s)
        s = self.up4(s)
        s = self.dec4(torch.cat([s, feats['block4']], dim=1))
        s = self.up3(s)
        s = self.dec3(torch.cat([s, feats['block3']], dim=1))
        s = self.up2(s)
        s = self.dec2(torch.cat([s, feats['block2']], dim=1))
        s = self.up1(s)
        s = self.dec1(torch.cat([s, feats['block1']], dim=1))
        seg_out = self.out_conv(s)

        return {
            'classification': cls_out,
            'localization': loc_out,
            'segmentation': seg_out,
        }
