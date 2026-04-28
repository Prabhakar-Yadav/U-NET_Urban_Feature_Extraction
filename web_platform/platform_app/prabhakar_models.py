from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models import resnet18


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PrabhakarUNet(nn.Module):
    def __init__(self, in_channels: int = 4, num_classes: int = 7):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.bottleneck = ConvBlock(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.out_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        def up_cat(upconv, feat, skip):
            feat = upconv(feat)
            if feat.shape[2:] != skip.shape[2:]:
                feat = nn.functional.interpolate(feat, size=skip.shape[2:])
            return torch.cat([feat, skip], dim=1)

        d4 = self.dec4(up_cat(self.up4, b, e4))
        d3 = self.dec3(up_cat(self.up3, d4, e3))
        d2 = self.dec2(up_cat(self.up2, d3, e2))
        d1 = self.dec1(up_cat(self.up1, d2, e1))
        return self.out_conv(d1)


def load_prabhakar_unet(checkpoint_path: str, device: torch.device) -> tuple[PrabhakarUNet, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = PrabhakarUNet(
        in_channels=ckpt.get("in_channels", 4),
        num_classes=ckpt.get("num_classes", 7),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt


def load_prabhakar_resnet18(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        n_cls = len(ckpt.get("class_names", [None] * 6))
    else:
        state_dict = ckpt
        n_cls = state_dict["fc.weight"].shape[0]
        ckpt = {
            "class_names": None,
            "crop_size": 256,
            "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225],
        }
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_cls)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, ckpt


def load_prabhakar_maskrcnn(checkpoint_path: str, device: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", 2)
    model = maskrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    in_feat_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, ckpt
