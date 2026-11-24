"""Unified 3-class change detection model factory for Milestone 4 recovery.

Supported architectures (all consume T1, T2 in [-1, 1] of shape [B, 3, H, W]
and return 3-class logits [B, 3, H, W] for 0=background, 1=road, 2=building):
  - bit_resnet18 : Official pinned BIT_CD BASE_Transformer (ResNet18 backbone)
  - ef_unet      : Early-Fusion U-Net (FC-EF style 4-stage Conv U-Net)
  - fc_siam_diff : Fully Convolutional Siamese-Difference U-Net (Daudt et al.)
  - changeformer : Official ChangeFormerV6 (Siamese MiT transformer)
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EarlyFusionUNet(nn.Module):
    """FC-EF style 4-stage U-Net taking concatenated [T1, T2] (6 channels)."""

    def __init__(self, num_classes: int = 3, base_ch: int = 32, dropout: float = 0.1):
        super().__init__()
        c = base_ch
        self.enc1 = ConvBlock(6, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4, dropout=dropout)
        self.enc4 = ConvBlock(c * 4, c * 8, dropout=dropout)
        self.bottleneck = ConvBlock(c * 8, c * 16, dropout=dropout)

        self.pool = nn.MaxPool2d(2)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class FCSiamDiff(nn.Module):
    """Daudt et al. FC-Siam-Diff: shared 4-stage Siamese encoder with |f1 - f2| skips."""

    def __init__(self, num_classes: int = 3, base_ch: int = 32, dropout: float = 0.1):
        super().__init__()
        c = base_ch
        self.enc1 = ConvBlock(3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4, dropout=dropout)
        self.enc4 = ConvBlock(c * 4, c * 8, dropout=dropout)
        self.bottleneck = ConvBlock(c * 8, c * 16, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.head = nn.Conv2d(c, num_classes, 1)

    def encode(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        return e1, e2, e3, e4, b

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        a1, a2, a3, a4, ab = self.encode(x1)
        b1, b2, b3, b4, bb = self.encode(x2)
        d = torch.abs(ab - bb)
        d4 = self.dec4(torch.cat([self.up4(d), torch.abs(a4 - b4)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), torch.abs(a3 - b3)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), torch.abs(a2 - b2)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), torch.abs(a1 - b1)], dim=1))
        return self.head(d1)


class SiameseResNetUNet(nn.Module):
    """ImageNet-pretrained Siamese ResNet18 + feature difference decoder (strong CNN baseline)."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        import torchvision.models as tvm

        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.ModuleDict({
            "stem": nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),
            "maxpool": resnet.maxpool,
            "layer1": resnet.layer1,
            "layer2": resnet.layer2,
            "layer3": resnet.layer3,
            "layer4": resnet.layer4,
        })
        # Difference + concat features at each scale: [64, 64, 128, 256, 512]
        self.dec4 = ConvBlock(512 * 2, 256)
        self.dec3 = ConvBlock(256 + 256 * 2, 128)
        self.dec2 = ConvBlock(128 + 128 * 2, 64)
        self.dec1 = ConvBlock(64 + 64 * 2, 64)
        self.dec0 = ConvBlock(64 + 64 * 2, 32)
        self.head = nn.Conv2d(32, num_classes, 1)

    def encode(self, x: torch.Tensor):
        f0 = self.backbone["stem"](x)         # /2, 64
        f1 = self.backbone["layer1"](self.backbone["maxpool"](f0))  # /4, 64
        f2 = self.backbone["layer2"](f1)      # /8, 128
        f3 = self.backbone["layer3"](f2)      # /16, 256
        f4 = self.backbone["layer4"](f3)      # /32, 512
        return f0, f1, f2, f3, f4

    @staticmethod
    def _fuse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.abs(a - b), a + b], dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        a0, a1, a2, a3, a4 = self.encode(x1)
        b0, b1, b2, b3, b4 = self.encode(x2)
        x = self.dec4(self._fuse(a4, b4))
        x = F.interpolate(x, size=a3.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, self._fuse(a3, b3)], dim=1))
        x = F.interpolate(x, size=a2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, self._fuse(a2, b2)], dim=1))
        x = F.interpolate(x, size=a1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat([x, self._fuse(a1, b1)], dim=1))
        x = F.interpolate(x, size=a0.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec0(torch.cat([x, self._fuse(a0, b0)], dim=1))
        x = F.interpolate(x, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x)


def _install_torchvision_legacy_shim() -> None:
    legacy_utils = ModuleType("torchvision.models.utils")
    legacy_utils.load_state_dict_from_url = torch.hub.load_state_dict_from_url
    sys.modules.setdefault(legacy_utils.__name__, legacy_utils)


def _reset_models_pkg(source: Path) -> None:
    for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
        sys.modules.pop(k, None)
    s = str(source)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


def build_model(
    arch: str,
    device: str,
    bit_source_dir: str | None = None,
    changeformer_source_dir: str | None = None,
) -> Tuple[nn.Module, List[nn.Parameter], List[nn.Parameter], nn.Module | None]:
    """Return (model, backbone_params, head_params, backbone_module_or_None)."""
    if arch == "bit_resnet18":
        if not bit_source_dir:
            raise ValueError("--bit-source-dir is required for bit_resnet18")
        source = Path(bit_source_dir).resolve()
        if not (source / "models/networks.py").is_file():
            raise FileNotFoundError(f"BIT source_dir missing models/networks.py: {source}")
        _install_torchvision_legacy_shim()
        _reset_models_pkg(source)
        from models.networks import BASE_Transformer

        model = BASE_Transformer(
            input_nc=3,
            output_nc=3,
            with_pos="learned",
            resnet_stages_num=4,
            token_len=4,
            enc_depth=1,
            dec_depth=8,
            decoder_dim_head=8,
            backbone="resnet18",
        ).to(device)
        backbone_params = list(model.resnet.parameters())
        bb_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model.parameters() if id(p) not in bb_ids]
        return model, backbone_params, head_params, model.resnet

    if arch == "ef_unet":
        model = EarlyFusionUNet(num_classes=3, base_ch=32, dropout=0.1).to(device)
        return model, [], list(model.parameters()), None

    if arch == "fc_siam_diff":
        model = FCSiamDiff(num_classes=3, base_ch=32, dropout=0.1).to(device)
        return model, [], list(model.parameters()), None

    if arch == "siam_resnet18_unet":
        model = SiameseResNetUNet(num_classes=3).to(device)
        backbone_params = list(model.backbone.parameters())
        bb_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model.parameters() if id(p) not in bb_ids]
        return model, backbone_params, head_params, model.backbone

    if arch == "changeformer":
        if not changeformer_source_dir:
            raise ValueError("--changeformer-source-dir is required for changeformer")
        source = Path(changeformer_source_dir).resolve()
        if not (source / "models/ChangeFormer.py").is_file():
            raise FileNotFoundError(f"ChangeFormer source_dir missing models/ChangeFormer.py: {source}")
        _install_torchvision_legacy_shim()
        _reset_models_pkg(source)
        from models.ChangeFormer import ChangeFormerV6

        class _CFWrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = ChangeFormerV6(input_nc=3, output_nc=3, embed_dim=256)

            def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
                out = self.net(x1, x2)
                return out[-1] if isinstance(out, (list, tuple)) else out

        model = _CFWrapper().to(device)
        return model, [], list(model.parameters()), None

    raise ValueError(f"Unknown detector architecture: {arch}")
