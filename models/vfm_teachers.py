# Add or modify code through Du Shenyu
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Teacher encoder wrappers for FGDC-Net VFM distillation."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detectron2_swin import SwinTransformer


def _load_state_dict(path):
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "teacher", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    return ckpt


class ImageNetNormalizeResize(nn.Module):
    """Resize to teacher input size and apply ImageNet normalization."""

    def __init__(self, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std


class ViTBasePatch16Teacher(nn.Module):
    """ViT-B/16 teacher returning final patch-token feature map."""

    def __init__(self, weights=None, img_size=224, freeze=True):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm is required for ViTBasePatch16Teacher. Install with `pip install timm`.") from e

        self.preprocess = ImageNetNormalizeResize(img_size)
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.load_info = None
        if weights:
            self.load_info = self.model.load_state_dict(_load_state_dict(weights), strict=False)
        if freeze:
            self.freeze()

    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.preprocess(x)
        tokens = self.model.forward_features(x)  # (B, 197, 768) for 224 input
        if tokens.ndim == 3 and tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]  # drop cls token
        b, n, c = tokens.shape
        h = w = int(n**0.5)
        return tokens.transpose(1, 2).reshape(b, c, h, w).contiguous()


class SwinBaseTeacher(nn.Module):
    """Detectron2/MMDetection-style Swin-B teacher returning multi-stage feature maps."""

    def __init__(self, weights=None, img_size=224, freeze=True):
        super().__init__()
        self.preprocess = ImageNetNormalizeResize(img_size)
        self.model = SwinTransformer(
            pretrain_img_size=img_size,
            patch_size=4,
            in_chans=3,
            embed_dim=128,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
        )
        self.load_info = None
        if weights:
            msg = self.model.load_state_dict(_load_state_dict(weights), strict=True)
            self.load_info = {"missing": list(msg.missing_keys), "unexpected": list(msg.unexpected_keys)}
        if freeze:
            self.freeze()

    def freeze(self):
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.preprocess(x)
        return self.model(x)  # {"p0": Bx128x56x56, ..., "p3": Bx1024x7x7}


def build_vfm_teachers(cls_weights=None, reg_weights=None, img_size=224, freeze=True):
    """Build classification and regression VFM teachers from checkpoint paths."""
    cls_teacher = ViTBasePatch16Teacher(cls_weights, img_size=img_size, freeze=freeze) if cls_weights else None
    reg_teacher = SwinBaseTeacher(reg_weights, img_size=img_size, freeze=freeze) if reg_weights else None
    return cls_teacher, reg_teacher
