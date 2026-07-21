# Add or modify code through Du Shenyu
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""FGDC-Net modules for Ultralytics YOLO models.

The modules in this file are optional and keep the official Detect output contract:
training returns {"boxes", "scores", "feats"}, while inference/export follows the
base Detect decoding path.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.vfm_guidance_loss import DualPathVFMGuidanceLoss

from .conv import Conv, DWConv
from .head import Detect


class ChannelImportancePartition(nn.Module):
    """Split a feature map into classification-oriented and regression-oriented paths."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self.score(x)
        return x * weight, x * (1.0 - weight)


class FGDH(nn.Module):
    """Fine-Grained Enhanced Classification Head.

    It combines a local texture stream and a context stream with compact bilinear fusion, then predicts dense class
    logits for each feature-map location.
    """

    def __init__(self, c1: int, nc: int, hidden: int | None = None, compact_dim: int = 128):
        super().__init__()
        hidden = hidden or max(c1, min(nc, 100))
        compact_dim = max(min(compact_dim, hidden), 16)

        self.stream_a = nn.Sequential(
            DWConv(c1, c1, 3),
            Conv(c1, hidden, 1),
            DWConv(hidden, hidden, 3),
            Conv(hidden, hidden, 1),
        )
        self.stream_b = nn.Sequential(
            Conv(c1, hidden, 1),
            Conv(hidden, hidden, 3, d=2),
            Conv(hidden, hidden, 1),
        )
        self.proj_a = nn.Conv2d(hidden, compact_dim, 1)
        self.proj_b = nn.Conv2d(hidden, compact_dim, 1)
        self.pred = nn.Conv2d(compact_dim, nc, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two streams encode local texture and semantic context separately.
        a = self.proj_a(self.stream_a(x))
        b = self.proj_b(self.stream_b(x))

        # Compact bilinear fusion with signed square-root and L2 normalization.
        z = a * b
        z = torch.sign(z) * torch.sqrt(torch.abs(z) + 1e-6)
        z = F.normalize(z, p=2, dim=1)
        return self.pred(z)


class DualPathVFMGuider(nn.Module):
    """Training-time dual-path feature distillation helper.

    Teacher features can be precomputed by external encoders and passed in as tensors. This class intentionally does not
    force a specific teacher implementation so the Detect graph remains export-friendly.
    """

    def __init__(
        self,
        c_cls: int,
        c_reg: int,
        distill_dim: int = 128,
        teacher_cls_channels: int | None = None,
        teacher_reg_channels: int | None = None,
        lambda_cls: float = 1.0,
        lambda_reg: float = 1.0,
        cls_loss: str = "MSE",
        reg_loss: str = "MSE",
        vfm_alpha: float = 0.5,
        vfm_beta: float = 0.5,
        vfm_max_tokens: int = 256,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.cls_loss_name = self._normalize_loss_name(cls_loss)
        self.reg_loss_name = self._normalize_loss_name(reg_loss)
        self.use_branch_specific_loss = self.cls_loss_name in {
            "branchspecific",
            "branch_specific",
            "dual_path_vfm",
        } and (self.reg_loss_name in {"branchspecific", "branch_specific", "dual_path_vfm"})
        self.branch_guidance_loss = (
            DualPathVFMGuidanceLoss(
                lambda_cls=lambda_cls,
                lambda_reg=lambda_reg,
                alpha=vfm_alpha,
                beta=vfm_beta,
                max_tokens=vfm_max_tokens,
            )
            if self.use_branch_specific_loss
            else None
        )
        self.cls_proj = nn.Conv2d(c_cls, distill_dim, 1)
        self.reg_proj = nn.Conv2d(c_reg, distill_dim, 1)
        self.teacher_cls_proj = (
            nn.Conv2d(teacher_cls_channels, distill_dim, 1) if teacher_cls_channels is not None else nn.Identity()
        )
        self.teacher_reg_proj = (
            nn.Conv2d(teacher_reg_channels, distill_dim, 1) if teacher_reg_channels is not None else nn.Identity()
        )
        self.last_cls_loss = None
        self.last_reg_loss = None

    @staticmethod
    def _normalize_loss_name(name: str) -> str:
        """Normalize yaml-configured VFM loss names."""
        return str(name or "MSE").replace("-", "_").lower()

    @staticmethod
    def _distill_loss(pred: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
        """Dispatch VFM distillation loss by name. Extend here for new loss methods."""
        if loss_name in {"mse", "mse_loss", "l2"}:
            return F.mse_loss(pred, target)
        if loss_name in {"l1", "mae"}:
            return F.l1_loss(pred, target)
        if loss_name in {"smooth_l1", "smoothl1", "huber"}:
            return F.smooth_l1_loss(pred, target)
        raise ValueError(f"Unsupported VFM loss: {loss_name}")

    def forward(
        self,
        f_cls: torch.Tensor,
        f_reg: torch.Tensor,
        t_cls: torch.Tensor | None = None,
        t_reg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        zero = f_cls.new_zeros(())
        self.last_cls_loss = zero
        self.last_reg_loss = zero
        if t_cls is None or t_reg is None:
            return zero

        f_cls = self.cls_proj(f_cls)
        f_reg = self.reg_proj(f_reg)
        t_cls = t_cls.detach().to(device=f_cls.device, dtype=f_cls.dtype)
        t_reg = t_reg.detach().to(device=f_reg.device, dtype=f_reg.dtype)
        self.teacher_cls_proj = self.teacher_cls_proj.to(device=f_cls.device, dtype=f_cls.dtype)
        self.teacher_reg_proj = self.teacher_reg_proj.to(device=f_reg.device, dtype=f_reg.dtype)
        t_cls = self.teacher_cls_proj(t_cls)
        t_reg = self.teacher_reg_proj(t_reg)
        t_cls = F.interpolate(t_cls, size=f_cls.shape[-2:], mode="bilinear", align_corners=False)
        t_reg = F.interpolate(t_reg, size=f_reg.shape[-2:], mode="bilinear", align_corners=False)
        if self.branch_guidance_loss is not None:
            loss_dict = self.branch_guidance_loss(f_cls, f_reg, t_cls, t_reg, fg_mask=None)
            self.last_cls_loss = loss_dict["loss_cls_cos"] + self.branch_guidance_loss.alpha * loss_dict["loss_cls_rel"]
            self.last_reg_loss = loss_dict["loss_reg_fg"] + self.branch_guidance_loss.beta * loss_dict["loss_reg_att"]
            return loss_dict["loss_vfm"]
        self.last_cls_loss = self._distill_loss(f_cls, t_cls, self.cls_loss_name)
        self.last_reg_loss = self._distill_loss(f_reg, t_reg, self.reg_loss_name)
        return self.lambda_cls * self.last_cls_loss + self.lambda_reg * self.last_reg_loss


class FGDCDetect(Detect):
    """FGDC-Net detection head for Ultralytics YOLO models.

    Args are intentionally compatible with YAML forms such as:
        - [[15, 18, 21], 1, FGDCDetect, [nc, {"use_fgdh": True}]]
    """

    def __init__(
        self,
        nc: int = 80,
        reg_max: int | dict = 16,
        end2end: bool = False,
        ch: tuple = (),
        proj_ratio: float = 1.0,
        min_proj_channels: int = 32,
        reduction: int = 16,
        use_fgdh: bool = True,
        fgdh_compact_dim: int = 128,
        use_vfm: bool = False,
        vfm_distill_dim: int = 128,
        vfm_lambda_cls: float = 1.0,
        vfm_lambda_reg: float = 1.0,
        VFM_cls_loss: str = "MSE",
        VFM_reg_loss: str = "MSE",
        vfm_alpha: float = 0.5,
        vfm_beta: float = 0.5,
        vfm_max_tokens: int = 256,
        vfm_cls_weights: str | None = None,
        vfm_reg_weights: str | None = None,
        vfm_imgsz: int = 224,
        **kwargs,
    ):
        parsed_reg_max, parsed_end2end = 16, end2end
        if isinstance(proj_ratio, (list, tuple)) and isinstance(reg_max, dict):
            # Ultralytics parse_model appends [reg_max, end2end, ch] after YAML args.
            parsed_reg_max, parsed_end2end, ch = end2end, ch, proj_ratio
            proj_ratio = 1.0
        elif isinstance(end2end, (list, tuple)) and not ch:
            ch = end2end
            end2end = False
        if ch is None:
            ch = ()
        if isinstance(reg_max, dict):
            opts = dict(reg_max)
            reg_max = opts.pop("reg_max", parsed_reg_max)
            end2end = opts.pop("end2end", parsed_end2end)
            proj_ratio = opts.pop("proj_ratio", proj_ratio)
            min_proj_channels = opts.pop("min_proj_channels", min_proj_channels)
            reduction = opts.pop("reduction", reduction)
            use_fgdh = opts.pop("use_fgdh", use_fgdh)
            fgdh_compact_dim = opts.pop("fgdh_compact_dim", fgdh_compact_dim)
            use_vfm = opts.pop("use_vfm", use_vfm)
            vfm_distill_dim = opts.pop("vfm_distill_dim", vfm_distill_dim)
            vfm_lambda_cls = opts.pop("vfm_lambda_cls", vfm_lambda_cls)
            vfm_lambda_reg = opts.pop("vfm_lambda_reg", vfm_lambda_reg)
            VFM_cls_loss = opts.pop("VFM_cls_loss", VFM_cls_loss)
            VFM_reg_loss = opts.pop("VFM_reg_loss", VFM_reg_loss)
            vfm_alpha = opts.pop("vfm_alpha", vfm_alpha)
            vfm_beta = opts.pop("vfm_beta", vfm_beta)
            vfm_max_tokens = opts.pop("vfm_max_tokens", vfm_max_tokens)
            vfm_cls_weights = opts.pop("vfm_cls_weights", vfm_cls_weights)
            vfm_reg_weights = opts.pop("vfm_reg_weights", vfm_reg_weights)
            vfm_imgsz = opts.pop("vfm_imgsz", vfm_imgsz)
            kwargs.update(opts)

        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=ch)
        self.use_fgdh = use_fgdh
        self.use_vfm = use_vfm
        self.vfm_loss = None
        self.vfm_cls_loss = None
        self.vfm_reg_loss = None
        self.vfm_teacher_features = None
        self.vfm_cls_weights = vfm_cls_weights
        self.vfm_reg_weights = vfm_reg_weights
        self.vfm_imgsz = vfm_imgsz
        self.vfm_weight_status = self._check_vfm_weight_paths(vfm_cls_weights, vfm_reg_weights)

        proj_ch = [max(min_proj_channels, int(c * proj_ratio)) for c in ch]
        self.proj = nn.ModuleList(Conv(c, p, 1) for c, p in zip(ch, proj_ch))
        self.partition = nn.ModuleList(ChannelImportancePartition(p, reduction) for p in proj_ch)

        c2 = max((16, proj_ch[0] // 4, self.reg_max * 4))
        c3 = max(proj_ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(p, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for p in proj_ch
        )
        if use_fgdh:
            self.cv3 = nn.ModuleList(FGDH(p, self.nc, hidden=c3, compact_dim=fgdh_compact_dim) for p in proj_ch)
        else:
            self.cv3 = nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(p, p, 3), Conv(p, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for p in proj_ch
            )

        self.vfm_guiders = (
            nn.ModuleList(
                DualPathVFMGuider(
                    p,
                    p,
                    distill_dim=vfm_distill_dim,
                    lambda_cls=vfm_lambda_cls,
                    lambda_reg=vfm_lambda_reg,
                    cls_loss=VFM_cls_loss,
                    reg_loss=VFM_reg_loss,
                    vfm_alpha=vfm_alpha,
                    vfm_beta=vfm_beta,
                    vfm_max_tokens=vfm_max_tokens,
                )
                for p in proj_ch
            )
            if use_vfm
            else None
        )

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @staticmethod
    def _check_vfm_weight_paths(cls_weights: str | None, reg_weights: str | None) -> dict[str, bool]:
        """Record whether configured VFM teacher checkpoint paths exist."""
        status = {"cls": False, "reg": False}
        if cls_weights:
            status["cls"] = Path(cls_weights).is_file()
        if reg_weights:
            status["reg"] = Path(reg_weights).is_file()
        return status

    def load_vfm_teacher_weights(self) -> dict[str, bool]:
        """Public hook for training code to verify configured teacher weight paths."""
        self.vfm_weight_status = self._check_vfm_weight_paths(self.vfm_cls_weights, self.vfm_reg_weights)
        return self.vfm_weight_status

    def set_vfm_teacher_features(self, features) -> None:
        """Set optional teacher feature tensors for the next training forward pass."""
        self.vfm_teacher_features = features

    def _teacher_pair(self, i: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        feats = self.vfm_teacher_features
        if feats is None:
            return None, None
        if isinstance(feats, dict):
            cls_feats = feats.get("cls") or feats.get("cls_features")
            reg_feats = feats.get("reg") or feats.get("reg_features")
        else:
            cls_feats, reg_feats = feats
        t_cls = cls_feats[i] if isinstance(cls_feats, (list, tuple)) else cls_feats
        t_reg = reg_feats[i] if isinstance(reg_feats, (list, tuple)) else reg_feats
        return t_cls, t_reg

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes, scores, feats = [], [], []
        vfm_loss = x[0].new_zeros(())
        vfm_cls_loss = x[0].new_zeros(())
        vfm_reg_loss = x[0].new_zeros(())

        for i in range(self.nl):
            feat = self.proj[i](x[i])
            f_cls, f_reg = self.partition[i](feat)
            feats.append(feat)
            boxes.append(box_head[i](f_reg).view(bs, 4 * self.reg_max, -1))
            scores.append(cls_head[i](f_cls).view(bs, self.nc, -1))
            if self.training and self.vfm_guiders is not None:
                t_cls, t_reg = self._teacher_pair(i)
                loss_i = self.vfm_guiders[i](f_cls, f_reg, t_cls, t_reg)
                vfm_loss = vfm_loss + loss_i
                vfm_cls_loss = vfm_cls_loss + self.vfm_guiders[i].lambda_cls * self.vfm_guiders[i].last_cls_loss
                vfm_reg_loss = vfm_reg_loss + self.vfm_guiders[i].lambda_reg * self.vfm_guiders[i].last_reg_loss

        if self.training and self.vfm_guiders is not None:
            self.vfm_loss = vfm_loss
            self.vfm_cls_loss = vfm_cls_loss.detach()
            self.vfm_reg_loss = vfm_reg_loss.detach()
        else:
            self.vfm_loss = self.vfm_cls_loss = self.vfm_reg_loss = None
        return dict(boxes=torch.cat(boxes, dim=-1), scores=torch.cat(scores, dim=-1), feats=feats)

    def bias_init(self):
        """Initialize FGDCDetect biases."""
        for i, box_head in enumerate(self.cv2):
            box_head[-1].bias.data[:] = 2.0
            cls_head = self.cv3[i].pred if isinstance(self.cv3[i], FGDH) else self.cv3[i][-1]
            cls_head.bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, box_head in enumerate(self.one2one_cv2):
                box_head[-1].bias.data[:] = 2.0
                cls_head = (
                    self.one2one_cv3[i].pred if isinstance(self.one2one_cv3[i], FGDH) else self.one2one_cv3[i][-1]
                )
                cls_head.bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
