# Add or modify code through Du Shenyu
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Dual-path VFM feature guider for FGDC-Net."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.vfm_guidance_loss import DualPathVFMGuidanceLoss


class DualPathVFMGuider(nn.Module):
    """Dual-path visual foundation model feature guider.

    The guider distills semantic knowledge from a DINO-style teacher into the
    classification branch, and localization knowledge from a Swin-style teacher
    into the regression branch. Teacher weight loading is intentionally left as
    an interface because DINOv3/Swin checkpoint key names may vary.

    Args:
        cls_channels: Channel count of the detector classification feature F_cls.
        reg_channels: Channel count of the detector regression feature F_reg.
        distill_dim: Shared feature dimension used for detector/teacher alignment.
        cls_teacher: Optional DINOv3/ViT teacher module.
        reg_teacher: Optional Swin Transformer teacher module.
        cls_weight_path: Optional path reserved for future DINOv3 weight loading.
        reg_weight_path: Optional path reserved for future Swin weight loading.
        enable_lora: If True, keeps teacher parameters with "lora" in their names trainable.
        lambda_cls: Weight for classification-branch VFM distillation.
        lambda_reg: Weight for regression-branch VFM distillation.
        freeze_teachers: Freeze non-LoRA teacher parameters.
    """

    def __init__(
        self,
        cls_channels,
        reg_channels,
        distill_dim=256,
        cls_teacher=None,
        reg_teacher=None,
        cls_teacher_channels=None,
        reg_teacher_channels=None,
        cls_weight_path=None,
        reg_weight_path=None,
        enable_lora=False,
        lambda_cls=1.0,
        lambda_reg=1.0,
        cls_loss="MSE",
        reg_loss="MSE",
        vfm_alpha=0.5,
        vfm_beta=0.5,
        vfm_max_tokens=256,
        freeze_teachers=True,
    ):
        super().__init__()
        self.cls_teacher = cls_teacher
        self.reg_teacher = reg_teacher
        self.cls_teacher_channels = cls_teacher_channels
        self.reg_teacher_channels = reg_teacher_channels
        self.cls_weight_path = Path(cls_weight_path) if cls_weight_path else None
        self.reg_weight_path = Path(reg_weight_path) if reg_weight_path else None
        self.enable_lora = enable_lora
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.cls_loss_name = self._normalize_loss_name(cls_loss)
        self.reg_loss_name = self._normalize_loss_name(reg_loss)
        self.use_branch_specific_loss = self.cls_loss_name in {"branchspecific", "branch_specific", "dual_path_vfm"} and (
            self.reg_loss_name in {"branchspecific", "branch_specific", "dual_path_vfm"}
        )
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
        self.last_cls_loss = None
        self.last_reg_loss = None
        self.last_cls_cos_loss = None
        self.last_cls_rel_loss = None
        self.last_reg_fg_loss = None
        self.last_reg_att_loss = None

        # Project detector branch features into the shared VFM distillation space.
        self.cls_proj = nn.Conv2d(cls_channels, distill_dim, 1)
        self.reg_proj = nn.Conv2d(reg_channels, distill_dim, 1)

        self.cls_teacher_proj = nn.Conv2d(cls_teacher_channels, distill_dim, 1) if cls_teacher_channels else None
        self.reg_teacher_proj = nn.Conv2d(reg_teacher_channels, distill_dim, 1) if reg_teacher_channels else None

        self._configure_teacher(self.cls_teacher, freeze_teachers)
        self._configure_teacher(self.reg_teacher, freeze_teachers)

    def set_teachers(
        self,
        cls_teacher=None,
        reg_teacher=None,
        cls_teacher_channels=None,
        reg_teacher_channels=None,
        freeze_teachers=True,
    ):
        """Attach or replace teacher encoders after initialization."""
        if cls_teacher is not None:
            self.cls_teacher = cls_teacher
            self.cls_teacher_channels = cls_teacher_channels or self.cls_teacher_channels
            if self.cls_teacher_channels:
                self.cls_teacher_proj = nn.Conv2d(self.cls_teacher_channels, self.cls_proj.out_channels, 1)
            self._configure_teacher(self.cls_teacher, freeze_teachers)
        if reg_teacher is not None:
            self.reg_teacher = reg_teacher
            self.reg_teacher_channels = reg_teacher_channels or self.reg_teacher_channels
            if self.reg_teacher_channels:
                self.reg_teacher_proj = nn.Conv2d(self.reg_teacher_channels, self.reg_proj.out_channels, 1)
            self._configure_teacher(self.reg_teacher, freeze_teachers)

    def set_weight_paths(self, cls_weight_path=None, reg_weight_path=None):
        """Store teacher checkpoint paths for later encoder-specific loading."""
        if cls_weight_path:
            self.cls_weight_path = Path(cls_weight_path)
        if reg_weight_path:
            self.reg_weight_path = Path(reg_weight_path)

    def forward(self, F_cls, F_reg, teacher_inputs=None, fg_mask=None, return_teacher=False):
        """Compute dual-path VFM distillation loss.

        Args:
            F_cls: Classification branch feature map, shape (B, C_cls, H, W).
            F_reg: Regression branch feature map, shape (B, C_reg, H, W).
            teacher_inputs: Optional input for teacher encoders. If omitted,
                F_cls is fed to the classification teacher and F_reg to the
                regression teacher. When using image-level VFM teachers, pass
                the corresponding image tensor or a dict with keys "cls"/"reg".
            return_teacher: If True, also return aligned teacher features.

        Returns:
            loss, or (loss, debug_dict) when return_teacher=True.
        """
        # Align detector features from both paths to the same distillation width.
        F_cls_proj = self.cls_proj(F_cls)
        F_reg_proj = self.reg_proj(F_reg)

        # If a teacher is not supplied yet, return a differentiable zero loss so
        # training code can keep the same call site while VFM setup is pending.
        zero = F_cls_proj.sum() * 0.0 + F_reg_proj.sum() * 0.0
        L_cls_vfm = zero
        L_reg_vfm = zero
        T_cls = None
        T_reg = None

        # DINOv3 semantic teacher guides classification-oriented features.
        cls_precomputed = self._teacher_input(teacher_inputs, "cls_feature", None)
        if cls_precomputed is not None:
            T_cls = self._to_feature_map(cls_precomputed, F_cls_proj.shape[-2:])
            T_cls = T_cls.to(device=F_cls_proj.device, dtype=F_cls_proj.dtype)
            if T_cls.shape[-2:] != F_cls_proj.shape[-2:]:
                T_cls = F.interpolate(T_cls, size=F_cls_proj.shape[-2:], mode="bilinear", align_corners=False)
            if self.cls_teacher_proj is None:
                self.cls_teacher_proj = nn.Conv2d(T_cls.shape[1], self.cls_proj.out_channels, 1).to(
                    device=F_cls_proj.device, dtype=F_cls_proj.dtype
                )
            else:
                self.cls_teacher_proj = self.cls_teacher_proj.to(device=F_cls_proj.device, dtype=F_cls_proj.dtype)
            T_cls = self.cls_teacher_proj(T_cls)
            L_cls_vfm = self._distill_loss(F_cls_proj, T_cls.detach(), self.cls_loss_name)
        elif self.cls_teacher is not None:
            cls_input = self._teacher_input(teacher_inputs, "cls", F_cls)
            if torch.is_tensor(cls_input):
                cls_input = cls_input.to(device=F_cls_proj.device, dtype=F_cls_proj.dtype)
            self.cls_teacher = self.cls_teacher.to(device=F_cls_proj.device)
            T_cls = self._teacher_feature(self.cls_teacher, cls_input, F_cls_proj.shape[-2:])
            T_cls = T_cls.to(device=F_cls_proj.device, dtype=F_cls_proj.dtype)
            self.cls_teacher_proj = self.cls_teacher_proj.to(device=F_cls_proj.device, dtype=F_cls_proj.dtype)
            T_cls = self.cls_teacher_proj(T_cls)
            L_cls_vfm = self._distill_loss(F_cls_proj, T_cls.detach(), self.cls_loss_name)

        # Swin localization teacher guides regression-oriented features.
        reg_precomputed = self._teacher_input(teacher_inputs, "reg_feature", None)
        if reg_precomputed is not None:
            T_reg = self._to_feature_map(reg_precomputed, F_reg_proj.shape[-2:])
            T_reg = T_reg.to(device=F_reg_proj.device, dtype=F_reg_proj.dtype)
            if T_reg.shape[-2:] != F_reg_proj.shape[-2:]:
                T_reg = F.interpolate(T_reg, size=F_reg_proj.shape[-2:], mode="bilinear", align_corners=False)
            if self.reg_teacher_proj is None:
                self.reg_teacher_proj = nn.Conv2d(T_reg.shape[1], self.reg_proj.out_channels, 1).to(
                    device=F_reg_proj.device, dtype=F_reg_proj.dtype
                )
            else:
                self.reg_teacher_proj = self.reg_teacher_proj.to(device=F_reg_proj.device, dtype=F_reg_proj.dtype)
            T_reg = self.reg_teacher_proj(T_reg)
            L_reg_vfm = self._distill_loss(F_reg_proj, T_reg.detach(), self.reg_loss_name)
        elif self.reg_teacher is not None:
            reg_input = self._teacher_input(teacher_inputs, "reg", F_reg)
            if torch.is_tensor(reg_input):
                reg_input = reg_input.to(device=F_reg_proj.device, dtype=F_reg_proj.dtype)
            self.reg_teacher = self.reg_teacher.to(device=F_reg_proj.device)
            T_reg = self._teacher_feature(self.reg_teacher, reg_input, F_reg_proj.shape[-2:])
            T_reg = T_reg.to(device=F_reg_proj.device, dtype=F_reg_proj.dtype)
            self.reg_teacher_proj = self.reg_teacher_proj.to(device=F_reg_proj.device, dtype=F_reg_proj.dtype)
            T_reg = self.reg_teacher_proj(T_reg)
            L_reg_vfm = self._distill_loss(F_reg_proj, T_reg.detach(), self.reg_loss_name)

        if self.branch_guidance_loss is not None and T_cls is not None and T_reg is not None:
            loss_dict = self.branch_guidance_loss(F_cls_proj, F_reg_proj, T_cls, T_reg, fg_mask=fg_mask)
            loss = loss_dict["loss_vfm"]
            L_cls_vfm = loss_dict["loss_cls_cos"] + self.branch_guidance_loss.alpha * loss_dict["loss_cls_rel"]
            L_reg_vfm = loss_dict["loss_reg_fg"] + self.branch_guidance_loss.beta * loss_dict["loss_reg_att"]
            self.last_cls_cos_loss = loss_dict["loss_cls_cos"].detach()
            self.last_cls_rel_loss = loss_dict["loss_cls_rel"].detach()
            self.last_reg_fg_loss = loss_dict["loss_reg_fg"].detach()
            self.last_reg_att_loss = loss_dict["loss_reg_att"].detach()
        else:
            loss = self.lambda_cls * L_cls_vfm + self.lambda_reg * L_reg_vfm
            self.last_cls_cos_loss = None
            self.last_cls_rel_loss = None
            self.last_reg_fg_loss = None
            self.last_reg_att_loss = None
        self.last_cls_loss = L_cls_vfm.detach()
        self.last_reg_loss = L_reg_vfm.detach()

        if not return_teacher:
            return loss
        return loss, {
            "F_cls_proj": F_cls_proj,
            "F_reg_proj": F_reg_proj,
            "T_cls": T_cls,
            "T_reg": T_reg,
            "L_cls_vfm": L_cls_vfm.detach(),
            "L_reg_vfm": L_reg_vfm.detach(),
            "L_cls_cos": self.last_cls_cos_loss,
            "L_cls_rel": self.last_cls_rel_loss,
            "L_reg_fg": self.last_reg_fg_loss,
            "L_reg_att": self.last_reg_att_loss,
        }

    @staticmethod
    def _normalize_loss_name(name):
        """Normalize yaml-configured VFM loss names."""
        return str(name or "MSE").replace("-", "_").lower()

    @staticmethod
    def _distill_loss(pred, target, loss_name):
        """Dispatch VFM distillation loss by name. Extend here for new loss methods."""
        if loss_name in {"mse", "mse_loss", "l2"}:
            return F.mse_loss(pred, target)
        if loss_name in {"l1", "mae"}:
            return F.l1_loss(pred, target)
        if loss_name in {"smooth_l1", "smoothl1", "huber"}:
            return F.smooth_l1_loss(pred, target)
        if loss_name in {"branchspecific", "branch_specific", "dual_path_vfm"}:
            return pred.sum() * 0.0
        raise ValueError(f"Unsupported VFM loss: {loss_name}")

    def _configure_teacher(self, teacher, freeze_teachers):
        """Set teacher trainability; LoRA parameters can remain trainable."""
        if teacher is None:
            return
        teacher.eval()
        if not freeze_teachers:
            return
        for name, p in teacher.named_parameters():
            is_lora = "lora" in name.lower()
            p.requires_grad = bool(self.enable_lora and is_lora)

    @staticmethod
    def _teacher_input(teacher_inputs, key, fallback):
        """Resolve which tensor to feed into a teacher encoder."""
        if teacher_inputs is None:
            return fallback
        if isinstance(teacher_inputs, dict):
            return teacher_inputs.get(key, teacher_inputs.get("image", fallback))
        return teacher_inputs

    def _teacher_feature(self, teacher, x, spatial_size):
        """Run a teacher and convert its output to a BCHW feature map."""
        with torch.set_grad_enabled(self.enable_lora):
            out = teacher(x)
        out = self._select_tensor(out)
        out = self._to_feature_map(out, spatial_size)
        if out.shape[-2:] != spatial_size:
            out = F.interpolate(out, size=spatial_size, mode="bilinear", align_corners=False)
        return out

    @staticmethod
    def _select_tensor(out):
        """Extract the most likely feature tensor from common teacher outputs."""
        if torch.is_tensor(out):
            return out
        if isinstance(out, dict):
            for key in ("last_hidden_state", "features", "feature", "x", "out"):
                if key in out and torch.is_tensor(out[key]):
                    return out[key]
            for value in out.values():
                if torch.is_tensor(value):
                    return value
        if isinstance(out, (list, tuple)):
            for value in reversed(out):
                if torch.is_tensor(value):
                    return value
        raise TypeError("Teacher encoder must return a Tensor, dict, list, or tuple containing a Tensor.")

    @staticmethod
    def _to_feature_map(x, spatial_size):
        """Convert teacher outputs in BCHW, BNC, or BC format to BCHW."""
        if x.ndim == 4:
            # Accept both BCHW and BHWC teacher features. Default to BCHW,
            # and only permute when the two middle dimensions look spatial.
            if x.shape[1] == x.shape[2] and x.shape[-1] > x.shape[1]:
                return x.permute(0, 3, 1, 2).contiguous()
            return x
        if x.ndim == 3:
            b, n, c = x.shape
            h, w = spatial_size
            if n == h * w + 1:
                x = x[:, 1:, :]  # drop ViT class token
                n -= 1
            if n == h * w:
                return x.transpose(1, 2).reshape(b, c, h, w).contiguous()
            pooled = x.mean(1).view(b, c, 1, 1)
            return F.interpolate(pooled, size=spatial_size, mode="bilinear", align_corners=False)
        if x.ndim == 2:
            return x[:, :, None, None].expand(-1, -1, *spatial_size)
        raise ValueError(f"Unsupported teacher feature shape: {tuple(x.shape)}")
