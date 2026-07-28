# Add or modify code through Du Shenyu
"""Branch-specific Dual-Path VFM guidance loss for FGDC-Net.

This module distills two VFM teachers into lightweight detector branches:
classification features are guided by semantic direction and token relations,
while regression features are guided by foreground-weighted feature alignment
and spatial attention alignment.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DualPathVFMGuidanceLoss(nn.Module):
    """Branch-specific VFM guidance loss.

    Args:
        lambda_cls: Weight of classification-branch guidance.
        lambda_reg: Weight of regression-branch guidance.
        alpha: Weight of classification relation consistency loss.
        beta: Weight of regression spatial attention loss.
        max_tokens: Maximum token count for token-token relation matrices.
        eps: Numerical stability epsilon.
        reduction: Currently supports "mean" for multi-scale averaging.
        Forward inputs:
        f_cls_student: Tensor or list of tensors, each with shape (B, C, H, W).
        f_reg_student: Tensor or list of tensors, each with shape (B, C, H, W).
        t_cls_teacher: Tensor or list of tensors aligned by scale.
        t_reg_teacher: Tensor or list of tensors aligned by scale.
        fg_mask: Optional Tensor or list of tensors. Shape can be (B, 1, H, W) or (B, H, W). If None, full-map
            foreground is used.

    Returns:
        Dict with "loss_vfm", "loss_cls_cos", "loss_cls_rel",
        "loss_reg_fg", and "loss_reg_att". All values are tensors and support
        backward through student features only.
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_reg: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5,
        max_tokens: int = 256,
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__()
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}.")
        if reduction != "mean":
            raise ValueError(f'Unsupported reduction "{reduction}". Only "mean" is currently supported.')
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.beta = beta
        self.max_tokens = max_tokens
        self.eps = eps
        self.reduction = reduction

    def forward(
        self,
        f_cls_student,
        f_reg_student,
        t_cls_teacher,
        t_reg_teacher,
        fg_mask=None,
    ) -> dict[str, torch.Tensor]:
        """Compute branch-specific VFM guidance loss."""
        f_cls_list = self._as_list(f_cls_student)
        f_reg_list = self._as_list(f_reg_student)
        t_cls_list = self._as_list(t_cls_teacher)
        t_reg_list = self._as_list(t_reg_teacher)
        mask_list = self._as_list(fg_mask) if fg_mask is not None else [None] * len(f_reg_list)

        n = len(f_cls_list)
        if not (len(f_reg_list) == len(t_cls_list) == len(t_reg_list) == n):
            raise ValueError(
                "Multi-scale VFM inputs must have the same number of scales: "
                f"got cls_student={len(f_cls_list)}, reg_student={len(f_reg_list)}, "
                f"cls_teacher={len(t_cls_list)}, reg_teacher={len(t_reg_list)}."
            )
        if len(mask_list) not in {1, n}:
            raise ValueError(f"fg_mask must be a Tensor or a list with 1 or {n} items, got {len(mask_list)}.")
        if len(mask_list) == 1 and n > 1:
            mask_list = mask_list * n

        loss_cls_cos = []
        loss_cls_rel = []
        loss_reg_fg = []
        loss_reg_att = []

        for i, (fs_cls, fs_reg, tt_cls, tt_reg, mask) in enumerate(
            zip(f_cls_list, f_reg_list, t_cls_list, t_reg_list, mask_list)
        ):
            self._check_feature(fs_cls, f"f_cls_student[{i}]")
            self._check_feature(fs_reg, f"f_reg_student[{i}]")
            self._check_feature(tt_cls, f"t_cls_teacher[{i}]")
            self._check_feature(tt_reg, f"t_reg_teacher[{i}]")

            tt_cls = self._resize_like(tt_cls.detach(), fs_cls)
            tt_reg = self._resize_like(tt_reg.detach(), fs_reg)
            self._check_same_shape(fs_cls, tt_cls, f"classification scale {i}")
            self._check_same_shape(fs_reg, tt_reg, f"regression scale {i}")
            mask_i = self._normalize_mask(mask, fs_reg.shape[-2:], fs_reg)

            loss_cls_cos.append(self.cosine_alignment_loss(fs_cls, tt_cls))
            loss_cls_rel.append(self.relation_consistency_loss(fs_cls, tt_cls))
            loss_reg_fg.append(self.foreground_alignment_loss(fs_reg, tt_reg, mask_i))
            loss_reg_att.append(self.spatial_attention_loss(fs_reg, tt_reg, mask_i))

        out_cls_cos = torch.stack(loss_cls_cos).mean()
        out_cls_rel = torch.stack(loss_cls_rel).mean()
        out_reg_fg = torch.stack(loss_reg_fg).mean()
        out_reg_att = torch.stack(loss_reg_att).mean()
        total = self.lambda_cls * (out_cls_cos + self.alpha * out_cls_rel) + self.lambda_reg * (
            out_reg_fg + self.beta * out_reg_att
        )

        return {
            "loss_vfm": total,
            "loss_cls_cos": out_cls_cos,
            "loss_cls_rel": out_cls_rel,
            "loss_reg_fg": out_reg_fg,
            "loss_reg_att": out_reg_att,
        }

    @staticmethod
    def _as_list(x):
        """Convert a Tensor to [Tensor], or a tuple/list to list."""
        if torch.is_tensor(x):
            return [x]
        if isinstance(x, (list, tuple)):
            return list(x)
        raise TypeError(f"Expected Tensor, list, or tuple, got {type(x).__name__}.")

    @staticmethod
    def _resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Resize src spatially to ref using bilinear interpolation."""
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def _normalize_mask(self, mask, target_hw, ref: torch.Tensor) -> torch.Tensor:
        """Convert mask to (B, 1, H, W), resize to target_hw, and clamp to [0, 1].

        If mask is None, full-map foreground is used as a safe fallback.
        """
        if mask is None:
            return ref.new_ones((ref.shape[0], 1, *target_hw))
        if not torch.is_tensor(mask):
            raise TypeError(f"fg_mask must be a Tensor, list, tuple, or None, got {type(mask).__name__}.")
        mask = mask.to(device=ref.device, dtype=ref.dtype)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4:
            raise ValueError(f"fg_mask must have shape (B,H,W) or (B,1,H,W), got {tuple(mask.shape)}.")
        if mask.shape[1] != 1:
            raise ValueError(f"fg_mask channel dimension must be 1, got shape {tuple(mask.shape)}.")
        if mask.shape[0] != ref.shape[0]:
            raise ValueError(f"fg_mask batch size {mask.shape[0]} does not match feature batch size {ref.shape[0]}.")
        if mask.shape[-2:] != target_hw:
            mask = F.interpolate(mask, size=target_hw, mode="nearest")
        return mask.clamp_(0, 1)

    def cosine_alignment_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Semantic direction alignment at each spatial location."""
        teacher = teacher.detach()
        cos = F.cosine_similarity(student, teacher, dim=1, eps=self.eps)
        return 1.0 - cos.mean()

    def relation_consistency_loss(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Token-token relation consistency with optional token sampling."""
        teacher = teacher.detach()
        b, c, h, w = student.shape
        n = h * w
        student_tokens = student.reshape(b, c, n)
        teacher_tokens = teacher.reshape(b, c, n)

        if n > self.max_tokens:
            index = torch.linspace(0, n - 1, self.max_tokens, device=student.device).round().long()
            student_tokens = student_tokens.index_select(2, index)
            teacher_tokens = teacher_tokens.index_select(2, index)

        student_tokens = F.normalize(student_tokens, p=2, dim=1, eps=self.eps)
        teacher_tokens = F.normalize(teacher_tokens, p=2, dim=1, eps=self.eps)
        student_sim = torch.bmm(student_tokens.transpose(1, 2), student_tokens)
        teacher_sim = torch.bmm(teacher_tokens.transpose(1, 2), teacher_tokens)
        return F.mse_loss(student_sim, teacher_sim)

    def foreground_alignment_loss(
        self, student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Foreground-weighted feature alignment for the regression branch."""
        teacher = teacher.detach()
        c = student.shape[1]
        diff = (student - teacher).pow(2)
        return (diff * mask).sum() / (c * mask.sum() + self.eps)

    def spatial_attention_map(self, feat: torch.Tensor) -> torch.Tensor:
        """Compute L2-normalized spatial response map A(F)."""
        response = feat.pow(2).sum(dim=1, keepdim=True)
        norm = response.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1)
        return response / (norm + self.eps)

    def spatial_attention_loss(self, student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Foreground-weighted spatial attention alignment."""
        teacher = teacher.detach()
        student_att = self.spatial_attention_map(student)
        teacher_att = self.spatial_attention_map(teacher)
        return ((student_att - teacher_att).pow(2) * mask).sum() / (mask.sum() + self.eps)

    @staticmethod
    def _check_feature(x: torch.Tensor, name: str) -> None:
        if not torch.is_tensor(x):
            raise TypeError(f"{name} must be a Tensor, got {type(x).__name__}.")
        if x.ndim != 4:
            raise ValueError(f"{name} must have shape (B,C,H,W), got {tuple(x.shape)}.")

    @staticmethod
    def _check_same_shape(student: torch.Tensor, teacher: torch.Tensor, context: str) -> None:
        if student.shape != teacher.shape:
            raise ValueError(
                f"Student and teacher features must have the same shape after resize for {context}. "
                f"Got student={tuple(student.shape)}, teacher={tuple(teacher.shape)}. "
                "Please align channel dimensions outside the loss with projection layers."
            )


if __name__ == "__main__":
    torch.manual_seed(0)
    B, C, H, W = 2, 64, 20, 20
    f_cls_student = torch.randn(B, C, H, W, requires_grad=True)
    f_reg_student = torch.randn(B, C, H, W, requires_grad=True)
    t_cls_teacher = torch.randn(B, C, H, W)
    t_reg_teacher = torch.randn(B, C, H, W)
    fg_mask = torch.rand(B, 1, H, W)

    loss_fn = DualPathVFMGuidanceLoss()
    loss_dict = loss_fn(f_cls_student, f_reg_student, t_cls_teacher, t_reg_teacher, fg_mask)
    for name, value in loss_dict.items():
        print(f"{name}: {value.item():.6f}")
    loss_dict["loss_vfm"].backward()
    print("backward ok")
