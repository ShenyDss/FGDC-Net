# Add or modify code through Du Shenyu
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Dual-path detection head modules for YOLOv5."""

import math

import torch
import torch.nn as nn

from models.common import Conv
from models.fgdh import FGDH
from models.vfm_guider import DualPathVFMGuider
from utils.general import check_version


class ChannelImportancePartition(nn.Module):
    """Build classification- and regression-oriented features with learned channel importance."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels * 2, 1, bias=True),
        )

    def forward(self, x):
        cls_score, reg_score = self.score(self.pool(x)).chunk(2, 1)
        weights = torch.softmax(torch.stack((cls_score, reg_score), dim=1), dim=1)
        cls_feat = x * weights[:, 0]
        reg_feat = x * weights[:, 1]
        return cls_feat, reg_feat


class DualPathDetect(nn.Module):
    """YOLOv5-compatible detection head with classification/regression feature partition."""

    stride = None
    dynamic = False
    export = False

    def __init__(
        self,
        nc=80,
        anchors=(),
        ch=(),
        inplace=True,
        proj_ratio=0.5,
        reduction=4,
        min_proj_channels=16,
        use_fgdh=False,
        fgdh_compact_dim=256,
        use_vfm=False,
        vfm_distill_dim=256,
        vfm_lambda_cls=1.0,
        vfm_lambda_reg=1.0,
        VFM_cls_loss="MSE",
        VFM_reg_loss="MSE",
        vfm_alpha=0.5,
        vfm_beta=0.5,
        vfm_max_tokens=256,
    ):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.empty(0) for _ in range(self.nl)]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.inplace = inplace
        self.use_fgdh = use_fgdh
        self.use_vfm = use_vfm
        self.use_vfm_guider = use_vfm
        self.vfm_loss = None
        self.vfm_cls_loss = None
        self.vfm_reg_loss = None
        self.vfm_cls_cos_loss = None
        self.vfm_cls_rel_loss = None
        self.vfm_reg_fg_loss = None
        self.vfm_reg_att_loss = None
        self.vfm_debug = None
        self.vfm_teacher_inputs = None
        self.vfm_targets = None

        proj_ch = [max(int(c * proj_ratio), min_proj_channels) for c in ch]
        self.proj = nn.ModuleList(Conv(c1, c2, 1, 1) for c1, c2 in zip(ch, proj_ch))
        self.partition = nn.ModuleList(ChannelImportancePartition(c, reduction) for c in proj_ch)
        self.cls_stem = (
            nn.ModuleList(
                FGDH(
                    c,
                    self.na * self.nc,
                    compact_dim=fgdh_compact_dim,
                    dense=True,
                    return_probs=False,
                )
                for c in proj_ch
            )
            if use_fgdh
            else nn.ModuleList(Conv(c, c, 3, 1) for c in proj_ch)
        )
        self.reg_stem = nn.ModuleList(Conv(c, c, 3, 1) for c in proj_ch)
        self.cls_pred = nn.ModuleList(nn.Identity() if use_fgdh else nn.Conv2d(c, self.na * self.nc, 1) for c in proj_ch)
        self.reg_pred = nn.ModuleList(nn.Conv2d(c, self.na * 4, 1) for c in proj_ch)
        self.obj_pred = nn.ModuleList(nn.Conv2d(c, self.na, 1) for c in proj_ch)
        vfm_cls_teacher_channels = 768
        vfm_reg_teacher_channels = [256, 512, 1024]
        self.vfm_guiders = (
            nn.ModuleList(
                DualPathVFMGuider(
                    c,
                    c,
                    distill_dim=vfm_distill_dim,
                    cls_teacher_channels=vfm_cls_teacher_channels,
                    reg_teacher_channels=vfm_reg_teacher_channels[i],
                    lambda_cls=vfm_lambda_cls,
                    lambda_reg=vfm_lambda_reg,
                    cls_loss=VFM_cls_loss,
                    reg_loss=VFM_reg_loss,
                    vfm_alpha=vfm_alpha,
                    vfm_beta=vfm_beta,
                    vfm_max_tokens=vfm_max_tokens,
                )
                for i, c in enumerate(proj_ch)
            )
            if use_vfm
            else nn.ModuleList()
        )

    def forward(self, x):
        """Return the same train/inference formats as YOLOv5 Detect."""
        z = []
        vfm_losses = []
        vfm_cls_losses = []
        vfm_reg_losses = []
        vfm_cls_cos_losses = []
        vfm_cls_rel_losses = []
        vfm_reg_fg_losses = []
        vfm_reg_att_losses = []
        self.vfm_loss = None
        self.vfm_cls_loss = None
        self.vfm_reg_loss = None
        self.vfm_cls_cos_loss = None
        self.vfm_cls_rel_loss = None
        self.vfm_reg_fg_loss = None
        self.vfm_reg_att_loss = None
        self.vfm_debug = None
        teacher_features = self._encode_vfm_teachers()
        for i in range(self.nl):
            feat = self.proj[i](x[i])
            cls_feat, reg_feat = self.partition[i](feat)
            if self.training and self.use_vfm:
                fg_mask = self._build_vfm_fg_mask(self.vfm_targets, reg_feat.shape[0], reg_feat.shape[-2:], reg_feat.device, reg_feat.dtype)
                vfm_loss = self.vfm_guiders[i](
                    cls_feat,
                    reg_feat,
                    teacher_inputs=self._vfm_features_for_scale(teacher_features, i),
                    fg_mask=fg_mask,
                )
                vfm_losses.append(vfm_loss)
                vfm_cls_losses.append(self.vfm_guiders[i].last_cls_loss)
                vfm_reg_losses.append(self.vfm_guiders[i].last_reg_loss)
                vfm_cls_cos_losses.append(self._loss_or_zero(self.vfm_guiders[i].last_cls_cos_loss, vfm_loss))
                vfm_cls_rel_losses.append(self._loss_or_zero(self.vfm_guiders[i].last_cls_rel_loss, vfm_loss))
                vfm_reg_fg_losses.append(self._loss_or_zero(self.vfm_guiders[i].last_reg_fg_loss, vfm_loss))
                vfm_reg_att_losses.append(self._loss_or_zero(self.vfm_guiders[i].last_reg_att_loss, vfm_loss))
            reg_feat = self.reg_stem[i](reg_feat)

            reg = self.reg_pred[i](reg_feat).view(x[i].shape[0], self.na, 4, x[i].shape[2], x[i].shape[3])
            obj = self.obj_pred[i](reg_feat).view(x[i].shape[0], self.na, 1, x[i].shape[2], x[i].shape[3])
            cls = self.cls_pred[i](self.cls_stem[i](cls_feat)).view(
                x[i].shape[0], self.na, self.nc, x[i].shape[2], x[i].shape[3]
            )
            x[i] = torch.cat((reg, obj, cls), 2).permute(0, 1, 3, 4, 2).contiguous()
            bs, _, ny, nx, _ = x[i].shape

            if not self.training:
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]
                wh = (wh * 2) ** 2 * self.anchor_grid[i]
                y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        if vfm_losses:
            self.vfm_loss = torch.stack(vfm_losses).mean()
            self.vfm_cls_loss = torch.stack(vfm_cls_losses).mean()
            self.vfm_reg_loss = torch.stack(vfm_reg_losses).mean()
            self.vfm_cls_cos_loss = torch.stack(vfm_cls_cos_losses).mean()
            self.vfm_cls_rel_loss = torch.stack(vfm_cls_rel_losses).mean()
            self.vfm_reg_fg_loss = torch.stack(vfm_reg_fg_losses).mean()
            self.vfm_reg_att_loss = torch.stack(vfm_reg_att_losses).mean()
        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def __getstate__(self):
        """Drop forward-time tensors so EMA/checkpoint deepcopy does not copy autograd graphs."""
        state = self.__dict__.copy()
        state["vfm_loss"] = None
        state["vfm_cls_loss"] = None
        state["vfm_reg_loss"] = None
        state["vfm_cls_cos_loss"] = None
        state["vfm_cls_rel_loss"] = None
        state["vfm_reg_fg_loss"] = None
        state["vfm_reg_att_loss"] = None
        state["vfm_debug"] = None
        state["vfm_teacher_inputs"] = None
        state["vfm_targets"] = None
        return state

    def initialize_biases(self, cf=None):
        """Initialize objectness/classification biases with YOLOv5 defaults."""
        for i, (obj, cls, s) in enumerate(zip(self.obj_pred, self.cls_pred, self.stride)):
            b = obj.bias.view(self.na, -1)
            b.data[:, 0] += math.log(8 / (640 / s) ** 2)
            obj.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

            if self.use_fgdh:
                cls = self.cls_stem[i].classifier
            b = cls.bias.view(self.na, -1)
            b.data += math.log(0.6 / (self.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            cls.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def set_vfm_teachers(self, cls_teacher=None, reg_teacher=None, freeze_teachers=True):
        """Attach DINO/Swin teacher encoders to all VFM guider scales."""
        if not self.use_vfm:
            raise RuntimeError("VFM guider is disabled. Set use_vfm=True in the model yaml first.")
        reg_teacher_channels = [256, 512, 1024]
        for i, guider in enumerate(self.vfm_guiders):
            guider.set_teachers(
                cls_teacher=cls_teacher,
                reg_teacher=reg_teacher,
                cls_teacher_channels=768,
                reg_teacher_channels=reg_teacher_channels[i],
                freeze_teachers=freeze_teachers,
            )

    def set_vfm_teacher_inputs(self, images=None):
        """Set image batch used by VFM teachers during the next forward."""
        self.vfm_teacher_inputs = images

    def set_vfm_targets(self, targets=None):
        """Set YOLO-format GT targets used to build foreground masks for VFM guidance."""
        self.vfm_targets = targets

    @staticmethod
    def _loss_or_zero(loss_value, ref):
        """Return a scalar loss or a differentiable zero matching ref."""
        return loss_value if loss_value is not None else ref.new_zeros(())

    @staticmethod
    def _build_vfm_fg_mask(targets, batch_size, hw, device, dtype):
        """Map normalized YOLO targets [image, cls, x, y, w, h] to a Bx1xHxW foreground mask."""
        h, w = hw
        mask = torch.zeros(batch_size, 1, h, w, device=device, dtype=dtype)
        if targets is None or targets.numel() == 0:
            return mask
        targets = targets.to(device=device)
        for target in targets:
            b = int(target[0].item())
            if b < 0 or b >= batch_size:
                continue
            x, y, bw, bh = target[2:6]
            x1 = torch.clamp(((x - bw / 2) * w).floor(), 0, w - 1).long()
            y1 = torch.clamp(((y - bh / 2) * h).floor(), 0, h - 1).long()
            x2 = torch.clamp(((x + bw / 2) * w).ceil(), 1, w).long()
            y2 = torch.clamp(((y + bh / 2) * h).ceil(), 1, h).long()
            if x2 > x1 and y2 > y1:
                mask[b, 0, y1:y2, x1:x2] = 1
        return mask

    def set_vfm_weight_paths(self, cls_weight_path=None, reg_weight_path=None):
        """Store VFM teacher checkpoint paths on all guider scales."""
        if not self.use_vfm:
            raise RuntimeError("VFM guider is disabled. Set use_vfm=True in the model yaml first.")
        for guider in self.vfm_guiders:
            guider.set_weight_paths(cls_weight_path=cls_weight_path, reg_weight_path=reg_weight_path)

    def _encode_vfm_teachers(self):
        """Run shared teachers once per batch and reuse their features across scales."""
        if not (self.training and self.use_vfm and self.vfm_teacher_inputs is not None and self.vfm_guiders):
            return None
        guider = self.vfm_guiders[0]
        images = self.vfm_teacher_inputs
        features = {}
        if guider.cls_teacher is not None:
            with torch.no_grad():
                features["cls_feature"] = guider.cls_teacher(images)
        if guider.reg_teacher is not None:
            with torch.no_grad():
                features["reg_features"] = guider.reg_teacher(images)
        return features

    def _vfm_features_for_scale(self, teacher_features, scale_index):
        """Select teacher features for the current detection scale."""
        if teacher_features is None:
            return None
        out = {}
        if "cls_feature" in teacher_features:
            out["cls_feature"] = teacher_features["cls_feature"]
        reg_features = teacher_features.get("reg_features")
        if isinstance(reg_features, dict):
            key = f"p{min(scale_index + 1, 3)}"
            out["reg_feature"] = reg_features.get(key, next(reversed(reg_features.values())))
        elif isinstance(reg_features, (list, tuple)):
            # For 224 teacher input, Swin stages are 56,28,14,7. YOLO P3/P4/P5
            # align naturally with stages 1/2/3, then interpolate as needed.
            idx = min(scale_index + 1, len(reg_features) - 1)
            out["reg_feature"] = reg_features[idx]
        elif reg_features is not None:
            out["reg_feature"] = reg_features
        return out or None

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
