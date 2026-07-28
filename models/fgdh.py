# Add or modify code through Du Shenyu
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Fine-Grained Enhanced Classification Head for FGDC-Net."""

import torch
import torch.nn.functional as F
from torch import nn

from models.common import Conv, DWConv


class FGDH(nn.Module):
    """Fine-Grained Enhanced Classification Head.

    Args:
        in_channels: Number of channels in the input classification feature map.
        num_classes: Number of output classes.
        hidden_channels: Internal stream width. Defaults to in_channels.
        compact_dim: Compact bilinear feature dimension before the classifier.
        dropout: Dropout probability before the final classifier.
        Input: F_in with shape (B, C, H, W)
        Output: Class probability vector with shape (B, num_classes) by default. If dense=True, dense
            logits/probabilities with shape (B, num_classes, H, W).
    """

    def __init__(
        self,
        in_channels,
        num_classes,
        hidden_channels=None,
        compact_dim=256,
        dropout=0.0,
        dense=False,
        return_probs=True,
    ):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        compact_dim = min(compact_dim, hidden_channels)
        self.dense = dense
        self.return_probs = return_probs

        # Stream A keeps the receptive field local and lightweight, which helps preserve
        # fine defect boundaries, edges, and texture changes.
        self.stream_a = nn.Sequential(
            Conv(in_channels, hidden_channels, 1, 1),
            DWConv(hidden_channels, hidden_channels, 3, 1),
            Conv(hidden_channels, hidden_channels, 1, 1),
        )

        # Stream B uses dilated convolution to collect wider context without reducing
        # spatial resolution, complementing the local texture stream.
        self.stream_b = nn.Sequential(
            Conv(in_channels, hidden_channels, 1, 1),
            Conv(hidden_channels, hidden_channels, 3, 1, d=2),
            Conv(hidden_channels, hidden_channels, 1, 1),
        )

        # Project both streams to a compact shared dimension before bilinear fusion.
        # This approximates bilinear pooling with much lower memory than an outer product.
        self.proj_a = nn.Conv2d(hidden_channels, compact_dim, 1, bias=False)
        self.proj_b = nn.Conv2d(hidden_channels, compact_dim, 1, bias=False)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(compact_dim, num_classes, 1) if dense else nn.Linear(compact_dim, num_classes)

    def forward(self, x):
        """Classify a batch of classification feature maps.

        Args:
            x: Classification feature map with shape (B, C, H, W).

        Returns:
            Softmax class probabilities with shape (B, num_classes), or dense: logits/probabilities with shape (B,
                num_classes, H, W) when dense=True.
        """
        # Extract local fine-grained cues such as boundaries and defect texture.
        feat_a = self.stream_a(x)

        # Extract broader contextual semantics with a larger effective receptive field.
        feat_b = self.stream_b(x)

        # Compact both streams before fusion to avoid the C x C memory cost of full
        # bilinear pooling.
        feat_a = self.proj_a(feat_a)
        feat_b = self.proj_b(feat_b)

        # Low-rank bilinear feature fusion: element-wise multiplication captures
        # pairwise interactions between the two streams at each spatial location.
        fused = feat_a * feat_b

        # Signed square-root and L2 normalization are common bilinear-pooling
        # stabilizers and keep feature magnitudes well behaved.
        fused = torch.sign(fused) * torch.sqrt(torch.abs(fused) + 1e-6)
        fused = F.normalize(fused, p=2, dim=1)

        # In dense mode, keep the spatial grid so a detector can predict one class
        # vector per location/anchor. In global mode, pool to one descriptor per image.
        if self.dense:
            logits = self.classifier(fused)
            return torch.softmax(logits, dim=1) if self.return_probs else logits

        fused = self.pool(fused).flatten(1)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        return F.softmax(logits, dim=1) if self.return_probs else logits
