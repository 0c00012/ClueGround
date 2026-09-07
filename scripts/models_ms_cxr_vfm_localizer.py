"""Model heads for MS-CXR claim-conditioned bbox localization.

MS-CXR boxes are phrase-grounding bboxes, not pixel-level lesion masks. These
heads operate on frozen VFM features and train only lightweight localization
layers unless the calling script explicitly unfreezes a visual encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(a[:, 0], b[:, 0])
    y1 = torch.maximum(a[:, 1], b[:, 1])
    x2 = torch.minimum(a[:, 2], b[:, 2])
    y2 = torch.minimum(a[:, 3], b[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a + area_b - inter
    return inter / union.clamp(min=1e-6)


def giou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    iou = box_iou_xyxy(a, b)
    cx1 = torch.minimum(a[:, 0], b[:, 0])
    cy1 = torch.minimum(a[:, 1], b[:, 1])
    cx2 = torch.maximum(a[:, 2], b[:, 2])
    cy2 = torch.maximum(a[:, 3], b[:, 3])
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)
    x1 = torch.maximum(a[:, 0], b[:, 0])
    y1 = torch.maximum(a[:, 1], b[:, 1])
    x2 = torch.minimum(a[:, 2], b[:, 2])
    y2 = torch.minimum(a[:, 3], b[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a + area_b - inter
    return iou - (c_area - union) / c_area.clamp(min=1e-6)


def bbox_loss(pred_cxcywh: torch.Tensor, target_cxcywh: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_xy = cxcywh_to_xyxy(pred_cxcywh).clamp(0, 1)
    target_xy = cxcywh_to_xyxy(target_cxcywh).clamp(0, 1)
    l1 = nn.functional.smooth_l1_loss(pred_cxcywh, target_cxcywh)
    giou = giou_xyxy(pred_xy, target_xy).mean()
    return l1 + (1.0 - giou), {"l1": float(l1.detach().cpu()), "giou": float(giou.detach().cpu())}


class BBoxMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(64, hidden // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, hidden // 2), 4),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.cat([out[:, :2], out[:, 2:].clamp(min=0.02, max=1.0)], dim=1)


class PatchHeatmapBBoxHead(nn.Module):
    """Patch-token conditional localization head.

    The model scores each VFM patch token under a finding/context condition,
    uses the score distribution as a soft heatmap, pools patch features, and
    regresses the final normalized bbox.
    """

    def __init__(self, token_dim: int, context_dim: int, hidden: int = 384, dropout: float = 0.1):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, hidden)
        self.context_proj = nn.Linear(context_dim, hidden) if context_dim > 0 else None
        self.score = nn.Linear(hidden, 1)
        self.bbox = nn.Sequential(
            nn.Linear(token_dim + context_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(64, hidden // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, hidden // 2), 4),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.token_proj(tokens)
        if self.context_proj is not None and context.shape[1] > 0:
            h = h + self.context_proj(context).unsqueeze(1)
        h = torch.nn.functional.gelu(h)
        logits = self.score(h).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        x = torch.cat([pooled, context], dim=1) if context.shape[1] > 0 else pooled
        out = self.bbox(x)
        return torch.cat([out[:, :2], out[:, 2:].clamp(min=0.02, max=1.0)], dim=1), logits


class FiLMPatchHeatmapBBoxHead(nn.Module):
    """Phrase-conditioned patch head with multiplicative feature modulation.

    ``PatchHeatmapBBoxHead`` adds one projected context vector to every patch.
    This variant keeps the same frozen token input and bbox loss, but lets the
    query amplify or suppress individual token channels before heatmap scoring.
    It is deliberately a lightweight head rather than a new visual backbone.
    """

    def __init__(self, token_dim: int, context_dim: int, hidden: int = 384, dropout: float = 0.1):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, hidden)
        self.context_scale = nn.Linear(context_dim, hidden) if context_dim > 0 else None
        self.context_shift = nn.Linear(context_dim, hidden) if context_dim > 0 else None
        self.score = nn.Linear(hidden, 1)
        self.bbox = nn.Sequential(
            nn.Linear(token_dim + context_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(64, hidden // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(64, hidden // 2), 4),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.token_proj(tokens)
        if self.context_scale is not None and context.shape[1] > 0:
            scale = torch.sigmoid(self.context_scale(context)).unsqueeze(1)
            shift = self.context_shift(context).unsqueeze(1)
            h = h * (0.5 + scale) + shift
        h = torch.nn.functional.gelu(h)
        logits = self.score(h).squeeze(-1)
        weights = torch.softmax(logits, dim=1)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        x = torch.cat([pooled, context], dim=1) if context.shape[1] > 0 else pooled
        out = self.bbox(x)
        return torch.cat([out[:, :2], out[:, 2:].clamp(min=0.02, max=1.0)], dim=1), logits


class CrossAttentionPatchHeatmapBBoxHead(PatchHeatmapBBoxHead):
    """Patch heatmap head with direct query-to-patch attention.

    The base head's additive context pathway is retained so a checkpoint from
    :class:`PatchHeatmapBBoxHead` can warm-start it exactly.  A gated
    multi-head query/key term is then learned on top.  The gate starts near
    zero, making this a controlled change to phrase-to-patch interaction
    rather than a replacement visual model.
    """

    def __init__(
        self,
        token_dim: int,
        context_dim: int,
        hidden: int = 384,
        dropout: float = 0.1,
        heads: int = 4,
    ):
        super().__init__(token_dim, context_dim, hidden=hidden, dropout=dropout)
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.query_attn = nn.Linear(context_dim, hidden, bias=False)
        self.key_attn = nn.Linear(hidden, hidden, bias=False)
        # sigmoid(-3) keeps the initial behavior close to the warm-started head.
        self.attn_gate = nn.Parameter(torch.tensor(-3.0))

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_hidden = self.token_proj(tokens)
        if self.context_proj is not None and context.shape[1] > 0:
            additive_context = self.context_proj(context).unsqueeze(1)
        else:
            additive_context = torch.zeros_like(token_hidden)
        base_hidden = torch.nn.functional.gelu(token_hidden + additive_context)
        base_logits = self.score(base_hidden).squeeze(-1)

        if context.shape[1] > 0:
            query = self.query_attn(context).view(-1, self.heads, self.head_dim)
            keys = self.key_attn(token_hidden).view(-1, token_hidden.shape[1], self.heads, self.head_dim)
            attention_logits = (keys * query.unsqueeze(1)).sum(dim=-1).mean(dim=-1)
            attention_logits = attention_logits / float(self.head_dim) ** 0.5
        else:
            attention_logits = torch.zeros_like(base_logits)
        logits = base_logits + torch.sigmoid(self.attn_gate) * attention_logits
        weights = torch.softmax(logits, dim=1)
        pooled = (tokens * weights.unsqueeze(-1)).sum(dim=1)
        x = torch.cat([pooled, context], dim=1) if context.shape[1] > 0 else pooled
        out = self.bbox(x)
        return torch.cat([out[:, :2], out[:, 2:].clamp(min=0.02, max=1.0)], dim=1), logits


class DensePatchProposalHead(nn.Module):
    """Query-conditioned dense patch proposals over frozen VFM tokens.

    Unlike :class:`PatchHeatmapBBoxHead`, each patch predicts its own box and
    objectness.  This permits several spatially separated phrase regions to be
    represented before the shared downstream cardinality decoder is applied.
    """

    def __init__(self, token_dim: int, context_dim: int, hidden: int = 384, dropout: float = 0.1):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, hidden)
        self.context_proj = nn.Linear(context_dim, hidden) if context_dim > 0 else None
        self.shared = nn.Sequential(nn.GELU(), nn.Dropout(dropout))
        self.objectness = nn.Linear(hidden, 1)
        self.box = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.token_proj(tokens)
        if self.context_proj is not None and context.shape[1] > 0:
            hidden = hidden + self.context_proj(context).unsqueeze(1)
        hidden = self.shared(hidden)
        boxes = self.box(hidden)
        boxes = torch.cat([boxes[..., :2], boxes[..., 2:].clamp(min=0.02, max=1.0)], dim=-1)
        return boxes, self.objectness(hidden).squeeze(-1)


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "FeatureScaler":
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return cls(mean.astype("float32"), std.astype("float32"))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype("float32")

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.mean, "std": self.std}
