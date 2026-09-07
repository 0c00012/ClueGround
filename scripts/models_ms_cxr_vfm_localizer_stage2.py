"""Stage 2 model heads for MS-CXR context-conditioned VFM localization."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from models_ms_cxr_vfm_localizer import PatchHeatmapBBoxHead


class AdapterPatchHeatmapBBoxHead(nn.Module):
    """Trainable residual adapter on frozen VFM patch tokens plus heatmap head."""

    def __init__(self, token_dim: int, context_dim: int, hidden: int = 384, bottleneck: int = 128, dropout: float = 0.1):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, token_dim),
        )
        self.head = PatchHeatmapBBoxHead(token_dim=token_dim, context_dim=context_dim, hidden=hidden, dropout=dropout)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        adapted = tokens + self.adapter(tokens)
        return self.head(adapted, context)
