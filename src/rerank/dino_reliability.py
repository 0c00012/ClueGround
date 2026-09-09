from __future__ import annotations

import numpy as np
import pandas as pd


def _onehot_value(row: pd.Series, prefix: str) -> str:
    cols = [c for c in row.index if c.startswith(prefix)]
    if not cols:
        return "unknown"
    best = max(cols, key=lambda c: float(row.get(c, 0.0) or 0.0))
    if float(row.get(best, 0.0) or 0.0) <= 0:
        return "unknown"
    return best[len(prefix) :]


def _position_match(value: str, cx: float, cy: float) -> bool | None:
    if value in {"unknown", "none", "whole", ""}:
        return None
    if value == "left":
        return cx >= 0.45
    if value == "right":
        return cx <= 0.55
    if value == "bilateral":
        return True
    if value in {"apical", "upper"}:
        return cy <= 0.55
    if value in {"lower", "basal"}:
        return cy >= 0.45
    if value == "mid":
        return 0.25 <= cy <= 0.75
    return None


def add_dino_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """Add a conservative reliability gate for RAD-DINO agreement features.

    The gate intentionally avoids using gold IoU.  It treats DINO overlap as a
    boost only when DINO does not visibly conflict with rule-context position
    and has at least weak consensus with the detector pool.
    """

    out = df.copy()
    if "xattn_iou" not in out.columns:
        out["xattn_iou"] = 0.0
    if "consensus_max" not in out.columns:
        out["consensus_max"] = 0.0
    if "box_cx_norm" not in out.columns:
        out["box_cx_norm"] = 0.5
    if "box_cy_norm" not in out.columns:
        out["box_cy_norm"] = 0.5

    reliabilities: list[float] = []
    conflicts: list[int] = []
    for _, row in out.iterrows():
        rel = 1.0
        conflict = 0
        cx = float(row.get("box_cx_norm", 0.5) or 0.5)
        cy = float(row.get("box_cy_norm", 0.5) or 0.5)
        lat = _onehot_value(row, "lat_") if any(str(c).startswith("lat_") for c in out.columns) else str(row.get("laterality", "unknown"))
        vert = _onehot_value(row, "v_") if any(str(c).startswith("v_") for c in out.columns) else str(row.get("vertical", "unknown"))
        lat_match = _position_match(lat, cx, cy)
        vert_match = _position_match(vert, cx, cy)
        if lat_match is False:
            rel *= 0.25
            conflict = 1
        if vert_match is False:
            rel *= 0.35
            conflict = 1
        if float(row.get("consensus_max", 0.0) or 0.0) < 0.05:
            rel *= 0.5
        area = float(row.get("box_area_norm", 0.0) or 0.0)
        if area <= 0.005 or area >= 0.75:
            rel *= 0.7
        reliabilities.append(float(np.clip(rel, 0.0, 1.0)))
        conflicts.append(conflict)
    out["dino_reliability"] = reliabilities
    out["dino_rule_conflict"] = conflicts
    out["dino_reliable_agreement"] = out["dino_reliability"].astype(float) * out["xattn_iou"].astype(float)
    return out

