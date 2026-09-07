from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


Box = Sequence[float]


def clip_box(box: Box, width: float | None = None, height: float | None = None) -> list[float]:
    x1, y1, x2, y2 = [float(x) for x in box]
    if width is not None:
        x1, x2 = max(0.0, x1), min(float(width), x2)
    if height is not None:
        y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def area_xyxy(box: Box) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def iou_xyxy(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = area_xyxy(a) + area_xyxy(b) - inter
    return 0.0 if denom <= 0 else float(inter / denom)


def overlap_ratios(candidate: Box, reference: Box | None) -> tuple[float, float]:
    if reference is None:
        return 0.0, 0.0
    x1 = max(float(candidate[0]), float(reference[0]))
    y1 = max(float(candidate[1]), float(reference[1]))
    x2 = min(float(candidate[2]), float(reference[2]))
    y2 = min(float(candidate[3]), float(reference[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    cand_area = max(area_xyxy(candidate), 1e-8)
    ref_area = max(area_xyxy(reference), 1e-8)
    return float(inter / cand_area), float(inter / ref_area)


def center_distance(a: Box, b: Box | None, width: float | None = None, height: float | None = None) -> float:
    if b is None:
        return 1.0
    acx = (float(a[0]) + float(a[2])) / 2.0
    acy = (float(a[1]) + float(a[3])) / 2.0
    bcx = (float(b[0]) + float(b[2])) / 2.0
    bcy = (float(b[1]) + float(b[3])) / 2.0
    dist = math.hypot(acx - bcx, acy - bcy)
    if width and height:
        return float(dist / max(math.hypot(float(width), float(height)), 1e-8))
    return float(dist)


def weighted_box(boxes: Iterable[Box], weights: Iterable[float]) -> list[float]:
    arr = np.asarray(list(boxes), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if len(arr) == 0:
        return [0.0, 0.0, 1.0, 1.0]
    if float(w.sum()) <= 1e-8:
        w = np.ones(len(arr), dtype=float)
    w = w / w.sum()
    return (arr * w[:, None]).sum(axis=0).tolist()


def dedupe_boxes(rows: list[dict], iou_threshold: float = 0.985, score_key: str = "score") -> list[dict]:
    out: list[dict] = []
    for row in sorted(rows, key=lambda x: float(x.get(score_key, 0.0)), reverse=True):
        if all(iou_xyxy(row["box"], old["box"]) < iou_threshold for old in out):
            out.append(row)
    return out


def add_box_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    x1, y1, x2, y2 = [out[c].astype(float) for c in ["pred_x1", "pred_y1", "pred_x2", "pred_y2"]]
    width = out.get("image_width", pd.Series(np.nan, index=out.index)).astype(float)
    height = out.get("image_height", pd.Series(np.nan, index=out.index)).astype(float)
    out["box_w_px"] = (x2 - x1).clip(lower=0.0)
    out["box_h_px"] = (y2 - y1).clip(lower=0.0)
    out["box_area_px"] = out["box_w_px"] * out["box_h_px"]
    out["box_aspect"] = out["box_w_px"] / out["box_h_px"].replace(0, np.nan)
    out["box_aspect"] = out["box_aspect"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["box_cx"] = (x1 + x2) / 2.0
    out["box_cy"] = (y1 + y2) / 2.0
    out["box_cx_norm"] = out["box_cx"] / width.replace(0, np.nan)
    out["box_cy_norm"] = out["box_cy"] / height.replace(0, np.nan)
    out["box_w_norm"] = out["box_w_px"] / width.replace(0, np.nan)
    out["box_h_norm"] = out["box_h_px"] / height.replace(0, np.nan)
    out["box_area_norm"] = out["box_w_norm"] * out["box_h_norm"]
    fill_cols = ["box_cx_norm", "box_cy_norm", "box_w_norm", "box_h_norm", "box_area_norm"]
    out[fill_cols] = out[fill_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out

