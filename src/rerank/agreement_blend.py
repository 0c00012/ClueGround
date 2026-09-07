from __future__ import annotations

from typing import Sequence

from .box_features import center_distance, iou_xyxy, weighted_box


def choose_or_blend(
    box_a: Sequence[float],
    score_a: float,
    box_b: Sequence[float],
    score_b: float,
    *,
    alpha: float = 0.15,
    iou_threshold: float = 0.1,
    center_threshold: float = 0.15,
    image_width: float | None = None,
    image_height: float | None = None,
) -> tuple[list[float], str]:
    iou = iou_xyxy(box_a, box_b)
    cdist = center_distance(box_a, box_b, image_width, image_height)
    if iou >= iou_threshold or cdist <= center_threshold:
        return weighted_box([box_a, box_b], [1.0 - alpha, alpha]), "blend"
    if score_b > score_a:
        return [float(x) for x in box_b], "switch"
    return [float(x) for x in box_a], "keep"

