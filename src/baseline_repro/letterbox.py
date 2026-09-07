from __future__ import annotations

from typing import Iterable


def _py3round(number: float) -> int:
    """Match Albumentations 1.3.1's Python-version-independent rounding."""
    if abs(round(number) - number) == 0.5:
        return int(2.0 * round(number / 2.0))
    return int(round(number))


def longest_max_size_geometry(
    width: float, height: float, model_size: int = 640
) -> dict[str, float]:
    width = float(width)
    height = float(height)
    if width <= 0 or height <= 0 or model_size <= 0:
        raise ValueError(f"Invalid letterbox geometry: {(width, height, model_size)}")
    scale = float(model_size) / max(width, height)
    resized_width = _py3round(width * scale)
    resized_height = _py3round(height * scale)
    pad_left = int((model_size - resized_width) / 2.0)
    pad_top = int((model_size - resized_height) / 2.0)
    return {
        "orig_width": width,
        "orig_height": height,
        "resized_width": float(resized_width),
        "resized_height": float(resized_height),
        "scale_x": float(resized_width) / width,
        "scale_y": float(resized_height) / height,
        "pad_left": float(pad_left),
        "pad_top": float(pad_top),
        "model_size": float(model_size),
    }


def canonical_xyxy_to_letterboxed_normalized_xyxy(
    box: Iterable[float], width: float, height: float, model_size: int = 640
) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    geometry = longest_max_size_geometry(width, height, model_size)
    size = geometry["model_size"]
    return [
        (x1 * geometry["scale_x"] + geometry["pad_left"]) / size,
        (y1 * geometry["scale_y"] + geometry["pad_top"]) / size,
        (x2 * geometry["scale_x"] + geometry["pad_left"]) / size,
        (y2 * geometry["scale_y"] + geometry["pad_top"]) / size,
    ]


def letterboxed_normalized_cxcywh_to_canonical_xyxy(
    box: Iterable[float], width: float, height: float, model_size: int = 640
) -> list[float]:
    cx, cy, box_width, box_height = [float(value) for value in box]
    geometry = longest_max_size_geometry(width, height, model_size)
    size = geometry["model_size"]
    transformed = [
        (cx - box_width / 2.0) * size,
        (cy - box_height / 2.0) * size,
        (cx + box_width / 2.0) * size,
        (cy + box_height / 2.0) * size,
    ]
    mapped = [
        (transformed[0] - geometry["pad_left"]) / geometry["scale_x"],
        (transformed[1] - geometry["pad_top"]) / geometry["scale_y"],
        (transformed[2] - geometry["pad_left"]) / geometry["scale_x"],
        (transformed[3] - geometry["pad_top"]) / geometry["scale_y"],
    ]
    mapped[0] = min(max(mapped[0], 0.0), float(width))
    mapped[2] = min(max(mapped[2], 0.0), float(width))
    mapped[1] = min(max(mapped[1], 0.0), float(height))
    mapped[3] = min(max(mapped[3], 0.0), float(height))
    return mapped
