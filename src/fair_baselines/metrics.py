from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from .mscxr_data import GroundingSample


def box_area(box: list[float] | tuple[float, ...]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def iou_xyxy(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    denom = box_area(a) + box_area(b) - inter
    return 0.0 if denom <= 0 else inter / denom


def hull_box(boxes: list[list[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def exact_rectangle_union_iou(pred: list[list[float]], gt: list[list[float]]) -> float:
    if not pred or not gt:
        return 0.0
    xs = sorted({float(box[index]) for box in pred + gt for index in (0, 2)})
    ys = sorted({float(box[index]) for box in pred + gt for index in (1, 3)})
    inter = 0.0
    union = 0.0
    for x1, x2 in zip(xs[:-1], xs[1:]):
        for y1, y2 in zip(ys[:-1], ys[1:]):
            if x2 <= x1 or y2 <= y1:
                continue
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            in_pred = any(box[0] <= mx < box[2] and box[1] <= my < box[3] for box in pred)
            in_gt = any(box[0] <= mx < box[2] and box[1] <= my < box[3] for box in gt)
            area = (x2 - x1) * (y2 - y1)
            if in_pred or in_gt:
                union += area
            if in_pred and in_gt:
                inter += area
    return 0.0 if union <= 0 else inter / union


def raster_union_iou(
    pred: list[list[float]], gt: list[list[float]], width: int, height: int, size: int = 224
) -> float:
    pred_mask = np.zeros((size, size), dtype=np.bool_)
    gt_mask = np.zeros((size, size), dtype=np.bool_)

    def fill(mask: np.ndarray, boxes: list[list[float]]) -> None:
        for box in boxes:
            x1 = int(np.floor(float(box[0]) / width * size))
            y1 = int(np.floor(float(box[1]) / height * size))
            x2 = int(np.ceil(float(box[2]) / width * size))
            y2 = int(np.ceil(float(box[3]) / height * size))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(size, x2), min(size, y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

    fill(pred_mask, pred)
    fill(gt_mask, gt)
    union = np.logical_or(pred_mask, gt_mask).sum()
    return 0.0 if union == 0 else float(np.logical_and(pred_mask, gt_mask).sum() / union)


def evaluate_set(
    gt_boxes: list[list[float]], pred_boxes: list[list[float]], width: int, height: int
) -> dict[str, float]:
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    gt_best = [max((iou_xyxy(gt, pred) for pred in pred_boxes), default=0.0) for gt in gt_boxes]

    def optimal_tp(threshold: float) -> int:
        if not gt_boxes or not pred_boxes:
            return 0
        ious = np.asarray(
            [[iou_xyxy(gt, pred) for pred in pred_boxes] for gt in gt_boxes],
            dtype=np.float64,
        )
        # A binary assignment cost maximizes the number of threshold-valid
        # one-to-one matches, independent of the order of boxes.
        rows, columns = linear_sum_assignment((ious < threshold).astype(np.int8))
        return int(np.sum(ious[rows, columns] >= threshold))

    result: dict[str, float] = {
        "coverage_mean_iou": float(np.mean(gt_best)) if gt_best else 0.0,
        "hull_union_iou": iou_xyxy(hull_box(gt_boxes), hull_box(pred_boxes)) if gt_boxes and pred_boxes else 0.0,
        "exact_union_iou": exact_rectangle_union_iou(pred_boxes, gt_boxes),
        "raster_union_iou": raster_union_iou(pred_boxes, gt_boxes, width, height),
        "gt_hit_rate_0_3": float(np.mean([value >= 0.3 for value in gt_best])) if gt_best else 0.0,
        "gt_hit_rate_0_5": float(np.mean([value >= 0.5 for value in gt_best])) if gt_best else 0.0,
        "n_gt": float(n_gt),
        "n_pred": float(n_pred),
        "pred_count_abs_error": float(abs(n_pred - n_gt)),
    }
    for name, threshold in (("0_3", 0.3), ("0_5", 0.5)):
        tp = optimal_tp(threshold)
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gt if n_gt else 0.0
        result[f"set_precision_{name}"] = precision
        result[f"set_recall_{name}"] = recall
        result[f"set_f1_{name}"] = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return result


def _inverse_letterbox_box(box: list[float], geometry: dict[str, Any]) -> list[float] | None:
    scale = float(geometry["scale"])
    pad_x, pad_y = float(geometry["pad_x"]), float(geometry["pad_y"])
    width, height = float(geometry["orig_w"]), float(geometry["orig_h"])
    mapped = [
        (box[0] - pad_x) / scale,
        (box[1] - pad_y) / scale,
        (box[2] - pad_x) / scale,
        (box[3] - pad_y) / scale,
    ]
    mapped[0], mapped[2] = np.clip(mapped[0], 0, width), np.clip(mapped[2], 0, width)
    mapped[1], mapped[3] = np.clip(mapped[1], 0, height), np.clip(mapped[3], 0, height)
    return [float(value) for value in mapped] if mapped[2] > mapped[0] and mapped[3] > mapped[1] else None


def decode_probability_mask(
    probability: np.ndarray,
    geometry: dict[str, Any],
    threshold: float,
    min_area_ratio: float,
    max_components: int,
) -> tuple[list[list[float]], list[float]]:
    probability = np.asarray(probability, dtype=np.float32).squeeze().copy()
    valid = np.zeros_like(probability, dtype=np.bool_)
    pad_x, pad_y = int(geometry["pad_x"]), int(geometry["pad_y"])
    resized_w, resized_h = int(geometry["resized_w"]), int(geometry["resized_h"])
    valid[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = True
    probability[~valid] = -1.0
    binary = probability >= threshold
    if not binary.any():
        valid_values = probability[valid]
        adaptive = float(np.quantile(valid_values, 0.98)) if valid_values.size else threshold
        binary = np.logical_and(valid, probability >= adaptive)
    labels, count = ndimage.label(binary)
    valid_area = max(1, resized_w * resized_h)
    components: list[tuple[float, int, list[float]]] = []
    for label_index in range(1, count + 1):
        ys, xs = np.where(labels == label_index)
        if len(xs) == 0:
            continue
        area_ratio = len(xs) / valid_area
        score = float(probability[ys, xs].mean())
        box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        components.append((score, len(xs), box))
    if not components:
        y, x = np.unravel_index(int(np.argmax(probability)), probability.shape)
        components = [(float(probability[y, x]), 1, [float(x), float(y), float(x + 1), float(y + 1)])]
    filtered = [item for item in components if item[1] / valid_area >= min_area_ratio]
    if not filtered:
        filtered = [max(components, key=lambda item: item[1])]
    filtered.sort(key=lambda item: (item[0], item[1]), reverse=True)
    boxes: list[list[float]] = []
    scores: list[float] = []
    for score, _, box in filtered[:max_components]:
        mapped = _inverse_letterbox_box(box, geometry)
        if mapped is not None:
            boxes.append(mapped)
            scores.append(score)
    return boxes, scores


def summarize_records(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, float]]:
    set_rows: list[dict[str, float]] = []
    singleton_iou: list[float] = []
    singleton_set_rows: list[dict[str, float]] = []
    for record in records:
        sample: GroundingSample = record["sample"]
        pred_boxes = [list(map(float, box)) for box in record["pred_boxes"]]
        gt_boxes = [list(map(float, box)) for box in sample.gt_boxes]
        set_rows.append(evaluate_set(gt_boxes, pred_boxes, sample.image_width, sample.image_height))
        if sample.gold_count == 1:
            top_box = pred_boxes[0] if pred_boxes else []
            singleton_iou.append(iou_xyxy(gt_boxes[0], top_box) if top_box else 0.0)
            singleton_set_rows.append(set_rows[-1])
    multi_keys = [
        "coverage_mean_iou",
        "hull_union_iou",
        "exact_union_iou",
        "raster_union_iou",
        "set_f1_0_3",
        "set_f1_0_5",
        "n_pred",
        "pred_count_abs_error",
    ]
    multi = {key: float(np.mean([row[key] for row in set_rows])) for key in multi_keys}
    multi.update({"n_groups": float(len(records)), "n_gt_boxes": float(sum(row["n_gt"] for row in set_rows))})
    single = {
        "n": float(len(singleton_iou)),
        "mean_iou": float(np.mean(singleton_iou)) if singleton_iou else 0.0,
        "median_iou": float(np.median(singleton_iou)) if singleton_iou else 0.0,
        "hit_0_3": float(np.mean([value >= 0.3 for value in singleton_iou])) if singleton_iou else 0.0,
        "hit_0_5": float(np.mean([value >= 0.5 for value in singleton_iou])) if singleton_iou else 0.0,
        "mean_pred_count": (
            float(np.mean([row["n_pred"] for row in singleton_set_rows]))
            if singleton_set_rows
            else 0.0
        ),
        "strict_set_f1_0_3": (
            float(np.mean([row["set_f1_0_3"] for row in singleton_set_rows]))
            if singleton_set_rows
            else 0.0
        ),
        "strict_set_f1_0_5": (
            float(np.mean([row["set_f1_0_5"] for row in singleton_set_rows]))
            if singleton_set_rows
            else 0.0
        ),
        "metric_note": "mean_iou is the conventional top-1 projection; strict_set_f1 penalizes extra boxes",
    }
    return single, multi


def tune_mask_decoder(
    probabilities: list[np.ndarray],
    samples: list[GroundingSample],
    geometries: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grid_rows: list[dict[str, Any]] = []
    for threshold, min_area, max_components in itertools.product(
        (0.25, 0.35, 0.45, 0.55, 0.65), (0.0005, 0.002, 0.005), (1, 2, 3)
    ):
        records = []
        for probability, sample, geometry in zip(probabilities, samples, geometries):
            boxes, scores = decode_probability_mask(probability, geometry, threshold, min_area, max_components)
            records.append({"sample": sample, "pred_boxes": boxes, "scores": scores})
        _, metrics = summarize_records(records)
        selection_score = float(
            0.25 * metrics["coverage_mean_iou"]
            + 0.25 * metrics["exact_union_iou"]
            + 0.25 * metrics["set_f1_0_3"]
            + 0.25 * metrics["set_f1_0_5"]
        )
        grid_rows.append(
            {
                "threshold": threshold,
                "min_area_ratio": min_area,
                "max_components": max_components,
                "selection_score": selection_score,
                **metrics,
            }
        )
    best = max(grid_rows, key=lambda row: (row["selection_score"], -row["max_components"]))
    return {
        "threshold": float(best["threshold"]),
        "min_area_ratio": float(best["min_area_ratio"]),
        "max_components": int(best["max_components"]),
        "selection_score": float(best["selection_score"]),
    }, grid_rows
