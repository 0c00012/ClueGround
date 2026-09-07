from __future__ import annotations

import json
import hashlib
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


def patient_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    identity_columns: list[str],
    metric_columns: list[str],
    cluster_column: str = "subject_id",
    reps: int = 2000,
    seed: int = 20260712,
) -> pd.DataFrame:
    """Compute deterministic patient-cluster bootstrap confidence intervals."""
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    for identity, part in frame.groupby(identity_columns, sort=True, dropna=False):
        identity_values = identity if isinstance(identity, tuple) else (identity,)
        clusters = sorted(part[cluster_column].astype(str).unique())
        if not clusters:
            continue
        token = "|".join(str(value) for value in identity_values)
        offset = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng((seed + offset) % (2**32))
        choices = rng.integers(0, len(clusters), size=(reps, len(clusters)))
        for metric in metric_columns:
            if metric not in part:
                continue
            sums = np.asarray(
                [
                    part.loc[part[cluster_column].astype(str).eq(cluster), metric]
                    .astype(float)
                    .sum()
                    for cluster in clusters
                ],
                dtype=np.float64,
            )
            counts = np.asarray(
                [int(part[cluster_column].astype(str).eq(cluster).sum()) for cluster in clusters],
                dtype=np.float64,
            )
            samples = sums[choices].sum(axis=1) / counts[choices].sum(axis=1)
            row = {
                column: value for column, value in zip(identity_columns, identity_values)
            }
            row.update(
                {
                    "metric": metric,
                    "point_estimate": float(part[metric].astype(float).mean()),
                    "ci_95_low": float(np.quantile(samples, 0.025)),
                    "ci_95_high": float(np.quantile(samples, 0.975)),
                    "n_clusters": len(clusters),
                    "n_rows": int(len(part)),
                    "bootstrap_reps": reps,
                    "bootstrap_unit": cluster_column,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def iou_matrix(pred: list[list[float]], gold: list[list[float]]) -> np.ndarray:
    a = np.asarray(pred, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(gold, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    width_height = np.clip(bottom_right - top_left, 0.0, None)
    intersection = width_height[..., 0] * width_height[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def rectangle_union_area(boxes: list[list[float]]) -> float:
    valid = [box for box in boxes if len(box) == 4 and box[2] > box[0] and box[3] > box[1]]
    if not valid:
        return 0.0
    xs = sorted({float(box[0]) for box in valid} | {float(box[2]) for box in valid})
    area = 0.0
    for left, right in zip(xs[:-1], xs[1:]):
        if right <= left:
            continue
        intervals = sorted(
            (float(box[1]), float(box[3]))
            for box in valid
            if float(box[0]) < right and float(box[2]) > left
        )
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for low, high in intervals[1:]:
                if low <= end:
                    end = max(end, high)
                else:
                    covered += end - start
                    start, end = low, high
            covered += end - start
        area += (right - left) * covered
    return area


def exact_union_iou(pred: list[list[float]], gold: list[list[float]]) -> float:
    pred_area = rectangle_union_area(pred)
    gold_area = rectangle_union_area(gold)
    intersections = []
    for a in pred:
        for b in gold:
            box = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
            if box[2] > box[0] and box[3] > box[1]:
                intersections.append(box)
    intersection = rectangle_union_area(intersections)
    union = pred_area + gold_area - intersection
    return intersection / union if union > 0 else 1.0


def hull_iou(pred: list[list[float]], gold: list[list[float]]) -> float:
    if not pred or not gold:
        return 0.0

    def hull(boxes: list[list[float]]) -> list[float]:
        return [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ]

    return float(iou_matrix([hull(pred)], [hull(gold)])[0, 0])


def greedy_set_f1(pred: list[list[float]], gold: list[list[float]], threshold: float) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    matrix = iou_matrix(pred, gold)
    pairs = sorted(
        (
            (float(matrix[i, j]), i, j)
            for i in range(len(pred))
            for j in range(len(gold))
            if matrix[i, j] >= threshold
        ),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_gold: set[int] = set()
    for _, pred_index, gold_index in pairs:
        if pred_index not in used_pred and gold_index not in used_gold:
            used_pred.add(pred_index)
            used_gold.add(gold_index)
    true_positive = len(used_pred)
    precision = true_positive / len(pred)
    recall = true_positive / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def maximum_set_f1(pred: list[list[float]], gold: list[list[float]], threshold: float) -> float:
    """Set F1 using maximum-cardinality bipartite matching at an IoU threshold."""
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    matrix = iou_matrix(pred, gold)
    adjacency = [
        [gold_index for gold_index in range(len(gold)) if matrix[pred_index, gold_index] >= threshold]
        for pred_index in range(len(pred))
    ]
    gold_to_pred = [-1] * len(gold)

    def augment(pred_index: int, seen: set[int]) -> bool:
        for gold_index in adjacency[pred_index]:
            if gold_index in seen:
                continue
            seen.add(gold_index)
            previous = gold_to_pred[gold_index]
            if previous < 0 or augment(previous, seen):
                gold_to_pred[gold_index] = pred_index
                return True
        return False

    true_positive = sum(augment(pred_index, set()) for pred_index in range(len(pred)))
    precision = true_positive / len(pred)
    recall = true_positive / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_rows(
    samples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    force_single_box: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_map = {str(sample["query_id"]): sample for sample in samples}
    prediction_map = {str(row["query_id"]): row for row in predictions}
    detail = []
    for query_id, sample in sample_map.items():
        raw = prediction_map.get(query_id, {}).get("pred_boxes", [])
        if isinstance(raw, str):
            raw = json.loads(raw) if raw else []
        pred = [[float(value) for value in box] for box in raw]
        if force_single_box:
            pred = pred[:1]
        gold = [[float(value) for value in box] for box in sample["gold_boxes"]]
        matrix = iou_matrix(pred, gold)
        top1 = float(matrix[0, 0]) if matrix.size and force_single_box else 0.0
        coverage = float(matrix.max(axis=0).mean()) if len(pred) and len(gold) else 0.0
        detail.append(
            {
                "query_id": query_id,
                "subject_id": sample.get("subject_id", ""),
                "finding": sample.get("finding", ""),
                "n_gt": len(gold),
                "n_pred": len(pred),
                "top1_iou": top1,
                "coverage_iou": coverage,
                "hull_iou": hull_iou(pred, gold),
                "exact_union_iou": exact_union_iou(pred, gold),
                "set_f1_0_3": greedy_set_f1(pred, gold, 0.3),
                "set_f1_0_5": greedy_set_f1(pred, gold, 0.5),
                "set_f1_optimal_0_3": maximum_set_f1(pred, gold, 0.3),
                "set_f1_optimal_0_5": maximum_set_f1(pred, gold, 0.5),
                "missing_prediction": int(query_id not in prediction_map),
            }
        )
    by_metric: dict[str, list[float]] = defaultdict(list)
    for row in detail:
        for key in (
            "top1_iou",
            "coverage_iou",
            "hull_iou",
            "exact_union_iou",
            "set_f1_0_3",
            "set_f1_0_5",
            "set_f1_optimal_0_3",
            "set_f1_optimal_0_5",
        ):
            by_metric[key].append(float(row[key]))
    summary = {
        "n": len(detail),
        "n_missing_predictions": sum(row["missing_prediction"] for row in detail),
        "mean_pred_count": float(np.mean([row["n_pred"] for row in detail])) if detail else 0.0,
        **{key: float(np.mean(values)) if values else 0.0 for key, values in by_metric.items()},
    }
    if force_single_box:
        ious = np.asarray(by_metric["top1_iou"], dtype=np.float64)
        summary.update(
            {
                "mean_iou": float(ious.mean()) if len(ious) else 0.0,
                "Hit@0.3": float((ious >= 0.3).mean()) if len(ious) else 0.0,
                "Hit@0.5": float((ious >= 0.5).mean()) if len(ious) else 0.0,
            }
        )
    return summary, detail
