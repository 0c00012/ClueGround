from __future__ import annotations

from typing import Any

import numpy as np

from src.fair_baselines.metrics import evaluate_set

from .contracts import ProtocolSpec


def evaluate_group(
    gold_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    return evaluate_set(gold_boxes, pred_boxes, image_width, image_height)


def summarize_protocol(
    spec: ProtocolSpec,
    inputs: list[dict[str, Any]],
    labels: dict[str, list[list[float]]],
    predictions: dict[str, list[list[float]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detail: list[dict[str, Any]] = []
    for row in inputs:
        group_id = str(row["group_id"])
        gold = labels[group_id]
        pred = predictions.get(group_id, [])
        scores = evaluate_group(
            gold,
            pred,
            image_width=int(row["image_width"]),
            image_height=int(row["image_height"]),
        )
        detail.append({"protocol_key": spec.key, "group_id": group_id, **scores})

    if not detail:
        return {"protocol_key": spec.key, "n_groups": 0}, detail
    if spec.output_mode == "one_box":
        top_iou = []
        for row in detail:
            # One-box protocols contain exactly one GT rectangle by contract.
            top_iou.append(float(row["coverage_mean_iou"]))
        values = np.asarray(top_iou, dtype=float)
        summary = {
            "protocol_key": spec.key,
            "n_groups": len(detail),
            "mean_iou": float(values.mean()),
            "median_iou": float(np.median(values)),
            "hit_0_3": float((values >= 0.3).mean()),
            "hit_0_5": float((values >= 0.5).mean()),
            "mean_pred_count": float(np.mean([row["n_pred"] for row in detail])),
            "target_semantics": spec.target_semantics,
        }
    else:
        summary = {
            "protocol_key": spec.key,
            "n_groups": len(detail),
            "n_gt_boxes": int(sum(row["n_gt"] for row in detail)),
            "coverage_mean_iou": float(np.mean([row["coverage_mean_iou"] for row in detail])),
            "exact_union_iou": float(np.mean([row["exact_union_iou"] for row in detail])),
            "hull_union_iou_diagnostic": float(np.mean([row["hull_union_iou"] for row in detail])),
            "set_f1_0_3": float(np.mean([row["set_f1_0_3"] for row in detail])),
            "set_f1_0_5": float(np.mean([row["set_f1_0_5"] for row in detail])),
            "mean_pred_count": float(np.mean([row["n_pred"] for row in detail])),
            "target_semantics": spec.target_semantics,
        }
    return summary, detail


def singleton_projection(
    inputs: list[dict[str, Any]],
    labels: dict[str, list[list[float]]],
    predictions: dict[str, list[list[float]]],
) -> dict[str, Any]:
    values = []
    strict_f1_03 = []
    strict_f1_05 = []
    pred_counts = []
    for row in inputs:
        group_id = str(row["group_id"])
        gold = labels[group_id]
        if len(gold) != 1:
            continue
        full_prediction = predictions.get(group_id, [])
        top_one_metrics = evaluate_group(
            gold,
            full_prediction[:1],
            image_width=int(row["image_width"]),
            image_height=int(row["image_height"]),
        )
        strict_metrics = evaluate_group(
            gold,
            full_prediction,
            image_width=int(row["image_width"]),
            image_height=int(row["image_height"]),
        )
        values.append(float(top_one_metrics["coverage_mean_iou"]))
        strict_f1_03.append(float(strict_metrics["set_f1_0_3"]))
        strict_f1_05.append(float(strict_metrics["set_f1_0_5"]))
        pred_counts.append(float(strict_metrics["n_pred"]))
    array = np.asarray(values, dtype=float)
    return {
        "singlebox_888_view": "same variable-cardinality predictions; top-1 mIoU plus full-set penalty audit",
        "singlebox_888_n": len(values),
        "singlebox_888_mean_iou": float(array.mean()) if len(array) else 0.0,
        "singlebox_888_hit_0_3": float((array >= 0.3).mean()) if len(array) else 0.0,
        "singlebox_888_hit_0_5": float((array >= 0.5).mean()) if len(array) else 0.0,
        "singlebox_888_strict_set_f1_0_3": float(np.mean(strict_f1_03)) if strict_f1_03 else 0.0,
        "singlebox_888_strict_set_f1_0_5": float(np.mean(strict_f1_05)) if strict_f1_05 else 0.0,
        "singlebox_888_mean_pred_count": float(np.mean(pred_counts)) if pred_counts else 0.0,
    }
