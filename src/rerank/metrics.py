from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .box_features import iou_xyxy


def singlebox_metrics(pred: pd.DataFrame, method: str) -> dict[str, Any]:
    if pred.empty:
        return {"method": method, "n": 0}
    ious = pred["iou"].astype(float).to_numpy()
    return {
        "method": method,
        "n": int(len(pred)),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "Hit@0.1": float(np.mean(ious >= 0.1)),
        "Hit@0.3": float(np.mean(ious >= 0.3)),
        "Hit@0.5": float(np.mean(ious >= 0.5)),
        "bbox_missing_rate": float(pred.get("bbox_missing", pd.Series(False, index=pred.index)).astype(bool).mean()),
        "bbox_invalid_rate": float(pred.get("bbox_invalid", pd.Series(False, index=pred.index)).astype(bool).mean()),
    }


def candidates_to_singlebox_predictions(df: pd.DataFrame, method: str, score_col: str) -> pd.DataFrame:
    rows = []
    for qid, part in df.groupby("query_id", sort=False):
        if part.empty:
            continue
        row = part.sort_values(score_col, ascending=False).iloc[0]
        gt = [row["gt_x1"], row["gt_y1"], row["gt_x2"], row["gt_y2"]]
        pred = [row["pred_x1"], row["pred_y1"], row["pred_x2"], row["pred_y2"]]
        iou = iou_xyxy(pred, gt)
        rows.append(
            {
                "method": method,
                "query_id": qid,
                "task_id": row.get("task_id", qid),
                "sample_id": row.get("sample_id", qid),
                "split": row.get("split", ""),
                "dicom_id": row.get("dicom_id", ""),
                "subject_id": row.get("subject_id", ""),
                "study_id": row.get("study_id", ""),
                "image_path": row.get("image_path", ""),
                "finding": row.get("finding", ""),
                "claim_sentence": row.get("claim_sentence", ""),
                "gt_x1": gt[0],
                "gt_y1": gt[1],
                "gt_x2": gt[2],
                "gt_y2": gt[3],
                "pred_x1": pred[0],
                "pred_y1": pred[1],
                "pred_x2": pred[2],
                "pred_y2": pred[3],
                "score": float(row.get(score_col, 0.0)),
                "candidate_id": row.get("candidate_id", ""),
                "candidate_source": row.get("candidate_source", row.get("source_model", "")),
                "iou": iou,
                "hit_0_1": iou >= 0.1,
                "hit_0_3": iou >= 0.3,
                "hit_0_5": iou >= 0.5,
                "bbox_missing": False,
                "bbox_invalid": False,
            }
        )
    return pd.DataFrame(rows)


def oracle_singlebox(df: pd.DataFrame, method: str = "topK_oracle") -> pd.DataFrame:
    score_col = "gold_iou" if "gold_iou" in df.columns else "target_iou"
    return candidates_to_singlebox_predictions(df, method, score_col)


def boxes_json(boxes: list[list[float]]) -> str:
    return json.dumps([[float(x) for x in b] for b in boxes])

