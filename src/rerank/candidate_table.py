from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_ms_cxr_multibox_phrase_grounding_v1 as mb  # noqa: E402
from scripts import run_ms_cxr_singlebox_fair_yolo_dino_fusion_v1 as sb  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v1 as ybase  # noqa: E402
from scripts import run_ms_cxr_yolo_rule_context_v2 as yv2  # noqa: E402

from .box_features import add_box_columns, center_distance, iou_xyxy, overlap_ratios
from .dino_reliability import add_dino_reliability


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


SINGLEBOX_DETECTOR = _path_from_env("MSCXR_SINGLEBOX_DETECTOR_DIR", PROJECT_ROOT / "experiments" / "ms_cxr_singlebox_fair_detector_sweep_v2" / "predictions")
SINGLEBOX_SEM = _path_from_env("MSCXR_SINGLEBOX_SEM_DIR", PROJECT_ROOT / "experiments" / "ms_cxr_singlebox_semantic_finegrid_fair_v1")
SINGLEBOX_FUSION = _path_from_env("MSCXR_SINGLEBOX_FUSION_DIR", PROJECT_ROOT / "experiments" / "ms_cxr_singlebox_fair_yolo_dino_fusion_v1")
ROW_SCORER = _path_from_env("MSCXR_ROW_SCORER_DIR", PROJECT_ROOT / "experiments" / "ms_cxr_rowlevel_candidate_set_scorer_yolo8_nsml_yolo11_sm_v1")
UNIFIED_FINEGRID = PROJECT_ROOT / "experiments" / "ms_cxr_semantic_finegrid_unified_yolo8_nsml_yolo11_sm_v1"
CONTRASTIVE_FINEGRID = PROJECT_ROOT / "experiments" / "ms_cxr_finegrid_plus_contrastive_expert_v1"
STAGE1_DATA = PROJECT_ROOT / "training" / "ms_cxr_vfm_localizer_stage1_p10_p19" / "datasets"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _candidate_id(df: pd.DataFrame) -> pd.Series:
    if "candidate_id" in df.columns:
        return df["candidate_id"].astype(str)
    return (
        df.get("task_id", df.get("sample_id", pd.Series("", index=df.index))).astype(str)
        + "::"
        + df.get("source_model", pd.Series("candidate", index=df.index)).astype(str)
        + "::"
        + df.get("rank", df.get("candidate_rank", pd.Series(0, index=df.index))).astype(str)
        + "::"
        + df.index.astype(str)
    )


def _normalize_candidate_schema(df: pd.DataFrame, protocol: str, split: str, query_col: str) -> pd.DataFrame:
    out = df.copy()
    out["protocol"] = protocol
    out["split"] = split
    out["query_id"] = out[query_col].astype(str)
    out["candidate_id"] = _candidate_id(out)
    out["candidate_source"] = out.get("source_model", pd.Series("unknown", index=out.index)).astype(str)
    out = add_box_columns(out)
    if "rule_context_score" not in out.columns and "region_score" in out.columns:
        out["rule_context_score"] = out["region_score"]
    if "train_prior_score" not in out.columns and "prior_iou" in out.columns:
        out["train_prior_score"] = out["prior_iou"]
    if "existing_hybrid_score" not in out.columns and "score_head" in out.columns:
        out["existing_hybrid_score"] = out["score_head"]
    return add_dino_reliability(out)


def _singlebox_rows(split: str) -> dict[str, dict[str, Any]]:
    return {str(r["sample_id"]): r for r in sb.row_dicts(split)}


def _singlebox_dino_map(split: str, allow_cache_fill: bool) -> dict[str, np.ndarray]:
    path = SINGLEBOX_FUSION / "predictions" / f"rad_dino_full_phrase_singlebox_{split}_predictions.csv"
    if not path.exists() and allow_cache_fill:
        # This creates a missing cache file only; it does not overwrite existing results.
        sb.load_rad_prediction("full_phrase", split, "cuda", False)
    df = _safe_read_csv(path)
    if df.empty:
        return {}
    return sb.dino_map(df)


def _norm_to_xyxy(norm_box: np.ndarray, width: float, height: float) -> list[float]:
    cx, cy, w, h = [float(x) for x in norm_box]
    return [
        (cx - w / 2.0) * width,
        (cy - h / 2.0) * height,
        (cx + w / 2.0) * width,
        (cy + h / 2.0) * height,
    ]


def build_singlebox_candidate_table(
    split: str,
    *,
    detector_tags: list[str],
    allow_cache_fill: bool = False,
) -> pd.DataFrame:
    """Build single-box 888 query-instance candidate table.

    Candidate rows are made from the strict single-box detector sweep.  Eval
    gold IoU is included in the saved table for diagnostics/metrics, but the
    runner must not use eval gold during hyperparameter search.
    """

    rows = _singlebox_rows(split)
    train_rows_for_prior = list(_singlebox_rows("train").values())
    priors = ybase.make_train_priors(train_rows_for_prior)
    dino = _singlebox_dino_map(split, allow_cache_fill)
    parts = []
    for tag in detector_tags:
        path = SINGLEBOX_DETECTOR / f"{tag}_{split}_conf0p001_candidates.csv"
        df = _safe_read_csv(path)
        if df.empty:
            continue
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No single-box detector candidates for split={split}, tags={detector_tags}")
    cand = pd.concat(parts, ignore_index=True)
    rows_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows.values():
        rows_by_key.setdefault((str(row["dicom_id"]), int(ybase.CLASS_TO_ID[row["finding"]])), []).append(row)
    out_rows: list[dict[str, Any]] = []
    for _, c in cand.iterrows():
        for row in rows_by_key.get((str(c["dicom_id"]), int(c["class_id"])), []):
            pred = [float(c["x1"]), float(c["y1"]), float(c["x2"]), float(c["y2"])]
            gt = [float(x) for x in row["gold_bbox_xyxy"]]
            width, height = float(row["image_width"]), float(row["image_height"])
            dino_norm = dino.get(str(row["sample_id"]))
            dino_box = _norm_to_xyxy(dino_norm, width, height) if dino_norm is not None else None
            q, prior = yv2.target_prior_for_row(row, priors, {"side_mode": "radiology_right", "prior_train_weight": 0.75})
            prior_box = _norm_to_xyxy(prior, width, height)
            cand_in_dino, dino_in_cand = overlap_ratios(pred, dino_box)
            out_rows.append(
                {
                    "task_id": row["task_id"],
                    "sample_id": row["sample_id"],
                    "dicom_id": row["dicom_id"],
                    "subject_id": row["subject_id"],
                    "study_id": row["study_id"],
                    "image_path": row["image_path"],
                    "finding": row["finding"],
                    "class_id": ybase.CLASS_TO_ID[row["finding"]],
                    "claim_sentence": row["claim_sentence"],
                    "laterality": q.get("laterality", "unknown"),
                    "vertical": q.get("vertical", "unknown"),
                    "gt_x1": gt[0],
                    "gt_y1": gt[1],
                    "gt_x2": gt[2],
                    "gt_y2": gt[3],
                    "pred_x1": pred[0],
                    "pred_y1": pred[1],
                    "pred_x2": pred[2],
                    "pred_y2": pred[3],
                    "image_width": width,
                    "image_height": height,
                    "confidence": float(c["score"]),
                    "source_model": str(c.get("source_model", tag)),
                    "candidate_rank": int(c.get("rank", 999)),
                    "rank_bonus": 1.0 / (1.0 + float(c.get("rank", 999))),
                    "rank_norm": float(c.get("rank", 999)) / 999.0,
                    "prior_iou": iou_xyxy(pred, prior_box),
                    "xattn_iou": iou_xyxy(pred, dino_box) if dino_box is not None else 0.0,
                    "candidate_in_dino": cand_in_dino,
                    "dino_in_candidate": dino_in_cand,
                    "center_distance_to_dino": center_distance(pred, dino_box, width, height),
                    "region_score": yv2.center_region_score_v2(ybase.xyxy_to_norm(pred, width, height), q, "radiology_right"),
                    "consensus_max": 0.0,
                    "consensus_mean": 0.0,
                    "gold_iou": iou_xyxy(pred, gt),
                }
            )
    out = pd.DataFrame(out_rows)
    out = _normalize_candidate_schema(out, "singlebox_888", split, "sample_id")
    return out


def build_multibox_candidate_table(split: str) -> pd.DataFrame:
    path = ROW_SCORER / "predictions" / f"{split}_scored_candidates.csv"
    if not path.exists():
        path = ROW_SCORER / "predictions" / f"{split}_row_candidates_light.csv"
    df = _safe_read_csv(path)
    if df.empty:
        raise FileNotFoundError(path)
    rows = _read_jsonl(STAGE1_DATA / f"{split}.jsonl")
    groups = mb.make_groups(rows)
    task_to_group = {str(t): gid for gid, g in groups.items() for t in g["task_ids"]}
    df = df.copy()
    df["group_id"] = df["task_id"].astype(str).map(task_to_group)
    df = df[df["group_id"].notna()].copy()
    group_info = {
        gid: {
            "gold_boxes": g["gt_boxes"],
            "gold_count": len(g["gt_boxes"]),
            "phrase": g["claim_sentence"],
        }
        for gid, g in groups.items()
    }
    best_iou = []
    best_idx = []
    gold_json = []
    for _, r in df.iterrows():
        pred = [float(r["pred_x1"]), float(r["pred_y1"]), float(r["pred_x2"]), float(r["pred_y2"])]
        boxes = group_info[str(r["group_id"])]["gold_boxes"]
        vals = [iou_xyxy(pred, b) for b in boxes]
        if vals:
            idx = int(np.argmax(vals))
            best_idx.append(idx)
            best_iou.append(float(vals[idx]))
        else:
            best_idx.append(-1)
            best_iou.append(0.0)
        gold_json.append(json.dumps(boxes))
    df["gold_boxes"] = gold_json
    df["gold_count"] = df["group_id"].map(lambda x: group_info[str(x)]["gold_count"])
    df["gold_best_iou"] = best_iou
    df["gold_best_index"] = best_idx
    df = _normalize_candidate_schema(df, "multibox_1444", split, "group_id")
    return df
